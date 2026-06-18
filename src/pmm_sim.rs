/// pmm-sim integration: Yellowstone gRPC account subscription + subprocess engine.
///
/// Architecture:
/// - `PmmAccountCache` subscribes to the Yellowstone Geyser gRPC stream and keeps
///   an in-memory snapshot of every PMM pool account, updated immediately on change.
/// - `PmmSimEngine` manages a long-lived `pmm-sim serve` subprocess and sends
///   simulation requests via newline-delimited JSON over stdin/stdout pipes.
///
/// When a SimulationRequest arrives from the queue, `PmmSimEngine::simulate` sends the
/// Metis swap instruction together with the freshest account state to the subprocess.
/// The subprocess executes the instruction in LiteSVM and returns the outcome (amount_out,
/// compute units, success/error) as a JSON line on stdout.
use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
    time::Duration,
};

use base64::{engine::general_purpose::STANDARD as B64, Engine};
use serde::{Deserialize, Serialize};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader as TokioBufReader},
    process::Command,
    sync::Mutex as TokioMutex,
    time::timeout,
};
use tracing::{error, warn};

use crate::config::PmmSimConfig;

// ── Account cache ────────────────────────────────────────────────────────────

/// A single cached account (mirroring what LiteSVM / Solana SDK needs).
#[derive(Clone, Debug)]
pub struct CachedAccount {
    pub lamports: u64,
    pub data: Vec<u8>,
    pub owner: [u8; 32],
    pub executable: bool,
    pub rent_epoch: u64,
}

/// Thread-safe map from base58 pubkey string → account state.
pub type AccountCache = Arc<RwLock<HashMap<String, CachedAccount>>>;

// ── IPC protocol types ────────────────────────────────────────────────────────

/// Full transaction simulation request sent to pmm-sim subprocess (JSON line).
#[derive(Serialize)]
pub struct FullSimRequest {
    pub id: u64,
    pub fee_payer: String,
    pub src_mint: String,
    pub src_amount: u64,
    pub token_mints: Vec<String>,
    /// ALL instructions: compute_budget + setup + swap + cleanup (in order).
    pub instructions: Vec<IpcInstruction>,
    /// Resolved ALT contents (addresses, not ALT account pubkeys).
    pub lookup_tables: Vec<IpcLookupTable>,
    /// Fresh account state from Yellowstone cache for accounts in instructions.
    pub accounts: Vec<IpcAccount>,
    pub jito_tip_lamports: u64,
    pub cu_limit: u32,
    pub route_sig: String,
    pub route_labels: Vec<String>,
}

#[derive(Serialize)]
pub struct IpcLookupTable {
    pub key: String,
    pub addresses: Vec<String>,
}

#[derive(Serialize)]
pub struct IpcInstruction {
    pub program_id: String,
    pub accounts: Vec<IpcAccountMeta>,
    pub data: String, // base64
}

#[derive(Serialize)]
pub struct IpcAccountMeta {
    pub pubkey: String,
    pub is_signer: bool,
    pub is_writable: bool,
}

#[derive(Serialize)]
pub struct IpcAccount {
    pub pubkey: String,
    pub lamports: u64,
    pub data: String, // base64
    pub owner: String,
    pub executable: bool,
    pub rent_epoch: u64,
}

/// Response received from pmm-sim subprocess stdout (JSON line).
#[derive(Deserialize, Debug)]
pub struct PmmSimResponse {
    pub id: u64,
    pub success: bool,
    pub amount_out: Option<u64>,
    pub compute_units: Option<u64>,
    pub error: Option<String>,
}

// ── BisonFi build+quote IPC (first-test mode) ─────────────────────────────────
//
// Distinct from the full-tx simulation path: this asks pmm-sim to BUILD a DFlow
// `swap2` (spoof-Magnus) instruction for a single BisonFi leg and simulate it,
// returning BOTH the predicted output AND the ready-to-send instruction.

/// Request to build + simulate a single BisonFi `WSOL -> USDC` leg.
#[derive(Serialize)]
pub struct BuildBisonRequest {
    /// Discriminates this op from the full-sim path. Always "build_bison".
    pub op: &'static str,
    pub id: u64,
    /// Real trading wallet — the returned instruction references its real ATAs.
    pub fee_payer: String,
    /// BisonFi market (pool) account.
    pub market: String,
    pub src_mint: String,
    pub dst_mint: String,
    pub amount_in: u64,
    /// Fresh pool account state (market + both vaults) from the Yellowstone cache.
    pub accounts: Vec<IpcAccount>,
}

/// One account meta of a built instruction (mirrors Metis `AccountMeta`).
#[derive(Deserialize, Debug, Clone)]
pub struct IpcInstructionAccount {
    pub pubkey: String,
    pub is_signer: bool,
    pub is_writable: bool,
}

/// A fully-built instruction returned by pmm-sim (base64 data, real accounts).
#[derive(Deserialize, Debug, Clone)]
pub struct IpcInstructionOut {
    pub program_id: String,
    pub accounts: Vec<IpcInstructionAccount>,
    pub data: String,
}

#[derive(Deserialize, Debug)]
pub struct BuildBisonResponse {
    #[allow(dead_code)]
    pub id: u64,
    pub success: bool,
    /// Predicted USDC out (raw units) for the simulated BisonFi leg.
    pub amount_out: Option<u64>,
    #[allow(dead_code)]
    pub compute_units: Option<u64>,
    /// The DFlow swap2 (spoof-Magnus) instruction to splice into the bundle.
    pub instruction: Option<IpcInstructionOut>,
    pub error: Option<String>,
}

// ── Subprocess handle ─────────────────────────────────────────────────────────

struct SubprocessHandle {
    stdin: tokio::process::ChildStdin,
    stdout: TokioBufReader<tokio::process::ChildStdout>,
}

/// Manages the long-lived `pmm-sim serve` subprocess.
///
/// All simulation requests are serialised through a single async mutex so that
/// requests and responses on the pipe stay correctly paired.
pub struct PmmSimEngine {
    handle: TokioMutex<Option<SubprocessHandle>>,
    cfg: PmmSimConfig,
    timeout: Duration,
}

impl PmmSimEngine {
    pub fn new(cfg: PmmSimConfig) -> Arc<Self> {
        let timeout = Duration::from_millis(cfg.timeout_ms.max(10));
        Arc::new(Self {
            handle: TokioMutex::new(None),
            cfg,
            timeout,
        })
    }

    /// Spawn (or re-spawn) the pmm-sim subprocess.
    pub async fn ensure_started(&self) {
        let mut guard = self.handle.lock().await;
        if guard.is_some() {
            return;
        }
        match self.spawn_subprocess().await {
            Ok(h) => {
                *guard = Some(h);
                eprintln!("[pmm_sim] subprocess started (binary={})", self.cfg.binary);
            }
            Err(e) => {
                let cwd = std::env::current_dir()
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|_| "?".to_string());
                let resolved = std::path::Path::new(&self.cfg.binary)
                    .canonicalize()
                    .map(|p| p.display().to_string())
                    .unwrap_or_else(|_| "<does not exist>".to_string());
                error!(
                    "[pmm_sim] failed to spawn subprocess: {e} | binary='{}' resolved='{resolved}' cwd='{cwd}' setup_path='{}' programs_path='{}' — check that this executable exists and is built (cd pmm-sim && cargo build --release)",
                    self.cfg.binary, self.cfg.setup_path, self.cfg.programs_path,
                );
            }
        }
    }

    async fn spawn_subprocess(&self) -> anyhow::Result<SubprocessHandle> {
        let mut child = Command::new(&self.cfg.binary)
            .args(["serve", "--setup-path", &self.cfg.setup_path, "--programs-path", &self.cfg.programs_path])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::inherit())
            .kill_on_drop(true)
            .spawn()?;

        let stdin = child.stdin.take().expect("subprocess stdin");
        let stdout = TokioBufReader::new(child.stdout.take().expect("subprocess stdout"));

        // Detach the child; it will be killed when dropped (kill_on_drop).
        tokio::spawn(async move { let _ = child.wait().await; });

        Ok(SubprocessHandle { stdin, stdout })
    }

    /// Send one simulation request and await the response.
    ///
    /// Returns `None` if the subprocess is unavailable or the call times out.
    pub async fn simulate(
        &self,
        request: &FullSimRequest,
    ) -> Option<PmmSimResponse> {
        let line = self.roundtrip(request).await?;
        match serde_json::from_str::<PmmSimResponse>(line.trim()) {
            Ok(resp) => Some(resp),
            Err(e) => {
                warn!("[pmm_sim] failed to parse response: {e} line={line:?}");
                *self.handle.lock().await = None;
                None
            }
        }
    }

    /// Whether the subprocess handle is currently live (spawned and not torn down).
    pub async fn is_running(&self) -> bool {
        self.handle.lock().await.is_some()
    }

    /// Build + simulate a single BisonFi `WSOL -> USDC` leg.
    pub async fn build_bison(
        &self,
        request: &BuildBisonRequest,
    ) -> Option<BuildBisonResponse> {
        let line = self.roundtrip(request).await?;
        match serde_json::from_str::<BuildBisonResponse>(line.trim()) {
            Ok(resp) => Some(resp),
            Err(e) => {
                warn!("[pmm_sim] failed to parse build_bison response: {e} line={line:?}");
                *self.handle.lock().await = None;
                None
            }
        }
    }

    /// Serialize `request` to one JSON line, write it to the subprocess, and
    /// return the raw response line. Closes the handle on any IO/timeout error.
    async fn roundtrip<T: serde::Serialize>(&self, request: &T) -> Option<String> {
        let mut guard = self.handle.lock().await;
        let handle = guard.as_mut()?;

        let line = {
            let mut s = serde_json::to_string(request).ok()?;
            s.push('\n');
            s
        };

        let result = timeout(self.timeout, async {
            handle.stdin.write_all(line.as_bytes()).await?;
            handle.stdin.flush().await?;
            let mut resp_line = String::new();
            handle.stdout.read_line(&mut resp_line).await?;
            Ok::<String, tokio::io::Error>(resp_line)
        })
        .await;

        match result {
            Ok(Ok(resp_line)) if !resp_line.trim().is_empty() => Some(resp_line),
            Ok(Ok(_)) => {
                warn!("[pmm_sim] empty response from subprocess");
                *guard = None;
                None
            }
            Ok(Err(e)) => {
                warn!("[pmm_sim] IO error with subprocess: {e}");
                *guard = None;
                None
            }
            Err(_) => {
                warn!("[pmm_sim] request timed out after {}ms", self.timeout.as_millis());
                *guard = None;
                None
            }
        }
    }
}

// ── Yellowstone gRPC subscription ────────────────────────────────────────────

pub mod proto {
    tonic::include_proto!("geyser");
}

use proto::{
    geyser_client::GeyserClient, CommitmentLevel, SubscribeRequest,
    SubscribeRequestFilterAccounts, SubscribeUpdate,
};
use tonic::{
    metadata::MetadataValue,
    transport::Channel,
    Request,
};

/// Optional live-update trigger: when a watched account whose pubkey is in
/// `keys` updates, `notify` is signalled. Used by the BisonFi test flow to react
/// the instant the pool state changes (rather than polling on a timer).
#[derive(Clone)]
pub struct UpdateTrigger {
    pub keys: Arc<std::collections::HashSet<String>>,
    pub notify: Arc<tokio::sync::Notify>,
    pub metrics: Arc<crate::bison_metrics::BisonMetrics>,
}

/// Spawn a background task that subscribes to Yellowstone gRPC and keeps the
/// account cache up to date. On disconnect the task reconnects automatically.
pub fn spawn_yellowstone_subscription(
    endpoint: String,
    x_token: String,
    watchlist: Vec<String>,
    cache: AccountCache,
    trigger: Option<UpdateTrigger>,
) {
    tokio::spawn(async move {
        loop {
            if let Err(e) =
                run_subscription(&endpoint, &x_token, &watchlist, cache.clone(), trigger.as_ref())
                    .await
            {
                warn!("[yellowstone] subscription error: {e} — reconnecting in 2s");
            }
            tokio::time::sleep(Duration::from_secs(2)).await;
        }
    });
}

async fn run_subscription(
    endpoint: &str,
    x_token: &str,
    watchlist: &[String],
    cache: AccountCache,
    trigger: Option<&UpdateTrigger>,
) -> anyhow::Result<()> {
    let channel = Channel::from_shared(endpoint.to_string())?
        .tls_config(tonic::transport::ClientTlsConfig::new().with_enabled_roots())?
        .connect()
        .await?;

    let token_meta = x_token.to_string();
    let mut client = GeyserClient::with_interceptor(channel, move |mut req: Request<()>| {
        if !token_meta.is_empty() {
            if let Ok(val) = MetadataValue::try_from(token_meta.as_str()) {
                req.metadata_mut().insert("x-token", val);
            }
        }
        Ok(req)
    });

    // Split watchlist into chunks of 1000 for the filter (provider limits may vary).
    let mut accounts_map = std::collections::HashMap::new();
    for (i, chunk) in watchlist.chunks(1000).enumerate() {
        accounts_map.insert(format!("accounts_{i}"), SubscribeRequestFilterAccounts {
            account: chunk.to_vec(),
            owner: vec![],
            nonempty_txn_signature: false,
        });
    }
    if accounts_map.is_empty() {
        // Empty watchlist — nothing to subscribe to; sleep and retry.
        tokio::time::sleep(Duration::from_secs(30)).await;
        return Ok(());
    }

    let sub_request = SubscribeRequest {
        accounts: accounts_map,
        commitment: Some(CommitmentLevel::Processed as i32),
        ..Default::default()
    };

    let stream_req = tokio_stream::once(sub_request);
    let mut stream = client.subscribe(stream_req).await?.into_inner();

    use futures::StreamExt;

    while let Some(msg) = stream.next().await {
        match msg {
            Ok(SubscribeUpdate { update_oneof: Some(proto::subscribe_update::UpdateOneof::Account(acc_update)), .. }) => {
                if let Some(info) = acc_update.account {
                    if info.pubkey.len() == 32 {
                        let pubkey = bs58::encode(&info.pubkey).into_string();
                        let cached = CachedAccount {
                            lamports: info.lamports,
                            data: info.data,
                            owner: info.owner.as_slice().try_into().unwrap_or([0u8; 32]),
                            executable: info.executable,
                            rent_epoch: info.rent_epoch,
                        };
                        if let Ok(mut w) = cache.write() {
                            w.insert(pubkey.clone(), cached);
                        }
                        // Fire the live-update trigger if this account is watched.
                        if let Some(t) = trigger {
                            if t.keys.contains(&pubkey) {
                                t.metrics.record_pool_update();
                                t.notify.notify_one();
                            }
                        }
                    }
                }
            }
            Ok(_) => {}
            Err(e) => {
                warn!("[yellowstone] stream error: {e}");
                break;
            }
        }
    }
    Ok(())
}

// ── Request builder helpers ───────────────────────────────────────────────────

/// Collect IpcAccounts for all pubkeys referenced in the given instructions, from the cache.
pub fn collect_all_instruction_accounts(
    instructions: &[&crate::metis::InstructionData],
    cache: &AccountCache,
) -> Vec<IpcAccount> {
    let mut seen = std::collections::HashSet::new();
    let mut result = Vec::new();
    let r = match cache.read() { Ok(r) => r, Err(_) => return result };
    for ix in instructions {
        for am in &ix.accounts {
            if seen.insert(am.pubkey.clone()) {
                if let Some(acc) = r.get(&am.pubkey) {
                    result.push(IpcAccount {
                        pubkey: am.pubkey.clone(),
                        lamports: acc.lamports,
                        data: B64.encode(&acc.data),
                        owner: bs58::encode(acc.owner).into_string(),
                        executable: acc.executable,
                        rent_epoch: acc.rent_epoch,
                    });
                }
            }
        }
    }
    result
}

/// Collect `IpcAccount`s for an explicit list of pubkeys from the cache.
/// Returns only the ones currently present; missing pubkeys are reported via
/// the second element so the caller can decide whether state is warm enough.
pub fn collect_accounts_by_pubkey(
    pubkeys: &[String],
    cache: &AccountCache,
) -> (Vec<IpcAccount>, Vec<String>) {
    let mut found = Vec::new();
    let mut missing = Vec::new();
    let r = match cache.read() {
        Ok(r) => r,
        Err(_) => return (found, pubkeys.to_vec()),
    };
    for pk in pubkeys {
        match r.get(pk) {
            Some(acc) => found.push(IpcAccount {
                pubkey: pk.clone(),
                lamports: acc.lamports,
                data: B64.encode(&acc.data),
                owner: bs58::encode(acc.owner).into_string(),
                executable: acc.executable,
                rent_epoch: acc.rent_epoch,
            }),
            None => missing.push(pk.clone()),
        }
    }
    (found, missing)
}

/// Bootstrap account cache from RPC using batched getMultipleAccounts.
pub fn bootstrap_from_rpc(
    watchlist: &[String],
    cache: &AccountCache,
    rpc: &solana_client::rpc_client::RpcClient,
    batch_size: usize,
) {
    use solana_sdk::pubkey::Pubkey;
    let batch_size = batch_size.min(100).max(1);
    let mut total = 0usize;
    let chunks: usize = (watchlist.len() + batch_size - 1) / batch_size;
    for chunk in watchlist.chunks(batch_size) {
        let pubkeys: Vec<Pubkey> = chunk.iter()
            .filter_map(|s| s.parse().ok())
            .collect();
        match rpc.get_multiple_accounts(&pubkeys) {
            Ok(accounts) => {
                let mut w = match cache.write() { Ok(w) => w, Err(_) => continue };
                for (pk, maybe_acc) in pubkeys.iter().zip(accounts.iter()) {
                    if let Some(acc) = maybe_acc {
                        w.insert(pk.to_string(), CachedAccount {
                            lamports: acc.lamports,
                            data: acc.data.clone(),
                            owner: acc.owner.to_bytes(),
                            executable: acc.executable,
                            rent_epoch: acc.rent_epoch,
                        });
                        total += 1;
                    }
                }
            }
            Err(e) => eprintln!("[cache_bootstrap] RPC batch error: {e}"),
        }
    }
    eprintln!("[cache_bootstrap] accounts={total} batches={chunks}");
}

/// Write a programs_registry.json file to programs_path for pmm-sim to load extra programs.
pub fn write_programs_registry(programs: &[crate::config::ProgramEntry], programs_path: &str) {
    use std::io::Write as _;
    if programs.is_empty() { return; }
    let path = format!("{programs_path}/programs_registry.json");
    let json = serde_json::to_string_pretty(programs).unwrap_or_default();
    match std::fs::File::create(&path) {
        Ok(mut f) => {
            let _ = f.write_all(json.as_bytes());
            eprintln!("[pmm_sim] wrote programs_registry: {path} ({} programs)", programs.len());
        }
        Err(e) => eprintln!("[pmm_sim] cannot write programs_registry {path}: {e}"),
    }
}

/// Create a new, empty account cache.
pub fn new_account_cache() -> AccountCache {
    Arc::new(RwLock::new(HashMap::new()))
}

/// Load cached accounts from pmm-sim's on-disk JSON files into the cache.
///
/// Provides a warm starting state so simulations can run immediately before the
/// first Yellowstone update arrives (typically within a few hundred milliseconds).
pub fn preload_cache_from_disk(accounts_path: &str, cache: &AccountCache) {
    let path = std::path::Path::new(accounts_path);
    if !path.exists() {
        return;
    }

    let entries = match std::fs::read_dir(path) {
        Ok(e) => e,
        Err(_) => return,
    };

    let mut loaded = 0usize;
    for entry in entries.flatten() {
        let p = entry.path();
        if p.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        if let Ok(contents) = std::fs::read_to_string(&p) {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&contents) {
                let pubkey = v["pubkey"].as_str().unwrap_or_default().to_string();
                let lamports = v["account"]["lamports"].as_u64().unwrap_or_default();
                let data_b64 = v["account"]["data"][0].as_str().unwrap_or_default();
                let data = B64.decode(data_b64).unwrap_or_default();
                let owner_str = v["account"]["owner"].as_str().unwrap_or("11111111111111111111111111111111");
                let owner_bytes = bs58::decode(owner_str).into_vec().unwrap_or_default();
                let owner: [u8; 32] = owner_bytes.try_into().unwrap_or([0u8; 32]);
                let executable = v["account"]["executable"].as_bool().unwrap_or(false);
                let rent_epoch = v["account"]["rentEpoch"].as_u64().unwrap_or(u64::MAX);

                if !pubkey.is_empty() {
                    if let Ok(mut w) = cache.write() {
                        w.insert(pubkey, CachedAccount { lamports, data, owner, executable, rent_epoch });
                        loaded += 1;
                    }
                }
            }
        }
    }
    eprintln!("[pmm_sim] preloaded {loaded} accounts from {accounts_path}");
}
