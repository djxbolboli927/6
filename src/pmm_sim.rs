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

/// All PMM pool account pubkeys we subscribe to via Yellowstone gRPC.
///
/// Extracted from pmm-sim/cfg/setup.toml. These are the mutable state accounts
/// (token reserves, market state, oracles) that change with every swap and need
/// real-time updates. Static accounts (programs, sysvars) are loaded from disk.
pub const PMM_WATCHED_ACCOUNTS: &[&str] = &[
    // ── HumidiFi markets (swap-v1/v2/v3 share the same market/base_ta/quote_ta) ──
    "FksffEqnBRixYGR791Qw2MgdU7zNCpHVFYBL4Fa4qVuH", // market
    "C3FzbX9n1YD2dow2dCmEv5uNyyf22Gb3TLAEqGBhw5fY", // base_ta
    "3RWFAQBRkNGq7CMGcTLK3kXDgFTe9jgMeFYqk8nHwcWh", // quote_ta
    "DB3sUCP2H4icbeKmK6yb6nUxU5ogbcRHtGuq7W2RoRwW",
    "8BrVfsvzb1DZqCactbYWoKSv24AfsLBuXJqzpzYCwznF",
    "HsQcHFFNUVTp3MWrXYbuZchBNd4Pwk8636bKzLvpfYNR",
    "6n9VhCwQ7EwK6NqFDjnHPzEk6wZdRBTfh43RFgHQWHuQ",
    "Cv9St5tDTGwpbG5UVvM6QvFmf3FYSXc14W9BYvQN5wAZ",
    "7Rf8Gu8YemSoGjZT3z1cL5BT9HLbGywcyaz8Mrbhd1MH",
    "AvGeFw71N5sNfV97mZ1uNrHg4yfufRicCJUrS9j2ehTX",
    "ECEPWwZJ1U1Vjsj1X5sUbZYETKMSCjYHuoTMVitCn64t",
    "FBWtVVvzsRuAAzVX8ua1hden9KmgPrC2rFijuwEn1ngJ",
    "5dhYayH9qvzNyCPoh2hKN8TJqumoGsWdyZ9UfPLXBfD9",
    "86DHdQfRpghMCmXYKDg93bXtmi3doBL5NeY9DYwAj6rz",
    "FqEVwrQJsJ4AciwJRYLPQEdstDfUYJp6z3f6XYb6MnSb",
    // HumidiFi v2/v3 extra accounts
    "4nXmcNY1NjNd9sjA3qajUdUYpVDvwxTzGZT9oS4KpaD1", // add1
    "J1to1yufRnoWn81KYg1XkTWzmKjnYSnmE2VY8DGUJ9Qv", // vote
    // ── Tessera ──────────────────────────────────────────────────────────────
    "FLckHLGMJy5gEoXWwcE68Nprde1D4araK4TGLw4pQq2n", // market
    "5pVN5XZB8cYBjNLFrsBCPWkCQBan5K5Mq2dWGzwPgGJV", // base_ta
    "9t4P5wMwfFkyn92Z7hf463qYKEZf8ERVZsGBEPNp8uJx", // quote_ta
    "8ekCy2jHHUbW2yeNGFWYJT9Hm9FW7SvZcZK66dSZCDiF", // global_state
    // ── GoonFi ───────────────────────────────────────────────────────────────
    "4uWuh9fC7rrZKrN8ZdJf69MN1e2S7FPpMqcsyY1aof6K", // market
    "pKiUC9hDXv52xqU1p3BKypV9AQjAMgfZUGRnoBsdkKm",  // base_ta
    "Gsy5Zr7Vxn5KckAbduPHHGR1qzPJ4w3GSYmcinWAkhrC", // quote_ta
    "7XqYD6DEGmDXooB1E8NNRWV9pWAmm1z6WYpsfjnABTUz", // blacklist
    // ── SolFi V2 ─────────────────────────────────────────────────────────────
    "65ZHSArs5XxPseKQbB1B4r16vDxMWnCxHMzogDAqiDUc", // market 1
    "CRo8DBwrmd97DJfAnvCv96tZPL5Mktf2NZy2ZnhDer1A",
    "GhFfLFSprPpfoRaWakPMmJTMJBHuz6C694jYwxy2dAic",
    "FmxXDSR9WvpJTCh738D1LEDuhMoA8geCtZgHb3isy7Dp", // cfg
    "2ny7eGyZCoeEVTkNLf5HcnJFBKkyA4p4gcrtb3b8y8ou", // oracle
    "FkEB6uvyzuoaGpgs4yRtFtxC4WJxhejNFbUkj5R6wR32", // market 2
    "5bHD9xdEzJdkVuhs54mGPC9BZgUshqgMg4tqmTwhWggc",
    "ARWaajRJyF6PKQryJ4HLzLBfTWM2qmVQUQVtBjk6PgPc",
    "QoFvFhDZg9TaZEi4SsasWpH5xXzk3zBqfRyicGexfNQ",  // cfg 2
    "CyCUgmaCYUZxbux3J2svDzxSryVFMtZNPrnMKS41nc4G", // oracle 2
    // ── ZeroFi ───────────────────────────────────────────────────────────────
    "2h9hhu3gxY9kCdXEwdTHV8yPAMYVoHgKopRyG1HbDwfi", // market
    "7RHJ2WfexqUxy7SXfbNZRZDgZi3D9jtMAQp9VhfzpU8T", // vault_info_base
    "ERP5RTV6cWmoGrv7r9W2V5pbgDFSepc4j97qNnx1Jris",  // vault_base
    "Ef7zPqj4NuZHwaTczUTY9oRbxXrfZseUcKcqPaidCZ5W",  // vault_info_quote
    "7wYJVD8iXmMQjND1fwi1hPr68QwruVVtirbotyJZXaVH",  // vault_quote
    // ── ObricV2 ──────────────────────────────────────────────────────────────
    "BWBHrYqfcjAh5dSiRwzPnY4656cApXVXmkeDmAfwBKQG", // market
    "GZsNmWKbqhMYtdSkkvMdEyQF9k5mLmP7tTKYWZjcHVPE", // second_ref_oracle
    "6YawcNeZ74tRyCv4UfGydYMr7eho7vbUR6ScVffxKAb3",  // third_ref_oracle
    "C3tPQ8TRcHybnPpR8KMASUVD3PukQRRHEsLwxorJMhgm",  // reserve_x
    "AAamGhyPfpQJWfZHTq944NM1cFvoVLDrQxt7HGjeRQUS",  // reserve_y
    "J4HJYz4p7TRP96WVFky3vh7XryxoFehHjoRySUTeSeXw",  // ref_oracle / price_feed
    // ── BisonFi ──────────────────────────────────────────────────────────────
    "51FQwjrvo8J8zXUaKyAznJ5NYpoiTCuqAqCu3HAMB9NZ", // market 1
    "FxGiN5NkigicwrnFshZEAUH9C13yrBALmgYxA9x8sfnQ",  // market_base_ta
    "6DMF4t6Ks8yXhG8K3rrTAeNYrqNrr1DwewHvBmH3a3FX",  // market_quote_ta
    "FC9pWtfdtbyGZ5WHTLneoMSUx6jmTDgqKaxDcm2trsND",  // market 2
    "CL9xU6uijD4FL3ximjt65R6YhUHm1qtiTbBvW1s9TXHZ",
    "Dp4c6UyCHy6N4Xmg2URwq77SGWCRFGKQJFuDnCkXVUF4",
];

// ── IPC protocol types ────────────────────────────────────────────────────────

/// Request sent to pmm-sim subprocess stdin (JSON line).
#[derive(Serialize)]
pub struct PmmSimRequest {
    pub id: u64,
    /// Pubkey of the actual trading keypair (base58).
    pub fee_payer: String,
    /// All token mints referenced in the route (used to compute ATAs to patch).
    pub token_mints: Vec<String>,
    /// Source mint for the first hop.
    pub src_mint: String,
    /// Amount of src tokens to start the simulation with.
    pub src_amount: u64,
    /// The actual swap instruction from Metis.
    pub swap_instruction: IpcInstruction,
    /// Fresh pool account state from the Yellowstone cache.
    pub accounts: Vec<IpcAccount>,
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
                error!("[pmm_sim] failed to spawn subprocess: {e}");
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
        request: &PmmSimRequest,
    ) -> Option<PmmSimResponse> {
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
            Ok(Ok(resp_line)) if !resp_line.trim().is_empty() => {
                match serde_json::from_str::<PmmSimResponse>(resp_line.trim()) {
                    Ok(resp) => Some(resp),
                    Err(e) => {
                        warn!("[pmm_sim] failed to parse response: {e} line={resp_line:?}");
                        // Subprocess may be in a bad state; close the handle.
                        *guard = None;
                        None
                    }
                }
            }
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
                warn!("[pmm_sim] simulation timed out after {}ms", self.timeout.as_millis());
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

/// Spawn a background task that subscribes to Yellowstone gRPC and keeps the
/// account cache up to date. On disconnect the task reconnects automatically.
pub fn spawn_yellowstone_subscription(
    endpoint: String,
    x_token: String,
    cache: AccountCache,
) {
    tokio::spawn(async move {
        loop {
            if let Err(e) = run_subscription(&endpoint, &x_token, cache.clone()).await {
                warn!("[yellowstone] subscription error: {e} — reconnecting in 2s");
            }
            tokio::time::sleep(Duration::from_secs(2)).await;
        }
    });
}

async fn run_subscription(
    endpoint: &str,
    x_token: &str,
    cache: AccountCache,
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

    let accounts_filter = SubscribeRequestFilterAccounts {
        account: PMM_WATCHED_ACCOUNTS.iter().map(|s| s.to_string()).collect(),
        owner: vec![],
        nonempty_txn_signature: false,
    };

    let sub_request = SubscribeRequest {
        accounts: [("pmm_pools".to_string(), accounts_filter)].into_iter().collect(),
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
                            w.insert(pubkey, cached);
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

/// Build a `PmmSimRequest` from a Metis swap instruction + current account cache.
///
/// Collects all accounts from the cache whose pubkeys appear in the instruction's
/// account list, so the subprocess always has the freshest state.
pub fn build_sim_request(
    id: u64,
    fee_payer: &str,
    swap_ix: &crate::metis::InstructionData,
    token_mints: Vec<String>,
    src_mint: &str,
    src_amount: u64,
    cache: &AccountCache,
) -> PmmSimRequest {
    let ix_pubkeys: Vec<String> = swap_ix.accounts.iter().map(|a| a.pubkey.clone()).collect();

    let accounts: Vec<IpcAccount> = if let Ok(r) = cache.read() {
        ix_pubkeys
            .iter()
            .filter_map(|pk| {
                r.get(pk).map(|acc| IpcAccount {
                    pubkey: pk.clone(),
                    lamports: acc.lamports,
                    data: B64.encode(&acc.data),
                    owner: bs58::encode(acc.owner).into_string(),
                    executable: acc.executable,
                    rent_epoch: acc.rent_epoch,
                })
            })
            .collect()
    } else {
        vec![]
    };

    PmmSimRequest {
        id,
        fee_payer: fee_payer.to_string(),
        token_mints,
        src_mint: src_mint.to_string(),
        src_amount,
        swap_instruction: IpcInstruction {
            program_id: swap_ix.program_id.clone(),
            accounts: swap_ix
                .accounts
                .iter()
                .map(|a| IpcAccountMeta {
                    pubkey: a.pubkey.clone(),
                    is_signer: a.is_signer,
                    is_writable: a.is_writable,
                })
                .collect(),
            data: swap_ix.data.clone(),
        },
        accounts,
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
