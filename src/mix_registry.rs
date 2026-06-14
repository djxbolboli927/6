#![allow(dead_code)]
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use solana_client::rpc_client::RpcClient;
use solana_sdk::{hash::hashv, pubkey::Pubkey};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const MIX2_SCHEMA_VERSION: u32 = 2;
const MAX_MIX_RPC_RPS: u64 = 5;
const RPC_CHUNK_SIZE: usize = 100;

#[derive(Clone)]
pub struct VerifiedMixRegistry {
    mix_path: PathBuf,
    output_root: PathBuf,
    source_mix_hash: String,
    rpc_url: String,
    inner: Arc<RwLock<MixRegistryInner>>,
}

#[derive(Clone, Default)]
struct MixRegistryInner {
    pools: Vec<MixPool>,
    valid_accounts: HashSet<Pubkey>,
    invalid_accounts: HashSet<Pubkey>,
    unverified_accounts: HashSet<Pubkey>,
    account_reasons: HashMap<Pubkey, String>,
    account_meta: HashMap<Pubkey, VerifiedAccountMeta>,
    account_importance: HashMap<Pubkey, MixAccountImportance>,
    valid_pools: HashSet<Pubkey>,
    invalid_pools: HashSet<Pubkey>,
    unverified_pools: HashSet<Pubkey>,
    account_to_pools: HashMap<Pubkey, Vec<Pubkey>>,
    pool_to_accounts: HashMap<Pubkey, Vec<Pubkey>>,
    pool_owner: HashMap<Pubkey, Pubkey>,
}

#[derive(Clone)]
struct MixPool {
    pool: Pubkey,
    owner: Option<Pubkey>,
    dex: Option<String>,
    params: serde_json::Value,
    original_mix_entry: serde_json::Value,
    required_accounts: Vec<MixAccountRef>,
}

#[derive(Clone)]
struct MixAccountRef {
    pubkey: Pubkey,
    source: String,
    role: String,
    is_variable: bool,
    importance: MixAccountImportance,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
enum MixAccountImportance {
    RequiredStatic,
    RequiredLive,
    OptionalStatic,
    TxOnly,
    ProgramOrSysvar,
}

impl MixAccountImportance {
    fn as_str(self) -> &'static str {
        match self {
            Self::RequiredStatic => "RequiredStatic",
            Self::RequiredLive => "RequiredLive",
            Self::OptionalStatic => "OptionalStatic",
            Self::TxOnly => "TxOnly",
            Self::ProgramOrSysvar => "ProgramOrSysvar",
        }
    }

    fn can_block_pool(self) -> bool {
        matches!(self, Self::RequiredStatic | Self::RequiredLive)
    }

    fn can_drop_route(self) -> bool {
        matches!(
            self,
            Self::RequiredStatic | Self::RequiredLive | Self::TxOnly
        )
    }
}

#[derive(Clone, Default)]
struct VerifiedAccountMeta {
    status: String,
    reason: Option<String>,
    owner: Option<Pubkey>,
    executable: Option<bool>,
    data_len: Option<usize>,
    fetched_slot: Option<u64>,
}

#[derive(Clone, Debug)]
pub struct MixDropReport {
    pub amm_keys: Vec<Pubkey>,
    pub invalid_pools: Vec<Pubkey>,
    pub unverified_pools: Vec<Pubkey>,
    pub invalid_accounts: Vec<Pubkey>,
    pub unverified_accounts: Vec<Pubkey>,
}

#[derive(Serialize, Deserialize)]
struct Mix2File {
    schema_version: u32,
    source_mix_path: String,
    source_mix_hash: String,
    created_at_unix: u64,
    rpc_url: String,
    pools_total: usize,
    pools_valid: usize,
    pools_invalid: usize,
    pools_unverified: usize,
    accounts_total: usize,
    accounts_valid: usize,
    accounts_invalid: usize,
    accounts_unverified: usize,
    pools: Vec<Mix2Pool>,
}

#[derive(Serialize, Deserialize, Clone)]
struct Mix2Pool {
    pubkey: String,
    owner: Option<String>,
    dex: Option<String>,
    status: String,
    reason: Option<String>,
    params: serde_json::Value,
    required_accounts: Vec<Mix2Account>,
    alt_accounts: Vec<Mix2AltAccount>,
    original_mix_entry: serde_json::Value,
}

#[derive(Serialize, Deserialize, Clone)]
struct Mix2Account {
    pubkey: String,
    source: String,
    role: String,
    importance: MixAccountImportance,
    status: String,
    reason: Option<String>,
    owner: Option<String>,
    executable: Option<bool>,
    data_len: Option<usize>,
    fetched_slot: Option<u64>,
    is_variable: bool,
}

#[derive(Serialize, Deserialize, Clone)]
struct Mix2AltAccount {
    pubkey: String,
    status: String,
    loaded_addresses: Vec<String>,
}

#[derive(Serialize)]
struct InvalidPoolReport {
    pool: String,
    dex: Option<String>,
    owner: Option<String>,
    status: String,
    reason: String,
    missing_accounts: Vec<InvalidAccountReport>,
    original_mix_entry: serde_json::Value,
}

#[derive(Serialize, Clone)]
struct InvalidAccountReport {
    pubkey: String,
    role: String,
    source: String,
    importance: MixAccountImportance,
    status: String,
    rpc_error: String,
    action: String,
    reason: String,
    pools: Vec<String>,
}

impl MixDropReport {
    pub fn is_unverified_only(&self) -> bool {
        self.invalid_pools.is_empty() && self.invalid_accounts.is_empty()
    }

    pub fn reason(&self) -> &'static str {
        if self.is_unverified_only() {
            "unverified_mix_account_or_pool"
        } else {
            "invalid_mix_account_or_pool"
        }
    }
}

pub fn maybe_mix_path(dex_dir: &str) -> Option<PathBuf> {
    let dex_path = Path::new(dex_dir);
    if dex_path.is_file() && dex_path.file_name().and_then(|n| n.to_str()) == Some("mix.json") {
        return Some(dex_path.to_path_buf());
    }

    let candidate = dex_path.join("mix.json");
    candidate.exists().then_some(candidate)
}

impl VerifiedMixRegistry {
    pub async fn load_and_verify(
        rpc: Arc<RpcClient>,
        dex_dir: &str,
        groups_per_second: u64,
        rpc_url: &str,
    ) -> Result<Option<Arc<Self>>> {
        let Some(mix_path) = maybe_mix_path(dex_dir) else {
            return Ok(None);
        };

        let source_content = std::fs::read_to_string(&mix_path)
            .with_context(|| format!("cannot read {}", mix_path.display()))?;
        let source_mix_hash = hashv(&[source_content.as_bytes()]).to_string();
        let output_root = mix_output_root(&mix_path);
        let mix2_path = output_root.join("mix2.json");

        if let Some(registry) = Self::load_cached_mix2(
            &mix_path,
            &output_root,
            &source_mix_hash,
            rpc_url,
            &mix2_path,
        )? {
            registry.log_ready("cache");
            registry.log_summary();
            return Ok(Some(registry));
        }

        let pools = parse_mix_pools_from_str(&source_content, &mix_path)?;
        let registry = Arc::new(Self {
            mix_path,
            output_root,
            source_mix_hash,
            rpc_url: rpc_url.to_string(),
            inner: Arc::new(RwLock::new(build_inner_from_pools(pools))),
        });

        registry.verify_startup(rpc, groups_per_second).await;
        registry.recompute_pools();
        registry.write_outputs("rebuilt")?;
        registry.log_ready("rebuilt");
        registry.log_summary();
        Ok(Some(registry))
    }

    fn load_cached_mix2(
        mix_path: &Path,
        output_root: &Path,
        source_mix_hash: &str,
        rpc_url: &str,
        mix2_path: &Path,
    ) -> Result<Option<Arc<Self>>> {
        if !mix2_path.exists() {
            eprintln!(
                "[mix2_cache] status=missing path={} action=rebuild",
                mix2_path.display()
            );
            return Ok(None);
        }

        let content = match std::fs::read_to_string(mix2_path) {
            Ok(content) => content,
            Err(e) => {
                eprintln!(
                    "[mix2_cache] status=read_error path={} error={} action=rebuild",
                    mix2_path.display(),
                    e
                );
                return Ok(None);
            }
        };
        let cached: Mix2File = match serde_json::from_str(&content) {
            Ok(file) => file,
            Err(e) => {
                eprintln!(
                    "[mix2_cache] status=parse_error path={} error={} action=rebuild",
                    mix2_path.display(),
                    e
                );
                return Ok(None);
            }
        };

        if cached.schema_version != MIX2_SCHEMA_VERSION {
            eprintln!(
                "[mix2_cache] status=stale_schema path={} cached_schema={} expected_schema={} action=rebuild",
                mix2_path.display(),
                cached.schema_version,
                MIX2_SCHEMA_VERSION
            );
            return Ok(None);
        }
        if cached.source_mix_hash != source_mix_hash {
            eprintln!(
                "[mix2_cache] status=stale_source path={} cached_hash={} current_hash={} action=rebuild",
                mix2_path.display(),
                cached.source_mix_hash,
                source_mix_hash
            );
            return Ok(None);
        }

        let inner = build_inner_from_mix2(&cached);
        let registry = Arc::new(Self {
            mix_path: mix_path.to_path_buf(),
            output_root: output_root.to_path_buf(),
            source_mix_hash: source_mix_hash.to_string(),
            rpc_url: if cached.rpc_url.is_empty() {
                rpc_url.to_string()
            } else {
                cached.rpc_url
            },
            inner: Arc::new(RwLock::new(inner)),
        });
        registry.recompute_pools();
        eprintln!(
            "[mix2_cache] status=hit path={} source_hash={} rpc_verify_skipped=true",
            mix2_path.display(),
            source_mix_hash
        );
        Ok(Some(registry))
    }

    pub fn spawn_unverified_retry_task(self: &Arc<Self>, rpc: Arc<RpcClient>, groups_per_second: u64) {
        let registry = self.clone();
        let rate = capped_rpc_rps(groups_per_second);
        let delay = Duration::from_millis((1000 / rate).max(1));
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(30));
            interval.tick().await;

            loop {
                interval.tick().await;
                let accounts = registry.unverified_accounts();
                if accounts.is_empty() {
                    continue;
                }

                let mut changed = 0usize;
                let mut requests = 0usize;
                for chunk in accounts.chunks(RPC_CHUNK_SIZE) {
                    log_rpc_fetch_reason("mix_verify", chunk);
                    let chunk_keys = chunk.to_vec();
                    let rpc = rpc.clone();
                    requests += 1;
                    let result =
                        tokio::task::spawn_blocking(move || rpc.get_multiple_accounts(&chunk_keys))
                            .await;

                    match result {
                        Ok(Ok(accounts)) => {
                            let mut inner = registry.inner.write().unwrap();
                            for (pk, acct_opt) in chunk.iter().zip(accounts.into_iter()) {
                                match acct_opt {
                                    Some(acct) => {
                                        set_account_found(&mut inner, *pk, acct, None);
                                    }
                                    None => {
                                        set_account_not_found(&mut inner, *pk);
                                    }
                                }
                                changed += 1;
                            }
                        }
                        Ok(Err(e)) => {
                            eprintln!(
                                "[mix_verify_retry] accounts={} error_kind={} error={}",
                                chunk.len(),
                                classify_rpc_error(&e.to_string()),
                                e
                            );
                        }
                        Err(e) => {
                            eprintln!(
                                "[mix_verify_retry] accounts={} error_kind=task error={:?}",
                                chunk.len(),
                                e
                            );
                        }
                    }

                    tokio::time::sleep(delay).await;
                }

                if changed > 0 {
                    registry.recompute_pools();
                    registry.write_outputs_snapshot("retry");
                    eprintln!(
                        "[mix_verify_retry] changed={} requests={} remaining_unverified={}",
                        changed,
                        requests,
                        registry.unverified_account_count()
                    );
                }
            }
        });
    }

    async fn verify_startup(&self, rpc: Arc<RpcClient>, groups_per_second: u64) {
        let rate = capped_rpc_rps(groups_per_second);
        let delay = Duration::from_millis((1000 / rate).max(1));
        let accounts = self.all_mix_accounts();
        let total_accounts = accounts.len();
        let total_chunks = total_accounts.div_ceil(RPC_CHUNK_SIZE);
        let mut requests = 0usize;
        eprintln!(
            "[mix_verify] start mix_path={} pools={} accounts={} chunk_size={} rpc_rate={}/sec",
            self.mix_path.display(),
            self.pool_count(),
            total_accounts,
            RPC_CHUNK_SIZE,
            rate
        );

        let fetched_slot_hint = {
            let rpc = rpc.clone();
            tokio::task::spawn_blocking(move || rpc.get_slot())
                .await
                .ok()
                .and_then(|slot_result| slot_result.ok())
        };

        for (chunk_idx, chunk) in accounts.chunks(RPC_CHUNK_SIZE).enumerate() {
            self.verify_chunk(rpc.clone(), chunk, fetched_slot_hint).await;
            requests += 1;

            if (chunk_idx + 1) % 10 == 0 || chunk_idx + 1 == total_chunks {
                eprintln!(
                    "[mix_verify] progress chunks={}/{} accounts_valid={} accounts_invalid={} accounts_unverified={}",
                    chunk_idx + 1,
                    total_chunks,
                    self.valid_account_count(),
                    self.invalid_account_count(),
                    self.unverified_account_count()
                );
            }

            tokio::time::sleep(delay).await;
        }

        eprintln!(
            "[rpc_fetch_summary] mix2_build=1 mix_verify={} alt_account=0 mint_account=0 tx_static_unknown=0 rpc_compare=0 blockhash=0 block_time=0",
            requests
        );
    }

    async fn verify_chunk(
        &self,
        rpc: Arc<RpcClient>,
        chunk: &[Pubkey],
        fetched_slot_hint: Option<u64>,
    ) {
        if chunk.is_empty() {
            return;
        }

        const MAX_ATTEMPTS: usize = 3;
        let chunk_keys = chunk.to_vec();
        for attempt in 1..=MAX_ATTEMPTS {
            log_rpc_fetch_reason("mix_verify", &chunk_keys);
            let rpc = rpc.clone();
            let request_keys = chunk_keys.clone();
            let result =
                tokio::task::spawn_blocking(move || rpc.get_multiple_accounts(&request_keys)).await;

            match result {
                Ok(Ok(accounts)) => {
                    let mut inner = self.inner.write().unwrap();
                    for (pk, acct_opt) in chunk_keys.iter().zip(accounts.into_iter()) {
                        match acct_opt {
                            Some(acct) => set_account_found(&mut inner, *pk, acct, fetched_slot_hint),
                            None => set_account_not_found(&mut inner, *pk),
                        }
                    }
                    return;
                }
                Ok(Err(e)) => {
                    let message = e.to_string();
                    let kind = classify_rpc_error(&message);
                    eprintln!(
                        "[mix_verify_retry] accounts={} attempt={} error_kind={} error={}",
                        chunk_keys.len(),
                        attempt,
                        kind,
                        message
                    );
                    if attempt == MAX_ATTEMPTS {
                        let mut inner = self.inner.write().unwrap();
                        for pk in &chunk_keys {
                            set_account_unverified(&mut inner, *pk, kind);
                        }
                    } else {
                        tokio::time::sleep(Duration::from_millis(
                            250_u64.saturating_mul(1_u64 << (attempt - 1).min(4)),
                        ))
                        .await;
                    }
                }
                Err(e) => {
                    eprintln!(
                        "[mix_verify_retry] accounts={} attempt={} error_kind=task error={:?}",
                        chunk_keys.len(),
                        attempt,
                        e
                    );
                    if attempt == MAX_ATTEMPTS {
                        let mut inner = self.inner.write().unwrap();
                        for pk in &chunk_keys {
                            set_account_unverified(&mut inner, *pk, "task");
                        }
                    }
                }
            }
        }
    }

    fn recompute_pools(&self) {
        let mut inner = self.inner.write().unwrap();
        inner.valid_pools.clear();
        inner.invalid_pools.clear();
        inner.unverified_pools.clear();

        for pool in inner.pools.clone() {
            if pool.required_accounts.iter().any(|account| {
                account.importance.can_block_pool()
                    && inner.invalid_accounts.contains(&account.pubkey)
            }) {
                inner.invalid_pools.insert(pool.pool);
            } else if pool.required_accounts.iter().any(|account| {
                account.importance.can_block_pool()
                    && inner.unverified_accounts.contains(&account.pubkey)
            }) {
                inner.unverified_pools.insert(pool.pool);
            } else {
                inner.valid_pools.insert(pool.pool);
            }
        }
    }

    pub fn filter_valid_accounts(&self, accounts: &mut Vec<Pubkey>) {
        accounts.retain(|account| self.is_valid_account(account));
        accounts.sort_unstable();
        accounts.dedup();
    }

    pub fn filter_valid_groups(&self, groups: &[Vec<Pubkey>]) -> Vec<Vec<Pubkey>> {
        groups
            .iter()
            .filter_map(|group| {
                let mut out = group.clone();
                self.filter_valid_accounts(&mut out);
                (!out.is_empty()).then_some(out)
            })
            .collect()
    }

    pub fn valid_variable_accounts(&self) -> Vec<Pubkey> {
        let inner = self.inner.read().unwrap();
        let mut out = inner
            .pools
            .iter()
            .flat_map(|pool| pool.required_accounts.iter())
            .filter(|account| account.is_variable)
            .map(|account| account.pubkey)
            .filter(|account| inner.valid_accounts.contains(account))
            .collect::<Vec<_>>();
        out.sort_unstable();
        out.dedup();
        out
    }

    pub fn alt_accounts(&self) -> Vec<Pubkey> {
        let inner = self.inner.read().unwrap();
        let mut out = inner
            .pools
            .iter()
            .flat_map(|pool| pool.required_accounts.iter())
            .filter(|account| account.source == "alt")
            .map(|account| account.pubkey)
            .collect::<Vec<_>>();
        out.sort_unstable();
        out.dedup();
        out
    }

    pub fn output_root(&self) -> PathBuf {
        self.output_root.clone()
    }

    pub fn is_valid_account(&self, account: &Pubkey) -> bool {
        self.inner.read().unwrap().valid_accounts.contains(account)
    }

    pub fn check_route_plan(&self, route_plan: &serde_json::Value) -> Option<MixDropReport> {
        let amm_keys = route_plan_amm_keys(route_plan);
        if amm_keys.is_empty() {
            return None;
        }

        let inner = self.inner.read().unwrap();
        let invalid_pools = amm_keys
            .iter()
            .copied()
            .filter(|pool| inner.invalid_pools.contains(pool))
            .collect::<Vec<_>>();
        let unverified_pools = amm_keys
            .iter()
            .copied()
            .filter(|pool| inner.unverified_pools.contains(pool))
            .collect::<Vec<_>>();

        if invalid_pools.is_empty() && unverified_pools.is_empty() {
            return None;
        }

        Some(MixDropReport {
            amm_keys,
            invalid_pools,
            unverified_pools,
            invalid_accounts: Vec::new(),
            unverified_accounts: Vec::new(),
        })
    }

    pub fn check_tx_accounts(&self, accounts: &[Pubkey]) -> Option<MixDropReport> {
        let inner = self.inner.read().unwrap();
        let invalid_accounts = accounts
            .iter()
            .copied()
            .filter(|account| {
                inner.invalid_accounts.contains(account)
                    && !is_data_hash_only_reason(&inner, account)
                    && inner
                        .account_importance
                        .get(account)
                        .copied()
                        .unwrap_or(MixAccountImportance::RequiredLive)
                        .can_drop_route()
            })
            .collect::<Vec<_>>();
        let unverified_accounts = accounts
            .iter()
            .copied()
            .filter(|account| {
                inner.unverified_accounts.contains(account)
                    && !is_data_hash_only_reason(&inner, account)
                    && inner
                        .account_importance
                        .get(account)
                        .copied()
                        .unwrap_or(MixAccountImportance::RequiredLive)
                        .can_drop_route()
            })
            .collect::<Vec<_>>();

        if invalid_accounts.is_empty() && unverified_accounts.is_empty() {
            return None;
        }

        Some(MixDropReport {
            amm_keys: Vec::new(),
            invalid_pools: Vec::new(),
            unverified_pools: Vec::new(),
            invalid_accounts,
            unverified_accounts,
        })
    }

    pub fn data_hash_only_tx_accounts(&self, accounts: &[Pubkey]) -> Vec<Pubkey> {
        let inner = self.inner.read().unwrap();
        accounts
            .iter()
            .copied()
            .filter(|account| {
                (inner.invalid_accounts.contains(account)
                    || inner.unverified_accounts.contains(account))
                    && is_data_hash_only_reason(&inner, account)
            })
            .collect()
    }

    pub fn account_count(&self) -> usize {
        self.inner.read().unwrap().account_to_pools.len()
    }

    pub fn valid_account_count(&self) -> usize {
        self.inner.read().unwrap().valid_accounts.len()
    }

    pub fn invalid_account_count(&self) -> usize {
        self.inner.read().unwrap().invalid_accounts.len()
    }

    pub fn unverified_account_count(&self) -> usize {
        self.inner.read().unwrap().unverified_accounts.len()
    }

    pub fn pool_count(&self) -> usize {
        self.inner.read().unwrap().pool_to_accounts.len()
    }

    pub fn valid_pool_count(&self) -> usize {
        self.inner.read().unwrap().valid_pools.len()
    }

    pub fn invalid_pool_count(&self) -> usize {
        self.inner.read().unwrap().invalid_pools.len()
    }

    pub fn unverified_pool_count(&self) -> usize {
        self.inner.read().unwrap().unverified_pools.len()
    }

    fn all_mix_accounts(&self) -> Vec<Pubkey> {
        let mut out = self
            .inner
            .read()
            .unwrap()
            .account_importance
            .iter()
            .filter_map(|(pubkey, importance)| {
                (*importance != MixAccountImportance::ProgramOrSysvar).then_some(*pubkey)
            })
            .collect::<Vec<_>>();
        out.sort_unstable();
        out.dedup();
        out
    }

    fn unverified_accounts(&self) -> Vec<Pubkey> {
        let mut out = self
            .inner
            .read()
            .unwrap()
            .unverified_accounts
            .iter()
            .copied()
            .collect::<Vec<_>>();
        out.sort_unstable();
        out
    }

    fn write_outputs_snapshot(&self, source: &str) {
        if let Err(e) = self.write_outputs(source) {
            eprintln!(
                "[mix2_write_error] output_root={} source={} error={}",
                self.output_root.display(),
                source,
                e
            );
        }
    }

    fn write_outputs(&self, source: &str) -> Result<()> {
        std::fs::create_dir_all(&self.output_root)
            .with_context(|| format!("cannot create {}", self.output_root.display()))?;

        let mix2 = self.build_mix2_file();
        let invalid_pools = self.build_invalid_pool_reports();
        let invalid_accounts = self.build_invalid_account_reports();
        let report = self.build_text_report(&invalid_pools, &invalid_accounts);

        let mix2_path = self.output_root.join("mix2.json");
        let invalid_pools_path = self.output_root.join("invalid_pools.json");
        let invalid_accounts_path = self.output_root.join("invalid_accounts.json");
        let report_path = self.output_root.join("mix_verify_report.txt");

        std::fs::write(&mix2_path, serde_json::to_vec_pretty(&mix2)?)
            .with_context(|| format!("cannot write {}", mix2_path.display()))?;
        std::fs::write(&invalid_pools_path, serde_json::to_vec_pretty(&invalid_pools)?)
            .with_context(|| format!("cannot write {}", invalid_pools_path.display()))?;
        std::fs::write(&invalid_accounts_path, serde_json::to_vec_pretty(&invalid_accounts)?)
            .with_context(|| format!("cannot write {}", invalid_accounts_path.display()))?;
        std::fs::write(&report_path, report)
            .with_context(|| format!("cannot write {}", report_path.display()))?;

        eprintln!(
            "[mix2_write] source={} mix2={} invalid_pools={} invalid_accounts={} report={}",
            source,
            mix2_path.display(),
            invalid_pools_path.display(),
            invalid_accounts_path.display(),
            report_path.display()
        );
        Ok(())
    }

    fn build_mix2_file(&self) -> Mix2File {
        let inner = self.inner.read().unwrap();
        let pools = inner
            .pools
            .iter()
            .map(|pool| {
                let status = pool_status(&inner, &pool.pool).to_string();
                let reason = pool_reason(&status);
                let required_accounts = pool
                    .required_accounts
                    .iter()
                    .map(|account_ref| {
                        let meta = inner
                            .account_meta
                            .get(&account_ref.pubkey)
                            .cloned()
                            .unwrap_or_else(|| VerifiedAccountMeta {
                                status: "unverified".to_string(),
                                reason: Some("not_checked".to_string()),
                                ..Default::default()
                            });
                        Mix2Account {
                            pubkey: account_ref.pubkey.to_string(),
                            source: account_ref.source.clone(),
                            role: account_ref.role.clone(),
                            importance: account_ref.importance,
                            status: meta.status,
                            reason: meta.reason,
                            owner: meta.owner.map(|owner| owner.to_string()),
                            executable: meta.executable,
                            data_len: meta.data_len,
                            fetched_slot: meta.fetched_slot,
                            is_variable: account_ref.is_variable,
                        }
                    })
                    .collect::<Vec<_>>();

                let alt_accounts = required_accounts
                    .iter()
                    .filter(|account| account.source == "alt" || account.role.contains("alt"))
                    .map(|account| Mix2AltAccount {
                        pubkey: account.pubkey.clone(),
                        status: account.status.clone(),
                        loaded_addresses: Vec::new(),
                    })
                    .collect::<Vec<_>>();

                Mix2Pool {
                    pubkey: pool.pool.to_string(),
                    owner: pool.owner.map(|owner| owner.to_string()),
                    dex: pool.dex.clone(),
                    status,
                    reason,
                    params: pool.params.clone(),
                    required_accounts,
                    alt_accounts,
                    original_mix_entry: pool.original_mix_entry.clone(),
                }
            })
            .collect::<Vec<_>>();

        Mix2File {
            schema_version: MIX2_SCHEMA_VERSION,
            source_mix_path: self.mix_path.display().to_string(),
            source_mix_hash: self.source_mix_hash.clone(),
            created_at_unix: unix_now(),
            rpc_url: self.rpc_url.clone(),
            pools_total: inner.pool_to_accounts.len(),
            pools_valid: inner.valid_pools.len(),
            pools_invalid: inner.invalid_pools.len(),
            pools_unverified: inner.unverified_pools.len(),
            accounts_total: inner.account_to_pools.len(),
            accounts_valid: inner.valid_accounts.len(),
            accounts_invalid: inner.invalid_accounts.len(),
            accounts_unverified: inner.unverified_accounts.len(),
            pools,
        }
    }

    fn build_invalid_pool_reports(&self) -> Vec<InvalidPoolReport> {
        let inner = self.inner.read().unwrap();
        inner
            .pools
            .iter()
            .filter(|pool| {
                inner.invalid_pools.contains(&pool.pool)
                    || inner.unverified_pools.contains(&pool.pool)
            })
            .map(|pool| {
                let status = pool_status(&inner, &pool.pool).to_string();
                let missing_accounts = pool
                    .required_accounts
                    .iter()
                    .filter(|account_ref| {
                        account_ref.importance.can_block_pool()
                            && (inner.invalid_accounts.contains(&account_ref.pubkey)
                                || inner.unverified_accounts.contains(&account_ref.pubkey))
                    })
                    .map(|account_ref| invalid_account_report(&inner, account_ref))
                    .collect::<Vec<_>>();
                InvalidPoolReport {
                    pool: pool.pool.to_string(),
                    dex: pool.dex.clone(),
                    owner: pool.owner.map(|owner| owner.to_string()),
                    status: status.clone(),
                    reason: pool_reason(&status)
                        .unwrap_or_else(|| "missing_required_account".to_string()),
                    missing_accounts,
                    original_mix_entry: pool.original_mix_entry.clone(),
                }
            })
            .collect()
    }

    fn build_invalid_account_reports(&self) -> Vec<InvalidAccountReport> {
        let inner = self.inner.read().unwrap();
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for pool in &inner.pools {
            for account_ref in &pool.required_accounts {
                let status = inner
                    .account_meta
                    .get(&account_ref.pubkey)
                    .map(|meta| meta.status.as_str())
                    .unwrap_or("not_checked");
                if status == "valid"
                    || status == "program_or_sysvar"
                    || status == "not_checked"
                {
                    continue;
                }
                if seen.insert(account_ref.pubkey) {
                    out.push(invalid_account_report(&inner, account_ref));
                }
            }
        }
        out
    }

    fn build_text_report(
        &self,
        invalid_pools: &[InvalidPoolReport],
        invalid_accounts: &[InvalidAccountReport],
    ) -> String {
        let inner = self.inner.read().unwrap();
        let mut out = String::new();
        out.push_str("MIX VERIFY REPORT\n");
        out.push_str(&format!("source={}\n", self.mix_path.display()));
        out.push_str(&format!("output={}\n\n", self.output_root.join("mix2.json").display()));
        out.push_str(&format!("source_mix_hash={}\n", self.source_mix_hash));
        out.push_str(&format!("schema_version={}\n\n", MIX2_SCHEMA_VERSION));
        out.push_str(&format!("pools_total={}\n", inner.pool_to_accounts.len()));
        out.push_str(&format!("pools_valid={}\n", inner.valid_pools.len()));
        out.push_str(&format!("pools_invalid={}\n", inner.invalid_pools.len()));
        out.push_str(&format!("pools_unverified={}\n", inner.unverified_pools.len()));
        out.push_str(&format!("accounts_total={}\n", inner.account_to_pools.len()));
        out.push_str(&format!("accounts_valid={}\n", inner.valid_accounts.len()));
        out.push_str(&format!("accounts_invalid={}\n", inner.invalid_accounts.len()));
        out.push_str(&format!(
            "accounts_unverified={}\n\n",
            inner.unverified_accounts.len()
        ));

        out.push_str("INVALID OR UNVERIFIED POOLS:\n");
        for pool in invalid_pools.iter().take(500) {
            out.push_str(&format!(
                "pool={} dex={} owner={} status={} reason={}\n",
                pool.pool,
                pool.dex.clone().unwrap_or_default(),
                pool.owner.clone().unwrap_or_default(),
                pool.status,
                pool.reason
            ));
            for account in pool.missing_accounts.iter().take(20) {
                out.push_str(&format!(
                    "  account={} role={} source={} importance={} rpc_status={} action={} reason={}\n",
                    account.pubkey,
                    account.role,
                    account.source,
                    account.importance.as_str(),
                    account.status,
                    account.action,
                    account.reason
                ));
            }
        }

        out.push_str("\nINVALID OR UNVERIFIED ACCOUNTS:\n");
        for account in invalid_accounts.iter().take(1000) {
            out.push_str(&format!(
                "account={} role={} source={} importance={} rpc_status={} action={} reason={} pools={:?}\n",
                account.pubkey,
                account.role,
                account.source,
                account.importance.as_str(),
                account.status,
                account.action,
                account.reason,
                account.pools
            ));
        }
        out
    }

    fn log_ready(&self, source: &str) {
        eprintln!(
            "[mix2_ready] true source={} path={} source_hash={} pools_valid={} pools_invalid={} pools_unverified={} accounts_valid={} accounts_invalid={} accounts_unverified={}",
            source,
            self.output_root.join("mix2.json").display(),
            self.source_mix_hash,
            self.valid_pool_count(),
            self.invalid_pool_count(),
            self.unverified_pool_count(),
            self.valid_account_count(),
            self.invalid_account_count(),
            self.unverified_account_count()
        );
    }

    fn log_summary(&self) {
        eprintln!(
            "[mix_verify_summary] mix_path={} pools_total={} accounts_total={} accounts_valid={} accounts_invalid_not_found={} accounts_unverified_rpc_error={} pools_valid={} pools_invalid={} pools_unverified={}",
            self.mix_path.display(),
            self.pool_count(),
            self.account_count(),
            self.valid_account_count(),
            self.invalid_account_count(),
            self.unverified_account_count(),
            self.valid_pool_count(),
            self.invalid_pool_count(),
            self.unverified_pool_count()
        );

        let inner = self.inner.read().unwrap();
        for account in inner.invalid_accounts.iter().take(20) {
            let reason = inner
                .account_reasons
                .get(account)
                .map(String::as_str)
                .unwrap_or("not_found");
            let pools = inner
                .account_to_pools
                .get(account)
                .cloned()
                .unwrap_or_default();
            eprintln!(
                "[mix_invalid_account] pubkey={} reason={} pools={}",
                account,
                reason,
                pubkeys_json(&pools)
            );
        }
        for account in inner.unverified_accounts.iter().take(20) {
            let reason = inner
                .account_reasons
                .get(account)
                .map(String::as_str)
                .unwrap_or("rpc_error");
            let pools = inner
                .account_to_pools
                .get(account)
                .cloned()
                .unwrap_or_default();
            eprintln!(
                "[mix_unverified_account] pubkey={} reason={} pools={}",
                account,
                reason,
                pubkeys_json(&pools)
            );
        }
        for pool in inner
            .invalid_pools
            .iter()
            .chain(inner.unverified_pools.iter())
            .take(20)
        {
            let accounts = inner.pool_to_accounts.get(pool).cloned().unwrap_or_default();
            let missing = accounts
                .iter()
                .copied()
                .filter(|account| {
                    inner.invalid_accounts.contains(account)
                        || inner.unverified_accounts.contains(account)
                })
                .collect::<Vec<_>>();
            let dex = inner
                .pool_owner
                .get(pool)
                .map(ToString::to_string)
                .unwrap_or_default();
            eprintln!(
                "[mix_invalid_pool] pool={} reason=missing_static_account missing_accounts={} dex={}",
                pool,
                pubkeys_json(&missing),
                dex
            );
        }
    }
}

fn is_data_hash_only_reason(inner: &MixRegistryInner, account: &Pubkey) -> bool {
    matches!(
        inner
        .account_reasons
        .get(account)
        .or_else(|| {
            inner
                .account_meta
                .get(account)
                .and_then(|meta| meta.reason.as_ref())
        })
        .map(String::as_str),
        Some("data_hash_mismatch" | "data_hash_only_mismatch" | "stale_data_hash_mismatch")
    )
}

fn build_inner_from_pools(pools: Vec<MixPool>) -> MixRegistryInner {
    let mut inner = MixRegistryInner {
        pools,
        ..Default::default()
    };

    for pool in &inner.pools {
        let accounts = unique_pubkeys(pool.required_accounts.iter().map(|a| a.pubkey));
        inner.pool_to_accounts.insert(pool.pool, accounts.clone());
        if let Some(owner) = pool.owner {
            inner.pool_owner.insert(pool.pool, owner);
        }
        for account_ref in &pool.required_accounts {
            let entry = inner
                .account_importance
                .entry(account_ref.pubkey)
                .or_insert(account_ref.importance);
            *entry = strongest_importance(*entry, account_ref.importance);
        }
        for account in &accounts {
            inner
                .account_to_pools
                .entry(*account)
                .or_default()
                .push(pool.pool);
        }
    }
    normalize_account_to_pools(&mut inner);
    inner
}

fn build_inner_from_mix2(cached: &Mix2File) -> MixRegistryInner {
    let mut pools = Vec::new();
    for cached_pool in &cached.pools {
        let Some(pool) = Pubkey::try_from(cached_pool.pubkey.as_str()).ok() else {
            continue;
        };
        let owner = cached_pool
            .owner
            .as_deref()
            .and_then(|owner| Pubkey::try_from(owner).ok());
        let required_accounts = cached_pool
            .required_accounts
            .iter()
            .filter_map(|account| {
                let pubkey = Pubkey::try_from(account.pubkey.as_str()).ok()?;
                Some(MixAccountRef {
                    pubkey,
                    source: account.source.clone(),
                    role: account.role.clone(),
                    is_variable: account.is_variable,
                    importance: account.importance,
                })
            })
            .collect::<Vec<_>>();
        pools.push(MixPool {
            pool,
            owner,
            dex: cached_pool.dex.clone(),
            params: cached_pool.params.clone(),
            original_mix_entry: cached_pool.original_mix_entry.clone(),
            required_accounts,
        });
    }

    let mut inner = build_inner_from_pools(pools);
    for cached_pool in &cached.pools {
        for account in &cached_pool.required_accounts {
            let Some(pubkey) = Pubkey::try_from(account.pubkey.as_str()).ok() else {
                continue;
            };
            let meta = VerifiedAccountMeta {
                status: account.status.clone(),
                reason: account.reason.clone(),
                owner: account
                    .owner
                    .as_deref()
                    .and_then(|owner| Pubkey::try_from(owner).ok()),
                executable: account.executable,
                data_len: account.data_len,
                fetched_slot: account.fetched_slot,
            };
            apply_cached_status(&mut inner, pubkey, meta);
        }
    }
    inner
}

fn normalize_account_to_pools(inner: &mut MixRegistryInner) {
    for pools in inner.account_to_pools.values_mut() {
        pools.sort_unstable();
        pools.dedup();
    }
}

fn set_account_found(
    inner: &mut MixRegistryInner,
    pk: Pubkey,
    acct: solana_sdk::account::Account,
    fetched_slot: Option<u64>,
) {
    inner.invalid_accounts.remove(&pk);
    inner.unverified_accounts.remove(&pk);
    inner.valid_accounts.insert(pk);
    inner.account_reasons.remove(&pk);
    inner.account_meta.insert(
        pk,
        VerifiedAccountMeta {
            status: "valid".to_string(),
            reason: None,
            owner: Some(acct.owner),
            executable: Some(acct.executable),
            data_len: Some(acct.data.len()),
            fetched_slot,
        },
    );
}

fn set_account_not_found(inner: &mut MixRegistryInner, pk: Pubkey) {
    let importance = inner
        .account_importance
        .get(&pk)
        .copied()
        .unwrap_or(MixAccountImportance::RequiredLive);
    let (status, route_blocking) = match importance {
        MixAccountImportance::RequiredStatic | MixAccountImportance::RequiredLive => {
            ("invalid", true)
        }
        MixAccountImportance::TxOnly => ("tx_only_missing", true),
        MixAccountImportance::OptionalStatic => ("optional_missing", false),
        MixAccountImportance::ProgramOrSysvar => ("program_or_sysvar", false),
    };

    inner.valid_accounts.remove(&pk);
    inner.unverified_accounts.remove(&pk);
    if route_blocking {
        inner.invalid_accounts.insert(pk);
    } else {
        inner.invalid_accounts.remove(&pk);
    }
    inner
        .account_reasons
        .insert(pk, "not_found".to_string());
    inner.account_meta.insert(
        pk,
        VerifiedAccountMeta {
            status: status.to_string(),
            reason: Some("not_found".to_string()),
            ..Default::default()
        },
    );
}

fn set_account_unverified(inner: &mut MixRegistryInner, pk: Pubkey, reason: &str) {
    let importance = inner
        .account_importance
        .get(&pk)
        .copied()
        .unwrap_or(MixAccountImportance::RequiredLive);
    let (status, route_blocking) = match importance {
        MixAccountImportance::RequiredStatic | MixAccountImportance::RequiredLive => {
            ("unverified", true)
        }
        MixAccountImportance::TxOnly => ("tx_only_unverified", true),
        MixAccountImportance::OptionalStatic => ("optional_unverified", false),
        MixAccountImportance::ProgramOrSysvar => ("program_or_sysvar", false),
    };

    inner.valid_accounts.remove(&pk);
    inner.invalid_accounts.remove(&pk);
    if route_blocking {
        inner.unverified_accounts.insert(pk);
    } else {
        inner.unverified_accounts.remove(&pk);
    }
    inner.account_reasons.insert(pk, reason.to_string());
    inner.account_meta.insert(
        pk,
        VerifiedAccountMeta {
            status: status.to_string(),
            reason: Some(reason.to_string()),
            ..Default::default()
        },
    );
}

fn apply_cached_status(inner: &mut MixRegistryInner, pk: Pubkey, meta: VerifiedAccountMeta) {
    match meta.status.as_str() {
        "valid" => {
            inner.invalid_accounts.remove(&pk);
            inner.unverified_accounts.remove(&pk);
            inner.valid_accounts.insert(pk);
            inner.account_reasons.remove(&pk);
        }
        "invalid" | "tx_only_missing" => {
            inner.valid_accounts.remove(&pk);
            inner.unverified_accounts.remove(&pk);
            inner.invalid_accounts.insert(pk);
            inner.account_reasons.insert(
                pk,
                meta.reason
                    .clone()
                    .unwrap_or_else(|| "not_found".to_string()),
            );
        }
        "unverified" | "tx_only_unverified" => {
            inner.valid_accounts.remove(&pk);
            inner.invalid_accounts.remove(&pk);
            inner.unverified_accounts.insert(pk);
            inner.account_reasons.insert(
                pk,
                meta.reason
                    .clone()
                .unwrap_or_else(|| "unverified".to_string()),
            );
        }
        _ => {
            inner.valid_accounts.remove(&pk);
            inner.invalid_accounts.remove(&pk);
            inner.unverified_accounts.remove(&pk);
            if let Some(reason) = meta.reason.clone() {
                inner.account_reasons.insert(pk, reason);
            }
        }
    }
    inner.account_meta.insert(pk, meta);
}

fn parse_mix_pools_from_str(content: &str, path: &Path) -> Result<Vec<MixPool>> {
    let json: serde_json::Value =
        serde_json::from_str(content).with_context(|| format!("invalid {}", path.display()))?;

    let root = json
        .get("pools")
        .or_else(|| json.get("Pools"))
        .unwrap_or(&json);

    let mut pools = Vec::new();
    match root {
        serde_json::Value::Array(items) => {
            for item in items {
                if let Some(pool) = parse_pool_value(item) {
                    pools.push(pool);
                }
            }
        }
        serde_json::Value::Object(map) => {
            for value in map.values() {
                if let Some(pool) = parse_pool_value(value) {
                    pools.push(pool);
                }
            }
        }
        _ => {}
    }

    if pools.is_empty() {
        let mut refs = Vec::new();
        collect_pubkey_refs(root, &mut Vec::new(), &mut refs);
        dedup_account_refs(&mut refs);
        if let Some(pool) = refs.first().map(|account| account.pubkey) {
            if !refs.iter().any(|account| account.pubkey == pool) {
                refs.push(MixAccountRef {
                    pubkey: pool,
                    source: "pool_pubkey".to_string(),
                    role: "pool".to_string(),
                    is_variable: true,
                    importance: MixAccountImportance::RequiredLive,
                });
            }
            assign_importance_to_refs(&mut refs, None, None);
            pools.push(MixPool {
                pool,
                owner: None,
                dex: None,
                params: serde_json::Value::Null,
                original_mix_entry: root.clone(),
                required_accounts: refs,
            });
        }
    }

    Ok(pools)
}

fn parse_pool_value(value: &serde_json::Value) -> Option<MixPool> {
    let pool = value
        .get("pubkey")
        .or_else(|| value.get("pool"))
        .or_else(|| value.get("ammKey"))
        .and_then(|v| v.as_str())
        .and_then(|s| Pubkey::try_from(s).ok());

    let owner = value
        .get("owner")
        .and_then(|v| v.as_str())
        .and_then(|s| Pubkey::try_from(s).ok());

    let dex = value
        .get("dex")
        .or_else(|| value.get("label"))
        .or_else(|| value.get("name"))
        .or_else(|| value.get("amm"))
        .and_then(|v| v.as_str())
        .map(str::to_string)
        .or_else(|| owner.map(|owner| owner.to_string()));

    let mut required_accounts = Vec::new();
    collect_pubkey_refs(value, &mut Vec::new(), &mut required_accounts);
    dedup_account_refs(&mut required_accounts);
    assign_importance_to_refs(&mut required_accounts, owner, dex.as_deref());

    let pool = pool.or_else(|| required_accounts.first().map(|account| account.pubkey))?;
    if !required_accounts
        .iter()
        .any(|account| account.pubkey == pool && account.source == "pool_pubkey")
    {
        required_accounts.push(MixAccountRef {
            pubkey: pool,
            source: "pool_pubkey".to_string(),
            role: "pool".to_string(),
            is_variable: true,
            importance: MixAccountImportance::RequiredLive,
        });
    }

    Some(MixPool {
        pool,
        owner,
        dex,
        params: value
            .get("params")
            .cloned()
            .unwrap_or_else(|| serde_json::Value::Null),
        original_mix_entry: value.clone(),
        required_accounts,
    })
}

fn collect_pubkey_refs(
    value: &serde_json::Value,
    path: &mut Vec<String>,
    out: &mut Vec<MixAccountRef>,
) {
    match value {
        serde_json::Value::String(s) => {
            if let Ok(pk) = Pubkey::try_from(s.as_str()) {
                let (source, role, is_variable) = classify_account_path(path);
                out.push(MixAccountRef {
                    pubkey: pk,
                    source,
                    role,
                    is_variable,
                    importance: MixAccountImportance::OptionalStatic,
                });
            }
        }
        serde_json::Value::Array(items) => {
            for (idx, item) in items.iter().enumerate() {
                path.push(idx.to_string());
                collect_pubkey_refs(item, path, out);
                path.pop();
            }
        }
        serde_json::Value::Object(map) => {
            for (key, value) in map {
                path.push(key.clone());
                collect_pubkey_refs(value, path, out);
                path.pop();
            }
        }
        _ => {}
    }
}

fn classify_account_path(path: &[String]) -> (String, String, bool) {
    let role = path
        .last()
        .cloned()
        .unwrap_or_else(|| "param".to_string());
    let role_l = role.to_ascii_lowercase();
    let joined = path.join(".").to_ascii_lowercase();

    if matches!(role_l.as_str(), "pubkey" | "pool" | "ammkey") {
        return ("pool_pubkey".to_string(), role, true);
    }
    if role_l == "owner" || joined.contains("program") {
        return ("program_id".to_string(), role, false);
    }
    if joined.contains("addresslookuptable")
        || role_l.contains("alt")
        || joined.contains("lookup_table")
    {
        return ("alt".to_string(), role, false);
    }
    if joined.contains("tokenaccount")
        || joined.contains("vault")
        || joined.contains("reserve")
    {
        return ("vault".to_string(), role, true);
    }
    if joined.contains("mint") || joined.contains("tokenment") {
        return ("mint".to_string(), role, false);
    }
    // Tessera V: mintA/mintB (AccountA/AccountB renamed) — static mint, not live
    if role_l == "minta" || role_l == "mint_a" || role_l == "mintb" || role_l == "mint_b"
        || role_l == "accounta" || role_l == "account_a"
        || role_l == "accountb" || role_l == "account_b"
    {
        return ("mint".to_string(), role, false);
    }
    if joined.contains("authority") {
        return ("authority".to_string(), role, false);
    }
    // Manifest: global (live market state) + globalVault (live token vault)
    // SolFi V2: globalState + globalFeeAccount (live protocol accounts)
    if role_l == "global" || role_l == "globalstate" || role_l == "global_state" {
        return ("pool_pubkey".to_string(), role, true);
    }
    if role_l == "globalvault" || role_l == "global_vault"
        || role_l == "globalfeeaccount" || role_l == "global_fee_account"
    {
        return ("vault".to_string(), role, true);
    }
    if role_l == "vendorkey" || role_l == "vendor_key" {
        return ("vault".to_string(), role, true);
    }
    if role_l == "marketstate" || role_l == "market_state" {
        return ("pool_pubkey".to_string(), role, true);
    }
    if joined.contains("oracle")
        || joined.contains("observation")
        || joined.contains("tick")
        || joined.contains("bin")
        || joined.contains("market")
        || joined.contains("orderbook")
    {
        return ("param".to_string(), role, true);
    }

    ("param".to_string(), role, false)
}

fn assign_importance_to_refs(
    refs: &mut [MixAccountRef],
    owner: Option<Pubkey>,
    dex: Option<&str>,
) {
    for account in refs {
        account.importance = classify_importance(owner, dex, account);
    }
}

fn classify_importance(
    owner: Option<Pubkey>,
    dex: Option<&str>,
    account: &MixAccountRef,
) -> MixAccountImportance {
    let role = account.role.to_ascii_lowercase();
    let source = account.source.to_ascii_lowercase();
    let dex_l = dex.unwrap_or_default().to_ascii_lowercase();
    let owner_s = owner.map(|p| p.to_string()).unwrap_or_default();
    let is_manifest = dex_l.contains("manifest")
        || dex_l.contains("mnfst")
        || owner_s == "MNFSTqtC93rEfYHB6hF82sKdZpUDFWkViLByLd1k1Ms";
    let is_whirlpool = dex_l.contains("whirlpool")
        || dex_l.contains("orca")
        || owner_s == "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc";

    if source == "program_id" || role == "owner" || role.contains("program") {
        return MixAccountImportance::ProgramOrSysvar;
    }

    if source == "pool_pubkey" {
        return MixAccountImportance::RequiredLive;
    }

    if is_manifest && (role == "global" || role == "globalvault") {
        return MixAccountImportance::OptionalStatic;
    }

    if is_whirlpool
        && (role.contains("tick") || role.contains("oracle") || role.contains("observation"))
    {
        return MixAccountImportance::TxOnly;
    }

    if source == "vault" {
        return MixAccountImportance::RequiredLive;
    }
    if source == "mint" || source == "alt" {
        return MixAccountImportance::RequiredStatic;
    }
    if source == "authority" {
        return MixAccountImportance::OptionalStatic;
    }

    if role.contains("tick")
        || role.contains("oracle")
        || role.contains("observation")
        || role.contains("bin")
    {
        return MixAccountImportance::TxOnly;
    }

    if is_manifest && (role.contains("market") || role.contains("orderbook")) {
        return MixAccountImportance::RequiredLive;
    }

    if role.contains("market") || role.contains("orderbook") {
        return MixAccountImportance::TxOnly;
    }

    MixAccountImportance::OptionalStatic
}

fn strongest_importance(
    current: MixAccountImportance,
    next: MixAccountImportance,
) -> MixAccountImportance {
    if importance_rank(next) > importance_rank(current) {
        next
    } else {
        current
    }
}

fn importance_rank(importance: MixAccountImportance) -> u8 {
    match importance {
        MixAccountImportance::RequiredLive => 5,
        MixAccountImportance::RequiredStatic => 4,
        MixAccountImportance::TxOnly => 3,
        MixAccountImportance::OptionalStatic => 2,
        MixAccountImportance::ProgramOrSysvar => 1,
    }
}

fn action_for_status(importance: MixAccountImportance, status: &str) -> String {
    match (importance, status) {
        (_, "valid") => "use_account".to_string(),
        (MixAccountImportance::RequiredLive, "invalid")
        | (MixAccountImportance::RequiredStatic, "invalid") => "blacklist_pool".to_string(),
        (MixAccountImportance::RequiredLive, "unverified")
        | (MixAccountImportance::RequiredStatic, "unverified") => {
            "hold_pool_until_verified".to_string()
        }
        (MixAccountImportance::TxOnly, "tx_only_missing")
        | (MixAccountImportance::TxOnly, "tx_only_unverified") => {
            "drop_route_only_if_actual_tx_needs_it".to_string()
        }
        (MixAccountImportance::OptionalStatic, "optional_missing")
        | (MixAccountImportance::OptionalStatic, "optional_unverified") => {
            "do_not_blacklist_pool".to_string()
        }
        (MixAccountImportance::ProgramOrSysvar, _) => "ignore_program_or_sysvar".to_string(),
        _ => "inspect".to_string(),
    }
}

fn reason_for_status(
    importance: MixAccountImportance,
    status: &str,
    rpc_error: &str,
) -> String {
    match (importance, status) {
        (MixAccountImportance::OptionalStatic, "optional_missing") => {
            "optional_or_tx_only_mix_param".to_string()
        }
        (MixAccountImportance::TxOnly, "tx_only_missing") => {
            "tx_only_account_missing_from_bootstrap".to_string()
        }
        (MixAccountImportance::RequiredLive, "invalid")
        | (MixAccountImportance::RequiredStatic, "invalid") => {
            "invalid_required_account".to_string()
        }
        _ => rpc_error.to_string(),
    }
}

fn dedup_account_refs(refs: &mut Vec<MixAccountRef>) {
    let mut seen = HashSet::new();
    refs.retain(|account| {
        seen.insert((
            account.pubkey,
            account.source.clone(),
            account.role.clone(),
        ))
    });
}

fn unique_pubkeys<I>(iter: I) -> Vec<Pubkey>
where
    I: IntoIterator<Item = Pubkey>,
{
    let mut out = iter.into_iter().collect::<Vec<_>>();
    out.sort_unstable();
    out.dedup();
    out
}

fn route_plan_amm_keys(route_plan: &serde_json::Value) -> Vec<Pubkey> {
    let mut out = route_plan
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|hop| hop.get("swapInfo"))
        .filter_map(|swap_info| swap_info.get("ammKey"))
        .filter_map(|amm_key| amm_key.as_str())
        .filter_map(|amm_key| Pubkey::try_from(amm_key).ok())
        .collect::<Vec<_>>();
    out.sort_unstable();
    out.dedup();
    out
}

pub fn pubkeys_json(pubkeys: &[Pubkey]) -> String {
    serde_json::to_string(
        &pubkeys
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>(),
    )
    .unwrap_or_else(|_| "[]".to_string())
}

fn invalid_account_report(
    inner: &MixRegistryInner,
    account_ref: &MixAccountRef,
) -> InvalidAccountReport {
    let status = inner
        .account_meta
        .get(&account_ref.pubkey)
        .map(|meta| meta.status.as_str())
        .unwrap_or_else(|| {
            if inner.invalid_accounts.contains(&account_ref.pubkey) {
                "invalid"
            } else if inner.unverified_accounts.contains(&account_ref.pubkey) {
                "unverified"
            } else {
                "not_checked"
            }
        });
    let rpc_error = inner
        .account_reasons
        .get(&account_ref.pubkey)
        .cloned()
        .unwrap_or_else(|| status.to_string());
    let action = action_for_status(account_ref.importance, status);
    let reason = reason_for_status(account_ref.importance, status, &rpc_error);
    let pools = inner
        .account_to_pools
        .get(&account_ref.pubkey)
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .map(|pool| pool.to_string())
        .collect::<Vec<_>>();
    InvalidAccountReport {
        pubkey: account_ref.pubkey.to_string(),
        role: account_ref.role.clone(),
        source: account_ref.source.clone(),
        importance: account_ref.importance,
        status: status.to_string(),
        rpc_error,
        action,
        reason,
        pools,
    }
}

fn pool_status(inner: &MixRegistryInner, pool: &Pubkey) -> &'static str {
    if inner.invalid_pools.contains(pool) {
        "invalid"
    } else if inner.unverified_pools.contains(pool) {
        "unverified"
    } else {
        "valid"
    }
}

fn pool_reason(status: &str) -> Option<String> {
    match status {
        "invalid" => Some("missing_required_account".to_string()),
        "unverified" => Some("unverified_required_account".to_string()),
        _ => None,
    }
}

fn mix_output_root(mix_path: &Path) -> PathBuf {
    let parent = mix_path.parent().unwrap_or_else(|| Path::new("."));
    if parent.file_name().and_then(|name| name.to_str()) == Some("1") {
        if let Some(metis_dir) = parent.parent() {
            if metis_dir.file_name().and_then(|name| name.to_str()) == Some("metis") {
                if let Some(root) = metis_dir.parent() {
                    return root.to_path_buf();
                }
            }
        }
    }
    parent.to_path_buf()
}

fn capped_rpc_rps(groups_per_second: u64) -> u64 {
    groups_per_second.max(1).min(MAX_MIX_RPC_RPS)
}

fn log_rpc_fetch_reason(reason: &str, pubkeys: &[Pubkey]) {
    if pubkeys.is_empty() {
        return;
    }
    let sample = pubkeys.iter().copied().take(10).collect::<Vec<_>>();
    eprintln!(
        "[rpc_fetch_reason] reason={} count={} pubkeys_sample={}",
        reason,
        pubkeys.len(),
        pubkeys_json(&sample)
    );
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn classify_rpc_error(message: &str) -> &'static str {
    let lower = message.to_ascii_lowercase();
    if lower.contains("429") || lower.contains("too many requests") || lower.contains("rate limit")
    {
        "rate_limited"
    } else if lower.contains("timeout") {
        "timeout"
    } else if lower.contains("transport")
        || lower.contains("connection")
        || lower.contains("request")
    {
        "transport"
    } else {
        "other"
    }
}