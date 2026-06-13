//! Hot account cache fed by a Yellowstone gRPC subscription.
//!
//! This is the same data Metis consumes. By keeping a parallel copy in our
//! own process we can hand it to LiteSVM for pre-flight simulation without
//! any RPC round-trip on the hot path (a getMultipleAccounts would add
//! 20-50ms and make simulation useless).
//!
//! The cache subscribes once at startup with two filter entries:
//!   1. all DEX program ids from `program_registry::PROGRAMS` (owner filter)
//!      -> every pool account owned by those programs streams in
//!   2. the user's WSOL ATA (specific account filter)
//!      -> so the simulated tx can read / debit it
//!
//! Missing entries (token mints, intermediate ATAs) are fetched lazily from
//! RPC the first time they're needed and then cached forever (their data
//! rarely changes).

use anyhow::{Context, Result};
use dashmap::DashMap;
use futures::{SinkExt, StreamExt};
use solana_account::Account;
use solana_address::Address;
use solana_client::rpc_client::RpcClient;
use solana_sdk::{pubkey::Pubkey, transaction::VersionedTransaction};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::sync::Notify;
use tracing::{debug, info, warn};
use yellowstone_grpc_client::GeyserGrpcClient;
use yellowstone_grpc_proto::prelude::{
    subscribe_update::UpdateOneof, CommitmentLevel, SubscribeRequest,
    SubscribeRequestFilterAccounts, SubscribeRequestPing,
};

/// Shared concurrent cache. Cloning an `AccountCache` is cheap; it's just
/// an Arc-wrapped DashMap plus an Arc-wrapped RpcClient for fallbacks.
#[derive(Clone)]
pub struct AccountCache {
    inner: Arc<DashMap<Pubkey, Account>>,
    rpc: Arc<RpcClient>,
    /// Extra RPC endpoints tried in order when `rpc` fails a
    /// `getMultipleAccounts` call. Used ONLY for account data — never for
    /// blockhash queries, transaction simulation, or Jito submission.
    fallback_rpcs: Arc<Vec<Arc<RpcClient>>>,
    /// Slot of the most recent Yellowstone account update. The simulator
    /// reads this to set LiteSVM's Clock.slot — no RPC call needed.
    stream_slot: Arc<AtomicU64>,
    stream_unix_timestamp: Arc<AtomicI64>,
    timestamp_seed_slot: Arc<AtomicU64>,
    timestamp_seed_unix: Arc<AtomicI64>,
    tx_static_cache_path: Arc<RwLock<Option<PathBuf>>>,
    tx_static_cache_lock: Arc<Mutex<()>>,
    /// Accounts to add to the live Yellowstone subscription at runtime
    /// (token vaults and other writable state NOT covered by the owner
    /// filter). Grows as the simulator discovers stale/missing accounts.
    dynamic_accounts: Arc<Mutex<HashSet<Pubkey>>>,
    /// Signals the stream task to re-send its SubscribeRequest after
    /// `dynamic_accounts` changes. Multiple rapid additions collapse into a
    /// single wakeup, so re-subscription is naturally debounced.
    resubscribe_notify: Arc<Notify>,
    /// DEX program ids already covered by the "dex_pools" owner filter. Used
    /// to skip redundant per-account subscriptions for pool state that the
    /// owner filter already streams.
    dex_owner_set: Arc<RwLock<HashSet<Pubkey>>>,
}

#[derive(Clone, Debug)]
pub enum AccountFetchResult {
    Found(Account),
    NotFound,
    Error { kind: String, message: String },
}

#[derive(Clone, Debug)]
pub struct AccountCompareFetchResult {
    pub accounts: HashMap<Pubkey, AccountFetchResult>,
    pub rpc_context_slot: Option<u64>,
}

#[derive(serde::Serialize, serde::Deserialize, Default)]
struct TxStaticAccountCacheFile {
    schema_version: u64,
    accounts: Vec<TxStaticAccountCacheEntry>,
}

#[derive(serde::Serialize, serde::Deserialize, Clone)]
struct TxStaticAccountCacheEntry {
    pubkey: String,
    owner: String,
    lamports: u64,
    executable: bool,
    rent_epoch: u64,
    data_base64: String,
    fetched_slot: Option<u64>,
    first_seen_unix: u64,
    last_seen_unix: u64,
    source: String,
    usage_count: u64,
    status: String,
    classification: String,
}

#[derive(serde::Serialize, serde::Deserialize, Default)]
struct AutoMissingCacheFile {
    schema_version: u32,
    accounts: Vec<AutoMissingCachedRecord>,
}

#[derive(serde::Serialize, serde::Deserialize, Clone)]
struct AutoMissingCachedRecord {
    pubkey: String,
    owner: String,
    lamports: u64,
    executable: bool,
    rent_epoch: u64,
    data_base64: String,
    data_len: usize,
    fetched_slot: u64,
    source: String,
    classification: String,
    is_writable_seen: bool,
    last_seen_unix: u64,
    seen_count: u32,
    status: String,
    needs_live_subscription: bool,
}

#[derive(serde::Serialize, serde::Deserialize, Clone)]
struct AutoMissingLiveRecord {
    pubkey: String,
    owner: String,
    classification: String,
    is_writable_seen: bool,
    first_seen_unix: u64,
}

#[derive(serde::Serialize, serde::Deserialize, Default)]
struct ManualAccountCacheFileForProblemLoader {
    schema_version: u32,
    accounts: Vec<ManualCachedAccountForProblemLoader>,
}

#[derive(serde::Serialize, serde::Deserialize, Clone)]
struct ManualCachedAccountForProblemLoader {
    pubkey: String,
    owner: String,
    lamports: u64,
    executable: bool,
    rent_epoch: u64,
    data_base64: String,
    fetched_slot: Option<u64>,
    status: String,
}

impl AccountCache {
    pub fn new(rpc: Arc<RpcClient>) -> Self {
        Self::new_with_fallbacks(rpc, Arc::new(vec![]))
    }

    pub fn new_with_fallbacks(rpc: Arc<RpcClient>, fallback_rpcs: Arc<Vec<Arc<RpcClient>>>) -> Self {
        Self {
            inner: Arc::new(DashMap::with_capacity(4096)),
            rpc,
            fallback_rpcs,
            stream_slot: Arc::new(AtomicU64::new(0)),
            stream_unix_timestamp: Arc::new(AtomicI64::new(fallback_unix_timestamp())),
            timestamp_seed_slot: Arc::new(AtomicU64::new(0)),
            timestamp_seed_unix: Arc::new(AtomicI64::new(fallback_unix_timestamp())),
            tx_static_cache_path: Arc::new(RwLock::new(None)),
            tx_static_cache_lock: Arc::new(Mutex::new(())),
            dynamic_accounts: Arc::new(Mutex::new(HashSet::new())),
            resubscribe_notify: Arc::new(Notify::new()),
            dex_owner_set: Arc::new(RwLock::new(HashSet::new())),
        }
    }

    /// Call `getMultipleAccounts` on the primary RPC. If it fails, try each
    /// fallback in order. Only account-data paths use this — never blockhash
    /// or transaction simulation.
    fn get_multiple_accounts_with_fallback(
        &self,
        keys: &[Pubkey],
    ) -> solana_client::client_error::Result<Vec<Option<solana_sdk::account::Account>>> {
        match self.rpc.get_multiple_accounts(keys) {
            Ok(r) => return Ok(r),
            Err(primary_err) => {
                for (i, fallback) in self.fallback_rpcs.iter().enumerate() {
                    match fallback.get_multiple_accounts(keys) {
                        Ok(r) => {
                            eprintln!(
                                "[rpc_fallback] accounts={} primary_err={} used_fallback_idx={}",
                                keys.len(),
                                primary_err,
                                i
                            );
                            return Ok(r);
                        }
                        Err(_) => continue,
                    }
                }
                Err(primary_err)
            }
        }
    }

    /// Add a single account to the live Yellowstone subscription at runtime.
    /// Used for writable token vaults and foreign-owned state that the
    /// owner filter does not capture, so the cache keeps them fresh instead
    /// of serving a stale startup snapshot. Returns true if newly added.
    pub fn subscribe_account(&self, pk: Pubkey) -> bool {
        let newly_added = {
            let mut set = self.dynamic_accounts.lock().unwrap();
            set.insert(pk)
        };
        if newly_added {
            self.resubscribe_notify.notify_one();
        }
        newly_added
    }


    /// Ensure a pubkey is present in the direct Yellowstone account subscription.
    /// Returns `(subscribed_after_call, newly_added)`.  This is intentionally
    /// different from `subscribe_account()`: a previously-added live account is
    /// still considered subscribed, so operator logs must show `subscribed=true`
    /// instead of `false` for already tracked token vaults.
    pub fn ensure_account_subscription(&self, pk: Pubkey) -> (bool, bool) {
        let newly_added = {
            let mut set = self.dynamic_accounts.lock().unwrap();
            set.insert(pk)
        };
        if newly_added {
            self.resubscribe_notify.notify_one();
        }
        (true, newly_added)
    }

    /// Subscribe a writable account to the live stream only if its owner is
    /// NOT already covered by the "dex_pools" owner filter. This targets the
    /// real staleness gap (SPL-Token vaults, foreign-owned PDAs) without
    /// redundantly re-subscribing pool state the owner filter already streams.
    pub fn note_uncovered_writable(&self, pk: Pubkey, owner: &Pubkey) {
        let covered = self
            .dex_owner_set
            .read()
            .map(|set| set.contains(owner))
            .unwrap_or(false);
        if !covered && self.subscribe_account(pk) {
            eprintln!(
                "[grpc_dynamic_subscribe] pubkey={} owner={} reason=uncovered_writable",
                pk, owner
            );
        }
    }

    /// The latest slot seen from the Yellowstone stream. The sim pool reads
    /// this instead of making its own `get_slot` RPC.
    pub fn stream_slot(&self) -> Arc<AtomicU64> {
        self.stream_slot.clone()
    }

    pub fn stream_unix_timestamp(&self) -> Arc<AtomicI64> {
        self.stream_unix_timestamp.clone()
    }

    /// Seed the stream slot with an initial value (from RPC at startup)
    /// so sims have a valid Clock.slot before the first Yellowstone message.
    pub fn seed_slot(&self, slot: u64) {
        let ts = self.timestamp_seed_unix.load(Ordering::Relaxed);
        self.seed_clock(slot, ts);
    }

    pub fn seed_clock(&self, slot: u64, unix_timestamp: i64) {
        self.stream_slot.store(slot, Ordering::Relaxed);
        self.timestamp_seed_slot.store(slot, Ordering::Relaxed);
        self.timestamp_seed_unix
            .store(unix_timestamp, Ordering::Relaxed);
        self.stream_unix_timestamp.store(
            estimate_unix_timestamp(slot, slot, unix_timestamp),
            Ordering::Relaxed,
        );
    }

    /// Fast path: read from the hot cache. Returns None if not yet populated.
    #[inline]
    pub fn get(&self, pubkey: &Pubkey) -> Option<Account> {
        self.inner.get(pubkey).map(|v| v.value().clone())
    }

    pub fn insert_manual(&self, pubkey: Pubkey, account: Account) {
        self.inner.insert(pubkey, account);
    }

    pub fn load_tx_static_account_cache(&self, path: impl AsRef<Path>) -> Result<usize> {
        let path = path.as_ref().to_path_buf();
        *self.tx_static_cache_path.write().unwrap() = Some(path.clone());
        if !path.exists() {
            eprintln!(
                "[tx_static_cache] loaded=0 skipped=0 path={} status=missing",
                path.display()
            );
            return Ok(0);
        }

        let bytes = std::fs::read(&path)
            .with_context(|| format!("cannot read {}", path.display()))?;
        let file: TxStaticAccountCacheFile = serde_json::from_slice(&bytes)
            .with_context(|| format!("cannot parse {}", path.display()))?;
        let mut loaded = 0usize;
        let mut skipped = 0usize;
        for entry in file.accounts {
            if entry.status != "valid" || !can_load_tx_static_classification(&entry.classification)
            {
                skipped += 1;
                continue;
            }
            let Ok(pubkey) = Pubkey::try_from(entry.pubkey.as_str()) else {
                skipped += 1;
                continue;
            };
            let Ok(owner) = Pubkey::try_from(entry.owner.as_str()) else {
                skipped += 1;
                continue;
            };
            let Ok(data) = base64::Engine::decode(
                &base64::engine::general_purpose::STANDARD,
                entry.data_base64.as_bytes(),
            ) else {
                skipped += 1;
                continue;
            };
            self.inner.insert(
                pubkey,
                Account {
                    lamports: entry.lamports,
                    data,
                    owner: Address::from(owner.to_bytes()),
                    executable: entry.executable,
                    rent_epoch: entry.rent_epoch,
                },
            );
            loaded += 1;
        }
        eprintln!(
            "[tx_static_cache] loaded={} skipped={} path={} status=ok",
            loaded,
            skipped,
            path.display()
        );
        Ok(loaded)
    }

    pub fn tx_static_refresh_pubkeys(&self, path: impl AsRef<Path>) -> Result<Vec<Pubkey>> {
        let path = path.as_ref();
        if !path.exists() {
            return Ok(Vec::new());
        }

        let bytes = std::fs::read(path)
            .with_context(|| format!("cannot read {}", path.display()))?;
        let file: TxStaticAccountCacheFile = serde_json::from_slice(&bytes)
            .with_context(|| format!("cannot parse {}", path.display()))?;
        let mut pubkeys = file
            .accounts
            .into_iter()
            .filter(|entry| {
                entry.status != "valid"
                    || !can_load_tx_static_classification(&entry.classification)
            })
            .filter_map(|entry| Pubkey::try_from(entry.pubkey.as_str()).ok())
            .collect::<Vec<_>>();
        pubkeys.sort_unstable();
        pubkeys.dedup();
        eprintln!(
            "[tx_static_cache] refresh_candidates={} path={}",
            pubkeys.len(),
            path.display()
        );
        Ok(pubkeys)
    }

    /// Slow path used only during startup warm-up and for rarely-changing
    /// accounts (token mints, ALTs) that aren't streamed over Yellowstone.
    pub fn get_or_fetch(&self, pubkey: &Pubkey) -> Result<Account> {
        if let Some(a) = self.get(pubkey) {
            return Ok(a);
        }
        log_rpc_fetch_reason("tx_static_unknown", &[*pubkey]);
        let acct = self
            .rpc
            .get_account(pubkey)
            .with_context(|| format!("RPC fetch of {pubkey} failed"))?;
        let account = rpc_account_to_cache_account(acct);
        self.inner.insert(*pubkey, account.clone());
        self.persist_tx_static_account(*pubkey, &account, "tx_static_unknown");
        Ok(account)
    }

    /// Batch fetch missing accounts with getMultipleAccounts. RPC errors are
    /// retried with backoff so rate limits do not become false missing
    /// accounts in the simulator.
    pub fn get_many_or_fetch(&self, pubkeys: &[Pubkey]) -> HashMap<Pubkey, AccountFetchResult> {
        let mut keys = pubkeys.to_vec();
        keys.sort_unstable();
        keys.dedup();

        let mut out = HashMap::with_capacity(keys.len());
        let mut missing = Vec::new();
        for pk in keys {
            if let Some(account) = self.get(&pk) {
                out.insert(pk, AccountFetchResult::Found(account));
            } else {
                missing.push(pk);
            }
        }

        const MAX_ATTEMPTS: usize = 5;
        for chunk in missing.chunks(100) {
            let chunk_keys = chunk.to_vec();
            let mut attempt = 0usize;
            loop {
                attempt += 1;
                log_rpc_fetch_reason("tx_static_unknown", &chunk_keys);
                match self.get_multiple_accounts_with_fallback(&chunk_keys) {
                    Ok(accounts) => {
                        for (pk, acct_opt) in chunk_keys.iter().zip(accounts.into_iter()) {
                            match acct_opt {
                                Some(acct) => {
                                    let account = rpc_account_to_cache_account(acct);
                                    self.inner.insert(*pk, account.clone());
                                    self.persist_tx_static_account(
                                        *pk,
                                        &account,
                                        "tx_static_unknown",
                                    );
                                    out.insert(*pk, AccountFetchResult::Found(account));
                                }
                                None => {
                                    out.insert(*pk, AccountFetchResult::NotFound);
                                }
                            }
                        }
                        break;
                    }
                    Err(e) => {
                        let message = e.to_string();
                        let kind = classify_rpc_error(&message).to_string();
                        eprintln!(
                            "[sim_batch_fetch_retry] accounts={} attempt={} error_kind={} error={}",
                            chunk_keys.len(),
                            attempt,
                            kind,
                            message
                        );
                        if attempt >= MAX_ATTEMPTS {
                            for pk in &chunk_keys {
                                out.insert(
                                    *pk,
                                    AccountFetchResult::Error {
                                        kind: kind.clone(),
                                        message: message.clone(),
                                    },
                                );
                            }
                            break;
                        }

                        let delay_ms = 250_u64.saturating_mul(1_u64 << (attempt - 1).min(4));
                        std::thread::sleep(Duration::from_millis(delay_ms));
                    }
                }
            }
        }

        out
    }


    /// Hot-path version of `get_many_or_fetch` used by LiteSVM preflight.
    ///
    /// Keep this as a separate method so litesvm_sim.rs can compile against the
    /// optimized hot-path API.  The current implementation delegates to the
    /// existing batch fetcher to preserve the project's fallback RPC behavior;
    /// if you want stricter latency later, reduce retries inside this method
    /// only instead of changing startup warm-up behavior globally.
    pub fn get_many_or_fetch_hot(&self, pubkeys: &[Pubkey]) -> HashMap<Pubkey, AccountFetchResult> {
        self.get_many_or_fetch(pubkeys)
    }

    /// Pre-fetch a batch of accounts (used at startup to warm up mints, ATAs,
    /// etc. that won't naturally stream in via the owner filter).
    pub fn prefetch(&self, pubkeys: &[Pubkey]) {
        for pk in pubkeys {
            if let Err(e) = self.get_or_fetch(pk) {
                warn!(pubkey = %pk, error = %e, "prefetch miss");
            }
        }
    }

    /// Startup-only warm-up. Each group is fetched with getMultipleAccounts and
    /// RPC errors are retried before the bot is allowed to continue.
    pub async fn prefetch_groups_rate_limited(
        &self,
        groups: &[Vec<Pubkey>],
        groups_per_second: u64,
    ) {
        let rate = groups_per_second.max(1).min(5);
        let delay = Duration::from_millis((1000 / rate).max(1));
        let total_groups = groups.len();
        let total_accounts: usize = groups.iter().map(|g| g.len()).sum();
        let mut fetched = 0usize;
        let mut missing = 0usize;
        let mut requests = 0usize;
        let mut retries = 0usize;

        eprintln!(
            "[sim_prefetch] start groups={total_groups} accounts={total_accounts} rate={rate}/sec"
        );

        for (group_idx, group) in groups.iter().enumerate() {
            let mut keys = group.clone();
            keys.sort_unstable();
            keys.dedup();
            keys.retain(|pk| self.get(pk).is_none());

            for chunk in keys.chunks(100) {
                let chunk_keys = chunk.to_vec();
                loop {
                    requests += 1;
                    log_rpc_fetch_reason("tx_static_unknown", &chunk_keys);
                    let rpc = self.rpc.clone();
                    let fallback_rpcs = self.fallback_rpcs.clone();
                    let request_keys = chunk_keys.clone();
                    let result = tokio::task::spawn_blocking(move || {
                        match rpc.get_multiple_accounts(&request_keys) {
                            Ok(r) => Ok(r),
                            Err(primary_err) => {
                                for (i, fb) in fallback_rpcs.iter().enumerate() {
                                    match fb.get_multiple_accounts(&request_keys) {
                                        Ok(r) => {
                                            eprintln!(
                                                "[rpc_fallback] accounts={} primary_err={} used_fallback_idx={}",
                                                request_keys.len(), primary_err, i
                                            );
                                            return Ok(r);
                                        }
                                        Err(_) => continue,
                                    }
                                }
                                Err(primary_err)
                            }
                        }
                    })
                    .await;

                    match result {
                        Ok(Ok(accounts)) => {
                            for (pk, acct_opt) in chunk_keys.iter().zip(accounts.into_iter()) {
                                match acct_opt {
                                    Some(acct) => {
                                        let account = rpc_account_to_cache_account(acct);
                                        self.inner.insert(*pk, account);
                                        fetched += 1;
                                    }
                                    None => {
                                        missing += 1;
                                        warn!(pubkey = %pk, "prefetch account not found");
                                    }
                                }
                            }
                            break;
                        }
                        Ok(Err(e)) => {
                            retries += 1;
                            eprintln!(
                                "[sim_prefetch_retry] group={}/{} accounts={} error={}",
                                group_idx + 1,
                                total_groups,
                                chunk_keys.len(),
                                e
                            );
                            tokio::time::sleep(Duration::from_secs(1)).await;
                        }
                        Err(e) => {
                            retries += 1;
                            eprintln!(
                                "[sim_prefetch_retry] group={}/{} accounts={} task_error={:?}",
                                group_idx + 1,
                                total_groups,
                                chunk_keys.len(),
                                e
                            );
                            tokio::time::sleep(Duration::from_secs(1)).await;
                        }
                    }
                }

                tokio::time::sleep(delay).await;
            }

            if (group_idx + 1) % 25 == 0 || group_idx + 1 == total_groups {
                eprintln!(
                    "[sim_prefetch] progress groups={}/{} fetched={} missing={} requests={} retries={}",
                    group_idx + 1,
                    total_groups,
                    fetched,
                    missing,
                    requests,
                    retries
                );
            }
        }

        eprintln!(
            "[sim_prefetch] complete groups={total_groups} fetched={fetched} missing={missing} requests={requests} retries={retries}"
        );
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn simulate_transaction_for_compare(
        &self,
        tx: &VersionedTransaction,
    ) -> Result<(String, Vec<String>)> {
        eprintln!("[rpc_fetch_reason] reason=rpc_compare count=1 pubkeys_sample=[]");
        let response = self
            .rpc
            .simulate_transaction(tx)
            .context("RPC simulateTransaction failed")?;
        Ok((
            format!("{:?}", response.value.err),
            response.value.logs.unwrap_or_default(),
        ))
    }

    pub fn fetch_accounts_for_compare(
        &self,
        pubkeys: &[Pubkey],
    ) -> AccountCompareFetchResult {
        let mut keys = pubkeys.to_vec();
        keys.sort_unstable();
        keys.dedup();

        let rpc_context_slot = self.rpc.get_slot().ok();
        let mut out = HashMap::with_capacity(keys.len());
        for chunk in keys.chunks(100) {
            let chunk_keys = chunk.to_vec();
            log_rpc_fetch_reason("failed_account_compare", &chunk_keys);
            match self.get_multiple_accounts_with_fallback(&chunk_keys) {
                Ok(accounts) => {
                    for (pk, acct_opt) in chunk_keys.iter().zip(accounts.into_iter()) {
                        match acct_opt {
                            Some(acct) => {
                                out.insert(
                                    *pk,
                                    AccountFetchResult::Found(rpc_account_to_cache_account(acct)),
                                );
                            }
                            None => {
                                out.insert(*pk, AccountFetchResult::NotFound);
                            }
                        }
                    }
                }
                Err(e) => {
                    let message = e.to_string();
                    let kind = classify_rpc_error(&message).to_string();
                    for pk in chunk_keys {
                        out.insert(
                            pk,
                            AccountFetchResult::Error {
                                kind: kind.clone(),
                                message: message.clone(),
                            },
                        );
                    }
                }
            }
        }
        AccountCompareFetchResult {
            accounts: out,
            rpc_context_slot,
        }
    }

    fn persist_tx_static_account(&self, pubkey: Pubkey, account: &Account, source: &str) {
        let path = self.tx_static_cache_path.read().unwrap().clone();
        let Some(path) = path else {
            return;
        };
        let classification = classify_tx_static_account(account);
        let status = if can_persist_tx_static_data(classification) {
            "valid"
        } else if classification.starts_with("volatile_") {
            "volatile"
        } else {
            "metadata_only"
        };
        let data_base64 = if status == "valid" {
            base64::Engine::encode(
                &base64::engine::general_purpose::STANDARD,
                account.data.as_slice(),
            )
        } else {
            String::new()
        };
        let now = unix_now();
        let fetched_slot = self.stream_slot.load(Ordering::Relaxed);
        let entry = TxStaticAccountCacheEntry {
            pubkey: pubkey.to_string(),
            owner: account.owner.to_string(),
            lamports: account.lamports,
            executable: account.executable,
            rent_epoch: account.rent_epoch,
            data_base64,
            fetched_slot: (fetched_slot > 0).then_some(fetched_slot),
            first_seen_unix: now,
            last_seen_unix: now,
            source: source.to_string(),
            usage_count: 1,
            status: status.to_string(),
            classification: classification.to_string(),
        };

        let _guard = self.tx_static_cache_lock.lock().unwrap();
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let mut file = match std::fs::read(&path) {
            Ok(bytes) => serde_json::from_slice::<TxStaticAccountCacheFile>(&bytes)
                .unwrap_or_else(|_| TxStaticAccountCacheFile {
                    schema_version: 1,
                    accounts: Vec::new(),
                }),
            Err(_) => TxStaticAccountCacheFile {
                schema_version: 1,
                accounts: Vec::new(),
            },
        };
        file.schema_version = 1;
        if let Some(existing) = file
            .accounts
            .iter_mut()
            .find(|existing| existing.pubkey == entry.pubkey)
        {
            let first_seen = existing.first_seen_unix;
            let usage_count = existing.usage_count.saturating_add(1);
            *existing = entry;
            existing.first_seen_unix = first_seen;
            existing.usage_count = usage_count;
        } else {
            file.accounts.push(entry);
        }
        let tmp = path.with_extension("json.tmp");
        if serde_json::to_vec_pretty(&file)
            .ok()
            .and_then(|json| std::fs::write(&tmp, json).ok())
            .and_then(|_| std::fs::rename(&tmp, &path).ok())
            .is_none()
        {
            eprintln!(
                "[tx_static_cache] persist_failed path={} pubkey={}",
                path.display(),
                pubkey
            );
        }
    }


    /// Watch `problem_sim_accounts.csv` in the simulation output root and
    /// promote reported accounts into real runtime actions.
    ///
    /// JSON/CSV files by themselves do nothing; this watcher is the code path
    /// that turns `needs_grpc_live_state` rows into startup RPC snapshots plus
    /// live Yellowstone specific-account subscriptions. It also keeps a
    /// deduplicated `load.txt` audit file in the same root so the operator can
    /// see what was actually loaded/subscribed/ignored.
    pub fn spawn_problem_accounts_watcher(&self, output_root: PathBuf, poll_secs: u64) {
        let cache = self.clone();
        let poll = Duration::from_secs(poll_secs.max(10));
        let processed_this_run: Arc<Mutex<HashSet<Pubkey>>> = Arc::new(Mutex::new(HashSet::new()));
        tokio::spawn(async move {
            loop {
                let cache2 = cache.clone();
                let root2 = output_root.clone();
                let root_for_task = root2.clone();
                let seen2 = processed_this_run.clone();
                let result = tokio::task::spawn_blocking(move || {
                    cache2.process_problem_accounts_once(&root_for_task, &seen2)
                })
                .await;
                match result {
                    Ok(Ok(stats)) => {
                        if stats.promoted_live > 0
                            || stats.manual_loaded > 0
                            || stats.ignored > 0
                            || stats.errors > 0
                        {
                            eprintln!(
                                "[problem_accounts_watcher] promoted_live={} manual_loaded={} ignored={} errors={} skipped_seen={} path={}",
                                stats.promoted_live,
                                stats.manual_loaded,
                                stats.ignored,
                                stats.errors,
                                stats.skipped_seen,
                                root2.join("problem_sim_accounts.csv").display()
                            );
                        }
                    }
                    Ok(Err(e)) => {
                        eprintln!("[problem_accounts_watcher] error={}", e);
                    }
                    Err(e) => {
                        eprintln!("[problem_accounts_watcher] task_error={:?}", e);
                    }
                }
                tokio::time::sleep(poll).await;
            }
        });
    }

    fn process_problem_accounts_once(
        &self,
        output_root: &Path,
        processed_this_run: &Arc<Mutex<HashSet<Pubkey>>>,
    ) -> Result<ProblemAccountWatchStats> {
        let csv_path = output_root.join("problem_sim_accounts.csv");
        if !csv_path.exists() {
            return Ok(ProblemAccountWatchStats::default());
        }

        let content = std::fs::read_to_string(&csv_path)
            .with_context(|| format!("cannot read {}", csv_path.display()))?;
        let mut rows = Vec::new();
        let mut header: Vec<String> = Vec::new();
        for (line_idx, raw) in content.lines().enumerate() {
            let line = raw.trim();
            if line.is_empty() {
                continue;
            }
            if line_idx == 0 && line.to_ascii_lowercase().contains("pubkey") {
                header = split_simple_csv_line(line);
                continue;
            }
            if let Some(row) = parse_problem_csv_row(line, &header) {
                rows.push(row);
            }
        }

        let mut stats = ProblemAccountWatchStats::default();
        if rows.is_empty() {
            return Ok(stats);
        }

        let mut load_map = read_load_txt_map(&output_root.join("load.txt"));
        let now = unix_now();

        for row in rows {
            let Ok(pubkey) = Pubkey::try_from(row.pubkey.as_str()) else {
                stats.errors += 1;
                continue;
            };
            {
                let mut seen = processed_this_run.lock().unwrap();
                if !seen.insert(pubkey) {
                    stats.skipped_seen += 1;
                    continue;
                }
            }

            if row.classification == "expected_readonly_pda_authority"
                || row.action.contains("ignore_not_found")
                || row.severity == "info"
            {
                stats.ignored += 1;
                upsert_load_line(
                    &mut load_map,
                    &pubkey,
                    format!(
                        "unix={} pubkey={} action=ignore_expected_pda severity={} classification={} pool={} role={} dex={}",
                        now,
                        pubkey,
                        row.severity,
                        row.classification,
                        empty_dash(&row.pool),
                        empty_dash(&row.role),
                        empty_dash(&row.dex)
                    ),
                );
                continue;
            }

            let should_live = row.severity == "needs_grpc_live_state"
                || row.classification == "live_state_stale"
                || matches!(
                    row.role.as_str(),
                    "pool" | "market" | "tokenAccountA" | "tokenAccountB" | "vault" | "reserve" | "baseVault" | "quoteVault" | "globalVault" | "oracle" | "tickArray" | "tickArray0" | "tickArray1" | "tickArray2" | "binArray" | "observation" | "observationState" | "observationAccount"
                );

            if should_live {
                let (subscribed, newly_added) = self.ensure_account_subscription(pubkey);
                let fetch = self.get_many_or_fetch(&[pubkey]);
                let status = describe_fetch_result(fetch.get(&pubkey));
                let mut persisted = false;
                if let Some(AccountFetchResult::Found(account)) = fetch.get(&pubkey) {
                    persist_problem_live_account(output_root, pubkey, account, &row, now)?;
                    persisted = true;
                }
                stats.promoted_live += 1;
                upsert_load_line(
                    &mut load_map,
                    &pubkey,
                    format!(
                        "unix={} pubkey={} action=grpc_live startup_fetch=true subscribed={} newly_added={} fetch={} persisted={} severity={} classification={} pool={} role={} dex={}",
                        now,
                        pubkey,
                        subscribed,
                        newly_added,
                        status,
                        persisted,
                        row.severity,
                        row.classification,
                        empty_dash(&row.pool),
                        empty_dash(&row.role),
                        empty_dash(&row.dex)
                    ),
                );
                eprintln!(
                    "[problem_account_promoted] pubkey={} action=grpc_live subscribed={} newly_added={} fetch={} persisted={} pool={} role={} dex={}",
                    pubkey,
                    subscribed,
                    newly_added,
                    status,
                    persisted,
                    empty_dash(&row.pool),
                    empty_dash(&row.role),
                    empty_dash(&row.dex)
                );
                continue;
            }

            // Static read accounts (global configs, authority PDAs) that rarely change.
            // Fetch once from RPC and persist as a manual cached copy. Unlike
            // needs_grpc_live_state, these do not need a live Yellowstone subscription.
            if row.severity == "needs_manual_static_or_synthetic" {
                let fetch = self.get_many_or_fetch(&[pubkey]);
                let status = describe_fetch_result(fetch.get(&pubkey));
                let mut persisted = false;
                if let Some(AccountFetchResult::Found(account)) = fetch.get(&pubkey) {
                    let class = classify_tx_static_account(account);
                    if can_persist_tx_static_data(class) {
                        if let Err(e) = persist_problem_manual_account(output_root, pubkey, account) {
                            eprintln!(
                                "[problem_account_static_persist_error] pubkey={} error={}",
                                pubkey, e
                            );
                        } else {
                            persisted = true;
                        }
                    }
                }
                stats.manual_loaded += 1;
                upsert_load_line(
                    &mut load_map,
                    &pubkey,
                    format!(
                        "unix={} pubkey={} action=static_rpc_loaded fetch={} persisted={} severity={} classification={} pool={} role={} dex={}",
                        now, pubkey, status, persisted,
                        row.severity, row.classification,
                        empty_dash(&row.pool), empty_dash(&row.role), empty_dash(&row.dex)
                    ),
                );
                eprintln!(
                    "[problem_account_promoted] pubkey={} action=static_manual fetch={} persisted={} pool={} role={} dex={}",
                    pubkey, status, persisted,
                    empty_dash(&row.pool), empty_dash(&row.role), empty_dash(&row.dex)
                );
                continue;
            }

            if row.severity == "needs_manual_check" {
                let fetch = self.get_many_or_fetch(&[pubkey]);
                let status = describe_fetch_result(fetch.get(&pubkey));
                let mut subscribed = false;
                if let Some(AccountFetchResult::Found(account)) = fetch.get(&pubkey) {
                    let owner = pubkey_from_address(&account.owner);
                    let class = classify_tx_static_account(account);
                    let mut persisted = false;
                    if class == "volatile_token_account" || class == "volatile_dex_state" {
                        let (subscribed_after, _newly_added) = self.ensure_account_subscription(pubkey);
                        subscribed = subscribed_after;
                        persist_problem_live_account(output_root, pubkey, account, &row, now)?;
                        persisted = true;
                    } else if can_persist_tx_static_data(class) {
                        persist_problem_manual_account(output_root, pubkey, account)?;
                        persisted = true;
                    }
                    upsert_load_line(
                        &mut load_map,
                        &pubkey,
                        format!(
                            "unix={} pubkey={} action=manual_rpc_loaded fetch={} subscribed={} persisted={} owner={} data_len={} class={} severity={} classification={} pool={} role={} dex={}",
                            now,
                            pubkey,
                            status,
                            subscribed,
                            persisted,
                            owner,
                            account.data.len(),
                            class,
                            row.severity,
                            row.classification,
                            empty_dash(&row.pool),
                            empty_dash(&row.role),
                            empty_dash(&row.dex)
                        ),
                    );
                } else {
                    upsert_load_line(
                        &mut load_map,
                        &pubkey,
                        format!(
                            "unix={} pubkey={} action=manual_rpc_failed fetch={} severity={} classification={} pool={} role={} dex={}",
                            now,
                            pubkey,
                            status,
                            row.severity,
                            row.classification,
                            empty_dash(&row.pool),
                            empty_dash(&row.role),
                            empty_dash(&row.dex)
                        ),
                    );
                }
                stats.manual_loaded += 1;
                eprintln!(
                    "[problem_account_promoted] pubkey={} action=manual_check fetch={} subscribed={} pool={} role={} dex={}",
                    pubkey,
                    status,
                    subscribed,
                    empty_dash(&row.pool),
                    empty_dash(&row.role),
                    empty_dash(&row.dex)
                );
                continue;
            }

            stats.ignored += 1;
            upsert_load_line(
                &mut load_map,
                &pubkey,
                format!(
                    "unix={} pubkey={} action=noop severity={} classification={} pool={} role={} dex={}",
                    now,
                    pubkey,
                    row.severity,
                    row.classification,
                    empty_dash(&row.pool),
                    empty_dash(&row.role),
                    empty_dash(&row.dex)
                ),
            );
        }

        write_load_txt_map(&output_root.join("load.txt"), &load_map)?;
        Ok(stats)
    }

    /// Spawn the Yellowstone subscription task. Reconnects with exponential
    /// backoff if the stream drops.
    pub fn spawn_subscription(
        &self,
        endpoint: String,
        x_token: String,
        dex_program_ids: Vec<String>,
        extra_accounts: Vec<Pubkey>,
    ) {
        // Record which program ids the owner filter already covers, so
        // note_uncovered_writable can skip redundant per-account subscriptions.
        {
            let mut owner_set = self.dex_owner_set.write().unwrap();
            for id in &dex_program_ids {
                if let Ok(pk) = Pubkey::try_from(id.as_str()) {
                    owner_set.insert(pk);
                }
            }
        }

        let cache = self.inner.clone();
        let stream_slot = self.stream_slot.clone();
        let stream_unix_timestamp = self.stream_unix_timestamp.clone();
        let timestamp_seed_slot = self.timestamp_seed_slot.clone();
        let timestamp_seed_unix = self.timestamp_seed_unix.clone();
        let dynamic_accounts = self.dynamic_accounts.clone();
        let resubscribe_notify = self.resubscribe_notify.clone();
        tokio::spawn(async move {
            let mut backoff = Duration::from_millis(500);
            loop {
                match run_stream(
                    &endpoint,
                    &x_token,
                    &dex_program_ids,
                    &extra_accounts,
                    &cache,
                    &stream_slot,
                    &stream_unix_timestamp,
                    &timestamp_seed_slot,
                    &timestamp_seed_unix,
                    &dynamic_accounts,
                    &resubscribe_notify,
                )
                .await
                {
                    Ok(()) => {
                        warn!("gRPC account stream ended cleanly, reconnecting");
                    }
                    Err(e) => {
                        warn!(error = %e, "gRPC account stream error, reconnecting");
                    }
                }
                tokio::time::sleep(backoff).await;
                backoff = (backoff * 2).min(Duration::from_secs(10));
            }
        });
    }
}


#[derive(Default)]
struct ProblemAccountWatchStats {
    promoted_live: usize,
    manual_loaded: usize,
    ignored: usize,
    errors: usize,
    skipped_seen: usize,
}

#[derive(Debug, Clone)]
struct ProblemCsvRow {
    pubkey: String,
    severity: String,
    classification: String,
    action: String,
    pool: String,
    role: String,
    dex: String,
}

fn split_simple_csv_line(line: &str) -> Vec<String> {
    // Project-generated CSV rows do not contain quoted commas; keep this
    // dependency-free and intentionally simple.
    line.split(',').map(|s| s.trim().to_string()).collect()
}

fn parse_problem_csv_row(line: &str, header: &[String]) -> Option<ProblemCsvRow> {
    let fields = split_simple_csv_line(line);
    if fields.is_empty() {
        return None;
    }

    let by_name = |name: &str| -> Option<String> {
        header
            .iter()
            .position(|h| h == name)
            .and_then(|idx| fields.get(idx).cloned())
    };

    if !header.is_empty() {
        if let (Some(pubkey), Some(severity), Some(classification), Some(action)) = (
            by_name("pubkey"),
            by_name("severity"),
            by_name("classification"),
            by_name("action"),
        ) {
            return Some(ProblemCsvRow {
                pubkey,
                severity,
                classification,
                action,
                pool: by_name("pool").unwrap_or_default(),
                role: by_name("role").unwrap_or_default(),
                dex: by_name("dex").unwrap_or_default(),
            });
        }
    }

    // Compact format:
    // key,pubkey,seen_count,first_seen,last_seen,severity,classification,action,pool,role,dex
    if fields.len() >= 11 {
        return Some(ProblemCsvRow {
            pubkey: fields[1].clone(),
            severity: fields[5].clone(),
            classification: fields[6].clone(),
            action: fields[7].clone(),
            pool: fields[8].clone(),
            role: fields[9].clone(),
            dex: fields[10].clone(),
        });
    }

    // Older format:
    // key,pubkey,severity,classification,action,pool,role,dex
    if fields.len() >= 8 {
        return Some(ProblemCsvRow {
            pubkey: fields[1].clone(),
            severity: fields[2].clone(),
            classification: fields[3].clone(),
            action: fields[4].clone(),
            pool: fields[5].clone(),
            role: fields[6].clone(),
            dex: fields[7].clone(),
        });
    }

    None
}

fn persist_problem_live_account(
    output_root: &Path,
    pubkey: Pubkey,
    account: &Account,
    row: &ProblemCsvRow,
    now: u64,
) -> Result<()> {
    let owner = pubkey_from_address(&account.owner).to_string();
    let data_base64 = base64::Engine::encode(
        &base64::engine::general_purpose::STANDARD,
        account.data.as_slice(),
    );
    let classification = if row.role == "tokenAccountA"
        || row.role == "tokenAccountB"
        || row.role.to_ascii_lowercase().contains("vault")
        || (owner == "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA" && account.data.len() == 165)
    {
        "token_account".to_string()
    } else if row.role == "pool" || row.role == "market" {
        "dex_owned_live".to_string()
    } else if row.classification.is_empty() {
        "problem_report_live".to_string()
    } else {
        row.classification.clone()
    };

    let cache_path = output_root.join("missing_sim_account_cache.json");
    let mut cache_file = read_auto_missing_cache_file(&cache_path);
    let fetched_slot = 0u64;
    let new_record = AutoMissingCachedRecord {
        pubkey: pubkey.to_string(),
        owner: owner.clone(),
        lamports: account.lamports,
        executable: account.executable,
        rent_epoch: account.rent_epoch,
        data_base64,
        data_len: account.data.len(),
        fetched_slot,
        source: "problem_sim_accounts".to_string(),
        classification: classification.clone(),
        is_writable_seen: true,
        last_seen_unix: now,
        seen_count: 1,
        status: "valid".to_string(),
        needs_live_subscription: true,
    };
    if let Some(existing) = cache_file.accounts.iter_mut().find(|r| r.pubkey == new_record.pubkey) {
        let seen_count = existing.seen_count.saturating_add(1);
        *existing = new_record;
        existing.seen_count = seen_count;
    } else {
        cache_file.accounts.push(new_record);
    }
    write_auto_missing_cache_file(&cache_path, &cache_file)?;

    let live_path = output_root.join("missing_sim_account_live.json");
    let mut live_records = read_auto_missing_live_file(&live_path);
    let pk_str = pubkey.to_string();
    if let Some(existing) = live_records.iter_mut().find(|r| r.pubkey == pk_str) {
        existing.owner = owner;
        existing.classification = classification;
        existing.is_writable_seen = true;
    } else {
        live_records.push(AutoMissingLiveRecord {
            pubkey: pk_str,
            owner,
            classification,
            is_writable_seen: true,
            first_seen_unix: now,
        });
    }
    write_auto_missing_live_file(&live_path, &live_records)?;
    Ok(())
}

fn persist_problem_manual_account(output_root: &Path, pubkey: Pubkey, account: &Account) -> Result<()> {
    let path = output_root.join("manual_account_cache.json");
    let mut file = read_manual_account_cache_for_problem_loader(&path);
    let data_base64 = base64::Engine::encode(
        &base64::engine::general_purpose::STANDARD,
        account.data.as_slice(),
    );
    let rec = ManualCachedAccountForProblemLoader {
        pubkey: pubkey.to_string(),
        owner: pubkey_from_address(&account.owner).to_string(),
        lamports: account.lamports,
        executable: account.executable,
        rent_epoch: account.rent_epoch,
        data_base64,
        fetched_slot: None,
        status: "valid".to_string(),
    };
    if let Some(existing) = file.accounts.iter_mut().find(|r| r.pubkey == rec.pubkey) {
        *existing = rec;
    } else {
        file.accounts.push(rec);
    }
    write_manual_account_cache_for_problem_loader(&path, &file)
}

fn read_auto_missing_cache_file(path: &Path) -> AutoMissingCacheFile {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|s| serde_json::from_str::<AutoMissingCacheFile>(&s).ok())
        .filter(|f| f.schema_version == 1)
        .unwrap_or_default()
}

fn write_auto_missing_cache_file(path: &Path, file: &AutoMissingCacheFile) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = AutoMissingCacheFile { schema_version: 1, accounts: file.accounts.clone() };
    file.accounts.sort_by(|a, b| a.pubkey.cmp(&b.pubkey));
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(&file)?)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

fn read_auto_missing_live_file(path: &Path) -> Vec<AutoMissingLiveRecord> {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|s| serde_json::from_str::<Vec<AutoMissingLiveRecord>>(&s).ok())
        .unwrap_or_default()
}

fn write_auto_missing_live_file(path: &Path, records: &[AutoMissingLiveRecord]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut records = records.to_vec();
    records.sort_by(|a, b| a.pubkey.cmp(&b.pubkey));
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(&records)?)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

fn read_manual_account_cache_for_problem_loader(path: &Path) -> ManualAccountCacheFileForProblemLoader {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|s| serde_json::from_str::<ManualAccountCacheFileForProblemLoader>(&s).ok())
        .filter(|f| f.schema_version == 1)
        .unwrap_or_default()
}

fn write_manual_account_cache_for_problem_loader(path: &Path, file: &ManualAccountCacheFileForProblemLoader) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = ManualAccountCacheFileForProblemLoader { schema_version: 1, accounts: file.accounts.clone() };
    file.accounts.sort_by(|a, b| a.pubkey.cmp(&b.pubkey));
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(&file)?)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

fn read_load_txt_map(path: &Path) -> HashMap<String, String> {
    let mut out = HashMap::new();
    let Ok(content) = std::fs::read_to_string(path) else {
        return out;
    };
    for line in content.lines() {
        if let Some(pk) = extract_pubkey_from_load_line(line) {
            out.insert(pk, line.to_string());
        }
    }
    out
}

fn extract_pubkey_from_load_line(line: &str) -> Option<String> {
    for token in line.split_whitespace() {
        if let Some(rest) = token.strip_prefix("pubkey=") {
            if !rest.is_empty() {
                return Some(rest.to_string());
            }
        }
    }
    None
}

fn upsert_load_line(map: &mut HashMap<String, String>, pubkey: &Pubkey, line: String) {
    map.insert(pubkey.to_string(), line);
}

fn write_load_txt_map(path: &Path, map: &HashMap<String, String>) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("cannot create {}", parent.display()))?;
    }
    let mut lines = map.values().cloned().collect::<Vec<_>>();
    lines.sort();
    let tmp = path.with_extension("txt.tmp");
    std::fs::write(&tmp, format!("{}\n", lines.join("\n")))
        .with_context(|| format!("cannot write {}", tmp.display()))?;
    std::fs::rename(&tmp, path)
        .with_context(|| format!("cannot rename {} to {}", tmp.display(), path.display()))?;
    Ok(())
}

fn describe_fetch_result(result: Option<&AccountFetchResult>) -> &'static str {
    match result {
        Some(AccountFetchResult::Found(_)) => "found",
        Some(AccountFetchResult::NotFound) => "not_found",
        Some(AccountFetchResult::Error { .. }) => "error",
        None => "missing_result",
    }
}

fn empty_dash(s: &str) -> &str {
    if s.is_empty() { "-" } else { s }
}

fn pubkey_from_address(address: &Address) -> Pubkey {
    Pubkey::new_from_array(address.to_bytes())
}

fn classify_tx_static_account(account: &Account) -> &'static str {
    let owner = account.owner.to_string();
    if account.executable {
        return "program";
    }
    if owner == "AddressLookupTab1e1111111111111111111111111" {
        return "alt";
    }
    if owner == "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
        || owner == "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"
    {
        if account.data.len() == 165 {
            return "volatile_token_account";
        }
        return "mint";
    }
    if crate::program_registry::PROGRAMS
        .iter()
        .any(|(program_id, _)| *program_id == owner)
    {
        return "volatile_dex_state";
    }
    if account.data.len() <= 512 {
        return "readonly_pda";
    }
    "unknown"
}

fn can_persist_tx_static_data(classification: &str) -> bool {
    matches!(
        classification,
        "alt" | "mint" | "readonly_pda" | "static_config" | "program_config"
    )
}

fn can_load_tx_static_classification(classification: &str) -> bool {
    can_persist_tx_static_data(classification)
}

fn rpc_account_to_cache_account(acct: solana_sdk::account::Account) -> Account {
    Account {
        lamports: acct.lamports,
        data: acct.data,
        owner: Address::from(acct.owner.to_bytes()),
        executable: acct.executable,
        rent_epoch: acct.rent_epoch,
    }
}

pub fn fallback_unix_timestamp() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn estimate_unix_timestamp(slot: u64, seed_slot: u64, seed_unix: i64) -> i64 {
    const SLOT_MS: u128 = 400;
    const SAFETY_SECONDS: i64 = 2;
    if seed_unix <= 0 || slot < seed_slot {
        return fallback_unix_timestamp().saturating_add(SAFETY_SECONDS);
    }

    let delta_slots = (slot - seed_slot) as u128;
    let delta_seconds = ((delta_slots * SLOT_MS) + 999) / 1000;
    seed_unix
        .saturating_add(delta_seconds as i64)
        .saturating_add(SAFETY_SECONDS)
}

fn classify_rpc_error(message: &str) -> &'static str {
    let lower = message.to_ascii_lowercase();
    if lower.contains("429") || lower.contains("too many requests") || lower.contains("rate limit")
    {
        "rate_limited"
    } else if lower.contains("not found") {
        "not_found"
    } else if lower.contains("decode") || lower.contains("deserialize") {
        "decode"
    } else if lower.contains("transport")
        || lower.contains("connection")
        || lower.contains("timeout")
        || lower.contains("request")
    {
        "transport"
    } else {
        "other"
    }
}

fn log_rpc_fetch_reason(reason: &str, pubkeys: &[Pubkey]) {
    let sample = pubkeys
        .iter()
        .take(10)
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    eprintln!(
        "[rpc_fetch_reason] reason={} count={} pubkeys_sample={}",
        reason,
        pubkeys.len(),
        serde_json::to_string(&sample).unwrap_or_else(|_| "[]".to_string())
    );
}

fn build_subscribe_request(
    dex_program_ids: &[String],
    extra_accounts: &[Pubkey],
    dynamic_accounts: &Arc<Mutex<HashSet<Pubkey>>>,
) -> SubscribeRequest {
    let mut accounts_filter: HashMap<String, SubscribeRequestFilterAccounts> = HashMap::new();

    accounts_filter.insert(
        "dex_pools".to_string(),
        SubscribeRequestFilterAccounts {
            account: vec![],
            owner: dex_program_ids.to_vec(),
            filters: vec![],
            nonempty_txn_signature: None,
        },
    );

    // Merge the static startup extras with the runtime-discovered accounts
    // (token vaults etc.) into one specific-account filter.
    let mut specific: HashSet<Pubkey> = extra_accounts.iter().copied().collect();
    {
        let dyn_set = dynamic_accounts.lock().unwrap();
        specific.extend(dyn_set.iter().copied());
    }
    if !specific.is_empty() {
        accounts_filter.insert(
            "extras".to_string(),
            SubscribeRequestFilterAccounts {
                account: specific.iter().map(|p| p.to_string()).collect(),
                owner: vec![],
                filters: vec![],
                nonempty_txn_signature: None,
            },
        );
    }

    SubscribeRequest {
        slots: HashMap::new(),
        accounts: accounts_filter,
        transactions: HashMap::new(),
        transactions_status: HashMap::new(),
        entry: HashMap::new(),
        blocks: HashMap::new(),
        blocks_meta: HashMap::new(),
        commitment: Some(CommitmentLevel::Processed as i32),
        accounts_data_slice: vec![],
        ping: None,
        from_slot: None,
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_stream(
    endpoint: &str,
    x_token: &str,
    dex_program_ids: &[String],
    extra_accounts: &[Pubkey],
    cache: &Arc<DashMap<Pubkey, Account>>,
    stream_slot: &Arc<AtomicU64>,
    stream_unix_timestamp: &Arc<AtomicI64>,
    timestamp_seed_slot: &Arc<AtomicU64>,
    timestamp_seed_unix: &Arc<AtomicI64>,
    dynamic_accounts: &Arc<Mutex<HashSet<Pubkey>>>,
    resubscribe_notify: &Arc<Notify>,
) -> Result<()> {
    let mut client = GeyserGrpcClient::build_from_shared(endpoint.to_string())?
        .x_token(Some(x_token.to_string()))?
        .tls_config(yellowstone_grpc_client::ClientTlsConfig::new().with_native_roots())?
        .max_decoding_message_size(64 * 1024 * 1024)
        .connect()
        .await
        .context("gRPC connect failed")?;

    info!(endpoint, "gRPC connected");

    let request = build_subscribe_request(dex_program_ids, extra_accounts, dynamic_accounts);

    let (mut tx, mut stream) = client
        .subscribe_with_request(Some(request))
        .await
        .context("gRPC subscribe failed")?;

    info!("gRPC subscription active; waiting for account updates");

    let mut count: u64 = 0;
    // P4 watchdog: track stream liveness. If the slot stops advancing while
    // we keep simulating, the cache is silently frozen — warn loudly.
    let mut watchdog = tokio::time::interval(Duration::from_secs(10));
    watchdog.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    watchdog.tick().await; // drain immediate first tick
    let mut last_watchdog_slot: u64 = stream_slot.load(Ordering::Relaxed);
    let mut last_watchdog_count: u64 = 0;

    loop {
        tokio::select! {
            biased;
            // Re-subscribe when new dynamic accounts are added.
            _ = resubscribe_notify.notified() => {
                let dyn_len = dynamic_accounts.lock().unwrap().len();
                let req = build_subscribe_request(dex_program_ids, extra_accounts, dynamic_accounts);
                match tx.send(req).await {
                    Ok(()) => eprintln!(
                        "[grpc_resubscribe] dynamic_accounts={} status=sent",
                        dyn_len
                    ),
                    Err(e) => {
                        warn!(error = %e, "gRPC re-subscribe failed; reconnecting");
                        return Err(anyhow::anyhow!("re-subscribe send failed: {e}"));
                    }
                }
            }
            _ = watchdog.tick() => {
                let now_slot = stream_slot.load(Ordering::Relaxed);
                let updates = count.saturating_sub(last_watchdog_count);
                if now_slot <= last_watchdog_slot || updates == 0 {
                    warn!(
                        cache_slot = now_slot,
                        last_slot = last_watchdog_slot,
                        updates_in_window = updates,
                        "[stream_stalled] Yellowstone slot not advancing; cache may be serving frozen pool state"
                    );
                    eprintln!(
                        "[stream_stalled] cache_slot={} last_slot={} updates_in_window={} action=stream_frozen_check_endpoint",
                        now_slot, last_watchdog_slot, updates
                    );
                } else {
                    eprintln!(
                        "[stream_health] cache_slot={} slot_advanced={} updates_in_window={} cache_size={}",
                        now_slot,
                        now_slot.saturating_sub(last_watchdog_slot),
                        updates,
                        cache.len()
                    );
                }
                last_watchdog_slot = now_slot;
                last_watchdog_count = count;
            }
            msg = stream.next() => {
                let Some(msg) = msg else { break };
                let msg = msg.context("stream yielded error")?;
                match msg.update_oneof {
                    Some(UpdateOneof::Account(a)) => {
                        stream_slot.store(a.slot, Ordering::Relaxed);
                        let seed_slot = timestamp_seed_slot.load(Ordering::Relaxed);
                        let seed_unix = timestamp_seed_unix.load(Ordering::Relaxed);
                        stream_unix_timestamp.store(
                            estimate_unix_timestamp(a.slot, seed_slot, seed_unix),
                            Ordering::Relaxed,
                        );

                        if let Some(info) = a.account {
                            let pk = match Pubkey::try_from(info.pubkey.as_slice()) {
                                Ok(p) => p,
                                Err(_) => continue,
                            };
                            let owner_bytes: [u8; 32] = info.owner.as_slice()
                                .try_into()
                                .unwrap_or([0u8; 32]);
                            let account = Account {
                                lamports: info.lamports,
                                data: info.data,
                                owner: Address::from(owner_bytes),
                                executable: info.executable,
                                rent_epoch: info.rent_epoch,
                            };
                            cache.insert(pk, account);
                            count += 1;
                            if count % 10_000 == 0 {
                                debug!(count, size = cache.len(), "cache growth");
                            }
                        }
                    }
                    Some(UpdateOneof::Ping(_)) => {
                        let _ = tx
                            .send(SubscribeRequest {
                                ping: Some(SubscribeRequestPing { id: 1 }),
                                ..Default::default()
                            })
                            .await;
                    }
                    _ => {}
                }
            }
        }
    }
    Ok(())
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
