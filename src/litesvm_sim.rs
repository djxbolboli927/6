//! Local LiteSVM simulation gate backed by a live account cache.
//!
//! Uses the vendored LiteSVM source (vendor/litesvm, GitHub master) rather
//! than a crates.io release.  Key capabilities used from LiteSVM 0.11:
//!
//! • `with_mainnet_features()` — activates every Solana feature gate live on
//!   mainnet-beta.  PMM DEXes (Tessera, SolFi, ZeroFi) rely on post-2.0
//!   features; without this they silently mis-execute or revert unexpectedly.
//!
//! • `warp_to_slot(slot)` — atomically advances Clock.slot, Clock.epoch,
//!   SlotHashes, and EpochSchedule to the live Yellowstone slot.  PMM DEXes
//!   check Clock.slot for price-staleness; the old manual sysvar approach
//!   could leave SlotHashes stale, causing oracle checks to mis-fire.
//!
//! • `with_default_programs()` — loads the full SPL + built-in program set
//!   (replaces the removed `with_spl_programs()` from earlier versions).
//!
//! • CPI bug-fixes — 0.9+ resolved return-data propagation across CPI hops,
//!   eliminating false-negative reverts on multi-hop routes.
//!
//! ## Type-conversion boundary
//!
//! Our bot is built on `solana-sdk 2.2` (monolithic SDK, uses
//! `solana-transaction 2.x` internally).  LiteSVM 0.11 uses the newer
//! granular crates (`solana-transaction 3.x`, `solana-address 2.x`).
//! Both share the same on-wire binary format (Solana maintains wire-format
//! compatibility across major releases), so the `to_litesvm_tx` adapter
//! does a cheap bincode round-trip at the simulate() call site.
//! Account state (solana-account 3.2.0) is shared by both and needs no
//! conversion.

use anyhow::{anyhow, Context, Result};
use litesvm::LiteSVM;
use solana_account::{Account, ReadableAccount};
use solana_address::Address as LsAddr;
use solana_clock::Clock;
#[allow(deprecated)]
use solana_sdk::{
    address_lookup_table::{
        self,
        state::{AddressLookupTable, LookupTableMeta},
        AddressLookupTableAccount,
    },
    message::{MessageHeader, VersionedMessage},
    pubkey::Pubkey,
    system_instruction,
    system_program,
    transaction::VersionedTransaction,
};
use std::borrow::Cow;
use std::collections::{hash_map::DefaultHasher, HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicI64, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use tracing::{debug, info, warn};

use crate::account_cache::{AccountCache, AccountFetchResult};
use crate::manual_sim_accounts::{self, RuntimeMissingAccount};
use crate::metrics::Metrics;

const ASSOCIATED_TOKEN_PROGRAM_ID: Pubkey =
    Pubkey::from_str_const("ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL");
const JUPITER_PROGRAM_ID: Pubkey =
    Pubkey::from_str_const("JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4");
const ALPHAQ_PROGRAM_ID: Pubkey =
    Pubkey::from_str_const("ALPHAQmeA7bjrVuccPsYPiCvsi428SNwte66Srvs4pHA");
const WHIRLPOOL_PROGRAM_ID: Pubkey =
    Pubkey::from_str_const("whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc");
const SOLFI_V2_PROGRAM_ID: Pubkey =
    Pubkey::from_str_const("SV2EYYJyRz2YhfXwXnhNAevDEui5Q6yrfyo13WtupPF");
const MAX_RPC_SIM_COMPARE_PER_PROCESS: u64 = 1_000;
const MAX_RPC_SNAPSHOT_RETRY_PER_PROCESS: u64 = u64::MAX;
const MAX_SIM_CLOCK_LOGS_PER_PROCESS: u64 = 100;
static RPC_SIM_COMPARE_COUNT: AtomicU64 = AtomicU64::new(0);
static RPC_SNAPSHOT_RETRY_COUNT: AtomicU64 = AtomicU64::new(0);
static SIM_CLOCK_LOG_COUNT: AtomicU64 = AtomicU64::new(0);

pub struct SimOutcome {
    pub compute_units: u64,
    pub wsol_before: u64,
    pub wsol_after: u64,
}

pub struct Simulator {
    svm: Mutex<LiteSVM>,
    wsol_ata: Pubkey,
    payer_pubkey: Pubkey,
    loaded_programs: HashSet<Pubkey>,
    loaded_program_files: HashMap<Pubkey, String>,
    /// Program bytecode kept in memory so the RPC-snapshot retry path can build
    /// a fresh LiteSVM without re-reading ~28 .so files from disk on every
    /// failed simulation (the retry runs per failure now that the cap is gone).
    loaded_program_bytes: HashMap<Pubkey, Vec<u8>>,
    jito_tip_accounts: HashSet<Pubkey>,
    fail_closed: bool,
    /// Live mainnet slot from the Yellowstone gRPC stream (zero RPC).
    current_slot: Arc<AtomicU64>,
    current_unix_timestamp: Arc<AtomicI64>,
    allow_hot_path_rpc_fetch: bool,
    manual_accounts_root: PathBuf,
    /// Role map extracted from mix.json. This is not used to fabricate price
    /// state. It is used only for diagnostics and for deciding whether a
    /// stale/missing account is live pool/vault state (gRPC) or a static/
    /// authority account (manual/static/synthetic).
    mix_roles: Arc<MixRoleMap>,
    missing_handle: Option<Arc<crate::auto_missing_accounts::AutoMissingAccountsHandle>>,
}

#[derive(Clone, Copy, Debug)]
struct TxAccountMeta {
    pubkey: Pubkey,
    is_signer: bool,
    is_writable: bool,
    source: TxAccountSource,
    /// Resolved transaction-key index after v0 ALT expansion. This is the
    /// position the program actually indexes in the message. Keeping it in
    /// logs is critical because an address can be correct but appear in the
    /// wrong role/position for a CPI.
    tx_key_index: Option<usize>,
    /// For ALT accounts: address lookup table account and index inside it.
    /// Static keys leave these as None.
    alt_table: Option<Pubkey>,
    alt_index: Option<u8>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TxAccountSource {
    Static,
    AltWritable,
    AltReadonly,
}

impl TxAccountSource {
    fn as_str(self) -> &'static str {
        match self {
            TxAccountSource::Static => "static",
            TxAccountSource::AltWritable => "alt_writable",
            TxAccountSource::AltReadonly => "alt_readonly",
        }
    }
}

#[derive(Clone, Debug)]
struct MixRoleInfo {
    dex: String,
    pool_pubkey: Option<Pubkey>,
    pool_owner: Option<Pubkey>,
    role: String,
    account_kind: String,
    recommended_action: String,
    manual_static_allowed: bool,
    source_path: String,
}

type MixRoleMap = HashMap<Pubkey, Vec<MixRoleInfo>>;

impl MixRoleInfo {
    fn dex_or_unknown(&self) -> &str {
        if self.dex.is_empty() { "unknown" } else { &self.dex }
    }

    fn pool_pubkey_string(&self) -> String {
        self.pool_pubkey.map(|p| p.to_string()).unwrap_or_else(|| "".to_string())
    }

    fn pool_owner_string(&self) -> String {
        self.pool_owner.map(|p| p.to_string()).unwrap_or_else(|| "".to_string())
    }
}


// ── Type-conversion helpers at the solana-sdk 2.x / LiteSVM 3.x boundary ───

/// Convert a solana-sdk 2.x `Pubkey` to the `solana-address 2.x` `Address`
/// type that LiteSVM 0.11 uses for all account-lookup APIs.
/// Both types are `[u8; 32]` wrappers; the conversion is a byte-level copy.
#[inline]
fn pk_to_addr(pk: Pubkey) -> LsAddr {
    LsAddr::from(pk.to_bytes())
}

fn resolve_program_file(so_dir: &str, fname: &str) -> Option<std::path::PathBuf> {
    let exact = Path::new(so_dir).join(fname);
    if exact.exists() {
        return Some(exact);
    }

    let wanted = normalize_filename(fname);
    let entries = std::fs::read_dir(so_dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("so") {
            continue;
        }
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if normalize_filename(name) == wanted {
            return Some(path);
        }
    }
    None
}

fn resolve_program_file_for_program(
    so_dir: &str,
    program_id: &str,
    fname: &str,
) -> Option<std::path::PathBuf> {
    if let Some(path) = resolve_program_file(so_dir, fname) {
        return Some(path);
    }

    let id_named = Path::new(so_dir).join(format!("{program_id}.so"));
    if id_named.exists() {
        return Some(id_named);
    }
    None
}

fn normalize_filename(name: &str) -> String {
    name.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .flat_map(|c| c.to_lowercase())
        .collect()
}

/// Translate a `solana-sdk 2.x` `VersionedTransaction` to the
/// `solana-transaction 3.x` type expected by `LiteSVM::simulate_transaction`.
///
/// Wire format is identical across Solana major releases, so a bincode
/// round-trip is a safe zero-semantic-change conversion.  Cost: one heap
/// allocation (~few hundred bytes) per simulation — negligible vs. the sim.
fn to_litesvm_tx(
    tx: &VersionedTransaction,
) -> Result<solana_transaction::versioned::VersionedTransaction> {
    let bytes = bincode::serialize(tx).context("serialize tx for litesvm boundary")?;
    bincode::deserialize(&bytes).context("deserialize tx for litesvm boundary")
}


fn load_mix_role_map(manual_accounts_root: &Path) -> Arc<MixRoleMap> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    candidates.push(manual_accounts_root.join("metis/1/mix.json"));
    candidates.push(manual_accounts_root.join("mix.json"));
    if let Some(parent) = manual_accounts_root.parent() {
        candidates.push(parent.join("metis/1/mix.json"));
        candidates.push(parent.join("mix.json"));
    }
    candidates.push(PathBuf::from("/root/s/metis/1/mix.json"));
    candidates.push(PathBuf::from("/root/s/mix.json"));

    let mut map: MixRoleMap = HashMap::new();
    for path in candidates {
        if !path.exists() {
            continue;
        }
        match std::fs::read_to_string(&path)
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        {
            Some(json) => {
                collect_mix_roles_from_json(&json, &path, &mut map);
                eprintln!(
                    "[mix_role_map_source] path={} accounts={} status=parsed",
                    path.display(),
                    map.len()
                );
                break;
            }
            None => eprintln!(
                "[mix_role_map_source] path={} accounts=0 status=parse_failed",
                path.display()
            ),
        }
    }
    Arc::new(map)
}

fn collect_mix_roles_from_json(json: &serde_json::Value, path: &Path, map: &mut MixRoleMap) {
    let pools = json
        .get("pools")
        .or_else(|| json.get("Pools"))
        .and_then(|v| v.as_array())
        .or_else(|| json.as_array());
    let Some(pools) = pools else { return; };

    for pool in pools {
        let Some(obj) = pool.as_object() else { continue; };
        let pool_pubkey = obj
            .get("pubkey")
            .and_then(|v| v.as_str())
            .and_then(|s| Pubkey::try_from(s).ok());
        let pool_owner = obj
            .get("owner")
            .and_then(|v| v.as_str())
            .and_then(|s| Pubkey::try_from(s).ok());
        let dex = obj
            .get("dex")
            .or_else(|| obj.get("label"))
            .or_else(|| obj.get("name"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .unwrap_or_else(|| pool_owner.map(|p| p.to_string()).unwrap_or_default());

        if let Some(pk) = pool_pubkey {
            insert_mix_role(
                map,
                pk,
                MixRoleInfo {
                    dex: dex.clone(),
                    pool_pubkey,
                    pool_owner,
                    role: "pool".to_string(),
                    account_kind: "live_pool_state".to_string(),
                    recommended_action: "owner_filter_grpc_or_direct_grpc".to_string(),
                    manual_static_allowed: false,
                    source_path: "pubkey".to_string(),
                },
            );
        }

        if let Some(params) = obj.get("params") {
            let mut stack = vec!["params".to_string()];
            collect_mix_roles_recursive(params, &mut stack, &dex, pool_pubkey, pool_owner, map);
        }
        let mut stack = Vec::new();
        for (k, v) in obj {
            if k == "params" || k == "pubkey" || k == "owner" { continue; }
            stack.push(k.clone());
            collect_mix_roles_recursive(v, &mut stack, &dex, pool_pubkey, pool_owner, map);
            stack.pop();
        }
        let _ = path;
    }
}

fn collect_mix_roles_recursive(
    value: &serde_json::Value,
    path_stack: &mut Vec<String>,
    dex: &str,
    pool_pubkey: Option<Pubkey>,
    pool_owner: Option<Pubkey>,
    map: &mut MixRoleMap,
) {
    match value {
        serde_json::Value::String(s) => {
            if let Ok(pk) = Pubkey::try_from(s.as_str()) {
                let role = path_stack
                    .last()
                    .cloned()
                    .unwrap_or_else(|| "unknown".to_string());
                let (kind, action, manual_static) = classify_mix_role(&role, path_stack);
                insert_mix_role(
                    map,
                    pk,
                    MixRoleInfo {
                        dex: dex.to_string(),
                        pool_pubkey,
                        pool_owner,
                        role,
                        account_kind: kind,
                        recommended_action: action,
                        manual_static_allowed: manual_static,
                        source_path: path_stack.join("."),
                    },
                );
            }
        }
        serde_json::Value::Array(items) => {
            for (idx, item) in items.iter().enumerate() {
                path_stack.push(idx.to_string());
                collect_mix_roles_recursive(item, path_stack, dex, pool_pubkey, pool_owner, map);
                path_stack.pop();
            }
        }
        serde_json::Value::Object(obj) => {
            for (key, child) in obj {
                path_stack.push(key.clone());
                collect_mix_roles_recursive(child, path_stack, dex, pool_pubkey, pool_owner, map);
                path_stack.pop();
            }
        }
        _ => {}
    }
}

fn insert_mix_role(map: &mut MixRoleMap, pk: Pubkey, info: MixRoleInfo) {
    let entry = map.entry(pk).or_default();
    if !entry.iter().any(|old| old.role == info.role && old.pool_pubkey == info.pool_pubkey) {
        entry.push(info);
    }
}

fn classify_mix_role(role: &str, path: &[String]) -> (String, String, bool) {
    let role_l = role.to_ascii_lowercase();
    let joined = path.join(".").to_ascii_lowercase();
    if role_l == "owner" || joined.contains("program") {
        return ("program_or_sysvar".to_string(), "ignore_compare".to_string(), false);
    }
    if joined.contains("addresslookuptable") || role_l.contains("alt") || joined.contains("lookup_table") {
        return ("alt_static".to_string(), "startup_fetch_or_alt_cache".to_string(), false);
    }
    if joined.contains("tokenaccount") || joined.contains("vault") || joined.contains("reserve") {
        return (
            "live_vault_token_account".to_string(),
            "direct_grpc_subscribe_and_startup_fetch".to_string(),
            false,
        );
    }
    if joined.contains("oracle") || joined.contains("observation") || joined.contains("tick") || joined.contains("bin") || joined.contains("orderbook") {
        return (
            "live_oracle_tick_bin_state".to_string(),
            "direct_grpc_subscribe_and_startup_fetch".to_string(),
            false,
        );
    }
    if role_l == "pubkey" || role_l == "pool" || role_l == "ammkey" || role_l.contains("market") {
        return (
            "live_pool_state".to_string(),
            "owner_filter_grpc_or_direct_grpc".to_string(),
            false,
        );
    }
    if joined.contains("authority") || role_l.contains("authpda") || role_l.contains("authority") {
        return (
            "readonly_authority_pda".to_string(),
            "allow_synthetic_readonly_or_manual_static".to_string(),
            true,
        );
    }
    if joined.contains("mint") || joined.contains("tokenment") {
        return ("mint_static".to_string(), "manual_static_or_startup_rpc".to_string(), true);
    }
    (
        "unknown_param".to_string(),
        "write_problem_report_for_manual_review".to_string(),
        false,
    )
}

fn best_mix_role<'a>(roles: &'a [MixRoleInfo], meta: &TxAccountMeta) -> Option<&'a MixRoleInfo> {
    if roles.is_empty() { return None; }
    if meta.is_writable {
        roles.iter().find(|r| r.account_kind.starts_with("live_")).or_else(|| roles.first())
    } else {
        roles.iter().find(|r| r.account_kind.contains("authority")).or_else(|| roles.first())
    }
}

fn json_str_array(items: &[String]) -> serde_json::Value {
    serde_json::Value::Array(items.iter().map(|s| serde_json::Value::String(s.clone())).collect())
}

fn parse_json_array_field(text: &str) -> Vec<String> {
    serde_json::from_str::<Vec<String>>(text).unwrap_or_default()
}

fn write_problem_sim_account(
    sim: &Simulator,
    meta: &TxAccountMeta,
    route_sig: u128,
    route_labels: &str,
    route_programs: &str,
    failed_program: &Pubkey,
    cache_state: &AccountDebugState,
    rpc_state: &AccountDebugState,
    outcome: &AccountCompareOutcome,
    classification: &str,
    action: &str,
) {
    if is_known_sysvar(&meta.pubkey) || is_builtin_program(&meta.pubkey) || sim.loaded_programs.contains(&meta.pubkey) {
        return;
    }
    let roles = sim.mix_roles.get(&meta.pubkey).cloned().unwrap_or_default();
    let role = best_mix_role(&roles, meta);
    let account_kind = role
        .map(|r| r.account_kind.clone())
        .unwrap_or_else(|| classification.to_string());
    let recommended_action = role
        .map(|r| r.recommended_action.clone())
        .unwrap_or_else(|| action.to_string());
    let dex = role.map(|r| r.dex_or_unknown().to_string()).unwrap_or_else(|| "unknown".to_string());
    let pool_pubkey = role.map(|r| r.pool_pubkey_string()).unwrap_or_default();
    let pool_owner = role.map(|r| r.pool_owner_string()).unwrap_or_default();
    let pool_role = role.map(|r| r.role.clone()).unwrap_or_else(|| "unknown".to_string());
    let source_path = role.map(|r| r.source_path.clone()).unwrap_or_default();
    let manual_static_allowed = role.map(|r| r.manual_static_allowed).unwrap_or(false);

    let severity = if classification == "live_state_stale" || account_kind.starts_with("live_") {
        "needs_grpc_live_state"
    } else if recommended_action.contains("manual") || manual_static_allowed {
        "needs_manual_static_or_synthetic"
    } else if !outcome.matches_rpc {
        "needs_manual_check"
    } else {
        "info"
    };

    let now = unix_now_for_problem_report();
    // Stable key: do NOT include route_sig. A single live/state account can appear
    // in thousands of profitable route attempts overnight.  We aggregate it once
    // and keep counters + route samples instead of appending a new CSV row per hit.
    let key = format!("{}:{}:{}:{}", meta.pubkey, pool_pubkey, pool_role, classification);
    let current_route_sig_hex = format!("{:032x}", route_sig);
    // Do not use serde_json::json! here.  This record has many fields and the
    // macro can hit rustc's macro-expansion recursion limit in release builds.
    // Build the Value explicitly instead.
    let mut rec_obj = serde_json::Map::new();
    rec_obj.insert("key".to_string(), serde_json::Value::String(key.clone()));
    rec_obj.insert("first_seen_unix".to_string(), serde_json::Value::Number(serde_json::Number::from(now)));
    rec_obj.insert("last_seen_unix".to_string(), serde_json::Value::Number(serde_json::Number::from(now)));
    rec_obj.insert("seen_count".to_string(), serde_json::Value::Number(serde_json::Number::from(1u64)));
    rec_obj.insert("pubkey".to_string(), serde_json::Value::String(meta.pubkey.to_string()));
    rec_obj.insert("dex".to_string(), serde_json::Value::String(dex.clone()));
    rec_obj.insert("pool_pubkey".to_string(), serde_json::Value::String(pool_pubkey.clone()));
    rec_obj.insert("pool_owner".to_string(), serde_json::Value::String(pool_owner));
    rec_obj.insert("pool_role".to_string(), serde_json::Value::String(pool_role.clone()));
    rec_obj.insert("role_source_path".to_string(), serde_json::Value::String(source_path));
    rec_obj.insert("account_kind".to_string(), serde_json::Value::String(account_kind));
    rec_obj.insert("last_route_sig".to_string(), serde_json::Value::String(current_route_sig_hex.clone()));
    rec_obj.insert(
        "route_sig_samples".to_string(),
        serde_json::Value::Array(vec![serde_json::Value::String(current_route_sig_hex)]),
    );
    let route_labels_vec = parse_json_array_field(route_labels);
    rec_obj.insert("last_route_labels".to_string(), json_str_array(&route_labels_vec));
    rec_obj.insert("route_labels_samples".to_string(), serde_json::Value::Array(vec![json_str_array(&route_labels_vec)]));
    rec_obj.insert("programs".to_string(), json_str_array(&parse_json_array_field(route_programs)));
    rec_obj.insert("failed_program".to_string(), serde_json::Value::String(failed_program.to_string()));
    rec_obj.insert("lite_reason".to_string(), serde_json::Value::String(outcome.reason.to_string()));
    rec_obj.insert("account_source".to_string(), serde_json::Value::String(meta.source.as_str().to_string()));
    rec_obj.insert("is_writable".to_string(), serde_json::Value::Bool(meta.is_writable));
    rec_obj.insert("is_signer".to_string(), serde_json::Value::Bool(meta.is_signer));
    rec_obj.insert("tx_key_index".to_string(), serde_json::to_value(meta.tx_key_index).unwrap_or(serde_json::Value::Null));
    rec_obj.insert(
        "alt_table".to_string(),
        meta.alt_table
            .map(|p| serde_json::Value::String(p.to_string()))
            .unwrap_or(serde_json::Value::Null),
    );
    rec_obj.insert("alt_index".to_string(), serde_json::to_value(meta.alt_index).unwrap_or(serde_json::Value::Null));
    rec_obj.insert("cache_status".to_string(), serde_json::Value::String(cache_state.status.clone()));
    rec_obj.insert("rpc_status".to_string(), serde_json::Value::String(rpc_state.status.clone()));
    rec_obj.insert("owner_cache".to_string(), serde_json::Value::String(cache_state.owner.clone()));
    rec_obj.insert("owner_rpc".to_string(), serde_json::Value::String(rpc_state.owner.clone()));
    rec_obj.insert("data_len_cache".to_string(), serde_json::Value::String(cache_state.data_len.clone()));
    rec_obj.insert("data_len_rpc".to_string(), serde_json::Value::String(rpc_state.data_len.clone()));
    rec_obj.insert("lamports_cache".to_string(), serde_json::Value::String(cache_state.lamports.clone()));
    rec_obj.insert("lamports_rpc".to_string(), serde_json::Value::String(rpc_state.lamports.clone()));
    rec_obj.insert("data_hash_cache".to_string(), serde_json::Value::String(cache_state.data_hash.clone()));
    rec_obj.insert("data_hash_rpc".to_string(), serde_json::Value::String(rpc_state.data_hash.clone()));
    rec_obj.insert("owner_match".to_string(), serde_json::Value::Bool(outcome.owner_match));
    rec_obj.insert("data_len_match".to_string(), serde_json::Value::Bool(outcome.data_len_match));
    rec_obj.insert("lamports_match".to_string(), serde_json::Value::Bool(outcome.lamports_match));
    rec_obj.insert("data_hash_match".to_string(), serde_json::Value::Bool(outcome.data_hash_match));
    rec_obj.insert("classification".to_string(), serde_json::Value::String(classification.to_string()));
    rec_obj.insert("recommended_action".to_string(), serde_json::Value::String(recommended_action));
    rec_obj.insert("manual_static_allowed".to_string(), serde_json::Value::Bool(manual_static_allowed));
    rec_obj.insert("severity".to_string(), serde_json::Value::String(severity.to_string()));
    let rec = serde_json::Value::Object(rec_obj);
    append_problem_record(&sim.manual_accounts_root.join("problem_sim_accounts.json"), rec);
    upsert_problem_csv(&sim.manual_accounts_root.join("problem_sim_accounts.csv"), &key, &meta.pubkey.to_string(), severity, classification, action, &pool_pubkey, &pool_role, &dex, now);

    // Minimal operator report requested by the user. This file is not read by
    // the loader, so it can stay intentionally small and human-readable.
    upsert_bad_account_review_csv(
        &sim.manual_accounts_root.join("bad_accounts_review.csv"),
        &meta.pubkey.to_string(),
        &pool_pubkey,
        &pool_role,
    );
}

fn unix_now_for_problem_report() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// True when a simulation revert is a Jupiter slippage abort (custom error
/// 6001 / 0x1771 = SlippageToleranceExceeded). These are expected price moves,
/// not account/loading problems, so the operator asked to exclude them from the
/// dedicated error file.
fn is_slippage_revert(lite_err: &str) -> bool {
    lite_err.contains("Custom(6001)") || lite_err.contains("0x1771")
}

/// Append a simulation error to `sim_errors.log`, but ONLY the first time a
/// given (failed_program + error-kind + culprit-account) combination is seen.
/// Slippage reverts are skipped entirely. The dedup set lives in a sidecar
/// `sim_errors_seen.txt` so it survives across the run without holding every
/// line in memory. The visible log stays small and human-readable: one line per
/// genuinely new error, which is exactly what the operator wants to send back.
fn log_new_sim_error(
    output_root: &Path,
    route_sig: u128,
    route_labels: &str,
    route_programs: &str,
    lite_err: &str,
    failed_program: Option<&Pubkey>,
    culprit_account: Option<&Pubkey>,
) {
    // Skip slippage — it is an expected price move, not a sim/account bug.
    if is_slippage_revert(lite_err) {
        return;
    }

    // Normalise the error string into a stable "kind" so price/amount specifics
    // do not make every revert look unique. We keep the InstructionError shape
    // (e.g. "InstructionError(5, InvalidAccountOwner)") which is what matters
    // for diagnosis, and drop any trailing detail.
    let err_kind = lite_err
        .split(" logs=")
        .next()
        .unwrap_or(lite_err)
        .trim()
        .to_string();

    let program_str = failed_program
        .map(|p| p.to_string())
        .unwrap_or_else(|| "unknown".to_string());
    let account_str = culprit_account
        .map(|p| p.to_string())
        .unwrap_or_else(|| "none".to_string());

    // Dedup key: same failing program + same error kind + same culprit account
    // is considered "the same error" and logged only once.
    let dedup_key = format!("{program_str}|{err_kind}|{account_str}");

    let seen_path = output_root.join("sim_errors_seen.txt");
    let log_path = output_root.join("sim_errors.log");
    if let Some(parent) = log_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    // Check the sidecar dedup file. If the key is already there, do nothing.
    if let Ok(seen) = std::fs::read_to_string(&seen_path) {
        if seen.lines().any(|l| l.trim() == dedup_key) {
            return;
        }
    }

    // Record the key, then append a single human-readable line to the log.
    {
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&seen_path)
        {
            let _ = writeln!(f, "{dedup_key}");
        }
    }

    let now = unix_now_for_problem_report();
    let line = format!(
        "unix={now} route_sig={route_sig:032x} failed_program={program_str} culprit_account={account_str} route_labels={route_labels} programs={route_programs} error={err_kind}\n"
    );
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
    {
        let _ = f.write_all(line.as_bytes());
    }
}

fn append_problem_record(path: &Path, mut rec: serde_json::Value) {
    if let Some(parent) = path.parent() { let _ = std::fs::create_dir_all(parent); }
    let key = rec.get("key").and_then(|v| v.as_str()).unwrap_or_default().to_string();
    let mut items = match std::fs::read(path) {
        Ok(bytes) => serde_json::from_slice::<Vec<serde_json::Value>>(&bytes).unwrap_or_default(),
        Err(_) => Vec::new(),
    };

    if let Some(old) = items.iter_mut().find(|old| old.get("key").and_then(|v| v.as_str()) == Some(key.as_str())) {
        let first_seen = old.get("first_seen_unix").cloned();
        let seen_count = old.get("seen_count").and_then(|v| v.as_u64()).unwrap_or(1).saturating_add(1);
        if let Some(first_seen) = first_seen { rec["first_seen_unix"] = first_seen; }
        rec["seen_count"] = serde_json::Value::Number(seen_count.into());

        // Keep a small bounded sample of routes where this account appeared.
        // This gives us enough context for manual diagnosis without making the file
        // grow by thousands of duplicate rows.
        let mut route_samples = old
            .get("route_sig_samples")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        if let Some(new_route) = rec.get("last_route_sig").cloned() {
            if !route_samples.iter().any(|v| v == &new_route) {
                route_samples.push(new_route);
                if route_samples.len() > 20 {
                    route_samples.remove(0);
                }
            }
        }
        rec["route_sig_samples"] = serde_json::Value::Array(route_samples);

        let mut label_samples = old
            .get("route_labels_samples")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();
        if let Some(new_labels) = rec.get("last_route_labels").cloned() {
            if !label_samples.iter().any(|v| v == &new_labels) {
                label_samples.push(new_labels);
                if label_samples.len() > 10 {
                    label_samples.remove(0);
                }
            }
        }
        rec["route_labels_samples"] = serde_json::Value::Array(label_samples);
        *old = rec;
    } else {
        items.push(rec);
    }
    let tmp = path.with_extension("json.tmp");
    if let Ok(bytes) = serde_json::to_vec_pretty(&items) {
        let _ = std::fs::write(&tmp, bytes).and_then(|_| std::fs::rename(&tmp, path));
    }
}

fn upsert_problem_csv(path: &Path, key: &str, pubkey: &str, severity: &str, classification: &str, action: &str, pool: &str, role: &str, dex: &str, now: u64) {
    if let Some(parent) = path.parent() { let _ = std::fs::create_dir_all(parent); }

    // The old implementation appended every hit and could create millions of lines.
    // Rebuild the CSV as a de-duplicated key->row table.  This also compacts an
    // already-bloated CSV the first time the patched bot writes to it.
    let mut rows: HashMap<String, Vec<String>> = HashMap::new();
    if let Ok(text) = std::fs::read_to_string(path) {
        for line in text.lines().skip(1) {
            if line.trim().is_empty() { continue; }
            let cols: Vec<String> = line.split(',').map(|s| s.to_string()).collect();
            if cols.is_empty() { continue; }
            let existing_key = cols[0].clone();
            rows.entry(existing_key).or_insert(cols);
        }
    }

    let mut seen_count = 1u64;
    let first_seen = match rows.get(key) {
        Some(cols) if cols.len() >= 4 => {
            seen_count = cols.get(2).and_then(|s| s.parse::<u64>().ok()).unwrap_or(1).saturating_add(1);
            cols.get(3).cloned().unwrap_or_else(|| now.to_string())
        }
        Some(_) => {
            seen_count = 2;
            now.to_string()
        }
        None => now.to_string(),
    };

    rows.insert(key.to_string(), vec![
        key.to_string(),
        pubkey.to_string(),
        seen_count.to_string(),
        first_seen,
        now.to_string(),
        severity.to_string(),
        classification.to_string(),
        action.to_string(),
        pool.to_string(),
        role.to_string(),
        dex.to_string(),
    ]);

    let mut keys: Vec<String> = rows.keys().cloned().collect();
    keys.sort();
    let mut out = String::from("key,pubkey,seen_count,first_seen_unix,last_seen_unix,severity,classification,action,pool,role,dex\n");
    for k in keys {
        if let Some(cols) = rows.get(&k) {
            out.push_str(&csv_join_problem(cols));
            out.push('\n');
        }
    }
    let tmp = path.with_extension("csv.tmp");
    let _ = std::fs::write(&tmp, out).and_then(|_| std::fs::rename(&tmp, path));
}

fn csv_join_problem(cols: &[String]) -> String {
    cols.iter().map(|s| csv_escape_problem(s)).collect::<Vec<_>>().join(",")
}

fn csv_escape_problem(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') || s.contains('\r') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}



/// Write the exact minimal operator-facing CSV requested by the user.
///
/// Output path: `<manual_accounts_root>/bad_accounts_review.csv`
/// Exact columns: `bad_account,fail_count,pool,role`
///
/// One row is kept per account+pool+role and `fail_count` is incremented every
/// time that combination appears in a failing simulation diagnosis.  If the
/// account is not found in mix.json, `pool` and/or `role` is written as
/// `unknown`.
fn upsert_bad_account_review_csv(path: &Path, bad_account: &str, pool: &str, role: &str) {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    let pool = if pool.trim().is_empty() { "unknown" } else { pool.trim() };
    let role = if role.trim().is_empty() { "unknown" } else { role.trim() };
    let key = format!("{}|{}|{}", bad_account, pool, role);

    let mut rows: HashMap<String, (String, u64, String, String)> = HashMap::new();
    if let Ok(text) = std::fs::read_to_string(path) {
        for line in text.lines().skip(1) {
            let line = line.trim();
            if line.is_empty() { continue; }
            let cols: Vec<String> = line
                .split(',')
                .map(|s| s.trim().trim_matches('"').to_string())
                .collect();
            if cols.len() < 4 { continue; }
            let account = cols[0].clone();
            let count = cols[1].parse::<u64>().unwrap_or(1);
            let pool = if cols[2].is_empty() { "unknown".to_string() } else { cols[2].clone() };
            let role = if cols[3].is_empty() { "unknown".to_string() } else { cols[3].clone() };
            rows.insert(format!("{}|{}|{}", account, pool, role), (account, count, pool, role));
        }
    }

    rows.entry(key)
        .and_modify(|row| row.1 = row.1.saturating_add(1))
        .or_insert_with(|| (bad_account.to_string(), 1, pool.to_string(), role.to_string()));

    let mut values: Vec<_> = rows.into_values().collect();
    values.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    let mut out = String::from("bad_account,fail_count,pool,role\n");
    for (account, count, pool, role) in values {
        out.push_str(&csv_join_problem(&[account, count.to_string(), pool, role]));
        out.push('\n');
    }

    let tmp = path.with_extension("csv.tmp");
    let _ = std::fs::write(&tmp, out).and_then(|_| std::fs::rename(&tmp, path));
}

// ────────────────────────────────────────────────────────────────────────────

impl Simulator {
    pub fn new(
        so_dir: &str,
        wsol_ata: Pubkey,
        payer_pubkey: Pubkey,
        fail_closed: bool,
        allow_hot_path_rpc_fetch: bool,
        manual_accounts_root: PathBuf,
        current_slot: Arc<AtomicU64>,
        current_unix_timestamp: Arc<AtomicI64>,
        missing_handle: Option<Arc<crate::auto_missing_accounts::AutoMissingAccountsHandle>>,
    ) -> Result<Self> {
        // Build the SVM with the full mainnet feature set.
        //
        // with_mainnet_features(): activates every Solana feature gate live on
        //   mainnet-beta.  PMM DEXes rely on post-2.0 features; without this
        //   they silently mis-execute.
        //
        // with_default_programs(): loads SPL Token, SPL Token-2022, ATA,
        //   System, Compute Budget, and other built-ins.
        //
        // with_sigverify(false): skip ed25519 sig checks — the bot already
        //   signs correctly; skipping saves ~0.5 ms per sim on the hot path.
        //
        // with_blockhash_check(false): we use a cached recent blockhash;
        //   skip the SVM's internal staleness check.
        let mut svm = LiteSVM::new()
            .with_sysvars()
            .with_sigverify(false)
            .with_blockhash_check(false)
            .with_default_programs()
            .with_mainnet_features()
            .with_feature_accounts();

        // warp_to_slot atomically advances Clock.slot, Clock.epoch,
        // SlotHashes, and EpochSchedule — everything PMM oracle staleness
        // checks read.
        let initial_slot = current_slot.load(Ordering::Relaxed);
        svm.warp_to_slot(initial_slot);
        let initial_unix_timestamp = current_unix_timestamp.load(Ordering::Relaxed);
        set_live_clock(&mut svm, initial_slot, initial_unix_timestamp);
        debug!(initial_slot, "sim slot initialised via warp_to_slot");

        for fname in crate::program_registry::KNOWN_PROGRAM_FILES {
            let path = resolve_program_file(so_dir, fname);
            eprintln!(
                "[sim_program_file_check] file={} exists={} resolved={}",
                fname,
                path.is_some(),
                path.as_ref()
                    .map(|p| p.display().to_string())
                    .unwrap_or_default()
            );
        }

        // Load DEX program bytecode (.so files) from disk.
        let mut loaded = 0usize;
        let mut missing = 0usize;
        let mut loaded_programs = HashSet::new();
        let mut loaded_program_files = HashMap::new();
        let mut loaded_program_bytes = HashMap::new();
        for (pid_str, fname) in crate::program_registry::PROGRAMS {
            if fname.is_empty() {
                continue;
            }
            let path = resolve_program_file_for_program(so_dir, pid_str, fname);
            if path.is_none() {
                eprintln!(
                    "[sim_program_load] program={pid_str} file={fname} exists=false loaded=false error=missing"
                );
                warn!(file = fname, so_dir, "program .so not found, skipping");
                missing += 1;
                continue;
            }
            let path = path.unwrap();
            let pid = Pubkey::try_from(*pid_str)
                .map_err(|e| anyhow!("bad program id {pid_str}: {e:?}"))?;
            let bytes = match std::fs::read(&path) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!(
                        "[sim_program_load] program={pid} registry_file={fname} resolved={} exists=true loaded=false error=read:{e:?}",
                        path.display()
                    );
                    warn!(program = %pid, path = %path.display(), error = ?e, "program read failed");
                    missing += 1;
                    continue;
                }
            };
            match svm.add_program(pk_to_addr(pid), &bytes) {
                Ok(()) => {
                    eprintln!(
                        "[sim_program_load] program={pid} registry_file={fname} resolved={} exists=true loaded=true",
                        path.display()
                    );
                    debug!(program = %pid, path = %path.display(), "program loaded");
                    loaded_programs.insert(pid);
                    loaded_program_files.insert(pid, path.display().to_string());
                    loaded_program_bytes.insert(pid, bytes);
                    loaded += 1;
                }
                Err(e) => {
                    eprintln!(
                        "[sim_program_load] program={pid} registry_file={fname} resolved={} exists=true loaded=false error={e:?}",
                        path.display()
                    );
                    warn!(program = %pid, path = %path.display(), error = ?e, "program load failed");
                    missing += 1;
                }
            }
        }
        info!(loaded, missing, "LiteSVM 0.11 programs loaded");

        let mix_roles = load_mix_role_map(&manual_accounts_root);
        eprintln!(
            "[mix_role_map] accounts={} source_root={} status=loaded",
            mix_roles.len(),
            manual_accounts_root.display()
        );

        Ok(Self {
            svm: Mutex::new(svm),
            wsol_ata,
            payer_pubkey,
            loaded_programs,
            loaded_program_files,
            loaded_program_bytes,
            jito_tip_accounts: crate::transaction::jito_tip_pubkeys()
                .into_iter()
                .collect(),
            fail_closed,
            current_slot,
            current_unix_timestamp,
            allow_hot_path_rpc_fetch,
            manual_accounts_root,
            mix_roles,
            missing_handle,
        })
    }

    fn should_skip_account(&self, pk: &Pubkey) -> bool {
        is_known_sysvar(pk)
            || self.loaded_programs.contains(pk)
            || is_builtin_program(pk)
            || self.jito_tip_accounts.contains(pk)
            || *pk == self.payer_pubkey
    }

    #[allow(clippy::too_many_arguments)]
    fn debug_failed_program_environment(
        &self,
        failed_program: &Pubkey,
        cache: &AccountCache,
        alt_cache: &crate::alt_cache::AltCache,
        tx: &VersionedTransaction,
        alts: &[AddressLookupTableAccount],
        account_metas: &[TxAccountMeta],
        synthetic_readonly_system_accounts: &HashSet<Pubkey>,
        created_by_setup: &HashSet<Pubkey>,
        route_sig: u128,
        route_labels: &str,
        route_programs: &str,
        lite_err: &str,
        ix_source: &str,
        metrics: &Metrics,
    ) {
        validate_alts_for_failed_tx(
            cache,
            alt_cache,
            alts,
            failed_program,
            route_sig,
            route_labels,
            route_programs,
            ix_source,
        );
        compare_tx_accounts_with_rpc_for_failed_program(
            self,
            failed_program,
            cache,
            account_metas,
            synthetic_readonly_system_accounts,
            created_by_setup,
            route_sig,
            route_labels,
            route_programs,
            ix_source,
            metrics,
        );
        dump_failed_program_context_window(
            failed_program,
            tx,
            alts,
            account_metas,
            cache,
            synthetic_readonly_system_accounts,
            created_by_setup,
            route_sig,
            route_labels,
            route_programs,
            lite_err,
            ix_source,
        );
    }

    fn fresh_snapshot_svm(&self, slot: u64, unix_timestamp: i64) -> Result<LiteSVM> {
        let mut svm = LiteSVM::new()
            .with_sysvars()
            .with_sigverify(false)
            .with_blockhash_check(false)
            .with_default_programs()
            .with_mainnet_features()
            .with_feature_accounts();
        svm.warp_to_slot(slot);
        set_live_clock(&mut svm, slot, unix_timestamp);

        // Reuse the in-memory bytecode captured at startup — no disk I/O on the
        // retry hot path.
        let mut programs = self.loaded_program_bytes.iter().collect::<Vec<_>>();
        programs.sort_by_key(|(program, _)| program.to_string());
        for (program, bytes) in programs {
            svm.add_program(pk_to_addr(*program), bytes)
                .with_context(|| format!("snapshot add_program program={program}"))?;
        }
        Ok(svm)
    }

    #[allow(clippy::too_many_arguments)]
    fn retry_with_rpc_snapshot(
        &self,
        failed_program: &Pubkey,
        cache: &AccountCache,
        tx: &VersionedTransaction,
        alts: &[AddressLookupTableAccount],
        account_metas: &[TxAccountMeta],
        synthetic_readonly_system_accounts: &HashSet<Pubkey>,
        created_by_setup: &HashSet<Pubkey>,
        min_wsol_gain: u64,
        metrics: &Metrics,
        route_sig: u128,
        route_labels: &str,
        route_programs: &str,
        lite_err: &str,
        ix_source: &str,
    ) -> Option<SimOutcome> {
        let attempt = RPC_SNAPSHOT_RETRY_COUNT.fetch_add(1, Ordering::Relaxed);
        if attempt >= MAX_RPC_SNAPSHOT_RETRY_PER_PROCESS {
            eprintln!(
                "[sim_retry_with_rpc_snapshot] route_sig={:032x} source={} route_labels={} programs={} failed_program={} status=skipped reason=process_cap lite_err={}",
                route_sig, ix_source, route_labels, route_programs, failed_program, lite_err
            );
            return None;
        }

        let mut metas = account_metas.to_vec();
        metas.sort_by(|a, b| a.pubkey.cmp(&b.pubkey));
        metas.dedup_by(|a, b| a.pubkey == b.pubkey);
        let keys = metas.iter().map(|meta| meta.pubkey).collect::<Vec<_>>();
        let rpc_fetch = cache.fetch_accounts_for_compare(&keys);
        let rpc_context_slot = rpc_fetch.rpc_context_slot.unwrap_or(0);
        let cache_slot = cache.stream_slot().load(Ordering::Relaxed);
        let retry_slot = if rpc_context_slot > 0 {
            rpc_context_slot
        } else {
            cache_slot
        };
        let retry_unix_timestamp = self.current_unix_timestamp.load(Ordering::Relaxed);
        let slot_delta = slot_delta(cache_slot, rpc_context_slot);

        let mut svm = match self.fresh_snapshot_svm(retry_slot, retry_unix_timestamp) {
            Ok(svm) => svm,
            Err(e) => {
                metrics
                    .sim_retry_rpc_snapshot_fail
                    .fetch_add(1, Ordering::Relaxed);
                eprintln!(
                    "[sim_retry_with_rpc_snapshot] route_sig={:032x} source={} route_labels={} programs={} failed_program={} status=setup_error cache_slot={} rpc_context_slot={} slot_delta={} error={} lite_err={}",
                    route_sig,
                    ix_source,
                    route_labels,
                    route_programs,
                    failed_program,
                    cache_slot,
                    rpc_context_slot,
                    slot_delta,
                    e,
                    lite_err
                );
                return None;
            }
        };

        let mut injected_rpc = 0usize;
        let mut injected_synthetic = 0usize;
        let mut synthetic_notfound_rpc = 0usize;
        let mut missing = 0usize;
        let mut rpc_errors = 0usize;

        // For AlphaQ retries, compute the route accounts so we can block
        // synthetic re-injection for accounts that are confirmed NotFound on RPC.
        // Re-injecting a synthetic would repeat the same InvalidAccountOwner error.
        let retry_alphaq_route_accounts = if *failed_program == ALPHAQ_PROGRAM_ID {
            jupiter_route_account_keys_for_program(tx, alts, &ALPHAQ_PROGRAM_ID)
        } else {
            HashSet::new()
        };

        for alt in alts {
            match synthetic_alt_account(alt).and_then(|raw| {
                svm.set_account(pk_to_addr(alt.key), raw)
                    .map_err(|e| anyhow!("set snapshot ALT {} failed: {e:?}", alt.key))
            }) {
                Ok(()) => injected_rpc += 1,
                Err(e) => eprintln!(
                    "[sim_retry_with_rpc_snapshot] route_sig={:032x} source={} failed_program={} status=alt_inject_error alt={} error={}",
                    route_sig, ix_source, failed_program, alt.key, e
                ),
            }
        }

        for meta in &metas {
            let pk = &meta.pubkey;
            if is_known_sysvar(pk) || self.loaded_programs.contains(pk) || is_builtin_program(pk) {
                continue;
            }
            if self.jito_tip_accounts.contains(pk) {
                if let Err(e) = svm.set_account(pk_to_addr(*pk), synthetic_system_account(1_000_000_000)) {
                    eprintln!(
                        "[sim_retry_with_rpc_snapshot] route_sig={:032x} source={} failed_program={} status=account_inject_error pubkey={} error={:?}",
                        route_sig, ix_source, failed_program, pk, e
                    );
                } else {
                    injected_synthetic += 1;
                }
                continue;
            }
            if *pk == self.payer_pubkey {
                if let Err(e) = svm.set_account(pk_to_addr(*pk), synthetic_system_account(10_000_000_000)) {
                    eprintln!(
                        "[sim_retry_with_rpc_snapshot] route_sig={:032x} source={} failed_program={} status=account_inject_error pubkey={} error={:?}",
                        route_sig, ix_source, failed_program, pk, e
                    );
                } else {
                    injected_synthetic += 1;
                }
                continue;
            }

            match rpc_fetch.accounts.get(pk) {
                Some(AccountFetchResult::Found(account)) => {
                    if account.executable() {
                        continue;
                    }
                    // Immediately update the live AccountCache so the next
                    // simulation of this route finds fresh state without
                    // another RPC round-trip. For writable pool-state accounts
                    // owned by a registered DEX program, Yellowstone's owner
                    // filter will keep the entry live going forward.
                    cache.insert_manual(*pk, account.clone());
                    // Writable accounts NOT covered by the owner filter
                    // (token vaults, foreign-owned PDAs) would otherwise go
                    // stale again after this one-shot insert. Add them to the
                    // live gRPC subscription so the stream keeps them fresh.
                    if meta.is_writable {
                        if let Ok(owner_pk) = Pubkey::try_from(account.owner().as_ref()) {
                            cache.note_uncovered_writable(*pk, &owner_pk);
                        }
                    }
                    if let Err(e) = svm.set_account(pk_to_addr(*pk), account.clone()) {
                        eprintln!(
                            "[sim_retry_with_rpc_snapshot] route_sig={:032x} source={} failed_program={} status=account_inject_error pubkey={} error={:?}",
                            route_sig, ix_source, failed_program, pk, e
                        );
                    } else {
                        injected_rpc += 1;
                    }
                }
                Some(AccountFetchResult::NotFound) | None => {
                    // AlphaQ route accounts that were synthetic in the initial sim must NOT be
                    // re-injected as synthetic in the retry. If the account is also NotFound on
                    // RPC it doesn't exist on-chain → the trade will fail on-chain too → count
                    // as missing so the retry aborts cleanly instead of repeating InvalidAccountOwner.
                    if !retry_alphaq_route_accounts.is_empty()
                        && retry_alphaq_route_accounts.contains(pk)
                        && synthetic_readonly_system_accounts.contains(pk)
                        && !is_expected_readonly_pda_authority_meta(meta)
                    {
                        missing += 1;
                        eprintln!(
                            "[sim_retry_alphaq_account_notfound_rpc] route_sig={:032x} source={} pk={} action=count_as_missing reason=alphaq_account_not_on_rpc_synthetic_would_repeat_invalid_owner",
                            route_sig, ix_source, pk
                        );
                        if let Some(handle) = &self.missing_handle {
                            handle.record(crate::auto_missing_accounts::MissingAccountEvent {
                                pubkey: *pk,
                                route_sig,
                                route_labels: route_labels.to_string(),
                                programs: route_programs.to_string(),
                                source: "retry_alphaq_notfound_rpc".to_string(),
                                is_signer: meta.is_signer,
                                is_writable: meta.is_writable,
                                created_by_setup: false,
                            });
                        }
                    } else if synthetic_readonly_system_accounts.contains(pk) {
                        let expected_pda_authority = is_expected_readonly_pda_authority_meta(meta);

                        if expected_pda_authority {
                            // Readonly PDA authorities can be valid tx accounts without an initialized
                            // on-chain account. Do not count them as missing pool/vault state.
                            eprintln!(
                                "[sim_expected_readonly_pda_authority] route_sig={:032x} source={} pk={} action=do_not_record_missing reason=rpc_not_found_readonly_pda_authority",
                                route_sig, ix_source, pk
                            );
                        } else {
                            // RPC also says this account doesn't exist. This is a real blocker candidate,
                            // so it should influence retry diagnosis and be learned by auto-missing.
                            synthetic_notfound_rpc += 1;
                            eprintln!(
                                "[sim_retry_synthetic_notfound_rpc] route_sig={:032x} source={} failed_program={} pk={} is_writable={} note=account_not_found_on_rpc_injecting_synthetic_owner_check_may_fail",
                                route_sig, ix_source, failed_program, pk, meta.is_writable
                            );
                            if let Some(handle) = &self.missing_handle {
                                handle.record(crate::auto_missing_accounts::MissingAccountEvent {
                                    pubkey: *pk,
                                    route_sig,
                                    route_labels: route_labels.to_string(),
                                    programs: route_programs.to_string(),
                                    source: "retry_synthetic_notfound_rpc".to_string(),
                                    is_signer: meta.is_signer,
                                    is_writable: meta.is_writable,
                                    created_by_setup: false,
                                });
                            }
                        }

                        if let Err(e) = svm.set_account(pk_to_addr(*pk), synthetic_system_account(0)) {
                            eprintln!(
                                "[sim_retry_with_rpc_snapshot] route_sig={:032x} source={} failed_program={} status=account_inject_error pubkey={} error={:?}",
                                route_sig, ix_source, failed_program, pk, e
                            );
                        } else {
                            injected_synthetic += 1;
                        }
                    } else if !created_by_setup.contains(pk) {
                        missing += 1;
                        // Queue to background fetcher — it will keep retrying
                        // until the account appears or crosses the 8000 threshold.
                        if let Some(handle) = &self.missing_handle {
                            handle.record(
                                crate::auto_missing_accounts::MissingAccountEvent {
                                    pubkey: *pk,
                                    route_sig,
                                    route_labels: route_labels.to_string(),
                                    programs: route_programs.to_string(),
                                    source: "rpc_retry_not_found".to_string(),
                                    is_signer: meta.is_signer,
                                    is_writable: meta.is_writable,
                                    created_by_setup: false,
                                },
                            );
                        }
                    }
                }
                Some(AccountFetchResult::Error { .. }) => {
                    rpc_errors += 1;
                }
            }
        }

        let injected = injected_rpc + injected_synthetic;

        if missing > 0 || rpc_errors > 0 {
            metrics
                .sim_retry_rpc_snapshot_fail
                .fetch_add(1, Ordering::Relaxed);
            eprintln!(
                "[sim_retry_with_rpc_snapshot] route_sig={:032x} source={} route_labels={} programs={} failed_program={} status=missing_snapshot_accounts injected={} injected_rpc={} injected_synthetic={} synthetic_notfound_rpc={} missing={} rpc_errors={} cache_slot={} rpc_context_slot={} slot_delta={} lite_err={}",
                route_sig,
                ix_source,
                route_labels,
                route_programs,
                failed_program,
                injected,
                injected_rpc,
                injected_synthetic,
                synthetic_notfound_rpc,
                missing,
                rpc_errors,
                cache_slot,
                rpc_context_slot,
                slot_delta,
                lite_err
            );
            return None;
        }

        let wsol_before = parse_wsol_amount(&svm, self.wsol_ata);
        let litesvm_tx = match to_litesvm_tx(tx) {
            Ok(tx) => tx,
            Err(e) => {
                metrics
                    .sim_retry_rpc_snapshot_fail
                    .fetch_add(1, Ordering::Relaxed);
                eprintln!(
                    "[sim_retry_with_rpc_snapshot] route_sig={:032x} source={} route_labels={} programs={} failed_program={} status=tx_convert_error error={} lite_err={}",
                    route_sig, ix_source, route_labels, route_programs, failed_program, e, lite_err
                );
                return None;
            }
        };

        match svm.simulate_transaction(litesvm_tx) {
            Ok(info) => {
                metrics
                    .sim_retry_rpc_snapshot_ok
                    .fetch_add(1, Ordering::Relaxed);
                let wsol_ata_addr = pk_to_addr(self.wsol_ata);
                let wsol_after = info
                    .post_accounts
                    .iter()
                    .find(|(addr, _)| *addr == wsol_ata_addr)
                    .and_then(|(_, acc)| parse_token_amount(acc.data()))
                    .unwrap_or(wsol_before);
                let min_wsol_after = wsol_before.saturating_add(min_wsol_gain);
                let (result_kind, profitable) = if wsol_after >= min_wsol_after {
                    ("pass_profitable", true)
                } else {
                    ("pass_unprofitable", false)
                };
                eprintln!(
                    "[sim_retry_with_rpc_snapshot] route_sig={:032x} source={} route_labels={} programs={} failed_program={} status=ok result={} injected={} injected_rpc={} injected_synthetic={} synthetic_notfound_rpc={} cache_slot={} rpc_context_slot={} slot_delta={} compute_units={} wsol_before={} wsol_after={} min_after={} diagnosis=stale_cache_possible lite_err={}",
                    route_sig,
                    ix_source,
                    route_labels,
                    route_programs,
                    failed_program,
                    result_kind,
                    injected,
                    injected_rpc,
                    injected_synthetic,
                    synthetic_notfound_rpc,
                    cache_slot,
                    rpc_context_slot,
                    slot_delta,
                    info.meta.compute_units_consumed,
                    wsol_before,
                    wsol_after,
                    min_wsol_after,
                    lite_err
                );
                if profitable {
                    return Some(SimOutcome {
                        compute_units: info.meta.compute_units_consumed,
                        wsol_before,
                        wsol_after,
                    });
                }
                None
            }
            Err(meta) => {
                metrics
                    .sim_retry_rpc_snapshot_fail
                    .fetch_add(1, Ordering::Relaxed);
                let retry_err = format!("{:?}", meta.err);
                let retry_diagnosis = if synthetic_notfound_rpc > 0 {
                    "synthetic_accounts_not_on_rpc_likely_uninitialized"
                } else {
                    "route_or_logic_issue_possible"
                };
                eprintln!(
                    "[sim_retry_with_rpc_snapshot] route_sig={:032x} source={} route_labels={} programs={} failed_program={} status=fail injected={} injected_rpc={} injected_synthetic={} synthetic_notfound_rpc={} cache_slot={} rpc_context_slot={} slot_delta={} retry_err={} compute_units={} diagnosis={} lite_err={}",
                    route_sig,
                    ix_source,
                    route_labels,
                    route_programs,
                    failed_program,
                    injected,
                    injected_rpc,
                    injected_synthetic,
                    synthetic_notfound_rpc,
                    cache_slot,
                    rpc_context_slot,
                    slot_delta,
                    retry_err,
                    meta.meta.compute_units_consumed,
                    retry_diagnosis,
                    lite_err
                );
                // On-chain this route succeeds but LiteSVM rejects it with
                // InvalidAccountOwner even on a 100%-fresh RPC snapshot. Dump the
                // owner LiteSVM actually presented for each account in the failing
                // program's instruction vs the owner RPC reports, so the exact
                // mismatching account is pinpointed.
                if is_invalid_account_owner(&retry_err, &meta.meta.logs) {
                    let owner_failed_program =
                        first_failed_program_for_invalid_owner(&meta.meta.logs)
                            .unwrap_or(*failed_program);
                    dump_failing_program_owner_diagnosis(
                        &svm,
                        &owner_failed_program,
                        tx,
                        alts,
                        &rpc_fetch.accounts,
                        route_sig,
                        ix_source,
                    );
                }
                None
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn retry_mix_gate_drop_with_rpc_snapshot(
        &self,
        cache: &AccountCache,
        tx: &VersionedTransaction,
        alts: &[AddressLookupTableAccount],
        min_wsol_gain: u64,
        metrics: &Metrics,
        route_sig: u128,
        route_labels: &str,
        route_programs: &str,
        ix_source: &str,
        drop_reason: &str,
    ) {
        let account_metas = collect_tx_account_metas(tx, alts);
        let created_by_setup = collect_accounts_created_by_setup(tx, alts);
        let synthetic_readonly_system_accounts = HashSet::new();
        let solfi_id = SOLFI_V2_PROGRAM_ID.to_string();
        let failed_program = if route_programs.contains(&solfi_id)
            || transaction_mentions_program(tx, alts, &SOLFI_V2_PROGRAM_ID)
        {
            SOLFI_V2_PROGRAM_ID
        } else {
            JUPITER_PROGRAM_ID
        };
        let lite_err = format!("mix_gate_drop_before_simulation:{drop_reason}");
        self.retry_with_rpc_snapshot(
            &failed_program,
            cache,
            tx,
            alts,
            &account_metas,
            &synthetic_readonly_system_accounts,
            &created_by_setup,
            min_wsol_gain,
            metrics,
            route_sig,
            route_labels,
            route_programs,
            &lite_err,
            ix_source,
        );
    }

    /// Simulate `tx` against the Yellowstone-fed `cache`. Returns
    /// `Ok(SimOutcome)` when the tx succeeds AND leaves at least
    /// `min_acceptable_out` lamports in the user's WSOL ATA. Returns `Err`
    /// for reverts or unprofitable outcomes — caller should drop the bundle.
    ///
    /// Reads from the Yellowstone-fed cache first. If Metis/Jupiter adds a
    /// transaction-local non-executable account that was not in mix.json, it is
    /// fetched once from RPC, cached, classified, and injected.
    pub fn simulate(
        &self,
        tx: &VersionedTransaction,
        alts: &[AddressLookupTableAccount],
        alt_cache: &crate::alt_cache::AltCache,
        cache: &AccountCache,
        min_wsol_gain: u64,
        metrics: &Metrics,
        route_sig: u128,
        route_labels: &str,
        route_programs: &str,
        ix_source: &str,
    ) -> Result<SimOutcome> {
        let account_metas = collect_tx_account_metas(tx, alts);
        let created_by_setup = collect_accounts_created_by_setup(tx, alts);
        let contains_alphaq = transaction_mentions_program(tx, alts, &ALPHAQ_PROGRAM_ID);
        let contains_whirlpool = transaction_mentions_program(tx, alts, &WHIRLPOOL_PROGRAM_ID);
        let alphaq_route_accounts =
            jupiter_route_account_keys_for_program(tx, alts, &ALPHAQ_PROGRAM_ID);
        let disable_synthetic_for_all_alphaq_accounts =
            contains_alphaq && alphaq_route_accounts.is_empty();
        if disable_synthetic_for_all_alphaq_accounts {
            eprintln!(
                "[sim_alphaq_route_accounts_missing] program={} reason=alphaq_mentioned_but_no_jupiter_route_ix_found action=disable_synthetic_for_tx",
                ALPHAQ_PROGRAM_ID
            );
        }

        // Missing transaction-local accounts are fetched once above and then
        // stay in the cache for subsequent simulations.
        let mut missing_accounts: Vec<TxAccountMeta> = Vec::new();
        let mut missing_programs: Vec<TxAccountMeta> = Vec::new();
        let mut synthetic_readonly_system_accounts: HashSet<Pubkey> = HashSet::new();
        let mut lazy_fetched = 0usize;
        let mut need_fetch = Vec::new();

        // Warm transaction-local accounts that are not part of mix.json.
        // Non-executable accounts are safe to fetch and inject. Executable
        // accounts must have a matching .so loaded through program_registry.
        for meta in &account_metas {
            let pk = &meta.pubkey;
            if self.should_skip_account(pk) {
                continue;
            }

            match cache.get(pk) {
                Some(acct) => {
                    if acct.executable() {
                        log_account_classification(
                            "sim_missing_full",
                            meta,
                            &acct,
                            "program_not_loaded",
                        );
                        missing_programs.push(*meta);
                    }
                }
                None => {
                    if created_by_setup.contains(pk) {
                        eprintln!(
                            "[sim_missing_skip_created_by_setup] pk={} is_writable={} source={}",
                            pk,
                            meta.is_writable,
                            meta.source.as_str()
                        );
                        continue;
                    }
                    need_fetch.push(*pk);
                }
            }
        }

        let missing_without_rpc = if self.allow_hot_path_rpc_fetch {
            HashSet::new()
        } else {
            need_fetch.iter().copied().collect::<HashSet<_>>()
        };
        if !need_fetch.is_empty() && !self.allow_hot_path_rpc_fetch {
            let sample = need_fetch
                .iter()
                .take(8)
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(",");
            eprintln!(
                "[hot_path_rpc_warning] reason=tx_account_missing_from_cache action=no_rpc_drop_or_synthetic count={} sample=[{}]",
                need_fetch.len(),
                sample
            );
        }
        let fetched_accounts = if self.allow_hot_path_rpc_fetch {
            cache.get_many_or_fetch_hot(&need_fetch)
        } else {
            HashMap::new()
        };
        for meta in &account_metas {
            let pk = &meta.pubkey;
            let fetch_result = fetched_accounts.get(pk);

            if fetch_result.is_none() && missing_without_rpc.contains(pk) {
                if should_allow_synthetic_missing(meta, disable_synthetic_for_all_alphaq_accounts) {
                    eprintln!(
                        "[sim_synthetic_readonly_system] pk={} reason=hot_path_rpc_disabled_readonly_non_signer is_writable={} source={}",
                        pk,
                        meta.is_writable,
                        meta.source.as_str()
                    );
                    synthetic_readonly_system_accounts.insert(*pk);
                    // Queue for background fetch: if this account is owned by a DEX
                    // program (e.g. AlphaQ tick arrays), giving it a synthetic
                    // System-owned account will fail the DEX's ownership check.
                    // The auto_missing service fetches it once; gRPC owner-filter
                    // keeps it fresh on subsequent slots.
                    if is_expected_readonly_pda_authority_meta(meta) {
                        eprintln!(
                            "[sim_expected_readonly_pda_authority] route_sig={:032x} source={} pk={} action=do_not_record_missing reason=synthetic_readonly_authority",
                            route_sig, ix_source, pk
                        );
                    } else if let Some(handle) = &self.missing_handle {
                        handle.record(crate::auto_missing_accounts::MissingAccountEvent {
                            pubkey: *pk,
                            route_sig,
                            route_labels: route_labels.to_string(),
                            programs: route_programs.to_string(),
                            source: "synthetic_readonly_system".to_string(),
                            is_signer: meta.is_signer,
                            is_writable: meta.is_writable,
                            created_by_setup: false,
                        });
                    }
                    if contains_alphaq && alphaq_route_accounts.contains(pk) {
                        eprintln!(
                            "[sim_alphaq_synthetic_candidate] pk={} source={} note=alphaq_may_check_owner_of_this_account_and_fail",
                            pk, meta.source.as_str()
                        );
                    }
                } else {
                    if contains_alphaq && alphaq_route_accounts.contains(pk) {
                        log_missing_alphaq_route_account(meta, "hot_path_rpc_disabled_no_synthetic");
                    }
                    manual_sim_accounts::append_missing_runtime_account(
                        &self.manual_accounts_root,
                        RuntimeMissingAccount {
                            pubkey: *pk,
                            route_sig,
                            route_labels: route_labels.to_string(),
                            programs: route_programs.to_string(),
                            source: meta.source.as_str().to_string(),
                            is_signer: meta.is_signer,
                            is_writable: meta.is_writable,
                            created_by_setup: created_by_setup.contains(pk),
                            from_cache: false,
                            reason: "hot_path_rpc_disabled_no_synthetic".to_string(),
                        },
                    );
                    if let Some(handle) = &self.missing_handle {
                        handle.record(crate::auto_missing_accounts::MissingAccountEvent {
                            pubkey: *pk,
                            route_sig,
                            route_labels: route_labels.to_string(),
                            programs: route_programs.to_string(),
                            source: meta.source.as_str().to_string(),
                            is_signer: meta.is_signer,
                            is_writable: meta.is_writable,
                            created_by_setup: created_by_setup.contains(pk),
                        });
                    }
                    if meta.is_writable {
                        log_missing_writable(meta, "hot_path_rpc_disabled");
                    }
                    log_missing_fetch_error(
                        meta,
                        "hot_path_rpc_disabled",
                        "account missing from cache/gRPC/mix2",
                    );
                    missing_accounts.push(*meta);
                }
                continue;
            }

            let Some(fetch_result) = fetch_result else {
                continue;
            };

            match fetch_result {
                AccountFetchResult::Found(acct) => {
                    lazy_fetched += 1;
                    log_account_classification(
                        "sim_missing_classify",
                        meta,
                        &acct,
                        "batch_rpc_fetched",
                    );
                    if acct.executable() {
                        log_account_classification(
                            "sim_missing_full",
                            meta,
                            &acct,
                            "program_not_loaded",
                        );
                        missing_programs.push(*meta);
                    }
                }
                AccountFetchResult::NotFound => {
                    // AlphaQ route accounts that don't exist on RPC must NOT receive a
                    // synthetic placeholder — the program will throw InvalidAccountOwner.
                    if contains_alphaq
                        && alphaq_route_accounts.contains(pk)
                        && !is_expected_readonly_pda_authority_meta(meta)
                    {
                        eprintln!(
                            "[sim_alphaq_route_account_blocked] pk={} source={} is_writable={} action=add_to_missing reason=alphaq_ownership_check_would_fail_not_found",
                            pk, meta.source.as_str(), meta.is_writable
                        );
                        if let Some(handle) = &self.missing_handle {
                            handle.record(crate::auto_missing_accounts::MissingAccountEvent {
                                pubkey: *pk,
                                route_sig,
                                route_labels: route_labels.to_string(),
                                programs: route_programs.to_string(),
                                source: "alphaq_route_account_not_found".to_string(),
                                is_signer: meta.is_signer,
                                is_writable: meta.is_writable,
                                created_by_setup: false,
                            });
                        }
                        missing_accounts.push(*meta);
                    } else if should_allow_synthetic_missing(meta, disable_synthetic_for_all_alphaq_accounts)
                    {
                        eprintln!(
                            "[sim_synthetic_readonly_system] pk={} reason=not_found is_writable={} source={}",
                            pk,
                            meta.is_writable,
                            meta.source.as_str()
                        );
                        synthetic_readonly_system_accounts.insert(*pk);
                        if is_expected_readonly_pda_authority_meta(meta) {
                            eprintln!(
                                "[sim_expected_readonly_pda_authority] route_sig={:032x} source={} pk={} action=do_not_record_missing reason=rpc_not_found_readonly_authority",
                                route_sig, ix_source, pk
                            );
                        } else if let Some(handle) = &self.missing_handle {
                            handle.record(crate::auto_missing_accounts::MissingAccountEvent {
                                pubkey: *pk,
                                route_sig,
                                route_labels: route_labels.to_string(),
                                programs: route_programs.to_string(),
                                source: "synthetic_alt_readonly_not_found".to_string(),
                                is_signer: meta.is_signer,
                                is_writable: meta.is_writable,
                                created_by_setup: false,
                            });
                        }
                    } else {
                        if contains_alphaq && alphaq_route_accounts.contains(pk) {
                            log_missing_alphaq_route_account(meta, "not_found_no_synthetic");
                        }
                        if meta.is_writable {
                            log_missing_writable(meta, "not_found");
                        } else {
                            log_missing_not_found(meta);
                        }
                        missing_accounts.push(*meta);
                    }
                }
                AccountFetchResult::Error { kind, message } => {
                    // AlphaQ route accounts: RPC error must not fall back to a synthetic.
                    if contains_alphaq
                        && alphaq_route_accounts.contains(pk)
                        && !is_expected_readonly_pda_authority_meta(meta)
                    {
                        eprintln!(
                            "[sim_alphaq_route_account_blocked] pk={} source={} is_writable={} action=add_to_missing reason=alphaq_ownership_check_would_fail_rpc_error error_kind={}",
                            pk, meta.source.as_str(), meta.is_writable, kind
                        );
                        if let Some(handle) = &self.missing_handle {
                            handle.record(crate::auto_missing_accounts::MissingAccountEvent {
                                pubkey: *pk,
                                route_sig,
                                route_labels: route_labels.to_string(),
                                programs: route_programs.to_string(),
                                source: "alphaq_route_account_rpc_error".to_string(),
                                is_signer: meta.is_signer,
                                is_writable: meta.is_writable,
                                created_by_setup: false,
                            });
                        }
                        missing_accounts.push(*meta);
                    } else if should_allow_synthetic_missing(meta, disable_synthetic_for_all_alphaq_accounts)
                    {
                        eprintln!(
                            "[sim_synthetic_readonly_system] pk={} reason=rpc_error error_kind={} error={} is_writable={} source={}",
                            pk,
                            kind,
                            message,
                            meta.is_writable,
                            meta.source.as_str()
                        );
                        synthetic_readonly_system_accounts.insert(*pk);
                        if let Some(handle) = &self.missing_handle {
                            handle.record(crate::auto_missing_accounts::MissingAccountEvent {
                                pubkey: *pk,
                                route_sig,
                                route_labels: route_labels.to_string(),
                                programs: route_programs.to_string(),
                                source: "synthetic_alt_readonly_rpc_error".to_string(),
                                is_signer: meta.is_signer,
                                is_writable: meta.is_writable,
                                created_by_setup: false,
                            });
                        }
                    } else {
                        if contains_alphaq && alphaq_route_accounts.contains(pk) {
                            log_missing_alphaq_route_account(meta, "rpc_error_no_synthetic");
                        }
                        if meta.is_writable {
                            log_missing_writable(meta, &kind);
                        }
                        log_missing_fetch_error(meta, &kind, &message);
                        missing_accounts.push(*meta);
                    }
                }
            }
        }

        if contains_alphaq {
            dump_jupiter_route_accounts_for_program(
                &ALPHAQ_PROGRAM_ID,
                "alphaq",
                tx,
                alts,
                &account_metas,
                cache,
                &synthetic_readonly_system_accounts,
                &created_by_setup,
            );
        }

        if !missing_accounts.is_empty() || !missing_programs.is_empty() {
            let missing_total = missing_accounts.len() + missing_programs.len();
            metrics
                .sim_missing_account
                .fetch_add(missing_total as u64, Ordering::Relaxed);
            for meta in &missing_accounts {
                eprintln!(
                    "[sim_not_executed] reason=preflight_missing_account route_sig={:032x} route_labels={} programs={} account={} source={} is_signer={} is_writable={} suggested_action=add_to_manual_sim_accounts",
                    route_sig,
                    route_labels,
                    route_programs,
                    meta.pubkey,
                    meta.source.as_str(),
                    meta.is_signer,
                    meta.is_writable
                );
            }
            for meta in &missing_programs {
                eprintln!(
                    "[sim_not_executed] reason=preflight_missing_program route_sig={:032x} route_labels={} programs={} account={} source={} is_signer={} is_writable={} suggested_action=fix_program_registry_or_so",
                    route_sig,
                    route_labels,
                    route_programs,
                    meta.pubkey,
                    meta.source.as_str(),
                    meta.is_signer,
                    meta.is_writable
                );
            }
            anyhow::bail!(
                "sim missing account data/programs; missing_accounts={} account_sample=[{}] missing_programs={} program_sample=[{}]",
                missing_accounts.len(),
                sample_account_metas(&missing_accounts),
                missing_programs.len(),
                sample_account_metas(&missing_programs)
            );
        }

        let mut svm = self.svm.lock().unwrap();

        // Advance the SVM clock to the live Yellowstone slot.
        let live_slot = self.current_slot.load(Ordering::Relaxed);
        let live_unix_timestamp = self.current_unix_timestamp.load(Ordering::Relaxed);
        svm.warp_to_slot(live_slot);
        set_live_clock(&mut svm, live_slot, live_unix_timestamp);
        if contains_whirlpool {
            log_sim_clock(live_slot, live_unix_timestamp);
        }

        // Inject ALT raw accounts so the SVM sanitizer can expand v0 address
        // lookups. The transaction compiler consumes AddressLookupTableAccount
        // directly, but LiteSVM sanitization reads raw ALT accounts.
        for alt in alts {
            let raw = synthetic_alt_account(alt)?;
            if let Err(e) = svm.set_account(pk_to_addr(alt.key), raw) {
                warn!(alt = %alt.key, error = ?e, "set_account(ALT) failed");
            }
        }

        // Inject live account state from the Yellowstone cache.
        // Executable accounts (programs) are already loaded via
        // add_program_from_file and must NOT be overwritten here.
        let mut injected = 0usize;
        for meta in &account_metas {
            let pk = &meta.pubkey;
            if is_known_sysvar(pk) || self.loaded_programs.contains(pk) || is_builtin_program(pk) {
                continue;
            }

            if self.jito_tip_accounts.contains(pk) {
                svm.set_account(pk_to_addr(*pk), synthetic_system_account(1_000_000_000))?;
                injected += 1;
                continue;
            }

            if *pk == self.payer_pubkey {
                svm.set_account(pk_to_addr(*pk), synthetic_system_account(10_000_000_000))?;
                injected += 1;
                continue;
            }

            if synthetic_readonly_system_accounts.contains(pk) {
                svm.set_account(pk_to_addr(*pk), synthetic_system_account(0))?;
                injected += 1;
                continue;
            }

            match cache.get(pk) {
                Some(acct) => {
                    if acct.executable() {
                        continue;
                    }
                    if let Err(e) = svm.set_account(pk_to_addr(*pk), acct) {
                        warn!(pubkey = %pk, error = ?e, "set_account failed");
                    } else {
                        injected += 1;
                    }
                }
                None => {
                    if created_by_setup.contains(pk) {
                        continue;
                    }
                }
            }
        }
        debug!(
            injected,
            lazy_fetched,
            accounts = account_metas.len(),
            "sim prepared"
        );

        let wsol_before = parse_wsol_amount(&svm, self.wsol_ata);

        // Convert solana-sdk 2.x VersionedTransaction → solana-transaction 3.x.
        let litesvm_tx = to_litesvm_tx(tx)?;

        // DEX programs touched by this route — attributed to per-DEX sim stats.
        let dex_programs = dex_programs_in_tx(tx, alts);

        eprintln!(
            "[sim_executed] route_sig={:032x} source={} route_labels={} programs={}",
            route_sig, ix_source, route_labels, route_programs
        );
        match svm.simulate_transaction(litesvm_tx) {
            Ok(info) => {
                // The route executed without reverting — every DEX in it is
                // "seen" with no culprit (unprofitable is not a DEX failure).
                metrics.record_dex_sim(&dex_programs, None);
                // post_accounts: Vec<(Address, AccountSharedData)>
                let wsol_ata_addr = pk_to_addr(self.wsol_ata);
                let wsol_after = info
                    .post_accounts
                    .iter()
                    .find(|(addr, _)| *addr == wsol_ata_addr)
                    .and_then(|(_, acc)| parse_token_amount(acc.data()))
                    .unwrap_or(wsol_before);

                let cu = info.meta.compute_units_consumed;

                let min_wsol_after = wsol_before.saturating_add(min_wsol_gain);
                if wsol_after < min_wsol_after {
                    metrics.tx_dropped.fetch_add(1, Ordering::Relaxed);
                    eprintln!(
                        "[sim_executed_revert] route_sig={:032x} reason=unprofitable wsol_before={} wsol_after={} min_after={}",
                        route_sig, wsol_before, wsol_after, min_wsol_after
                    );
                    anyhow::bail!(
                        "sim unprofitable: wsol_before={} wsol_after={} min_after={}",
                        wsol_before,
                        wsol_after,
                        min_wsol_after
                    );
                }
                eprintln!(
                    "[sim_executed_ok] route_sig={:032x} compute_units={} wsol_before={} wsol_after={}",
                    route_sig, cu, wsol_before, wsol_after
                );
                Ok(SimOutcome {
                    compute_units: cu,
                    wsol_before,
                    wsol_after,
                })
            }
            Err(meta) => {
                let lite_err = format!("{:?}", meta.err);
                eprintln!(
                    "[sim_executed_revert] route_sig={:032x} source={} err={}",
                    route_sig, ix_source, lite_err
                );
                let generic_failed_program = first_failed_program_for_revert(&meta.meta.logs);
                let mut debugged_failed_program = false;
                let mut compared_alphaq_invalid_owner = false;
                let invalid_owner_program =
                    if is_invalid_account_owner(&lite_err, &meta.meta.logs) {
                        first_failed_program_for_invalid_owner(&meta.meta.logs)
                    } else {
                        None
                    };
                // Attribute this revert to the responsible DEX (prefer the inner
                // program that hit InvalidAccountOwner, else the first failure).
                metrics.record_dex_sim(
                    &dex_programs,
                    invalid_owner_program.or(generic_failed_program),
                );
                // Record genuinely-new, non-slippage sim errors to a small
                // operator-facing file. Deduped by (program + error-kind +
                // culprit account) so repeats and slippage aborts are skipped.
                log_new_sim_error(
                    &self.manual_accounts_root,
                    route_sig,
                    route_labels,
                    route_programs,
                    &lite_err,
                    invalid_owner_program.or(generic_failed_program).as_ref(),
                    None,
                );
                if let Some(program) = invalid_owner_program {
                    eprintln!(
                        "[sim_invalid_account_owner] failed_program={} route_mentions_alphaq={} alphaq_route_accounts={} hint=account owner mismatch in local SVM; compare same tx with RPC and inspect route account dump",
                        program,
                        contains_alphaq,
                        alphaq_route_accounts.len()
                    );
                    dump_jupiter_route_accounts_for_program(
                        &program,
                        "invalid_account_owner",
                        tx,
                        alts,
                        &account_metas,
                        cache,
                        &synthetic_readonly_system_accounts,
                        &created_by_setup,
                    );
                    if program == ALPHAQ_PROGRAM_ID {
                        metrics
                            .sim_alphaq_invalid_owner
                            .fetch_add(1, Ordering::Relaxed);
                        compare_revert_with_rpc(
                            cache,
                            tx,
                            "alphaq_invalid_account_owner",
                            &program,
                            &lite_err,
                            metrics,
                            route_sig,
                            route_labels,
                            route_programs,
                            ix_source,
                        );
                        self.debug_failed_program_environment(
                            &program,
                            cache,
                            alt_cache,
                            tx,
                            alts,
                            &account_metas,
                            &synthetic_readonly_system_accounts,
                            &created_by_setup,
                            route_sig,
                            route_labels,
                            route_programs,
                            &lite_err,
                            ix_source,
                            metrics,
                        );
                        debugged_failed_program = true;
                        compared_alphaq_invalid_owner = true;
                        // Retry with fresh RPC snapshot: if local cache had a stale/wrong-owner
                        // account, the snapshot will correct it. If the retry passes and the
                        // route is profitable, return it so the bundle can be submitted.
                        if let Some(outcome) = self.retry_with_rpc_snapshot(
                            &program,
                            cache,
                            tx,
                            alts,
                            &account_metas,
                            &synthetic_readonly_system_accounts,
                            &created_by_setup,
                            min_wsol_gain,
                            metrics,
                            route_sig,
                            route_labels,
                            route_programs,
                            &lite_err,
                            ix_source,
                        ) {
                            metrics.sim_alphaq_owner_rpc_retry_ok.fetch_add(1, Ordering::Relaxed);
                            return Ok(outcome);
                        }
                    } else {
                        compare_revert_with_rpc(
                            cache,
                            tx,
                            "invalid_account_owner",
                            &program,
                            &lite_err,
                            metrics,
                            route_sig,
                            route_labels,
                            route_programs,
                            ix_source,
                        );
                    }
                }
                if contains_alphaq && !compared_alphaq_invalid_owner {
                    compare_revert_with_rpc(
                        cache,
                        tx,
                        "alphaq",
                        &ALPHAQ_PROGRAM_ID,
                        &lite_err,
                        metrics,
                        route_sig,
                        route_labels,
                        route_programs,
                        ix_source,
                    );
                }
                if contains_whirlpool && is_whirlpool_invalid_timestamp(&lite_err, &meta.meta.logs)
                {
                    eprintln!(
                        "[sim_whirlpool_invalid_timestamp] slot={} unix_timestamp={} program={} code=6022 hex=0x1786 reason=InvalidTimestamp",
                        live_slot, live_unix_timestamp, WHIRLPOOL_PROGRAM_ID
                    );
                    compare_revert_with_rpc(
                        cache,
                        tx,
                        "whirlpool",
                        &WHIRLPOOL_PROGRAM_ID,
                        &lite_err,
                        metrics,
                        route_sig,
                        route_labels,
                        route_programs,
                        ix_source,
                    );
                }
                if is_declared_program_id_mismatch(&lite_err, &meta.meta.logs) {
                    if let Some(program) = first_failed_program_from_logs(&meta.meta.logs) {
                        let registry_file = self
                            .loaded_program_files
                            .get(&program)
                            .map(String::as_str)
                            .unwrap_or("unknown");
                        eprintln!(
                            "[sim_program_id_mismatch] invoked_program={} registry_file={} hint=wrong .so mapped to program id; dump exact on-chain program binary and remap registry",
                            program, registry_file
                        );
                        compare_revert_with_rpc(
                            cache,
                            tx,
                            "declared_program_id_mismatch",
                            &program,
                            &lite_err,
                            metrics,
                            route_sig,
                            route_labels,
                            route_programs,
                            ix_source,
                        );
                    } else {
                        eprintln!(
                            "[sim_program_id_mismatch] invoked_program=unknown registry_file=unknown hint=wrong .so mapped to program id; dump exact on-chain program binary and remap registry"
                        );
                    }
                }
                if let Some(program) = generic_failed_program {
                    if ix_source == "route_template" && program == SOLFI_V2_PROGRAM_ID {
                        eprintln!(
                            "[template_disable] route_sig={:032x} route_labels={} programs={} reason=sim_failed_solfi_v2_0x17 action=fresh_metis_only",
                            route_sig,
                            route_labels,
                            route_programs
                        );
                    }
                    if !debugged_failed_program {
                        compare_revert_with_rpc(
                            cache,
                            tx,
                            "failed_program",
                            &program,
                            &lite_err,
                            metrics,
                            route_sig,
                            route_labels,
                            route_programs,
                            ix_source,
                        );
                        self.debug_failed_program_environment(
                            &program,
                            cache,
                            alt_cache,
                            tx,
                            alts,
                            &account_metas,
                            &synthetic_readonly_system_accounts,
                            &created_by_setup,
                            route_sig,
                            route_labels,
                            route_programs,
                            &lite_err,
                            ix_source,
                            metrics,
                        );
                        if should_retry_with_rpc_snapshot(&program, &lite_err, &meta.meta.logs) {
                            if let Some(outcome) = self.retry_with_rpc_snapshot(
                                &program,
                                cache,
                                tx,
                                alts,
                                &account_metas,
                                &synthetic_readonly_system_accounts,
                                &created_by_setup,
                                min_wsol_gain,
                                metrics,
                                route_sig,
                                route_labels,
                                route_programs,
                                &lite_err,
                                ix_source,
                            ) {
                                return Ok(outcome);
                            }
                        }
                    }
                }
                if self.fail_closed {
                    anyhow::bail!(
                        "sim reverted: err={:?} logs={:#?}",
                        meta.err,
                        meta.meta.logs
                    );
                } else {
                    warn!(
                        err = ?meta.err,
                        logs = ?meta.meta.logs,
                        "sim reverted but fail_open=true, allowing send"
                    );
                    Ok(SimOutcome {
                        compute_units: meta.meta.compute_units_consumed,
                        wsol_before,
                        wsol_after: 0,
                    })
                }
            }
        }
    }
}

/// Read the SPL Token amount field from raw account data.
/// Offset 64..72 is the `amount` field in the spl_token::state::Account layout.
fn parse_token_amount(data: &[u8]) -> Option<u64> {
    if data.len() < 72 {
        return None;
    }
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&data[64..72]);
    Some(u64::from_le_bytes(buf))
}

fn parse_wsol_amount(svm: &LiteSVM, wsol_ata: Pubkey) -> u64 {
    let addr = pk_to_addr(wsol_ata);
    svm.get_account(&addr)
        .and_then(|a| parse_token_amount(a.data()))
        .unwrap_or(0)
}

fn set_live_clock(svm: &mut LiteSVM, slot: u64, unix_timestamp: i64) {
    let mut clock = svm.get_sysvar::<Clock>();
    clock.slot = slot;
    if unix_timestamp > 0 {
        clock.unix_timestamp = unix_timestamp;
    }
    svm.set_sysvar::<Clock>(&clock);
}

fn log_sim_clock(slot: u64, unix_timestamp: i64) {
    if SIM_CLOCK_LOG_COUNT.fetch_add(1, Ordering::Relaxed) >= MAX_SIM_CLOCK_LOGS_PER_PROCESS {
        return;
    }
    eprintln!(
        "[sim_clock] slot={} unix_timestamp={} estimated_from_slot=true",
        slot, unix_timestamp
    );
}

fn is_whirlpool_invalid_timestamp(lite_err: &str, logs: &[String]) -> bool {
    lite_err.contains("Custom(6022)")
        || logs.iter().any(|line| {
            line.contains("InvalidTimestamp")
                || line.contains("Error Number: 6022")
                || line.contains("custom program error: 0x1786")
        })
}

fn is_declared_program_id_mismatch(lite_err: &str, logs: &[String]) -> bool {
    lite_err.contains("Custom(4100)")
        || logs.iter().any(|line| {
            line.contains("DeclaredProgramIdMismatch")
                || line.contains("Error Number: 4100")
                || line.contains("custom program error: 0x1004")
        })
}

fn should_retry_with_rpc_snapshot(
    _failed_program: &Pubkey,
    _lite_err: &str,
    _logs: &[String],
) -> bool {
    true
}

fn is_invalid_account_owner(lite_err: &str, logs: &[String]) -> bool {
    lite_err.contains("InvalidAccountOwner")
        || lite_err.contains("Invalid account owner")
        || logs.iter().any(|line| {
            line.contains("InvalidAccountOwner") || line.contains("Invalid account owner")
        })
}

fn first_failed_program_from_logs(logs: &[String]) -> Option<Pubkey> {
    let mut outer_failure = None;
    for line in logs {
        if line.contains("failed: custom program error: 0x1004") {
            if let Some(pk) = parse_log_program_id(line) {
                if pk != JUPITER_PROGRAM_ID {
                    return Some(pk);
                }
                outer_failure.get_or_insert(pk);
            }
        }
    }
    if outer_failure.is_some() {
        return outer_failure;
    }

    let mismatch_idx = logs.iter().position(|line| {
        line.contains("DeclaredProgramIdMismatch") || line.contains("Error Number: 4100")
    })?;
    logs[..mismatch_idx]
        .iter()
        .rev()
        .find_map(|line| parse_log_program_id(line))
}

fn first_failed_program_for_revert(logs: &[String]) -> Option<Pubkey> {
    let mut outer_failure = None;
    for line in logs {
        if line.contains(" failed:") {
            if let Some(pk) = parse_log_program_id(line) {
                if pk != JUPITER_PROGRAM_ID {
                    return Some(pk);
                }
                outer_failure.get_or_insert(pk);
            }
        }
    }
    outer_failure
}

fn first_failed_program_for_invalid_owner(logs: &[String]) -> Option<Pubkey> {
    for line in logs {
        if line.contains("failed:")
            && (line.contains("InvalidAccountOwner") || line.contains("Invalid account owner"))
        {
            if let Some(pk) = parse_log_program_id(line) {
                return Some(pk);
            }
        }
    }

    logs.iter()
        .find(|line| line.contains("failed:"))
        .and_then(|line| parse_log_program_id(line))
}

fn parse_log_program_id(line: &str) -> Option<Pubkey> {
    let rest = line.strip_prefix("Program ")?;
    let id = rest.split_whitespace().next()?;
    Pubkey::try_from(id).ok()
}

fn synthetic_alt_account(alt: &AddressLookupTableAccount) -> Result<Account> {
    let table = AddressLookupTable {
        meta: LookupTableMeta::default(),
        addresses: Cow::Owned(alt.addresses.clone()),
    };
    let data = table
        .serialize_for_tests()
        .map_err(|e| anyhow!("serialize ALT {} failed: {e:?}", alt.key))?;
    Ok(Account {
        lamports: 1,
        data,
        owner: LsAddr::from(address_lookup_table::program::id().to_bytes()),
        executable: false,
        rent_epoch: 0,
    })
}

fn synthetic_system_account(lamports: u64) -> Account {
    Account {
        lamports,
        data: vec![],
        owner: LsAddr::from(system_program::id().to_bytes()),
        executable: false,
        rent_epoch: 0,
    }
}

fn is_known_sysvar(pk: &Pubkey) -> bool {
    *pk == solana_sdk::sysvar::clock::id()
        || *pk == solana_sdk::sysvar::epoch_schedule::id()
        || *pk == solana_sdk::sysvar::fees::id()
        || *pk == solana_sdk::sysvar::instructions::id()
        || *pk == solana_sdk::sysvar::recent_blockhashes::id()
        || *pk == solana_sdk::sysvar::rent::id()
        || *pk == solana_sdk::sysvar::slot_hashes::id()
        || *pk == solana_sdk::sysvar::slot_history::id()
        || *pk == solana_sdk::sysvar::stake_history::id()
}

fn is_builtin_program(pk: &Pubkey) -> bool {
    *pk == system_program::id()
        || *pk == solana_sdk::compute_budget::id()
        || *pk == solana_sdk::bpf_loader::id()
        || *pk == solana_sdk::bpf_loader_deprecated::id()
        || *pk == solana_sdk::bpf_loader_upgradeable::id()
        || *pk == solana_sdk::address_lookup_table::program::id()
        || *pk == Pubkey::from_str_const("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")
        || *pk == Pubkey::from_str_const("TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb")
        || *pk == Pubkey::from_str_const("ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL")
        || *pk == Pubkey::from_str_const("MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr")
}

fn collect_accounts_created_by_setup(
    tx: &VersionedTransaction,
    alts: &[AddressLookupTableAccount],
) -> HashSet<Pubkey> {
    let resolved_keys = resolve_tx_account_keys(tx, alts);
    let mut created = HashSet::new();

    match &tx.message {
        VersionedMessage::Legacy(msg) => {
            for ix in &msg.instructions {
                collect_created_from_instruction(ix.program_id_index, &ix.accounts, &ix.data, &resolved_keys, &mut created);
            }
        }
        VersionedMessage::V0(v0) => {
            for ix in &v0.instructions {
                collect_created_from_instruction(ix.program_id_index, &ix.accounts, &ix.data, &resolved_keys, &mut created);
            }
        }
    }

    created
}

fn collect_created_from_instruction(
    program_id_index: u8,
    accounts: &[u8],
    data: &[u8],
    resolved_keys: &[Pubkey],
    created: &mut HashSet<Pubkey>,
) {
    let Some(program_id) = resolved_keys.get(program_id_index as usize) else {
        return;
    };

    if *program_id == ASSOCIATED_TOKEN_PROGRAM_ID {
        if let Some(account_idx) = accounts.get(1) {
            if let Some(created_account) = resolved_keys.get(*account_idx as usize) {
                created.insert(*created_account);
            }
        }
        return;
    }

    if *program_id == system_program::id()
        && is_system_create_account_instruction(data)
        && accounts.len() > 1
    {
        if let Some(created_account) = resolved_keys.get(accounts[1] as usize) {
            created.insert(*created_account);
        }
    }
}

fn is_system_create_account_instruction(data: &[u8]) -> bool {
    match bincode::deserialize::<system_instruction::SystemInstruction>(data) {
        Ok(system_instruction::SystemInstruction::CreateAccount { .. }) => true,
        Ok(system_instruction::SystemInstruction::CreateAccountWithSeed { .. }) => true,
        _ => false,
    }
}

fn transaction_mentions_program(
    tx: &VersionedTransaction,
    alts: &[AddressLookupTableAccount],
    program: &Pubkey,
) -> bool {
    resolve_tx_account_keys(tx, alts)
        .iter()
        .any(|pk| pk == program)
}

/// The set of registered DEX/aggregator program ids referenced by this tx.
/// Used to attribute simulation outcomes to specific exchanges.
fn dex_programs_in_tx(
    tx: &VersionedTransaction,
    alts: &[AddressLookupTableAccount],
) -> Vec<Pubkey> {
    let registry: HashSet<Pubkey> = crate::program_registry::PROGRAMS
        .iter()
        .filter_map(|(id, _)| Pubkey::try_from(*id).ok())
        .collect();
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for pk in resolve_tx_account_keys(tx, alts) {
        if registry.contains(&pk) && seen.insert(pk) {
            out.push(pk);
        }
    }
    out
}

fn jupiter_route_account_keys_for_program(
    tx: &VersionedTransaction,
    alts: &[AddressLookupTableAccount],
    target_program: &Pubkey,
) -> HashSet<Pubkey> {
    let resolved_keys = resolve_tx_account_keys(tx, alts);
    let mut out = HashSet::new();

    match &tx.message {
        VersionedMessage::Legacy(msg) => {
            for ix in &msg.instructions {
                collect_jupiter_route_keys_for_instruction(
                    ix.program_id_index,
                    &ix.accounts,
                    &resolved_keys,
                    target_program,
                    &mut out,
                );
            }
        }
        VersionedMessage::V0(v0) => {
            for ix in &v0.instructions {
                collect_jupiter_route_keys_for_instruction(
                    ix.program_id_index,
                    &ix.accounts,
                    &resolved_keys,
                    target_program,
                    &mut out,
                );
            }
        }
    }

    out
}

fn collect_jupiter_route_keys_for_instruction(
    program_id_index: u8,
    account_indexes: &[u8],
    resolved_keys: &[Pubkey],
    target_program: &Pubkey,
    out: &mut HashSet<Pubkey>,
) {
    let Some(program_id) = resolved_keys.get(program_id_index as usize) else {
        return;
    };
    if *program_id != JUPITER_PROGRAM_ID {
        return;
    }

    let mentions_target = account_indexes.iter().any(|raw_idx| {
        resolved_keys
            .get(*raw_idx as usize)
            .is_some_and(|pk| pk == target_program)
    });
    if !mentions_target {
        return;
    }

    for raw_idx in account_indexes {
        if let Some(pubkey) = resolved_keys.get(*raw_idx as usize) {
            out.insert(*pubkey);
        }
    }
}

fn resolve_tx_account_keys(
    tx: &VersionedTransaction,
    alts: &[AddressLookupTableAccount],
) -> Vec<Pubkey> {
    let mut out: Vec<Pubkey> = tx.message.static_account_keys().to_vec();
    if let VersionedMessage::V0(v0) = &tx.message {
        for lookup in &v0.address_table_lookups {
            let Some(alt) = alts.iter().find(|a| a.key == lookup.account_key) else {
                continue;
            };
            for &idx in &lookup.writable_indexes {
                if let Some(addr) = alt.addresses.get(idx as usize) {
                    out.push(*addr);
                }
            }
            for &idx in &lookup.readonly_indexes {
                if let Some(addr) = alt.addresses.get(idx as usize) {
                    out.push(*addr);
                }
            }
        }
    }
    out
}

/// Collect every unique account key referenced by the transaction, including
/// signer/writable flags and v0 ALT source metadata for missing-account logs.
fn collect_tx_account_metas(
    tx: &VersionedTransaction,
    alts: &[AddressLookupTableAccount],
) -> Vec<TxAccountMeta> {
    let mut out = Vec::new();
    let mut seen = HashMap::new();

    match &tx.message {
        VersionedMessage::Legacy(msg) => {
            collect_static_account_metas(&msg.header, &msg.account_keys, &mut out, &mut seen);
        }
        VersionedMessage::V0(v0) => {
            collect_static_account_metas(&v0.header, &v0.account_keys, &mut out, &mut seen);
            let mut resolved_idx = v0.account_keys.len();
            for lookup in &v0.address_table_lookups {
                let alt = match alts.iter().find(|a| a.key == lookup.account_key) {
                    Some(a) => a,
                    None => continue,
                };
                for &idx in &lookup.writable_indexes {
                    if let Some(addr) = alt.addresses.get(idx as usize) {
                        push_account_meta(
                            &mut out,
                            &mut seen,
                            TxAccountMeta {
                                pubkey: *addr,
                                is_signer: false,
                                is_writable: true,
                                source: TxAccountSource::AltWritable,
                                tx_key_index: Some(resolved_idx),
                                alt_table: Some(lookup.account_key),
                                alt_index: Some(idx),
                            },
                        );
                        resolved_idx += 1;
                    }
                }
                for &idx in &lookup.readonly_indexes {
                    if let Some(addr) = alt.addresses.get(idx as usize) {
                        push_account_meta(
                            &mut out,
                            &mut seen,
                            TxAccountMeta {
                                pubkey: *addr,
                                is_signer: false,
                                is_writable: false,
                                source: TxAccountSource::AltReadonly,
                                tx_key_index: Some(resolved_idx),
                                alt_table: Some(lookup.account_key),
                                alt_index: Some(idx),
                            },
                        );
                        resolved_idx += 1;
                    }
                }
            }
        }
    }

    out.sort_unstable_by_key(|meta| meta.pubkey);
    out
}

fn collect_static_account_metas(
    header: &MessageHeader,
    keys: &[Pubkey],
    out: &mut Vec<TxAccountMeta>,
    seen: &mut HashMap<Pubkey, usize>,
) {
    for (idx, key) in keys.iter().enumerate() {
        let is_signer = idx < header.num_required_signatures as usize;
        push_account_meta(
            out,
            seen,
            TxAccountMeta {
                pubkey: *key,
                is_signer,
                is_writable: is_static_account_writable(idx, keys.len(), header),
                source: TxAccountSource::Static,
                tx_key_index: Some(idx),
                alt_table: None,
                alt_index: None,
            },
        );
    }
}

fn is_static_account_writable(idx: usize, key_count: usize, header: &MessageHeader) -> bool {
    let signer_count = header.num_required_signatures as usize;
    if idx < signer_count {
        let readonly_signed_start =
            signer_count.saturating_sub(header.num_readonly_signed_accounts as usize);
        idx < readonly_signed_start
    } else {
        let readonly_unsigned_start =
            key_count.saturating_sub(header.num_readonly_unsigned_accounts as usize);
        idx < readonly_unsigned_start
    }
}

fn push_account_meta(
    out: &mut Vec<TxAccountMeta>,
    seen: &mut HashMap<Pubkey, usize>,
    meta: TxAccountMeta,
) {
    if let Some(idx) = seen.get(&meta.pubkey).copied() {
        let existing = &mut out[idx];
        existing.is_signer |= meta.is_signer;
        existing.is_writable |= meta.is_writable;
        if meta.source == TxAccountSource::Static
            || (existing.source != TxAccountSource::Static && meta.is_writable)
        {
            existing.source = meta.source;
            existing.tx_key_index = meta.tx_key_index;
            existing.alt_table = meta.alt_table;
            existing.alt_index = meta.alt_index;
        } else if existing.tx_key_index.is_none() {
            existing.tx_key_index = meta.tx_key_index;
            existing.alt_table = meta.alt_table;
            existing.alt_index = meta.alt_index;
        }
        return;
    }
    seen.insert(meta.pubkey, out.len());
    out.push(meta);
}

fn sample_account_metas(metas: &[TxAccountMeta]) -> String {
    metas
        .iter()
        .take(8)
        .map(|meta| meta.pubkey.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

fn log_account_classification(tag: &str, meta: &TxAccountMeta, acct: &Account, note: &str) {
    eprintln!(
        "[{tag}] pk={} kind={} owner={} executable={} data_len={} is_signer={} is_writable={} source={} note={}",
        meta.pubkey,
        account_kind(acct),
        acct.owner(),
        acct.executable(),
        acct.data().len(),
        meta.is_signer,
        meta.is_writable,
        meta.source.as_str(),
        note
    );
}

fn is_registered_program_id(pk: &Pubkey) -> bool {
    crate::program_registry::PROGRAMS
        .iter()
        .filter_map(|(id, _)| Pubkey::try_from(*id).ok())
        .any(|program_id| program_id == *pk)
}

fn should_synthetic_readonly_system(meta: &TxAccountMeta) -> bool {
    // Fail-open only for readonly, non-signer data accounts when hot-path RPC
    // is disabled. This lets LiteSVM actually execute and, if the program
    // needs real data/owner, the existing RPC-snapshot retry path can replace
    // the synthetic account with chain state.
    //
    // Never synthesize executable/runtime accounts: sysvars, built-ins, and
    // registered DEX/Jupiter program IDs are handled by LiteSVM or by the
    // program loader path and must not become empty System accounts.
    if meta.is_writable || meta.is_signer {
        return false;
    }
    if is_known_sysvar(&meta.pubkey) || is_builtin_program(&meta.pubkey) {
        return false;
    }
    if is_registered_program_id(&meta.pubkey) {
        return false;
    }
    true
}

fn should_allow_synthetic_missing(
    meta: &TxAccountMeta,
    disable_synthetic_for_all_alphaq_accounts: bool,
) -> bool {
    should_synthetic_readonly_system(meta) && !disable_synthetic_for_all_alphaq_accounts
}

fn is_registered_dex_owner_str(owner: &str) -> bool {
    crate::program_registry::PROGRAMS
        .iter()
        .any(|(program_id, _)| *program_id == owner)
}

fn is_live_state_owner_str(owner: &str) -> bool {
    owner == "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
        || owner == "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"
        || is_registered_dex_owner_str(owner)
}

fn is_expected_readonly_pda_authority_meta(meta: &TxAccountMeta) -> bool {
    !meta.is_writable && !meta.is_signer && meta.source == TxAccountSource::AltReadonly
}

fn is_expected_readonly_pda_authority_compare(
    meta: &TxAccountMeta,
    cache_state: &AccountDebugState,
    rpc_state: &AccountDebugState,
) -> bool {
    is_expected_readonly_pda_authority_meta(meta)
        && cache_state.status == "synthetic_system"
        && rpc_state.status == "not_found"
}

fn is_live_state_hash_mismatch(
    meta: &TxAccountMeta,
    cache_state: &AccountDebugState,
    rpc_state: &AccountDebugState,
    outcome: &AccountCompareOutcome,
) -> bool {
    meta.is_writable
        && !outcome.matches_rpc
        && outcome.reason == "data_hash_mismatch"
        && outcome.owner_match
        && outcome.data_len_match
        && outcome.lamports_match
        && cache_state.status == "found"
        && rpc_state.status == "found"
        && is_live_state_owner_str(&cache_state.owner)
}

fn log_missing_not_found(meta: &TxAccountMeta) {
    eprintln!(
        "[sim_missing_full] pk={} kind=not_found owner=? executable=false data_len=0 is_signer={} is_writable={} source={} error_kind=not_found",
        meta.pubkey,
        meta.is_signer,
        meta.is_writable,
        meta.source.as_str()
    );
}

fn log_missing_writable(meta: &TxAccountMeta, reason: &str) {
    eprintln!(
        "[sim_missing_writable] pk={} source={} is_signer={} is_writable={} created_by_setup=false reason={}",
        meta.pubkey,
        meta.source.as_str(),
        meta.is_signer,
        meta.is_writable,
        reason
    );
}

fn log_missing_fetch_error(meta: &TxAccountMeta, error_kind: &str, error: &str) {
    eprintln!(
        "[sim_missing_full] pk={} kind=missing owner=? executable=? data_len=? is_signer={} is_writable={} source={} error_kind={} error={}",
        meta.pubkey,
        meta.is_signer,
        meta.is_writable,
        meta.source.as_str(),
        error_kind,
        error
    );
}

fn log_missing_alphaq_route_account(meta: &TxAccountMeta, reason: &str) {
    eprintln!(
        "[sim_missing_alphaq_route_account] pk={} source={} is_signer={} is_writable={} reason={} action=no_synthetic_system_account",
        meta.pubkey,
        meta.source.as_str(),
        meta.is_signer,
        meta.is_writable,
        reason
    );
}

fn dump_jupiter_route_accounts_for_program(
    target_program: &Pubkey,
    label: &str,
    tx: &VersionedTransaction,
    alts: &[AddressLookupTableAccount],
    metas: &[TxAccountMeta],
    cache: &AccountCache,
    synthetic_readonly_system_accounts: &HashSet<Pubkey>,
    created_by_setup: &HashSet<Pubkey>,
) {
    let resolved_keys = resolve_tx_account_keys(tx, alts);
    let meta_by_key: HashMap<Pubkey, TxAccountMeta> =
        metas.iter().map(|meta| (meta.pubkey, *meta)).collect();

    match &tx.message {
        VersionedMessage::Legacy(msg) => {
            for (ix_index, ix) in msg.instructions.iter().enumerate() {
                dump_jupiter_route_instruction_accounts_for_program(
                    target_program,
                    label,
                    ix_index,
                    ix.program_id_index,
                    &ix.accounts,
                    &resolved_keys,
                    &meta_by_key,
                    cache,
                    synthetic_readonly_system_accounts,
                    created_by_setup,
                );
            }
        }
        VersionedMessage::V0(v0) => {
            for (ix_index, ix) in v0.instructions.iter().enumerate() {
                dump_jupiter_route_instruction_accounts_for_program(
                    target_program,
                    label,
                    ix_index,
                    ix.program_id_index,
                    &ix.accounts,
                    &resolved_keys,
                    &meta_by_key,
                    cache,
                    synthetic_readonly_system_accounts,
                    created_by_setup,
                );
            }
        }
    }
}

/// For a route that LiteSVM rejected with `InvalidAccountOwner` (but which
/// succeeds on-chain), dump — for every account in the failing program's
/// Jupiter instruction — the owner LiteSVM actually held vs the owner RPC
/// reports. Any line with `owner_match=false` is the smoking gun: an account
/// whose owner LiteSVM got wrong (e.g. a synthetic System-owned placeholder
/// injected over a real DEX/token account), which is exactly what makes the
/// program's internal owner check fail locally while passing on-chain.
fn dump_failing_program_owner_diagnosis(
    svm: &LiteSVM,
    failing_program: &Pubkey,
    tx: &VersionedTransaction,
    alts: &[AddressLookupTableAccount],
    rpc_accounts: &HashMap<Pubkey, AccountFetchResult>,
    route_sig: u128,
    ix_source: &str,
) {
    let resolved_keys = resolve_tx_account_keys(tx, alts);

    let mut handle_instruction = |program_id_index: u8, account_indexes: &[u8], ix_index: usize| {
        let Some(program_id) = resolved_keys.get(program_id_index as usize) else {
            return;
        };
        if *program_id != JUPITER_PROGRAM_ID {
            return;
        }
        let mentions = account_indexes
            .iter()
            .any(|i| resolved_keys.get(*i as usize) == Some(failing_program));
        if !mentions {
            return;
        }
        for (slot, raw_idx) in account_indexes.iter().enumerate() {
            let Some(pk) = resolved_keys.get(*raw_idx as usize) else {
                continue;
            };
            let svm_owner = svm
                .get_account(&pk_to_addr(*pk))
                .map(|a| a.owner().to_string())
                .unwrap_or_else(|| "absent".to_string());
            let (rpc_owner, rpc_status) = match rpc_accounts.get(pk) {
                Some(AccountFetchResult::Found(a)) => (a.owner().to_string(), "found"),
                Some(AccountFetchResult::NotFound) => ("none".to_string(), "not_found"),
                Some(AccountFetchResult::Error { .. }) => ("?".to_string(), "rpc_error"),
                None => ("?".to_string(), "no_result"),
            };
            let is_runtime_sysvar = is_known_sysvar(pk);
            let is_runtime_program = is_builtin_program(pk);
            let expected_runtime_account_gap =
                is_runtime_sysvar && svm_owner == "absent" && rpc_status == "not_found";
            let runtime_loader_compat =
                is_runtime_program && rpc_status == "found" && svm_owner != rpc_owner;

            let owner_match = if expected_runtime_account_gap || runtime_loader_compat {
                true
            } else {
                rpc_status == "found" && svm_owner == rpc_owner
            };

            // Do not report runtime/sysvar implementation details as bad DEX accounts.
            //
            // * Sysvar Instructions is transaction-scoped; RPC commonly returns not_found and
            //   LiteSVM may not expose it through get_account even though the runtime can still
            //   supply instruction-sysvar data to programs.
            // * SPL Token can be executed by LiteSVM as a default/builtin program while the
            //   mainnet program account is owned by the upgradeable loader. That loader-owner
            //   difference is not a pool/vault account problem.
            let flag = if expected_runtime_account_gap {
                "EXPECTED_RUNTIME_SYSVAR"
            } else if runtime_loader_compat {
                "RUNTIME_PROGRAM_LOADER_COMPAT"
            } else if rpc_status == "found" && svm_owner != rpc_owner {
                "OWNER_MISMATCH"
            } else if svm_owner == "absent" {
                "ABSENT_IN_SVM"
            } else {
                "ok"
            };
            eprintln!(
                "[sim_owner_diagnosis] route_sig={:032x} source={} failing_program={} jupiter_ix_index={} account_slot={} pubkey={} svm_owner={} rpc_owner={} rpc_status={} owner_match={} flag={}",
                route_sig,
                ix_source,
                failing_program,
                ix_index,
                slot,
                pk,
                svm_owner,
                rpc_owner,
                rpc_status,
                owner_match,
                flag
            );
        }
    };

    match &tx.message {
        VersionedMessage::Legacy(msg) => {
            for (ix_index, ix) in msg.instructions.iter().enumerate() {
                handle_instruction(ix.program_id_index, &ix.accounts, ix_index);
            }
        }
        VersionedMessage::V0(v0) => {
            for (ix_index, ix) in v0.instructions.iter().enumerate() {
                handle_instruction(ix.program_id_index, &ix.accounts, ix_index);
            }
        }
    }
}

fn validate_alts_for_failed_tx(
    cache: &AccountCache,
    alt_cache: &crate::alt_cache::AltCache,
    alts: &[AddressLookupTableAccount],
    failed_program: &Pubkey,
    route_sig: u128,
    route_labels: &str,
    route_programs: &str,
    ix_source: &str,
) {
    let alt_keys = alts.iter().map(|alt| alt.key).collect::<Vec<_>>();
    if alt_keys.is_empty() {
        return;
    }

    let rpc_fetch = cache.fetch_accounts_for_compare(&alt_keys);
    let rpc_accounts = &rpc_fetch.accounts;
    for alt in alts {
        let cached_hash = hash_pubkeys_hex(&alt.addresses);
        match rpc_accounts.get(&alt.key) {
            Some(AccountFetchResult::Found(rpc_account)) => {
                match crate::transaction::deserialize_alt_addresses(rpc_account.data()) {
                    Ok(rpc_addresses) => {
                        let rpc_hash = hash_pubkeys_hex(&rpc_addresses);
                        let status = if rpc_addresses == alt.addresses {
                            "same"
                        } else {
                            "different"
                        };
                        eprintln!(
                            "[alt_validate_for_failed_tx] route_sig={:032x} source={} route_labels={} programs={} failed_program={} alt={} cached_addresses_len={} rpc_addresses_len={} cached_hash={} rpc_hash={} status={}",
                            route_sig,
                            ix_source,
                            route_labels,
                            route_programs,
                            failed_program,
                            alt.key,
                            alt.addresses.len(),
                            rpc_addresses.len(),
                            cached_hash,
                            rpc_hash,
                            status
                        );
                        if status == "different" {
                            match alt_cache.update_from_account_data(alt.key, rpc_account) {
                                Ok(_) => eprintln!(
                                    "[alt_validate_for_failed_tx] route_sig={:032x} source={} failed_program={} alt={} action=refreshed_alt_cache_from_rpc note=rebuild_tx_before_retry",
                                    route_sig, ix_source, failed_program, alt.key
                                ),
                                Err(e) => eprintln!(
                                    "[alt_validate_for_failed_tx] route_sig={:032x} source={} failed_program={} alt={} action=refresh_failed error={}",
                                    route_sig, ix_source, failed_program, alt.key, e
                                ),
                            }
                        }
                    }
                    Err(e) => eprintln!(
                        "[alt_validate_for_failed_tx] route_sig={:032x} source={} route_labels={} programs={} failed_program={} alt={} cached_addresses_len={} rpc_addresses_len=0 cached_hash={} rpc_hash=decode_error status=decode_error error={}",
                        route_sig,
                        ix_source,
                        route_labels,
                        route_programs,
                        failed_program,
                        alt.key,
                        alt.addresses.len(),
                        cached_hash,
                        e
                    ),
                }
            }
            Some(AccountFetchResult::NotFound) => eprintln!(
                "[alt_validate_for_failed_tx] route_sig={:032x} source={} route_labels={} programs={} failed_program={} alt={} cached_addresses_len={} rpc_addresses_len=0 cached_hash={} rpc_hash=missing status=missing",
                route_sig,
                ix_source,
                route_labels,
                route_programs,
                failed_program,
                alt.key,
                alt.addresses.len(),
                cached_hash
            ),
            Some(AccountFetchResult::Error { kind, message }) => eprintln!(
                "[alt_validate_for_failed_tx] route_sig={:032x} source={} route_labels={} programs={} failed_program={} alt={} cached_addresses_len={} rpc_addresses_len=0 cached_hash={} rpc_hash=rpc_error status=rpc_error error_kind={} error={}",
                route_sig,
                ix_source,
                route_labels,
                route_programs,
                failed_program,
                alt.key,
                alt.addresses.len(),
                cached_hash,
                kind,
                message
            ),
            None => eprintln!(
                "[alt_validate_for_failed_tx] route_sig={:032x} source={} route_labels={} programs={} failed_program={} alt={} cached_addresses_len={} rpc_addresses_len=0 cached_hash={} rpc_hash=no_result status=no_result",
                route_sig,
                ix_source,
                route_labels,
                route_programs,
                failed_program,
                alt.key,
                alt.addresses.len(),
                cached_hash
            ),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn compare_tx_accounts_with_rpc_for_failed_program(
    sim: &Simulator,
    failed_program: &Pubkey,
    cache: &AccountCache,
    account_metas: &[TxAccountMeta],
    synthetic_readonly_system_accounts: &HashSet<Pubkey>,
    created_by_setup: &HashSet<Pubkey>,
    route_sig: u128,
    route_labels: &str,
    route_programs: &str,
    ix_source: &str,
    metrics: &Metrics,
) {
    let mut metas = account_metas.to_vec();
    metas.sort_by(|a, b| a.pubkey.cmp(&b.pubkey));
    metas.dedup_by(|a, b| a.pubkey == b.pubkey);
    let keys = metas.iter().map(|meta| meta.pubkey).collect::<Vec<_>>();
    let rpc_fetch = cache.fetch_accounts_for_compare(&keys);
    let rpc_accounts = &rpc_fetch.accounts;
    let cache_slot = cache.stream_slot().load(Ordering::Relaxed);
    let rpc_context_slot = rpc_fetch.rpc_context_slot.unwrap_or(0);
    let slot_delta = slot_delta(cache_slot, rpc_context_slot);
    let mut mismatches = 0usize;

    for meta in &metas {
        let cache_account = cache.get(&meta.pubkey);
        let rpc_result = rpc_accounts.get(&meta.pubkey);
        let cache_state = account_debug_state(
            cache_account.as_ref(),
            synthetic_readonly_system_accounts.contains(&meta.pubkey),
            sim.loaded_programs.contains(&meta.pubkey),
            created_by_setup.contains(&meta.pubkey),
            sim.should_skip_account(&meta.pubkey),
        );
        let rpc_state = match rpc_result {
            Some(AccountFetchResult::Found(account)) => account_debug_state(
                Some(account),
                false,
                false,
                false,
                false,
            ),
            Some(AccountFetchResult::NotFound) => AccountDebugState::marker("not_found"),
            Some(AccountFetchResult::Error { kind, message }) => {
                AccountDebugState::marker(format!("rpc_error:{kind}:{message}"))
            }
            None => AccountDebugState::marker("no_result"),
        };

        let mut outcome = account_compare_result(&cache_state, &rpc_state, cache_account.is_some());
        let mut classification = "normal";
        let mut action = "none";
        let role_infos = sim.mix_roles.get(&meta.pubkey).cloned().unwrap_or_default();
        let role_info = best_mix_role(&role_infos, meta);
        if let Some(info) = role_info {
            if info.account_kind.starts_with("live_") && meta.is_writable && cache_state.status == "found" && rpc_state.status == "found" {
                // Give the account-cache a chance to keep Token-owned vaults and
                // foreign-owned live state fresh via direct Yellowstone account
                // subscriptions. DEX-owned pool state is already covered by the
                // owner filter and note_uncovered_writable will no-op.
                if let Ok(owner_pk) = Pubkey::try_from(cache_state.owner.as_str()) {
                    cache.note_uncovered_writable(meta.pubkey, &owner_pk);
                }
            }
        }

        if is_expected_readonly_pda_authority_compare(meta, &cache_state, &rpc_state) {
            outcome = AccountCompareOutcome::new(
                true,
                "expected_pda_authority_not_initialized",
                false,
                false,
                false,
                false,
            );
            classification = "expected_readonly_pda_authority";
            action = "ignore_not_found_do_not_fetch";
        } else if is_live_state_hash_mismatch(meta, &cache_state, &rpc_state, &outcome) {
            outcome = AccountCompareOutcome::new(
                true,
                "live_state_data_hash_mismatch_refresh_needed",
                outcome.owner_match,
                outcome.data_len_match,
                outcome.lamports_match,
                outcome.data_hash_match,
            );
            classification = "live_state_stale";
            action = "refresh_from_grpc_or_rpc_snapshot";
            eprintln!(
                "[sim_live_state_stale] route_sig={:032x} source={} failed_program={} pubkey={} owner={} account_source={} is_writable={} reason=data_hash_mismatch action=refresh_from_grpc_or_rpc_snapshot",
                route_sig,
                ix_source,
                failed_program,
                meta.pubkey,
                cache_state.owner,
                meta.source.as_str(),
                meta.is_writable
            );
        }

        if !outcome.matches_rpc {
            mismatches += 1;
            metrics
                .sim_state_mismatch_total
                .fetch_add(1, Ordering::Relaxed);
            if meta.is_writable {
                metrics
                    .sim_state_mismatch_writable
                    .fetch_add(1, Ordering::Relaxed);
            } else {
                metrics
                    .sim_state_mismatch_readonly
                    .fetch_add(1, Ordering::Relaxed);
            }
        }
        eprintln!(
            "[sim_account_state_compare] route_sig={:032x} source={} route_labels={} programs={} failed_program={} pubkey={} account_source={} is_writable={} is_signer={} cache_status={} rpc_status={} cache_owner={} rpc_owner={} cache_data_len={} rpc_data_len={} cache_lamports={} rpc_lamports={} cache_data_hash={} rpc_data_hash={} match={} reason={} owner_match={} data_len_match={} lamports_match={} data_hash_match={} classification={} action={} cache_slot={} rpc_context_slot={} slot_delta={}",
            route_sig,
            ix_source,
            route_labels,
            route_programs,
            failed_program,
            meta.pubkey,
            meta.source.as_str(),
            meta.is_writable,
            meta.is_signer,
            cache_state.status,
            rpc_state.status,
            cache_state.owner,
            rpc_state.owner,
            cache_state.data_len,
            rpc_state.data_len,
            cache_state.lamports,
            rpc_state.lamports,
            cache_state.data_hash,
            rpc_state.data_hash,
            outcome.matches_rpc,
            outcome.reason,
            outcome.owner_match,
            outcome.data_len_match,
            outcome.lamports_match,
            outcome.data_hash_match,
            classification,
            action,
            cache_slot,
            rpc_context_slot,
            slot_delta
        );
        if !outcome.matches_rpc {
            eprintln!(
                "[sim_account_state_mismatch] route_sig={:032x} source={} failed_program={} pubkey={} account_source={} is_writable={} is_signer={} cache_owner={} rpc_owner={} cache_data_len={} rpc_data_len={} cache_lamports={} rpc_lamports={} cache_data_hash={} rpc_data_hash={} reason={} owner_match={} data_len_match={} lamports_match={} data_hash_match={} classification={} action={} cache_slot={} rpc_context_slot={} slot_delta={} stale_state_hint={}",
                route_sig,
                ix_source,
                failed_program,
                meta.pubkey,
                meta.source.as_str(),
                meta.is_writable,
                meta.is_signer,
                cache_state.owner,
                rpc_state.owner,
                cache_state.data_len,
                rpc_state.data_len,
                cache_state.lamports,
                rpc_state.lamports,
                cache_state.data_hash,
                rpc_state.data_hash,
                outcome.reason,
                outcome.owner_match,
                outcome.data_len_match,
                outcome.lamports_match,
                outcome.data_hash_match,
                classification,
                action,
                cache_slot,
                rpc_context_slot,
                slot_delta,
                meta.is_writable && outcome.reason == "data_hash_mismatch"
            );
        }

        if classification != "normal" || !outcome.matches_rpc {
            write_problem_sim_account(
                sim,
                meta,
                route_sig,
                route_labels,
                route_programs,
                failed_program,
                &cache_state,
                &rpc_state,
                &outcome,
                classification,
                action,
            );
        }
    }

    eprintln!(
        "[sim_account_state_compare_summary] route_sig={:032x} source={} route_labels={} programs={} failed_program={} cache_slot={} rpc_context_slot={} slot_delta={} accounts={} mismatches={}",
        route_sig,
        ix_source,
        route_labels,
        route_programs,
        failed_program,
        cache_slot,
        rpc_context_slot,
        slot_delta,
        metas.len(),
        mismatches
    );
}

#[allow(clippy::too_many_arguments)]
fn dump_failed_program_context_window(
    failed_program: &Pubkey,
    tx: &VersionedTransaction,
    alts: &[AddressLookupTableAccount],
    metas: &[TxAccountMeta],
    cache: &AccountCache,
    synthetic_readonly_system_accounts: &HashSet<Pubkey>,
    created_by_setup: &HashSet<Pubkey>,
    route_sig: u128,
    route_labels: &str,
    route_programs: &str,
    lite_err: &str,
    ix_source: &str,
) {
    let resolved_keys = resolve_tx_account_keys(tx, alts);
    let meta_by_key: HashMap<Pubkey, TxAccountMeta> =
        metas.iter().map(|meta| (meta.pubkey, *meta)).collect();

    match &tx.message {
        VersionedMessage::Legacy(msg) => {
            for (ix_index, ix) in msg.instructions.iter().enumerate() {
                dump_failed_program_context_window_for_instruction(
                    failed_program,
                    ix_index,
                    ix.program_id_index,
                    &ix.accounts,
                    &resolved_keys,
                    &meta_by_key,
                    cache,
                    synthetic_readonly_system_accounts,
                    created_by_setup,
                    route_sig,
                    route_labels,
                    route_programs,
                    lite_err,
                    ix_source,
                );
            }
        }
        VersionedMessage::V0(v0) => {
            for (ix_index, ix) in v0.instructions.iter().enumerate() {
                dump_failed_program_context_window_for_instruction(
                    failed_program,
                    ix_index,
                    ix.program_id_index,
                    &ix.accounts,
                    &resolved_keys,
                    &meta_by_key,
                    cache,
                    synthetic_readonly_system_accounts,
                    created_by_setup,
                    route_sig,
                    route_labels,
                    route_programs,
                    lite_err,
                    ix_source,
                );
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn dump_failed_program_context_window_for_instruction(
    failed_program: &Pubkey,
    ix_index: usize,
    program_id_index: u8,
    account_indexes: &[u8],
    resolved_keys: &[Pubkey],
    meta_by_key: &HashMap<Pubkey, TxAccountMeta>,
    cache: &AccountCache,
    synthetic_readonly_system_accounts: &HashSet<Pubkey>,
    created_by_setup: &HashSet<Pubkey>,
    route_sig: u128,
    route_labels: &str,
    route_programs: &str,
    lite_err: &str,
    ix_source: &str,
) {
    let Some(program_id) = resolved_keys.get(program_id_index as usize) else {
        return;
    };
    if *program_id != JUPITER_PROGRAM_ID {
        return;
    }

    if let Some(failed_program_account_index) = account_indexes
        .iter()
        .position(|raw_idx| resolved_keys.get(*raw_idx as usize) == Some(failed_program))
    {
        let start = failed_program_account_index.saturating_sub(20);
        let end = (failed_program_account_index + 21).min(account_indexes.len());
        for account_index in start..end {
            let raw_idx = account_indexes[account_index];
            let Some(pubkey) = resolved_keys.get(raw_idx as usize) else {
                continue;
            };
            let meta = meta_by_key.get(pubkey).copied().unwrap_or(TxAccountMeta {
                pubkey: *pubkey,
                is_signer: false,
                is_writable: false,
                source: TxAccountSource::Static,
                tx_key_index: Some(raw_idx as usize),
                alt_table: None,
                alt_index: None,
            });
            let cached = cache.get(pubkey);
            let synthetic = synthetic_readonly_system_accounts.contains(pubkey);
            let owner = cached
                .as_ref()
                .map(|acct| acct.owner().to_string())
                .unwrap_or_else(|| {
                    if synthetic {
                        system_program::id().to_string()
                    } else {
                        "?".to_string()
                    }
                });
            let data_len = cached
                .as_ref()
                .map(|acct| acct.data().len().to_string())
                .unwrap_or_else(|| {
                    if synthetic {
                        "0".to_string()
                    } else {
                        "?".to_string()
                    }
                });
            let flag = context_window_flag(
                cached.is_some(),
                synthetic,
                created_by_setup.contains(pubkey),
                &owner,
                meta.is_writable,
                is_expected_readonly_pda_authority_meta(&meta),
                is_known_sysvar(pubkey) || is_builtin_program(pubkey) || is_registered_program_id(pubkey),
            );
            eprintln!(
                "[failed_program_context_window] route_sig={:032x} source={} route_labels={} programs={} failed_program={} lite_err={} jupiter_ix_index={} failed_program_index={} account_index={} tx_key_index={} pubkey={} owner={} data_len={} is_writable={} is_signer={} account_source={} from_cache={} synthetic={} created_by_setup={} flag={}",
                route_sig,
                ix_source,
                route_labels,
                route_programs,
                failed_program,
                lite_err,
                ix_index,
                failed_program_account_index,
                account_index,
                raw_idx,
                pubkey,
                owner,
                data_len,
                meta.is_writable,
                meta.is_signer,
                meta.source.as_str(),
                cached.is_some(),
                synthetic,
                created_by_setup.contains(pubkey),
                flag
            );
        }
    }
}

struct AccountDebugState {
    status: String,
    owner: String,
    data_len: String,
    lamports: String,
    data_hash: String,
}

struct AccountCompareOutcome {
    matches_rpc: bool,
    reason: &'static str,
    owner_match: bool,
    data_len_match: bool,
    lamports_match: bool,
    data_hash_match: bool,
}

impl AccountCompareOutcome {
    fn new(
        matches_rpc: bool,
        reason: &'static str,
        owner_match: bool,
        data_len_match: bool,
        lamports_match: bool,
        data_hash_match: bool,
    ) -> Self {
        Self {
            matches_rpc,
            reason,
            owner_match,
            data_len_match,
            lamports_match,
            data_hash_match,
        }
    }
}

impl AccountDebugState {
    fn marker(status: impl Into<String>) -> Self {
        Self {
            status: status.into(),
            owner: "?".to_string(),
            data_len: "?".to_string(),
            lamports: "?".to_string(),
            data_hash: "?".to_string(),
        }
    }
}

fn account_debug_state(
    account: Option<&Account>,
    synthetic: bool,
    loaded_program: bool,
    created_by_setup: bool,
    skipped_runtime: bool,
) -> AccountDebugState {
    if let Some(account) = account {
        return AccountDebugState {
            status: "found".to_string(),
            owner: account.owner().to_string(),
            data_len: account.data().len().to_string(),
            lamports: account.lamports().to_string(),
            data_hash: hash_bytes_hex(account.data()),
        };
    }
    if synthetic {
        return AccountDebugState {
            status: "synthetic_system".to_string(),
            owner: system_program::id().to_string(),
            data_len: "0".to_string(),
            lamports: "0".to_string(),
            data_hash: hash_bytes_hex(&[]),
        };
    }
    if loaded_program {
        return AccountDebugState::marker("loaded_program");
    }
    if created_by_setup {
        return AccountDebugState::marker("created_by_setup");
    }
    if skipped_runtime {
        return AccountDebugState::marker("skipped_runtime");
    }
    AccountDebugState::marker("missing")
}

fn account_compare_result(
    cache_state: &AccountDebugState,
    rpc_state: &AccountDebugState,
    cache_has_real_account: bool,
) -> AccountCompareOutcome {
    let owner_match = cache_state.owner == rpc_state.owner;
    let data_len_match = cache_state.data_len == rpc_state.data_len;
    let lamports_match = cache_state.lamports == rpc_state.lamports;
    let data_hash_match = cache_state.data_hash == rpc_state.data_hash;

    if !cache_has_real_account {
        return match cache_state.status.as_str() {
            "synthetic_system" => {
                if rpc_state.status == "found"
                    && rpc_state.owner == system_program::id().to_string()
                    && rpc_state.data_len == "0"
                {
                    AccountCompareOutcome::new(
                        true,
                        "synthetic_matches_rpc_system_empty",
                        owner_match,
                        data_len_match,
                        lamports_match,
                        data_hash_match,
                    )
                } else {
                    AccountCompareOutcome::new(
                        false,
                        "synthetic_or_missing_without_matching_rpc",
                        owner_match,
                        data_len_match,
                        lamports_match,
                        data_hash_match,
                    )
                }
            }
            "loaded_program" | "created_by_setup" | "skipped_runtime" => {
                AccountCompareOutcome::new(
                    true,
                    "not_compared_runtime_account",
                    owner_match,
                    data_len_match,
                    lamports_match,
                    data_hash_match,
                )
            }
            _ if rpc_state.status == "found" => AccountCompareOutcome::new(
                false,
                "cache_missing_rpc_found",
                owner_match,
                data_len_match,
                lamports_match,
                data_hash_match,
            ),
            _ => AccountCompareOutcome::new(
                true,
                "both_missing_or_rpc_unavailable",
                owner_match,
                data_len_match,
                lamports_match,
                data_hash_match,
            ),
        };
    }

    if rpc_state.status != "found" {
        return AccountCompareOutcome::new(
            false,
            "cache_found_rpc_not_found_or_error",
            owner_match,
            data_len_match,
            lamports_match,
            data_hash_match,
        );
    }
    if !owner_match {
        return AccountCompareOutcome::new(
            false,
            "owner_mismatch",
            owner_match,
            data_len_match,
            lamports_match,
            data_hash_match,
        );
    }
    if !data_len_match {
        return AccountCompareOutcome::new(
            false,
            "data_len_mismatch",
            owner_match,
            data_len_match,
            lamports_match,
            data_hash_match,
        );
    }
    if !lamports_match {
        return AccountCompareOutcome::new(
            false,
            "lamports_mismatch",
            owner_match,
            data_len_match,
            lamports_match,
            data_hash_match,
        );
    }
    if !data_hash_match {
        return AccountCompareOutcome::new(
            false,
            "data_hash_mismatch",
            owner_match,
            data_len_match,
            lamports_match,
            data_hash_match,
        );
    }
    AccountCompareOutcome::new(
        true,
        "same",
        owner_match,
        data_len_match,
        lamports_match,
        data_hash_match,
    )
}

fn slot_delta(left: u64, right: u64) -> u64 {
    if left == 0 || right == 0 {
        0
    } else if left >= right {
        left - right
    } else {
        right - left
    }
}

fn context_window_flag(
    from_cache: bool,
    synthetic: bool,
    created_by_setup: bool,
    owner: &str,
    is_writable: bool,
    expected_readonly_pda_authority: bool,
    runtime_or_program_account: bool,
) -> &'static str {
    if expected_readonly_pda_authority && synthetic && !is_writable {
        "expected_readonly_pda_authority"
    } else if runtime_or_program_account && !from_cache && !synthetic {
        "program_or_sysvar"
    } else if !from_cache && !synthetic && !created_by_setup {
        "owner_unknown"
    } else if synthetic && is_writable {
        "bad_writable_synthetic"
    } else if is_writable && owner == system_program::id().to_string() {
        "writable_system_owner"
    } else if synthetic {
        "synthetic"
    } else {
        "ok"
    }
}

fn hash_bytes_hex(data: &[u8]) -> String {
    let mut hasher = DefaultHasher::new();
    data.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

fn hash_pubkeys_hex(pubkeys: &[Pubkey]) -> String {
    let mut hasher = DefaultHasher::new();
    pubkeys.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

fn dump_jupiter_route_instruction_accounts_for_program(
    target_program: &Pubkey,
    label: &str,
    ix_index: usize,
    program_id_index: u8,
    account_indexes: &[u8],
    resolved_keys: &[Pubkey],
    meta_by_key: &HashMap<Pubkey, TxAccountMeta>,
    cache: &AccountCache,
    synthetic_readonly_system_accounts: &HashSet<Pubkey>,
    created_by_setup: &HashSet<Pubkey>,
) {
    let Some(program_id) = resolved_keys.get(program_id_index as usize) else {
        return;
    };
    if *program_id != JUPITER_PROGRAM_ID {
        return;
    }

    let mentions_target = account_indexes.iter().any(|raw_idx| {
        resolved_keys
            .get(*raw_idx as usize)
            .is_some_and(|pk| pk == target_program)
    });
    if !mentions_target {
        return;
    }

    let mut owner_counts: HashMap<String, usize> = HashMap::new();
    for (account_index, raw_idx) in account_indexes.iter().enumerate() {
        let Some(pubkey) = resolved_keys.get(*raw_idx as usize) else {
            continue;
        };
        let meta = meta_by_key.get(pubkey).copied().unwrap_or(TxAccountMeta {
            pubkey: *pubkey,
            is_signer: false,
            is_writable: false,
            source: TxAccountSource::Static,
            tx_key_index: Some(*raw_idx as usize),
            alt_table: None,
            alt_index: None,
        });
        let cached = cache.get(pubkey);
        let (owner, executable, data_len, from_cache) = match cached.as_ref() {
            Some(acct) => (
                acct.owner().to_string(),
                acct.executable().to_string(),
                acct.data().len().to_string(),
                "true",
            ),
            None => ("?".to_string(), "?".to_string(), "?".to_string(), "false"),
        };
        *owner_counts.entry(owner.clone()).or_insert(0) += 1;
        eprintln!(
            "[sim_route_accounts_for_program] target_program={} label={} jupiter_ix_index={} account_index={} pubkey={} is_signer={} is_writable={} source={} owner={} executable={} data_len={} from_cache={} synthetic={} created_by_setup={}",
            target_program,
            label,
            ix_index,
            account_index,
            pubkey,
            meta.is_signer,
            meta.is_writable,
            meta.source.as_str(),
            owner,
            executable,
            data_len,
            from_cache,
            synthetic_readonly_system_accounts.contains(pubkey),
            created_by_setup.contains(pubkey)
        );
    }

    let mut owner_counts = owner_counts.into_iter().collect::<Vec<_>>();
    owner_counts.sort_by(|a, b| a.0.cmp(&b.0));
    for (owner, count) in owner_counts {
        eprintln!(
            "[sim_alphaq_owner_summary] target_program={} label={} jupiter_ix_index={} owner={} count={}",
            target_program,
            label,
            ix_index,
            owner,
            count
        );
    }
}

fn compare_revert_with_rpc(
    cache: &AccountCache,
    tx: &VersionedTransaction,
    compare_reason: &str,
    program: &Pubkey,
    lite_err: &str,
    metrics: &Metrics,
    route_sig: u128,
    route_labels: &str,
    route_programs: &str,
    ix_source: &str,
) {
    let attempt = RPC_SIM_COMPARE_COUNT.fetch_add(1, Ordering::Relaxed);
    if attempt >= MAX_RPC_SIM_COMPARE_PER_PROCESS {
        return;
    }

    match cache.simulate_transaction_for_compare(tx) {
        Ok((rpc_err, logs)) => {
            if rpc_err == "None" {
                metrics.sim_rpc_compare_ok.fetch_add(1, Ordering::Relaxed);
            } else {
                metrics
                    .sim_rpc_compare_same_fail
                    .fetch_add(1, Ordering::Relaxed);
            }
            let rpc_logs = logs
                .iter()
                .take(30)
                .map(|line| line.replace('\n', "\\n"))
                .collect::<Vec<_>>()
                .join(" | ");
            if compare_reason == "alphaq_invalid_account_owner" {
                eprintln!(
                    "[sim_compare_alphaq] route_sig={:032x} source={} route_labels={} programs={} lite_err={} rpc_err={} same_tx=true rpc_logs={}",
                    route_sig,
                    ix_source,
                    route_labels,
                    route_programs,
                    lite_err,
                    rpc_err,
                    rpc_logs
                );
            }
            eprintln!(
                "[sim_compare_failed_program] route_sig={:032x} source={} route_labels={} programs={} failed_program={} compare_reason={} lite_err={} rpc_err={} same_tx=true",
                route_sig,
                ix_source,
                route_labels,
                route_programs,
                program,
                compare_reason,
                lite_err,
                rpc_err
            );
            eprintln!(
                "[sim_compare] same_tx=true route_sig={:032x} source={} compare_reason={} route_labels={} programs={} failed_program={} lite_err={} rpc_err={} rpc_logs={}",
                route_sig,
                ix_source,
                compare_reason,
                route_labels,
                route_programs,
                program,
                lite_err,
                rpc_err,
                rpc_logs
            );
        }
        Err(e) => {
            metrics
                .sim_rpc_compare_error
                .fetch_add(1, Ordering::Relaxed);
            eprintln!(
                "[sim_compare_failed_program] route_sig={:032x} source={} route_labels={} programs={} failed_program={} compare_reason={} lite_err={} rpc_error={} same_tx=true",
                route_sig,
                ix_source,
                route_labels,
                route_programs,
                program,
                compare_reason,
                lite_err,
                e
            );
            eprintln!(
                "[sim_compare] same_tx=true route_sig={:032x} source={} compare_reason={} route_labels={} programs={} failed_program={} lite_err={} rpc_error={}",
                route_sig,
                ix_source,
                compare_reason,
                route_labels,
                route_programs,
                program,
                lite_err,
                e
            );
        }
    }
}

fn account_kind(acct: &Account) -> &'static str {
    if acct.executable() {
        "program"
    } else if owner_is(acct, &system_program::id()) {
        "system_account"
    } else if owner_is(
        acct,
        &Pubkey::from_str_const("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"),
    ) {
        "spl_token_account"
    } else if owner_is(
        acct,
        &Pubkey::from_str_const("TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"),
    ) {
        "token_2022_account"
    } else {
        "unknown"
    }
}

fn owner_is(acct: &Account, owner: &Pubkey) -> bool {
    *acct.owner() == LsAddr::from(owner.to_bytes())
}

/// Pool of independent Simulator instances for concurrent simulation.
/// Each worker holds its own LiteSVM instance (mutex-protected) so multiple
/// profitable opportunities can be simulated in parallel without contention.
pub struct SimulatorPool {
    sims: Vec<Arc<Simulator>>,
    next: AtomicUsize,
}

impl SimulatorPool {
    pub fn new(
        workers: usize,
        so_dir: &str,
        wsol_ata: Pubkey,
        payer_pubkey: Pubkey,
        fail_closed: bool,
        allow_hot_path_rpc_fetch: bool,
        manual_accounts_root: PathBuf,
        current_slot: Arc<AtomicU64>,
        current_unix_timestamp: Arc<AtomicI64>,
        missing_handle: Option<Arc<crate::auto_missing_accounts::AutoMissingAccountsHandle>>,
    ) -> Result<Self> {
        let workers = workers.max(1);
        let mut sims = Vec::with_capacity(workers);
        for i in 0..workers {
            let sim = Simulator::new(
                so_dir,
                wsol_ata,
                payer_pubkey,
                fail_closed,
                allow_hot_path_rpc_fetch,
                manual_accounts_root.clone(),
                current_slot.clone(),
                current_unix_timestamp.clone(),
                missing_handle.clone(),
            )
            .with_context(|| format!("failed to build sim worker #{i}"))?;
            sims.push(Arc::new(sim));
            info!(worker = i, "sim worker initialised");
        }
        info!(workers, "SimulatorPool ready (LiteSVM 0.11, mainnet features)");
        Ok(Self {
            sims,
            next: AtomicUsize::new(0),
        })
    }

    #[inline]
    pub fn acquire(&self) -> Arc<Simulator> {
        let idx = self.next.fetch_add(1, Ordering::Relaxed) % self.sims.len();
        self.sims[idx].clone()
    }
}

pub fn resolve_alts(
    alt_addresses: &[String],
    alt_cache: &crate::alt_cache::AltCache,
    rpc: &solana_client::rpc_client::RpcClient,
) -> Result<Vec<AddressLookupTableAccount>> {
    let mut out = Vec::with_capacity(alt_addresses.len());
    for s in alt_addresses {
        let pk = match Pubkey::try_from(s.as_str()) {
            Ok(pk) => pk,
            Err(e) => {
                eprintln!(
                    "[resolve_alts_error] status=bad_pubkey alt={} error={:?} action=fail_closed",
                    s, e
                );
                return Err(anyhow!("bad ALT pubkey {s}: {e:?}"));
            }
        };

        match alt_cache.get_or_fetch(&pk, rpc) {
            Ok(alt) => out.push(alt),
            Err(e) => {
                eprintln!(
                    "[resolve_alts_error] status=fetch_failed alt={} error={} action=fail_closed",
                    pk, e
                );
                return Err(e).with_context(|| format!("ALT fetch for sim failed: {pk}"));
            }
        }
    }
    Ok(out)
}

pub fn tx_account_keys_for_mix_gate(
    tx: &VersionedTransaction,
    alts: &[AddressLookupTableAccount],
) -> Vec<Pubkey> {
    let created_by_setup = collect_accounts_created_by_setup(tx, alts);
    let mut keys = resolve_tx_account_keys(tx, alts);
    keys.retain(|pk| !created_by_setup.contains(pk));
    keys.sort_unstable();
    keys.dedup();
    keys
}

pub fn registered_programs_mentioned_in_tx(
    tx: &VersionedTransaction,
    alts: &[AddressLookupTableAccount],
) -> Vec<String> {
    let registry = crate::program_registry::PROGRAMS
        .iter()
        .filter_map(|(program_id, _)| Pubkey::try_from(*program_id).ok())
        .collect::<HashSet<_>>();
    let mut programs = resolve_tx_account_keys(tx, alts)
        .into_iter()
        .filter(|key| registry.contains(key))
        .map(|key| key.to_string())
        .collect::<Vec<_>>();
    programs.sort();
    programs.dedup();
    programs
}