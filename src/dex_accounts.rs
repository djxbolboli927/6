//! Pool account registry.
//!
//! Reads `<dex_dir>/<DEX_NAME>/<pool>.toml` files at startup and returns:
//!   - `all_accounts` — every static pubkey to pre-fetch via RPC so the sim
//!     cache is fully populated before the first trade
//!   - `subscribe_accounts` — vault accounts that change on every swap and
//!     must be subscribed individually on Yellowstone for live updates
//!     (the Yellowstone owner-filter already covers accounts owned by DEX
//!     programs; vaults are owned by SPL Token and need a direct subscription)
//!
//! Directory layout:
//!   dex_dir/
//!     Goonfi_V2/
//!       hype_usdc.toml
//!       wsol_usdc.toml
//!     SomeOtherDex/
//!       ...
//!
//! Files starting with `_` (e.g. `_template.toml`) are skipped.

use serde::Deserialize;
use solana_sdk::pubkey::Pubkey;
use std::path::Path;
use tracing::{info, warn};

#[derive(Deserialize)]
struct PoolFile {
    name: Option<String>,
    market: Option<String>,
    vault_a: Option<String>,
    vault_b: Option<String>,
    mint_a: Option<String>,
    mint_b: Option<String>,
    #[serde(default)]
    extra: Vec<String>,
}

pub struct DexPools {
    /// Every static account to pre-fetch via RPC at startup.
    pub all_accounts: Vec<Pubkey>,
    /// Volatile accounts (vaults, oracle states, global vaults, etc.) that
    /// change on every swap and must be subscribed to Yellowstone individually.
    pub subscribe_accounts: Vec<Pubkey>,
    /// Account groups fetched during startup. For mix.json each group should
    /// correspond to one pool so warm-up can obey a pools/sec rate limit.
    pub prefetch_groups: Vec<Vec<Pubkey>>,
    /// Address Lookup Table pubkeys that must be loaded into AltCache so v0
    /// transactions referencing them can be resolved by the simulator.
    pub alt_accounts: Vec<Pubkey>,
}

/// Load all pool files from `dex_dir/<DEX>/<pool>.toml`.
/// Returns an empty `DexPools` if the directory does not exist.
pub fn load(dex_dir: &str) -> DexPools {
    let dex_path = Path::new(dex_dir);
    if !dex_path.exists() {
        return DexPools {
            all_accounts: vec![],
            subscribe_accounts: vec![],
            prefetch_groups: vec![],
            alt_accounts: vec![],
        };
    }

    let mix_path = if dex_path.is_file()
        && dex_path.file_name().and_then(|n| n.to_str()) == Some("mix.json")
    {
        Some(dex_path.to_path_buf())
    } else {
        let candidate = dex_path.join("mix.json");
        candidate.exists().then_some(candidate)
    };
    if let Some(path) = mix_path {
        return load_mix_json(&path);
    }

    let mut all: Vec<Pubkey> = Vec::new();
    let mut subs: Vec<Pubkey> = Vec::new();
    let mut pool_count = 0usize;

    let dex_entries = match std::fs::read_dir(dex_path) {
        Ok(e) => e,
        Err(e) => {
            warn!(dex_dir, error = %e, "cannot read dex_dir");
            return DexPools {
                all_accounts: all,
                subscribe_accounts: subs,
                prefetch_groups: vec![],
                alt_accounts: vec![],
            };
        }
    };

    let mut prefetch_groups: Vec<Vec<Pubkey>> = Vec::new();
    for dex_entry in dex_entries.flatten() {
        if !dex_entry.file_type().map_or(false, |t| t.is_dir()) {
            continue;
        }
        let dex_name = dex_entry.file_name();
        let pool_entries = match std::fs::read_dir(dex_entry.path()) {
            Ok(e) => e,
            Err(_) => continue,
        };

        for pool_entry in pool_entries.flatten() {
            let path = pool_entry.path();
            // Skip template files and non-TOML files
            let fname = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if fname.starts_with('_') || path.extension().and_then(|e| e.to_str()) != Some("toml") {
                continue;
            }

            let content = match std::fs::read_to_string(&path) {
                Ok(s) => s,
                Err(e) => {
                    warn!(path = %path.display(), error = %e, "cannot read pool file");
                    continue;
                }
            };

            let pool: PoolFile = match toml::from_str(&content) {
                Ok(p) => p,
                Err(e) => {
                    warn!(path = %path.display(), error = %e, "invalid pool TOML");
                    continue;
                }
            };

            let pool_name = pool.name.as_deref().unwrap_or(fname);
            let mut parsed = 0usize;
            let mut group: Vec<Pubkey> = Vec::new();

            let mut push = |s: Option<&str>, is_vault: bool| {
                let s = match s { Some(s) => s, None => return };
                match Pubkey::try_from(s) {
                    Ok(pk) => {
                        all.push(pk);
                        group.push(pk);
                        if is_vault { subs.push(pk); }
                        parsed += 1;
                    }
                    Err(_) => warn!(
                        dex = ?dex_name, pool = pool_name, addr = s,
                        "invalid pubkey in pool file — skipped"
                    ),
                }
            };

            push(pool.market.as_deref(), false);
            push(pool.vault_a.as_deref(), true);   // vault → live subscription
            push(pool.vault_b.as_deref(), true);   // vault → live subscription
            push(pool.mint_a.as_deref(), false);
            push(pool.mint_b.as_deref(), false);
            for addr in &pool.extra {
                push(Some(addr.as_str()), false);
            }

            info!(
                dex = ?dex_name, pool = pool_name,
                accounts = parsed,
                "pool loaded"
            );
            group.sort_unstable();
            group.dedup();
            if !group.is_empty() {
                prefetch_groups.push(group);
            }
            pool_count += 1;
        }
    }

    // Deduplicate — mints and global protocol accounts appear across pools
    all.sort_unstable();
    all.dedup();
    subs.sort_unstable();
    subs.dedup();

    info!(
        pools = pool_count,
        prefetch = all.len(),
        live_subs = subs.len(),
        groups = prefetch_groups.len(),
        "dex pool registry loaded"
    );

    DexPools {
        all_accounts: all,
        subscribe_accounts: subs,
        prefetch_groups,
        alt_accounts: vec![],
    }
}

fn load_mix_json(path: &Path) -> DexPools {
    let content = match std::fs::read_to_string(path) {
        Ok(content) => content,
        Err(e) => {
            warn!(path = %path.display(), error = %e, "cannot read mix.json");
            return DexPools {
                all_accounts: vec![],
                subscribe_accounts: vec![],
                prefetch_groups: vec![],
                alt_accounts: vec![],
            };
        }
    };

    let json: serde_json::Value = match serde_json::from_str(&content) {
        Ok(json) => json,
        Err(e) => {
            warn!(path = %path.display(), error = %e, "invalid mix.json");
            return DexPools {
                all_accounts: vec![],
                subscribe_accounts: vec![],
                prefetch_groups: vec![],
                alt_accounts: vec![],
            };
        }
    };

    let mut all = Vec::new();
    collect_pubkeys_from_json(&json, &mut all);
    all.sort_unstable();
    all.dedup();
    let mut groups = collect_prefetch_groups_from_mix(&json);
    if groups.is_empty() {
        groups = all.chunks(100).map(|chunk| chunk.to_vec()).collect();
    }

    info!(
        mix = %path.display(),
        accounts_total = all.len(),
        groups = groups.len(),
        "sim account mix loaded"
    );

    DexPools {
        all_accounts: all.clone(),
        subscribe_accounts: all,
        prefetch_groups: groups,
        alt_accounts: vec![],
    }
}

fn collect_pubkeys_from_json(value: &serde_json::Value, out: &mut Vec<Pubkey>) {
    match value {
        serde_json::Value::String(s) => {
            if let Ok(pk) = Pubkey::try_from(s.as_str()) {
                out.push(pk);
            }
        }
        serde_json::Value::Array(items) => {
            for item in items {
                collect_pubkeys_from_json(item, out);
            }
        }
        serde_json::Value::Object(map) => {
            for value in map.values() {
                collect_pubkeys_from_json(value, out);
            }
        }
        _ => {}
    }
}

fn collect_prefetch_groups_from_mix(value: &serde_json::Value) -> Vec<Vec<Pubkey>> {
    let root = value
        .get("pools")
        .or_else(|| value.get("Pools"))
        .unwrap_or(value);
    collect_pool_like_groups(root)
}

/// Load all pool accounts from `pools_by_dex/*.json` files.
///
/// Each JSON file is named after the DEX program ID and contains an array of
/// pool entries with a rich `params` object.  Every pool is fully described
/// in its JSON file — there are no external config files, no mix.json, no TOML
/// templates.  All accounts are fetched from RPC at startup and kept fresh via
/// Yellowstone gRPC.
///
/// ## Account classification
///
/// | Category     | Action              | Examples                                  |
/// |-------------|---------------------|-------------------------------------------|
/// | Vault        | subscribe + prefetch| tokenAccountA, tokenAccountB, globalVault |
/// | Volatile state | subscribe + prefetch | oracle, observationState, tickmap, state |
/// | Mint / static | prefetch only      | tokenmentA, authority, poolMint           |
/// | ALT          | alt_accounts list   | addressLookupTableAddress                 |
/// | Executable / sysvar | skip         | tokenProgram, sysvarInstructions          |
///
/// Pool state owned by the DEX program is covered by the Yellowstone owner
/// filter and does not need an individual subscription.
pub fn load_pools_by_dex_dir(pools_dir: &str) -> DexPools {
    // Params that need live Yellowstone subscriptions because they hold
    // volatile on-chain state (token balances, oracle/observation slots, etc.)
    // that changes on every swap.
    const SUBSCRIBE_FIELDS: &[&str] = &[
        // Always volatile — token vault balances
        "tokenAccountA",
        "tokenAccountB",
        // Raydium CPMM oracle
        "observationState",
        // Meteora DLMM vaults
        "vaultToken",
        "vaultLp",
        "protocolTokenFee",
        // Manifest global vaults
        "global",
        "globalVault",
        // Invariant AMM state
        "state",
        "tickmap",
        // Whirlpool oracle
        "oracle",
        // BiSoN / Bonk inner pool state
        "market",
        "poolA",
        "poolB",
        // DEXY metadata state
        "metadataState",
    ];

    // Params that are static or mint accounts — fetch from RPC at startup but
    // do not need individual Yellowstone subscriptions.
    const STATIC_ONLY_FIELDS: &[&str] = &[
        "tokenmentA",
        "tokenmentB",
        "poolMint",
        "vaultLpMint",
        "authority",
        "vaultAuthority",
        "programAuthority",
    ];

    // Params that are Address Lookup Tables — loaded separately into AltCache.
    const ALT_FIELDS: &[&str] = &["addressLookupTableAddress"];

    // Params that are executable programs or sysvars — LiteSVM provides them
    // natively; no need to prefetch.
    const SKIP_FIELDS: &[&str] = &[
        "tokenProgramA",
        "tokenProgramB",
        "tokenProgram",
        "vaultProgram",
        "sysvarInstructions",
    ];

    let dir_path = std::path::Path::new(pools_dir);
    if !dir_path.exists() {
        return DexPools {
            all_accounts: vec![],
            subscribe_accounts: vec![],
            prefetch_groups: vec![],
            alt_accounts: vec![],
        };
    }

    let entries = match std::fs::read_dir(dir_path) {
        Ok(e) => e,
        Err(e) => {
            warn!(pools_dir, error = %e, "cannot read pools_by_dex dir");
            return DexPools {
                all_accounts: vec![],
                subscribe_accounts: vec![],
                prefetch_groups: vec![],
                alt_accounts: vec![],
            };
        }
    };

    let mut all: Vec<Pubkey> = Vec::new();
    let mut subs: Vec<Pubkey> = Vec::new();
    let mut alts: Vec<Pubkey> = Vec::new();
    let mut prefetch_groups: Vec<Vec<Pubkey>> = Vec::new();
    let mut total_pools = 0usize;

    for dir_entry in entries.flatten() {
        let path = dir_entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("json") {
            continue;
        }
        let fname = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("?")
            .to_string();

        let content = match std::fs::read_to_string(&path) {
            Ok(s) => s,
            Err(e) => {
                warn!(file = %fname, error = %e, "cannot read pools_by_dex json");
                continue;
            }
        };

        let pool_list: Vec<serde_json::Value> = match serde_json::from_str(&content) {
            Ok(v) => v,
            Err(e) => {
                warn!(file = %fname, error = %e, "invalid pools_by_dex json");
                continue;
            }
        };

        let mut file_pools = 0usize;

        for pool_json in &pool_list {
            let pool_pk_str = pool_json
                .get("pubkey")
                .and_then(|v| v.as_str())
                .unwrap_or("");

            let mut group: Vec<Pubkey> = Vec::new();

            // Pool state is DEX-program-owned → Yellowstone owner filter covers it.
            if let Ok(pk) = Pubkey::try_from(pool_pk_str) {
                all.push(pk);
                group.push(pk);
            }

            // Walk every key in params and classify it.
            if let Some(serde_json::Value::Object(params)) = pool_json.get("params") {
                for (field, value) in params {
                    let addr = match value.as_str() {
                        Some(s) if !s.is_empty() => s,
                        _ => continue, // integer, bool, nested object, or empty string
                    };

                    let field_str = field.as_str();

                    if SKIP_FIELDS.contains(&field_str) {
                        continue;
                    }

                    let Ok(pk) = Pubkey::try_from(addr) else {
                        continue; // not a valid base58 pubkey
                    };

                    if ALT_FIELDS.contains(&field_str) {
                        alts.push(pk);
                        continue;
                    }

                    // Everything else goes into all_accounts (prefetch from RPC).
                    all.push(pk);
                    group.push(pk);

                    // Subscribe if the field is known-volatile OR if it's not in
                    // the static-only list (unknown extra fields default to subscribe
                    // so we never miss a writable account).
                    if SUBSCRIBE_FIELDS.contains(&field_str)
                        || !STATIC_ONLY_FIELDS.contains(&field_str)
                    {
                        subs.push(pk);
                    }
                }
            }

            if !group.is_empty() {
                group.sort_unstable();
                group.dedup();
                prefetch_groups.push(group);
                file_pools += 1;
                total_pools += 1;
            }
        }

        eprintln!("[pools_by_dex] file={} pools={}", fname, file_pools);
    }

    all.sort_unstable();
    all.dedup();
    subs.sort_unstable();
    subs.dedup();
    alts.sort_unstable();
    alts.dedup();

    eprintln!(
        "[pools_by_dex] total_pools={} prefetch_accounts={} subscribe_accounts={} alt_accounts={}",
        total_pools,
        all.len(),
        subs.len(),
        alts.len(),
    );

    DexPools {
        all_accounts: all,
        subscribe_accounts: subs,
        prefetch_groups,
        alt_accounts: alts,
    }
}

fn collect_pool_like_groups(value: &serde_json::Value) -> Vec<Vec<Pubkey>> {
    let mut own = Vec::new();
    collect_pubkeys_from_json(value, &mut own);
    own.sort_unstable();
    own.dedup();

    let children: Vec<&serde_json::Value> = match value {
        serde_json::Value::Array(items) => items.iter().collect(),
        serde_json::Value::Object(map) => map.values().collect(),
        _ => vec![],
    };

    let mut nested = Vec::new();
    for child in children {
        if matches!(
            child,
            serde_json::Value::Array(_) | serde_json::Value::Object(_)
        ) {
            nested.extend(collect_pool_like_groups(child));
        }
    }

    if nested.len() > 4 {
        nested
    } else if !own.is_empty() {
        vec![own]
    } else {
        nested
    }
}
