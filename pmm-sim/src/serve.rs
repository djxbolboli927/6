/// pmm-sim serve mode: JSON IPC server for the bot's simulation stage.
///
/// Reads newline-delimited JSON requests from stdin, executes each swap in a
/// fresh LiteSVM snapshot (with all PMM programs pre-loaded at startup), and
/// writes the JSON result to stdout.
///
/// The bot (SDK 2.2) communicates with this process via stdio pipes.  Using a
/// subprocess allows pmm-sim (SDK 3.0 / litesvm 0.10) to coexist with the bot's
/// older SDK without any linker conflicts.
///
/// # Protocol
///
/// Request (one JSON line from bot → pmm-sim stdin):
/// ```json
/// {
///   "id": 42,
///   "fee_payer": "<base58 trading keypair pubkey>",
///   "token_mints": ["<base58 mint>", ...],
///   "src_mint": "<base58 WSOL>",
///   "src_amount": 1000000000,
///   "swap_instruction": {
///     "program_id": "<base58 router program>",
///     "accounts": [{"pubkey":"<b58>","is_signer":false,"is_writable":true}, ...],
///     "data": "<base64>"
///   },
///   "accounts": [
///     {"pubkey":"<b58>","lamports":123,"data":"<b64>","owner":"<b58>","executable":false,"rent_epoch":0},
///     ...
///   ]
/// }
/// ```
///
/// Response (one JSON line from pmm-sim stdout → bot):
/// ```json
/// {"id":42,"success":true,"amount_out":12345678,"compute_units":450123,"error":null}
/// ```
use std::{
    collections::HashMap,
    io::{BufRead, Write},
    str::FromStr,
};

use base64::{Engine as _, engine::general_purpose::STANDARD as B64};
use litesvm::LiteSVM;
use magnus_shared::{pmm_bisonfi, pmm_goonfi, pmm_humidifi, pmm_obric_v2, pmm_solfi_v2, pmm_tessera, pmm_zerofi};
use serde::{Deserialize, Serialize};
use solana_compute_budget::compute_budget::ComputeBudget;
use solana_sdk::{
    account::Account,
    instruction::{AccountMeta, Instruction},
    pubkey::Pubkey,
    signature::Keypair,
    signer::Signer,
    transaction::Transaction,
};
use spl_associated_token_account::get_associated_token_address;

use crate::Aggregator;
use crate::builder::ConstructSwap;
use crate::cfg::Cfg;
use crate::misc::Misc;
use crate::consts;

use magnus_router_client::types::SwapArgs;
use magnus_shared::{Dex, Route};

// ── IPC types ─────────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct ServeRequest {
    id: u64,
    fee_payer: String,
    token_mints: Vec<String>,
    src_mint: String,
    src_amount: u64,
    /// ALL instructions: compute_budget + setup + swap + cleanup.
    instructions: Vec<ServeIx>,
    #[serde(default)]
    lookup_tables: Vec<ServeLookupTable>,
    accounts: Vec<ServeAccount>,
    #[serde(default = "default_jito_tip")]
    jito_tip_lamports: u64,
    #[serde(default = "default_cu_limit")]
    cu_limit: u32,
    #[serde(default)]
    route_sig: String,
}

fn default_jito_tip() -> u64 { 1600 }
fn default_cu_limit() -> u32 { 1_400_000 }

#[derive(Deserialize)]
struct ServeLookupTable {
    #[allow(dead_code)]
    key: String,
    #[allow(dead_code)]
    addresses: Vec<String>,
}

#[derive(Deserialize)]
struct ServeIx {
    program_id: String,
    accounts: Vec<ServeAccountMeta>,
    data: String, // base64
}

#[derive(Deserialize)]
struct ServeAccountMeta {
    pubkey: String,
    is_signer: bool,
    is_writable: bool,
}

#[derive(Deserialize)]
struct ServeAccount {
    pubkey: String,
    lamports: u64,
    data: String, // base64
    owner: String,
    executable: bool,
    rent_epoch: u64,
}

#[derive(Serialize)]
struct ServeResponse {
    id: u64,
    success: bool,
    amount_out: Option<u64>,
    compute_units: Option<u64>,
    error: Option<String>,
}

// ── BisonFi build+quote op ──────────────────────────────────────────────────
//
// Distinct from the full-tx sim path: the bot asks us to BUILD a DFlow `swap2`
// (spoof-Magnus) instruction for a single BisonFi leg, simulate it, and return
// BOTH the predicted output AND the instruction (referencing the real wallet).

/// Lightweight probe to route a request line to the right handler.
#[derive(Deserialize)]
struct OpProbe {
    #[serde(default)]
    op: String,
}

#[derive(Deserialize)]
struct BuildBisonRequest {
    id: u64,
    fee_payer: String,
    market: String,
    src_mint: String,
    dst_mint: String,
    amount_in: u64,
    /// Chain slot the injected account state belongs to. The LiteSVM clock is
    /// warped to it so time/slot-sensitive PMMs (BisonFi) accept the snapshot.
    #[serde(default)]
    slot: u64,
    /// When true, call the BisonFi program DIRECTLY (no Magnus/DFlow spoof
    /// router) — a plain top-level swap. Used to read the price without the
    /// router CPI path that v3 pools reject.
    #[serde(default)]
    direct: bool,
    accounts: Vec<ServeAccount>,
}

#[derive(Serialize)]
struct OutAccountMeta {
    pubkey: String,
    is_signer: bool,
    is_writable: bool,
}

#[derive(Serialize)]
struct OutInstruction {
    program_id: String,
    accounts: Vec<OutAccountMeta>,
    data: String, // base64
}

#[derive(Serialize)]
struct BuildBisonResponse {
    id: u64,
    success: bool,
    amount_out: Option<u64>,
    compute_units: Option<u64>,
    instruction: Option<OutInstruction>,
    error: Option<String>,
}

// ── Router program IDs ────────────────────────────────────────────────────────

fn router_program_id() -> Pubkey {
    magnus_router_client::programs::ROUTER_ID
}

/// Derive a BisonFi market's REAL base/quote vault token accounts from the
/// market account data. Layout (observed, stable across BisonFi markets):
///   bytes[0..8]   = b"POOLSTAT" magic
///   bytes[120..152] = base vault token account
///   bytes[152..184] = quote vault token account
///   bytes[184..216] = base mint, bytes[216..248] = quote mint
fn bisonfi_vaults_from_market(data: &[u8]) -> Option<(Pubkey, Pubkey)> {
    if data.len() < 248 || &data[0..8] != b"POOLSTAT" {
        return None;
    }
    let base: [u8; 32] = data[120..152].try_into().ok()?;
    let quote: [u8; 32] = data[152..184].try_into().ok()?;
    Some((Pubkey::from(base), Pubkey::from(quote)))
}

// ── Serve entry point ─────────────────────────────────────────────────────────

/// Run the serve loop: initialise LiteSVM with all programs, then process
/// JSON requests from stdin until EOF.
pub fn run(cfg: &Cfg, programs_path: &str) -> eyre::Result<()> {
    let mut svm = build_base_svm(programs_path)?;

    // One persistent simulation wallet — we patch the fee_payer in every request
    // to use this keypair so we can sign transactions without the real keypair.
    let sim_wallet = Keypair::new();

    // Fund the simulation wallet with a large SOL balance once.
    // The SVM state persists between requests; the wallet lamports don't drain
    // significantly because we re-inject the account each request.
    fund_wallet(&mut svm, &sim_wallet.pubkey(), consts::AIRDROP_AMOUNT);

    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut stdout_lock = stdout.lock();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) if !l.trim().is_empty() => l,
            Ok(_) => continue,
            Err(e) => {
                eprintln!("[pmm-sim serve] stdin read error: {e}");
                break;
            }
        };

        // Route by `op`: "build_bison" → BisonFi build+quote; otherwise full-tx sim.
        let op = serde_json::from_str::<OpProbe>(&line).map(|p| p.op).unwrap_or_default();
        let out = if op == "build_bison" {
            let resp = match process_build_bison(&mut svm, cfg, &sim_wallet, &line) {
                Ok(resp) => resp,
                Err(e) => BuildBisonResponse {
                    id: extract_id(&line).unwrap_or(0),
                    success: false,
                    amount_out: None,
                    compute_units: None,
                    instruction: None,
                    error: Some(e.to_string()),
                },
            };
            serde_json::to_string(&resp)
        } else {
            let resp = match process_request(&mut svm, &sim_wallet, &line) {
                Ok(resp) => resp,
                Err(e) => ServeResponse {
                    id: extract_id(&line).unwrap_or(0),
                    success: false,
                    amount_out: None,
                    compute_units: None,
                    error: Some(e.to_string()),
                },
            };
            serde_json::to_string(&resp)
        };

        let mut out = out.unwrap_or_else(|_| r#"{"id":0,"success":false,"error":"serialization error"}"#.to_string());
        out.push('\n');
        let _ = stdout_lock.write_all(out.as_bytes());
        let _ = stdout_lock.flush();
    }

    Ok(())
}

// ── BisonFi build+quote handler ─────────────────────────────────────────────

fn process_build_bison(
    svm: &mut LiteSVM,
    cfg: &Cfg,
    sim_wallet: &Keypair,
    line: &str,
) -> eyre::Result<BuildBisonResponse> {
    let req: BuildBisonRequest = serde_json::from_str(line)?;

    let fee_payer = Pubkey::from_str(&req.fee_payer).map_err(|e| eyre::eyre!("fee_payer: {e}"))?;
    let market = Pubkey::from_str(&req.market).map_err(|e| eyre::eyre!("market: {e}"))?;
    let src_mint = Pubkey::from_str(&req.src_mint).map_err(|e| eyre::eyre!("src_mint: {e}"))?;
    let dst_mint = Pubkey::from_str(&req.dst_mint).map_err(|e| eyre::eyre!("dst_mint: {e}"))?;

    // Inject fresh pool state (market + both vaults) from the Yellowstone cache.
    for acc in &req.accounts {
        let pk = Pubkey::from_str(&acc.pubkey).map_err(|e| eyre::eyre!("acc {}: {e}", acc.pubkey))?;
        let data = B64.decode(&acc.data).map_err(|e| eyre::eyre!("data {}: {e}", acc.pubkey))?;
        let owner_bytes: [u8; 32] = bs58::decode(&acc.owner)
            .into_vec()
            .map_err(|e| eyre::eyre!("owner {}: {e}", acc.pubkey))?
            .try_into()
            .map_err(|_| eyre::eyre!("owner not 32 bytes for {}", acc.pubkey))?;
        svm.set_account(pk, Account {
            lamports: acc.lamports,
            data,
            owner: Pubkey::from(owner_bytes),
            executable: acc.executable,
            rent_epoch: acc.rent_epoch,
        })?;
    }

    // Warp the clock to the snapshot's slot so slot/time-sensitive PMMs
    // (BisonFi validates against the current slot) accept the injected state.
    // Without this the SVM runs at genesis (slot 0) and BisonFi rejects early.
    if req.slot > 0 {
        svm.warp_to_slot(req.slot);
    }

    // Mints (WSOL = 9, USDC = 6).
    svm.set_account(src_mint, Misc::mk_mint_acc(9))?;
    svm.set_account(dst_mint, Misc::mk_mint_acc(6))?;

    // Sim wallet ATAs: src funded with amount_in, dst empty.
    let sim_src_ta = get_associated_token_address(&sim_wallet.pubkey(), &src_mint);
    let sim_dst_ta = get_associated_token_address(&sim_wallet.pubkey(), &dst_mint);
    svm.set_account(sim_src_ta, Misc::mk_ata(&src_mint, &sim_wallet.pubkey(), req.amount_in))?;
    svm.set_account(sim_dst_ta, Misc::mk_ata(&dst_mint, &sim_wallet.pubkey(), 0))?;

    // Re-fund sim wallet for fees.
    svm.set_account(sim_wallet.pubkey(), Account {
        lamports: consts::AIRDROP_AMOUNT,
        data: vec![],
        owner: Pubkey::from_str("11111111111111111111111111111111").unwrap(),
        executable: false,
        rent_epoch: u64::MAX,
    })?;

    // Real wallet ATAs — referenced by the instruction we RETURN (for the live tx).
    let real_src_ta = get_associated_token_address(&fee_payer, &src_mint);
    let real_dst_ta = get_associated_token_address(&fee_payer, &dst_mint);

    // Derive the BisonFi market's REAL vault token accounts from the market
    // account data itself (layout: magic "POOLSTAT", base_ta@120, quote_ta@152,
    // baseMint@184, quoteMint@216). This is authoritative — it does NOT trust
    // the setup.toml / mix.json vault addresses, which are a common source of
    // wrong-account errors (BisonFi custom error 0x1a / 26).
    let market_data: Option<Vec<u8>> = req
        .accounts
        .iter()
        .find(|a| a.pubkey == req.market)
        .and_then(|a| B64.decode(&a.data).ok());
    let cfg = match market_data.as_deref().and_then(bisonfi_vaults_from_market) {
        Some((base_ta, quote_ta)) => {
            eprintln!(
                "[pmm-sim build_bison] derived vaults market={} base_ta={base_ta} quote_ta={quote_ta}",
                req.market
            );
            let mut c = cfg.clone();
            let mut swap_v1 = indexmap::IndexMap::new();
            swap_v1.insert(market, crate::cfg::BisonfiSwapV1 { market, market_base_ta: base_ta, market_quote_ta: quote_ta });
            c.bisonfi = Some(crate::cfg::BisonfiCfg { swap_v1 });
            c
        }
        None => {
            eprintln!(
                "[pmm-sim build_bison] WARN could not derive vaults from market {} data (len={:?}) — falling back to setup.toml",
                req.market,
                market_data.as_ref().map(|d| d.len())
            );
            cfg.clone()
        }
    };

    // Build the swap instruction. Either a DIRECT BisonFi call (no router) or
    // the DFlow swap2 (spoof-Magnus) router path.
    let real_ix = if req.direct {
        // Direct top-level BisonFi swap (selector 0x02 + amount + min(0) + dir).
        // Derive base/quote vault + base mint from the market data to set direction.
        let (base_ta, quote_ta) = market_data
            .as_deref()
            .and_then(bisonfi_vaults_from_market)
            .ok_or_else(|| eyre::eyre!("cannot derive vaults for direct BisonFi"))?;
        let base_mint = market_data
            .as_deref()
            .and_then(|d| {
                if d.len() >= 216 {
                    let arr: [u8; 32] = d[184..216].try_into().ok()?;
                    Some(Pubkey::from(arr))
                } else {
                    None
                }
            })
            .ok_or_else(|| eyre::eyre!("cannot derive base mint"))?;
        // src is base => direction 0 and (user_base, user_quote) = (src, dst).
        let src_is_base = src_mint == base_mint;
        let (dir, user_base, user_quote) = if src_is_base {
            (0u8, real_src_ta, real_dst_ta)
        } else {
            (1u8, real_dst_ta, real_src_ta)
        };
        let mut data = vec![0x02u8];
        data.extend_from_slice(&req.amount_in.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.push(dir);
        let bisonfi_pid = Pubkey::new_from_array(magnus_shared::pmm_bisonfi::id().to_bytes());
        eprintln!(
            "[pmm-sim build_bison] DIRECT market={} base_ta={base_ta} quote_ta={quote_ta} dir={dir}",
            req.market
        );
        Instruction {
            program_id: bisonfi_pid,
            accounts: vec![
                AccountMeta::new(fee_payer, true),
                AccountMeta::new(market, false),
                AccountMeta::new(base_ta, false),
                AccountMeta::new(quote_ta, false),
                AccountMeta::new(user_base, false),
                AccountMeta::new(user_quote, false),
                AccountMeta::new_readonly(Pubkey::new_from_array(magnus_shared::spl_token::id().to_bytes()), false),
                AccountMeta::new_readonly(Pubkey::new_from_array(magnus_shared::spl_token::id().to_bytes()), false),
                AccountMeta::new_readonly(solana_sdk::sysvar::instructions::id(), false),
            ],
            data,
        }
    } else {
        // Build the DFlow swap2 (spoof-Magnus) instruction for BisonFi.
        let routes: Vec<Vec<magnus_router_client::types::Route>> =
            vec![vec![Route { dexes: vec![Dex::BisonFi], weights: vec![100] }.into()]];
        let data = SwapArgs {
            amount_in: req.amount_in,
            expect_amount_out: 1,
            min_return: 1,
            amounts: vec![req.amount_in],
            routes,
        };
        let mut construct = ConstructSwap {
            cfg: cfg.clone(),
            remaining_accounts: vec![],
            payer: fee_payer,
            src_ta: real_src_ta,
            dst_ta: real_dst_ta,
            src_mint,
            dst_mint,
        };
        construct.attach_pmm_accs(&Dex::BisonFi, &market);
        construct.instruction(Some(Aggregator::DFlow), data, Misc::gen_order_id())
    };

    // Simulate a copy with the real fee_payer/ATAs remapped onto the sim wallet.
    let mut replace: HashMap<Pubkey, Pubkey> = HashMap::new();
    replace.insert(fee_payer, sim_wallet.pubkey());
    replace.insert(real_src_ta, sim_src_ta);
    replace.insert(real_dst_ta, sim_dst_ta);
    let sim_accounts: Vec<AccountMeta> = real_ix.accounts.iter().map(|a| {
        let pk = *replace.get(&a.pubkey).unwrap_or(&a.pubkey);
        match (a.is_writable, a.is_signer) {
            (true, true) => AccountMeta::new(pk, true),
            (true, false) => AccountMeta::new(pk, false),
            (false, true) => AccountMeta::new_readonly(pk, true),
            (false, false) => AccountMeta::new_readonly(pk, false),
        }
    }).collect();
    let sim_ix = Instruction {
        program_id: real_ix.program_id,
        accounts: sim_accounts,
        data: real_ix.data.clone(),
    };

    // Full instruction dump BEFORE simulation (so we can compare against the
    // real DFlow instruction as ground truth, even when the sim fails).
    dump_ix("REAL", &real_ix);
    dump_ix("SIM ", &sim_ix);

    // Build the out_ix once so we can return it on BOTH success and failure.
    let out_ix = OutInstruction {
        program_id: real_ix.program_id.to_string(),
        accounts: real_ix.accounts.iter().map(|a| OutAccountMeta {
            pubkey: a.pubkey.to_string(),
            is_signer: a.is_signer,
            is_writable: a.is_writable,
        }).collect(),
        data: B64.encode(&real_ix.data),
    };

    let initial_dst = token_balance_from_svm(svm, &sim_dst_ta);
    let tx = Transaction::new_signed_with_payer(
        &[sim_ix],
        Some(&sim_wallet.pubkey()),
        &[sim_wallet],
        svm.latest_blockhash(),
    );

    match svm.send_transaction(tx) {
        Ok(meta) => {
            let final_dst = token_balance_from_svm(svm, &sim_dst_ta);
            let amount_out = final_dst.checked_sub(initial_dst);
            eprintln!(
                "[pmm-sim build_bison] OK market={} amount_in={} amount_out_usdc={} direct={} slot={} cu={}",
                req.market, req.amount_in, amount_out.unwrap_or(0), req.direct, req.slot, meta.compute_units_consumed,
            );
            Ok(BuildBisonResponse {
                id: req.id,
                success: true,
                amount_out,
                compute_units: Some(meta.compute_units_consumed),
                instruction: Some(out_ix),
                error: None,
            })
        }
        Err(failed) => {
            // Include the program logs — they carry the real cause behind the
            // bare error code (e.g. BisonFi "slippage"/"direction"/"amount").
            let logs = failed.meta.logs.join(" | ");
            eprintln!("[pmm-sim build_bison] FAILED err={:?} logs=[{}]", failed.err, logs);
            Ok(BuildBisonResponse {
                id: req.id,
                success: false,
                amount_out: None,
                compute_units: Some(failed.meta.compute_units_consumed),
                // Return the instruction even on failure for debugging.
                instruction: Some(out_ix),
                error: Some(format!("{:?} | logs: {}", failed.err, logs)),
            })
        }
    }
}

/// Log a full instruction: program id, every account (pubkey/signer/writable in
/// exact order), and the data as byte array + base64 + length.
fn dump_ix(tag: &str, ix: &Instruction) {
    eprintln!("[ix_dump {tag}] program_id={}", ix.program_id);
    for (i, a) in ix.accounts.iter().enumerate() {
        eprintln!(
            "[ix_dump {tag}]   [{i}] {} signer={} writable={}",
            a.pubkey, a.is_signer, a.is_writable
        );
    }
    eprintln!(
        "[ix_dump {tag}] data_len={} data_b64={} data_bytes={:?}",
        ix.data.len(),
        B64.encode(&ix.data),
        ix.data
    );
}

// ── SVM initialisation ────────────────────────────────────────────────────────

fn load_extra_programs(svm: &mut LiteSVM, programs_path: &str) {
    let registry_path = format!("{programs_path}/programs_registry.json");
    let content = match std::fs::read_to_string(&registry_path) {
        Ok(c) => c,
        Err(_) => return, // no registry = no extra programs
    };
    #[derive(serde::Deserialize)]
    struct PEntry { label: String, program_id: String, so_path: String }
    let entries: Vec<PEntry> = match serde_json::from_str(&content) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("[pmm-sim serve] error parsing {registry_path}: {e}");
            return;
        }
    };
    for e in &entries {
        if !std::path::Path::new(&e.so_path).exists() {
            eprintln!("[pmm-sim serve] program .so not found: {} ({})", e.label, e.so_path);
            continue;
        }
        match Pubkey::from_str(&e.program_id) {
            Ok(pk) => match svm.add_program_from_file(pk, &e.so_path) {
                Ok(_) => eprintln!("[pmm-sim serve] loaded: {} ({})", e.label, e.program_id),
                Err(err) => eprintln!("[pmm-sim serve] failed to load {}: {err}", e.label),
            },
            Err(_) => eprintln!("[pmm-sim serve] invalid program_id for {}: {}", e.label, e.program_id),
        }
    }
}

fn build_base_svm(programs_path: &str) -> eyre::Result<LiteSVM> {
    let mut budget = ComputeBudget::new_with_defaults(false, false);
    budget.compute_unit_limit = consts::COMPUTE_UNITS_LIMIT;

    let mut svm = LiteSVM::new()
        .with_default_programs()
        .with_sysvars()
        .with_sigverify(false) // We sign with sim_wallet but instruction references fee_payer.
        .with_compute_budget(budget);

    // Load all router variants (spoof + plain).
    load_router_programs(&mut svm, programs_path)?;

    // Load all supported PMM programs.
    load_pmm_programs(&mut svm, programs_path)?;

    // Load extra programs from programs_registry.json (external DEX programs).
    load_extra_programs(&mut svm, programs_path);

    eprintln!("[pmm-sim serve] LiteSVM ready (programs_path={programs_path})");
    Ok(svm)
}

fn load_router_programs(svm: &mut LiteSVM, programs_path: &str) -> eyre::Result<()> {
    // Plain Magnus router.
    svm.add_program_from_file(
        router_program_id(),
        format!("{programs_path}/magnus-router.so"),
    )?;

    // Spoofed router variants.
    for aggr in [Aggregator::Jupiter, Aggregator::DFlow, Aggregator::OkxLabs, Aggregator::Titan] {
        let path = format!("{programs_path}/magnus-router-spoof-{aggr}.so");
        if std::path::Path::new(&path).exists() {
            svm.add_program_from_file(aggr.program_id(), path)?;
        }
    }
    Ok(())
}

fn load_pmm_programs(svm: &mut LiteSVM, programs_path: &str) -> eyre::Result<()> {
    let programs: &[(Pubkey, &str)] = &[
        (Pubkey::new_from_array(pmm_humidifi::id().to_bytes()), "humidifi"),
        (Pubkey::new_from_array(pmm_bisonfi::id().to_bytes()),  "bisonfi"),
        (Pubkey::new_from_array(pmm_goonfi::id().to_bytes()),   "goonfi"),
        (Pubkey::new_from_array(pmm_obric_v2::id().to_bytes()), "obric-v2"),
        (Pubkey::new_from_array(pmm_solfi_v2::id().to_bytes()), "solfi-v2"),
        (Pubkey::new_from_array(pmm_tessera::id().to_bytes()),  "tessera"),
        (Pubkey::new_from_array(pmm_zerofi::id().to_bytes()),   "zerofi"),
    ];

    for (program_id, name) in programs {
        let path = format!("{programs_path}/{name}.so");
        if std::path::Path::new(&path).exists() {
            svm.add_program_from_file(*program_id, path)?;
        } else {
            eprintln!("[pmm-sim serve] warning: {path} not found, skipping");
        }
    }
    Ok(())
}

fn fund_wallet(svm: &mut LiteSVM, pubkey: &Pubkey, lamports: u64) {
    svm.airdrop(pubkey, lamports).expect("airdrop failed");
}

// ── Per-request processing ────────────────────────────────────────────────────

const COMPUTE_BUDGET_PROGRAM_ID: &str = "ComputeBudget111111111111111111111111111111";

fn process_request(svm: &mut LiteSVM, sim_wallet: &Keypair, line: &str) -> eyre::Result<ServeResponse> {
    let req: ServeRequest = serde_json::from_str(line)?;

    let fee_payer = Pubkey::from_str(&req.fee_payer)
        .map_err(|e| eyre::eyre!("invalid fee_payer: {e}"))?;
    let src_mint = Pubkey::from_str(&req.src_mint)
        .map_err(|e| eyre::eyre!("invalid src_mint: {e}"))?;

    let token_mints: Vec<Pubkey> = req.token_mints.iter()
        .filter_map(|s| Pubkey::from_str(s).ok())
        .collect();

    // Inject pool accounts into SVM.
    for acc in &req.accounts {
        let pk = Pubkey::from_str(&acc.pubkey)
            .map_err(|e| eyre::eyre!("invalid account pubkey {}: {e}", acc.pubkey))?;
        let data = B64.decode(&acc.data)
            .map_err(|e| eyre::eyre!("invalid account data for {}: {e}", acc.pubkey))?;
        let owner_bytes = bs58::decode(&acc.owner).into_vec()
            .map_err(|e| eyre::eyre!("invalid owner for {}: {e}", acc.pubkey))?;
        let owner: [u8; 32] = owner_bytes.try_into()
            .map_err(|_| eyre::eyre!("owner not 32 bytes for {}", acc.pubkey))?;
        svm.set_account(pk, Account {
            lamports: acc.lamports,
            data,
            owner: Pubkey::from(owner),
            executable: acc.executable,
            rent_epoch: acc.rent_epoch,
        })?;
    }

    // Set up mint accounts.
    svm.set_account(src_mint, Misc::mk_mint_acc(9))?;
    for mint in &token_mints {
        if *mint != src_mint {
            svm.set_account(*mint, Misc::mk_mint_acc(6))?;
        }
    }

    // Build replacement map: fee_payer → sim_wallet, ATAs → sim ATAs.
    let mut replace: HashMap<Pubkey, Pubkey> = HashMap::new();
    replace.insert(fee_payer, sim_wallet.pubkey());
    for mint in &token_mints {
        let orig_ata = get_associated_token_address(&fee_payer, mint);
        let sim_ata  = get_associated_token_address(&sim_wallet.pubkey(), mint);
        replace.insert(orig_ata, sim_ata);
    }

    // Set up simulation wallet ATAs.
    for mint in &token_mints {
        let amount = if *mint == src_mint { req.src_amount } else { 0 };
        let sim_ata = get_associated_token_address(&sim_wallet.pubkey(), mint);
        svm.set_account(sim_ata, Misc::mk_ata(mint, &sim_wallet.pubkey(), amount))?;
    }

    // Re-fund simulation wallet.
    svm.set_account(sim_wallet.pubkey(), Account {
        lamports: consts::AIRDROP_AMOUNT,
        data: vec![],
        owner: Pubkey::from_str("11111111111111111111111111111111").unwrap(),
        executable: false,
        rent_epoch: u64::MAX,
    })?;

    // Read initial src balance for profit calculation.
    let src_ata = get_associated_token_address(&sim_wallet.pubkey(), &src_mint);
    let initial_balance = token_balance_from_svm(svm, &src_ata);

    // Build patched instructions (skip ComputeBudget, patch fee_payer/ATAs).
    let mut ixs: Vec<Instruction> = Vec::new();
    for serve_ix in &req.instructions {
        if serve_ix.program_id == COMPUTE_BUDGET_PROGRAM_ID {
            continue; // LiteSVM has its own budget; skip these
        }
        let program_id = Pubkey::from_str(&serve_ix.program_id)
            .map_err(|e| eyre::eyre!("invalid program_id {}: {e}", serve_ix.program_id))?;
        let ix_data = B64.decode(&serve_ix.data)
            .map_err(|e| eyre::eyre!("invalid instruction data: {e}"))?;
        let accounts: Vec<AccountMeta> = serve_ix.accounts.iter()
            .map(|a| {
                let pk = Pubkey::from_str(&a.pubkey).unwrap_or_default();
                let pk = *replace.get(&pk).unwrap_or(&pk);
                match (a.is_writable, a.is_signer) {
                    (true,  true)  => AccountMeta::new(pk, true),
                    (true,  false) => AccountMeta::new(pk, false),
                    (false, true)  => AccountMeta::new_readonly(pk, true),
                    (false, false) => AccountMeta::new_readonly(pk, false),
                }
            })
            .collect();
        ixs.push(Instruction { program_id, accounts, data: ix_data });
    }

    if ixs.is_empty() {
        return Err(eyre::eyre!("no executable instructions after filtering"));
    }

    let tx = Transaction::new_signed_with_payer(
        &ixs,
        Some(&sim_wallet.pubkey()),
        &[sim_wallet],
        svm.latest_blockhash(),
    );

    match svm.send_transaction(tx) {
        Ok(meta) => {
            // Use balance change as profit source; fall back to log parsing.
            let final_balance = token_balance_from_svm(svm, &src_ata);
            let amount_out = if final_balance > initial_balance {
                Some(final_balance - initial_balance)
            } else {
                // Fallback: try parsing "after_destination_balance: N" from logs.
                meta.logs.iter().find_map(|log| {
                    log.split("after_destination_balance: ")
                        .nth(1)?
                        .split(|c: char| !c.is_ascii_digit())
                        .next()?
                        .parse::<u64>()
                        .ok()
                })
            };
            Ok(ServeResponse {
                id: req.id,
                success: true,
                amount_out,
                compute_units: Some(meta.compute_units_consumed),
                error: None,
            })
        }
        Err(failed) => Ok(ServeResponse {
            id: req.id,
            success: false,
            amount_out: None,
            compute_units: Some(failed.meta.compute_units_consumed),
            error: Some(format!("{:?}", failed.err)),
        }),
    }
}

fn token_balance_from_svm(svm: &LiteSVM, ata: &Pubkey) -> u64 {
    svm.get_account(ata)
        .and_then(|a| {
            // SPL token account: amount is at bytes 64..72 (little-endian u64).
            if a.data.len() >= 72 {
                Some(u64::from_le_bytes(a.data[64..72].try_into().ok()?))
            } else {
                None
            }
        })
        .unwrap_or(0)
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn extract_id(line: &str) -> Option<u64> {
    #[derive(Deserialize)]
    struct IdOnly { id: u64 }
    serde_json::from_str::<IdOnly>(line).ok().map(|v| v.id)
}
