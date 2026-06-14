/// sim-server: Full-transaction LiteSVM simulation server.
///
/// Reads newline-delimited JSON requests from stdin, executes the complete
/// Metis transaction (compute_budget + setup + swap + cleanup) in LiteSVM,
/// and writes the JSON result to stdout.
///
/// No private dependencies required — compiles independently of pmm-sim.
/// Handles all standard DEX programs (Raydium, Orca, Meteora, etc.) and any
/// extra programs provided via programs_registry.json or pubkey-named .so files.
///
/// CLI:
///   sim-server serve --programs-path <path>
///
/// The <programs-path> directory should contain:
///   - programs_registry.json  (written by the bot from [[pmm_sim.programs]] config)
///   - *.so files named as their base58 program ID (auto-loaded)
///
/// IPC Protocol (identical to pmm-sim serve):
///   Request:  one JSON line (FullSimRequest)
///   Response: one JSON line (ServeResponse)
use std::{
    collections::{HashMap, HashSet},
    io::{BufRead, Write},
    str::FromStr,
};

use base64::{Engine as _, engine::general_purpose::STANDARD as B64};
use litesvm::LiteSVM;
use serde::{Deserialize, Serialize};
use solana_sdk::{
    account::Account,
    instruction::{AccountMeta, Instruction},
    program_option::COption,
    pubkey::Pubkey,
    rent::Rent,
    signature::Keypair,
    signer::Signer,
    transaction::Transaction,
};
use spl_associated_token_account::get_associated_token_address;

// ── Constants ─────────────────────────────────────────────────────────────────

const AIRDROP_AMOUNT: u64 = 100_000_000_000; // 100 SOL for the sim wallet
const COMPUTE_BUDGET_PROGRAM: &str = "ComputeBudget111111111111111111111111111111";

// ── IPC types (must match bot's FullSimRequest / ServeResponse) ───────────────

#[derive(Deserialize)]
struct SimRequest {
    id: u64,
    fee_payer: String,
    token_mints: Vec<String>,
    src_mint: String,
    src_amount: u64,
    /// ALL instructions: compute_budget + setup + swap + cleanup.
    instructions: Vec<SimIx>,
    #[serde(default)]
    lookup_tables: Vec<SimLookupTable>,
    accounts: Vec<SimAccount>,
    #[serde(default = "default_tip")]
    jito_tip_lamports: u64,
    #[serde(default = "default_cu")]
    cu_limit: u32,
    #[serde(default)]
    route_sig: String,
}

fn default_tip() -> u64 { 1600 }
fn default_cu() -> u32 { 1_400_000 }

#[derive(Deserialize)]
struct SimIx {
    program_id: String,
    accounts: Vec<SimAccountMeta>,
    data: String, // base64
}

#[derive(Deserialize)]
struct SimAccountMeta {
    pubkey: String,
    is_signer: bool,
    is_writable: bool,
}

#[derive(Deserialize)]
struct SimLookupTable {
    key: String,
    addresses: Vec<String>,
}

#[derive(Deserialize)]
struct SimAccount {
    pubkey: String,
    lamports: u64,
    data: String,   // base64
    owner: String,  // base58
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

// ── Known DEX program ID → .so filename mapping ───────────────────────────────
//
// These are the public, well-known on-chain program IDs for each DEX.
// The .so filenames match what you'd get from `solana program dump`.
// Programs are loaded from `programs_path`/`filename`.
//
// ⚠ VERIFY: Program IDs marked with [?] should be verified on-chain with:
//   solana program show <PROGRAM_ID>
// ⚠ MISSING: AlphaQ, PancakeSwap program IDs are not publicly documented.
//   Add them to [[pmm_sim.programs]] in config.toml and they'll load via
//   programs_registry.json.

fn known_dex_programs() -> Vec<(&'static str, &'static str)> {
    vec![
        // Raydium ──────────────────────────────────────────────────────────────
        ("675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8", "Raydium_AMM_v4"),
        ("CAMMCzo5YL8w4VFF8KVHrK22GGUsp5VTaW7grrKgrWqK", "Raydium_CLMM"),
        ("CAMMCzo5YL8w4VFF8KVHrK22GGUsp5VTaW7grrKgrWqK", "Raydium_ConcentratedLiquidity"), // same program
        ("CPMMoo8L3F4NbTegBCKVNunggL7H1ZpdTHKxQB5qKP1C", "Raydium_CPMM"),

        // Orca ─────────────────────────────────────────────────────────────────
        ("whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc",  "Whirlpools_Program"),
        ("9W959DqEETiGZocYWCQPaJ6sBmUzgfxXfqGeTEdp3aQP", "Orca_Token_Swap_V2"),

        // Meteora ──────────────────────────────────────────────────────────────
        ("LBUZKhRxPF3XUpBCjp4YzTKgLccjZhTSDM9YuVaPwxo", "Meteora_DLMM_Program"),
        // [?] Meteora Dynamic AMM (Pools):
        ("Eo7WjKq67rjJQDd1d4t4fBRDCYGGGZnHVTsC3AaUyKS", "Meteora_Pools_Program"),
        // [?] Meteora DAMM v2:
        ("cpamdpRtUpGnoRjDngDsg8JySFGuqUCEiWTLqF99gCNE", "Meteora_DAMM_v2"),
        // [?] Meteora Vault:
        ("24Uqj9JCLxUeoC3hGfh5W3s9FM9uCHDS2SG3LYwBpyTi", "Meteora_Vault_Program"),
        // Mercurial Stable Swap (legacy, but still used):
        ("MERLuDFBMmsHnsBPZw2sDQZHvXFMwp8EdjudcU2HKky",  "Mercurial_Stable_Swap"),

        // Invariant ────────────────────────────────────────────────────────────
        // [?] Verify with: solana program show HyaB3W9q6XdA5xwpU4XnSZV94htfmbmqJXZcEbRaJutt
        ("HyaB3W9q6XdA5xwpU4XnSZV94htfmbmqJXZcEbRaJutt", "Invariant_Swap"),

        // Manifest ─────────────────────────────────────────────────────────────
        // [?] Verify with: solana program show MNFSTqtC93rEfYHB6hF82sKdZpUDFWkViLhhLc65fR4
        ("MNFSTqtC93rEfYHB6hF82sKdZpUDFWkViLhhLc65fR4", "Manifest"),

        // Jupiter (router used by Metis) ───────────────────────────────────────
        ("JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4", "Jupiter_Aggregator_v6"),

        // SPL Token programs ───────────────────────────────────────────────────
        ("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",  "spl_token"),
        ("TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb",  "token_2022"),
        ("ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJe8bXh", "spl_associated_token_account-1.1.1"),

        // ⚠ MISSING — add to [[pmm_sim.programs]] in config.toml:
        //   AlphaQ.so       → program_id unknown, add as [[pmm_sim.programs]]
        //   PancakeSwap.so  → program_id unknown, add as [[pmm_sim.programs]]
        //   1Dex_Program.so → program_id unknown, add as [[pmm_sim.programs]]
        //
        // PropAMM programs (HumidiFi, BisonFi, GoonFi, SolFi, Tessera, ZeroFi, ObricV2)
        // also go in [[pmm_sim.programs]] — program IDs come from the user's deployment.
    ]
}

// ── Program loading ───────────────────────────────────────────────────────────

fn load_all_programs(svm: &mut LiteSVM, programs_path: &str) {
    let mut loaded: HashSet<String> = HashSet::new();

    // 1. Load from programs_registry.json (written by the bot from config.toml [[pmm_sim.programs]])
    load_from_registry(svm, programs_path, &mut loaded);

    // 2. Auto-load any *.so file whose stem is a valid base58 pubkey.
    load_pubkey_named(svm, programs_path, &mut loaded);

    // 3. Load known DEX programs by hardcoded (program_id → filename) mapping.
    load_known(svm, programs_path, &mut loaded);

    eprintln!("[sim-server] programs loaded: {}", loaded.len());
}

fn load_from_registry(svm: &mut LiteSVM, programs_path: &str, loaded: &mut HashSet<String>) {
    let registry_path = format!("{programs_path}/programs_registry.json");
    let content = match std::fs::read_to_string(&registry_path) {
        Ok(c) => c,
        Err(_) => return,
    };
    #[derive(Deserialize)]
    struct PEntry { label: String, program_id: String, so_path: String }
    let entries: Vec<PEntry> = match serde_json::from_str(&content) {
        Ok(e) => e,
        Err(e) => { eprintln!("[sim-server] registry parse error: {e}"); return; }
    };
    for e in &entries {
        if loaded.contains(&e.program_id) { continue; }
        if !std::path::Path::new(&e.so_path).exists() {
            eprintln!("[sim-server] .so not found: {} ({})", e.label, e.so_path);
            continue;
        }
        match Pubkey::from_str(&e.program_id) {
            Ok(pk) => match svm.add_program_from_file(pk, &e.so_path) {
                Ok(_) => { eprintln!("[sim-server] loaded (registry): {}", e.label); loaded.insert(e.program_id.clone()); }
                Err(e2) => eprintln!("[sim-server] failed to load {}: {e2}", e.label),
            },
            Err(_) => eprintln!("[sim-server] invalid program_id for {}: {}", e.label, e.program_id),
        }
    }
}

fn load_pubkey_named(svm: &mut LiteSVM, programs_path: &str, loaded: &mut HashSet<String>) {
    let Ok(dir) = std::fs::read_dir(programs_path) else { return };
    for entry in dir.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("so") { continue; }
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
        // Check if stem is a valid base58 Pubkey (32 bytes when decoded)
        let is_pubkey = bs58::decode(stem).into_vec()
            .map(|v| v.len() == 32)
            .unwrap_or(false);
        if !is_pubkey { continue; }
        if loaded.contains(stem) { continue; }
        match Pubkey::from_str(stem) {
            Ok(pk) => match svm.add_program_from_file(pk, &path) {
                Ok(_) => { eprintln!("[sim-server] loaded (auto): {stem}"); loaded.insert(stem.to_string()); }
                Err(e) => eprintln!("[sim-server] failed to auto-load {stem}: {e}"),
            },
            Err(_) => {}
        }
    }
}

fn load_known(svm: &mut LiteSVM, programs_path: &str, loaded: &mut HashSet<String>) {
    let known = known_dex_programs();
    // Deduplicate: same program_id may map to multiple filenames (e.g. CLMM aliases).
    let mut seen_ids: HashSet<&str> = HashSet::new();
    for (program_id, filename) in &known {
        if loaded.contains(*program_id) { continue; }
        if !seen_ids.insert(program_id) { continue; } // skip duplicate IDs
        let path = format!("{programs_path}/{filename}.so");
        if !std::path::Path::new(&path).exists() { continue; }
        match Pubkey::from_str(program_id) {
            Ok(pk) => match svm.add_program_from_file(pk, &path) {
                Ok(_) => { eprintln!("[sim-server] loaded (known): {filename}"); loaded.insert(program_id.to_string()); }
                Err(e) => eprintln!("[sim-server] failed to load {filename}: {e}"),
            },
            Err(_) => {}
        }
    }
}

// ── SPL account helpers ───────────────────────────────────────────────────────

fn make_mint_account(decimals: u8) -> Account {
    use spl_token::state::Mint;
    use solana_sdk::program_pack::Pack;
    let mint = Mint {
        mint_authority: COption::None,
        supply: u64::MAX,
        decimals,
        is_initialized: true,
        freeze_authority: COption::None,
    };
    let mut data = vec![0u8; Mint::LEN];
    Mint::pack(mint, &mut data).unwrap();
    Account {
        lamports: Rent::default().minimum_balance(data.len()),
        data,
        owner: spl_token::id(),
        executable: false,
        rent_epoch: u64::MAX,
    }
}

fn make_token_account(mint: &Pubkey, owner: &Pubkey, amount: u64) -> Account {
    use spl_token::state::Account as TokenAccount;
    use solana_sdk::program_pack::Pack;
    let ta = TokenAccount {
        mint: *mint,
        owner: *owner,
        amount,
        state: spl_token::state::AccountState::Initialized,
        ..Default::default()
    };
    let mut data = vec![0u8; TokenAccount::LEN];
    TokenAccount::pack(ta, &mut data).unwrap();
    Account {
        lamports: Rent::default().minimum_balance(data.len()),
        data,
        owner: spl_token::id(),
        executable: false,
        rent_epoch: u64::MAX,
    }
}

fn token_balance(svm: &LiteSVM, ata: &Pubkey) -> u64 {
    // SPL token account: amount is at bytes 64..72 (little-endian u64 after mint[32] + owner[32]).
    svm.get_account(ata)
        .and_then(|a| {
            if a.data.len() >= 72 {
                let bytes: [u8; 8] = a.data[64..72].try_into().ok()?;
                Some(u64::from_le_bytes(bytes))
            } else {
                None
            }
        })
        .unwrap_or(0)
}

// ── Per-request processing ────────────────────────────────────────────────────

fn process(svm: &mut LiteSVM, sim_wallet: &Keypair, line: &str) -> eyre::Result<ServeResponse> {
    let req: SimRequest = serde_json::from_str(line)
        .map_err(|e| eyre::eyre!("parse error: {e}"))?;

    let fee_payer = Pubkey::from_str(&req.fee_payer)
        .map_err(|e| eyre::eyre!("invalid fee_payer: {e}"))?;
    let src_mint = Pubkey::from_str(&req.src_mint)
        .map_err(|e| eyre::eyre!("invalid src_mint: {e}"))?;

    let token_mints: Vec<Pubkey> = req.token_mints.iter()
        .filter_map(|s| Pubkey::from_str(s).ok())
        .collect();

    // Inject provided pool accounts into SVM.
    for acc in &req.accounts {
        let pk = Pubkey::from_str(&acc.pubkey)
            .map_err(|e| eyre::eyre!("invalid pubkey {}: {e}", acc.pubkey))?;
        let data = B64.decode(&acc.data)
            .map_err(|e| eyre::eyre!("bad data for {}: {e}", acc.pubkey))?;
        let owner_bytes: [u8; 32] = bs58::decode(&acc.owner)
            .into_vec()
            .map_err(|e| eyre::eyre!("bad owner for {}: {e}", acc.pubkey))?
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

    // Set up mint accounts.
    svm.set_account(src_mint, make_mint_account(9))?; // WSOL = 9 decimals
    for mint in &token_mints {
        if *mint != src_mint {
            svm.set_account(*mint, make_mint_account(6))?;
        }
    }

    // Build replacement map: fee_payer → sim_wallet, all ATAs → sim ATAs.
    let mut replace: HashMap<Pubkey, Pubkey> = HashMap::new();
    replace.insert(fee_payer, sim_wallet.pubkey());
    for mint in &token_mints {
        let orig_ata = get_associated_token_address(&fee_payer, mint);
        let sim_ata  = get_associated_token_address(&sim_wallet.pubkey(), mint);
        replace.insert(orig_ata, sim_ata);
    }

    // Create sim_wallet's token accounts.
    for mint in &token_mints {
        let amount  = if *mint == src_mint { req.src_amount } else { 0 };
        let sim_ata = get_associated_token_address(&sim_wallet.pubkey(), mint);
        svm.set_account(sim_ata, make_token_account(mint, &sim_wallet.pubkey(), amount))?;
    }

    // Fund the simulation wallet.
    svm.set_account(sim_wallet.pubkey(), Account {
        lamports: AIRDROP_AMOUNT,
        data: vec![],
        owner: Pubkey::from_str("11111111111111111111111111111111").unwrap(),
        executable: false,
        rent_epoch: u64::MAX,
    })?;

    // Read initial src balance (for profit calculation).
    let src_ata = get_associated_token_address(&sim_wallet.pubkey(), &src_mint);
    let initial_balance = token_balance(svm, &src_ata);

    // Build patched instruction list (skip ComputeBudget, replace pubkeys).
    let mut ixs: Vec<Instruction> = Vec::new();
    for six in &req.instructions {
        if six.program_id == COMPUTE_BUDGET_PROGRAM {
            continue; // LiteSVM uses its own budget
        }
        let program_id = Pubkey::from_str(&six.program_id)
            .map_err(|e| eyre::eyre!("invalid program_id {}: {e}", six.program_id))?;
        let ix_data = B64.decode(&six.data)
            .map_err(|e| eyre::eyre!("bad instruction data: {e}"))?;
        let accounts: Vec<AccountMeta> = six.accounts.iter().map(|a| {
            let pk = Pubkey::from_str(&a.pubkey).unwrap_or_default();
            let pk = *replace.get(&pk).unwrap_or(&pk);
            match (a.is_writable, a.is_signer) {
                (true,  true)  => AccountMeta::new(pk, true),
                (true,  false) => AccountMeta::new(pk, false),
                (false, true)  => AccountMeta::new_readonly(pk, true),
                (false, false) => AccountMeta::new_readonly(pk, false),
            }
        }).collect();
        ixs.push(Instruction { program_id, accounts, data: ix_data });
    }

    if ixs.is_empty() {
        return Err(eyre::eyre!("no executable instructions (only ComputeBudget?)"));
    }

    let tx = Transaction::new_signed_with_payer(
        &ixs,
        Some(&sim_wallet.pubkey()),
        &[sim_wallet],
        svm.latest_blockhash(),
    );

    match svm.send_transaction(tx) {
        Ok(meta) => {
            // Primary: balance change of WSOL ATA.
            let final_balance = token_balance(svm, &src_ata);
            let amount_out = if final_balance > initial_balance {
                Some(final_balance - initial_balance)
            } else {
                // Fallback: parse "after_destination_balance: N" from program logs.
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

fn extract_id(line: &str) -> u64 {
    #[derive(Deserialize)]
    struct IdOnly { id: u64 }
    serde_json::from_str::<IdOnly>(line).map(|v| v.id).unwrap_or(0)
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    // Parse args: sim-server serve --programs-path <path>
    // (--setup-path is silently ignored for compatibility with pmm-sim CLI)
    let args: Vec<String> = std::env::args().collect();
    let programs_path = args.windows(2)
        .find(|w| w[0] == "--programs-path")
        .map(|w| w[1].clone())
        .unwrap_or_else(|| "./cfg/programs".to_string());

    // Build LiteSVM with sigverify disabled (we sign with sim_wallet but instructions
    // reference the original fee_payer pubkey before patching).
    use solana_compute_budget::compute_budget::ComputeBudget;
    let mut budget = ComputeBudget::new_with_defaults(false, false);
    budget.compute_unit_limit = 20_000_000;

    let mut svm = LiteSVM::new()
        .with_default_programs()
        .with_sysvars()
        .with_sigverify(false)
        .with_compute_budget(budget);

    // Load all programs.
    load_all_programs(&mut svm, &programs_path);

    // One persistent simulation wallet — re-funded on every request.
    let sim_wallet = Keypair::new();

    eprintln!("[sim-server] ready (programs_path={programs_path} wallet={})", sim_wallet.pubkey());

    let stdin  = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) if !l.trim().is_empty() => l,
            Ok(_) => continue,
            Err(e) => { eprintln!("[sim-server] stdin error: {e}"); break; }
        };

        let response = match process(&mut svm, &sim_wallet, &line) {
            Ok(r) => r,
            Err(e) => ServeResponse {
                id: extract_id(&line),
                success: false,
                amount_out: None,
                compute_units: None,
                error: Some(e.to_string()),
            },
        };

        let mut json = serde_json::to_string(&response)
            .unwrap_or_else(|_| r#"{"id":0,"success":false,"error":"serialization error"}"#.to_string());
        json.push('\n');
        let _ = out.write_all(json.as_bytes());
        let _ = out.flush();
    }
}
