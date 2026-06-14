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
use crate::misc::Misc;
use crate::consts;

// ── IPC types ─────────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct ServeRequest {
    id: u64,
    fee_payer: String,
    token_mints: Vec<String>,
    src_mint: String,
    src_amount: u64,
    swap_instruction: ServeIx,
    accounts: Vec<ServeAccount>,
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

// ── Router program IDs ────────────────────────────────────────────────────────

fn router_program_id() -> Pubkey {
    magnus_router_client::programs::ROUTER_ID
}

fn aggregator_for_program(program_id: &Pubkey) -> Option<Aggregator> {
    if *program_id == Aggregator::Jupiter.program_id() {
        Some(Aggregator::Jupiter)
    } else if *program_id == Aggregator::DFlow.program_id() {
        Some(Aggregator::DFlow)
    } else if *program_id == Aggregator::OkxLabs.program_id() {
        Some(Aggregator::OkxLabs)
    } else if *program_id == Aggregator::Titan.program_id() {
        Some(Aggregator::Titan)
    } else {
        None
    }
}

// ── Serve entry point ─────────────────────────────────────────────────────────

/// Run the serve loop: initialise LiteSVM with all programs, then process
/// JSON requests from stdin until EOF.
pub fn run(programs_path: &str) -> eyre::Result<()> {
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

        let response = match process_request(&mut svm, &sim_wallet, &line) {
            Ok(resp) => resp,
            Err(e) => {
                // Try to extract the id for the error response.
                let id = extract_id(&line).unwrap_or(0);
                ServeResponse { id, success: false, amount_out: None, compute_units: None, error: Some(e.to_string()) }
            }
        };

        let mut out = serde_json::to_string(&response).unwrap_or_else(|_| r#"{"id":0,"success":false,"error":"serialization error"}"#.to_string());
        out.push('\n');
        let _ = stdout_lock.write_all(out.as_bytes());
        let _ = stdout_lock.flush();
    }

    Ok(())
}

// ── SVM initialisation ────────────────────────────────────────────────────────

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

fn process_request(svm: &mut LiteSVM, sim_wallet: &Keypair, line: &str) -> eyre::Result<ServeResponse> {
    let req: ServeRequest = serde_json::from_str(line)?;

    // Parse pubkeys.
    let fee_payer = Pubkey::from_str(&req.fee_payer)
        .map_err(|e| eyre::eyre!("invalid fee_payer: {e}"))?;
    let src_mint = Pubkey::from_str(&req.src_mint)
        .map_err(|e| eyre::eyre!("invalid src_mint: {e}"))?;

    // Parse all token mints for ATA patching.
    let token_mints: Vec<Pubkey> = req.token_mints.iter()
        .filter_map(|s| Pubkey::from_str(s).ok())
        .collect();

    // Inject fresh pool accounts into the SVM.
    for acc in &req.accounts {
        let pk = Pubkey::from_str(&acc.pubkey)
            .map_err(|e| eyre::eyre!("invalid account pubkey {}: {e}", acc.pubkey))?;
        let data = B64.decode(&acc.data)
            .map_err(|e| eyre::eyre!("invalid account data for {}: {e}", acc.pubkey))?;
        let owner_bytes = bs58::decode(&acc.owner).into_vec()
            .map_err(|e| eyre::eyre!("invalid owner for {}: {e}", acc.pubkey))?;
        let owner: [u8; 32] = owner_bytes.try_into()
            .map_err(|_| eyre::eyre!("owner not 32 bytes for {}", acc.pubkey))?;
        let account = Account {
            lamports: acc.lamports,
            data,
            owner: Pubkey::from(owner),
            executable: acc.executable,
            rent_epoch: acc.rent_epoch,
        };
        svm.set_account(pk, account)?;
    }

    // Set up mint accounts (needed for ATA creation).
    svm.set_account(src_mint, Misc::mk_mint_acc(9))?; // assume 9 decimals (WSOL)
    for mint in &token_mints {
        if *mint != src_mint {
            // Assume 6 decimals for non-WSOL tokens; exact decimals don't affect swap logic.
            svm.set_account(*mint, Misc::mk_mint_acc(6))?;
        }
    }

    // Compute original ATAs (fee_payer's) and replacement ATAs (sim_wallet's).
    // Build a replacement map: original_pubkey → replacement_pubkey.
    let mut replace: HashMap<Pubkey, Pubkey> = HashMap::new();
    replace.insert(fee_payer, sim_wallet.pubkey());
    for mint in &token_mints {
        let orig_ata = get_associated_token_address(&fee_payer, mint);
        let sim_ata  = get_associated_token_address(&sim_wallet.pubkey(), mint);
        replace.insert(orig_ata, sim_ata);
    }

    // Set up the simulation wallet's ATAs with correct starting balances.
    for mint in &token_mints {
        let amount = if *mint == src_mint { req.src_amount } else { 0 };
        let sim_ata = get_associated_token_address(&sim_wallet.pubkey(), mint);
        svm.set_account(sim_ata, Misc::mk_ata(mint, &sim_wallet.pubkey(), amount))?;
    }

    // Re-fund the simulation wallet to cover transaction fees.
    svm.set_account(
        sim_wallet.pubkey(),
        Account {
            lamports: consts::AIRDROP_AMOUNT,
            data: vec![],
            owner: solana_sdk::system_program::id(),
            executable: false,
            rent_epoch: u64::MAX,
        },
    )?;

    // Build the patched instruction.
    let program_id = Pubkey::from_str(&req.swap_instruction.program_id)
        .map_err(|e| eyre::eyre!("invalid program_id: {e}"))?;
    let ix_data = B64.decode(&req.swap_instruction.data)
        .map_err(|e| eyre::eyre!("invalid instruction data: {e}"))?;

    let accounts: Vec<AccountMeta> = req.swap_instruction.accounts.iter()
        .map(|a| {
            let pk = Pubkey::from_str(&a.pubkey).unwrap_or_default();
            // Apply replacement: use sim_wallet's accounts where fee_payer's were.
            let pk = *replace.get(&pk).unwrap_or(&pk);
            match (a.is_writable, a.is_signer) {
                (true,  true)  => AccountMeta::new(pk, true),
                (true,  false) => AccountMeta::new(pk, false),
                (false, true)  => AccountMeta::new_readonly(pk, true),
                (false, false) => AccountMeta::new_readonly(pk, false),
            }
        })
        .collect();

    let ix = Instruction { program_id, accounts, data: ix_data };

    // Create and sign the transaction (sigverify is disabled in the SVM).
    let tx = Transaction::new_signed_with_payer(
        &[ix],
        Some(&sim_wallet.pubkey()),
        &[sim_wallet],
        svm.latest_blockhash(),
    );

    match svm.send_transaction(tx) {
        Ok(meta) => {
            let amount_out = meta.logs.iter().find_map(|log| {
                log.split("after_destination_balance: ")
                    .nth(1)?
                    .split(|c: char| !c.is_ascii_digit())
                    .next()?
                    .parse::<u64>()
                    .ok()
            });
            Ok(ServeResponse {
                id: req.id,
                success: true,
                amount_out,
                compute_units: Some(meta.compute_units_consumed),
                error: None,
            })
        }
        Err(failed) => {
            let err_msg = format!("{:?}", failed.err);
            Ok(ServeResponse {
                id: req.id,
                success: false,
                amount_out: None,
                compute_units: Some(failed.meta.compute_units_consumed),
                error: Some(err_msg),
            })
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn extract_id(line: &str) -> Option<u64> {
    #[derive(Deserialize)]
    struct IdOnly { id: u64 }
    serde_json::from_str::<IdOnly>(line).ok().map(|v| v.id)
}
