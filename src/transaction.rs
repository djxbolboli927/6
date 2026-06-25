use anyhow::{Context, Result};
use rand::seq::SliceRandom;
use solana_client::rpc_client::RpcClient;
#[allow(deprecated)]
use solana_sdk::{
    address_lookup_table::AddressLookupTableAccount,
    hash::Hash,
    instruction::{AccountMeta, Instruction},
    message::{v0, VersionedMessage},
    pubkey::Pubkey,
    signature::Keypair,
    signer::Signer,
    system_instruction,
    transaction::VersionedTransaction,
};
use std::collections::HashMap;
use std::str::FromStr;
use std::sync::{Arc, Mutex};

use crate::metis::{InstructionData, SwapInstructionsResponse};

/// Simple in-memory cache for Address Lookup Table accounts.
/// Thread-safe; fetches from RPC on miss and caches the result.
pub type AltLookup = Arc<Mutex<HashMap<Pubkey, Vec<Pubkey>>>>;

pub fn new_alt_lookup() -> AltLookup {
    Arc::new(Mutex::new(HashMap::new()))
}

/// Jito tip account addresses -- pick one at random for each bundle.
/// Per Jito docs: do NOT use ALTs for tip accounts.
const JITO_TIP_ACCOUNTS: &[&str] = &[
    "96gYZGLnJYVFmbjzopPSU6QiEV5fGqZNyN9nmNhvrZU5",
    "HFqU5x63VTqvQss8hp11i4wVV8bD44PvwucfZ2bU7gRe",
    "Cw8CFyM9FkoMi7K7Crf6HNQqf4uEMzpKw6QNghXLvLkY",
    "ADaUMid9yfUytqMBgopwjb2DTLSokTSzL1zt6iGPaS49",
    "DfXygSm4jCyNCybVYYK6DwvWqjKee8pbDmJGcLWNDXjh",
    "ADuUkR4vqLUMWXxW9gh6D6L8pMSawimctcNZ5pGwDcEt",
    "DttWaMuVvTiduZRnguLF7jNxTgiMBZ1hyAumKUiL2KRL",
    "3AVi9Tg9Uo68tJfuvoKvqKNWKkC5wPdSSdeBnizKZ6jT",
];

/// Convert a Metis instruction into a Solana SDK Instruction.
fn to_sdk_instruction(ix: &InstructionData) -> Result<Instruction> {
    let program_id = Pubkey::from_str(&ix.program_id)?;
    let accounts: Vec<AccountMeta> = ix
        .accounts
        .iter()
        .map(|a| {
            let pubkey = Pubkey::from_str(&a.pubkey).expect("invalid pubkey in instruction");
            if a.is_writable {
                AccountMeta::new(pubkey, a.is_signer)
            } else {
                AccountMeta::new_readonly(pubkey, a.is_signer)
            }
        })
        .collect();
    let data = base64::Engine::decode(
        &base64::engine::general_purpose::STANDARD,
        &ix.data,
    )
    .context("failed to decode instruction data")?;
    Ok(Instruction {
        program_id,
        accounts,
        data,
    })
}

/// Fetch and cache an Address Lookup Table account.
fn resolve_alt(
    pubkey: &Pubkey,
    cache: &AltLookup,
    rpc: &RpcClient,
) -> Result<AddressLookupTableAccount> {
    {
        let map = cache.lock().unwrap();
        if let Some(addresses) = map.get(pubkey) {
            return Ok(AddressLookupTableAccount {
                key: *pubkey,
                addresses: addresses.clone(),
            });
        }
    }
    let account = rpc
        .get_account(pubkey)
        .with_context(|| format!("fetch ALT account {pubkey}"))?;
    #[allow(deprecated)]
    let state =
        solana_sdk::address_lookup_table::state::AddressLookupTable::deserialize(&account.data)
            .with_context(|| format!("deserialize ALT {pubkey}"))?;
    let addresses: Vec<Pubkey> = state.addresses.iter().copied().collect();
    cache.lock().unwrap().insert(*pubkey, addresses.clone());
    Ok(AddressLookupTableAccount {
        key: *pubkey,
        addresses,
    })
}

/// Build a versioned transaction:
///
/// #1 - Compute Budget: SetComputeUnitLimit
/// #2 - optional Metis setup instructions
/// #3 - Jupiter Aggregator: route_v2 (entire circular arb)
/// #4 - optional Metis cleanup instruction
/// #5 - System Program: Transfer (Jito tip, MUST be last)
pub fn build_arb_transaction(
    swap_ixs: &SwapInstructionsResponse,
    payer: &Keypair,
    tip_lamports: u64,
    cu_limit: u32,
    recent_blockhash: Hash,
    alt_lookup: &AltLookup,
    rpc_client: &RpcClient,
) -> Result<VersionedTransaction> {
    let mut instructions: Vec<Instruction> = Vec::new();

    const SETUP_IX_CU_BUDGET: u32 = 15_000;
    const MAX_TX_CU_LIMIT: u32 = 1_400_000;
    let aux_ix_count = swap_ixs.setup_instructions.len() as u32
        + u32::from(swap_ixs.cleanup_instruction.is_some());
    let effective_cu_limit = cu_limit
        .saturating_add(aux_ix_count.saturating_mul(SETUP_IX_CU_BUDGET))
        .min(MAX_TX_CU_LIMIT);
    let cu_limit_ix = Instruction {
        program_id: Pubkey::from_str("ComputeBudget111111111111111111111111111111")?,
        accounts: vec![],
        data: {
            let mut data = vec![0x02];
            data.extend_from_slice(&effective_cu_limit.to_le_bytes());
            data
        },
    };
    instructions.push(cu_limit_ix);

    for ix in &swap_ixs.setup_instructions {
        instructions.push(to_sdk_instruction(ix)?);
    }

    instructions.push(to_sdk_instruction(&swap_ixs.swap_instruction)?);

    if let Some(ix) = &swap_ixs.cleanup_instruction {
        instructions.push(to_sdk_instruction(ix)?);
    }

    let mut alt_pubkeys: Vec<Pubkey> = Vec::new();
    for addr in &swap_ixs.address_lookup_table_addresses {
        let pk = Pubkey::from_str(addr)?;
        if !alt_pubkeys.contains(&pk) {
            alt_pubkeys.push(pk);
        }
    }

    let mut address_lookup_tables: Vec<AddressLookupTableAccount> = Vec::new();
    for pk in &alt_pubkeys {
        address_lookup_tables.push(resolve_alt(pk, alt_lookup, rpc_client)?);
    }

    let tip_candidates: Vec<Pubkey> = JITO_TIP_ACCOUNTS
        .iter()
        .filter_map(|addr| Pubkey::from_str(addr).ok())
        .filter(|tip| {
            !address_lookup_tables
                .iter()
                .any(|alt| alt.addresses.contains(tip))
        })
        .collect();
    let tip_account = {
        let mut rng = rand::thread_rng();
        *tip_candidates
            .choose(&mut rng)
            .context("all Jito tip accounts are present in route ALTs")?
    };
    #[allow(deprecated)]
    instructions.push(system_instruction::transfer(
        &payer.pubkey(),
        &tip_account,
        tip_lamports,
    ));

    let message = v0::Message::try_compile(
        &payer.pubkey(),
        &instructions,
        &address_lookup_tables,
        recent_blockhash,
    )
    .context("failed to compile v0 message")?;

    let versioned_message = VersionedMessage::V0(message);
    let tx = VersionedTransaction::try_new(versioned_message, &[payer])
        .context("failed to sign versioned transaction")?;

    Ok(tx)
}

/// Build the BisonFi single-pool first-test transaction.
///
/// Instruction order (token-ledger semantics REQUIRE this exact layout):
///   1. Metis computeBudget instructions (SetComputeUnitLimit / data-size limit).
///      If Metis returned none, a single SetComputeUnitLimit(`cu_limit`) is added.
///   2. Metis setup instructions (ATA creation etc.), if any.
///   3. Metis tokenLedgerInstruction  — snapshots the USDC balance.
///   4. BisonFi DFlow swap2 instruction (from pmm-sim) — deposits USDC.
///   5. Metis swap instruction (route_with_token_ledger) — consumes the delta.
///   6. Metis cleanup instruction, if any.
///   7. System Program transfer — Jito tip (MUST be last).
///
/// `bison_ix` is the instruction built and simulated by pmm-sim, already carrying
/// the real wallet's accounts. `extra_alts` lets the caller add the BisonFi leg's
/// own Address Lookup Table in addition to the Metis-provided ones.
pub fn build_bison_test_transaction(
    swap_ixs: &SwapInstructionsResponse,
    bison_ix: &InstructionData,
    payer: &Keypair,
    tip_account: &Pubkey,
    tip_lamports: u64,
    cu_limit: u32,
    recent_blockhash: Hash,
    alt_lookup: &AltLookup,
    rpc_client: &RpcClient,
    extra_alts: &[Pubkey],
) -> Result<VersionedTransaction> {
    let mut instructions: Vec<Instruction> = Vec::new();

    // 1. Compute budget.
    if swap_ixs.compute_budget_instructions.is_empty() {
        instructions.push(Instruction {
            program_id: Pubkey::from_str("ComputeBudget111111111111111111111111111111")?,
            accounts: vec![],
            data: {
                let mut data = vec![0x02];
                data.extend_from_slice(&cu_limit.to_le_bytes());
                data
            },
        });
    } else {
        for ix in &swap_ixs.compute_budget_instructions {
            instructions.push(to_sdk_instruction(ix)?);
        }
    }

    // 2. Setup instructions.
    for ix in &swap_ixs.setup_instructions {
        instructions.push(to_sdk_instruction(ix)?);
    }

    // 3. Token ledger — MUST come before BisonFi.
    let token_ledger = swap_ixs
        .token_ledger_instruction
        .as_ref()
        .context("missing tokenLedgerInstruction (useTokenLedger not honoured by Metis)")?;
    instructions.push(to_sdk_instruction(token_ledger)?);

    // 4. BisonFi leg.
    instructions.push(to_sdk_instruction(bison_ix)?);

    // 5. Jupiter route_with_token_ledger.
    instructions.push(to_sdk_instruction(&swap_ixs.swap_instruction)?);

    // 6. Cleanup.
    if let Some(ix) = &swap_ixs.cleanup_instruction {
        instructions.push(to_sdk_instruction(ix)?);
    }

    // 7. Jito tip (last).
    #[allow(deprecated)]
    instructions.push(system_instruction::transfer(
        &payer.pubkey(),
        tip_account,
        tip_lamports,
    ));

    // Resolve ALTs: Metis-provided + any extra (BisonFi leg) tables.
    let mut alt_pubkeys: Vec<Pubkey> = Vec::new();
    for addr in &swap_ixs.address_lookup_table_addresses {
        let pk = Pubkey::from_str(addr)?;
        if !alt_pubkeys.contains(&pk) {
            alt_pubkeys.push(pk);
        }
    }
    for pk in extra_alts {
        if !alt_pubkeys.contains(pk) {
            alt_pubkeys.push(*pk);
        }
    }

    let mut address_lookup_tables: Vec<AddressLookupTableAccount> = Vec::new();
    for pk in &alt_pubkeys {
        address_lookup_tables.push(resolve_alt(pk, alt_lookup, rpc_client)?);
    }

    let message = v0::Message::try_compile(
        &payer.pubkey(),
        &instructions,
        &address_lookup_tables,
        recent_blockhash,
    )
    .context("failed to compile v0 message")?;

    let tx = VersionedTransaction::try_new(VersionedMessage::V0(message), &[payer])
        .context("failed to sign versioned transaction")?;
    Ok(tx)
}

/// Pick a random Jito tip account that is not present in any of the given ALTs.
pub fn pick_tip_account(alts: &[AddressLookupTableAccount]) -> Result<Pubkey> {
    let tip_candidates: Vec<Pubkey> = JITO_TIP_ACCOUNTS
        .iter()
        .filter_map(|addr| Pubkey::from_str(addr).ok())
        .filter(|tip| !alts.iter().any(|alt| alt.addresses.contains(tip)))
        .collect();
    let mut rng = rand::thread_rng();
    tip_candidates
        .choose(&mut rng)
        .copied()
        .context("all Jito tip accounts are present in route ALTs")
}

/// Number of distinct accounts the transaction locks.
pub fn account_lock_count(tx: &VersionedTransaction) -> usize {
    match &tx.message {
        VersionedMessage::Legacy(m) => m.account_keys.len(),
        VersionedMessage::V0(m) => {
            m.account_keys.len()
                + m.address_table_lookups
                    .iter()
                    .map(|l| l.writable_indexes.len() + l.readonly_indexes.len())
                    .sum::<usize>()
        }
    }
}
