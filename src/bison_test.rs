//! BisonFi single-pool first-test flow.
//!
//! This module replaces the multi-token scanner while `config.bison_test.enabled`
//! is true. It runs ONE fixed opportunity, re-evaluated live on every Yellowstone
//! update of the configured BisonFi pool:
//!
//! ```text
//!   amount_in = 0.04 SOL (40_000_000 lamports), WSOL
//!     │
//!     ├─ pmm-sim: build + simulate BisonFi WSOL→USDC  ⇒ predicted_usdc, DFlow swap2 ix
//!     │
//!     ├─ Metis:   quote USDC→WSOL (amount = predicted_usdc, V2, slippage 0)
//!     │
//!     ├─ profit:  quote.outAmount >= amount_in + min_profit_lamports ?
//!     │
//!     ├─ Metis:   /swap-instructions (useTokenLedger=true) ⇒ tokenLedger + route_with_token_ledger
//!     │
//!     └─ bundle:  computeBudget · setup · tokenLedger · BisonFi · swap · cleanup · tip ⇒ Jito
//! ```
//!
//! Jupiter instructions come ENTIRELY from Metis (no RAM/offline route building).
//! Only the BisonFi leg is built locally (by pmm-sim), as a DFlow `swap2`.

use std::sync::Arc;
use std::time::{Duration, Instant};

use solana_sdk::signer::Signer;

use crate::arbitrage::CalcCtx;
use crate::bison_metrics::BisonMetrics;
use crate::config::BisonTestConfig;
use crate::metis::{InstructionData, SwapInstructionsResponse};
use crate::pmm_sim::{
    self, AccountCache, BuildBisonRequest, IpcInstructionOut, PmmSimEngine,
};
use crate::tokens::{USDC_MINT, WSOL_MINT};

/// Anchor `global:<name>` 8-byte discriminator, computed with solana's sha256.
fn anchor_discriminator(name: &str) -> [u8; 8] {
    let h = solana_sdk::hash::hash(format!("global:{name}").as_bytes());
    let mut d = [0u8; 8];
    d.copy_from_slice(&h.to_bytes()[..8]);
    d
}

/// True if the Metis swap instruction is a token-ledger route variant
/// (`route_with_token_ledger` or `shared_accounts_route_with_token_ledger`).
fn is_token_ledger_route(ix: &InstructionData) -> bool {
    let data = match base64::Engine::decode(
        &base64::engine::general_purpose::STANDARD,
        &ix.data,
    ) {
        Ok(d) => d,
        Err(_) => return false,
    };
    if data.len() < 8 {
        return false;
    }
    let disc = &data[..8];
    disc == anchor_discriminator("route_with_token_ledger")
        || disc == anchor_discriminator("shared_accounts_route_with_token_ledger")
}

/// Convert a pmm-sim-built instruction into the Metis instruction shape so it can
/// be spliced into the bundle with the existing transaction builder.
fn ipc_to_instruction_data(ix: &IpcInstructionOut) -> InstructionData {
    InstructionData {
        program_id: ix.program_id.clone(),
        accounts: ix
            .accounts
            .iter()
            .map(|a| crate::metis::AccountMeta {
                pubkey: a.pubkey.clone(),
                is_signer: a.is_signer,
                is_writable: a.is_writable,
            })
            .collect(),
        data: ix.data.clone(),
    }
}

static REQ_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// Build the live-update trigger so `main` can attach it to the Yellowstone
/// subscription BEFORE the subscription starts (otherwise early updates would be
/// missed). Returns the trigger and the shared notify to hand to `run_with_notify`.
pub fn make_trigger(
    cfg: &BisonTestConfig,
    metrics: Arc<BisonMetrics>,
) -> (pmm_sim::UpdateTrigger, Arc<tokio::sync::Notify>) {
    let watched: std::collections::HashSet<String> =
        [cfg.market.clone(), cfg.base_ta.clone(), cfg.quote_ta.clone()]
            .into_iter()
            .collect();
    let notify = Arc::new(tokio::sync::Notify::new());
    (
        pmm_sim::UpdateTrigger {
            keys: Arc::new(watched),
            notify: notify.clone(),
            metrics,
        },
        notify,
    )
}

/// Variant of `run` that uses an externally-created notify (shared with the
/// Yellowstone trigger) so no live update is lost during startup.
pub async fn run_with_notify(
    cfg: BisonTestConfig,
    ctx: Arc<CalcCtx>,
    engine: Arc<PmmSimEngine>,
    cache: AccountCache,
    min_profit_lamports: u64,
    notify: Arc<tokio::sync::Notify>,
    metrics: Arc<BisonMetrics>,
) {
    if cfg.market.is_empty() || cfg.base_ta.is_empty() || cfg.quote_ta.is_empty() {
        eprintln!("[bison_test] FATAL: market/base_ta/quote_ta must be set in [bison_test]");
        return;
    }
    engine.ensure_started().await;
    eprintln!(
        "[bison_test] ready | market={} amount_in={} tip={} fee={} min_profit={} required_out={}",
        cfg.market,
        cfg.amount_in_lamports,
        cfg.jito_tip_lamports,
        cfg.network_fee_lamports,
        min_profit_lamports,
        cfg.amount_in_lamports + min_profit_lamports,
    );

    spawn_metrics_reporter(metrics.clone(), cache.clone(), engine.clone(), &cfg);

    loop {
        run_once(&cfg, &ctx, &engine, &cache, min_profit_lamports, &metrics).await;
        // Re-evaluate the instant the pool changes (live, low-latency). The
        // fallback sleep is only a safety net so the test keeps running even if
        // the Yellowstone stream is quiet — it never adds latency to a real
        // update, which wakes the `notified()` arm immediately.
        tokio::select! {
            _ = notify.notified() => {}
            _ = tokio::time::sleep(Duration::from_millis(FALLBACK_TICK_MS)) => {}
        }
    }
}

/// Spawn the 30-second diagnostic report for the BisonFi test flow.
fn spawn_metrics_reporter(
    metrics: Arc<BisonMetrics>,
    cache: AccountCache,
    engine: Arc<PmmSimEngine>,
    cfg: &BisonTestConfig,
) {
    use std::sync::atomic::Ordering::Relaxed;
    let keys = vec![cfg.market.clone(), cfg.base_ta.clone(), cfg.quote_ta.clone()];
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(30));
        interval.tick().await;
        loop {
            interval.tick().await;

            let updates = metrics.grpc_pool_updates.swap(0, Relaxed);
            let total_updates = metrics.grpc_total_updates.swap(0, Relaxed);
            let iv_sum = metrics.grpc_interval_sum_ms.swap(0, Relaxed);
            let iv_n = metrics.grpc_interval_samples.swap(0, Relaxed);
            let reqs = metrics.build_requests.swap(0, Relaxed);
            let resps = metrics.build_responses.swap(0, Relaxed);
            let succ = metrics.build_success.swap(0, Relaxed);
            let rt_sum = metrics.build_resp_us_sum.swap(0, Relaxed);
            let rt_n = metrics.build_resp_samples.swap(0, Relaxed);
            let avg_iv = if iv_n > 0 { iv_sum / iv_n } else { 0 };
            let avg_rt = if rt_n > 0 { rt_sum / rt_n } else { 0 };

            // Live cache warmth for the pool accounts.
            let (found, missing, slot) = pmm_sim::collect_accounts_by_pubkey(&keys, &cache);
            let pmm_up = engine.is_running().await;
            metrics.set_pmm_up(pmm_up);

            eprintln!(
                "[bison30s] pmm_up={pmm_up} \
grpc_total_updates={total_updates} pool_grpc_updates={updates} avg_update_interval_ms={avg_iv} \
pool_accounts_in_cache={}/{} cache_slot={slot} missing=[{}] \
bot->pmm_requests={reqs} pmm->bot_responses={resps} build_success={succ} \
avg_pmm_resp_us={avg_rt}",
                found.len(),
                keys.len(),
                missing.join(","),
            );
        }
    });
}

/// Safety-net re-evaluation interval when no live pool update arrives.
const FALLBACK_TICK_MS: u64 = 2000;

async fn run_once(
    cfg: &BisonTestConfig,
    ctx: &Arc<CalcCtx>,
    engine: &Arc<PmmSimEngine>,
    cache: &AccountCache,
    min_profit_lamports: u64,
    metrics: &Arc<BisonMetrics>,
) {
    let started = Instant::now();
    let amount_in = cfg.amount_in_lamports;
    let required_out = amount_in.saturating_add(min_profit_lamports);
    // On-chain break-even floor embedded in the return swap (positive slippage kept).
    let on_chain_floor = amount_in
        .saturating_add(cfg.jito_tip_lamports)
        .saturating_add(cfg.network_fee_lamports);

    // ── 1. Gather fresh pool state from the cache ──────────────────────────────
    // The market account is authoritative: it stores its real vault token
    // accounts (base_ta@120, quote_ta@152). We DERIVE them from the market data
    // rather than trusting mix.json/config (whose vaults caused BisonFi 0x1a).
    let market_data = match pmm_sim::get_cached_data(cache, &cfg.market) {
        Some(d) => d,
        None => {
            eprintln!("[bison_test] skip: market state not warm yet ({})", cfg.market);
            return;
        }
    };
    let (base_ta, quote_ta) = match pmm_sim::parse_bisonfi_vaults(&market_data) {
        Some(v) => v,
        None => {
            eprintln!(
                "[bison_test] skip: cannot parse BisonFi market {} (len={}, not a POOLSTAT account?)",
                cfg.market, market_data.len()
            );
            return;
        }
    };
    if base_ta != cfg.base_ta || quote_ta != cfg.quote_ta {
        eprintln!(
            "[bison_test] note: using vaults derived from market data base_ta={base_ta} quote_ta={quote_ta} (config had base_ta={} quote_ta={})",
            cfg.base_ta, cfg.quote_ta
        );
    }

    let pool_keys = vec![cfg.market.clone(), base_ta.clone(), quote_ta.clone()];
    let (mut accounts, mut missing, mut slot) =
        pmm_sim::collect_accounts_by_pubkey(&pool_keys, cache);
    if !missing.is_empty() {
        // The derived vaults may not be in the cache yet — fetch them once from RPC.
        let rpc = ctx.rpc_client.clone();
        let cache_c = cache.clone();
        let to_fetch = missing.clone();
        let _ = tokio::task::spawn_blocking(move || {
            pmm_sim::bootstrap_from_rpc(&to_fetch, &cache_c, &rpc, to_fetch.len().max(1));
        })
        .await;
        let r = pmm_sim::collect_accounts_by_pubkey(&pool_keys, cache);
        accounts = r.0;
        missing = r.1;
        slot = r.2;
    }
    if !missing.is_empty() {
        eprintln!(
            "[bison_test] skip: pool state not warm (missing after RPC fetch=[{}])",
            missing.join(",")
        );
        return;
    }

    // ── 2. pmm-sim: build + simulate BisonFi WSOL→USDC ─────────────────────────
    let id = REQ_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let build_req = BuildBisonRequest {
        op: "build_bison",
        id,
        fee_payer: ctx.user_pubkey.clone(),
        market: cfg.market.clone(),
        src_mint: WSOL_MINT.to_string(),
        dst_mint: USDC_MINT.to_string(),
        amount_in,
        slot,
        accounts,
    };

    metrics.record_build_request();
    let build_started = Instant::now();
    let build = match engine.build_bison(&build_req).await {
        Some(b) => b,
        None => {
            metrics.set_pmm_up(false);
            eprintln!("[bison_test] skip: pmm-sim unavailable / timed out");
            return;
        }
    };
    let build_us = build_started.elapsed().as_micros() as u64;
    let build_ok = build.success && build.amount_out.map(|v| v > 0).unwrap_or(false);
    metrics.record_build_response(build_us, build_ok);
    if !build.success {
        eprintln!(
            "[bison_test] skip: BisonFi sim failed slot={slot} error={}",
            build.error.as_deref().unwrap_or("unknown")
        );
        return;
    }
    let predicted_usdc = match build.amount_out.filter(|&v| v > 0) {
        Some(v) => v,
        None => {
            eprintln!("[bison_test] skip: predicted_usdc_out is zero/none");
            return;
        }
    };

    // ── Price log (acceptance criteria) ────────────────────────────────────────
    eprintln!(
        "[bison_price] market={} market_base_ta={} market_quote_ta={} amount_in={} amount_out_usdc={} spoof=dflow slot={} source=live_cache cu={}",
        cfg.market, base_ta, quote_ta, amount_in, predicted_usdc, slot,
        build.compute_units.unwrap_or(0),
    );

    // Price-only mode: we have the BisonFi WSOL->USDC rate for caller=DFlow.
    // Do not call Metis, build a bundle, or send to Jito.
    if cfg.price_only {
        return;
    }

    let bison_ix_out = match build.instruction {
        Some(ix) => ix,
        None => {
            eprintln!("[bison_test] skip: pmm-sim returned no BisonFi instruction");
            return;
        }
    };

    // ── 3. Metis quote USDC→WSOL ───────────────────────────────────────────────
    let quote = match ctx
        .metis
        .get_quote(USDC_MINT, WSOL_MINT, predicted_usdc, false)
        .await
    {
        Ok(q) => q,
        Err(e) => {
            eprintln!("[bison_test] skip: Metis quote failed: {e}");
            return;
        }
    };
    let metis_in: u64 = quote.in_amount.parse().unwrap_or(0);
    let out_wsol: u64 = quote.out_amount.parse().unwrap_or(0);

    // ── 4. Profitability ───────────────────────────────────────────────────────
    let is_profitable = predicted_usdc > 0 && out_wsol >= required_out;

    if !is_profitable {
        eprintln!(
            "[bison_test] not_profitable | amount_in={amount_in} predicted_usdc={predicted_usdc} metis_in={metis_in} out_wsol={out_wsol} required_out={required_out} min_profit={min_profit_lamports} elapsed_us={}",
            started.elapsed().as_micros()
        );
        return;
    }

    // ── 5. Metis /swap-instructions with token ledger ──────────────────────────
    // Embed the on-chain floor (amount_in + tip + fee) as the route's minimum so
    // the tx lands at break-even-or-better and keeps any positive slippage.
    let mut floored_quote = quote.clone();
    floored_quote.out_amount = on_chain_floor.to_string();
    floored_quote.other_amount_threshold = on_chain_floor.to_string();

    let swap_ixs: SwapInstructionsResponse = match ctx
        .metis
        .get_swap_instructions_token_ledger(&ctx.user_pubkey, &floored_quote)
        .await
    {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[bison_test] abort: /swap-instructions failed: {e:?}");
            return;
        }
    };

    // ── 6. Verify token-ledger structure ───────────────────────────────────────
    let has_token_ledger = swap_ixs.token_ledger_instruction.is_some();
    let swap_is_token_ledger = is_token_ledger_route(&swap_ixs.swap_instruction);
    if !has_token_ledger || !swap_is_token_ledger {
        eprintln!(
            "[bison_test] abort: token-ledger not honoured (has_token_ledger={has_token_ledger} swap_is_token_ledger={swap_is_token_ledger}) — not sending"
        );
        return;
    }

    // ── 7. Build the bundle transaction ────────────────────────────────────────
    let bison_ix = ipc_to_instruction_data(&bison_ix_out);
    let recent_blockhash = ctx.blockhash_cache.get();
    // Only used if Metis returns no computeBudget instructions of its own.
    // Generous for the BisonFi + token-ledger return route to avoid CU reverts.
    let cu_limit = ctx.cu_limits.first().copied().unwrap_or(200_000).max(400_000);
    let tip_account = match crate::transaction::pick_tip_account(&[]) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("[bison_test] abort: no tip account: {e}");
            return;
        }
    };
    let extra_alts: Vec<solana_sdk::pubkey::Pubkey> = if cfg.alt.is_empty() {
        vec![]
    } else {
        match cfg.alt.parse() {
            Ok(pk) => vec![pk],
            Err(_) => {
                eprintln!("[bison_test] warning: invalid bison_test.alt, ignoring");
                vec![]
            }
        }
    };

    let tx = match crate::transaction::build_bison_test_transaction(
        &swap_ixs,
        &bison_ix,
        &ctx.trading_keypair,
        &tip_account,
        cfg.jito_tip_lamports,
        cu_limit,
        recent_blockhash,
        &ctx.alt_lookup,
        &ctx.rpc_client,
        &extra_alts,
    ) {
        Ok(tx) => tx,
        Err(e) => {
            eprintln!("[bison_test] abort: tx build failed: {e}");
            return;
        }
    };

    // Size / lock sanity checks.
    let lock_count = crate::transaction::account_lock_count(&tx);
    let tx_size = bincode::serialize(&tx).map(|b| b.len()).unwrap_or(usize::MAX);
    if lock_count > 64 {
        eprintln!("[bison_test] abort: account_locks={lock_count} > 64");
        return;
    }
    if tx_size > 1232 {
        eprintln!("[bison_test] abort: tx_size={tx_size} > 1232");
        return;
    }

    // ── 8. Full pre-send log (friend's required fields) ────────────────────────
    eprintln!(
        "[bison_test] SEND | amount_in_wsol={amount_in} predicted_usdc_out={predicted_usdc} metis_quote_in={metis_in} metis_quote_out={out_wsol} required_out={required_out} min_profit={min_profit_lamports} is_profitable=true has_token_ledger_instruction=true jupiter_swap_instruction_type=route_with_token_ledger jito_tip={} on_chain_floor={on_chain_floor} ix_order=[computeBudget,setup,tokenLedger,bisonFi,swap,cleanup,tip] tx_size={tx_size} account_locks={lock_count} fee_payer={} elapsed_us={}",
        cfg.jito_tip_lamports,
        ctx.trading_keypair.pubkey(),
        started.elapsed().as_micros(),
    );

    // ── 9. Send the bundle to Jito ─────────────────────────────────────────────
    let result = if let Some(grpc) = &ctx.jito_grpc {
        grpc.send_bundle(&tx).await
    } else {
        ctx.jito.send_bundle(&tx).await
    };
    match result {
        Ok(sig) => eprintln!("[bison_test] jito_sent bundle={sig}"),
        Err(e) => eprintln!("[bison_test] jito_send_failed: {e}"),
    }
}
