use anyhow::Result;
use futures::stream::{self, StreamExt};
use solana_client::rpc_client::RpcClient;
use solana_sdk::signature::Keypair;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::blockhash_cache::BlockhashCache;
use crate::config::Config;
use crate::jito::JitoClient;
use crate::jito_grpc::JitoGrpcClient;
use crate::metis::{MetisClient, QuoteResponse, SwapInstructionsResponse};
use crate::metrics::Metrics;
use crate::rate_limiter::RateLimiter;
use crate::sim;
use crate::token_metrics::TokenMetrics;
use crate::tokens::WSOL_MINT;
use crate::transaction::{self, AltLookup};

const LAMPORTS_PER_SOL: f64 = 1_000_000_000.0;
const JITO_TIP_LAMPORTS: u64 = 1_600;
const NETWORK_FEE_LAMPORTS: u64 = 5_000;
const RATE_RETRY_BACKOFF_MS: u64 = 20;

// ─── Route sig ───────────────────────────────────────────────────────────────

fn compute_route_sig(route_plan: &serde_json::Value) -> u128 {
    use std::collections::hash_map::DefaultHasher;
    let s = route_plan.to_string();
    let mut h1 = DefaultHasher::new();
    s.hash(&mut h1);
    let v1 = h1.finish();
    let mut h2 = DefaultHasher::new();
    v1.hash(&mut h2);
    ((v1 as u128) << 64) | (h2.finish() as u128)
}

// ─── Route helpers ───────────────────────────────────────────────────────────

fn route_labels(route_plan: &serde_json::Value) -> Vec<String> {
    route_plan
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|hop| hop.get("swapInfo"))
        .filter_map(|swap_info| swap_info.get("label"))
        .filter_map(|label| label.as_str())
        .map(str::to_string)
        .collect::<Vec<_>>()
}

fn route_labels_summary(route_plan: &serde_json::Value) -> String {
    let labels = route_labels(route_plan);
    serde_json::to_string(&labels).unwrap_or_else(|_| "[]".to_string())
}

/// Returns true if both quotes share at least one pool (round-trip always loses).
fn routes_share_pool(q1: &QuoteResponse, q2: &QuoteResponse) -> bool {
    let keys1: Vec<&str> = q1
        .route_plan
        .as_array()
        .map(|hops| {
            hops.iter()
                .filter_map(|h| h.get("swapInfo")?.get("ammKey")?.as_str())
                .collect()
        })
        .unwrap_or_default();
    if keys1.is_empty() {
        return false;
    }
    let arr2 = match q2.route_plan.as_array() {
        Some(a) => a,
        None => return false,
    };
    for hop in arr2 {
        if let Some(key) = hop
            .get("swapInfo")
            .and_then(|s| s.get("ammKey"))
            .and_then(|v| v.as_str())
        {
            if keys1.contains(&key) {
                return true;
            }
        }
    }
    false
}

fn lookup_cu_limit(hop_count: usize, cu_limits: &[u32]) -> u32 {
    if cu_limits.is_empty() {
        return 200_000;
    }
    let index = hop_count.saturating_sub(2);
    cu_limits[index.min(cu_limits.len() - 1)]
}

// ─── Stage 1 output ──────────────────────────────────────────────────────────

struct QuotePair {
    #[allow(dead_code)]
    token_mint: String,
    amount: u64,
    output_wsol: u64,
    net_profit: i64,
    quote1: QuoteResponse,
    quote2: QuoteResponse,
    hop_count: usize,
    only_direct: bool,
}

// ─── LIFO queue item ──────────────────────────────────────────────────────────

struct ReadyInstruction {
    swap_ixs: SwapInstructionsResponse,
    hop_count: usize,
    min_wsol_gain: u64,
    route_sig: u128,
    route_labels: String,
    arrived_at: Instant,
    waited_for_slot: bool,
}

#[derive(Clone)]
struct SwapCandidate {
    merged: QuoteResponse,
    amount: u64,
    output_wsol: u64,
    net_profit_estimate: i64,
    hop_count: usize,
    only_direct: bool,
    route_sig: u128,
    route_labels: String,
    min_wsol_gain: u64,
    created_at: Instant,
}

struct InflightSwapIx {
    current_profit: i64,
    pending: Option<SwapCandidate>,
}

pub struct SwapIxState {
    inflight: Mutex<HashMap<u128, InflightSwapIx>>,
    budget: Arc<Semaphore>,
    max_inflight: usize,
}

enum SwapIxStart {
    Start(OwnedSemaphorePermit),
    DroppedInflight,
    DroppedBudget { inflight: usize },
}

enum SwapIxDispatchOutcome {
    Sent,
    DroppedInflight,
    DroppedBudget,
}

impl SwapIxState {
    pub fn new(max_inflight: usize) -> Self {
        let max_inflight = max_inflight.max(1);
        Self {
            inflight: Mutex::new(HashMap::new()),
            budget: Arc::new(Semaphore::new(max_inflight)),
            max_inflight,
        }
    }

    fn try_start(&self, candidate: &SwapCandidate) -> SwapIxStart {
        let mut inflight = self.inflight.lock().unwrap();
        if let Some(existing) = inflight.get_mut(&candidate.route_sig) {
            if candidate.net_profit_estimate > existing.current_profit {
                let replace_pending = existing
                    .pending
                    .as_ref()
                    .map(|pending| candidate.net_profit_estimate > pending.net_profit_estimate)
                    .unwrap_or(true);
                if replace_pending {
                    existing.pending = Some(candidate.clone());
                }
            }
            return SwapIxStart::DroppedInflight;
        }
        match self.budget.clone().try_acquire_owned() {
            Ok(permit) => {
                inflight.insert(
                    candidate.route_sig,
                    InflightSwapIx {
                        current_profit: candidate.net_profit_estimate,
                        pending: None,
                    },
                );
                SwapIxStart::Start(permit)
            }
            Err(_) => SwapIxStart::DroppedBudget {
                inflight: inflight.len(),
            },
        }
    }

    fn complete(&self, route_sig: u128) -> Option<SwapCandidate> {
        let mut inflight = self.inflight.lock().unwrap();
        let Some(mut existing) = inflight.remove(&route_sig) else {
            return None;
        };
        existing.pending.take()
    }

    fn max_inflight(&self) -> usize {
        self.max_inflight
    }

    fn inflight_len(&self) -> usize {
        self.inflight.lock().unwrap().len()
    }
}

// ─── Shared context ───────────────────────────────────────────────────────────

pub struct CalcCtx {
    pub metis: Arc<MetisClient>,
    pub blockhash_cache: Arc<BlockhashCache>,
    pub trading_keypair: Arc<Keypair>,
    pub rpc_client: Arc<RpcClient>,
    pub alt_lookup: AltLookup,
    pub jito: Arc<JitoClient>,
    pub jito_grpc: Option<Arc<JitoGrpcClient>>,
    pub jito_limiter: Arc<Mutex<RateLimiter>>,
    pub jito_grpc_limiter: Option<Arc<Mutex<RateLimiter>>>,
    pub cu_limits: Vec<u32>,
    pub user_pubkey: String,
    pub swap_ix_state: Arc<SwapIxState>,
    /// All candidates with successful Metis instructions are pushed here.
    pub sim_queue: Arc<sim::SimQueue>,
}

// ─── Pipeline handle ──────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct Pipeline {
    lifo: Arc<Mutex<Vec<ReadyInstruction>>>,
    lifo_sem: Arc<tokio::sync::Semaphore>,
}

impl Pipeline {
    /// Push a ready instruction into the LIFO queue.
    fn push(&self, item: ReadyInstruction, metrics: &Metrics) {
        let route_sig = item.route_sig;
        let route_labels = item.route_labels.clone();
        self.lifo.lock().unwrap().push(item);
        metrics.queue_in.fetch_add(1, Ordering::Relaxed);
        let depth = metrics.queue_depth.fetch_add(1, Ordering::Relaxed) + 1;
        eprintln!(
            "[queue_push] route_sig={route_sig:032x} route_labels={route_labels} queue_depth={depth}"
        );
        self.lifo_sem.add_permits(1);
    }
}

// ─── Stage 1: Quote scanner ───────────────────────────────────────────────────

async fn quote_check(
    metis: &MetisClient,
    token_mint: &str,
    amount: u64,
    only_direct: bool,
    min_profit_lamports: u64,
    metrics: &Metrics,
    token_metrics: &TokenMetrics,
) -> Option<QuotePair> {
    metrics.metis_req_sent.fetch_add(2, Ordering::Relaxed);
    let ts = token_metrics.get(token_mint);
    if let Some(ts) = ts {
        ts.q_sent.fetch_add(2, Ordering::Relaxed);
    }

    let quote1 = match metis.get_quote(WSOL_MINT, token_mint, amount, only_direct).await {
        Ok(q) => q,
        Err(_) => {
            if let Some(ts) = ts {
                ts.route_fail.fetch_add(1, Ordering::Relaxed);
            }
            return None;
        }
    };

    let token_amount: u64 = match quote1.out_amount.parse::<u64>().ok().filter(|&v| v > 0) {
        Some(v) => v,
        None => {
            if let Some(ts) = ts {
                ts.route_fail.fetch_add(1, Ordering::Relaxed);
            }
            return None;
        }
    };

    let quote2 = match metis.get_quote(token_mint, WSOL_MINT, token_amount, only_direct).await {
        Ok(q) => q,
        Err(_) => {
            if let Some(ts) = ts {
                ts.route_fail.fetch_add(1, Ordering::Relaxed);
            }
            return None;
        }
    };

    let output_wsol: u64 = quote2.out_amount.parse().unwrap_or(0);
    metrics.metis_resp_total.fetch_add(1, Ordering::Relaxed);
    if let Some(ts) = ts {
        ts.route_ok.fetch_add(1, Ordering::Relaxed);
    }

    let on_chain_floor = amount
        .saturating_add(JITO_TIP_LAMPORTS)
        .saturating_add(NETWORK_FEE_LAMPORTS);
    if output_wsol <= on_chain_floor.saturating_add(min_profit_lamports) {
        if let Some(ts) = ts {
            ts.not_profitable.fetch_add(1, Ordering::Relaxed);
        }
        return None;
    }

    if routes_share_pool(&quote1, &quote2) {
        metrics.dropped_same_pool.fetch_add(1, Ordering::Relaxed);
        if let Some(ts) = ts {
            ts.not_profitable.fetch_add(1, Ordering::Relaxed);
        }
        return None;
    }

    if let Some(ts) = ts {
        ts.profitable.fetch_add(1, Ordering::Relaxed);
    }

    let hop_count = {
        let n1 = quote1.route_plan.as_array().map(|a| a.len()).unwrap_or(1);
        let n2 = quote2.route_plan.as_array().map(|a| a.len()).unwrap_or(1);
        n1 + n2
    };

    metrics.metis_resp_ok.fetch_add(1, Ordering::Relaxed);

    let on_chain_floor = amount
        .saturating_add(JITO_TIP_LAMPORTS)
        .saturating_add(NETWORK_FEE_LAMPORTS);
    let net_profit = output_wsol as i64 - on_chain_floor as i64;
    Some(QuotePair {
        token_mint: token_mint.to_string(),
        amount,
        output_wsol,
        net_profit,
        quote1,
        quote2,
        hop_count,
        only_direct,
    })
}

// ─── Stage 3: Jito workers ────────────────────────────────────────────────────

pub fn spawn_workers(
    ctx: Arc<CalcCtx>,
    metrics: Arc<Metrics>,
    worker_count: usize,
    queue_max_age_ms: u64,
) -> Pipeline {
    let lifo: Arc<Mutex<Vec<ReadyInstruction>>> = Arc::new(Mutex::new(Vec::new()));
    let lifo_sem = Arc::new(tokio::sync::Semaphore::new(0));

    for _ in 0..worker_count {
        let lifo_c = lifo.clone();
        let sem_c = lifo_sem.clone();
        let ctx_c = ctx.clone();
        let met_c = metrics.clone();
        tokio::spawn(async move {
            loop {
                sem_c.acquire().await.unwrap().forget();

                let item = lifo_c.lock().unwrap().pop();
                let mut item = match item {
                    Some(i) => {
                        met_c.queue_depth.fetch_sub(1, Ordering::Relaxed);
                        i
                    }
                    None => continue,
                };

                if item.arrived_at.elapsed().as_millis() as u64 > queue_max_age_ms {
                    met_c.dropped_stale.fetch_add(1, Ordering::Relaxed);
                    continue;
                }

                let cu_limit = lookup_cu_limit(item.hop_count, &ctx_c.cu_limits);
                let recent_blockhash = ctx_c.blockhash_cache.get();
                let keypair = ctx_c.trading_keypair.clone();
                let alt_lookup = ctx_c.alt_lookup.clone();
                let rpc = ctx_c.rpc_client.clone();
                let min_wsol_gain = item.min_wsol_gain;
                let route_sig = item.route_sig;
                let route_labels = item.route_labels.clone();
                let swap_ixs = item.swap_ixs.clone();

                let tip_lamports = min_wsol_gain.max(JITO_TIP_LAMPORTS);

                let tx = match transaction::build_arb_transaction(
                    &swap_ixs,
                    &keypair,
                    tip_lamports,
                    cu_limit,
                    recent_blockhash,
                    &alt_lookup,
                    &rpc,
                ) {
                    Ok(tx) => tx,
                    Err(e) => {
                        met_c.tx_build_failed.fetch_add(1, Ordering::Relaxed);
                        eprintln!("[tx_build_failed] route_sig={route_sig:032x} error={e}");
                        continue;
                    }
                };

                if transaction::account_lock_count(&tx) > 64 {
                    met_c.dropped_account_locks.fetch_add(1, Ordering::Relaxed);
                    continue;
                }

                match bincode::serialize(&tx) {
                    Ok(bytes) if bytes.len() > 1232 => {
                        met_c.tx_too_large.fetch_add(1, Ordering::Relaxed);
                        continue;
                    }
                    Err(_) => {
                        met_c.tx_build_failed.fetch_add(1, Ordering::Relaxed);
                        continue;
                    }
                    Ok(_) => {}
                }

                met_c.calc_done.fetch_add(1, Ordering::Relaxed);

                let use_grpc = if ctx_c.jito_limiter.lock().unwrap().try_acquire() {
                    false
                } else if ctx_c
                    .jito_grpc_limiter
                    .as_ref()
                    .map(|gl| gl.lock().unwrap().try_acquire())
                    .unwrap_or(false)
                {
                    true
                } else {
                    if !item.waited_for_slot {
                        item.waited_for_slot = true;
                        met_c.rate_requeued.fetch_add(1, Ordering::Relaxed);
                    }
                    lifo_c.lock().unwrap().push(item);
                    met_c.queue_depth.fetch_add(1, Ordering::Relaxed);
                    sem_c.add_permits(1);
                    tokio::time::sleep(std::time::Duration::from_millis(RATE_RETRY_BACKOFF_MS))
                        .await;
                    continue;
                };

                let result = if use_grpc {
                    match &ctx_c.jito_grpc {
                        Some(grpc) => grpc.send_bundle(&tx).await,
                        None => ctx_c.jito.send_bundle(&tx).await,
                    }
                } else {
                    ctx_c.jito.send_bundle(&tx).await
                };

                match result {
                    Ok(_) => {
                        met_c.jito_sent.fetch_add(1, Ordering::Relaxed);
                        eprintln!(
                            "[jito_sent] route_sig={route_sig:032x} route_labels={route_labels}"
                        );
                    }
                    Err(e) => {
                        met_c.jito_send_failed.fetch_add(1, Ordering::Relaxed);
                        eprintln!("[jito_send_failed] route_sig={route_sig:032x} error={e}");
                    }
                }
            }
        });
    }

    Pipeline { lifo, lifo_sem }
}

// ─── Stage 2: Simulation workers ─────────────────────────────────────────────

/// Spawn simulation workers that pop from sim_queue, classify venues,
/// optionally run LiteSVM simulation via the pmm-sim subprocess, then forward
/// to the Jito LIFO queue.
pub fn spawn_sim_workers(
    sim_queue: Arc<sim::SimQueue>,
    pipeline: Pipeline,
    metrics: Arc<Metrics>,
    worker_count: usize,
    queue_max_age_ms: u64,
    engine: Arc<crate::pmm_sim::PmmSimEngine>,
    cache: crate::pmm_sim::AccountCache,
    fee_payer: String,
    simulation_gate: bool,
    min_profit_after_sim_lamports: i64,
) {
    for _ in 0..worker_count {
        let sim_q = sim_queue.clone();
        let pipe = pipeline.clone();
        let met = metrics.clone();
        let eng = engine.clone();
        let acct_cache = cache.clone();
        let fp = fee_payer.clone();
        tokio::spawn(async move {
            loop {
                let req = sim_q.pop().await;

                if req.arrived_at.elapsed().as_millis() as u64 > queue_max_age_ms {
                    eprintln!(
                        "[sim_stale] route_sig={:032x} labels={}",
                        req.route_sig, req.route_labels
                    );
                    met.sim_stale.fetch_add(1, Ordering::Relaxed);
                    continue;
                }

                // Classify venues and run pmm-sim LiteSVM simulation for PropAMM routes.
                let result = sim::simulate(&req, &eng, &acct_cache, &fp).await;

                met.sim_classified.fetch_add(1, Ordering::Relaxed);

                if !result.unsupported_venues.is_empty() {
                    met.sim_unsupported.fetch_add(1, Ordering::Relaxed);
                    eprintln!(
                        "[sim_unsupported_venue] route_sig={:032x} venues=[{}]",
                        result.route_sig,
                        result.unsupported_venues.join(",")
                    );
                }

                eprintln!(
                    "[sim_classify] route_sig={:032x} sim_kind={} hops=[{}] elapsed_us={}",
                    result.route_sig,
                    result.sim_kind,
                    result.logs.join(","),
                    result.elapsed_us,
                );

                // Gate: if simulation_gate is enabled, only forward to Jito when sim passed.
                if simulation_gate {
                    let gate_ok = result.executed
                        && result.success
                        && result.simulated_profit.unwrap_or(i64::MIN) >= min_profit_after_sim_lamports;
                    if !gate_ok {
                        let reason = if !result.executed { "sim_not_executed" }
                                     else if !result.success { "sim_failed" }
                                     else { "insufficient_profit" };
                        eprintln!(
                            "[sim_gate_drop] route_sig={:032x} reason={reason} labels={}",
                            result.route_sig, result.route_labels,
                        );
                        met.sim_failed.fetch_add(1, Ordering::Relaxed);
                        continue;
                    }
                    eprintln!(
                        "[sim_gate] route_sig={:032x} action=jito_send profit={} labels={}",
                        result.route_sig,
                        result.simulated_profit.unwrap_or(0),
                        result.route_labels,
                    );
                }

                pipe.push(
                    ReadyInstruction {
                        swap_ixs: req.instructions,
                        hop_count: req.hop_count,
                        min_wsol_gain: req.min_wsol_gain,
                        route_sig: req.route_sig,
                        route_labels: req.route_labels,
                        arrived_at: req.arrived_at,
                        waited_for_slot: false,
                    },
                    &met,
                );
            }
        });
    }
}

// ─── Stage 1.5: Metis instruction fetch ──────────────────────────────────────

/// Dispatch a candidate to fetch /swap-instructions from Metis.
/// On success the SimulationRequest is pushed to ctx.sim_queue.
/// No ranking is applied — ALL profitable candidates are dispatched.
fn spawn_fresh_metis_candidate(
    candidate: SwapCandidate,
    ctx: Arc<CalcCtx>,
    metrics: Arc<Metrics>,
    timeout_ms: u64,
) -> SwapIxDispatchOutcome {
    match ctx.swap_ix_state.try_start(&candidate) {
        SwapIxStart::Start(permit) => {
            metrics.metis_req_sent.fetch_add(1, Ordering::Relaxed);
            metrics.swap_ix_sent_total.fetch_add(1, Ordering::Relaxed);
            eprintln!(
                "[fresh_metis_budget] inflight={} max_inflight={} sent=1",
                ctx.swap_ix_state.inflight_len(),
                ctx.swap_ix_state.max_inflight()
            );
            tokio::spawn(async move {
                let route_sig = candidate.route_sig;
                let candidate_age_ms = candidate.created_at.elapsed().as_millis();
                let t = Instant::now();
                eprintln!(
                    "[fresh_metis_start] route_sig={route_sig:032x} route_labels={} amount={} output_wsol={} net_profit_estimate={} only_direct={} candidate_age_ms={candidate_age_ms}",
                    candidate.route_labels,
                    candidate.amount,
                    candidate.output_wsol,
                    candidate.net_profit_estimate,
                    candidate.only_direct,
                );
                let result = ctx
                    .metis
                    .get_swap_instructions_with_timeout(
                        &ctx.user_pubkey,
                        &candidate.merged,
                        timeout_ms,
                    )
                    .await;
                let fetch_ms = t.elapsed().as_millis() as u64;

                let swap_ixs = match result {
                    Ok(s) => s,
                    Err(e) => {
                        let (kind, error) = match e {
                            crate::metis::SwapIxError::Timeout => {
                                ("timeout", "timeout".to_string())
                            }
                            crate::metis::SwapIxError::Http(status) => {
                                ("http", format!("http_status_{status}"))
                            }
                            crate::metis::SwapIxError::Network => {
                                ("network", "network".to_string())
                            }
                            crate::metis::SwapIxError::Parse => ("parse", "parse".to_string()),
                        };
                        eprintln!(
                            "[fresh_metis_err] route_sig={route_sig:032x} elapsed_ms={fetch_ms} kind={kind} error={error}"
                        );
                        metrics.swap_ix_failed.fetch_add(1, Ordering::Relaxed);
                        match e {
                            crate::metis::SwapIxError::Timeout => {
                                metrics.swap_ix_timeout.fetch_add(1, Ordering::Relaxed)
                            }
                            crate::metis::SwapIxError::Http(_) => {
                                metrics.swap_ix_http.fetch_add(1, Ordering::Relaxed)
                            }
                            crate::metis::SwapIxError::Network => {
                                metrics.swap_ix_network.fetch_add(1, Ordering::Relaxed)
                            }
                            crate::metis::SwapIxError::Parse => {
                                metrics.swap_ix_parse.fetch_add(1, Ordering::Relaxed)
                            }
                        };
                        drop(permit);
                        if let Some(pending) = ctx.swap_ix_state.complete(route_sig) {
                            spawn_fresh_metis_candidate(
                                pending,
                                ctx.clone(),
                                metrics.clone(),
                                timeout_ms,
                            );
                        }
                        return;
                    }
                };

                eprintln!(
                    "[fresh_metis_ok] route_sig={route_sig:032x} elapsed_ms={fetch_ms} setup_ix={} swap_ix=1 cleanup_ix={} alts={} action=sim_queue",
                    swap_ixs.setup_instructions.len(),
                    if swap_ixs.cleanup_instruction.is_some() { 1 } else { 0 },
                    swap_ixs.address_lookup_table_addresses.len()
                );
                metrics.metis_fetch_ms_total.fetch_add(fetch_ms, Ordering::Relaxed);
                metrics.metis_fetch_samples.fetch_add(1, Ordering::Relaxed);
                metrics.swap_ix_ok.fetch_add(1, Ordering::Relaxed);

                // Push to simulation queue — not directly to Jito.
                ctx.sim_queue.push(sim::SimulationRequest {
                    route_sig,
                    route_labels: candidate.route_labels.clone(),
                    hop_count: candidate.hop_count,
                    min_wsol_gain: candidate.min_wsol_gain,
                    net_profit_estimate: candidate.net_profit_estimate,
                    instructions: swap_ixs,
                    merged_quote: candidate.merged.clone(),
                    arrived_at: Instant::now(),
                });
                metrics.sim_queued.fetch_add(1, Ordering::Relaxed);

                drop(permit);
                if let Some(pending) = ctx.swap_ix_state.complete(route_sig) {
                    spawn_fresh_metis_candidate(
                        pending,
                        ctx.clone(),
                        metrics.clone(),
                        timeout_ms,
                    );
                }
            });
            SwapIxDispatchOutcome::Sent
        }
        SwapIxStart::DroppedInflight => {
            metrics
                .candidate_dropped_inflight_total
                .fetch_add(1, Ordering::Relaxed);
            eprintln!(
                "[fresh_metis_budget] inflight={} max_inflight={} dropped=inflight route_sig={:032x}",
                ctx.swap_ix_state.inflight_len(),
                ctx.swap_ix_state.max_inflight(),
                candidate.route_sig
            );
            SwapIxDispatchOutcome::DroppedInflight
        }
        SwapIxStart::DroppedBudget { inflight } => {
            metrics
                .swap_ix_budget_dropped_total
                .fetch_add(1, Ordering::Relaxed);
            eprintln!(
                "[fresh_metis_budget] inflight={inflight} max_inflight={} dropped=budget route_sig={:032x}",
                ctx.swap_ix_state.max_inflight(),
                candidate.route_sig
            );
            SwapIxDispatchOutcome::DroppedBudget
        }
    }
}

/// Dispatch ALL candidates in the batch to Metis instruction fetch.
/// No ranking or coalescing is applied — every profitable candidate that
/// passes the same-pool and merge checks is sent to /swap-instructions.
fn flush_candidate_batch(
    candidates: &mut Vec<SwapCandidate>,
    config: &Config,
    ctx: &Arc<CalcCtx>,
    metrics: &Arc<Metrics>,
) {
    if candidates.is_empty() {
        return;
    }

    let timeout_ms = config.performance.swap_instructions_timeout_ms;
    let total = candidates.len();
    let mut dispatched = 0usize;
    let mut dropped_by_inflight = 0usize;
    let mut dropped_by_budget = 0usize;

    for candidate in candidates.drain(..) {
        match spawn_fresh_metis_candidate(candidate, ctx.clone(), metrics.clone(), timeout_ms) {
            SwapIxDispatchOutcome::Sent => dispatched += 1,
            SwapIxDispatchOutcome::DroppedInflight => dropped_by_inflight += 1,
            SwapIxDispatchOutcome::DroppedBudget => dropped_by_budget += 1,
        }
    }

    eprintln!(
        "[flush_batch] total={total} dispatched={dispatched} drop_inflight={dropped_by_inflight} drop_budget={dropped_by_budget}"
    );
}

// ─── Main scan entry ─────────────────────────────────────────────────────────

pub async fn scan_all_tokens(
    token_mints: &[String],
    config: &Config,
    ctx: &Arc<CalcCtx>,
    metrics: &Arc<Metrics>,
    token_metrics: &Arc<TokenMetrics>,
) -> Result<()> {
    let min_lamports = (config.trading.min_amount_sol * LAMPORTS_PER_SOL) as u64;
    let max_lamports = (config.trading.max_amount_sol * LAMPORTS_PER_SOL) as u64;
    let step_lamports = (config.trading.step_sol * LAMPORTS_PER_SOL) as u64;
    let min_profit_lamports = config
        .trading
        .min_profit_lamports
        .saturating_add(JITO_TIP_LAMPORTS)
        .saturating_add(NETWORK_FEE_LAMPORTS)
        .saturating_add(config.performance.metis_latency_margin_lamports);

    let all_pairs: Vec<(u64, String, bool)> = {
        let mut pairs = Vec::new();
        let mut amount = min_lamports;
        while amount <= max_lamports {
            for token_mint in token_mints {
                pairs.push((amount, token_mint.clone(), false));
                pairs.push((amount, token_mint.clone(), true));
            }
            amount += step_lamports;
        }
        pairs
    };
    let max_concurrent = config
        .performance
        .max_concurrent_quotes
        .max(1)
        .min(all_pairs.len().max(1));

    let metis_ref: &MetisClient = &ctx.metis;
    let met_ref: &Metrics = metrics;
    let tok_met_ref: &TokenMetrics = token_metrics;

    let mut opps = stream::iter(all_pairs)
        .map(move |(amt, tok, direct)| async move {
            quote_check(
                metis_ref, &tok, amt, direct, min_profit_lamports, met_ref, tok_met_ref,
            )
            .await
        })
        .buffer_unordered(max_concurrent);

    let mut candidate_batch = Vec::new();
    let mut candidate_batch_started = Instant::now();
    let candidate_batch_window =
        Duration::from_millis(config.performance.candidate_batch_window_ms.max(1));

    while let Some(result) = opps.next().await {
        let pair = match result {
            Some(p) => p,
            None => continue,
        };

        // Direct routes must be exactly 2-hop.
        if pair.only_direct && pair.hop_count != 2 {
            metrics.dropped_multi_hop.fetch_add(1, Ordering::Relaxed);
            continue;
        }

        let on_chain_floor = pair.amount + JITO_TIP_LAMPORTS + NETWORK_FEE_LAMPORTS;

        let merged = match MetisClient::merge_quotes(&pair.quote1, &pair.quote2, on_chain_floor) {
            Ok(m) => m,
            Err(_) => {
                metrics.dropped_merge_fail.fetch_add(1, Ordering::Relaxed);
                continue;
            }
        };

        let hop_count = pair.hop_count;
        let amount = pair.amount;
        let sig = compute_route_sig(&merged.route_plan);
        let min_wsol_gain = on_chain_floor.saturating_sub(amount);
        let route_labels = route_labels_summary(&merged.route_plan);

        if candidate_batch.is_empty() {
            candidate_batch_started = Instant::now();
        }
        candidate_batch.push(SwapCandidate {
            merged,
            amount,
            output_wsol: pair.output_wsol,
            net_profit_estimate: pair.net_profit,
            hop_count,
            only_direct: pair.only_direct,
            route_sig: sig,
            route_labels,
            min_wsol_gain,
            created_at: Instant::now(),
        });
        metrics
            .candidate_profitable_total
            .fetch_add(1, Ordering::Relaxed);

        if candidate_batch_started.elapsed() >= candidate_batch_window {
            flush_candidate_batch(&mut candidate_batch, config, ctx, metrics);
            candidate_batch_started = Instant::now();
        }
    }

    flush_candidate_batch(&mut candidate_batch, config, ctx, metrics);

    Ok(())
}
