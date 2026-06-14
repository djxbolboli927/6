use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokio::sync::Semaphore;

use crate::metis::{QuoteResponse, SwapInstructionsResponse};

// ─── Venue classification ─────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SimulationVenueKind {
    /// Proprietary AMMs simulated via LiteSVM (pmm-sim approach).
    PropAmmPmmSim,
    /// Standard AMMs simulated via manual mathematical adapters.
    ManualAmmAdapter,
    /// Venue not yet classified; simulation not possible.
    Unsupported,
}

impl SimulationVenueKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::PropAmmPmmSim => "prop_amm_pmm_sim",
            Self::ManualAmmAdapter => "manual_amm_adapter",
            Self::Unsupported => "unsupported",
        }
    }
}

fn normalize_label(label: &str) -> String {
    label
        .to_lowercase()
        .replace([' ', '_', '-', '.', '/'], "")
}

/// Classify a single venue label returned by Metis routePlan.
pub fn classify_venue(label: &str) -> SimulationVenueKind {
    match normalize_label(label).as_str() {
        // ── Prop AMMs (LiteSVM / pmm-sim) ─────────────────────────────────────
        "bisonfi"
        | "humidifi" | "humidifiv1" | "humidifiv2" | "humidifiv3"
        | "solfi" | "solfiv2"
        | "obricv2"
        | "zerofi"
        | "tesserav"
        | "goonfi" | "gonfi" | "goonfiv2" => SimulationVenueKind::PropAmmPmmSim,

        // ── Standard AMMs (manual adapters) ───────────────────────────────────
        "raydiumammv4" | "raydiumamm" | "raydium"
        | "raydiumcpmm"
        | "raydiumclmm" | "raydiumconcentratedliquidity"
        | "orca" | "orcav2" | "orcawhirlpool" | "whirlpool"
        | "meteora" | "meteoradlmm" | "meteoradamm" | "meteoradammv2"
        | "meteorapools" | "meteoraammv2"
        | "pancakeswap" | "pancake"
        | "invariant"
        | "manifest"
        | "1dex"
        | "alphaq" => SimulationVenueKind::ManualAmmAdapter,

        _ => SimulationVenueKind::Unsupported,
    }
}

/// Classify every swap hop in a merged routePlan.
/// Returns (label, kind) for each hop.
pub fn classify_route(route_plan: &serde_json::Value) -> Vec<(String, SimulationVenueKind)> {
    route_plan
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|hop| {
            let label = hop.get("swapInfo")?.get("label")?.as_str()?.to_string();
            let kind = classify_venue(&label);
            Some((label, kind))
        })
        .collect()
}

// ─── Request / Result ─────────────────────────────────────────────────────────

pub struct SimulationRequest {
    pub route_sig: u128,
    pub route_labels: String,
    pub hop_count: usize,
    pub min_wsol_gain: u64,
    #[allow(dead_code)]
    pub net_profit_estimate: i64,
    /// Swap instructions returned by Metis — the ONLY valid input for simulation.
    pub instructions: SwapInstructionsResponse,
    /// Merged circular quote used to call /swap-instructions.
    pub merged_quote: QuoteResponse,
    /// When this request entered the simulation queue.
    pub arrived_at: Instant,
}

// Phase 2+ fields are defined now for interface stability but not yet consumed.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct SimulationResult {
    pub route_sig: u128,
    pub route_labels: String,
    /// true if simulation ran successfully for all hops.
    pub success: bool,
    /// true if the simulation was actually executed (not just classified).
    pub executed: bool,
    /// Phase 2: actual output amount from LiteSVM execution.
    pub final_out_amount: Option<u64>,
    /// Phase 2: profit as determined by simulation.
    pub simulated_profit: Option<i64>,
    /// Phase 2: simulated_profit − metis_estimated_profit.
    pub difference_vs_metis_quote: Option<i64>,
    /// Phase 2: compute units consumed by the simulated transaction.
    pub consumed_compute_units: Option<u64>,
    /// Phase 2: index of the first failing instruction (0-based).
    pub failed_instruction_index: Option<usize>,
    /// Per-hop venue classification or execution log lines.
    pub logs: Vec<String>,
    /// Venue labels for which no simulation path exists.
    pub unsupported_venues: Vec<String>,
    /// "prop_amm_pmm_sim" | "manual_amm_adapter" | "mixed" | "unsupported"
    pub sim_kind: String,
    pub elapsed_us: u64,
}

// ─── Simulation queue ─────────────────────────────────────────────────────────

/// FIFO queue between the Metis instruction stage and the simulation workers.
pub struct SimQueue {
    inner: Arc<Mutex<VecDeque<SimulationRequest>>>,
    sem: Arc<Semaphore>,
}

impl SimQueue {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            inner: Arc::new(Mutex::new(VecDeque::new())),
            sem: Arc::new(Semaphore::new(0)),
        })
    }

    pub fn push(&self, req: SimulationRequest) {
        self.inner.lock().unwrap().push_back(req);
        self.sem.add_permits(1);
    }

    pub async fn pop(&self) -> SimulationRequest {
        loop {
            self.sem.acquire().await.unwrap().forget();
            if let Some(req) = self.inner.lock().unwrap().pop_front() {
                return req;
            }
        }
    }

    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().len()
    }
}

// ─── Phase 1: venue-classification stub ──────────────────────────────────────
//
// The function below is Phase 1: it classifies venues and returns structured
// metadata but does NOT execute a simulation.
//
// TODO Phase 3: for ManualAmmAdapter venues, implement per-AMM mathematical
//   swap adapters (constant-product, concentrated-liquidity, DLMM, etc.).

pub fn simulate_stub(req: &SimulationRequest) -> SimulationResult {
    let t = std::time::Instant::now();
    let venues = classify_route(&req.merged_quote.route_plan);

    let unsupported: Vec<String> = venues
        .iter()
        .filter(|(_, k)| *k == SimulationVenueKind::Unsupported)
        .map(|(l, _)| l.clone())
        .collect();

    let prop_count = venues
        .iter()
        .filter(|(_, k)| *k == SimulationVenueKind::PropAmmPmmSim)
        .count();
    let manual_count = venues
        .iter()
        .filter(|(_, k)| *k == SimulationVenueKind::ManualAmmAdapter)
        .count();

    let sim_kind = match (prop_count, manual_count, unsupported.is_empty()) {
        (p, 0, _) if p > 0 => "prop_amm_pmm_sim",
        (0, m, _) if m > 0 => "manual_amm_adapter",
        (p, m, _) if p > 0 && m > 0 => "mixed",
        _ => "unsupported",
    };

    let logs: Vec<String> = venues
        .iter()
        .map(|(label, kind)| format!("{}:{}", label, kind.as_str()))
        .collect();

    SimulationResult {
        route_sig: req.route_sig,
        route_labels: req.route_labels.clone(),
        success: unsupported.is_empty(),
        executed: false,
        final_out_amount: None,
        simulated_profit: None,
        difference_vs_metis_quote: None,
        consumed_compute_units: None,
        failed_instruction_index: None,
        logs,
        unsupported_venues: unsupported,
        sim_kind: sim_kind.to_string(),
        elapsed_us: t.elapsed().as_micros() as u64,
    }
}

// ─── Phase 2: pmm-sim LiteSVM simulation ─────────────────────────────────────
//
// For routes that contain only PropAMM hops, the actual swap instruction from
// Metis is forwarded to the pmm-sim subprocess which executes it in LiteSVM
// with the latest Yellowstone-sourced pool state.  All other routes fall back
// to simulate_stub (classification only).

static SIM_REQUEST_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// Extract all unique token mints from the route_plan (inputMint + outputMint of each hop).
pub fn extract_route_mints(route_plan: &serde_json::Value) -> Vec<String> {
    let mut mints = std::collections::HashSet::new();
    if let Some(arr) = route_plan.as_array() {
        for hop in arr {
            if let Some(info) = hop.get("swapInfo") {
                if let Some(m) = info.get("inputMint").and_then(|v| v.as_str()) {
                    mints.insert(m.to_string());
                }
                if let Some(m) = info.get("outputMint").and_then(|v| v.as_str()) {
                    mints.insert(m.to_string());
                }
            }
        }
    }
    mints.into_iter().collect()
}

pub async fn simulate(
    req: &SimulationRequest,
    engine: &Arc<crate::pmm_sim::PmmSimEngine>,
    cache: &crate::pmm_sim::AccountCache,
    fee_payer: &str,
) -> SimulationResult {
    let mut result = simulate_stub(req);

    // Gather ALL instructions for full transaction simulation.
    let mut all_ixs: Vec<&crate::metis::InstructionData> = Vec::new();
    for ix in &req.instructions.compute_budget_instructions {
        all_ixs.push(ix);
    }
    for ix in &req.instructions.setup_instructions {
        all_ixs.push(ix);
    }
    all_ixs.push(&req.instructions.swap_instruction);
    if let Some(ix) = &req.instructions.cleanup_instruction {
        all_ixs.push(ix);
    }

    let ipc_instructions: Vec<crate::pmm_sim::IpcInstruction> = all_ixs.iter().map(|ix| {
        crate::pmm_sim::IpcInstruction {
            program_id: ix.program_id.clone(),
            accounts: ix.accounts.iter().map(|a| crate::pmm_sim::IpcAccountMeta {
                pubkey: a.pubkey.clone(),
                is_signer: a.is_signer,
                is_writable: a.is_writable,
            }).collect(),
            data: ix.data.clone(),
        }
    }).collect();

    let accounts = crate::pmm_sim::collect_all_instruction_accounts(&all_ixs, cache);
    let token_mints = extract_route_mints(&req.merged_quote.route_plan);
    let src_mint = req.merged_quote.input_mint.clone();
    let src_amount: u64 = req.merged_quote.in_amount.parse().unwrap_or(0);
    let id = SIM_REQUEST_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    let cache_hits = accounts.len();
    eprintln!(
        "[sim_start] route_sig={:032x} ix_count={} account_hits={}",
        req.route_sig, all_ixs.len(), cache_hits,
    );

    let ipc_req = crate::pmm_sim::FullSimRequest {
        id,
        fee_payer: fee_payer.to_string(),
        src_mint,
        src_amount,
        token_mints,
        instructions: ipc_instructions,
        lookup_tables: vec![],
        accounts,
        jito_tip_lamports: 1600,
        cu_limit: 1_400_000,
        route_sig: format!("{:032x}", req.route_sig),
        route_labels: req.route_labels.split(',').map(|s| s.trim().to_string()).collect(),
    };

    let t = std::time::Instant::now();
    match engine.simulate(&ipc_req).await {
        Some(resp) if resp.success => {
            result.executed = true;
            result.final_out_amount = resp.amount_out;
            result.consumed_compute_units = resp.compute_units;
            result.success = true;
            if let (Some(out), Ok(input)) = (resp.amount_out, req.merged_quote.in_amount.parse::<u64>()) {
                result.simulated_profit = Some(out as i64 - input as i64);
            }
            eprintln!(
                "[sim_result] route_sig={:032x} executed=true success=true final_out={} cu={} elapsed_us={}",
                req.route_sig,
                resp.amount_out.unwrap_or(0),
                resp.compute_units.unwrap_or(0),
                t.elapsed().as_micros(),
            );
        }
        Some(resp) => {
            result.executed = true;
            result.success = false;
            eprintln!(
                "[sim_result] route_sig={:032x} executed=true success=false error={} elapsed_us={}",
                req.route_sig,
                resp.error.as_deref().unwrap_or("unknown"),
                t.elapsed().as_micros(),
            );
        }
        None => {
            result.executed = false;
            eprintln!(
                "[sim_result] route_sig={:032x} executed=false reason=engine_unavailable elapsed_us={}",
                req.route_sig,
                t.elapsed().as_micros(),
            );
        }
    }

    result
}
