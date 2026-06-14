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
    let t = Instant::now();
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

// ─── Phase 2: SolFi LiteSVM simulation ───────────────────────────────────────
//
// Extends simulate_stub with real SolFi swap simulation via LiteSVM.
// SolFi hops are detected in the route_plan, simulated with fresh accounts
// from the SolFiAccountCache, and the results are logged.
// Phase 2 still forwards ALL candidates to Jito — Phase 3 will gate on results.

pub fn simulate(
    req: &SimulationRequest,
    solfi_cache: &Arc<crate::solfi_sim::SolFiAccountCache>,
) -> SimulationResult {
    let mut result = simulate_stub(req);

    // Attempt SolFi simulation if a fresh snapshot is available.
    if let Some(snapshot) = solfi_cache.get_snapshot() {
        let hop_results =
            crate::solfi_sim::simulate_solfi_hops(&snapshot, &req.merged_quote.route_plan);
        if !hop_results.is_empty() {
            for hop in &hop_results {
                if hop.success {
                    result.logs.push(format!(
                        "[solfi_sim_ok] market={} amount_in={} amount_out={}",
                        hop.market, hop.amount_in, hop.amount_out
                    ));
                } else {
                    result.logs.push(format!(
                        "[solfi_sim_err] market={} amount_in={} error={}",
                        hop.market,
                        hop.amount_in,
                        hop.error.as_deref().unwrap_or("unknown")
                    ));
                }
            }
        }
    }

    result
}
