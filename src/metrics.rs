use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

const WINDOW_SECS: u64 = 30;

pub struct Metrics {
    // ── Stage 1: quoting ─────────────────────────────────────────────────────
    pub metis_req_sent: AtomicU64,
    pub metis_resp_total: AtomicU64,
    pub metis_resp_ok: AtomicU64,
    pub swap_ix_ok: AtomicU64,

    // ── Stage 1.5: swap_instructions ─────────────────────────────────────────
    pub swap_ix_failed: AtomicU64,
    pub swap_ix_timeout: AtomicU64,
    pub swap_ix_http: AtomicU64,
    pub swap_ix_network: AtomicU64,
    pub swap_ix_parse: AtomicU64,
    pub candidate_profitable_total: AtomicU64,
    /// Reserved — ranking disabled; will always be zero.
    #[allow(dead_code)]
    pub candidate_coalesced_total: AtomicU64,
    /// Reserved — ranking disabled; will always be zero.
    #[allow(dead_code)]
    pub candidate_dropped_rank_total: AtomicU64,
    pub candidate_dropped_inflight_total: AtomicU64,
    pub swap_ix_budget_dropped_total: AtomicU64,
    pub swap_ix_sent_total: AtomicU64,
    pub dropped_multi_hop: AtomicU64,
    pub dropped_merge_fail: AtomicU64,
    pub dropped_no_serve: AtomicU64,
    pub dropped_same_pool: AtomicU64,
    pub queue_in: AtomicU64,
    pub queue_depth: AtomicI64,

    // ── Stage 2: simulation ───────────────────────────────────────────────────
    /// Candidates pushed into the simulation queue.
    pub sim_queued: AtomicU64,
    /// Candidates that completed venue classification (Phase 1).
    pub sim_classified: AtomicU64,
    /// Candidates with at least one unsupported venue.
    pub sim_unsupported: AtomicU64,
    /// Candidates dropped from sim_queue as stale (age > queue_max_age_ms).
    pub sim_stale: AtomicU64,

    // ── Stage 3: worker processing ────────────────────────────────────────────
    pub dropped_stale: AtomicU64,
    pub tx_build_failed: AtomicU64,
    pub tx_too_large: AtomicU64,
    pub dropped_account_locks: AtomicU64,
    pub calc_done: AtomicU64,

    // ── Stage 4: Jito send ────────────────────────────────────────────────────
    pub rate_requeued: AtomicU64,
    pub jito_send_failed: AtomicU64,
    pub jito_sent: AtomicU64,

    // ── swap_instructions latency ─────────────────────────────────────────────
    pub metis_fetch_ms_total: AtomicU64,
    pub metis_fetch_samples: AtomicU64,
}

impl Metrics {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            metis_req_sent: AtomicU64::new(0),
            metis_resp_total: AtomicU64::new(0),
            metis_resp_ok: AtomicU64::new(0),
            swap_ix_ok: AtomicU64::new(0),
            swap_ix_failed: AtomicU64::new(0),
            swap_ix_timeout: AtomicU64::new(0),
            swap_ix_http: AtomicU64::new(0),
            swap_ix_network: AtomicU64::new(0),
            swap_ix_parse: AtomicU64::new(0),
            candidate_profitable_total: AtomicU64::new(0),
            candidate_coalesced_total: AtomicU64::new(0),
            candidate_dropped_rank_total: AtomicU64::new(0),
            candidate_dropped_inflight_total: AtomicU64::new(0),
            swap_ix_budget_dropped_total: AtomicU64::new(0),
            swap_ix_sent_total: AtomicU64::new(0),
            queue_in: AtomicU64::new(0),
            queue_depth: AtomicI64::new(0),
            sim_queued: AtomicU64::new(0),
            sim_classified: AtomicU64::new(0),
            sim_unsupported: AtomicU64::new(0),
            sim_stale: AtomicU64::new(0),
            dropped_stale: AtomicU64::new(0),
            tx_build_failed: AtomicU64::new(0),
            tx_too_large: AtomicU64::new(0),
            dropped_account_locks: AtomicU64::new(0),
            calc_done: AtomicU64::new(0),
            rate_requeued: AtomicU64::new(0),
            jito_send_failed: AtomicU64::new(0),
            jito_sent: AtomicU64::new(0),
            metis_fetch_ms_total: AtomicU64::new(0),
            metis_fetch_samples: AtomicU64::new(0),
            dropped_multi_hop: AtomicU64::new(0),
            dropped_merge_fail: AtomicU64::new(0),
            dropped_no_serve: AtomicU64::new(0),
            dropped_same_pool: AtomicU64::new(0),
        })
    }

    pub fn spawn_reporter(self: &Arc<Self>, queue_max_age_ms: u64) {
        let m = self.clone();
        let ttl_secs = queue_max_age_ms as f64 / 1000.0;
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(WINDOW_SECS));
            interval.tick().await;
            loop {
                interval.tick().await;

                let sent      = m.metis_req_sent.swap(0, Ordering::Relaxed);
                let routes    = m.metis_resp_total.swap(0, Ordering::Relaxed);
                let profit    = m.metis_resp_ok.swap(0, Ordering::Relaxed);
                let sw_ok     = m.swap_ix_ok.swap(0, Ordering::Relaxed);

                let swap_fail = m.swap_ix_failed.swap(0, Ordering::Relaxed);
                let sf_to     = m.swap_ix_timeout.swap(0, Ordering::Relaxed);
                let sf_http   = m.swap_ix_http.swap(0, Ordering::Relaxed);
                let sf_net    = m.swap_ix_network.swap(0, Ordering::Relaxed);
                let sf_parse  = m.swap_ix_parse.swap(0, Ordering::Relaxed);
                let cand_total = m.candidate_profitable_total.swap(0, Ordering::Relaxed);
                let cand_inflight_drop = m.candidate_dropped_inflight_total.swap(0, Ordering::Relaxed);
                let swap_budget_drop = m.swap_ix_budget_dropped_total.swap(0, Ordering::Relaxed);
                let swap_sent_total = m.swap_ix_sent_total.swap(0, Ordering::Relaxed);
                let q_in      = m.queue_in.swap(0, Ordering::Relaxed);

                let sim_q     = m.sim_queued.swap(0, Ordering::Relaxed);
                let sim_cls   = m.sim_classified.swap(0, Ordering::Relaxed);
                let sim_unsup = m.sim_unsupported.swap(0, Ordering::Relaxed);
                let sim_st    = m.sim_stale.swap(0, Ordering::Relaxed);

                let stale     = m.dropped_stale.swap(0, Ordering::Relaxed);
                let build     = m.tx_build_failed.swap(0, Ordering::Relaxed);
                let too_big   = m.tx_too_large.swap(0, Ordering::Relaxed);
                let too_locks = m.dropped_account_locks.swap(0, Ordering::Relaxed);
                let calc      = m.calc_done.swap(0, Ordering::Relaxed);
                let requeued  = m.rate_requeued.swap(0, Ordering::Relaxed);
                let jfail     = m.jito_send_failed.swap(0, Ordering::Relaxed);
                let jito      = m.jito_sent.swap(0, Ordering::Relaxed);

                let ms_ms     = m.metis_fetch_ms_total.swap(0, Ordering::Relaxed);
                let ms_n      = m.metis_fetch_samples.swap(0, Ordering::Relaxed);

                let depth     = m.queue_depth.load(Ordering::Relaxed);

                let drop_hop    = m.dropped_multi_hop.swap(0, Ordering::Relaxed);
                let drop_merge  = m.dropped_merge_fail.swap(0, Ordering::Relaxed);
                let drop_no_srv = m.dropped_no_serve.swap(0, Ordering::Relaxed);
                let drop_pool   = m.dropped_same_pool.swap(0, Ordering::Relaxed);

                let avg_ms    = if ms_n > 0 { ms_ms / ms_n } else { 0 };

                eprintln!(
                    "[{WINDOW_SECS}s] \
metis_sent={sent} routes={routes} quoted_profitable={profit}\n  \
  FUNNEL    : profitable={profit}  drop_same_pool={drop_pool}  drop_multi_hop={drop_hop}  drop_merge={drop_merge}  drop_no_serve={drop_no_srv}  -> swap_ix_ok={sw_ok}\n  \
  CANDIDATE : admitted={cand_total}  drop_inflight={cand_inflight_drop}  swap_budget_drop={swap_budget_drop}  swap_sent={swap_sent_total}\n  \
  SIM-QUEUE : queued={sim_q}  classified={sim_cls}  unsupported_venue={sim_unsup}  stale={sim_st}\n  \
PRE-QUEUE : swap_ix_ok={sw_ok}  swap_ix_fail={swap_fail} [timeout={sf_to} http={sf_http} net={sf_net} parse={sf_parse}] -> queue_in={q_in}  (depth_now={depth})\n  \
  IN-QUEUE  : stale={stale} (waited >{ttl_secs}s)\n  \
  TX-BUILD  : build_fail={build}  too_large={too_big}  too_many_locks={too_locks}  calc_ok={calc}\n  \
  JITO      : sent={jito}  send_fail={jfail}  waited_for_slot={requeued}\n  \
SWAP-IX   : avg_metis={avg_ms}ms"
                );
            }
        });
    }
}
