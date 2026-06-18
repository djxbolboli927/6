//! Diagnostic counters for the BisonFi first-test flow, reported every 30s.
//!
//! These exist to pinpoint where the pipeline stalls: pool updates from gRPC,
//! account state warmth, and the bot↔pmm-sim request/response loop.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Default)]
pub struct BisonMetrics {
    /// Number of live Yellowstone updates for the watched pool accounts.
    pub grpc_pool_updates: AtomicU64,
    /// Sum / count of inter-update intervals (ms) — used for the average.
    pub grpc_interval_sum_ms: AtomicU64,
    pub grpc_interval_samples: AtomicU64,
    /// Internal: timestamp (unix ms) of the previous pool update.
    grpc_last_update_ms: AtomicU64,

    /// build_bison requests the bot sent to pmm-sim (each carries the gRPC state).
    pub build_requests: AtomicU64,
    /// build_bison responses pmm-sim returned to the bot.
    pub build_responses: AtomicU64,
    /// Responses that were successful with a non-zero predicted output.
    pub build_success: AtomicU64,
    /// Sum / count of pmm-sim round-trip latency (µs) — used for the average.
    pub build_resp_us_sum: AtomicU64,
    pub build_resp_samples: AtomicU64,

    /// Whether the pmm-sim subprocess was reachable on the last check.
    pub pmm_up: AtomicBool,
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

impl BisonMetrics {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// Record one live pool-account update (called from the Yellowstone stream).
    pub fn record_pool_update(&self) {
        let now = now_ms();
        let prev = self.grpc_last_update_ms.swap(now, Ordering::Relaxed);
        if prev != 0 && now >= prev {
            self.grpc_interval_sum_ms
                .fetch_add(now - prev, Ordering::Relaxed);
            self.grpc_interval_samples.fetch_add(1, Ordering::Relaxed);
        }
        self.grpc_pool_updates.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_build_request(&self) {
        self.build_requests.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_build_response(&self, elapsed_us: u64, success: bool) {
        self.build_responses.fetch_add(1, Ordering::Relaxed);
        self.build_resp_us_sum.fetch_add(elapsed_us, Ordering::Relaxed);
        self.build_resp_samples.fetch_add(1, Ordering::Relaxed);
        if success {
            self.build_success.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn set_pmm_up(&self, up: bool) {
        self.pmm_up.store(up, Ordering::Relaxed);
    }
}
