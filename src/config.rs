use anyhow::Result;
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    pub metis: MetisConfig,
    pub trading: TradingConfig,
    pub jito: JitoConfig,
    pub rpc: RpcConfig,
    pub performance: PerformanceConfig,
    #[serde(default)]
    pub jito_grpc: JitoGrpcConfig,
}

#[derive(Debug, Deserialize, Clone)]
pub struct MetisConfig {
    pub url: String,
    #[allow(dead_code)]
    pub binary_key: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct TradingConfig {
    pub min_amount_sol: f64,
    pub max_amount_sol: f64,
    pub step_sol: f64,
    pub min_profit_lamports: u64,
    /// Standard Solana transaction fee in lamports (5000 = one signature fee).
    #[allow(dead_code)]
    pub base_fee_lamports: u64,
    pub tokens_file: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct JitoConfig {
    /// Multiple Jito block engine URLs -- bundles are sent to ALL concurrently.
    pub urls: Vec<String>,
    pub uuid: String,
    pub trading_keypair: String,
    #[allow(dead_code)]
    pub tip_min_lamports: u64,
    #[allow(dead_code)]
    pub tip_max_lamports: u64,
    #[allow(dead_code)]
    pub tip_profit_percent: f64,
    pub max_bundles_per_second: u32,
}

#[derive(Debug, Deserialize, Clone)]
pub struct RpcConfig {
    pub url: String,
    #[serde(default = "default_rpc_commitment")]
    pub commitment: String,
}

fn default_rpc_commitment() -> String {
    "processed".to_string()
}

/// Second Jito submission path via SearcherService gRPC.
///
/// Runs alongside the REST UUID client in `jito.rs`. Each path has its
/// own rate limiter, so the effective Jito throughput is
/// `jito.max_bundles_per_second + jito_grpc.max_bundles_per_second`.
///
/// Like the REST client, gRPC fans out to every regional Block Engine
/// endpoint concurrently — first regional success wins. Per-region auth
/// is attempted using the whitelisted keypair, which gives 5 req/s per
/// region. Regions whose auth fails downgrade to no-auth mode (1 req/s).
#[derive(Debug, Deserialize, Clone)]
pub struct JitoGrpcConfig {
    /// If false, only the REST UUID path is used (pre-gRPC behaviour).
    #[serde(default)]
    pub enabled: bool,
    /// All Jito Block Engine gRPC endpoints. Bundles are broadcast to ALL
    /// of these per send call, mirroring the REST multi-region fan-out.
    #[serde(default = "default_jito_grpc_endpoints")]
    pub endpoints: Vec<String>,
    /// Path to the Solana keypair JSON whose pubkey Jito has whitelisted
    /// for gRPC auth. This wallet holds no funds — it is an identity only.
    /// If empty or auth fails, regions fall back to no-auth (1 req/s).
    #[serde(default)]
    pub auth_keypair: String,
    /// Per-second rate limit applied *before* the gRPC SendBundle call.
    /// REST and gRPC limiters operate independently.
    #[serde(default = "default_grpc_rate")]
    pub max_bundles_per_second: u32,
}

impl Default for JitoGrpcConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            endpoints: default_jito_grpc_endpoints(),
            auth_keypair: String::new(),
            max_bundles_per_second: default_grpc_rate(),
        }
    }
}

fn default_jito_grpc_endpoints() -> Vec<String> {
    vec![
        "https://amsterdam.mainnet.block-engine.jito.wtf".to_string(),
        "https://dublin.mainnet.block-engine.jito.wtf".to_string(),
        "https://frankfurt.mainnet.block-engine.jito.wtf".to_string(),
        "https://london.mainnet.block-engine.jito.wtf".to_string(),
        "https://ny.mainnet.block-engine.jito.wtf".to_string(),
        "https://slc.mainnet.block-engine.jito.wtf".to_string(),
        "https://singapore.mainnet.block-engine.jito.wtf".to_string(),
        "https://tokyo.mainnet.block-engine.jito.wtf".to_string(),
    ]
}

fn default_grpc_rate() -> u32 {
    5
}

#[derive(Debug, Deserialize, Clone)]
pub struct PerformanceConfig {
    /// Number of tokio worker threads (multi-thread runtime).
    pub threads: usize,
    pub quote_timeout_ms: u64,
    /// CU limits per hop count: index 0 = 2 hops, index 1 = 3 hops, etc.
    /// If hops exceed the array, the last value is used.
    pub cu_limits: Vec<u32>,
    /// Maximum in-flight Metis quote requests per scan chunk.
    /// Keeps the HTTP connection pool from being overwhelmed.
    #[serde(default = "default_max_concurrent_quotes")]
    pub max_concurrent_quotes: usize,
    /// Micro-batch window before firing /swap-instructions. Candidates inside
    /// this short window are ranked and coalesced by route before Metis.
    #[serde(default = "default_candidate_batch_window_ms")]
    pub candidate_batch_window_ms: u64,
    /// Keep at most this many profitable candidates per route signature.
    #[serde(default = "default_candidate_top_per_route")]
    pub candidate_top_per_route: usize,
    /// Keep at most this many candidates globally per micro-batch.
    #[serde(default = "default_candidate_global_top_n")]
    pub candidate_global_top_n: usize,
    /// Hard cap for concurrent /swap-instructions requests.
    #[serde(default = "default_max_concurrent_swap_instructions")]
    pub max_concurrent_swap_instructions: usize,
    /// Timeout for /swap-instructions, separate from quote timeout.
    #[serde(default = "default_swap_instructions_timeout_ms")]
    pub swap_instructions_timeout_ms: u64,
    /// Extra quote-stage margin before a candidate is allowed to request
    /// /swap-instructions. This absorbs latency and reduces Metis bursts.
    #[serde(default = "default_metis_latency_margin_lamports")]
    pub metis_latency_margin_lamports: u64,
    /// Maximum concurrent Stage-2 calc workers (merge quotes + fire
    /// swap_instructions). With fire-and-forget each worker holds its slot
    /// only for microseconds, so this can be set high to rule out the calc
    /// stage as a bottleneck. Default 6 (legacy value).
    #[serde(default = "default_calc_workers")]
    pub calc_workers: usize,
    /// Max time (ms) a swap_instructions result may wait in the LIFO queue
    /// before being dropped by a calc worker. Tune higher to tolerate slower
    /// Metis responses; lower to discard stale opportunities faster.
    #[serde(default = "default_queue_max_age_ms")]
    pub queue_max_age_ms: u64,
    #[serde(default)]
    pub bot_cpu_cores: Vec<usize>,
}

fn default_max_concurrent_quotes() -> usize {
    512
}

fn default_candidate_batch_window_ms() -> u64 {
    30
}

fn default_candidate_top_per_route() -> usize {
    1
}

fn default_candidate_global_top_n() -> usize {
    40
}

fn default_max_concurrent_swap_instructions() -> usize {
    32
}

fn default_swap_instructions_timeout_ms() -> u64 {
    800
}

fn default_metis_latency_margin_lamports() -> u64 {
    5_000
}

fn default_calc_workers() -> usize {
    6
}

fn default_queue_max_age_ms() -> u64 {
    5000
}

impl Config {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Config = toml::from_str(&content)?;
        Ok(config)
    }
}