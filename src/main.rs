mod arbitrage;
mod blockhash_cache;
mod config;
mod jito;
#[allow(dead_code)]
mod jito_grpc;
mod metis;
mod metrics;
mod rate_limiter;
mod sim;
mod solfi_sim;
mod token_metrics;
mod tokens;
mod transaction;
mod wallet;

use anyhow::Result;
use solana_client::rpc_client::RpcClient;
use solana_sdk::signer::Signer;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};

use tracing::error;

use blockhash_cache::BlockhashCache;
use rate_limiter::RateLimiter;

fn parse_commitment(level: &str) -> solana_sdk::commitment_config::CommitmentConfig {
    use solana_sdk::commitment_config::CommitmentConfig;
    match level.trim().to_ascii_lowercase().as_str() {
        "finalized" => CommitmentConfig::finalized(),
        "confirmed" => CommitmentConfig::confirmed(),
        _ => CommitmentConfig::processed(),
    }
}

fn main() -> Result<()> {
    let log_filter = std::env::var("RUST_LOG").unwrap_or_else(|_| "error".to_string());
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::new(
            format!("{log_filter},hyper_util=error,hyper=error,reqwest=error,h2=error,tonic=error"),
        ))
        .init();

    let config = config::Config::load("config.toml")?;

    let worker_threads = config.performance.threads.max(1);
    let pinned_cores: Vec<usize> = config.performance.bot_cpu_cores.clone();
    let available_cores = core_affinity::get_core_ids().unwrap_or_default();
    let next_worker = Arc::new(AtomicUsize::new(0));

    let mut builder = tokio::runtime::Builder::new_multi_thread();
    builder.worker_threads(worker_threads).enable_all();
    builder.thread_name("arb-worker");

    if !pinned_cores.is_empty() {
        let cores = pinned_cores.clone();
        let available = available_cores.clone();
        let counter = next_worker.clone();
        builder.on_thread_start(move || {
            let idx = counter.fetch_add(1, Ordering::SeqCst);
            let target = cores[idx % cores.len()];
            if let Some(core_id) = available.iter().find(|c| c.id == target) {
                core_affinity::set_for_current(*core_id);
            }
        });
    }

    let runtime = builder.build()?;
    runtime.block_on(async_main(config))
}

async fn async_main(config: config::Config) -> Result<()> {
    let token_mints = tokens::load_tokens(&config.trading.tokens_file)?;

    let trading_keypair = Arc::new(wallet::read_keypair(&config.jito.trading_keypair)?);

    let rpc_commitment = parse_commitment(&config.rpc.commitment);
    eprintln!("[rpc_commitment] level={}", config.rpc.commitment);
    let rpc_client = Arc::new(RpcClient::new_with_commitment(
        config.rpc.url.clone(),
        rpc_commitment,
    ));

    let alt_lookup = transaction::new_alt_lookup();

    let metrics = metrics::Metrics::new();
    metrics.spawn_reporter(config.performance.queue_max_age_ms);

    let token_metrics = token_metrics::TokenMetrics::new(&token_mints);
    token_metrics.spawn_reporter();

    let metis = Arc::new(metis::MetisClient::new(
        &config.metis.url,
        config.performance
            .quote_timeout_ms
            .max(config.performance.swap_instructions_timeout_ms),
    ));

    let jito_client = Arc::new(jito::JitoClient::new(&config.jito.urls, &config.jito.uuid));

    let jito_limiter = Arc::new(Mutex::new(RateLimiter::new(
        config.jito.max_bundles_per_second,
    )));

    let (jito_grpc_client, jito_grpc_limiter) = if config.jito_grpc.enabled {
        match jito_grpc::JitoGrpcClient::new(
            &config.jito_grpc.endpoints,
            &config.jito_grpc.auth_keypair,
        )
        .await
        {
            Ok(client) => {
                let limiter = Arc::new(Mutex::new(RateLimiter::new(
                    config.jito_grpc.max_bundles_per_second,
                )));
                (Some(Arc::new(client)), Some(limiter))
            }
            Err(e) => {
                eprintln!("Jito gRPC init failed: {e} — continuing REST-only");
                (None, None)
            }
        }
    } else {
        (None, None)
    };

    let solfi_cache = solfi_sim::SolFiAccountCache::new(config.rpc.url.clone());

    let blockhash_cache = Arc::new(BlockhashCache::new(rpc_client.clone()));

    // Simulation queue: sits between Metis instruction fetch and Jito send.
    let sim_queue = sim::SimQueue::new();

    let calc_ctx = Arc::new(arbitrage::CalcCtx {
        metis: metis.clone(),
        blockhash_cache: blockhash_cache.clone(),
        trading_keypair: trading_keypair.clone(),
        rpc_client: rpc_client.clone(),
        alt_lookup,
        jito: jito_client,
        jito_grpc: jito_grpc_client,
        jito_limiter: jito_limiter.clone(),
        jito_grpc_limiter: jito_grpc_limiter.clone(),
        cu_limits: config.performance.cu_limits.clone(),
        user_pubkey: trading_keypair.pubkey().to_string(),
        swap_ix_state: Arc::new(arbitrage::SwapIxState::new(
            config.performance.max_concurrent_swap_instructions,
        )),
        sim_queue: sim_queue.clone(),
    });

    let worker_count = config.performance.calc_workers.max(1);
    let sim_worker_count = config.performance.sim_workers.max(1);
    let jito_capacity = config.jito.max_bundles_per_second as usize
        + jito_grpc_limiter
            .as_ref()
            .map(|_| config.jito_grpc.max_bundles_per_second as usize)
            .unwrap_or(0);

    // Stage 3: Jito LIFO workers
    let pipeline = arbitrage::spawn_workers(
        calc_ctx.clone(),
        metrics.clone(),
        worker_count,
        config.performance.queue_max_age_ms,
    );

    // Stage 2: simulation workers (pop from sim_queue → classify → push to pipeline)
    arbitrage::spawn_sim_workers(
        sim_queue,
        pipeline.clone(),
        metrics.clone(),
        sim_worker_count,
        config.performance.queue_max_age_ms,
        solfi_cache,
    );

    eprintln!(
        "scanner ready | tokens={} | pairs_per_scan={} | calc_workers={worker_count} | sim_workers={sim_worker_count} | jito_capacity_per_sec={jito_capacity} | quote_concurrency={}",
        token_mints.len(),
        {
            let steps = ((config.trading.max_amount_sol - config.trading.min_amount_sol)
                / config.trading.step_sol) as usize
                + 1;
            steps * token_mints.len() * 2
        },
        config.performance.max_concurrent_quotes.max(1),
    );

    loop {
        if let Err(e) = arbitrage::scan_all_tokens(
            &token_mints,
            &config,
            &calc_ctx,
            &metrics,
            &token_metrics,
        )
        .await
        {
            error!(error = %e, "scan cycle error");
        }
    }
}
