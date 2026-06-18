mod arbitrage;
mod bison_test;
mod blockhash_cache;
mod config;
mod jito;
#[allow(dead_code)]
mod jito_grpc;
mod metis;
mod metrics;
mod mix_watchlist;
mod pmm_sim;
mod rate_limiter;
mod sim;
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

    // ── PMM simulation: account cache (Yellowstone) + pmm-sim subprocess ───────
    let pmm_cache = pmm_sim::new_account_cache();
    pmm_sim::preload_cache_from_disk(&config.pmm_sim.accounts_path, &pmm_cache);

    // ── BisonFi first-test mode ────────────────────────────────────────────────
    // When enabled, the scanner is fully disabled and a single fixed flow runs,
    // re-triggered live on every pool update. Build the trigger BEFORE the
    // Yellowstone subscription so no early update is lost.
    let (bison_trigger, bison_notify) = if config.bison_test.enabled {
        let (t, n) = bison_test::make_trigger(&config.bison_test);
        (Some(t), Some(n))
    } else {
        (None, None)
    };

    // Load dynamic watchlist from mix.json (all pool accounts for Yellowstone).
    let mut mix_watchlist_set = mix_watchlist::load_mix_watchlist(&config.pmm_sim.mix_json_path);
    // In BisonFi test mode we only need the one pool's accounts streamed live.
    if config.bison_test.enabled {
        mix_watchlist_set.insert(config.bison_test.market.clone());
        mix_watchlist_set.insert(config.bison_test.base_ta.clone());
        mix_watchlist_set.insert(config.bison_test.quote_ta.clone());
    }
    let mix_watchlist: Vec<String> = mix_watchlist_set.into_iter().collect();

    // Bootstrap account state from RPC before first simulation (optional).
    if config.pmm_sim.rpc_bootstrap && !mix_watchlist.is_empty() {
        eprintln!("[main] starting RPC account bootstrap ({} accounts)...", mix_watchlist.len());
        let rpc_clone = rpc_client.clone();
        let cache_clone = pmm_cache.clone();
        let batch = config.pmm_sim.rpc_bootstrap_batch_size;
        let wl = mix_watchlist.clone();
        tokio::task::spawn_blocking(move || {
            pmm_sim::bootstrap_from_rpc(&wl, &cache_clone, &rpc_clone, batch);
        }).await.ok();
    }

    // Write programs registry for pmm-sim serve to load extra .so files.
    pmm_sim::write_programs_registry(&config.pmm_sim.programs, &config.pmm_sim.programs_path);

    pmm_sim::spawn_yellowstone_subscription(
        config.yellowstone_grpc.endpoint.clone(),
        config.yellowstone_grpc.x_token.clone(),
        mix_watchlist.clone(),
        pmm_cache.clone(),
        bison_trigger,
    );
    let pmm_engine = pmm_sim::PmmSimEngine::new(config.pmm_sim.clone());
    if config.pmm_sim.enabled {
        pmm_engine.ensure_started().await;
    }

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

    // ── BisonFi first-test mode: run the single fixed flow and never return ─────
    if config.bison_test.enabled {
        let notify = bison_notify.expect("bison_notify is Some when bison_test.enabled");
        bison_test::run_with_notify(
            config.bison_test.clone(),
            calc_ctx.clone(),
            pmm_engine.clone(),
            pmm_cache.clone(),
            config.trading.min_profit_lamports,
            notify,
        )
        .await;
        return Ok(());
    }

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
    let fee_payer_str = trading_keypair.pubkey().to_string();
    arbitrage::spawn_sim_workers(
        sim_queue,
        pipeline.clone(),
        metrics.clone(),
        sim_worker_count,
        config.performance.queue_max_age_ms,
        pmm_engine,
        pmm_cache,
        fee_payer_str,
        config.pmm_sim.simulation_gate,
        config.pmm_sim.min_profit_after_sim_lamports,
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
