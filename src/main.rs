#[allow(dead_code)]
mod account_cache;
mod alt_cache;
mod arbitrage;
mod auto_missing_accounts;
mod blockhash_cache;
mod config;
mod dex_accounts;
mod jito;
#[allow(dead_code)]
mod jito_grpc;
#[allow(dead_code)]
mod litesvm_sim;
mod manual_sim_accounts;
mod metis;
mod metrics;
mod mix_registry;
mod program_registry;
mod rate_limiter;
mod template_cache;
mod token_metrics;
mod tokens;
mod transaction;
mod wallet;

use anyhow::{bail, Result};
use solana_client::rpc_client::RpcClient;
use solana_sdk::{pubkey::Pubkey, signer::Signer};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};
use std::time::{Duration, Instant};

use tracing::error;

use alt_cache::AltCache;
use blockhash_cache::BlockhashCache;
use rate_limiter::RateLimiter;

const SIM_STATIC_EXTRA_ACCOUNTS: &[&str] = &[
    "D8cy77BBepLMngZx6ZukaTff5hCt1HrWyKk3Hnd9oitf",
    "DuFXxPxAyJhHj4gMpE8As1Ta4nSSVXv8xfEDRrWQmJ9G",
    // NOTE: Enc6rB84ZwGxZU8aqAF41dRJxg3yesiJgD7uJFVhMraM was removed here. It is
    // NULL on-chain (getMultipleAccounts returns null: an uninitialized PDA), so
    // prefetching it is pointless and it is correctly handled by the
    // expected_readonly_pda_authority synthetic-system path at sim time.
    "GswwnegnBMWEuEsptDCBDmRB9YtG5zjetTSw7RunUQMY",
];

/// Parse a config commitment string into a `CommitmentConfig`. Defaults to
/// `processed` for anything unrecognised so the RPC stays aligned with the
/// Yellowstone stream.
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
    report_simulation_dex_config(&config.simulation.enabled_dexes);

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

    // Match the RPC commitment to the Yellowstone stream commitment
    // (processed) so account fetches, sim-compare, and retry snapshots read
    // the same slot the cache is fed from. Reading finalized state (the
    // RpcClient::new default) makes every hot pool look stale and feeds the
    // retry path data OLDER than the cache it is trying to correct.
    let rpc_commitment = parse_commitment(&config.rpc.commitment);
    eprintln!(
        "[rpc_commitment] level={} (stream is processed; keep these aligned)",
        config.rpc.commitment
    );
    let rpc_client = Arc::new(RpcClient::new_with_commitment(
        config.rpc.url.clone(),
        rpc_commitment,
    ));
    let fallback_rpcs: Arc<Vec<Arc<RpcClient>>> = Arc::new(
        config
            .rpc
            .fallback_rpc_urls
            .iter()
            .map(|url| Arc::new(RpcClient::new_with_commitment(url.clone(), rpc_commitment)))
            .collect(),
    );
    if !fallback_rpcs.is_empty() {
        eprintln!("[rpc_fallback] configured {} fallback RPC(s)", fallback_rpcs.len());
    }

    let wsol_mint = solana_sdk::pubkey::Pubkey::from_str_const(tokens::WSOL_MINT);
    let wsol_ata = spl_associated_token_account::get_associated_token_address(
        &trading_keypair.pubkey(),
        &wsol_mint,
    );

    // ── Template cache: load hop templates from disk and start periodic flush ─
    let template_store = template_cache::TemplateStore::new();
    if config.template_cache.save_new || config.template_cache.serve_route {
        let hops_loaded = template_store.load_from_disk();
        let routes_loaded = template_store.load_routes_from_disk();
        eprintln!(
            "[template] loaded {hops_loaded} hop templates and {routes_loaded} route templates from /root/c/cache/"
        );
        template_store.spawn_flush_task(60);
    }

    let metrics = metrics::Metrics::new();
    metrics.spawn_reporter(config.performance.queue_max_age_ms, template_store.clone());

    let token_metrics = token_metrics::TokenMetrics::new(&token_mints);
    token_metrics.spawn_reporter();

    let tip_pubkeys = transaction::jito_tip_pubkeys();
    let alt_cache = AltCache::new(tip_pubkeys);

    let metis = Arc::new(metis::MetisClient::new(
        &config.metis.url,
        config.performance.quote_timeout_ms.max(config.performance.swap_instructions_timeout_ms),
    ));

    let jito_client = Arc::new(jito::JitoClient::new(&config.jito.urls, &config.jito.uuid));

    let jito_limiter = Arc::new(Mutex::new(
        RateLimiter::new(config.jito.max_bundles_per_second),
    ));

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

    let (sim_cache, sim_pool, mix_registry) = if config.simulation.enabled {
        let cache = account_cache::AccountCache::new_with_fallbacks(
            rpc_client.clone(),
            fallback_rpcs.clone(),
        );
        let manual_accounts_root =
            manual_sim_accounts::output_root_from_dex_dir(&config.simulation.dex_dir);
        let manual_sim_accounts_path = manual_accounts_root.join("manual_sim_accounts.json");
        let manual_account_cache_path = manual_accounts_root.join("manual_account_cache.json");
        let manual_account_errors_path = manual_accounts_root.join("manual_account_errors.json");
        let tx_static_account_cache_path = manual_accounts_root.join("tx_static_account_cache.json");

        manual_sim_accounts::load_cached_accounts_into_cache(&manual_account_cache_path, &cache)?;
        auto_missing_accounts::load_cache_into_account_cache(&manual_accounts_root, &cache);
        cache.load_tx_static_account_cache(&tx_static_account_cache_path)?;
        // Real runtime bridge for problem_sim_accounts.csv:
        // every 60s it promotes needs_grpc_live_state rows into startup RPC
        // snapshots + live gRPC subscriptions, and writes a deduped load.txt.
        cache.spawn_problem_accounts_watcher(manual_accounts_root.clone(), 60);
        let tx_static_refresh_accounts =
            cache.tx_static_refresh_pubkeys(&tx_static_account_cache_path)?;
        manual_sim_accounts::fetch_and_cache_startup_accounts(
            &manual_sim_accounts_path,
            &manual_account_cache_path,
            &manual_account_errors_path,
            rpc_client.clone(),
            &cache,
            config.simulation.prefetch_pools_per_second,
        )
        .await?;

        let missing_handle = Arc::new(auto_missing_accounts::start(
            &manual_accounts_root,
            rpc_client.clone(),
            fallback_rpcs.clone(),
            cache.clone(),
        ));

        eprintln!("[rpc_fetch_reason] reason=block_time count=1 pubkeys_sample=[]");
        if let Ok(s) = rpc_client.get_slot() {
            eprintln!("[rpc_fetch_reason] reason=block_time count=1 pubkeys_sample=[]");
            let ts = match rpc_client.get_block_time(s) {
                Ok(ts) => ts,
                Err(e) => {
                    let fallback = account_cache::fallback_unix_timestamp();
                    eprintln!(
                        "[sim_clock_seed] slot={} block_time_error={} fallback_unix_timestamp={}",
                        s, e, fallback
                    );
                    fallback
                }
            };
            cache.seed_clock(s, ts);
            eprintln!("[sim_clock_seed] slot={} unix_timestamp={}", s, ts);
        }

        let dex_pools = dex_accounts::load(&config.simulation.dex_dir);
        let mix_registry = mix_registry::VerifiedMixRegistry::load_and_verify(
            rpc_client.clone(),
            &config.simulation.dex_dir,
            config.simulation.prefetch_pools_per_second,
            &config.rpc.url,
        )
        .await?;
        if let Some(registry) = &mix_registry {
            let alt_cache_path = registry.output_root().join("alt_accounts_cache.json");
            alt_cache.load_from_disk(&alt_cache_path)?;
            let mix_alts = registry.alt_accounts();
            alt_cache
                .prefetch_missing_rate_limited(
                    &mix_alts,
                    rpc_client.clone(),
                    config.simulation.prefetch_pools_per_second,
                )
                .await;

            metrics.invalid_mix_accounts.fetch_add(
                registry.invalid_account_count() as u64,
                Ordering::Relaxed,
            );
            metrics.invalid_mix_pools.fetch_add(
                (registry.invalid_pool_count() + registry.unverified_pool_count()) as u64,
                Ordering::Relaxed,
            );
            registry.spawn_unverified_retry_task(
                rpc_client.clone(),
                config.simulation.prefetch_pools_per_second,
            );
        }

        let mut sim_all_accounts = dex_pools.all_accounts.clone();
        let mut sim_subscribe_accounts = dex_pools.subscribe_accounts.clone();
        let mut sim_prefetch_groups = dex_pools.prefetch_groups.clone();
        if let Some(registry) = &mix_registry {
            registry.filter_valid_accounts(&mut sim_all_accounts);
            registry.filter_valid_accounts(&mut sim_subscribe_accounts);
            sim_subscribe_accounts.extend(registry.valid_variable_accounts());
            sim_subscribe_accounts.sort_unstable();
            sim_subscribe_accounts.dedup();
            sim_prefetch_groups = registry.filter_valid_groups(&sim_prefetch_groups);
        }

        // Load ALL pool accounts from pools_by_dex/*.json as the authoritative
        // source for the 713+ fixed pools.  This is done AFTER mix_registry
        // filtering so these accounts are never removed.
        //
        // Classification inside load_pools_by_dex_dir:
        //   - tokenAccountA/B + DEX-specific volatile state → subscribe (Yellowstone)
        //   - mints, authority PDAs                         → prefetch (RPC only)
        //   - addressLookupTableAddress                     → alt_accounts (AltCache)
        //
        // This replaces the old approach of only loading vaults A/B: every
        // DEX-specific writable account (oracle, observationState, globalVault,
        // tickmap, etc.) is now also subscribed so the sim cache stays current.
        let fixed_pools = dex_accounts::load_pools_by_dex_dir("pools_by_dex");
        sim_all_accounts.extend_from_slice(&fixed_pools.all_accounts);
        sim_all_accounts.sort_unstable();
        sim_all_accounts.dedup();
        sim_subscribe_accounts.extend_from_slice(&fixed_pools.subscribe_accounts);
        sim_subscribe_accounts.sort_unstable();
        sim_subscribe_accounts.dedup();
        sim_prefetch_groups.extend(fixed_pools.prefetch_groups);

        // Pre-load ALT contents for every addressLookupTableAddress in pools_by_dex
        // so v0 transactions that reference them can be resolved by the simulator.
        if !fixed_pools.alt_accounts.is_empty() {
            eprintln!(
                "[pools_by_dex_alts] loading {} ALTs into alt_cache",
                fixed_pools.alt_accounts.len()
            );
            alt_cache
                .prefetch_missing_rate_limited(
                    &fixed_pools.alt_accounts,
                    rpc_client.clone(),
                    config.simulation.prefetch_pools_per_second,
                )
                .await;
        }

        let mut live_extra = vec![wsol_ata];
        live_extra.extend_from_slice(&sim_subscribe_accounts);
        for s in SIM_STATIC_EXTRA_ACCOUNTS {
            if let Ok(pk) = solana_sdk::pubkey::Pubkey::try_from(*s) {
                live_extra.push(pk);
            }
        }

        cache.spawn_subscription(
            config.yellowstone_grpc.endpoint.clone(),
            config.yellowstone_grpc.x_token.clone(),
            program_registry::all_program_ids(),
            live_extra,
        );

        let mut warm_extra: Vec<solana_sdk::pubkey::Pubkey> = token_mints
            .iter()
            .filter_map(|s| solana_sdk::pubkey::Pubkey::try_from(s.as_str()).ok())
            .collect();
        warm_extra.push(wsol_mint);
        warm_extra.push(wsol_ata);
        warm_extra.push(trading_keypair.pubkey());
        for s in SIM_STATIC_EXTRA_ACCOUNTS {
            if let Ok(pk) = solana_sdk::pubkey::Pubkey::try_from(*s) {
                warm_extra.push(pk);
            }
        }
        for mint_str in &token_mints {
            if let Ok(mint) = solana_sdk::pubkey::Pubkey::try_from(mint_str.as_str()) {
                let ata = spl_associated_token_account::get_associated_token_address(
                    &trading_keypair.pubkey(),
                    &mint,
                );
                warm_extra.push(ata);
            }
        }
        warm_extra.sort_unstable();
        warm_extra.dedup();

        let mut prefetch_groups = sim_prefetch_groups;
        if prefetch_groups.is_empty() && !sim_all_accounts.is_empty() {
            prefetch_groups.extend(sim_all_accounts.chunks(100).map(|chunk| chunk.to_vec()));
        }
        if !sim_subscribe_accounts.is_empty() {
            prefetch_groups.extend(sim_subscribe_accounts.chunks(100).map(|chunk| chunk.to_vec()));
        }
        if !tx_static_refresh_accounts.is_empty() {
            prefetch_groups.extend(
                tx_static_refresh_accounts
                    .chunks(100)
                    .map(|chunk| chunk.to_vec()),
            );
        }
        prefetch_groups.extend(warm_extra.chunks(100).map(|chunk| chunk.to_vec()));

        cache
            .prefetch_groups_rate_limited(
                &prefetch_groups,
                config.simulation.prefetch_pools_per_second,
            )
            .await;

        wait_for_live_cache_ready(
            &cache,
            &sim_subscribe_accounts,
            mix_registry.as_deref(),
        )
        .await?;

        let pool = litesvm_sim::SimulatorPool::new(
            config.simulation.workers,
            &config.simulation.so_dir,
            wsol_ata,
            trading_keypair.pubkey(),
            config.simulation.fail_closed,
            config.simulation.allow_hot_path_rpc_fetch,
            manual_accounts_root.clone(),
            cache.stream_slot(),
            cache.stream_unix_timestamp(),
            Some(missing_handle),
        )?;
        eprintln!(
            "[simulator_ready] true workers={} so_dir={}",
            config.simulation.workers,
            config.simulation.so_dir
        );
        (
            Some(Arc::new(cache)),
            Some(Arc::new(pool)),
            mix_registry,
        )
    } else {
        (None, None, None)
    };

    let blockhash_cache = Arc::new(BlockhashCache::new(rpc_client.clone()));

    // ── Build shared CalcCtx ─────────────────────────────────────────────────
    let calc_ctx = Arc::new(arbitrage::CalcCtx {
        metis: metis.clone(),
        blockhash_cache: blockhash_cache.clone(),
        trading_keypair: trading_keypair.clone(),
        rpc_client: rpc_client.clone(),
        alt_cache: alt_cache.clone(),
        jito: jito_client,
        jito_grpc: jito_grpc_client,
        jito_limiter: jito_limiter.clone(),
        jito_grpc_limiter: jito_grpc_limiter.clone(),
        cu_limits: config.performance.cu_limits.clone(),
        user_pubkey: trading_keypair.pubkey().to_string(),
        sim_cache,
        sim_pool,
        mix_registry,
        template_store,
        template_config: config.template_cache.clone(),
        swap_ix_state: Arc::new(arbitrage::SwapIxState::new(
            config.performance.max_concurrent_swap_instructions,
        )),
    });

    let worker_count = config.performance.calc_workers.max(1);
    let jito_capacity = config.jito.max_bundles_per_second as usize
        + jito_grpc_limiter
            .as_ref()
            .map(|_| config.jito_grpc.max_bundles_per_second as usize)
            .unwrap_or(0);
    let pipeline = arbitrage::spawn_workers(
        calc_ctx.clone(),
        metrics.clone(),
        worker_count,
        config.performance.queue_max_age_ms,
    );

    eprintln!(
        "scanner ready | tokens={} | pairs_per_scan={} | calc_workers={worker_count} | jito_capacity_per_sec={jito_capacity} | quote_concurrency={}",
        token_mints.len(),
        {
            let steps = ((config.trading.max_amount_sol - config.trading.min_amount_sol)
                / config.trading.step_sol) as usize
                + 1;
            steps * token_mints.len() * 2 // ×2: free + direct route per pair
        },
        config.performance.max_concurrent_quotes.max(1),
    );

    loop {
        if let Err(e) = arbitrage::scan_all_tokens(
            &token_mints,
            &config,
            &calc_ctx,
            &pipeline,
            &metrics,
            &token_metrics,
        )
        .await
        {
            error!(error = %e, "scan cycle error");
        }
    }
}

async fn wait_for_live_cache_ready(
    cache: &account_cache::AccountCache,
    accounts: &[Pubkey],
    mix_registry: Option<&mix_registry::VerifiedMixRegistry>,
) -> Result<()> {
    let mut targets = accounts.to_vec();
    targets.sort_unstable();
    targets.dedup();

    let timeout = Duration::from_secs(30);
    let deadline = Instant::now() + timeout;
    loop {
        let missing = targets
            .iter()
            .copied()
            .filter(|account| cache.get(account).is_none())
            .collect::<Vec<_>>();
        if missing.is_empty() {
            eprintln!(
                "[grpc_live_cache_ready] true live_accounts_ready={} live_accounts_total={} missing_live_accounts=0 valid_pools_ready={} invalid_pools={}",
                targets.len(),
                targets.len(),
                mix_registry.map(|registry| registry.valid_pool_count()).unwrap_or(0),
                mix_registry
                    .map(|registry| registry.invalid_pool_count() + registry.unverified_pool_count())
                    .unwrap_or(0)
            );
            return Ok(());
        }

        if Instant::now() >= deadline {
            // Before failing, do a final RPC check to distinguish accounts that
            // genuinely exist on-chain (Yellowstone lag → real error) from
            // accounts that don't exist at all (closed/inactive pools → safe to
            // skip).  pools_by_dex/*.json may reference vaults that were closed
            // since the file was generated; those should never block startup.
            let still_missing_after_rpc = {
                let rpc = cache.rpc_client();
                let chunk_size = 100;
                let mut still_missing = Vec::new();
                let mut not_found_on_rpc = 0usize;
                for chunk in missing.chunks(chunk_size) {
                    match rpc.get_multiple_accounts(chunk) {
                        Ok(results) => {
                            for (pk, maybe_acct) in chunk.iter().zip(results.into_iter()) {
                                match maybe_acct {
                                    Some(acct) => {
                                        // Account exists on-chain but Yellowstone hasn't
                                        // delivered it yet — this is a real readiness problem.
                                        let account = account_cache::rpc_account_to_cache_account(acct);
                                        cache.insert_manual(*pk, account);
                                        // Now it's in cache; don't count as missing.
                                    }
                                    None => {
                                        // Account doesn't exist on-chain; will never arrive
                                        // from Yellowstone.  Log and skip.
                                        eprintln!(
                                            "[grpc_live_cache_skip_notfound] pk={} reason=account_not_on_chain",
                                            pk
                                        );
                                        not_found_on_rpc += 1;
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            // RPC error: can't distinguish; treat as still missing.
                            for pk in chunk {
                                still_missing.push(*pk);
                            }
                            eprintln!("[grpc_live_cache_rpc_check_error] error={e}");
                        }
                    }
                }
                // Re-check after the RPC top-up.
                let remaining: Vec<Pubkey> = targets
                    .iter()
                    .copied()
                    .filter(|pk| cache.get(pk).is_none())
                    .filter(|pk| still_missing.contains(pk))
                    .collect();
                eprintln!(
                    "[grpc_live_cache_ready_rpc_check] missing_before={} not_found_on_rpc={} rpc_errors={} still_missing={}",
                    missing.len(),
                    not_found_on_rpc,
                    missing.len().saturating_sub(not_found_on_rpc + remaining.len()),
                    remaining.len()
                );
                remaining
            };

            // Re-count after the RPC top-up.
            let final_missing: Vec<Pubkey> = targets
                .iter()
                .copied()
                .filter(|pk| cache.get(pk).is_none())
                .collect();

            if final_missing.is_empty() {
                eprintln!(
                    "[grpc_live_cache_ready] true live_accounts_ready={} live_accounts_total={} missing_live_accounts=0 valid_pools_ready={} invalid_pools={} note=resolved_by_rpc_topup",
                    targets.len(),
                    targets.len(),
                    mix_registry.map(|registry| registry.valid_pool_count()).unwrap_or(0),
                    mix_registry
                        .map(|registry| registry.invalid_pool_count() + registry.unverified_pool_count())
                        .unwrap_or(0)
                );
                return Ok(());
            }

            // If accounts still missing are only because of RPC errors (not
            // confirmed on-chain), treat it as a soft warning and continue.
            if still_missing_after_rpc.is_empty() {
                // All were NotFound on RPC — safe to proceed.
                eprintln!(
                    "[grpc_live_cache_ready] true live_accounts_ready={} live_accounts_total={} missing_live_accounts={} note=all_missing_are_not_found_on_chain",
                    targets.len().saturating_sub(final_missing.len()),
                    targets.len(),
                    final_missing.len(),
                );
                return Ok(());
            }

            let sample = crate::mix_registry::pubkeys_json(
                &final_missing.iter().copied().take(20).collect::<Vec<_>>(),
            );
            eprintln!(
                "[grpc_live_cache_ready] false live_accounts_ready={} live_accounts_total={} missing_live_accounts={} sample={}",
                targets.len().saturating_sub(final_missing.len()),
                targets.len(),
                final_missing.len(),
                sample
            );
            bail!(
                "live account cache is not ready after {}s; missing={} sample={}",
                timeout.as_secs(),
                final_missing.len(),
                sample
            );
        }

        tokio::time::sleep(Duration::from_millis(250)).await;
    }
}

fn report_simulation_dex_config(enabled_dexes: &[String]) {
    let normalized = enabled_dexes
        .iter()
        .map(|raw| {
            program_registry::dex_key_from_program_id(raw)
                .or_else(|| program_registry::dex_key_from_label(raw))
                .map(str::to_string)
                .unwrap_or_else(|| program_registry::normalize_dex_key(raw))
        })
        .collect::<Vec<_>>();
    eprintln!(
        "[simulation_dexes] configured={} normalized={} note=simulation_enabled_routes_are_gated_by_loaded_programs_and_mix",
        enabled_dexes.len(),
        serde_json::to_string(&normalized).unwrap_or_else(|_| "[]".to_string())
    );
}