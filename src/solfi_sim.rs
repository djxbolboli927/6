use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use solana_sdk::commitment_config::CommitmentConfig;
use solana_sdk::pubkey::Pubkey;
use spl_associated_token_account::get_associated_token_address;

use solfi_sim::constants::{SOLFI_MARKETS, USDC, WSOL};
use solfi_sim::swap::SwapDirection;

const REFRESH_INTERVAL_MS: u64 = 400;
const MAX_STALE_MS: u64 = 2_000;

type RawAccount = (Pubkey, solana_sdk::account::Account);

pub struct SolFiAccountCache {
    snapshot: Arc<RwLock<Option<Arc<SolFiSnapshot>>>>,
}

pub struct SolFiSnapshot {
    pub accounts: Vec<RawAccount>,
    pub slot: u64,
    pub fetched_at: Instant,
}

pub struct SolFiHopResult {
    pub market: Pubkey,
    pub amount_in: u64,
    pub amount_out: u64,
    pub success: bool,
    pub error: Option<String>,
}

impl SolFiAccountCache {
    pub fn new(rpc_url: String) -> Arc<Self> {
        let cache = Arc::new(Self {
            snapshot: Arc::new(RwLock::new(None)),
        });
        let snap = cache.snapshot.clone();
        tokio::spawn(async move {
            loop {
                let url = rpc_url.clone();
                match tokio::task::spawn_blocking(move || fetch_solfi_accounts(&url)).await {
                    Ok(Ok((accounts, slot))) => {
                        let mut guard = snap.write().unwrap();
                        *guard = Some(Arc::new(SolFiSnapshot {
                            accounts,
                            slot,
                            fetched_at: Instant::now(),
                        }));
                    }
                    Ok(Err(e)) => eprintln!("[solfi_cache] fetch error: {e}"),
                    Err(e) => eprintln!("[solfi_cache] spawn_blocking panic: {e}"),
                }
                tokio::time::sleep(Duration::from_millis(REFRESH_INTERVAL_MS)).await;
            }
        });
        cache
    }

    pub fn get_snapshot(&self) -> Option<Arc<SolFiSnapshot>> {
        let guard = self.snapshot.read().unwrap();
        let snap = guard.clone()?;
        if snap.fetched_at.elapsed().as_millis() as u64 > MAX_STALE_MS {
            return None;
        }
        Some(snap)
    }
}

fn fetch_solfi_accounts(rpc_url: &str) -> anyhow::Result<(Vec<RawAccount>, u64)> {
    use solana_client::rpc_client::RpcClient;

    let addresses: Vec<Pubkey> = [WSOL, USDC]
        .into_iter()
        .chain(SOLFI_MARKETS.iter().flat_map(|m| {
            [
                *m,
                get_associated_token_address(m, &WSOL),
                get_associated_token_address(m, &USDC),
            ]
        }))
        .collect();

    let client = RpcClient::new_with_commitment(rpc_url.to_string(), CommitmentConfig::processed());
    let resp = client.get_multiple_accounts_with_commitment(&addresses, CommitmentConfig::processed())?;

    let accounts = resp
        .value
        .into_iter()
        .zip(addresses)
        .filter_map(|(acct, addr)| acct.map(|a| (addr, a)))
        .collect();

    Ok((accounts, resp.context.slot))
}

pub fn simulate_solfi_hops(
    snapshot: &SolFiSnapshot,
    route_plan: &serde_json::Value,
) -> Vec<SolFiHopResult> {
    let hops = extract_solfi_hops(route_plan);
    hops.into_iter()
        .map(|(market, direction, amount_in)| {
            match solfi_sim::cmd::simulate_with_accounts(
                &snapshot.accounts,
                market,
                direction,
                amount_in,
                Some(snapshot.slot),
            ) {
                Ok(amount_out) => SolFiHopResult {
                    market,
                    amount_in,
                    amount_out,
                    success: true,
                    error: None,
                },
                Err(e) => SolFiHopResult {
                    market,
                    amount_in,
                    amount_out: 0,
                    success: false,
                    error: Some(e.to_string()),
                },
            }
        })
        .collect()
}

const WSOL_STR: &str = "So11111111111111111111111111111111111111112";

fn extract_solfi_hops(
    route_plan: &serde_json::Value,
) -> Vec<(Pubkey, SwapDirection, u64)> {
    route_plan
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|hop| {
            let swap_info = hop.get("swapInfo")?;
            let label = swap_info.get("label")?.as_str()?;
            let normalized = label.to_lowercase().replace([' ', '_', '-', '.', '/'], "");
            if !matches!(normalized.as_str(), "solfi" | "solfiv2") {
                return None;
            }
            let market: Pubkey = swap_info.get("ammKey")?.as_str()?.parse().ok()?;
            let input_mint = swap_info.get("inputMint")?.as_str()?;
            let amount_in: u64 = swap_info.get("inAmount")?.as_str()?.parse().ok()?;
            let direction = if input_mint == WSOL_STR {
                SwapDirection::SolToUsdc
            } else {
                SwapDirection::UsdcToSol
            };
            Some((market, direction, amount_in))
        })
        .collect()
}
