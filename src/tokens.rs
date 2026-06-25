use anyhow::Result;
use std::path::Path;

/// Base mint for arbitrage cycles (Wrapped SOL).
pub const WSOL_MINT: &str = "So11111111111111111111111111111111111111112";

/// USDC mint — quote leg of the BisonFi first-test pool.
pub const USDC_MINT: &str = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v";

/// Read token mint addresses from a file (one per line).
/// Blank lines and lines starting with '#' are ignored.
pub fn load_tokens<P: AsRef<Path>>(path: P) -> Result<Vec<String>> {
    let content = std::fs::read_to_string(path)?;
    let tokens: Vec<String> = content
        .lines()
        .map(|l| l.split('#').next().unwrap_or("").trim().to_string())
        .filter(|l| !l.is_empty())
        .collect();
    anyhow::ensure!(!tokens.is_empty(), "tokens file is empty");
    Ok(tokens)
}
