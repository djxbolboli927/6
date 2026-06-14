use anyhow::Result;
use solana_sdk::signature::Keypair;
use std::path::Path;

/// Read a Solana keypair from a JSON file (byte-array format).
pub fn read_keypair<P: AsRef<Path>>(path: P) -> Result<Keypair> {
    let data = std::fs::read_to_string(path)?;
    let bytes: Vec<u8> = serde_json::from_str(&data)?;
    let keypair = Keypair::from_bytes(bytes.as_slice())
        .map_err(|e| anyhow::anyhow!("invalid keypair: {}", e))?;
    Ok(keypair)
}
