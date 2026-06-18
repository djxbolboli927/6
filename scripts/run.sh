#!/usr/bin/env bash
# One-command launcher for the arbitrage bot.
#
# You run ONLY this. The bot automatically spawns pmm-sim as a child subprocess
# (communication over a local stdin/stdout pipe — microsecond latency, started
# once and kept warm), so there is NO separate window and NO extra latency from
# pmm-sim being a separate build. pmm-sim is its own Cargo workspace (private
# magnus deps), so the bot's `cargo run` does NOT build it — this script makes
# sure its binary exists first.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Raise the open-file limit (many concurrent sockets to Metis/Jito/Yellowstone).
ulimit -n 1000000 || true

# Ensure the pmm-sim binary exists (config.toml [pmm_sim].binary points at it).
PMM_BIN="$ROOT_DIR/pmm-sim/target/release/pmm-sim"
if [[ ! -x "$PMM_BIN" ]]; then
    echo "==> pmm-sim binary missing — building it once (this can take a few minutes)…"
    cargo build --release --manifest-path "$ROOT_DIR/pmm-sim/Cargo.toml"
fi

echo "==> Starting bot (it spawns pmm-sim automatically)…"
exec env RUST_LOG="${RUST_LOG:-info}" cargo run --release
