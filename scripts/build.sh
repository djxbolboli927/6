#!/usr/bin/env bash
# Build BOTH halves of the system:
#   1. the bot + sim-server  (root workspace, SDK 2.2, no private deps)
#   2. pmm-sim               (standalone workspace, SDK 3.0 + private magnus deps)
#
# pmm-sim is intentionally a SEPARATE Cargo workspace: it pulls the private
# `magnus` GitHub crates, and keeping it out of the root workspace means the bot
# can build without that private access. The downside is it is NOT built by a
# plain `cargo build` at the repo root — this script builds it for you.
#
# The bot launches pmm-sim as a subprocess; config.toml [pmm_sim].binary must
# point at the produced ./pmm-sim/target/release/pmm-sim.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "==> Building bot + sim-server (root workspace)…"
cargo build --release --workspace

echo "==> Building pmm-sim (standalone workspace)…"
cargo build --release --manifest-path pmm-sim/Cargo.toml

BIN="$ROOT_DIR/pmm-sim/target/release/pmm-sim"
if [[ -x "$BIN" ]]; then
    echo "==> OK. pmm-sim binary: $BIN"
    echo "    Set this absolute path in config.toml under [pmm_sim].binary"
else
    echo "!! pmm-sim binary not found at $BIN — check the build output above." >&2
    exit 1
fi
