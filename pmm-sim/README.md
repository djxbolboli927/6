# pmm-sim

Simulation & benchmark environment for Solana's proprietary AMMs, built on LiteSVM.

It can simulate and benchmark swaps across the major Solana prop AMMs locally, and
runs as a JSON-over-stdio server (`pmm-sim serve`) used by the arbitrage bot to:

- simulate full Metis transactions in LiteSVM, and
- build + simulate a single BisonFi `WSOL→USDC` DFlow `swap2` leg (the `build_bison`
  IPC op) for the first-test flow, returning both the predicted output and the
  ready-to-send instruction.

See `cfg/setup.toml` for pool/market configuration and `cfg/programs/` for the
on-chain program `.so` files loaded into the simulator at startup.
