use clap::Parser;
use pmm_sim::{App, CliArgs, cfg::Cfg};
use tracing::{debug, info};
use tracing_subscriber::{EnvFilter, fmt::time::UtcTime};

fn main() -> eyre::Result<()> {
    // CRITICAL: log to STDERR, not stdout. In `serve` mode the bot reads this
    // process's STDOUT as a newline-delimited JSON IPC channel; any tracing line
    // on stdout corrupts the protocol and the bot fails to parse responses.
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_thread_ids(true)
        .with_line_number(true)
        .with_target(true)
        .with_timer(UtcTime::rfc_3339())
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::default().add_directive(tracing::Level::INFO.into())),
        )
        .init();

    let args = CliArgs::parse();
    let cfg = Cfg::load(args.cmd.setup_path())?;

    info!(cmd = args.cmd.name(), ?args);
    debug!(?cfg);

    App::new(args, cfg).start()
}
