//! MCP/LSP server binary for ripvec semantic search.
//!
//! By default, exposes seven tools over stdin/stdout using the MCP protocol:
//! `search_code`, `search_text`, `find_similar`, `reindex`, `index_status`,
//! `get_repo_map`, and `up_to_date`.
//!
//! With `--lsp`, starts a Language Server Protocol server over stdio instead,
//! providing code intelligence (symbols, definitions, references, hover).
//!
//! Both modes share the same search index and file watcher. Set `RIPVEC_ROOT`
//! to override the default project root (current directory).

mod lsp;
mod result;
mod server;
mod tools;

use rmcp::ServiceExt;
use tracing_subscriber::prelude::*;

/// Start the MCP or LSP server: build the background index, then serve over stdio.
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();

    // Handle --version before anything else (no clap needed for MCP server)
    if args.iter().any(|a| a == "--version" || a == "-V") {
        println!("ripvec-mcp {}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }

    let lsp_mode = args.iter().any(|a| a == "--lsp");

    // Initialize tracing to stderr (both MCP and LSP use stdin/stdout for transport).
    // Uses a reload handle so the filter can be changed at runtime via the log_level tool.
    // Default: RIPVEC_LOG env var, or info for ripvec crates + warn for everything else.
    let env_filter =
        tracing_subscriber::EnvFilter::try_from_env("RIPVEC_LOG").unwrap_or_else(|_| {
            tracing_subscriber::EnvFilter::new("ripvec_mcp=info,ripvec_core=info,warn")
        });

    let (filter_layer, reload_handle) = tracing_subscriber::reload::Layer::new(env_filter);

    let log_buffer = server::LogBuffer::new(100);
    let buffer_layer = server::BufferLayer::new(log_buffer.clone());

    tracing_subscriber::registry()
        .with(filter_layer)
        .with(
            tracing_subscriber::fmt::layer()
                .with_writer(std::io::stderr)
                .with_target(false)
                .compact(),
        )
        .with(buffer_layer)
        .init();

    let mode_label = if lsp_mode { "LSP" } else { "MCP" };
    tracing::info!("{mode_label} mode — on-demand indexing, all tools require explicit root");

    // No default project root. Every tool call must provide a root parameter.
    // The on-demand index path (ensure_root) handles caching, disk cache
    // discovery, and .ripvec/config.toml resolution per-call.
    let root = std::env::var("RIPVEC_ROOT").map_or_else(
        |_| std::env::current_dir().expect("cannot determine current directory"),
        std::path::PathBuf::from,
    );

    let server = server::RipvecServer::new(root, log_buffer, reload_handle);

    // No background indexing. No file watcher. Indices are built on-demand
    // when tools are called with a root parameter, and cached in memory
    // (root_indices) and on disk (~/.cache/ripvec/ or .ripvec/cache/).

    if lsp_mode {
        lsp::run(server).await;
        Ok(())
    } else {
        let service = server
            .serve(rmcp::transport::stdio())
            .await
            .map_err(|e| anyhow::anyhow!("MCP serve error: {e}"))?;
        service.waiting().await?;
        Ok(())
    }
}
