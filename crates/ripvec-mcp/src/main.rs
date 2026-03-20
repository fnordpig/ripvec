//! MCP server binary for ripvec semantic search.
//!
//! Exposes five tools over stdin/stdout using the MCP protocol:
//! `search_code`, `search_text`, `find_similar`, `reindex`, and `index_status`.
//!
//! The search index is built in the background on startup. Set `RIPVEC_ROOT`
//! to override the default project root (current directory).

mod result;
mod server;
mod tools;

use rmcp::ServiceExt;

/// Start the MCP server: build the background index, then serve over stdio.
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing to stderr (MCP uses stdin/stdout for transport)
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();

    let root = std::env::var("RIPVEC_ROOT").map_or_else(
        |_| std::env::current_dir().expect("cannot determine current directory"),
        std::path::PathBuf::from,
    );

    eprintln!("[ripvec-mcp] project root: {}", root.display());

    let server = server::RipvecServer::new(root);

    // Spawn background indexing (non-blocking)
    let bg_server = server.clone();
    tokio::spawn(async move {
        server::run_background_index(&bg_server).await;
    });

    let service = server
        .serve(rmcp::transport::stdio())
        .await
        .map_err(|e| anyhow::anyhow!("MCP serve error: {e}"))?;
    service.waiting().await?;
    Ok(())
}
