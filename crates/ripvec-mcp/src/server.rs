//! MCP server state and background indexing.
//!
//! [`RipvecServer`] holds the shared search index, embedding backends,
//! and tokenizers. Background indexing runs on startup and on `reindex`.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use rmcp::handler::server::tool::ToolRouter;
use tokio::sync::{OnceCell, RwLock};
use tracing::error;

/// RAII guard that resets the indexing flag on drop (including panics).
struct IndexingGuard(Arc<AtomicBool>);

impl Drop for IndexingGuard {
    fn drop(&mut self) {
        self.0.store(false, Ordering::SeqCst);
    }
}

/// The ripvec MCP server, holding shared backend and tokenizer state.
///
/// All fields use `Arc` / `OnceCell` / `RwLock` so the struct is cheaply
/// cloneable and safe to share across async tasks.
#[derive(Clone)]
pub struct RipvecServer {
    /// Pre-built search index (populated after background indexing).
    pub index: Arc<RwLock<Option<ripvec_core::index::SearchIndex>>>,
    /// Text embedding backend (BGE-small), lazy-loaded on first search.
    pub text_backend: Arc<OnceCell<Arc<dyn ripvec_core::backend::EmbedBackend>>>,
    /// Tokenizer for the text model, lazy-loaded.
    pub text_tokenizer: Arc<OnceCell<Arc<tokenizers::Tokenizer>>>,
    /// Root directory of the project being indexed.
    pub project_root: PathBuf,
    /// Whether background indexing is currently in progress.
    pub indexing: Arc<AtomicBool>,
    /// Generated tool router mapping tool names to handlers.
    pub tool_router: ToolRouter<Self>,
}

impl rmcp::ServerHandler for RipvecServer {
    fn get_info(&self) -> rmcp::model::ServerInfo {
        rmcp::model::ServerInfo::new(
            rmcp::model::ServerCapabilities::builder()
                .enable_tools()
                .build(),
        )
        .with_instructions(
            "Semantic code search server (ripvec). Tools: \
             search_code (semantic code search with BGE embeddings), \
             search_text (general text search with BGE embeddings), \
             find_similar (find chunks similar to a given location), \
             reindex (rebuild the search index), \
             index_status (check indexing state).",
        )
    }

    fn call_tool(
        &self,
        request: rmcp::model::CallToolRequestParams,
        context: rmcp::service::RequestContext<rmcp::service::RoleServer>,
    ) -> impl std::future::Future<Output = Result<rmcp::model::CallToolResult, rmcp::ErrorData>>
    + Send
    + '_ {
        let ctx = rmcp::handler::server::tool::ToolCallContext::new(self, request, context);
        async move { self.tool_router.call(ctx).await }
    }

    fn list_tools(
        &self,
        _request: Option<rmcp::model::PaginatedRequestParams>,
        _context: rmcp::service::RequestContext<rmcp::service::RoleServer>,
    ) -> impl std::future::Future<Output = Result<rmcp::model::ListToolsResult, rmcp::ErrorData>>
    + Send
    + '_ {
        let tools = self.tool_router.list_all();
        async move {
            Ok(rmcp::model::ListToolsResult {
                tools,
                next_cursor: None,
                meta: None,
            })
        }
    }
}

/// Run background indexing: walk, chunk, embed, and build the search index.
///
/// Sets `server.indexing` to `true` during the operation and `false` when done.
/// The indexing flag is managed via an RAII guard so it resets even on panic.
/// Errors are logged to stderr rather than propagated.
pub async fn run_background_index(server: &RipvecServer) {
    if server
        .indexing
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        // Already indexing
        return;
    }

    // RAII guard ensures the flag resets even if we panic
    let _guard = IndexingGuard(Arc::clone(&server.indexing));

    let root = server.project_root.clone();
    let index_lock = Arc::clone(&server.index);

    let result = tokio::task::spawn_blocking(move || {
        let backends = ripvec_core::backend::detect_backends("BAAI/bge-small-en-v1.5")?;
        let tokenizer = ripvec_core::tokenize::load_tokenizer("BAAI/bge-small-en-v1.5")?;
        let cfg = ripvec_core::embed::SearchConfig::default();
        let profiler = ripvec_core::profile::Profiler::noop();

        let backend_refs: Vec<&dyn ripvec_core::backend::EmbedBackend> =
            backends.iter().map(AsRef::as_ref).collect();
        let (chunks, embeddings) =
            ripvec_core::embed::embed_all(&root, &backend_refs, &tokenizer, &cfg, &profiler)?;

        let search_index = ripvec_core::index::SearchIndex::new(chunks, &embeddings);
        Ok::<_, ripvec_core::Error>(search_index)
    })
    .await;

    match result {
        Ok(Ok(new_index)) => {
            eprintln!(
                "[ripvec-mcp] indexed {} chunks from {}",
                new_index.chunks.len(),
                server.project_root.display()
            );
            let mut idx = index_lock.write().await;
            *idx = Some(new_index);
        }
        Ok(Err(e)) => {
            error!("background indexing failed: {e}");
            eprintln!("[ripvec-mcp] indexing error: {e}");
        }
        Err(e) => {
            error!("background indexing task panicked: {e}");
            eprintln!("[ripvec-mcp] indexing panic: {e}");
        }
    }

    // Guard drop resets indexing flag automatically
}
