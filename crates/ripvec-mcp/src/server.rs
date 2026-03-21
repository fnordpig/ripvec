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
        let model_repo = "BAAI/bge-small-en-v1.5";
        let backends = ripvec_core::backend::detect_backends(model_repo)?;
        let tokenizer = ripvec_core::tokenize::load_tokenizer(model_repo)?;
        let cfg = ripvec_core::embed::SearchConfig::default();
        let profiler = ripvec_core::profile::Profiler::noop();

        let backend_refs: Vec<&dyn ripvec_core::backend::EmbedBackend> =
            backends.iter().map(AsRef::as_ref).collect();

        // Use persistent cache for instant restarts
        let (index, stats) = ripvec_core::cache::reindex::incremental_index(
            &root,
            &backend_refs,
            &tokenizer,
            &cfg,
            &profiler,
            model_repo,
            None,
        )?;

        Ok::<_, ripvec_core::Error>((index, stats))
    })
    .await;

    match result {
        Ok(Ok((new_index, stats))) => {
            eprintln!(
                "[ripvec-mcp] indexed {} chunks ({} cached, {} re-embedded, {}ms)",
                stats.chunks_total,
                stats.files_unchanged,
                stats.chunks_reembedded,
                stats.duration_ms,
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

/// Watch the project directory for file changes and re-index after a 2-second quiet period.
///
/// Uses the `notify` crate for OS-level file watching (`FSEvents` on macOS,
/// inotify on Linux). After detecting changes, waits 2 seconds for quiet
/// then runs `run_background_index` to incrementally update the index.
pub async fn run_file_watcher(server: &RipvecServer) {
    use notify::{RecursiveMode, Watcher};

    let (tx, mut rx) = tokio::sync::mpsc::channel::<()>(100);

    let _watcher = match notify::recommended_watcher(move |res: notify::Result<notify::Event>| {
        if let Ok(event) = res {
            use notify::EventKind;
            if matches!(
                event.kind,
                EventKind::Create(_) | EventKind::Modify(_) | EventKind::Remove(_)
            ) {
                let _ = tx.blocking_send(());
            }
        }
    }) {
        Ok(mut w) => {
            if let Err(e) = w.watch(&server.project_root, RecursiveMode::Recursive) {
                eprintln!(
                    "[ripvec-mcp] file watcher failed to watch {}: {e}",
                    server.project_root.display()
                );
                return;
            }
            eprintln!(
                "[ripvec-mcp] file watcher active on {}",
                server.project_root.display()
            );
            w
        }
        Err(e) => {
            eprintln!("[ripvec-mcp] file watcher failed to start: {e}");
            return;
        }
    };

    while rx.recv().await.is_some() {
        // Debounce: wait 2 seconds for quiet period
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;

        // Drain any events that arrived during the debounce window
        while rx.try_recv().is_ok() {}

        eprintln!("[ripvec-mcp] changes detected, re-indexing...");
        run_background_index(server).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_indexing_guard_resets_on_drop() {
        let flag = Arc::new(AtomicBool::new(true));
        let guard = IndexingGuard(Arc::clone(&flag));
        assert!(flag.load(Ordering::SeqCst));
        drop(guard);
        assert!(!flag.load(Ordering::SeqCst));
    }

    #[test]
    fn test_indexing_guard_resets_on_panic() {
        let flag = Arc::new(AtomicBool::new(true));
        let flag_clone = Arc::clone(&flag);
        let handle = std::thread::spawn(move || {
            let _guard = IndexingGuard(flag_clone);
            panic!("intentional panic to test RAII guard");
        });
        // The thread panicked, but the guard's Drop should have run
        let _ = handle.join();
        assert!(!flag.load(Ordering::SeqCst));
    }
}
