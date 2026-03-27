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
    /// Cached PageRank-weighted repository graph (built after indexing).
    pub repo_graph: Arc<std::sync::RwLock<Option<ripvec_core::repo_map::RepoGraph>>>,
}

impl rmcp::ServerHandler for RipvecServer {
    fn get_info(&self) -> rmcp::model::ServerInfo {
        rmcp::model::ServerInfo::new(
            rmcp::model::ServerCapabilities::builder()
                .enable_tools()
                .enable_resources()
                .build(),
        )
        .with_instructions(
            "Semantic code search and structural analysis. Finds code by MEANING \
             (not text match) using vector embeddings and PageRank dependency graphs.\n\n\
             ## WORKFLOW: broad → narrow\n\
             1. get_repo_map → architecture overview (which files matter most)\n\
             2. search_code / search_text → find specific code by meaning\n\
             3. LSP tools → precise navigation (definitions, references, calls)\n\n\
             ## TOOLS\n\n\
             **get_repo_map** — START HERE for orientation.\n\
             Returns files ranked by structural importance (PageRank on import graph) \
             with signatures, callers, and callees. Use focus_file when editing a \
             specific file to see its dependency neighborhood.\n\n\
             **search_code** — semantic search over code.\n\
             Understands meaning: \"retry logic with exponential backoff\" finds the \
             actual retry implementation even if the code never uses those exact words. \
             Results include full source in fenced code blocks.\n\n\
             **search_text** — semantic search over documentation and comments.\n\n\
             **find_similar** — given a file and line, find structurally similar code.\n\n\
             **reindex** — force re-embedding. Normally auto-updates on file change.\n\n\
             **index_status** — check readiness and chunk count.\n\n\
             ## WHEN TO USE (by language / task)\n\n\
             Rust:\n\
             - \"How does the trait system work here?\" → get_repo_map (shows trait defs + impls)\n\
             - \"Find error handling with thiserror\" → search_code\n\
             - \"What implements EmbedBackend?\" → search_code then LSP findReferences\n\n\
             TypeScript / React:\n\
             - \"Find components that handle form validation\" → search_code\n\
             - \"How are API routes organized?\" → get_repo_map (shows route files ranked)\n\
             - \"Find hooks similar to useAuth\" → find_similar on useAuth.ts\n\n\
             Python / Django / FastAPI:\n\
             - \"Find database migration logic\" → search_code(\"database migration\")\n\
             - \"Which views handle authentication?\" → search_code(\"authentication middleware\")\n\
             - \"Show the project structure\" → get_repo_map\n\n\
             SQL / dbt:\n\
             - \"Find queries that join users and orders\" → search_code(\"join users orders\")\n\
             - \"Which models depend on the staging layer?\" → get_repo_map (PageRank shows \
               downstream dependencies)\n\
             - \"Find slowly changing dimension logic\" → search_code(\"SCD type 2\")\n\n\
             Full-stack:\n\
             - \"How does the frontend call the backend?\" → get_repo_map (shows API boundary)\n\
             - \"Find WebSocket handling\" → search_code(\"WebSocket connection handler\")\n\
             - \"What's similar to the payment processing flow?\" → find_similar\n\n\
             ## COMBINING WITH OTHER TOOLS\n\n\
             search_code results include lsp_location → pass to LSP goToDefinition, \
             findReferences, incomingCalls.\n\
             get_repo_map identifies central files → use LSP documentSymbol for details.\n\
             After search_code, use Grep for exact string matches within the found files.\n\
             Results include full code — review before calling read_file.\n\n\
             ## TIPS\n\
             - Natural language queries work best (not regex or exact code)\n\
             - search_code with --code flag uses CodeRankEmbed (optimized for code)\n\
             - Index auto-updates when files change (2s debounce)\n\
             - focus_file on get_repo_map: great when you're editing a specific file",
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

    async fn list_resources(
        &self,
        _request: Option<rmcp::model::PaginatedRequestParams>,
        _context: rmcp::service::RequestContext<rmcp::service::RoleServer>,
    ) -> Result<rmcp::model::ListResourcesResult, rmcp::ErrorData> {
        let resource = rmcp::model::Resource {
            raw: rmcp::model::RawResource {
                uri: "ripvec://repo-map".to_string(),
                name: "Repository Structure Map".to_string(),
                title: None,
                description: Some("PageRank-weighted structural overview".to_string()),
                mime_type: Some("text/markdown".to_string()),
                size: None,
                icons: None,
                meta: None,
            },
            annotations: None,
        };
        Ok(rmcp::model::ListResourcesResult {
            resources: vec![resource],
            next_cursor: None,
            meta: None,
        })
    }

    async fn read_resource(
        &self,
        request: rmcp::model::ReadResourceRequestParams,
        _context: rmcp::service::RequestContext<rmcp::service::RoleServer>,
    ) -> Result<rmcp::model::ReadResourceResult, rmcp::ErrorData> {
        if request.uri != "ripvec://repo-map" {
            return Err(rmcp::ErrorData::internal_error(
                format!("unknown resource URI: {}", request.uri),
                None,
            ));
        }

        let graph_guard = self
            .repo_graph
            .read()
            .map_err(|e| rmcp::ErrorData::internal_error(e.to_string(), None))?;

        let rendered = match graph_guard.as_ref() {
            Some(graph) => ripvec_core::repo_map::render(graph, 1500, None),
            None => "Repository graph not yet available. Index is still building.".to_string(),
        };

        Ok(rmcp::model::ReadResourceResult::new(vec![
            rmcp::model::ResourceContents::text(rendered, "ripvec://repo-map"),
        ]))
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
        let backends = ripvec_core::backend::detect_backends(model_repo, None)?;
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

    // Build the repository graph (non-fatal — search works without it)
    let graph_root = server.project_root.clone();
    let graph_lock = Arc::clone(&server.repo_graph);
    let graph_result =
        tokio::task::spawn_blocking(move || ripvec_core::repo_map::build_graph(&graph_root)).await;

    match graph_result {
        Ok(Ok(graph)) => {
            let file_count = graph.files.len();
            match graph_lock.write() {
                Ok(mut g) => {
                    *g = Some(graph);
                    eprintln!("[ripvec-mcp] repo graph built ({file_count} files)");
                }
                Err(e) => {
                    error!("repo graph lock poisoned: {e}");
                    eprintln!("[ripvec-mcp] repo graph lock error: {e}");
                }
            }
        }
        Ok(Err(e)) => {
            error!("repo graph build failed: {e}");
            eprintln!("[ripvec-mcp] repo graph error: {e}");
        }
        Err(e) => {
            error!("repo graph task panicked: {e}");
            eprintln!("[ripvec-mcp] repo graph panic: {e}");
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
