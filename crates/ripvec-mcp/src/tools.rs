//! MCP tool handlers for semantic search, similarity, and index management.
//!
//! All five tools are defined in a single `#[tool_router]` impl block on
//! [`RipvecServer`]. Each tool returns JSON via `CallToolResult::success`.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Instant;

use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::{CallToolResult, Content};
use rmcp::{tool, tool_router};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::result::{SearchResponse, SearchResultItem};
use crate::server::{RipvecServer, run_background_index};

/// Parameters for the `search_code` and `search_text` tools.
#[derive(Deserialize, JsonSchema)]
pub struct SearchParams {
    /// Natural-language query describing the code or text to find.
    pub query: String,
    /// Maximum number of results to return.
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    /// Minimum similarity threshold (0.0 to 1.0).
    #[serde(default = "default_threshold")]
    pub threshold: f32,
}

/// Parameters for the `find_similar` tool.
#[derive(Deserialize, JsonSchema)]
pub struct FindSimilarParams {
    /// Path to the source file (relative or absolute).
    pub file_path: String,
    /// 0-based line number within the file.
    pub line: usize,
    /// Maximum number of results to return.
    #[serde(default = "default_top_k")]
    pub top_k: usize,
}

fn default_top_k() -> usize {
    10
}

fn default_threshold() -> f32 {
    0.3
}

#[tool_router]
impl RipvecServer {
    /// Create a new server for the given project root.
    ///
    /// The search index starts empty; call [`run_background_index`] to populate it.
    pub fn new(project_root: std::path::PathBuf) -> Self {
        Self {
            index: Arc::new(tokio::sync::RwLock::new(None)),
            chunks: Arc::new(tokio::sync::RwLock::new(Vec::new())),
            code_backend: Arc::new(tokio::sync::OnceCell::new()),
            text_backend: Arc::new(tokio::sync::OnceCell::new()),
            code_tokenizer: Arc::new(tokio::sync::OnceCell::new()),
            text_tokenizer: Arc::new(tokio::sync::OnceCell::new()),
            project_root,
            indexing: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            tool_router: Self::tool_router(),
        }
    }

    /// Search code semantically using `CodeRankEmbed` embeddings.
    ///
    /// Prepends a code-search query prefix and ranks indexed chunks by
    /// cosine similarity.
    #[tool(
        name = "search_code",
        description = "Search code by semantic meaning using CodeRankEmbed. Returns ranked code chunks with LSP locations."
    )]
    async fn search_code(
        &self,
        Parameters(params): Parameters<SearchParams>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        // Ensure index is ready
        let idx_guard = self.index.read().await;
        let index = idx_guard.as_ref().ok_or_else(|| {
            if self.indexing.load(Ordering::SeqCst) {
                rmcp::ErrorData::internal_error(
                    "Index is still building. Try again shortly.".to_string(),
                    None,
                )
            } else {
                rmcp::ErrorData::internal_error(
                    "No index available. Call reindex first.".to_string(),
                    None,
                )
            }
        })?;

        // Lazy-load code backend (CodeRankEmbed / nomic-ai/CodeRankEmbed)
        let code_backend = self
            .code_backend
            .get_or_init(|| async {
                let backends = tokio::task::spawn_blocking(|| {
                    ripvec_core::backend::detect_backends("nomic-ai/CodeRankEmbed")
                })
                .await
                .expect("spawn_blocking panicked")
                .expect("failed to load CodeRankEmbed backend");
                Arc::from(backends.into_iter().next().expect("no backend available"))
            })
            .await;

        let code_tokenizer = self
            .code_tokenizer
            .get_or_init(|| async {
                let t = tokio::task::spawn_blocking(|| {
                    ripvec_core::tokenize::load_tokenizer("nomic-ai/CodeRankEmbed")
                })
                .await
                .expect("spawn_blocking panicked")
                .expect("failed to load CodeRankEmbed tokenizer");
                Arc::new(t)
            })
            .await;

        // Prepend code search query prefix
        let prefixed_query = format!(
            "Represent this query for searching relevant code: {}",
            params.query
        );

        let backend = Arc::clone(code_backend);
        let tokenizer = Arc::clone(code_tokenizer);
        let threshold = params.threshold;
        let top_k = params.top_k;

        let query_embedding = tokio::task::spawn_blocking(move || {
            let max_tokens = backend.max_tokens();
            let enc =
                ripvec_core::tokenize::tokenize_query(&prefixed_query, &tokenizer, max_tokens)?;
            let mut results = backend.embed_batch(&[enc])?;
            results.pop().ok_or_else(|| {
                ripvec_core::Error::Other(anyhow::anyhow!(
                    "backend returned no embedding for query"
                ))
            })
        })
        .await
        .map_err(|e| rmcp::ErrorData::internal_error(e.to_string(), None))?
        .map_err(|e| rmcp::ErrorData::internal_error(e.to_string(), None))?;

        let ranked = index.rank(&query_embedding, threshold);
        let chunks_guard = self.chunks.read().await;

        let results: Vec<SearchResultItem> = ranked
            .into_iter()
            .take(top_k)
            .filter_map(|(idx, score)| {
                chunks_guard
                    .get(idx)
                    .map(|chunk| SearchResultItem::from_chunk(chunk, score))
            })
            .collect();

        let response = SearchResponse { results };
        let json = serde_json::to_string_pretty(&response)
            .map_err(|e| rmcp::ErrorData::internal_error(e.to_string(), None))?;

        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    /// Search text semantically using BGE-small embeddings.
    ///
    /// Uses the general-purpose BGE model without a query prefix.
    #[tool(
        name = "search_text",
        description = "Search text by semantic meaning using BGE-small. Returns ranked text chunks with LSP locations."
    )]
    async fn search_text(
        &self,
        Parameters(params): Parameters<SearchParams>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let idx_guard = self.index.read().await;
        let index = idx_guard.as_ref().ok_or_else(|| {
            if self.indexing.load(Ordering::SeqCst) {
                rmcp::ErrorData::internal_error(
                    "Index is still building. Try again shortly.".to_string(),
                    None,
                )
            } else {
                rmcp::ErrorData::internal_error(
                    "No index available. Call reindex first.".to_string(),
                    None,
                )
            }
        })?;

        // Lazy-load text backend (BGE-small)
        let text_backend = self
            .text_backend
            .get_or_init(|| async {
                let backends = tokio::task::spawn_blocking(|| {
                    ripvec_core::backend::detect_backends("BAAI/bge-small-en-v1.5")
                })
                .await
                .expect("spawn_blocking panicked")
                .expect("failed to load BGE backend");
                Arc::from(backends.into_iter().next().expect("no backend available"))
            })
            .await;

        let text_tokenizer = self
            .text_tokenizer
            .get_or_init(|| async {
                let t = tokio::task::spawn_blocking(|| {
                    ripvec_core::tokenize::load_tokenizer("BAAI/bge-small-en-v1.5")
                })
                .await
                .expect("spawn_blocking panicked")
                .expect("failed to load BGE tokenizer");
                Arc::new(t)
            })
            .await;

        let backend = Arc::clone(text_backend);
        let tokenizer = Arc::clone(text_tokenizer);
        let threshold = params.threshold;
        let top_k = params.top_k;
        let query = params.query;

        let query_embedding = tokio::task::spawn_blocking(move || {
            let max_tokens = backend.max_tokens();
            let enc = ripvec_core::tokenize::tokenize_query(&query, &tokenizer, max_tokens)?;
            let mut results = backend.embed_batch(&[enc])?;
            results.pop().ok_or_else(|| {
                ripvec_core::Error::Other(anyhow::anyhow!(
                    "backend returned no embedding for query"
                ))
            })
        })
        .await
        .map_err(|e| rmcp::ErrorData::internal_error(e.to_string(), None))?
        .map_err(|e| rmcp::ErrorData::internal_error(e.to_string(), None))?;

        let ranked = index.rank(&query_embedding, threshold);
        let chunks_guard = self.chunks.read().await;

        let results: Vec<SearchResultItem> = ranked
            .into_iter()
            .take(top_k)
            .filter_map(|(idx, score)| {
                chunks_guard
                    .get(idx)
                    .map(|chunk| SearchResultItem::from_chunk(chunk, score))
            })
            .collect();

        let response = SearchResponse { results };
        let json = serde_json::to_string_pretty(&response)
            .map_err(|e| rmcp::ErrorData::internal_error(e.to_string(), None))?;

        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    /// Find chunks similar to the chunk at a given file location.
    ///
    /// Locates the chunk containing the specified 0-based line, retrieves
    /// its embedding from the index, and ranks all other chunks against it.
    #[tool(
        name = "find_similar",
        description = "Find code chunks similar to the chunk at a given file path and line number (0-based)."
    )]
    async fn find_similar(
        &self,
        Parameters(params): Parameters<FindSimilarParams>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let idx_guard = self.index.read().await;
        let index = idx_guard.as_ref().ok_or_else(|| {
            rmcp::ErrorData::internal_error("No index available.".to_string(), None)
        })?;

        let chunks_guard = self.chunks.read().await;

        // Convert 0-based input line to 1-based chunk lines
        let target_line = params.line + 1;

        // Find the chunk containing this location
        let source_idx = chunks_guard
            .iter()
            .position(|chunk| {
                chunk.file_path.ends_with(&params.file_path)
                    && chunk.start_line <= target_line
                    && target_line <= chunk.end_line
            })
            .ok_or_else(|| {
                rmcp::ErrorData::internal_error(
                    format!("No chunk found at {}:{}", params.file_path, params.line),
                    None,
                )
            })?;

        // Get the source chunk's embedding
        let source_embedding = index.embedding(source_idx).ok_or_else(|| {
            rmcp::ErrorData::internal_error(
                "Embedding not found for source chunk.".to_string(),
                None,
            )
        })?;

        // Rank all chunks against the source embedding
        let ranked = index.rank(&source_embedding, 0.0);

        let results: Vec<SearchResultItem> = ranked
            .into_iter()
            .filter(|(idx, _)| *idx != source_idx) // Exclude the source chunk
            .take(params.top_k)
            .filter_map(|(idx, score)| {
                chunks_guard
                    .get(idx)
                    .map(|chunk| SearchResultItem::from_chunk(chunk, score))
            })
            .collect();

        let response = SearchResponse { results };
        let json = serde_json::to_string_pretty(&response)
            .map_err(|e| rmcp::ErrorData::internal_error(e.to_string(), None))?;

        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    /// Rebuild the search index from scratch.
    ///
    /// Clears the current index, re-walks and re-embeds all files, then
    /// returns statistics about the new index. Blocks until indexing is done.
    #[tool(
        name = "reindex",
        description = "Rebuild the search index from scratch. Returns chunk and file counts when done."
    )]
    async fn reindex(&self) -> Result<CallToolResult, rmcp::ErrorData> {
        // Clear current index
        {
            let mut idx = self.index.write().await;
            *idx = None;
            let mut ch = self.chunks.write().await;
            ch.clear();
        }

        let start = Instant::now();
        run_background_index(self).await;
        let duration_ms = start.elapsed().as_millis();

        let chunks_guard = self.chunks.read().await;
        let chunk_count = chunks_guard.len();
        let file_count = chunks_guard
            .iter()
            .map(|c| c.file_path.as_str())
            .collect::<HashSet<_>>()
            .len();

        let response = serde_json::json!({
            "chunks": chunk_count,
            "files": file_count,
            "duration_ms": duration_ms,
        });

        let json = serde_json::to_string_pretty(&response)
            .map_err(|e| rmcp::ErrorData::internal_error(e.to_string(), None))?;

        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    /// Return the current index status.
    ///
    /// Always available, even during background indexing. Returns readiness
    /// state, chunk/file counts, file extension breakdown, and project root.
    #[tool(
        name = "index_status",
        description = "Check the current index status: readiness, chunk/file counts, and project root."
    )]
    async fn index_status(&self) -> Result<CallToolResult, rmcp::ErrorData> {
        let is_indexing = self.indexing.load(Ordering::SeqCst);
        let idx_guard = self.index.read().await;
        let ready = idx_guard.is_some();

        let chunks_guard = self.chunks.read().await;
        let chunk_count = chunks_guard.len();

        let mut files_set = HashSet::new();
        let mut ext_counts: HashMap<String, usize> = HashMap::new();
        for chunk in chunks_guard.iter() {
            files_set.insert(chunk.file_path.as_str());
            let ext = std::path::Path::new(&chunk.file_path)
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("(none)")
                .to_string();
            *ext_counts.entry(ext).or_insert(0) += 1;
        }

        let response = serde_json::json!({
            "ready": ready,
            "indexing": is_indexing,
            "chunks": chunk_count,
            "files": files_set.len(),
            "extensions": ext_counts,
            "project_root": self.project_root.display().to_string(),
        });

        let json = serde_json::to_string_pretty(&response)
            .map_err(|e| rmcp::ErrorData::internal_error(e.to_string(), None))?;

        Ok(CallToolResult::success(vec![Content::text(json)]))
    }
}
