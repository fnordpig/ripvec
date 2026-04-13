//! MCP tool handlers for semantic search, similarity, and index management.
//!
//! All seven tools are defined in a single `#[tool_router]` impl block on
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

/// Deserialize a value that may arrive as either a number or a string.
fn deserialize_number_or_string<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: serde::Deserializer<'de>,
    T: std::str::FromStr + serde::Deserialize<'de>,
    <T as std::str::FromStr>::Err: std::fmt::Display,
{
    use serde::de::Error;
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum StringOrNum<T> {
        Num(T),
        Str(String),
    }
    match StringOrNum::<T>::deserialize(deserializer)? {
        StringOrNum::Num(v) => Ok(v),
        StringOrNum::Str(s) => s.parse().map_err(D::Error::custom),
    }
}

/// Parameters for the `search_code` and `search_text` tools.
#[derive(Deserialize, JsonSchema)]
pub struct SearchParams {
    /// Natural-language query describing the code or text to find.
    pub query: String,
    /// Maximum number of results to return.
    #[serde(
        default = "default_top_k",
        deserialize_with = "deserialize_number_or_string"
    )]
    pub top_k: usize,
    /// Minimum similarity threshold (0.0 to 1.0).
    #[serde(
        default = "default_threshold",
        deserialize_with = "deserialize_number_or_string"
    )]
    pub threshold: f32,
    /// Skip the first N results (for pagination). Default: 0.
    #[serde(
        default = "default_offset",
        deserialize_with = "deserialize_number_or_string"
    )]
    pub offset: usize,
    /// Project root directory to search. Uses the pre-built index if available
    /// for this path, otherwise falls back to the server's default project root.
    /// Accepts absolute paths or paths relative to the server's project root.
    pub root: Option<String>,
}

/// Parameters for the `find_similar` tool.
#[derive(Deserialize, JsonSchema)]
pub struct FindSimilarParams {
    /// Path to the source file (relative or absolute).
    pub file_path: String,
    /// 0-based line number within the file.
    #[serde(deserialize_with = "deserialize_number_or_string")]
    pub line: usize,
    /// Maximum number of results to return.
    #[serde(
        default = "default_top_k",
        deserialize_with = "deserialize_number_or_string"
    )]
    pub top_k: usize,
    /// Skip the first N results (for pagination). Default: 0.
    #[serde(
        default = "default_offset",
        deserialize_with = "deserialize_number_or_string"
    )]
    pub offset: usize,
    /// Project root directory. Uses the server's default project root if omitted.
    pub root: Option<String>,
}

/// Default number of results to return.
fn default_top_k() -> usize {
    10
}

/// Default similarity threshold.
fn default_threshold() -> f32 {
    0.3
}

/// Default pagination offset.
fn default_offset() -> usize {
    0
}

/// Parameters for the `get_repo_map` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct RepoMapParams {
    /// Maximum tokens in output (default: 2000).
    #[serde(
        default = "default_repo_map_tokens",
        deserialize_with = "deserialize_number_or_string"
    )]
    pub max_tokens: usize,
    /// Focus file for topic-sensitive `PageRank` (relative path).
    pub focus_file: Option<String>,
    /// Project root directory. Uses the server's default project root if omitted.
    pub root: Option<String>,
}

/// Parameters for the `find_duplicates` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct FindDuplicatesParams {
    /// Project root directory to search. Uses the server's default project root if omitted.
    pub root: Option<String>,
    /// Minimum cosine similarity threshold for a pair to be considered a duplicate.
    /// Default: 0.85. Higher values (0.90+) find near-exact copies; lower values
    /// (0.75-0.85) find similar patterns that may be refactorable.
    pub threshold: Option<f32>,
    /// Maximum number of duplicate pairs to return. Default: 20.
    pub max_pairs: Option<usize>,
}

/// Parameters for the `reindex` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ReindexParams {
    /// Project root directory to reindex. Uses the server's default project root if omitted.
    pub root: Option<String>,
    /// Store the index in `.ripvec/cache/` at the project root (repo-local).
    /// Creates `.ripvec/config.toml` on first use. Commit the `.ripvec/` directory
    /// to git so teammates get instant search without re-embedding.
    #[serde(default)]
    pub repo_level: bool,
}

/// Parameters for the `index_status` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct IndexStatusParams {
    /// Project root directory to check. Uses the server's default project root if omitted.
    pub root: Option<String>,
}

/// Parameters for the `debug_log` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct DebugLogParams {
    /// Maximum number of recent log lines to return. Default: 50.
    #[serde(default = "default_debug_log_lines")]
    pub lines: usize,
}

/// Default number of log lines to return.
fn default_debug_log_lines() -> usize {
    50
}

/// Parameters for the `log_level` tool.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct LogLevelParams {
    /// Filter directive. Examples: "debug", "warn", "ripvec_mcp=debug,ripvec_core=trace".
    pub level: String,
}

/// Default token budget for repo map rendering.
fn default_repo_map_tokens() -> usize {
    2000
}

#[tool_router]
impl RipvecServer {
    /// Create a new server for the given project root.
    ///
    /// The search index starts empty; call [`run_background_index`] to populate it.
    pub fn new(
        project_root: std::path::PathBuf,
        log_buffer: crate::server::LogBuffer,
        reload_handle: crate::server::FilterReloadHandle,
    ) -> Self {
        Self {
            index: Arc::new(tokio::sync::RwLock::new(None)),
            text_backend: Arc::new(tokio::sync::OnceCell::new()),
            text_tokenizer: Arc::new(tokio::sync::OnceCell::new()),
            project_root,
            indexing: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            tool_router: Self::tool_router(),
            repo_graph: Arc::new(std::sync::RwLock::new(None)),
            root_indices: Arc::new(tokio::sync::RwLock::new(std::collections::HashMap::new())),
            root_graphs: Arc::new(std::sync::RwLock::new(std::collections::HashMap::new())),
            progress: Arc::new(crate::server::IndexProgress::default()),
            log_buffer,
            reload_handle: Arc::new(reload_handle),
        }
    }

    /// Lazy-load the text embedding backend (BGE-small).
    ///
    /// Returns the cached backend on subsequent calls.
    async fn load_text_backend(
        &self,
    ) -> Result<&Arc<dyn ripvec_core::backend::EmbedBackend>, rmcp::ErrorData> {
        self.text_backend
            .get_or_try_init(|| async {
                let backends = tokio::task::spawn_blocking(|| {
                    ripvec_core::backend::detect_backends("nomic-ai/modernbert-embed-base")
                })
                .await
                .map_err(|e| rmcp::ErrorData::internal_error(e.to_string(), None))?
                .map_err(|e| rmcp::ErrorData::internal_error(e.to_string(), None))?;
                let first = backends.into_iter().next().ok_or_else(|| {
                    rmcp::ErrorData::internal_error("no backend available".to_string(), None)
                })?;
                Ok(Arc::from(first))
            })
            .await
    }

    /// Lazy-load the text tokenizer (BGE-small).
    ///
    /// Returns the cached tokenizer on subsequent calls.
    async fn load_text_tokenizer(&self) -> Result<&Arc<tokenizers::Tokenizer>, rmcp::ErrorData> {
        self.text_tokenizer
            .get_or_try_init(|| async {
                let t = tokio::task::spawn_blocking(|| {
                    ripvec_core::tokenize::load_tokenizer("nomic-ai/modernbert-embed-base")
                })
                .await
                .map_err(|e| rmcp::ErrorData::internal_error(e.to_string(), None))?
                .map_err(|e| rmcp::ErrorData::internal_error(e.to_string(), None))?;
                Ok(Arc::new(t))
            })
            .await
    }

    /// Resolve a `root` parameter to a canonical path, ensuring the index
    /// and repo graph are cached for that root. Returns `None` if root is
    /// absent (use default project root).
    async fn ensure_root(
        &self,
        root: Option<&str>,
        repo_level: bool,
    ) -> Result<Option<std::path::PathBuf>, rmcp::ErrorData> {
        let Some(root_str) = root else {
            return Ok(None);
        };
        let root_path = std::path::PathBuf::from(root_str);
        if !root_path.is_dir() {
            return Err(rmcp::ErrorData::internal_error(
                format!("root is not a directory: {root_str}"),
                None,
            ));
        }
        let canonical = root_path
            .canonicalize()
            .unwrap_or_else(|_| root_path.clone());

        // Ensure index is cached for this root.
        {
            let cache = self.root_indices.read().await;
            if !cache.contains_key(&canonical) {
                drop(cache);
                tracing::info!(root = %canonical.display(), "ensure_root: building index");
                let text_backend = self.load_text_backend().await?;
                let text_tokenizer = self.load_text_tokenizer().await?;
                let backend = Arc::clone(text_backend);
                let tokenizer = Arc::clone(text_tokenizer);
                let canon2 = canonical.clone();
                let (idx, _stats) = tokio::task::spawn_blocking(move || {
                    let backend_refs: Vec<&dyn ripvec_core::backend::EmbedBackend> =
                        vec![backend.as_ref()];
                    let cfg = ripvec_core::embed::SearchConfig::default();
                    let profiler = ripvec_core::profile::Profiler::noop();
                    ripvec_core::cache::reindex::incremental_index(
                        &canon2,
                        &backend_refs,
                        &tokenizer,
                        &cfg,
                        &profiler,
                        "nomic-ai/modernbert-embed-base",
                        None,
                        repo_level,
                    )
                })
                .await
                .map_err(|e| rmcp::ErrorData::internal_error(e.to_string(), None))?
                .map_err(|e| rmcp::ErrorData::internal_error(e.to_string(), None))?;
                let mut cache = self.root_indices.write().await;
                cache.insert(canonical.clone(), idx);
            }
        }

        // Ensure repo graph is cached for this root.
        {
            let has_graph = self
                .root_graphs
                .read()
                .map_err(|e| rmcp::ErrorData::internal_error(e.to_string(), None))?
                .contains_key(&canonical);
            if !has_graph {
                let canon2 = canonical.clone();
                let graph = tokio::task::spawn_blocking(move || {
                    ripvec_core::repo_map::build_graph(&canon2)
                })
                .await
                .map_err(|e| rmcp::ErrorData::internal_error(e.to_string(), None))?
                .map_err(|e| rmcp::ErrorData::internal_error(e.to_string(), None))?;
                self.root_graphs
                    .write()
                    .map_err(|e| rmcp::ErrorData::internal_error(e.to_string(), None))?
                    .insert(canonical.clone(), graph);
            }
        }

        Ok(Some(canonical))
    }

    /// Shared search implementation used by both `search_code` and `search_text`.
    #[expect(clippy::too_many_lines, reason = "PageRank boost adds graph lookup")]
    async fn run_search(
        &self,
        query: &str,
        top_k: usize,
        threshold: f32,
        offset: usize,
        root: Option<&str>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let custom_root = self.ensure_root(root, false).await?;

        if custom_root.is_none() {
            let idx_guard = self.index.read().await;
            if idx_guard.is_none() {
                return Err(if self.indexing.load(Ordering::SeqCst) {
                    rmcp::ErrorData::internal_error(self.progress.format_message(), None)
                } else {
                    rmcp::ErrorData::internal_error(
                        "No index available. Call reindex first.".to_string(),
                        None,
                    )
                });
            }
        }

        // Lazy-load backend and tokenizer (no index lock held)
        let text_backend = self.load_text_backend().await?;
        let text_tokenizer = self.load_text_tokenizer().await?;

        let backend = Arc::clone(text_backend);
        let tokenizer = Arc::clone(text_tokenizer);
        let query_owned = query.to_string();

        // Embed the query (no index lock held)
        let query_embedding = tokio::task::spawn_blocking(move || {
            let max_tokens = backend.max_tokens();
            let enc = ripvec_core::tokenize::tokenize_query(&query_owned, &tokenizer, max_tokens)?;
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

        // Use cached custom-root index or the default.
        let root_cache_guard;
        let idx_guard;
        let index: &ripvec_core::hybrid::HybridIndex = if let Some(ref canon) = custom_root {
            root_cache_guard = self.root_indices.read().await;
            root_cache_guard.get(canon).ok_or_else(|| {
                rmcp::ErrorData::internal_error(
                    "Cached index disappeared. Try again.".to_string(),
                    None,
                )
            })?
        } else {
            idx_guard = self.index.read().await;
            idx_guard.as_ref().ok_or_else(|| {
                rmcp::ErrorData::internal_error(
                    "Index was cleared during search. Call reindex.".to_string(),
                    None,
                )
            })?
        };

        let mut ranked = index.search(
            &query_embedding,
            query,
            top_k,
            threshold,
            ripvec_core::hybrid::SearchMode::Hybrid,
        );

        // Apply PageRank boost if the repo graph is available.
        // Multiplicative: amplifies relevant results from structurally important
        // files without promoting irrelevant ones.
        if let Some(ref canon) = custom_root {
            let rg = self
                .root_graphs
                .read()
                .map_err(|e| rmcp::ErrorData::internal_error(e.to_string(), None))?;
            if let Some(graph) = rg.get(canon) {
                let pr = ripvec_core::hybrid::pagerank_lookup(graph);
                ripvec_core::hybrid::boost_with_pagerank(
                    &mut ranked,
                    index.chunks(),
                    &pr,
                    graph.alpha,
                );
            }
        } else {
            let rg = self
                .repo_graph
                .read()
                .map_err(|e| rmcp::ErrorData::internal_error(e.to_string(), None))?;
            if let Some(graph) = rg.as_ref() {
                let pr = ripvec_core::hybrid::pagerank_lookup(graph);
                ripvec_core::hybrid::boost_with_pagerank(
                    &mut ranked,
                    index.chunks(),
                    &pr,
                    graph.alpha,
                );
            }
        }

        let results: Vec<SearchResultItem> = ranked
            .into_iter()
            .skip(offset)
            .take(top_k)
            .filter_map(|(idx, score)| {
                index
                    .chunks()
                    .get(idx)
                    .map(|chunk| SearchResultItem::from_chunk(chunk, score))
            })
            .collect();

        let response = SearchResponse { results };
        let json = serde_json::to_string_pretty(&response)
            .map_err(|e| rmcp::ErrorData::internal_error(e.to_string(), None))?;

        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    /// Search code by semantic meaning.
    ///
    /// Uses BGE-small embeddings to rank indexed chunks by cosine similarity
    /// to the query. Returns ranked chunks with LSP locations.
    #[tool(
        name = "search_code",
        description = "Search code by semantic meaning. Returns ranked chunks with LSP locations."
    )]
    async fn search_code(
        &self,
        Parameters(params): Parameters<SearchParams>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        tracing::debug!(query = %params.query, top_k = params.top_k, "tool: search_code");
        self.run_search(
            &params.query,
            params.top_k,
            params.threshold,
            params.offset,
            params.root.as_deref(),
        )
        .await
    }

    /// Search text by semantic meaning.
    ///
    /// Uses BGE-small embeddings to rank indexed chunks by cosine similarity
    /// to the query. Returns ranked chunks with LSP locations.
    #[tool(
        name = "search_text",
        description = "Search text by semantic meaning. Returns ranked text chunks with LSP locations."
    )]
    async fn search_text(
        &self,
        Parameters(params): Parameters<SearchParams>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        tracing::debug!(query = %params.query, top_k = params.top_k, "tool: search_text");
        self.run_search(
            &params.query,
            params.top_k,
            params.threshold,
            params.offset,
            params.root.as_deref(),
        )
        .await
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
        tracing::debug!(file = %params.file_path, line = params.line, "tool: find_similar");
        let custom_root = self.ensure_root(params.root.as_deref(), false).await?;

        let root_cache_guard;
        let idx_guard;
        let index: &ripvec_core::hybrid::HybridIndex = if let Some(ref canon) = custom_root {
            root_cache_guard = self.root_indices.read().await;
            root_cache_guard.get(canon).ok_or_else(|| {
                rmcp::ErrorData::internal_error(
                    "Cached index disappeared. Try again.".to_string(),
                    None,
                )
            })?
        } else {
            idx_guard = self.index.read().await;
            idx_guard.as_ref().ok_or_else(|| {
                rmcp::ErrorData::internal_error("No index available.".to_string(), None)
            })?
        };

        // Convert 0-based input line to 1-based chunk lines
        let target_line = params.line + 1;

        // Find the chunk containing this location
        let source_idx = index
            .chunks()
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
        let source_embedding = index.semantic.embedding(source_idx).ok_or_else(|| {
            rmcp::ErrorData::internal_error(
                "Embedding not found for source chunk.".to_string(),
                None,
            )
        })?;

        // Rank all chunks against the source embedding
        let ranked = index.semantic.rank(&source_embedding, 0.0);

        let results: Vec<SearchResultItem> = ranked
            .into_iter()
            .filter(|(idx, _)| *idx != source_idx) // Exclude the source chunk
            .skip(params.offset)
            .take(params.top_k)
            .filter_map(|(idx, score)| {
                index
                    .chunks()
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
        name = "find_duplicates",
        description = "Find duplicate or near-duplicate code by pairwise embedding similarity. Returns pairs of chunks with cosine similarity above the threshold, sorted by similarity. Useful for detecting copy-paste code, refactoring candidates, and similar patterns across the codebase."
    )]
    async fn find_duplicates(
        &self,
        Parameters(params): Parameters<FindDuplicatesParams>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        tracing::debug!(threshold = ?params.threshold, max_pairs = ?params.max_pairs, "tool: find_duplicates");
        let threshold = params.threshold.unwrap_or(0.85);
        let max_pairs = params.max_pairs.unwrap_or(20);

        let custom_root = self.ensure_root(params.root.as_deref(), false).await?;

        let root_cache_guard;
        let idx_guard;
        let index: &ripvec_core::hybrid::HybridIndex = if let Some(ref canon) = custom_root {
            root_cache_guard = self.root_indices.read().await;
            root_cache_guard.get(canon).ok_or_else(|| {
                rmcp::ErrorData::internal_error(
                    "Cached index disappeared. Try again.".to_string(),
                    None,
                )
            })?
        } else {
            idx_guard = self.index.read().await;
            idx_guard.as_ref().ok_or_else(|| {
                let msg = self.progress.format_message();
                rmcp::ErrorData::internal_error(msg, None)
            })?
        };

        let pairs = index.semantic.find_duplicates(threshold, max_pairs);

        let chunks = index.chunks();
        let results: Vec<serde_json::Value> = pairs
            .iter()
            .map(|&(a, b, score)| {
                let chunk_a = &chunks[a];
                let chunk_b = &chunks[b];
                serde_json::json!({
                    "similarity": (score * 1000.0).round() / 1000.0,
                    "a": {
                        "file": &chunk_a.file_path,
                        "name": &chunk_a.name,
                        "kind": &chunk_a.kind,
                        "lines": format!("{}-{}", chunk_a.start_line, chunk_a.end_line),
                    },
                    "b": {
                        "file": &chunk_b.file_path,
                        "name": &chunk_b.name,
                        "kind": &chunk_b.kind,
                        "lines": format!("{}-{}", chunk_b.start_line, chunk_b.end_line),
                    },
                })
            })
            .collect();

        let response = serde_json::json!({
            "pairs": results,
            "total": pairs.len(),
            "threshold": threshold,
        });

        Ok(CallToolResult::success(vec![Content::json(response)?]))
    }

    #[tool(
        name = "reindex",
        description = "Rebuild the search index from scratch. Returns chunk and file counts when done."
    )]
    async fn reindex(
        &self,
        Parameters(params): Parameters<ReindexParams>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        tracing::debug!(root = ?params.root, repo_level = params.repo_level, "tool: reindex");
        if let Some(root_str) = params.root.as_deref() {
            // Reindex a custom root: evict from cache, then re-ensure.
            let root_path = std::path::PathBuf::from(root_str);
            if !root_path.is_dir() {
                return Err(rmcp::ErrorData::internal_error(
                    format!("root is not a directory: {root_str}"),
                    None,
                ));
            }
            let canonical = root_path
                .canonicalize()
                .unwrap_or_else(|_| root_path.clone());

            // Evict cached index and graph.
            {
                let mut cache = self.root_indices.write().await;
                cache.remove(&canonical);
            }
            {
                let mut cache = self
                    .root_graphs
                    .write()
                    .map_err(|e| rmcp::ErrorData::internal_error(e.to_string(), None))?;
                cache.remove(&canonical);
            }

            let start = Instant::now();
            self.ensure_root(Some(root_str), params.repo_level).await?;
            let duration_ms = start.elapsed().as_millis();

            let cache = self.root_indices.read().await;
            let (chunk_count, file_count) = match cache.get(&canonical) {
                Some(index) => {
                    let files = index
                        .chunks()
                        .iter()
                        .map(|c| c.file_path.as_str())
                        .collect::<HashSet<_>>()
                        .len();
                    (index.chunks().len(), files)
                }
                None => (0, 0),
            };

            let mut response = serde_json::json!({
                "chunks": chunk_count,
                "files": file_count,
                "duration_ms": duration_ms,
                "root": canonical.display().to_string(),
            });
            if let Some(msg) = ripvec_core::cache::reindex::check_auto_stash(&canonical) {
                response["auto_stash_hint"] = serde_json::Value::String(msg);
            }
            let json = serde_json::to_string_pretty(&response)
                .map_err(|e| rmcp::ErrorData::internal_error(e.to_string(), None))?;
            Ok(CallToolResult::success(vec![Content::text(json)]))
        } else {
            // Reindex default project root.
            {
                let mut idx = self.index.write().await;
                *idx = None;
            }

            let start = Instant::now();
            run_background_index(self, params.repo_level).await;
            let duration_ms = start.elapsed().as_millis();

            let idx_guard = self.index.read().await;
            let (chunk_count, file_count) = match idx_guard.as_ref() {
                Some(index) => {
                    let files = index
                        .chunks()
                        .iter()
                        .map(|c| c.file_path.as_str())
                        .collect::<HashSet<_>>()
                        .len();
                    (index.chunks().len(), files)
                }
                None => (0, 0),
            };

            let mut response = serde_json::json!({
                "chunks": chunk_count,
                "files": file_count,
                "duration_ms": duration_ms,
                "root": self.project_root.display().to_string(),
            });
            if let Some(msg) = ripvec_core::cache::reindex::check_auto_stash(&self.project_root) {
                response["auto_stash_hint"] = serde_json::Value::String(msg);
            }
            let json = serde_json::to_string_pretty(&response)
                .map_err(|e| rmcp::ErrorData::internal_error(e.to_string(), None))?;
            Ok(CallToolResult::success(vec![Content::text(json)]))
        }
    }

    /// Return the current index status.
    ///
    /// Always available, even during background indexing. Returns readiness
    /// state, chunk/file counts, file extension breakdown, and project root.
    #[tool(
        name = "index_status",
        description = "Check the current index status: readiness, chunk/file counts, and project root."
    )]
    async fn index_status(
        &self,
        Parameters(params): Parameters<IndexStatusParams>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        tracing::debug!(root = ?params.root, "tool: index_status");
        if let Some(root_str) = params.root.as_deref() {
            // Status for a custom root.
            let root_path = std::path::PathBuf::from(root_str);
            let canonical = root_path
                .canonicalize()
                .unwrap_or_else(|_| root_path.clone());

            let cache = self.root_indices.read().await;
            let (ready, chunk_count, files_count, ext_counts) = if let Some(index) =
                cache.get(&canonical)
            {
                let mut files_set = HashSet::new();
                let mut exts: HashMap<String, usize> = HashMap::new();
                for chunk in index.chunks() {
                    files_set.insert(chunk.file_path.as_str());
                    let ext = std::path::Path::new(&chunk.file_path)
                        .extension()
                        .and_then(|e| e.to_str())
                        .unwrap_or("(none)")
                        .to_string();
                    *exts.entry(ext).or_insert(0) += 1;
                }
                (true, index.chunks().len(), files_set.len(), exts)
            } else {
                // Check disk cache, or rebuild manifest from objects if gitignored.
                let cache_dir = ripvec_core::cache::reindex::resolve_cache_dir(
                    &canonical,
                    "nomic-ai/modernbert-embed-base",
                    None,
                );
                let manifest_path = cache_dir.join("manifest.json");
                if let Some(manifest) = ripvec_core::cache::manifest::Manifest::load(&manifest_path)
                    .ok()
                    .or_else(|| {
                        ripvec_core::cache::reindex::rebuild_manifest_from_objects(
                            &cache_dir,
                            &canonical,
                            "nomic-ai/modernbert-embed-base",
                        )
                    })
                {
                    let file_count = manifest.files.len();
                    let chunk_count: usize = manifest.files.values().map(|f| f.chunk_count).sum();
                    let mut exts: HashMap<String, usize> = HashMap::new();
                    for key in manifest.files.keys() {
                        let ext = std::path::Path::new(key)
                            .extension()
                            .and_then(|e| e.to_str())
                            .unwrap_or("(none)")
                            .to_string();
                        *exts.entry(ext).or_insert(0) += chunk_count / file_count.max(1);
                    }
                    (true, chunk_count, file_count, exts)
                } else {
                    (false, 0, 0, HashMap::new())
                }
            };

            let cache_dir = ripvec_core::cache::reindex::resolve_cache_dir(
                &canonical,
                "nomic-ai/modernbert-embed-base",
                None,
            );
            let is_repo_local = ripvec_core::cache::reindex::is_repo_local(&cache_dir);
            let response = serde_json::json!({
                "ready": ready,
                "indexing": false,
                "chunks": chunk_count,
                "files": files_count,
                "extensions": ext_counts,
                "project_root": canonical.display().to_string(),
                "cache_location": cache_dir.display().to_string(),
                "repo_local": is_repo_local,
            });
            let json = serde_json::to_string_pretty(&response)
                .map_err(|e| rmcp::ErrorData::internal_error(e.to_string(), None))?;
            Ok(CallToolResult::success(vec![Content::text(json)]))
        } else {
            // Status for default project root.
            let is_indexing = self.indexing.load(Ordering::SeqCst);
            let idx_guard = self.index.read().await;

            // "ready" is a 3-state value:
            //   true       — in-memory index loaded, ready to search
            //   "verifying" — disk cache found, loading + validating mtimes
            //   false      — no index anywhere
            let (ready, chunk_count, files_count, ext_counts) = if let Some(index) =
                idx_guard.as_ref()
            {
                let mut files_set = HashSet::new();
                let mut exts: HashMap<String, usize> = HashMap::new();
                for chunk in index.chunks() {
                    files_set.insert(chunk.file_path.as_str());
                    let ext = std::path::Path::new(&chunk.file_path)
                        .extension()
                        .and_then(|e| e.to_str())
                        .unwrap_or("(none)")
                        .to_string();
                    *exts.entry(ext).or_insert(0) += 1;
                }
                (
                    serde_json::json!(true),
                    index.chunks().len(),
                    files_set.len(),
                    exts,
                )
            } else {
                // Check disk cache — index may exist on disk even if not loaded.
                // Also try rebuilding from objects if manifest is gitignored.
                let cache_dir = ripvec_core::cache::reindex::resolve_cache_dir(
                    &self.project_root,
                    "nomic-ai/modernbert-embed-base",
                    None,
                );
                let manifest_path = cache_dir.join("manifest.json");
                if let Some(manifest) = ripvec_core::cache::manifest::Manifest::load(&manifest_path)
                    .ok()
                    .or_else(|| {
                        ripvec_core::cache::reindex::rebuild_manifest_from_objects(
                            &cache_dir,
                            &self.project_root,
                            "nomic-ai/modernbert-embed-base",
                        )
                    })
                {
                    let file_count = manifest.files.len();
                    let chunk_count: usize = manifest.files.values().map(|f| f.chunk_count).sum();
                    let mut exts: HashMap<String, usize> = HashMap::new();
                    for key in manifest.files.keys() {
                        let ext = std::path::Path::new(key)
                            .extension()
                            .and_then(|e| e.to_str())
                            .unwrap_or("(none)")
                            .to_string();
                        *exts.entry(ext).or_insert(0) += 1;
                    }

                    // Kick off background load — the disk cache will be validated
                    // (mtime heal) and loaded into memory. Next index_status call
                    // will return ready: true.
                    //
                    // run_background_index does its own atomic compare_exchange
                    // on the indexing flag — don't pre-set it here or we'd deadlock
                    // the task into an early return.
                    if !is_indexing {
                        drop(idx_guard); // release read lock before spawning writer
                        let bg_server = self.clone();
                        tokio::spawn(async move {
                            crate::server::run_background_index(&bg_server, false).await;
                        });
                    }

                    (
                        serde_json::json!("verifying"),
                        chunk_count,
                        file_count,
                        exts,
                    )
                } else {
                    (serde_json::json!(false), 0, 0, HashMap::new())
                }
            };

            let cache_dir = ripvec_core::cache::reindex::resolve_cache_dir(
                &self.project_root,
                "nomic-ai/modernbert-embed-base",
                None,
            );
            let is_repo_local = ripvec_core::cache::reindex::is_repo_local(&cache_dir);
            let response = serde_json::json!({
                "ready": ready,
                "indexing": is_indexing,
                "chunks": chunk_count,
                "files": files_count,
                "extensions": ext_counts,
                "project_root": self.project_root.display().to_string(),
                "cache_location": cache_dir.display().to_string(),
                "repo_local": is_repo_local,
            });
            let json = serde_json::to_string_pretty(&response)
                .map_err(|e| rmcp::ErrorData::internal_error(e.to_string(), None))?;
            Ok(CallToolResult::success(vec![Content::text(json)]))
        }
    }

    /// Check whether the running ripvec-mcp binary is up-to-date with its source.
    ///
    /// Compares the binary's modification time against the newest source file
    /// in the ripvec workspace. Returns stale=true when the binary is older
    /// than the source, meaning `cargo build --release` should be run.
    #[tool(
        name = "up_to_date",
        description = "Check if the running ripvec-mcp binary is newer than its source code. \
            Returns version, binary age, and whether a rebuild is needed."
    )]
    async fn up_to_date(&self) -> Result<CallToolResult, rmcp::ErrorData> {
        fn walk_newest(
            dir: &std::path::Path,
            newest: &mut std::time::SystemTime,
            newest_file: &mut String,
        ) {
            let Ok(entries) = std::fs::read_dir(dir) else {
                return;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    walk_newest(&path, newest, newest_file);
                } else if let Some(ext) = path.extension().and_then(|e| e.to_str())
                    && (ext == "rs" || path.file_name().is_some_and(|n| n == "Cargo.toml"))
                    && let Ok(mtime) = entry.metadata().and_then(|m| m.modified())
                    && mtime > *newest
                {
                    *newest = mtime;
                    *newest_file = path.display().to_string();
                }
            }
        }

        let binary_path = std::env::current_exe()
            .map_err(|e| rmcp::ErrorData::internal_error(e.to_string(), None))?;
        let binary_mtime = std::fs::metadata(&binary_path)
            .and_then(|m| m.modified())
            .map_err(|e| rmcp::ErrorData::internal_error(e.to_string(), None))?;

        // Walk crates/ for the newest .rs or Cargo.toml source file.
        let crates_dir = self.project_root.join("crates");
        let mut newest_source = std::time::SystemTime::UNIX_EPOCH;
        let mut newest_file = String::new();
        walk_newest(&crates_dir, &mut newest_source, &mut newest_file);

        let stale = newest_source > binary_mtime;
        let binary_age = binary_mtime.elapsed().unwrap_or_default();
        let source_age = newest_source.elapsed().unwrap_or_default();

        let response = serde_json::json!({
            "version": env!("CARGO_PKG_VERSION"),
            "stale": stale,
            "binary_age_secs": binary_age.as_secs(),
            "newest_source_age_secs": source_age.as_secs(),
            "newest_source_file": newest_file,
            "binary_path": binary_path.display().to_string(),
            "rebuild_command": if stale { "cargo build --release" } else { "" },
        });

        let json = serde_json::to_string_pretty(&response)
            .map_err(|e| rmcp::ErrorData::internal_error(e.to_string(), None))?;

        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    /// Get a PageRank-weighted structural overview of the codebase.
    ///
    /// Renders the most architecturally important files first with their key
    /// definitions, dependency relationships, and function signatures. Supports
    /// topic-sensitive `PageRank` when a focus file is provided.
    #[tool(
        name = "get_repo_map",
        description = "Get a PageRank-weighted structural overview of the codebase. \
            Use FIRST when exploring unfamiliar code or asked about architecture. \
            Shows the most architecturally important files first with their key \
            definitions, dependency relationships, and function signatures."
    )]
    async fn get_repo_map(
        &self,
        Parameters(params): Parameters<RepoMapParams>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        tracing::debug!(max_tokens = params.max_tokens, focus_file = ?params.focus_file, "tool: get_repo_map");
        let custom_root = self.ensure_root(params.root.as_deref(), false).await?;

        // Get the appropriate repo graph (custom root or default).
        let root_graphs_guard;
        let default_guard;
        let graph: &ripvec_core::repo_map::RepoGraph = if let Some(ref canon) = custom_root {
            root_graphs_guard = self
                .root_graphs
                .read()
                .map_err(|e| rmcp::ErrorData::internal_error(e.to_string(), None))?;
            root_graphs_guard.get(canon).ok_or_else(|| {
                rmcp::ErrorData::internal_error(
                    "Cached repo graph disappeared. Try again.".to_string(),
                    None,
                )
            })?
        } else {
            default_guard = self
                .repo_graph
                .read()
                .map_err(|e| rmcp::ErrorData::internal_error(e.to_string(), None))?;
            default_guard.as_ref().ok_or_else(|| {
                if self.indexing.load(std::sync::atomic::Ordering::SeqCst) {
                    rmcp::ErrorData::internal_error(self.progress.format_message(), None)
                } else {
                    rmcp::ErrorData::internal_error(
                        "No repository graph available. Call reindex first.".to_string(),
                        None,
                    )
                }
            })?
        };

        // Resolve focus file to an index in the graph
        let focus_idx = params.focus_file.as_deref().and_then(|focus| {
            graph
                .files
                .iter()
                .position(|f| f.path == focus || f.path.ends_with(focus))
        });

        let rendered = ripvec_core::repo_map::render(graph, params.max_tokens, focus_idx);

        Ok(CallToolResult::success(vec![Content::text(rendered)]))
    }

    /// Get recent log messages from the ripvec-mcp server.
    ///
    /// Useful for diagnosing why operations are slow or failing.
    /// Returns the most recent log lines captured since server start.
    #[tool(
        name = "debug_log",
        description = "Get recent log messages from the ripvec-mcp server. Useful for diagnosing why operations are slow or failing."
    )]
    async fn debug_log(
        &self,
        Parameters(params): Parameters<DebugLogParams>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let lines = self.log_buffer.snapshot();
        let total = lines.len();
        let output: Vec<&str> = lines
            .iter()
            .rev()
            .take(params.lines)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .map(String::as_str)
            .collect();
        let response = serde_json::json!({
            "lines": output,
            "total_buffered": total,
            "returned": output.len(),
        });
        let json = serde_json::to_string_pretty(&response)
            .map_err(|e| rmcp::ErrorData::internal_error(e.to_string(), None))?;
        Ok(CallToolResult::success(vec![Content::text(json)]))
    }

    /// Change the log verbosity at runtime.
    ///
    /// Accepts standard tracing filter directives like "debug", "warn",
    /// or fine-grained per-module filters like "ripvec_mcp=debug,ripvec_core=trace".
    #[tool(
        name = "log_level",
        description = "Change the log verbosity at runtime. Levels: error, warn, info, debug, trace. Also accepts module-level filters like 'ripvec_mcp=debug,ripvec_core=trace'."
    )]
    async fn log_level(
        &self,
        Parameters(params): Parameters<LogLevelParams>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        let new_filter = tracing_subscriber::EnvFilter::try_new(&params.level).map_err(|e| {
            rmcp::ErrorData::internal_error(
                format!("invalid filter '{0}': {e}", params.level),
                None,
            )
        })?;
        self.reload_handle.reload(new_filter).map_err(|e| {
            rmcp::ErrorData::internal_error(format!("failed to reload filter: {e}"), None)
        })?;
        tracing::info!(filter = %params.level, "log level changed");
        let response = serde_json::json!({
            "filter": params.level,
            "status": "applied",
        });
        let json = serde_json::to_string_pretty(&response)
            .map_err(|e| rmcp::ErrorData::internal_error(e.to_string(), None))?;
        Ok(CallToolResult::success(vec![Content::text(json)]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_params_from_number() {
        let json = r#"{"query": "test", "top_k": 5, "threshold": 0.5}"#;
        let params: SearchParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.query, "test");
        assert_eq!(params.top_k, 5);
        assert!((params.threshold - 0.5).abs() < f32::EPSILON);
        assert_eq!(params.offset, 0);
    }

    #[test]
    fn test_search_params_from_string() {
        let json = r#"{"query": "test", "top_k": "5", "threshold": "0.5"}"#;
        let params: SearchParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.query, "test");
        assert_eq!(params.top_k, 5);
        assert!((params.threshold - 0.5).abs() < f32::EPSILON);
        assert_eq!(params.offset, 0);
    }

    #[test]
    fn test_search_params_defaults() {
        let json = r#"{"query": "test"}"#;
        let params: SearchParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.query, "test");
        assert_eq!(params.top_k, 10);
        assert!((params.threshold - 0.3).abs() < f32::EPSILON);
        assert_eq!(params.offset, 0);
    }

    #[test]
    fn test_search_params_with_offset() {
        let json = r#"{"query": "test", "offset": 5}"#;
        let params: SearchParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.offset, 5);
    }

    #[test]
    fn test_search_params_with_offset_string() {
        let json = r#"{"query": "test", "offset": "10"}"#;
        let params: SearchParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.offset, 10);
    }

    #[test]
    fn test_find_similar_params_from_string() {
        let json = r#"{"file_path": "foo.rs", "line": "42"}"#;
        let params: FindSimilarParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.file_path, "foo.rs");
        assert_eq!(params.line, 42);
        assert_eq!(params.top_k, 10); // default
        assert_eq!(params.offset, 0); // default
    }

    #[test]
    fn test_find_similar_params_with_offset() {
        let json = r#"{"file_path": "foo.rs", "line": 10, "offset": 3}"#;
        let params: FindSimilarParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.offset, 3);
    }

    #[test]
    fn test_search_params_invalid_string() {
        let json = r#"{"query": "test", "top_k": "not_a_number"}"#;
        let result: Result<SearchParams, _> = serde_json::from_str(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_repo_map_params_defaults() {
        let json = r"{}";
        let params: RepoMapParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.max_tokens, 2000);
        assert!(params.focus_file.is_none());
    }

    #[test]
    fn test_repo_map_params_with_values() {
        let json = r#"{"max_tokens": 500, "focus_file": "src/main.rs"}"#;
        let params: RepoMapParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.max_tokens, 500);
        assert_eq!(params.focus_file.as_deref(), Some("src/main.rs"));
    }

    #[test]
    fn test_repo_map_params_string_tokens() {
        let json = r#"{"max_tokens": "1000"}"#;
        let params: RepoMapParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.max_tokens, 1000);
    }
}
