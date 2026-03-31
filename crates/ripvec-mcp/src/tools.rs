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
    pub fn new(project_root: std::path::PathBuf) -> Self {
        Self {
            index: Arc::new(tokio::sync::RwLock::new(None)),
            text_backend: Arc::new(tokio::sync::OnceCell::new()),
            text_tokenizer: Arc::new(tokio::sync::OnceCell::new()),
            project_root,
            indexing: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            tool_router: Self::tool_router(),
            repo_graph: Arc::new(std::sync::RwLock::new(None)),
            root_indices: Arc::new(tokio::sync::RwLock::new(std::collections::HashMap::new())),
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

    /// Shared search implementation used by both `search_code` and `search_text`.
    ///
    /// When `root` is `Some`, loads the index for that directory from the
    /// on-disk cache (instant if already indexed). Otherwise uses the
    /// pre-built in-memory index for the server's default project root.
    async fn run_search(
        &self,
        query: &str,
        top_k: usize,
        threshold: f32,
        offset: usize,
        root: Option<&str>,
    ) -> Result<CallToolResult, rmcp::ErrorData> {
        // Resolve which index to use: custom root (cached) or default.
        let custom_root: Option<std::path::PathBuf>;
        if let Some(root_str) = root {
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

            // Check if we already have this index cached.
            {
                let cache = self.root_indices.read().await;
                if cache.contains_key(&canonical) {
                    // Cache hit — skip re-indexing.
                    custom_root = Some(canonical);
                } else {
                    drop(cache);
                    // Cache miss — load from on-disk cache (instant if pre-indexed).
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
                        )
                    })
                    .await
                    .map_err(|e| rmcp::ErrorData::internal_error(e.to_string(), None))?
                    .map_err(|e| rmcp::ErrorData::internal_error(e.to_string(), None))?;
                    let mut cache = self.root_indices.write().await;
                    cache.insert(canonical.clone(), idx);
                    custom_root = Some(canonical);
                }
            }
        } else {
            custom_root = None;
            // Check default index exists
            let idx_guard = self.index.read().await;
            if idx_guard.is_none() {
                return Err(if self.indexing.load(Ordering::SeqCst) {
                    rmcp::ErrorData::internal_error(
                        "Index is still building. Try again shortly.".to_string(),
                        None,
                    )
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

        let ranked = index.search(
            &query_embedding,
            query,
            top_k,
            threshold,
            ripvec_core::hybrid::SearchMode::Hybrid,
        );

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
        self.run_search(&params.query, params.top_k, params.threshold, params.offset, params.root.as_deref())
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
        self.run_search(&params.query, params.top_k, params.threshold, params.offset, params.root.as_deref())
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
        let idx_guard = self.index.read().await;
        let index = idx_guard.as_ref().ok_or_else(|| {
            rmcp::ErrorData::internal_error("No index available.".to_string(), None)
        })?;

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
        name = "reindex",
        description = "Rebuild the search index from scratch. Returns chunk and file counts when done."
    )]
    async fn reindex(&self) -> Result<CallToolResult, rmcp::ErrorData> {
        // Clear current index
        {
            let mut idx = self.index.write().await;
            *idx = None;
        }

        let start = Instant::now();
        run_background_index(self).await;
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

        let (chunk_count, files_count, ext_counts) = match idx_guard.as_ref() {
            Some(index) => {
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
                (index.chunks().len(), files_set.len(), exts)
            }
            None => (0, 0, HashMap::new()),
        };

        let response = serde_json::json!({
            "ready": ready,
            "indexing": is_indexing,
            "chunks": chunk_count,
            "files": files_count,
            "extensions": ext_counts,
            "project_root": self.project_root.display().to_string(),
        });

        let json = serde_json::to_string_pretty(&response)
            .map_err(|e| rmcp::ErrorData::internal_error(e.to_string(), None))?;

        Ok(CallToolResult::success(vec![Content::text(json)]))
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
        let binary_path = std::env::current_exe()
            .map_err(|e| rmcp::ErrorData::internal_error(e.to_string(), None))?;
        let binary_mtime = std::fs::metadata(&binary_path)
            .and_then(|m| m.modified())
            .map_err(|e| rmcp::ErrorData::internal_error(e.to_string(), None))?;

        // Walk crates/ for the newest .rs or Cargo.toml source file.
        let crates_dir = self.project_root.join("crates");
        let mut newest_source = std::time::SystemTime::UNIX_EPOCH;
        let mut newest_file = String::new();
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
                } else if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                    if (ext == "rs" || path.file_name().is_some_and(|n| n == "Cargo.toml"))
                        && let Ok(mtime) = entry.metadata().and_then(|m| m.modified())
                        && mtime > *newest
                    {
                        *newest = mtime;
                        *newest_file = path.display().to_string();
                    }
                }
            }
        }
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
        let graph_guard = self
            .repo_graph
            .read()
            .map_err(|e| rmcp::ErrorData::internal_error(e.to_string(), None))?;

        let graph = graph_guard.as_ref().ok_or_else(|| {
            if self.indexing.load(std::sync::atomic::Ordering::SeqCst) {
                rmcp::ErrorData::internal_error(
                    "Repository graph is still building. Try again shortly.".to_string(),
                    None,
                )
            } else {
                rmcp::ErrorData::internal_error(
                    "No repository graph available. Call reindex first.".to_string(),
                    None,
                )
            }
        })?;

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
