//! MCP server binary for ripvec semantic search.
//!
//! Exposes a `semantic_search` tool over stdin/stdout using the MCP protocol.
//! The embedding model and tokenizer are loaded once at startup and shared
//! across all tool calls via [`Arc`] and [`Mutex`].

use std::sync::{Arc, Mutex};

use rmcp::{
    ServerHandler,
    handler::server::tool::ToolRouter,
    handler::server::wrapper::Parameters,
    model::{Content, ServerCapabilities, ServerInfo},
    tool, tool_router,
};
use schemars::JsonSchema;
use serde::Deserialize;

/// The ripvec MCP server, holding shared model and tokenizer state.
#[derive(Clone)]
pub struct RipvecServer {
    /// The ONNX embedding model, guarded by a mutex because `embed` requires `&mut self`.
    model: Arc<Mutex<ripvec_core::model::EmbeddingModel>>,
    /// The HuggingFace tokenizer, `Send + Sync` so it can be shared across tasks.
    tokenizer: Arc<tokenizers::Tokenizer>,
    /// The generated tool router mapping tool names to handlers.
    tool_router: ToolRouter<Self>,
}

/// Parameters for the `semantic_search` tool call.
#[derive(Deserialize, JsonSchema)]
pub struct SearchRequest {
    /// Natural language query describing the code you're looking for.
    pub query: String,
    /// Root directory to search (defaults to current directory).
    #[serde(default = "default_path")]
    pub path: Option<String>,
    /// Maximum number of results to return (defaults to 10).
    #[serde(default = "default_top_k")]
    pub top_k: Option<usize>,
}

fn default_path() -> Option<String> {
    Some(".".to_string())
}

fn default_top_k() -> Option<usize> {
    Some(10)
}

#[tool_router]
impl RipvecServer {
    /// Create a new server with an already-loaded model and tokenizer.
    fn new(
        model: Arc<Mutex<ripvec_core::model::EmbeddingModel>>,
        tokenizer: Arc<tokenizers::Tokenizer>,
    ) -> Self {
        Self {
            model,
            tokenizer,
            tool_router: Self::tool_router(),
        }
    }

    /// Search code semantically by meaning using vector embeddings.
    #[tool(
        name = "semantic_search",
        description = "Search code semantically by meaning using vector embeddings."
    )]
    async fn semantic_search(
        &self,
        Parameters(req): Parameters<SearchRequest>,
    ) -> Result<rmcp::model::CallToolResult, rmcp::ErrorData> {
        let path = req.path.unwrap_or_else(|| ".".to_string());
        let top_k = req.top_k.unwrap_or(10);
        let model = Arc::clone(&self.model);
        let tokenizer = Arc::clone(&self.tokenizer);
        let query = req.query.clone();

        let results = tokio::task::spawn_blocking(move || {
            ripvec_core::embed::search(
                std::path::Path::new(&path),
                &query,
                &model,
                &tokenizer,
                top_k,
            )
        })
        .await
        .map_err(|e| rmcp::ErrorData::internal_error(e.to_string(), None))?
        .map_err(|e| rmcp::ErrorData::internal_error(e.to_string(), None))?;

        let text = if results.is_empty() {
            "No results found.".to_string()
        } else {
            results
                .iter()
                .enumerate()
                .map(|(i, r)| {
                    format!(
                        "{}. {} ({}:{}-{}, similarity: {:.3})\n```\n{}\n```",
                        i + 1,
                        r.chunk.name,
                        r.chunk.file_path,
                        r.chunk.start_line,
                        r.chunk.end_line,
                        r.similarity,
                        r.chunk.content,
                    )
                })
                .collect::<Vec<_>>()
                .join("\n\n")
        };

        Ok(rmcp::model::CallToolResult::success(vec![Content::text(
            text,
        )]))
    }
}

impl ServerHandler for RipvecServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_instructions("Semantic code search: find functions and definitions by meaning.")
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

/// Load the model and tokenizer, start the MCP server over stdin/stdout.
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let model = Arc::new(Mutex::new(
        ripvec_core::model::EmbeddingModel::load("BAAI/bge-small-en-v1.5", "onnx/model.onnx")
            .map_err(|e| anyhow::anyhow!("failed to load embedding model: {e}"))?,
    ));
    let tokenizer = Arc::new(
        ripvec_core::tokenize::load_tokenizer("BAAI/bge-small-en-v1.5")
            .map_err(|e| anyhow::anyhow!("failed to load tokenizer: {e}"))?,
    );

    let server = RipvecServer::new(model, tokenizer);
    let service = server
        .serve(rmcp::transport::stdio())
        .await
        .map_err(|e| anyhow::anyhow!("MCP serve error: {e}"))?;
    service.waiting().await?;
    Ok(())
}
