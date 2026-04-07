//! LSP server for ripvec semantic code intelligence.
//!
//! Implements the Language Server Protocol over stdio, sharing the same
//! search index and file watcher as the MCP server. Started with `--lsp`.

mod call_hierarchy;
mod diagnostics;
mod hover;
mod navigation;
mod symbols;

use std::fmt;
use std::path::PathBuf;
use std::sync::Arc;

use tower_lsp_server::jsonrpc::Result;
use tower_lsp_server::ls_types::{
    CallHierarchyIncomingCall, CallHierarchyIncomingCallsParams, CallHierarchyItem,
    CallHierarchyOutgoingCall, CallHierarchyOutgoingCallsParams, CallHierarchyPrepareParams,
    CallHierarchyServerCapability, DidChangeTextDocumentParams, DidOpenTextDocumentParams,
    DocumentSymbolParams, DocumentSymbolResponse, GotoDefinitionParams, GotoDefinitionResponse,
    Hover, HoverParams, HoverProviderCapability, ImplementationProviderCapability,
    InitializeParams, InitializeResult, InitializedParams, Location, MessageType, OneOf,
    ReferenceParams, ServerCapabilities, TextDocumentSyncCapability, TextDocumentSyncKind,
    WorkspaceSymbolParams, WorkspaceSymbolResponse, request,
};
use tower_lsp_server::{Client, LanguageServer, LspService, Server};

use crate::server::{IndexProgress, RipvecServer};

/// The ripvec LSP backend, holding shared state from [`RipvecServer`].
///
/// Manual `Debug` impl because `HybridIndex` and `IndexProgress` don't
/// derive `Debug`.
struct RipvecLsp {
    /// LSP client handle for sending notifications (diagnostics, logs).
    client: Client,
    /// Pre-built search index (shared with MCP server / file watcher).
    index: Arc<tokio::sync::RwLock<Option<ripvec_core::hybrid::HybridIndex>>>,
    /// PageRank-weighted repository graph.
    repo_graph: Arc<std::sync::RwLock<Option<ripvec_core::repo_map::RepoGraph>>>,
    /// Root directory of the indexed project.
    project_root: PathBuf,
    /// Shared indexing progress state.
    #[expect(
        dead_code,
        reason = "stub — will be used once handlers are implemented"
    )]
    progress: Arc<IndexProgress>,
}

impl fmt::Debug for RipvecLsp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RipvecLsp")
            .field("project_root", &self.project_root)
            .finish_non_exhaustive()
    }
}

impl LanguageServer for RipvecLsp {
    async fn initialize(&self, _params: InitializeParams) -> Result<InitializeResult> {
        Ok(InitializeResult {
            capabilities: ServerCapabilities {
                text_document_sync: Some(TextDocumentSyncCapability::Kind(
                    TextDocumentSyncKind::FULL,
                )),
                document_symbol_provider: Some(OneOf::Left(true)),
                workspace_symbol_provider: Some(OneOf::Left(true)),
                definition_provider: Some(OneOf::Left(true)),
                implementation_provider: Some(ImplementationProviderCapability::Simple(true)),
                references_provider: Some(OneOf::Left(true)),
                hover_provider: Some(HoverProviderCapability::Simple(true)),
                call_hierarchy_provider: Some(CallHierarchyServerCapability::Simple(true)),
                ..Default::default()
            },
            ..Default::default()
        })
    }

    async fn initialized(&self, _params: InitializedParams) {
        self.client
            .log_message(MessageType::INFO, "ripvec LSP server initialized")
            .await;
    }

    async fn shutdown(&self) -> Result<()> {
        Ok(())
    }

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        diagnostics::publish(
            &self.client,
            &params.text_document.uri,
            &params.text_document.text,
        )
        .await;
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        // With FULL sync, the last content change contains the entire document.
        if let Some(change) = params.content_changes.last() {
            diagnostics::publish(&self.client, &params.text_document.uri, &change.text).await;
        }
    }

    async fn document_symbol(
        &self,
        params: DocumentSymbolParams,
    ) -> Result<Option<DocumentSymbolResponse>> {
        symbols::document_symbol(params).await
    }

    async fn symbol(
        &self,
        params: WorkspaceSymbolParams,
    ) -> Result<Option<WorkspaceSymbolResponse>> {
        symbols::workspace_symbol(params, &self.index, &self.repo_graph, &self.project_root).await
    }

    async fn goto_definition(
        &self,
        params: GotoDefinitionParams,
    ) -> Result<Option<GotoDefinitionResponse>> {
        navigation::goto_definition(params, &self.index, &self.repo_graph, &self.project_root).await
    }

    async fn goto_implementation(
        &self,
        params: request::GotoImplementationParams,
    ) -> Result<Option<request::GotoImplementationResponse>> {
        // Reuse goto_definition for now — implementation will diverge later.
        let def_params = GotoDefinitionParams {
            text_document_position_params: params.text_document_position_params,
            work_done_progress_params: params.work_done_progress_params,
            partial_result_params: params.partial_result_params,
        };
        let result = navigation::goto_definition(
            def_params,
            &self.index,
            &self.repo_graph,
            &self.project_root,
        )
        .await?;
        // GotoDefinitionResponse and GotoImplementationResponse are the same type alias.
        Ok(result)
    }

    async fn references(&self, params: ReferenceParams) -> Result<Option<Vec<Location>>> {
        navigation::find_references(params, &self.index, &self.repo_graph, &self.project_root).await
    }

    async fn hover(&self, params: HoverParams) -> Result<Option<Hover>> {
        hover::hover(params, &self.index, &self.project_root).await
    }

    async fn prepare_call_hierarchy(
        &self,
        params: CallHierarchyPrepareParams,
    ) -> Result<Option<Vec<CallHierarchyItem>>> {
        call_hierarchy::prepare(params, &self.repo_graph, &self.project_root).await
    }

    async fn incoming_calls(
        &self,
        params: CallHierarchyIncomingCallsParams,
    ) -> Result<Option<Vec<CallHierarchyIncomingCall>>> {
        call_hierarchy::incoming(params, &self.repo_graph, &self.project_root).await
    }

    async fn outgoing_calls(
        &self,
        params: CallHierarchyOutgoingCallsParams,
    ) -> Result<Option<Vec<CallHierarchyOutgoingCall>>> {
        call_hierarchy::outgoing(params, &self.repo_graph, &self.project_root).await
    }
}

/// Start the LSP server over stdio, borrowing shared state from `mcp_server`.
pub async fn run(mcp_server: RipvecServer) {
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();

    let (service, socket) = LspService::new(|client| RipvecLsp {
        client,
        index: Arc::clone(&mcp_server.index),
        repo_graph: Arc::clone(&mcp_server.repo_graph),
        project_root: mcp_server.project_root.clone(),
        progress: Arc::clone(&mcp_server.progress),
    });

    Server::new(stdin, stdout, socket).serve(service).await;
}
