//! Persistent index cache with content-addressed object store.
//!
//! Stores per-file chunks and embeddings in a git-style object store,
//! tracks file state in a Merkle-tree manifest, and supports incremental
//! reindexing by diffing against the filesystem.

pub mod file_cache;
pub mod store;
