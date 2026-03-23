//! Per-file cache entry storing chunks and their embeddings.
//!
//! Uses `rkyv` for zero-copy deserialization — the on-disk format can be
//! memory-mapped and accessed directly without parsing.

use crate::chunk::CodeChunk;

/// Cached chunks and embeddings for a single source file.
///
/// Stored as an rkyv archive in the object store, keyed by the blake3
/// hash of the source file content.
#[derive(Debug, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct FileCache {
    /// The semantic chunks extracted from this file.
    pub chunks: Vec<CodeChunk>,
    /// Flat embedding data: `[n_chunks × hidden_dim]` contiguous f32 values.
    pub embeddings: Vec<f32>,
    /// The embedding dimension (e.g., 384 for BGE-small, 768 for `CodeRankEmbed`).
    pub hidden_dim: usize,
}

impl FileCache {
    /// Serialize to rkyv bytes.
    ///
    /// # Panics
    ///
    /// Panics if serialization fails (should not happen for valid data).
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        rkyv::to_bytes::<rkyv::rancor::Error>(self)
            .expect("FileCache serialization should never fail")
            .to_vec()
    }

    /// Deserialize from rkyv bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the bytes are not a valid rkyv archive.
    pub fn from_bytes(bytes: &[u8]) -> crate::Result<Self> {
        rkyv::from_bytes::<Self, rkyv::rancor::Error>(bytes)
            .map_err(|e| crate::Error::Other(anyhow::anyhow!("rkyv deserialization failed: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let fc = FileCache {
            chunks: vec![CodeChunk {
                file_path: "test.rs".into(),
                name: "foo".into(),
                kind: "function".into(),
                start_line: 1,
                end_line: 10,
                enriched_content: "fn foo() {}".into(),
                content: "fn foo() {}".into(),
            }],
            embeddings: vec![1.0, 2.0, 3.0, 4.0],
            hidden_dim: 4,
        };
        let bytes = fc.to_bytes();
        let loaded = FileCache::from_bytes(&bytes).unwrap();
        assert_eq!(loaded.chunks.len(), 1);
        assert_eq!(loaded.chunks[0].name, "foo");
        assert_eq!(loaded.embeddings.len(), 4);
        assert_eq!(loaded.hidden_dim, 4);
    }

    #[test]
    fn empty_cache() {
        let fc = FileCache {
            chunks: vec![],
            embeddings: vec![],
            hidden_dim: 384,
        };
        let bytes = fc.to_bytes();
        let loaded = FileCache::from_bytes(&bytes).unwrap();
        assert_eq!(loaded.chunks.len(), 0);
        assert_eq!(loaded.embeddings.len(), 0);
        assert_eq!(loaded.hidden_dim, 384);
    }

    #[test]
    fn invalid_bytes_returns_error() {
        let result = FileCache::from_bytes(b"garbage data");
        assert!(result.is_err());
    }
}
