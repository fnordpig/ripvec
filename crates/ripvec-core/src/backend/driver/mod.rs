//! Hardware-agnostic compute driver trait.
//!
//! The [`Driver`] trait exposes low-level compute primitives (GEMM, layer-norm,
//! activations, etc.) that each hardware backend implements. Model architectures
//! are generic over `D: Driver` and compose these primitives into a forward pass.
//!
//! # Design
//!
//! - **Associated type `Tensor`**: each driver defines its own opaque tensor
//!   handle (Metal: buffer+offset, CUDA: device pointer, CPU: ndarray).
//! - **Not object-safe**: architectures use `D: Driver` generics so the compiler
//!   can monomorphize and inline driver calls.
//! - **Send + Sync**: drivers are shared across the pipeline.

#[cfg(any(feature = "cpu", feature = "cpu-accelerate"))]
pub mod cpu;
#[cfg(feature = "metal")]
pub mod metal;
#[cfg(feature = "mlx")]
pub mod mlx;

use super::Encoding;

/// Hardware-agnostic compute primitives for BERT inference.
///
/// Each method corresponds to one operation in the forward pass. Drivers handle
/// memory allocation, kernel dispatch, and synchronization. Architectures
/// compose these primitives via the [`super::arch::ModelArch`] trait.
pub trait Driver: Send + Sync {
    /// Opaque tensor handle.
    ///
    /// Metal: `MTLBuffer` + byte offset. CUDA: `CUdeviceptr`. CPU: `Array2<f32>`.
    type Tensor;

    // --- Batching ---

    /// Begin batched mode: all subsequent operations encode into one dispatch.
    ///
    /// GPU drivers accumulate into a single command buffer; CPU is a no-op.
    /// Call [`end_batch`] to commit. This eliminates per-call overhead.
    fn begin_batch(&self) -> crate::Result<()> {
        Ok(())
    }

    /// End batched mode: commit all accumulated operations and wait.
    fn end_batch(&self) -> crate::Result<()> {
        Ok(())
    }

    /// Reset workspace allocation cursor so the next layer reuses buffers
    /// from the previous layer. Without this, each of N layers allocates
    /// fresh buffers, accumulating N × (buffers-per-layer) in memory.
    /// With this, peak memory is 1 × (buffers-per-layer).
    fn reset_layer_workspace(&self) {}

    // --- Allocation ---

    /// Allocate a zero-initialized tensor with `n` float elements on device.
    ///
    /// Used by architectures to create workspace buffers (QKV projections,
    /// attention scores, intermediate activations, etc.).
    ///
    /// # Errors
    ///
    /// Returns an error if device memory allocation fails.
    fn alloc_zeros(&self, n: usize) -> crate::Result<Self::Tensor>;

    /// Clone a tensor, producing an independent copy of the data.
    ///
    /// Used when an operation needs both the original and a mutable output
    /// referencing the same logical data (e.g., in-place layer normalization
    /// where input == output).
    ///
    /// # Errors
    ///
    /// Returns an error if device memory allocation or the copy fails.
    fn clone_tensor(&self, tensor: &Self::Tensor, n: usize) -> crate::Result<Self::Tensor>;

    // --- Batch preparation ---

    /// Prepare a batch of encodings for inference, returning input tensors on device.
    ///
    /// Pads all sequences to `max_seq` and uploads `input_ids`, `attention_mask`,
    /// `token_type_ids`, `position_ids`, and a float attention mask to device memory.
    fn prepare_batch(
        &self,
        encodings: &[Encoding],
        max_seq: usize,
    ) -> crate::Result<BatchInputs<Self::Tensor>>;

    /// Prepare a batch WITHOUT padding — concatenate all tokens flat.
    ///
    /// Returns `BatchInputs` with `total_tokens` actual tokens (no padding),
    /// `cu_seqlens` for attention boundaries, and per-token position IDs.
    /// Linear layers (GEMM, LN, GELU) process `total_tokens` rows.
    /// Attention must pad/unpad around the per-head operations.
    fn prepare_batch_unpadded(
        &self,
        encodings: &[Encoding],
    ) -> crate::Result<BatchInputs<Self::Tensor>> {
        // Default: fall back to padded (backends override for unpadded support)
        let max_seq = encodings
            .iter()
            .map(|e| e.input_ids.len())
            .max()
            .unwrap_or(0)
            .next_multiple_of(8);
        self.prepare_batch(encodings, max_seq)
    }

    /// Scatter flat `[total_tokens, dim]` tensor into padded `[batch, max_seq, dim]`.
    ///
    /// Used before attention: linear layers produce unpadded output, but the
    /// QKV split + batched attention GEMM need aligned `[batch*heads, seq, head_dim]`.
    /// Padding positions are zeroed.
    fn pad_to_batch(
        &self,
        flat: &Self::Tensor,
        padded: &mut Self::Tensor,
        seq_lengths: &[usize],
        max_seq: usize,
        dim: usize,
    ) -> crate::Result<()>;

    /// Gather padded `[batch, max_seq, dim]` back to flat `[total_tokens, dim]`.
    ///
    /// Used after attention: extracts only the real tokens, discarding padding.
    fn unpad_from_batch(
        &self,
        padded: &Self::Tensor,
        flat: &mut Self::Tensor,
        seq_lengths: &[usize],
        max_seq: usize,
        dim: usize,
    ) -> crate::Result<()>;

    // --- Embedding operations ---

    /// Word/position/token-type embedding lookup via gather.
    ///
    /// Reads `seq_len` token IDs from `word_ids`, gathers rows from
    /// `embedding_table`, and writes `[seq_len, hidden]` floats to the result.
    fn embedding_lookup(
        &self,
        word_ids: &Self::Tensor,
        embedding_table: &Self::Tensor,
        seq_len: usize,
        hidden: usize,
    ) -> crate::Result<Self::Tensor>;

    /// Element-wise add an embedding table lookup into `hidden`.
    ///
    /// Used for position and token-type embeddings:
    /// `hidden[i] += table[ids[i]]` for each token position.
    fn add_embeddings(
        &self,
        hidden: &mut Self::Tensor,
        table: &Self::Tensor,
        ids: &Self::Tensor,
        seq_len: usize,
        hidden_dim: usize,
    ) -> crate::Result<()>;

    // --- Normalization ---

    /// Layer normalization: `output = (input - mean) / sqrt(var + eps) * weight + bias`.
    fn layer_norm(
        &self,
        output: &mut Self::Tensor,
        input: &Self::Tensor,
        weight: &Self::Tensor,
        bias: &Self::Tensor,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> crate::Result<()>;

    // --- Linear algebra ---

    /// General matrix multiply: `output = A * B` (or `A * B^T` if `transpose_b`).
    ///
    /// Dimensions: A is `[m, k]`, B is `[k, n]` (or `[n, k]` if transposed),
    /// output is `[m, n]`.
    fn gemm(
        &self,
        a: &Self::Tensor,
        b: &Self::Tensor,
        output: &mut Self::Tensor,
        m: usize,
        n: usize,
        k: usize,
        transpose_b: bool,
    ) -> crate::Result<()>;

    /// Batched GEMM for multi-head attention.
    ///
    /// Performs `batch_count` independent GEMMs with strided access into
    /// contiguous buffers. Used for per-head Q*K^T and attn*V.
    fn gemm_batched(
        &self,
        a: &Self::Tensor,
        b: &Self::Tensor,
        output: &mut Self::Tensor,
        m: usize,
        n: usize,
        k: usize,
        transpose_b: bool,
        stride_a: usize,
        stride_b: usize,
        stride_c: usize,
        batch_count: usize,
    ) -> crate::Result<()>;

    // --- Attention ---

    /// Fused scale + mask + softmax for attention scores.
    ///
    /// `scores = softmax(scores * scale + mask)` computed per-head.
    fn fused_scale_mask_softmax(
        &self,
        scores: &mut Self::Tensor,
        mask: &Self::Tensor,
        batch: usize,
        num_heads: usize,
        seq_len: usize,
        scale: f32,
    ) -> crate::Result<()>;

    /// Fused scale + mask + sliding window + softmax for attention scores.
    ///
    /// Like [`fused_scale_mask_softmax`](Driver::fused_scale_mask_softmax) but
    /// additionally masks out positions where `|query_pos - key_pos| > window_size / 2`.
    /// Used by `ModernBERT`'s local attention layers.
    fn fused_scale_mask_softmax_windowed(
        &self,
        scores: &mut Self::Tensor,
        mask: &Self::Tensor,
        batch: usize,
        num_heads: usize,
        seq_len: usize,
        scale: f32,
        window_size: usize,
    ) -> crate::Result<()>;

    /// Build a float attention mask from an integer mask.
    ///
    /// Converts `[batch * seq]` int mask (0/1) to `[batch * seq]` float mask
    /// (0.0 / -10000.0) for use with [`fused_scale_mask_softmax`](Driver::fused_scale_mask_softmax).
    fn build_attn_mask(
        &self,
        output: &mut Self::Tensor,
        int_mask: &Self::Tensor,
        n: usize,
    ) -> crate::Result<()>;

    /// Split a fused QKV projection into separate Q, K, V tensors.
    fn qkv_split(
        &self,
        q: &mut Self::Tensor,
        k: &mut Self::Tensor,
        v: &mut Self::Tensor,
        qkv: &Self::Tensor,
        batch: usize,
        seq: usize,
        hidden: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> crate::Result<()>;

    /// Reshape attention output from `[batch, num_heads, seq, head_dim]` to
    /// `[batch * seq, hidden]`.
    fn attn_reshape(
        &self,
        output: &mut Self::Tensor,
        input: &Self::Tensor,
        batch: usize,
        seq: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> crate::Result<()>;

    /// Apply Rotary Position Embedding (RoPE) to Q/K tensors.
    ///
    /// Used by NomicBert and ModernBERT (not ClassicBert which uses learned
    /// position embeddings).
    fn apply_rope(
        &self,
        qk: &mut Self::Tensor,
        cos: &Self::Tensor,
        sin: &Self::Tensor,
        num_rows: usize,
        seq_len: usize,
        head_dim: usize,
        num_heads: usize,
    ) -> crate::Result<()>;

    // --- Tensor manipulation ---

    /// Split a `[rows, 2*cols]` matrix into two `[rows, cols]` halves.
    ///
    /// Each row of `input` is `[first_half | second_half]`. The first `cols`
    /// elements go to `first`, the remaining `cols` to `second`.
    /// Used by `ModernBERT` and `NomicBert` for gated MLP splits.
    fn split_gate_value(
        &self,
        first: &mut Self::Tensor,
        second: &mut Self::Tensor,
        input: &Self::Tensor,
        rows: usize,
        cols: usize,
    ) -> crate::Result<()>;

    // --- Activations ---

    /// GELU activation (Gaussian Error Linear Unit), applied in-place.
    fn gelu(&self, x: &mut Self::Tensor, n: usize) -> crate::Result<()>;

    /// SwiGLU gated activation: `output = value * silu(gate)`.
    ///
    /// Used by NomicBert. The gate and value come from splitting the
    /// intermediate projection.
    fn swiglu(
        &self,
        value: &Self::Tensor,
        gate: &Self::Tensor,
        output: &mut Self::Tensor,
        n: usize,
    ) -> crate::Result<()>;

    /// `GeGLU` gated activation: `output = gelu(value) * gate`.
    ///
    /// Used by `ModernBERT`. The value and gate come from splitting the
    /// MLP `Wi` projection output in half.
    fn geglu(
        &self,
        value: &Self::Tensor,
        gate: &Self::Tensor,
        output: &mut Self::Tensor,
        n: usize,
    ) -> crate::Result<()>;

    /// Fused bias + GELU: `x = gelu(x + bias)` row-wise.
    fn fused_bias_gelu(
        &self,
        x: &mut Self::Tensor,
        bias: &Self::Tensor,
        rows: usize,
        cols: usize,
    ) -> crate::Result<()>;

    // --- Fused residual operations ---

    /// Fused bias + residual add: `output = input + bias + residual`.
    ///
    /// Bias is broadcast row-wise (`cols`-wide) across `n / cols` rows.
    fn fused_bias_residual(
        &self,
        output: &mut Self::Tensor,
        input: &Self::Tensor,
        bias: &Self::Tensor,
        residual: &Self::Tensor,
        n: usize,
        cols: usize,
    ) -> crate::Result<()>;

    /// Fused residual add + layer normalization.
    ///
    /// `output = layer_norm(hidden + residual, weight, bias, eps)`.
    fn fused_residual_layernorm(
        &self,
        output: &mut Self::Tensor,
        hidden: &Self::Tensor,
        residual: &Self::Tensor,
        weight: &Self::Tensor,
        bias: &Self::Tensor,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> crate::Result<()>;

    /// Residual add without bias: `output = hidden + residual`.
    ///
    /// Used by `ModernBERT` which has no bias terms.
    fn residual_add(
        &self,
        output: &mut Self::Tensor,
        hidden: &Self::Tensor,
        residual: &Self::Tensor,
        n: usize,
    ) -> crate::Result<()>;

    /// Add bias to a matrix row-wise: `x[row] += bias` for each row.
    fn add_bias(
        &self,
        x: &mut Self::Tensor,
        bias: &Self::Tensor,
        rows: usize,
        cols: usize,
    ) -> crate::Result<()>;

    // --- Pooling ---

    /// CLS pooling: extract the first token's hidden state per batch element.
    fn cls_pool(
        &self,
        output: &mut Self::Tensor,
        hidden: &Self::Tensor,
        batch: usize,
        seq: usize,
        hidden_dim: usize,
    ) -> crate::Result<()>;

    /// Mean pooling: attention-mask-weighted average of hidden states.
    fn mean_pool(
        &self,
        output: &mut Self::Tensor,
        hidden: &Self::Tensor,
        mask: &Self::Tensor,
        batch: usize,
        seq: usize,
        hidden_dim: usize,
    ) -> crate::Result<()>;

    // --- Post-processing ---

    /// L2-normalize each row vector in-place.
    fn l2_normalize(&self, data: &mut Self::Tensor, rows: usize, cols: usize) -> crate::Result<()>;

    /// Copy tensor data back to host memory as `Vec<Vec<f32>>`.
    ///
    /// Returns one `Vec<f32>` of length `dim` per batch element.
    fn to_host(
        &self,
        tensor: &Self::Tensor,
        batch: usize,
        dim: usize,
    ) -> crate::Result<Vec<Vec<f32>>>;
}

/// Batch input tensors on device, produced by [`Driver::prepare_batch`].
///
/// Supports both padded and unpadded modes:
/// - **Padded**: all sequences padded to `max_seq`. `cu_seqlens` is `None`.
/// - **Unpadded**: sequences concatenated without padding. `cu_seqlens`
///   contains cumulative lengths `[0, len0, len0+len1, ...]` so attention
///   knows where each sequence starts. Eliminates ALL padding compute.
pub struct BatchInputs<T> {
    /// Token IDs — `[batch * max_seq]` (padded) or `[total_tokens]` (unpadded).
    pub input_ids: T,
    /// Attention mask `[batch * max_seq]` as int32 (0 or 1). Unused in unpadded mode.
    pub attention_mask: T,
    /// Token type IDs — same layout as `input_ids`.
    pub token_type_ids: T,
    /// Position IDs — same layout as `input_ids`.
    pub position_ids: T,
    /// Float attention bias mask `[batch * max_seq]` (0.0 or -1e9) for softmax.
    pub float_mask: T,
    /// Float pooling mask `[batch * max_seq]` (1.0 or 0.0) for mean pooling.
    pub pooling_mask: T,
    /// Number of sequences in this batch.
    pub batch: usize,
    /// Maximum sequence length (all sequences padded to this). In unpadded mode,
    /// this is the longest sequence (used for workspace sizing, not padding).
    pub max_seq: usize,
    /// Total actual tokens across all sequences (no padding).
    pub total_tokens: usize,
    /// Per-sequence lengths: `[batch]` — each element is the actual token count.
    pub seq_lengths: Vec<usize>,
    /// Cumulative sequence lengths for unpadded attention: `[batch + 1]`.
    /// `cu_seqlens[i]..cu_seqlens[i+1]` is the token range for sequence `i`.
    /// `None` in padded mode (all sequences padded to max_seq).
    pub cu_seqlens: Option<Vec<usize>>,
}
