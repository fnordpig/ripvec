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

#[cfg(feature = "metal")]
pub mod metal;

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
/// All tensors are padded to `max_seq` tokens per sequence.
pub struct BatchInputs<T> {
    /// Token IDs `[batch * max_seq]` as int32 on device.
    pub input_ids: T,
    /// Attention mask `[batch * max_seq]` as int32 (0 or 1).
    pub attention_mask: T,
    /// Token type IDs `[batch * max_seq]` as int32.
    pub token_type_ids: T,
    /// Position IDs `[batch * max_seq]` as int32.
    pub position_ids: T,
    /// Float attention mask `[batch * max_seq]` (0.0 or -10000.0).
    pub float_mask: T,
    /// Number of sequences in this batch.
    pub batch: usize,
    /// Maximum sequence length (all sequences padded to this).
    pub max_seq: usize,
}
