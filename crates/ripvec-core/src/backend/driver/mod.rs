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
#[cfg(feature = "cuda")]
pub mod cuda;
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

    /// Create a new driver instance for a cloned worker thread.
    ///
    /// CPU drivers are zero-size and always succeed. GPU drivers typically
    /// cannot be cloned this way (they share device state) and should leave
    /// the default panic implementation.
    fn new_for_clone() -> crate::Result<Self>
    where
        Self: Sized,
    {
        Err(crate::Error::Other(anyhow::anyhow!(
            "this driver does not support cloning"
        )))
    }

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

    /// Flush the current command buffer and start a new one, preserving pool
    /// state. Use mid-forward-pass to prevent GPU timeouts on deep models.
    fn flush_batch(&self) -> crate::Result<()> {
        Ok(())
    }

    /// Close and reopen the compute encoder within the same command buffer.
    ///
    /// This segments a long sequence of compute dispatches into multiple
    /// encoders without committing or waiting. Metal processes encoders
    /// back-to-back from the same CB — zero sync overhead.
    ///
    /// Use every few layers to prevent encoder state overflow (>~60 dispatches
    /// per encoder can cause hangs on some Apple Silicon GPUs).
    fn segment_encoder(&self) {
        // No-op for non-Metal backends
    }

    /// Save the current pool cursor position. Call BEFORE a layer's work.
    fn save_pool_cursor(&self) -> usize {
        0
    }

    /// Restore the pool cursor to a previously saved position. Call AFTER
    /// a layer's transient tensors have been dropped (out of scope).
    ///
    /// The architecture must ensure only the output tensor (`hidden_states`)
    /// survives — all layer-internal tensors (qkv, scores, context, etc.)
    /// must be dropped before this call so their pool slots can be recycled.
    fn restore_pool_cursor(&self, _saved: usize) {}

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

    // --- Banded (local/sliding-window) attention ---

    /// Banded Q@K^T: compute attention scores only within a sliding window.
    ///
    /// Output shape: `[batch * num_heads, seq, window]` (NOT `[seq, seq]`).
    /// `scores[h, i, w]` = dot(Q[h, i, :], K[h, i - window/2 + w, :])
    /// where out-of-bounds positions are set to `-inf` (masked in softmax).
    ///
    /// Reduces attention compute from O(seq²) to O(seq × window).
    /// For `seq=512, window=128`: **4× less compute** per local layer.
    fn banded_qk(
        &self,
        q: &Self::Tensor,
        k: &Self::Tensor,
        scores: &mut Self::Tensor,
        batch_heads: usize,
        seq: usize,
        head_dim: usize,
        window: usize,
        stride_qk: usize,
        stride_scores: usize,
    ) -> crate::Result<()>;

    /// Banded scores@V: weighted sum using banded attention scores.
    ///
    /// Input scores: `[batch * num_heads, seq, window]` (from `banded_qk`).
    /// Output: `[batch * num_heads, seq, head_dim]`.
    /// `output[h, i, d]` = sum_w scores[h, i, w] * V[h, i - window/2 + w, d]
    fn banded_sv(
        &self,
        scores: &Self::Tensor,
        v: &Self::Tensor,
        output: &mut Self::Tensor,
        batch_heads: usize,
        seq: usize,
        head_dim: usize,
        window: usize,
        stride_scores: usize,
        stride_v: usize,
        stride_out: usize,
    ) -> crate::Result<()>;

    /// Fused scale + softmax over the window dimension (no padding mask needed).
    ///
    /// Operates on `[batch * num_heads * seq, window]` rows.
    fn banded_softmax(
        &self,
        scores: &mut Self::Tensor,
        total_rows: usize,
        window: usize,
        scale: f32,
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
    /// Used by ModernBERT (not ClassicBert which uses learned position embeddings).
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
    /// Used by `ModernBERT` for gated MLP splits.
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
    /// The gate and value come from splitting the intermediate projection.
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

    // =======================================================================
    // FP16 operations for full half-precision pipeline
    //
    // These methods mirror the FP32 counterparts but operate on FP16 tensors.
    // Internal reductions (softmax, layer-norm) use FP32 accumulators but
    // all tensor I/O is half precision. Default implementations return an
    // error — only backends with FP16 support override them.
    // =======================================================================

    /// Allocate a zero-initialized FP16 tensor with `n` half-precision elements.
    ///
    /// # Errors
    ///
    /// Returns an error if device memory allocation fails or FP16 is unsupported.
    fn alloc_zeros_f16(&self, _n: usize) -> crate::Result<Self::Tensor> {
        Err(crate::Error::Metal(
            "FP16 not supported by this driver".into(),
        ))
    }

    /// Convert FP32 tensor to FP16 (element-wise narrowing).
    fn f32_to_f16(
        &self,
        _output: &mut Self::Tensor,
        _input: &Self::Tensor,
        _n: usize,
    ) -> crate::Result<()> {
        Err(crate::Error::Metal(
            "FP16 not supported by this driver".into(),
        ))
    }

    /// Convert FP16 tensor back to FP32 (element-wise widening).
    fn f16_to_f32(
        &self,
        _output: &mut Self::Tensor,
        _input: &Self::Tensor,
        _n: usize,
    ) -> crate::Result<()> {
        Err(crate::Error::Metal(
            "FP16 not supported by this driver".into(),
        ))
    }

    /// Mixed-precision GEMM: FP16 inputs → FP32 output via native simdgroup ops.
    fn gemm_mixed(
        &self,
        _a_f16: &Self::Tensor,
        _b_f16: &Self::Tensor,
        _output_f32: &mut Self::Tensor,
        _m: usize,
        _n: usize,
        _k: usize,
        _transpose_b: bool,
    ) -> crate::Result<()> {
        Err(crate::Error::Metal(
            "gemm_mixed not supported by this driver".into(),
        ))
    }

    /// FP16 GEMM: `output = A * B` (or `A * B^T`). All tensors are half.
    fn gemm_f16(
        &self,
        _a: &Self::Tensor,
        _b: &Self::Tensor,
        _output: &mut Self::Tensor,
        _m: usize,
        _n: usize,
        _k: usize,
        _transpose_b: bool,
    ) -> crate::Result<()> {
        Err(crate::Error::Metal(
            "FP16 not supported by this driver".into(),
        ))
    }

    /// FP16 batched GEMM for multi-head attention. All tensors are half.
    #[expect(
        clippy::too_many_arguments,
        reason = "matches FP32 gemm_batched signature"
    )]
    fn gemm_batched_f16(
        &self,
        _a: &Self::Tensor,
        _b: &Self::Tensor,
        _output: &mut Self::Tensor,
        _m: usize,
        _n: usize,
        _k: usize,
        _transpose_b: bool,
        _stride_a: usize,
        _stride_b: usize,
        _stride_c: usize,
        _batch_count: usize,
    ) -> crate::Result<()> {
        Err(crate::Error::Metal(
            "FP16 not supported by this driver".into(),
        ))
    }

    /// FP16 layer normalization. Half I/O, FP32 reductions.
    fn layer_norm_f16(
        &self,
        _output: &mut Self::Tensor,
        _input: &Self::Tensor,
        _weight: &Self::Tensor,
        _bias: &Self::Tensor,
        _rows: usize,
        _cols: usize,
        _eps: f32,
    ) -> crate::Result<()> {
        Err(crate::Error::Metal(
            "FP16 not supported by this driver".into(),
        ))
    }

    /// FP16 fused scale + mask + softmax. Half scores, FP32 reductions.
    fn fused_scale_mask_softmax_f16(
        &self,
        _scores: &mut Self::Tensor,
        _mask: &Self::Tensor,
        _batch: usize,
        _num_heads: usize,
        _seq_len: usize,
        _scale: f32,
    ) -> crate::Result<()> {
        Err(crate::Error::Metal(
            "FP16 not supported by this driver".into(),
        ))
    }

    /// FP16 fused scale + mask + sliding window + softmax.
    fn fused_scale_mask_softmax_windowed_f16(
        &self,
        _scores: &mut Self::Tensor,
        _mask: &Self::Tensor,
        _batch: usize,
        _num_heads: usize,
        _seq_len: usize,
        _scale: f32,
        _window_size: usize,
    ) -> crate::Result<()> {
        Err(crate::Error::Metal(
            "FP16 not supported by this driver".into(),
        ))
    }

    /// FP16 QKV split: `[batch*seq, 3*hidden]` into Q, K, V per-head layout.
    fn qkv_split_f16(
        &self,
        _q: &mut Self::Tensor,
        _k: &mut Self::Tensor,
        _v: &mut Self::Tensor,
        _qkv: &Self::Tensor,
        _batch: usize,
        _seq: usize,
        _hidden: usize,
        _num_heads: usize,
        _head_dim: usize,
    ) -> crate::Result<()> {
        Err(crate::Error::Metal(
            "FP16 not supported by this driver".into(),
        ))
    }

    /// FP16 attention output reshape: `[batch*num_heads, seq, head_dim]` to
    /// `[batch*seq, hidden]`.
    fn attn_reshape_f16(
        &self,
        _output: &mut Self::Tensor,
        _input: &Self::Tensor,
        _batch: usize,
        _seq: usize,
        _num_heads: usize,
        _head_dim: usize,
    ) -> crate::Result<()> {
        Err(crate::Error::Metal(
            "FP16 not supported by this driver".into(),
        ))
    }

    /// FP16 scatter flat `[total_tokens, dim]` to padded `[batch, max_seq, dim]`.
    fn pad_to_batch_f16(
        &self,
        _flat: &Self::Tensor,
        _padded: &mut Self::Tensor,
        _seq_lengths: &[usize],
        _max_seq: usize,
        _dim: usize,
    ) -> crate::Result<()> {
        Err(crate::Error::Metal(
            "FP16 not supported by this driver".into(),
        ))
    }

    /// FP16 gather padded `[batch, max_seq, dim]` back to flat `[total_tokens, dim]`.
    fn unpad_from_batch_f16(
        &self,
        _padded: &Self::Tensor,
        _flat: &mut Self::Tensor,
        _seq_lengths: &[usize],
        _max_seq: usize,
        _dim: usize,
    ) -> crate::Result<()> {
        Err(crate::Error::Metal(
            "FP16 not supported by this driver".into(),
        ))
    }

    /// FP16 RoPE: apply rotary position embedding. Half Q/K, float cos/sin tables.
    fn rope_encode_f16(
        &self,
        _qk: &mut Self::Tensor,
        _cos: &Self::Tensor,
        _sin: &Self::Tensor,
        _num_rows: usize,
        _seq_len: usize,
        _head_dim: usize,
        _num_heads: usize,
    ) -> crate::Result<()> {
        Err(crate::Error::Metal(
            "FP16 not supported by this driver".into(),
        ))
    }

    /// FP16 `GeGLU` gated activation: `output = gelu(value) * gate`. Half I/O.
    fn geglu_f16(
        &self,
        _value: &Self::Tensor,
        _gate: &Self::Tensor,
        _output: &mut Self::Tensor,
        _n: usize,
    ) -> crate::Result<()> {
        Err(crate::Error::Metal(
            "FP16 not supported by this driver".into(),
        ))
    }

    /// FP16 fused residual add + layer normalization.
    fn fused_residual_layernorm_f16(
        &self,
        _output: &mut Self::Tensor,
        _hidden: &Self::Tensor,
        _residual: &Self::Tensor,
        _weight: &Self::Tensor,
        _bias: &Self::Tensor,
        _rows: usize,
        _cols: usize,
        _eps: f32,
    ) -> crate::Result<()> {
        Err(crate::Error::Metal(
            "FP16 not supported by this driver".into(),
        ))
    }

    /// FP16 residual add (no bias): `output = hidden + residual`.
    fn residual_add_f16(
        &self,
        _output: &mut Self::Tensor,
        _hidden: &Self::Tensor,
        _residual: &Self::Tensor,
        _n: usize,
    ) -> crate::Result<()> {
        Err(crate::Error::Metal(
            "FP16 not supported by this driver".into(),
        ))
    }

    /// FP16 split `[rows, 2*cols]` into two `[rows, cols]` halves.
    fn split_gate_value_f16(
        &self,
        _first: &mut Self::Tensor,
        _second: &mut Self::Tensor,
        _input: &Self::Tensor,
        _rows: usize,
        _cols: usize,
    ) -> crate::Result<()> {
        Err(crate::Error::Metal(
            "FP16 not supported by this driver".into(),
        ))
    }

    /// Fused split + `GeGLU`: read `[rows, 2*cols]`, write `[rows, cols]`.
    ///
    /// Combines [`split_gate_value_f16`](Driver::split_gate_value_f16) and
    /// [`geglu_f16`](Driver::geglu_f16) into a single kernel, eliminating
    /// two intermediate `[rows, cols]` buffers and halving HBM round-trips.
    ///
    /// Default falls back to separate split + geglu calls.
    fn fused_split_geglu_f16(
        &self,
        output: &mut Self::Tensor,
        input: &Self::Tensor,
        rows: usize,
        cols: usize,
    ) -> crate::Result<()> {
        // Default: allocate intermediates and call separately.
        let n = rows * cols;
        let mut value = self.alloc_zeros_f16(n)?;
        let mut gate = self.alloc_zeros_f16(n)?;
        self.split_gate_value_f16(&mut value, &mut gate, input, rows, cols)?;
        self.geglu_f16(&value, &gate, output, n)
    }
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
