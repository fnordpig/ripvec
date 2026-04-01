//! MLX lazy-evaluation GPU driver for Apple Silicon.
//!
//! Implements the [`Driver`] trait using [`MlxTensor`] as the tensor type.
//! Every driver method builds a lazy computation graph — no GPU work happens
//! until [`to_host`](Driver::to_host) calls `eval()`. This preserves MLX's
//! graph fusion optimisations.
//!
//! # Design
//!
//! - **`type Tensor = MlxTensor`**: wraps `mlx_rs::Array` with `Send + Sync`.
//! - **`begin_batch` / `end_batch`**: no-ops. MLX is always lazy; the
//!   architecture calls `eval()` implicitly via `to_host`.
//! - **Reshapes are free**: MLX reshapes are metadata-only (no data movement).
//! - **`Array::clone()` is cheap**: reference-counted, no copy.

use std::collections::HashMap;

use hf_hub::api::sync::Api;
use mlx_rs::Array;
use mlx_rs::ops::indexing::TryIndexOp;

use super::{BatchInputs, Driver};
use crate::backend::Encoding;
use crate::backend::arch::classic_bert::{
    ClassicBertArch, ClassicBertLayerWeights, ClassicBertWeights,
};
use crate::backend::generic::GenericBackend;

/// Convert an MLX exception into our crate error type.
fn mlx_err(e: impl std::fmt::Display) -> crate::Error {
    crate::Error::Other(anyhow::anyhow!("MLX driver: {e}"))
}

// ---------------------------------------------------------------------------
// MlxDriver
// ---------------------------------------------------------------------------

/// Wrapper around `mlx_rs::Array` that is `Send + Sync`.
///
/// MLX arrays are reference-counted handles to lazy computation graph nodes.
/// The underlying Metal resources are managed by MLX's runtime, which is
/// thread-safe at the API level (one eval at a time). We enforce single-
/// threaded access through the `embed_distributed` architecture (GPU backends
/// are not cloned).
pub struct MlxTensor(pub Array);

// SAFETY: MLX arrays are reference-counted handles. The MLX runtime serialises
// GPU operations internally. Our pipeline ensures single-threaded forward-pass
// access (GPU backends are not cloned).
#[expect(unsafe_code, reason = "MLX runtime serialises GPU access")]
unsafe impl Send for MlxTensor {}
#[expect(unsafe_code, reason = "MLX runtime serialises GPU access")]
unsafe impl Sync for MlxTensor {}

impl Clone for MlxTensor {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

/// Stateless MLX compute driver.
///
/// MLX manages its own Metal device, command queues, and memory internally.
/// The driver is just a namespace for [`Driver`] trait methods — all state
/// lives in the lazy [`MlxTensor`] handles themselves.
pub struct MlxDriver;

impl MlxDriver {
    /// Create a new MLX driver.
    ///
    /// MLX initialises its Metal backend on first array operation, so this
    /// is essentially free.
    pub fn new() -> crate::Result<Self> {
        Ok(Self)
    }
}

// ---------------------------------------------------------------------------
// Driver trait implementation
// ---------------------------------------------------------------------------

impl Driver for MlxDriver {
    type Tensor = MlxTensor;

    // begin_batch / end_batch: no-ops. MLX is always lazy.

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "mlx-rs requires i32 for shape params; ML tensor dims fit in i32"
    )]
    fn alloc_zeros(&self, n: usize) -> crate::Result<MlxTensor> {
        Array::zeros::<f32>(&[n as i32])
            .map_err(mlx_err)
            .map(MlxTensor)
    }

    fn clone_tensor(&self, tensor: &MlxTensor, _n: usize) -> crate::Result<MlxTensor> {
        // MLX arrays are reference-counted; clone just bumps the refcount.
        Ok(tensor.clone())
    }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_precision_loss,
        reason = "token IDs, masks, and type IDs from tokenizer are always small non-negative values"
    )]
    fn prepare_batch(
        &self,
        encodings: &[Encoding],
        max_seq: usize,
    ) -> crate::Result<BatchInputs<MlxTensor>> {
        let batch = encodings.len();
        let total = batch * max_seq;

        let mut input_ids = vec![0_i32; total];
        let mut attn_mask_int = vec![0_i32; total];
        let mut token_type_ids = vec![0_i32; total];
        let mut position_ids = vec![0_i32; total];
        let mut float_mask = vec![0.0_f32; total];
        let mut pooling_mask = vec![0.0_f32; total];

        for (b, enc) in encodings.iter().enumerate() {
            let offset = b * max_seq;
            let seq_len = enc.input_ids.len();
            for (i, (&id, (&mask, &typ))) in enc
                .input_ids
                .iter()
                .zip(enc.attention_mask.iter().zip(enc.token_type_ids.iter()))
                .enumerate()
            {
                input_ids[offset + i] = id as i32;
                attn_mask_int[offset + i] = mask as i32;
                token_type_ids[offset + i] = typ as i32;
                position_ids[offset + i] = i as i32;
                // Real token: 0.0 (pass-through); pad: -1e9 (kill softmax)
                float_mask[offset + i] = if mask == 1 { 0.0 } else { -1e9 };
                pooling_mask[offset + i] = mask as f32;
            }
            // Pad positions beyond seq_len stay 0 (default).
            // Float mask for pad positions is already -1e9 (default 0.0 for
            // unset means pad, but we explicitly set real tokens above).
            // Fix: pad positions default to 0.0 float_mask, need -1e9.
            for i in seq_len..max_seq {
                float_mask[offset + i] = -1e9;
            }
        }

        let seq_lengths: Vec<usize> = encodings.iter().map(|e| e.input_ids.len()).collect();
        let total_tokens: usize = seq_lengths.iter().sum();
        let total_i32 = total as i32;
        Ok(BatchInputs {
            input_ids: MlxTensor(Array::from_slice(&input_ids, &[total_i32])),
            attention_mask: MlxTensor(Array::from_slice(&attn_mask_int, &[total_i32])),
            token_type_ids: MlxTensor(Array::from_slice(&token_type_ids, &[total_i32])),
            position_ids: MlxTensor(Array::from_slice(&position_ids, &[total_i32])),
            float_mask: MlxTensor(Array::from_slice(&float_mask, &[total_i32])),
            pooling_mask: MlxTensor(Array::from_slice(&pooling_mask, &[total_i32])),
            batch,
            max_seq,
            total_tokens,
            seq_lengths,
            cu_seqlens: None, // Padded mode
        })
    }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "mlx-rs requires i32 for shape params; ML tensor dims fit in i32"
    )]
    fn pad_to_batch(
        &self,
        flat: &MlxTensor,
        padded: &mut MlxTensor,
        seq_lengths: &[usize],
        max_seq: usize,
        dim: usize,
    ) -> crate::Result<()> {
        // Perform on CPU — pad/unpad is a small, infrequent operation.
        // The large GEMMs remain fully lazy in MLX.
        let batch = seq_lengths.len();
        let total_out = batch * max_seq * dim;
        let mut out = vec![0.0_f32; total_out];

        // Eval flat to read data — synchronisation point.
        flat.0.eval().map_err(mlx_err)?;
        let flat_data: &[f32] = flat.0.as_slice();

        let mut offset = 0;
        for (b, &len) in seq_lengths.iter().enumerate() {
            for t in 0..len {
                let src = (offset + t) * dim;
                let dst = (b * max_seq + t) * dim;
                out[dst..dst + dim].copy_from_slice(&flat_data[src..src + dim]);
            }
            offset += len;
        }
        *padded = MlxTensor(Array::from_slice(&out, &[total_out as i32]));
        Ok(())
    }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "mlx-rs requires i32 for shape params; ML tensor dims fit in i32"
    )]
    fn unpad_from_batch(
        &self,
        padded: &MlxTensor,
        flat: &mut MlxTensor,
        seq_lengths: &[usize],
        max_seq: usize,
        dim: usize,
    ) -> crate::Result<()> {
        // Perform on CPU — pad/unpad is a small, infrequent operation.
        let _batch = seq_lengths.len();
        let total_tokens: usize = seq_lengths.iter().sum();
        let mut out = vec![0.0_f32; total_tokens * dim];

        // Eval padded to read data — synchronisation point.
        padded.0.eval().map_err(mlx_err)?;
        let padded_data: &[f32] = padded.0.as_slice();

        let mut offset = 0;
        for (b, &len) in seq_lengths.iter().enumerate() {
            for t in 0..len {
                let src = (b * max_seq + t) * dim;
                let dst = (offset + t) * dim;
                out[dst..dst + dim].copy_from_slice(&padded_data[src..src + dim]);
            }
            offset += len;
        }
        *flat = MlxTensor(Array::from_slice(&out, &[(total_tokens * dim) as i32]));
        Ok(())
    }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "mlx-rs requires i32 for shape params; ML tensor dims fit in i32"
    )]
    fn embedding_lookup(
        &self,
        word_ids: &MlxTensor,
        embedding_table: &MlxTensor,
        seq_len: usize,
        hidden: usize,
    ) -> crate::Result<MlxTensor> {
        let ids = mlx_rs::ops::reshape(&word_ids.0, &[seq_len as i32]).map_err(mlx_err)?;
        let table =
            mlx_rs::ops::reshape(&embedding_table.0, &[-1, hidden as i32]).map_err(mlx_err)?;
        let emb = table.try_index(&ids).map_err(mlx_err)?;
        // Flatten back to 1D: [seq_len * hidden]
        let flat = mlx_rs::ops::reshape(&emb, &[(seq_len * hidden) as i32]).map_err(mlx_err)?;
        Ok(MlxTensor(flat))
    }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "mlx-rs requires i32 for shape params; ML tensor dims fit in i32"
    )]
    fn add_embeddings(
        &self,
        hidden: &mut MlxTensor,
        table: &MlxTensor,
        ids: &MlxTensor,
        seq_len: usize,
        hidden_dim: usize,
    ) -> crate::Result<()> {
        let ids_2d = mlx_rs::ops::reshape(&ids.0, &[seq_len as i32]).map_err(mlx_err)?;
        let table_2d = mlx_rs::ops::reshape(&table.0, &[-1, hidden_dim as i32]).map_err(mlx_err)?;
        let emb = table_2d.try_index(&ids_2d).map_err(mlx_err)?;
        let emb_flat =
            mlx_rs::ops::reshape(&emb, &[(seq_len * hidden_dim) as i32]).map_err(mlx_err)?;
        let h =
            mlx_rs::ops::reshape(&hidden.0, &[(seq_len * hidden_dim) as i32]).map_err(mlx_err)?;
        hidden.0 = mlx_rs::ops::add(&h, &emb_flat).map_err(mlx_err)?;
        Ok(())
    }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "mlx-rs requires i32 for shape params; ML tensor dims fit in i32"
    )]
    fn layer_norm(
        &self,
        output: &mut MlxTensor,
        input: &MlxTensor,
        weight: &MlxTensor,
        bias: &MlxTensor,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> crate::Result<()> {
        let x = mlx_rs::ops::reshape(&input.0, &[rows as i32, cols as i32]).map_err(mlx_err)?;
        let w = mlx_rs::ops::reshape(&weight.0, &[1, cols as i32]).map_err(mlx_err)?;
        let b = mlx_rs::ops::reshape(&bias.0, &[1, cols as i32]).map_err(mlx_err)?;

        // Use mlx_rs::fast::layer_norm which is optimised
        let normed = mlx_rs::fast::layer_norm(&x, Some(&w), Some(&b), eps).map_err(mlx_err)?;
        output.0 = mlx_rs::ops::reshape(&normed, &[(rows * cols) as i32]).map_err(mlx_err)?;
        Ok(())
    }

    #[expect(
        clippy::many_single_char_names,
        reason = "matches Driver trait signature (m, n, k)"
    )]
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "mlx-rs requires i32 for shape params; ML tensor dims fit in i32"
    )]
    fn gemm(
        &self,
        a: &MlxTensor,
        b: &MlxTensor,
        output: &mut MlxTensor,
        m: usize,
        n: usize,
        k: usize,
        transpose_b: bool,
    ) -> crate::Result<()> {
        let a_2d = mlx_rs::ops::reshape(&a.0, &[m as i32, k as i32]).map_err(mlx_err)?;
        let b_2d = if transpose_b {
            // B is [n, k], need [k, n] for matmul
            let b_shaped = mlx_rs::ops::reshape(&b.0, &[n as i32, k as i32]).map_err(mlx_err)?;
            b_shaped.t()
        } else {
            mlx_rs::ops::reshape(&b.0, &[k as i32, n as i32]).map_err(mlx_err)?
        };
        let result = mlx_rs::ops::matmul(&a_2d, &b_2d).map_err(mlx_err)?;
        output.0 = mlx_rs::ops::reshape(&result, &[(m * n) as i32]).map_err(mlx_err)?;
        Ok(())
    }

    #[expect(
        clippy::many_single_char_names,
        reason = "matches Driver trait signature (m, n, k)"
    )]
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "mlx-rs requires i32 for shape params; ML tensor dims fit in i32"
    )]
    fn gemm_batched(
        &self,
        a: &MlxTensor,
        b: &MlxTensor,
        output: &mut MlxTensor,
        m: usize,
        n: usize,
        k: usize,
        transpose_b: bool,
        _stride_a: usize,
        _stride_b: usize,
        _stride_c: usize,
        batch_count: usize,
    ) -> crate::Result<()> {
        let bc = batch_count as i32;
        let a_3d = mlx_rs::ops::reshape(&a.0, &[bc, m as i32, k as i32]).map_err(mlx_err)?;
        let b_3d = if transpose_b {
            // B is [batch, n, k] stored flat, need [batch, k, n] for matmul
            let b_shaped =
                mlx_rs::ops::reshape(&b.0, &[bc, n as i32, k as i32]).map_err(mlx_err)?;
            mlx_rs::ops::transpose_axes(&b_shaped, &[0, 2, 1]).map_err(mlx_err)?
        } else {
            mlx_rs::ops::reshape(&b.0, &[bc, k as i32, n as i32]).map_err(mlx_err)?
        };
        let result = mlx_rs::ops::matmul(&a_3d, &b_3d).map_err(mlx_err)?;
        output.0 =
            mlx_rs::ops::reshape(&result, &[(batch_count * m * n) as i32]).map_err(mlx_err)?;
        Ok(())
    }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "mlx-rs requires i32 for shape params; ML tensor dims fit in i32"
    )]
    fn fused_scale_mask_softmax(
        &self,
        scores: &mut MlxTensor,
        mask: &MlxTensor,
        batch: usize,
        num_heads: usize,
        seq_len: usize,
        scale: f32,
    ) -> crate::Result<()> {
        let total_heads = (batch * num_heads) as i32;
        let s = seq_len as i32;

        // scores: [batch*num_heads, seq, seq], mask: [batch*seq] (flat)
        let sc = mlx_rs::ops::reshape(&scores.0, &[total_heads, s, s]).map_err(mlx_err)?;

        // Scale
        let scale_arr = Array::from_f32(scale);
        let scaled = mlx_rs::ops::multiply(&sc, &scale_arr).map_err(mlx_err)?;

        // Reshape mask to [batch, 1, 1, seq] for broadcasting over heads and query positions
        let mask_2d = mlx_rs::ops::reshape(&mask.0, &[batch as i32, s]).map_err(mlx_err)?;
        let mask_4d = mlx_rs::ops::reshape(&mask_2d, &[batch as i32, 1, 1, s]).map_err(mlx_err)?;

        // Reshape scores to [batch, num_heads, seq, seq] for broadcast with mask
        let scaled_4d = mlx_rs::ops::reshape(&scaled, &[batch as i32, num_heads as i32, s, s])
            .map_err(mlx_err)?;

        let masked = mlx_rs::ops::add(&scaled_4d, &mask_4d).map_err(mlx_err)?;

        // Softmax along last axis
        let softmaxed = mlx_rs::ops::softmax_axis(&masked, -1, None).map_err(mlx_err)?;

        // Flatten back
        let total = batch * num_heads * seq_len * seq_len;
        scores.0 = mlx_rs::ops::reshape(&softmaxed, &[total as i32]).map_err(mlx_err)?;
        Ok(())
    }

    fn fused_scale_mask_softmax_windowed(
        &self,
        scores: &mut MlxTensor,
        mask: &MlxTensor,
        batch: usize,
        num_heads: usize,
        seq_len: usize,
        scale: f32,
        window_size: usize,
    ) -> crate::Result<()> {
        // For now, delegate to non-windowed version.
        // Windowed attention is only used by ModernBERT; ClassicBert doesn't need it.
        // TODO: implement sliding window mask when adding ModernBERT MlxDriver support.
        let _ = window_size;
        self.fused_scale_mask_softmax(scores, mask, batch, num_heads, seq_len, scale)
    }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "mlx-rs requires i32 for shape params; ML tensor dims fit in i32"
    )]
    fn build_attn_mask(
        &self,
        output: &mut MlxTensor,
        int_mask: &MlxTensor,
        n: usize,
    ) -> crate::Result<()> {
        // Convert int mask (0/1) to float mask (0.0 / -1e9)
        let mask_f32 = int_mask
            .0
            .as_dtype(mlx_rs::Dtype::Float32)
            .map_err(mlx_err)?;
        let ones = Array::ones::<f32>(&[n as i32]).map_err(mlx_err)?;
        let inverted = mlx_rs::ops::subtract(&ones, &mask_f32).map_err(mlx_err)?;
        let neg = Array::from_f32(-1e9_f32);
        output.0 = mlx_rs::ops::multiply(&inverted, &neg).map_err(mlx_err)?;
        Ok(())
    }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "mlx-rs requires i32 for shape params; ML tensor dims fit in i32"
    )]
    fn qkv_split(
        &self,
        q: &mut MlxTensor,
        k: &mut MlxTensor,
        v: &mut MlxTensor,
        qkv: &MlxTensor,
        batch: usize,
        seq: usize,
        _hidden: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> crate::Result<()> {
        let total_tokens = batch * seq;

        // qkv is flat [total_tokens * 3 * hidden]. Reshape to [total_tokens, 3, num_heads, head_dim].
        let qkv_4d = mlx_rs::ops::reshape(
            &qkv.0,
            &[total_tokens as i32, 3, num_heads as i32, head_dim as i32],
        )
        .map_err(mlx_err)?;

        // Extract Q, K, V along axis 1
        let q_3d = qkv_4d.try_index((.., 0, .., ..)).map_err(mlx_err)?;
        let k_3d = qkv_4d.try_index((.., 1, .., ..)).map_err(mlx_err)?;
        let v_3d = qkv_4d.try_index((.., 2, .., ..)).map_err(mlx_err)?;

        // Reshape from [total_tokens, num_heads, head_dim] to
        // [batch, seq, num_heads, head_dim] -> transpose to [batch, num_heads, seq, head_dim]
        // -> reshape to [batch*num_heads, seq, head_dim] -> flatten to 1D
        let reshape_head = |x: Array| -> crate::Result<Array> {
            let x = mlx_rs::ops::reshape(
                &x,
                &[batch as i32, seq as i32, num_heads as i32, head_dim as i32],
            )
            .map_err(mlx_err)?;
            let x = mlx_rs::ops::transpose_axes(&x, &[0, 2, 1, 3]).map_err(mlx_err)?;
            let x = mlx_rs::ops::reshape(
                &x,
                &[(batch * num_heads) as i32, seq as i32, head_dim as i32],
            )
            .map_err(mlx_err)?;
            mlx_rs::ops::reshape(&x, &[(batch * num_heads * seq * head_dim) as i32])
                .map_err(mlx_err)
        };

        q.0 = reshape_head(q_3d)?;
        k.0 = reshape_head(k_3d)?;
        v.0 = reshape_head(v_3d)?;
        Ok(())
    }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "mlx-rs requires i32 for shape params; ML tensor dims fit in i32"
    )]
    fn attn_reshape(
        &self,
        output: &mut MlxTensor,
        input: &MlxTensor,
        batch: usize,
        seq: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> crate::Result<()> {
        let hidden = num_heads * head_dim;
        // input: flat [batch * num_heads * seq * head_dim]
        // -> [batch, num_heads, seq, head_dim]
        // -> transpose [batch, seq, num_heads, head_dim]
        // -> reshape [batch*seq, hidden]
        // -> flatten [batch*seq*hidden]
        let x = mlx_rs::ops::reshape(
            &input.0,
            &[batch as i32, num_heads as i32, seq as i32, head_dim as i32],
        )
        .map_err(mlx_err)?;
        let x = mlx_rs::ops::transpose_axes(&x, &[0, 2, 1, 3]).map_err(mlx_err)?;
        output.0 = mlx_rs::ops::reshape(&x, &[(batch * seq * hidden) as i32]).map_err(mlx_err)?;
        Ok(())
    }

    fn apply_rope(
        &self,
        _qk: &mut MlxTensor,
        _cos: &MlxTensor,
        _sin: &MlxTensor,
        _num_rows: usize,
        _seq_len: usize,
        _head_dim: usize,
        _num_heads: usize,
    ) -> crate::Result<()> {
        // RoPE is used by ModernBERT, not ClassicBert. Implement when adding ModernBERT MLX support.
        Err(crate::Error::Other(anyhow::anyhow!(
            "MLX driver: apply_rope not yet implemented (ModernBERT support pending)"
        )))
    }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "mlx-rs requires i32 for shape params; ML tensor dims fit in i32"
    )]
    fn split_gate_value(
        &self,
        first: &mut MlxTensor,
        second: &mut MlxTensor,
        input: &MlxTensor,
        rows: usize,
        cols: usize,
    ) -> crate::Result<()> {
        // input is flat [rows * 2 * cols]. Reshape to [rows, 2*cols], split at cols.
        let x =
            mlx_rs::ops::reshape(&input.0, &[rows as i32, (2 * cols) as i32]).map_err(mlx_err)?;
        let first_half = x.try_index((.., ..cols as i32)).map_err(mlx_err)?;
        let second_half = x.try_index((.., cols as i32..)).map_err(mlx_err)?;
        first.0 = mlx_rs::ops::reshape(&first_half, &[(rows * cols) as i32]).map_err(mlx_err)?;
        second.0 = mlx_rs::ops::reshape(&second_half, &[(rows * cols) as i32]).map_err(mlx_err)?;
        Ok(())
    }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "mlx-rs requires i32 for shape params; ML tensor dims fit in i32"
    )]
    fn gelu(&self, x: &mut MlxTensor, n: usize) -> crate::Result<()> {
        let reshaped = mlx_rs::ops::reshape(&x.0, &[n as i32]).map_err(mlx_err)?;
        let activated = mlx_rs::nn::gelu(&reshaped).map_err(mlx_err)?;
        x.0 = activated;
        Ok(())
    }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "mlx-rs requires i32 for shape params; ML tensor dims fit in i32"
    )]
    fn swiglu(
        &self,
        value: &MlxTensor,
        gate: &MlxTensor,
        output: &mut MlxTensor,
        n: usize,
    ) -> crate::Result<()> {
        let v = mlx_rs::ops::reshape(&value.0, &[n as i32]).map_err(mlx_err)?;
        let g = mlx_rs::ops::reshape(&gate.0, &[n as i32]).map_err(mlx_err)?;
        let gate_activated = mlx_rs::nn::silu(&g).map_err(mlx_err)?;
        output.0 = mlx_rs::ops::multiply(&v, &gate_activated).map_err(mlx_err)?;
        Ok(())
    }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "mlx-rs requires i32 for shape params; ML tensor dims fit in i32"
    )]
    fn geglu(
        &self,
        value: &MlxTensor,
        gate: &MlxTensor,
        output: &mut MlxTensor,
        n: usize,
    ) -> crate::Result<()> {
        let v = mlx_rs::ops::reshape(&value.0, &[n as i32]).map_err(mlx_err)?;
        let g = mlx_rs::ops::reshape(&gate.0, &[n as i32]).map_err(mlx_err)?;
        let gelu_v = mlx_rs::nn::gelu(&v).map_err(mlx_err)?;
        output.0 = mlx_rs::ops::multiply(&gelu_v, &g).map_err(mlx_err)?;
        Ok(())
    }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "mlx-rs requires i32 for shape params; ML tensor dims fit in i32"
    )]
    fn fused_bias_gelu(
        &self,
        x: &mut MlxTensor,
        bias: &MlxTensor,
        rows: usize,
        cols: usize,
    ) -> crate::Result<()> {
        let x_2d = mlx_rs::ops::reshape(&x.0, &[rows as i32, cols as i32]).map_err(mlx_err)?;
        let b = mlx_rs::ops::reshape(&bias.0, &[1, cols as i32]).map_err(mlx_err)?;
        let biased = mlx_rs::ops::add(&x_2d, &b).map_err(mlx_err)?;
        let activated = mlx_rs::nn::gelu(&biased).map_err(mlx_err)?;
        x.0 = mlx_rs::ops::reshape(&activated, &[(rows * cols) as i32]).map_err(mlx_err)?;
        Ok(())
    }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "mlx-rs requires i32 for shape params; ML tensor dims fit in i32"
    )]
    fn fused_bias_residual(
        &self,
        output: &mut MlxTensor,
        input: &MlxTensor,
        bias: &MlxTensor,
        residual: &MlxTensor,
        n: usize,
        cols: usize,
    ) -> crate::Result<()> {
        let rows = n / cols;
        let inp = mlx_rs::ops::reshape(&input.0, &[rows as i32, cols as i32]).map_err(mlx_err)?;
        let b = mlx_rs::ops::reshape(&bias.0, &[1, cols as i32]).map_err(mlx_err)?;
        let res =
            mlx_rs::ops::reshape(&residual.0, &[rows as i32, cols as i32]).map_err(mlx_err)?;
        let biased = mlx_rs::ops::add(&inp, &b).map_err(mlx_err)?;
        let sum = mlx_rs::ops::add(&biased, &res).map_err(mlx_err)?;
        output.0 = mlx_rs::ops::reshape(&sum, &[n as i32]).map_err(mlx_err)?;
        Ok(())
    }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "mlx-rs requires i32 for shape params; ML tensor dims fit in i32"
    )]
    fn fused_residual_layernorm(
        &self,
        output: &mut MlxTensor,
        hidden: &MlxTensor,
        residual: &MlxTensor,
        weight: &MlxTensor,
        bias: &MlxTensor,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> crate::Result<()> {
        let h = mlx_rs::ops::reshape(&hidden.0, &[rows as i32, cols as i32]).map_err(mlx_err)?;
        let r = mlx_rs::ops::reshape(&residual.0, &[rows as i32, cols as i32]).map_err(mlx_err)?;
        let sum = mlx_rs::ops::add(&h, &r).map_err(mlx_err)?;
        let w = mlx_rs::ops::reshape(&weight.0, &[1, cols as i32]).map_err(mlx_err)?;
        let b = mlx_rs::ops::reshape(&bias.0, &[1, cols as i32]).map_err(mlx_err)?;
        let normed = mlx_rs::fast::layer_norm(&sum, Some(&w), Some(&b), eps).map_err(mlx_err)?;
        output.0 = mlx_rs::ops::reshape(&normed, &[(rows * cols) as i32]).map_err(mlx_err)?;
        Ok(())
    }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "mlx-rs requires i32 for shape params; ML tensor dims fit in i32"
    )]
    fn residual_add(
        &self,
        output: &mut MlxTensor,
        hidden: &MlxTensor,
        residual: &MlxTensor,
        n: usize,
    ) -> crate::Result<()> {
        let h = mlx_rs::ops::reshape(&hidden.0, &[n as i32]).map_err(mlx_err)?;
        let r = mlx_rs::ops::reshape(&residual.0, &[n as i32]).map_err(mlx_err)?;
        output.0 = mlx_rs::ops::add(&h, &r).map_err(mlx_err)?;
        Ok(())
    }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "mlx-rs requires i32 for shape params; ML tensor dims fit in i32"
    )]
    fn add_bias(
        &self,
        x: &mut MlxTensor,
        bias: &MlxTensor,
        rows: usize,
        cols: usize,
    ) -> crate::Result<()> {
        let x_2d = mlx_rs::ops::reshape(&x.0, &[rows as i32, cols as i32]).map_err(mlx_err)?;
        let b = mlx_rs::ops::reshape(&bias.0, &[1, cols as i32]).map_err(mlx_err)?;
        let result = mlx_rs::ops::add(&x_2d, &b).map_err(mlx_err)?;
        x.0 = mlx_rs::ops::reshape(&result, &[(rows * cols) as i32]).map_err(mlx_err)?;
        Ok(())
    }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "mlx-rs requires i32 for shape params; ML tensor dims fit in i32"
    )]
    fn cls_pool(
        &self,
        output: &mut MlxTensor,
        hidden: &MlxTensor,
        batch: usize,
        seq: usize,
        hidden_dim: usize,
    ) -> crate::Result<()> {
        // hidden: flat [batch * seq * hidden_dim]
        // -> [batch, seq, hidden_dim] -> take [:, 0, :] -> flatten [batch * hidden_dim]
        let h = mlx_rs::ops::reshape(&hidden.0, &[batch as i32, seq as i32, hidden_dim as i32])
            .map_err(mlx_err)?;
        let cls = h.try_index((.., 0, ..)).map_err(mlx_err)?;
        output.0 = mlx_rs::ops::reshape(&cls, &[(batch * hidden_dim) as i32]).map_err(mlx_err)?;
        Ok(())
    }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "mlx-rs requires i32 for shape params; ML tensor dims fit in i32"
    )]
    fn mean_pool(
        &self,
        output: &mut MlxTensor,
        hidden: &MlxTensor,
        mask: &MlxTensor,
        batch: usize,
        seq: usize,
        hidden_dim: usize,
    ) -> crate::Result<()> {
        // hidden: [batch*seq*hidden_dim] -> [batch, seq, hidden_dim]
        // mask: [batch*seq] -> [batch, seq, 1] for broadcasting
        let h = mlx_rs::ops::reshape(&hidden.0, &[batch as i32, seq as i32, hidden_dim as i32])
            .map_err(mlx_err)?;
        let m = mlx_rs::ops::reshape(&mask.0, &[batch as i32, seq as i32, 1]).map_err(mlx_err)?;

        // Masked sum: sum(hidden * mask, axis=1)
        let masked = mlx_rs::ops::multiply(&h, &m).map_err(mlx_err)?;
        let sum = masked.sum_axis(1, false).map_err(mlx_err)?; // [batch, hidden_dim]

        // Count: sum(mask, axis=1), clamp to avoid div-by-zero
        let count = m.sum_axis(1, false).map_err(mlx_err)?; // [batch, 1]
        let eps = Array::from_f32(1e-9_f32);
        let count = mlx_rs::ops::maximum(&count, &eps).map_err(mlx_err)?;

        let mean = mlx_rs::ops::divide(&sum, &count).map_err(mlx_err)?;
        output.0 = mlx_rs::ops::reshape(&mean, &[(batch * hidden_dim) as i32]).map_err(mlx_err)?;
        Ok(())
    }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "mlx-rs requires i32 for shape params; ML tensor dims fit in i32"
    )]
    fn l2_normalize(&self, data: &mut MlxTensor, rows: usize, cols: usize) -> crate::Result<()> {
        let x = mlx_rs::ops::reshape(&data.0, &[rows as i32, cols as i32]).map_err(mlx_err)?;
        let sq = x.square().map_err(mlx_err)?;
        let norms = sq
            .sum_axis(-1, true)
            .map_err(mlx_err)?
            .sqrt()
            .map_err(mlx_err)?;
        let eps = Array::from_f32(1e-12_f32);
        let norms = mlx_rs::ops::maximum(&norms, &eps).map_err(mlx_err)?;
        let normalized = mlx_rs::ops::divide(&x, &norms).map_err(mlx_err)?;
        data.0 = mlx_rs::ops::reshape(&normalized, &[(rows * cols) as i32]).map_err(mlx_err)?;
        Ok(())
    }

    fn banded_qk(
        &self,
        _q: &MlxTensor,
        _k: &MlxTensor,
        _scores: &mut MlxTensor,
        _bh: usize,
        _seq: usize,
        _hd: usize,
        _w: usize,
        _sqk: usize,
        _ss: usize,
    ) -> crate::Result<()> {
        Err(crate::Error::Other(anyhow::anyhow!(
            "banded_qk not implemented for MLX"
        )))
    }

    fn banded_sv(
        &self,
        _s: &MlxTensor,
        _v: &MlxTensor,
        _o: &mut MlxTensor,
        _bh: usize,
        _seq: usize,
        _hd: usize,
        _w: usize,
        _ss: usize,
        _sv: usize,
        _so: usize,
    ) -> crate::Result<()> {
        Err(crate::Error::Other(anyhow::anyhow!(
            "banded_sv not implemented for MLX"
        )))
    }

    fn banded_softmax(
        &self,
        _s: &mut MlxTensor,
        _rows: usize,
        _w: usize,
        _scale: f32,
    ) -> crate::Result<()> {
        Err(crate::Error::Other(anyhow::anyhow!(
            "banded_softmax not implemented for MLX"
        )))
    }

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        reason = "mlx-rs requires i32 for shape params; ML tensor dims fit in i32"
    )]
    fn to_host(
        &self,
        tensor: &MlxTensor,
        batch: usize,
        dim: usize,
    ) -> crate::Result<Vec<Vec<f32>>> {
        let t = mlx_rs::ops::reshape(&tensor.0, &[batch as i32, dim as i32]).map_err(mlx_err)?;
        let t = t.as_dtype(mlx_rs::Dtype::Float32).map_err(mlx_err)?;
        // ALL computation happens here — eval triggers the entire lazy graph.
        t.eval().map_err(mlx_err)?;
        let flat: &[f32] = t.as_slice();
        let mut results = Vec::with_capacity(batch);
        for b in 0..batch {
            results.push(flat[b * dim..(b + 1) * dim].to_vec());
        }
        Ok(results)
    }
}

// ---------------------------------------------------------------------------
// ClassicBert weight loading
// ---------------------------------------------------------------------------

/// Configuration parsed from a ClassicBert `config.json`.
pub struct ClassicBertConfig {
    /// Hidden dimension (e.g. 384 for BGE-small).
    pub hidden_size: usize,
    /// Number of transformer layers.
    pub num_hidden_layers: usize,
    /// Number of attention heads.
    pub num_attention_heads: usize,
    /// FFN intermediate dimension.
    pub intermediate_size: usize,
    /// Maximum position embedding length.
    pub max_position_embeddings: usize,
    /// Layer normalization epsilon.
    pub layer_norm_eps: f32,
}

impl ClassicBertConfig {
    /// Parse from a `config.json` serde value.
    ///
    /// # Errors
    ///
    /// Returns an error if required keys are missing.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "HuggingFace config ints always fit in usize"
    )]
    pub fn from_json(v: &serde_json::Value) -> crate::Result<Self> {
        let get = |key: &str| -> crate::Result<usize> {
            v.get(key)
                .and_then(serde_json::Value::as_u64)
                .map(|n| n as usize)
                .ok_or_else(|| crate::Error::Other(anyhow::anyhow!("missing config key: {key}")))
        };
        let get_f64 = |key: &str| -> crate::Result<f64> {
            v.get(key)
                .and_then(serde_json::Value::as_f64)
                .ok_or_else(|| crate::Error::Other(anyhow::anyhow!("missing config key: {key}")))
        };

        #[expect(
            clippy::cast_possible_truncation,
            reason = "layer_norm_eps is always a small float"
        )]
        let layer_norm_eps =
            get_f64("layer_norm_eps").or_else(|_| get_f64("layer_norm_epsilon"))? as f32;

        Ok(Self {
            hidden_size: get("hidden_size")?,
            num_hidden_layers: get("num_hidden_layers")?,
            num_attention_heads: get("num_attention_heads")?,
            intermediate_size: get("intermediate_size")?,
            max_position_embeddings: get("max_position_embeddings").unwrap_or(512),
            layer_norm_eps,
        })
    }
}

/// Remove a weight from the map by name, returning an error if missing.
fn take_weight(weights: &mut HashMap<String, Array>, name: &str) -> crate::Result<Array> {
    weights
        .remove(name)
        .ok_or_else(|| crate::Error::Other(anyhow::anyhow!("missing weight: {name}")))
}

/// Load a `ClassicBert` model (e.g. BGE-small) for the MLX driver.
///
/// Downloads model files via `hf-hub`, loads safetensors weights into MLX
/// arrays wrapped as [`MlxTensor`], fuses Q/K/V per layer, and returns an
/// [`EmbedBackend`] backed by [`GenericBackend`]. Because [`MlxTensor`]
/// implements `Send + Sync`, no extra mutex is required.
///
/// # Errors
///
/// Returns an error if the model cannot be downloaded, config cannot be
/// parsed, or weight loading fails.
#[expect(
    clippy::too_many_lines,
    reason = "weight loading is inherently verbose per-field"
)]
pub fn load_classic_mlx(model_repo: &str) -> crate::Result<Box<dyn crate::backend::EmbedBackend>> {
    let api = Api::new().map_err(|e| crate::Error::Download(e.to_string()))?;
    let repo = api.model(model_repo.to_string());

    let config_path = repo
        .get("config.json")
        .map_err(|e| crate::Error::Download(e.to_string()))?;
    let weights_path = repo
        .get("model.safetensors")
        .map_err(|e| crate::Error::Download(e.to_string()))?;

    // Parse config
    let config_str = std::fs::read_to_string(&config_path).map_err(|e| crate::Error::Io {
        path: config_path.display().to_string(),
        source: e,
    })?;
    let config_json: serde_json::Value = serde_json::from_str(&config_str)
        .map_err(|e| crate::Error::Other(anyhow::anyhow!("config parse error: {e}")))?;
    let config = ClassicBertConfig::from_json(&config_json)?;

    // Load weights into MLX arrays
    let mut weights = Array::load_safetensors(&weights_path).map_err(mlx_err)?;

    // Helper: flatten an Array to 1D and wrap in MlxTensor.
    let flatten = |a: Array| -> crate::Result<MlxTensor> {
        let n: i32 = a.shape().iter().product();
        mlx_rs::ops::reshape(&a, &[n])
            .map_err(mlx_err)
            .map(MlxTensor)
    };

    // Build per-layer weights, fusing Q/K/V
    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for i in 0..config.num_hidden_layers {
        let prefix = format!("encoder.layer.{i}");

        // Fuse Q/K/V weights into single [3*hidden, hidden] matrix
        let q_w = take_weight(
            &mut weights,
            &format!("{prefix}.attention.self.query.weight"),
        )?;
        let k_w = take_weight(&mut weights, &format!("{prefix}.attention.self.key.weight"))?;
        let v_w = take_weight(
            &mut weights,
            &format!("{prefix}.attention.self.value.weight"),
        )?;
        let qkv_weight = mlx_rs::ops::concatenate_axis(&[&q_w, &k_w, &v_w], 0).map_err(mlx_err)?;

        // Fuse Q/K/V biases
        let q_b = take_weight(&mut weights, &format!("{prefix}.attention.self.query.bias"))?;
        let k_b = take_weight(&mut weights, &format!("{prefix}.attention.self.key.bias"))?;
        let v_b = take_weight(&mut weights, &format!("{prefix}.attention.self.value.bias"))?;
        let qkv_bias = mlx_rs::ops::concatenate_axis(&[&q_b, &k_b, &v_b], 0).map_err(mlx_err)?;

        layers.push(ClassicBertLayerWeights {
            qkv_weight: flatten(qkv_weight)?,
            qkv_bias: flatten(qkv_bias)?,
            output_weight: flatten(take_weight(
                &mut weights,
                &format!("{prefix}.attention.output.dense.weight"),
            )?)?,
            output_bias: flatten(take_weight(
                &mut weights,
                &format!("{prefix}.attention.output.dense.bias"),
            )?)?,
            output_ln_weight: flatten(take_weight(
                &mut weights,
                &format!("{prefix}.attention.output.LayerNorm.weight"),
            )?)?,
            output_ln_bias: flatten(take_weight(
                &mut weights,
                &format!("{prefix}.attention.output.LayerNorm.bias"),
            )?)?,
            ffn_inter_weight: flatten(take_weight(
                &mut weights,
                &format!("{prefix}.intermediate.dense.weight"),
            )?)?,
            ffn_inter_bias: flatten(take_weight(
                &mut weights,
                &format!("{prefix}.intermediate.dense.bias"),
            )?)?,
            ffn_out_weight: flatten(take_weight(
                &mut weights,
                &format!("{prefix}.output.dense.weight"),
            )?)?,
            ffn_out_bias: flatten(take_weight(
                &mut weights,
                &format!("{prefix}.output.dense.bias"),
            )?)?,
            ffn_ln_weight: flatten(take_weight(
                &mut weights,
                &format!("{prefix}.output.LayerNorm.weight"),
            )?)?,
            ffn_ln_bias: flatten(take_weight(
                &mut weights,
                &format!("{prefix}.output.LayerNorm.bias"),
            )?)?,
        });
    }

    let hidden = config.hidden_size;
    let num_heads = config.num_attention_heads;
    let head_dim = hidden / num_heads;
    let max_tokens = config.max_position_embeddings;

    let arch = ClassicBertArch {
        weights: ClassicBertWeights {
            word_embeddings: flatten(take_weight(
                &mut weights,
                "embeddings.word_embeddings.weight",
            )?)?,
            position_embeddings: flatten(take_weight(
                &mut weights,
                "embeddings.position_embeddings.weight",
            )?)?,
            token_type_embeddings: flatten(take_weight(
                &mut weights,
                "embeddings.token_type_embeddings.weight",
            )?)?,
            emb_ln_weight: flatten(take_weight(&mut weights, "embeddings.LayerNorm.weight")?)?,
            emb_ln_bias: flatten(take_weight(&mut weights, "embeddings.LayerNorm.bias")?)?,
            layers,
            num_heads,
            head_dim,
            hidden_dim: hidden,
            intermediate_dim: config.intermediate_size,
            layer_norm_eps: config.layer_norm_eps,
        },
    };

    let driver = MlxDriver::new()?;

    // MLX copies weights into its own arrays, but GenericBackend needs an mmap
    // to satisfy the lifetime invariant. Create a dummy mmap from the safetensors
    // file (it's already cached by hf-hub).
    let file = std::fs::File::open(&weights_path).map_err(|e| crate::Error::Io {
        path: weights_path.display().to_string(),
        source: e,
    })?;
    #[expect(unsafe_code, reason = "mmap of read-only cached file is safe")]
    let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| crate::Error::Io {
        path: weights_path.display().to_string(),
        source: e,
    })?;

    tracing::info!(
        model_repo,
        hidden,
        layers = config.num_hidden_layers,
        heads = num_heads,
        intermediate = config.intermediate_size,
        max_tokens,
        "ClassicBert loaded on MLX (driver/arch)"
    );

    Ok(Box::new(GenericBackend::new(
        driver, arch, max_tokens, true, mmap,
    )))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{EmbedBackend, Encoding};

    const BGE_SMALL: &str = "BAAI/bge-small-en-v1.5";

    /// Verify that the MlxDriver can be constructed.
    #[test]
    fn mlx_driver_creates() {
        let driver = MlxDriver::new().unwrap();
        let zeros = driver.alloc_zeros(16).unwrap();
        zeros.0.eval().map_err(mlx_err).unwrap();
        assert_eq!(zeros.0.as_slice::<f32>().len(), 16);
    }

    /// Load ClassicBert via driver/arch and embed a short sequence.
    /// Verifies the full pipeline produces a 384-dim unit vector.
    #[test]
    #[ignore = "requires model download"]
    fn mlx_driver_arch_bge_small() {
        let backend = load_classic_mlx(BGE_SMALL).unwrap();
        let enc = Encoding {
            input_ids: vec![101, 2023, 2003, 1037, 3231, 102],
            attention_mask: vec![1, 1, 1, 1, 1, 1],
            token_type_ids: vec![0, 0, 0, 0, 0, 0],
        };
        let results = backend.embed_batch(&[enc]).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].len(), 384);

        let norm: f32 = results[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        eprintln!("L2 norm = {norm:.6}");
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "L2 norm should be ~1.0, got {norm}"
        );
    }

    /// Compare MLX driver/arch output against the monolithic MLX backend.
    #[test]
    #[ignore = "requires model download"]
    fn mlx_driver_matches_monolithic() {
        let driver_backend = load_classic_mlx(BGE_SMALL).unwrap();
        let mono_backend =
            crate::backend::mlx::MlxBackend::load(BGE_SMALL, &crate::backend::DeviceHint::Auto)
                .unwrap();

        let enc = Encoding {
            input_ids: vec![101, 2023, 2003, 1037, 3231, 102],
            attention_mask: vec![1, 1, 1, 1, 1, 1],
            token_type_ids: vec![0, 0, 0, 0, 0, 0],
        };

        let driver_result = driver_backend.embed_batch(&[enc.clone()]).unwrap();
        let mono_result = mono_backend.embed_batch(&[enc]).unwrap();

        let cosine: f32 = driver_result[0]
            .iter()
            .zip(&mono_result[0])
            .map(|(a, b)| a * b)
            .sum();
        eprintln!("cosine(MLX driver/arch, MLX monolithic) = {cosine:.6}");
        assert!(
            cosine > 0.99,
            "driver/arch and monolithic should produce near-identical embeddings, got cosine={cosine:.6}"
        );
    }

    /// Multi-sequence batch test — verifies batching and padding work.
    #[test]
    #[ignore = "requires model download"]
    fn mlx_driver_batch() {
        let backend = load_classic_mlx(BGE_SMALL).unwrap();
        let enc1 = Encoding {
            input_ids: vec![101, 7592, 102],
            attention_mask: vec![1, 1, 1],
            token_type_ids: vec![0, 0, 0],
        };
        let enc2 = Encoding {
            input_ids: vec![101, 2023, 2003, 1037, 3231, 102],
            attention_mask: vec![1, 1, 1, 1, 1, 1],
            token_type_ids: vec![0, 0, 0, 0, 0, 0],
        };
        let results = backend.embed_batch(&[enc1, enc2]).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].len(), 384);
        assert_eq!(results[1].len(), 384);

        // Both should be unit vectors
        for (i, r) in results.iter().enumerate() {
            let norm: f32 = r.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-4,
                "batch element {i}: L2 norm should be ~1.0, got {norm}"
            );
        }
    }
}
