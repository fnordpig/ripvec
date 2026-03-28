//! CPU compute driver using ndarray + system BLAS.
//!
//! Implements the [`Driver`] trait using flat `Vec<f32>` tensors and scalar
//! loops for all operations except GEMM, which delegates to ndarray (backed
//! by Accelerate on macOS, OpenBLAS on Linux) for hardware-optimised matrix
//! multiplication.
//!
//! # Design
//!
//! - **`type Tensor = Vec<f32>`**: simplest possible representation. The
//!   `Driver` trait passes explicit dimensions (`m`, `n`, `k`, `rows`, `cols`)
//!   to every operation, so tensors need not track their own shape.
//! - **`begin_batch` / `end_batch`**: no-ops (no GPU command buffers).
//! - **Input IDs as f32**: token IDs are small positive integers (0-50000),
//!   exactly representable as `f32`, and cast back to `usize` in
//!   [`embedding_lookup`](Driver::embedding_lookup).

// Ensure the BLAS linker symbols are pulled in.
#[cfg(feature = "cpu-accelerate")]
extern crate accelerate_src;
#[cfg(feature = "cpu")]
extern crate blas_src;
#[cfg(all(feature = "cpu", not(feature = "cpu-accelerate")))]
extern crate openblas_src;

use std::path::Path;

use safetensors::SafeTensors;

use super::{BatchInputs, Driver};
use crate::backend::arch::classic_bert::{
    ClassicBertArch, ClassicBertLayerWeights, ClassicBertWeights,
};
use crate::backend::arch::modern_bert::{
    ModernBertArch, ModernBertLayerWeights, ModernBertWeights, RopeCache,
};
use crate::backend::Encoding;
// ---------------------------------------------------------------------------
// CpuDriver
// ---------------------------------------------------------------------------

/// CPU compute driver: flat `Vec<f32>` tensors, ndarray BLAS for GEMM.
///
/// All operations are synchronous and run on the calling thread. There is no
/// device memory, no command buffers, and no kernel dispatch overhead.
pub struct CpuDriver;

impl CpuDriver {
    /// Create a new CPU driver.
    ///
    /// Always succeeds -- there is no device to initialise.
    pub fn new() -> crate::Result<Self> {
        Ok(Self)
    }
}

// ---------------------------------------------------------------------------
// Config types (mirrors metal.rs config structs)
// ---------------------------------------------------------------------------

/// Parsed `ClassicBert` model configuration from `config.json`.
///
/// Contains geometry and hyperparameters needed to build the `ClassicBert`
/// architecture and load weights (e.g. `BAAI/bge-small-en-v1.5`).
pub struct ClassicBertConfig {
    /// Hidden dimension (384 for BGE-small).
    pub hidden_size: usize,
    /// FFN intermediate dimension (1536 for BGE-small).
    pub intermediate_size: usize,
    /// Number of encoder layers (12 for BGE-small).
    pub num_hidden_layers: usize,
    /// Number of attention heads (12 for BGE-small).
    pub num_attention_heads: usize,
    /// Layer normalization epsilon (typically 1e-12).
    pub layer_norm_eps: f32,
    /// Maximum position embeddings / sequence length (512 for BGE-small).
    pub max_position_embeddings: usize,
    /// Vocabulary size (30522 for BGE-small).
    pub vocab_size: usize,
}

impl ClassicBertConfig {
    /// Parse a `ClassicBert` config from a `config.json` value.
    ///
    /// Expects standard BERT config keys (`hidden_size`, `intermediate_size`, etc.).
    ///
    /// # Errors
    ///
    /// Returns an error if any required field is missing or has an unexpected type.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "config ints are small positive values"
    )]
    pub fn from_json(json: &serde_json::Value) -> crate::Result<Self> {
        let get_usize = |key: &str| -> crate::Result<usize> {
            json.get(key)
                .and_then(serde_json::Value::as_u64)
                .map(|v| v as usize)
                .ok_or_else(|| crate::Error::Cpu(format!("config.json missing or invalid: {key}")))
        };
        let get_f64 = |key: &str| -> crate::Result<f64> {
            json.get(key)
                .and_then(serde_json::Value::as_f64)
                .ok_or_else(|| crate::Error::Cpu(format!("config.json missing or invalid: {key}")))
        };

        Ok(Self {
            hidden_size: get_usize("hidden_size")?,
            intermediate_size: get_usize("intermediate_size")?,
            num_hidden_layers: get_usize("num_hidden_layers")?,
            num_attention_heads: get_usize("num_attention_heads")?,
            layer_norm_eps: get_f64("layer_norm_eps")
                .or_else(|_| get_f64("layer_norm_epsilon"))
                .unwrap_or(1e-12) as f32,
            max_position_embeddings: get_usize("max_position_embeddings").unwrap_or(512),
            vocab_size: get_usize("vocab_size")?,
        })
    }
}

// ---------------------------------------------------------------------------
// Safetensors -> Vec<f32> helpers
// ---------------------------------------------------------------------------

/// Load a named tensor from safetensors as flat `Vec<f32>`.
///
/// The tensor must be stored in `f32` (little-endian) format.
fn load_tensor_flat(tensors: &SafeTensors<'_>, name: &str) -> crate::Result<Vec<f32>> {
    let tensor = tensors
        .tensor(name)
        .map_err(|_| crate::Error::Cpu(format!("missing weight: {name}")))?;
    let data: Vec<f32> = tensor
        .data()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    Ok(data)
}

// ---------------------------------------------------------------------------
// Weight loading
// ---------------------------------------------------------------------------

impl CpuDriver {
    /// Load `ClassicBert` weights from a safetensors file into `Vec<f32>` tensors.
    ///
    /// Fuses separate Q, K, V weight matrices into a single `[3*hidden, hidden]`
    /// tensor (and `[3*hidden]` bias) at load time for a single GEMM per layer.
    ///
    /// Returns `(arch, mmap)` -- the mmap is kept alive for API consistency with
    /// the Metal driver, though CPU tensors are independent copies.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened, safetensors parsing fails,
    /// or any expected weight tensor is missing.
    #[expect(unsafe_code, reason = "memmap2::Mmap::map requires unsafe")]
    pub fn load_classic_bert_weights(
        &self,
        weights_path: &Path,
        config: &ClassicBertConfig,
    ) -> crate::Result<(ClassicBertArch<Vec<f32>>, memmap2::Mmap)> {
        let file = std::fs::File::open(weights_path).map_err(|e| crate::Error::Io {
            path: weights_path.display().to_string(),
            source: e,
        })?;
        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| crate::Error::Io {
            path: weights_path.display().to_string(),
            source: e,
        })?;

        let tensors = SafeTensors::deserialize(&mmap)
            .map_err(|e| crate::Error::Cpu(format!("safetensors parse: {e}")))?;

        let hidden = config.hidden_size;
        let num_layers = config.num_hidden_layers;
        let num_heads = config.num_attention_heads;
        let head_dim = hidden / num_heads;
        let intermediate = config.intermediate_size;

        // Build per-layer weights with fused QKV
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let prefix = format!("encoder.layer.{i}");

            // Fuse Q+K+V weights into [3*hidden, hidden] and bias into [3*hidden].
            let q_w = load_tensor_flat(&tensors, &format!("{prefix}.attention.self.query.weight"))?;
            let k_w = load_tensor_flat(&tensors, &format!("{prefix}.attention.self.key.weight"))?;
            let v_w = load_tensor_flat(&tensors, &format!("{prefix}.attention.self.value.weight"))?;
            let q_b = load_tensor_flat(&tensors, &format!("{prefix}.attention.self.query.bias"))?;
            let k_b = load_tensor_flat(&tensors, &format!("{prefix}.attention.self.key.bias"))?;
            let v_b = load_tensor_flat(&tensors, &format!("{prefix}.attention.self.value.bias"))?;

            let mut fused_qkv_w = Vec::with_capacity(3 * hidden * hidden);
            fused_qkv_w.extend_from_slice(&q_w);
            fused_qkv_w.extend_from_slice(&k_w);
            fused_qkv_w.extend_from_slice(&v_w);

            let mut fused_qkv_b = Vec::with_capacity(3 * hidden);
            fused_qkv_b.extend_from_slice(&q_b);
            fused_qkv_b.extend_from_slice(&k_b);
            fused_qkv_b.extend_from_slice(&v_b);

            layers.push(ClassicBertLayerWeights {
                qkv_weight: fused_qkv_w,
                qkv_bias: fused_qkv_b,
                output_weight: load_tensor_flat(
                    &tensors,
                    &format!("{prefix}.attention.output.dense.weight"),
                )?,
                output_bias: load_tensor_flat(
                    &tensors,
                    &format!("{prefix}.attention.output.dense.bias"),
                )?,
                output_ln_weight: load_tensor_flat(
                    &tensors,
                    &format!("{prefix}.attention.output.LayerNorm.weight"),
                )?,
                output_ln_bias: load_tensor_flat(
                    &tensors,
                    &format!("{prefix}.attention.output.LayerNorm.bias"),
                )?,
                ffn_inter_weight: load_tensor_flat(
                    &tensors,
                    &format!("{prefix}.intermediate.dense.weight"),
                )?,
                ffn_inter_bias: load_tensor_flat(
                    &tensors,
                    &format!("{prefix}.intermediate.dense.bias"),
                )?,
                ffn_out_weight: load_tensor_flat(
                    &tensors,
                    &format!("{prefix}.output.dense.weight"),
                )?,
                ffn_out_bias: load_tensor_flat(&tensors, &format!("{prefix}.output.dense.bias"))?,
                ffn_ln_weight: load_tensor_flat(
                    &tensors,
                    &format!("{prefix}.output.LayerNorm.weight"),
                )?,
                ffn_ln_bias: load_tensor_flat(
                    &tensors,
                    &format!("{prefix}.output.LayerNorm.bias"),
                )?,
            });
        }

        // Embedding weights
        let weights = ClassicBertWeights {
            word_embeddings: load_tensor_flat(&tensors, "embeddings.word_embeddings.weight")?,
            position_embeddings: load_tensor_flat(
                &tensors,
                "embeddings.position_embeddings.weight",
            )?,
            token_type_embeddings: load_tensor_flat(
                &tensors,
                "embeddings.token_type_embeddings.weight",
            )?,
            emb_ln_weight: load_tensor_flat(&tensors, "embeddings.LayerNorm.weight")?,
            emb_ln_bias: load_tensor_flat(&tensors, "embeddings.LayerNorm.bias")?,
            layers,
            num_heads,
            head_dim,
            hidden_dim: hidden,
            intermediate_dim: intermediate,
            layer_norm_eps: config.layer_norm_eps,
        };

        Ok((ClassicBertArch { weights }, mmap))
    }

    /// Load `ModernBERT` weights from a safetensors file into `Vec<f32>` tensors.
    ///
    /// Weight names follow the `nomic-ai/modernbert-embed-base` convention:
    /// `layers.{i}.attn.Wqkv.weight`, `layers.{i}.mlp.Wi.weight`, etc.
    ///
    /// Returns `(arch, mmap)` where mmap is kept alive for API consistency.
    pub fn load_modern_bert_weights(
        &self,
        weights_path: &Path,
        config: &ModernBertConfig,
    ) -> crate::Result<(ModernBertArch<Vec<f32>>, memmap2::Mmap)> {
        let file = std::fs::File::open(weights_path).map_err(|e| crate::Error::Io {
            path: weights_path.display().to_string(),
            source: e,
        })?;
        #[expect(unsafe_code, reason = "memmap2 requires unsafe for mmap")]
        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| crate::Error::Io {
            path: weights_path.display().to_string(),
            source: e,
        })?;

        let tensors = SafeTensors::deserialize(&mmap)
            .map_err(|e| crate::Error::Cpu(format!("safetensors parse: {e}")))?;

        let hidden = config.hidden_size;
        let num_layers = config.num_hidden_layers;
        let num_heads = config.num_attention_heads;
        let head_dim = hidden / num_heads;
        let intermediate = config.intermediate_size;
        let global_attn_every_n = config.global_attn_every_n_layers;

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let qkv_weight = load_tensor_flat(&tensors, &format!("layers.{i}.attn.Wqkv.weight"))?;
            let output_weight = load_tensor_flat(&tensors, &format!("layers.{i}.attn.Wo.weight"))?;
            let attn_norm_weight = if i == 0 {
                None
            } else {
                Some(load_tensor_flat(
                    &tensors,
                    &format!("layers.{i}.attn_norm.weight"),
                )?)
            };
            let mlp_wi_weight = load_tensor_flat(&tensors, &format!("layers.{i}.mlp.Wi.weight"))?;
            let mlp_wo_weight = load_tensor_flat(&tensors, &format!("layers.{i}.mlp.Wo.weight"))?;
            let mlp_norm_weight =
                load_tensor_flat(&tensors, &format!("layers.{i}.mlp_norm.weight"))?;

            let is_global = i % global_attn_every_n == 0;

            layers.push(ModernBertLayerWeights {
                qkv_weight,
                output_weight,
                attn_norm_weight,
                mlp_wi_weight,
                mlp_wo_weight,
                mlp_norm_weight,
                is_global,
            });
        }

        let tok_embeddings = load_tensor_flat(&tensors, "embeddings.tok_embeddings.weight")?;
        let emb_norm_weight = load_tensor_flat(&tensors, "embeddings.norm.weight")?;
        let final_norm_weight = load_tensor_flat(&tensors, "final_norm.weight")?;
        let zero_bias = vec![0.0f32; hidden];

        let weights = ModernBertWeights {
            tok_embeddings,
            emb_norm_weight,
            final_norm_weight,
            zero_bias,
            layers,
            num_heads,
            head_dim,
            hidden_dim: hidden,
            intermediate_dim: intermediate,
            layer_norm_eps: config.norm_eps,
            local_window: config.local_attention,
        };

        // Build RoPE caches
        let max_seq = config.max_position_embeddings;
        let global_rope = build_rope_cache_cpu(head_dim, max_seq, config.global_rope_theta);
        let local_rope = build_rope_cache_cpu(head_dim, max_seq, config.local_rope_theta);

        let arch = ModernBertArch {
            weights,
            global_rope,
            local_rope,
            max_layers: None,
        };

        Ok((arch, mmap))
    }
}

/// Parsed `ModernBERT` config from `config.json`.
pub struct ModernBertConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub global_attn_every_n_layers: usize,
    pub local_attention: usize,
    pub global_rope_theta: f32,
    pub local_rope_theta: f32,
    pub norm_eps: f32,
    pub max_position_embeddings: usize,
    pub vocab_size: usize,
}

impl ModernBertConfig {
    /// Parse from a `config.json` value.
    pub fn from_json(json: &serde_json::Value) -> crate::Result<Self> {
        let get_usize = |key: &str| -> crate::Result<usize> {
            json.get(key)
                .and_then(serde_json::Value::as_u64)
                .map(|v| v as usize)
                .ok_or_else(|| crate::Error::Cpu(format!("config.json missing or invalid: {key}")))
        };
        let get_f64 = |key: &str| -> crate::Result<f64> {
            json.get(key)
                .and_then(serde_json::Value::as_f64)
                .ok_or_else(|| crate::Error::Cpu(format!("config.json missing or invalid: {key}")))
        };

        Ok(Self {
            hidden_size: get_usize("hidden_size")?,
            intermediate_size: get_usize("intermediate_size")?,
            num_hidden_layers: get_usize("num_hidden_layers")?,
            num_attention_heads: get_usize("num_attention_heads")?,
            global_attn_every_n_layers: get_usize("global_attn_every_n_layers")?,
            local_attention: get_usize("local_attention")?,
            global_rope_theta: get_f64("global_rope_theta")? as f32,
            local_rope_theta: get_f64("local_rope_theta")? as f32,
            norm_eps: get_f64("norm_eps").unwrap_or(1e-5) as f32,
            max_position_embeddings: get_usize("max_position_embeddings")?,
            vocab_size: get_usize("vocab_size")?,
        })
    }
}

/// Build RoPE cos/sin tables as `Vec<f32>` for CPU.
fn build_rope_cache_cpu(head_dim: usize, max_seq: usize, theta: f32) -> RopeCache<Vec<f32>> {
    let half_dim = head_dim / 2;
    let n = max_seq * half_dim;
    let mut cos_data = Vec::with_capacity(n);
    let mut sin_data = Vec::with_capacity(n);

    for pos in 0..max_seq {
        for d in 0..half_dim {
            let freq = (pos as f32) / theta.powf(2.0 * d as f32 / head_dim as f32);
            cos_data.push(freq.cos());
            sin_data.push(freq.sin());
        }
    }

    RopeCache {
        cos: cos_data,
        sin: sin_data,
    }
}

// ---------------------------------------------------------------------------
// Activation helpers
// ---------------------------------------------------------------------------

/// GELU activation (tanh approximation).
///
/// `x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`
fn gelu_scalar(x: f32) -> f32 {
    x * 0.5 * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (x + 0.044_715 * x * x * x)).tanh())
}

/// Softmax over a mutable slice, in-place.
///
/// Uses the numerically stable `exp(x - max) / sum(exp(x - max))` form.
fn softmax_inplace(vals: &mut [f32]) {
    let max = vals.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for v in vals.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv_sum = 1.0 / sum;
    for v in vals.iter_mut() {
        *v *= inv_sum;
    }
}

// ---------------------------------------------------------------------------
// Driver trait implementation
// ---------------------------------------------------------------------------

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "dimension values and token IDs are small positive integers"
)]
impl Driver for CpuDriver {
    type Tensor = Vec<f32>;

    // begin_batch / end_batch: no-ops for CPU.

    fn alloc_zeros(&self, n: usize) -> crate::Result<Vec<f32>> {
        Ok(vec![0.0; n])
    }

    fn clone_tensor(&self, tensor: &Vec<f32>, _n: usize) -> crate::Result<Vec<f32>> {
        Ok(tensor.clone())
    }

    fn prepare_batch(
        &self,
        encodings: &[Encoding],
        max_seq: usize,
    ) -> crate::Result<BatchInputs<Vec<f32>>> {
        let batch = encodings.len();
        let total = batch * max_seq;

        // Input IDs stored as f32 (cast to usize in embedding_lookup).
        let mut input_ids = vec![0.0_f32; total];
        let mut token_type_ids = vec![0.0_f32; total];
        let mut position_ids = vec![0.0_f32; total];
        let mut attn_mask_int = vec![0.0_f32; total];

        for (b, enc) in encodings.iter().enumerate() {
            let seq_len = enc.input_ids.len();
            let offset = b * max_seq;
            for (i, &id) in enc.input_ids.iter().enumerate() {
                input_ids[offset + i] = id as f32;
            }
            for (i, &id) in enc.token_type_ids.iter().enumerate() {
                token_type_ids[offset + i] = id as f32;
            }
            for i in 0..seq_len {
                position_ids[offset + i] = i as f32;
            }
            for (i, &m) in enc.attention_mask.iter().enumerate() {
                attn_mask_int[offset + i] = m as f32;
            }
        }

        // Float attention bias mask: 0.0 for real tokens, -1e9 for padding.
        let float_mask: Vec<f32> = attn_mask_int
            .iter()
            .map(|&m| if m > 0.5 { 0.0 } else { -1e9 })
            .collect();

        // Pooling mask: 1.0 for real tokens, 0.0 for padding.
        let pooling_mask: Vec<f32> = attn_mask_int
            .iter()
            .map(|&m| if m > 0.5 { 1.0 } else { 0.0 })
            .collect();

        // Per-sequence lengths and total token count.
        let seq_lengths: Vec<usize> = encodings.iter().map(|e| e.input_ids.len()).collect();
        let total_tokens: usize = seq_lengths.iter().sum();

        Ok(BatchInputs {
            input_ids,
            attention_mask: attn_mask_int,
            token_type_ids,
            position_ids,
            float_mask,
            pooling_mask,
            batch,
            max_seq,
            total_tokens,
            seq_lengths,
            cu_seqlens: None, // padded mode
        })
    }

    fn pad_to_batch(
        &self,
        flat: &Vec<f32>,
        padded: &mut Vec<f32>,
        seq_lengths: &[usize],
        max_seq: usize,
        dim: usize,
    ) -> crate::Result<()> {
        let batch = seq_lengths.len();
        padded.resize(batch * max_seq * dim, 0.0);
        padded.fill(0.0);
        let mut offset = 0;
        for (b, &len) in seq_lengths.iter().enumerate() {
            for t in 0..len {
                let src = (offset + t) * dim;
                let dst = (b * max_seq + t) * dim;
                padded[dst..dst + dim].copy_from_slice(&flat[src..src + dim]);
            }
            offset += len;
        }
        Ok(())
    }

    fn unpad_from_batch(
        &self,
        padded: &Vec<f32>,
        flat: &mut Vec<f32>,
        seq_lengths: &[usize],
        max_seq: usize,
        dim: usize,
    ) -> crate::Result<()> {
        let total_tokens: usize = seq_lengths.iter().sum();
        flat.resize(total_tokens * dim, 0.0);
        let mut offset = 0;
        for (b, &len) in seq_lengths.iter().enumerate() {
            for t in 0..len {
                let src = (b * max_seq + t) * dim;
                let dst = (offset + t) * dim;
                flat[dst..dst + dim].copy_from_slice(&padded[src..src + dim]);
            }
            offset += len;
        }
        Ok(())
    }

    fn embedding_lookup(
        &self,
        word_ids: &Vec<f32>,
        embedding_table: &Vec<f32>,
        seq_len: usize,
        hidden: usize,
    ) -> crate::Result<Vec<f32>> {
        let mut output = vec![0.0; seq_len * hidden];
        for (i, &wid) in word_ids.iter().enumerate().take(seq_len) {
            let id = wid as usize;
            let src_start = id * hidden;
            let dst_start = i * hidden;
            output[dst_start..dst_start + hidden]
                .copy_from_slice(&embedding_table[src_start..src_start + hidden]);
        }
        Ok(output)
    }

    fn add_embeddings(
        &self,
        hidden: &mut Vec<f32>,
        table: &Vec<f32>,
        ids: &Vec<f32>,
        seq_len: usize,
        hidden_dim: usize,
    ) -> crate::Result<()> {
        for (i, &id_f) in ids.iter().enumerate().take(seq_len) {
            let id = id_f as usize;
            let tbl_start = id * hidden_dim;
            let hid_start = i * hidden_dim;
            for j in 0..hidden_dim {
                hidden[hid_start + j] += table[tbl_start + j];
            }
        }
        Ok(())
    }

    fn layer_norm(
        &self,
        output: &mut Vec<f32>,
        input: &Vec<f32>,
        weight: &Vec<f32>,
        bias: &Vec<f32>,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> crate::Result<()> {
        output.resize(rows * cols, 0.0);
        for r in 0..rows {
            let base = r * cols;
            let row = &input[base..base + cols];

            let mean: f32 = row.iter().sum::<f32>() / cols as f32;
            let var: f32 = row.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / cols as f32;
            let inv_std = 1.0 / (var + eps).sqrt();

            for c in 0..cols {
                output[base + c] = (row[c] - mean) * inv_std * weight[c] + bias[c];
            }
        }
        Ok(())
    }

    #[expect(
        clippy::many_single_char_names,
        reason = "a, b, m, n, k are standard GEMM parameter names from BLAS"
    )]
    fn gemm(
        &self,
        a: &Vec<f32>,
        b: &Vec<f32>,
        output: &mut Vec<f32>,
        m: usize,
        n: usize,
        k: usize,
        transpose_b: bool,
    ) -> crate::Result<()> {
        output.resize(m * n, 0.0);

        let a_view = ndarray::ArrayView2::from_shape((m, k), a)
            .map_err(|e| crate::Error::Cpu(format!("GEMM a shape error: {e}")))?;

        if transpose_b {
            // B is [n, k], need B^T = [k, n]
            let b_view = ndarray::ArrayView2::from_shape((n, k), b)
                .map_err(|e| crate::Error::Cpu(format!("GEMM b shape error: {e}")))?;
            let bt = b_view.t();
            let mut c = ndarray::Array2::zeros((m, n));
            ndarray::linalg::general_mat_mul(1.0, &a_view, &bt, 0.0, &mut c);
            output.clear();
            output.extend(c.iter());
        } else {
            // B is [k, n]
            let b_view = ndarray::ArrayView2::from_shape((k, n), b)
                .map_err(|e| crate::Error::Cpu(format!("GEMM b shape error: {e}")))?;
            let mut c = ndarray::Array2::zeros((m, n));
            ndarray::linalg::general_mat_mul(1.0, &a_view, &b_view, 0.0, &mut c);
            output.clear();
            output.extend(c.iter());
        }

        Ok(())
    }

    #[expect(
        clippy::many_single_char_names,
        reason = "a, b, m, n, k are standard GEMM parameter names from BLAS"
    )]
    fn gemm_batched(
        &self,
        a: &Vec<f32>,
        b: &Vec<f32>,
        output: &mut Vec<f32>,
        m: usize,
        n: usize,
        k: usize,
        transpose_b: bool,
        stride_a: usize,
        stride_b: usize,
        stride_c: usize,
        batch_count: usize,
    ) -> crate::Result<()> {
        output.resize(batch_count * stride_c, 0.0);

        for batch in 0..batch_count {
            let a_off = batch * stride_a;
            let b_off = batch * stride_b;
            let c_off = batch * stride_c;

            let a_slice = &a[a_off..a_off + m * k];
            let b_slice = if transpose_b {
                &b[b_off..b_off + n * k]
            } else {
                &b[b_off..b_off + k * n]
            };

            let a_view = ndarray::ArrayView2::from_shape((m, k), a_slice)
                .map_err(|e| crate::Error::Cpu(format!("batched GEMM a shape: {e}")))?;

            let mut c = ndarray::Array2::zeros((m, n));

            if transpose_b {
                let b_view = ndarray::ArrayView2::from_shape((n, k), b_slice)
                    .map_err(|e| crate::Error::Cpu(format!("batched GEMM b shape: {e}")))?;
                ndarray::linalg::general_mat_mul(1.0, &a_view, &b_view.t(), 0.0, &mut c);
            } else {
                let b_view = ndarray::ArrayView2::from_shape((k, n), b_slice)
                    .map_err(|e| crate::Error::Cpu(format!("batched GEMM b shape: {e}")))?;
                ndarray::linalg::general_mat_mul(1.0, &a_view, &b_view, 0.0, &mut c);
            }

            output[c_off..c_off + m * n].copy_from_slice(c.as_slice().unwrap());
        }

        Ok(())
    }

    fn fused_scale_mask_softmax(
        &self,
        scores: &mut Vec<f32>,
        mask: &Vec<f32>,
        batch: usize,
        num_heads: usize,
        seq_len: usize,
        scale: f32,
    ) -> crate::Result<()> {
        // scores: [batch * num_heads, seq_len, seq_len]
        // mask:   [batch, seq_len] (float mask: 0.0 or -1e9)
        for b in 0..batch {
            for h in 0..num_heads {
                for q in 0..seq_len {
                    let row_off = ((b * num_heads + h) * seq_len + q) * seq_len;
                    let row = &mut scores[row_off..row_off + seq_len];

                    // Scale and add mask
                    for kk in 0..seq_len {
                        row[kk] = row[kk] * scale + mask[b * seq_len + kk];
                    }

                    softmax_inplace(row);
                }
            }
        }
        Ok(())
    }

    fn fused_scale_mask_softmax_windowed(
        &self,
        scores: &mut Vec<f32>,
        mask: &Vec<f32>,
        batch: usize,
        num_heads: usize,
        seq_len: usize,
        scale: f32,
        window_size: usize,
    ) -> crate::Result<()> {
        let half_window = window_size / 2;

        for b in 0..batch {
            for h in 0..num_heads {
                for q in 0..seq_len {
                    let row_off = ((b * num_heads + h) * seq_len + q) * seq_len;
                    let row = &mut scores[row_off..row_off + seq_len];

                    for kk in 0..seq_len {
                        let dist = q.abs_diff(kk);
                        let window_mask = if dist > half_window { -1e9 } else { 0.0 };
                        row[kk] = row[kk] * scale + mask[b * seq_len + kk] + window_mask;
                    }

                    softmax_inplace(row);
                }
            }
        }
        Ok(())
    }

    fn build_attn_mask(
        &self,
        output: &mut Vec<f32>,
        int_mask: &Vec<f32>,
        n: usize,
    ) -> crate::Result<()> {
        output.resize(n, 0.0);
        for i in 0..n {
            output[i] = if int_mask[i] > 0.5 { 0.0 } else { -1e9 };
        }
        Ok(())
    }

    fn qkv_split(
        &self,
        q: &mut Vec<f32>,
        k: &mut Vec<f32>,
        v: &mut Vec<f32>,
        qkv: &Vec<f32>,
        batch: usize,
        seq: usize,
        hidden: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> crate::Result<()> {
        let total_head = batch * num_heads * seq * head_dim;
        q.resize(total_head, 0.0);
        k.resize(total_head, 0.0);
        v.resize(total_head, 0.0);

        // qkv is [batch*seq, 3*hidden]
        // output: Q, K, V each [batch*num_heads, seq, head_dim]
        for b in 0..batch {
            for s in 0..seq {
                let src_row = (b * seq + s) * 3 * hidden;
                for h in 0..num_heads {
                    for d in 0..head_dim {
                        let src_q = src_row + h * head_dim + d;
                        let src_k = src_row + hidden + h * head_dim + d;
                        let src_v = src_row + 2 * hidden + h * head_dim + d;
                        let dst = (b * num_heads + h) * seq * head_dim + s * head_dim + d;
                        q[dst] = qkv[src_q];
                        k[dst] = qkv[src_k];
                        v[dst] = qkv[src_v];
                    }
                }
            }
        }
        Ok(())
    }

    fn attn_reshape(
        &self,
        output: &mut Vec<f32>,
        input: &Vec<f32>,
        batch: usize,
        seq: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> crate::Result<()> {
        let hidden = num_heads * head_dim;
        let total = batch * seq * hidden;
        output.resize(total, 0.0);

        // input: [batch*num_heads, seq, head_dim]
        // output: [batch*seq, hidden]
        for b in 0..batch {
            for s in 0..seq {
                for h in 0..num_heads {
                    let src_off = (b * num_heads + h) * seq * head_dim + s * head_dim;
                    let dst_off = (b * seq + s) * hidden + h * head_dim;
                    output[dst_off..dst_off + head_dim]
                        .copy_from_slice(&input[src_off..src_off + head_dim]);
                }
            }
        }
        Ok(())
    }

    fn apply_rope(
        &self,
        qk: &mut Vec<f32>,
        cos: &Vec<f32>,
        sin: &Vec<f32>,
        num_rows: usize,
        seq_len: usize,
        head_dim: usize,
        _num_heads: usize,
    ) -> crate::Result<()> {
        let half = head_dim / 2;

        // qk: [batch*num_heads, seq_len, head_dim]
        // cos/sin: [max_seq, half_dim]
        for row_idx in 0..num_rows {
            let row_off = row_idx * seq_len;

            for s in 0..seq_len {
                let base = (row_off + s) * head_dim;
                let cache_base = s * half;

                for d in 0..half {
                    let first = qk[base + d];
                    let second = qk[base + d + half];
                    let c = cos[cache_base + d];
                    let sn = sin[cache_base + d];
                    qk[base + d] = first * c - second * sn;
                    qk[base + d + half] = first * sn + second * c;
                }
            }
        }
        Ok(())
    }

    fn split_gate_value(
        &self,
        first: &mut Vec<f32>,
        second: &mut Vec<f32>,
        input: &Vec<f32>,
        rows: usize,
        cols: usize,
    ) -> crate::Result<()> {
        first.resize(rows * cols, 0.0);
        second.resize(rows * cols, 0.0);

        // input: [rows, 2*cols], first = left half, second = right half
        for r in 0..rows {
            let src = r * 2 * cols;
            let dst = r * cols;
            first[dst..dst + cols].copy_from_slice(&input[src..src + cols]);
            second[dst..dst + cols].copy_from_slice(&input[src + cols..src + 2 * cols]);
        }
        Ok(())
    }

    fn gelu(&self, x: &mut Vec<f32>, n: usize) -> crate::Result<()> {
        for v in x.iter_mut().take(n) {
            *v = gelu_scalar(*v);
        }
        Ok(())
    }

    fn swiglu(
        &self,
        value: &Vec<f32>,
        gate: &Vec<f32>,
        output: &mut Vec<f32>,
        n: usize,
    ) -> crate::Result<()> {
        output.resize(n, 0.0);
        for i in 0..n {
            // SwiGLU: value * silu(gate) = value * gate * sigmoid(gate)
            let g = gate[i];
            let silu = g / (1.0 + (-g).exp());
            output[i] = value[i] * silu;
        }
        Ok(())
    }

    fn geglu(
        &self,
        value: &Vec<f32>,
        gate: &Vec<f32>,
        output: &mut Vec<f32>,
        n: usize,
    ) -> crate::Result<()> {
        output.resize(n, 0.0);
        for i in 0..n {
            // GeGLU: gelu(value) * gate
            output[i] = gelu_scalar(value[i]) * gate[i];
        }
        Ok(())
    }

    fn fused_bias_gelu(
        &self,
        x: &mut Vec<f32>,
        bias: &Vec<f32>,
        rows: usize,
        cols: usize,
    ) -> crate::Result<()> {
        for r in 0..rows {
            let base = r * cols;
            for c in 0..cols {
                x[base + c] = gelu_scalar(x[base + c] + bias[c]);
            }
        }
        Ok(())
    }

    fn fused_bias_residual(
        &self,
        output: &mut Vec<f32>,
        input: &Vec<f32>,
        bias: &Vec<f32>,
        residual: &Vec<f32>,
        n: usize,
        cols: usize,
    ) -> crate::Result<()> {
        output.resize(n, 0.0);
        for i in 0..n {
            output[i] = input[i] + bias[i % cols] + residual[i];
        }
        Ok(())
    }

    fn fused_residual_layernorm(
        &self,
        output: &mut Vec<f32>,
        hidden: &Vec<f32>,
        residual: &Vec<f32>,
        weight: &Vec<f32>,
        bias: &Vec<f32>,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> crate::Result<()> {
        output.resize(rows * cols, 0.0);
        for r in 0..rows {
            let base = r * cols;

            // First: add residual
            let mean: f32 = (0..cols)
                .map(|c| hidden[base + c] + residual[base + c])
                .sum::<f32>()
                / cols as f32;

            let var: f32 = (0..cols)
                .map(|c| {
                    let v = hidden[base + c] + residual[base + c] - mean;
                    v * v
                })
                .sum::<f32>()
                / cols as f32;

            let inv_std = 1.0 / (var + eps).sqrt();

            for c in 0..cols {
                let v = hidden[base + c] + residual[base + c];
                output[base + c] = (v - mean) * inv_std * weight[c] + bias[c];
            }
        }
        Ok(())
    }

    fn residual_add(
        &self,
        output: &mut Vec<f32>,
        hidden: &Vec<f32>,
        residual: &Vec<f32>,
        n: usize,
    ) -> crate::Result<()> {
        output.resize(n, 0.0);
        for i in 0..n {
            output[i] = hidden[i] + residual[i];
        }
        Ok(())
    }

    fn add_bias(
        &self,
        x: &mut Vec<f32>,
        bias: &Vec<f32>,
        rows: usize,
        cols: usize,
    ) -> crate::Result<()> {
        for r in 0..rows {
            let base = r * cols;
            for c in 0..cols {
                x[base + c] += bias[c];
            }
        }
        Ok(())
    }

    fn cls_pool(
        &self,
        output: &mut Vec<f32>,
        hidden: &Vec<f32>,
        batch: usize,
        seq: usize,
        hidden_dim: usize,
    ) -> crate::Result<()> {
        output.resize(batch * hidden_dim, 0.0);
        for b in 0..batch {
            let src = b * seq * hidden_dim;
            let dst = b * hidden_dim;
            output[dst..dst + hidden_dim].copy_from_slice(&hidden[src..src + hidden_dim]);
        }
        Ok(())
    }

    fn mean_pool(
        &self,
        output: &mut Vec<f32>,
        hidden: &Vec<f32>,
        mask: &Vec<f32>,
        batch: usize,
        seq: usize,
        hidden_dim: usize,
    ) -> crate::Result<()> {
        output.resize(batch * hidden_dim, 0.0);
        for b in 0..batch {
            // Sum mask values for this batch element
            let mask_sum: f32 = (0..seq).map(|s| mask[b * seq + s]).sum();
            let inv_sum = if mask_sum > 0.0 { 1.0 / mask_sum } else { 0.0 };

            for d in 0..hidden_dim {
                let mut sum = 0.0_f32;
                for s in 0..seq {
                    sum += hidden[(b * seq + s) * hidden_dim + d] * mask[b * seq + s];
                }
                output[b * hidden_dim + d] = sum * inv_sum;
            }
        }
        Ok(())
    }

    fn l2_normalize(&self, data: &mut Vec<f32>, rows: usize, cols: usize) -> crate::Result<()> {
        for r in 0..rows {
            let base = r * cols;
            let row = &data[base..base + cols];
            let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            let inv_norm = if norm > 0.0 { 1.0 / norm } else { 0.0 };
            for c in 0..cols {
                data[base + c] *= inv_norm;
            }
        }
        Ok(())
    }

    fn banded_qk(
        &self,
        q: &Vec<f32>,
        k: &Vec<f32>,
        scores: &mut Vec<f32>,
        batch_heads: usize,
        seq: usize,
        head_dim: usize,
        window: usize,
        stride_qk: usize,
        stride_scores: usize,
    ) -> crate::Result<()> {
        let half_w = window / 2;
        for h in 0..batch_heads {
            for i in 0..seq {
                for w in 0..window {
                    let k_pos = i as isize - half_w as isize + w as isize;
                    if k_pos < 0 || k_pos >= seq as isize {
                        scores[h * stride_scores + i * window + w] = -1e9;
                    } else {
                        let mut dot = 0.0_f32;
                        for d in 0..head_dim {
                            dot += q[h * stride_qk + i * head_dim + d]
                                * k[h * stride_qk + k_pos as usize * head_dim + d];
                        }
                        scores[h * stride_scores + i * window + w] = dot;
                    }
                }
            }
        }
        Ok(())
    }

    fn banded_sv(
        &self,
        scores: &Vec<f32>,
        v: &Vec<f32>,
        output: &mut Vec<f32>,
        batch_heads: usize,
        seq: usize,
        head_dim: usize,
        window: usize,
        stride_scores: usize,
        stride_v: usize,
        stride_out: usize,
    ) -> crate::Result<()> {
        let half_w = window / 2;
        for h in 0..batch_heads {
            for i in 0..seq {
                for d in 0..head_dim {
                    let mut sum = 0.0_f32;
                    for w in 0..window {
                        let v_pos = i as isize - half_w as isize + w as isize;
                        if v_pos >= 0 && v_pos < seq as isize {
                            sum += scores[h * stride_scores + i * window + w]
                                * v[h * stride_v + v_pos as usize * head_dim + d];
                        }
                    }
                    output[h * stride_out + i * head_dim + d] = sum;
                }
            }
        }
        Ok(())
    }

    fn banded_softmax(
        &self,
        scores: &mut Vec<f32>,
        total_rows: usize,
        window: usize,
        scale: f32,
    ) -> crate::Result<()> {
        for r in 0..total_rows {
            let row = &mut scores[r * window..(r + 1) * window];
            for v in row.iter_mut() {
                *v *= scale;
            }
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0_f32;
            for v in row.iter_mut() {
                *v = (*v - max).exp();
                sum += *v;
            }
            let inv = 1.0 / sum.max(1e-12);
            for v in row.iter_mut() {
                *v *= inv;
            }
        }
        Ok(())
    }

    fn to_host(&self, tensor: &Vec<f32>, batch: usize, dim: usize) -> crate::Result<Vec<Vec<f32>>> {
        let mut results = Vec::with_capacity(batch);
        for b in 0..batch {
            results.push(tensor[b * dim..(b + 1) * dim].to_vec());
        }
        Ok(results)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_driver_creates() {
        let _driver = CpuDriver::new().unwrap();
    }

    /// Verify that `CpuDriver` satisfies `Send + Sync` bounds.
    #[test]
    fn driver_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<CpuDriver>();
    }

    /// Basic GELU test: apply GELU to known values and verify output.
    #[test]
    fn gelu_smoke_test() {
        let driver = CpuDriver::new().unwrap();
        let mut tensor = vec![0.0_f32, 1.0, -1.0, 2.0];
        driver.gelu(&mut tensor, 4).unwrap();

        // GELU(0) = 0
        assert!(
            tensor[0].abs() < 1e-4,
            "GELU(0) should be ~0, got {}",
            tensor[0]
        );
        // GELU(1) ~= 0.8412
        assert!(
            (tensor[1] - 0.8412).abs() < 0.01,
            "GELU(1) should be ~0.8412, got {}",
            tensor[1]
        );
        // GELU(-1) ~= -0.1588
        assert!(
            (tensor[2] - (-0.1588)).abs() < 0.01,
            "GELU(-1) should be ~-0.1588, got {}",
            tensor[2]
        );
    }

    /// Basic layer norm test.
    #[test]
    fn layer_norm_smoke_test() {
        let driver = CpuDriver::new().unwrap();
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 rows x 3 cols
        let weight = vec![1.0, 1.0, 1.0];
        let bias = vec![0.0, 0.0, 0.0];
        let mut output = vec![];

        driver
            .layer_norm(&mut output, &input, &weight, &bias, 2, 3, 1e-5)
            .unwrap();

        // Each row should be zero-mean, unit-variance (approx)
        let row0_mean: f32 = output[0..3].iter().sum::<f32>() / 3.0;
        assert!(
            row0_mean.abs() < 1e-5,
            "layer norm row mean should be ~0, got {row0_mean}"
        );
    }

    /// Basic GEMM test.
    #[test]
    fn gemm_smoke_test() {
        let driver = CpuDriver::new().unwrap();
        // A = [[1, 2], [3, 4]] (2x2)
        // B = [[5, 6], [7, 8]] (2x2)
        // A @ B = [[19, 22], [43, 50]]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut output = vec![];

        driver.gemm(&a, &b, &mut output, 2, 2, 2, false).unwrap();

        assert!((output[0] - 19.0).abs() < 1e-4, "got {}", output[0]);
        assert!((output[1] - 22.0).abs() < 1e-4, "got {}", output[1]);
        assert!((output[2] - 43.0).abs() < 1e-4, "got {}", output[2]);
        assert!((output[3] - 50.0).abs() < 1e-4, "got {}", output[3]);
    }

    /// GEMM with transpose_b.
    #[test]
    fn gemm_transpose_b_test() {
        let driver = CpuDriver::new().unwrap();
        // A = [[1, 2], [3, 4]] (2x2)
        // B = [[5, 7], [6, 8]] (2x2, transposed form = [[5,6],[7,8]])
        // A @ B^T = [[19, 22], [43, 50]]
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0]; // [n=2, k=2] when transpose_b=true
        let mut output = vec![];

        driver.gemm(&a, &b, &mut output, 2, 2, 2, true).unwrap();

        assert!((output[0] - 17.0).abs() < 1e-4, "got {}", output[0]);
        assert!((output[1] - 23.0).abs() < 1e-4, "got {}", output[1]);
        assert!((output[2] - 39.0).abs() < 1e-4, "got {}", output[2]);
        assert!((output[3] - 53.0).abs() < 1e-4, "got {}", output[3]);
    }

    /// Embedding lookup test.
    #[test]
    fn embedding_lookup_test() {
        let driver = CpuDriver::new().unwrap();
        // 3 tokens, hidden=2, table has 5 entries
        let table = vec![
            0.1, 0.2, // id 0
            0.3, 0.4, // id 1
            0.5, 0.6, // id 2
            0.7, 0.8, // id 3
            0.9, 1.0, // id 4
        ];
        let ids = vec![2.0, 0.0, 4.0]; // look up ids 2, 0, 4

        let output = driver.embedding_lookup(&ids, &table, 3, 2).unwrap();
        assert_eq!(output, vec![0.5, 0.6, 0.1, 0.2, 0.9, 1.0]);
    }

    /// CLS pooling test.
    #[test]
    fn cls_pool_test() {
        let driver = CpuDriver::new().unwrap();
        // batch=2, seq=3, hidden=2
        let hidden = vec![
            1.0, 2.0, // batch 0, token 0 (CLS)
            3.0, 4.0, // batch 0, token 1
            5.0, 6.0, // batch 0, token 2
            7.0, 8.0, // batch 1, token 0 (CLS)
            9.0, 10.0, // batch 1, token 1
            11.0, 12.0, // batch 1, token 2
        ];
        let mut output = vec![];
        driver.cls_pool(&mut output, &hidden, 2, 3, 2).unwrap();
        assert_eq!(output, vec![1.0, 2.0, 7.0, 8.0]);
    }
}
