//! Metal GPU embedding backend for Apple Silicon.
//!
//! Provides low-level Metal device initialization, MSL kernel compilation, and
//! compute dispatch infrastructure. Uses `objc2-metal` Rust bindings to Apple's
//! Metal framework.
//!
//! # Architecture
//!
//! The Metal backend detects the GPU family at init time ([`ChipFamily`]) and
//! stores a pre-compiled compute pipeline alongside the device and command
//! queue. All GPU resources use `StorageModeShared` (unified memory on Apple
//! Silicon) to avoid explicit CPU-GPU copies.
//!
//! Model weights are loaded via zero-copy mmap: the safetensors file is
//! memory-mapped and wrapped as a single Metal buffer using
//! `newBufferWithBytesNoCopy`. Individual tensors are addressed via
//! [`WeightRef`] offsets into this buffer.
//!
//! # Thread safety
//!
//! Metal device and command queue are thread-safe. `MetalBackend` implements
//! `Send + Sync` so it can be shared across the ring-buffer pipeline.

use std::collections::HashMap;
use std::path::Path;

use hf_hub::api::sync::Api;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSString, NSUInteger};
use objc2_metal::{
    MTLBuffer, MTLCommandQueue, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLGPUFamily, MTLLibrary, MTLResourceOptions, MTLSize,
};
use safetensors::SafeTensors;

use super::{DeviceHint, EmbedBackend, Encoding};

// ---------------------------------------------------------------------------
// CoreGraphics linkage (required for MTLCreateSystemDefaultDevice)
// ---------------------------------------------------------------------------

#[expect(unsafe_code, reason = "extern block required for CoreGraphics linkage")]
#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}

// ---------------------------------------------------------------------------
// Device + pipeline helpers
// ---------------------------------------------------------------------------

/// Create the default Metal GPU device.
fn create_device() -> crate::Result<Retained<ProtocolObject<dyn MTLDevice>>> {
    MTLCreateSystemDefaultDevice()
        .ok_or_else(|| crate::Error::Metal("no Metal device available".into()))
}

/// Create a command queue for submitting work.
fn create_queue(
    device: &ProtocolObject<dyn MTLDevice>,
) -> crate::Result<Retained<ProtocolObject<dyn MTLCommandQueue>>> {
    device
        .newCommandQueue()
        .ok_or_else(|| crate::Error::Metal("failed to create command queue".into()))
}

/// Compile MSL source code into a Metal library.
fn compile_library(
    device: &ProtocolObject<dyn MTLDevice>,
    source: &str,
) -> crate::Result<Retained<ProtocolObject<dyn MTLLibrary>>> {
    let ns_source = NSString::from_str(source);
    device
        .newLibraryWithSource_options_error(&ns_source, None)
        .map_err(|e| crate::Error::Metal(format!("MSL compilation failed: {e}")))
}

/// Create a compute pipeline state from a named kernel function.
fn create_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
    name: &str,
) -> crate::Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
    let ns_name = NSString::from_str(name);
    let function = library
        .newFunctionWithName(&ns_name)
        .ok_or_else(|| crate::Error::Metal(format!("function '{name}' not found in library")))?;
    device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|e| crate::Error::Metal(format!("pipeline creation failed: {e}")))
}

// ---------------------------------------------------------------------------
// Kernel pipelines (compiled MSL compute functions)
// ---------------------------------------------------------------------------

/// Pre-compiled Metal compute pipeline states for all BERT inference kernels.
///
/// Created once at model load time by compiling the MSL source from
/// [`super::metal_kernels::KERNELS`] and extracting each named function.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "fields will be used in BERT inference dispatch")
)]
struct KernelPipelines {
    /// Embedding table lookup.
    embedding_lookup: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Add embedding table values to existing output.
    add_embeddings: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Layer normalization with threadgroup reduction.
    layer_norm: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// GELU activation (tanh approximation, in-place).
    gelu: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// `SwiGLU` activation (value * silu(gate)).
    swiglu: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// `RoPE` with pre-computed cos/sin tables.
    rope_cached: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Fused scale + attention-mask + softmax.
    fused_scale_mask_softmax: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Fused residual add + layer norm.
    fused_residual_layernorm: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Fused bias + GELU activation for `ClassicBert` FFN.
    fused_bias_gelu: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Fused bias + residual add for output projections.
    fused_bias_residual: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Unified `SwiGLU` kernel handling both bias and no-bias paths.
    fused_swiglu: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Split QKV `[batch*seq, 3*hidden]` into Q,K,V `[batch*num_heads, seq, head_dim]`.
    qkv_split: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Reshape attention output `[batch*num_heads, seq, head_dim]` to `[batch*seq, hidden]`.
    attn_reshape: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// CLS pooling (extract row 0 per batch element).
    cls_pool: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// L2 normalize each row.
    l2_normalize: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Build float attention mask from int mask.
    build_attn_mask: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    /// Convert FP32 to FP16.
    f32_to_f16: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl KernelPipelines {
    /// Compile all MSL kernels and create pipeline states.
    fn compile(device: &ProtocolObject<dyn MTLDevice>) -> crate::Result<Self> {
        let library = compile_library(device, super::metal_kernels::KERNELS)?;
        let p = |name: &str| create_pipeline(device, &library, name);

        Ok(Self {
            embedding_lookup: p("embedding_lookup_kernel")?,
            add_embeddings: p("add_embeddings_kernel")?,
            layer_norm: p("layer_norm_kernel")?,
            gelu: p("gelu_kernel")?,
            swiglu: p("swiglu_kernel")?,
            rope_cached: p("rope_cached_kernel")?,
            fused_scale_mask_softmax: p("fused_scale_mask_softmax_kernel")?,
            fused_residual_layernorm: p("fused_residual_layernorm_kernel")?,
            fused_bias_gelu: p("fused_bias_gelu_kernel")?,
            fused_bias_residual: p("fused_bias_residual_kernel")?,
            fused_swiglu: p("fused_swiglu_kernel")?,
            qkv_split: p("qkv_split_kernel")?,
            attn_reshape: p("attn_reshape_kernel")?,
            cls_pool: p("cls_pool_kernel")?,
            l2_normalize: p("l2_normalize_kernel")?,
            build_attn_mask: p("build_attn_mask_kernel")?,
            f32_to_f16: p("f32_to_f16_kernel")?,
        })
    }
}

// ---------------------------------------------------------------------------
// Dispatch helpers
// ---------------------------------------------------------------------------

/// Dispatch a 1D compute kernel over `n` elements.
///
/// Automatically chooses a threadgroup size based on the pipeline's maximum.
/// Uses `dispatchThreads:threadsPerThreadgroup:` for non-uniform grids.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "will be used in BERT inference dispatch")
)]
fn dispatch_1d(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    n: usize,
) {
    let max_threads = pipeline.maxTotalThreadsPerThreadgroup();
    let threads_per_tg = max_threads.min(n).max(1);
    let grid = MTLSize {
        width: n,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: threads_per_tg,
        height: 1,
        depth: 1,
    };
    encoder.dispatchThreads_threadsPerThreadgroup(grid, tg);
}

/// Dispatch a per-row kernel using threadgroups (one threadgroup per row).
///
/// Used for kernels like `layer_norm`, `l2_normalize`, `fused_scale_mask_softmax`,
/// and `fused_residual_layernorm` that use threadgroup reductions. Each row is
/// handled by one threadgroup with up to 256 threads.
#[expect(dead_code, reason = "will be used in BERT inference dispatch")]
fn dispatch_rows(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    num_rows: usize,
    threads_per_row: usize,
) {
    let max_threads = pipeline.maxTotalThreadsPerThreadgroup();
    let tpg = max_threads.min(threads_per_row).clamp(1, 256);
    let grid = MTLSize {
        width: num_rows,
        height: 1,
        depth: 1,
    };
    let tg = MTLSize {
        width: tpg,
        height: 1,
        depth: 1,
    };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
}

// ---------------------------------------------------------------------------
// Chip family detection
// ---------------------------------------------------------------------------

/// Apple GPU family classification for tuning compute kernels.
///
/// Different chip families have different optimal GEMM tile sizes and
/// support different hardware features (e.g. async copies).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChipFamily {
    /// M1, M2 (Apple7/8) -- prefer async copy, 48x48 GEMM tiles.
    Apple7_8,
    /// M3+ (Apple9) -- no async copy, 32x32 GEMM tiles.
    Apple9,
}

impl ChipFamily {
    /// Detect the GPU family from the device capabilities.
    fn detect(device: &ProtocolObject<dyn MTLDevice>) -> Self {
        if device.supportsFamily(MTLGPUFamily::Apple9) {
            Self::Apple9
        } else {
            Self::Apple7_8
        }
    }
}

// ---------------------------------------------------------------------------
// Model variant detection + BertConfig (mirrors cpu.rs)
// ---------------------------------------------------------------------------

/// Which BERT variant the loaded weights correspond to.
///
/// `ClassicBert` uses learned position embeddings, GELU activation, and
/// biased QKV projections. `NomicBert` uses `RoPE`, `SwiGLU`, and unbiased
/// projections.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelVariant {
    /// Standard BERT / BGE models (e.g. `BAAI/bge-small-en-v1.5`).
    ClassicBert,
    /// `NomicBert` models (e.g. `nomic-ai/CodeRankEmbed`, `nomic-embed-text-v1.5`).
    NomicBert,
}

/// Detect the model variant by inspecting weight names.
///
/// `ClassicBert` has `embeddings.position_embeddings.weight`; `NomicBert`
/// does not (it uses rotary position encoding instead).
fn detect_variant(tensors: &SafeTensors<'_>) -> ModelVariant {
    if tensors
        .tensor("embeddings.position_embeddings.weight")
        .is_ok()
    {
        ModelVariant::ClassicBert
    } else {
        ModelVariant::NomicBert
    }
}

/// Configuration for a BERT-style encoder model.
///
/// Matches the `config.json` schema from `HuggingFace` model repos.
/// Supports both `ClassicBert` and `NomicBert` config key names.
#[derive(Debug, Clone)]
struct BertConfig {
    /// Which variant this config describes.
    variant: ModelVariant,
    /// Hidden dimension (384 for bge-small, 768 for nomic).
    hidden_size: i32,
    /// Number of transformer layers.
    num_hidden_layers: i32,
    /// Number of attention heads.
    num_attention_heads: i32,
    /// Maximum sequence length (512 for classic, 8192 for nomic).
    max_position_embeddings: i32,
    /// Base for rotary embeddings (only used by `NomicBert`).
    rotary_emb_base: f32,
    /// Layer norm epsilon.
    layer_norm_eps: f32,
}

impl BertConfig {
    /// Parse from a `config.json` value, dispatching on `variant`.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "config values are small ints/floats that fit in i32/f32"
    )]
    fn from_json(v: &serde_json::Value, variant: ModelVariant) -> crate::Result<Self> {
        let get_i32 = |key: &str| -> crate::Result<i32> {
            v.get(key)
                .and_then(serde_json::Value::as_i64)
                .map(|n| n as i32)
                .ok_or_else(|| crate::Error::Other(anyhow::anyhow!("missing config key: {key}")))
        };
        let get_f64 = |key: &str| -> crate::Result<f64> {
            v.get(key)
                .and_then(serde_json::Value::as_f64)
                .ok_or_else(|| crate::Error::Other(anyhow::anyhow!("missing config key: {key}")))
        };

        let layer_norm_eps =
            get_f64("layer_norm_epsilon").or_else(|_| get_f64("layer_norm_eps"))? as f32;

        match variant {
            ModelVariant::ClassicBert => Ok(Self {
                variant,
                hidden_size: get_i32("hidden_size")?,
                num_hidden_layers: get_i32("num_hidden_layers")?,
                num_attention_heads: get_i32("num_attention_heads")?,
                max_position_embeddings: get_i32("max_position_embeddings").unwrap_or(512),
                rotary_emb_base: 10000.0,
                layer_norm_eps,
            }),
            ModelVariant::NomicBert => {
                let hidden_size = get_i32("n_embd").or_else(|_| get_i32("hidden_size"))?;
                let num_hidden_layers =
                    get_i32("n_layer").or_else(|_| get_i32("num_hidden_layers"))?;
                let num_attention_heads =
                    get_i32("n_head").or_else(|_| get_i32("num_attention_heads"))?;
                let max_position_embeddings = get_i32("n_positions")
                    .or_else(|_| get_i32("max_position_embeddings"))
                    .unwrap_or(8192);
                let rotary_emb_base = get_f64("rotary_emb_base")
                    .map(|v| v as f32)
                    .unwrap_or(10000.0);

                Ok(Self {
                    variant,
                    hidden_size,
                    num_hidden_layers,
                    num_attention_heads,
                    max_position_embeddings,
                    rotary_emb_base,
                    layer_norm_eps,
                })
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Weight references (zero-copy offsets into mmap'd safetensors)
// ---------------------------------------------------------------------------

/// Reference to a weight tensor within the mmap'd safetensors buffer.
///
/// Instead of copying tensor data, `WeightRef` stores the byte offset and
/// shape so kernels can index directly into the Metal buffer backed by the
/// mmap'd file.
struct WeightRef {
    /// Byte offset from the start of the mmap into the safetensors file.
    offset: usize,
    /// Tensor shape (e.g. `[384, 384]`).
    shape: Vec<usize>,
    /// Number of f32 elements.
    numel: usize,
}

/// Parse safetensors header and extract byte offsets for all weight tensors.
///
/// Returns a map of tensor name to [`WeightRef`] plus the byte offset where
/// the tensor data region starts (after the JSON header).
fn extract_weight_refs(mmap: &[u8]) -> crate::Result<(HashMap<String, WeightRef>, usize)> {
    let tensors = SafeTensors::deserialize(mmap)
        .map_err(|e| crate::Error::Metal(format!("safetensors parse: {e}")))?;

    // safetensors format: [8-byte header_len][JSON header][tensor data...]
    // data_offsets() returns offsets relative to the DATA region start
    let header_len_u64 = u64::from_le_bytes(
        mmap[..8]
            .try_into()
            .map_err(|_| crate::Error::Metal("safetensors file too short".into()))?,
    );
    let header_len = usize::try_from(header_len_u64)
        .map_err(|_| crate::Error::Metal("safetensors header length overflow".into()))?;
    let data_start = 8 + header_len;

    let mmap_base = mmap.as_ptr() as usize;
    let mut refs = HashMap::new();
    for (name, view) in tensors.tensors() {
        // Compute offset by finding where view.data() sits within the mmap.
        // view.data() returns a slice into the mmap, so pointer arithmetic
        // gives us the absolute byte offset from the start of the file.
        let offset = view.data().as_ptr() as usize - mmap_base;

        let shape: Vec<usize> = view.shape().to_vec();
        let numel: usize = shape.iter().product();
        refs.insert(
            name.clone(),
            WeightRef {
                offset,
                shape,
                numel,
            },
        );
    }
    Ok((refs, data_start))
}

/// Look up a required weight tensor by name.
fn get_weight_ref<'a>(
    refs: &'a HashMap<String, WeightRef>,
    name: &str,
) -> crate::Result<&'a WeightRef> {
    refs.get(name)
        .ok_or_else(|| crate::Error::Metal(format!("missing weight: {name}")))
}

/// Look up an optional weight tensor by name.
fn try_get_weight_ref<'a>(
    refs: &'a HashMap<String, WeightRef>,
    name: &str,
) -> Option<&'a WeightRef> {
    refs.get(name)
}

// ---------------------------------------------------------------------------
// Zero-copy Metal buffer from mmap
// ---------------------------------------------------------------------------

/// Apple Silicon system page size (16KB).
const PAGE_SIZE: usize = 16384;

/// Memory-map a safetensors file and wrap it as a zero-copy Metal buffer.
///
/// The returned `Mmap` must outlive the `MTLBuffer` -- callers store the mmap
/// in a field declared AFTER the buffer to ensure correct drop order.
///
/// # Safety
///
/// Uses unsafe for `Mmap::map` (file must not be modified while mapped) and
/// `newBufferWithBytesNoCopy` (pointer must be page-aligned, which mmap
/// guarantees).
#[expect(
    unsafe_code,
    reason = "mmap + newBufferWithBytesNoCopy require unsafe FFI"
)]
fn load_weights_zero_copy(
    device: &ProtocolObject<dyn MTLDevice>,
    weights_path: &Path,
) -> crate::Result<(memmap2::Mmap, Retained<ProtocolObject<dyn MTLBuffer>>)> {
    let file = std::fs::File::open(weights_path).map_err(|e| crate::Error::Io {
        path: weights_path.display().to_string(),
        source: e,
    })?;

    let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| crate::Error::Io {
        path: weights_path.display().to_string(),
        source: e,
    })?;

    // Round up to page boundary (16KB on Apple Silicon)
    let aligned_len = mmap.len().next_multiple_of(PAGE_SIZE);

    let buffer = unsafe {
        device.newBufferWithBytesNoCopy_length_options_deallocator(
            std::ptr::NonNull::new(mmap.as_ptr() as *mut _)
                .ok_or_else(|| crate::Error::Metal("mmap returned null pointer".into()))?,
            aligned_len as NSUInteger,
            MTLResourceOptions::StorageModeShared,
            None, // Rust owns the mmap lifetime via _mmap field
        )
    }
    .ok_or_else(|| {
        crate::Error::Metal("zero-copy buffer creation failed (pointer not page-aligned?)".into())
    })?;

    Ok((mmap, buffer))
}

// ---------------------------------------------------------------------------
// Metal BERT model structs (WeightRef-based)
// ---------------------------------------------------------------------------

/// Embedding layer weight references.
#[expect(dead_code, reason = "fields used in BERT inference dispatch")]
struct MetalBertEmbeddings {
    /// Word embedding table `[vocab_size, hidden_size]`.
    word_embeddings: WeightRef,
    /// Learned position embeddings (`ClassicBert` only).
    position_embeddings: Option<WeightRef>,
    /// Token type embeddings (segment embeddings).
    token_type_embeddings: Option<WeightRef>,
    /// Layer norm gamma.
    layer_norm_weight: WeightRef,
    /// Layer norm beta.
    layer_norm_bias: WeightRef,
    /// Layer norm epsilon.
    layer_norm_eps: f32,
}

/// Self-attention weight references for one encoder layer.
///
/// For `ClassicBert`, Q/K/V are stored as three separate `WeightRef`s (since
/// we use zero-copy and cannot concatenate in-place). For `NomicBert`, the
/// fused QKV weight is a single tensor in the safetensors file.
#[expect(dead_code, reason = "fields used in BERT inference dispatch")]
struct MetalBertSelfAttention {
    /// Fused QKV weight (`NomicBert`) or Q weight (`ClassicBert`).
    qkv_weight: WeightRef,
    /// K weight (`ClassicBert` only; `None` for `NomicBert`).
    k_weight: Option<WeightRef>,
    /// V weight (`ClassicBert` only; `None` for `NomicBert`).
    v_weight: Option<WeightRef>,
    /// Fused QKV bias (`NomicBert`) or Q bias (`ClassicBert`).
    qkv_bias: Option<WeightRef>,
    /// K bias (`ClassicBert` only).
    k_bias: Option<WeightRef>,
    /// V bias (`ClassicBert` only).
    v_bias: Option<WeightRef>,
    /// Output projection weight.
    output_weight: WeightRef,
    /// Output projection bias.
    output_bias: Option<WeightRef>,
    /// Post-attention layer norm gamma.
    output_ln_weight: WeightRef,
    /// Post-attention layer norm beta.
    output_ln_bias: WeightRef,
    /// Number of attention heads.
    num_heads: i32,
    /// Dimension per head.
    head_dim: i32,
    /// Layer norm epsilon.
    layer_norm_eps: f32,
    /// Rotary embedding base (`NomicBert` only).
    rotary_emb_base: Option<f32>,
}

/// Feed-forward network weight references for one encoder layer.
#[expect(dead_code, reason = "fields used in BERT inference dispatch")]
struct MetalBertFfn {
    /// Intermediate (up) projection weight.
    intermediate_weight: WeightRef,
    /// Intermediate (up) projection bias.
    intermediate_bias: Option<WeightRef>,
    /// For `NomicBert` `SwiGLU`: gate projection weight.
    gate_weight: Option<WeightRef>,
    /// Output (down) projection weight.
    output_weight: WeightRef,
    /// Output (down) projection bias.
    output_bias: Option<WeightRef>,
    /// Post-FFN layer norm gamma.
    output_ln_weight: WeightRef,
    /// Post-FFN layer norm beta.
    output_ln_bias: WeightRef,
    /// Layer norm epsilon.
    layer_norm_eps: f32,
    /// Model variant for activation function selection.
    variant: ModelVariant,
}

/// One transformer encoder layer.
#[expect(dead_code, reason = "fields used in BERT inference dispatch")]
struct MetalBertLayer {
    /// Self-attention sub-layer.
    attention: MetalBertSelfAttention,
    /// Feed-forward sub-layer.
    ffn: MetalBertFfn,
}

/// Complete BERT model as weight references into a Metal buffer.
#[expect(dead_code, reason = "fields used in BERT inference dispatch")]
struct MetalBertModel {
    /// Embedding sub-layer.
    embeddings: MetalBertEmbeddings,
    /// Transformer encoder layers.
    layers: Vec<MetalBertLayer>,
    /// Model configuration.
    config: BertConfig,
}

// ---------------------------------------------------------------------------
// Layer loading (WeightRef extraction)
// ---------------------------------------------------------------------------

/// Extract `ClassicBert` encoder layer weight refs.
///
/// Q/K/V are stored as three separate refs because we cannot concatenate
/// with zero-copy. The compute kernel will dispatch three separate GEMMs
/// (or use offset-based dispatch).
fn load_classic_layer_refs(
    refs: &HashMap<String, WeightRef>,
    i: i32,
    config: &BertConfig,
) -> crate::Result<(MetalBertSelfAttention, MetalBertFfn)> {
    let prefix = format!("encoder.layer.{i}");

    let attention = MetalBertSelfAttention {
        // Store Q/K/V separately (zero-copy cannot concatenate)
        qkv_weight: take_weight_ref(refs, &format!("{prefix}.attention.self.query.weight"))?,
        k_weight: Some(take_weight_ref(
            refs,
            &format!("{prefix}.attention.self.key.weight"),
        )?),
        v_weight: Some(take_weight_ref(
            refs,
            &format!("{prefix}.attention.self.value.weight"),
        )?),
        qkv_bias: try_take_weight_ref(refs, &format!("{prefix}.attention.self.query.bias")),
        k_bias: try_take_weight_ref(refs, &format!("{prefix}.attention.self.key.bias")),
        v_bias: try_take_weight_ref(refs, &format!("{prefix}.attention.self.value.bias")),
        output_weight: take_weight_ref(refs, &format!("{prefix}.attention.output.dense.weight"))?,
        output_bias: try_take_weight_ref(refs, &format!("{prefix}.attention.output.dense.bias")),
        output_ln_weight: take_weight_ref(
            refs,
            &format!("{prefix}.attention.output.LayerNorm.weight"),
        )?,
        output_ln_bias: take_weight_ref(
            refs,
            &format!("{prefix}.attention.output.LayerNorm.bias"),
        )?,
        num_heads: config.num_attention_heads,
        head_dim: config.hidden_size / config.num_attention_heads,
        layer_norm_eps: config.layer_norm_eps,
        rotary_emb_base: None,
    };

    let ffn = MetalBertFfn {
        intermediate_weight: take_weight_ref(refs, &format!("{prefix}.intermediate.dense.weight"))?,
        intermediate_bias: try_take_weight_ref(refs, &format!("{prefix}.intermediate.dense.bias")),
        gate_weight: None,
        output_weight: take_weight_ref(refs, &format!("{prefix}.output.dense.weight"))?,
        output_bias: try_take_weight_ref(refs, &format!("{prefix}.output.dense.bias")),
        output_ln_weight: take_weight_ref(refs, &format!("{prefix}.output.LayerNorm.weight"))?,
        output_ln_bias: take_weight_ref(refs, &format!("{prefix}.output.LayerNorm.bias"))?,
        layer_norm_eps: config.layer_norm_eps,
        variant: config.variant,
    };

    Ok((attention, ffn))
}

/// Extract `NomicBert` encoder layer weight refs.
fn load_nomic_layer_refs(
    refs: &HashMap<String, WeightRef>,
    i: i32,
    config: &BertConfig,
) -> crate::Result<(MetalBertSelfAttention, MetalBertFfn)> {
    let prefix = format!("encoder.layers.{i}");

    let attention = MetalBertSelfAttention {
        qkv_weight: take_weight_ref(refs, &format!("{prefix}.attn.Wqkv.weight"))?,
        k_weight: None,
        v_weight: None,
        qkv_bias: None,
        k_bias: None,
        v_bias: None,
        output_weight: take_weight_ref(refs, &format!("{prefix}.attn.out_proj.weight"))?,
        output_bias: None,
        output_ln_weight: take_weight_ref(refs, &format!("{prefix}.norm1.weight"))?,
        output_ln_bias: take_weight_ref(refs, &format!("{prefix}.norm1.bias"))?,
        num_heads: config.num_attention_heads,
        head_dim: config.hidden_size / config.num_attention_heads,
        layer_norm_eps: config.layer_norm_eps,
        rotary_emb_base: Some(config.rotary_emb_base),
    };

    // SwiGLU: fc11 = value/up, fc12 = gate, fc2 = down
    let ffn = MetalBertFfn {
        intermediate_weight: take_weight_ref(refs, &format!("{prefix}.mlp.fc11.weight"))?,
        intermediate_bias: None,
        gate_weight: Some(take_weight_ref(refs, &format!("{prefix}.mlp.fc12.weight"))?),
        output_weight: take_weight_ref(refs, &format!("{prefix}.mlp.fc2.weight"))?,
        output_bias: None,
        output_ln_weight: take_weight_ref(refs, &format!("{prefix}.norm2.weight"))?,
        output_ln_bias: take_weight_ref(refs, &format!("{prefix}.norm2.bias"))?,
        layer_norm_eps: config.layer_norm_eps,
        variant: config.variant,
    };

    Ok((attention, ffn))
}

/// Take a required weight ref by cloning its data from the map.
fn take_weight_ref(refs: &HashMap<String, WeightRef>, name: &str) -> crate::Result<WeightRef> {
    let r = get_weight_ref(refs, name)?;
    Ok(WeightRef {
        offset: r.offset,
        shape: r.shape.clone(),
        numel: r.numel,
    })
}

/// Take an optional weight ref by cloning its data from the map.
fn try_take_weight_ref(refs: &HashMap<String, WeightRef>, name: &str) -> Option<WeightRef> {
    try_get_weight_ref(refs, name).map(|r| WeightRef {
        offset: r.offset,
        shape: r.shape.clone(),
        numel: r.numel,
    })
}

// ---------------------------------------------------------------------------
// MetalBertModel construction
// ---------------------------------------------------------------------------

impl MetalBertModel {
    /// Build the model from weight refs extracted from a safetensors file.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_possible_wrap,
        reason = "hidden_size and num_layers are small positive ints from config"
    )]
    fn from_weight_refs(
        refs: &HashMap<String, WeightRef>,
        config: &BertConfig,
    ) -> crate::Result<Self> {
        let embeddings = match config.variant {
            ModelVariant::ClassicBert => MetalBertEmbeddings {
                word_embeddings: take_weight_ref(refs, "embeddings.word_embeddings.weight")?,
                position_embeddings: Some(take_weight_ref(
                    refs,
                    "embeddings.position_embeddings.weight",
                )?),
                token_type_embeddings: Some(take_weight_ref(
                    refs,
                    "embeddings.token_type_embeddings.weight",
                )?),
                layer_norm_weight: take_weight_ref(refs, "embeddings.LayerNorm.weight")?,
                layer_norm_bias: take_weight_ref(refs, "embeddings.LayerNorm.bias")?,
                layer_norm_eps: config.layer_norm_eps,
            },
            ModelVariant::NomicBert => MetalBertEmbeddings {
                word_embeddings: take_weight_ref(refs, "embeddings.word_embeddings.weight")?,
                position_embeddings: None,
                token_type_embeddings: try_take_weight_ref(
                    refs,
                    "embeddings.token_type_embeddings.weight",
                ),
                layer_norm_weight: take_weight_ref(refs, "emb_ln.weight")?,
                layer_norm_bias: take_weight_ref(refs, "emb_ln.bias")?,
                layer_norm_eps: config.layer_norm_eps,
            },
        };

        // Validate embedding dimension matches config
        if embeddings.word_embeddings.shape.len() >= 2 {
            let emb_dim = embeddings.word_embeddings.shape[1] as i32;
            if emb_dim != config.hidden_size {
                return Err(crate::Error::Metal(format!(
                    "hidden_size mismatch: config says {} but word_embeddings has dim {}",
                    config.hidden_size, emb_dim
                )));
            }
        }

        let mut layers = Vec::with_capacity(config.num_hidden_layers as usize);
        for i in 0..config.num_hidden_layers {
            let (attention, ffn) = match config.variant {
                ModelVariant::ClassicBert => load_classic_layer_refs(refs, i, config)?,
                ModelVariant::NomicBert => load_nomic_layer_refs(refs, i, config)?,
            };
            layers.push(MetalBertLayer { attention, ffn });
        }

        Ok(Self {
            embeddings,
            layers,
            config: config.clone(),
        })
    }
}

// ---------------------------------------------------------------------------
// MetalBackend
// ---------------------------------------------------------------------------

/// Metal GPU embedding backend.
///
/// Holds the device, command queue, chip family, and zero-copy model weights
/// needed to dispatch BERT inference on Apple Silicon GPUs. Weights are
/// memory-mapped from the safetensors file and wrapped as a single Metal
/// buffer via `newBufferWithBytesNoCopy`.
pub struct MetalBackend {
    /// The Metal GPU device.
    #[expect(
        dead_code,
        reason = "will be used for buffer allocation in BERT inference"
    )]
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    /// Command queue for submitting compute work.
    #[expect(
        dead_code,
        reason = "will be used for compute dispatch in BERT inference"
    )]
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    /// Detected GPU chip family for kernel tuning.
    #[expect(
        dead_code,
        reason = "will be used for kernel selection in BERT inference"
    )]
    chip_family: ChipFamily,
    /// Zero-copy Metal buffer wrapping the mmap'd safetensors data.
    /// Must be declared BEFORE `_mmap` so it is dropped first.
    #[expect(
        dead_code,
        reason = "will be used for weight data access in BERT inference"
    )]
    weight_buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// BERT model with weight refs indexing into `weight_buffer`.
    #[expect(dead_code, reason = "will be used for BERT inference dispatch")]
    model: MetalBertModel,
    /// Pre-compiled compute pipeline states for all MSL kernels.
    #[expect(dead_code, reason = "will be used for BERT inference dispatch")]
    kernels: KernelPipelines,
    /// Hidden dimension for output vector size.
    #[cfg_attr(not(test), expect(dead_code, reason = "will be used in embed_batch"))]
    hidden_size: i32,
    /// Maximum sequence length supported by the model.
    max_position_embeddings: i32,
    /// Memory-mapped safetensors file backing the Metal buffer.
    /// Must be declared AFTER `weight_buffer` to ensure correct drop order
    /// (buffer dropped before mmap).
    _mmap: memmap2::Mmap,
}

// SAFETY: Metal device, command queue, and shared-mode buffers are thread-safe
// (Apple documents MTLDevice and MTLCommandQueue as safe to use from multiple
// threads, and StorageModeShared buffers are readable from any thread).
#[expect(
    unsafe_code,
    reason = "Metal device/queue/shared buffers are documented as thread-safe"
)]
unsafe impl Send for MetalBackend {}
// SAFETY: Same rationale -- Metal's device, queue, and shared buffers are thread-safe.
#[expect(
    unsafe_code,
    reason = "Metal device/queue/shared buffers are documented as thread-safe"
)]
unsafe impl Sync for MetalBackend {}

impl MetalBackend {
    /// Load a BERT embedding model onto Metal via zero-copy mmap.
    ///
    /// Downloads `model.safetensors` and `config.json` via `hf-hub`, then
    /// memory-maps the safetensors file and wraps it as a single Metal buffer
    /// using `newBufferWithBytesNoCopy`. Weight tensors are addressed via
    /// [`WeightRef`] byte offsets.
    ///
    /// # Errors
    ///
    /// Returns an error if no Metal device is available, the model cannot
    /// be downloaded, or weight loading fails.
    pub fn load(model_repo: &str, _device_hint: &DeviceHint) -> crate::Result<Self> {
        let device = create_device()?;
        let queue = create_queue(&device)?;
        let chip_family = ChipFamily::detect(&device);

        // Download model files via hf-hub
        let api = Api::new().map_err(|e| crate::Error::Download(e.to_string()))?;
        let repo = api.model(model_repo.to_string());

        let config_path = repo
            .get("config.json")
            .map_err(|e| crate::Error::Download(e.to_string()))?;
        let weights_path = repo
            .get("model.safetensors")
            .map_err(|e| crate::Error::Download(e.to_string()))?;

        // Parse config.json
        let config_str = std::fs::read_to_string(&config_path).map_err(|e| crate::Error::Io {
            path: config_path.display().to_string(),
            source: e,
        })?;
        let config_json: serde_json::Value = serde_json::from_str(&config_str)
            .map_err(|e| crate::Error::Other(anyhow::anyhow!("config parse error: {e}")))?;

        // Mmap safetensors and create zero-copy Metal buffer
        let (mmap, weight_buffer) = load_weights_zero_copy(&device, &weights_path)?;

        // Extract weight refs from the mmap'd data
        let (refs, _data_start) = extract_weight_refs(&mmap)?;

        // Detect variant from weight names
        let tensors = SafeTensors::deserialize(&mmap)
            .map_err(|e| crate::Error::Metal(format!("safetensors parse: {e}")))?;
        let variant = detect_variant(&tensors);
        drop(tensors); // done with the parsed header

        let config = BertConfig::from_json(&config_json, variant)?;
        let hidden_size = config.hidden_size;
        let max_position_embeddings = config.max_position_embeddings;

        // Build model from weight refs
        let model = MetalBertModel::from_weight_refs(&refs, &config)?;

        // Compile all MSL kernels into pipeline states
        let kernels = KernelPipelines::compile(&device)?;

        tracing::info!(
            device = %device.name(),
            chip_family = ?chip_family,
            hidden_size = hidden_size,
            layers = config.num_hidden_layers,
            variant = ?variant,
            weights_bytes = mmap.len(),
            "Metal backend initialized with zero-copy weights + 17 MSL kernels"
        );

        Ok(Self {
            device,
            queue,
            chip_family,
            weight_buffer,
            model,
            kernels,
            hidden_size,
            max_position_embeddings,
            _mmap: mmap,
        })
    }
}

impl EmbedBackend for MetalBackend {
    fn embed_batch(&self, _encodings: &[Encoding]) -> crate::Result<Vec<Vec<f32>>> {
        Err(crate::Error::Metal(
            "Metal BERT inference not yet implemented".into(),
        ))
    }

    fn supports_clone(&self) -> bool {
        false
    }

    fn clone_backend(&self) -> Box<dyn EmbedBackend> {
        panic!("MetalBackend does not support cloning");
    }

    fn is_gpu(&self) -> bool {
        true
    }

    #[expect(
        clippy::cast_sign_loss,
        reason = "max_position_embeddings is always positive from config"
    )]
    fn max_tokens(&self) -> usize {
        self.max_position_embeddings as usize
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[expect(
    unsafe_code,
    reason = "Metal buffer creation/readback requires unsafe FFI"
)]
mod tests {
    use core::ffi::c_void;
    use core::ptr::NonNull;

    use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLSize};

    use super::*;

    #[test]
    fn metal_device_creation() {
        let device = create_device().unwrap();
        let _queue = create_queue(&device).unwrap();
        let name = device.name().to_string();
        assert!(!name.is_empty(), "device should have a name");
    }

    #[test]
    fn metal_chip_family_detection() {
        let device = create_device().unwrap();
        let family = ChipFamily::detect(&device);
        // On any Apple Silicon Mac, should be one of the two variants.
        assert!(
            family == ChipFamily::Apple7_8 || family == ChipFamily::Apple9,
            "unexpected chip family: {family:?}"
        );
    }

    #[test]
    fn metal_kernel_compilation() {
        let device = create_device().unwrap();
        let library = compile_library(&device, super::super::metal_kernels::TEST_KERNEL).unwrap();
        let pipeline = create_pipeline(&device, &library, "add_one").unwrap();
        assert!(
            pipeline.maxTotalThreadsPerThreadgroup() > 0,
            "pipeline should support at least 1 thread per threadgroup"
        );
    }

    #[test]
    fn metal_compute_dispatch() {
        let device = create_device().unwrap();
        let queue = create_queue(&device).unwrap();
        let library = compile_library(&device, super::super::metal_kernels::TEST_KERNEL).unwrap();
        let pipeline = create_pipeline(&device, &library, "add_one").unwrap();

        // Create a buffer with [1.0, 2.0, 3.0, 4.0]
        let data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let buf_size = (data.len() * core::mem::size_of::<f32>()) as NSUInteger;
        let buffer = unsafe {
            device.newBufferWithBytes_length_options(
                NonNull::new(data.as_ptr() as *mut c_void).unwrap(),
                buf_size,
                MTLResourceOptions::StorageModeShared,
            )
        }
        .expect("failed to create Metal buffer");

        // Encode and dispatch the add_one kernel
        let cmd_buf = queue
            .commandBuffer()
            .expect("failed to create command buffer");
        let encoder = cmd_buf
            .computeCommandEncoder()
            .expect("failed to create compute encoder");
        encoder.setComputePipelineState(&pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&buffer), 0, 0);
        }

        let grid = MTLSize {
            width: 4,
            height: 1,
            depth: 1,
        };
        let threadgroup = MTLSize {
            width: 4,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreads_threadsPerThreadgroup(grid, threadgroup);
        encoder.endEncoding();

        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        // Read back results -- should be [2.0, 3.0, 4.0, 5.0]
        let result =
            unsafe { core::slice::from_raw_parts(buffer.contents().as_ptr() as *const f32, 4) };
        assert_eq!(result, &[2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    #[ignore = "requires model download"]
    fn metal_zero_copy_weight_loading() {
        let backend = MetalBackend::load("BAAI/bge-small-en-v1.5", &DeviceHint::Auto).unwrap();
        assert_eq!(backend.hidden_size, 384);
        assert_eq!(backend.max_tokens(), 512);
        assert!(backend.is_gpu());
        assert!(!backend.supports_clone());
    }

    #[test]
    fn metal_all_kernels_compile() {
        let device = create_device().unwrap();
        let pipelines = KernelPipelines::compile(&device).unwrap();
        // Verify all 17 pipelines have non-zero max threadgroup size
        assert!(pipelines.embedding_lookup.maxTotalThreadsPerThreadgroup() > 0);
        assert!(pipelines.add_embeddings.maxTotalThreadsPerThreadgroup() > 0);
        assert!(pipelines.layer_norm.maxTotalThreadsPerThreadgroup() > 0);
        assert!(pipelines.gelu.maxTotalThreadsPerThreadgroup() > 0);
        assert!(pipelines.swiglu.maxTotalThreadsPerThreadgroup() > 0);
        assert!(pipelines.rope_cached.maxTotalThreadsPerThreadgroup() > 0);
        assert!(
            pipelines
                .fused_scale_mask_softmax
                .maxTotalThreadsPerThreadgroup()
                > 0
        );
        assert!(
            pipelines
                .fused_residual_layernorm
                .maxTotalThreadsPerThreadgroup()
                > 0
        );
        assert!(pipelines.fused_bias_gelu.maxTotalThreadsPerThreadgroup() > 0);
        assert!(
            pipelines
                .fused_bias_residual
                .maxTotalThreadsPerThreadgroup()
                > 0
        );
        assert!(pipelines.fused_swiglu.maxTotalThreadsPerThreadgroup() > 0);
        assert!(pipelines.qkv_split.maxTotalThreadsPerThreadgroup() > 0);
        assert!(pipelines.attn_reshape.maxTotalThreadsPerThreadgroup() > 0);
        assert!(pipelines.cls_pool.maxTotalThreadsPerThreadgroup() > 0);
        assert!(pipelines.l2_normalize.maxTotalThreadsPerThreadgroup() > 0);
        assert!(pipelines.build_attn_mask.maxTotalThreadsPerThreadgroup() > 0);
        assert!(pipelines.f32_to_f16.maxTotalThreadsPerThreadgroup() > 0);
    }

    /// Helper: create a Metal buffer from a slice of f32 values.
    fn make_f32_buffer(
        device: &ProtocolObject<dyn MTLDevice>,
        data: &[f32],
    ) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        let size = core::mem::size_of_val(data) as NSUInteger;
        unsafe {
            device.newBufferWithBytes_length_options(
                NonNull::new(data.as_ptr() as *mut c_void).unwrap(),
                size,
                MTLResourceOptions::StorageModeShared,
            )
        }
        .expect("failed to create Metal buffer")
    }

    /// Helper: create a Metal buffer from a slice of i32 values.
    fn make_i32_buffer(
        device: &ProtocolObject<dyn MTLDevice>,
        data: &[i32],
    ) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        let size = core::mem::size_of_val(data) as NSUInteger;
        unsafe {
            device.newBufferWithBytes_length_options(
                NonNull::new(data.as_ptr() as *mut c_void).unwrap(),
                size,
                MTLResourceOptions::StorageModeShared,
            )
        }
        .expect("failed to create Metal buffer")
    }

    /// Helper: create a zero-initialized Metal buffer of `n` f32 elements.
    fn make_zero_buffer(
        device: &ProtocolObject<dyn MTLDevice>,
        n: usize,
    ) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        let size = (n * core::mem::size_of::<f32>()) as NSUInteger;
        device
            .newBufferWithLength_options(size, MTLResourceOptions::StorageModeShared)
            .expect("failed to create Metal buffer")
    }

    /// Helper: create a Metal buffer containing a single i32 constant.
    fn make_i32_const_buffer(
        device: &ProtocolObject<dyn MTLDevice>,
        val: i32,
    ) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        make_i32_buffer(device, &[val])
    }

    /// Helper: create a Metal buffer containing a single f32 constant.
    fn make_f32_const_buffer(
        device: &ProtocolObject<dyn MTLDevice>,
        val: f32,
    ) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        make_f32_buffer(device, &[val])
    }

    /// Helper: read back f32 values from a Metal buffer.
    fn read_f32_buffer(buffer: &ProtocolObject<dyn MTLBuffer>, n: usize) -> Vec<f32> {
        unsafe { core::slice::from_raw_parts(buffer.contents().as_ptr() as *const f32, n) }.to_vec()
    }

    #[test]
    fn metal_layer_norm_kernel() {
        let device = create_device().unwrap();
        let queue = create_queue(&device).unwrap();
        let pipelines = KernelPipelines::compile(&device).unwrap();

        // Input: [1.0, 2.0, 3.0, 4.0] (1 row, 4 cols)
        // Weight: [1.0, 1.0, 1.0, 1.0], Bias: [0.0, 0.0, 0.0, 0.0], eps: 1e-5
        // Expected: standard layernorm of [1,2,3,4]
        // mean = 2.5, var = 1.25, std = sqrt(1.25) = 1.11803...
        // normalized = [-1.3416, -0.4472, 0.4472, 1.3416]
        let input = [1.0_f32, 2.0, 3.0, 4.0];
        let weight = [1.0_f32, 1.0, 1.0, 1.0];
        let bias = [0.0_f32, 0.0, 0.0, 0.0];

        let output_buf = make_zero_buffer(&device, 4);
        let input_buf = make_f32_buffer(&device, &input);
        let weight_buf = make_f32_buffer(&device, &weight);
        let bias_buf = make_f32_buffer(&device, &bias);
        let rows_buf = make_i32_const_buffer(&device, 1);
        let cols_buf = make_i32_const_buffer(&device, 4);
        let eps_buf = make_f32_const_buffer(&device, 1e-5);

        let cmd_buf = queue.commandBuffer().unwrap();
        let encoder = cmd_buf.computeCommandEncoder().unwrap();
        encoder.setComputePipelineState(&pipelines.layer_norm);

        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&output_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&input_buf), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&weight_buf), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&bias_buf), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&rows_buf), 0, 4);
            encoder.setBuffer_offset_atIndex(Some(&cols_buf), 0, 5);
            encoder.setBuffer_offset_atIndex(Some(&eps_buf), 0, 6);
        }

        // One threadgroup for one row, with 4 threads
        let grid = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: 4,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
        encoder.endEncoding();

        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        let result = read_f32_buffer(&output_buf, 4);
        let expected = [-1.3416, -0.4472, 0.4472, 1.3416];
        for (got, exp) in result.iter().zip(expected.iter()) {
            assert!(
                (got - exp).abs() < 1e-3,
                "layer_norm mismatch: got {got}, expected {exp}"
            );
        }
    }

    #[test]
    fn metal_gelu_kernel() {
        let device = create_device().unwrap();
        let queue = create_queue(&device).unwrap();
        let pipelines = KernelPipelines::compile(&device).unwrap();

        // GELU(0) = 0, GELU(1) ≈ 0.8412, GELU(-1) ≈ -0.1588
        let data = [0.0_f32, 1.0, -1.0, 2.0];
        let buf = make_f32_buffer(&device, &data);
        let n_buf = make_i32_const_buffer(&device, 4);

        let cmd_buf = queue.commandBuffer().unwrap();
        let encoder = cmd_buf.computeCommandEncoder().unwrap();
        encoder.setComputePipelineState(&pipelines.gelu);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&n_buf), 0, 1);
        }
        dispatch_1d(&encoder, &pipelines.gelu, 4);
        encoder.endEncoding();

        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        let result = read_f32_buffer(&buf, 4);
        let expected = [0.0, 0.8412, -0.1588, 1.9545];
        for (got, exp) in result.iter().zip(expected.iter()) {
            assert!(
                (got - exp).abs() < 1e-3,
                "GELU mismatch: got {got}, expected {exp}"
            );
        }
    }

    #[test]
    fn metal_l2_normalize_kernel() {
        let device = create_device().unwrap();
        let queue = create_queue(&device).unwrap();
        let pipelines = KernelPipelines::compile(&device).unwrap();

        // [3.0, 4.0] -> L2 norm = 5.0 -> [0.6, 0.8]
        let data = [3.0_f32, 4.0];
        let buf = make_f32_buffer(&device, &data);
        let rows_buf = make_i32_const_buffer(&device, 1);
        let cols_buf = make_i32_const_buffer(&device, 2);

        let cmd_buf = queue.commandBuffer().unwrap();
        let encoder = cmd_buf.computeCommandEncoder().unwrap();
        encoder.setComputePipelineState(&pipelines.l2_normalize);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&rows_buf), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&cols_buf), 0, 2);
        }
        // dispatch_rows: 1 row, 2 threads per row
        let grid = MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        };
        let tg = MTLSize {
            width: 2,
            height: 1,
            depth: 1,
        };
        encoder.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
        encoder.endEncoding();

        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        let result = read_f32_buffer(&buf, 2);
        assert!(
            (result[0] - 0.6).abs() < 1e-5,
            "expected 0.6, got {}",
            result[0]
        );
        assert!(
            (result[1] - 0.8).abs() < 1e-5,
            "expected 0.8, got {}",
            result[1]
        );
    }

    #[test]
    fn metal_embedding_lookup_kernel() {
        let device = create_device().unwrap();
        let queue = create_queue(&device).unwrap();
        let pipelines = KernelPipelines::compile(&device).unwrap();

        // Embedding table: 3 tokens x 2 dims
        // [[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]]
        let table = [10.0_f32, 11.0, 20.0, 21.0, 30.0, 31.0];
        let indices = [2_i32, 0]; // look up token 2 and token 0
        let hidden_dim = 2;
        let batch_seq = 2;

        let output_buf = make_zero_buffer(&device, 4);
        let table_buf = make_f32_buffer(&device, &table);
        let indices_buf = make_i32_buffer(&device, &indices);
        let batch_seq_buf = make_i32_const_buffer(&device, batch_seq);
        let hidden_dim_buf = make_i32_const_buffer(&device, hidden_dim);

        let cmd_buf = queue.commandBuffer().unwrap();
        let encoder = cmd_buf.computeCommandEncoder().unwrap();
        encoder.setComputePipelineState(&pipelines.embedding_lookup);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&output_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&table_buf), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&indices_buf), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&batch_seq_buf), 0, 3);
            encoder.setBuffer_offset_atIndex(Some(&hidden_dim_buf), 0, 4);
        }
        dispatch_1d(&encoder, &pipelines.embedding_lookup, 4);
        encoder.endEncoding();

        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        let result = read_f32_buffer(&output_buf, 4);
        // Token 2: [30.0, 31.0], Token 0: [10.0, 11.0]
        assert_eq!(result, vec![30.0, 31.0, 10.0, 11.0]);
    }

    #[test]
    fn metal_build_attn_mask_kernel() {
        let device = create_device().unwrap();
        let queue = create_queue(&device).unwrap();
        let pipelines = KernelPipelines::compile(&device).unwrap();

        let mask = [1_i32, 1, 0, 1];
        let output_buf = make_zero_buffer(&device, 4);
        let mask_buf = make_i32_buffer(&device, &mask);
        let total_buf = make_i32_const_buffer(&device, 4);

        let cmd_buf = queue.commandBuffer().unwrap();
        let encoder = cmd_buf.computeCommandEncoder().unwrap();
        encoder.setComputePipelineState(&pipelines.build_attn_mask);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&output_buf), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&mask_buf), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&total_buf), 0, 2);
        }
        dispatch_1d(&encoder, &pipelines.build_attn_mask, 4);
        encoder.endEncoding();

        cmd_buf.commit();
        cmd_buf.waitUntilCompleted();

        let result = read_f32_buffer(&output_buf, 4);
        assert!(
            (result[0] - 0.0_f32).abs() < f32::EPSILON,
            "expected 0.0, got {}",
            result[0]
        );
        assert!(
            (result[1] - 0.0_f32).abs() < f32::EPSILON,
            "expected 0.0, got {}",
            result[1]
        );
        assert!(
            (result[2] - (-1e9_f32)).abs() < 1.0,
            "expected -1e9, got {}",
            result[2]
        );
        assert!(
            (result[3] - 0.0_f32).abs() < f32::EPSILON,
            "expected 0.0, got {}",
            result[3]
        );
    }
}
