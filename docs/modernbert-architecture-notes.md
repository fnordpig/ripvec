# ModernBERT Architecture Notes

From HuggingFace `modeling_modernbert.py` + `nomic-ai/modernbert-embed-base` config.json.

## Config (modernbert-embed-base)

```
hidden_size: 768
intermediate_size: 1152 (1.5x hidden, NOT 4x)
num_hidden_layers: 22
num_attention_heads: 12
head_dim: 64
global_attn_every_n_layers: 3
local_attention: 128 (sliding window size)
global_rope_theta: 160000.0
local_rope_theta: 10000.0
hidden_activation: "gelu"
mlp_bias: false
attention_bias: false
norm_bias: false
norm_eps: 1e-5
classifier_pooling: "mean"
max_position_embeddings: 8192
vocab_size: 50368
```

## Weight Names (safetensors)

```
embeddings.tok_embeddings.weight  [50368, 768]    — NO position embeddings
embeddings.norm.weight            [768]           — post-embedding LN (no bias)
final_norm.weight                 [768]           — final LN before pooling

layers.{i}.attn.Wqkv.weight      [2304, 768]     — fused QKV
layers.{i}.attn.Wo.weight        [768, 768]      — output projection
layers.{i}.attn_norm.weight       [768]           — pre-attn LN (MISSING for layer 0)
layers.{i}.mlp.Wi.weight         [2304, 768]     — gated MLP input (chunk→1152+1152)
layers.{i}.mlp.Wo.weight         [768, 1152]     — MLP output
layers.{i}.mlp_norm.weight       [768]           — pre-MLP LN
```

NO bias tensors anywhere. Total ~568MB at FP32.

## Architecture Details

### MLP: Gated GELU (NOT plain GELU)

```python
input, gate = self.Wi(hidden_states).chunk(2, dim=-1)  # each [batch, seq, 1152]
return self.Wo(self.act(input) * gate)
```

Wi is [2304, 768], output chunks into [1152] + [1152].
act = GELU on the first half, elementwise multiply with second half.
This is identical to SwiGLU except GELU replaces SiLU.

For our Driver trait: reuse the `swiglu_two_input` pattern but replace silu with gelu.
New kernel: `geglu_kernel(value, gate)` = `gelu(value) * gate`.

### Attention: Alternating Local/Global

Global layers: 0, 3, 6, 9, 12, 15, 18, 21 (every 3rd, starting from 0)
Local layers: 1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20

Global attention: standard full [seq, seq] scores matrix.
Local attention: sliding window mask — token i can only attend to tokens [i-64, i+64].

Implementation: same GEMM for Q@K^T, but apply a DIFFERENT mask:
- Global: the standard padding mask (0 for real, -inf for pad)
- Local: padding mask AND window mask (additional -inf for |i-j| > 64)

The window mask is static per sequence length — precompute once.

For our Driver trait: add `build_sliding_window_mask(seq_len, window_size)` or
include window_size as a parameter to `fused_scale_mask_softmax`.

### RoPE: Two Theta Values

Global layers use theta=160000 (longer wavelength, better for long-range).
Local layers use theta=10000 (standard, fine for short-range within window).

Need two cos/sin caches. The architecture passes the appropriate one per layer.

### Layer 0: Skip Attention Pre-Norm

Layer 0's `attn_norm` is Identity (not LayerNorm). All other layers have it.
The weight tensor `layers.0.attn_norm.weight` does NOT exist in safetensors.

### Final Norm

After all 22 layers, apply `final_norm` (LayerNorm) before pooling.
This is new — NomicBert/ClassicBert don't have it.

### Embedding: Token Only

Just `tok_embeddings.weight` + `norm.weight` (LayerNorm).
No position embeddings (RoPE handles position).
No token_type embeddings (single-sequence model).

## Implementation Priority for Each Backend

### Metal (primary)

1. GeGLU kernel: `geglu_kernel(value, gate, n)` — trivial (GELU * gate)
2. Sliding window mask: modify `build_attn_mask_kernel` to accept window_size
3. Two RoPE caches: extend `MetalRopeCache` with separate global/local
4. Layer 0 identity norm: just skip the LN dispatch
5. Final norm: one extra LN dispatch after encoder loop

### CPU

Same changes but with ndarray. The gated GELU is:
```rust
let (input, gate) = wi_output.split_at(Axis(1), inter_dim);
let activated = input.mapv(gelu) * gate;
```

### MLX

MLX's lazy eval handles this naturally:
```rust
let wi_out = mlx::ops::matmul(&hidden, &wi_weight)?;
let (input, gate) = wi_out.split_equal(2, -1)?;
let activated = mlx::ops::multiply(&mlx::ops::gelu(&input)?, &gate)?;
```

### CUDA

Same kernels as Metal but in CUDA C:
1. `geglu_kernel<<<>>>` — trivial
2. Sliding window mask — modify existing mask kernel
3. Two RoPE caches — extend existing cache
