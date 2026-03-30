---
name: metal-kernel-development
description: This skill should be used when writing or modifying Metal compute shaders (MSL kernels) for ripvec, implementing GEMM kernels, working with simdgroup matrix operations, optimizing threadgroup memory layouts, or porting llama.cpp kernel patterns. Also use when the user mentions "MSL", "simdgroup", "threadgroup memory", "tiled GEMM", "cooperative load", or "kernel optimization".
---

# Metal Kernel Development for ripvec

Guide to writing and optimizing MSL compute kernels for BERT embedding inference on Apple Silicon.

## Architecture: llama.cpp-Style Tiled GEMM

The target kernel architecture (proven by llama.cpp to match MPS throughput):

```
device half*  activations → cooperative tgmem load → threadgroup half*
device half*  weights     → cooperative tgmem load → threadgroup half*
simdgroup_load(half8x8, tgmem, stride=8) → simdgroup registers
simdgroup_multiply_accumulate(float8x8, half8x8, half8x8, float8x8)
simdgroup_store(float8x8) → tgmem scratch → convert float→half → device half*
```

## 8×8-Block Threadgroup Memory Layout

All tiled GEMM kernels use the llama.cpp block layout:

- Each 8×8 tile occupies 64 contiguous half elements with stride 8
- Blocks indexed by `ib = 8 * K_blk + M_blk` (for A) or `8 * K_blk + N_blk` (for B)
- Address within block: `sa[64 * ib + 8 * row + col]`
- `simdgroup_load(tile, ptr, 8, 0, false)` reads one 8×8 block optimally

## Cooperative Load Pattern (Critical)

128 threads cooperatively load a 64×32 tile from device memory into threadgroup memory. The pattern MUST cover all 64 rows:

```metal
constexpr short NL = 2;          // 128 threads / 64 rows = 2 threads per row
short lr = min(short(tiitg / NL), short(63));  // row index 0..63
short il = short(tiitg % NL);                  // which half of 32 K-elements

for (short i = 0; i < 16; i++) {  // each thread loads 16 elements
    short sx = short(2 * il + i / 8);   // K-block index
    short lx = short(i % 8);            // K position within block
    short ib = short(8 * sx + (lr / 8));
    half val = *(ptr + 16 * il + i);
    *(tgmem + 64 * ib + 8 * (lr % 8) + lx) = val;  // or 8*lx + (lr%8) for transposed
}
```

**NL=4 is a bug.** With 128 threads and NL=4, only 32 rows are loaded (tiitg/4 = 0..31). Simdgroups accessing rows 32-63 read uninitialized data. Always use NL=2 for 64-row tiles.

## Simdgroup Compute Loop

4 simdgroups in 2×2 layout, 16 accumulators each (32×32 output per simdgroup):

```metal
threadgroup half* base_sa = sa + 4 * 64 * (sgitg % 2);  // M half
threadgroup half* base_sb = sb + 4 * 64 * (sgitg / 2);  // N half

for (short ik = 0; ik < TBK/8; ik++) {
    simdgroup_half8x8 ma[4], mb[4];

    simdgroup_barrier(mem_flags::mem_none);
    for (short i = 0; i < 4; i++)
        simdgroup_load(ma[i], base_sa + 64*i, 8, 0, false);

    simdgroup_barrier(mem_flags::mem_none);
    for (short j = 0; j < 4; j++)
        simdgroup_load(mb[j], base_sb + 64*j, 8, 0, false);

    simdgroup_barrier(mem_flags::mem_none);
    for (short i = 0; i < 16; i++)
        simdgroup_multiply_accumulate(mc[i], ma[i/4], mb[i%4], mc[i]);

    base_sa += 8 * 64;  // advance to next K-block set
    base_sb += 8 * 64;
}
```

## Fused Float→Half Store

Since `simdgroup_store(float8x8)` cannot write to `device half*`, use per-tile scratch:

```metal
threadgroup float scratch[4 * 64];  // 4 simdgroups × 64 floats each
threadgroup float* my_scratch = scratch + sgitg * 64;

for (short i = 0; i < 16; i++) {
    simdgroup_store(mc[i], my_scratch, 8, 0, false);
    for (ushort e = lane_id; e < 64; e += 32) {
        uint gm = m_base + e/8, gn = n_base + e%8;
        if (gm < M && gn < N)
            C[gm * N + gn] = half(my_scratch[e]);
    }
}
```

No cross-simdgroup barriers needed — each simdgroup uses its own scratch region.

## INT8 Block Quantization (block_q8_0)

```metal
struct block_q8_0 { half d; int8_t qs[32]; };  // 34 bytes per block
```

Dequantize during cooperative B load:
```metal
uint k_pos = loop_k + 16 * il1 + i;
device block_q8_0* blk = y_row + k_pos / 32;
half val = half(float(blk->qs[k_pos % 32]) * float(blk->d));
```

## Key Invariants

- **Pointer advance**: `x += 32; y += 32;` MUST appear after the cooperative load, before the barrier. Missing this = reading the same K-tile every iteration.
- **Pipeline state types**: All buffer arguments MUST be `device float*` in the signature, cast to actual type inside the kernel. `device half*` triggers a 20× Metal driver regression.
- **Separate Metal library**: Compile native simdgroup kernels in `NATIVE_GEMM_KERNEL`, not `GEMM_KERNEL`, to avoid MSL compiler cross-effects.
- **Tile dimensions**: BM=64, BN=64, BK=32. 4 simdgroups (128 threads). 16 accumulators per simdgroup.

## Additional Resources

- `references/llama-cpp-comparison.md` — Line-by-line comparison of our kernel vs llama.cpp's kernel_mul_mm
