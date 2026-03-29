---
name: deepgemm-grouped-gemm
description: DeepGEMM m_grouped FP8 contiguous GEMM — file map, call chain, and guide for implementing a custom SM100 grouped GEMM kernel within the DeepGEMM framework
---

# DeepGEMM Grouped GEMM — File Map & Implementation Guide

> **Goal:** Know exactly which files constitute the CUDA grouped GEMM in DeepGEMM, and how to start implementing your own SM100 grouped GEMM kernel within the same framework.

> **DeepGEMM root:** `/home/xule/DeepGEMM/`

---

## Call Chain: Python → CUDA Kernel

```
bench_m_grouped_fp8_contiguous.py
  │  calls deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(a, b, d, grouped_layout, ...)
  │
  ▼
csrc/python_api.cpp                          ← pybind11 module, exposes function to Python
  │
  ▼
csrc/apis/gemm.hpp                           ← C++ dispatcher: picks SM90 or SM100 path
  │  SM100 → sm100_m_grouped_fp8_fp4_gemm_contiguous_1d1d()
  │
  ▼
csrc/jit_kernels/impls/sm100_fp8_gemm_1d1d.hpp   ← JIT wrapper: builds CUDA code string,
  │                                                   compiles via NVRTC, caches result
  │  #include's the .cuh kernel file
  ▼
deep_gemm/include/deep_gemm/impls/sm100_fp8_gemm_1d1d.cuh   ← THE ACTUAL KERNEL
  │  #include's common utilities:
  ├── deep_gemm/include/deep_gemm/common/sm100_utils.cuh
  ├── deep_gemm/include/deep_gemm/common/tma_utils.cuh
  ├── deep_gemm/include/deep_gemm/common/epilogue_utils.cuh
  ├── deep_gemm/include/deep_gemm/common/scheduler.cuh
  ├── deep_gemm/include/deep_gemm/common/utils.cuh
  ├── deep_gemm/include/deep_gemm/common/reduction.cuh
  └── deep_gemm/include/deep_gemm/common/cute_tie.cuh
```

---

## File Map

### Layer 1: The Kernel (.cuh) — what you write

| File | Role |
|------|------|
| `deep_gemm/include/deep_gemm/impls/sm100_fp8_gemm_1d1d.cuh` | **The SM100 kernel.** TMA loads, MMA, epilogue, tile scheduling — all in one file. This is where grouped GEMM logic lives. |

### Layer 2: Common utilities (.cuh) — what you reuse

| File | Role |
|------|------|
| `deep_gemm/include/deep_gemm/common/sm100_utils.cuh` | SM100-specific helpers (warp group ops, TMEM, etc.) |
| `deep_gemm/include/deep_gemm/common/tma_utils.cuh` | TMA descriptor creation, on-device descriptor updates |
| `deep_gemm/include/deep_gemm/common/epilogue_utils.cuh` | Output writeback, scaling, type conversion |
| `deep_gemm/include/deep_gemm/common/scheduler.cuh` | Persistent kernel tile scheduler, group-to-tile mapping |
| `deep_gemm/include/deep_gemm/common/utils.cuh` | General CUDA utilities (barriers, sync, math) |
| `deep_gemm/include/deep_gemm/common/reduction.cuh` | Reduction operations |
| `deep_gemm/include/deep_gemm/common/cute_tie.cuh` | CuTe library integration helpers |

### Layer 3: JIT compilation (.hpp) — how the kernel gets compiled

| File | Role |
|------|------|
| `csrc/jit_kernels/impls/sm100_fp8_gemm_1d1d.hpp` | JIT wrapper: generates CUDA source string including the .cuh kernel, compiles via NVRTC at runtime, caches the binary. Also sets tile sizes, stages, thread counts based on heuristics. |
| `csrc/jit_kernels/heuristics/sm100.hpp` | Tile size / stage count / thread count selection for SM100 |
| `csrc/jit_kernels/heuristics/common.hpp` | Common heuristic utilities |
| `csrc/jit_kernels/impls/epilogue.hpp` | Epilogue code generation |
| `csrc/jit_kernels/impls/runtime_utils.hpp` | Runtime utilities for kernel launch |

### Layer 4: API & binding (.hpp / .cpp) — how Python calls it

| File | Role |
|------|------|
| `csrc/apis/gemm.hpp` | C++ API: `m_grouped_fp8_fp4_gemm_nt_contiguous()`, dispatches to SM90/SM100 |
| `csrc/python_api.cpp` | pybind11 module definition, exposes all functions to Python |

### Layer 5: JIT infrastructure — you don't touch these

| File | Role |
|------|------|
| `csrc/jit/compiler.hpp` | JIT compiler interface (wraps NVRTC) |
| `csrc/jit/device_runtime.hpp` | Device runtime management |
| `csrc/jit/kernel_runtime.hpp` | Kernel launch runtime |
| `csrc/jit/cache.hpp` | Compilation cache (avoids recompilation) |

### Other files

| File | Role |
|------|------|
| `csrc/indexing/main.cu` | The only standalone .cu file — indexing operations |
| `csrc/utils/*.hpp` | General C++ utilities (math, format, hash, exceptions, etc.) |

---

## How Many Files Do You Need?

To implement a custom SM100 grouped GEMM within DeepGEMM:

| What | Files | You write / modify |
|------|-------|--------------------|
| **Kernel** | 1 `.cuh` | Write from scratch or fork `sm100_fp8_gemm_1d1d.cuh` |
| **Common utilities** | 7 `.cuh` | Reuse as-is (TMA, scheduler, epilogue, etc.) |
| **JIT wrapper** | 1 `.hpp` | Fork `sm100_fp8_gemm_1d1d.hpp`, change kernel name / template params |
| **Heuristics** | 1 `.hpp` | Add your tile size / stage count configs to `sm100.hpp` |
| **API dispatcher** | 1 `.hpp` | Add your new function to `gemm.hpp` |
| **Python binding** | 1 `.cpp` | Add your new function to `python_api.cpp` |
| **Total** | ~12 files | **2 new files** (kernel + JIT wrapper), **3 files to modify** (heuristics, API, binding) |

---

## How to Start: Step-by-Step

### Step 1: Fork the kernel

Copy `deep_gemm/include/deep_gemm/impls/sm100_fp8_gemm_1d1d.cuh` to your new file (e.g., `sm100_my_grouped_gemm.cuh`). This file contains:
- TMA descriptor setup
- Persistent kernel main loop (iterates over tile assignments)
- Group switching logic (TMA descriptor updates when `did_batch_change`)
- MMA computation
- Epilogue (scaling + writeback)

### Step 2: Fork the JIT wrapper

Copy `csrc/jit_kernels/impls/sm100_fp8_gemm_1d1d.hpp` to your new file (e.g., `sm100_my_grouped_gemm.hpp`). Modify:
- The `#include` to point to your new .cuh kernel
- Template parameter generation (tile sizes, data types, etc.)
- The kernel launch function signature

### Step 3: Add heuristics

In `csrc/jit_kernels/heuristics/sm100.hpp`, add tile size / stage count / thread count selection for your new kernel. The existing heuristics select configs based on problem dimensions (M, N, K).

### Step 4: Wire up the API

In `csrc/apis/gemm.hpp`, add a new function (e.g., `my_grouped_gemm(...)`) that validates inputs and calls your JIT wrapper.

### Step 5: Expose to Python

In `csrc/python_api.cpp`, add a pybind11 binding for your new function.

### Step 6: Test

Write a test script similar to `tests/bench_m_grouped_fp8_contiguous.py` that calls your new function.

---

## SM90 Kernel Analysis: `sm90_fp8_gemm_1d2d.cuh`

> **Source:** `/home/xule/DeepGEMM/deep_gemm/include/deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh`
> This is the SM90 (Hopper) FP8 grouped GEMM kernel. "1D2D" refers to scale factor layout: 1D for A (per-token, one SF per row per K-group) and 2D for B (per-block, one SF per 128×128 block).

### Pipeline Schedule

The kernel uses a **warp-specialized producer-consumer pipeline** with `kNumStages` software pipeline stages:

**Thread organization:**
- **TMA warp-group** (`kNumTMAThreads`, typically 128 threads = 4 warps): loads data via TMA. Only **one thread** (warp 2, lane 0) actually issues TMA instructions — the rest are idle.
- **Math warp-groups** (`kNumMathThreads`, typically 128 or 256 threads): execute WGMMA and scale factor promotion.
- Total threads per CTA = `kNumTMAThreads + kNumMathThreads`.
- Register reconfig: TMA warps deallocate to `kNumTMARegisters=40`, math warps allocate up to `kNumMathRegisters=248`.

**Pipeline flow (persistent kernel):**

```
TMA producer thread (1 thread):                Math consumer warp-groups:
┌─────────────────────────────────┐            ┌─────────────────────────────────┐
│ while (scheduler.get_next_block)│            │ while (scheduler.get_next_block)│
│   for each k_block:             │            │   Load B scale factors (SFB)    │
│     wait(empty_barrier[stage])  │            │     from GMEM → SMEM via __ldg  │
│     TMA_copy(A → smem_a[stage]) │            │   NamedBarrier::sync            │
│     TMA_copy(SFA → smem_sfa[s]) │            │   for each k_block:             │
│     TMA_copy(B → smem_b[stage]) │            │     wait(full_barrier[stage])   │
│     full_barrier.arrive_and_    │            │     read SFA from SMEM          │
│       expect_tx(bytes)          │            │     read SFB from SMEM          │
│   endfor                        │            │     WGMMA(smem_a, smem_b →accum)│
│ endwhile                        │            │     promote: accum *= sfa * sfb │
└─────────────────────────────────┘            │     empty_barrier.arrive()      │
                                               │   endfor                        │
                                               │   STSM(accum → smem_d)          │
                                               │   TMA_store(smem_d → GMEM D)    │
                                               └─────────────────────────────────┘
```

**Barrier protocol:**
- `full_barriers[stage]` (ClusterTransactionBarrier): TMA producer signals arrival with expected bytes; math consumer waits.
- `empty_barriers[stage]` (ClusterTransactionBarrier): math consumer signals done; TMA producer waits before reusing the stage.
- Phase flipping: `phase ^= (stage_idx == 0)` — alternates between 0/1 each full pipeline cycle.

**Pipeline depth:** `kNumStages` stages (typically 2–4), with a possible `kNumLastStages` for the tail.

### Input Layouts & Memory Flow

**Data types:** A=FP8(e4m3), B=FP8(e4m3), D=BF16, Accumulator=FP32, SFA=FP32, SFB=FP32.

**Global memory layout:**
- **A** (activation): shape `(M, K)`, K-major (row-major). Passed as TMA descriptor `tensor_map_a`.
- **B** (weight): shape `(N, K)` per group (or `(num_groups, N, K)` for batched), K-major. Passed as TMA descriptor `tensor_map_b`.
- **SFA** (A scale factors): shape `(M, K/128)`, one FP32 scale per 128-element K-group per row. Passed as TMA descriptor `tensor_map_sfa`.
- **SFB** (B scale factors): shape `(N/128, K/128)`, one FP32 scale per 128×128 block. Layout controlled by `kMajorSFB`:
  - **K-major** (`Major::K`): K is the contiguous (fast) dim, `stride(-1)==1`. Memory layout: `sfb[n_block * (K/128) + k_block]`. **This is the default** — `per_block_cast_to_fp8()` returns `sf.view(N/128, K/128)` row-contiguous, and `get_major_type_ab()` returns `Major::K` when `stride(-1)==1`. All standard DeepGEMM tests use this layout.
  - **MN-major** (`Major::MN`): N is the contiguous dim, `stride(-2)==1`. Memory layout: `sfb[k_block * (N/128) + n_block]`. Supported but not used in default tests.
  - SM90 SFB validation (`layout.hpp:110-114`) accepts both: `(stride(-1)==1 and stride(-2)==size(-1))` or `(stride(-2)==1 and stride(-1)==size(-2))`.
  - Passed as raw `float*` pointer `sfb` (NOT via TMA — loaded by math threads via `__ldg` + `st_shared`).
- **D** (output): shape `(M, N)`, BF16. Written via TMA store `tensor_map_d`.

**GMEM → SMEM (via TMA):**
- `smem_a[stage]`: `BLOCK_M × BLOCK_K` FP8 bytes, swizzle mode `kSwizzleAMode` (typically 128B).
- `smem_b[stage]`: `BLOCK_N × BLOCK_K` FP8 bytes, swizzle mode `kSwizzleBMode`.
- `smem_sfa[stage]`: `BLOCK_M` FP32 values (one scale per row of the tile), aligned to 128 bytes.
- `smem_sfb`: Loaded once per M-N tile (NOT per K-block stage) by math threads via `__ldg` + `st_shared`. Contains `shape_k_scales` (or `2 × shape_k_scales` if B scale straddles two K-groups within a BLOCK_N).

**SMEM → Registers (for WGMMA):**
- WGMMA reads A and B directly from SMEM via **SMEM descriptors** (GmmaDescriptor). No explicit `ldmatrix`; WGMMA SS mode reads SMEM implicitly.
- A descriptor: base pointer `smem_a[stage] + math_wg_idx * WGMMA::M * BLOCK_K`, stride = K-major swizzle.
- B descriptor: base pointer `smem_b[stage]`, stride = K-major swizzle.
- SFA: read from SMEM via `ld_shared(smem_sfa[stage] + row_idx)` → FP32 register.
- SFB: read from SMEM via `ld_shared(smem_sfb + k_block_idx)` → FP32 register.

**Registers → SMEM → GMEM (epilogue):**
- `final_accum[]` (FP32 registers) → convert to BF16 → `stmatrix.sync.aligned.x2.m8n8.shared.b16` → `smem_d`.
- `smem_d` → `SM90_TMA_STORE_2D::copy` or `SM90_TMA_STORE_3D::copy` → GMEM D.

### Quantization Sizes

| Operand | Quant granularity | Scale factor shape | Layout | Description |
|---------|------------------|--------------------|--------|-------------|
| **A** (activation) | **1 × 128** (per-token, per-128-K-group) | `(M, K/128)` FP32 | K-major (K/128 is contiguous dim) | One SF per row per 128-element K-chunk. "1D" — scales vary along M and K. |
| **B** (weight) | **128 × 128** (per-block) | `(N/128, K/128)` FP32 | **K-major by default** (K/128 is contiguous dim) | One SF per 128×128 block. "2D" — scales form a grid. Hence "1D2D" in the kernel name. |

The assertion `BLOCK_K == 128` enforces that each K-tile aligns exactly with one scale factor group.

**SFB layout proof from source:**
- `per_block_cast_to_fp8()` (`math.py:51`) returns `sf.view(N/128, K/128)` — a standard row-contiguous tensor with `stride(-1)==1`, meaning K is the fast (contiguous) dimension.
- `get_major_type_ab()` (`layout.hpp:21-23`) returns `Major::K` when `stride(-1)==1`.
- `cast_fp8_fp4_with_major()` (`generators.py:229`) never transposes the SF tensor — even when B data is transposed to MN-major, SFB stays K-major.
- SM90 FP8 m-grouped tests (`generators.py:164`) only test `major_b=KMajor` on SM90, so SFB is always K-major in practice.
- The kernel template parameter `kMajorSFB` supports both `Major::K` and `Major::MN`, but no standard test exercises the MN-major path.

### Dequantization (Scale Factor Promotion)

**Yes, the kernel performs dequantization.** It is done as a **post-MMA scale factor promotion**, not as an explicit type conversion before MMA.

The FP8 tensor core (WGMMA) computes `accum = A_fp8 × B_fp8` in FP32 without any scaling — raw FP8 integer values are multiplied. After each WGMMA, the result is **promoted** (rescaled) by multiplying with the appropriate scale factors:

```cpp
// Lines 317-333: Scale factor promotion
float scale_0_0 = scale_a_0 * scale_b_0;  // sfa[row0] * sfb[k_block]
float scale_1_0 = scale_a_1 * scale_b_0;  // sfa[row1] * sfb[k_block]

for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++i) {
    // Choose scale_b_0 or scale_b_1 based on which K-group this column belongs to
    const bool& predicate = kMustUseUniformedScaleB or i < num_former_iters;
    final_accum[i*4+0] += (predicate ? scale_0_0 : scale_0_1) * accum[i*4+0];
    final_accum[i*4+1] += (predicate ? scale_0_0 : scale_0_1) * accum[i*4+1];
    final_accum[i*4+2] += (predicate ? scale_1_0 : scale_1_1) * accum[i*4+2];
    final_accum[i*4+3] += (predicate ? scale_1_0 : scale_1_1) * accum[i*4+3];
}
```

**How it works:**
1. WGMMA outputs raw FP32 accumulator values (FP8 × FP8 without scaling).
2. Each WGMMA covers a `64 × BLOCK_N × 32` tile (M=64 rows from the warp group, K=32 per WGMMA instruction).
3. After WGMMA, each thread holds `kNumAccum = 64*N/128` FP32 values.
4. For each 4-element group in the accumulator: multiply by `sfa[row] * sfb[k_block]`.
5. The `num_former_iters` logic handles the case where BLOCK_N straddles two B scale factor groups (when `BLOCK_N > BLOCK_K=128` or `BLOCK_N` is not a multiple of `BLOCK_K`).
6. Results accumulate across K-blocks into `final_accum[]`.

**There is no explicit FP8→FP16/FP32 conversion instruction.** The tensor core handles FP8 multiplication natively and produces FP32 output. The dequantization is purely the FP32 multiply by scale factors.

### Explicit PTX Instructions Used

| PTX Instruction | Location | Purpose |
|----------------|----------|---------|
| `wgmma.fence.sync.aligned` | `warpgroup_arrive()` | Fence before issuing WGMMA |
| `wgmma.commit_group.sync.aligned` | `warpgroup_commit_batch()` | Commit WGMMA group |
| `wgmma.wait_group.sync.aligned N` | `warpgroup_wait<N>()` | Wait for WGMMA group N to complete |
| `wgmma.mma_async.sync.aligned.*` | via `WGMMA::fma()` (CuTe) | FP8 WGMMA: `MMA_64xNx32_F32E4M3E4M3_SS_TN` |
| `stmatrix.sync.aligned.x2.m8n8.shared.b16` | `SM90_U32x2_STSM_N::copy()` | Store accumulators from registers to SMEM |
| `ld.shared.f32` / `.u32` / `.v2.f32` / `.v4.f32` | `ld_shared()` overloads | Read scale factors and data from SMEM |
| `st.shared.f32` / `.u32` / `.v2.u32` | `st_shared()` overloads | Write SFB to SMEM |
| `cp.async.bulk.tensor` | via `tma_copy()` (CuTe) | TMA load: GMEM → SMEM |
| `cp.async.bulk.tensor.store` | via `SM90_TMA_STORE_2D/3D::copy()` | TMA store: SMEM → GMEM |
| `cp.async.bulk.commit_group` | via `tma_store_arrive()` | Commit TMA store group |
| `cp.async.bulk.wait_group` | via `tma_store_wait<0>()` | Wait for TMA stores |
| `fence.proxy.async` | via `tma_store_fence()` | Fence before TMA store |
| `mov.u32 %0, %%smid` | `get_sm_idx()` | Read SM index |
| `mov.u32 %0, %laneid` | `get_lane_idx()` | Read lane index |
| `prefetch.global.L1` | `prefetch_l1()` | L1 prefetch |
| `prefetch.tensormap` | `cute::prefetch_tma_descriptor()` | Prefetch TMA descriptor |
| `cutlass::arch::warpgroup_reg_alloc<N>` / `reg_dealloc<N>` | Register reconfig | Dynamic register allocation per warp group |
| `cutlass::arch::fence_barrier_init` | Barrier init | Make barrier visible in async proxy |
| `cutlass::arch::ClusterTransactionBarrier::init/wait/arrive` | Pipeline barriers | Producer-consumer synchronization |

### Grouped GEMM Specifics

The kernel supports multiple `GemmType` modes via template parameter, all sharing the same pipeline:

| GemmType | How groups are handled |
|----------|----------------------|
| `Normal` | No grouping — single GEMM |
| `MGroupedContiguous` | `grouped_layout[m_row]` = group index for each row. B indexed by group via `get_global_idx`. |
| `MGroupedMasked` | `grouped_layout[group]` = number of valid M rows. Scheduler skips invalid rows. |
| `MGroupedContiguousWithPsumLayout` | `grouped_layout[group]` = prefix-sum of M rows. Scheduler computes per-group M-block ranges. |
| `Batched` | A and B are 3D tensors. `current_group_idx` selects the batch via TMA 3D copy. |
| `KGroupedContiguous` | Groups along K dimension. Scheduler tracks `current_k_cumsum`. |

The scheduler (`Scheduler` struct in `scheduler.cuh`) is a persistent-kernel tile scheduler. `get_next_block()` assigns the next (m_block, n_block) to each CTA, with L2-friendly swizzling (`get_swizzled_block_idx`). For grouped GEMM, it tracks group boundaries and adjusts global memory offsets accordingly.

---

## Key Design Decisions in DeepGEMM

- **JIT compilation, not ahead-of-time.** Kernels are compiled at first call via NVRTC, then cached. This allows tile sizes and other parameters to be baked in as compile-time constants, which is more efficient than passing them as kernel arguments.
- **One kernel file per architecture variant.** `sm100_fp8_gemm_1d1d.cuh` handles both regular and m_grouped contiguous GEMM — the grouped logic is controlled by template parameters and `if constexpr`.
- **Common utilities are header-only.** All `.cuh` files in `common/` are `#include`-able — no separate compilation units.
- **The kernel is self-contained.** `sm100_fp8_gemm_1d1d.cuh` + the 7 common headers is everything the GPU executes. No CUTLASS dependency at the kernel level (CuTe is used for TMA descriptor creation, but the kernel itself uses raw PTX/inline ASM).

---

## Testing & Correctness Verification

### How DeepGEMM verifies FP8/MXFP8 GEMM correctness

DeepGEMM uses a **BF16 reference → quantize → run kernel → compare** approach. It does NOT compare against cuBLAS FP8 output. Instead, it computes the mathematically exact answer in higher precision and checks that the FP8 kernel result is close enough.

**The flow (from `tests/generators.py` and `tests/test_fp8_fp4.py`):**

1. **Generate BF16 inputs:** Create random `a` (BF16) and `b` (BF16) on GPU.
2. **Compute reference in FP32:** `ref_d = (a.float() @ b.float().t()).to(out_dtype)` — matmul in FP32, then cast to output dtype. This is the ground truth.
3. **Quantize to FP8/FP4:** Cast `a` and `b` to FP8 (e4m3) with scale factors using `per_token_cast_to_fp8()`, `per_block_cast_to_fp8()`, or `per_token_cast_to_fp4()`. For MXFP8 (UE8M0 scale factors), pass `use_ue8m0=True`.
4. **Run kernel:** Call the DeepGEMM kernel with the quantized inputs.
5. **Compare:** Use `calc_diff(d, ref_d)` and assert below threshold.

### The similarity metric: `calc_diff()`

**Source:** `deep_gemm/testing/numeric.py`

```python
def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    if denominator == 0:
        return 0.0
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim
```

This is `1 - cosine_similarity` (specifically, `1 - 2*dot(x,y)/(||x||² + ||y||²)`). A value of 0 means perfect match. The metric is scale-invariant and measures directional agreement between the two output tensors.

### Thresholds by quantization type

**Source:** `QuantConfig.max_diff()` in `tests/generators.py`

| Quantization | Threshold | Meaning |
|-------------|-----------|---------|
| FP8 × FP8 (standard) | `< 0.001` | Very tight — FP8 is close to BF16 reference |
| FP8 × FP4 or FP4 × FP8 (mixed) | `< 0.01` | Looser due to FP4 precision loss |
| FP4 × FP4 | `< 0.02` | Loosest — both operands lose precision |
| K-grouped FP8 | `< 0.001` | Same as standard FP8 |

### Quantization configs tested

**Source:** `QuantConfig` class in `tests/generators.py`

| Config | `gran_k_a` | `gran_k_b` | `is_fp4_a` | `is_fp4_b` | Description |
|--------|-----------|-----------|-----------|-----------|-------------|
| Legacy (default) | 128 | 128 | False | False | Standard per-token FP8 with group size 128 along K |
| SM100 mixed | 128 | 32 | False | True | A=FP8 (gran_k=128), B=FP4 (gran_k=32) |

### What `generate_m_grouped_contiguous()` does

**Source:** `tests/generators.py`, lines 281–320

1. Creates `num_groups` sub-problems, each with `expected_m_per_group` rows (aligned to `get_mk_alignment_for_contiguous_layout()`).
2. `a` shape: `(m_total, k)` — all groups' A rows concatenated contiguously.
3. `b` shape: `(num_groups, n, k)` — one weight matrix per group.
4. `grouped_layout`: either a per-row group index (`shape=(m_total,)`, each entry = group id or -1 for padding) or a prefix-sum layout (`shape=(num_groups,)`, each entry = cumulative row end).
5. Reference: `ref_d[start:end] = a[start:end] @ b[i].t()` — BF16 matmul per group.
6. Padding rows (between `actual_m` and `aligned_m`) are zeroed in A and marked as `-1` in `grouped_layout`.

### Test variants exercised

**Source:** `test_fp8_fp4.py`

| Test function | What it tests |
|--------------|---------------|
| `test_gemm()` | Regular (non-grouped) FP8 GEMM: forward (m=1,128,4096), backward (dgrad+wgrad), multiple N/K combos, accumulate mode, alias layouts |
| `test_m_grouped_gemm_contiguous()` | m-grouped contiguous: (4 groups × 8192 rows, 8 groups × 4096 rows) × multiple N/K × psum layout on/off × layout aliases |
| `test_m_grouped_gemm_masked()` | m-grouped masked: random per-group M sizes, checks each group slice individually, tests both masked and psum-layout paths |
| `test_k_grouped_gemm_contiguous()` | k-grouped contiguous: groups along K dimension, tests empty groups (K=0), EP16/32/64 configs |

### Performance benchmarking

Two methods are used:

1. **Kineto profiler** (`bench_kineto()` in `deep_gemm/testing/bench.py`): Uses `torch.profiler` to measure actual GPU kernel time, filtering by kernel name. Flushes L2 cache (8 GB memset) between iterations. Reports average kernel time.

2. **CUPTI-based** (`bench_gpu_time()` in `tests/cupti_perf.py`): Used by `bench_m_grouped_fp8_contiguous.py`. Reports median/std of GPU times across iterations.

Both report TFLOPS and GB/s. The Kineto path also compares against cuBLASLt for non-FP4 configs and reports speedup ratio.

### How to write your own correctness test

Follow the pattern in `test_m_grouped_gemm_contiguous()`:

```python
# 1. Generate data
m, a, b, grouped_layout, d, ref_d = generate_m_grouped_contiguous(
    num_groups, expected_m_per_group, n, k, major_a, major_b,
    use_ue8m0=use_ue8m0, quant_config=quant_config)

# 2. Run your kernel
deep_gemm.your_kernel(a, b, d, grouped_layout, ...)

# 3. Check correctness
diff = calc_diff(d, ref_d)
assert diff < quant_config.max_diff(), f'{diff=}'
```

The reference (`ref_d`) is computed from BF16 inputs before quantization, so it represents the "ideal" answer. The diff threshold accounts for quantization error inherent in the data format.

### Quantization functions used in testing

**Source:** `deep_gemm/utils/math.py`

These are **host-side Python functions** used to prepare FP8 test inputs. They are NOT part of the kernel — they simulate the quantization that a real inference pipeline would do before calling the GEMM.

**`per_token_cast_to_fp8(x, use_ue8m0, gran_k=128)`** — Row-wise (per-token) quantization

Given a 2D tensor `x` of shape `(M, K)`:
1. Pad K to a multiple of `gran_k` (default 128).
2. Reshape to `(M, K/gran_k, gran_k)` — each row is split into groups of `gran_k` elements.
3. Compute one scale factor per group: `sf = amax(abs(group)) / 448.0` (448 = max representable value in FP8 e4m3). Shape of `sf`: `(M, K/gran_k)`.
4. If `use_ue8m0=True` (MXFP8): round `sf` up to the nearest power of 2 via `ceil_to_ue8m0()` — this gives a UE8M0 (8-bit unsigned exponent, no mantissa) scale factor.
5. Scale and cast: `(x / sf).to(float8_e4m3fn)`.
6. Return `(x_fp8, sf)`.

Each row of A gets its own set of scale factors (one per `gran_k` chunk along K). This is "1D" scaling — scales vary along M but are shared within each K-group.

**`per_block_cast_to_fp8(x, use_ue8m0, gran_k=128)`** — Block-wise (2D) quantization

Given a 2D tensor `x` of shape `(M, K)`:
1. Pad both M and K to multiples of `gran_k`.
2. Reshape to `(M/gran_k, gran_k, K/gran_k, gran_k)` — a 2D grid of `gran_k × gran_k` blocks.
3. Compute one scale factor per block: `sf = amax(abs(block)) / 448.0`. Shape of `sf`: `(M/gran_k, K/gran_k)`.
4. If `use_ue8m0=True`: round `sf` to nearest power of 2.
5. Scale and cast to FP8.
6. Return `(x_fp8, sf)`.

Each `gran_k × gran_k` block gets one shared scale factor. This is "2D" scaling — scales vary along both M and K. Used for the B (weight) matrix in grouped GEMM where `use_block_cast_for_fp8=True`.

**Key difference:** `per_token` produces `(M, K/gran_k)` scale factors (one per row-chunk). `per_block` produces `(M/gran_k, K/gran_k)` scale factors (one per 2D block). Block-wise is coarser but more compatible with the weight matrix's reuse pattern across groups.

**`ceil_to_ue8m0(x)`** — MXFP8 scale factor rounding

```python
def ceil_to_ue8m0(x):
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))
```

Rounds each scale factor up to the nearest power of 2. This converts an arbitrary FP32 scale into a UE8M0 format (exponent-only, no mantissa), which is what the SM100 MXFP8 tensor core instructions expect. The "ceil" ensures no overflow — the quantized values are always <= 448 after scaling.
