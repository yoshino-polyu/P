---
name: cutlass-grouped-gemm
description: CUTLASS Grouped GEMM — minimal implementation framework, kernel architecture, and implementation details for building grouped GEMM with minimal abstractions (1-2 header files, no heavy CUTLASS encapsulation)
---

# CUTLASS Grouped GEMM — Minimal Implementation Guide

> **Goal:** Understand how to implement CUTLASS-based grouped GEMM from scratch with minimal abstraction layers (1-2 header files), bypassing CUTLASS's deep encapsulation hierarchy.

---

## Implementation Framework

*(To be filled: overall architecture, dispatch logic, problem grouping strategy)*

---

## Kernel Architecture

*(To be filled: tile scheduling, CTA mapping across groups, persistent kernel design)*

---

## Memory Layout & Data Flow

*(To be filled: pointer arrays, stride arrays, group indexing, GMEM→SMEM→RF pipeline)*

---

## Core Computation

*(To be filled: MMA operations, warp-level GEMM, accumulator handling, epilogue)*

---

## Pipeline Schedule

*(To be filled: producer/consumer warp groups, TMA load stages, MMA compute stages, epilogue overlap, software pipelining, barrier synchronization)*

---

## Minimal Header Structure

*(To be filled: what goes in each header, API surface, template parameters)*

---

## Implementation Details & Pitfalls

*(To be filled: edge cases, group boundaries, tail handling, performance considerations)*

---

## CUTLASS Grouped GEMM Example Source Files

> **CUTLASS root:** `/home/xule/cutlass/`
> Each entry maps a numbered example folder to its `.cu` source file(s).

| # | Example Folder | Source File(s) |
|---|---------------|----------------|
| 92 | `examples/92_blackwell_moe_gemm/` | `92_blackwell_moe_gemm_rcgrouped.cu`, `92_blackwell_moe_gemm_blockscaled_rcgrouped.cu` |

### Full Paths (for Claude Code reference)

- `/home/xule/cutlass/examples/92_blackwell_moe_gemm/92_blackwell_moe_gemm_rcgrouped.cu` — SM100 Blackwell, Ragged Contiguous FP8 grouped GEMM (TMA ptr-array)
- `/home/xule/cutlass/examples/92_blackwell_moe_gemm/92_blackwell_moe_gemm_blockscaled_rcgrouped.cu` — SM100 Blackwell, MoE MX block-scaled ragged contiguous grouped GEMM

---

## Blackwell Grouped GEMM Comparison

> Summary of all Blackwell (SM100) grouped GEMM examples. Pick which to study further.

### 92_blackwell_moe_gemm (focused on Pattern 3 & 4)

All 92_* examples target **MoE (Mixture of Experts) decoding** where N (tokens per expert) varies wildly and is typically small.

> **CPASYNC vs TMA — why it matters for grouped GEMM:**
>
> **cp.async** is a per-thread PTX instruction (introduced on Ampere SM80) that copies 4, 8, or 16 bytes from global memory to shared memory. Each thread computes its own source address and issues its own copy. There is no tensor map, no hardware accelerator, and no automatic swizzling or multidimensional indexing — just raw non-blocking global→shared copies tracked via `mbarrier` or `cp.async.wait_group`. To load a tile, many threads each issue many 16-byte copies in a loop.
>
> **TMA** (Tensor Memory Accelerator, Hopper SM90+) is a dedicated hardware unit that copies entire multidimensional tiles in one instruction. The programmer creates a **tensor map** (descriptor) on the host that encodes base pointer, shape, strides, and swizzle pattern. A single thread issues `cp.async.bulk.tensor` and the TMA engine handles the rest. This is far more efficient for large, regular tiles — but the tensor map must be **updated** whenever the base pointer or shape changes (i.e., switching between groups).
>
> In MoE grouped GEMM, the activation matrix B changes shape (different N) for every expert. Updating the TMA descriptor per group adds overhead. Using cp.async for B avoids this: each thread just recomputes the source address with the new pointer/N — no descriptor update needed. The weight matrix A (same shape across groups) still uses TMA since its descriptor never changes.

#### Pattern 3: `92_blackwell_moe_gemm_rcgrouped.cu` — Ragged Contiguous FP8

- **Data types:** A=FP8(e4m3), B=FP8(e4m3), C/D=FP16, Acc=FP32
- **Problem shape:** `MoEProblemShape` — same M and K across groups, N varies
- **Data loading:** TMA for both A and B — `KernelPtrArrayTmaWarpSpecialized1SmSm100`
- **Input passing:** A is batched (TMA, same weight), B uses pointer arrays (TMA descriptor updates per group)
- **Tile shape:** 1SM: 128x128x64, 2SM: 256x256x64 (larger tiles than pattern 2)
- **Cluster:** Runtime (default 4x2), supports sparse groups (N=0 for some experts)
- **Key feature:** "Ragged Contiguous" — weights (A) shared via batched TMA load, activations (B) vary per group. Uses standard TMA for both, bigger tiles for higher throughput.

> **What is Ptr-Array TMA?**
>
> Recall from `tma_usage_quick_guide.md` that TMA requires a `CUtensorMap` descriptor created via `cuTensorMapEncodeTiled()`, encoding: `globalAddress`, `globalDim`, `globalStrides`, `boxDim`, `swizzle`, etc. In a standard batched GEMM, one descriptor suffices because all batches share the same shape and a constant batch stride (the L dimension in the tensor map).
>
> **Ptr-Array TMA** is the technique of creating **one** TMA descriptor at host time (with group 0's parameters), then **replacing its fields on-device** as the kernel iterates over groups. "Iterates over groups" means: the persistent kernel's main loop assigns tiles to CTAs; when a CTA's next tile belongs to a different group (i.e., `did_batch_change == true`), the CTA updates the descriptor before issuing TMA loads for the new group. The "pointer array" is a `const Element**` in GMEM (global memory / HBM) — an array of per-group base addresses that the kernel indexes into when updating the descriptor.
>
> **Components of Ptr-Array TMA:**
>
> 1. **One `CUtensorMap` created at host time** via `cuTensorMapEncodeTiled()`, using group 0's shape. For B in rcgrouped, the initial address is even set to `nullptr` — it will be replaced before any load.
>    (Source: `sm100_mma_array_warpspecialized_rcggemm.hpp`, Params construction, lines 320–402)
>
> 2. **One copy in shared memory (SMEM)** (`TensorMapStorage::smem_tensormap_B`, 128-byte aligned). At kernel start, `tensormaps_init()` copies the descriptor from Params into SMEM. This SMEM copy serves as a **staging buffer for modifications** — it is NOT where TMA reads from. Only B needs this — A's descriptor is static and stays in the TMA atom.
>
> 3. **A GMEM workspace** of `sm_count × NumTmaDescriptorsPerSm` descriptor copies. **TMA hardware reads the descriptor from GMEM, not SMEM.** After modifying the SMEM copy, the kernel copies it to a GMEM slot, and TMA loads reference that GMEM slot.
>    - `NumTmaDescriptorsPerSm = SchedulerPipelineStageCount + Stages + 2` (line 131).
>    - `SchedulerPipelineStageCount` = how many tiles the scheduler can queue ahead.
>    - `Stages` = how many TMA pipeline stages (in-flight loads).
>    - `+ 2` = buffer for consumer and producer in-flight accesses.
>    - Multiple GMEM slots per SM allow **pipelining**: while TMA hardware reads descriptor N for an in-flight load, the kernel can write the updated descriptor N+1 for the next group into a different GMEM slot, avoiding stalls.
>    - The slots are cycled round-robin: `TensorMapArray::operator[](idx)` returns `tma_desc_b + (idx % NumTmaDescriptorsPerSm)`.
>
> 4. **Per-group on-device updates** — the full read/write flow:
>
>    **Step 1: Wait for in-flight TMA loads to finish** (one elected thread)
>    - `cp.async.bulk.commit_group` — commits all pending TMA transactions
>    - `cp.async.bulk.wait_group.read 0` — waits until all committed transactions complete
>    - This ensures the previous GMEM descriptor is no longer being read by TMA hardware.
>
>    **Step 2: Modify the SMEM descriptor** (one elected thread, pure writes from registers into SMEM)
>    - The `tensormap.replace.*` PTX instructions are **pure writes** — the new value comes from a register and is written directly into the specified field of the 128-byte SMEM descriptor. There is no read-modify-write; no old value is read first.
>
>    | PTX instruction | Read from | Write to |
>    |----------------|-----------|----------|
>    | `tensormap.replace.tile.global_address.shared::cta.b1024.b64 [smem], reg` | Register holding `ptr_B[next_group]` (from GMEM pointer array) | SMEM descriptor's `globalAddress` field |
>    | `tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [smem], dim_idx, reg` (×5) | Registers holding new N, K, 1, 1, 1 (computed from `MoEProblemShape`) | SMEM descriptor's `globalDim[0..4]` fields |
>    | `tensormap.replace.tile.global_stride.shared::cta.b1024.b64 [smem], stride_idx, reg` (×4) | Registers holding new byte strides (computed from new N, K) | SMEM descriptor's `globalStrides[0..3]` fields |
>
>    (Source: `cute/arch/copy_sm90_desc.hpp`, lines 342–417)
>
>    **Step 3: Copy modified SMEM descriptor to GMEM + fence** (entire warp, aligned)
>    - `tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned [gmem], [smem], 128`
>    - **Read from:** SMEM descriptor (128 bytes, the just-modified copy)
>    - **Write to:** GMEM descriptor slot (the next slot in this SM's round-robin buffer)
>    - Also issues a GPU-wide **release fence**, making the new GMEM descriptor visible to TMA hardware.
>    (Source: `cute/arch/copy_sm90_desc.hpp`, lines 425–436)
>
>    **Step 4: Before next TMA load, acquire fence** (entire warp)
>    - `fence.proxy.tensormap::generic.acquire.gpu [gmem], 128`
>    - **Read from:** GMEM descriptor (the slot that was just written)
>    - This **acquire fence** ensures the updated GMEM descriptor is visible before TMA reads it.
>    - Then the TMA load instruction references this GMEM descriptor: `copy(tma_load_b.with(gmem_descriptor, ...), ...)`.
>    (Source: `cute/arch/copy_sm90_desc.hpp`, lines 457–471)
>
> 5. **A pointer array in GMEM** (`const ElementB** ptr_B`) — the kernel reads `ptr_B[group_idx]` to get the new `globalAddress` for each group. This is a simple indirection: `mainloop_params.ptr_B[next_batch]`.
>
> **Why SMEM as staging buffer instead of modifying GMEM directly?**
> The `tensormap.replace.*` instructions only operate on SMEM descriptors (note `shared::cta` in the instruction encoding). There is no PTX instruction to replace fields of a GMEM descriptor directly. The full flow is:
> 1. **Host creates** one `CUtensorMap` via `cuTensorMapEncodeTiled()` (with group 0's shape). This 128-byte descriptor is embedded in the kernel's `Params` struct.
> 2. **Kernel launch** passes `Params` (including the embedded descriptor) to the kernel.
> 3. **`tensormaps_init()`** (one elected thread) copies the 128-byte descriptor from `Params` into SMEM (`TensorMapStorage::smem_tensormap_B`) via a 128-bit `copy`. Now SMEM holds a complete, valid descriptor.
> 4. **Per-group: modify SMEM** — `tensormap.replace.*` writes new address/dims/strides into specific fields of the SMEM descriptor. The rest of the 128-byte descriptor (boxDim, swizzle, oobFill, etc.) remains unchanged from the initial copy.
> 5. **Copy SMEM → GMEM** — `tensormap.cp_fenceproxy` copies the full 128 bytes to a GMEM workspace slot.
> 6. **Fence** — release fence makes it visible to TMA hardware.
> 7. **TMA reads from GMEM** — the load instruction references the GMEM descriptor slot.
>
> So the `tensormap.replace.*` instructions are **field-level overwrites** into an already-complete descriptor in SMEM — they don't create a descriptor from scratch, they only patch the fields that change between groups.
>
> **Is B non-contiguous in memory?** Not in this example. Line 554 allocates one flat buffer (`block_B.reset(total_elements_B)`), and line 586 sets `ptr_B[i] = block_B.get() + offset_B[i]` — all groups packed contiguously.
>
> **Why use Ptr-Array TMA then?** The motivation is about **variable-sized groups**, not contiguity:
> 1. **Batched TMA requires a constant batch stride.** A `CUtensorMap` encodes one fixed stride for the L (batch) dimension. When N differs across groups, the offset between groups is `K × N_i` which varies — a single batch stride cannot express this.
> 2. **TMA must know each group's shape.** The `globalDim` fields in the descriptor determine out-of-bounds masking (`oobFill`). If the descriptor says `dim[0]=N_max` but the actual group has `N < N_max`, TMA would read garbage. The dims must match the actual group.
> 3. **Also supports non-contiguous memory.** In production MoE, activations may be scattered after routing. The pointer array works identically either way.
>
> **Batched TMA (A) vs Ptr-Array TMA (B):**
>
> | Aspect | Batched TMA (A) | Ptr-Array TMA (B) |
> |--------|-----------------|-------------------|
> | Layout arg | `LayoutA` (no `*`) | `LayoutB *` (with `*`) |
> | Host-time descriptors | 1 `CUtensorMap`, static | 1 `CUtensorMap`, used as template |
> | Where TMA reads descriptor | Kernel arg (Params) | GMEM workspace slot |
> | Memory layout | Contiguous, **fixed** batch stride (`M × K`) | Contiguous or scattered — abstracted by pointer array |
> | Why it works | M and K same across groups → constant stride | N varies → no constant stride possible |
> | On-device updates | None — batch index via TMA's built-in L dim | Modify SMEM → copy to GMEM → fence → TMA reads GMEM |
> | GMEM workspace | None | `sm_count × NumTmaDescriptorsPerSm × 128 bytes` |
>
> The `*` suffix on layout types (e.g., `LayoutB *` vs `LayoutB`) is the compile-time marker that selects this path. Internally, `cute::remove_pointer_t<LayoutB *>` recovers the actual layout type for computation.
>
> **`MoEProblemShape` vs `GroupProblemShape`:**
>
> `GroupProblemShape` stores an array of complete `{M,N,K}` tuples — each group can have fully independent dimensions. `MoEProblemShape` stores fixed `max_m` and `max_k` plus a `tokens_per_expert` array for variable N. It reconstructs per-group shapes as `{max_m, tokens_per_expert[i], max_k}`. This is more memory-efficient for MoE where only N varies.
> (Source: `cutlass/gemm/group_array_problem_shape.hpp`)

#### Pattern 4: `92_blackwell_moe_gemm_blockscaled_rcgrouped.cu` — MX Block-Scaled Ragged Contiguous

- **Data types:** A=MX_FP8(e4m3), B=MX_FP8(e4m3), C/D=BF16, Acc=FP32, ScaleFactors=UE8M0
- **Problem shape:** `MoEProblemShape` — same M and K across groups, N varies
- **Data loading:** TMA for A (batched), Ptr-Array TMA for B — `KernelPtrArrayTmaWarpSpecialized1SmMxf8f6f4Sm100`
- **Input passing:** A is a flat buffer with scale factors (SFA), B uses pointer arrays with per-group scale factors (SFB)
- **Tile shape:** 1SM: 128x256x128, 2SM: 256x256x128
- **Operator class:** `OpClassBlockScaledTensorOp` — uses MX block-scaled tensor cores
- **Extra inputs:** Scale factor tensors SFA (batched, shared across groups) and SFB (pointer array, per group), with interleaved layouts
- **Key feature:** Same Ptr-Array TMA pattern as rcgrouped but with MX block-scaled quantization. Most complex data management (SF tensors for A, B, and optionally D output).

### Decision Matrix

| Example | Data Types | A Loading | B Loading | M,N,K per group | Best For |
|---------|-----------|-----------|-----------|-----------------|----------|
| **92_rcgrouped** | FP8 | TMA(batched) | TMA(ptr array) | Same M,K; diff N | MoE FP8, larger tiles, higher throughput |
| **92_blockscaled_rc** | MX_FP8+SF | TMA(batched) | TMA(ptr array) | Same M,K; diff N | MoE MX block-scaled |

### Recommendations for Further Study

- **FP8 grouped GEMM:** Start with `92_blackwell_moe_gemm_rcgrouped.cu` — uses Ptr-Array TMA for B, larger tiles, higher throughput.
- **MX block-scaled quantization:** `92_blackwell_moe_gemm_blockscaled_rcgrouped.cu` — same Ptr-Array TMA pattern, adds MX scale factor tensors.
