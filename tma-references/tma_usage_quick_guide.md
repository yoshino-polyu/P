# TMA Usage Quick Guide

Sources: CUDA Programming Guide (Table 23) and CUDA Toolkit Driver API (`cuTensorMapEncodeTiled`).

Requires compute capability 9.0+.

---

## Tensor Map Creation API

### `cuTensorMapEncodeTiled`

Creates a tensor map for tiled memory access patterns.

```c
CUresult cuTensorMapEncodeTiled(
    CUtensorMap*             tensorMap,       // [out] descriptor to populate
    CUtensorMapDataType      tensorDataType,  // element data type
    cuuint32_t               tensorRank,      // number of dimensions
    void*                    globalAddress,   // base pointer to global memory tensor
    const cuuint64_t*        globalDim,       // tensor size per dimension (in elements)
    const cuuint64_t*        globalStrides,   // stride per dimension (in bytes), length = tensorRank - 1
    const cuuint32_t*        boxDim,          // tile (box) size per dimension (in elements)
    const cuuint32_t*        elementStrides,  // iteration step per dimension
    CUtensorMapInterleave    interleave,      // interleaved layout mode
    CUtensorMapSwizzle       swizzle,         // shared memory bank swizzle pattern
    CUtensorMapL2promotion   l2Promotion,     // L2 cache fetch granularity
    CUtensorMapFloatOOBfill  oobFill          // out-of-bounds fill value (zero or NaN)
);
```

### `cuTensorMapReplaceAddress`

Modifies only the `globalAddress` of an existing tensor map. All other fields remain unchanged.

```c
CUresult cuTensorMapReplaceAddress(
    CUtensorMap*  tensorMap,      // [in/out] existing tensor map to modify
    void*         globalAddress   // new base pointer (must satisfy original alignment requirements)
);
```

---

## Parameter Reference

### `tensorMap`

The `CUtensorMap` descriptor struct pointer. **Must be 64-byte aligned.** This is the alignment of the descriptor object itself, not the data it describes.
- On host: use `alignas(64)`.
- In shared memory: use `__shared__ alignas(64)` (examples use `alignas(128)` which exceeds the minimum).
- Via `cudaMalloc`: automatically satisfied (returns 256-byte aligned pointers).

### `tensorDataType`

Element data type enum (`CUtensorMapDataType`):

| Enum | Size |
|------|------|
| `UINT8` | 1 byte |
| `UINT16`, `FLOAT16`, `BFLOAT16` | 2 bytes |
| `UINT32`, `INT32`, `FLOAT32`, `FLOAT32_FTZ`, `TFLOAT32`, `TFLOAT32_FTZ` | 4 bytes |
| `UINT64`, `INT64`, `FLOAT64` | 8 bytes |
| `16U4_ALIGN8B` | 4 bits (packed, 8-byte aligned) |
| `16U4_ALIGN16B` | 4 bits (packed, 16-byte aligned with 8-byte gaps) |
| `16U6_ALIGN16B` | 6 bits (packed, 16-byte aligned with 4-byte gaps) |

### `tensorRank`

Number of tensor dimensions. Must be non-zero and <= 5. Must be >= 3 if `interleave` != `INTERLEAVE_NONE`.

### `globalAddress`

Base pointer to the global memory tensor — i.e., the starting address of the data that TMA will read from or write to. **This is a device pointer**, not a host pointer. It points to GPU global memory (DRAM/HBM visible to the GPU).

However, `cuTensorMapEncodeTiled` itself is a **host-side API** — the driver dereferences `globalAddress` only at TMA execution time on the device, not during the encode call. So the host code passes a device pointer it obtained earlier.

**How to obtain a valid `globalAddress`:**

```c
// Option 1: cudaMalloc (always 256-byte aligned, exceeds the 16-byte minimum)
float* d_tensor;
cudaMalloc(&d_tensor, M * N * sizeof(float));
// d_tensor is the globalAddress

// Option 2: PyTorch tensor (DeepGEMM style)
//   torch::Tensor t = ...;  // a CUDA tensor
//   t.data_ptr()            // returns the device pointer
cuTensorMapEncodeTiled(&tensor_map, ..., t.data_ptr(), ...);
```

Alignment requirements:
- Must be **16-byte aligned** (guaranteed by `cudaMalloc` / PyTorch CUDA allocator).
- **32-byte aligned** when `interleave` = `INTERLEAVE_32B`.
- **32-byte aligned** when `tensorDataType` is `16U4_ALIGN16B` or `16U6_ALIGN16B`.

**Passing the tensor map to the device** (after the encode call):

```c
// Recommended: pass as a const __grid_constant__ kernel parameter
__global__ void kernel(const __grid_constant__ CUtensorMap tensor_map) {
    // use tensor_map here
}
int main() {
    CUtensorMap map;
    cuTensorMapEncodeTiled(&map, ...);
    kernel<<<grid, block>>>(map);
}

// Alternative: copy into __constant__ memory
__constant__ CUtensorMap global_tensor_map;
cudaMemcpyToSymbol(global_tensor_map, &map, sizeof(CUtensorMap));
```

### `globalDim`

Array of `tensorRank` elements specifying tensor size (in number of elements) per dimension. Fastest-moving dimension first.
- Each element must be non-zero and <= 2^32.
- When `tensorDataType` is `16U4_ALIGN16B` or `16U6_ALIGN16B`: `globalDim[0]` must be a multiple of 128.
- When `tensorDataType` is `16U4_ALIGN8B`: `globalDim[0]` must be a multiple of 2.

### `globalStrides`

Array of `tensorRank - 1` elements specifying stride (in **bytes**) for each dimension (dimension 0 stride is implicit from element size).
- Must be a **multiple of 16** bytes and < 2^40.
- Must be a **multiple of 32** when `interleave` = `INTERLEAVE_32B`, or when `tensorDataType` is `16U4_ALIGN16B` / `16U6_ALIGN16B`.

Stride formula:
```
globalStrides[0] = globalDim[0] * elementSizeInBytes(tensorDataType) + padding[0]
for i = 1 to tensorRank - 2:
    globalStrides[i] = globalStrides[i-1] * (globalDim[i] + padding[i])
```
Each row must be padded so that the stride is 16-byte aligned.

### `boxDim`

Array of `tensorRank` elements specifying the tile/box size (in elements) to transfer per dimension.
- Each element must be non-zero and <= 256.
- When `interleave` = `INTERLEAVE_NONE`: `boxDim[0] * elementSize` must be a **multiple of 16 bytes**.
- When `tensorDataType` is `16U4_ALIGN16B` or `16U6_ALIGN16B`: `boxDim[0]` must be 128.

### `elementStrides`

Array of `tensorRank` elements specifying the iteration step per dimension.
- Each element must be non-zero and <= 8.
- When `interleave` = `INTERLEAVE_NONE`, `elementStrides[0]` is ignored (TMA does not support stride for dimension 0).
- When `elementStrides[i]` != 1, TMA loads `ceil(boxDim[i] / elementStrides[i])` elements along that dimension.

### `interleave`

Interleaved layout mode (`CUtensorMapInterleave`):

| Enum | Description |
|------|-------------|
| `INTERLEAVE_NONE` | Standard layout (default) |
| `INTERLEAVE_16B` | NC/8HWC8 style (16 bytes per group) |
| `INTERLEAVE_32B` | NC/16HWC16 style (32 bytes per group) |

### `swizzle`

Shared memory bank swizzling pattern (`CUtensorMapSwizzle`). Rearranges data in shared memory to avoid bank conflicts.

| Enum | Description |
|------|-------------|
| `SWIZZLE_NONE` | No swizzling |
| `SWIZZLE_32B` | Swizzle 16B chunks within 32B span |
| `SWIZZLE_64B` | Swizzle 16B chunks within 64B span |
| `SWIZZLE_128B` | Swizzle 16B chunks within 128B span |
| `SWIZZLE_128B_ATOM_32B` | Swizzle 32B chunks within 128B span |
| `SWIZZLE_128B_ATOM_32B_FLIP_8B` | Swizzle 32B + swap lower/upper 8B per 16B on alternate rows |
| `SWIZZLE_128B_ATOM_64B` | Swizzle 64B chunks within 128B span |

**Constraints:**
- When `interleave` = `INTERLEAVE_32B`, `swizzle` must be `SWIZZLE_32B`.
- When `interleave` = `INTERLEAVE_NONE` and `swizzle` != `SWIZZLE_NONE`, the bounding box inner dimension (`boxDim[0] * elementSize`) must be <= the swizzle size (32, 64, or 128 bytes respectively).

### `l2Promotion`

L2 cache fetch granularity (`CUtensorMapL2promotion`):

| Enum | Description |
|------|-------------|
| `L2_PROMOTION_NONE` | Default |
| `L2_PROMOTION_L2_64B` | 64-byte L2 fetch |
| `L2_PROMOTION_L2_128B` | 128-byte L2 fetch |
| `L2_PROMOTION_L2_256B` | 256-byte L2 fetch |

### `oobFill`

Out-of-bounds fill mode (`CUtensorMapFloatOOBfill`):

| Enum | Description |
|------|-------------|
| `FLOAT_OOB_FILL_NONE` | Fill with zero |
| `FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA` | Fill with special NaN (FP types only, not for packed U4/U6 types) |

---

## Alignment Summary (at the point of TMA load/store)

| Item | Requirement |
|------|-------------|
| Global memory address (`globalAddress`) | 16-byte aligned (32-byte when `interleave` = `INTERLEAVE_32B`, or packed `ALIGN16B` types) |
| Global memory strides (`globalStrides`) | Multiple of 16 bytes, < 2^40 (multiple of 32 when interleaved or packed) |
| Global memory sizes (`globalDim`) | Non-zero, <= 2^32. Does **not** need to be a multiple of 16 |
| Shared memory address | **128-byte aligned** |
| Shared memory barrier address | 8-byte aligned (guaranteed by `cuda::barrier`) |
| Size of transfer | Multiple of 16 bytes |
| Tensor map (`CUtensorMap`) address | **64-byte aligned** |

---

## Store-Specific Notes

- When writing from shared to global memory, parts of the tile may be out of bounds, but the **top-left corner cannot have negative indices** (unlike loads, which allow negative corner indices).
- Store completion uses a **bulk async-group** mechanism (thread-local), not a shared memory barrier.
- The same alignment requirements on addresses and sizes apply to both loads and stores.

---

## On-Device Modification of Tensor Maps

A tensor map is normally created on the host via `cuTensorMapEncodeTiled` and passed to a kernel as a `const __grid_constant__` parameter or through `__constant__` memory [CUDA Programming Guide, §4.11.2.2]. However, the tensor map can also be modified on the device at runtime. This is useful when a single kernel launch must process multiple tensors with different base addresses, dimensions, or strides — for example, "when processing a batch of tensors of various sizes in a single kernel launch" [CUDA Programming Guide, §4.11.2.2.1].

"On-device modification is only supported for tiled-type tensor maps; other tensor map types cannot be modified on device" [CUDA Programming Guide, §4.11.2.2.2].

### Modifiable Fields

The PTX instruction `tensormap.replace` "replaces the field, specified by `.field` qualifier, of the tensormap object at the location specified by the address operand `addr` with a new value" [PTX ISA, §9.7.9.26]. The descriptor can reside in shared memory (`.shared::cta`) or global memory (`.global`). The tensor map object is 1024 bits (128 bytes), indicated by the instruction type `.b1024` [PTX ISA, §9.7.9.26].

The following fields can be replaced:

| Field | Type | Description |
|-------|------|-------------|
| `.global_address` | `.b64` | Base pointer to global memory |
| `.rank` | `.b32` | Number of dimensions (new value must be rank − 1, zero-based) |
| `.global_dim` | `.b32` | Tensor size per dimension (indexed by ordinal) |
| `.global_stride` | `.b64` | Stride per dimension in bytes (indexed by ordinal) |
| `.box_dim` | `.b32` | Tile/box size per dimension (indexed by ordinal) |
| `.element_stride` | `.b32` | Iteration step per dimension (indexed by ordinal) |
| `.elemtype` | `.b32` | Element data type (immediate constant) |
| `.interleave_layout` | `.b32` | Interleaved layout mode (immediate constant) |
| `.swizzle_mode` | `.b32` | Swizzle pattern (immediate constant) |
| `.swizzle_atomicity` | `.b32` | Swizzle atomicity (immediate constant) |
| `.fill_mode` | `.b32` | Out-of-bounds fill mode (immediate constant) |

[PTX ISA, §9.7.9.26]

"tensormap.replace is treated as a weak memory operation, on the entire 1024-bit opaque tensormap object, in the Memory Consistency Model" [PTX ISA, §9.7.9.26].

### Modification Workflow

The recommended workflow proceeds in four steps [CUDA Programming Guide, §4.11.2.2.2]:

1. **Pass a template tensor map to the kernel.** Create a tensor map on the host using `cuTensorMapEncodeTiled` with initial or placeholder parameters. Pass it to the kernel as a `const __grid_constant__` parameter, a pointer to global memory, or a `__constant__` variable.

2. **Copy the template to shared memory.** One thread copy-initializes a `CUtensorMap` in shared memory from the template. On `sm_90a`, "a zero-initialized buffer in shared memory may also be used as the initial tensor map value", which "enables encoding a tensor map purely on device, without using the driver API" [CUDA Programming Guide, §4.11.2.2.2].

3. **Modify fields in shared memory.** Use `tensormap.replace` (wrapped by `cuda::ptx::tensormap_replace_*` in C++) to overwrite specific fields. The new value is provided by the operand `new_val` [PTX ISA, §9.7.9.26]. Only the fields that change need to be replaced; unchanged fields retain their values from the template.

4. **Copy the modified tensor map from shared memory to global memory with a release fence.** The `tensormap.cp_fenceproxy` instruction (wrapped by `cuda::ptx::tensormap_cp_fenceproxy`) copies the 128-byte descriptor from shared memory to a global memory location and issues a release fence. This makes the updated descriptor visible to the TMA hardware [CUDA Programming Guide, §4.11.2.2.2].

```
// Pseudocode (PTX-level steps):
// Step 2: copy template → SMEM
smem_tmap = template_tensor_map;

// Step 3: modify fields in SMEM
tensormap.replace.tile.global_address.shared::cta.b1024.b64 [smem_tmap], new_address;
tensormap.replace.tile.global_dim.shared::cta.b1024.b32     [smem_tmap], 0, new_dim0;
tensormap.replace.tile.global_stride.shared::cta.b1024.b64  [smem_tmap], 0, new_stride0;

// Step 4: copy SMEM → GMEM with release fence
tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned
    [gmem_tmap], [smem_tmap], 128;
```

### Fencing Requirements

"Using a tensor map in global memory requires explicitly establishing a release-acquire pattern in the tensor map proxy between the threads that modify the tensor map and the threads that use it" [CUDA Programming Guide, §4.11.2.2.3]:

- **Release** (writer side): performed by `tensormap.cp_fenceproxy` when copying the modified descriptor from shared memory to global memory.
- **Acquire** (reader side): before any thread uses the updated tensor map in a `cp.async.bulk.tensor` instruction, it must execute `fence.proxy.tensormap::generic.acquire` (wrapped by `cuda::ptx::fence_proxy_tensormap_generic`) on the global memory descriptor.

The scope of the fence depends on the relationship between the writer and reader. "If the two threads participating in the release-acquire pattern are on the same device, the `.gpu` scope suffices. If the threads are on different devices, the `.sys` scope must be used" [CUDA Programming Guide, §4.11.2.2.3].

"Once a tensor map has been acquired by one thread, it can be used by other threads in the block after sufficient synchronization", for example using `__syncthreads()`. Threads in other blocks must perform their own acquire fence [CUDA Programming Guide, §4.11.2.2.3].

"If there are no intermediate modifications, the fence does not have to be repeated before each `cp.async.bulk.tensor` instruction" [CUDA Programming Guide, §4.11.2.2.3].

### Waiting Before Modification

Before modifying a tensor map that is being read by in-flight TMA operations, the thread must wait for those operations to finish reading the descriptor. The `cp.async.bulk.wait_group` instruction with the `.read` modifier serves this purpose. Per the PTX ISA, "the optional `.read` modifier indicates that the waiting has to be done until all the bulk async operations in the specified bulk async-group have completed: 1. reading from the tensormap 2. the reading from their source locations" [PTX ISA, §9.7.9.25.6.2]. This ensures the previous descriptor is no longer in use before it is overwritten.

### Host-Side Address Replacement

For the common case where only the base address needs to change (and all other fields remain the same), the CUDA driver provides a host-side API [CUDA Toolkit Driver API, `cuTensorMapReplaceAddress`]:

```c
CUresult cuTensorMapReplaceAddress(
    CUtensorMap*  tensorMap,      // existing tensor map to modify
    void*         globalAddress   // new base pointer (must satisfy original alignment)
);
```

This API will "modify an existing tensor map descriptor with an updated `globalAddress`" [CUDA Toolkit Driver API, `cuTensorMapReplaceAddress`]. All other fields remain unchanged. The new address "must follow previous alignment requirements" [CUDA Toolkit Driver API, `cuTensorMapReplaceAddress`].

### Architecture Support

- `tensormap.replace`: introduced in PTX ISA version 8.3. Supported on `sm_90a`, `sm_100a`, `sm_110a`, `sm_120a`, and family-specific architectures from PTX ISA version 8.8 [PTX ISA, §9.7.9.26].
- `.swizzle_atomicity` field replacement: supported from `sm_100a` onwards [PTX ISA, §9.7.9.26].
- "The format of the tensor map may change over time. Therefore, the `cuda::ptx::tensormap_replace` functions and corresponding `tensormap.replace.tile` PTX instructions are marked as specific to `sm_90a`" [CUDA Programming Guide, §4.11.2.2.2].

### References

1. **CUDA Programming Guide** — §4.11.2.2.1 "Encoding a Tensor Map on Device", §4.11.2.2.2 "Device-side Encoding and Modification of a Tensor Map", §4.11.2.2.3 "Usage of a Modified Tensor Map"
2. **PTX ISA** — §9.7.9.26 "tensormap.replace", §9.7.9.25.6.2 "cp.async.bulk.wait_group"
3. **CUDA Toolkit Driver API** — `cuTensorMapReplaceAddress`
