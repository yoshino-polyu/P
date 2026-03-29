---
name: k-gemm-tma
description: Our K-GEMM methods to bypass explicit padding
---

# Note: Replacing Explicit Token Padding with Virtual TMA Zero-Fill for Weight-Gradient GEMM

## Short summary

The proposal is to avoid **physically padding each expert’s token dimension in global memory** for the weight-gradient GEMM, and instead perform **virtual padding only at the point of MMA consumption**. For each expert, we keep the true unpadded token count `M` in global memory, create a per-expert TMA descriptor, and use TMA `oobFill` so that out-of-bounds elements along the ragged tail are synthesized as zeros when the tile is materialized in shared memory. Because `K` is already aligned, the load path is regular and descriptor-friendly, while the shared-memory tile seen by MMA can be rounded up to a legal aligned extent such as 64 or 128. Under the assumption that alignment is required **only before MMA consumption**, this virtual shared-memory padding is equivalent to explicit global-memory padding for correctness, while avoiding the storage and movement overhead of padded tokens.

---

## Setup

For weight gradient, we compute:

\[
dW = X^T dY
\]

with shapes:

- \(X^T \in \mathbb{R}^{K \times M}\)
- \(dY \in \mathbb{R}^{M \times N}\)

So the **GEMM contraction axis is \(M\)**.

In MoE training, for a given expert, `M` is the number of routed tokens assigned to that expert. This count is dynamic and may be ragged, for example:

- `M = 37`
- `M = 91`
- `M = 243`

Many MMA-friendly kernels want the contraction tile to look aligned, for example rounded up to a multiple such as 16, 32, 64, or 128 depending on the recipe and implementation.

The key assumption here is:

> The implementation does **not** require `M` to be rounded up earlier for quantization, scaling, or grouped GEMM scheduling. It only requires an aligned tile **right before MMA consumes it**.

That assumption is exactly what makes this idea work.

---

## The idea

Instead of explicitly padding the expert-token buffer in global memory, do the following:

1. Keep the expert data in global memory with its **true** token count `M`.
2. Create a **per-expert TMA descriptor** for the logical tile to be consumed.
3. Let TMA load all valid in-bounds elements from global memory.
4. Use **`oobFill`** so any out-of-bounds positions in the logical tile are synthesized as zeros.
5. Materialize an **aligned shared-memory tile**.
6. Feed that aligned tile into MMA.

This replaces:

- **physical padding in global memory**

with:

- **virtual padding in shared memory**

---

## High-level picture

### 1. Original unpadded expert data in global memory

```text
Example expert:
M = 37, K aligned

X^T in GMEM:  [ K x 37 ]
dY  in GMEM:  [ 37 x N ]
```

### 2. TMA materializes a larger logical tile

```text
Logical MMA-facing tile along M:
round_up(37) = 64

Requested logical extent:
[ 0 ............................. 36 ][ 37 .............. 63 ]
[          valid region             ][    OOB tail         ]
```

### 3. TMA with `oobFill` writes shared memory

```text
SMEM tile after TMA:
[ real real real real ... real ][ 0 0 0 0 0 ... 0 ]
  <------ 37 valid values ---->   <--- zero tail --->
```

### 4. MMA consumes the aligned tile

```text
MMA sees contraction length 64 instead of 37

sum_{m=0}^{63} a_m b_m
=
sum_{m=0}^{36} a_m b_m
+
sum_{m=37}^{63} 0
```

So the compute path is aligned, but global memory only stores the real 37-token payload.

---

## Why this works

### 1. Alignment is needed only at consumption time

Under the given assumption, the system does not care whether the expert-token buffer is padded in global memory. It only cares that the tile **presented to MMA** has the required aligned extent.

That means the implementation requirement is about the **consumer view**, not the **storage format**.

So if TMA can synthesize:

```text
[ valid data ][ zero tail ]
```

in shared memory, then MMA gets exactly what it needs.

### 2. Zero-filled padding is mathematically equivalent to explicit padding

Suppose actual `M = 37`, but the aligned MMA tile uses `M' = 64`.

Then for a contraction:

\[
\sum_{m=0}^{M'-1} a_m b_m
=
\sum_{m=0}^{M-1} a_m b_m
+
\sum_{m=M}^{M'-1} 0
\]

As long as the out-of-bounds region contributes zeros, the padded computation is identical to the original ragged computation.

So explicit padding in global memory and virtual padding in shared memory are equivalent for correctness.

### 3. `K` alignment helps make the TMA path regular

It is still important that `K` is already aligned.

That does **not** mean the original contraction length `M` is aligned. What it does mean is that the memory layout is regular enough that per-expert TMA descriptors and aligned accesses are practical and efficient.

A concise way to say it is:

> `K` alignment makes the descriptor-based TMA load path regular; `oobFill` makes the MMA-facing reduction tile aligned.

---

## Diagram: explicit padding vs virtual padding

### A. Explicit padding in global memory

```text
Global memory:
| real real real real real ... real | 0 0 0 0 0 ... 0 |
|<------------- M=37 ------------->|<--- pad tail --->|

TMA load:
| real real real real real ... real | 0 0 0 0 0 ... 0 |

Shared memory:
| real real real real real ... real | 0 0 0 0 0 ... 0 |

MMA consumes aligned tile
```

### B. Virtual padding with TMA `oobFill`

```text
Global memory:
| real real real real real ... real |
|<------------- M=37 ------------->|

TMA descriptor requests a larger logical tile:
| real real real real real ... real | OOB OOB OOB OOB ... |

TMA + oobFill writes shared memory:
| real real real real real ... real | 0 0 0 0 0 ... 0 |

MMA consumes the same aligned tile
```

### Conclusion from the diagram

From MMA’s point of view, the two cases are equivalent.

The only difference is where the padded zeros are introduced:

- **explicit approach:** zeros are stored in global memory
- **virtual approach:** zeros are synthesized by TMA in shared memory

If the alignment requirement is only for MMA consumption, then synthesizing the tail in shared memory is sufficient.

---

## Concrete example

Take one expert with:

- `K = 4096` (already aligned)
- `N = 14336`
- `M = 37`

Then:

- `X^T` has shape `[4096 x 37]`
- `dY` has shape `[37 x 14336]`

Without explicit global-memory padding, we keep exactly those shapes in GMEM.

Now build a TMA descriptor that logically rounds `M` up to `64` for the MMA tile:

```text
GMEM:
[4096 x 37] and [37 x 14336]

TMA logical tile:
[4096 x 64] and [64 x 14336]
             ^^^^^^^^^^^^^^^
             tail is out-of-bounds and zero-filled
```

After TMA with `oobFill`, shared memory contains a 64-deep contraction tile. MMA consumes a legal aligned reduction tile, but only the first 37 positions contain real data.

So the kernel gets the aligned tile shape it needs without storing 27 zero-token positions in global memory.

---

## Why this can replace explicit padding

This proposal can replace explicit padding precisely because the role of padding here is **representational**, not **semantic**.

Padding is not changing the meaning of the computation. It is only turning a ragged reduction into an aligned tile representation that the MMA path can consume.

If that representation is needed only in the final shared-memory tile, then there is no need to pre-materialize it in global memory.

That is the central reason the idea works:

> Explicit padding is unnecessary when the only consumer that cares about alignment is MMA, because TMA can synthesize the required zero tail directly into shared memory.

---

## Practical takeaway

The proposal is:

- store only real expert tokens in global memory,
- avoid explicit token padding in GMEM,
- use per-expert TMA descriptors,
- use `oobFill` to synthesize the ragged tail as zeros,
- materialize aligned shared-memory tiles,
- let MMA consume those aligned tiles.

Under the stated assumption, this is a valid replacement for explicit padding.

---

## One-sentence conclusion

If the implementation requires alignment **only before MMA consumption**, then per-expert TMA loading with `oobFill` can replace explicit global-memory padding by generating an aligned zero-padded tile in shared memory on the fly, with identical math and lower padding overhead.
