---
name: system-paper-writings
description: LiquidGEMM Paper Analysis — structure, methodology, writing patterns, and Q&A for SC '25 systems paper
---

# LiquidGEMM Paper Analysis

## Paper Structure

**LiquidGEMM: Hardware-Efficient W4A8 GEMM Kernel for High-Performance LLM Serving**
Published at SC '25, co-authored by Shanghai Jiao Tong University and ByteDance Seed.

### Section Outline

| # | Section | Pages | Purpose |
|---|---------|-------|---------|
| 1 | Introduction | 1-2 | Motivation: W4A8 is theoretically superior but existing kernels underperform |
| 2 | Preliminary | 2-3 | Background on integer quantization and GEMM on GPUs |
| 3 | Motivation | 3-5 | Profiling gap between roofline analysis and practice; cost model development |
| 4 | Quantization Algorithm (LiquidQuant) | 5-6 | Proposes LQQ: overflow-free dequantization via rotation + XOR trick |
| 5 | High Performance W4A8 GEMM Kernel | 6-9 | ImFP pipeline, Dual-MMA packed layout, hardware-efficient dequantization |
| 6 | LLM Serving System and Offline Quantization | 9 | End-to-end system integration |
| 7 | Experiments | 9-11 | System throughput, GEMM kernel benchmarks, ablation study |
| 8 | Related Work | 11 | Quantization methods, LLM serving systems |
| 9 | Conclusion | 11 | Summary of contributions |

### Key Contributions
1. **LiquidQuant (LQQ)** - A quantization scheme that shifts INT8 into UINT8 before quantizing to UINT4, enabling dequantization with just 2 hardware instructions (IMAD + XOR) per 4 elements, avoiding overflow entirely.
2. **Implicit Fine-Grained Pipeline (ImFP)** - A single-producer multi-consumer pipeline that overlaps weight loading, dequantization, and MMA across warp groups without software synchronization.
3. **Dual-MMA Packed Layout** - A memory layout that packs data for two consecutive MMA ops per thread, enabling a single LDS.128 instruction to load all needed data, eliminating bank conflicts.

### Results Highlights
- Up to **2.90x** speedup over QServe (state-of-the-art W4A8)
- Up to **4.94x** end-to-end system-level speedup
- **1.12-1.63x** over NVIDIA TensorRT-LLM across various precisions

---

## Thoughts on Why They Draft Like This

### 1. The "Gap-Driven" Narrative Structure

The paper follows a very deliberate **"theory vs. reality gap"** narrative:

- **Section 1 (Intro)**: Sets up the *expectation* from roofline analysis that W4A8 should beat W8A8.
- **Section 3 (Motivation)**: Immediately *demolishes* that expectation with profiling data showing W4A8 is actually 2x *slower* than W8A8 at batch size 256.
- **Sections 4-5**: Presents the solution.

This is a textbook systems-paper storytelling technique: establish what *should* work, prove it doesn't, diagnose *why*, then fix it. SC reviewers appreciate this because it turns what could be "just another kernel optimization" into a story about *understanding hardware*. The gap itself is the hook.

### 2. The Cost Model as the Structural Backbone

Section 3 develops a formal cost model (Equations 3-6) that decomposes GEMM time into $T_{LD}$, $T_{DQ}$, and $T_{MMA}$. This is not just analysis for its own sake -- it serves as the **design specification** for everything that follows:

- LQQ (Section 4) is designed to minimize $\alpha$ (instructions per dequantized element) so that $T_{DQ} \leq T_{LD}$.
- ImFP (Section 5) is designed to overlap all three terms so that total time = $\max(T_{LD}, T_{DQ}, T_{MMA})$ rather than their sum.

By anchoring the design in a quantitative model, the authors can make precise claims like "$\alpha \leq 5.07$ is required on H100" and then show their method achieves $\alpha = 2$ (or ~1.75 including unpacking). This gives the paper a **deductive structure**: problem model -> design constraints -> solution. SC/ASPLOS/ISCA reviewers respect this because it shows the work isn't ad-hoc.

### 3. Separating Algorithm from System Engineering

The paper cleanly separates:
- **Section 4** (LQQ): The *mathematical insight* -- rotation into UINT8 domain, two's complement trick, XOR for MSB flip. This is the "clever idea" section.
- **Section 5** (Kernel): The *systems engineering* -- pipeline design, memory layout, warp group scheduling. This is the "hard engineering" section.

This separation is intentional. Section 4 could stand alone as a short theory contribution. Section 5 could stand alone as a GPU kernel optimization paper. Together they present a **co-design** story: the algorithm is designed *for* the hardware, and the pipeline is designed *for* the algorithm. SC specifically values this kind of cross-layer co-design.

### 4. The ExCP-then-ImFP Presentation

The authors first present ExCP (the "obvious" 3-WG pipeline), explain why it fails (round-trip RF/SMEM traffic, software sync overhead), and then present ImFP as the refinement. This is a **strawman-then-solution** pattern:

- It shows the authors considered the natural approach first.
- It clarifies *what's hard* about the problem (not just "pipeline it" -- the data movement and sync overhead matter).
- It makes ImFP's contribution clearer by contrast.

The ablation study in Section 7.3 (Figure 13) directly validates this: ExCP sometimes *hurts* at small batch sizes while ImFP always helps.

### 5. Dual Evaluation Strategy

The experiments are split into:
1. **System-level** (Table 1, Figures 10-11): End-to-end throughput with LiquidServe vs. QServe vs. TRT-LLM. Shows practical impact.
2. **Kernel-level** (Figures 12-13): Isolated GEMM benchmarks in a unified framework. Shows the kernel's contribution vs. confounds.

This dual approach is necessary because:
- System-level alone is unfair (different attention kernels, KV cache schemes).
- Kernel-level alone undersells the impact (reviewers want to see end-to-end numbers).

The `LiquidServe/wo` variant (LiquidServe with QServe's GEMM swapped in) is particularly smart -- it isolates exactly how much of the system speedup comes from the GEMM kernel vs. other system differences.

### 6. Why This Structure Works for SC

SC (Supercomputing) values:
- **Hardware awareness**: The paper is deeply grounded in H800 architecture (TMA, warp groups, SMEM banking, PTX instructions).
- **Quantitative modeling**: The cost model gives reviewers something concrete to evaluate.
- **Production relevance**: The paper explicitly states LiquidGEMM is deployed in ByteDance's production serving infrastructure.
- **Comprehensive evaluation**: 8 models, multiple baselines, both system-level and kernel-level benchmarks.

The paper reads like an engineering report backed by theory, which is exactly the SC audience's preference -- as opposed to, say, ICML which would want more focus on the quantization accuracy and less on PTX-level tricks.

### 7. What's Notably Absent (and Why)

- **No accuracy tables in the main paper** -- they promise a "full-version technical report" for detailed accuracy results. This is a deliberate tradeoff: SC reviewers care more about performance than accuracy, and the 12-page limit is tight. The authors use the space for hardware analysis instead.
- **No comparison with FP4/FP8 quantization schemes** beyond TRT-FP8 -- because the focus is specifically on the *integer* W4A8 path and its dequantization bottleneck; FP8 is a different hardware path entirely.
- **Minimal discussion of quantization-aware training** -- because LQQ is a PTQ method and the paper's contribution is the kernel, not the quantization algorithm per se.

---

## Q&A

### Q1: Which sections are related to the methodology?

**Sections 3, 4, and 5** together form the methodology, but they play distinct roles:

| Section | Role in Methodology |
|---------|-------------------|
| **Section 3 (Motivation)** | *Analytical methodology* -- develops the cost model (Eqs. 3-6) that formalizes $T = \max(T_{LD}, T_{DQ} + T_{MMA})$ and derives the design constraint $\alpha \leq 5.07$. This is methodology because it defines *what a solution must satisfy*, not just *what's wrong*. |
| **Section 4 (LiquidQuant)** | *Algorithm methodology* -- the quantization/dequantization scheme. Covers the rotation from INT8→UINT8→UINT4, the two's complement overflow-free proof, and the final IMAD+XOR hardware mapping. |
| **Section 5 (High Performance W4A8 GEMM Kernel)** | *Systems methodology* -- the kernel-level engineering. Covers ImFP pipeline design (Section 5.1), Dual-MMA packed memory layout (Section 5.2), hardware-efficient dequantization integration (Section 5.3), and other GEMM optimizations like persistent kernels and transposed computation (Section 5.4). |

Section 6 (LLM Serving System) is *integration*, not methodology -- it assembles existing components (FlashAttention, PagedAttention, SmoothQuant) around LiquidGEMM without proposing new techniques.

### Q2: What is the biggest contribution that obtains the largest claimed performance gain?

The **Implicit Fine-Grained Pipeline (ImFP)** from Section 5.1 is the single biggest contributor to the largest performance gains.

**Evidence from the ablation study (Figure 13):**
- LQQ alone gives up to **1.29x** speedup (only helps when compute-bound).
- ExCP (explicit pipeline) sometimes *hurts* at small batch sizes due to RF↔SMEM round-trip overhead and sync cost.
- ImFP gives the **full 2.90x** kernel speedup over QServe. It is the only technique that consistently improves performance across *all* batch sizes.

**Why ImFP dominates:** LQQ reduces $\alpha$ from ~12+ instructions (QServe's vadd-based path) to ~1.75 per element. But even with low $\alpha$, $T_{DQ}$ is still additive with $T_{MMA}$ unless they are *overlapped*. ImFP converts $T_{DQ} + T_{MMA}$ into $\max(T_{DQ}, T_{MMA})$ by having multiple Compute WGs dequantize and MMA on different fragments concurrently. This overlap is what enables the jump from 1.29x (LQQ-only) to 2.90x (full LiquidGEMM).

The **4.94x system-level** speedup (the largest number claimed) comes from the compound effect: LiquidGEMM's kernel speedup allows LiquidServe to scale to larger batch sizes (e.g., 184 on LLaMA2-70B vs. QServe's 64), which multiplies throughput. But the root enabler is ImFP.

### Q3: The delivery flow in the methodology sections and the linkage between them

The methodology flows as a **three-stage pipeline** where each stage's output is the next stage's input constraint:

```
Section 3 (Cost Model)
    │
    │  Output: Design constraints
    │  ├─ "α ≤ 5.07 instructions per element" (for T_DQ ≤ T_LD)
    │  └─ "T_LD, T_DQ, T_MMA must be overlapped, not summed"
    │
    ▼
Section 4 (LiquidQuant) ◄── satisfies constraint 1
    │
    │  Input:  Need α ≤ 5.07
    │  Method: INT8 → UINT8 rotation, UINT8 → UINT4 quantization,
    │          two's complement trick to stay in UINT8 domain
    │  Output: Dequantization = IMAD + XOR = 2 instructions per 4 elements
    │          (α ≈ 1.75 including unpacking, well below 5.07)
    │
    │  Linkage to Section 5: LQQ produces UINT4 weights with precomputed
    │  scale s_u8 and offset a = 2^7 + min(Q_i8). These are the operands
    │  that Section 5's dequantization logic consumes.
    │
    ▼
Section 5 (GEMM Kernel) ◄── satisfies constraint 2, consumes LQQ's output
    │
    ├─ 5.1 Pipeline Design
    │      Input:  Need to overlap T_LD, T_DQ, T_MMA
    │      Method: ImFP -- single Load WG (producer) + two Compute WGs (consumers)
    │              Each Compute WG does both dequant and MMA on its own fragment.
    │              Overlap is achieved *across* WGs: WG0 does MMA while WG1 dequants.
    │      Linkage: ImFP *requires* fast dequant from LQQ. If dequant were slow
    │              (α >> 5), it would dominate even with overlap, and ImFP would
    │              degrade to ExCP-like behavior.
    │
    ├─ 5.2 Memory Layout (Dual-MMA Packed)
    │      Input:  ImFP needs efficient SMEM→RF data loading for Compute WGs
    │      Method: Pack 2 MMAs' worth of UINT4 data contiguously per thread.
    │              One LDS.128 loads 32 UINT4 elements = both MMAs' data.
    │              1D layout eliminates bank conflicts and swizzling.
    │      Linkage: The packed layout is designed around LQQ's UINT4 format.
    │              The 32 elements map directly to the 4 registers (R0-R3)
    │              shown in Figure 8, which LQQ's dequant logic operates on.
    │
    ├─ 5.3 Hardware-Efficient Dequantization
    │      Input:  Registers R0-R3 loaded by 5.2, scale/offset from LQQ (Sec 4)
    │      Method: Unpack UINT4→UINT8 (AND + SHIFT), then apply Eq. 12:
    │              Q_i8 = (Q_u4 * s_u8 + a) XOR 0x80
    │              All via IMAD and XOR -- native 32-bit PTX instructions.
    │      Linkage: This is the *hardware realization* of LQQ's math (Sec 4).
    │              The unpacking step is co-designed with the Dual-MMA layout (5.2).
    │              The instruction count (7 instructions per 8 elements) is what
    │              makes ImFP (5.1) effective -- it's fast enough to hide behind MMA.
    │
    └─ 5.4 Other Optimizations
           Y = XW^T rewritten as Y = (WX^T)^T for better Tensor Core utilization
           at small batch sizes. Persistent kernels. These are standard techniques
           but necessary for the full system to work.
```

**The critical linkage chain:**

1. **Section 3 → Section 4**: The cost model says "reduce α." LQQ delivers α ≈ 1.75 (well below the 5.07 threshold). Without Section 3's analysis, the reader wouldn't know *how much* improvement is needed or *why* 2 instructions per 4 elements matters.

2. **Section 4 → Section 5.3**: LQQ's math (Eq. 12: multiply, add offset, XOR) maps directly to the IMAD+XOR instruction sequence in 5.3. The offline-precomputed values ($s_{u8}$ and $a$) are the glue -- they encode all the quantization parameters into two constants that the kernel consumes at runtime.

3. **Section 5.2 → Section 5.3**: The Dual-MMA packed layout determines *how* data lands in registers. The dequantization in 5.3 operates on exactly the 4-register layout (R0-R3 in Figure 8) that the packed load produces. If the layout were different, the unpack+dequant logic would need additional shuffle instructions.

4. **Section 5.3 → Section 5.1**: ImFP's effectiveness depends on dequant being fast. The pipeline's overlap only works because a Compute WG finishes dequant quickly enough that another WG's MMA can start before the next weight tile arrives. If dequant were 10x slower, the pipeline would stall regardless of structure.

5. **Section 3 → Section 5.1**: The cost model says "overlap, don't serialize." ImFP converts $T_{COMP} = T_{DQ} + T_{MMA}$ into effective $T_{COMP} \approx \max(T_{DQ}, T_{MMA})$ across WGs, which is exactly the prescription from Section 3's analysis.

In short: **Section 3 defines "what", Section 4 solves "how fast to dequant", and Section 5 solves "how to hide what remains."** Each section's output is tightly consumed by the next.

---

## Summary

The paper's structure is a masterclass in systems-paper writing for a hardware-focused venue. Every section serves a purpose in the deductive chain: **roofline says W4A8 should win -> profiling shows it doesn't -> cost model explains why (CUDA Core bottleneck) -> LQQ minimizes the bottleneck -> ImFP hides what remains -> experiments validate both the model and the solution**. The narrative never loses sight of the hardware, which is exactly what SC reviewers want to see.
