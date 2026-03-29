---
name: moe-kernel-experiments
description: How existing moe training framework conduct experiments.
---

# SonicMoE Experiments

## Symbol definitions

| Symbol | Meaning |
|---|---|
| T | Total tokens in the batch (= batch_size × seq_len) |
| d | Model hidden dimension (embedding size) |
| n | Expert intermediate size (FFN inner dim per expert) |
| E | Total number of experts |
| K | Number of experts activated per token |

Code mapping (`moe-cute.py --thiek`): T, H(=d), I(=n), E, K

## Key formulas

| Formula | Meaning |
|---|---|
| ρ = K / E | MoE activation ratio — fraction of experts each token uses |
| G = d / n | Expert granularity — higher G means more fine-grained (smaller) experts |
| P_layer = 3dnE | Total parameters in one MoE layer (SwiGLU: up-proj d×2n + down-proj n×d, times E experts) |
| FLOPs_fwd = 6TdnK | Forward FLOPs per layer (SwiGLU); each token goes through K experts |
| FLOPs_fwd+bwd = 18TdnK | Forward + backward FLOPs per layer (SwiGLU; 3× forward) |
| nK | Controls FLOPs per token per layer (iso-FLOPs ⟺ nK = const, since FLOPs ∝ TdnK and T,d fixed) |
| nE | Controls total params per layer (iso-params ⟺ nE = const, since P_layer = 3dnE and d fixed) |
| nE = nK / ρ | When ρ is held constant, iso-FLOPs (nK=const) automatically implies iso-params (nE=const) |

## Why "model size" appears in benchmark tables

Model size (1.4B, 7B, 30B, 120B) constrains MoE layer dimensions:
1. **d is fixed by model architecture.** (e.g. 1.4B→768, 7B→1536, 30B/120B→4096)
2. **nE is fixed by per-layer parameter budget.** (P_layer = 3dnE, so nE = P_layer / 3d)

| Model size | d | nE (fixed) |
|---|---|---|
| 1.4B | 768 | 32,768 |
| 7B | 1,536 | 32,768 |
| 30B | 4,096 | 65,536 |
| 120B | 4,096 | 131,072 |

---

## Experiment types by concept being varied

All MoE experiments in the paper vary one of three high-level concepts while holding others constant.

### 1. Varying Expert Granularity (G = d/n)

**Question:** Given a fixed compute and parameter budget, does splitting capacity into more smaller experts help?

**What changes:** n↓, E↑, K↑ (experts get smaller but more numerous)
**What is held constant:** nK (iso-FLOPs), nE (iso-params), ρ = K/E, d, T

Note: holding ρ constant while holding nK constant automatically gives nE constant (since nE = nK/ρ). So this sweep is simultaneously iso-FLOPs and iso-params.

**How to generate configs:** Pick a base (n₀, E₀, K₀), then scale n down by factor α while scaling E and K up by α:

| n | E | K | nK | nE | ρ | G = d/n |
|---|---|---|---|---|---|---|
| 1024 | 16 | 2 | 2048 | 16384 | 1/8 | 0.75 |
| 256 | 64 | 8 | 2048 | 16384 | 1/8 | 3 |
| 64 | 256 | 32 | 2048 | 16384 | 1/8 | 12 |

**Measured on two axes:**

- **Kernel throughput** (Figures 3, 13, 14; Table 9a, 9b): Benchmark a single MoE layer's forward/backward TFLOPS and activation memory. No training needed — just allocate weights and run timed forward/backward on one GPU.
- **Model quality** (Table 5): Train full models end-to-end, compare validation PPL and downstream task accuracy. Include two dense baselines:
  - Dense iso-FLOPs: FFN intermediate = nK (same compute, fewer params)
  - Dense iso-params: FFN intermediate = nE (same params, more compute)

### 2. Varying Sparsity (ρ = K/E)

**Question:** As we make the MoE sparser (each token uses a smaller fraction of experts), how does kernel efficiency degrade?

**What changes:** E↑ (more experts), so ρ = K/E↓
**What is held constant:** T, d, n, K (iso-FLOPs per token since FLOPs ∝ TdnK)

This is NOT iso-params: increasing E while fixing n means total parameters (3dnE) grow linearly with E. The point is to isolate the effect of sparsity on kernel throughput — more experts means more weight IO and more tile padding waste, even though each token does the same amount of compute.

**How to generate configs:** Fix (T, d, n, K), sweep E:

| E | K | ρ = K/E | nE (params) | nK (FLOPs) |
|---|---|---|---|---|
| 64 | 8 | 1/8 | 16384 | 2048 |
| 128 | 8 | 1/16 | 32768 | 2048 |
| 256 | 8 | 1/32 | 65536 | 2048 |
| 512 | 8 | 1/64 | 131072 | 2048 |

**Measured on:**

- **Kernel throughput** (Figure 16): TFLOPS drops as E increases due to (1) more expert weight IO, (2) more wasted FLOPs from tile padding in Grouped GEMM. This is where token rounding (TR) shows its advantage over vanilla TC routing — TR's throughput degrades more slowly.

### 3. Varying Routing Method

**Question:** Does the routing algorithm affect model quality and/or training throughput?

**What changes:** The routing algorithm only (TC top-K, EC, token rounding, etc.)
**What is held constant:** Everything else — same (d, n, E, K), same model architecture, same training budget

**Routing methods compared** (Table 3):
- **TC top-K**: Each token picks its top-K experts by score (standard)
- **EC (expert choice)**: Each expert picks its top tokens (load-balanced but breaks causality)
- **EC (aux router)**: EC with auxiliary router for inference
- **EC (ft TC router)**: EC-trained model finetuned with TC router
- **TC (token drop)**: TC top-K but drop tokens to align with tile size (always round down)
- **TR (token rounding)**: Round per-expert token counts to tile-size multiples (at most 1 tile deviation per expert)

**Measured on:**

- **Model quality** (Table 3): Train full models, compare PPL and downstream accuracy. Key finding: TR ≈ TC quality (near-lossless), EC has train-test gap.
- **Kernel throughput** (Figure 16): TR vs TC forward+backward TFLOPS. TR reduces wasted FLOPs from padding, advantage grows with sparsity.

---

## How to design a new experiment

1. **Decide which concept you are studying** (granularity, sparsity, or routing)
2. **Fix everything else:**
   - Granularity sweep → fix nK, ρ, d, T
   - Sparsity sweep → fix T, d, n, K
   - Routing sweep → fix all architecture params
3. **Decide what you are measuring:**
   - Kernel efficiency → single-layer benchmark, no training needed
   - Model quality → full model training + eval, needs dense baselines for granularity sweeps
4. **Generate configs** from the constraint equations above
