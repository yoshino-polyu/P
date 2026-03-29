# SonicMoE: Accelerating MoE with IO and Tile-aware Optimizations

Wentao Guo1 , Mayank Mishra2 , Xinle Cheng1 , Ion Stoica2 , and Tri Dao1,3 

1 Princeton University 2 University of California, Berkeley 3 Together AI 

Correspondence to: wg0420@princeton.edu, tri@tridao.me 

December 17, 2025 

# Abstract

Mixture of Experts (MoE) models have emerged as the de facto architecture for scaling up language models without significantly increasing the computational cost. Recent MoE models demonstrate a clear trend towards high expert granularity (smaller expert intermediate dimension) and higher sparsity (constant number of activated experts with higher number of total experts), which improve model quality per FLOP. However, fine-grained MoEs suffer from increased activation memory footprint and reduced hardware efficiency due to higher IO costs, while sparser MoEs suffer from wasted computations due to padding in Grouped GEMM kernels. In response, we propose a memory-efficient algorithm to compute the forward and backward passes of MoEs with minimal activation caching for the backward pass. We also design GPU kernels that overlap memory IO with computation benefiting all MoE architectures. Finally, we propose a novel “token rounding” method that minimizes the wasted compute due to padding in Grouped GEMM kernels. As a result, our method SonicMoE reduces activation memory by $45 \%$ and achieves a 1.86x compute throughput improvement on Hopper GPUs compared to ScatterMoE’s BF16 MoE kernel for a fine-grained 7B MoE. Concretely, SonicMoE on 64 H100s achieves a training throughput of 213 billion tokens per day comparable to ScatterMoE’s 225 billion tokens per day on 96 H100s for a 7B MoE model training with FSDP-2 using the lm-engine codebase1. Under high MoE sparsity settings, our tile-aware token rounding algorithm yields an additional 1.16x speedup on kernel execution time compared to vanilla top- $K$ routing while maintaining similar downstream performance. We open-source all our kernels2 to enable faster MoE model training. 

# 1 Introduction

Mixture of Experts (MoE) (Shazeer et al. 2017) models have emerged as a key technique for scaling up parameters (Kimi et al. 2025; Zhao et al. 2025a) without increasing the training computational requirements. Modern transformers often have layers comprised of a sequence mixer block (e.g. Multi-head Attention (Vaswani et al. 2017)), followed by a channel mixer block (e.g. dense MLPs) where MoEs are an excellent substitute for dense MLPs for FLOPs efficiency. A MoE block is typically composed of a token router and multiple smaller and often equal-sized subnetworks, called “experts”. MoEs can reduce FLOPs consumption during training by only activating a subset of all experts per token. However, reducing FLOPs does not directly translate to better hardware utilization since MoE computation features more dynamic IO accesses when each expert needs to gather token embeddings from different positions, and also scatter the results back to the original positions. Moreover, such hardware-unfriendliness becomes worse as experts become more granular (experts have smaller intermediate sizes) and sparser (experts are increased while keeping the number of activated experts constant), shown in Table 1. 

MoE scaling laws (Clark et al. 2022; Krajewski et al. 2024; Tian et al. 2025) predict better model quality per FLOP with increasing expert granularity (ratio between the model’s embedding dimension and each expert’s intermediate size) and sparsity. Recent MoE models like DeepSeek V3 (DeepSeek-AI et al. 2024), Qwen3 MoE (QwenLM 2025) and gpt-oss-120b (OpenAI 2025), have demonstrated superior performance of “fine-grained” MoEs over “coarse-grained” MoEs at scale. Besides granularity, the pursuit of MoEs with better model quality while keeping computational requirements constant has also led to modern MoEs becoming sparser. For example, Kimi K2 (Kimi et al. 2025) has the same amount of activated parameters as DeepSeek V3 (DeepSeek-AI et al. 2024) but much larger total parameter count. Overall, granularity and sparsity for MoEs have only increased over time as shown in Table 1. We also note that the pursuit of granularity and sparsity is also adopted by recent alternative architectures to MoE such as PEER (He 2024), Memory Layers (Berges et al. 2024), and 

Ultra-Mem (Huang et al. 2025). 

Though more granular and sparser MoEs increase model quality per FLOP, they suffer from hardware inefficiency due to: (1) larger activation memory footprint for granular MoE models as activation size typically scales linearly with the number of activated experts, (2) lower arithmetic intensity and increased IO cost due to granular experts and (3) wasted computations due to tile quantization effects of grouped GEMM for highly sparse MoEs. The high granularity and sparsity both push MoE training towards the memory-bound regime requiring carefully designed MoE kernels to hide the increased IO costs. Existing state-of-the-art MoE kernels such as ScatterMoE (Tan et al. 2024) and MoMoE (Costin et al. 2025) are not designed to handle these high IO costs and they suffer significant training throughput degradation. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-26/0390e304-b35b-4a5f-9ad3-e0d35edf7b5c/364163b5a38bea7fe5dbe509d422e3d8b41118cb2cd88ae1559e2490312ca59f.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-26/0390e304-b35b-4a5f-9ad3-e0d35edf7b5c/31101d8e3898ddd72dd24923c8d9d21148fe242e305464a7a00f6b01a804513e.jpg)

Figure 1: SonicMoE’s per-layer activation memory footprint (left) stays constant even when expert granularity $( d / n$ where $d$ is the embedding dimension and $n$ is the expert intermediate dimension) increases, and is $0 . 2 0 – 1 . 5 9 \mathrm { x }$ more memory-efficient than other baselines. SonicMoE’s forward computation throughput (right) reaches an average of $8 8 \%$ (max $91 \%$ , min $86 \%$ ) of the upper bound (cuBLAS BMM $^ +$ activation $^ +$ cuBLAS BMM $^ +$ aggregation on H100). Note that the cuBLAS upper bound baseline does not include the router computation. Here we use a 30B MoE configuration with microbatch size of 32768 tokens, and we vary the activated experts / total number of experts as 2/32, 4/64, 8/128, and 16/256 from left to right.


We propose to co-design the MoE architecture with a GPU kernel tailored to NVIDIA Blackwell and Hopper generation GPUs and a novel routing method. (1) We derive an algorithm to compute the MoE backward pass more efficiently leading to a much smaller activation memory footprint that does not increase with increasing expert granularity. (2) We leverage new hardware features on Blackwell and Hopper GPUs to overlap memory IO with computation which can benefit all MoEs, and, in particular, fine-grained MoEs. (3) We propose a hardware-aware token rounding routing method where the routed number of tokens to an expert is always a multiple of the GEMM tile size. Using extensive experiments, we show that token rounding routing is $16 \%$ faster than the baseline token-choice routing when we scale up the number of experts 4 times from a 30B MoE. We also validate that TR preserves the MoE inference quality on 2B parameter scale. With (1) and (2), we can increase the end-to-end training throughput of a 7B MoE model by $50 \%$ (without changing the top- $K$ token choice routing). Our token rounding routing method further improves training throughput by $16 \%$ when we scale up the number of experts without any accuracy loss. 

Summary of contributions. We propose SonicMoE, a hardware and model architecture co-design solution to address MoE training efficiency problems, making the following contributions: 

• MoE training with minimum possible activation memory footprint without increasing FLOPs: We analyze the impact of MoE granularity on the MoE layer’s forward and backward passes and observe that increasing MoE granularity while maintaining constant FLOPs leads to a linear increase in activation memory required by the backward pass. Leveraging this observation, we carefully redesign the computation graph to avoid caching the activations for the router gradient computation while maintaining the mathematical equivalence to the original MoE formulation. As a result, for a fine-grained 7B MoE, SonicMoE reduces activation memory usage per layer by up to $45 \%$ . 

• Efficient MoE kernel that overlaps IO with computation to yield SOTA training throughput: We show that increasing both granularity and sparsity leads to MoEs becoming increasingly memory bandwidth bound. To alleviate this bottleneck, we exploit the asynchrony of the GEMM and IO operations by overlapping them to maximize throughput. For the same fine-grained 7B MoE model, our approach increases relative speedup by $43 \%$ on the forward pass compared to a highly optimized DeepGEMM baseline, and by $83 \%$ and $11 5 \%$ on the backward pass compared to the state-of-the-art MoE baselines ScatterMoE and MoMoE, respectively. To evaluate the performance of these techniques, we conduct an extensive performance analysis through comprehensive kernel-level profiling and an IO-aware exploration of the MoE computational paths. 

• Token rounding routing that eliminates wasted FLOPs from sparse MoEs: We introduce a drop-in routing algorithm that rounds the per-expert token counts to multiples of the tile size (e.g., 128) used by grouped GEMM in MoE kernels. This rounding reduces compute wasted on padding while preserving the original token-to-expert 

assignment as much as possible. The algorithm ensures that, for each expert, the maximum deviation from the original top- $K$ token-choice result is bounded by one tile. This method effectively eliminates padding waste in grouped GEMM while maintaining the same total number of tokens in expectation, and it delivers robust token-choice accuracy even under highly sparse MoE training regimes. We validate the performance of this token-rounding strategy in a 1.4B-parameter sparse training setting, demonstrating that its compute throughput consistently exceeds that of the vanilla top- $K$ token-choice routing. In highly sparse regimes, the improvement reaches up to $16 \%$ higher TFLOPS for end-to-end MoE computation. 

We release SonicMoE, mainly written in CuTe-DSL (NVIDIA 2025c) with a PyTorch interface, with a permissive license to benefit researchers and practitioners. The GitHub link is https://github.com/Dao-AILab/sonic-moe. 


Table 1: MoE Scaling Trends: Here, we show the activation ratio as experts activated per token $K$ / total experts $E$ and expert granularity is shown as model embedding dimension (d) / expert intermediate size (n) for frontier open source models. We do not include the shared experts for the MoE sparsity calculation. The trend indicates new open-source MoE models tend to be more granular and sparser.


<table><tr><td>Model</td><td>Release date</td><td>Parameters</td><td>Expert activation ratio (K/E)</td><td>Expert granularity (d/n)</td></tr><tr><td>Mixtral 8x22B (Mistral 2024)</td><td>11/23</td><td>131B</td><td>25.0% (2/8)</td><td>6144/16384 = 0.38</td></tr><tr><td>DBRX (The Mosaic Research Team 2024)</td><td>03/24</td><td>132B</td><td>25.0% (4/16)</td><td>6144/10752 = 0.57</td></tr><tr><td>Phi-3.5-MoE (Microsoft 2024)</td><td>09/24</td><td>42B</td><td>12.5% (2/16)</td><td>4096/6400 = 0.64</td></tr><tr><td>OLMoE (Muennighoff et al. 2025)</td><td>09/24</td><td>7B</td><td>12.5% (8/64)</td><td>2048/1024 = 2.00</td></tr><tr><td>Granite 3.1-MoE (Granite 2024)</td><td>12/24</td><td>3B</td><td>20.0% (8/40)</td><td>1536/512 = 3.00</td></tr><tr><td>DeepSeek-V3 (DeepSeek-AI et al. 2024)</td><td>12/24</td><td>671B</td><td>3.13% (8/256)</td><td>7168/2048 = 3.50</td></tr><tr><td>Qwen3 MoE (QwenLM 2025)</td><td>04/25</td><td>235B</td><td>6.25% (8/128)</td><td>2048/1536 = 1.33</td></tr><tr><td>QWen3-30B-A3B (Qwen 2025)</td><td>05/25</td><td>30.5B</td><td>6.25% (8/128)</td><td>2048/768 = 2.67</td></tr><tr><td>Kimi K2 (Kimi et al. 2025)</td><td>07/25</td><td>1.04T</td><td>2.08% (8/384)</td><td>7168/2048 = 3.50</td></tr><tr><td>gpt-oss-120b (OpenAI 2025)</td><td>08/25</td><td>120B</td><td>3.13% (4/128)</td><td>2880/2880 = 1.00</td></tr><tr><td>GLM-4.5-Air (Zeng et al. 2025)</td><td>08/25</td><td>106B</td><td>6.25% (8/128)</td><td>4096/1408 = 2.91</td></tr><tr><td>Qwen3-Next-80B-A3B-Instruct (Qwen 2025)</td><td>09/25</td><td>81B</td><td>1.95% (10/512)</td><td>2048/512 = 4.00</td></tr><tr><td>DeepSeek-V3.2-Exp (DeepSeek-AI 2025)</td><td>10/25</td><td>685B</td><td>3.13% (8/256)</td><td>7168/2048 = 3.50</td></tr></table>

# 2 Background

We first provide an overview of the MoE architecture and a standard MoE kernel employing grouped GEMM in Section 2.1. In Section 2.2, we discuss how granularity and MoE sparsity will affect MoE’s training efficiency. We then examine the impact of MoE routing method on the MoE model quality and training efficiency in Section 2.3. 

# 2.1 MoE using Grouped GEMM

Modern GPUs support Tensor Cores; specialized hardware units with high matrix multiplication throughput (NVIDIA 2022). A GEMM (general matrix multiply) (Lawson et al. 1979) kernel often has 3 stages: prologue (start input loading), mainloop (keep loading inputs and compute GEMM) and epilogue (miscellaneous IO/math operations on GEMM outputs). The kernel tiles computations (dividing large matrices into small tiles), and optionally pads dimensions so computation aligns with hardware-permissible tile sizes. In this paper, we follow standard GEMM notations in most BLAS (Lawson et al. 1979) libraries: we have $A \in \mathbb { R } ^ { \mathbf { M } \times \mathbf { K } }$ , $B \in \mathbb { R } ^ { \bar { \mathbf { K } } \times \bar { \mathbf { N } } }$ , $C \in \mathbb { R } ^ { \mathbf { M } \times \mathbf { N } }$ for $C = A B$ with problem shape $( \mathbf { M } , \mathbf { N } , \mathbf { K } )$ . This notation is adopted by CUTLASS (NVIDIA 2025a) which implements efficient GEMM on CUDA. 

On NVIDIA Hopper GPUs, GEMM is performed asynchronously with a producer-consumer paradigm (Shah et al. 2024) where producers are dedicated to load a tile of data from High Bandwidth Memory (HBM), or global memory (GMEM) logically, to shared memory (SMEM) while consumer warpgroups3 are responsible for GEMM computation (Shah et al. 2024). In prologue and mainloop, producer warpgroups fetch a tile of data and cache to a dedicated pipeline while the consumer warpgroups read from the cached tile from this pipeline, perform tiled matrix multiply (MMA) and accumulate over the K dimension of GEMM. After the mainloop, we enter the epilogue stage where the consumer warpgroups apply post-processing (activation function and write results back to HBM) on the final MMA results. 

A MoE block is typically composed of a token router and multiple smaller and often equal-sized subnetworks, called “experts”. The router is responsible for dispatching tokens to the experts which are subsequently used by the specific expert for 


Algorithm 1 MoE forward with Grouped GEMM


Input: $X \in \mathbb{R}^{T \times d}$ , $W_1 = \{W_{1,e}\}_{e \in [E]} \in \mathbb{R}^{d \times 2n}$ , $W_2 = \{W_{2,e}\}_{e \in [E]} \in \mathbb{R}^{n \times d}$ , routing scores $S \in \mathbb{R}^{T \times E}$ , $\pi \in \{0,1\}^{T \times E}$ as a binary-valued mask matrix where $\pi_{t,e}$ represents whether token $t$ is routed to expert $e$ . 


Output :output activation $O \in \mathbb { R } ^ { T \times d }$


Parallel for $e\in [E]$ do //up-proj $X_{e}\gets$ Gather $(X,\pi_{;,e})$ //varlen-M Grouped GEMM $H_{e}\leftarrow X_{e}W_{1,e}$ //applyactivationfunction,e.g.SwiGLU $A_{e}\gets$ act_func $(H_{e})$ //down-proj,varlen-MGroupedGEMM $Y_{e}\gets A_{e}W_{2,e}$ 


Parallel for $t \in [ T ]$ do


// expert aggregation $O_{t} = \sum_{e\in [E]}\pi_{t,e}S_{t,e}Y_{e,t}$ 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-26/0390e304-b35b-4a5f-9ad3-e0d35edf7b5c/0528aeeb06dca0d559ca7fccc05fa8c09da98763d7c2690b5e6036b26c182cd2.jpg)



Figure 2: MoE computation often requires a Grouped GEMM. Each expert gathers inputs from different positions on an input tensor (top) or reads a contiguous chunk on a grouped input array (bottom). This figure is adapted from Tan et al. (2024)’s Figure 2.


computation. The outputs from all experts in the layer are then aggregated and passed onto the next layer. MoE computation4 can be performed using Grouped GEMM (a list of GEMMs with possibly different $\{ { \bf M } , { \bf N } , { \bf K } \}$ dimensions). Algorithm 1 illustrates running MoE forward with Grouped GEMM. 

As shown in Algorithm 1, during the forward pass (and backward activation gradient computation), we have variable number of tokens routed to every expert. A Grouped GEMM operation with fixed $( \mathbf { N } , \mathbf { K } )$ dim (as the expert weight matrix) but variable M (token dim) is then performed. We refer to this Grouped GEMM as “varlen-M Grouped GEMM”. During the backward weight gradient computation, the embedding dimension (M for backward) and intermediate hidden size (N for backward) are constant and instead, we reduce over the token dimension $( \mathbf { K } )$ , which we refer to as “varlen-K Grouped GEMM”. For each Grouped GEMM, we often have inputs gathered from different positions or contiguously-packed, as illustrated in Figure 2. For example in Algorithm 1, the inputs to up-proj are gathered while the inputs to down-proj are already contiguously-packed. 

# 2.2 MoE computation

Arithmetic intensity, defined as the ratio of FLOPs over the number of transferred bytes (IO), is a metric to quantify whether a kernel is memory-bound (kernel runtime dominated by memory IO cost) or compute-bound (kernel runtime dominated by compute throughput). 

The standard MoE computation for an expert e with SwiGLU activation can be broken down into following components: 

$$
H _ {e} = \operatorname {u p - p r o j e c t i o n} \left(X _ {e}\right) = X _ {e} W _ {1, e}: \mathbb {R} ^ {T _ {e} \times d} \rightarrow \mathbb {R} ^ {T _ {e} \times 2 n} \tag {1}
$$

$$
A _ {e} = \operatorname {S w i G L U} \left(H _ {e}\right): \mathbb {R} ^ {T _ {e} \times 2 n} \rightarrow \mathbb {R} ^ {T _ {e} \times n} \tag {2}
$$

$$
Y _ {e} = \text {d o w n - p r o j e c t i o n} \left(A _ {e}\right) = A _ {e} W _ {2, e}: \mathbb {R} ^ {T _ {e} \times n} \rightarrow \mathbb {R} ^ {T _ {e} \times d} \tag {3}
$$

where $X _ { e } \in \mathbb { R } ^ { T _ { e } \times d }$ denotes the input received by expert $e$ 

Here, the up-projection uses $2 T _ { e } \cdot 2 n \cdot d$ FLOPs and $2 T _ { e } d + 2 \cdot 2 n \cdot d + 2 T _ { e } n$ HBM memory transfer bytes (we ignore the writes for $H _ { e }$ here). Similarly, down-projection uses $2 T _ { e } n d$ FLOPs with $2 T _ { e } n + 2 n d + 2 T _ { e } d$ bytes. Assuming $\textstyle \rho = { \frac { K } { E } }$ as the MoE activation ratio, $\begin{array} { r } { G = { \frac { d } { n } } } \end{array}$ as the granularity and uniform routing i.e $T _ { e } = T \rho$ , the arithmetic intensity (ignoring the writes for $H _ { e }$ ) for the forward pass of an expert is 

$$
\frac {2 T _ {e} \cdot 2 n \cdot d + 2 T _ {e} n d}{4 T _ {e} n + 6 n d + 4 T _ {e} d} = \frac {3}{\frac {2}{d} + \frac {2}{n} + \frac {3}{T _ {e}}} = \frac {3}{\frac {2 + 2 G}{d} + \frac {3}{T \rho}} \tag {4}
$$

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-26/0390e304-b35b-4a5f-9ad3-e0d35edf7b5c/6c01ca4a8a24f90127cafa04f76e4615e3ff4f5b7f6eea6aed05242a94589aa2.jpg)



Figure 3: IO cost of MoE’s forward pass for one layer w.r.t. expert granularity across MoE configurations under iso-FLOPs training from 1.4B to 120B (configurations in Table 9a). We keep MoE activation ratio $\rho = K / E$ and the number of parameters of each MoE layer $3 d n E$ constant. When we scale up expert granularity $d / n$ , we scale down expert intermediate size $n$ while keeping both $n E$ and $n K$ constant.


For a specific model size (constant $d$ ), it can be seen that increasing granularity (increasing $G$ ) or increasing sparsity (decreasing $\rho \mathrm { \hbar }$ ) leads to a decreasing arithmetic intensity. This is caused by the linear scaling of IO cost w.r.t. expert granularity, as illustrated in Figure 3. Therefore for the case of fine-grained MoEs $( \mathrm { h i g h ~ } G ) ^ { 5 }$ , it becomes increasingly important to address the increased IO cost by maximally reducing IO access and hiding IO latency. We examine a memoryefficient MoE kernel design in Section 3 and discuss techniques to reduce IO access and latency in Section 4. 

Existing MoE kernel designs. There are multiple MoE implementations available: ScatterMoE (Tan et al. 2024), MoMoE (Costin et al. 2025), MegaBlocks (Gale et al. 2023), and Megatron (Shoeybi et al. 2019). However, they do not specialize for the setting of fine-grained MoEs that have linearly-increasing IO cost w.r.t. increasing expert granularity illustrated in Figure 3. In contrast, our kernel design, SonicMoE, minimizes the impact of IO cost on the training throughput. In Section 4 and Figure 14, we show that when expert granularity $G$ increases, SonicMoE demonstrates a greater relative speedup over existing MoE kernel designs due to the IO-aware optimizations. We elaborate on the technical differences between SonicMoE and prior MoE kernels in Appendix B and include an overview in Table 2. 

# 2.3 MoE routing methods

In MoE, routing determines which experts to activate for each token. Token choice (TC) routing where each token independently selects the activated expert is often the default routing method for MoE models (Shazeer et al. 2017). We often have top- $K$ TC routing where the routing decision for token $t$ is $\mathrm { T o p K } _ { e \in [ E ] } ( S _ { t , e } , K )$ and $S _ { t , e }$ is the expert score for token $t$ . Besides top- $K$ , Huang et al. (2024) introduce token-choice top- $P$ routing to flexibly allocate compute during training. However, it introduces nondeterminism in the number of activated experts and consumed FLOPs per token. Zeng et al. (2024b) also propose a similar idea that uses “null experts” to dynamically adjust the number of activated experts. 

Besides TC routing, expert choice (EC) routing is developed to avoid load imbalance for expert parallelism (Zhou et al. 2022) by letting experts choose the tokens. However, EC routing is not directly usable for inference because it is incompatible with autoregressive decoding, and switching back to TC at inference time leads to a mismatch. In addition, EC breaks causality by future token information leakage (Wang et al. 2024). To address the inference issue of EC routing, Raposo et al. (2024) introduce an auxiliary loss to promote the agreement between TC and EC routing results, or train an auxiliary router to explicitly predict the routing result of EC router and use this auxiliary router during inference. 

In this paper, we propose a novel Grouped GEMM tile-aware token rounding method that rounds the number of received tokens per expert (“expert frequency”) to nearby multiples of Grouped GEMM tile sizes and alters at most one tile of tokens per expert. This approach effectively reduces wasted FLOPs caused by Grouped GEMM padding during sparse MoE training while preserving inference quality of trained MoE models. There are similar works that propose to drop and reroute tokens, including Rectify-Router (Zeng et al. 2024a), but they do not focus on the tile structure of Grouped GEMM. Other works such as TMA-adaptive FP8 Grouped GEMM (Fu et al. 2025) focus on reducing padding-related load traffic but the FLOPs wasted by non-aligned tile size in GEMM computation is not addressed. 

# 3 Memory-efficient MoE algorithm

We first describe SonicMoE’s high-level kernel design in Section 3.1 that illustrates SonicMoE’s MoE computation4 as shown in Algorithm 2, 3, and 5. We then focus on activation memory usage of SonicMoE in Section 3.2. 

# 3.1 Overview of SonicMoE’s MoE kernels

The MoE computation4 in SonicMoE launches 8 kernels: during forward, we have up-proj $( A )$ , down-proj $( Y )$ , and expert aggregation $( O )$ kernels; during backward, we have activation gradient kernels for $d H$ (down-proj), $d \tilde { X }$ (up-proj), $d X$ (aggregating $d \tilde { X }$ across experts), and weight gradient kernels $d W _ { 1 }$ and $d W _ { 2 }$ . Figure 4 illustrates the computational workflow of these 8 kernels. We provide an efficient TC top- $K$ router, and an interface that accepts arbitrary routing input. However, it should be noted that SonicMoE’s MoE computation is independent of the MoE router choice and is thus compatible with arbitrary router logic. 

The implementation of SonicMoE’s MoE computation is highly modularized: it only consists of (1) optimized grouped GEMM kernel with modularized fusion (2) optimized expert aggregation kernel. The host dispatches to the best GEMM config and load/store strategies to launch the 8 kernels listed above. Besides such high modularity, SonicMoE still exhibits state-of-the-art training throughput and minimum activation memory usage which we describe below. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-26/0390e304-b35b-4a5f-9ad3-e0d35edf7b5c/957c983dd2679d4b8a5b385ed4389380b4c54d9e4708f827e84cbfbac58b7be4.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-26/0390e304-b35b-4a5f-9ad3-e0d35edf7b5c/30068cb5d2808819a82e16d1efd9d44588983dce5be540f3b56e24c7eb6c710b.jpg)



Figure 4: Computational workflow of SonicMoE’s 8 launched kernels, grouped by yellow boxes. 3 and 5 kernels are launched during forward and backward computation respectively. The incoming arrows to a yellow circle indicate a variable loaded from HBM to SRAM, and an outgoing arrow represents a variable stored to HBM. We color the boxes of all variables on HBM, with purple boxes indicating the output of forward and backward while blue boxes indicate intermediate variables or weights $( W _ { 1 } , W _ { 2 } )$ . We color all cached activations $X$ , $H , \pi , S$ in red. Algorithm 2 formally describes SonicMoE’s forward pass, and Algorithm 3 and 5 describe the backward pass.


# 3.2 Activation memory efficiency

The FLOPs of MoE forward and backward computation is $( 6 + 1 2 ) T n K d$ . For a given $T , d ^ { 6 }$ , we need to keep $n K$ constant for constant FLOPs. Therefore, increasing granularity requires decreasing $n$ and proportionally increasing $K$ . Hence, any activations with memory $O ( T K d )$ should not be cached for backward computation to avoid activation memory scaling with granularity. For current MoE kernels like ScatterMoE, activations scale linearly with expert granularity. Activations $Y$ (down-proj output) and $X _ { e }$ (gathered $X$ ) have size $T K d$ and avoiding caching them eliminates activation memory dependency on granularity. We avoid writing $d Y$ (gradient for $Y$ ) and $d O _ { e }$ (gathered $d O$ ) to HBM as they increase the peak activation memory during the backward computation: 

• For $X$ and $d O$ , fusing the gather operation with the HBM load eliminates the need for materialization and activation caching in HBM. We show in Figures 6 and 20 that this gather fusion significantly improves the throughput for fine-grained MoEs. 

• A naive implementation to compute $d S$ and $d H$ would need $Y$ and $d Y$ . Instead, we identify an alternative computation path to compute $d S$ and $d H$ without increasing FLOPs. This is achieved via expanding $d S$ and $d H$ into an equation that does not involve using $Y$ and $d Y$ , as illustrated in Appendix C. SonicMoE’s $d H$ kernel is shown in Algorithm 3. 

As a result, we only cache $X$ and $H$ along with routing metadata for a total size $2 T d + 4 T K n ^ { 7 }$ bytes per layer. This activation memory usage is the same as a dense model with same number of activated parameters, which is the minimum activation memory required for backward computation without doing activation recomputation with GEMM.89 In Figure 13, we profile SonicMoE’s activation memory for a 7B MoE training configuration and demonstrate that the activation memory of SonicMoE is independent of expert granularity. More results from 1.4B to 120B are included in Figure 13. 

Algorithm 2 SonicMoE’s MoE kernel forward pass. Variables stored in HBM are colored blue. load and store means load from / store into HBM respectively. 

Input : X, S, π, $W _ { 1 }$ , $W _ { 2 }$ same as Algorithm 1. 

Output :MoE layer output $O$ 

Up-proj A kernel (X, W1, π) → (H, A): 

// Gather + varlen-M Grouped GEMM + SwiGLU 

Parallel for $e \in [ E ]$ do 

$$
X _ {e}, W _ {1, e}, \pi_ {:, e} \leftarrow \operatorname {l o a d} \left(X _ {e}, W _ {1, e}, \pi_ {:, e}\right)
$$

$$
X _ {e} \leftarrow \operatorname {G a t h e r} (X, \pi_ {:, e})
$$

$$
H _ {e} \leftarrow X _ {e} W _ {1, e}
$$

// apply activation function, e.g. SwiGLU 

$$
A _ {e} \leftarrow \operatorname {a c t \_ f u n c} (H _ {e})
$$

$$
H _ {e}, A _ {e} \gets \operatorname {s t o r e} \left(H _ {e}, A _ {e}\right)
$$

Down-proj Y kernel $( A , W _ { 2 } )  Y$ : 

// varlen-M Grouped GEMM 

Parallel for $e \in [ E ]$ do 

$$
A _ {e}, W _ {2, e} \leftarrow \operatorname {l o a d} \left(A _ {e}, W _ {2, e}\right)
$$

$$
Y _ {e} \leftarrow A _ {e} W _ {2, e}
$$

$$
Y _ {e} \leftarrow \operatorname {s t o r e} \left(Y _ {e}\right)
$$

Expert aggregation O Kernel $( Y , S , \pi ) \to O$ : 

// Gather and sum 

Parallel for $t \in [ T ]$ do 

$$
Y _ {e, t}, S _ {t, e}, \pi_ {t, e} \leftarrow \operatorname {l o a d} \left(Y _ {e, t}, S _ {t, e}, \pi_ {t, e}\right)
$$

$$
O _ {t} \leftarrow \sum_ {e \in [ E ]} \pi_ {t, e} S _ {t, e} Y _ {e, t}
$$

$$
O _ {t} \leftarrow \operatorname {s t o r e} \left(\dot {O} _ {t}\right)
$$

Algorithm 3 SonicMoE’s MoE kernel backward pass of down projection. 

Input :S, π, $W _ { 2 }$ , dO. 

Output :dH, $d W _ { 2 }$ , dS. 

Down-proj act dH kernel (dO, W2, S, π) → (dH, dS, A′): 

// Gather + varlen-M Grouped GEMM + dSwiGLU + dS 

// Appendix C elaborates this algorithm in more detail 

Parallel for $e \in [ E ]$ do 

$$
d O _ {e}, W _ {2, e}, S, \pi_ {:, e} \leftarrow \operatorname {l o a d} \left(d O _ {e}, W _ {2, e}, S, \pi_ {:, e}\right)
$$

$$
d O _ {e} \leftarrow \operatorname {G a t h e r} (d O, \pi_ {:, e})
$$

// $d A ^ { \prime }$ is a temp variable for computing dA, $d S _ { \mathrm { \Omega } }$ and $A ^ { \prime }$ 

$$
d A _ {e} ^ {\prime} \leftarrow d O _ {e} W _ {2, e} ^ {\top} \quad / / d A _ {e} ^ {\prime} \in \mathbb {R} ^ {T _ {e} \times n}
$$

$$
\mathbf {s} _ {e} \leftarrow \operatorname {G a t h e r} (S, \pi_ {:; e})
$$

$$
d A _ {e} \gets \mathrm {B r o a d c a s t} (\mathbf {s} _ {e}) d A _ {e} ^ {\prime}
$$

// compute fwd act and bwd act grad simultaneously in dAct call 

$$
A _ {e}, d H _ {e} \leftarrow \mathrm {d A c t \_ f u n c} (d A _ {e}, H _ {e})
$$

$$
A _ {e} ^ {\prime} \leftarrow \operatorname {B r o a d c a s t} (\mathbf {s} _ {e}) A _ {e} / / A ^ {\prime} \in \mathbb {R} ^ {T _ {e} \times n}, \text {i n p u t f o r} d W _ {2}
$$

$$
d S _ {e, t} \leftarrow \langle d A _ {e, t} ^ {\prime}, A _ {e, t} \rangle \quad / / r e d u c e o v e r n d i m
$$

$$
d H _ {e}, d S, A _ {e} ^ {\prime} \leftarrow \operatorname {s t o r e} \left(d H _ {e}, d S, A _ {e} ^ {\prime}\right)
$$

Down-proj weight $\cdot$ kernel $( d O , \ A ^ { \prime } , \ \pi )  d W _ { 2 }$ 

// Gather + varlen-K Grouped GEMM 

Parallel for $e \in [ E ]$ do 

$$
d O _ {e}, A _ {e} ^ {\prime}, \pi_ {:, e} \leftarrow \operatorname {l o a d} \left(d O _ {e}, A _ {e} ^ {\prime}, \pi_ {:, e}\right)
$$

$$
d O _ {e} \leftarrow \operatorname {G a t h e r} (d O, \pi_ {:; e})
$$

$$
d W _ {2, e} \leftarrow A _ {e} ^ {\top} d O _ {e} ^ {\top}
$$

$$
d W _ {2, e} \leftarrow \operatorname {s t o r e} \left(d W _ {2, e}\right)
$$

# 4 IO-aware kernel design

The expressivity of fine-grained MoE comes from the diversity of every token’s expert selection, which in turn leads to linearly-scaled IO cost w.r.t. expert granularity (Figure 3). To sustain high throughput, we need to maximally (1) reduce IO access via fusion (2) overlap the IO latency with compute. We first examine the token gather fusion with computation, and math and IO fusion with epilogue in Section 4.1.1 and 4.1.2 respectively. We then describe the techniques to overlap MMA with IO in Section 4.2. We examine SonicMoE’s top- $K$ sorting kernel in Section 4.3 respectively. In Appendix B, we compare SonicMoE with other MoE kernel designs with a summary in Table 2. 

<table><tr><td>Features\Methods</td><td>SonicMoE</td><td>ScatterMoE</td><td>MoMoE</td><td>MegaBlocks</td><td>Megatron</td><td>DeepGEMM</td></tr><tr><td>Gather fused with GMEM-to-SMEM (HBM) load (Sec. 4.1.1)</td><td>✓</td><td>fwd✓, bwd✗</td><td>fwd✓, bwd✗</td><td>✗</td><td>✗</td><td>✗</td></tr><tr><td>SwiGLU and dSwiGLU fused with epilogue (Sec. 4.1.2)</td><td>✓</td><td>✗</td><td>✓</td><td>✗</td><td>✓</td><td>NA</td></tr><tr><td>dS computed as ⟨dA′e,t, Ae,t⟩ (Sec. 4.1.2, App. C.1)</td><td>✓</td><td>✗</td><td>✗</td><td>✗</td><td>✓</td><td>NA</td></tr><tr><td>Backward epilogue that computes dH, dS together (Sec. 4.1.2)</td><td>✓</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td><td>NA</td></tr><tr><td>Overlap MMA with epilogue/IO (Sec. 4.2)</td><td>✓</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td></tr><tr><td>Do not need a separate scatter kernel</td><td>✓</td><td>✓</td><td>✓</td><td>✗</td><td>NA</td><td>NA</td></tr><tr><td>Efficient top-K sorting (Sec. 4.3)</td><td>✓</td><td>✗</td><td>✗</td><td>✗</td><td>✗</td><td>NA</td></tr><tr><td>Do not need shape-alignment efforts outside GEMM kernels</td><td>✓</td><td>✓</td><td>✓</td><td>✗</td><td>✗</td><td>✗</td></tr></table>


Table 2: Comparison between SonicMoE and prior MoE kernels. $\checkmark$ means that the kernel implements the feature or a functionality similar in semantics, and $x$ means the feature is missing from the kernel. “NA” means that the feature is out of the expected scope. We use the GroupedMLP for Megatron and ParallelDroplessMLP for MegaBlocks. More discussion is included in Appendix B.


# 4.1 SonicMoE’s Grouped GEMM

SonicMoE is built on top of an efficient varlen-M and varlen-K Grouped GEMM. Inside the Grouped GEMM, we fuse the gather operations with the activation loading (4.1.1), and fuse SwiGLU/dSwiGLU/dS with epilogue (4.1.2). The gather fusion helps SonicMoE to be faster than MoE kernel designs that require a separate gather kernel such as MegaBlocks, Megatron, and DeepGEMM++, an optimized MoE forward pass implementation built on top of DeepGEMM library (Zhao et al. 2025b). The epilogue fusion boosts SonicMoE to be faster than ScatterMoE in the backward pass. These fusions reduce unnecessary IO access and can be overlapped with compute MMA, as we discuss in Section 4.2. 

# 4.1.1 Gather fusion with HBM load

SonicMoE’s Grouped GEMM accepts either contiguously-packed inputs or inputs gathered from different positions, illustrated in Figure 2. For the latter case, we fuse the input gather with the input loads from global memory (GMEM, often the HBM) to shared memory (SMEM) so we can batch them to perform GEMM on Tensor Core (Costin et al. 2025; Tan et al. 2024). This involves (1) fetching the routed token indices for each expert and then (2) using these indices to gather activations via Blackwell and Hopper’s cp.async instruction. We often have no better alternatives for the second step10, but synchronous index fetching is still optimizable by prefetching and cooperative fetching among producer warps. We illustrate our strategies in Figure 18. 

Hopper GPUs. As shown in Figure 6, Gather fusion provides SonicMoE with a major advantage over existing MoE kernel designs on H100 such as DeepGEMM. Although DeepGEMM’s varlen-M Grouped GEMM kernel is highly optimized11, DeepGEMM assumes the inputs are already contiguously packed and padded to multiples of 128, which requires a separate kernel launch for gather and pad (either all2all in expert parallelism, or a gather kernel on a single GPU in our case) before the Grouped GEMM. In Figure 6, even though we can provide an optimized gather kernel and DeepGEMM’s varlen-M Grouped GEMM is also highly-optimized, the large amount of IO to gather $X$ (2T Kd bytes) still makes DeepGEMM++ (for definition of DeepGEMM++, refer to Figure 6) slower than SonicMoE. 

In the backward pass, weight gradients for up-proj and down-proj ( $d W _ { 1 }$ and $d W _ { 2 }$ respectively) need to gather $X$ and $d O$ and the activation gradient for $d H$ also needs to gather dO. Despite the backward having more kernels requiring the gather operation, existing approaches including ScatterMoE (Tan et al. $2 0 2 4 ) ^ { 1 2 }$ and MoMoE (Costin et al. 2025)13 fuse the gather during forward but still launch a separate gather kernel during backward. Fusing this gather reduces the IO cost by $2 T K d$ bytes and cuts down a major portion of fine-grained MoE training time. For example in Figure 6, the 2 gathers in backward of ScatterMoE and MoMoE consume $1 9 . 6 \%$ and $2 0 . 6 \%$ total backward time which is even longer than their up-proj weight gradient $d W _ { 1 }$ kernel time. 

Blackwell GPUs. SonicMoE supports the varlen-M Grouped GEMM and its gather fusion for Blackwell GPUs at the time of writing this paper. On Blackwell GPUs, the gather fusion with cp.async encounters an architectural challenge when using 2-CTA clusters (Figure 5) for GEMM computation. The cp.async instruction, introduced in the Ampere generation, can only signal completion within the same CTA. However, Blackwell’s 2-CTA GEMM requires the MMA instruction in the leader CTA (CTA 0) to wait for gather completion from both CTAs. To work around this limitation, CTA 1 

requires a dedicated relay warp that receives the cp.async completion signal and forwards it to CTA 0’s MMA warp using cluster-level synchronization primitives (e.g., mbarrier with cluster scope). This relay mechanism adds scheduling complexity but enables efficient gather fusion across the 2-CTA cluster, maintaining high throughput for Grouped GEMM. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-26/0390e304-b35b-4a5f-9ad3-e0d35edf7b5c/6157dd2418458fbe94f798edf8487ccd235620f06a1cb60c66affd633e019d74.jpg)



Figure 5: Pipeline structure for gather fusion with cp.async on Blackwell GPUs using 2-CTA clusters.


# 4.1.2 Epilogue fusion

We exploit the epilogue computation to maximally reduce unnecessary IO accesses with the following design choices: 

• SwiGLU and dSwiGLU fusion: We fuse the SwiGLU and backward of SwiGLU with the epilogue of forward up-proj and backward down-proj activation gradient kernel respectively (Costin et al. 2025). In Figure 6, even though DeepGEMM++ has a highly-optimized Grouped GEMM and SwiGLU kernel, the total time (0.60ms) of DeepGEMM $^ { + + }$ ’s up-proj (0.49ms, 629TFLOPS) and SwiGLU (0.11ms, 2.88TB/s) is still longer than SonicMoE’s up-proj (0.55ms, 559TFLOPS) despite SonicMoE having an additional Gather fusion besides SwiGLU. 

• Computing $d H$ and $d S$ in backward down-proj activation gradient (dH) kernel’s epilogue: This heavy epilogue fusion benefits SonicMoE with a major speedup over alternative designs. Our dH kernel (0.47ms, 328TFLOPS) produces the same output with far less total time than ScatterMoE’s down-proj act (0.43ms, 364TFLOPS), dS (0.24ms), 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-26/0390e304-b35b-4a5f-9ad3-e0d35edf7b5c/41a818df78c939d02c789b70f1a4c6f0702d99a24c4ba54751868c0da5fa025e.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-26/0390e304-b35b-4a5f-9ad3-e0d35edf7b5c/4d0a517cd347a216aecc934d266bfcab51055d2a348fc98622f32b8f2c80e050.jpg)



Figure 6: Runtime breakdown of different MoE kernels $( \mathrm { m s } \downarrow )$ ) on the 7B MoE training with $( T , d , n , E , K ) = ( 2 4 5 7 6 , 1 5 3 6 , 2 5 6 , 1 2 8 , 8 )$ on H100. We annotate the model memory bandwidth (TB/s ↑) for memory-bound kernels (gather, SwiGLU/dSwiGLU, and expert aggregation kernel) and compute throughput (TFLOPS $\uparrow$ , abbr as TF/s in the figure) for grouped GEMM kernels. Note that this profile is grouped by kernel runtime semantics and one block can contain multiple actual kernel timing results. For example, the “router related” on left subfigure includes both router GEMM and routing metadata computation time. In addition, we do not consider the CUDA stream bubble time across kernels in this figure. We use the GroupedMLP for Megatron, and ParallelDroplessMLP for MegaBlocks. DeepGEMM does not provide an efficient router implementation, gather and expert aggregation kernels during the forward pass, where we use a standard PyTorch implementation (“DeepGEMM-pt”) or our highly optimized kernels (“DeepGEMM ) for them. During the backward pass, both “DeepGEMM++” and “DeepGEMM-pt” use the same computational path as SonicMoE, except we launch separate kernel(s) that compute dS, $A ^ { \prime }$ , and dSwiGLU together. DeepGEMM++ is effectively the best possible MoE implementation built on top of DeepGEMM SM90 BF16 Grouped GEMM kernels without modifying DeepGEMM’s source code.


and dSwiGLU (0.33ms) combined together (0.99ms) during 7B MoE training in Figure 6. In addition, SonicMoE also demonstrates a speedup over DeepGEMM $^ { + + }$ (0.57ms), where we launch an efficient Grouped GEMM (0.32ms, 480TFLOPS) and a separate optimized kernel (0.25ms, 2.43TB/s) that computes dSwiGLU and $d S$ together. 

In Appendix C.1, we show that SonicMoE’s $d S = \langle d A , \ A ^ { \prime } \rangle = \langle d A , \mathrm { B r o a d c a s t } ( \mathbf { s } ) A \rangle$ is the computationally and activation memory-efficient choice for fine-grained MoEs. However, both ScatterMoE and MoMoE choose to compute $d S$ as $\langle d O , Y \rangle$ , which requires an additional 2T Kd HBM load cost and requires caching $2 T K d$ bytes of activation memory. In Figure 6 (right subfigure), ScatterMoE14 launches a separate kernel for dS (0.24ms) while MoMoE15 fuses $d S$ with up-proj activation gradient (in total 1.58ms, 196TFLOPS) which takes much longer time than SonicMoE’s up-proj activation gradient (0.50ms, 618TFLOPS). 

The throughput of heavy epilogue fusion on backward down-proj activation gradient $d H$ kernel is boosted by the overlap of asynchronous IO and MMA, which we will elaborate on Section 4.2. Such overlap helps SonicMoE to sustain a reasonable training throughput (328TFLOPS) and memory bandwidth (2.14TB/s) simultaneously even with the heavy epilogue fusion (load $H$ and $S$ , compute dH, dS, and $A ^ { \prime }$ as inputs to $d W _ { 2 }$ ) in dH kernel. 

# 4.2 GEMM MMA Overlapping with Asynchronous IO

Hopper GPUs. In NVIDIA Hopper GPUs, GEMM is performed asynchronously with a producer-consumer paradigm (Shah et al. 2024). Suppose we have 2 consumer warpgroups, we can either let them cooperatively issue the WGMMA instruction with a large tile size, or overlap the IO of 1 warpgroup with GEMM of another warpgroup with a smaller tile size. Once this is finished, we switch the roles of the warpgroups (effectively interleaving IO and GEMM). This is often referred to as Ping-Pong scheduling (Shah et al. 2024; Wright and Hoque 2024b) on Hopper GPUs in Figure 7. 

Ping-Pong scheduling16 is particularly useful to maintain high Tensor Core throughput with heavy epilogue. For example, the down-proj forward $Y$ kernel’s epilogue has heavy HBM store IO $2 T K d$ bytes) relative to the mainloop. In the down-proj activation gradient $( d H )$ kernel’s epilogue, we need to load $H$ ( $4 T K n$ bytes) and execute multiple activation and reduction operations to compute and store dH, dS, and $A ^ { \prime }$ as inputs for $d W _ { 2 }$ . We note that the concept of overlapping MMA with IO and Ping-Pong scheduling is known in other places such as Flash Attention 3 (Shah et al. 2024), but the application of 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-26/0390e304-b35b-4a5f-9ad3-e0d35edf7b5c/e0cae14141ac387ae8185a1350297b84bd6c56d4a7e7448533c8b9b2dbe17988.jpg)



Figure 7: SonicMoE’s Ping-Pong warpgroup scheduling on Hopper GPUs. The green arrows indicate that a consumer warpgroup signals the start of the epilogue and the other consumer warpgroup can proceed with the MMA. Once this step is complete, the roles of 2 consumer warpgroups is switched. SonicMoE mainly uses Ping-Pong for forward down-proj $Y$ kernel and backward down-proj activation gradient dH kernel as they both have heavy epilogue. In dH kernel, SonicMoE has an asynchronous TMA load during epilogue, and producer warps need to issue cp.async for gathering $d O$ and load expert weights with TMA. This figure is adapted from Wright and Hoque (2024a)’s blog on Ping-Pong scheduling.


Ping-Pong scheduling to address the increasing IO costs of fine-grained MoE kernel design is novel. 

DeepGEMM SM90 BF16 varlen-M Grouped GEMM kernel11 does not implement Ping-Pong scheduling. This design choice is suitable for the case of lightweight epilogue (e.g. up-proj forward) but underperforms (413 TFLOPS and 2.15 TB/s vs. SonicMoE’s 485 TFLOPS and $2 . 5 2 \mathrm { T B } / \mathrm { s } $ ) for the case of heavy epilogue in down-proj forward, as shown in Figure 6. In Figure 19, SonicMoE’s down-proj has $1 0 . 0 \%$ higher TFLOPS than DeepGEMM on average. 

Besides Ping-Pong scheduling, SonicMoE also relies on asynchronous TMA operations to perform GMEM-to-SMEM load and SMEM-to-GMEM store. We overlap the following asynchronous IO with the MMA operations: 

• Asynchronous TMA load during dH kernel’s epilogue: In the $d H$ kernel’s epilogue, we need to load $H$ to compute $d H$ from $d A$ . We create a dedicated pipeline for asynchronous TMA load of $H$ to overlap with other epilogue operations across epilogue stages. In Figure 7, the transparent TMA block in the consumer warpgroups illustrates such asynchronous epilogue load. 

• Asynchronous TMA store in forward down-proj $Y$ and backward up-proj activation gradient $d \tilde { X }$ kernel: SonicMoE applies asynchronous TMA store for all 6 Grouped GEMMs. In forward down-proj and backward up-proj activation gradient, SonicMoE does not fuse the scatter with HBM store where ScatterMoE17 and $\mathbf { M o M o E } ^ { 1 8 }$ both choose to fuse the HBM store with scatter. This is because the scatter fusion (1) has more synchronous index fetching and address calculations19 and (2) requires a synchronous SMEM-to-GMEM store instruction on Hopper GPUs.20 The synchronous GMEM store blocks the execution of MMA of next tile and largely degrade the TFLOPS $( \sim 2 0 \% )$ in the case of heavy HBM store during forward down-proj and backward up-proj activation gradient kernel computation for fine-grained MoEs, as illustrated by Figure 8. We also note that Ping-Pong warpgroup scheduling cannot fully restore the throughput degradation for synchronous epilogue IO operations as the epilogue consumer warpgroup would be blocked and cannot switch the role with MMA warpgroup until the current synchronous GMEM store is finished. 

Blackwell GPUs. On NVIDIA Blackwell GPUs, GEMM kernels use the same “Ping-Pong” scheduling in spirit, but the implementation differs from Hopper. Blackwell introduces Tensor Memory (TMEM), a dedicated 256KB on-chip memory per SM organized as 512 columns $\times \ 1 2 8$ rows of 32-bit cells (NVIDIA 2025b; Research 2024). The accumulator results from matrix multiplication are stored directly in TMEM rather than in registers, with the 512-column structure naturally enabling a two-stage accumulator pipeline. Each stage uses 256 columns: while one stage performs MMA operations via the new UMMA (Unified Matrix Multiply-Accumulate) instruction, the other stage executes the epilogue. Unlike Hopper’s WGMMA which required warpgroup-level coordination and consumed significant register memory, Blackwell’s UMMA is a single-threaded asynchronous operation that eliminates register pressure for accumulation. This architectural change allows 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-26/0390e304-b35b-4a5f-9ad3-e0d35edf7b5c/fcae50740aafe6c1817b7196bf0237adf0e9906e07c239b3817892de85f62e18.jpg)



Figure 8: Illustration to show asynchronous TMA store (top) has higher memory bandwidth and can naturally overlap with TensorCore MMA while synchronous st.global (bottom) PTX instruction, necessary for scatter fusion on Hopper GPUs, blocks the execution of next Tensor Core MMA tile and leads to longer kernel runtime. This figure is supported by the $2 0 . 1 \%$ speedup on average of ”SonicMoE (gemm + gth w. sum)” (TMA store) over ”SonicMoE (gemm w. sct + sum)” (st.global) in Figure 22’s transparent bars. As a result, SonicMoE does not fuse scatter with HBM store and instead, lets each token gather the expert results in the expert aggregation kernel. Both ScatterMoE and MoMoE do not adopt such design and SonicMoE can achieve $1 . 7 5 \mathrm { x }$ and 3.11x speedup respectively on average during forward down-proj kernel in Figure 6.


epilogue warps to read and process results from one TMEM stage concurrently with MMA warps accumulating into the other stage, enabling better overlap of epilogue and MMA operations compared to Hopper’s ping-pong scheduling. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-26/0390e304-b35b-4a5f-9ad3-e0d35edf7b5c/9ee6e18f0e4829561cd51a5dc59f0bd06136fc785d1695e8880a5676949ecb9b.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-26/0390e304-b35b-4a5f-9ad3-e0d35edf7b5c/3883d323b1317f33e652e6c384e09daa283798ba865b488bc07011efac34708d.jpg)



Figure 9: Possible strategies for storing the results and aggregating the results for each token. SonicMoE chooses the first strategy (left) in which each expert directly stores contiguously-packed outputs via TMA in the GEMM epilogue. In the expert aggregation kernel, each token gathers and sums over activated expert outputs. ScatterMoE and MoMoE (middle) choose to fuse HBM store with scatter in epilogue and launch a summation kernel afterwards. We note that each token gathering (left) the Grouped GEMM result is equivalent to each expert scattering (middle) the Grouped GEMM outputs. In Figure 22, we implement both strategies on SonicMoE and observe the left strategy can have $17 \%$ speedup over the middle strategy. It is also possible to fuse atomic add in the epilogue to circumvent the requirement of an expert aggregation kernel as the right subfigure illustrated. However, this atomic add operation creates new issues like non-determinism (He and Machines 2025) and numerical accuracy (for BF16 atomic add). This figure is adapted from Tan et al. (2024)’s Figure 2.


# 4.3 Efficient top- $K$ sorting kernel for MoE

Existing MoE approaches such as ScatterMoE21, $\mathbf { M o M o E } ^ { 2 2 }$ , and MegaBlocks23 use the PyTorch top- $K$ (torch.topk) to compute the expert assignments for each token. We find that the PyTorch top- $K$ kernel can take approximately $40 \%$ of the router’s computation time. We implement an efficient top- $K$ kernel in SonicMoE to reduce the overhead due to PyTorch top- $K$ . Our top- $K$ kernel supports $E$ and $K$ when $E \le 4 0 9 6$ and $K \leq 1 6 ^ { 2 4 }$ and is optimized for the case of large number of tokens $T$ . We also offer an optional softmax fusion on top- $K$ values within the top- $K$ kernel. 

The top- $K$ kernel accepts the router output with shape $( T , E )$ and parallelizes over $T$ . The kernel uses bitonic sort (Batcher 1968) over every row (sorts $E$ values) and selects the first $K$ columns as the sort output. After loading the input, we pack the column indices of the first $K$ columns (for argtopK) to the lower $\log _ { 2 } ( E )$ mantissa bits of FP32 values in registers25, except that we specialize the base sorting cases (number of values $\leq 6 4$ ) to follow the comparison strategies obtained from optimal low-latency sorting networks (Dobbelaere 2025), which provide the minimum number of parallel operation steps and required compare and swap calls. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-26/0390e304-b35b-4a5f-9ad3-e0d35edf7b5c/c28f5b1fcc8a1ead643d8b827cdb52635ca20f146bc80704456da4f05d9c6897.jpg)


new FP32 val after packing column idx 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-26/0390e304-b35b-4a5f-9ad3-e0d35edf7b5c/a0feb76fc06081e229e80d4baf8fcbf1d99c52062c6acb11ea76cfefb478f689.jpg)



Figure 10: The sorting is conducted over values after we pack the column index bits into lower mantissa bits. This value format ensures a stable sorting result. Triton’s official top- $K$ kernel follows a similar format.


The bitonic compare and merging occurs within the same thread or the same warp via warp-shuffle. Therefore, every swap and merge operation only uses intra-thread or intrawarp registers. This achieves a higher memory bandwidth for the kernel over alternative kernel designs such as PyTorch TopK (Paszke et al. 2019), the Triton (Tillet, Kung, and Cox 2019) and Tilelang (Wang et al. 2025) official example, and RTop-K (Xie et al. 2025) in Figure 23. 

Since the assigned column indices for values on each row are always unique, there will not be any equal numbers after we pack the column index to the lower mantissa bits. Therefore, SonicMoE’s top- $K$ kernel is always stable as there will not be any tie-breaking scenarios during bitonic compare and merge. 

# 5 Token rounding routing

In this section, we analyze the hardware efficiency under sparse MoE training regime and identify that as MoEs become sparser, the wasted compute on padded GEMM tiles accumulate to a nontrivial amount, known as “tile quantization” effects. In response, we propose a novel routing method “token rounding” to eliminate tile quantization effects. 

# 5.1 Training efficiency of sparse MoE

# Algorithm 4 Token rounding routing

Input : $X ~ \in ~ \mathbb { R } ^ { T \times d }$ ; number of experts $E$ and expected activated number of experts $K$ per token; tile size $M _ { \mathrm { t i l e } }$ ; router scores $S ~ \in ~ [ 0 , 1 ] ^ { T \times E }$ . round and sparsify that determines rounding up or down. 

Output : $M _ { \mathrm { t i l e } }$ -rounded router scores $\lfloor S \rceil _ { M _ { \mathrm { t i l e } } }$ 

(1) Top- $K$ token choice sorting 

$$
\left(S _ {\text {t o p K}}, I _ {\text {t o p K}}\right) \leftarrow \operatorname {T o p K} (S, K)
$$

(2) Calculate each expert’s received token frequencies and its $M _ { \mathrm { t i l e } }$ rounded multiples 

$$
f _ {e} \leftarrow \sum_ {t} \mathbf {1} _ {\{e \in I _ {\text {t o p K ,}} \}}
$$

$$
\left[ f _ {e} \right] _ {M _ {\text {t i l e}}} \leftarrow \left[ f _ {e} / M _ {\text {t i l e}} \right] \cdot M _ {\text {t i l e}}
$$

$$
\left\lfloor f _ {e} \right\rfloor_ {M _ {\text {t i l e}}} \leftarrow \left\lfloor f _ {e} / M _ {\text {t i l e}} \right\rfloor \cdot M _ {\text {t i l e}}
$$

(3) Build Top- $K$ -preferred $S ^ { \prime }$ for expert-wise ranking 

// ensure non-top- $K$ entries are smaller 

$$
S _ {e} ^ {\prime} \leftarrow S _ {e} - 1
$$

for $t \in [ T ]$ & $k \in [ K ]$ in parallel do 

$$
S _ {t, I _ {\mathrm {t o p K}} (t, k)} ^ {\prime} \leftarrow S _ {\mathrm {t o p K}, t, k}
$$

(4) Token rounding per expert 

for $e \in [ E ]$ do // token ordering and sorted scores $\pi _ { e }$ , $s _ { e }  \mathrm { s o r t } ( S _ { e } ^ { \prime } )$ // token rounding for expert e $\pi _ { e } ^ { \prime }$ , $s _ { e } ^ { \prime } \gets$ round and sparsify $\cdot ( \pi _ { e } , s _ { e } , f _ { e } , \lceil f _ { e } \rceil _ { M _ { \mathrm { t i l e } } } , \lfloor f _ { e } \rfloor _ { M _ { \mathrm { t i l e } } } )$ $\lfloor S \rceil _ { M _ { \mathrm { t i l e } } , e } \gets \mathrm { G a t h e r } ( S , \pi _ { e } )$ 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-26/0390e304-b35b-4a5f-9ad3-e0d35edf7b5c/d1f53eabed1fec5cd9af8f9bbab7e8c6196d60ef929e332f933b7a65589fabde.jpg)



Figure 11: Wasted FLOPs by padding during MoE forward & backward pass with $T = 1 6 k$ , $d = 4 k$ , $n = 1 k$ , $K = 4$ as illustrated in the bottom right 2 subfigures of Figure 16.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-26/0390e304-b35b-4a5f-9ad3-e0d35edf7b5c/7f723c9ceb118f2313269e4c3b19dbab5dea5dffb932a36caa89a2a722ebfc64.jpg)



Figure 12: A demonstration of tile quantization effect for sparse MoE. The rounding subroutine in TR makes a binary decision for discarding or padding tokens to guarantee that each expert receives $M _ { \mathrm { t i l e } }$ -multiple number of tokens.


Besides granularity, the arithmetic intensity of MoE also depends on the MoE activation ratio $\rho$ as shown in Equation 4. When we scale down $\rho$ , the expected number of received tokens per expert $\mathbb { E } _ { e \in [ E ] } T _ { e } = \bar { T } _ { e } = T \rho$ will also linearly decrease and the GEMM computation shifts towards a memory-bound regime. 

Tile quantization effect. GEMM on modern GPUs is often computed in tiles (NVIDIA 2022) and we always need to pad to the next tile-sized multiples if any dimensions of $\mathbf { M } , \mathbf { N } , \mathbf { K }$ are not fully divisible by tile sizes. Once the size of input (e.g. token dimension per expert) is small, the wasted TFLOPS by padding can be nontrivial. 

Therefore, we propose to use token rounding to avoid launching such extra tiles, thereby leading to more efficient training. We also empirically show that our token rounding method does not affect model quality while achieving much higher training throughput. 

# 5.2 Token rounding routing

As such, we propose to use token rounding (TR) method as a 2-step sorting algorithm as shown in Algorithm 4. The token rounding algorithm first computes the vanilla token-choice (TC) routing results and applies a sorting of the router score over each expert’s tokens, similar to EC sorting step. We then choose to either discard tokens selected in the first step of TC top- $K$ routing or pad additional tokens on the second step of sorting. Between these 2 steps, we process the routing weight matrix such that the TC tokens are always preferred over EC tokens. This is done so that both discarding or padding only affects the last input tile for each expert. 

Token rounding requires a round and sparsify subroutine for making a binary decision for discarding or padding. Our default choice for such subroutine is to round expert frequency to the nearest $M _ { \mathrm { t i l e } }$ multiples: we choose to pad EC selected tokens if $\lceil f _ { e } \rceil _ { M _ { \mathrm { t i l e } } } - f _ { e }$ is smaller than $f _ { e } - \lfloor f _ { e } \rfloor _ { M _ { \mathrm { t i l e } } }$ . 26 We further conduct an ablation in Table 6 and find that (1) our TR algorithm is quite robust w.r.t. the underlying rounding subroutine (2) this simple strategy of nearest rounding on expert frequency is often sufficient to yield excellent task performance. More detailed discussion on different rounding subroutines are included in Appendix F.2. 

MoE training & inference quality. This simple algorithm guarantees that for each expert, the maximum deviation from token-choice routing is at most 1 tile. We find that this property has a surprisingly robust performance even under sparse MoE training regime and can serve as a substitute for token-choice under sparse MoE training settings, which is shown in Table 3. We also conduct an ablation study on the effect of microbatch size $T$ and tile size $M _ { \mathrm { t i l e } }$ on the quality of trained MoE model with TR in Table 7 and 8, and we find token rounding routing is generally robust when $\bar { T } _ { e } / M _ { \mathrm { t i l e } } \geq 2$ . 

MoE training throughput. TR guarantees no tile quantization effects and in Section 6.3.3, we show that TR training throughput over vanilla TC top- $K$ is consistently higher when in the highly sparse MoE training regime and can achieve $16 \%$ higher TFLOPS for the kernel runtime as we scale up $E$ while keeping $K$ constant. 

# 6 Experiments

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-26/0390e304-b35b-4a5f-9ad3-e0d35edf7b5c/d6642d36fd7aedd31d9ddd7138c8fab424e123d2c5fbac8974fb3edc53e6313c.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-26/0390e304-b35b-4a5f-9ad3-e0d35edf7b5c/3e0e0dd06c7ce3791f9ebd23e809067cb3ada6dc19c61f643e3d5e7283040bec.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-26/0390e304-b35b-4a5f-9ad3-e0d35edf7b5c/b2b56b682373fa22f3f065bbb1e1ffead8c1eb4984f5a102562c77a0536342f2.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-26/0390e304-b35b-4a5f-9ad3-e0d35edf7b5c/be91be306fb4936d0361bbe5ef3d7249c8b6aa701c529944d2130ff6ac2ff020.jpg)



Figure 13: Peak activation memory usage per layer across different model scales (1.4B–120B). MegaBlocks does not support small $n$ The benchmark configurations are listed in Table 9b. We only cache $X$ , gathered $X _ { e }$ , $H _ { e }$ for each expert $e$ and routing metadata which is the minimum amount of activation memory required for backward computation without GEMM recomputation.


We evaluate SonicMoE’s activation memory footprint (Section 6.1) and training throughput (Section 6.2) compared to other baseline MoE implementations. We also demonstrate the efficacy of token rounding routing strategy and show that it is possible to use token choice as a drop-in replacement after training with token rounding in Section 6.3.1. We also show that token rounding can maintain the training throughput under sparse MoE configuration in Section 6.3.3. 

# 6.1 SonicMoE’s activation memory

We demonstrate that the peak activation memory for SonicMoE has the lowest activation memory footprint for a single MoE layer as shown in Figure 13 across all scales. For the 7B model with $n = 2 5 6$ , our approach reduces memory usage by $45 \%$ compared to ScatterMoE, and more significantly compared to MoMoE. For 30B and 120B models, the gap becomes even wider: at 120B scale, our method saves more than 3GiB memory per layer compared to MoMoE. We also validate that SonicMoE’s activation memory stays constant w.r.t. expert granularity as shown in Figure 1. 

# 6.2 SonicMoE’s training throughput

# 6.2.1 Entire forward and backward throughput

Figure 14 reports the compute throughput of forward and backward pass of one MoE layer in various MoE training configurations. Across all model scales, our method consistently achieves the highest TFLOPS. In 1.4B and 7B settings, our approach improves TFLOPS by $40 \%$ compared to ScatterMoE and MoMoE. This throughput gap becomes wider for 30B and 120B MoE as SonicMoE achieves over 500 TFLOPS in forward and backward passes, whereas other baselines either fail to support certain $n$ sizes (MegaBlocks) or suffer from significant performance degradation (MoMoE). SonicMoE also demonstrates speedup over DeepGEMM $^ { + + }$ in the forward pass, which mainly arises from the gather $X$ kernel and Ping-Pong scheduling. The effect of both features increases as the MoE becomes more fine-grained (as we move from right to left across all configs in Figure 14) and thus SonicMoE’s relative speedup over DeepGEMM++ becomes larger. 

We further measure the real training throughput of a 7B MoE model with FSDP-2: SonicMoE on 64 H100s gets 213 billion tokens per day which achieves similar throughput as ScatterMoE on 96 H100s with 225 billion tokens per day. The throughput for this is measured using the lm-engine codebase27 (Mishra 2024). We shard the model using ZeRO-3 within a single node (8x H100s) and replicate this sharded unit across nodes for this experiment. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-26/0390e304-b35b-4a5f-9ad3-e0d35edf7b5c/e3583474c23dc8bd3e127cbb714dbb299aca28b072c0ffe21c3201dc96e8d3cf.jpg)



Figure 14: Forward & backward TFLOPS for different MoE kernels on H100. DeepGEMM does not provide an efficient router implementation, gather and expert aggregation kernels, where we use a standard PyTorch implementation (“DeepGEMM-pt”) or our highly optimized kernels (“DeepGEMM++”) for them. During the backward pass, both “DeepGEMM++” and “DeepGEMM-pt” use the same computational path as SonicMoE, except we launch separate kernel(s) that compute dS, $A ^ { \prime }$ , and dSwiGLU together. The MoE configurations are the same as in Figure 13.


In addition, we measure the training throughput of a single MoE layer with configurations from recent open-source MoEs in Figure 15. SonicMoE generally achieves more than 550 TFLOPS during both forward and backward pass, and consistently surpasses all baselines. We note that ScatterMoE, MoMoE, DeepGEMM-pt, and DeepGEMM++ all fail to run at the configuration for DeepSeek-V3.2-Exp, a 685B MoE model, while SonicMoE successfully runs on a single H100 GPU without expert parallelism achieving 534.8 TFLOPS during forward and 480.1 TFLOPS during backward. We also note that SonicMoE’s IO-aware kernel design can achieve a greater relative speedup over baselines $61 \%$ over ScatterMoE, $92 \%$ over MoMoE during forward, $85 \%$ over ScatterMoE, $120 \%$ over MoMoE during backward) for sparse and fine-grained MoEs (e.g. $\rho = 1 0 / 5 1 2$ and $G = 2 0 4 8 / 5 1 2$ for Qwen3-Next-80B-A3B-Thinking, $4 ^ { \mathrm { t h } }$ column for Firgure 15). 

# 6.3 Token rounding

# 6.3.1 Token rounding’s general task evaluation

In this section, we assess the quality of trained MoEs using our token rounding (“TR”) algorithm. We use TR for training and during evaluation we switch to token-choice top- $K$ (“TC top- $K ^ { \mathbf { \mathfrak { s } } , \mathbf { \mathfrak { s } } }$ ) routing. This assesses the capability of replacement of TR with TC after training.28 We use the OLMoE codebase and construct MoE models with OLMoE base architecture 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-26/0390e304-b35b-4a5f-9ad3-e0d35edf7b5c/6078bad49baa3620c153f2868efae45e39979dcca71f1ff4911c010cd924293e.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-26/0390e304-b35b-4a5f-9ad3-e0d35edf7b5c/5fceee5ab01919013444ed815191640a9bcedfd0e354732de21a54f7bcdb9a82.jpg)



Figure 15: Forward & backward TFLOPS of a single MoE layer for different MoE kernels for different configurations ranging from 7B to 685B parameters on H100. The MoE configurations from left to right adopt the model size of OLMoE-1B-7B-0125 (Muennighoff et al. 2025), gpt-oss-20b (OpenAI 2025), Kimi-Linear-48B-A3B-Base (Zhang et al. 2025), Qwen3-Next-80B-A3B-Thinking (Qwen 2025), Qwen3-235B-A22B-Thinking-2507 (Qwen 2025), and DeepSeek-V3.2-Exp (DeepSeek-AI 2025). For fair comparison, we do not consider shared experts and expert biases, and we always use TC top- $K$ router with softmax scores. ScatterMoE, MoMoE, DeepGEMM-pt, and DeepGEMM $^ { + + }$ all fail to run (either due to index overflow or CUDA OOM errors) for the DeepSeek-V3.2-Exp configuration.


(Muennighoff et al. 2025). We use a deduplicated version of FineWeb-Edu (Ben Allal et al. 2024) for training all our models. More details are included in Appendix H. 

We consistently use $M _ { \mathrm { t i l e } } = 1 2 8$ in Table 3, and the round and sparsify subroutine always rounds the expert frequency to the nearest multiple of $M _ { \mathrm { t i l e } }$ (“NR-f”, see Appendix F.2). We also use softmax renormalization for TR. We compare TR to token-choice (TC) top- $K$ routing and expert-choice (EC) routing (Zhou et al. 2022). However, EC routing results in future token leakage causing problems for autoregressive generation resulting in a performance drop during evaluation (Raposo et al. 2024; Wang et al. 2024). To address this issue, we consider MoD’s approach (Raposo et al. 2024) that trains an auxiliary router to predict the EC router’s selection during inference29. This baseline is referred as “EC (aux router)” in each subtable in Table 3. We also adapt EC routing to TC routing by finetuning a learned TC top- $K$ router and compare its task performance against TR’s task performance without any adaptation. This is the “EC (ft TC router)” baseline in Table 3. Finally, we consider a token dropping baseline in which we set the capacity of each expert as the largest multiple of $M _ { \mathrm { t i l e } }$ not exceeding its token frequencies and we discard the tokens with lowest scores. This is the “TC (token drop)” baseline, and we note that this is equivalent as we always round down in TR. 

TR’s train-test gap. We validate TR’s performance on a 0.5B (subtable 3a) and 1.4B (subtable 3c) MoE model. We then increase the MoE sparsity by either decreasing $K$ while keeping $E$ constant (from 3a to 3b, from 3d to 3e) or increasing $E$ while keeping $K$ constant (from 3a to 3c). Across these sparse MoE configurations, we consistently observe similar model quality between TR and TC. In fact, TR achieves slightly lower validation perplexity and higher average accuracy under the extremely sparse MoE $( K / E < 1 / 3 2 )$ settings for the 3c and 3e. There is a noticeable discrepancy between EC and TC as the train and val PPL for EC can have ${ \gtrap{ \approx } } 3$ gap for 3c,3d and 3e compared to TC and TR’s usual $\lessapprox 0 . 3$ gap. TC finetuning is more effective than the auxiliary router to close this gap, but TR’s task evaluation is still always better. In addition, when we compare TR with the token dropping baseline, we also find TR consistently yields lower validation perplexity, and has higher average task accuracy for 3a, 3c, 3e. In this case, TR can serve as an in-place substitute for TC during training. 

# 6.3.2 Ablation studies on token rounding routing

There are 3 variables that can affect the trained MoE quality with token rounding routing: (1) rounding subroutine round and sparsify (2) microbatch size $T$ , and (3) tile size $M _ { \mathrm { t i l e } }$ for rounding. We analyze their impacts: 

• Choice of rounding subroutine: In Table 6, we assess the choice of different routing subroutines to train MoEs using TR. We find that our token rounding algorithm in general is robust to the specific rounding subroutines, and nearest rounding expert frequency to multiples of $M _ { \mathrm { t i l e } }$ (“NR-f” in Table 6) is often sufficient for providing an excellent downstream task performance despite its simplicity. Therefore, we choose NR-f as the default rounding subroutine. 

• Effect of microbatch size $T$ and tile size $M _ { \mathrm { t i l e } }$ : The token rounding is applied on the microbatch level so varying the microbatch size $T$ will result in different qualitative results for TR. This also holds true for EC routing. For example, 

Table 3: Comparison of different routing methods’ task evaluation. “Train” and “Val” refer to the perplexity towards the end of training and on the validation set respectively. The next 11 columns are downstream tasks evaluated at the end of training and we report the accuracy for each. “Avg” is the mean accuracy across these 11 downstream tasks. We use TC top- $K$ routing for TR, token dropping, and EC baselines when evaluating validation perplexity and task performance. $\bar { T } _ { e }$ represents the average number of received tokens in each microbatch per expert. 


(a) 0.5B params, 20B tokens, 8/64 activated $\hat { T } _ { e } = 4 0 9 6$ , $M _ { \mathrm { t i l e } } = 1 2 8$


<table><tr><td>Method</td><td>Train</td><td>Val</td><td>Wino</td><td>SIQA</td><td>SciQ</td><td>PIQA</td><td>OBQA</td><td>HS</td><td>COPA</td><td>CSQA</td><td>BoolQ</td><td>ArcE</td><td>ArcC</td><td>Avg</td></tr><tr><td>TR</td><td>15.91</td><td>15.94</td><td>51.9</td><td>41.3</td><td>80.8</td><td>65.5</td><td>35.0</td><td>38.7</td><td>63.0</td><td>31.2</td><td>61.4</td><td>58.9</td><td>27.1</td><td>50.4</td></tr><tr><td>TC top-K</td><td>16.04</td><td>16.01</td><td>51.0</td><td>41.4</td><td>79.2</td><td>65.5</td><td>31.6</td><td>38.4</td><td>66.0</td><td>31.5</td><td>60.2</td><td>57.5</td><td>25.7</td><td>49.8</td></tr><tr><td>TC (token drop)</td><td>16.52</td><td>16.46</td><td>51.1</td><td>41.1</td><td>79.5</td><td>64.6</td><td>30.2</td><td>37.3</td><td>63.0</td><td>31.8</td><td>58.2</td><td>57.9</td><td>28.4</td><td>49.4</td></tr><tr><td>EC</td><td>16.25</td><td>17.23</td><td>51.0</td><td>41.0</td><td>78.3</td><td>63.8</td><td>33.4</td><td>37.5</td><td>69.0</td><td>31.4</td><td>54.4</td><td>56.1</td><td>29.4</td><td>49.7</td></tr><tr><td>EC (aux router)</td><td>16.25</td><td>17.40</td><td>52.6</td><td>41.5</td><td>77.3</td><td>64.4</td><td>31.4</td><td>37.5</td><td>65.0</td><td>30.9</td><td>55.4</td><td>55.8</td><td>30.4</td><td>49.3</td></tr><tr><td>EC (ft TC router)</td><td>16.34</td><td>16.40</td><td>49.3</td><td>41.4</td><td>78.0</td><td>64.4</td><td>33.4</td><td>37.5</td><td>67.0</td><td>30.8</td><td>56.1</td><td>55.4</td><td>29.4</td><td>49.3</td></tr></table>


(b) 0.5B params, 40B tokens, 2/64 activated $\bar { T } _ { e } = 5 1 2$ , $M _ { \mathrm { t i l e } } = 1 2 8$ )


<table><tr><td>TR</td><td>16.22</td><td>15.92</td><td>51.4</td><td>41.6</td><td>78.4</td><td>65.4</td><td>31.6</td><td>38.1</td><td>65.0</td><td>31.0</td><td>61.1</td><td>57.4</td><td>29.1</td><td>50.0</td></tr><tr><td>TC top-K</td><td>16.34</td><td>15.94</td><td>51.0</td><td>41.9</td><td>78.5</td><td>64.8</td><td>33.0</td><td>38.1</td><td>67.0</td><td>30.8</td><td>54.7</td><td>55.8</td><td>30.1</td><td>49.6</td></tr><tr><td>TC (token drop)</td><td>16.44</td><td>16.10</td><td>51.1</td><td>41.4</td><td>78.7</td><td>64.9</td><td>31.6</td><td>38.0</td><td>62.0</td><td>32.8</td><td>61.9</td><td>58.9</td><td>30.8</td><td>50.2</td></tr><tr><td>EC</td><td>16.83</td><td>18.61</td><td>49.6</td><td>41.4</td><td>79.1</td><td>64.4</td><td>33.4</td><td>36.9</td><td>62.0</td><td>32.8</td><td>60.2</td><td>55.8</td><td>29.1</td><td>49.5</td></tr><tr><td>EC (aux router)</td><td>16.80</td><td>21.80</td><td>50.0</td><td>40.9</td><td>75.2</td><td>63.7</td><td>28.2</td><td>35.2</td><td>61.0</td><td>31.5</td><td>57.2</td><td>53.3</td><td>24.7</td><td>47.4</td></tr><tr><td>EC (ft TC router)</td><td>16.81</td><td>16.98</td><td>50.0</td><td>41.7</td><td>79.7</td><td>64.9</td><td>31.6</td><td>36.8</td><td>63.0</td><td>32.1</td><td>60.7</td><td>54.6</td><td>27.4</td><td>49.3</td></tr></table>


(c) 1.8B params, 40B tokens, 8/256 activated $\bar { T } _ { e } = 5 1 2$ , $M _ { \mathrm { t i l e } } = 1 2 8$ )


<table><tr><td>TR</td><td>13.34</td><td>13.10</td><td>53.4</td><td>42.1</td><td>81.7</td><td>69.6</td><td>35.2</td><td>45.3</td><td>70.0</td><td>33.2</td><td>61.4</td><td>63.0</td><td>33.4</td><td>53.5</td></tr><tr><td>TC top-K</td><td>13.51</td><td>13.12</td><td>50.1</td><td>42.9</td><td>81.3</td><td>69.8</td><td>33.8</td><td>45.2</td><td>71.0</td><td>34.1</td><td>56.7</td><td>64.6</td><td>31.1</td><td>52.8</td></tr><tr><td>TC (token drop)</td><td>13.62</td><td>13.19</td><td>55.4</td><td>41.6</td><td>82.2</td><td>68.6</td><td>34.8</td><td>45.0</td><td>69.0</td><td>34.0</td><td>54.4</td><td>63.5</td><td>31.4</td><td>52.7</td></tr><tr><td>EC</td><td>14.92</td><td>19.82</td><td>51.9</td><td>40.8</td><td>77.7</td><td>65.8</td><td>30.0</td><td>39.8</td><td>67.0</td><td>30.9</td><td>60.7</td><td>56.0</td><td>28.4</td><td>49.9</td></tr><tr><td>EC (aux router)</td><td>14.94</td><td>18.01</td><td>50.6</td><td>41.8</td><td>79.8</td><td>65.8</td><td>31.6</td><td>39.3</td><td>62.0</td><td>31.8</td><td>59.7</td><td>55.8</td><td>29.8</td><td>49.8</td></tr><tr><td>EC (ft TC router)</td><td>14.81</td><td>15.01</td><td>52.7</td><td>41.1</td><td>79.6</td><td>66.9</td><td>30.6</td><td>40.2</td><td>66.0</td><td>31.9</td><td>60.5</td><td>57.2</td><td>30.8</td><td>50.7</td></tr></table>


(d) 1.4B params, 50B tokens, 8/128 activated $\bar { T } _ { e } = 2 0 4 8$ , $M _ { \mathrm { t i l e } } = 1 2 8$


<table><tr><td>TR</td><td>13.51</td><td>13.28</td><td>52.6</td><td>42.6</td><td>81.5</td><td>69.6</td><td>33.6</td><td>45.4</td><td>67.0</td><td>34.8</td><td>57.3</td><td>63.7</td><td>28.1</td><td>52.4</td></tr><tr><td>TC top-K</td><td>13.50</td><td>13.32</td><td>51.8</td><td>41.7</td><td>81.5</td><td>69.3</td><td>32.4</td><td>45.3</td><td>68.0</td><td>34.5</td><td>56.6</td><td>63.2</td><td>28.4</td><td>52.1</td></tr><tr><td>TC (token drop)</td><td>13.52</td><td>13.30</td><td>51.8</td><td>42.2</td><td>84.1</td><td>69.2</td><td>34.4</td><td>45.2</td><td>70.0</td><td>35.1</td><td>61.2</td><td>64.2</td><td>31.4</td><td>53.5</td></tr><tr><td>EC</td><td>14.41</td><td>17.37</td><td>51.4</td><td>42.0</td><td>79.7</td><td>66.3</td><td>32.2</td><td>40.7</td><td>64.0</td><td>31.8</td><td>59.0</td><td>57.4</td><td>27.4</td><td>50.2</td></tr><tr><td>EC (aux router)</td><td>14.34</td><td>26.96</td><td>49.8</td><td>41.5</td><td>79.1</td><td>63.1</td><td>30.2</td><td>37.6</td><td>61.0</td><td>31.0</td><td>60.9</td><td>46.7</td><td>25.1</td><td>47.8</td></tr><tr><td>EC (ft TC router)</td><td>14.67</td><td>14.90</td><td>51.9</td><td>41.8</td><td>80.1</td><td>66.4</td><td>32.6</td><td>41.1</td><td>65.0</td><td>32.4</td><td>57.7</td><td>57.7</td><td>27.8</td><td>50.4</td></tr></table>


(e) 1.4B params, 100B tokens, 2/128 activated ( $\bar { T } _ { e } = 5 1 2$ , $M _ { \mathrm { t i l e } } = 1 2 8$


<table><tr><td>TR</td><td>13.31</td><td>13.22</td><td>52.8</td><td>41.8</td><td>80.8</td><td>68.7</td><td>33.0</td><td>43.4</td><td>67.0</td><td>33.6</td><td>60.2</td><td>60.7</td><td>29.8</td><td>52.0</td></tr><tr><td>TC top-K</td><td>13.50</td><td>13.32</td><td>51.3</td><td>42.0</td><td>83.2</td><td>68.2</td><td>34.0</td><td>43.4</td><td>66.0</td><td>35.4</td><td>57.9</td><td>61.6</td><td>29.4</td><td>52.0</td></tr><tr><td>TC (token drop)</td><td>13.35</td><td>13.29</td><td>50.0</td><td>42.2</td><td>81.7</td><td>68.3</td><td>31.2</td><td>43.3</td><td>66.0</td><td>34.3</td><td>56.6</td><td>59.5</td><td>30.8</td><td>51.3</td></tr><tr><td>EC</td><td>14.08</td><td>24.79</td><td>51.5</td><td>41.7</td><td>81.0</td><td>66.1</td><td>33.2</td><td>40.6</td><td>64.0</td><td>34.0</td><td>56.3</td><td>56.5</td><td>27.4</td><td>50.2</td></tr><tr><td>EC (aux router)</td><td>14.01</td><td>37.52</td><td>49.7</td><td>40.2</td><td>73.6</td><td>57.5</td><td>27.6</td><td>33.2</td><td>61.0</td><td>27.8</td><td>58.8</td><td>45.2</td><td>24.2</td><td>45.3</td></tr><tr><td>EC (ft TC router)</td><td>14.24</td><td>14.75</td><td>52.2</td><td>42.6</td><td>79.4</td><td>65.7</td><td>32.8</td><td>40.8</td><td>64.0</td><td>34.9</td><td>58.3</td><td>57.2</td><td>27.1</td><td>50.5</td></tr></table>

EC over sequence will result in different model quality as EC over a text segment. Nevertheless, in Table 7, we find that TR perserves its trained MoE quality when $\bar { T } _ { e } / M _ { \mathrm { t i l e } } \geq 2$ , and even if $\bar { T } _ { e } / M _ { \mathrm { t i l e } } = 1$ (the last row in both subtables), the trained MoE inference quality is still better than training with EC and finetuning with TC top- $K$ routing. Similarly in Table 8, we can find that TR is generally robust w.r.t. $M _ { \mathrm { t i l e } }$ when $\bar { T } _ { e } / M _ { \mathrm { t i l e } } \geq 2$ . However, when $\bar { T } _ { e } / M _ { \mathrm { t i l e } } = 1$ there is a noticeable degradation compared to TC baseline but the model quality is still better than the EC baseline. 

# 6.3.3 Token rounding’s training throughput

In Figure 16, we benchmark the token rounding’s MoE main kernel runtime (without router) against top- $K$ token choice routing. We focus on the iso-FLOPs setting by keeping $T$ , $n$ and $K$ constant. We linearly increase the number of experts $E$ while keeping $K$ constant to increase MoE sparsity. As we linearly increase $E$ , we observe a drop in TFLOPS for token-choice routing. This is due to the (1) tile quantization effect as the wasted FLOPs spent on padding roughly linearly increases with the MoE sparsity as shown in Figure 11 and (2) the linearly increased IO due to more expert weights. We observe a drop in both TC and TR drop FLOPs as we increase $E$ , but the drop is more pronounced for TC as shown in Figure 16. 

For the $3 ^ { \mathrm { r d } }$ and $4 ^ { \mathrm { t h } }$ column in the top row in Figure 16, an MoE model with 128 experts $( K / E = 1 / 6 4 )$ and $n = 1 k$ with 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-26/0390e304-b35b-4a5f-9ad3-e0d35edf7b5c/4ecb24c94738b44b0ba37c8fd09f7cc0f78ad54e26e1ee1d0b5281fbf49f6e08.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-26/0390e304-b35b-4a5f-9ad3-e0d35edf7b5c/846ee1b87b5c9df6b254a4482607b1120a6036f3f0df18a3fcd38a646582de6c.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-26/0390e304-b35b-4a5f-9ad3-e0d35edf7b5c/50c3783aa898de0ead575873f09f3870f8c4a621f03138601461b3a8d6509f86.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-26/0390e304-b35b-4a5f-9ad3-e0d35edf7b5c/a90a43eba86e5eedf6da432e8ff8dfb18bc935a6dd0ab1645c144113f6db9131.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-26/0390e304-b35b-4a5f-9ad3-e0d35edf7b5c/b21201279a9dc5a9fdde151495d0c013b298813f132f89925d5fe80890d9b1be.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-26/0390e304-b35b-4a5f-9ad3-e0d35edf7b5c/07f8fbb0b2b3708be30473960f7e5c4a4bb62530ea054a6404dd14e454316692.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-26/0390e304-b35b-4a5f-9ad3-e0d35edf7b5c/717811eebb0518d5d62821681695e8ec68f4d3ddcd20aa62517bd8bca39bcbd1.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-26/0390e304-b35b-4a5f-9ad3-e0d35edf7b5c/ffb13c255f94d987665a9c7a03527b2732bd784e811e76e0a72c8fd4eaf51216.jpg)



Figure 16: Forward & backward model TFLOPS for SonicMoE MoE kernels with different routing methods. We compare TR equipped with “nearest rounding to $M _ { \mathrm { t i l e } }$ -multiples via expert frequency” subroutine against TC top- $K$ routing. Configuration details are in Appendix G.


token rounding routing achieves $1 6 . 5 \%$ model TFLOPS30 improvement in forward and $6 . 1 \%$ in backward resulting in an end-to-end improvement of $9 . 4 \%$ . For the $3 ^ { \mathrm { r d } }$ and $4 ^ { \mathrm { t h } }$ column on the bottom row in Figure 16, when we have a MoE with 256 experts $( K / E = 1 / 1 2 8 )$ , token rounding routing achieves a $2 5 . 7 \%$ TFLOPS improvement in forward and $1 1 . 8 \%$ in backward resulting in an end-to-end improvement of $1 5 . 9 \%$ . In general, we observe that as we move to larger intermediate sizes (more compute-bound) and higher MoE sparsity, the gap between TR and TC top- $K$ becomes larger. 

This trend also holds with configurations from recent open-source MoEs in Figure 17. When we equip SonicMoE’s MoE kernel with TR router instead of TC top- $K$ router, we observe a larger relative speedup for highly sparse MoEs such as Qwen3-Next-80B-A3B-Thinking $( K / E = 1 0 / 5 1 2 )$ , where TR achieves $1 9 . 6 \%$ and $7 . 9 \%$ speedup over TC top- $K$ router during forward and backward pass respectively. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-26/0390e304-b35b-4a5f-9ad3-e0d35edf7b5c/65fa0c8358645f674fd7bd8218ed03f4d2f9a9aed935657d1d30a0b0509fa8ff.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-26/0390e304-b35b-4a5f-9ad3-e0d35edf7b5c/499ef6fc831baa1b843b01aed11ca5c306dc17c6304c1e2d59f1facfb390d315.jpg)



Figure 17: Forward & backward TFLOPS of a single MoE layer of SonicMoE equipped with different routing methods for different configurations ranging from 7B to 685B parameters on H100. The MoE configurations from left to right adopt the model size of OLMoE-1B-7B-0125, gpt-oss-20b, Kimi-Linear-48B-A3B-Base, Qwen3-Next-80B-A3B-Thinking, Qwen3-235B-A22B-Thinking-2507, and DeepSeek-V3.2-Exp (configurations identical to Figure 15). We compare TR equipped with “nearest rounding to $M _ { \mathrm { t i l e } }$ -multiples via expert frequency” subroutine against TC top- $K$ routing.


# 7 Conclusion

We present SonicMoE, a co-design solution that jointly optimizes MoE architecture and GPU kernels to address the training challenges of granular and sparse MoEs. Our contributions include: (1) a memory-efficient algorithm that minimizes 

activation size as MoEs become more fine-grained, (2) GPU kernels that overlap IO with computation for throughput improvement, and (3) tile-aware token rounding that yields additional speedup without quality loss. Future directions include extending to low-precision and microscaling formats (FP8, MXFP8, MXFP4) for further memory savings, and overlapping communication with computation in distributed settings like expert parallelism. We envision future model architecture designs that optimize for quality per compute hour rather than just quality per FLOP—jointly considering algorithmic and hardware efficiency. 

# Acknowledgment

We gratefully acknowledge the support of Schmidt Sciences AI2050 fellowship, the Google ML and Systems Junior Faculty Awards, and the Google Research Scholar program. We also thank the Princeton Language Intelligence program for the computing resources support. We thank Shawn Tan for his generous support on our experiments. We also thank Songlin Yang, Yilong Zhao, Bharat Runwal, Xinyu Yang, Andrew Sheinberg, Lijie Yang, Yongye Zhu, Zhuoqing Song and numerous anonymous reviewers for providing valuable feedback. 

# References



[1] K. E. Batcher. “Sorting networks and their applications”. In: Proceedings of the April 30–May 2, 1968, Spring Joint Computer Conference. AFIPS ’68 (Spring). Atlantic City, New Jersey: Association for Computing Machinery, 1968, 307–314. ISBN: 9781450378970. DOI: 10.1145/1468075.1468121. URL: https://doi.org/10.1145/ 1468075.1468121. 





[2] Loubna Ben Allal, Anton Lozhkov, Guilherme Penedo, Thomas Wolf, and Leandro von Werra. SmolLM-Corpus. 2024. URL: https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus. 





[3] Vincent-Pierre Berges, Barlas Oguz, Daniel Haziza, Wen-tau Yih, Luke Zettlemoyer, and Gargi Ghosh. “Memory ˘ layers at scale”. In: arXiv preprint arXiv:2412.09764 (2024). 





[4] Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. “PIQA: Reasoning about Physical Commonsense in Natural Language”. In: Thirty-Fourth AAAI Conference on Artificial Intelligence. 2020. 





[5] Aidan Clark, Diego de Las Casas, Aurelia Guy, Arthur Mensch, Michela Paganini, Jordan Hoffmann, Bogdan Damoc, Blake Hechtman, Trevor Cai, Sebastian Borgeaud, et al. “Unified scaling laws for routed language models”. In: International conference on machine learning. PMLR. 2022, pp. 4057–4086. 





[6] Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. “BoolQ: Exploring the Surprising Difficulty of Natural Yes/No Questions”. In: Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers). 2019, pp. 2924–2936. 





[7] Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. “Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge”. In: arXiv:1803.05457v1 (2018). 





[8] A Feder Cooper, Wentao Guo, Duc Khiem Pham, Tiancheng Yuan, Charlie Ruan, Yucheng Lu, and Christopher M De Sa. “Coordinating distributed example orders for provably accelerated training”. In: Advances in Neural Information Processing Systems 36 (2023), pp. 56198–56210. 





[9] NVIDIA Corporation. 2025. URL: https://docs.nvidia.com/cutlass/media/docs/cpp/cutlass_ 3x_backwards_compatibility.html. 





[10] Bobby Costin, Timor Averbuch, Dhruv Pai, Nathan Chen, and Ben Keigwin. “MoMoE: Memory optimized Mixture of Experts”. In: Tilde Research Blog (July 2025). Blog post. URL: https://www.tilderesearch.com/blog/ momoe. 





[11] DeepSeek-AI. DeepSeek-V3.2-Exp: Boosting Long-Context Efficiency with DeepSeek Sparse Attention. 2025. 





[12] DeepSeek-AI, Aixin Liu, Bei Feng, Bing Xue, and et al. DeepSeek-V3 Technical Report. 2024. arXiv: 2412.19437 [cs.CL]. URL: https://arxiv.org/abs/2412.19437. 





[13] Bert Dobbelaere. 2025. URL: https://bertdobbelaere.github.io/sorting_networks.html. 





[14] Raaz Dwivedi and Lester Mackey. “Kernel thinning”. In: Journal of Machine Learning Research 25.152 (2024), pp. 1–77. 





[15] Efficient GEMM in CUDA. 2025. URL: https://docs.nvidia.com/cutlass/media/docs/cpp/ efficient_gemm.html#hopper-warp-specialization. 





[16] William Fedus, Barret Zoph, and Noam Shazeer. Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity. 2022. arXiv: 2101.03961 [cs.LG]. URL: https://arxiv.org/abs/2101. 03961. 





[17] Rong Fu, Weihan Cao, Jianfei Gao, Minxi Jin, Hui Wang, et al. “TMA-Adaptive FP8 Grouped GEMM: Eliminating Padding Requirements in Low-Precision Training and Inference on Hopper”. In: ES-FoMo III: 3rd Workshop on Efficient Systems for Foundation Models. 2025. 





[18] Trevor Gale, Deepak Narayanan, Cliff Young, and Matei Zaharia. “Megablocks: Efficient sparse training with mixtureof-experts”. In: Proceedings of Machine Learning and Systems 5 (2023), pp. 288–304. 





[19] IBM Granite. Granite 3.1 Language Models. https://github.com/ibm- granite/granite- 3.1- language-models. GitHub repository. 2024. 





[20] Horace He and Thinking Machines. Defeating Nondeterminism in LLM Inference. 2025. URL: https://thinkingmachines. ai/blog/defeating-nondeterminism-in-llm-inference/#true-on-policy-rl. 





[21] Xu Owen He. “Mixture of a million experts”. In: arXiv preprint arXiv:2407.04153 (2024). 





[22] Quzhe Huang, Zhenwei An, Nan Zhuang, Mingxu Tao, Chen Zhang, Yang Jin, Kun Xu, Liwei Chen, Songfang Huang, and Yansong Feng. “Harder Task Needs More Experts: Dynamic Routing in MoE Models”. In: Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024, pp. 12883–12895. 





[23] Zihao Huang, Qiyang Min, Hongzhi Huang, Yutao Zeng, Defa Zhu, Ran Guo, et al. “Ultra-Sparse Memory Network”. In: The Thirteenth International Conference on Learning Representations. 2025. 





[24] Matt Gardner Johannes Welbl Nelson F. Liu. “Crowdsourcing Multiple Choice Science Questions”. In: 2017. 





[25] Team Kimi, Yifan Bai, Yiping Bao, Guanduo Chen, and et al. Kimi K2: Open Agentic Intelligence. 2025. arXiv: 2507.20534 [cs.LG]. URL: https://arxiv.org/abs/2507.20534. 





[26] Jakub Krajewski, Jan Ludziejewski, Kamil Adamczewski, Maciej Pioro, Michał Krutul, Szymon Antoniak, Kamil ´ Ciebiera, Krystian Krol, Tomasz Odrzyg ´ o´zd´ z, Piotr Sankowski, et al. “Scaling laws for fine-grained mixture of ´ experts”. In: arXiv preprint arXiv:2402.07871 (2024). 





[27] Chuck L Lawson, Richard J. Hanson, David R Kincaid, and Fred T. Krogh. “Basic linear algebra subprograms for Fortran usage”. In: ACM Transactions on Mathematical Software (TOMS) 5.3 (1979), pp. 308–323. 





[28] Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. “GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding”. In: International Conference on Learning Representations. 2024. 





[29] Yucheng Lu, Wentao Guo, and Christopher M De Sa. “Grab: Finding provably better data permutations than random reshuffling”. In: Advances in Neural Information Processing Systems 35 (2022), pp. 8969–8981. 





[30] Weile Luo, Ruibo Fan, Zeyu Li, Dayou Du, Hongyuan Liu, Qiang Wang, and Xiaowen Chu. “Dissecting the NVIDIA Hopper Architecture through Microbenchmarking and Multiple Level Analysis”. In: arXiv preprint arXiv:2501.12084 (2025). 





[31] Microsoft. Announcing the Availability of PHI-3.5 MoE in Azure AI Studio and GitHub. https://techcommunity. microsoft.com/blog/azure-ai-foundry-blog/announcing-the-availability-of-phi-3-5-moe-in-azure-ai-studio-and-github/4256278. Microsoft Tech Community. 2024. 





[32] Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. “Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering”. In: EMNLP. 2018. 





[33] Mayank Mishra. LM Engine: A Hyper-Optimized Library for Pretraining and Finetuning. June 2024. URL: https: //github.com/ibm/lm-engine. 





[34] Mistral. Mixtral of Experts: A high quality Sparse Mixture-of-Experts. https://mistral.ai/news/mixtralof-experts. Mistral AI News. 2024. 





[35] Niklas Muennighoff, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Jacob Morrison, Sewon Min, Weijia Shi, Evan Pete Walsh, Oyvind Tafjord, Nathan Lambert, et al. “OLMoE: Open Mixture-of-Experts Language Models”. In: The Thirteenth International Conference on Learning Representations. 2025. 





[36] NVIDIA. NVIDIA H100 Tensor Core GPU Architecture: Exceptional Performance, Scalability, and Security for the Data Center. Whitepaper V1.01. Grace Hopper “Hopper” Architecture. NVIDIA, 2022. URL: https://www. advancedclustering . com / wp - content / uploads / 2022 / 03 / gtc22 - whitepaper - hopper . pdf. 





[37] NVIDIA. CUTLASS: CUDA Templates for Linear Algebra Subroutines. https://github.com/NVIDIA/ cutlass. Version 4.2.0, Accessed: 2025-09-19. 2025. 





[38] NVIDIA. NVIDIA Blackwell Architecture Technical Brief. Whitepaper. Blackwell Architecture. NVIDIA, 2025. URL: https://resources.nvidia.com/en-us-blackwell-architecture?ncid $\underline { { \underline { { \mathbf { \Pi } } } } } =$ no-ncid. 





[39] NVIDIA. NVIDIA CUTLASS Documentation. 2025. URL: https://docs.nvidia.com/cutlass/media/ docs/pythonDSL/cute_dsl_general/dsl_introduction.html. 





[40] OpenAI. gpt-oss-120b & gpt-oss-20b Model Card. 2025. arXiv: 2508.10925 [cs.CL]. URL: https://arxiv. org/abs/2508.10925. 





[41] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zach DeVito, Martin Raison, ¨ 





Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. PyTorch: An Imperative Style, High-Performance Deep Learning Library. 2019. arXiv: 1912.01703 [cs.LG]. URL: https: //arxiv.org/abs/1912.01703. 





[42] Team Qwen. Qwen3 Technical Report. 2025. arXiv: 2505.09388 [cs.CL]. URL: https://arxiv.org/ abs/2505.09388. 





[43] QwenLM. Qwen3: Think Deeper, Act Faster. https://qwenlm.github.io/blog/qwen3/. Official Blog. 2025. 





[44] David Raposo, Sam Ritter, Blake Richards, Timothy Lillicrap, Peter Conway Humphreys, and Adam Santoro. “Mixture-of-depths: Dynamically allocating compute in transformer-based language models”. In: arXiv preprint arXiv:2404.02258 (2024). 





[45] Colfax Research. CUTLASS Tutorial: Writing GEMM Kernels Using Tensor Memory For NVIDIA Blackwell GPUs. Accessed: 2025-09-21. 2024. URL: https://research.colfax- intl.com/cutlass- tutorialwriting-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/. 





[46] Melissa Roemmele, Cosmin Adrian Bejan, and Andrew S Gordon. “Choice of plausible alternatives: An evaluation of commonsense causal reasoning”. In: 2011 AAAI Spring Symposium Series. 2011. 





[47] Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. “WinoGrande: An Adversarial Winograd Schema Challenge at Scale”. In: Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 34. 2020, pp. 8732–8740. 





[48] Maarten Sap, Hannah Rashkin, Derek Chen, Ronan LeBras, and Yejin Choi. “SocialIQA: Commonsense Reasoning about Social Interactions”. In: Conference on Empirical Methods in Natural Language Processing. 2019. 





[49] Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, and Tri Dao. FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision. 2024. arXiv: 2407.08608 [cs.LG]. URL: https: //arxiv.org/abs/2407.08608. 





[50] Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. 2017. arXiv: 1701.06538 [cs.LG]. URL: https://arxiv.org/abs/1701.06538. 





[51] Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. “Megatronlm: Training multi-billion parameter language models using model parallelism”. In: arXiv preprint arXiv:1909.08053 (2019). 





[52] Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. “CommonsenseQA: A Question Answering Challenge Targeting Commonsense Knowledge”. In: Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers). Minneapolis, Minnesota: Association for Computational Linguistics, June 2019, pp. 4149–4158. DOI: 10.18653/v1/N19-1421. arXiv: 1811.00937 [cs]. URL: https://aclanthology.org/N19-1421. 





[53] Shawn Tan, Yikang Shen, Rameswar Panda, and Aaron Courville. “Scattered mixture-of-experts implementation”. In: arXiv preprint arXiv:2403.08245 (2024). 





[54] Databricks The Mosaic Research Team. Introducing DBRX: A New State-of-the-Art Open LLM. https://www. databricks.com/blog/introducing-dbrx-new-state-art-open-llm. Databricks Blog, March 27, 2024. 2024. 





[55] Changxin Tian, Kunlong Chen, Jia Liu, Ziqi Liu, Zhiqiang Zhang, and Jun Zhou. “Towards greater leverage: Scaling laws for efficient mixture-of-experts language models”. In: arXiv preprint arXiv:2507.17702 (2025). 





[56] Philippe Tillet, H. T. Kung, and David Cox. “Triton: an intermediate language and compiler for tiled neural network computations”. In: Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages. MAPL 2019. Phoenix, AZ, USA: Association for Computing Machinery, 2019, 10–19. ISBN: 9781450367196. DOI: 10.1145/3315508.3329973. URL: https://doi.org/10.1145/3315508. 3329973. 





[57] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention Is All You Need. 2017. arXiv: 1706.03762 [cs.CL]. URL: https://arxiv.org/ abs/1706.03762. 





[58] Lean Wang, Huazuo Gao, Chenggang Zhao, Xu Sun, and Damai Dai. “Auxiliary-loss-free load balancing strategy for mixture-of-experts”. In: arXiv preprint arXiv:2408.15664 (2024). 





[59] Lei Wang, Yu Cheng, Yining Shi, Zhengju Tang, Zhiwen Mo, Wenhao Xie, Lingxiao Ma, Yuqing Xia, Jilong Xue, Fan Yang, et al. “TileLang: A Composable Tiled Programming Model for AI Systems”. In: arXiv preprint arXiv:2504.17577 (2025). 





[60] Less Wright and Adnan Hoque. Deep dive on Cutlass Ping-Pong Gemm Kernel. 2024. URL: https://pytorch. org/blog/cutlass-ping-pong-gemm-kernel/. 





[61] Less Wright and Adnan Hoque. Deep Dive on CUTLASS Ping-Pong GEMM Kernel. Accessed: 2025-09-21. 2024. URL: https://docs.pytorch.org/blog/cutlass-ping-pong-gemm-kernel/. 





[62] Xi Xie, Yuebo Luo, Hongwu Peng, and Caiwen Ding. “RTop-K: Ultra-Fast Row-Wise Top-K Selection for Neural Network Acceleration on GPUs”. In: The Thirteenth International Conference on Learning Representations. 2025. URL: https://openreview.net/forum?id $\underline { { \underline { { \mathbf { \Pi } } } } } =$ PHg4rAXFVH. 





[63] Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. “HellaSwag: Can a Machine Really Finish Your Sentence?” In: Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics. 2019. 





[64] Aohan Zeng, Xin Lv, Qinkai Zheng, Zhenyu Hou, Bin Chen, Chengxing Xie, Cunxiang Wang, Da Yin, Hao Zeng, Jiajie Zhang, et al. “Glm-4.5: Agentic, reasoning, and coding (arc) foundation models”. In: arXiv preprint arXiv:2508.06471 (2025). 





[65] Zhiyuan Zeng, Qipeng Guo, Zhaoye Fei, Zhangyue Yin, Yunhua Zhou, Linyang Li, Tianxiang Sun, Hang Yan, Dahua Lin, and Xipeng Qiu. “Turn Waste into Worth: Rectifying Top-k Router of MoE”. In: EMNLP. 2024. 





[66] Zihao Zeng, Yibo Miao, Hongcheng Gao, Hao Zhang, and Zhijie Deng. “AdaMoE: Token-Adaptive Routing with Null Experts for Mixture-of-Experts Language Models”. In: Findings of the Association for Computational Linguistics: EMNLP 2024. Ed. by Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen. Miami, Florida, USA: Association for Computational Linguistics, Nov. 2024, pp. 6223–6235. DOI: 10.18653/v1/2024.findings-emnlp.361. URL: https://aclanthology.org/2024.findings-emnlp.361/. 





[67] Yu Zhang, Zongyu Lin, Xingcheng Yao, and et al. Kimi Linear: An Expressive, Efficient Attention Architecture. 2025. arXiv: 2510.26692 [cs.CL]. 





[68] Chenggang Zhao, Chengqi Deng, Chong Ruan, Damai Dai, Huazuo Gao, Jiashi Li, Liyue Zhang, Panpan Huang, Shangyan Zhou, Shirong Ma, Wenfeng Liang, Ying He, Yuqing Wang, Yuxuan Liu, and Y.X. Wei. “Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures”. In: Proceedings of the 52nd Annual International Symposium on Computer Architecture. ISCA ’25. New York, NY, USA: Association for Computing Machinery, 2025, 1731–1745. ISBN: 9798400712616. DOI: 10.1145/3695053.3731412. URL: https://doi.org/10.1145/3695053.3731412. 





[69] Chenggang Zhao, Liang Zhao, Jiashi Li, and Zhean Xu. DeepGEMM: clean and efficient FP8 GEMM kernels with fine-grained scaling. 2025. URL: https://github.com/deepseek-ai/DeepGEMM. 





[70] Chenggang Zhao, Shangyan Zhou, Liyue Zhang, Chengqi Deng, Zhean Xu, Yuxuan Liu, Kuai Yu, Jiashi Li, and Liang Zhao. DeepEP: an efficient expert-parallel communication library. https://github.com/deepseekai/DeepEP. 2025. 





[71] Yanqi Zhou, Tao Lei, Hanxiao Liu, Nan Du, Yanping Huang, Vincent Zhao, Andrew M Dai, Quoc V Le, James Laudon, et al. “Mixture-of-experts with expert choice routing”. In: Advances in Neural Information Processing Systems 35 (2022), pp. 7103–7114. 





[72] Barret Zoph, Irwan Bello, Sameer Kumar, Nan Du, Yanping Huang, Jeff Dean, Noam Shazeer, and William Fedus. “St-moe: Designing stable and transferable sparse expert models”. In: arXiv preprint arXiv:2202.08906 (2022). 



# Appendix

We provide a table listing all notations and their explanations in Table 4. In Section B, we compare SonicMoE’s kernel design with other open-source MoE kernel designs. In Section C, we elaborate on SonicMoE’s computational path for $d S$ and $d H$ that does not use $Y$ and $d Y$ . In Section C.1, we justify SonicMoE’s computational path for $d S$ is both activation memory and computationally-efficient. In Section D, we illustrate SonicMoE’s up-projection backward is included in Algorithm 5. In Section E, we present ablation studies of training throughput for SonicMoE’s MoE computation kernels to examine the impact of each design choice made for SonicMoE. In Section F, we assess the quality improvements of MoE models trained by varying expert granularity. We then focus on various ablation studies on our token rounding routing algorithm to assess the quality difference of the trained MoE models from the choice of rounding subroutine. We also study the effect of microbatch size $T$ and tile size $M _ { \mathrm { t i l e } }$ on token rounding. In Section G, we describe the configurations for benchmarking the memory and training throughput. In Section H, we include the details of model training and the evaluation setup. 

# A Notations

In Table 4, we describe the notations used in this paper. 


Table 4: Notations and their explanations


<table><tr><td>Notations</td><td>Explanation</td></tr><tr><td>T</td><td>number of tokens in a microbatch</td></tr><tr><td>d</td><td>model embedding dimension (hidden size)</td></tr><tr><td>n</td><td>each expert&#x27;s intermediate dimension</td></tr><tr><td>E</td><td>total number of experts</td></tr><tr><td>K</td><td>number of activated experts</td></tr><tr><td>ρ</td><td>ρ = K/E represents MoE activation ratio</td></tr><tr><td>Te</td><td>Te = Ee∈[E][Te] = Tρ represents the expected number of received tokens in each microbatch by expert e</td></tr><tr><td>G</td><td>G = d/n represents the MoE expert granularity. Greater G means a MoE is more fine-grained</td></tr><tr><td>M, N, K</td><td>Dimensions for GEMM in CUTLASS. We define A ∈ RM×K, B ∈ RKN, and C ∈ RM×N for AB = C</td></tr><tr><td>Mtile, Ntile, Ktile</td><td>tile size of M, N, K dimension for a single GEMM tile</td></tr><tr><td>Re</td><td>tile quantization residue Re := Te mod Mtile</td></tr><tr><td>X</td><td>X ∈ RT×d, input token embeddings for an MoE layer</td></tr><tr><td>W1</td><td>W1 ∈ RE×d×2n, weight of up projection</td></tr><tr><td>W2</td><td>W2 ∈ RE×n×d, weight of down projection</td></tr><tr><td>π</td><td>π ∈ {0,1}T×E, a binary-valued matrix where πt,e represents if token t is routed to expert e</td></tr><tr><td>S</td><td>S ∈ RT×E, router scores. In practice, we only materialize the sparsified S instead of the full S</td></tr><tr><td>H</td><td>H ∈ TK×2n, output of up projection</td></tr><tr><td>A</td><td>A ∈ TK×n, output of SwiGLU</td></tr><tr><td>Y</td><td>Y ∈ TK×d, output of down projection</td></tr><tr><td>O</td><td>O ∈ TK×d, output of expert aggregation, also output of the entire MoE layer</td></tr><tr><td>dO</td><td>dO ∈ RT×d, activation gradient for O</td></tr><tr><td>dA&#x27;</td><td>dA&#x27; = dO W2 ∈ RT×n, GEMM output of dO and W2. Intermediate result for computing dA and dS</td></tr><tr><td>dA</td><td>dA = Broadcast(s) dA&#x27; ∈ RT×n, activation gradient for A</td></tr><tr><td>dY</td><td>dY = Broadcast(s) dO ∈ TK×d, activation gradient for Y. dY is not used in SonicMoE.</td></tr><tr><td>dS</td><td>dS ∈ RT×E, activation gradient for S</td></tr><tr><td>A&#x27;</td><td>A&#x27; = Broadcast(s) A ∈ RT×n, intermediate result and input for computing dW2</td></tr><tr><td>dH</td><td>dH ∈ RT×2n, activation gradient for H</td></tr><tr><td>dX̅</td><td>dX̅ ∈ RTK×d, activation gradient for X before aggregation</td></tr><tr><td>dX</td><td>dX ∈ RT×d, activation gradient for X after aggregation</td></tr><tr><td>dW1</td><td>dW1 ∈ RE×d×2n, weight gradient for W1</td></tr><tr><td>dW2</td><td>dW1 ∈ RE×n×d, weight gradient for W2</td></tr><tr><td>A kernel</td><td>forward up-proj kernel</td></tr><tr><td>Y kernel</td><td>forward down-proj kernel</td></tr><tr><td>O kernel</td><td>forward expert aggregation kernel where each token aggregates all routed expert&#x27;s result as the final forward output</td></tr><tr><td>dH kernel</td><td>backward down-proj activation gradient kernel</td></tr><tr><td>dW2 kernel</td><td>backward down-proj weight gradient kernel</td></tr><tr><td>dX̅kernel</td><td>backward up-proj activation gradient kernel</td></tr><tr><td>dW1 kernel</td><td>backward up-proj weight gradient kernel</td></tr><tr><td>dX kernel</td><td>backward expert aggregation kernel where each token aggregates the routed expert&#x27;s dX̅</td></tr><tr><td>[fet]Mtile, [fe]MTile</td><td>MTile-rounded multiples of expert frequency fe. [fe]MTile = [fe/Mtile]·MTile [fe]MTile = [fe/Mtile]·MTile</td></tr><tr><td>[S]MTile, [fe]MTile</td><td>[fe]MTile is MTile-rounded multiples of expert frequency fe. [fe]MTile ∈ { [fe]MTile, [fe]MTile}, and [S]MTile is the score after rounding in Algorithm 4.</td></tr></table>

# B SonicMoE’s comparison with existing MoE kernel design

Existing efficient MoE kernels also frame MoE computation as a Grouped GEMM, but their ingredients are different from SonicMoE. Here we provide an overview (but not a complete list) of key differences: 

• ScatterMoE (Tan et al. $2 0 2 4 ) ^ { 3 1 }$ implements gather fusion for varlen-M Grouped GEMM but not for varlen-K Grouped GEMM. ScatterMoE also does not overlap MMA computation with memory IO. Moreover, ScatterMoE is also built on older versions of Triton where TMA is not supported. ScatterMoE computes $d S$ as $d S = \langle d O , Y \rangle$ which requires caching $Y$ . This results in large IO cost and activation memory requirement. ScatterMoE’s both forward and backward pass have limited fusion and hence it is much slower than SonicMoE, especially for backward computation. 

• MoMoE (Costin et al. 2025)32 also implements the gather fusion for varlen-M but not varlen-K Grouped GEMM similar to ScatterMoE. Although fused with up-proj activation gradient, the $d S$ computation still utilizes $d S = \langle d O , Y \rangle$ Similar to ScatterMoE, MoMoE does not use TMA for IO. The scatter operation in MoMoE is (much) slower than SonicMoE, as shown in Figure 22. 

• MegaBlocks (Gale et al. 2023) has multiple MoE implementations and we focus on ParallelDroplessMLP33 which is built on top of block-sparse matrix multiplication34. ParallelDroplessMLP first gathers and pads the tokens and then launches block-sparse GEMM for up and down-proj. Then, it launches a scatter kernel before reducing across the expert results. These sparse matrix multiplications usually take a longer time than the highly-optimized Grouped GEMM, as shown in Figure 6, and the gather and scatter kernel have a total IO cost of $8 T K d$ bytes which can be a bottleneck for fine-grained MoEs. We consider MegaBlocks’s ParallelDroplessMLP as a block-sparse GEMM baseline in our benchmark and find that MoE implemented via Grouped GEMMs often have a higher training throughput than MoEs implemented via block-sparse GEMM. 

• Megatron-LM (Shoeybi et al. 2019) also have multiple MoE implementations and we focus on GroupedMLP35, which uses Grouped GEMM36 from the CUTLASS library (Corporation 2025) with JIT epilogue fusion as the GEMM backend. Similar to DeepGEMM, GroupedMLP does not fuse gather with the prologue (it assumes contiguouslypacked inputs). A recent memory-efficient patch37 fuses $S$ weighting with SwiGLU computation during forward, and during backward which allows the PyTorch autograd engine (Paszke et al. 2019) to follow similar computational path as SonicMoE. 

Megatron-LM also implements TEGroupedMLP38 which launches 4 CUDA streams to execute a list of GEMM (without contiguously-packed inputs, and without a persistent tile scheduler). In this case, each expert independently launches a new GEMM kernel leading to “bubbles” on the CUDA streams. This leads to underutilization of the GPU resources. We empirically find that TEGroupedMLP runs slower slower than GroupedMLP and so we use GroupedMLP across all benchmarks. 

• DeepGEMM (Zhao et al. 2025b) design a Grouped GEMM kernel for contiguously-packed inputs. They also don’t implement any other fusion for SM90 (Hopper) BF16 Grouped GEMM. DeepGEMM specializes more on distributed training with expert parallelism (Zhao et al. 2025c), and it is common to launch a separate all2all kernel (Lepikhin et al. 2024) which is then followed by a contiguous Grouped GEMM. DeepGEMM SM90 BF16 kernel also assumes that each expert receives a multiple of $M _ { \mathrm { t i l e } }$ tokens as it does not implement the TMA tensor descriptor online update during the Grouped GEMM computation. DeepGEMM’s BF16 GEMM on $\mathbf { S } \mathbf { M } 9 0 ^ { 1 1 }$ also does not employ Ping-Pong scheduling. 

Additionally, ScatterMoE and MoMoE are both implemented in Triton (Tillet, Kung, and Cox 2019) for the ease of development at the expense of losing full programmability of the asynchronous compute and memory IO of Hopper and Blackwell GPUs (NVIDIA 2022, 2025b). For example, they cannot implement fine-grained control of asynchronous load and store during the GEMM’s epilogue. They also cannot overlap MMA with heavy epilogue operations using Ping-Pong scheduling. It becomes increasingly important to overlap IO operations in epilogue when the GEMM computations are small 

in size (as in the case of fine-grained MoEs) to achieve high GPU utilization. 

# C Gradient computation

For an expert e, let 

$$
X _ {e} \in \mathbb {R} ^ {T _ {e} \times d}, \quad W _ {1, e} \in \mathbb {R} ^ {d \times 2 n}, \quad W _ {2, e} \in \mathbb {R} ^ {n \times d} \tag {5}
$$

The forward activation computation is given by: 

$$
H _ {e} = X _ {e} W _ {1, e} \in \mathbb {R} ^ {T _ {e} \times 2 n}, \quad A _ {e} = \operatorname {S w i G L U} \left(H _ {e}\right) \in \mathbb {R} ^ {T _ {e} \times n}, \quad Y _ {e} = A _ {e} W _ {2, e} \in \mathbb {R} ^ {T _ {e} \times d}. \tag {6}
$$

The token aggregation with scores $S = \{ s _ { t , e } \}$ is given by 

$$
O _ {t} = \sum_ {e} s _ {t, e} Y _ {e, t}, \quad d O _ {t} \in \mathbb {R} ^ {T \times d} \text {a s t h e g a t h e r e d r e s u l t s f r o m} d O. \tag {7}
$$

We know 

$$
d Y _ {e, t} = s _ {t, e} d O _ {t} \quad \Longrightarrow \quad d Y _ {e} = \operatorname {B r o a d c a s t} \left(\mathbf {s} _ {e}\right) d O _ {e}. \tag {8}
$$

Define the Grouped GEMM output as $d A _ { e } ^ { \prime } : = d O _ { e } W _ { 2 , e } ^ { \top } \in \mathbb { R } ^ { T _ { e } \times n }$ 

Then from Equation 8 

$$
d A _ {e} = d Y _ {e} W _ {2, e} ^ {\top} = \operatorname {B r o a d c a s t} (\mathbf {s} _ {e}) d A _ {e} ^ {\prime}. \tag {9}
$$

The activation gradient for score $d S$ is39 

$$
\boxed {d S _ {t, e} = \left\langle d O _ {t}, Y _ {e, t} \right\rangle = \left\langle d O _ {t} W _ {2, e} ^ {\top}, A _ {e, t} \right\rangle = \left\langle d A _ {e, t} ^ {\prime}, A _ {e, t} \right\rangle .} \tag {10}
$$

In addition, we can derive $d H _ { e }$ from $d A _ { e }$ and $A _ { e }$ (recomputed from $H _ { e }$ ) as: 

$$
d H _ {e} = \mathrm {d S w i G L U} \left(d A _ {e}, H _ {e}\right). \tag {11}
$$

Using Equation 8, 

$$
d W _ {2, e} = A _ {e} ^ {\top} d Y _ {e} = A _ {e} ^ {\top} \left(\operatorname {B r o a d c a s t} \left(\mathbf {s} _ {e}\right) d O _ {e}\right) = \left(\underbrace {\operatorname {B r o a d c a s t} \left(\mathbf {s} _ {e}\right) A _ {e}} _ {A _ {e} ^ {\prime}}\right) ^ {\top} d O _ {e}. \tag {12}
$$

# C.1 Computational choices for $d S$

If we do not implement custom kernels and rely solely on PyTorch’s autograd (AD) engine, we can add the expert weighting $( S )$ either (1) before down-proj forward or (2) after down-proj forward. Both yield identical results for forward and backward, but the computation for $d S$ is different. For (1), we need to compute $\left. d A _ { e , t } ^ { \prime } , \ A _ { e , t } \right.$ which is used by SonicMoE and Megatron40. MoMoE41, ScatterMoE42, and MegaBlocks4344 compute $\langle d O _ { t }$ , $Y _ { e , t } \rangle$ as required in (2). 

Note that $d S$ can be computed as any of $d S _ { t , e } = \langle d A _ { e , t } ^ { \prime } , A _ { e , t } \rangle = \langle d O _ { t } , Y _ { e , t } \rangle $ , however computing it as $d S _ { t , e } ~ =$ $\langle d A _ { e , t } ^ { \prime } , \ A _ { e , t } \rangle$ $A _ { e , t } \rangle$ is a computationally and activation memory-efficient choice due to the following reasons: 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-26/0390e304-b35b-4a5f-9ad3-e0d35edf7b5c/cafb6ffa7f91bc8dae0cf9cd9d9a9a4f5cdb821856f873bf2eaff42a09520932.jpg)



Figure 18: Index prefetching strategies for gathering on M dim of varlen-M Grouped GEMM (left) and K dim of varlen-K Grouped GEMM (right) on H100 GPU. For gather on M dim (left), we let each thread independently prefetch indices to their own registers before the mainloop. For gather on K dim (right), we create a buffer on SMEM and let 4 producer warps cooperatively prefetch indices to SMEM and each producer thread read from this SMEM buffer to their registers.


• Additional HBM traffic (0 vs. 2T Kd bytes): $\left. d A _ { e , t } ^ { \prime } , \ A _ { e , t } \right.$ $A _ { e , t } \rangle$ requires $d A _ { e , t } ^ { \prime }$ and $A _ { e , t }$ are already computed during the dH kernel, so we can avoid extra unnecessary loads. 

• Extra cached activation memory (0 vs. 2T Kd bytes): One of the reasons why the cached activation memory for ScatterMoE, MoMoE and MegaBlocks fails to stay constant w.r.t. expert granularity is the required caching of $Y$ for computing dS. 

• Parallel reduction rounds $( \log _ { 2 } ( n )$ vs. $\log _ { 2 } ( d ) )$ : 
dA′e,t, Ae,t reduces over $n$ while $\langle d O _ { t } , Y _ { e , t } \rangle$ reduces over $d$ This difference saves at least $\log _ { 2 } ( d / n )$ rounds of reduction. 

# D SonicMoE algorithm

In this section, we present the referenced diagrams and algorithms in the main paper. We further describe the index prefetching strategies incorporated in our group GEMM kernel on Hopper GPUs, as illustrated in Figure 18. 


Algorithm 5 SonicMoE’s MoE kernel backward pass of up-proj.


Input: $X, \pi, W_1, dH$ Output: $dX, dW_1$ .  
Up-proj act $d\tilde{X}$ kernel $(dH, W_1) \to d\tilde{X}$ : // varlen-M Grouped GEMM  
Parallel for $e \in [E]$ do $dH_e, W_{1,e} \gets \text{load}(dH_e, W_{1,e})$ $d\tilde{X}_e \gets dH_e W_{1,e}^\top$ $d\tilde{X}_e \gets \text{store}(d\tilde{X}_e)$ Up-proj weight $dW_1$ kernel $(X, dH, \pi) \to dW_1$ : // Gather + varlen-K Grouped GEMM  
Parallel for $e \in [E]$ do $X, \pi::e, dH_e \gets \text{load}(X, \pi::e, dH_e)$ $X_e \gets \text{Gather}(X, \pi::e)$ $dW_{1,e} \gets X_e^\top dH_e$ $dW_{1,e} \gets \text{store}(dW_{1,e})$ Expert aggregation $dX$ kernel $(d\tilde{X}, \pi) \to dX$ : // Gather and sum  
Parallel for $t \in [T]$ do $d\tilde{X}, \pi_{t,e} \gets \text{load}(d\tilde{X}, \pi_{t,e})$ $dX_t \gets \sum_{e \in [E]} \pi_{t,e} d\tilde{X}_{e,t}$ $dX_t \gets \text{store}(dX_t)$ 

# E SonicMoE’s ablation study on kernel-level throughput

In this section, we present kernel-level ablation studies on training throughput to examine the impact of each of the implemented features on SonicMoE. In Section E.1, we investigate on the Grouped GEMM throughput with and without the gather fusion on Hopper GPUs. In Section E.2, we profile the memory bandwidth of expert aggregation kernels. In Section E.3, we compare SonicMoE’s top- $K$ kernels with other efficient top- $K$ implementations. 

# E.1 Grouped GEMM on Hopper GPUs

We also benchmark SonicMoE’s base Grouped GEMM kernel with both contiguously-packed inputs and gathered inputs without any epilogue fusion on H100 GPU. For contiguously-packed inputs, we mainly compare with DeepGEMM kernels (sm90 m grouped bf16 gemm contiguous45 and sm90 bf16 k grouped gemm46). We note that at the time of writing, DeepGEMM’s k Grouped GEMM kernel only accepts the form of $D = A B + C$ for $\mathrm { G E M M ^ { 4 7 } }$ where for correctness, we use a zero-filled FP32 $C$ weight gradient buffer as accumulator input, but we use $C$ as uninitialized weight gradient buffer during benchmarking. For inputs requiring a gather operation, we mainly compare with ScatterMoE and MoMoE. We also benchmark cuBLAS dense BMM (CUDA toolkit v12.9) assuming each expert receives the same amount of tokens. 

Grouped GEMM with contiguously-packed inputs. In Figure 19, we compare SonicMoE with DeepGEMM on varlen-M Grouped GEMM on H100 GPU without any other fusion. We also benchmark cuBLAS dense BMM (perfect load balance and no tensormap update needed) as a reference for the TFLOPS upper bound for Grouped GEMM. We find that SonicMoE’s up-proj has $2 . 7 \%$ higher TFLOPS while down-proj has $1 0 . 0 \%$ higher TFLOPS than DeepGEMM. SonicMoE’s relative TFLOPS speedup over DeepGEMM is $5 7 . 4 \%$ , $1 4 . 0 \%$ , and $5 . 3 \%$ for 30B down-proj config. We use Ping-Pong scheduling for down-proj with $n < 1 0 2 4$ while DeepGEMM (Zhao et al. 2025b) uses cooperative scheduling (Efficient GEMM in CUDA 2025; Zhao et al. 2025b). 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-26/0390e304-b35b-4a5f-9ad3-e0d35edf7b5c/1aaebb8d623262ad5a3f09ac268cd03a4f19786398f8bfc40b7294ff0990eb21.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-26/0390e304-b35b-4a5f-9ad3-e0d35edf7b5c/b97470f9cba049ec437ed924b605f305502d49bd6cd705dc0d1d18fbe47a3398.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-26/0390e304-b35b-4a5f-9ad3-e0d35edf7b5c/cc963f5fb13911d081dd7100e2e4b19d9c3e30e6abb814ad68f82ac7fce965be.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-26/0390e304-b35b-4a5f-9ad3-e0d35edf7b5c/bea6a07b945ff5980b0175a7c948ddb190166c07bd25141481dfb1e8b91a9f63.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-26/0390e304-b35b-4a5f-9ad3-e0d35edf7b5c/0d81416f67134110a7bf06ac95003532dfa7421e9403dcc621b5d160d64173d4.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-26/0390e304-b35b-4a5f-9ad3-e0d35edf7b5c/cc8c78b6f33b46340d8ccb6957a52b2e88adc971a713159780c83c31d9d8e3cf.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-26/0390e304-b35b-4a5f-9ad3-e0d35edf7b5c/385114f2251f97d3f5b2c489fb761a868f0d1496128582ffe78aebe6270f6e60.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-26/0390e304-b35b-4a5f-9ad3-e0d35edf7b5c/b263d31c62d5b7fa479982766d85dab21d4a97a153e7967182c6a91ea9ef52d4.jpg)



Figure 19: Varlen-M Grouped GEMM with contiguously-packed inputs to up and down-proj during forward pass on H100 GPU. We use the same configurations as in Figure 13. “cuBLAS BMM” is a dense BMM baseline equivalent to all expert receiving equal number of tokens (perfectly load balanced), whose TFLOPS can be considered as an upper bound for any Grouped GEMM kernel.


Grouped GEMM with gather fusion. In Figure 20, we report SonicMoE, ScatterMoE, MoMoE, and DeepGEMM with and without gather fusion (as opaque and transparent bars for each method) on Hopper GPU. ScatterMoE and MoMoE both have gather fusion for varlen-M but not varlen-K Grouped GEMM, so we benchmark their gather with varlen-K Grouped GEMM time (opaque bar) by adding up the time of their contiguously-packed weight gradient kernel (transparent bar) with the time of their own gather kernel. We benchmark the gather with Grouped GEMM time of DeepGEMM for both varlen-M and varlen-K cases using the same approach. We also provide a cuBLAS dense BMM (transparent bar) baseline and the gather with BMM GEMM time (opaque bar) by adding up the time with a heavily-tuned gather kernel’s time with the same 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-26/0390e304-b35b-4a5f-9ad3-e0d35edf7b5c/a2b66d15ffa79ef43d2206317670a77c1d7c0048c51d9cfbac3c5d69dab950b7.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-26/0390e304-b35b-4a5f-9ad3-e0d35edf7b5c/00c58ac8da081a3523be1ad1fb5f4fc619a29a788acf8e2eeeb92974c06cc750.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-26/0390e304-b35b-4a5f-9ad3-e0d35edf7b5c/45d1548c8fa2bd829b76b05adff33f6c047de8f9ee2d65a6d958347f8f39b925.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-26/0390e304-b35b-4a5f-9ad3-e0d35edf7b5c/94fee012a87f44d7f7e76fff40e3c88b2c67a661a86d605433a1771eff674073.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-26/0390e304-b35b-4a5f-9ad3-e0d35edf7b5c/e5dfcd81f706647c9c89879f9b78cd7405fdb76ba77ab99552ac013a958a8457.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-26/0390e304-b35b-4a5f-9ad3-e0d35edf7b5c/6ddd16afb0203c2ea8f1fb5be2e1e60ef5ce28a8d1eaa97e3e592ac4a88bf519.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-26/0390e304-b35b-4a5f-9ad3-e0d35edf7b5c/682641f9e05186c4352e38ded46c38d037a0d651672fe7999326f977a7aadb9d.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-26/0390e304-b35b-4a5f-9ad3-e0d35edf7b5c/e5253c869dcfb23d982e9b19e0a05c243a52aa11d33bde27f52e66d46ffb35f0.jpg)



Figure 20: Forward pass up-proj (gather on M dim) and backward up-proj weight gradient $d W _ { 1 }$ (gather on K dim) kernel on H100 GPU. SonicMoE supports both inputs gathered from different positions (opaque bars) and contiguously-packed inputs (transparent bars). ScatterMoE and MoMoE both have gather fusion for varlen-M but not varlen-K, so we benchmark their gather with varlen-K Grouped GEMM time (opaque bar) by adding up the time of their contiguously-packed weight gradient kernel (transparent bar) with the time of their gather kernel. DeepGEMM does not have gather fusion for both varlen-M and varlen-K Grouped GEMM, so we provide an optimized gather kernel in both cases. We also provide a “cuBLAS dense BMM” (transparent bar) baseline and the gather with GEMM time (opaque bar) by adding up the time with a heavily-tuned gather kernel’s time with the same input shape, which can be considered as the upper bound TFLOPS for any Grouped GEMM kernel without gather fusion.


input shape, which can be considered as the upper bound of the TFLOPS for any Grouped GEMM kernel without gather fusion. We discuss the effect gather on $M$ and $K$ dimension for SonicMoE: 

• Gather on M dimension. The average relative TFLOPS difference of SonicMoE with and without gather fusion on M dim is $6 . 3 \%$ in Figure 20. SonicMoE consistently achieves higher TFLOPS than ScatterMoE (avg $9 . 7 \%$ ), MoMoE (avg $3 0 . 9 \%$ ), and DeepGEMM (avg 38.3%) with gather fusion. 

• Gather on K dimension. The average relative TFLOPS difference of SonicMoE with and without gather fusion on K dim is $8 . 5 \%$ in Figure 20. SonicMoE already achieves higher TFLOPS than ScatterMoE (avg $2 3 . 5 \%$ higher), MoMoE (avg $4 . 3 \%$ higher), and DeepGEMM (avg $4 1 . 4 \%$ higher) without gather fusion (transparent bars). When we compare SonicMoE with gather fusion (opaque bars) against ScatterMoE and MoMoE with their gather kernel together, we observe a wider gap as expert granularity increases (from right to left on each 3 bar groups). On average, SonicMoE with gather fusion achieves $5 5 . 1 \%$ , $4 2 . 4 \%$ , $7 1 . 8 \%$ higher TFLOPS than ScatterMoE, MoMoE, and DeepGEMM with gather fusion respectively. 

# E.2 Expert aggregation on Hopper GPUs

We benchmark the bandwidth of SonicMoE’s aggregation kernel on H100 in Figure 21. We compare SonicMoE’s gatherand-sum aggregation (Figure 9 left) with ScatterMoE’s torch.bmm and MoMoE torch.sum aggregation (Figure 9 middle). In addition, we implement a highly-optimized triton aggregation kernel (“triton sum (contig. Y )”) with extensive kernel configuration tuning. This kernel achieves $2 . 8 5 +$ TB/s ( $8 5 \% +$ peak) for most MoE configurations so we consider it as an upper bound for any aggregation kernel on H100. Although SonicMoE’s aggregation kernel requires a gather fusion during HBM load, the memory bandwidth of SonicMoE’s still surpasses ScatterMoE ( $2 . 9 2 \mathrm { x }$ on average) and MoMoE ( $1 . 0 5 \mathrm { x }$ on average), and is only slightly slower (0.98x on average) than the triton upper bound of summing over contiguous $Y$ . 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-26/0390e304-b35b-4a5f-9ad3-e0d35edf7b5c/553afdecdb9357c3b3d4afe6db7e8f2221cb00f96812859460a04cade127995a.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-26/0390e304-b35b-4a5f-9ad3-e0d35edf7b5c/007f59733cfea704528f3a39293241b4f571b1c5b71fb4d99936e06ee6612c82.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-26/0390e304-b35b-4a5f-9ad3-e0d35edf7b5c/a7c8b61646a8684133e9d3cdc9c955c6a312e049f7536824b649d8af9725932e.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-26/0390e304-b35b-4a5f-9ad3-e0d35edf7b5c/26dc54018b8653090ee08bb3d645d25a60079bf8019c1a48c4ee48217c2411b1.jpg)



Figure 21: Expert aggregation kernels $\boldsymbol { O }$ kernel) during forward pass of MoE. Same configurations as in Figure 13. “ScatterMoE (contig. Y )” is the expert aggregation strategy employed by ScatterMoE. The implementation uses torch.bmm call to reduce over $K$ , “MoMoE (contig. Y )” is a torch.sum call for MoMoE. We take the maximum bandwidth between PyTorch eager and PyTorch compile on default mode with PyTorch 2.9.0. We also implement an optimized triton kernel for summing over contiguously-packed $Y$ inputs with Triton 3.3.1 as “triton sum (contig. Y )”.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-26/0390e304-b35b-4a5f-9ad3-e0d35edf7b5c/fa84d78a0474e9cb41bcbe75adb868fa963c5a05936b135a734e78878bed6e2d.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-26/0390e304-b35b-4a5f-9ad3-e0d35edf7b5c/2b8545e56510e2882c684d2581650adf6083aca58fc5c9a205c6a0d1597c404f.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-26/0390e304-b35b-4a5f-9ad3-e0d35edf7b5c/e5d9e500f075610cc2307a2e5490c0e54b4d9a1c4e5ba769d7c1deabf643a0d5.jpg)


![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-26/0390e304-b35b-4a5f-9ad3-e0d35edf7b5c/87fa6283c1119763b9e2fe5c3ad8fd2733121b6ece48d7a09b264255a734d969.jpg)



Figure 22: Throughput of Grouped GEMM and expert aggregation kernel on H100. “SonicMoE (gemm $^ +$ gth w. sum)” is the final design choice for SonicMoE as illustrated in Figure 9 left strategy. We compare this design against “SonicMoE (gemm w. sct + sum)” that implements the Figure 9 middle strategy on SonicMoE. We use identical tile sizes and other GEMM configs for both “SonicMoE (gemm $^ +$ gth w. sum)” and “SonicMoE (gemm w. sct + sum)”. We also compare with ScatterMoE’s design (fused scatter with GEMM $^ +$ torch.bmm, labeled as “ScatterMoE (gemm w. sct + BMM)”) and MoMoE’s design (fused scatter with GEMM $^ +$ torch.sum, labeled as “MoMoE (gemm w. sct + sum)”). For each method, we report the GEMM TFLOPS in transparent bars and TFLOPS of total runtime of GEMM and expert aggregation in the opaque bars.


# E.3 Top- $K$ sorting on Hopper GPUs

We benchmark the bandwidth of SonicMoE’s top- $K$ kernel on H100 in Figure 23. We compare SonicMoE with PyTorch48, triton official example49, tilelang official example50, and RTop- $K ^ { 5 1 }$ (Xie et al. 2025) on BF16 and FP32 inputs. 

• PyTorch single block Top- $K$ : PyTorch (Paszke et al. 2019) implements a radix-select followed by a gather algorithm for top- $K$ , and it dispatches to a single or multiple block version depending on $T , E , K$ . For large $T$ with modest $E$ and $K$ , PyTorch uses the single-block version that performs 2 SMEM scans. In this case, SonicMoE’s sorting networks with pure register-based communication is much faster. 

• Triton official example: Triton (Tillet, Kung, and Cox 2019) provides a top- $K$ example kernel that is also based on bit packing and bitonic merge. The main algorithmic difference is that SonicMoE relies on optimal sorting networks on the base cases while the Triton implementation directly calls triton.language.topk. During the top- $K$ benchmark in Figure 23, we observe that Triton is much faster than PyTorch torch.topk but it is still consistently slower than SonicMoE’s top- $K$ across all configurations. 

• Tilelang official example: Tilelang (Wang et al. 2025) provides a top- $K$ example kernel that performs $K$ -pass 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-03-26/0390e304-b35b-4a5f-9ad3-e0d35edf7b5c/41a82824ef2a8aa82d6298ef32ec6b7e0a75c45e5c4d0b9ac4832c4956716e98.jpg)



Figure 23: Top- $K$ kernels with BF16 inputs $1 ^ { \mathrm { s t } }$ row) and FP32 inputs ( $2 ^ { \mathrm { n d } }$ row) during forward pass of MoE. Same configurations as in Figure 13. “torch” is a direct torch.topk call. “triton” and “tilelang” are taken from their official examples with slight modifications to support BF16 inputs. For the triton official kernel, we remove the unnecessary bit matrix store and disable the softmax fusion in this example for a fair comparison. “RTop-K”(Xie et al. 2025) only supports FP32 inputs. We set $\epsilon = 0$ and maximum iteration as 8 for RTop-K.


maximum reduction. This design is more targeted for small $K$ and we observe that as both $E$ and $K$ become larger, the Tilelang top- $K$ kernel’s throughput decreases compared to other baselines (and SonicMoE)’s increasing trend. Such trend makes Tilelang’s example top- $K$ kernel ( $K$ -pass top- $K$ kernel) unsuitable for fine-grained MoEs. 

RTop-K: RTop-K (Xie et al. 2025) follows a threshold-based binary search. Each bisection step utilizes warp-level primitives and follows a selection-by-count method instead of SonicMoE’s sorting network. RTop-K is also an iterative algorithm with iterations dependent on the value range and vector size. In addition, RTop-K heavily utilizes SMEM for scanning while SonicMoE solely relies on registers for its compare and swap subroutine. We find that SonicMoE’s top- $K$ consistently achieves higher throughput for 1.4B and 7B model configurations. For 30B and 120B, SonicMoE is slower when $( E , K ) = ( 2 5 6 , 1 6 )$ but still outperforms when $( E , K ) = ( 1 2 8 , 8 ) , ( 6 4 , 4 )$ . 

# F More experiments

In this section, we investigate the qualitative improvement from fine-grained MoE in Section F.1. We also investigate the effect of rounding subroutines round and sparsify in Algorithm 4 and the effects from microbatch size $T$ and tile size $M _ { \mathrm { t i l e } }$ for token rounding in Section F.3. 

# F.1 Effect of expert granularity

Here we validate the effectiveness of adopting fine-grained MoE. We fix the MoE activation ratio $\rho = K / E$ for the 0.5B and 1.4B model and we proportionally scale up $K$ and $E$ while linearly decreasing $n$ from row 1 to row 3 in Table 5a and 5b. 

In general, we observe a better performance for $n = 2 5 6$ than $n = 1 0 2 4$ which is also consistent with the MoE scaling trends mentioned in Table 1. In Figure 1 right subfigure, we find both SonicMoE and cuBLAS can still sustain the throughput from $n = 1 0 2 4$ to $n = 2 5 6$ under iso-FLOPs, but starting from $n = 2 5 6$ FLOPs will drop linearly w.r.t. granularity. Therefore, we choose $n = 2 5 6$ for all experiments in Table 3. 

# F.2 Ablation study on different rounding subroutines for token rounding

We conduct ablation studies to study the effect of the different routing subroutines on the trained MoE by TR. We compare token rounding with nearest rounding (“NR”) with per-expert token counts with other rounding methods. Specifically, we compare against stochastic rounding with per-expert token count (“SR”), always round up (“UP”), and always round down (“DOWN”). The results are shown in Table 6 and we find that our token rounding algorithm in general is robust to the specific rounding subroutines. 

Table 5: Evaluation of MoE w.r.t. granularity with iso-FLOPs $n K$ is constant) and iso-params ( $_ { R E }$ is constant) settings. “PPL” refers to the validation perplexity at the end of training. “Avg” is the mean accuracy across the 11 downstream tasks. The “dense, iso-FLOPs” refers to a dense model with $n K$ as the intermediate size, while the “dense, iso-params” refers to a dense model with $n E$ as the intermediate size. 


(a) 0.5B params, 20B tokens, 8/64 activated


<table><tr><td>(E, K, n)</td><td>PPL</td><td>Wino</td><td>SIQA</td><td>SciQ</td><td>PIQA</td><td>OBQA</td><td>HS</td><td>COPA</td><td>CSQA</td><td>BoolQ</td><td>ArcE</td><td>ArcC</td><td>Avg</td></tr><tr><td>16, 2, 1024</td><td>16.23</td><td>53.0</td><td>41.3</td><td>79.8</td><td>65.0</td><td>32.6</td><td>37.8</td><td>66.0</td><td>32.2</td><td>55.8</td><td>53.9</td><td>29.1</td><td>49.7</td></tr><tr><td>64, 8, 256</td><td>16.01</td><td>51.0</td><td>41.4</td><td>79.2</td><td>65.5</td><td>31.6</td><td>38.4</td><td>66.0</td><td>31.5</td><td>60.2</td><td>57.5</td><td>25.7</td><td>49.8</td></tr><tr><td>256, 32, 64</td><td>16.13</td><td>51.2</td><td>41.5</td><td>78.9</td><td>65.3</td><td>34.2</td><td>38.4</td><td>63.0</td><td>32.4</td><td>60.6</td><td>59.5</td><td>28.1</td><td>50.3</td></tr><tr><td>Dense, iso-FLOPs</td><td>19.90</td><td>48.9</td><td>41.4</td><td>74.9</td><td>62.2</td><td>30.2</td><td>32.6</td><td>62.0</td><td>31.6</td><td>61.7</td><td>53.2</td><td>27.1</td><td>47.8</td></tr><tr><td>Dense, iso-params</td><td>15.46</td><td>52.1</td><td>41.5</td><td>78.9</td><td>65.3</td><td>34.0</td><td>39.2</td><td>69.0</td><td>32.2</td><td>58.5</td><td>59.3</td><td>28.8</td><td>50.8</td></tr></table>


(b) 1.4B params, 50B tokens, 8/128 activated


<table><tr><td>32, 2, 1024</td><td>13.38</td><td>52.2</td><td>41.7</td><td>81.7</td><td>69.2</td><td>33.6</td><td>44.3</td><td>64.0</td><td>33.5</td><td>61.1</td><td>60.9</td><td>29.8</td><td>52.0</td></tr><tr><td>128, 8, 256</td><td>13.32</td><td>51.8</td><td>41.7</td><td>81.5</td><td>69.3</td><td>32.4</td><td>45.3</td><td>68.0</td><td>34.5</td><td>56.6</td><td>63.2</td><td>28.4</td><td>52.1</td></tr><tr><td>512, 32, 64</td><td>13.50</td><td>52.5</td><td>41.2</td><td>82.9</td><td>68.9</td><td>34.4</td><td>44.7</td><td>69.0</td><td>33.6</td><td>58.7</td><td>62.6</td><td>30.1</td><td>52.6</td></tr><tr><td>Dense, iso-FLOPs</td><td>17.90</td><td>52.2</td><td>41.0</td><td>79.2</td><td>63.4</td><td>31.0</td><td>34.7</td><td>61.0</td><td>30.5</td><td>60.3</td><td>51.8</td><td>25.1</td><td>48.2</td></tr><tr><td>Dense, iso-params</td><td>12.74</td><td>52.2</td><td>42.6</td><td>83.3</td><td>70.1</td><td>34.8</td><td>46.8</td><td>67.0</td><td>35.1</td><td>61.7</td><td>63.5</td><td>31.8</td><td>53.5</td></tr></table>

Following Algorithm 4, for expert e, we denote the expert frequency from the TC sorting as $f _ { e }$ , and the last $M _ { \mathrm { t i l e } }$ -divisible expert frequency as $\lfloor f _ { e } \rfloor _ { M _ { \mathrm { t i l e } } }$ , and the next $M _ { \mathrm { t i l e } }$ -divisible expert frequency as $\lceil f _ { e } \rceil _ { M _ { \mathrm { t i l e } } }$ . We also denote the expert scores from TC sorting as $s _ { e }$ , the expert scores from the selected tokens in $\pi _ { e } [$ [: $\lfloor f _ { e } \rfloor _ { M _ { \mathrm { t i l e } } } ]$ as $\lfloor s _ { e } \rfloor _ { M _ { \mathrm { t i l e } } }$ and the scores for π $\boldsymbol { \mathbf { \ell } } _ { e } [ : \left. \lceil f _ { e } \right\rceil _ { M _ { \mathrm { t i l e } } } ]$ as $\lceil s _ { e } \rceil _ { M _ { \mathrm { t i l e } } }$ . We note that all rounding algorithms only make a binary decision between discarding TC tokens and padding EC tokens for each expert. Following are simple heuristics to perform rounding: 

• NR-f: nearest rounding to $M _ { \mathrm { t i l e } }$ -multiples via expert frequency: We pad EC tokens if $\lceil f _ { e } \rceil _ { M _ { \mathrm { t i l e } } - f _ { e } } < f _ { e } - \lfloor f _ { e } \rfloor _ { M _ { \mathrm { t i l e } } }$ “NR-f” is our default choice of token rounding and we use it for Table 3, 7, 8, and Figures 11 and 16. 

• SR-f: stochastic rounding to $M _ { \mathrm { t i l e } }$ -multiples via expert frequency: We sample from Bernoulli $\left( \frac { f _ { e } - \lfloor f _ { e } \rfloor _ { M _ { \mathrm { t i l e } } } } { M _ { \mathrm { t i l e } } } \right)$ Mtile distribution for deciding whether to pad EC tokens for expert e. 

• NR-s: nearest rounding to $M _ { \mathrm { t i l e } }$ -multiples via expert scores: We sample from the following distribution for deciding whether to pad EC tokens for expert $e$ : 

$$
\text {B e r n o u l l i} \left(\frac {\sum_ {t} s _ {e , t} - \sum_ {t} \left\lfloor s _ {e , t} \right\rfloor M _ {\text {t i l e}}}{\sum_ {t} \left\lceil s _ {e , t} \right\rceil M _ {\text {t i l e}} - \sum_ {t} \left\lfloor s _ {e , t} \right\rfloor M _ {\text {t i l e}}}\right) \tag {13}
$$

• Balance-f: balanced rounding to $M _ { \mathrm { t i l e } }$ -multiples via expert frequency: The Balance algorithm (Cooper et al. 2023; Dwivedi and Mackey 2024; Lu, Guo, and De Sa 2022) can be adapted to ensure the total number of routed tokens to all experts after tile-rounding is preserved regardless of the number of experts $E$ . Algorithm 6 is such an example that ensures 

$$
\max  _ {e \in [ E ]} | \left\lceil f _ {e} \left. \right\rfloor_ {M _ {\text {t i l e}}} - f _ {e} | \leq M _ {\text {t i l e}} / 2, \quad \left| \right. \sum_ {e = 1} ^ {E} \left\lceil f _ {e} \left. \right\rfloor_ {M _ {\text {t i l e}}} - \sum_ {e = 1} ^ {E} f _ {e} \left. \right| \leq M _ {\text {t i l e}} / 2 \tag {14}
$$

where the other rounding subroutine will have an expected deviation of $O \left( M _ { \mathrm { t i l e } } { \sqrt { E } } \right)$ for $\textstyle \sum _ { e = 1 } ^ { E } \lceil f _ { e } \rfloor _ { M _ { \mathrm { t i l e } } }$ 

• UP: always round up expert frequency as $\lceil f _ { e } \rceil _ { M _ { \mathrm { t i l e } } }$ : We always pad EC tokens chosen in the second step of sorting in Algorithm 4. This gives a model TFLOPS lower-bound for Figure 16. 

• DOWN: always round down expert frequency as $\lfloor f _ { e } \rfloor _ { M _ { \mathrm { t i l e } } }$ : We always discard TC top- $K$ tokens chosen in the first step of sorting in Algorithm 4. This gives a model TFLOPS upper-bound for Figure 16. 

Always discarding TC tokens (“DOWN”). “DOWN” is a baseline in which we always drop the last TC tile if the expert frequency is not $M _ { \mathrm { t i l e } }$ -divisible. This idea is similar to the idea of token dropping in expert parallelism where the expert will sort and drop the token with the lowest scores when it receives too many tokens (Fedus, Zoph, and Shazeer 2022). We note that “DOWN” produces the shortest MoE kernel runtime for any rounding algorithm. However, in Table 6, we observe that “DOWN” yields a much higher validation perplexity than “NR-f”, “SR-f” and “NR-s”. Although we can expect a shorter MoE kernel runtime by always discarding TC tokens, such quality degradation might not be acceptable in practice. 

Algorithm 6 Balanced rounding to $M _ { \mathrm { t i l e } }$ -multiples via expert frequency (“Balance-f” in Table 6). 

This algorithm satisfies $\begin{array} { r } { \operatorname* { m a x } _ { e \in [ E ] } \left| \left\lceil f _ { e } \right\rfloor _ { M _ { \mathrm { t i l e } } } - f _ { e } \right| \leq M _ { \mathrm { t i l e } } / 2 } \end{array}$ and $\begin{array} { r } { \left| \sum _ { e = 1 } ^ { E } [ f _ { e } ] _ { M _ { \mathrm { t i l e } } } - \sum _ { e = 1 } ^ { E } f _ { e } \right| \leq M _ { \mathrm { t i l e } } / 2 } \end{array}$ 

Input : $f ^ { \mathrm { T C } } = \{ f _ { e } \} _ { e \in [ E ] }$ as a list of expert frequency with TC top- $K$ routing, $\{ \lceil f _ { e } \rceil _ { M _ { \mathrm { t i l e } } } \} _ { e \in [ E ] }$ as a list of expert frequency with TC top- $K$ routing and with potential EC padding to ensure each expert receives a multiple of $M _ { \mathrm { t i l e } }$ tokens, $\{ \lfloor f _ { e } \rfloor _ { M _ { \mathrm { t i l e } } } \} _ { e \in [ E ] }$ as a list of expert frequency with TC top- $K$ routing and with potential token dropping to ensure each expert receives a multiple of $M _ { \mathrm { t i l e } }$ tokens; We should ensure $\begin{array} { r } { \operatorname* { m a x } _ { e \in \left[ E \right] } \left( \left\lceil f _ { e } \right\rceil _ { M _ { \mathrm { t i l e } } } - \left\lfloor f _ { e } \right\rfloor _ { M _ { \mathrm { t i l e } } } \right) \leq M _ { \mathrm { t i l e } } } \end{array}$ . 

Output : $f ^ { \mathrm { T R } } = \{ \lceil f _ { e } \rfloor _ { M _ { \mathrm { t i l e } } } \} _ { e \in [ E ] }$ as a list of expert frequency that ensures each expert receives a multiple of $M _ { \mathrm { t i l e } }$ tokens 

// an accumulator that ensures the preservation of total expert frequency 

$$
z \leftarrow 0;
$$

for $e \in [ E ]$ do 

// calculate the residual error of both rounding choice 

$$
\begin{array}{l} r _ {e} ^ {\mathrm {u p}} \leftarrow \left\lceil f _ {e} \right\rceil_ {M _ {\text {t i l e}}} - f _ {e}; \\ r _ {e} ^ {\text {d o w n}} \leftarrow \left\lfloor f e \right\rfloor_ {M _ {\text {t i l e}}} - f _ {e}; \\ \text {i f} \left| r _ {e} ^ {u p} + z \right| <   \left| r _ {e} ^ {\text {d o w n}} + z \right| \text {t h e n} \\ / / \text {c h o i s e t o p a d w i t h E C t o k e n s} \\ \begin{array}{l} \left[ f _ {e} \right] _ {M _ {\text {t i l e}}} \leftarrow \left[ f _ {e} \right] _ {M _ {\text {t i l e}}}; \\ z \leftarrow z + r _ {e} ^ {\text {u p}}; \end{array} \\ \end{array}
$$

$$
\begin{array}{l} \left\lceil f _ {e} \left. \right\rfloor_ {M _ {\text {t i l e}}} \leftarrow \left\lfloor f _ {e} \right\rfloor_ {M _ {\text {t i l e}}}; \\ z \leftarrow z + r _ {e} ^ {\text {d o w n}}; \\ \end{array}
$$


Table 6: Evaluation of token rounding algorithms equipped with different round and sparsify subroutines in Algorithm 4. “PPL” refers to the validation perplexity at the end of training. “Avg” is the mean accuracy across the 11 downstream tasks.



(a) 0.5B params, 40B tokens, 2/64 activated $\bar { T } _ { e } = 5 1 2$ , $M _ { \mathrm { t i l e } } = 1 2 8$


<table><tr><td>Method</td><td>PPL</td><td>Wino</td><td>SIQA</td><td>SciQ</td><td>PIQA</td><td>OBQA</td><td>HS</td><td>COPA</td><td>CSQA</td><td>BoolQ</td><td>ArcE</td><td>ArcC</td><td>Avg</td></tr><tr><td>TR (NR-f)</td><td>15.92</td><td>51.4</td><td>41.6</td><td>78.4</td><td>65.4</td><td>31.6</td><td>38.1</td><td>65.0</td><td>31.0</td><td>61.1</td><td>57.4</td><td>29.1</td><td>50.0</td></tr><tr><td>TR (SR-f)</td><td>15.93</td><td>50.8</td><td>40.9</td><td>77.4</td><td>66.9</td><td>33.0</td><td>38.4</td><td>64.0</td><td>31.1</td><td>60.7</td><td>55.8</td><td>28.1</td><td>49.7</td></tr><tr><td>TR (NR-s)</td><td>15.91</td><td>51.3</td><td>40.9</td><td>80.3</td><td>65.4</td><td>30.8</td><td>37.7</td><td>67.0</td><td>31.0</td><td>61.6</td><td>55.4</td><td>28.4</td><td>50.0</td></tr><tr><td>TR (Balance-f)</td><td>15.93</td><td>51.9</td><td>41.8</td><td>78.8</td><td>65.9</td><td>32.6</td><td>38.4</td><td>66.0</td><td>31.6</td><td>60.3</td><td>56.8</td><td>27.1</td><td>50.1</td></tr><tr><td>TR (UP)</td><td>15.89</td><td>50.5</td><td>40.9</td><td>78.6</td><td>64.5</td><td>32.2</td><td>38.2</td><td>68.0</td><td>29.9</td><td>55.2</td><td>54.2</td><td>30.1</td><td>49.3</td></tr><tr><td>TR (DOWN)</td><td>16.10</td><td>51.1</td><td>41.4</td><td>78.7</td><td>64.9</td><td>31.6</td><td>38.0</td><td>62.0</td><td>32.8</td><td>61.9</td><td>58.9</td><td>30.8</td><td>50.2</td></tr><tr><td>TC top-K</td><td>15.94</td><td>51.0</td><td>41.9</td><td>78.5</td><td>64.8</td><td>33.0</td><td>38.1</td><td>67.0</td><td>30.8</td><td>54.7</td><td>55.8</td><td>30.1</td><td>49.6</td></tr></table>


(b) 1.8B params, 40B tokens, 8/256 activated $\bar { T } _ { e } = 5 1 2$ , $M _ { \mathrm { t i l e } } = 1 2 8$ )


<table><tr><td>TR (NR-f)</td><td>13.10</td><td>53.4</td><td>42.1</td><td>81.7</td><td>69.6</td><td>35.2</td><td>45.3</td><td>70.0</td><td>33.2</td><td>61.4</td><td>63.0</td><td>33.4</td><td>53.5</td></tr><tr><td>TR (SR-f)</td><td>13.08</td><td>52.7</td><td>41.6</td><td>82.6</td><td>69.4</td><td>34.4</td><td>45.6</td><td>70.0</td><td>33.0</td><td>59.1</td><td>62.5</td><td>34.8</td><td>53.2</td></tr><tr><td>TR (NR-s)</td><td>13.09</td><td>54.1</td><td>42.3</td><td>82.8</td><td>69.3</td><td>33.8</td><td>45.7</td><td>70.0</td><td>34.1</td><td>59.0</td><td>64.6</td><td>32.4</td><td>53.5</td></tr><tr><td>TR (Balance-f)</td><td>13.08</td><td>52.5</td><td>42.0</td><td>82.7</td><td>70.0</td><td>33.2</td><td>45.3</td><td>68.0</td><td>34.6</td><td>59.4</td><td>63.3</td><td>33.4</td><td>53.1</td></tr><tr><td>TR (UP)</td><td>13.07</td><td>50.4</td><td>41.7</td><td>81.4</td><td>68.4</td><td>37.2</td><td>45.4</td><td>69.0</td><td>31.9</td><td>51.7</td><td>62.2</td><td>33.4</td><td>52.1</td></tr><tr><td>TR (DOWN)</td><td>13.19</td><td>55.4</td><td>41.6</td><td>82.2</td><td>68.6</td><td>34.8</td><td>45.0</td><td>69.0</td><td>34.0</td><td>54.4</td><td>63.5</td><td>31.4</td><td>52.7</td></tr><tr><td>TC top-K</td><td>13.12</td><td>50.1</td><td>42.9</td><td>81.3</td><td>69.8</td><td>33.8</td><td>45.2</td><td>71.0</td><td>34.1</td><td>56.7</td><td>64.6</td><td>31.1</td><td>52.8</td></tr></table>

Always padding EC tokens (“UP”). “UP” is a baseline in which we always pad extra EC tokens to the last TC tile if the expert frequency is not $M _ { \mathrm { t i l e } }$ -divisible. Contrary to “DOWN”, “UP” produces the longest MoE kernel runtime for any rounding algorithm. In Table 6, we find that “UP” often produces lower validation perplexity, but the average downstream task accuracy is not necessarily higher than other rounding algorithms. Given the longer MoE kernel runtime but not necessarily better trained MoE quality, we do not recommend the usage of always rounding up. We speculate this is due to the train-test gap between TC and EC routing and “UP” reinforces the bias towards EC more strongly than other TR algorithms. 

For a balance between training efficiency and trained MoE quality, neither always discarding TC tokens nor padding EC tokens is the right solution. In Table 3, we pick “NR-f” as the round and sparsify subroutine for TR’s main experiments. 

# F.3 Ablation study on the effects of microbatch size $T$ and tile size $M _ { \mathrm { t i l e } }$

Effect of microbatch size $T$ . Since the token rounding is applied on the microbatch level, the choice of microbatch size $T$ will result in different qualitative results for TR. Note that this also holds true for EC routing. For example, EC over sequence 

will result in different model quality as EC over a text segment. In Table 7, we vary the microbatch size while keeping the minibatch size (consumed tokens per optimization step) constant. 

We find that TR will preserve its trained MoE quality when $\bar { T } _ { e } / M _ { \mathrm { t i l e } } \geq 2$ , but if $\bar { T } _ { e } / M _ { \mathrm { t i l e } } = 1$ (the last row in both subtables), there is a noticeable quality degradation for both validation perplexity and downstream task performance. However, the trained MoE quality with $\bar { T } _ { e } / M _ { \mathrm { t i l e } } = 1$ is still better than training with EC and finetuning with TC top- $K$ routing. 

Effect of the tile quantization size $M _ { \mathrm { t i l e } }$ . Similarly in Table 8, we can find that TR is generally robust w.r.t. $M _ { \mathrm { t i l e } }$ when $\bar { T } _ { e } / M _ { \mathrm { t i l e } } \geq 2$ , and when $\bar { T } _ { e } / M _ { \mathrm { t i l e } } = 1$ there is a noticeable degradation but the overall result is still better than EC baseline. 

Table 7: Evaluation of token rounding algorithms when we vary microbatch size $T$ to change average number of tokens per expert $( \hat { T } _ { e } )$ For each trial, we vary the microbatch size from 4 $\bar { T } _ { e } = 5 1 2$ ) to 1 $\bar { T } _ { e } = 1 2 8 )$ and keep minibatch size constant. The $M _ { \mathrm { t i l e } }$ is always kept as 128. “PPL” refers to the validation perplexity at the end of training. “Avg” is the mean accuracy across the 11 downstream tasks. 


(a) 0.5B params, 40B tokens, 2/64 activated $M _ { \mathrm { t i l e } } = 1 2 8$


<table><tr><td>Method</td><td>PPL</td><td>Wino</td><td>SIQA</td><td>SciQ</td><td>PIQA</td><td>OBQA</td><td>HS</td><td>COPA</td><td>CSQA</td><td>BoolQ</td><td>ArcE</td><td>ArcC</td><td>Avg</td></tr><tr><td>TR (Te=1024)</td><td>15.91</td><td>52.9</td><td>41.0</td><td>80.1</td><td>65.1</td><td>31.0</td><td>37.9</td><td>63.0</td><td>32.3</td><td>59.3</td><td>54.9</td><td>28.1</td><td>49.6</td></tr><tr><td>TR (Te=512)</td><td>15.92</td><td>51.4</td><td>41.6</td><td>78.4</td><td>65.4</td><td>31.6</td><td>38.1</td><td>65.0</td><td>31.0</td><td>61.1</td><td>57.4</td><td>29.1</td><td>50.0</td></tr><tr><td>TR (Te=256)</td><td>15.98</td><td>52.2</td><td>41.4</td><td>77.7</td><td>66.1</td><td>32.2</td><td>37.9</td><td>66.0</td><td>31.0</td><td>59.6</td><td>57.2</td><td>30.1</td><td>50.1</td></tr><tr><td>TR (Te=128)</td><td>16.11</td><td>51.7</td><td>41.7</td><td>77.9</td><td>66.1</td><td>30.8</td><td>37.7</td><td>67.0</td><td>31.9</td><td>61.2</td><td>54.7</td><td>29.1</td><td>50.0</td></tr><tr><td>TC top-K</td><td>15.94</td><td>51.0</td><td>41.9</td><td>78.5</td><td>64.8</td><td>33.0</td><td>38.1</td><td>67.0</td><td>30.8</td><td>54.7</td><td>55.8</td><td>30.1</td><td>49.6</td></tr><tr><td>EC (ft TC router)</td><td>16.98</td><td>50.0</td><td>41.7</td><td>79.7</td><td>64.9</td><td>31.6</td><td>36.8</td><td>63.0</td><td>32.1</td><td>60.7</td><td>54.6</td><td>27.4</td><td>49.3</td></tr></table>


(b) 1.8B params, 40B tokens, 8/256 activated $( M _ { \mathrm { t i l e } } = 1 2 8 )$ )


<table><tr><td>TR (Te=1024)</td><td>13.08</td><td>51.5</td><td>42.0</td><td>81.7</td><td>68.9</td><td>34.8</td><td>45.7</td><td>72.0</td><td>32.6</td><td>59.5</td><td>61.4</td><td>32.1</td><td>52.9</td></tr><tr><td>TR (Te=512)</td><td>13.10</td><td>53.4</td><td>42.1</td><td>81.7</td><td>69.6</td><td>35.2</td><td>45.3</td><td>70.0</td><td>33.2</td><td>61.4</td><td>63.0</td><td>33.4</td><td>53.5</td></tr><tr><td>TR (Te=256)</td><td>13.12</td><td>51.9</td><td>41.2</td><td>81.8</td><td>69.7</td><td>33.6</td><td>45.2</td><td>73.0</td><td>34.2</td><td>56.9</td><td>63.2</td><td>34.1</td><td>53.2</td></tr><tr><td>TR (Te=128)</td><td>13.55</td><td>51.9</td><td>41.5</td><td>82.0</td><td>69.2</td><td>32.8</td><td>44.4</td><td>69.0</td><td>34.4</td><td>59.8</td><td>64.0</td><td>30.4</td><td>52.7</td></tr><tr><td>TC top-K</td><td>13.12</td><td>50.1</td><td>42.9</td><td>81.3</td><td>69.8</td><td>33.8</td><td>45.2</td><td>71.0</td><td>34.1</td><td>56.7</td><td>64.6</td><td>31.1</td><td>52.8</td></tr><tr><td>EC (ft TC router)</td><td>15.01</td><td>52.7</td><td>41.1</td><td>79.6</td><td>66.9</td><td>30.6</td><td>40.2</td><td>66.0</td><td>31.9</td><td>60.5</td><td>57.2</td><td>30.8</td><td>50.7</td></tr></table>


Table 8: Evaluation of token rounding algorithms when we vary the size of tile $M _ { \mathrm { t i l e } }$ for token rounding. “PPL” refers to the validation perplexity at the end of training. “Avg” is the mean accuracy across the 11 downstream tasks.



(a) 0.5B params, 40B tokens, 2/64 activated $( \bar { T } _ { e } = 5 1 2 $ )


<table><tr><td>Method</td><td>PPL</td><td>Wino</td><td>SIQA</td><td>SciQ</td><td>PIQA</td><td>OBQA</td><td>HS</td><td>COPA</td><td>CSQA</td><td>BoolQ</td><td>ArcE</td><td>ArcC</td><td>Avg</td></tr><tr><td>TR (Mtile = 64)</td><td>15.90</td><td>51.3</td><td>41.7</td><td>78.1</td><td>65.6</td><td>31.4</td><td>37.9</td><td>67.0</td><td>32.4</td><td>59.8</td><td>57.2</td><td>28.8</td><td>50.1</td></tr><tr><td>TR (Mtile = 128)</td><td>15.92</td><td>51.4</td><td>41.6</td><td>78.4</td><td>65.4</td><td>31.6</td><td>38.1</td><td>65.0</td><td>31.0</td><td>61.1</td><td>57.4</td><td>29.1</td><td>50.0</td></tr><tr><td>TR (Mtile = 256)</td><td>16.00</td><td>51.7</td><td>41.4</td><td>78.7</td><td>66.3</td><td>32.4</td><td>37.7</td><td>67.0</td><td>31.3</td><td>60.1</td><td>58.2</td><td>29.1</td><td>50.4</td></tr><tr><td>TR (Mtile = 512)</td><td>16.17</td><td>52.5</td><td>41.2</td><td>80.2</td><td>65.2</td><td>32.0</td><td>37.9</td><td>62.0</td><td>31.0</td><td>59.4</td><td>57.2</td><td>30.4</td><td>49.9</td></tr><tr><td>TC top-K</td><td>15.94</td><td>51.0</td><td>41.9</td><td>78.5</td><td>64.8</td><td>33.0</td><td>38.1</td><td>67.0</td><td>30.8</td><td>54.7</td><td>55.8</td><td>30.1</td><td>49.6</td></tr><tr><td>EC (ft TC router)</td><td>16.98</td><td>50.0</td><td>41.7</td><td>79.7</td><td>64.9</td><td>31.6</td><td>36.8</td><td>63.0</td><td>32.1</td><td>60.7</td><td>54.6</td><td>27.4</td><td>49.3</td></tr></table>


(b) 1.8B params, 40B tokens, 8/256 activated $( \bar { T } _ { e } = 5 1 2 $ )


<table><tr><td>TR (Mtile = 64)</td><td>13.07</td><td>52.3</td><td>42.9</td><td>82.7</td><td>69.4</td><td>35.4</td><td>45.6</td><td>70.0</td><td>32.4</td><td>56.6</td><td>64.4</td><td>31.4</td><td>53.0</td></tr><tr><td>TR (Mtile = 128)</td><td>13.10</td><td>53.4</td><td>42.1</td><td>81.7</td><td>69.6</td><td>35.2</td><td>45.3</td><td>70.0</td><td>33.2</td><td>61.4</td><td>63.0</td><td>33.4</td><td>53.5</td></tr><tr><td>TR (Mtile = 256)</td><td>13.13</td><td>52.0</td><td>41.6</td><td>82.1</td><td>69.2</td><td>35.4</td><td>45.3</td><td>69.0</td><td>34.2</td><td>58.0</td><td>65.6</td><td>32.1</td><td>53.1</td></tr><tr><td>TR (Mtile = 512)</td><td>13.56</td><td>53.0</td><td>41.8</td><td>81.2</td><td>68.4</td><td>34.0</td><td>44.2</td><td>68.0</td><td>33.3</td><td>58.1</td><td>59.5</td><td>30.1</td><td>52.0</td></tr><tr><td>TC top-K</td><td>13.12</td><td>50.1</td><td>42.9</td><td>81.3</td><td>69.8</td><td>33.8</td><td>45.2</td><td>71.0</td><td>34.1</td><td>56.7</td><td>64.6</td><td>31.1</td><td>52.8</td></tr><tr><td>EC (ft TC router)</td><td>15.01</td><td>52.7</td><td>41.1</td><td>79.6</td><td>66.9</td><td>30.6</td><td>40.2</td><td>66.0</td><td>31.9</td><td>60.5</td><td>57.2</td><td>30.8</td><td>50.7</td></tr></table>

# G Activation memory and training throughput benchmark configurations

The configurations for calculating IO cost in Figure 3 are presented in Table 9a and the configurations of Figure 13 and 14 are included in Table 9b. 

The configurations for the 4 subfigures in Figure 16 are listed below as a list. Notice that we consistently use $M _ { \mathrm { t i l e } }$ as 128 when we benchmark the TR’s speed. 


Table 9: Benchmark configurations for Figure 3, 13 and 14.


(a) Benchmark configurations for memory IO cost in Figure 3. 

<table><tr><td>Model Size</td><td>T</td><td>d</td><td>n</td><td>E</td><td>K</td></tr><tr><td rowspan="5">1.4B</td><td>40960</td><td>768</td><td>64</td><td>512</td><td>32</td></tr><tr><td>40960</td><td>768</td><td>128</td><td>256</td><td>16</td></tr><tr><td>40960</td><td>768</td><td>256</td><td>128</td><td>8</td></tr><tr><td>40960</td><td>768</td><td>512</td><td>64</td><td>4</td></tr><tr><td>40960</td><td>768</td><td>1024</td><td>32</td><td>2</td></tr><tr><td rowspan="5">7B</td><td>24576</td><td>1536</td><td>64</td><td>512</td><td>32</td></tr><tr><td>24576</td><td>1536</td><td>128</td><td>256</td><td>16</td></tr><tr><td>24576</td><td>1536</td><td>256</td><td>128</td><td>8</td></tr><tr><td>24576</td><td>1536</td><td>512</td><td>64</td><td>4</td></tr><tr><td>24576</td><td>1536</td><td>1024</td><td>32</td><td>2</td></tr><tr><td rowspan="5">30B</td><td>32768</td><td>4096</td><td>64</td><td>1024</td><td>64</td></tr><tr><td>32768</td><td>4096</td><td>128</td><td>512</td><td>32</td></tr><tr><td>32768</td><td>4096</td><td>256</td><td>256</td><td>16</td></tr><tr><td>32768</td><td>4096</td><td>512</td><td>128</td><td>8</td></tr><tr><td>32768</td><td>4096</td><td>1024</td><td>64</td><td>4</td></tr><tr><td rowspan="5">120B</td><td>32768</td><td>4096</td><td>128</td><td>1024</td><td>64</td></tr><tr><td>32768</td><td>4096</td><td>256</td><td>512</td><td>32</td></tr><tr><td>32768</td><td>4096</td><td>512</td><td>256</td><td>16</td></tr><tr><td>32768</td><td>4096</td><td>1024</td><td>128</td><td>8</td></tr><tr><td>32768</td><td>4096</td><td>2048</td><td>64</td><td>4</td></tr></table>

(b) Benchmark configurations used by Figure 13 and 14, and all other kernel-level ablation studies. 

<table><tr><td>Model Size</td><td>T</td><td>d</td><td>n</td><td>E</td><td>K</td></tr><tr><td rowspan="3">1.4B</td><td>40960</td><td>768</td><td>256</td><td>128</td><td>8</td></tr><tr><td>40960</td><td>768</td><td>512</td><td>64</td><td>4</td></tr><tr><td>40960</td><td>768</td><td>1024</td><td>32</td><td>2</td></tr><tr><td rowspan="3">7B</td><td>24576</td><td>1536</td><td>256</td><td>128</td><td>8</td></tr><tr><td>24576</td><td>1536</td><td>512</td><td>64</td><td>4</td></tr><tr><td>24576</td><td>1536</td><td>1024</td><td>32</td><td>2</td></tr><tr><td rowspan="3">30B</td><td>32768</td><td>4096</td><td>256</td><td>256</td><td>16</td></tr><tr><td>32768</td><td>4096</td><td>512</td><td>128</td><td>8</td></tr><tr><td>32768</td><td>4096</td><td>1024</td><td>64</td><td>4</td></tr><tr><td rowspan="3">120B</td><td>32768</td><td>4096</td><td>512</td><td>256</td><td>16</td></tr><tr><td>32768</td><td>4096</td><td>1024</td><td>128</td><td>8</td></tr><tr><td>32768</td><td>4096</td><td>2048</td><td>64</td><td>4</td></tr></table>

• Top-left 2 subfigures: We use $( T , d , n , K ) = ( 1 6 3 8 4 , 1 5 3 6 , 2 5 6 , 8 )$ and we vary $E$ from 64 to 512. 

• Top-right 2 subfigures: We use $( T , d , n , K ) = ( 1 6 3 8 4 , 1 5 3 6 , 1 0 2 4 , 2 )$ and we vary $E$ from 16 to 128. 

• Bottom-left 2 subfigures: We use $( T , d , n , K ) = ( 1 6 3 8 4 , 4 0 9 6 , 5 1 2 , 8 )$ and we vary $E$ from 64 to 512. 

• Bottom-right 2 subfigures: We use $( T , d , n , K ) = ( 1 6 3 8 4 , 4 0 9 6 , 1 0 2 4 , 4 )$ and we vary $E$ from 32 to 256. 

# H Hyperparameter details for LM training

We use the OLMoE codebase (Muennighoff et al. 2025) and its downstream tasks in the default configuration52 except for MMLU: WinoGrande (“wino”) (Sakaguchi et al. 2020), Social IQA (“SIQA”) (Sap et al. 2019), SciQ (Johannes Welbl 2017), PIQA (Bisk et al. 2020), OpenBookQA (“OBQA”) (Mihaylov et al. 2018), HellaSwag (“HS”) (Zellers et al. 2019), COPA (Roemmele, Bejan, and Gordon 2011), CommonsenseQA (“CSQA”) (Talmor et al. 2019), BoolQ (Clark et al. 2019), Arc-Easy and Arc-Challenge (“ArcE” and “ArcC”) (Clark et al. 2018) datasets. We use a deduplicated version of FineWeb-Edu (Ben Allal et al. 2024)53 for pretraining corpus, and train all models with context length of 4096 tokens. 

We always use MoE with SwiGLU for the MoE layers and we use an auxiliary load balancing loss (Shazeer et al. 2017) with coefficient 0.01 but we do not use the router Z loss (Zoph et al. 2022). Our attention block architecture is identical to OLMoE’s attention block. We always tie the weight of the LM head with the weight of the token embedding matrix. 


Table 10: Common configurations for MoE pretraining experiment


<table><tr><td>Config name in Tables 3 and 6</td><td># layers</td><td># attn heads</td><td>d</td><td>n</td><td>E</td><td>K</td><td># tokens in a minibatch</td><td>LR</td><td>WD</td><td>LR scheduler</td></tr><tr><td>0.5B params, 20B tokens, 8/64 activated</td><td>12</td><td>12</td><td>768</td><td>256</td><td>64</td><td>8</td><td>0.5M</td><td>6e-4</td><td>0.01</td><td>cosine w./ warmup (10% steps)</td></tr><tr><td>0.5B params, 40B tokens, 2/64 activated</td><td>12</td><td>12</td><td>768</td><td>256</td><td>64</td><td>2</td><td>1M</td><td>6e-4</td><td>0.01</td><td>cosine w./ warmup (10% steps)</td></tr><tr><td>1.8B params, 40B tokens, 8/256 activated</td><td>12</td><td>12</td><td>768</td><td>256</td><td>256</td><td>8</td><td>1M</td><td>6e-4</td><td>0.01</td><td>cosine w./ warmup (10% steps)</td></tr><tr><td>1.4B params, 50B tokens, 8/128 activated</td><td>18</td><td>12</td><td>768</td><td>256</td><td>128</td><td>8</td><td>1M</td><td>4e-4</td><td>0.01</td><td>cosine w./ warmup (10% steps)</td></tr><tr><td>1.4B params, 100B tokens, 2/128 activated</td><td>18</td><td>12</td><td>768</td><td>256</td><td>128</td><td>2</td><td>2M</td><td>4e-4</td><td>0.01</td><td>cosine w./ warmup (10% steps)</td></tr></table>

For all EC with finetuned TC router experiments in Table 3, we use an additional 4B tokens and we only finetune the router weights with TC top- $K$ routing (all other parameters are frozen). We always use a learning rate of 2e-4, weight decay of 0.01 

and cosine learning rate scheduler with $10 \%$ warmup steps. The number of tokens per minibatch during finetuning is 1M. We do disable auxiliary load balancing loss during TC finetuning. 

For all EC with auxiliary router experiments, we use a 2-layer MLP (each linear layer has size $E \times E$ with SiLU activation) which takes as input the raw router logits and make $E$ independent binary predictions for all experts. We compute the averaged binary cross entropy loss over $E$ labels using the multi-label prediction loss, and scale the loss by 0.01. During the evaluation, we will let EC router compute the raw logits and raw scores and let the auxiliary router mask the token-expert pair with its own confidence score. 

We implement “TC (token drop)” by discarding tokens selected from the TC top- $K$ sorting, or always round down. 