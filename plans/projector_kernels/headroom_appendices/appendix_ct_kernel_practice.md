<!-- Appendix to plans/projector_kernels/gpu_headroom_plan.md.
Produced 2026-07-12 by a parallel research agent during the headroom-investigation kickoff
(five-agent workflow; this file is one agent's report, reproduced verbatim).
Claims marked "verified" were checked against the repo / the pinned jax 0.10.1 env / cited
sources by that agent; quantitative traffic models are PRE-E0 estimates pending the HLO/ncu
verification pass. -->

# Survey: How State-of-the-Art GPU Tomography Codes Implement Projectors

**Scope note on sources.** Everything in sections 1–5 marked "established" is sourced from fetched papers/docs (URLs inline). Normalized throughput numbers in §2 are **my own arithmetic** on published figures; the final calibration conclusions are **inference** and labeled as such.

---

## 1. Projector taxonomy across the major codes (established)

| Code | Forward | Back | Matched pair? | Texture/HW interp | Language |
|---|---|---|---|---|---|
| **ASTRA Toolbox** | Ray-driven (Joseph kernel): one thread per detector pixel steps along the ray, trilinear texture fetches; line lands exactly on detector-pixel midpoint so no sinogram re-interpolation | Voxel-driven: thread per (x,y) column, registers accumulate a stack of N̄z=6 voxels over N̄θ=32 angles per launch, bilinear texture fetch into sinogram | **No — explicitly unmatched** | Yes, 3D CUDA Arrays, HW bi/trilinear | CUDA (KernelKit: NVRTC runtime-compiled CUDA via CuPy) |
| **TIGRE** | Siddon (exact ray-voxel intersection) **or** interpolated (samples along ray at Δl via 3D texture, trilinear) | Voxel-driven with texture read of sinogram; two weightings: FDK weight (**unmatched**) or "matched" weight per Jia et al. | Optional pseudo-matched mode | Yes | CUDA |
| **LEAP (LLNL)** | Separable-Footprint (SF)-class model; parallelized **over output samples in each direction** (detector pixels for A, voxels for Aᵀ) — gather both ways, same weight model | Same SF weights, voxel-parallel | **Yes — matched pairs are the headline feature** | 3D texture for the *input* array in each direction | CUDA |
| **RTK (Kitware)** | Joseph (ray-driven) CUDA forward | CUDA voxel-based backprojector (also Joseph-transpose option) | Mostly unmatched by default | Yes | CUDA/ITK |
| **svMBIR / MBIRCONE (Bouman group)** | **Precomputed system matrix** (trapezoid/SV footprints), encoded in a cache-friendly super-voxel layout, stored on disk keyed by geometry (`~/.cache/svmbir/sysmatrix`) | Same stored matrix (exact transpose) | **Yes — same A both ways** (like mbirjax) | n/a | C, CPU-only |
| **PYRO-NN (2025 update)** | Ray-driven (rays cast per detector pixel) | Voxel-driven | Not verified adjoint; "analytical gradients" | Texture-based *or* kernel interpolation (memory/speed tradeoff) | CUDA in TF/PyTorch layers |
| **CTorch (2025)** | Implements all four (voxel-, ray-, distance-driven, SF); ray-driven is their fastest forward, voxel-driven fastest back | Exact adjoint of each forward is implemented (voxel-driven forward / ray-driven back exist *only* to serve as gradients of their adjoints) | **Yes — strict adjoint pairs** | CUDA C + pybind11 | Raw CUDA |
| **Branchless distance-driven (GE-lineage, multi-GPU helical)** | DD factored into integration → interpolation → differentiation on pre-accumulated slabs; forward is read-only gather | Transposed pipeline ("anterpolation"); scatter contention solved with per-GPU private partial accumulators summed after a barrier | Matched in model (transposed pipeline) | Yes | CUDA |

Key structural facts (all established):

- **The forward-scatter problem is universally avoided by gathering.** Every GPU code parallelizes the forward pass over *detector* elements (ray-driven/detector-driven) or, in LEAP/CTorch's matched designs, over whichever array is the output of that operator. ASTRA's authors state the unmatched design "is advantageous for an implementation on GPUs. All parallel threads are independent... which avoids potential race conditions" ([KernelKit paper, §2.1](https://ir.cwi.nl/pub/34711/34711.pdf), also [doi:10.3934/ammc.2024004](https://www.aimsciences.org//article/doi/10.3934/ammc.2024004)).
- **Matched pairs do not require scatter.** LEAP and CTorch prove the pattern: implement A and Aᵀ as two *separate gather kernels sharing one weight function*. LEAP: "parallelization is done over the samples in the output space," with 3D texture on the input array ([arXiv:2307.05801](https://arxiv.org/html/2307.05801); [LEAP features](https://github.com/LLNL/LEAP/blob/main/LEAP_features.md)).
- **ASTRA's back kernel is the canonical register-tile-across-views design**: per-thread registers hold 6 voxels updated across 32 angles per kernel launch, with block/chunk constants (16,32,6,32) chosen by brute-force search; retuning them per problem yields 8–44% ([KernelKit](https://ir.cwi.nl/pub/34711/34711.pdf)).
- **Hardware texture interpolation is a large part of hand-CUDA speed** (10–16× vs global memory in one controlled study: [FSNP, PMC4664243](https://pmc.ncbi.nlm.nih.gov/articles/PMC4664243/)). *Inference relevant to mbirjax:* XLA/JAX cannot emit texture-object lerps, and CUDA texture lerp uses low-precision fixed-point weights (9-bit, per CUDA programming guide) — a float-correctness hazard for MBIR-grade gates anyway.

---

## 2. Throughput calibration (published numbers → voxel·view updates/s; normalization is my arithmetic)

| Source | Problem | GPU (≈BW) | Time | Updates/s | Updates/s per TB/s |
|---|---|---|---|---|---|
| **mbirjax** (your baseline) | parallel fwd 1024³×1024 | H100 (3.35 TB/s SXM assumed) | 8.19 s | 134 G/s | **40 G** |
| **mbirjax** | parallel back, same | H100 | 10.92 s | 101 G/s | **30 G** |
| **mbirjax** | cone fwd, same | H100 | 19.4 s | 57 G/s | **17 G** |
| **LEAP** (Table 1) | parallel 1024³×720 fwd | Tesla P100 (0.73 TB/s) | 11.5 s | 67 G/s | **92 G** |
| **LEAP** | cone 1024³×720 | P100 | 37.1 s | 21 G/s | **28 G** |
| **LTT** (same table) | parallel 1024³×720 | P100 | 17.4 s | 44 G/s | 61 G |
| **Branchless DD** | helical fwd 512²×164 vol, 13,920 views | TITAN X (0.34 TB/s) | 15 s | 40 G/s | **119 G** |
| **Branchless DD** | back, same | TITAN X | 22 s | 27 G/s | 81 G |
| **ASTRA/KernelKit** tuned quasi-3D BP | 2016² slice × 300 views | RTX A6000 (0.77 TB/s) | 15.7 ms | 78 G/s | 101 G |
| **ASTRA/KernelKit** tuned single kernel | 2000² slice × 32 views | A6000 | 0.397 ms | 322 G/s | 420 G (cache-friendly small-Nθ regime) |

Sources: [LEAP paper](https://arxiv.org/html/2307.05801), [multi-GPU branchless DD, PMC5448423](https://pmc.ncbi.nlm.nih.gov/articles/PMC5448423/), [KernelKit](https://ir.cwi.nl/pub/34711/34711.pdf). torch-radon reports >40× (fp32) / 125× (fp16) vs ASTRA for *batched 2D* — what full specialization + fp16 textures buys in the 2D regime ([arXiv:2009.14788](https://arxiv.org/pdf/2009.14788), [repo](https://github.com/matteo-ronchetti/torch-radon)).

**Calibration conclusions (inference, but numerically convergent):**

1. Large-3D state of the art clusters at **≈80–120 G voxel·view updates/s per TB/s** of DRAM bandwidth across three independent codebases/architectures. mbirjax parallel forward sits at 40 G (≈**2.3× below** SOTA), back at 30 G (≈**3×**), cone forward at 17 G (vs LEAP cone's 28 G, ≈**1.7×**). If the H100 is PCIe (2 TB/s), the gaps shrink to ~1.4–2×.
2. This cross-checks your own control: fwd_noscatter ≈1.1–1.4 s is the compute floor; SOTA-normalized expectation for a hand-written H100 forward is 1024³×1024/(0.3–0.4 T/s) ≈ **2.7–3.7 s** vs the current 8.2 s. So the realistic custom-kernel prize is **~2–3×, not 10×**. The internal "10× above compute-only bounds" is a roofline-vs-compute statement; *published SOTA kernels are also far above compute-only bounds* — everyone is access-pattern-limited, matching your June ncu finding (L1-bound, 8% HBM).
3. The one regime where published kernels hit 3–4× higher (KernelKit's 420 G per TB/s) is small view-count per launch with register accumulation — i.e., precisely the tiling regime, which supports the register-tile back-kernel plan.

---

## 3. Matched vs unmatched pairs in iterative reconstruction

**Established literature:**

- **Zeng & Gullberg (2000)** — the foundational analysis. With backprojector B ≠ Aᵀ, the iteration converges only if eigenvalues of the pair product lie in (0,2) (positivity required), and it converges to a *different* fixed point, (BᵀC)⁻¹BᵀP. With noise, both matched and unmatched show semi-convergence (improve then diverge). They also note an unmatched ray-driven-forward/voxel-driven-back pair can *suppress* ring artifacts. [IEEE TMI](https://ieeexplore.ieee.org/document/870265/), [PMC5297459](https://pmc.ncbi.nlm.nih.gov/articles/PMC5297459/)
- **Dong, Hansen et al., SIAM SISC 2019** — SIRT/Landweber-type methods with unmatched B generically *fail to converge* (nonsymmetric iteration matrix); fix = a computed shift parameter (estimated from the leftmost eigenvalue via Krylov methods), converging to a *perturbed* problem with bounds. [arXiv:1902.04282](https://arxiv.org/abs/1902.04282), [SISC](https://epubs.siam.org/doi/10.1137/18M1206448)
- **AB-/BA-GMRES (Hansen et al. 2022)** — reformulate so the unmatched pair is handled by GMRES on ABy or BAx; reduces to LSQR/LSMR when matched. [J. Comp. Appl. Math](https://www.sciencedirect.com/science/article/pii/S037704272200156X). Hybrid (Tikhonov-in-Krylov) variants that tame semi-convergence appeared in 2026: [arXiv:2602.17892](https://arxiv.org/pdf/2602.17892).
- **The toolbox authors themselves take sides.** ASTRA (unmatched, speed-first): "unmatched projectors lead to nonconvergence in iterative algorithms... In the presence of noise, this does not always pose a problem" ([KernelKit](https://ir.cwi.nl/pub/34711/34711.pdf)). LEAP (matched-first): unmatched projectors "may produce artifacts when used over enough iterations"; matched pairs "ensure convergence" ([LEAP features](https://github.com/LLNL/LEAP/blob/main/LEAP_features.md)). TIGRE ships both weightings ([TIGRE paper](https://iopscience.iop.org/article/10.1088/2057-1976/2/5/055010)).

**Implication for VCD (my inference, clearly flagged):** the entire unmatched-convergence literature and all its fixes (shift, GMRES wrapping) apply to gradient/Krylov outer loops. There is **no literature supporting unmatched operators inside coordinate descent**, where the update uses the exact column A_j and a Hessian term built from A_j² — the majorization argument requires the true columns. So: (a) adjoint-breaking algorithmic rewrites are **inadmissible for the VCD inner loop** without new theory; admissible only for one-shot operations (FDK-style initialization, filtering, weight builds). (b) The reorganizations you're considering (precomputed CSR of the *same* trapezoid weights; register-tiled back) compute the **same matrix A** and therefore preserve matchedness *by construction* — only float summation *order* changes, which is a reproducibility-gate concern, not a convergence-theory concern. svMBIR is the existence proof that an MBIR coordinate-descent code built on a precomputed, cached, exactly-shared system matrix is standard practice ([svmbir docs](https://svmbir.readthedocs.io/en/latest/overview.html)).

---

## 4. Prior art for the specific planned pattern (precomputed per-view CSR pixel→channel segments; register-tile back)

- **PyFAI (ESRF) is the canonical published example of exactly this transform.** Azimuthal integration is the same scatter problem (pixels → bins with fractional "pixel splitting" weights). They: (1) precompute the pixel→bin mapping with weights once; (2) store it as **CSR** ("struct of arrays... better suited to GPUs"; 2–3× smaller than LUT); (3) thereby switch "from a 'linear read / random write' forward algorithm to a 'random read / linear write' backward algorithm which is more suitable for parallelisation"; (4) compute each bin by a **workgroup-cooperative parallel reduction over its CSR segment**; (5) use **Kahan compensated summation** to keep fp32 accurate. [arXiv:1412.6367](https://arxiv.org/pdf/1412.6367). Every element of your proposed forward (concrete channel centers → precomputed variable-length segments → per-channel segment-gather-reduce) has a direct published counterpart here.
- **Sparse Matrix-Based HPC Tomography (2020)**: full precomputed CSR system matrix per slice, **A and Aᵀ stored separately**, cached to disk keyed by a hash of the geometry; forward/back = SpMV; scaled to 16 V100s. [PMC7302278](https://pmc.ncbi.nlm.nih.gov/articles/PMC7302278/). svMBIR does the same on CPU with the super-voxel cache layout (100–1000× vs naive MBIR claimed) ([overview](https://svmbir.readthedocs.io/en/latest/overview.html)).
- **Segmented-reduction kernel machinery** is mature: ModernGPU `segreduce` ([moderngpu.github.io/segreduce](https://moderngpu.github.io/segreduce.html)) and CSR5's load-balanced SpMV ([arXiv:1503.05032](https://arxiv.org/pdf/1503.05032)) are the standard warp-level building blocks for variable-length segments — directly what a hand-written per-channel gather would use. Note your current `sort_key_val + segment_sum(indices_are_sorted)` is a *runtime-sorted* software segmented reduction; the CSR plan removes the sort from the inner loop, which is exactly PyFAI's win.
- **Register-tile accumulation across views in the back kernel**: ASTRA's voxel-driven BP (KernelKit Algorithm 1) *is* this structure — per-thread register accumulation of a 6-voxel z-stack across 32 views per launch, single write-out, constants found by exhaustive tuning. The branchless-DD multi-GPU paper adds the scatter-side recipe: private per-device partial accumulators, summed after a barrier ([PMC5448423](https://pmc.ncbi.nlm.nih.gov/articles/PMC5448423/)). The FSNP paper is precedent for **precomputing per-view geometry tables and exploiting view symmetry** to keep the tables small (2.5 GB → tractable) ([PMC4664243](https://pmc.ncbi.nlm.nih.gov/articles/PMC4664243/)).
- "Detector-driven" as a named formulation: what you'd hand-write for the forward is what the field calls ray-driven/detector-driven gather; the *novel* part of your plan is only that the segment lists come precomputed from voxel-side trapezoid weights (preserving matchedness), rather than being re-derived ray-side — for which PyFAI + the CSR tomography codes are the precedent, not the CT-projector papers.

---

## 5. CT projectors in Triton / Pallas / compiler DSLs (2023–2026)

- **GPAIR (Feb 2026, photoacoustic CT)** — the only published tomographic forward/adjoint pair I found written in **Triton**: closed-form Gaussian-kernel forward operator + adjoint as custom autograd functions, "optimized GPU-native Triton kernels," RTX 5090, sub-second iterative recon of 8.4M-voxel volumes with 1024 detectors, 200–900× vs the prior IR baseline. [arXiv:2602.03893](https://arxiv.org/pdf/2602.03893). Notably they made the problem DSL-friendly by choosing *fixed-size analytic kernels* (regular per-source work) — analogous to your fixed T=3 taps.
- **No published X-ray CT projector in Triton or Pallas found** (searched multiple phrasings). The 2025 wave of projector toolboxes — CTorch ([arXiv:2503.16741](https://arxiv.org/html/2503.16741v3)), PYRO-NN update ([arXiv:2511.08427](https://arxiv.org/html/2511.08427)), LEAP — are all raw CUDA C. A Julia KernelAbstractions.jl Radon implementation exists ([GitHub topic](https://github.com/topics/radon-transform?o=desc&s=)).
- Closest "compiler-era" practice: **ASTRA KernelKit** — CUDA kernels runtime-compiled via NVRTC/CuPy, Jinja2-templated code paths, autotuned with Kernel Tuner; tuning alone gave 8–44% and texture-backend choice mattered (layered CUDA arrays best for one-shot BP, pitched linear memory competitive when y needs write access) ([paper](https://ir.cwi.nl/pub/34711/34711.pdf)).
- *Inference:* a Pallas cone/parallel projector would be publishable novelty, with two known risks: (1) block-oriented DSLs dislike variable-length segments (mitigated by CSR padding/bucketing to fixed block sizes, as PyFAI/CSR5 do); (2) no access to texture hardware — but matched MBIR-grade weights can't use 9-bit texture lerp anyway, so the DSL route forfeits less than it would for an FDK code.

---

## 6. Bottom line for the mbirjax investigation (inference)

1. **Recalibrate the target:** vs published, bandwidth-normalized SOTA (LEAP, branchless DD, tuned ASTRA), the current kernels are ~1.7–3.4× off, not 10×. A hand-written H100 kernel plausibly lands at ~2.7–3.7 s forward / ~3.5–4 s back at 1024³×1024 — worth having, but plan around ~2–3×.
2. **The planned structures are well-precedented:** precomputed-CSR segment-gather forward (PyFAI, sparse-HPC tomography, svMBIR) and register-tiled multi-view back (ASTRA's own BP) are the two proven patterns; nothing exotic is required.
3. **Adjointness constraint is real:** unmatched-pair fixes exist only for gradient/Krylov methods; for VCD, only reorganizations that compute the same A are admissible — which the CSR plan satisfies by construction (watch summation-order reproducibility, and Kahan/fp32 per PyFAI if needed).
4. **DSL gap:** no X-ray CT projector in Triton/Pallas exists in the literature; GPAIR (Triton, PACT, 2026) is the nearest existence proof that a tomographic matched pair in a block DSL can hit SOTA speeds when per-element work is regularized — which mbirjax's fixed-tap trapezoid model already is.
