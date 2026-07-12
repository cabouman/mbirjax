<!-- Appendix to plans/projector_kernels/gpu_headroom_plan.md.
Produced 2026-07-12 by a parallel research agent during the headroom-investigation kickoff
(five-agent workflow; this file is one agent's report, reproduced verbatim).
Claims marked "verified" were checked against the repo / the pinned jax 0.10.1 env / cited
sources by that agent; quantitative traffic models are PRE-E0 estimates pending the HLO/ncu
verification pass. -->

# Amdahl accounting: where a large GPU projector-kernel win pays off end to end

All paths absolute; `…/mbirjax` = `/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirjax`, `…/mbirjax_metrics` = `/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirjax_metrics`.

## 1. Code-confirmed structure of the VCD loop (`…/mbirjax/mbirjax/tomography_model.py`)

**Per recon (once):**
- Partitions generated ONCE: `initialize_recon` line 2738 (`gen_set_of_pixel_partitions`); partition sequence generated once (line 2742, default `[0,2,4,6,7]` over granularity `[1,2,4,8,16,32,64,128,256]` → granularities `[1,4,16,64,128]`, extended with 128 — `…/mbirjax/mbirjax/_utils.py:116-117`, `vcd_utils.py:326`).
- Only subset ORDER is permuted per iteration: `vcd_partition_iterator` line 3249 (`np.random.permutation(partition.shape[0])`). Partition CONTENTS are fixed for the whole recon → per-(view, subset) sort permutations for the sorted channel reduce are cacheable across all iterations sharing a partition (11 of 15 default iterations use granularity 128).
- Init when `init_recon=None` (`vcd_recon` lines 2991–3054): 1 direct recon (FBP/FDK = filter + **1 full back projection**) + **1 full forward projection** (init error sinogram) + **1 Hessian** = `compute_hessian_diagonal` → `sparse_back_project(weights, full_indices, coeff_power=2)` (**1 full back-projection-equivalent, once per recon** — lines 3110–3124; line 2210).

**Per subset** (`vcd_subset_updater`, lines 3309–3469): 1 `sparse_back_project` (gradient, line 3370) + qGGMRF prior grad/hess (line 3354; halos staged once per partition pass, line 3263) + 1 `sparse_forward_project` (delta, line 3398; a **second** forward per subset if `positivity_flag`, line 3432) + line search kept ON-device (no per-subset alpha host sync; the only per-subset host ops are a `block_until_ready` on two scalars, line 3394, and Python loop dispatch). Stats sync to host once per ITERATION (line 3169).

**Subset sizes at 1024³** (1024×1008×992, ROR mask ≈ π/4 → ~811k pixels): granularity 128 → **~6.3k pixels per subset call**; granularity 64 → ~12.7k (matches `phase_d_design.md`'s "1.6e4"); i.e. a subset is ≤ one forward pixel-batch (8192) — one scan step, minimal sort amortization (the cone VCD guard cell +1.9% watch item in `fwd_back_findings.md` line 232 is exactly this regime).

## 2. Measured anchors used below

- Post-campaign kernels (H100 n=1, `…/mbirjax/plans/projector_kernels/fwd_back_findings.md`): parallel fwd 8.19 s / back 10.92 s at 1024³; cone fwd 19.4 s; 513³ fwd 342 ms / back 314 ms; 200³ fwd 10.5 ms / back 15.1 ms. Cone back 18.9 s (unchanged).
- Pre-campaign nightly (prerelease, 2026-07-10, `…/mbirjax_metrics/results/gpu/prerelease/regression_gpu_20260710T204130Z_84e9f18d_table.yaml`): parallel 1024³ n=1 fwd 35.0 s / back 18.2 s / `vcd_nonconst` **254.8 s**; 512-class vcd 8.12 s; 200-class vcd 348 ms; cone 1024³ vcd 268.0 s (n=2 only 1.26×; cone back n=2 = **0.87×**).
- Intra-kernel floors: `fwd_noscatter` = 3–4% of the pre-campaign 35 s kernel (≈1.1–1.4 s pure compute → ~15% of the CURRENT 8.19 s fwd kernel); `back_nogather` = 0.08–0.10× of the CURRENT back kernel (~90% gather).
- VCD 200³ (A/B harness cell): ~2.0 s wall, ~95% HOST time / ~0.1 s device kernels (`…/mbirjax/plans/current_plans.md` §3 line 92; `…/mbirjax/.claude/lessons.md` line 133). Note the nightly 200-class vcd cell (348 ms, 3 coarse iterations) shows ~58% projector by algebra — the kernel share collapses as granularity gets finer, which is the production regime.
- **The nightly vcd cell runs only 3 iterations at granularities [1,4,16]** (`…/mbirjax_metrics/tooling/scaling_tests/performance_tracking.py` line 112) — it UNDER-represents fine-granularity (128-subset) host overhead relative to a production 15–50-iteration recon (~1493 subsets ≈ ~3000 sparse projector calls).

## 3. The table

f = fraction of workload wall time in fwd/back GPU device kernels. Max end-to-end speedup if kernels get 5× = 1/((1−f)+f/5). Provenance tags: **[M]** measured, **[D]** derived by algebra from measured pieces (assumption: per-subset projector work over a partition sums to ≈ one full-projection-equivalent; exact at granularity 1, degrades at 128), **[X]** does not exist.

| Workload (H100, n=1) | Wall time | f (fwd/back kernel fraction) | Max E2E speedup @ 5× kernels | Notes / provenance |
|---|---|---|---|---|
| Full fwd projection, parallel 1024³ | 8.19 s [M] | ≈1.0 (device-bound single driver call) | ≈5×, but capped ≈3.1× if only the channel reduction (85% of kernel) speeds up; absolute floor ≈6.8× (`fwd_noscatter`) | [M] findings.md scoreboard + noscatter control |
| Full fwd projection, cone 1024³ | 19.4 s [M] | ≈1.0; hfan/vfan split at 1024³ **not measured** [X] | ≈5× only if the untouched vertical fan is also rewritten; hfan-only win bounded by vfan share (unknown) | June 256³ trace: sinogram scatter ~25% of the pre-campaign kernel; cone-fwd ncu is open item #1 in `…/mbirjax_metrics/experiments/profiling/key_findings.md` |
| Full back, parallel 1024³ | 10.92 s [M] | ≈1.0; ~90% of kernel is the gather | ≈3.6× (5× on the gather); floor 10–12× (`back_nogather`) | [M] |
| Full back, cone 1024³ | 18.9 s [M] | ≈1.0 | needs a whole-kernel restructure — per-fan wins are a measured composition no-op (gather hides behind vfan band work) | [M] `cone_back_kernel_ab.py` result |
| Hessian (coeff_power=2, once/recon) | ≈ 1 full back [D] | same as back | same as back; but only ~3% of a 15-iteration recon (~1 of ~32 full-projection-equivalents) | code-confirmed once per recon |
| FBP/FDK init (direct_recon) | filter 0.12 s + full back [M pieces] | ≈0.99 | ≈4.5–5× | filter is ~1% at 1024³ |
| VCD recon 1024³ (nightly config: 3 coarse iters + init) | 254.8 s pre-campaign [M]; post-campaign wall **not measured** [X], ≈111 s [D] | ≈0.91 pre [D]; ≈0.78 post [D] | ≈2.7× (post-campaign composition) | derived: init (back+fwd+hessian) + 3×(fwd+back) ≈ 231 s of 254.8 s |
| VCD recon 512³-class (same config) | 8.12 s pre [M]; ≈3.4 s post [D] | ≈0.94 pre / ≈0.85 post [D] | ≈3.1× | same derivation; residual ~0.5 s host/prior/stats |
| **VCD production regime (15–50 iters, granularity 128, 512³/1024³)** | **[X] no measurement exists anywhere** | unknown — between the 78–85% coarse-granularity bound and the 200³ fine-granularity 5% | **the deciding number for the whole custom-kernel investment** | see measurement proposal below |
| VCD recon 200³ (A/B cell) | ~2.0 s [M] | ≈0.05 [M, device trace] | ≈1.04× — kernel wins pay nothing here | host-dispatch pool (current_plans §3) |
| FDK init + MAR fit | [X] no wall split recorded | projector content = (1+num_metal) full fwd projections (`…/mbirjax/mbirjax/preprocess/mar.py` lines 202–207); the fit itself is elementwise H-column recompute (the current bottleneck per current_plans §4) | kernel wins do NOT touch the fit; they speed only the (1+num_metal) fwd projections + the downstream MBIR recon | [X] |
| Multi-device back (band path) | cone 1024³ n=2 = 0.87× of n=1 [M] | limiter = band transpose in `back_project_one_view_to_band`, **L1/TEX-bound (99.9% L1, 6% HBM)**; reduce-scatter cheap (~3.4 ms) | separate beneficiary: a custom band kernel avoiding the transpose (the "B4.5 lever") unlocks multi-GPU back scaling and hence cone VCD n≥2 (now 1.26× vs parallel's 2.05×) | [M] ncu, key_findings.md |

## 4. Where the leverage actually is

1. **Kernel wins pay ~fully in the full-projection workloads** (forward_project/back_project, FBP/FDK init, Hessian, MAR's forward projections): f ≈ 1, so a 5× kernel → ~3–5× end to end (bounded by intra-kernel compute floors: fwd ~15%, back ~10%).
2. **In VCD they pay only at large size AND only as far as the per-subset regime allows.** Coarse-granularity VCD at 1024³ is ~78–91% projector [D] → a 5× kernel is worth ~2.7×. But production iterations run at granularity 128 (~6.3k-pixel subsets = single pixel-batch calls), where per-call fixed costs and host dispatch grow — 200³ shows the limit (~95% host, ≤1.04× from any kernel win). The 1024³ fine-granularity number does not exist and is the gating measurement.
3. **Real scanner data is cone** (preprocess ingestion: `nsi.py`, `zeiss.py` cone; `zeiss_tct.py` translation; helical demo = cone; flash-remediation validation = real cone data) — so a custom-kernel effort that only reproduces the parallel-beam wins misses the dominant geometry: cone forward's vertical fan is untouched, and cone back needs a whole-kernel restructure (per-fan substitutions are a composition no-op).
4. **Multi-device is a distinct beneficiary**: the band transpose (L1-bound, not comms) currently makes cone back anti-scale (0.87× at n=2), which caps cone VCD multi-GPU at 1.26×. A custom band kernel is the one change that converts kernel work into SCALING rather than single-device time.

## 5. Missing numbers and the cheapest way to get each

| Missing number | Cheapest measurement |
|---|---|
| Device-kernel vs host share of VCD at 512³/1024³ at **production granularity** | `jax.profiler.trace` around a warm 2-iteration `vcd_recon` with `partition_sequence=[7]` (granularity 128) on one H100; attribute with the existing tooling in `…/mbirjax_metrics/experiments/profiling/` (`trace_utils.py`, `region_attribution.py`, `profile_measure.py` — the same device-trace flow that produced the 200³ ~95%-host number). Minutes at 512³, ~10 min at 1024³. |
| Post-campaign VCD wall at 512³/1024³ | one nightly-harness run of the `vcd_nonconst` cells on `greg/kernel_investigation` (no kernel-branch results dir exists yet under `…/mbirjax_metrics/results/gpu/`). |
| Cone forward vfan/hfan split at 1024³ | replicate the `cone_back_kernel_ab.py` split-bench pattern for forward (open item #1 in `key_findings.md`), or ncu the vfan fusions — single sbatch cell. |
| MAR wall split (projections vs H-column fit) | `time.perf_counter` brackets around `_est_plastic_metal_sinos_from_recon` vs `_estimate_BH_model_params` in the existing real-data MAR script (context in `mar_refactor_plan.md`). |
| Per-subset fixed cost vs pixel-proportional cost at 6.3k-pixel subsets | sweep `sparse_forward/back_project` wall vs pixel count {1.6k, 6.3k, 12.7k, 50k, full} at 1024³ — one script in the `…/mbirjax/plans/experiments/projector_kernels/` bench style; directly predicts fine-granularity VCD kernel share without a full recon. |

**Bottom line:** a 5× projector-kernel win is worth ~3–5× on every full-projection workload (init, Hessian, forward/back APIs, MAR's projections), a derived ~2.7–3.1× on coarse-granularity large VCD, ~nothing at 200³, and an unmeasured amount — the single most decision-relevant gap — on production fine-granularity VCD at 512³/1024³. The multi-device band transpose is a second, independent payoff (scaling, not single-device time), and cone (the real-data geometry) requires rewriting the vertical fan and the composed cone back kernel, not just the horizontal-fan primitives the campaign already optimized.
