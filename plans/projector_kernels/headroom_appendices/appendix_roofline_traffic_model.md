<!-- Appendix to plans/projector_kernels/gpu_headroom_plan.md.
Produced 2026-07-12 by a parallel research agent during the headroom-investigation kickoff
(five-agent workflow; this file is one agent's report, reproduced verbatim).
Claims marked "verified" were checked against the repo / the pinned jax 0.10.1 env / cited
sources by that agent; quantitative traffic models are PRE-E0 estimates pending the HLO/ncu
verification pass. -->

# Bytes-and-FLOPs traffic model: parallel-beam GPU projectors at the 1024³ cell (H100, n=1)

## 1. Verified configuration (what I verified vs assumed)

**Verified by executing the library on CPU** (model construction + direct `_select_tile_policy(on_gpu=True, ...)` call, matching how `fwd_band_pixel_sweep.py`/`back_tile_sweep.py` build the cell):

| quantity | value | source |
|---|---|---|
| sinogram | (V=1024 views, R=1008 det rows, C=992 channels) fp32 | `SINO_SHAPES` in both sweep scripts |
| recon | (992, 992, S=1008), auto-derived | `model.get_params('recon_shape')` |
| pixels P_tot | **771,240** (ROR mask, 78.4% of 992²) | `gen_full_indices(..., use_ror_mask=True)` — the sweeps use the ROR-masked grid |
| psf taps T | 3 (psf_radius=1) | `model.get_psf_radius()` |
| fwd policy | view_batch=128, pixel_batch=8192, slice_band=256 → **4 balanced bands of B=252**, `sort_by_channel=True` | TilePolicy + `_balanced_slice_bounds(1008, 256)` = [(0,252),(252,504),(504,756),(756,1008)] |
| back policy | view_batch=128, pixel_batch=2048, `back_stacked_gather=True`; n=1 GPU **monolithic short-circuit** (`_sparse_back_project_single_device`), transfer chunks = 4 × 192,810 pixels × 8 view batches | TilePolicy + tomography_model.py:1882–1895 |
| measured times | fwd **8.19 s**, back **10.92 s** — model-level, single trial (`TRIALS_1024=1`) | findings doc Phase B / back tables |

**Assumed:** H100 SXM numbers as given (3.35 TB/s HBM, 60 TFLOP/s fp32, 50 MB L2, 132 SM × 228 KB L1/SMEM); XLA's materialization/fusion behavior (flagged below — the decisive unknowns); radix-sort pass count (8-pass classic vs ~2-pass onesweep). GB = 10⁹ B throughout.

## 2. Corrections to the framing in the prompt

1. **Forward at 1024³ n=1 is not one scan-over-pixels program.** It is 4 slice-bands (B=252, not 1008) × 8 view chunks (the wrapper's `N_PC_SINGLE_CALL_MAX_BYTES`=256 MB rule: 4·1024·771240 = 3.16 GB > 256 MB → chunk by `fwd_view_batch`=128) = **32 jit calls**, each a 95-step pixel scan (initial partial 1,192 px + 94 × 8,192). Inside each jit call there is exactly **one** view batch — `concatenate_function_in_batches` is a pass-through. The (V,B,C) carry is (128, 252, 992) = **128 MB**, not ~512 MB (that figure was pre-Phase-B).
2. **Back at n=1 is an eager host loop**, not a scan: Python loop over 8 view batches × 4 pixel-transfer chunks, with `block_until_ready()` per chunk (32 syncs) and an eager 3.11 GB accumulator (`recon_at_indices + concatenate(...)`, ~22 ms + dispatch bubbles). The 128-view sum is a single vmap+`jnp.sum` inside each jit call. Back pixel batch = **2048**, not 8192.
3. `n_p_centers` for forward is **recomputed per band** (`scatter_centers` inside `one_call`, called per band×chunk): 4× redundant, 12.6 GB written → ~4 ms. Negligible time, ~0.4 GB resident per chunk.
4. The (C, rows) channel-major view for back is produced by a transpose **inside the kernel under vmap**; assuming XLA's while-loop invariant code motion hoists it out of the pixel `lax.map`, it costs ~10 ms total. If NOT hoisted it is 1.03 GB/step → **0.94 s** — worth one HLO check.

Both kernels execute exactly **3,040 steps** (fwd: 4×8×95; back: 8×4×95) — measured 2.69 ms/step fwd, 3.59 ms/step back.

## 3. Forward kernel traffic model (per step: V=128, P=8192, B=252, C=992, T=3)

Per-step array sizes: voxel tile (8192×252) = 8.26 MB; `updates` (V, T·P, B) = **3.171 GB**; segment-sum output / carry (V, 992, 252) = 128 MB; sort keys+payload = 25.2 MB/pass.

| term (GB/step) | (a) naive | (b) perfect-L2-in-step | notes |
|---|---|---|---|
| voxel reads | 3.171 (gather-amplified, all miss) | 0.008 (source is 8.26 MB, L2-resident) | T× row amplification |
| n_pc tile | 0.004 | 0.004 | |
| in-kernel sort (r+w) | 0.403 (8-pass) | ~0 (25 MB fits L2) | |
| `updates` write + scatter re-read | 6.342 | 0 (fused) | **the big one** |
| scatter RMW on (C,B) cells | 6.343 | 0 (on-chip) | sorted → cache-absorbed in practice |
| segsum out + carry RMW | 0.512 | 0.256 | carry is irreducible per step |
| **total × 3040 steps** | **51.1 TB → 15.24 s** | **0.82 TB → 0.244 s** | |

**(c) ideal fused:** cylinder 3.11 GB re-read per view chunk (×8) + n_pc 3.16 GB + sinogram write 4.10 GB = 32.1 GB → **9.6 ms**; with once-total voxel reads 10.4 GB → 3.1 ms. **FLOP floor:** 2·V·P·T·S = 4.78e12 → **0.080 s** (binds over (c)).

**Sanity crosscheck:** measured 8.19 s ≈ `updates` write+read at peak BW (**5.76 s**) + in-kernel sort (findings: ~0.65 ms/call at P=2048 → est. 0.7–1.5 ms/step at P=8192 → **2.1–4.6 s** over 3040 steps) = 7.9–10.4 s. The measured time is fully explained by *(materialized updates stream) + (sort latency)*; the sorted-scatter RMW itself is already cache-absorbed (that was Phase A's win — measured is 0.54× the naive bound).

## 4. Back kernel traffic model (per step: V=128, P=2048, R=1008, C=992, T=3)

Per-step: `gathered` (V, T·P=6144, R) = **3.171 GB**; per-view stack (V, P, R) = 1.057 GB; distinct source = 128 views × 4.0 MB = 512 MB (× ~2/π ≈ 0.64 mean channel footprint per view → 335 MB); output (P,R) = 8.26 MB.

| term (GB/step) | (a) naive | (b) perfect-L2-in-step |
|---|---|---|
| gather source reads | 3.171 (every row miss) | 0.335–0.512 (distinct bytes only) |
| `gathered` write + `weighted` r+w + tap-sum read | 12.68 | 0 (fused) |
| tap-sum write + view-sum read | 2.114 | 0 |
| output + n_pc | 0.009 | 0.009 |
| **total × 3040** | **54.7 TB → 16.31 s** | **1.02–1.58 TB → 0.30–0.47 s** |

Note (b) is a *per-step* bound: the 512 MB view working set cannot sit in 50 MB L2, so re-reading it every step is compulsory without restructuring. **(c) ideal:** sinogram 4.10 + n_pc 3.16 + output 3.11 = 10.4 GB → **3.1 ms**; **FLOP floor** 2·V·P·T·R + V·P·R = 5.57e12 → **0.093 s**.

Measured 10.92 s = 0.67× the full-materialization bound (a) → the chain is **partially but not fully fused** (a fully-fused program would be ≤ 2.89 s, see below).

## 5. Headroom scoreboard

| | measured | FLOP floor | (a) naive HBM | (b) L2-perfect HBM | (c) ideal fused |
|---|---|---|---|---|---|
| **forward** | 8.19 s | 0.080 s (**103×**) | 15.24 s (0.54× — already beaten) | 0.244 s (**34×**) | 0.010 s; floor→FLOP 0.080 s |
| **back** | 10.92 s | 0.093 s (**118×**) | 16.31 s (0.67×) | 0.30–0.47 s (**23–36×**) | 0.003 s; floor→FLOP 0.093 s |

Caveat: HBM-floor arithmetic understates the true floor for these kernels — June ncu (pre-campaign) showed the accumulate fusion at 97% **L1** utilization / 8% HBM, i.e. the on-chip gather/scatter pipes, not HBM, are the wall once traffic is fused away. The realistic custom-kernel floor is max(FLOP floor, L1-throughput of ~T·P·B lane-work) — likely ~0.1–0.3 s each, still **~30–50×** below measured.

## 6. Dominant terms and what removes them

**Forward — dominant: the materialized (V, T·P, B) `updates` array (write+scatter-re-read = 19.3 TB = 5.8 s at peak BW) plus the per-step sort (~2–4.6 s).**
- *XLA-level:* fuse `A[order]·values[order % P]` into the segment-sum's scatter input so `updates` never hits HBM. If XLA can be induced to do this (HLO check: is there a 3.17 GB fusion output per step?), the traffic endpoint is ~0.63 GB/step → **0.57 s** + sort. Realistic XLA-only endpoint ≈ 2.5–5 s (1.6–3×).
- *Driver-level (no custom kernel): kill the sort.* `order`/`sorted_n` are pure functions of `n_pc`, which is **already precomputed eagerly** per (view, pixel-batch). Precomputing the permutation the same way (once per chunk, reused across the 4 bands — and across repeated VCD calls with fixed pixel sets) removes ≥¾ of the 2–4.6 s sort cost. Cost: ~1.2 GB resident per chunk (3×n_pc).
- *Custom kernel only:* on-chip accumulation of the (C,B) output tile across pixel batches (kills the 128 MB carry RMW ×3040 = 0.35 s AND the segsum output write), cross-view voxel-tile reuse from SMEM, tap dedup — the path from ~0.5 s to the ~0.1–0.25 s floor.

**Back — dominant: materialized gather intermediates (the (V,T·P,R) `gathered`/`weighted` streams + the (V,P,R) per-view stack = 14.8 of 18.0 GB/step).**
- *XLA-level (highest-leverage single target):* one fusion gather→·A→tap-sum→view-sum. Bound if every gathered row misses: 3.18 GB/step → **2.89 s (3.8×)**. Tap overlap (rows n−1,n,n+1 of one pixel; ~6.2 pixels/channel within a batch, channel-monotone pixel order) is cache-catchable even inside a fused kernel → **0.97–1.1 s** plausible. Also verify the view-transpose is hoisted (§2.4).
- *Custom kernel only:* cross-pixel-batch reuse of the 512 MB view set (explicit (view-tile × pixel-tile) blocking so a view tile stays in L2 across many pixel batches) to reach (b) 0.30–0.47 s robustly, and cross-view-batch reuse toward the 0.09 s floor.

**Not worth touching at n=1:** n_pc recompute (4 ms), transposes (~10 ms), eager back accumulator (22 ms), chunk-concat copies (~10 ms), 32 host syncs — all <1% each.

## 7. Recommended verification (cheap, decisive)

1. One `XLA_FLAGS=--xla_dump_to` HLO dump of `_jit_sparse_forward_project`/`_jit_sparse_back_project` on the GPU node: does a ≥3 GB fusion output exist per step (updates/gathered materialized)? Is back's view-sum a separate kernel? Is the transpose hoisted? This discriminates the (a)-vs-fused stories above without any timing.
2. ncu on the CURRENT kernels (the June data predates the campaign): confirm the sorted segment-sum and stacked gather are L1-bound vs HBM-bound — this decides whether Pallas effort should target traffic (fusion/tiling) or access-pattern (coalescing/vectorization) first.

Files: `/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirjax/mbirjax/projectors.py` (kernels 107–125, 203–302; drivers 801–948; wrapper chunking 505–563), `/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirjax/mbirjax/parallel_beam.py` (policy 86–117, kernels 236–332), `/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirjax/mbirjax/tomography_model.py` (banded fwd 1626–1680, back short-circuit 1709–1765, 1882–1895), benchmark shapes `/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirjax/plans/experiments/projector_kernels/fwd_band_pixel_sweep.py` + `back_tile_sweep.py`, measured times `/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirjax/plans/projector_kernels/fwd_back_findings.md`.
