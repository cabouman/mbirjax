# GPU headroom investigation — measured record (E0, E1, …)

Companion to `gpu_headroom_plan.md` (the plan of record); this file accumulates the
MEASURED results, in the style of `fwd_back_findings.md`.  Numbers here supersede the
pre-E0 estimates in `headroom_appendices/appendix_roofline_traffic_model.md` where they
conflict.  All cells: gautschi H100 (single GPU unless stated), jax 0.10.1, dedicated
`mbirjax_headroom` env (editable install → the worktree), branch `greg/gpu_headroom`.

## E0 — attribution repair (2026-07-12; job 13439991; scripts `e0_hlo_dump.py`)

**Setup:** optimized-HLO dump of the production `_jit_sparse_forward_project` /
`_jit_sparse_back_project` programs at the 1024³ cell (V=1024, R=1008, C=992, ~771k ROR
pixels, n=1; compilation cache disabled so the dump fires).  HLO texts saved at
`~/headroom/results/e0_hlo_{fwd,back}.txt` (gautschi).

**Verdict: the roofline appendix's dominant-traffic story is FALSIFIED — XLA already
fuses both chains completely.**  The per-step structure (while-body fusions, from the
dump):

- **Forward** (per step: 8192 pixels × 128 views × B=252): dynamic-slice of the voxel
  tile (8 MB) → clamp (tap indices) → **`cub_sort_pairs` custom-call** on s32[128, 24576]
  pairs (per-step, UNFUSED — the one kernel boundary) → weights gather (12.6 MB) →
  zeros-broadcast (128 MB) → **`input_scatter_fusion`** (contains the (V, T·P, B) gather ×
  multiply × sorted scatter-add — the 2.95 GiB "updates stream" is fusion-INTERIOR,
  never materialized to HBM) → add into the 128 MB loop carry.  Both scatters carry
  **`indices_are_sorted=true`** (the ScatterWithDistributedIndices fast-path
  precondition holds at production shape).
- **Back** (per step: 2048 pixels × 128 views, 94-step pixel scan × 4 chunks × 8 view
  batches): centers transpose (tiny, s32) → weights select (3 MB) →
  **ONE `input_reduce_fusion`** producing f32[2048, 1008] — the gather + weight +
  tap-sum + VIEW-sum chain is a single fusion; the modeled (V, T·P, R) / (V, P, R)
  intermediates do NOT exist → dynamic-update-slice into the chunk output.

**Rewritten attribution (structure-level; kernel-share numbers pending E0b):**

- Forward's candidate costs: the per-step CUB sort (share = THE pending number, job
  13440013), the L1/atomic behavior INSIDE the scatter fusion (its gather source is the
  L2-resident 8 MB tile — not an HBM problem), and the 128 MB zeros+add accumulator
  pattern (~0.5–0.7 s class over the call).
- Back's cost: the fused gather reads 3×2048×128 rows of 4 KB per step from a
  **512 MB per-step view working set** (128 views × 4 MB channel-major views) — far
  beyond the 50 MB L2, so row gathers miss to HBM with poor efficiency (~3.7× above the
  3.17 GB/step streaming bound; consistent with the June L1-bound ncu signature).
- **Consequences for the plan:** approach A3 (fusion repair) is DEAD — nothing left to
  fuse.  A1 (sort hoist) hinges entirely on E0b's sort share.  NEW cheap lever (added to
  E2): **shrink `back_view_batch` (128 → ~16)** so the per-step gather source (~64 MB)
  approaches L2 residency — same total gather volume, pure TilePolicy knob.  The Pallas
  view-tile × pixel-tile blocking remains the structural fix for back; forward's fix
  remains sort removal + on-chip accumulation.

Note: `nsys` is unavailable in batch shells (`module` not initialized under
`#!/bin/bash`); E0b uses the jax-trace + `trace_utils` machinery instead.

## E1a — pixel-count sweep (2026-07-12; job 13439992; script `e1_pixel_sweep.py`)

Per-call wall (median of 5, warm; seeded random ROR-subset draws — VCD-like locality) at
the 1024³ cell, and the per-iteration projector-seconds prediction
(= granularity × per-call fwd+back):

| geometry | granularity | pixels/call | fwd s | back s | predicted iter proj-s |
|---|---|---|---|---|---|
| parallel | 512 | 1,507 | 0.032 | 0.060 | 46.9 |
| parallel | 128 | 6,026 | 0.072 | 0.198 | 34.5 |
| parallel | 64 | 12,051 | 0.138 | 0.352 | 31.4 |
| parallel | 16 | 48,203 | 0.531 | 0.935 | 23.5 |
| parallel | 4 | 192,810 | 2.005 | 2.708 | 18.9 |
| parallel | 1 (full) | 771,240 | 7.947 | 10.482 | 18.4 |
| cone | 512 | 1,507 | 0.077 | 0.052 | 66.3 |
| cone | 128 | 6,026 | 0.188 | 0.166 | 45.3 |
| cone | 64 | 12,051 | 0.360 | 0.317 | 43.3 |
| cone | 16 | 48,203 | 1.303 | 1.221 | 40.4 |
| cone | 4 | 192,810 | 4.898 | 4.868 | 39.1 |

(cone granularity-1 cell + E1b VCD traces pending at time of writing.)

**Reads:**

- **Fine granularity is NOT host-collapsed at 1024³** (unlike 200³): a granularity-128
  subset call is 70–200 ms — device-scale work, not dispatch-scale.  The
  granularity-128 penalty over full-grid is only ~1.9× for parallel (34.5 vs 18.4
  proj-s/iteration) and ~1.2× for cone — so production-granularity VCD at this size
  remains projector-dominated, and kernel wins SHOULD largely transfer (E1b's device
  share confirms/denies).
- Per-pixel cost at 6k pixels vs full grid: parallel fwd 11.9 vs 10.3 µs (+15%), parallel
  back 32.8 vs 13.6 µs (**+2.4×** — parallel back is the fine-granularity casualty; its
  fixed cost intercept is ~18 ms/call with ~28 µs/pixel slope).  Cone back at subset
  sizes is FASTER than parallel back (166 vs 198 ms at 6k) — the cone pixel-kernel
  short-circuit composes better at small pixel counts.
- Full-grid medians replicate the campaign scoreboard (7.95 s fwd / 10.48 s back vs
  8.19 / 10.92 in `fwd_back_findings.md` — different node/day, −3%).

## Pending

- **E0b (job 13440013):** fwd sort share + back kernel composition from a traced window.
- **E1b (job 13439992, running):** window-difference VCD device share at granularity 128,
  parallel+cone × 512³+1024³ — the (a)-track gate.
- ncu round (pipe-level: L1/LSU vs HBM on the two big fusions; the band kernel at n≥2)
  — needs kernel names from E0b; `ncu` availability on compute nodes to be checked.
