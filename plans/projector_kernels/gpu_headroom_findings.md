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

## E0b — kernel shares, round 1 (2026-07-12; job 13440013; `e0b_kernel_share.py`)

Traced window of 2 warm full-grid calls each (parallel 1024³ n=1):

- **Back: unambiguous.**  Wall 11.24 s/call (+7% profiler overhead vs E1a's 10.48);
  ~95% of wall is device; **98.3% of device time is the single `input_reduce_fusion`**
  (10.5 s/call).  Back = one kernel; its fix is that kernel's access pattern.  Also
  visible: cuGraphLaunch events — XLA command buffers ARE capturing the loops.
- **Forward: composition contaminated by overlapping trace tracks** (event-name
  self-time summed to 162% of wall — graph-node/annotation tracks double-count kernels).
  Trustworthy pieces: `input_scatter_fusion` = 6.46 s/call (≥67% of the 9.59 s/call
  traced wall), and a 7.5 s/call `<UNKNOWN>` bucket of ~44k events (~7 per sort call —
  the CUB sub-kernel signature) that overlaps it.  **The sort share therefore remains
  open** — the unambiguous instrument is the A1 prototype itself (identical kernel fed
  PRE-SORTED inputs; promoted into the next round).  Round 2 of this script reports
  device busy from the STREAM-track totals (single timeline, no double counting) and
  adds a 6,026-pixel variant (the E1b replacement, below).

## E1b — VCD iteration walls (2026-07-12; job 13439992; `e1_vcd_trace.py`)

**The traced-window-difference design FAILED — and the failure is itself a finding:**
device-STREAM totals were identical between the 1- and 2-iteration windows to 3 digits
(e.g. 40.85 vs 40.84 s at parallel 1024³) while wall grew 37.5 s — the signature of the
profiler's event buffer saturating early in a whole-recon window and dropping everything
after.  Traced differencing over long windows is unusable; per-call SHORT traces (E0b
round 2) replace it.  The cone-1024³ cell segfaulted (rc=-11, no Python traceback —
native, likely the profiler on the largest window); its wall-only rerun is queued for a
later round.  (Lesson recorded: whole-recon jax traces at 1024³ overflow the event
buffer SILENTLY — trace windows must be a few calls, not a recon.)

**What survives (untraced walls, trustworthy) — joined with E1a's prediction:**

| cell | iter wall (untraced) | E1a predicted projector-call wall | projector share of iteration |
|---|---|---|---|
| parallel 512³-class | 2.84 s | (not swept at 512³) | — |
| cone 512³-class | 3.80 s | (not swept) | — |
| parallel 1024³ | **36.7 s** | **34.5 s** | **~94%** |

**The (a)-track gate, provisionally: PASSED at 1024³** — a production-granularity
(128-subset) VCD iteration is ~94% projector CALLS by wall.  What fraction of those
70–200 ms calls is device-busy (vs per-call host/dispatch) is the one remaining piece —
E0b round 2's 6,026-pixel cells measure exactly that.

## E0b round 2 — kernel shares, resolved (2026-07-12; job 13440901)

Stream-track busy (one timeline — no double counting) + composition, parallel 1024³ n=1:

| cell | traced wall/call | device busy/call | busy vs UNTRACED wall | dominant kernels |
|---|---|---|---|---|
| fwd @ full grid | 9.63 s | 7.37 s (77%) | ~93% of E1a's 7.95 s | `input_scatter_fusion` 6.45 s (**86–88% of busy**) |
| back @ full grid | 11.34 s | 10.63 s (94%) | ~100% of E1a's 10.48 s | `input_reduce_fusion` 10.5 s (**98%**) |
| fwd @ 6,026 px | 0.122 s | 0.066 s (55%) | **~92% of E1a's 0.072 s** | scatter fusion 82% |
| back @ 6,026 px | 0.277 s | 0.190 s (69%) | **~96% of E1a's 0.198 s** | reduce fusion 63% + **`input_concatenate_fusion` 32%** |

**Verdicts:**

1. **The sort share is ~2–3% — approach A1 (sort hoist) is DEAD.**  The named-op track
   total (≈7.65 s/call: scatter 6.45 + clamp 0.53 + add 0.38 + iota 0.20 + small) matches
   the stream busy (7.37 s/call) within noise, leaving ≲0.2 s/call for the batched CUB
   sorts.  The roofline appendix's 2.1–4.6 s sort extrapolation scaled per-call sort cost
   linearly in pixels; at production shape the 128×24576 batched sorts are tiny next to
   the 252-column scatter work.  (A1's memory quantification in the plan §5 is moot.)
2. **Forward = the scatter fusion's in-fusion access pattern (86–88% of device); back =
   the reduce fusion's 512 MB-working-set gathers (98%).**  With A1 and A3 both dead, the
   remaining XLA-level levers are A2 (index-read vectorization inside the scatter fusion —
   small, cheap A/B), A5 (multi-device band restructure), and the NEW back_view_batch
   L2-residency sweep (E2a).  Everything else is custom-kernel (Pallas) territory, as the
   plan's tiering anticipated.
3. **Subset-sized calls are DEVICE-bound** (~92% fwd / ~96% back against untraced walls)
   — combined with E1b's 94%-projector-calls iteration wall: **production-granularity VCD
   at 1024³ is ~85–90% projector device time.  The (a)-track Amdahl gate is PASSED** —
   kernel wins transfer nearly fully at this size.
4. Small-call overhead attribution: at 6,026 pixels, back spends 32% of device in
   `input_concatenate_fusion` (the pixel-chunk concat) — the fixed cost behind E1a's 2.4×
   per-pixel penalty; a subset-call fast path (skip concat when one chunk) is a candidate
   small VCD win, noted for later.

## Pending

- **E2a (submitted):** back_view_batch 128 → {64, 32, 16, 8} at 1024³ n=1, full-grid +
  subset shapes — the L2-residency lever on the back reduce fusion.
- Cone 1024³ VCD iteration wall (wall-only rerun); cone fwd hfan/vfan split at 1024³.
- A2 flatten A/B (small); the subset-call concat fast path (observation 4).
- ncu round (pipe-level attribution of the two big fusions; the band kernel at n≥2);
  `ncu` availability on compute nodes to be checked.
- Then: the Pallas spike per the plan (band kernel first — the (c) main target).
