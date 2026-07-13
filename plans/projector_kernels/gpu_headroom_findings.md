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

## E2a — back_view_batch L2-residency sweep: HYPOTHESIS REFUTED (2026-07-12; job 13444276)

Parallel 1024³ n=1; median wall (2 full / 5 subset trials, tight repeats); working set =
view_batch × 4 MB channel-major views:

| view_batch | working set | full back s | subset (6,026 px) back s | peak GB |
|---|---|---|---|---|
| 128 (default) | 512 MB | **10.56** | 0.200 | 15.96 |
| 64 | 256 MB | 13.45 | 0.256 | 15.69 |
| 32 | 128 MB | 11.65 | **0.179** | 15.62 |
| 16 | 64 MB | 13.25 | 0.181 | 15.58 |
| 8 | 32 MB | 11.48 | 0.207 | 15.58 |

**Verdict: shrinking the per-step view working set toward L2 residency does NOT recover
back's gap — the default 128 is the FASTEST full-grid setting, and even a 32 MB working
set (comfortably inside the 50 MB L2) is 9% SLOWER.**  If the reduce fusion's cost were
L2-capacity misses on the 512 MB set, vb=8–16 would have won large.  It didn't, so the
E0 working-set framing was too coarse: within one view the gather's 992 distinct channel
rows are only a 4 MB source with ~6 reads per row — reuse is per-view and L2 captures it
at ANY view batch.  The cost is the TRANSACTION pattern of the uncoalesced 4 KB row
gathers + the tap/view-sum structure — exactly the June ncu signature (97% L1, 8% HBM),
now confirmed by ablation rather than extrapolation.  Non-monotonicity across settings
(64/16 worse than 32/8) looks like per-shape launch/autotune variation, not signal.

**Consequences:** the last cheap XLA-level lever for back is closed — back projection's
remaining ~10× is CUSTOM-KERNEL-ONLY (coalescing/register-tiling inside the fusion
replacement, per the plan's Pallas design).  Two small notes: subset-shaped calls got a
real but modest ~10% at vb=32 (a possible VCD-only tuning crumb, below the
bother-threshold alone); peak memory eases slightly at small vb (−0.4 GB).

## Cone forward vfan/hfan split (2026-07-12; job 13473088; `cone_fwd_split_ab.py`)

Kernel level at the 1024³-class cell (P=4096, V=128, H100): full 15.15 ms; vfan-only
6.27 ms (**41%**); hfan-only 10.84 ms (**72%**; sum 1.13 — overlap, the cone-back
lesson).  Reads: (1) the vfan-elimination prize on cone forward is bounded ~1.4×; the
hfan (sorted channel reduce over the FULL detector rows) is the bigger share.  (2) For
slice-parity efficiency (slice_parity_plan.md R1 cost accounting): the hfan's cost is
SLICE-COUNT-INDEPENDENT (its reduce columns are detector rows), so slice-set-aware
kernels recover only the vfan share of the P=2 forward penalty — refined idealized
charge ≈ **1.3–1.4×** per P=2 iteration (protocol used 1.5×, slightly generous to
parity; direction unchanged).  Flow this to the parity session's R1 reading.

## E3 step zero — Pallas smoke, round 1 (2026-07-12; job 13473088; `e3_pallas_smoke.py`)

Vfan-shaped computed-index gather kernel (P=8192, R=1008, slice-SET of 126 @ stride 8,
T=3); XLA reference ~93 µs.  **The backend-constraint ladder, one layer per round**
(every variant passes interpret gates; failures are GPU lowerings — jobs 13473088,
13473710, 13474601):

| round | Triton verdict | Mosaic GPU verdict |
|---|---|---|
| 1 naive (block-load + jnp gather) | power-of-2 shapes required (1008) | async-copy cap 256 elements/dim |
| 2 padded / R-chunked | `slice` primitive unimplemented (w[...,t]) | 128-byte warpgroup alignment on copies |
| 3 per-tap weights, aligned | **`gather` primitive unimplemented** | **`gather` unimplemented (Lane/Warpgroup semantics)** |

**Round-3 verdict is the structural finding: NEITHER backend lowers an in-kernel HLO
gather at jax 0.10.1** — array-level `take_along_axis`/integer indexing of a LOADED
block is a dead end.  The working idiom (from in-tree `paged_attention`, which gathers
at this pin) is **REF-level integer-array indexing** (`ref[idx_array]` → pointer loads,
never materializing the block).  Round 4 (`e3_pallas_smoke4.py`, job 13475379) tests
three ref-gather variants on Triton (TILE=1 1-D, TILE=8 2-D advanced, flat-index whole-
array window); MGPU parked (its gather gap is at lane semantics — deeper).  Implication
for the E3 kernel designs regardless of outcome: all gathers must be expressed at the
ref level, which also means SMEM staging of gathered values (the cross-view-reuse
lever) is NOT free-form at this pin — the Triton backend has no SMEM scratch, so reuse
must come from L1/L2 behavior of pointer loads, and the MGPU route (explicit SMEM)
needs its gather gap resolved upstream or a different formulation.  If round 4 fails
too: probe a newer jax in a spike-only side env, or jax-triton raw (library pin
untouched either way).

## E3 step zero — CLOSED, GREEN LIGHT (round 4, job 13475379; `e3_pallas_smoke4.py`)

**All three ref-level gather variants compile, match the XLA reference BITWISE (rel err
0), and run at 1.03–1.04× the XLA time untuned** (tile1 97.8 µs / tile8 98.2 / flat
98.5 vs XLA 94.5 µs at the vfan-shaped production case).  The working recipe at the
pin: **Triton backend + ref-level integer-array indexing** (1-D, 2-D advanced, and
flat-index whole-array windows all work — full layout freedom for CSR segment walks).
Read: a naive Pallas kernel TIES XLA before any tuning (num_warps/num_stages, tap-load
dedup, view-tiling, load balance all untouched) — the tooling is not the obstacle; the
E3 bar (≥1.5–2×) is now a kernel-design question, as intended.  Design constraint to
carry: no SMEM scratch on Triton, so reuse comes from registers + L1/L2 behavior of
pointer loads (the ASTRA register-tile pattern — registers, not shared memory).

## E3 hfan-forward kernel v1 (2026-07-12; job 13476543; `e3_hfan_pallas_v1.py`)

First real kernel, first attempt — the CSR segment-walk (one program per (view,
channel), dynamic-trip fori_loop, ref-level row gathers, register accumulation, one
store; streams precomputed eagerly from concrete centers).  1024³-class parallel cell
(P=8192, C=992, band 252→256, V=128), H100, values ≤6.3e-7 rel:

| case | taps/channel skew | XLA sorted reduce | Pallas v1 | speedup |
|---|---|---|---|---|
| subset (uniform — the VCD shape) | 2.2× | 0.666 ms | 0.340 ms | **1.96×** |
| raster (full-grid batch) | 4.6× | 0.689 ms | 0.484 ms | **1.42×** |

**The success bar (≥1.5–2× at both cases) is MET on the subset case immediately and
missed by 0.08× on the raster case — by exactly the predicted mechanism** (channel-skew
stragglers in the one-program-per-channel grid: identical work volume runs 0.34 vs
0.49 ms).  num_warps ∈ {1,2,4} is flat (~±3%); nw=1 suffices — the per-program work is
one warp's worth, as designed.

Open accounting items before composition claims: (1) the stream precompute (sort +
searchsorted, eager) is uncounted — fair for VCD (fixed partitions amortize it across
iterations) but must be charged for one-shot full projections; (2) the isolated-bench
XLA baseline (0.69 ms) differs from the in-scan production step (~2.1 ms of scatter
fusion) — the campaign's compose-don't-extrapolate lesson applies; model-level A/B is
the E4 gate.  **v2 (queued): even-partitioned tap stream with boundary fixup (the
ModernGPU segreduce pattern) for the raster skew; then the cone variant (per-pixel
weight scale + full-rows band) and the precompute-cost measurement.**

## E3 hfan kernel v2 (2026-07-12; job 13478705; `e3_hfan_pallas_v2.py`)

Cap-and-split skew fix, two store strategies (all-atomic into zero-aliased output vs
hybrid store/atomic), caps {16, 32, 64}:

- **Raster: bar crossed** — 1.59× (hybrid, cap 64; was 1.42× in v1); the split bounds
  the stragglers as designed, and larger caps trend better (fewer atomic segments).
- **Subset: regressed to 1.78×** (was 1.96×) — attributable mostly to the 130 MB
  zeros-init write the atomic path requires (~35 µs/call ≈ the gap at cap 64, where
  only ~1 segment/view splits).
- Precompute on the H100 node: ~145 ms warm for the (V=128, P=8192) streams (63 ms on
  the M3 — the device sort path is oddly slower; optimization candidate, amortized in
  VCD regardless).  Values pass throughout (≤4e-7).
- **v3 (queued, job 13479640): two-phase launch** — phase 1 direct-stores every
  channel's first segment (every row written exactly once — no zeros pass, no race;
  empty channels materialize as store-zero segments), phase 2 atomic-adds only leftover
  split segments (~450/view raster, ~1/view subset) via input_output_aliases on the
  phase-1 result.  Caps {32, 64, 96, 128}.  Interpret gates pass.  Also for the record:
  v2's interpret gate caught a real pre-GPU correctness bug (whole-array out block
  indexed with a literal view 0 instead of program_id — all views would have collided).

## E3 hfan kernel v3 — THE SPIKE BAR IS MET (2026-07-12; job 13479640)

Two-phase store+atomic (no zeros pass): **subset 2.13×** (caps ≥64; n2 = 1 atomic
segment/view — phase 2 nearly empty as designed), best of all rounds; raster regressed
to 1.41× (phase 1 spans all 992 channels including ~410 empty/view + a second launch —
costs the skewed case roughly what the zeros pass cost v2).  Values ≤6e-7 everywhere.

**Composite verdict — the plan's E3 success bar (≥1.5–2× at both cases) is MET by a
variant policy**: v3-two-phase for uniform/subset batches (**2.13×**), v2b-hybrid for
skewed raster batches (**1.59×**); the precompute computes the segment statistics that
discriminate them, so the policy is free (and matches the TilePolicy kernel-selection
pattern).  Iteration ledger: v1 structure → v2 skew fix → v3 init-tax fix; the interpret
gate caught one real correctness bug (v2 view-index collision) before any GPU time.

Remaining tuning upside (not needed for the bar, noted for E4+): raster's contiguous
pixel ranges per channel should make it FASTER than subset, not slower — a zeros-free
single-launch formulation (e.g. split-channel row pre-zeroing at ~14 MB instead of
130 MB) and multi-channel programs are the known next levers; the H100 precompute at
~145 ms (vs 63 ms on the M3 CPU) is its own small oddity to chase.

**Next per plan: the cone hfan variant (per-pixel weight scale, full-rows band — the
72%-share case), then E4 composition (TilePolicy flag, wrapper integration with the
precompute placed per the Phase-D idiom, model-level A/B), then the band back kernel.**

## E3 cone hfan v1+v2 (2026-07-12; jobs 13480264, 13481121)

v1 (parallel design transplanted): values pass (≤5e-7) but subset 1.14× / raster 0.84×
— cone forfeits parallel's shared-L2-tile advantage (per-view values), moves 4 KB rows
with parallel-tuned num_warps=1, and has half-length segments.  **v2 (row-chunked grid
+ warp sweep, two-phase only): subset = 2.13× (rc=256, w=1) — identical to parallel's
best; the row-chunk restored the vector/register balance.**  Raster improved to 1.15×
(rc=256, w=4) but remains below bar: cone raster combines skew WITH short segments
(mean 12.4 taps) — largely MOOT under the E4 policy (one-shot coarse full-grid
iterations keep the XLA path; the pallas path serves the repeated fine tail, where the
kernel is at bar).  Note the warp story flipped by case: raster wants w=4, subset w=1 —
another policy-selected knob.

**E3 spike verdict, both geometries: the production (VCD fine-tail) case is at 2.13×
on parallel AND cone; skewed one-shot cases stay on XLA by policy.  Next: the
back-projection kernel (design at plan §E3b), then E4 integration of the pair.**

## E3b back-projection kernel v1 (2026-07-12; job 13483553; `e3_back_pallas_v1.py`)

**The largest win of the campaign, first attempt**: register-tile across views +
row-chunk L2 phases, at the 1024³ parallel cell vs the library stacked-gather baseline
(vmap + view-sum, the production composition):

| case | best | speedup | Hessian (cp=2) |
|---|---|---|---|
| raster | rc256 w1 | **16.3×** | 14.9× |
| subset | rc256 w2 | **26.0×** | 22.7× |

Ruler checks (a result this size demands them): the baseline extrapolates correctly
from E0b's measured 3.59 ms/step (~14 ms at this P); the kernel's HBM traffic under the
L2-phase design (~0.6 GB/call) predicts ~0.9 ms — the observed ratio IS the roofline
(b)-bound headroom for back (23–36×), realized by the design's two levers.  Values
~1e-6 (view-sum reorder); **adjoint ⟨Ax,y⟩=⟨x,By⟩ prints exactly equal** (agreement
below f32 quantization at that magnitude).  rc=256 dominates (fewer weight re-reads);
warps flat-ish.  No skew sensitivity — back is uniform work, as designed.

**Composition consequences for E4 (the next design deliverable):** at 15–26× kernel
speed, the back DRIVER becomes the limiter — but the pallas kernel has no scan carry,
so the driver simplifies: the whole pixel set can go in ONE grid per view-chunk (no
94-step pixel scan, possibly no transfer chunking), which is where much of the
kernel-level win can survive composition.  The E4 model-level A/B remains the gate
(compose, don't extrapolate).  E3 spike status: parallel fwd 2.13×/1.59×, cone fwd
2.13× (fine-tail), parallel back 16–26× + Hessian — the pair is ready for the E4
integration design.

## E4 preview — composed back projection (2026-07-12; job 13484992; `e4_back_composed.py`)

Full production shape (771k ROR pixels × 1024 views, 8 view-chunks, every cost charged)
vs the warm library `sparse_back_project`:

**Composed 3.54× (10.56 s → 2.98 s), rel 5.5e-7; Hessian 3.53×.**  Breakdown: kernel
1.11 s, **weights precompute 1.83 s (61% — the new limiter)**, centers 6 ms, layout
13 ms, accumulation 22 ms.  Verdicts: (1) the simplified no-scan driver COMPOSES — the
94-step scan and transfer chunks deleted at ~zero cost; (2) the 16–26× kernel win is
real in composition but currently capped by building the (V, T, P) weights array; (3)
the fix is designed, not speculative: **compute weights IN-KERNEL** from (n_p, W, scale)
— the plan shrinks from 3 arrays of taps to 2 per-pixel floats (2/3 the bytes), the
1.8 s becomes ~6 ALU ops per (view, tap) reused across the row chunk, and the projected
composed time ≈ 1.2–1.3 s ≈ **8–9×**.  For VCD the point is already moot in the good
direction: plans amortize across iterations, so fine-tail subset calls see
near-kernel-level gains regardless — 3.54× is the ONE-SHOT floor, not the ceiling.

## E4 preview, corrected — the retrace artifact (2026-07-12; job 13486103)

The 1.83 s weights cost in the first composed run was a BENCH ARTIFACT: the builder
constructed a fresh `jax.jit` per chunk → full host retrace each call (the module-level
centers jit doing near-identical work cost 6 ms — the tell).  With the builder hoisted
(traced once, the production structure): **weights 1,828 → 7 ms (261×); COMPOSED BACK
9.07× (10.48 s → 1.16 s); Hessian 9.11×; values unchanged (5.5e-7).**  Kernel = 96% of
composed time; all plan+driver overheads ≈ 44 ms.  Consequences: (1) the one-shot back
number is 9×, matching the projection; (2) the plan-builder cost floor is ~1 ms/chunk —
the fwd break-even drops to a few reuses pending the same hoisted-jit measurement of the
fwd stream builder (sort included, expect ~2–5 ms/chunk); (3) the "H100 precompute
oddity" is CLOSED (host tracing, not device — also explains the M3 being faster); (4)
the in-kernel-weights design change is now OPTIONAL (a memory nicety: 0.10-shard plan vs
0.19), not performance-critical.  Lesson (the eager-gather episode's sibling): benches
that construct jits per call measure TRACING, not the operation — hoist and warm before
attributing.

## E4 increment 1 — SHIPPED TO THE BRANCH AND GATED (2026-07-12; commit 2174fb7 + afc8a76; job 13486197)

The pallas parallel-beam back path is in the library (module `_pallas_kernels.py`,
TilePolicy `back_pallas`, the n=1 dispatch, `compute_hfan_data`, tests, dev docs).
Model-level A/B, flag-on vs kill-switch, fresh subprocesses:

| cell | XLA → pallas | speedup | peak memory | values |
|---|---|---|---|---|
| back full-grid (1024³) | 10.55 → 1.15 s | **9.17×** | 18.89 → 17.61 GB (−1.3) | 5.5e-7 PASS |
| back subset (6,026 px) | 198 → 24 ms | **8.25×** | 5.35 → 5.84 GB (+0.49) | 4.4e-7 PASS |
| Hessian full-grid | 10.48 → 1.15 s | **9.10×** | −1.3 GB | 5.1e-7 PASS |
| VCD guard (256³, 5 it) | 16.95 → 15.87 s | 1.07× | −0.03 GB | **7.6e-7** PASS (1e-4 gate) |

Reads: full-grid memory DROPS 1.3 GB (the old path's transfer-chunk concat transients
are gone); the subset +0.49 GB is the channel-major chunk copy (~0.52 GB — matches the
pre-declared ack; a transpose-free gather variant could reclaim it later); the VCD
guard is POSITIVE at an interactive size (the CUDA-graph/dispatch caveat did not bite)
with recon values at 7.6e-7 after 5 iterations — far inside the iterated gate.  The
CPU suite is green (one PRE-EXISTING unrelated failure, flagged separately); interpret
gates + the adjoint identity run on CPU CI.

Operational lesson from the first A/B attempt (job 13486147, FAILED silently at 3m15s):
**the cluster HOME quota is 25 GB and a full home kills writes silently** — no
traceback, no log tail, even the shell's echo lost.  Big regenerable artifacts go to
SCRATCH (the runner now does; the parity session independently adopted the same policy
and symlinked its staging).  Tell: sacct shows FAILED with a short elapsed while the
log just stops.

**Next: increment 2 — the forward kernels (two-phase + hybrid variants, fine-tail
policy, ProjectorPlan for the streams), same rails; then the model-level pair A/B and
the nightly soak.**

## E4 increment 2 — forward subset path: first gate FAILED, fixed, RE-GATED CLEAN (2026-07-12; commits 6268ba4 + 541b0b3; jobs 13486271 → 13487918)

The pallas forward horizontal fan is in the library for SUBSET-SIZED calls only
(TilePolicy `fwd_pallas` + the pixel-count guard in `sparse_forward_project_public`;
full-grid forward keeps the XLA sorted reduce by design — the python view-chunk loop
at ~3000 pixel batches would erode the win).  No ProjectorPlan object was needed:
post-retrace-fix the streams build in-call at ~ms cost.

**First gate (6268ba4, job 13486271): the fwd cells FAILED on speed** — values passed
everywhere, but fwd_subset measured **0.68×** (73 → 108 ms) and the VCD guard fell to
a 0.99× wash.  Diagnosis (code inspection, confirmed by the fix): the driver synced
`starts` to host **per view chunk** (`np.asarray` → 8 pipeline stalls per call) for a
host-numpy cap-and-split, and phase 2 was sized by the **data-dependent** segment
count, so every distinct VCD subset changed a pallas cache key → Triton recompiles
inside the timed loop.  The spike's 2.13× was kernel-only timing; the driver around it
is part of what gates.

**Fix (541b0b3), reviewed pre-gate by a 3-lens adversarial pass** (split-bound math
brute-forced over 4,481 adversarial + 200k random cases; jit/caching-trap audit;
GPU-perf audit): the split runs ON DEVICE under the static bound n2 = (T·P)//cap
(provably sufficient; shapes only, never data), each view chunk is ONE cached fused
jit (streams → split → both phases → trim+reorient), pad-slot atomics are
`pl.when`-guarded (the reviewers estimated 5–15 ms of same-address adds-of-zero at the
gate shape), and view bookkeeping stays in numpy (zero-eager-ops contract; ~1 eager
dispatch per call remains, the multi-chunk concat).  A new test forces an over-cap
regime so the guard is exercised on segments that must NOT be skipped.

Re-gate, same six cells, flag-on vs kill-switch, fresh subprocesses:

| cell | XLA → pallas | speedup | peak memory | values |
|---|---|---|---|---|
| back full-grid (1024³) | 10.48 → 1.15 s | **9.10×** | −1.3 GB | 5.5e-7 PASS |
| back subset (6,026 px) | 198 → 24 ms | **8.25×** | +0.49 GB (acked) | 4.4e-7 PASS |
| Hessian full-grid | 10.48 → 1.15 s | **9.09×** | −1.3 GB | 5.1e-7 PASS |
| **fwd subset (6,026 px)** | 72 → 28 ms | **2.57×** | equal | 2.5e-7 PASS |
| fwd full-grid (policy check) | 7.97 → 7.97 s | 1.00× | equal | **rel 0 (bitwise)** PASS |
| VCD guard (both paths, 5 it) | 17.33 → 16.34 s | **1.06×** | −0.03 GB | 7.6e-7 PASS |

Reads: the gated fwd subset (2.57×) now EXCEEDS the kernel-only spike number (2.13×)
— the fused chunk jit is cheaper glue than the spike harness's; fwd_full at rel 0
proves the guard keeps full-grid calls bitwise on XLA; the VCD guard's 1.06× shows the
forward acceleration adds little END-TO-END at 256³ (VCD there is host-dispatch-bound
— the §3 pool-1 story, not a kernel problem).  Job wall fell 18:24 → 5:38, which is
the recompile diagnosis confirmed operationally.  Suite: 297 passed (14 pallas gates,
interpret mode on CPU CI).

## Compile-stall attribution — NOT REPRODUCIBLE, transient (2026-07-13; jobs 13497683/13497717/13497719; `compile_attribution.py`)

Trigger: Greg's cold demo-1 run (kernel_investigation, 1024³, his interactive node,
06:33) tripped XLA's slow-compile alarm — `_jit_compute_scatter_centers` took
**2m19s** to compile; the second run was fast (the library's persistent cache).
Question: is a multi-minute cold compile what a new user sees at production sizes?

Six isolated cold cells (fresh persistent-cache dir each; per-module compile logs on
a shared clock; H100 batch node), each varying ONE thing vs Greg's run:

| cell | variable tested | cold fwd_project | in-process warm | ⇒ compile cost |
|---|---|---|---|---|
| cone_default | (reproduction baseline) | 25.6 s | 24.1 s | **~1.5 s** |
| cone_autotune0 | autotuning off | 25.5 s | 24.1 s | ~1.4 s |
| par_default | geometry | 10.3 s | 9.4 s | ~0.9 s |
| cone_warm | cache-hit control | 25.5 s | 24.1 s | ~1.4 s |
| ki_default | **Greg's exact code+env** (kernel_investigation, `mbirjax` env) | 25.6 s | 24.1 s | ~1.5 s |
| cone_2cpu | compile-CPU starvation (taskset 2 CPUs) | 26.4 s | 24.2 s | ~2.3 s |
| ki_homecache | cache dir on NFS HOME (the library default) | 25.5 s | 24.1 s | ~1.5 s |

No cell tripped any slow-compile alarm; the entire demo-1 projector chain compiles
cold in **1.5–2.5 s** at 1024³ under every variation (jax 0.10.1 in both envs; Greg's
interactive allocation has the same 14 CPUs as the batch nodes).  **Verdict: the
2m19s was a transient stall specific to that node/session at that moment** (most
plausible: momentary NFS or CPU contention on the shared interactive node during the
in-compile autotune-cache I/O — this jax pin reads/writes the per-fusion autotune
cache inside the timed compile), not a property of the library, the branch, or a
typical first-run cost.  New-user expectation at 1024³: ~1.5–2.5 s of one-time
compile, then the persistent cache removes even that across processes and sessions.
If the alarm recurs, capture `myquota`, `uptime`, and `nfsstat` from the node before
suspecting the code.

## Pending
- Cone 1024³ VCD iteration wall (wall-only rerun); cone fwd hfan/vfan split at 1024³.
- A2 flatten A/B (small); the subset-call concat fast path (observation 4).
- ncu round (pipe-level attribution of the two big fusions; the band kernel at n≥2);
  `ncu` availability on compute nodes to be checked.
- Then: the Pallas spike per the plan (band kernel first — the (c) main target).
