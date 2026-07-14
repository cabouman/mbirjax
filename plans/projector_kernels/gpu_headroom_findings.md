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

## Wave-2 baseline — the n=1/2/4 scaling curve, post-campaign (2026-07-13; job 13497778; `w2_scaling_baseline.py`)

The (c)-target gap, re-measured on the current branch (1024³ cell, H100 node, median
of 3 warm trials; VCD = 4 iterations including compile):

| cell | n=1 | n=2 | n=4 | reading |
|---|---|---|---|---|
| cone back_full | 18.95 s | **27.51 s** | 14.35 s | **n=2 ANTI-SCALES (1.45× slower)**; 4 GPUs buy only 1.32× |
| cone VCD (4 it) | 262 s | **276 s** | 181 s | same shape end-to-end: n=2 loses, n=4 = 1.45× |
| cone peak/GPU | 53.8 GB | 27.2 GB | 23.0 GB | capacity story intact — sharding is doing its memory job |
| parallel back (shipped) | **1.57 s** (pallas) | 5.39 s | 3.29 s | **one pallas GPU beats FOUR XLA-band GPUs (2.1×)** |
| parallel back (XLA n=1 ref) | 10.90 s | — | — | band path scales 10.9 → 5.4 → 3.3 (≈2×/≈3.3×, normal) |

Readings: (1) the June cone-back n=2 anti-scaling fully reproduces post-campaign —
the band-path restructure is still THE (c) blocker; (2) the parallel row reframes the
prize: the register-tile kernel on ONE device already beats the 4-GPU band path, so a
band-capable pallas kernel converts added GPUs into real wall-clock instead of
overhead; (3) parallel's XLA band path scales normally (2.02×/3.31×) while cone's
collapses — the cone-specific vertical-fan/transpose structure is implicated, matching
the June ncu attribution (to be re-confirmed by an n≥2 trace before A5).

Bench side-catch: the device-line pallas token (added the same day) read "(pallas:
back)" at n≥2 while the walls prove the pallas kernel never runs there (the dispatch
lives in the n=1 short-circuit).  Fixed: `back_pallas` is now n=1-gated in the policy
like `fwd_pallas`, so the token and `get_compute_config` report reachability, not
wishfulness.

## Wave-2 band trace — June's transpose attribution CONFIRMED on current code (2026-07-13; job 13497819; `w2_band_trace.py`)

Per-kernel device self-time of `sparse_back_project` (3 calls, 1024³, XLA path):

| cell | dominant fusions (self-time over 3 calls) | reading |
|---|---|---|
| cone n=2 | `input_transpose_fusion` 41.7 s + `_2` 48.0 s (**~90 s**), loop_add 52.9 s, MemcpyD2D 42.9 s, hfan gather (`input_reduce_fusion`) only **5.9 s** | transposes = the largest compute bucket; the actual projection gather is ~6% of it |
| cone n=4 | transposes 59.4 s, MemcpyD2D **45.4 s** (grows with n), loop_add 41.6 s | same shape; D2D copies become co-dominant at n=4 |
| parallel n=2 | `input_reduce_fusion` 31.4 s dominates; **zero transpose fusions** | the normally-scaling band path has no transpose — cone-specific, as the scaling table implied |

(Stream overlap makes absolute shares fuzzy — multiple GPUs and copy streams double-count
against wall — but the ranking is unambiguous and matches the June ncu: transpose fusions
L1-bound at 6% HBM.)  Execution counts localize the structure: 8,020 transpose-fusion
runs per call at n=2 ≈ views(512) × bands(8) × devices(2) — **the transposes run once per
(view, band)**, i.e. the band-INDEPENDENT per-view work (the `sinogram_view.T` at
cone_beam.py:650 and a second layout fusion) is recomputed for every band, and the band
count grows with device count.  `cone_n1` traced cell crashed with
CUDA_ERROR_LAUNCH_FAILED (the known cone-1024³-under-profiler failure, as in E1b) — its
wall is known from the scaling bench instead.

A5 candidate forms, to be picked by an HLO dump of the band jit (which ops live in each
transpose fusion):
1. **Pre-transpose the view shard once per call** (+1 transient sino-shard per owner,
   no loop change) — kills the input `.T` if both fusions are view-layout ops.
2. **Hoist the whole horizontal fan out of the band loop** (vmap bands inside one call
   or invert the loop) — kills all per-(view,band) redundancy but needs the full
   (pixels × slices) partial per owner (+3.4 GB at 1024³ n=2).

**HLO probe verdict (job 13497836): NEITHER — the transposes are the vfan's
materialized per-view partials.**  The compiled band module shows the two dominant
fusions each produce `f32[2048, 115, 512]` = (pixel_batch, band_slices, VIEWS) —
~480 MB apiece — combined by a third and collapsed by `input_reduce_fusion` to
(2048, 115).  The whole hfan+vfan chain is fused into these: XLA materializes the
full (P_batch × L × V) stack, transposed view-last, before summing views.  This is
the same re-materialize-the-reduction weakness the E3b register-tile analysis named
for n=1 back — pre-transposing inputs would not touch it (option 1 DEAD), and
hoisting the hfan would leave the materialization in place (option 2 secondary).

**A5, reformulated: scan-over-view-chunks with carry accumulation.**  Restructure the
banded reduction from vmap-all-views→reduce to `lax.scan` over view CHUNKS (vmap
within a chunk, chunk partial added into a (P_batch, L) carry): the cross-view
intermediate shrinks from (P, L, 512) to (P, L, chunk) — bounded, tunable — and the
view-last transposes disappear (nothing to reorder; the carry add is in place).
Chunk size trades launch count against intermediate size (sweep {16, 32, 64}).  The
pallas band register-tile kernel (E3b) remains the deeper fix that eliminates the
class entirely; per the plan's gates, if this XLA-level form restores n=2 scaling,
the pallas band kernel's bar rises accordingly.  Next: single-device A/B bench at
the per-owner shard shape (V=512, L=115, P=823k) — the band kernel cost is
per-device, so no multi-GPU allocation is needed to gate the kernel itself — then
the n=1/2/4 model-level curve if it clears ≥1.5×.

## Wave-2 band A/B — A5 knob verdict + the parallel adoption probe (2026-07-13; job 13505309; `w2_band_ab.py`)

Single-device replications of one n=2 view-owner's full band sweep (512 views,
1024×1024 det, L=115; cone = 11 bands over the extended 1152 slices, ~1.1× production's
absolute walls with ratios exact; harness review wf_2a3f9c18 fixed two operating-point
blockers pre-run).  Values PASS everywhere (≤2.4e-6).

**Cone (the A5 view-batch knob):**

| back_view_batch | 512 (production) | 256 | 128 | 64 | 32 | 16 |
|---|---|---|---|---|---|---|
| band-sweep wall | 27.83 s | 30.88 | 29.82 | 29.67 | 27.58 | **21.15 s** |
| peak GB | 13.9 | 13.4 | 13.0 | 12.9 | 12.9 | 13.0 |

**A5 verdict: the XLA lever tops out at 1.32×** (vb16), non-monotonically (mid values
WORSE than default).  Shrinking the (P_batch, L, view_batch) stack 32× bought only
24% — the materialization is real but the XLA gather kernels' access patterns keep
most of the cost.  Per the plan's decision gate ("bar not met → stop at the XLA level
and document"): **the XLA-level (c) track is CLOSED**; 1.32× does not restore cone
n=2 scaling (27.5 → ~21 s vs n=1's 18.9 s).

**Parallel (the adoption probe): the shipped register-tile kernel is 8.9× on the
per-owner band work** — 5.49 s (XLA at the true n≥2 entry point and vb=512) →
**0.618 s** (pallas, same calls routed through `back_project_single_device`), rel
2.3e-6 PASS.  Projected: parallel n=2 back ≈ 0.6–0.8 s + reduce-scatter, vs today's
5.39 s and n=1-pallas's 1.57 s — the existing kernel turns parallel multi-GPU into a
real scaling curve with DRIVER work only (per-band dispatch + owned_view_indices
plumbing in the per-owner override).  Ack to size in integration: the pallas cell
peaked +3.4 GB (15.3 vs 11.9 GB — the channel-major copies; bounded per view chunk).

Consequences for the (c) sequencing: (1) increment 3 = parallel multi-device band
adoption (driver-level, kernels already shipped and gated); (2) the cone fused-vfan
pallas band kernel is now the load-bearing (c) work — bar: beat the best XLA form
(21.2 s per-owner sweep) by ≥1.5×, with the parallel probe suggesting ~5–9× is
available to a kernel that fuses the vertical fan; (3) the vb16 cone knob is a
possible interim ~1.3× (platform-gated) but is superseded if the cone kernel lands.

## Increment 3 — parallel multi-device band adoption: SHIPPED AND GATED (2026-07-13; commits through the chunk cap; jobs 13508069 → 13508447)

The shipped register-tile back kernel now serves the n≥2 per-owner slice-band calls
(`back_pallas_band`; driver per-owner mode = global view indices, caller placement
trusted, no resharding; the banded reduce-scatter orchestration unchanged).  Final
gate, flag-on vs kill-switch, values cross-checked on-vs-off, seeded partitions:

| cell | XLA → pallas | speedup | values | peak memory |
|---|---|---|---|---|
| back full-grid n=2 | 5.37 → **0.716 s** | **7.5×** | 1.3e-6 PASS | +0.3 GB |
| back full-grid n=4 | 3.29 → **0.443 s** | **7.4×** | 8.7e-7 PASS | +3.1 GB (ack) |
| VCD n=2 (4 it, 1024³) | 97.9 → **50.3 s** | **1.95×** | 6.7e-6 PASS | equal |

**The (c) deliverable for parallel beam — the scaling curve is real now**: back
full-grid 1.57 s (n=1) → 0.716 (n=2, 2.2×) → 0.443 (n=4, 3.5×), where pre-adoption
n=4 LOST to n=1.  End-to-end VCD at n=2 nearly halves.  Two gate lessons en route:
(1) the first VCD value FAIL (rel 0.116) was the unseeded-partition trap — off/on
cells drew different random subsets; `np.random.seed(0)` before recon restored
6.7e-6 (the load-bearing reproducibility gotcha, again); (2) the first run's +5.4 GB
peak at n=2 was the per-owner weights transient at the sharded view batch of 512 —
weights/shard = T·(P/(rows·C))·(chunk/(V/n)) ≈ T·π/4 ≈ 2.4 when the whole shard is
one chunk, size- and n-independent — fixed by capping the driver's chunk at 128
(`BACK_VIEW_CHUNK_CAP`; chunking measured ~free), which cut it to +0.3 GB for 5.6%
of wall.  The n=4 +3.1 GB residual is accepted (18.4 GB absolute) and unattributed —
revisit if it grows at larger shapes.  **CORRECTION (2026-07-13, increment-4
review finding): these peak numbers were RSS-contaminated — the harness maxed
`peak_bytes_in_use` over `get_memory_stats`' trailing 'CPU' entry (process RSS).
RE-MEASURED with the per-GPU reader (job 13515412, on the branch WITH increment 4,
so the on-cells exercise both band adoptions):**

| cell | off → on (true per-GPU peak) | delta |
|---|---|---|
| back n=2 | 13.96 → 13.72 GB | **−0.2 GB** (the chunk cap over-delivers) |
| back n=4 | 10.87 → 11.90 GB | +1.0 GB (was misreported +3.1) |
| VCD n=2 (inc3+inc4 both on) | 25.01 → 26.71 GB | **+1.7 GB for the 2.83× speedup** (99.5 → 35.2 s) |

The composed-workload memory cost is modest: the large fwd acks (+6.8/+7.4 GB)
appear only at dedicated full-grid forward cells; in VCD the forward calls are
subset-sized and the net footprint is +1.7 GB.  One environmental rerun (both n=4 cells
SIGABRT in XLA GPU-client init immediately after the `ai`-partition outage; clean on
the healthy partition).

Remaining wave-2 work: the cone fused-vfan pallas band kernel (the load-bearing (c)
item — cone still anti-scales; bar ≥1.5× over the 21.2 s best-XLA per-owner sweep,
with 5–9× indicated); candidate increment 4 = per-owner banded FORWARD adoption
(the guard-sweep session's band-256 column measured the pallas forward driver
1.7–3.8× at exactly the banded-forward width).

## Increment 4 — parallel multi-device banded FORWARD adoption: SHIPPED AND GATED (2026-07-13; commit 487236a; job 13515398)

The shipped pallas forward kernel now serves the n≥2 per-owner banded-forward calls
(`fwd_pallas_band`).  The per-owner forward call — previously INLINED in the base
`_forward_project_band_to_local_views` worker — was extracted into a seam
`_forward_project_band_to_view_shard` (base = XLA `sparse_forward_project`,
byte-identical to the old inline call; `ParallelBeamModel` overrides it to route through
the pallas `forward_project_subset` when the flag is set).  The driver already supported
the per-owner mode (global view indices, no resharding), so `_pallas_kernels.py` was
untouched and the band-broadcast orchestration is unchanged.  Final gate, flag-on vs
kill-switch, values cross-checked on-vs-off, seeded partitions, GPU-filtered per-device
peaks (1024³, 4-GPU node):

| cell | XLA → pallas | speedup | values | peak memory (per-GPU) |
|---|---|---|---|---|
| fwd full-grid n=2 | 4.22 → **1.31 s** | **3.23×** | 5.8e-6 PASS | 11.5 → 18.3 GB (+6.8, ack) |
| fwd full-grid n=4 | 2.11 → **0.672 s** | **3.14×** | 5.9e-6 PASS | 9.2 → 16.6 GB (+7.4, ack) |
| VCD n=2 (4 it, 1024³) | 98.9 → **34.9 s** | **2.84×** | 6.7e-6 PASS | 25.0 → 26.8 GB (+1.8) |

**End-to-end VCD n=2 nearly triples with BOTH bands pallas**: 97.9 s (pure XLA, the
inc3 baseline) → 50.3 s (inc3, back band only, 1.95×) → **34.9 s** (both bands, 2.84×)
— the forward adoption adds 1.44× on top of increment 3.  The forward already SCALED
with XLA (n=2 4.22 → n=4 2.11 s = 2.0×; pallas 1.31 → 0.672 s = 1.95×, both near-ideal —
unlike back, which anti-scaled pre-inc3), so the kernel is a flat ~3.2× on top at every
device count.

**Value margins** (per the gate instruction): the isolated forward pallas rel is
5.8–5.9e-6 single-shot, and the VCD rel is **6.7e-6 — identical to inc3's back-only VCD
rel**, i.e. adopting the forward band did NOT inflate the iterated floor (the review's
thinner-margin concern did not materialize).  Comfortable headroom below the 1e-4 gate
and well under the ~3e-5 want-to-know line — routine.

**Memory ack:** the isolated forward pallas adds +6.8 GB (n=2) / +7.4 GB (n=4) per GPU —
the per-owner weights/streams transient (`forward_project_subset`'s (P, band) gather
tiles); absolute peaks 18.3 / 16.6 GB, well within the 80 GB H100.  Larger than inc3's
post-cap back +0.3 GB; the forward analog of the inc3 chunk cap (`fwd_view_batch`, the
existing per-chunk knob) could bound it if it grows at larger shapes — accepted for now,
revisit on OOM.  The end-to-end VCD peak moves only +1.8 GB (the sino-shaped updater
transients set the VCD peak, not the forward band).  These peaks ARE per-GPU (the
GPU-filtered reader, this diff's response to the inc3 review finding); one benign
`module: command not found` from `load_conda_cuda.sh` in the non-interactive shell (the
job ran on GPU, results correct).

## Cone fused-vfan back kernel — DESIGN DRAFT (2026-07-13; the load-bearing (c) item; discussion-first, no code yet)

**Operation.**  Cone banded back, per (pixel p, band slice l):
`out[p, l] = Σ_v Σ_tr Σ_tc  Wrow[v,p,l,tr] · Wchan[v,p,tc] · sino[v, m(v,p,l)+tr−r, c(v,p)+tc−r]`
— the horizontal fan's 3 channel taps × the vertical fan's 3 row taps.

**The geometry fact that enables fusion** (verified in `cone_beam.py`
recon_ijk_to_xyz → geometry_xyz_to_uv_mag → detector_uv_to_mn): z is affine in slice
index k; `v = pixel_mag(x,y) · z` with pixel_mag per-(view,pixel); `m` affine in v —
so **m(v,p,l) = m0(v,p) + slope(v,p)·l EXACTLY, flat and curved detectors alike**
(only u differs by detector type), and `W_p_r = pixel_mag·Δslice/Δrow` is
l-independent.  The l-dependent weight divisor cos φ = sdd/hypot(sdd, v_p) is 3 ALU
ops from the same affine.  Consequence: the vertical fan needs NO per-(v,p,l)
precompute — 3 scalars per (view, pixel) beyond the parallel kernel's existing
{c0, 3 channel weights}: {m0, slope, W_p_r}.

**Kernel (register-tile, extending increment 1's design):** grid = (slice-chunk,
pixel); the program holds out[p, l0:l0+LC] in registers and loops the view chunk
(BACK_VIEW_CHUNK_CAP=128): load the per-(v,p) scalars (ref-level), then per l:
compute m(l), row-center it, form the 3 trapezoid row weights (÷ cos φ), and
accumulate the 3×3 tap products (factored as 3 row-weight × 3 channel-scalar FMAs)
from the channel-major (V, C, rows) sinogram.  PANEL AMENDMENTS (wf_0fc29e84, both
agents' numerical checks in their scratch scripts): (1) `jnp.round` has NO Triton
lowering at the pin — emulate with floor(m + 0.5); safety rests on the verified
invariant W_p_r ≤ 2·psf_radius (a center flip moves a tap whose weight is EXACTLY
zero — forced-flip check: output delta 0.0), NOT on matching round-half-even
semantics; (2) the divisor must be the Inf-safe form 1/hypot(1, v_p/sdd)
(sdd = Inf is a supported configuration; sdd/hypot(sdd, v_p) is NaN there) and
division-free in-kernel; (3) `slope == W_p_r` EXACTLY (both are
pixel_mag·Δslice/Δrow — verified analytically and numerically), so the vfan
precompute is {m0, W_p_r} only; (4) day-0 lowering probe for the vector ref-gather
`sino_ref[v, cc, m_vec]` — the lowering table supports int-array indexers but no
shipped kernel exercises one, and the two-stage fallback needs the same construct.

**Traffic/flop model (panel-corrected):** load side = 9·V·P·L taps at the parallel
kernel's measured 1.81e12 taps/s ≈ 2.15 s per-owner sweep.  ALU side is NOT free
here (~50–60 FMA-eq per (v,l) element vs parallel's ~6): 2.2–2.6e13 FMA-eq ≈ 0.7 s
at f32 peak, 1.3–2.6 s at realistic utilization.  Honest expectation **2.5–4.5 s ≈
5–8×** (was 7–10×); the ≥1.5× bar (>14 s) keeps ≥3× margin.  L2 story corrected:
the per-view row window is dominated by the pixel-magnification SPREAD
(~110–170 rows at 1024-scale, LC-independent), so the nominal (128-view, C, window)
set is 68–90 MB > L2 — residency holds via the DRIFT-BAND mechanism (~70–95 views
of program drift fit in L2), the same mechanism the shipped parallel kernel already
relies on (its nominal phase set is 134 MB); the spike must report edge-chunk vs
central-chunk throughput separately.  LC sweep extended to {16, 32, 64, **128**} —
each slice-chunk phase re-streams the whole per-(v,p) scalar array, so large LC
amortizes it (the rc=256 lesson), and the register budget is comfortable at 128.

**Precompute:** one builder jit mirroring `_jit_compute_back_weights`: hfan
{c0, Wchan(T)} (identical math) + vfan {m0, W_p_r} — 24 B/(view, pixel) →
**2.53 GB** at chunk=128 × P=823k, bounded by BACK_VIEW_CHUNK_CAP.

**OPEN DECISION (panel blocker) — the Hessian value gate.**  The affinity is exact
in real arithmetic (f64 deviation 5.7e-14) but the in-kernel f32 `m0 + slope·l` is a
different ROUNDING SEQUENCE than the XLA chain: δm ≈ 1–2 ULP of m (~1e-4 abs at 1024
rows).  The GRADIENT is immune (trapezoid weights are a partition of unity — Σ taps
= W_p_r independent of m, so δm only transports weight between adjacent rows;
end-to-end emulation passes at 1.5–2.3e-6).  The HESSIAN (squared weights, no
cancellation) measures **1.5e-5 flat / 3.4e-5 curved vs the 1e-5 gate — FAIL**;
ablation confirms the affine-m recomputation is the sole driver.  Options:
(a) relax the Hessian gate to 1e-4 with the accuracy argument (the VCD fixed point
is set by the gradient; the Hessian diagonal only preconditions the path, so a
~3e-5 relative perturbation is limit-neutral and rate-negligible — verify
empirically in the spike with a VCD-trajectory comparison), keeping 1e-5 for the
gradient; (b) dispatch coeff_power=2 to the XLA band path (Hessian is computed once
per recon — costs one XLA-speed pass per recon, zero accuracy questions);
(c) reproduce the exact f32 op sequence in-kernel — fragile across XLA/Triton FMA
contraction, rejected.

**DECIDED (Greg, 2026-07-13): option (a)** — the Hessian is a preconditioner, a
little error is acceptable; gates are rel ≤1e-5 for the gradient, ≤1e-4 for the
Hessian, with a VCD-trajectory comparison in the spike as the empirical check.

**Fallback variant (record, not first):** two-stage in pallas (the existing hfan
spike kernel → HBM cylinder → a vfan register-tile) — fewer in-kernel loads but pays
the cylinder materialization; the plan's E0 finding (cone hfan hides behind the
vfan) argues the fused form first.

**Spike plan:** e3-style standalone bench at the per-owner shard shape (512 views,
1024×1024, L=115): fused kernel vs the XLA band call; sweeps LC ∈ {16, 32, 64} ×
num_warps ∈ {1, 2}; gates = rel ≤1e-5 vs XLA (gradient + Hessian via squared
weights) + the adjoint pair check; then E4-style integration behind
`back_pallas_band` for ConeBeamModel (the dispatch plumbing from increment 3 is
geometry-generic already).

## E5 — the cone fused-vfan back kernel: SPIKE BAR SMASHED (2026-07-13; jobs 13515406 fail → 13515407; `e5_cone_fused_back.py`)

Best config **lc=128, warps=1: 2.97 s for the full 11-band per-owner sweep vs 27.8 s
XLA (9.4×; 7.1× vs the best-XLA vb16 form)** — inside the panel's honest 2.5–4.5 s
window, 6× above the ≥1.5× bar.  Gates: gradient rel 6.5e-6 (≤1e-5 ✓), **Hessian
2.04e-5 (≤1e-4 ✓ — and inside the panel's predicted 1.5–3.4e-5 window for the
affine-m ULP effect; the numerical forecast was exact)**, adjoint vs the XLA cone
forward 0.0e+00.  The sweep matched every panel prediction: lc=128 best (per-(v,p)
scalar re-stream amortization — the rc=256 lesson again), warps=1 dominant, the
lc=16→128 curve monotone.

Two spike lessons: (1) a bare `pallas_call` (no compiler_params) selects the MOSAIC
backend on Hopper — warpgroup-divisible-copy errors; Triton `compiler_params` are
backend SELECTION, not tuning (probe round 1 failed on its own scaffolding);
(2) the interpret smoke caught an inverted cos-φ divisor pre-GPU via the residual's
magnitude ((v_p/sdd)²/2 ≈ the observed 7e-3).

Projection for integration: cone back n=2 per-owner work 27.8 → ~3 s; the trace's
MemcpyD2D (14 s/call self-time at n=2) overlaps compute rather than serializing —
parallel's gate proved that (its trace showed 3.7 s/call of D2D yet gated at
0.716 s) — so integrated cone back n=2 should land in the 3–4 s class vs today's
27.5 s, ending the anti-scaling.  Integration increment (next): promote
`cone_hfan_data` to `ConeBeamModel.compute_hfan_data`, move the fused kernel +
driver into `_pallas_kernels.py`, wire cone `back_pallas_band` (n≥2) AND cone n=1
`back_pallas` (full rows = one band — the (a)-track bonus, ~18.9 → ~2 s class)
through the same driver, gates = the increment-3 pattern + the VCD-trajectory
check backing the Hessian-gate decision (a).

## inc5 VCD divergence: INTRINSIC edge conditioning, not a kernel defect (2026-07-13; jobs 13518353/13518443/13518498/13519365)

The cone VCD n=2 trajectory gate (rel ≤1e-4, 4 seeded iterations) failed at 8.5e-3.
The diagnosis chain, each step a single-variable experiment:

1. **Hessian ablation** (cp=2 → XLA, the `_PALLAS_BACK_COEFF_POWERS` policy): rel
   UNCHANGED (8.47e-3 → 8.52e-3) — the low-Hessian-amplification hypothesis REFUTED
   (the Hessian-policy cell reads rel 0.0, confirming the policy works).
2. **Per-call gates at every VCD call shape**: full-grid 6.5e-6 (GPU), subsets
   sorted AND unsorted ≤4.7e-7 (interpret) — no per-call defect.
3. **Localization** (iters 1/2/4 + slice/radial profiles): divergence grows
   ~4×/iteration (1.1e-4 → 4.5e-4 → 8.5e-3), lives ENTIRELY in the outermost radial
   bin (9e-3 vs ≤4e-4 interior) and seeds in the flash-extension axial zone (iter-1
   worst slices all ≥952) — the minimal-ray-coverage voxels.  No band-seam period.
4. **THE CONTROL** (two pure-XLA runs, back_view_batch 512 vs 96 — a legitimate
   tuning knob, same reordering float class): **rel 1.33e-4 — ALSO FAILS the 1e-4
   gate**, same signature (interior 1–2e-6, edge bin 1e-4).

**Reading: linear response of ill-conditioned voxels.**  Amplification ≈10³ at
low-coverage voxels over 4 iterations for BOTH pairs (control per-call ~1e-7-class →
1.33e-4; pallas per-call 6.5e-6 → 8.5e-3; ratios match at ~64-65×).  The max-norm
trajectory gate over the full support is passable only by bitwise-identical
implementations — it is MISCALIBRATED for cone at this scale, not a kernel verdict.
Interior behavior is likewise linear (pallas interior 2-4e-4 = control interior
1-2e-6 × the per-call ratio).

**Open decision (Greg): the cone VCD value-gate criterion.**  Options: (a)
interior-masked rel with a control-calibrated tolerance; (b) convergence-equivalence
— run off/on to production iterations against a deep reference and require equal
NRMSE trajectories within the control's noise band (the parity-study methodology),
plus a visual check; (c) both.  Also decidable: with the intrinsic explanation
confirmed, the cone cp=2 → XLA policy (worth ~30 s/recon) was based on a
misattributed signal and can be reverted — or kept as cheap conservatism.

## Increment 5 CLOSED: the convergence-equivalence gate PASSES on the full configuration (2026-07-13; job 13520862; `w2_inc5_convergence.py`)

Real data + the parity study's 150-iteration depot references (Greg's design: an
occasional gate, never nightly); off / on / XLA-control × iters {8, 15} at n=2,
seeded; cropped log10 NRMSE per the parity conventions; PASS = |on−off| ≤ 3× the
control band.  Cone cp=2 was REVERTED to the fused kernel first (convert-then-gate,
bisect only on failure — Greg):

| case, iters | NRMSE off | NRMSE on | \|on−off\| | control band | Δlog10 |
|---|---|---|---|---|---|
| lilly_ds8, 8 | 0.145033 | 0.145033 | 6.0e-8 | 4.4e-5 | −0.0000 PASS |
| lilly_ds8, 15 | 0.111308 | 0.111308 | 8.2e-8 | 3.3e-5 | −0.0000 PASS |
| z62, 8 | 0.043103 | 0.043103 | 0.0 | 1.3e-5 | +0.0000 PASS |
| z62, 15 | 0.032022 | 0.032022 | 3.7e-9 | 9.6e-6 | −0.0000 PASS |

Vacuity check: the on-cells' device lines read `(pallas: band-back)`, the off-cells'
don't.  Reading: at the reconstruction level the paths are equivalent to ~8 digits of
interior NRMSE — the edge-voxel pointwise noise (the intrinsic-conditioning story)
contributes nothing to converged quality.  The Hessian revert stands validated; no
bisect needed.  **Increment 5 is complete**: cone back 3.9×/9.9×/6.0× at n=1/2/4,
both coeff powers through the fused kernel, cone anti-scaling dead, value equivalence
gated at the level that matters.

## Cone forward kernel — DESIGN OPENING (2026-07-13; increment 6 candidate)

Structure (cone_beam.forward_project_pixel_batch_to_one_view): **vfan first** —
x(P, slices) → cylinder(P, det_rows) per view, the affine row spread — **then hfan**,
the channel-sorted two-phase scatter the shipped forward kernel already solves.
Baseline: XLA cone fwd 19.4 s full-grid at the 1024³ cell (campaign); the E1 split:
hfan 72% / vfan 41% (overlap 1.13).  The E3 cone-hfan spike kernel exists (v2
row-chunked, 2.13× subset).

Structural difference from parallel fwd: the post-vfan values are PER-VIEW (m0/W_p_r
are view-dependent), so the shared L2-hot values tile that powers the parallel
forward kernel does not exist — wins must come from sort/scatter elimination and
fused resampling, not cross-view tile reuse.  The vfan itself is collision-free in
BOTH directions: forward spread per pixel column, or a gather via the INVERSE affine
l*(m) = (m − m0)/W_p_r (each output row gathers its ~T slice taps).

Candidate architectures for the design pass:
* **(A) hfan-stage adoption**: the spiked cone-hfan kernel serves the hfan stage,
  vfan stays XLA — Amdahl ≈ 1.6× cone fwd; smallest change, needs the two-stage
  library restructure of the cone forward internals.
* **(B) two-stage pallas**: an inverse-affine vfan gather kernel producing per-
  (view, pixel-tile) cylinders + the hfan kernel — pays the cylinder materialization
  per view tile; pixel tiling requires the hfan kernel's multi-tile atomic variant
  (the deferred increment-2 "batched-grid" machinery).
* **(C) fully fused**: the hfan segment-walk whose per-(pixel, row) reads compute the
  vfan in-kernel via the inverse affine (~3 gathers + weights per element, ~3× the
  parallel fwd per-element cost; zero intermediate).  Rough ceiling ~2.5× vs XLA
  from the per-element model — the design pass must sharpen this before choosing.

Next: full design + adversarial panel (the E5 pattern), then the spike.  The
Hessian/rounding lessons carry over: channel centers concrete, row rounding
in-kernel under the W ≤ 2r invariant, value gates gradient-class only (forward has
no squared-weight path), convergence-level equivalence for the VCD composition.

## Cone forward fused kernel — DESIGN (architecture (C), Greg-approved 2026-07-13; discussion-first, no code yet)

**Correction to the design opening**: the shared-tile premise HOLDS for (C) — the
gather source is the raw, VIEW-INDEPENDENT ``x(P, slices)``; only the in-kernel
indices are view-dependent.  (B)'s per-view cylinders are what break sharing.

**Why the inverse formulation is forced, not chosen.**  The natural forward vfan
spreads each x[p, l] into rows m(l)±r — an accumulator SCATTER at dynamic REGISTER
indices, which Triton cannot express (dynamic indexing is legal only on refs, i.e.
reads).  Inverting the affine turns every dynamic index into a ref-gather: for
output row m, the contributing slices are l ∈ round((m − m0)/W_p_r) ± T_l, and the
weight is the SAME trapezoid evaluated at m_p(l) − m (symmetric |·|, so the pair is
adjoint-matched by construction).

**The kernel = the shipped forward kernel + one substitution.**  The two-phase
channel-sorted segment walk is unchanged; the per-pixel contribution
``wt · vals_ref[pix, :]`` becomes ``wchan · resample(vals_ref[pix, :])`` where
resample, vectorized over the rows axis (rows-length vectors are proven — the
shipped kernel ran band=1024): l_c = the inverse affine of the row vector, then T_l
static taps of {vector ref-gather x[pix, l_c+tl] × trapezoid(m_p(l_c+tl) − m) ÷ the
cos φ divisor matching the XLA forward vfan}.  Precompute = exactly inc5's
builders: {c0, Wchan(T)} + {m0, W_p_r}.  No coeff_power axis (forward has none).

**PANEL AMENDMENTS (wf_39d30f52; two blockers, both fixed by simplifications).**

*Tap window = ``gp.bp_psf_radius``, not the derived T_l formula.*  The panel's
central discovery: the XLA forward vfan is ITSELF an inverse-affine gather
(``create_det_column_rows``, k_offset ∈ ±bp_psf_radius) — the fused kernel is a
transplant of XLA's own structure, and using XLA's own window is gate-safe BY
CONSTRUCTION: it matches XLA window-for-window (verified 3.2–6.6e-6 nrmse across
geometries incl. helical + offsets), is support-complete exactly when XLA is
(bp ≥ ceil(1/(2W)) verified over 5000-point isotropic sweeps), and inherits XLA's
own truncation on anisotropic-pixel pathologies (ddc < ddr, W ≲ 0.5 — a
pre-existing XLA approximation, ~8% of a stress scan, none with square pixels) so
user-visible values never change.  It is also SMALLER: bp=1 (3 taps) in the default
geometry where my formula took 5 — restoring the upper end of the model.  Dispatch
guard: fall back to XLA when bp_psf_radius > 2 (the MAX_PSF_RADIUS precedent).

*The 1e-5 single-shot gate is below the f32 noise floor and must be calibrated.*
At 1024 detector rows, two CORRECT f32 implementations of the same vfan differ by
~1.3e-4 max-rel (the trapezoid arguments are differences of O(1024)-scale values;
ulp(1024) = 1.2e-4); f64-vs-f64 agrees to 2e-13.  The E5 back precedent does not
transfer: back outputs average over ~all views (~√V noise reduction), forward's
single-view outputs do not.  The spike measures the floor (f64-truth two-sided
and/or an XLA-reorder control) and sets the calibrated single-shot contract
(~1e-4-class max-rel at production rows; nrmse lands ~5e-6-class).

*Feasibility notes to pre-commit*: register budget at bp=1 ≈ 5–6 live rows-length
vectors (~160–192 regs/thread, inside the ceiling); if the spike spills, sweep
num_warps ∈ {1,2,4} and add an optional row-chunk grid dim (the lc pattern, ~1–2%
stream re-walk cost) BEFORE reading it as an architecture failure.  Acks to
pre-register: +2×(V_chunk × P) f32 m0/W_p_r refs (~0.8 GB per full-grid chunk, the
mid-size-fwd ack class); the padded row tail computes real garbage made inert only
by the chunk-fn trim — the same "not zeroed here" contract note as the cone back
driver.  The zero-weight slice mask ((l ≥ 0) & (l < num_slices) on the WEIGHT, with
a clamped gather index) is load-bearing — omitting it measured order-1 errors.

*Verified clean*: the inverse gather is an exact summation reorder of the XLA vfan
(f64 agreement ≤2.6e-14 across flat/curved/helical/offset configs; the per-tap cos φ
divisor placement confirmed), tap coverage never misses a contributor, negative/
clamped inverse indices safe, two-phase machinery carries verbatim (stream counts
still sum to exactly T·P).

**Model and bar (amended)**: per-element ≈ (2·bp+1)/3 × the parallel forward's
loads + inverse ALU; anchors: parallel pallas fwd full-grid 2.78 s, XLA cone fwd
19.4 s ⇒ expected **2.2–3.9× at bp=1**; bar ≥1.5×.  Gates: calibrated single-shot
contract (above) at full-grid and subset+band shapes, the PAIR adjoint against the
inc5 cone back kernel, interpret mode on CPU CI; VCD composition via the occasional
convergence gate.  Dispatch (post-spike): cone fwd_pallas / fwd_pallas_band through
the inc5 geometry-hook pattern, guarded on bp_psf_radius ≤ 2.

## E6 — the cone fused forward kernel: SPIKE BAR CLEARED (2026-07-13; job 13530564; `e6_cone_fused_fwd.py`)

| cell | XLA | E6 (warps=1) | E6 (warps=2) | values (nrmse / max-rel) |
|---|---|---|---|---|
| full grid (821,904 px, 1152 slices) | 23.03 s | 10.86 s (2.12x) | **8.85 s (2.60x)** | 1.1e-6 / 8.9e-6 PASS |
| subset (6,026 px) | 0.189 s | 0.089 s (2.12x) | **0.071 s (2.65x)** | 1.2e-5 / 4.3e-5 PASS |
| adjoint vs the XLA cone back | — | — | — | 7.6e-8 PASS |

Inside the panel-amended 2.2-3.9x window; the bar (1.5x) cleared at 2.6x.  warps=2
wins everywhere -- the panel's register-pressure escape hatch was needed exactly as
predicted.  Values sit comfortably inside the floor-calibrated gates (full-grid
max-rel 8.9e-6 is far under the ~1.3e-4 measured floor -- the many-view nrmse
averaging helps more than the single-view analysis assumed).  bp=1 at this geometry
(3 taps).  Baseline note: 23.0 s here vs the campaign's 19.4 s = the extended
1152-slice/821.9k-pixel cell, not a regression.

NEXT (increment 6 integration, the inc5 pattern): kernel + driver into
`_pallas_kernels.py` (the spike file is the transplant source; CONE_FWD_NUM_WARPS=2),
cone fwd_pallas / fwd_pallas_band via the geometry hooks with the bp_psf_radius <= 2
dispatch guard, tests mirroring the parallel forward set + a bp>2-fallback test,
model-level gate (fwd n=1/2/4 off/on + the seeded cone VCD walls; value criterion =
the calibrated contract), then the occasional convergence gate on the final
configuration.

## Soak: repeated-run stability + demo sanity + CPU suite — ALL PASS (2026-07-13; jobs 13528061 soak-gates, 13528062 soak-demo; CPU suite ×3 local)

Composed, repeated-run validation of everything that shipped today (fwd driver fix +
get_compute_config; guard drop + 1280-col cap; parallel band inc3/inc4; cone fused-vfan
inc5) — not re-deriving the individual gates, but confirming they are stable run-to-run,
end-to-end, and on CPU.  `soak_gates.py` ran the three A/B harnesses ×3 back-to-back +
`w2_inc5_convergence.py` ×1 in one sbatch (1 h 23 m); `soak_demo_sanity.py` ran the
demo-1 workflow flag-on vs kill-switch at 1024³, n=1 (37 m).

**Gate stability — on-cell (the pallas path), 3 reps.**  Walls in seconds, rel = value
gate vs flag-off, peaks per-GPU.  Every per-cell wall max/min ≤ 1.026 (bar: < 1.1).

| gate | on-cell | wall min–max (ratio) | rel (3 reps) | peak GB | verdict |
|---|---|---|---|---|---|
| inc3 | par_back_n2 | 0.714–0.717 (1.004) | 1.31e-6 | 13.7–14.1 | 3/3 PASS |
| inc3 | par_back_n4 | 0.441–0.451 (1.023) | 8.74e-7 | 11.8–11.9 | 3/3 PASS |
| inc3 | par_vcd_n2  | 34.66–35.56 (1.026) | 6.7–7.2e-6 | 26.7–26.8 | 3/3 PASS |
| inc4 | par_fwd_n2  | 1.309 (1.000) | 5.8–5.9e-6 | 18.1–18.3 | 3/3 PASS |
| inc4 | par_fwd_n4  | 0.671–0.672 (1.001) | 5.82e-6 | 17.4–17.5 | 3/3 PASS |
| inc4 | par_vcd_n2  | 34.60–35.17 (1.016) | 6.2–8.2e-6 | 26.8 | 3/3 PASS |
| inc5 | cone_back_n1 | 4.890 (1.000) | 6.52e-6 | 21.15 | 3/3 PASS |
| inc5 | cone_back_n2 | 2.780–2.789 (1.003) | 6.52e-6 | 16.6 | 3/3 PASS |
| inc5 | cone_back_n4 | 2.459–2.470 (1.004) | 6.52e-6 | 13.7–13.8 | 3/3 PASS |
| inc5 | cone_hess_n1 | 5.011–5.022 (1.002) | 1.97e-5 | 21.15 | 3/3 PASS |
| inc5 | cone_vcd_n2 | 117.7–120.0 (1.019) | 8.44–8.47e-3 | 27.14 | 3/3 **EXPECTED-FAIL** |

Off-cell (XLA baseline) walls are equally stable (par_back_n2 5.394 exact ×3; cone_vcd
268.9–272.6, ratio 1.014).  Speedups reproduce the shipped gates exactly (inc3 back
7.5×/7.4×, inc4 fwd 3.2×, inc5 cone back 3.9×/9.9×/6.0×; VCD n=2 both-bands 2.8×).  The
lone non-pass is the DOCUMENTED `cone_vcd` expected-fail (8.47/8.47/8.44e-3 — the
intrinsic edge-conditioning of "inc5 VCD divergence" above; stable across reps, not
chased).

**Convergence-equivalence gate (once)** — all four cases PASS, reproducing
Increment-5-CLOSED: `|on−off|` ≤ 8.2e-8 vs control bands 9.6e-6–4.4e-5 (lilly_ds8 / z62
at iters 8 / 15).

**Demo-level sanity (1024³, n=1, flag-on vs MBIRJAX_DISABLE_PALLAS=1)** — both geometries
clear the rel-max ≤ 1e-3 real-workflow gate with large margin AND show the expected
device tokens:

| geometry | tokens | rel-max (interior) | wall off → on | peak GB off → on |
|---|---|---|---|---|
| parallel | (pallas: back+fwd) | 5.13e-6 (5.13e-6) | 585.1 → 149.1 s (**3.92×**) | 49.98 → 49.98 |
| cone | (pallas: back) | 6.85e-6 (6.85e-6) | 812.8 → 547.3 s (**1.49×**) | 54.26 → 57.76 (+3.5) |

Notable: the demo cone rel-max is 6.85e-6 (float-noise class), NOT the 8.5e-3 of the inc5
`cone_vcd` cell, and the interior crop equals the full rel — the edge divergence is
specific to the real-data minimal-coverage geometry (the flash-extension zero-coverage
zone) and does not appear in the shepp-logan demo, where a fully-converged recon agrees
to float noise everywhere.  This reinforces the "intrinsic edge conditioning, not a
kernel defect" reading.  End-to-end the single-GPU pallas paths give 3.92× (parallel
back+fwd) / 1.49× (cone back) on the whole recon.

**CPU suite ×3 (local)** — 313 passed / 2 skipped / 72 subtests, identical all three runs;
no flake among the pallas tests (the `test_qggmrf` precedent did not recur).

**Verdict: the shipped configuration is stable across repeated runs, end-to-end, and on
CPU.**  No cell failed reproducibly outside the documented `cone_vcd` expected-fail.
(Benign `module: command not found` from `load_conda_cuda.sh` in the non-interactive
shell, as in every campaign job — the jobs ran on GPU with correct results.)

## Increment 6 — the cone fused forward kernel: SHIPPED AND GATED (2026-07-13; commit 8f2278f + review fixes; job 13530895)

The E6 kernel integrated behind the UNIFIED wrapper dispatch (one seam for all four
forward flows via the `_pallas_forward_project` geometry hook; increment 4's parallel
band override folded in); cone policy gates on bp_psf_radius <= 2.  Review
wf_fd13091b: clean on the dispatch audit (no unintended pallas routing at any
internal caller), transplant fidelity, and policy init-order; two harness/test fixes
applied pre-gate.

| cell | XLA -> pallas | speedup | values (tol 3e-4) | per-GPU peak |
|---|---|---|---|---|
| fwd n=1 | 22.97 -> 8.84 s | **2.60x** | 9.0e-6 PASS | equal |
| fwd n=2 | 13.99 -> 6.58 s | **2.13x** | 8.9e-6 PASS | -2.4 GB |
| fwd n=4 | 10.82 -> 2.97 s | **3.64x** | 8.9e-6 PASS | **-8.7 GB** |
| VCD n=2 (4 it) | 271.7 -> **85.0 s** | **3.20x** | 8.3e-3 INFO (intrinsic, stable) | equal |

Tokens verified: n=1 `(pallas: back+fwd)`, n>=2 `(pallas: band-back+band-fwd)`.
**Cone VCD n=2 composes to 3.20x end-to-end** (the back adoption alone gave ~1.8x;
the forward adds ~1.76x).  The occasional convergence gate on this final
configuration (both cone kernels) is the last box; submitted as the closing check.

## Increment 6 CLOSED: the convergence gate passes on the final configuration (2026-07-13; job 13530971)

Both cone kernels active (band-back + band-fwd), real data vs the depot references:
|on-off| NRMSE <= 6.7e-8 across lilly_ds8/z62 x iters {8,15} -- hundreds of times
inside the control bands, dlog10 = 0.0000 everywhere.  **The kernel campaign's
increment series is complete**: parallel back 9.1x/7.5x/7.4x and forward
3.2-3.8x/3.2x/3.1x (n=1/2/4); cone back 3.9x/9.9x/6.0x and forward 2.6x/2.1x/3.6x;
end-to-end VCD n=2: parallel 97.9 -> 34.9 s (2.83x), cone 271.7 -> 85.0 s (3.20x);
both anti-scaling regimes dead; every path value-gated per-call at calibrated
contracts and at the reconstruction level against deep references.

## Pending
- Cone 1024³ VCD iteration wall (wall-only rerun); cone fwd hfan/vfan split at 1024³.
- A2 flatten A/B (small); the subset-call concat fast path (observation 4).
- ncu round (pipe-level attribution of the two big fusions; the band kernel at n≥2);
  `ncu` availability on compute nodes to be checked.
- Then: the Pallas spike per the plan (band kernel first — the (c) main target).
