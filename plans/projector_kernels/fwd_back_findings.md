# Forward-projection kernel investigation: attribution, alternatives, and results

**Written 2026-07-07, final update 2026-07-08** (branch `greg/kernel_investigation`).
Companion to the benches in plans/experiments/projector_kernels/ (`fwd_back_kernel_ab.py`,
`fwd_band_pixel_sweep.py`, `mt_back_kernel_ab.py`, `translation_fwd_psf_ab.py`,
`pixel_count_crossover_ab.py`, `phased_overhead_ab.py`), which produced every number below.

## Developer overview — what changed relative to prerelease (the campaign's full delta)

- **Projector tiling is consolidated into a `TilePolicy`** (`model.tiles`): every batching /
  banding knob and kernel-algorithm flag, selected in ONE method
  (`TomographyModel._select_tile_policy`, re-run on each device re-layout) and read late-bound
  by all consumers.  Geometry classes override only what they have measured.  Experiment
  override idiom: `model.tiles = model.tiles._replace(...)`.  The retired attribute names
  (`view_batch_size_for_vmap` and its fwd/back split, `pixel_batch_size_for_vmap`,
  `transfer_pixel_batch_size`) raise loudly with guidance.
- **Per-op view batches** (from the view-batch work that preceded this investigation): forward
  keeps an OOM-safe width (cap 128); back single-vmaps its per-device view shard (cap 512
  sharded / 128 single-device) to avoid the accumulating-scan carry.
- **GPU forward horizontal fans use the SORTED channel reduction** (all four geometries carry
  the branch; parallel, cone, and multiaxis ENABLE it by policy — translation deliberately
  does not, see the collision-cliff section below) behind a single static flag,
  `ProjectorParams.sort_by_channel`.  The reduction lives in
  `projectors.channel_scatter_reduce`; the sorted form uses `lax.sort_key_val` with the
  sorted keys AS the segment ids.  Three measured guard constants (min columns, max psf
  radius, min collision ratio) gate the policies.  CPU keeps the original scatter loop.
- **The GPU parallel-beam back kernel gathers all psf taps at once** (one
  (psf_width · num_pixels, num_rows) gather + reshape-sum) behind a second flag,
  `ProjectorParams.back_stacked_gather`.  Every geometry's back kernel honors the flag, but
  only parallel's policy enables it: for the vertical-fan geometries the gather hides behind
  the band work (a measured composition no-op — three confirmations below).
- **The horizontal-fan and banded-vertical-fan kernels are DRY**: the trapezoid tap machinery
  lives once in `projectors.py` (`horizontal_fan_project` / `horizontal_fan_back` /
  `vertical_fan_band_gather`); geometry files keep only their coordinate stages and weight
  scales.  Consolidation was value-gated bitwise (one accepted last-bit reassociation in
  translation).
- **The horizontal fans' integer channel centers are concrete inputs** (the rounding-bug
  fix): computed in a separate small jit (`projectors._jit_compute_scatter_centers` from each
  geometry's `compute_channel_coordinate`) and passed into the projector programs, whose
  compiled forms are round-free for parallel beam.  The wrappers use a 256 MB single-call /
  view-chunked hybrid and carry a no-eager-array-ops performance contract.  See the Phase D
  sections below and `plans/bugs_and_artifacts/jax rounding bug/phase_d_design.md`.
- **Parallel-beam GPU forward tiling** (measured): slice band 256, pixel batch 8192; cone
  gets pixel batch 4096 above 768 slices.  Back tiling was swept and PARKED — its bands
  protect reduce-scatter memory, so only memory-expensive trades exist.
- **Float32 matmuls default to full float32** (TF32 opt-out via
  `JAX_DEFAULT_MATMUL_PRECISION=float32` in `_device_setup`, environment-overridable).
- **Final scoreboard (H100, model level), investigation start → campaign end at 1024³ n=1:**
  parallel forward 35.0 → 7.9 s (4.4×), parallel back 18.2 → 10.5 s (1.7×), cone forward
  41.5 → 18.8 s (2.2×); multiaxis forward 1.2–1.4× at its cells; parallel VCD neutral;
  memory flat at capacity (cone 1024³ −3.2 GB).
- **Accepted trades (memory-gate acks pending):** parallel forward small cells +27–58%
  relative (≤0.65 GB absolute); cone forward 1024³ +9.6/29/46% at n=1/2/4; mid-size forward
  cells +~0.5 GB from the materialized centers + chunk concat; 513-class back +8% (chunked
  accumulation).

The original question: why is `parallel_beam.forward_project_pixel_batch_to_one_view` slower
than `back_project_one_view_to_pixel_batch` (nightly model-level: fwd/back ≈ 1.4–1.9× GPU,
≈ 1.5× CPU)?

**Answer in one line:** forward's psf loop is a **scatter-add with duplicated channel indices**
(~T·P/C pixels collide per channel); back's is a conflict-free **gather** + dense accumulate.  The
`fwd_noscatter` control (identical arithmetic, no channel scatter) runs at 1–6% of the forward
kernel on both platforms, so the scatter is essentially the whole kernel-level story.

**Notation** (matching `tomography_model.py` where it overlaps):

- **P** — pixels per kernel call (the driver's pixel batch, `tiles.fwd_pixel_batch`;
  2048 default, 8192 for parallel-GPU forward).
- **C** — detector channels.
- **B** — the slice-band width: how many recon slices (equivalently detector rows, by the
  parallel-beam row r ↔ slice r identity) one kernel call processes.  This is the band length
  `B` of `tomography_model._slice_band_length`; an unbanded call has B = the full slice count.
  B is the column count of the channel reduction, so it is what the sorted reduce's fixed sort
  cost amortizes over.
- **V** — the view vmap width (`tiles.fwd_view_batch`, 128 in production).
- **T** — psf taps per pixel (2·psf_radius + 1; 3 for these geometries).

## Method

Kernel-level timings use the PRODUCTION kernel shape: a 2048-pixel batch (the driver's
pixel scan unit) vmapped over 128 view angles (the forward view batch), on one device.  Back is timed as its driver composes it
(vmap over views, then the view sum).  A driver-level ground truth
(`projector_functions.sparse_forward_project` / `sparse_back_project` on the full pixel grid)
anchors the kernel numbers to reality.  Values are checked with a robust
fraction-of-deviating-elements metric (see "Numerics note").

## Variants and results

All variants are value-equal reformulations of the forward kernel (same `n`, `A` from the same
geometry code; only the channel reduction differs), except `fwd_noscatter`.

| variant | what it does | CPU (M3) | GPU (H100) |
|---|---|---|---|
| `fwd_asis` | library kernel: 3 sequential `.at[n,:].add` scatters, one per psf tap | 2.7–3.4× back | 1.07× back (448-size), 1.29× (1024-size) |
| `fwd_1scatter` | ONE stacked scatter: (T·P,) indices, (T·P, B) updates | ~1.2× worse than as-is | ~1.45× worse than as-is |
| `fwd_segsum` | `jax.ops.segment_sum` over the stacked indices (unsorted; lowers to scatter) | ≈ `fwd_1scatter` | ≈ `fwd_1scatter` |
| `fwd_sortsegsum` | argsort the stacked indices IN-kernel, `segment_sum(indices_are_sorted=True)` | **2–4× WORSE** | **2.3–3.2× BETTER** (0.34–0.57× back) |
| `fwd_matmul` | dense (C, P) one-hot weight matrix @ (P, B) voxels — C/T× redundant flops | ties at C=160, loses at C=384 | **5.8× better — but TF32 only**; forced float32 it loses at 1024-size (1.93× back) |
| `fwd_gathertable` | host-precomputed per-view channel→pixel tables padded to (C,K); kernel = pure gather + reshape-reduce | 19× worse | 16× worse |
| `fwd_noscatter` | lower bound: all compute, psf reduce, NO channel scatter (wrong values) | 1–6% of as-is | 3–4% of as-is |

Notes per variant:

- **`fwd_1scatter` / `fwd_segsum`:** merging the 3 taps into one scatter does NOT help — the cost
  is the scattered accumulation itself, not the kernel-launch count.
- **`fwd_sortsegsum` — the GPU winner.**  With sorted indices XLA lowers the segment reduction to
  an efficient contiguous reduce instead of atomics; the win *includes* the in-kernel argsort
  (~6K elements/view at production shape).  On CPU the same path is far slower than the plain
  scatter — XLA's CPU sort+segment lowering is poor — so this is a **platform-conditional** kernel
  (precedent: back projection already splits monolithic-vs-band by platform).
- **`fwd_matmul`:** the spectacular GPU number rides TF32 tensor cores (JAX default matmul
  precision on H100).  Forced to full float32 it is no longer a win at scale.  Using it would be
  a *precision-policy* decision, not a drop-in optimization.
- **`fwd_gathertable`:** dead end as padded tables — per-channel pixel counts skew ~17× above the
  mean (K = 19439 vs mean ~1150 at the 448-size full grid), so padding waste dominates.  The
  unpadded form of this idea IS `fwd_sortsegsum`.

## The three tiers of the gap (GPU)

| tier | fwd/back | where the extra time lives |
|---|---|---|
| kernel (production shape) | 1.07–1.29× | the channel scatter |
| raw driver (full grid) | 1.06× (448) → **1.56×** (1024) | forward's ~480-step pixel scan accumulating into a ~512 MB (V, B, C) carry; back's driver has no equivalent |
| nightly model level | 1.4–1.9× | banded forward vs back's n=1 monolithic short-circuit; ragged view batches at odd view counts |

On CPU the kernel tier dominates outright (kernel 2.7–3.4×, driver 2.4–3.1×); the nightly's
smaller 1.5× is because the deployed CPU paths use band kernels on both sides.

## Strategy ranking (assessment of 2026-07-07; status updated 2026-07-08)

1. **[DONE — Phase A + cone rollout]** **GPU platform-conditional sorted-segment-sum reduction** — verified 2.3–3.2× kernel win,
   exact values, pure XLA.  Structure: keep ONE shared kernel body (geometry → `n`, `A`) and
   platform-dispatch only the ~10-line channel reduction.  The same primitive drops into the
   cone/multiaxis/translation horizontal fans (same scatter structure).
2. **[DONE — Phase B]** **Driver tier:** revisit the forward pixel batch for parallel (fewer scan steps →
   less carry traffic; 2048 was tuned for cone's gather path).
3. **[DONE via the band policy — Phase B]** **Model tier:** quantify banded-forward overhead
   at n=1.  The band × pixel grid answered it empirically: band 256 beats BOTH the default
   narrow bands and the monolithic (whole-shard) form at n=1, so no short-circuit is needed.
4. **[REJECTED]** **TF32 matmul:** only viable with a reduced-precision policy the
   correctness gates rule out; instead the library now OPTS OUT of TF32 globally
   (`_device_setup`).
5. **[ACCEPTED/PARKED]** **CPU:** the scatter is XLA-CPU-inherent; every alternative measured
   worse on CPU (kernels, bands, and pixel batches alike), so all CPU paths are unchanged.
   The honest alternative remains an algorithmic rewrite (shear/resample forward) — parked.

## Phase A implementation results (2026-07-07, branch `greg/kernel_investigation`)

Strategy 1 was implemented: `channel_scatter_reduce` in `projectors.py` (scatter-add vs
`lax.sort_key_val`-based sorted segment-sum, with the sorted keys used AS the segment ids so
they cannot disagree — robust against the rounding-bug divergence class), dispatched in the
parallel forward kernel on `ProjectorParams.backend_gpu` (an int leaf — a str breaks where
the namedtuple is traced) AND on the band width.  The CPU path keeps the original loop
verbatim (compiled CPU HLO proven bit-identical to HEAD, modulo source-location metadata).

H100 model-level A/B (isolation-clean harness cells, old → new; memory unchanged ≤2%):

| parallel forward | old ms | new ms | speedup |
|---|---|---|---|
| 1024³ n=1 / n=2 / n=4 | 34974 / 17473 / 9325 | 16656 / 8312 / 7083 | 2.10 / 2.10 / 1.32× |
| 513³ n=1 / n=2 | 1117 / 611 | 439 / 400 | 2.54 / 1.53× |
| 513³ n=4 (bands B=24) | 285 | 525 → threshold added | see below |
| 200³ n=1 | 33.5 | 11.0 | 3.05× |

Forward is now FASTER than back at n=1 (439 vs 777 ms at 513³).  Guards: parallel back and
cone forward byte-flat.

**Band-width threshold.**  The sorted reduce's per-call cost is sort-dominated and nearly
independent of the band width B (~0.65 ms per 128-view/2048-pixel call), while the scatter
scales with B — so narrow bands (the banded forward at high device counts) favor the scatter.
Standalone-kernel sweep crossover ≈ B=96 (both problem sizes); end-to-end anchors: sorted
LOSES at B=24 (513³ n=4), WINS at B=63 (1024³ n=4) — the in-scan context favors sorted more
than standalone timing suggests.  Dispatch threshold `SORTED_CHANNEL_REDUCE_MIN_COLS = 48`
sits between the end-to-end anchors.  Follow-up: forward band sizing is inherited from back's
memory-driven policy; longer forward bands (Phase B/C) would push everything to B ≥ 96.

## Phase B implementation results (2026-07-08, branch `greg/kernel_investigation`)

Strategies 2 and 3 landed together via the **TilePolicy consolidation**: every projector
batching/banding knob and kernel-algorithm flag now lives in one immutable namedtuple
(`model.tiles`), selected in ONE method (`_select_tile_policy`) on every device re-layout and
read late-bound by all consumers.  The kernel's compound branch collapsed to a single
precomputed flag (`ProjectorParams.sort_by_channel`, decided by the policy).  Experiment
override idiom: `model.tiles = model.tiles._replace(...)`; the per-instance
`fwd/back_project_slice_band` attributes remain the top-priority hook.

`ParallelBeamModel`'s GPU policy (from the band × pixel-batch grid, `fwd_band_pixel_sweep.py`):
`fwd_slice_band = 256` (whole-shard is WORSE at 1024³ n=1: 0.97×, +3 GB), `fwd_pixel_batch =
8192` (16384 is non-monotonic), sorted reduce on unless the balanced band falls below the
crossover.  CPU measured the opposite direction on every knob, so the base (CPU) policy is
untouched and the CPU compiled program is bit-identical (HLO-proven, model level).

H100 model-level A/B (old = post-Phase-A HEAD, isolation-clean cells):

| parallel forward | old ms | new ms | speedup | memory |
|---|---|---|---|---|
| 1024³ n=1 / n=2 / n=4 | 16727 / 8308 / 7086 | 8187 / 4103 / 1973 | 2.04 / 2.03 / 3.59× | +3% / −8% / +22% |
| 513³ n=1 / n=2 / n=4 | 438 / 398 / 285 | 342 / 201 / 148 | 1.28 / 1.97 / 1.93× | −15% / +27% / +58% |
| 200³ n=1 | 11.0 | 10.5 | 1.05× | +54% (88→136 MB) |

Guards: parallel back and cone forward byte-flat; parallel VCD 1.05× (its interior forward).
CUMULATIVE (Phase A + B) vs the original kernel: 1024³ forward 34.97 s → **8.19 s at n=1
(4.3×)** and 9.32 → **1.97 s at n=4 (4.7×)**; forward is now ~2.2× FASTER than back at
1024³ n=1.  Back projection is the new long pole.

NOTE for the nightly: the relative memory increases at the small cells (+27–58%, absolute
≤ 0.65 GB) exceed the hard 8% GPU memory gate — a deliberate time-for-memory trade that needs
the acknowledged-regression path (or re-baseline) when this lands.

## Cone horizontal-fan rollout results (2026-07-08, branch `greg/kernel_investigation`)

Cone's forward horizontal fan is line-for-line the same channel scatter as parallel's kernel,
so it received the same single-flag branch (`sort_by_channel` → stacked taps + sorted reduce;
else the original loop verbatim).  Cone-specific details: `W_p_c` / `L_max` / `footprint_xy`
are PER-PIXEL arrays (per-pixel magnification) — plain broadcasting covers them and both
detector types (flat and curved are kernel-equality-tested); the reduce columns are the FULL
detector rows (cone forward is not row-banded), so `ConeBeamModel._select_tile_policy` sets
the flag whenever `num_det_rows >=` the crossover.  Band/pixel-batch knobs stay inherited —
cone's own sweep is future work.

H100 model-level A/B (old = pre-sorted-reduce cone):

| cone forward | old ms | new ms | speedup | memory |
|---|---|---|---|---|
| 1024³ n=1 / n=2 / n=4 | 41491 / 22507 / 15329 | 29245 / 15648 / 10504 | 1.42 / 1.44 / 1.46× | flat |
| 512³ n=1 / n=2 / n=4 | 1288 / 695 / 462 | 803 / 445 / 298 | 1.60 / 1.56 / 1.55× | flat |
| 200³ n=1 | 42.1 | 30.7 | 1.37× | flat |

Guards: cone back byte-flat; parallel forward at its Phase B value.  The smaller speedup than
parallel's (1.4–1.6× vs 2–3×) is the vertical fan's untouched share of cone forward.  Watch
item: the cone VCD guard cell moved +1.9% (single trial) — VCD calls forward on SMALL pixel
subsets where the sort has less to amortize; if the nightly confirms it, the clean fix is a
static pixel-count guard INSIDE `channel_scatter_reduce` (the kernel's single-flag branch
stays).

## Back-projection results (2026-07-08, branch `greg/kernel_investigation`)

With forward fixed, back became the long pole (18.2 vs 8.2 s at 1024³ n=1).  Attribution
(`back_kernel_ab.py`): back's kernel is the platform-MIRROR of forward's — on CPU the
data-dependent gather is only ~10% of the kernel (FMA-bound; nothing to gain), on GPU it is
~95% (`back_nogather` = 0.05×).  The tile-knob sweep (`back_tile_sweep.py`) found ONLY
memory-expensive trades (best 1024³ n=2 cell: 1.57× time for 3.3× memory; n=1 tops out at
1.08× at 61 GB) — back's bands exist to protect reduce-scatter memory, and widening them
spends exactly that memory.  Tiling PARKED; defaults unchanged.

The clean win: **stack the psf taps into ONE (T·P, rows) gather + reshape-sum** — 1.6–1.8×
kernel-level on H100, exact values, but 3.6–4.4× WORSE on CPU (FMA-bound + cache).  Landed as
the second kernel-algorithm flag, `back_stacked_gather` (TilePolicy → ProjectorParams), same
pattern as `sort_by_channel`; CPU keeps the loop verbatim (HLO-proven identical).

H100 model-level A/B:

| parallel back | old ms | new ms | speedup | memory |
|---|---|---|---|---|
| 1024³ n=1 / n=2 / n=4 | 18231 / 11069 / 4803 | 10919 / 4930 / 2673 | 1.67 / 2.25 / 1.80× | flat / −5% / flat |
| 513³ n=1 / n=2 / n=4 | 777 / 429 / 204 | 314 / 120 / 59 | 2.47 / 3.58 / 3.44× | −46% / −42% / −10% |
| 200³ n=1 | 13.6 | 15.1 | 0.90× | −68% |

Multi-device beats the kernel-level prediction (the band path repeats the kernel with narrow
rows, so collapsing 3 gather passes to 1 pays per band), and memory DROPS at mid sizes (no
per-tap accumulator coexistence).  Guards: forward and cone back flat; parallel VCD +10%
(the Hessian's coeff_power=2 rides the same branch).  Accepted trade: the tiny 200³ n=1 cell
pays +1.5 ms for −226 MB.  Remaining headroom: the kernel is still ~90% gather-bound
(`back_nogather` = 0.08–0.10×) — data-layout / custom-kernel territory.

**Combined scoreboard at 1024³ n=1 (H100), investigation start → now:** forward 34.97 →
8.19 s (4.3×), back 18.23 → 10.92 s (1.67×) — the pair is roughly balanced again.

## Cone follow-ups: back attribution (no-op) and forward pixel batch (2026-07-08)

**Cone back: measured, and deliberately NOT changed.**  `cone_back_kernel_ab.py` split the
monolithic kernel: the stacked-gather horizontal fan wins in ISOLATION (0.57–0.59× vs the
loop's 0.91–1.04×), but substituted into the FULL kernel it changes nothing (1.00×, values
identical) — and the shares don't add (hfan 1.04 + vfan 0.46 ≈ 1.5 ≫ 1.0), i.e. XLA already
overlaps the horizontal-fan gather with the vertical-fan band work.  Parallel back has no
vertical fan to hide behind, which is why the same change won 1.7–3.6× there.  The cone
policy documents this as a deliberate non-setting of `back_stacked_gather`.

**Cone forward: size-conditional pixel batch.**  `cone_fwd_tile_sweep.py` (H100, model
level): at 1024³-class, `fwd_pixel_batch = 4096` gives **1.51 / 1.53 / 1.13× at n=1/2/4**
(29.2 → 19.4 s at n=1); larger batches add little beyond 4096 while memory balloons (16384 →
30–38 GB).  At 512³-class it is neutral-to-worse (0.95–1.00×) while paying +41% memory.
Policy: `ConeBeamModel` uses 4096 on GPU only when `num_slices >= 768` (between the measured
sizes).  Memory at 1024³: +9.6/+29/+46% at n=1/2/4 (1.5–4.1 GB absolute) — same
memory-gate-ack situation as parallel forward.  View-batch 256 spot checks: small extra gains
only at large memory multiples; not adopted.

## Multiaxis + translation rollout results (2026-07-08, branch `greg/kernel_investigation`)

Both geometries' forward horizontal fans are the same channel scatter as parallel/cone, and
both received the same single-flag branch (`sort_by_channel` → stacked taps +
`channel_scatter_reduce`; else the original loop verbatim).  CPU compiled programs
bit-identical to HEAD (metadata-stripped HLO diff, model level, fwd + back, both
geometries).  Outcome: **multiaxis enables the sorted reduce (guarded); translation keeps
the scatter path** — the investigation below found a production-shape cliff that the
harness's small cells could not see.

**Multiaxis: H100 harness-cell A/B (isolated cells, old → new; memory flat ≤1%):**

| multiaxis forward | old ms | new ms | speedup |
|---|---|---|---|
| 129x113x97 n=1 | 8.4 | 7.0 | 1.20× |
| 256x224x192 n=1 | 122.4 | 87.1 | 1.41× |
| 512x448x384 n=1 / n=2 / n=4 | 2301 / 1232 / 850 | 1738 / 930 / 638 | 1.32 / 1.32 / 1.33× |
| 513x449x385 n=1 | 2402 | 1953 | 1.23× |

Multiaxis back guard: 593 → 596 ms (flat, memory identical).

**Back composition (`mt_back_kernel_ab.py`): cone's lesson replicates in BOTH geometries.**
The stacked horizontal-fan gather wins in isolation (multiaxis 0.64–0.65×, translation
0.78–0.83×, measured at psf_radius 1 AND 3) but substituted into the FULL kernels it changes
nothing (1.00–1.02×, values identical) — the gather hides behind the vertical-fan band work.
`back_stacked_gather` therefore stays deliberately unset for both (documented in the
policies).  Three geometries now confirm the rule: measure compositions, not pieces.

**The translation investigation — the channel-collision cliff.**  The harness cells were
mixed (15x256 won 1.20×; 15x257 a REAL, repeat-verified 0.86×; 15x65 neutral), and the
hypothesis chain ran: ragged-pixel-batch → REFUTED (an evenly-dividing pixel batch changed
nothing, `pixel_count_crossover_ab.py` Part 2); detector-size sweep at square detectors
256..1025 → sorted wins 0.65–0.90× at powers of 2, ties at odd sizes; then the decisive
probe at the REAL TCT shapes (`demo_TCT_simulation`: 1936×3064; the phantom's 1883 rows):
**sorted is 4.5–6.5× SLOWER**.  The controlling variable is the mean channel-collision
count `psf_width · num_pixels / num_det_channels`: the scatter's cost IS duplicate-channel
collisions, so at translation's few-views/wide-detector shape (~2 hits per channel) there
is nothing for the sort to win back, and XLA's near-empty segment-sum lowering is a cliff
(all parallel/cone/multiaxis wins sit at ratio ≥ 6).  Policy: translation's
`_select_tile_policy` deliberately does not set `sort_by_channel` (the branch remains, for
experiments); the 0.4 ms cell win at 15x256 is forfeited in favor of the production shapes.

**Two new shared-reduce guard constants** (projectors.py), from the same investigation:

- `SORTED_CHANNEL_REDUCE_MAX_PSF_RADIUS = 2` — the sorted reduce also inverts at wide psf
  (`translation_fwd_psf_ab.py`, 256-column reduce, full forward kernel: 1.27× at psf_width
  3, 1.02× at 5, **0.85× at 7**; compiled temps flat, so compute, not memory).  Multiaxis'
  shared radius can widen via elevation, so its policy gates on this.
- `SORTED_CHANNEL_REDUCE_MIN_COLLISION_RATIO = 4` — splits the measured bracket (ratio 2
  loses 4.5–6.5×, ratio 6 wins 0.73–0.90×).  Multiaxis' policy gates on it (protects the
  unmeasured wide-detector regime).  Follow-up: parallel/cone policies predate this
  constant (their measured cells all sit at ratio ≥ 6); add the guard there if very wide
  detectors with modest pixel batches become a real configuration.

## The DRY horizontal-fan refactor (2026-07-08, branch `greg/kernel_investigation`)

With all four geometries sharing the same trapezoid-rule fan structure, the duplicated tap
loops and their kernel-algorithm branches (8 sites) were consolidated into two helpers in
projectors.py: `horizontal_fan_project` (forward: scatter loop | sorted reduce, chosen by
`sort_by_channel`) and `horizontal_fan_back` (adjoint: gather+FMA loop | stacked gather,
chosen by `back_stacked_gather`).  The geometry enters only through `(n_p, n_p_center,
W_p_c)` plus a per-geometry `weight_scale` (scalar or per-pixel) — for parallel / cone /
multiaxis the scale was already a subexpression, so precomputing it reorders nothing;
translation's `dvr * L / cos` became `(dvr / cos) * L`, a deliberate accepted ULP-class
reassociation.  A side effect: every geometry's back kernel now HONORS `back_stacked_gather`
(previously only parallel's had the branch); the policies still enable it only for parallel
(measured composition no-op elsewhere), and a kernel-equality test pins the branch for
multiaxis (scalar weights) and translation (per-pixel weights, psf_radius 3).

**Vertical-fan twins (follow-up, same day):** the cone and translation banded back
vertical fans were word-for-word twins (same L / cos_alpha tap loop, padded-slice zeroing);
they now share `projectors.vertical_fan_band_gather` VERBATIM — bitwise-preserving, no ULP
license needed.  Multiaxis' vertical fan deliberately stays its own (structurally different:
pure-L weights, mass-conserving amplitude applied post-loop — documented at the function).
The forward vertical fans remain per-geometry (genuinely divergent batching structures).

Verification (old = the committed rollout 9cfe52e):
- **Value gate (the primary gate, stronger than HLO parity): parallel, cone, and multiaxis
  kernels are BITWISE EQUAL on every production path** (fwd both flag states; back both
  coeff_powers), CPU.  Translation ≤ 1.1e-7 max relative (the reassociation).
- Compiled-HLO diff (metadata + source tables + SSA numbering normalized): parallel
  IDENTICAL; cone identical modulo instruction names; multiaxis same instructions reordered
  within fusions (the commuted multiply — consistent with bitwise-equal outputs);
  translation genuinely changed (the reassociation).
- Full CPU + 4-device sharding suites green; per-geometry fwd+back GPU spot cells A/B'd on
  H100 (old vs new snapshots).

## Phase D: concrete n_p_center inputs — the Class-H rounding-bug fix (2026-07-08)

The horizontal fans' integer channel centers are now computed OUTSIDE the projector
programs (`projectors._jit_compute_scatter_centers`, fed by each geometry's new
`compute_channel_coordinate` — the [0] element of the existing float chain) and passed into
the kernels/drivers as CONCRETE arrays.  This removes the round-inside-vmap/map/scatter
precondition of the known XLA rounding bug for all 8 Class-H sites; the vertical fans'
per-slice rounds (Class V) remain documented accepted risk.  Design and trade discussion:
`plans/bugs_and_artifacts/jax rounding bug/phase_d_design.md`.

Mechanics: kernels gain an `n_p_centers` argument (per view, (num_pixels,) int32); the
drivers thread a (P, V)/(V, P) centers array through the existing batching helpers (forward
pixels-major, back views-major, one batch-sized transpose each); the public wrappers compute
the centers eagerly, with a tracer-guard assert (the concreteness contract) and a HYBRID:
one driver call when the centers array is <= `N_PC_SINGLE_CALL_MAX_BYTES` (256 MB, initial),
else a view-chunk loop sized by the op's view batch (forward concatenates chunk outputs,
back accumulates with a donated add; chunking never slices a multi-device view axis).
Forward and back share ONE center value per (view, pixel) — adjoint at rounding ties by
construction.

Verification:
- **The bug repro on this exact toolchain (jax 0.10.1): T14 still FIRES (0.4995
  antisymmetric signature), every in-jit mitigation still fails (T15a–h), and T15j — the
  exact Phase D pattern — is clean on all 24 batches.**
- **Round-free proof:** the compiled model-level parallel forward AND back programs contain
  ZERO round ops (parallel has no vertical fan, so the confirmed production site's
  precondition is provably gone); cone/multiaxis/translation retain only their Class-V
  vertical-fan rounds (6/12/6 fwd, 1/2/1 back).
- **Value gate vs pre-Phase-D:** parallel, cone, translation kernels BITWISE EQUAL on every
  path; multiaxis fwd bitwise, multiaxis back <= 9e-6 rel on <= 4% of elements —
  discriminated by an hfan-only probe (bitwise equal) to be downstream fusion/FMA context
  in the vertical fan, not center divergence.
- New tests (`tests/test_scatter_centers.py`): centers == round(coordinate) in both
  layouts; chunked == single-call (incl. ragged owned-view tails); the wrapper REFUSES to
  run under an outer jit.  Full CPU + sharding suites green.
- **H100 A/B (final, after the eager-gather fix below):** 1024³ parallel fwd/back −3% with
  memory flat (cone fwd 1024³ −3.2 GB); VCD 200³ neutral-to-better (2092 → 1950 ms);
  513-class fwd flat.  Accepted costs: mid-size fwd cells +~0.5 GB (materialized centers +
  chunk-output concat — memory-gate ack, same class as the Phase B acks) and 513-class back
  +8% (the chunked accumulation); translation ms-cells within their ±0.5 ms band.
- **The eager-gather episode (a lesson now in .claude/lessons.md §3):** the first GPU round
  showed VCD +35%.  The culprit — found by cProfile after a device trace showed the cell is
  ~95% HOST-bound — was ONE eager `view_params[asarray(owned)]` gather per projector call in
  the new wrappers (~1 ms host each, 547×/recon); every kernel-level probe was flat because
  the micro-bench used the empty-default owned path.  Fix: owned_view_indices passes THROUGH
  to the jitted programs (in-jit gathers, as the drivers always did); the wrappers now carry
  an explicit no-eager-array-ops contract.

## Numerics note (ties to the known JAX rounding bug)

Early full-grid runs showed two *value-equal* compiled programs (`fwd_asis` vs the stacked
variants) differing by ~4e-3 relative on isolated elements: 2 of 32 views, channels `c−1`/`c+1`
around a rounding tie, `c` itself clean, ~0.3–0.5 magnitude per affected cluster.  That is the
**antisymmetric ±1-channel signature of the known rounding bug**
(`plans/bugs_and_artifacts/jax rounding bug/jax_rounding_bug.md`): the projection is
mathematically continuous at exact .5 ties (PSF clipping), so a compilation-dependent value
change means the scatter destination `n` and the weight's `|n_p − n|` disagreed about
`n_p_center` — the same family as the vmap→map→scatter mis-optimization documented there, here
surfaced by comparing two differently-optimized programs.  Consistent with that bug's known
input sensitivity, the effect vanished at production shapes (frac > 1e-4 = 0 everywhere).
The planned host-side precompute of `n_p_center` (that doc, §3–4) removes the ambiguity at the
root and would also let the sort permutation be precomputed/cached per (view, pixel batch).
The bench therefore checks values with a fraction-deviating metric, never max-error.
