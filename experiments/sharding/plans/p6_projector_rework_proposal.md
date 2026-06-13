# P6 projector rework — scoping proposal (2026-06-12; REVIEWED with Greg, cone-first)

*The design that the cone port implements first, then translation/multiaxis adopt.
Inputs: the (g0,L) design note + anchor rule + forward-accumulation note (v2 §P3),
`.claude/back_projection_overview.md`, and a fresh read of the parallel/cone/
translation/multiaxis kernels.*

**Review status (2026-06-12).**  Reviewed with Greg; decisions folded in below.
Settled: **cone first**, land it well before the other geometries (§8); **row
window from day one** (§2); **anchor rule from params** (§3); cone padding
**exactly inert** via the global validity clip (§5); delete
`entries_per_cylinder_batch` (§6); de-closure with the **two top-level signatures
the only preserved surface** — all other assembly code is a simplification target
(§7).  **Unified assembly direction (§4):** drive forward-band assembly through ONE
scatter-add accumulator for every geometry (parallel = degenerate disjoint window),
deleting the per-geometry hook IF an A/B ablation shows parallel does not regress;
the `_forward_band_assembly` flag is the fallback.  **Provisional (Greg, to watch):**
the kernel returns a band CONTRIBUTION and assembly resolves overlap by add (§3-overlap)
— accepted for now; the overlap size scales with cone angle, so if cone deviates badly
from parallel under a parameter sweep we revisit (a stateful donated-accumulator kernel
is the held-in-reserve alternative).  **Channel-major cone kernels** carried over from
parallel's cache fix, A/B-measured (§5-transpose).

---

## 0. Goal and scope

One template, built for and proven on **cone beam first**, then adopted by
translation and multiaxis.  Six interlocking changes, designed together because the
cone port touches all of them at once:

1. The banded `(g0, L)` kernel interface, uniform across geometries.
2. The **detector-row window** — the companion to (1) that keeps per-band work ∝
   band (without it the cone port is correct but unacceptably slow); built in from
   day one and, per §4, likely the same plumbing for every geometry.
3. The anchor rule (kernel coordinates from params + global indices).
4. Forward-band assembly via a single scatter-add accumulator (unified across
   geometries pending the parallel A/B); overlap resolved by add.
5. Deletion of `entries_per_cylinder_batch` (subsumed by the outer band loop).
6. De-closuring: module-level jitted projector drivers shared across model
   instances, one traced program per shape (tail batches padded), `view_indices`
   removed from the hot path.

**Cone-first staging discipline (§8).**  The new module-level banded drivers are
built *as part of* the cone port (cone is their first adopter).  ParallelBeam keeps
its existing closure-based row-crop drivers until cone lands and validates, behind a
transitional "geometry has banded kernels?" branch tagged RETIRE-AFTER.  Once cone
is validated, parallel converts and the old drivers are deleted promptly (the F1
delete-the-loser discipline — do not let two driver stacks linger).  Consequences
accepted: that transitional branch, and the trace-sharing win not being
suite-wide-measurable until parallel converts.

Out of scope here: the retirement cascade itself (P6 step 4) and the docs page —
this design just has to not block them.

---

## 1. The banded kernel interface

Each geometry replaces its two kernels with banded forms.  Shared conventions:
`g0` is a **traced** global slice index; `L` is **static** (≤2 values from
`_balanced_slice_bounds`, so ≤2 programs); the sinogram view passed to the back
kernel is the **full resident view** (all rows — a view-owner always holds every
row of its own views); each geometry maps the band to the rows it needs
*internally*.

**Back** (per view): `back_project_one_view_to_band(sinogram_view, pixel_indices,
single_view_params, projector_params, g0, num_band_slices, coeff_power)` →
`(num_pixels, L)`, exactly global slices `[g0, g0+L)`.

**Forward** (per view): `forward_project_band_to_one_view(band_values,
pixel_indices, single_view_params, projector_params, g0)` where `band_values` is
`(num_pixels, L)` (L static via shape).  **Returns a band CONTRIBUTION**, not a full
view: the window rows `(W, C)` plus the traced offset `r0` (the rows the band
projects into — see §2).  The kernel is **pure** (band in → contribution out); the
caller's assembly resolves band-to-band overlap by ADD (§4 / overlap note below).
Parallel is the degenerate case (`r0 = g0`, `W = L`, no margin, disjoint windows).

Per-geometry internals:

| geometry | back: band → rows | forward: band → window |
|---|---|---|
| parallel | `dynamic_slice` rows `[g0:g0+L)` of the view (replaces today's *external* row-crop and the kernel's input-length sizing); body unchanged | emits rows `[g0:g0+L)` → returns `(L, C)` at `r0=g0`; windows are disjoint so the add degenerates to writes |
| cone, translation (gather-form vertical fan) | vertical fan outputs global slices `k = g0 + arange(L)`, gathering from the row **window** (§2); horizontal fan runs on the window only | vertical fan scatters the band's slices into the row window; horizontal fan produces the window's `(W, C)` rows; returned with `r0` |
| multiaxis (scatter-form) | same window structure | its slice loop runs `k = g0 + local`, scattering into the window |

**Overlap (the provisional point, Greg to watch).**  Adjacent forward bands project
into row windows that OVERLAP (by the psf blur + the window margin; the overlap grows
with cone angle).  Resolution: the kernel returns only its band's contribution, and
assembly does `acc[:, r0:r0+W, :] += window` — overlapping rows from adjacent bands
simply both add, which is exactly correct because forward projection is linear in the
slices.  So the kernel never takes an accumulating view as input (keeps it pure and
keeps the adjoint round-trip gate clean).  **Accepted for now**; if a cone parameter
sweep shows large deviation from parallel (assembly-add cost or memory), the
held-in-reserve alternative is a stateful kernel taking a donated accumulator and
adding internally (fuses the pass, at the cost of a stateful, harder-to-test kernel).

The generic driver change: `_back_project_local_views_to_band` passes `(g0, L)`
instead of row-cropping (`data[:, g0:g1, :]` disappears from tomography_model);
`_forward_project_all_bands` calls the unified scatter-add assembly (§4).  Back
assembly (concat of bands per slice-owner) is output-side banding and stays valid
for every geometry — unchanged (the band sweep already settled back's assembly).

---

## 2. The detector-row window (the load-bearing cone decision)

**The cost problem.**  Today's sharded band loop hands the kernel the full view.
Parallel beam's row-crop makes per-band work ∝ L, so total work stays ∝ S.  For
cone, the horizontal fan runs over **all** detector rows and is band-independent —
naive banding recomputes it once per band.  Bands per recon: ~n_dev² multi-device,
and ~S/L single-device under the compute cap once the single-device path unifies
onto band-looping (1024³: num_pixels ≈ 1M ⇒ L ≈ 95 ⇒ ~11 bands).  The horizontal
and vertical fans are comparable cost (both ~num_pixels × extent × psf), so naive
banding multiplies roughly half the kernel cost by the band count — a 5–15×
back-projection regression.  Banding cone without restricting rows is not viable.

**The fix.**  A band of slices `[g0, g0+L)` projects, for one view, into a
bounded detector-row window `[r0, r0+W)`:

- `r0` — **traced**, computed in-kernel from `g0`, the view params (helical
  z-shift enters here, so per-view shifts cost no recompile), and the volume-wide
  magnification bounds.
- `W` — **static**, computed at `get_psf_radius()` time from the worst-case
  (volume-wide) magnification already derived there:
  `W ≈ ceil(L · max_magnification · Δ_slice/Δ_det_row) + 2·psf_margin + 1`.
  One W per L value ⇒ still ≤2 programs.
  *(The existing `slice_range_length` — computed, shipped in geometry params, and
  consumed by NOTHING — is exactly this quantity sized for
  `entries_per_cylinder_batch`; it gets re-derived for L and finally used, or
  retired.)*

Kernel flow (back): `dynamic_slice` rows `[r0, r0+W)` → horizontal fan on
`(W, C)` → `(P, W)` det window → vertical fan gathers within the window, with
validity masks doing the *precise* work (row in-window ∩ `[0, R)`; slice
`k < S_real`).  The window is only a conservative superset — correctness never
depends on its tightness.  Forward is the transpose: vertical fan scatters into a
`(P, W)` window, horizontal fan → `(W, C)`, returned with `r0` for assembly.

Conservatism: volume-wide max/min magnification makes W loose by roughly the
mag ratio `(D+w)/(D−w)` — typically tens of percent of L, plus the psf margin.
That bounds the wasted work at far below the naive blowup; per-pixel-batch
tightening is a later optimization if it ever shows up.

**Decided: build the window in from day one.**  The naive kernel is not enough
simpler to justify a throwaway, and its measurements would be discarded.
Validation handles the risk: implement the window as an in-kernel choice with a
"full rows" setting (`r0=0, W=R`) so window-vs-full-rows is a cheap
single-variable ablation, gated by tight allclose.

**One bit of plumbing for everything (Greg).**  The window machinery (traced `r0`,
static `W`, the `dynamic_slice`/scatter at `r0`) is the SAME for back and forward
and for every geometry; parallel beam is the degenerate window (`r0=g0`, `W=L`).
So the intent is one shared window/assembly code path, not a cone-only branch —
this is the same unification as §4.  The cost to check is whether the degenerate
window adds measurable overhead to parallel beam vs. its current row-crop; that is
exactly the parallel A/B in §4 (and the window's "full rows" toggle gives the
ablation knob).  If parallel regresses unacceptably, parallel keeps its row-crop
and the window stays the z-based path — but the expectation is one path.

**Translation caveat (record now, decide at the translation port).**  W from a
volume-wide worst-case magnification assumes magnification does not vary wildly
across the recon.  For the **translation model** the per-view translation changes
the source-object distance, so magnification can vary a lot and a single
worst-case W could be very loose (much wasted window).  Bounded escape if it bites:
a few per-view-bucket W values (each static ⇒ a handful of programs), NOT
per-pixel tightening.  This is a translation-port decision with its own
measurement; cone and multiaxis are fine with one W.

---

## 3. Anchor rule (mechanical, bit-exact today)

Kernel physical coordinates come from PROBLEM shapes + GLOBAL indices; input
lengths size loops/allocations only:
`z(k_global) = Δ_slice · (k_global − (S_real − 1)/2) + shift`, with
`S_real = projector_params.recon_shape[2]` and `k_global = g0 + k_local`.

Sites (all currently derive the center from the input cylinder length — a latent
identity that both banding and padding break the same way):
- `cone_beam.py` forward vertical fan: 390→402 (z from `voxel_cylinder.shape[0]`),
  434 (`k_m` recentering), 448 (validity clip → global `S_real`, see §5).
- `translation_model.py` forward vertical fan: 315/360, clip 374.
- `multiaxis_parallel.py` forward vertical fan: 207/225, clips 247/254.
- Back-projection vertical-fan data (`compute_vertical_data_single_pixel` in cone
  and translation) already anchors from params `recon_shape` — its
  `slice_indices` argument just becomes `g0 + arange(L)`; nothing else changes.

Corollary (recorded at P5, restated as a port invariant): padded or
device-count-dependent shapes never enter `projector_params` or any jit
closure/static.

---

## 4. Forward-band assembly — one unified scatter-add accumulator

Forward banding is input-side, so a band does NOT own a fixed row range of the
output: view-owners must SUM per-band contributions.  **Direction (reviewed):
unify on ONE assembly path for every geometry** instead of a concat/accumulate
split.

- **Unified accumulator.**  Each view-owner allocates its `(V_local, R, C)`
  view-shard accumulator once (zeros — this is just its output shard, already in
  the memory budget), and per band scatter-adds the kernel's `(V_local, W, C)`
  window at the per-view `r0` under a **donated jit** (`donate_argnums` on the
  accumulator).  Donation makes the per-band update in-place — no per-band
  full-shard copy.  These are local single-device arrays, so they free on
  refcount (no NamedSharding reference-cycle exposure; the donation is for the
  copy, not the leak).  `r0` varies per view (helical/translation z-shift), so
  the update is vmapped over the view batch with per-view offsets.
- **Parallel is the degenerate case**: `r0 = g0`, `W = L`, windows disjoint, so
  the scatter-add reduces to plain writes onto zeros — no double-counting, and it
  *replaces* today's concat path, which holds the full row-band list AND allocates
  the concatenated output on top (a ~2× transient at assembly).  So unification
  plausibly *improves* parallel forward memory while deleting a code path.
- **The gate (the A/B Greg asked for).**  Before committing to unification, run
  the single-variable ablation on PARALLEL forward: concat (today) vs. scatter-add
  accumulator, two sizes × {1,2,4} devices, time + `peak_bytes_in_use`, tight
  allclose on output.  If scatter-add is in the noise or better → unify and delete
  the concat path.  If concat genuinely wins on parallel → keep a
  `_forward_band_assembly = 'concat' | 'accumulate'` flag (concat for parallel,
  accumulate for z-based) as the documented fallback.  Either way the padded-view
  mask stays POST-assembly (already designed to survive this).
- **Cone needs NO row padding** (rows aren't tied to slices; `_sino_row_padding`
  stays ParallelBeam-only).  Forward inertness of padded slices is free: their
  values are identically zero (forced-zero invariant) and a zero band contributes
  a zero window.

---

## 5. Padding semantics for cone: EXACTLY INERT, via the global validity clip

Decision recorded in the design note as open ("inert vs enlarge-the-volume";
clip global vs band-local).  **Recommendation: exactly inert, with the in-kernel
validity clip in GLOBAL terms (`k_global < S_real`).**

- Inert keeps the P5 invariant and its N-independence proof intact; "enlarge the
  volume" would change the reconstruction problem with the device count — exactly
  what "padding must be exactly inert" exists to prevent.
- The global clip makes it free in both directions: back projection simply never
  computes a padded slice's value (the gather mask kills it), and forward
  projection rejects padded-slice contributions even if the forced-zero invariant
  were ever violated.  `_mask_padded_slices` / the forward mask remain as the
  one-site defensive postconditions (cheap, already landed, geometry-blind).
- The padded-slice exact-zero tests (constructed-zero invariants — exact equality
  is correct there) then gate cone identically to parallel.

---

## 5b. Channel-major (transposed) cone horizontal fans

ParallelBeam's kernels were rewritten to operate on the **transpose** of the view
(channel-major: `(channels, slices)` forward, `(channels, rows)` back) so the
per-pixel channel scatter/gather hits a CONTIGUOUS row (stride 1) instead of a
column of stride `num_det_channels`; a power-of-2 channel count otherwise aliases
the CPU cache and runs several-× slower at large slice counts (lessons / the
parallel kernel comments).

Cone's horizontal fans have the SAME exposure and have NOT been fixed:
- forward `forward_horizontal_fan_pixel_batch_to_one_view`:
  `sinogram_view.at[:, n].add(...)` — stride-`num_det_channels` scatter
  (cone_beam.py ~356).
- back `back_horizontal_fan_one_view_to_pixel_batch`: `sinogram_view[:, n].T`
  gather — same stride (cone_beam.py ~529).

Since the banded port rewrites these fans anyway (the horizontal fan now runs on
the `(W, C)` window), **carry the channel-major layout into cone from the start**
and A/B it at the KERNEL level on CPU (where the aliasing bites), with a GPU sanity
check.  The vertical-fan accesses are already contiguous (per-pixel detector
columns), so the horizontal fans are the whole story.  Translation/multiaxis
horizontal fans get the same treatment at their ports.

---

## 6. `entries_per_cylinder_batch` deletion

The outer band loop subsumes the internal chunking once single-device also drives
band-looping (the trivial-placement unification, already true for ParallelBeam):

- cone back `create_voxel_cylinder_slices` (lax.map over slice chunks) — gone;
  the band IS the chunk.
- cone forward `create_det_column_rows` (lax.map over det-row chunks) — replaced
  by the row window (W is band-sized; no internal chunking needed).
- translation + multiaxis equivalents — same.
- The attribute, its geometry-params entries (cone/translation/multiaxis), and
  `slice_range_length` (cone) all retire; `get_psf_radius` keeps computing the
  magnification bounds (now consumed by W).

Memory audit: per-band working set = `(P_batch, W)` det window +
`(P_batch, L)` band output, times the view-batch factor — bounded by the existing
band sizing (`_slice_band_length` compute cap makes one device stream), same
structure as parallel beam.  Single-device cone memory is then bounded by the
band loop where it used to be bounded by the internal chunking — same knob, one
layer up, shared with every geometry.

---

## 7. De-closuring: module-level jitted drivers

Today `create_projectors` rebuilds and re-jits four closures per model instance
(closing over the kernels, `projector_params`, batch sizes), so every fresh model
re-traces byte-identical programs — measured: tracing/lowering, not XLA compile,
dominates first-call cost at test sizes.

- Lift `sparse_forward_project_fcn` / `sparse_forward_project_pixel_batch` /
  `sparse_back_project_fcn` / `sparse_back_project_view_batch` to **module-level
  functions** with explicit arguments.  Static args: the geometry kernel pair
  (module-level static methods — hashable by identity), `projector_params`
  (hashable namedtuple — precedent: it is already `static_argnames` on the
  per-view kernel jits), batch sizes, `coeff_power`, `L`.  Traced: view params
  array, data, `g0`.  The jit cache then keys on (kernel, params, shapes) and is
  shared across model INSTANCES — sweeps and the vcls 1-view sibling stop
  retracing; with the legacy single-device bodies retired, roughly half the
  per-model programs disappear outright.
- `Projectors` shrinks to a thin holder: the current `view_params_array` + public
  partials binding the statics.
- **`view_indices` leaves the hot path** (its P6 retirement enabler): the
  module-level drivers take the view-params array FOR THE VIEWS BEING PROJECTED
  as a traced argument — the full array single-device, the view-owner's local
  slice in the sharded path.  No index indirection inside the jit; the
  padded-view clamp moves to where the local slice is built (host-side, cheap).
- **Tail batches → one signature**: instead of the current
  initial-partial-batch + scan (two traced programs when a batch doesn't divide),
  pad to full batches and scan over all of them.  Pixels: pad `pixel_indices`
  (repeat index 0) with zero `voxel_values` — zero forward contribution; back
  outputs for padded pixels are garbage and cropped.  Views (back): pad with zero
  views — zero contribution to the sum.  Crop after.  One program per shape
  bucket.
- **GPU item (flagged; cluster measurement the plan already calls for):**
  instrument first-call overhead (trace + lower + compile, separately) at
  production size before/alongside this work, so the win is sized honestly and a
  regression in either component is visible.

---

## 8. Staging — CONE FIRST (each stage lands CPU-green for review)

Cone is where all the design risk lives (window, accumulation, anchor), so we
confront it first and land it WELL before touching the other geometries.  The new
module-level banded drivers are built as cone's drivers; ParallelBeam keeps its
existing closure-based row-crop drivers behind a transitional branch until cone
validates, then converts and the old drivers are deleted (don't let two stacks
linger).

0. **Baselines first** (§8a): capture cone single-device time + memory on CURRENT
   code, CPU (local) and GPU (cluster), recorded as a table in this doc.  These are
   the "no regression (not literal)" reference.
A. **Parallel forward assembly A/B** (§4 gate): concat vs scatter-add accumulator
   on ParallelBeam, the window's "full rows" toggle as the knob.  Decides whether
   §4/§2 unify into one path or keep the `_forward_band_assembly` fallback.  Cheap,
   isolates the one parallel-perf question before cone work starts.
B. **Module-level banded drivers + cone banded kernels** — the (g0,L) interface,
   the row window (with the full-rows toggle), the anchor rule, the global clip,
   the unified scatter-add assembly, channel-major cone horizontal fans, and
   `entries_per_cylinder_batch` deletion (cone only).  `_supports_sharding()=True`
   for cone.  Transitional branch: sharded orchestration routes
   "geometry has banded kernels?" to the new driver, else the existing parallel
   row-crop path (tagged RETIRE-AFTER).  CPU-green; then GPU validation at scale
   (stage2 pattern, cone edition) — including the scaling check that the
   horizontal-fan blowup is absent and the window-vs-full-rows ablation.
C. **Parallel conversion + old-driver deletion** — once cone is validated, port
   ParallelBeam onto the banded drivers (degenerate window), delete the closure
   drivers and the transitional branch, and confirm trace-sharing across model
   instances (program-cache hits) + full suite green.  Behavior gate: tight
   allclose vs the recorded parallel behavior (never exact — project rule).
D. **Translation port**, then **multiaxis** (its own `_supports_sharding` flip —
   it extends TomographyModel directly); translation may need per-view-bucket W
   (§2 caveat) — decide with its own measurement.
E. **Retirement cascade** (v2 §P6 step 4 — only after D): legacy single-device
   bodies, `main_device`/`sinogram_device`, `view_indices` machinery,
   `initialize_recon` early device_put, `compute_hessian_diagonal`'s
   `output_device`, RETIRE-AFTER sweep.

Validation per port: the **(g0,L) adjoint round-trip at arbitrary bands** (the
strong per-geometry gate), banded-vs-baseline full recon at 1e-5/1e-4 allclose,
window-vs-full-rows ablation (cone), padded exact-zero invariants, full suite.

---

## 8a. Baseline measurements (no-regression reference)

Goal (Greg): no regression, judged not literal.  From ParallelBeam we expect good
per-device and per-size scaling and a 1-device-mesh case comparable-to-or-better
than the existing single-device path; cone differs (the window, accumulation,
channel-major), so measure carefully and evaluate as we go.

Tooling (AS BUILT, 2026-06-13): a thin dedicated driver
`scaling_tests/cone_baseline_scaling.py` that REUSES `scaling_common`'s
measurement primitives (isolated-subprocess harness, `time_op` warmup/trials,
free-previous-result discipline, `peak_memory_mb`, throttle sampling, YAML) rather
than editing the three device-sweep drivers.  Rationale: cone is single-device-only
on the current code, so the sweep drivers' sharding machinery (configure_sharding,
pre-shard, reduce-scatter, device-forcing, YAML naming) is irrelevant and threading
a geometry knob through all three risks the WORKING parallel harness for no
measurement benefit.  The dedicated driver keeps identical timing discipline and
makes the pre-port-vs-post-port cone comparison apples-to-apples (both from this one
tool); `GEOMETRY='parallel'` reproduces the parallel single-device reference with
the same discipline.  Measures forward / back / vcd_const / vcd_nonc, one fresh
worker per (op, size).

- **Pre-port, CURRENT code — cone single-device baselines**: forward, back, VCD
  (const + non-const weights); FDK optional.  CPU locally 64³–256³; GPU on the
  cluster 256³–1024³ (Greg runs — GPU item).  Capture time + `peak_bytes_in_use`,
  one subprocess per config, JAX-free orchestrator, free-previous-result, dmon
  preflight, fresh `pip install -e .`.  YAMLs in gitignored `results/`; record the
  numbers as a TABLE here (binaries don't go on github).
- **Post-port gates**: trivial-1-device-mesh vs the recorded baseline
  (comparable-or-better, not literal); multi-device sweep n_dev∈{1,2,4} × two
  sizes expecting parallel-like behavior (memory ~1/n_dev; time scaling at least
  the projectors' measured ceilings); NRMSE vs single-device ≤1e-4 iterated.
- **Discipline**: same-allocation pairing for any GPU before/after (thermal +
  hardware variance); fold the first-call trace/lower/compile instrumentation
  (§7) into the same runs so the de-closuring win is sized on the same ruler.

### 8a-results — CPU baseline captured 2026-06-13 (M3 Max, single device)

Geometry: cone, magnification 2 (sdd = 4·channels).  Recon shape auto-derived =
sinogram size here (NxNxN).  15 VCD iterations.  min over 3 trials (1 warmup);
RSS (process-level, CPU).  YAML: `results/cone_baseline_cone_cpu.yaml`.

| size (sino=recon) | forward ms | back ms | vcd_const ms | vcd_nonc ms | RSS@vcd MB |
|---|---|---|---|---|---|
| 64³  |     41.8 |     15.1 |    5 490 |    8 148 | ~1 456 |
| 128³ |  1 011.1 |    197.5 |   35 424 |   68 551 | ~4 087 |
| 256³ | 42 637.3 | 22 812.2 | 1 139 239† | 1 368 561 | ~5 908 |

All cells min/mean spread ≤0.2% (rock-stable) EXCEPT †vcd_const 256³ =
**16% spread, std 129 s — thermal-throttle-contaminated** (a ~19-min sustained
single recon on a laptop; the CPU throttle sampler is GPU-only so this is not
auto-flagged).  forward/back 256³ are clean (std ≤0.2%), so the projector cliff
below is real, not throttle.

**Findings (these MOTIVATE the port, and become the post-port gate):**
1. **Super-N⁴ projector scaling on CPU, with a sharp BACK cliff at 256³.**  Ideal
   doubling = ×16 (time ∝ N⁴ = voxels·views).  Measured: forward ×24.2 then ×42.2
   (exponent 4.6→5.4); **back ×13.1 then ×115.5** (exponent 3.7→**6.85**).  Back's
   ×115 jump from 128³→256³ is reproducible (std 0.03%), so it is a real
   working-set/cache cliff — almost certainly the strided channel access
   (`sinogram_view[:, n]`, stride = num_det_channels, power-of-2 → cache aliasing)
   that parallel beam already fixed and cone has NOT (§5b) plus the unbounded
   single-device working set (§6).  The channel-major rewrite + band-streaming the
   working set should flatten this; the 256³ back number is the headline
   pre-port datum to beat.
2. **vcd_nonc > vcd_const** (weighted error-sino path): ×1.48 / ×1.93 / ×1.20 — as
   expected; both dominated by the projectors.
3. **VCD at 15 iters is an impractical CPU ruler at ≥256³** (~19–23 min/recon,
   throttle-prone).  Per-iteration cost is the comparable quantity, so for the
   POST-port CPU comparison: drop VCD to ~3 iterations and push ≥256³ VCD to the
   GPU (fast + throttle-detected).  forward/back stay the primary cone-specific CPU
   baseline (the VCD loop itself is geometry-independent — test-suite redesign
   rationale).
4. **Memory** (CPU RSS, process-level, approximate): forward 256³ 2.9 GB; VCD 256³
   5.9 GB.  GPU `peak_bytes_in_use` (the real per-device ruler) is the cluster run.

GPU baseline (256³–1024³, `peak_bytes_in_use`, throttle-checked): **Greg, cluster**
— same driver, `GEOMETRY='cone'`, sizes already wired in the config block.

---

## 9. Risks / things to verify at the port

- **Curved detector** (cone): the row mapping v→m is linear even when curved
  (curvature affects u/channels), so the window math should hold — verify
  against `geometry_xyz_to_uv_mag` / `detector_uv_to_mn` before trusting W.
- **Multiaxis elevation ≈ 90°** (top-down): `slope_k_to_m → 0`, all slices land
  in a few rows — window tiny, fine; but W must use |cos(elevation)| bounds over
  the actual view params.  Translation's per-view magnification likewise enters
  its W bound.
- **Helical**: z-shift enters `r0` (traced) — confirm no other per-view
  quantity wants to be static.
- **Forward overlap-add (the provisional point):** the scatter-add per band over
  the view-shard adds one pass; overlap between adjacent windows grows with cone
  angle.  Expected in the noise (band count small, add bandwidth-trivial next to
  projection) — but this is the thing to WATCH across a cone-angle / parameter
  sweep.  Large deviation from parallel ⇒ revisit (stateful donated-accumulator
  kernel held in reserve).
- The `(g0,L)` parallel-beam conversion replaces an *external* crop that was
  exact; in-kernel `dynamic_slice` should compile identically on CPU, but gates
  stay allclose regardless.

---

## 10. Decisions (resolved with Greg 2026-06-12) + what's left to measure

Resolved:
1. **Cone first**, land well before other geometries.  ✅
2. **Row window from day one.**  ✅
3. **W from volume-wide worst-case magnification** for cone/multiaxis; translation
   may need per-view-bucket W (decide at its port, §2 caveat).  ✅
4. **Forward kernel returns a band contribution `(r0, window)`**; assembly resolves
   overlap by ADD; kernel stays pure.  Provisional — watch cone-angle sweeps. ✅
5. **One unified assembly path** intended (scatter-add accumulator, parallel =
   degenerate window).  `_forward_band_assembly` flag is the FALLBACK only if the
   parallel A/B (stage A) shows concat genuinely wins.  ✅
6. **De-closure preserves only the two top-level signatures**
   (`sparse_forward_project` / `sparse_back_project`).  Everything else in
   projectors.py assembly — including the `forward_project_pixel_batch` /
   `back_project_view_batch` exposed attributes — is a simplification target, not
   preserved verbatim.  ✅

Left to MEASURE (the "evaluate as we go" items):
- Stage 0 cone baselines (CPU + GPU) — the no-regression reference table (§8a).
- Stage A parallel forward concat-vs-scatter-add A/B — decides unify vs flag (§4).
- Cone window-vs-full-rows ablation + horizontal-fan-blowup-absent scaling check.
- Channel-major cone horizontal fans A/B (CPU, §5b).
- First-call trace/lower/compile instrumentation at production size (§7, GPU).
