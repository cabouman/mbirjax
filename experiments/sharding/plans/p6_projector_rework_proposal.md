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

## 1. The banded projector interface (data-driven — canonical statement in §8a-design)

**This section was rewritten 2026-06-13 after the CPU+GPU fan-split measurements
(§8a-split).**  The original design (a detector-row WINDOW + a forward CONTRIBUTION
return + an optional fused/sino-accumulator) was *measured out*: on GPU the
horizontal fan dominates BOTH directions and the row window would recompute it per
band (~2× the dominant stage), while fusion showed no time win.  The window is
demoted to a "considered, measured out" note (§2); what remains is the durable part.

The cone port replaces each geometry's monolithic per-view kernel with a **two-stage
decomposition** that bands the VERTICAL (slice) fan and computes the HORIZONTAL
(channel) fan ONCE per view (Option B):

- **Horizontal fan — once per view, channel-major (§5b).**  The dominant GPU stage
  (the channel scatter/gather).  Back: sino `(rows×channels)` → det cylinder
  `(pixels×rows)` (a channel gather).  Forward: det cylinder `(pixels×rows)` → view
  `(rows×channels)` (a channel scatter).  Pixel-batched INTERNALLY
  (`pixel_batch_size`, `projectors.py` — see the layer note below) so the
  `pixels×rows` transient is bounded — **no row window needed**.
- **Vertical fan — banded by global slice `(g0, L)`.**  `g0` traced, `L` static
  (≤2 values from `_balanced_slice_bounds` → ≤2 programs).  Back: det cylinder →
  `(pixels, L)` for global slices `[g0, g0+L)`.  Forward: a `(pixels, L)` band →
  its contribution to the view, ACCUMULATED across bands (§4 — input-side banding
  needs a sum).  Global slice `k = g0 + k_local`; the physical-coordinate anchor
  comes from PARAMS (§3), never from the band length.

Banding the vertical fan is exactly what the multi-device reduce-scatter needs (the
recon is slice-sharded): each view-owner computes its horizontal det cylinder once,
then per slice-band runs the vertical fan → `(pixels, L)` partial → summed onto the
slice-owner.  Parallel beam is the degenerate case (vertical fan = the slice=row
identity; its "band" is the existing row-crop).

**Batching layers (the distinction Greg flagged):** *slices* are banded OUTSIDE, in
TomographyModel's sharded driver (`_back_project_all_bands`), for the reduce-scatter;
*pixels* are batched INSIDE the projector functions (`pixel_batch_size_for_vmap`, via
`concatenate_function_in_batches`/`sum_function_in_batches` in `projectors.py`),
which is what bounds the `pixels×rows` horizontal transient to
`pixel_batch_size × num_det_rows`.  Option B leans on the existing inside knob — no
new TomographyModel-level pixel loop (the legacy `transfer_pixel_batch_size` host
loop retires with the legacy single-device path).

**Implementation note (why this is a kernel restructure, not a flag).**  The per-view
kernel is vmapped by the shared generic layer (`projectors.py`
`sparse_back_project_view_batch`), so splitting horizontal-once from banded-vertical
changes the generic projector call structure and every geometry's kernel
decomposition.  Build it on cone first (§8 staging); parallel keeps its row-crop
band until it converts.

---

## 2. Detector-row window — CONSIDERED, MEASURED OUT (2026-06-13)

The original design made a per-band detector-row window `[r0, r0+W)` the
"load-bearing cone decision" — the premise being that the horizontal fan is
band-independent and expensive, so naive per-band banding would recompute it (a
5–15× regression) and the window restricts it to `W` rows.

**The fan-split measurement (§8a-split) refuted the premise and the fix:**
- The horizontal fan is NOT the back-projection bottleneck the window targeted —
  on CPU it is ~2% of back (vertical dominates); on GPU it is the *dominant* stage
  but the window would RECOMPUTE it per band (~2× the dominant cost).
- So instead of a window, **compute the horizontal fan ONCE per view (Option B)**
  and bound its transient by the existing internal pixel batching (§1).  No window,
  no `r0`/`W` machinery, no per-band horizontal recompute, cone-angle-independent.

What carried over from the window analysis (still true, now elsewhere): the anchor
rule (§3) and the fact that the dead `slice_range_length` (computed, shipped, used
by nothing) is retired with `entries_per_cylinder_batch` (§6).

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

## 4. Forward-band assembly — accumulate (no window)

Forward banding is input-side: a band of input slices `(pixels, L)` does NOT map to
a fixed row range of the output (cone: overlapping, pixel-dependent rows), so the
view-owner must SUM per-band contributions.  With Option B (no window, §1/§2) each
band's forward kernel produces a full view-shaped contribution `(rows, channels)`
for its slices (zero outside the rows those slices reach), and the view-owner
ACCUMULATES across bands:
- `view += forward_band(band_k)` over the bands.  Since the horizontal fan is
  computed inside the per-band forward (channel-major), and the contributions sum,
  no window/`r0` bookkeeping is needed — overlapping rows from adjacent bands simply
  both add (forward is linear in the slices).
- Parallel beam keeps its existing row-concat assembly (the slice=row identity
  makes each row a single producer; no add pass).  The earlier "unify parallel onto
  scatter-add" A/B is DROPPED — the fan-split showed no fusion/restructure time win
  and parallel's concat is already good; not worth disturbing the working path.
- The padded-view mask stays POST-assembly (already designed to survive this).
  **Cone needs NO row padding** (rows aren't tied to slices; `_sino_row_padding`
  stays ParallelBeam-only); a zero (padded) slice contributes a zero band.

(`donate_argnums` on the accumulator is the right in-place mechanism if/when the
band loop is jitted; in the host-level band loop it is a per-band add into the
local view-shard, freed on refcount — no NamedSharding cycle.)

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

**MEASURED (§8a-split, 2026-06-13) — on GPU (the production target) the horizontal
fan dominates BOTH directions, so channel-major is the #1 cone lever for FORWARD
AND BACK.**  GPU H/V: forward ≈ 7→3.4, back ≈ 1.3→2.6 (back-horizontal is the
*larger* stage on GPU — opposite to CPU, where back is vertical-bound and
back-horizontal is ~2%).  So: **apply channel-major to BOTH cone horizontal fans**
(forward scatter + back gather), A/B at the KERNEL level — CPU shows the forward win
sharply (forward-horizontal owns CPU's ×49 cliff); GPU shows both.  The CPU
back-vertical ×62 cliff does NOT reproduce on GPU (cache artifact), so it is not a
channel-layout issue and not a GPU lever.  Translation/multiaxis horizontal fans get
the same treatment at their ports.  **This is the first coding increment** (most
value, best-localized, independently testable via the adjoint identity).

**LANDED (increment A, 2026-06-13, CPU-green).**  Both cone horizontal fans converted
to channel-major (transpose, stride-1 channel access; mirror of ParallelBeam).  Cone
projector adjoint-identity tests (9) + cone FBP/FDK + cone VCD (incl. helical /
anisotropic) all pass.  **Measured CPU win (fan-split):** FORWARD horizontal 256³
**102.8 → 9.05 ms/view (~11×)**, cliff gone (scaling ×47.6 → ×9.7) — forward is now
vertical-bound on CPU.  Back-horizontal was already ~2% on CPU (2.1→2.0, unchanged);
its win is on GPU where it was the dominant stage.  **GPU re-run (Greg): expect the
analogous win for BOTH directions** (the GPU's dominant stage).

---

## 6. `entries_per_cylinder_batch` deletion

The outer band loop subsumes the internal chunking once single-device also drives
band-looping (the trivial-placement unification, already true for ParallelBeam):

- cone back `create_voxel_cylinder_slices` (lax.map over slice chunks) — gone;
  the band IS the chunk (the vertical fan is banded by `(g0, L)`).
- cone forward `create_det_column_rows` (lax.map over det-row chunks) — gone; the
  vertical fan produces the band's full-row contribution (no internal chunking).
- translation + multiaxis equivalents — same.
- The attribute, its geometry-params entries (cone/translation/multiaxis), and the
  dead `slice_range_length` (cone — computed, shipped, used by nothing) all retire.
- NOTE: on GPU the chunking is NOT a perf cliff (the CPU ×62 was cache-specific);
  this deletion is a SIMPLIFICATION + the band loop is needed for sharding anyway.

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

Cone is where the value and the risk live, so we confront it first and land it WELL
before touching the other geometries.  Staging revised 2026-06-13 to the data-driven
design (no window, no fusion; channel-major is the #1 lever).

0. **Baselines — DONE** (§8a): cone single-device CPU + GPU captured; the GPU table
   is the no-regression reference and resolved the design (§8a-design).
A. **Channel-major the cone horizontal fans (FIRST coding increment)** — apply the
   parallel-beam channel-major transpose to BOTH cone horizontal fans (forward
   scatter `forward_horizontal_fan_*`, back gather `back_horizontal_fan_*`).  The
   #1 GPU lever, well-localized (2 functions), independently testable (the existing
   cone adjoint identity + allclose vs current).  No structural change yet.  Land
   CPU-green; measure the win with the fan-split bench (CPU) + GPU re-run.
B. **Module-level banded drivers + cone two-stage kernels** — the `(g0, L)` banded
   VERTICAL fan + horizontal-once (Option B, §1), the anchor rule (§3), the global
   validity clip (§5), forward accumulation (§4), `entries_per_cylinder_batch`
   deletion (cone), de-closuring (§7).  `_supports_sharding()=True` for cone.
   Transitional branch: sharded orchestration routes "geometry has banded kernels?"
   to the new driver, else the existing parallel row-crop path (tagged RETIRE-AFTER).
   CPU-green; then GPU validation at scale (stage2 pattern, cone edition).
C. **Parallel conversion + old-driver deletion** — once cone validates, port
   ParallelBeam onto the banded drivers, delete the closure drivers + transitional
   branch, confirm trace-sharing across model instances + full suite green.
D. **Translation port**, then **multiaxis** (its own `_supports_sharding` flip — it
   extends TomographyModel directly); translation may need per-view-bucket geometry
   bounds (the magnification-spread caveat, §9) — decide with its own measurement.
E. **Retirement cascade** (v2 §P6 step 4 — only after D): legacy single-device
   bodies, `main_device`/`sinogram_device`, `view_indices` machinery,
   `initialize_recon` early device_put, `compute_hessian_diagonal`'s
   `output_device`, RETIRE-AFTER sweep.

Validation per port: the **(g0,L) adjoint round-trip at arbitrary bands** (the
strong per-geometry gate), banded-vs-baseline full recon at 1e-5/1e-4 allclose,
padded exact-zero invariants, full suite.

---

## 8a. Baseline measurements (no-regression reference)

Goal (Greg): no regression, judged not literal.  From ParallelBeam we expect good
per-device and per-size scaling and a 1-device-mesh case comparable-to-or-better
than the existing single-device path; cone differs (channel-major, banded vertical,
accumulation), so measure carefully and evaluate as we go.

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
sinogram size here (NxNxN).  RSS (process-level, CPU).  **Two runs**: the first
was partly contaminated (another app held ~148 GB → swap on this unified-memory
box); the **CLEAN re-run** below is authoritative (projectors 3 trials; VCD 1
timed pass at **3 iterations**).  YAML: `results/cone_baseline_cone_clean_cpu.yaml`
(contaminated first run kept at `cone_baseline_cone_cpu.yaml` for comparison).

| size (sino=recon) | forward ms | back ms | vcd_const ms (3 it) | vcd_nonc ms (3 it) | RSS@vcd MB |
|---|---|---|---|---|---|
| 64³  |     40.9 |     14.7 |     350 |     358 | ~1 280 |
| 128³ |    990.5 |    202.6 |   5 130 |   5 126 | ~3 105 |
| 256³ | 42 304.2 | 22 216.1 | 294 533 | 288 421 | ~5 620 |

What swap touched and what it didn't (verified by comparing the two runs):
- **Projectors were NOT contaminated.**  forward/back are within noise across the
  two runs (forward 256³ 42 637→42 304; back 256³ 22 812→22 216) — they are short
  and low-memory (back 256³ = 2.3 GB RSS), so swap never reached them.  All
  projector cells are rock-stable (std ≤0.7%).
- **Only the long VCD recons were contaminated** (vcd_const 256³ 1 139 239→294 533;
  the ~19-min sustained recon is what hit swap).  Clean VCD spreads are 0% (single
  timed pass).

**Findings (these MOTIVATE the port, and become the post-port gate):**
1. **Super-N⁴ projector scaling on CPU, with a sharp, REAL BACK cliff at 256³.**
   Ideal doubling = ×16 (time ∝ N⁴ = voxels·views).  Measured (clean): forward
   ×24.2 then ×42.7 (exponent 4.6→5.4); **back ×13.8 then ×109.6 (exponent
   3.8→6.78)**.  The ×110 back jump 128³→256³ reproduces across BOTH runs (clean
   and contaminated agree within 2.6%), so it is a genuine working-set/cache cliff,
   NOT swap.  **LOCALIZED (§8a-split): the cliff is the VERTICAL fan (×62), not the
   horizontal channel access** — my initial channel-aliasing hypothesis was wrong
   (corrected below).  **The 256³ back number (22.2 s) is the headline pre-port
   datum to beat.**
2. **vcd_nonc ≈ vcd_const** (clean): ×1.02 / ×1.00 / ×0.98 — essentially equal, as
   expected (the weighted path adds only an elementwise multiply + a buffer).  The
   earlier 1.2–1.9× gap was contamination, now gone.
3. **Few-iteration VCD is a poor SCALING ruler (Greg's amortization point,
   confirmed quantitatively).**  Clean vcd_const 256³ = 294 s for 3 iters ≈ 98
   s/iter, vs the contaminated 15-iter run's 76 s/iter — per-iter looks *worse* at
   fewer iters because the direct-recon init + first-call compile amortize over
   fewer iterations.  So VCD time is a correctness/integration anchor, not a
   scaling number; the projectors are the scaling ruler (the VCD loop itself is
   geometry-independent — test-suite redesign rationale).
4. **Memory** (CPU RSS, process-level, approximate): forward 256³ 2.9 GB; VCD 256³
   5.6 GB.  GPU `peak_bytes_in_use` (the real per-device ruler) is the cluster run.

**GPU baseline — H100 80GB, single device (2026-06-13), `peak_bytes_in_use`:**

| size | forward ms / GB | back ms / GB | vcd_const ms / GB (3 it) | vcd_nonc ms / GB (3 it) |
|---|---|---|---|---|
| 256³  |   296 / 1.3 |   158 / 1.2 |  2 961 / 2.1 |  3 076 / 2.1 |
| 512³  | 4 820 / 6.0 | 2 287 / 5.6 | 32 347 / 12.1 | 29 022 / 12.1 |
| 1024³ | 78 946 / 16.0 | 35 888 / 17.3 | **OOM** | 256 480 / **64.8** |

- **Clean ~N⁴ scaling, NO cliff on GPU.**  forward ×16.3 then ×16.4; back ×14.5
  then ×15.7 (ideal ×16).  So the CPU back-vertical ×62 cliff was a **CPU cache
  artifact** — gone on GPU.  (Confirms the §8a-split CPU/GPU caveat.)
- **Projectors FIT 1024³ single-device on one H100** (~16–17 GB) — I predicted OOM;
  wrong (80 GB is plenty for a single projector op).  **Full VCD is the capacity
  wall:** vcd 1024³ is ~marginal (~65 GB) — vcd_nonc fit, vcd_const OOM'd.  So cone
  needs multi-GPU for VCD at 1024³+, not for the projectors alone.
- **Anomaly (flagged, not chased):** vcd_const 1024³ OOM but vcd_nonc fit at 64.8 GB
  is backwards (const should use ≤ nonconst).  Likely GPU-occupancy variance on a
  shared node or a const-path transient at the 80 GB edge; the "~marginal at 1024³"
  conclusion holds either way.  Re-run on an empty GPU if a precise wall is needed.

### 8a-split — horizontal vs vertical fan cost (informs A vs B vs fused)

Open question (Greg + design): cone back = horizontal fan (band-independent,
materializes the pixels×rows transient) then vertical fan.  Three structures on
the table — **A** per-band row window (recompute horizontal per band, ~2× horizontal
FLOPs, cone-angle-dependent), **B** horizontal-once + pixel-batched memory, and the
**fused / sino-accumulator** structure (Greg: per pixel-batch, accumulate vertical
output into the sino (fwd) / gather from it (back), never materialize pixels×rows;
the two fans do NOT strictly commute — the vertical fan is pixel_mag-dependent and
horizontal mixes pixels↔channels — but fusing per pixel-batch with the sino/recon as
the in-place accumulator achieves the same benefit).  Decide with a micro-bench:
time the horizontal and vertical fans SEPARATELY (which stage dominates → is fusing
worth the restructure?) + a fused-vs-separate timing check (proxy for whether XLA
elides the pixels×rows intermediate).  Bench: `cone_fan_split_microbench.py`
(8-view batch, full pixels).

**RESULTS (CPU, 2026-06-13, 8-view batch) — these INVERT the premise of
A/B/window/fusion, and the two DIRECTIONS have cliffs in OPPOSITE stages:**

| size | BACK vert ms | BACK horiz ms | FWD vert ms | FWD horiz ms |
|---|---|---|---|---|
| 64³  |   2.12 |  0.65 |   3.08 |   2.09 |
| 128³ |  11.26 |  1.46 |  19.39 |  16.68 |
| 256³ | 693.24 | 16.15 | 492.60 | 814.36 |

128³→256³ scaling: **BACK vertical ×61.6** (horiz ×11.0); **FORWARD horizontal
×48.8** (vert ×25.4).  Back fused/separate ≈ 1.0 (no fusion time win).
Cross-check: back-vertical 8 views × (256/8) ≈ 22 300 ms ≈ the full back 22 216 ms
(§8a-results) — so **back ≈ its vertical fan; forward is split, horizontal-heavy at
scale.**

**Conclusions (supersede §2's row-window rationale; re-scope §5b):**
- **BACK cliff = the VERTICAL fan, not channel access.**  My "strided
  `sinogram_view[:, n]` cache-aliasing" attribution for back (§5b / §8a finding 1)
  was WRONG-DIRECTION — back-horizontal scales a tame ×11.  (Mechanism
  over-attributed before measuring — the project's own ruler lesson, self-inflicted.)
- **FORWARD cliff = the HORIZONTAL fan** (the channel SCATTER
  `sinogram_view.at[:, n].add`, stride = num_det_channels → cache aliasing), ×48.8.
  So **channel-major IS warranted — for FORWARD horizontal** (§5b corrected).
  Forward-vertical also blows up ×25, so forward has two heavy stages.
- **The row WINDOW (§2) optimizes a ~2% stage for BACK → NOT a perf lever.**
  Slice-banding is still right for back, but because the EXPENSIVE stage (vertical)
  is the one that bands by output slices — not because a window restricts
  horizontal rows.  So back: horizontal full-rows (cheap) per view-owner; band the
  vertical fan by slices for the reduce-scatter; pixel-batch to bound the
  pixels×rows cylinder (GPU-memory, ~3.2 GB/view at 1024³).  Option B, NO window.
- **Fusion buys ~0 time** (back fused/separate ≈ 1.0): the pixels×rows intermediate
  is not the bottleneck → the fused/sino-accumulator restructure is not justified
  on time grounds.
- **Levers, by direction:** BACK — the vertical fan's super-N⁴ blowup (×62 vs ×8
  of pixels×slices), likely working set (per-pixel gather from the pixels×rows det
  cylinder crossing cache at 256³) and/or `entries_per_cylinder_batch` chunking +
  per-slice geometry recompute → the band loop + chunk deletion (§6) targets this.
  FORWARD — channel-major the horizontal scatter (§5b), plus the vertical (input-
  band accumulation, §4).
- **CPU vs GPU caveat:** these cliffs are CPU cache effects; GPU will look
  different (Greg's cluster run).  Treat the DIRECTIONAL attribution (which stage
  dominates) as robust; treat the absolute cliff factors as CPU-specific.

**GPU RESULTS (H100, 2026-06-13, per-view ms) — these OVERRIDE the CPU conclusions
for the design (GPU is the production target):**

| size | BACK horiz | BACK vert | FWD horiz | FWD vert | fused/sep |
|---|---|---|---|---|---|
| 256³  |  0.201 | 0.158 | 1.257 | 0.181 | 1.72 |
| 512³  |  2.519 | 1.429 | 7.780 | 1.988 | 1.06 |
| 1024³ | 21.535 | 8.405 | 57.204 | 16.754 | 1.15 |

Per-view scaling (×/doubling; ideal ×16): all SMOOTH, sub-N⁴ (BACK h ×12.6/×8.5,
v ×9.0/×5.9; FWD v ×11/×8.4, h ×6.2/×7.4) — **no cliff on GPU.**

**GPU conclusions (the design basis — they REVERSE the CPU back finding):**
- **The HORIZONTAL fan dominates BOTH directions on GPU** — forward strongly
  (H/V ≈ 7→3.4), back moderately but really (H/V ≈ 1.3→2.6, i.e. back-horizontal is
  the LARGER stage on GPU, opposite to CPU's 2%).  The strided channel scatter/
  gather is the GPU bottleneck (poor coalescing), so **channel-major (§5b) is THE
  primary cone optimization on GPU, for FORWARD and BACK.**
- **The row WINDOW (§2) is now clearly WRONG on GPU**: it recomputes the dominant
  (horizontal) stage per band (~2× the biggest cost).  **Option B — horizontal-once
  + pixel-batched memory, band the slice axis only for the multi-device
  reduce-scatter (back) / accumulate (forward) — wins decisively.**  No window.
- **Fusion: no win on GPU either** (fused/separate 1.06–1.72, *hurts* at 256³).
  Off the table.
- The CPU back-vertical cliff does NOT reproduce on GPU (cache artifact), so the
  `entries_per_cylinder_batch` chunking is a SIMPLIFICATION + sharding enabler (§6),
  not a GPU perf fix.

Fused-vs-separate PEAK MEMORY: deprioritized (no time win anywhere; the baseline's
per-device `peak_bytes_in_use` is the memory ruler — projectors ~16 GB at 1024³,
VCD ~marginal).

### 8a-design — the data-driven cone design (supersedes the §1/§2/§4/§5 lean)

Settled by CPU+GPU measurement, GPU weighted (production target):
1. **Channel-major both horizontal fans** (forward + back) — the dominant GPU
   stage; the single biggest lever.  Kernel-level A/B at the port.
2. **Structure = Option B, no row window**: horizontal-once over full rows
   (pixel-batched to bound the pixels×rows cylinder), band the slice axis only for
   the multi-device reduce-scatter (back) and accumulate per band (forward, §4).
3. **No fusion / sino-accumulator restructure** (no measured time win).
4. **Delete `entries_per_cylinder_batch`** (§6) — simplification + the band loop is
   needed for sharding anyway; not a GPU perf fix.
5. **Anchor rule** (§3) unchanged — mechanical, bit-exact.
6. The (g0,L) banded INTERFACE (§1) stays (it is how the slice axis is banded for
   the reduce-scatter); the **row WINDOW within it is dropped**.

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
