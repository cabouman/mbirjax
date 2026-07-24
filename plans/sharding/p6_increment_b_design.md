# P6 increment B — cone two-stage banded projector: design note (2026-06-13, DRAFT for review)

*Implementation plan for increment B of the cone port.  High-level design + the
measurements that justify it: `p6_projector_rework_proposal.md` (read §8a-design).
Increment A — channel-major horizontal fans — is committed (CPU win, GPU-neutral,
build-verified).  This note is the concrete structure + the STAGED checkpoints, each
with correctness / memory / timing gates (Greg's request).*

**PROGRESS (2026-06-13):**
- **B1 DONE (staged, CPU-green):** banded cone kernels + `tests/geometries/test_cone_banded.py`
  (circular + helical: back band-decomposition, full-band==monolithic, adjoint-at-(g0,L),
  Hessian).  Review fixes applied (RETIRE markers, plain comments, renames, helical).
- **Forward structure DECIDED = C** (per-pixel-batch all-gather + monolithic; harness
  `cone_forward_structure_compare.py`): B (banded/streamed) is 5–14× slower on CPU
  (dispatch-bound) for ~13–23% memory; C ≈ current (no regression).  So the cone sharded
  **forward does NOT band**; **back stays banded** (reduce-scatter).
- **DONE (2026-06-13c):** removed the forward banded kernel
  (`forward_project_band_to_one_view` / `forward_vertical_fan_band_*`) and its three
  forward-dependent tests in test_cone_banded (the two forward band-decomposition
  tests + the forward-band adjoint test).  The adjoint test was DROPPED, not
  re-expressed: the banded BACK kernel is fully gated by the back band-decomposition
  against the monolithic back, which test_projectors already gates for adjointness
  (`⟨forward_mono x, y⟩ = ⟨x, concat back_band(y)⟩`).  Section header in cone_beam.py
  retargeted to back-only with a note recording why forward does not band (so it is
  not re-added).  Cone banded + cone projector suites green.
- **B2 commit 1 DONE (2026-06-13c, staged — additive, nothing deleted yet):** rewired the
  single-device cone `back_project_one_view_to_pixel_batch` to horizontal-fan-once + a
  ROLLED `jax.lax.map` over slice bands of `CONE_SLICE_BAND_SIZE` (module constant = 128,
  the old `entries_per_cylinder_batch` default) → reshape/crop.  The loop is ROLLED
  (lax.map, not a Python unroll), so the compiled program is independent of the slice
  count (Greg's unroll concern — slice counts span tens→thousands).  Added a
  `slice_band_size` static kwarg (None→constant) so a small test geometry exercises the
  multi-band assembly.  Monolithic vertical-fan primitives KEPT as the A/B reference.
  Tests: new `test_cone_banded` driver A/B (production lax.map+reshape+crop == monolithic
  across band sizes incl. a non-dividing crop, coeff_power 1+2); `_back_monolithic`
  repointed to assemble from the monolithic primitives (no longer the now-banded
  production method).  GATES GREEN: test_cone_banded, cone test_projectors/test_fbp_fdk/
  test_vcd (15), full test_projectors (21).
- **B2 commit 2 DONE (2026-06-13c, staged — the deletions):** deleted the monolithic cone
  vertical fan (`back_vertical_fan_one_view_to_pixel_batch` / `..._to_one_pixel`, 89 lines);
  swapped the FORWARD vertical fan's det-row chunk to the new module constant
  `CONE_FORWARD_DET_ROW_BATCH = 128` (bit-identical — only the source of 128 changed);
  deleted `entries_per_cylinder_batch` + the dead `slice_range_length` from cone (instance
  attrs, the geometry-param namedtuple, the `slice_range_length` computation) — runtime-only,
  no save/load migration (they were instance-attr namedtuple fields, not registered params);
  removed the banded-section RETIRE note (the banded kernels are production now).
  `test_cone_banded` consolidated to ONE self-contained gate `test_back_production_matches_
  band_concat` (production uniform-band lax.map == explicit non-uniform-band np.concatenate,
  band-size sweep incl. the non-dividing crop, coeff_power 1+2) — no monolithic reference, so
  permanent; the banded vertical fan's physical correctness is gated by test_projectors
  (adjoint) + test_vcd (convergence).  Stale `slice_range_length` comment in test_view_params
  fixed.  GATES GREEN: cone test_cone_banded/test_projectors/test_fbp_fdk/test_vcd (16), full
  test_projectors + test_view_params (25).
- **B2 §8a memory/timing — DONE; B2 CLOSED, no regression (2026-06-13c).**  Compared
  `cone_baseline_scaling.py` OLD (pre-B2) vs NEW (post-B2), 3-iter both, single-device
  (results/cone_baseline_cone_clean_{cpu,gpu}.yaml vs `..._{cpu,gpu} 2.yaml`, gitignored).
  **Method — forward as the bit-identical control:** B2 left the FORWARD projector
  bit-identical (commit 1 never touched it; commit 2's only forward change was
  `gp.entries_per_cylinder_batch → CONE_FORWARD_DET_ROW_BATCH`, same value 128), so any
  forward delta is NON-B2 and exposes the run's ruler error; the BACK delta relative to
  that control is the real B2 signal.
  - **GPU (H100, 256³–1024³, n=1): peak memory BYTE-IDENTICAL** old-vs-new for forward and
    VCD; back differs by ~0.5 KB.  ⇒ B2 changes single-device GPU peak by ≤0.5 KB —
    **capacity preserved exactly.**  Time: forward (control) and back BOTH ~0.53× (≈1.9×
    faster) — i.e. the documented ~1.9× GPU run-to-run variance; back TRACKS the control ⇒
    **time-neutral.**  (Lone outlier vcd_nonc 1024³ 1.03× = within that variance band,
    hottest single-trial.)
  - **CPU (64³–256³, n=1): back ~1.0× time (neutral)**, peak within the ≤1.1× gate (+1–3%).
    The CPU forward 0.38× at 256³ is NOT B2 — the OLD CPU yaml (6:25am) PREDATES the
    channel-major sinogram conversion (increment A, the ~13× horizontal CPU win,
    Greg-confirmed), so forward and back move by DIFFERENT factors (forward 0.38×, back
    1.0×) ⇒ not uniform variance ⇒ the forward delta is increment A, the back ~1.0× is the
    clean B2 signal.
  - **Verdict: B2 is memory- and time-NEUTRAL on both platforms.** The banded rolled
    `lax.map` (band=128) reproduces the old `entries_per_cylinder_batch=128` chunking, as
    designed.  GPU is the gate that matters (capacity) and its peak is byte-identical.
- **B3a DONE (2026-06-13c, staged) — de-closuring + the real blocker fixed.**  Lifted the
  four per-instance projector closures in `projectors.py` to MODULE-LEVEL jitted functions
  (`_sparse_forward_project` / `_sparse_back_project`) with the per-view kernel,
  projector_params, and the two batch sizes as STATIC args and the view-param array TRACED;
  dropped the unused exposed inner fns; kept `self._jit_sparse_*` as references to the module
  globals (introspection/`_cache_size` unchanged).  **KEY DISCOVERY: de-closuring alone did
  NOT share the cache** — `get_geometry_parameters()` calls `namedtuple('GeometryParams', …)`
  on every invocation, so each instance gets a DISTINCT namedtuple CLASS; jax keys the
  static-arg cache on the pytree treedef (which includes the class), so distinct-but-identical
  classes ⇒ re-trace.  **Fixed at the SOURCE (Greg's call, option b):**
  `ParameterHandler.make_geometry_params(field_names, values)` builds `GeometryParams` from a
  class cached by field-name tuple (one class per geometry, shared across instances); all 4
  geometries (`cone_beam`, `parallel_beam`, `multiaxis_parallel`, `translation_model`) route
  through it, and `ProjectorParams` is hoisted to module level in `projectors.py`.  (First
  landed as a `projectors.py` canonicalization shim, then moved to the source so the per-view
  kernels + the B4 sharded driver also get the shared class; **shim retired**.)  **Gate (new
  `tests/test_projector_cache_sharing.py`):** a 2nd fresh same-geometry model adds ZERO new
  programs to the shared module cache; measured a 2nd cone model's first forward 16× faster
  (201→13 ms, trace+compile reused).  Full suite green (160p/3s).
  - **Bonus from fixing at source:** the per-view kernels
    (`back_project_one_view_to_pixel_batch`, `@jit static projector_params`) now also get the
    shared `GeometryParams` class, so their caches share across instances too (not just the
    driver) — and the B4 sharded driver inherits this for free.  Verified the class is now
    identical across instances (was distinct before).
  - **General lesson (→ lessons.md candidate):** never call `namedtuple()` (or build any
    pytree-typed object) inside a function that feeds a jit STATIC arg — jax keys the
    static-arg cache on the pytree treedef, which includes the class, so a per-call class
    silently defeats cross-instance cache sharing.  Build the type once.
  - **B3b (tail-batch padding) — deferred/measured, unchanged.**
- **B4 plan (reviewed with Greg) + B4.1 DONE.**  The reduce-scatter/all-gather infra in
  tomography_model is GEOMETRY-AGNOSTIC; only two pieces are parallel-specific — the back
  per-band worker's ROW-CROP and the banded FORWARD.  B4 routes those by geometry; everything
  else (band loops, `_balanced_slice_bounds`, `sum_band_to_owner`/`broadcast_band_to_views`,
  `_mask_padded_*`, `assemble_sharded`, placements) is reused.  **Decision (A)** (back: recompute
  horizontal per band vs hoist) — START SIMPLE (recompute), MEASURE the penalty vs §8a, hoist
  (B4.5) only if needed.  **Memory (Greg's question):** the simple per-band worker is BOUNDED by
  view_batch_size×pixel_batch_size — it batches+sums over views (not V_d copies), and the full
  view-shard is read in place (not copied); the cone-vs-parallel extra is the full-detector-row
  transient recreated per band = the recompute cost itself.
  - **B4.1 DONE (2026-06-13d, staged) — cone back sharded path (banded driver + geometry hook).**
    Added `projectors._sparse_back_project_band` (banded back driver batched over views+pixels;
    g0 traced, num_band_slices static; bounded by the batch sizes) exposed as
    `Projectors.sparse_back_project_band` for geometries with a banded kernel (cone).  Geometry
    hook `TomographyModel._back_project_view_shard_to_band` — **BASE = the geometry-neutral BANDED
    path** (cone + future geometries use it as is); **`ParallelBeamModel` OVERRIDES** with its
    cheaper detector-row crop (a specialization exploiting row r → slice r, not the default).  The
    sharded back worker routes through the hook.  (Refactored from an initial base-row-crop/cone-
    override after review — base now holds the general case, not parallel's; also safer, since a
    future sharded geometry defaults to correct banded behavior instead of inheriting row-crop.)
    Gate: new `test_cone_banded` driver test (`sparse_back_project_band(full_sino, g0, L) ==
    sparse_back_project(full_sino)[:, g0:g0+L]`, coeff_power 1+2, circular+helical) green;
    parallel sharding suite unchanged (103p @4dev, now via the override); full suite 161p/3s.
  - **B4.2 DONE (2026-06-13d, staged) — cone forward sharded path (gather + monolithic).**
    Refactored `_sparse_forward_project_sharded` to call a `_forward_project_to_view_shards` hook
    (mirrors the back): BASE = the geometry-neutral GATHER+MONOLITHIC path (each view-owner gathers
    the full slice cylinder PER PIXEL-BATCH via `move_shard`+concat, runs the monolithic forward,
    sums over pixel-batches — decision C, the structure `cone_forward_structure_compare.py`
    validated); `ParallelBeamModel` OVERRIDES with its banded forward (broadcast band → project
    rows [g0:g1), never gathers).  Assemble shape now geometry-aware via `_sino_device_shape()`
    (cone keeps its real detector rows; parallel pads rows with slices — value-identical for
    parallel).  Gate: parallel sharding unchanged (103p @4dev, now via the override); full suite
    161p/3s.  The cone gathered path is DORMANT until the flag flip (B4.3) — exercised there.
  - **B4.3 DONE (2026-06-13d, staged) — cone `_supports_sharding()=True`; CPU-VALIDATED end to end.**
    Flipped the flag (cone defaults to the always-on placement path: trivial 1-device mesh
    single-device, shards multi-device).  n_dev=1 gate: the EXISTING cone gates (test_projectors
    adjoint, test_fbp_fdk, test_vcd convergence — circular+helical) all pass through the sharded
    path; full suite 161p/3s.  Multi-device gate: new `tests/sharding/test_cone_sharded.py` —
    back/forward/Hessian (coeff_power=2) at **1e-5** and a 3-iter VCD recon at **1e-4**, sharded
    (n=2,4) == single-device, circular+helical; full sharding suite 107p @4dev (cone+parallel
    coexist).  ⇒ the cone reduce-scatter back + gather-forward are CORRECT end to end on CPU
    AT DIVIDING COUNTS.  (`tests/sharding/test_cone_sharded.py` is UNTRACKED — commit it.)
  - **⚠ KNOWN FAILURES — DEFERRED TO B5 (Greg's call; do NOT chase before B5).**  The flag flip
    enabled the geometry-agnostic P5 PADDING for cone, but cone padding IS B5 (not done).  So at
    NON-dividing slice counts (≥4 devices) cone auto-pads and 4 tests FAIL on multi-device:
    `test_{adjoint,hessian}_anisotropic_cone` (test_projectors), `test_split_sino`,
    `test_vcd_anisotropic_cone` (test_vcd).  Reproduce on CPU with `MBIRJAX_NUM_CPU_DEVICES=4`
    (default 2 → no padding → why CPU was green; GPU box has 4 → red).  Cause: anisotropic cone
    (voxel_slice_aspect=2.9 → 14 slices) padded 14→16 at 4 dev; the forward gather assembles the
    PADDED cylinder and the device-form padded shape leaks to tests assuming the real shape.
    Scaling tests UNAFFECTED (256/512/1024³ divide 2/4/8 → no padding).
  - **B4.4 NEXT (GPU — Greg running):** `cone_baseline_scaling.py` multi-device sweep (n_dev 1/2/4)
    at DIVIDING sizes: per-device peak ~1/n_dev; the CAPACITY win (a 1024³ VCD that OOM'd
    single-device now fits sharded); the back horizontal-recompute penalty vs §8a.
  - **B4.5 (REFRAMED 2026-06-16, GPU-confirmed):** the band (reduce-scatter) kernel is **~2.25× slower
    than the monolithic pixel kernel ON GPU** — now the limiter of multi-device back scaling (n=2 back is
    SLOWER than n=1; crossover at n≈2.25, so ≥3 GPUs are needed before sharding back pays in TIME; VCD
    stays monotonic only because the forward parallelizes and masks it).  The n=1 case is already handled
    by the GPU-only n=1 back short-circuit (single-GPU → the pixel kernel; see sharding_status.md
    2026-06-16 handoff + lessons.md "Platform-divergent back-projection kernel").
    **KEY TAKEAWAY — the two kernels are PLATFORM-OPPOSITE:** GPU loves the rolled-pixel structure (the
    single-band kernel is 2.25× slower there); CPU loves the single-band kernel (the rolled-pixel
    `lax.map`+transpose hits the ×62 back-vertical cache cliff, ~8× slower).  So **B4.5 is NOT just
    "hoist the horizontal fan"** — it is "make the band kernel GPU-competitive (e.g. a rolled /
    pixel-like internal vertical structure) WITHOUT reintroducing the CPU cliff."  A single kernel that
    is fast on BOTH is the real design challenge; absent that, platform-specific kernel selection (what
    the n=1 short-circuit already does) is the fallback pattern.  Horizontal-fan hoist is one sub-idea,
    pursued only if the per-band recompute (at multi-band sizes) is shown to dominate the band-kernel cost.
  - **B5 DONE (2026-06-18, staged) — exactly-inert slice padding for cone; CPU-validated.**  The 4
    deferred failures (`test_{adjoint,hessian}_anisotropic_cone`, `test_split_sino`,
    `test_vcd_anisotropic_cone`) PASS at 4 devices; the full suite is green at the default 4 CPU
    devices (169p).  **One root cause for all four:** the cone sharded forward GATHERS the device-form
    (padded) cylinder and feeds the monolithic kernel, which anchors on / asserts the REAL slice count
    (`forward_project_pixel_batch_to_one_view`).
    - **Fix (1) — load-bearing.**  `_forward_project_to_view_shards` (the base/cone gather path) crops
      the gathered cylinder to `recon_placement.real_size` before the monolithic forward
      (`full_cyl = full_cyl[:, :real_slices]`).  EXACT, not an approximation: the padded slices are
      zero (forced-zero invariant), so dropping them changes nothing; a no-op at dividing counts.
      ParallelBeam OVERRIDES this method (banded forward), so it is cone/base-only.
    - **Fix (2) — the test contract.**  `sparse_back_project` STAYS device-form (architecturally
      required: the VCD loop and `output_sharded` need a shardable padded array; cropping to a
      non-dividing real count would break the sharding).  So the geometry tests
      `verify_adjoint`/`verify_hessian` crop the device-form back projection to real slices
      (`[:, :num_recon_slices]`) — a no-op without padding.  This is a **LATENT, geometry-agnostic**
      test bug, NOT cone-specific: parallel fails identically at 3 devices (40 slices → 42), hidden
      only because 40 divides 2/4.  `sparse_back_project`'s docstring now states the device-form return.
    - **Fix (3) — `_supports_slice_padding()` hook DROPPED** (Greg): after B5 both sharding geometries
      (parallel + cone) support padding, so the hook would be all-True/dead.  Revisit when a future
      geometry is `_supports_sharding=True` before its padding work is done.
    - **Bonus real bug (found by the new helical padding test): `helical_fdk_z_weight`.**  It built
      its per-recon-slice z-weight at the REAL slice count and applied it to the device-form (padded)
      recon → broadcast crash under slice padding (helical cone is 11 slices, not num_det_rows).  Now
      built at the device-form length, z anchored on the real count, with the padded slices forced to
      0 weight (guards `0 * inf → NaN`; an out-of-coverage padded slice has coverage 0).  A no-op when
      nothing is padded (this is why helical passed in test_cone_sharded at dividing counts).
    - **Also:** `verify_adjoint` now zeros the random `y`'s padded VIEWS — the back projector relies on
      padded views being zero (production zero-fills at entry); a hand-built device-form `y` with a
      nonzero padded tail contaminated `Aᵀy` at the clamped padding angle (~3% adjoint gap, view-padding
      counts only).  This + the slice crop make the verify tests robust at ANY device count (now pass
      at 2/3/4).
    - **Tests added/extended:** new `tests/sharding/test_padding.py::TestPaddedSlicesCone` (a prime
      7-slice cone, circular+helical: back/forward/Hessian + VCD const & non-const weights
      sharded==single-device, and the forward/back device-form **exact-zero** invariant — padded
      slices on back, padded views on forward, real detector rows preserved); the parallel
      `TestPaddedSlices` exact-zero test extended to the FORWARD direction (Greg's "both directions").
    - **Gates:** 4 deferred @4 ✓; full suite @4 169p ✓; sharding suite @4 107p+cone ✓; padding @3 & @4 ✓;
      `test_projectors` @3 21p ✓.  **GPU confirmation at a non-dividing slice count is pending**
      (CPU `MBIRJAX_NUM_CPU_DEVICES=4` is the proxy; the GPU box has 4 devices).  NEXT: C (+ the FDK
      filter → sharded-contract cleanup below, which also touches the cone FDK init path).
  - **FOLLOW-UP — convert cone `fdk_filter` to the sharded-contract `fbp_filter` pattern. ✅ DONE**
    (landed interspersed with B4/B5; see `sharding_implementation_plan_v3.md` §4).  `fdk_filter` now
    uses the shared `_apply_direct_recon_filter` (the FDK cosine pre-weight folded into `row_weight`)
    and `fdk_recon` stays sharded throughout — no gather, no single-device init bottleneck.  The
    original analysis is kept below for the record.  Now
    that cone is on the placement path (B4), `ConeBeamModel.fdk_filter` (cone_beam.py ~974) ~~still
    runs SINGLE-DEVICE~~ — its docstring "Accepted for API uniformity. Cone beam runs single-device
    UNTIL THE PLACEMENT PORT" is now STALE (the placement port = B4, done).  So a sharded cone
    `direct_recon`/`fdk_recon` (the VCD FDK init) gathers the sinogram, filters single-device, then
    re-shards for the sharded `back_project` — a host round-trip + single-device bottleneck at init
    for large sharded cone recons.  **The KEY POINT of `fbp_filter`'s sharded change was wrapping
    `tomography_utils.apply_row_filter` in `mjs.run_per_device`** (parallel_beam.py ~503-534): on a
    sharded model each device's worker filters its OWN local view-shard on-device
    (`apply_row_filter(shard, filter)` under `run_per_device`), then `assemble_sharded` stitches the
    results back with no data movement — the memory/time win (no gather, no host round-trip, no f64
    promotion, parallel across devices).  Cone's conversion does the ANALOGOUS: dispatch the cone
    FDK filtering (the per-(row,channel) FDK weighting + ramp) per view-shard via `run_per_device`
    (not `apply_row_filter` — cone's filter is its own per-view kernel), honor `output_sharded`
    (view-sharded internal / gather at exit), so the filtered sinogram feeds the sharded back
    projector with NO gather.  Not hot-path (FDK = init only), so it's an efficiency/consistency
    cleanup, not blocking.  (Stale `fdk_filter` docstring "single-device UNTIL THE PLACEMENT PORT"
    FIXED — now a TODO pointing at this pattern.)  Likely lands with C (parallel/cleanup) or as its
    own small task.

---

## 0. Goal and non-goals

**Goal.**  Give cone the banded projector structure the multi-GPU sharded path needs
(slice-banded recon ⇄ view-sharded sinogram via a reduce-scatter), with NO
single-device regression and bounded per-device memory, then turn on
`_supports_sharding()` for cone.

**The value is CAPACITY, not single-GPU speed** (§8a-design): on GPU the cone
projectors already scale ~N⁴ with no cliff and fit 1024³ single-device for the
projectors; sharding is what lets *VCD* exceed one GPU at 1024³+ (the measured wall).
So B is judged on correctness + memory-shards-1/n_dev + no-single-device-regression,
NOT on a single-GPU kernel speedup.

**Non-goals (deferred):** translation/multiaxis ports (increment D); the retirement
cascade (increment E); any row-window or fused/sino-accumulator structure (measured
out — proposal §2/§4).

---

## 1. The two-stage decomposition (what changes in the kernels)

Today cone's per-view kernels are monolithic two-stage compositions:
- back: `back_project_one_view_to_pixel_batch` = horizontal_fan → vertical_fan.
- forward: `forward_project_pixel_batch_to_one_view` = vertical_fan → horizontal_fan.

B splits them so the driver can compute the **horizontal fan ONCE per view** and
**band the VERTICAL fan by global slice `(g0, L)`** (Option B; horizontal dominates
on GPU and recomputing it per band would ~n_dev²× the dominant stage).  The four
sub-kernels already exist as static methods — B exposes them as the driver's seams
and adds the `(g0, L)` band argument to the vertical fans:

| stage | existing method | B change |
|---|---|---|
| back horizontal | `back_horizontal_fan_one_view_to_pixel_batch` | none (channel-major, increment A) — computed once per view → det cylinder `(pixels, num_det_rows)` |
| back vertical | `back_vertical_fan_one_view_to_pixel_batch` → `..._to_one_pixel` | **add `(g0, L)`**: produce global slices `[g0, g0+L)` only; `compute_vertical_data_single_pixel`'s `slice_indices = g0 + arange(L)` (the seam already takes slice indices — cone_beam.py:617) |
| forward vertical | `forward_vertical_fan_pixel_batch_to_one_view` → `..._one_pixel...` | **add `(g0, L)`**: input is a band of L slices; emit the full `(pixels, num_det_rows)` contribution for those slices |
| forward horizontal | `forward_horizontal_fan_pixel_batch_to_one_view` | none (channel-major) — computed once per view on the accumulated det cylinder |

**New per-view banded kernels** (the driver-facing interface, added ALONGSIDE the
monolithic ones; F1 "build next to, delete loser"):
- `back_project_one_view_to_band(sinogram_view, pixel_indices, single_view_params, projector_params, g0, num_band_slices, coeff_power=1)` → `(num_pixels, L)`.
  Body: `det_cyl = back_horizontal_fan(view, ...)` (full rows, once if the driver
  hoists it — see §3); `band = back_vertical_fan_band(det_cyl, g0, L, ...)`.
- `forward_project_band_to_one_view(band_values, pixel_indices, single_view_params, projector_params, g0)` → `(num_det_rows, num_det_channels)` contribution.
  Body: `det_cyl_contrib = forward_vertical_fan_band(band_values, g0, ...)`;
  `view_contrib = forward_horizontal_fan(det_cyl_contrib, ...)`.

`g0` is **traced** (dynamic scalar); `L`=`num_band_slices` is **static** (≤2 values
from `_balanced_slice_bounds` → ≤2 programs).

---

## 2. Anchor rule + global validity clip (the correctness-critical math)

**Anchor (proposal §3):** physical coordinates from PROBLEM shapes + GLOBAL indices,
never the band length.  In the vertical fans, the slice→z map must use
`S_real = projector_params.recon_shape[2]` and `k_global = g0 + k_local`:
`z(k_global) = Δ_slice·(k_global − (S_real−1)/2) + (recon_slice_offset − helical_z_shift)`.
- Back: `compute_vertical_data_single_pixel` already derives z from `recon_shape`
  (params) — it just needs `slice_indices = g0 + arange(L)` instead of the chunk
  indices.  **Bit-exact today** on full cylinders (g0=0, L=S_real).
- Forward: `forward_vertical_fan_one_pixel_to_one_view` currently derives
  `num_slices = voxel_cylinder.shape[0]` (the INPUT length) and centers on it
  (cone_beam.py:390,402,434) — this BREAKS under banding (a length-L band would
  shift z).  **Fix: source `S_real` from params and `k_global = g0 + arange(L)`.**

**Global validity clip (proposal §5 — exactly-inert padding):** the vertical fans'
slice-validity becomes a GLOBAL test `k_global < S_real` (and `≥ 0`):
- Back-vertical: zero output slices with `g0 + local ≥ S_real` (padded slices, inert).
- Forward-vertical: input band slices with `g0 + local ≥ S_real` contribute zero
  (already zero by the forced-zero invariant; the clip is defense).
This makes padded slices exactly inert in-kernel, so pad-to-multiple-of-n is
provably N-independent (the P5 invariant), and `_mask_padded_slices` / the
forward-mask stay as the one-site postconditions.

---

## 3. Driver structure (horizontal-once + pixel-batch + band)

### 3a. Single-device (increment B2)
The single-device projector must drive band-looping too (so `entries_per_cylinder_batch`
can be deleted and the kernel path is unified).  The memory knob is the EXISTING
internal `pixel_batch_size` (projectors.py); the band is the new slice knob.

Back (pseudocode; one view-batch, inside the existing pixel-batch loop):
```
for pixel_batch P:                      # bounds the pixels×rows transient (existing knob)
    det_cyl = back_horizontal_fan(views, P)          # (P, num_det_rows)  -- ONCE
    for band (g0, L) in balanced_slice_bounds(S_real):
        out[P, g0:g0+L] = back_vertical_fan_band(det_cyl, g0, L)   # cheap
# sum over views happens as today (vmap over views + sum)
```
Forward (adjoint):
```
for pixel_batch P:
    det_cyl = zeros(P, num_det_rows)
    for band (g0, L):
        det_cyl += forward_vertical_fan_band(voxels[P, g0:g0+L], g0)   # accumulate
    view += forward_horizontal_fan(det_cyl, P)        # ONCE
```
Horizontal computed once per pixel-batch (not per band).  `det_cyl` transient =
`pixel_batch × num_det_rows` — same floor as parallel.  `entries_per_cylinder_batch`
chunking deleted (the band loop subsumes it).

### 3b. Sharded (increment B4)
Recon slice-sharded, sino view-sharded; back is a reduce-scatter.  Restructure
`_back_project_all_bands` so each VIEW-OWNER computes its horizontal det cylinder
ONCE and reuses it across all bands (else horizontal is recomputed n_dev²×):
```
for pixel_batch P:                                  # bounds per-device pixels×rows
    for view-owner d:  cyl[d] = back_horizontal_fan(view_shard[d], P)   # ONCE, on d
    for slice-owner t:
        for band (g0, L) in t's slice-shard:
            partials = [back_vertical_fan_band(cyl[d], g0, L) on each d]  # (P, L)
            band_t   = reduce-scatter(partials) onto t                    # sum over views
        owned_p[t] = concat_bands
    accumulate owned_p into owned (over pixel-batches, on the pixel axis)
owned = _mask_padded_slices(owned)                  # postcondition (proposal §5)
```
Forward is the all-gather adjoint: per pixel-batch, broadcast each slice-band to all
view-owners, `forward_vertical_fan_band` → accumulate into each view-owner's det
cylinder, then `forward_horizontal_fan` once → view-shard; `_mask_padded_views`
postcondition.  The transitional driver branch routes "geometry has banded kernels?"
→ this path, else the existing parallel row-crop path (tagged RETIRE-AFTER).

**Loop-ordering note (the one real subtlety):** pixel-batch must be the OUTER loop
so the held `cyl[d]` is `P×rows` not `pixels×rows`.  This differs from today's driver
(bands outer, full pixels to the projector).  This is the bulk of B4's code and its
main risk — gated by the memory check below.

---

## 4. De-closuring (increment B3)

Lift the projector drivers from per-instance closures (projectors.py
`sparse_back_project_fcn` etc.) to MODULE-LEVEL functions taking explicit args
(`projector_params` static — already a hashable namedtuple; the kernel pair static;
view-params traced).  Shares the jit cache across model INSTANCES (sweeps, the vcls
sibling stop re-tracing) and is the template the settable-view-params lift already
established.  **Preserve ONLY the two top-level signatures** (`sparse_forward_project`,
`sparse_back_project`); everything else is a simplification target.  Tail batches pad
to one signature (proposal §7).  Can land before or after B4; recommended AFTER B4
(so the band structure is settled first), but the gate is just allclose + a
trace-sharing check.

---

## 5. Staging with per-stage gates (correctness / memory / timing)

Each stage lands CPU-green for review.  Hard gate = CORRECTNESS (allclose; project
rule — never exact for computed floats).  MEMORY/TIMING are REPORTED vs the recorded
§8a baselines (machine-dependent → guard-rails, not hard fails), via the scaling
harness; GPU runs are Greg's.

| stage | what | CORRECTNESS gate | MEMORY gate | TIMING gate |
|---|---|---|---|---|
| **B1** | banded kernels added (cone `*_to_band`), anchor rule, global clip; NOT wired | NEW unit tests: band-decomposition (full == Σ/concat of bands at arbitrary (g0,L), 1e-5) + adjoint-at-(g0,L) (`<A_band x,y>=<x,A_bandᵀy>`, rel 1e-5/1e-6 per the projector-noise rule); g0=0,L=S_real == legacy kernel (1e-5) | — (pure kernel) | — |
| **B2** | single-device cone on banded kernel; horizontal-once + band; delete `entries_per_cylinder_batch` | cone test_projectors / test_fbp_fdk / test_vcd (existing convergence gates) green; allclose vs pre-B output 1e-5 | `cone_baseline_scaling.py` (CPU 64–256³, GPU 256³–1024³): peak ≤ recorded §8a baseline (report ratio; ≤1.1× ok) | same harness: time ≤ ~1.1× baseline (report; the §8a numbers are the ruler) |
| **B3** | de-closuring (module-level banded drivers) | allclose vs B2 (1e-5); full suite green | — | trace-sharing: program-cache HIT across two fresh model instances (first-call cost drops) |
| **B4** | sharded cone (reduce-scatter on banded vertical); `_supports_sharding()=True` | NEW `tests/sharding/` cone tests: sharded vs single-device 1e-4 iterated (const + non-const weights), adjoint identity; trivial-1-device-mesh vs single-device 1e-5 | `cone_baseline_scaling.py` multi-device sweep (n_dev 1/2/4, GPU): per-device peak ~1/n_dev | multi-device speedup ≥ projectors' measured ceiling; VCD 1024³ that OOM'd single-device now FITS sharded (the capacity win) |
| **B5** | exactly-inert padding for cone (global clip already in B1; wire masks + non-dividing n) | NEW: padded-slice exact-zero invariant (constructed-zero → EXACT equality); padded recon vs unpadded 1e-4; auto-pads a prime slice count | per-device peak unchanged by padding (inert) | padding cost in the noise (report) |

Parallel beam is untouched through B1–B5 (its row-crop path stays).  (SUPERSEDED: this
note originally said "increment C converts it and deletes the monolithic cone kernel +
the transitional branch."  As it played out — see `sharding_implementation_plan_v3.md`
§5 item 2 — the transitional branch became polymorphic override dispatch in B4 (no
deletion needed), ParallelBeam's row-crop / banded-forward overrides are KEPT by decision
2026-06-18 (cheaper for parallel), and the monolithic single-device cone back kernel is
KEPT (the GPU n=1 short-circuit routes to it).  So C's substance landed interspersed with
B4/B5 rather than as a separate phase.)

### B1 — LANDED (CPU-green, 2026-06-13)
Added to `cone_beam.py` ALONGSIDE the monolithic kernels (not yet wired):
`forward_vertical_fan_one_pixel_to_one_view_band` / `..._band_pixel_batch`,
`back_vertical_fan_one_view_to_one_pixel_band` / `..._band_pixel_batch`, and the
driver-facing `back_project_one_view_to_band` (g0 traced, num_band_slices static) /
`forward_project_band_to_one_view`.  Anchor fix applied to the forward vertical fan
(z from params `S_real` + global `k = g0 + arange(L)`, gather via local index);
back vertical already anchored on params (just takes global band indices).  Global
validity clip (`k_global < S_real`) in both — a no-op for g0+L ≤ S_real, the inert-
padding hook for B5.  Forward banded drops the `entries_per_cylinder_batch` det-row
chunking (produces all rows directly).
Tests: `tests/geometries/test_cone_banded.py` (6 tests × {circular, helical} subtests,
all pass) — back/forward band-decomposition, full-band == monolithic,
adjoint-at-(g0,L), Hessian (coeff_power=2) decomposition.  Monolithic cone suite
(test_projectors / test_fbp_fdk cone, 12) unaffected.

Review fixes (Greg, 2026-06-13): helical coverage added (exercises the anchor's
z-shift term); the monolithic-comparison tests marked RETIRE (retire when the
monolithic kernels are deleted), the adjoint test kept as a permanent self-contained
gate; plan notation (§/increment refs) removed from cone_beam.py in favor of plain
descriptions, with the only RETIRE marker on the genuinely-transient "coexists with
monolithic" note; the one-pixel banded fns renamed to `*_band_one_pixel` ("band" =
the slice band, not the output).

**OPEN (raised by Greg's review) — the forward banded vertical over-allocates.**
A band of L input slices reaches only a limited detector-row window, but the current
`forward_vertical_fan_band_one_pixel` evaluates ALL `num_det_rows` per band (most
zero) and the caller sums full columns — O(num_det_rows) work per band, i.e.
num_bands× the monolithic vertical work, plus a full-column intermediate per band.
This is the FORWARD-vertical analogue of the row window we measured out for BACK-
horizontal — but here it is genuinely needed, because back's vertical OUTPUT is the
small thing (L slices) while forward's vertical OUTPUT is the big thing (det rows)
and the band fills only ~W of them.  It is asymmetric: back has no over-allocation
(output is exactly L); forward does.  **Resolved at B4** (it depends on how slice
bands are delivered to each device), with these options to measure:
  (i) **Windowed forward vertical** — evaluate only the band's row window
      `[r0, r0+W)` (W static from worst-case magnification; r0 PER-PIXEL traced,
      since magnification/`t` vary per pixel) and scatter-add into the full column
      (which the horizontal-once stage needs anyway).  Saves the per-band vertical
      compute + intermediate; the per-pixel r0 placement is the added complexity.
  (ii) **All-gather full cylinder + monolithic forward** — don't band the forward at
      all; gather the whole slice cylinder onto each view-owner (pixel-batched to
      bound memory) and run the monolithic forward.  No window, no banded-forward
      kernel; departs from the current band-streaming (`broadcast_band_to_views`).
For B1 the banded-forward kernel is a CORRECTNESS REFERENCE (it validates back_band's
adjoint + the decomposition); its full-column body is annotated (TODO) as such.

**HARNESS + CPU MEASUREMENT (2026-06-13): `cone_forward_structure_compare.py`.**
Direct, empirical B-vs-C(-vs-mono) comparison at the per-view-owner level (recon
slice-sharded across n devices; one owner gathers (C) or streams (B); isolated
subprocess per variant for clean peak).  B implemented as option (i) with a PER-BAND
shared window (r0 per band, static W from worst-case magnification) + det-cylinder
accumulator + horizontal once; all three structures JITTED (apples-to-apples).
- **Correctness: mono == C == B** at every size/n_dev (CPU) — validates the windowed
  vertical fan + the gather/stream structures.
- **Memory (CPU RSS, n=2/4): B beats C at scale** — B/C peak **0.75–0.76× at 256³**
  (B never holds the full cylinder), ~1.0 at tiny sizes (RSS baseline).  Confirms B's
  capacity advantage.
- **Time (CPU): B is 4.9–13.7× slower than C** — even after hoisting the per-pixel
  geometry out of the band loop (which helped: 256³ 7.4→4.8× at n=2).  The remaining
  cost is per-band DISPATCH: B makes `num_bands` jitted calls per pixel-batch, which
  STREAMING REQUIRES (one band at a time from the remote slice-owner).  Fusing the
  band loop into one jit would remove the dispatch but force holding all bands at
  once — killing B's only advantage (the streamed memory).  So B is structurally
  dispatch-bound on CPU and cannot be rescued without giving up streaming.

**DECISION (measured, 2026-06-13): adopt C; B is out.**  CPU is a target, and B's
~5–14× CPU time penalty for a ~13–23% transient-memory edge is a bad trade (Greg's
"can't absorb ~10× for 25%").  C's memory is bounded by pixel-batch (gather only
`(P_b, S)` per pixel-batch), so the sharding CAPACITY goal is preserved by the recon's
sharded STORAGE — the forward just gathers transiently.  **Consequences (simplifying):**
- The cone sharded **forward = per-pixel-batch all-gather of the slice cylinder +
  the existing MONOLITHIC forward**.  No forward banding, no windowed vertical fan,
  no accumulator.
- The B1 **forward banded kernel is no longer needed for production**
  (`forward_project_band_to_one_view` / `forward_vertical_fan_band_*`).  Keep it only
  if it still earns its place as a test helper for the back_band adjoint; otherwise
  remove it (back_band correctness is already gated by the back band-decomposition).
- **Back projection still bands** (the reduce-scatter — back's vertical OUTPUT is the
  small thing, no over-allocation); the B1 back banded kernel stays.
- The earlier window-vs-no-window / fused-accumulator debate is fully resolved by
  this: forward doesn't band at all.

GPU run (Greg) is now optional CONFIRMATION (does C's pixel-batched gather stay
bounded at 1024³? is B's edge ever big enough to reconsider a second, GPU-only path?
— unlikely given the single-path + CPU-target requirements), not a decision input.

NEXT: B2 (single-device + sharded driver: back on the banded kernel; forward = gather
+ monolithic; delete entries_per_cylinder_batch; memory/timing vs the §8a baseline).

---

## 6. Tests to BUILD (Greg's "tests at multiple points")

**Correctness (pytest — fast, CPU, in the suite):**
- `tests/geometries/test_projectors.py`: extend with **band-decomposition** and
  **adjoint-at-(g0,L)** for cone (B1).  These are the strong per-geometry gates and
  they run on every suite invocation.
- existing cone recon/fbp/vcd convergence gates: re-run each stage (regression net).
- `tests/sharding/`: cone sharded-vs-single-device + trivial-mesh + padded
  exact-zero (B4/B5), mirroring the ParallelBeam sharding tests.

**Memory + timing (scaling harness — isolated subprocess; CPU local + GPU cluster):**
- `cone_baseline_scaling.py` is the ruler; re-run at B2 (single-device) and B4
  (multi-device sweep).  Add a small **`--compare <baseline.yaml>` report mode** (or
  a `replot`-style helper) that prints time/peak DELTAS vs the recorded §8a baseline
  per (op,size), flagging >1.1× memory or >1.3× time as ⚠ (reported, not asserted —
  timing is machine-dependent; the hard gate stays correctness).
- `cone_fan_split_microbench.py` / `cone_channel_major_ablation.py` remain available
  for localized re-measurement if a stage moves the cost structure unexpectedly.

**Why memory/timing are reported, not hard-gated:** the project rule is exact
equality is never the gate for computed floats; likewise absolute time/memory are
machine- and occupancy-dependent (the swap-contamination and GPU-variance episodes).
So correctness is the hard gate; memory/timing are tracked against the recorded
baseline with guard-rails and human judgment ("comparable or better, not literal" —
Greg).

---

## 7. Risks / open questions for review

1. **B4 loop reorder (pixel-batch outer) is the main risk** — it changes the sharded
   driver's structure and the per-device transient bound.  Mitigation: the memory
   gate (per-device peak ~1/n_dev) catches a wrong bound; build B2's single-device
   horizontal-once first so the ordering is proven before adding the reduce-scatter.
2. **Curved detector:** the v→m (row) map is linear even when curved (curvature is in
   u/channels), so banding the vertical fan should be curvature-agnostic — verify
   against `geometry_xyz_to_uv_mag`/`detector_uv_to_mn` in B1's tests.
3. **Helical z-shift** enters z per-view (traced via `single_view_params`) — confirm
   the banded vertical handles it (it reads `helical_z_shift` from `single_view_params`
   today; unchanged).
4. **Hessian diagonal** (`coeff_power=2`) goes through the back banded kernel — the
   `coeff_power` arg threads through unchanged; B1's adjoint test should also cover
   `coeff_power=2` (it is a back-projection).
5. **B3 ordering:** de-closure after B4 (recommended) vs before — does settling the
   band structure first ease the module-level lift, or does lifting first ease B4?
   Lean: B4 first (band structure is the harder unknown).
6. **Delete-the-loser timing:** the monolithic cone kernel + `entries_per_cylinder_batch`
   are deleted at B2 (single-device) — confirm nothing else (FDK? `compute_horizontal_data`
   callers?) depends on them before deleting.
