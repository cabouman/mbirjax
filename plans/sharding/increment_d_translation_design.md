# Increment D — TranslationModel port: design note (2026-06-18; ✅ COMPLETE 2026-06-19/20)

> **STATUS: T1–T5 DONE, CPU + GPU validated.**  TranslationModel is on the always-on placement path
> (sharded recon-by-slice / sino-by-view), correct at all device counts incl. padding, no single-device
> regression.  Lives in `greg/sharding_extensions` (rebased onto prerelease, PR-ready).  Per-stage
> record in the §3 table (each row marked ✅).  Post-port: scale-invariant sharded-test gate
> (`conftest.assert_sharded_allclose`) + dropped the redundant sharded VCD-recon test (`RUN_SHARDED_VCD`,
> §5).  **Open (GPU, Greg's cluster — not blocking):** the n=1 back short-circuit band-vs-pixel platform
> split for translation (decision 2b, UNMEASURED); per-device memory/time scaling.  **Prereq:**
> `translation` baselines in mbirjax_metrics (§6).  This doc is now the COMPLETED-WORK record for
> translation; **multiaxis is the next sub-effort** (its FBP fix in v3 §6 — NOT covered here).

*The staged plan to put `TranslationModel` on the always-on placement path (sharded recon-by-slice /
sino-by-view), mirroring the cone port.  Read `p6_increment_b_design.md` (the cone port) for the
techniques this reuses, and `sharding_implementation_plan_v3.md` §5 for where D sits in the overall
order.  Multiaxis is the **sibling** sub-effort of increment D and has its own open item (the FBP
angular-weighting fix) in v3 §6 — it is NOT covered here.*

---

## 0. Key finding — `TranslationModel` is `ConeBeamModel` *pre-port*

The two classes share the two-stage fan architecture, and translation is essentially where cone was
before increment B:

- Its `fdk_filter` is a near-verbatim copy of cone's *old* one: the same cosine pre-weight
  `weight_map = sdd/√(sdd²+u²+v²)`, the same `alpha = delta_det_row/(voxel_volume·M_0)`, the same
  trailing `*= π/num_views`, and it even reuses `ConeBeamModel.detector_mn_to_uv`.  It is in the
  pre-cleanup style (out-of-place `sinogram*weight_map`, `fftconvolve` + `lax.map` per row — the
  jax#27591 fragility).
- It still carries `entries_per_cylinder_batch = 128` (the instance attr B2 deleted from cone), used
  in **both** the back vertical fan (slice batching) and the forward vertical fan (det-row batching,
  `forward_vertical_fan_one_pixel_to_one_view` ~line 344).
- `_supports_sharding` is not overridden → inherits `False` → runs the legacy single-device path.
- Confirmed: the forward vertical fan has cone's B1 anchor bug — `k = jnp.arange(len(voxel_cylinder))`
  (band-local index) with z centered on `num_recon_slices` from params, so a length-L band would map
  to the wrong global z.  Needs the B1 fix (`k_global = g0 + arange(L)`, `S_real` from params, global
  validity clip).

**Consequence:** the cone increments map ~1:1, and the geometry-agnostic sharding infrastructure is
already in place (see §4), so the port is mostly **kernel + filter** work, not driver work.

---

## 1. Goal and non-goals

**Goal.**  `TranslationModel` runs on the always-on placement path: trivial 1-device mesh single-device,
shards multi-device (recon by slice, sino by view), correct at **all** device counts (inert padding),
with **no single-device regression**.  As with cone, the value is **capacity** (per-device memory
≈ 1/N so iterative recon can exceed one GPU), judged on correctness + memory-shards-1/N +
no-single-device-regression.

**Non-goals (out of scope here):**
- **Multiaxis** — the sibling D sub-effort; its FBP angular-weighting fix is decided-in-principle in
  v3 §6 and is separate from this port.
- **Kernel consolidation between cone and translation** — deliberately deferred (§2); the home for it
  is the v3 §6 "simplify the sparse-projector batching machinery" refactor, *after* both are ported.
- **Row-sharding the sinogram** — a future exploration (`plans/sharding/sinogram_sharding.md`); kept open by
  NOT consolidating the kernels (§2).
- The deeper "is `π/num_views` the right weight for a *non-rotational* geometry?" question — the port
  preserves current behavior; the question is parked alongside the multiaxis one (§7).

---

## 2. Design decisions (with Greg, 2026-06-18)

### 2a. Mirror cone; DEFER kernel consolidation
Port translation by mirroring the cone structure — give `TranslationModel` its **own** copies of the
banded kernels (with translation's geometry math) — and do **not** factor the banded kernels into a
shared cone/translation base as part of this work.  Rationale (Greg's row-sharding goal is the decider):
- **Row-sharding is per-geometry.**  Row-sharding the sinogram intervenes at the row-handling +
  driver-distribution layer — exactly what a shared banded kernel would couple.  Separate kernels keep
  "try row-sharding for one geometry" a local change.
- **Lower port risk.**  Porting and refactoring at once means the per-stage gates can't separate a
  port bug from a refactor bug.  Mirror first; each stage gates against translation's existing tests.
- **The true seam is clearer afterward**, and the cone kernels carry platform-specific perf structure
  (band vs pixel, the back-vertical cache cliff) a premature abstraction could obscure.

**Distinction that matters:** *using the already-shared infrastructure is NOT the contentious
consolidation.*  T1 should adopt the shared `_apply_direct_recon_filter`, and T4 the shared driver
hooks — both geometry-agnostic.  What is deferred is factoring the *per-view banded kernels* into a
common base.  So: **consolidate via shared infra (do it); duplicate the geometry kernels (defer the
base-class factoring to §6, after both geometries are ported).**

### 2b. Keep BOTH back kernels + a GPU n=1 short-circuit — but MEASURE
By analogy with cone, plan for translation to keep both a single-device **pixel** kernel
(`back_project_one_view_to_pixel_batch`) and a **band** kernel (`back_project_one_view_to_band`), with
the GPU n=1 back short-circuit routing single-GPU recons to the pixel kernel.  The need is *expected*
(same fan structure → likely the same platform split: band ~slower on GPU, ~faster on CPU via the
back-vertical cache cliff) — but it must be **measured** for translation, not assumed, before finalizing
the short-circuit.

### 2c. Forward = monolithic gather (cone decision C, inherited)
The sharded forward uses the geometry-neutral base hook (per-pixel-batch all-gather + monolithic
forward + the inert-padding crop) — **no banded forward kernel**.  The only forward change is the
anchor fix (T3).

---

## 3. Staged plan (T1–T5), each landing single-device-green for review

Hard gate = **correctness** (tight `allclose`; never exact for computed floats).  Memory/time are
**reported** vs baseline (machine-dependent), via the metrics harness once translation is added there
(§6).

| Stage | Work | Maps to (cone) | Correctness gate |
|---|---|---|---|
| **T1 — FDK filter** ✅ DONE 2026-06-19 | Convert `fdk_filter` to call `self._apply_direct_recon_filter(filter_name, filter_scale=alpha, output_sharded=…, row_weight=weight_map)` (identical to cone, but keeping `mj.ConeBeamModel.detector_mn_to_uv` — translation has no detector map of its own).  Removes the out-of-place weight/π multiplies + the `fftconvolve`/`lax.map` fragility, and yields the per-view-shard sharded contract (dormant: translation's `is_sharded` is `False` so the shared method takes its single-device `else` branch — value-identical).  Added the equispaced-angle caveat note to `fdk_recon` (as on parallel/cone). | cone FDK cleanup | **`fdk_filter`/`fdk_recon` allclose vs pre-change** (1e-5) via a transient old-vs-new ablation — DONE, worst rel-diff 2.9e-7 across both translation geometries.  Component gating thereafter: filter via the mbirjax_metrics `.npy` baseline (§6), projectors via `test_projectors`/`test_vcd`.  **translation is NOT in `test_fbp_fdk`** (FDK on a limited-angle TCT geometry has no meaningful recon-NRMSE tolerance — a NOTE in that file now records why).  **No sharding flip.** |
| **T2 — banded back kernel + single-device rewire** ✅ DONE 2026-06-19 | Added `back_vertical_fan_band_one_pixel` / `back_vertical_fan_band_pixel_batch` / `back_project_one_view_to_band(…, g0, num_band_slices, coeff_power)` (anchor already on `recon_shape` params via `compute_vertical_data_single_pixel`; global `k = g0 + arange(L)`; global clip `k_global < S_real`).  Rewired `back_project_one_view_to_pixel_batch` to horizontal-once + a **rolled** `lax.map` over slice bands (module const `TRANSLATION_SLICE_BAND_SIZE=128`, `slice_band_size` test hook), deleting the monolithic `back_vertical_fan_one_view_to_{pixel_batch,one_pixel}`.  Deleted `entries_per_cylinder_batch` (instance attr + geometry-param namedtuple field) and swapped its forward det-row use to `TRANSLATION_FORWARD_DET_ROW_BATCH=128`.  Forward **anchor left for T3** (still monolithic/full-cylinder, so the band-local-index bug is unreachable).  Band driver auto-wires (`getattr(model,'back_project_one_view_to_band')` in projectors.py). | B1 + B2 | **New `tests/geometries/test_translation_banded.py`** (mirror `test_cone_banded`, isotropic + anisotropic, nonzero-z views): production rolled-band == explicit non-uniform band-concat across band sizes incl. non-divisor crop + coeff_power 1&2; `sparse_back_project_band` == full back sliced to the band.  `test_projectors`/`test_vcd` translation + anisotropic green; **full suite 175p/2s @4 CPU dev**.  **allclose vs pre-change: BIT-IDENTICAL (0.0)** at the production band size (single band → byte-identical to the old single-chunk path), transient ablation. |
| **T3 — forward anchor fix** ✅ DONE 2026-06-19 | Fixed `forward_vertical_fan_one_pixel_to_one_view` to source the slice count from params (`num_slices = num_recon_slices`, `k = arange(num_slices)`).  The z-map already centered on `num_recon_slices`; the **inverse map `k_m` and the validity clip centered on the INPUT length** (`voxel_cylinder.shape[0]`) — a latent params-vs-input-length inconsistency.  It is benign today (the batch method **asserts** the per-pixel cylinder is `num_recon_slices` long, so input-length == params), so the fix is **bit-identical** now; its value is removing the inconsistency and making the clip a GLOBAL test (`k < S_real`) — the inert-padding anchor for T5.  Forward stays **monolithic** (no `g0` param; g0=0 for the full cylinder). | B1 forward anchor | **Forward bit-identical (0.0)** vs pre-change (transient ablation); adjoint round-trip green (`test_projectors` translation + anisotropic); `test_translation_banded` + `test_vcd` green; **full suite 175p/2s @4 CPU dev**. |
| **T4 — flip `_supports_sharding()=True`** ✅ DONE 2026-06-19 | Added the one-line `_supports_sharding()` override (returns True) — **the only change needed**; translation adds NO projector/driver overrides (it uses the geometry-neutral base hooks: back = banded reduce-scatter via the T2 `back_project_one_view_to_band`, forward = gather+monolithic+crop).  The GPU n=1 back short-circuit ([tomography_model.py:1952]) is **geometry-agnostic and already active** for translation (routes a single-GPU mesh to the pixel kernel) — correctness is metadata-only-wrapped; the *performance* rationale (band-vs-pixel platform split, decision 2b) still needs **GPU measurement (Greg's cluster)**, not assumed. | B4 + short-circuit | **New `tests/sharding/test_translation_sharded.py`** (mirror `test_cone_sharded`, isotropic + anisotropic): back/forward/Hessian single-shot, sharded == single-device at dividing counts (n=2,4), trivial-1-device-mesh via the per-geometry gates.  *(The short sharded VCD recon originally added here was REMOVED 2026-06-19 as redundant — the sharded VCD loop is geometry-independent, gated on parallel + cone; see §5.)*  all comparisons use a **scale-invariant relative-max gate** `max\|out−ref\|/max\|ref\| ≤ TOL` (1e-5 projectors / 1e-4 VCD), NOT a fixed atol.  *Why (a ruler finding):* the sharded-vs-single difference is XLA reduction-ORDER noise and is **process-nondeterministic on CPU** — usually exactly 0, occasionally ~1e-7 of the peak (identical across device counts within a process → a per-process compile/autotune choice, not per-call reorder).  A fixed `atol=1e-5` is therefore both scale-fragile AND flaky: it false-fails the large-magnitude Hessian (peak ~5e3 → noise ~5e-4 ≫ atol) and is even marginal for back/forward (peak ~140–200), while silently passing small-magnitude cone (peak ~1.8).  The relative-max gate clears the worst case with ~100× margin and still catches a real bug (which gives rel_max ~O(1)).  **Full suite 179p/2s @4 CPU dev.**  *(Cone's `test_cone_sharded` still uses a flat 1e-5 — safe only because cone's values are O(1); a consistency retrofit to the same gate is an optional separate change.)*  **Bonus finding:** the B5 padding infra is geometry-agnostic, so translation is ALREADY correct at non-dividing (auto-padded) counts — the existing `test_projectors`/`test_vcd` auto-shard anisotropic to 3 devices (5→6 padded) and pass, with no helical-z-weight-style per-slice seam to fix.  ⇒ T5 is mostly just the dedicated padding test. |
| **T5 — inert slice padding** ✅ DONE 2026-06-19 | **Zero production changes** (as predicted): the masks + forward crop are geometry-agnostic and the global validity clips landed in T2 (back) / T3 (forward), so translation padding was already inert at the T4 flip (the existing `test_projectors`/`test_vcd` auto-shard anisotropic to a padded count and pass).  **Test-only:** added `TestPaddedSlicesTranslation(_PaddedReconMixin)` — isotropic + anisotropic (voxel_slice_aspect=2.9) variants, both z-range-tuned to a **prime 7-slice** count (pads at every device count > 1; a `test_prime_slice_count` guard catches auto-sizer drift), `PADS_ROWS=False` (cone-like). | B5 | **`tests/sharding/test_padding.py::TestPaddedSlicesTranslation` — 6 tests / 8 subtests green @4 CPU dev:** projectors+Hessian sharded == single-device at padded n=2,3,4; forward/back device-form exact-zero (both directions); `test_forward_inert_to_nonzero_recon_padding` (poison the padded slices → forward bit-identical); `test_fully_padded_trailing_shard` (tiny 3-slice on 4 dev → the `n_valid<=0` branch).  Full suite green. |

De-closuring (cone B3) is **already done** for translation — `make_geometry_params` is shared and the
module-level `_sparse_*` take the per-view kernel as a static arg, so translation's kernels already
share the jit cache.  No separate stage.

---

## 4. Already done / free (geometry-agnostic infrastructure)

The cone/ParallelBeam work made these geometry-neutral, so translation gets them at the T4 flip:
- **Driver hooks** — `_back_project_view_shard_to_band` (base = the banded path the band kernel feeds)
  and `_forward_project_to_view_shards` (base = gather+monolithic + the inert-padding crop).  Translation
  uses the base versions as-is (ParallelBeam is the only override).
- **Padding** — `_mask_padded_slices` / `_mask_padded_views`, the entry zero-fill, the forward-gather
  crop, and `Placement.padded_size`.  All keyed on the one predicate `k_global < real_count`.
- **qGGMRF** — the interface mask + halo machinery is geometry-agnostic; the VCD loop is too.
- **Tests** — `_PaddedReconMixin` (in `tests/sharding/test_padding.py`) reads all real sizes from
  params, so a `TranslationModel` subclass is a few lines.

---

## 5. Tests to build

- **`tests/geometries/test_translation_banded.py`** (mirror `test_cone_banded`): band-decomposition,
  adjoint-at-(g0,L), Hessian; self-contained (production uniform-band `lax.map` == explicit
  non-uniform-band concat), so it survives the monolithic-kernel deletion.
- **`tests/sharding/test_translation_sharded.py`**: back/forward/Hessian-diagonal single-shot
  sharded == single-device, isotropic + anisotropic.  **NO sharded VCD-recon test** (decision
  2026-06-19, supersedes the original "mirror `test_cone_sharded` + a short VCD"): the sharded VCD
  LOOP (reduce-scatter back / gather forward / halo qGGMRF prior / partitioning / donation) is
  geometry-INDEPENDENT and is gated sharded by **parallel + cone**; translation adds no loop
  overrides and its sharded projectors are gated by the single-shot tests, so a per-geometry sharded
  recon only re-runs shared machinery.  (Same rationale as `tests/geometries/test_vcd.py` gating full
  convergence on parallel + cone only; also removes the tiny-recon partition-granularity warning at
  the root.)  Multiaxis follows the same lean pattern.
- **`_PaddedReconMixin` subclass** for translation (T5), with **`RUN_SHARDED_VCD = False`** (a mixin
  flag, default True for parallel/cone): skips the mixin's sharded VCD-recon checks (the
  geometry-independent iterated-loop padding inertness, gated on parallel + cone) while keeping all
  the single-shot padding checks — projectors/Hessian, forward/back device-form exact-zero, the
  poison-the-padding forward-inert test, and the fully-padded-shard branch.
- **Regression net:** keep `test_projectors` (adjoint/Hessian) and `test_vcd` (convergence/sanity)
  green for `translation` and `anisotropic_translation` at every stage — these gate the projectors
  single-device.  **`test_fbp_fdk` does NOT cover translation** (corrected 2026-06-19 — the earlier
  "they already gate single-device" was wrong; `test_fbp_fdk` explicitly removes both translation
  geometries because FDK on a limited-angle TCT geometry has no meaningful recon-NRMSE tolerance; a
  NOTE in that file records the exclusion + where the components are gated).  `fdk_filter`
  correctness is gated by the mbirjax_metrics per-component `.npy` baseline (§6, captured from
  current `main` — itself pre-port — so the harness comparison doubles as an old-vs-new gate); each
  stage's conversion is additionally proved value-identical by a transient old-vs-new allclose
  ablation at change time.

---

## 6. Metrics / baseline (Greg + a separate session — prerequisite for tracking the port)

`translation` is **not** in the mbirjax_metrics harness geometries yet (`[parallel, cone]`).  To track
the port's correctness / memory / time it must be added, with baselines captured, mirroring the cone
cells — including a **dividing/non-dividing size pair** (like the cone `128/129` and `512/513`) so the
padded path is exercised, and an anisotropic-translation size.  Greg is handling this in a separate
session; it may need a dashboard redesign.  This is a prerequisite for the memory/time half of the
per-stage gates (correctness is gated by the unit tests regardless).

---

## 7. Risks / open questions

1. **Band-vs-pixel platform split (2b)** — expected but must be measured for translation before
   finalizing the dual-kernel + GPU n=1 short-circuit.
2. **`entries_per_cylinder_batch` is used in two places** (back slice-batch + forward det-row batch) —
   the B2 cleanup must swap both to module constants; don't delete the attr until both are rerouted.
3. **`π/num_views` for a non-rotational geometry** — translation already uses it (copied from cone), so
   the port *preserves* current behavior.  Whether it's the right normalization for a translation scan
   is the same class of question as multiaxis (v3 §6) and is parked there; not a blocker.
4. **anisotropic_translation** (`voxel_slice_aspect`) gives a slice count ≠ `num_det_rows`, so it
   exercises the anchor/clip and the padding more than the isotropic case — keep it in the gates.
5. **Curved detector / small-angle assumption** — translation shares cone's `|phi_p| < 45°` assumption;
   confirm the banded vertical fan is curvature/angle-agnostic in the band tests (as cone did).
