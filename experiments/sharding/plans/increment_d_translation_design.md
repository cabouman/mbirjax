# Increment D — TranslationModel port: design note (2026-06-18, forward plan)

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
- **Row-sharding the sinogram** — a future exploration (`.claude/sinogram_sharding.md`); kept open by
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
| **T2 — banded back kernel + single-device rewire** | Add `back_project_one_view_to_band(…, g0, num_band_slices, coeff_power)` (anchor: coords from `recon_shape` params + global `k = g0 + arange(L)`; global clip `k_global < S_real`).  Rewire `back_project_one_view_to_pixel_batch` to horizontal-once + a **rolled** `lax.map` over slice bands; delete `entries_per_cylinder_batch` and swap its forward use (det-row batch) to a module constant. | B1 + B2 | New band tests (mirror `test_cone_banded`): full-band == monolithic (1e-5), adjoint-at-(g0,L), Hessian (coeff_power=2); `test_projectors`/`test_vcd` translation + anisotropic_translation green; allclose vs pre-change. |
| **T3 — forward anchor fix** | Fix `forward_vertical_fan_one_pixel_to_one_view`: `k_global = g0 + arange(L)`, source `S_real` from params (already centers on `num_recon_slices` — just the index), global clip.  Forward stays **monolithic**. | B1 forward anchor | Adjoint round-trip; full-band == monolithic at arbitrary (g0, L). |
| **T4 — flip `_supports_sharding()=True`** | The geometry-neutral base hooks drive it: back = banded reduce-scatter, forward = gather+monolithic+crop.  Add the GPU n=1 back short-circuit if measurement confirms the platform split (2b). | B4 + short-circuit | `tests/sharding/test_translation_sharded.py`: sharded == single-device (1e-5 single-shot, 1e-4 VCD), circular + anisotropic, at dividing counts; trivial-1-device-mesh == single-device. |
| **T5 — inert slice padding** | Almost free: the masks + forward crop are geometry-agnostic and the kernel anchor/clip landed in T2/T3.  Add a `TranslationModel` subclass to the test `_PaddedReconMixin`. | B5 | Padding mixin: sharded == single-device at non-dividing counts; forward/back device-form exact-zero; `test_forward_inert_to_nonzero_recon_padding`.  Include anisotropic_translation (a non-`num_det_rows` slice count). |

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
- **`tests/sharding/test_translation_sharded.py`** (mirror `test_cone_sharded`): back/forward/Hessian +
  a short VCD, sharded == single-device, circular + anisotropic.
- **`_PaddedReconMixin` subclass** for translation (T5).
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
