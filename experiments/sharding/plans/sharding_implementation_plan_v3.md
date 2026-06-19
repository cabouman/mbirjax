# Sharding implementation plan v3 — current state + forward plan

*Created 2026-06-17.  This is the **primary forward-plan / orientation doc**: read it
first for where the sharding work is now and where it goes next.  It supersedes
`sharding_implementation_plan_v2.md` for forward planning (v2 is retained as a
**detailed-design archive** — the placement / `(g0,L)` / anchor design rationale).
`sharding_implementation_plan.md` (v1) is the **completed-work record + cross-cutting
principles + verified hardware facts**.  `sharding_status.md` is the **living handoff
log** (its TOP handoff is the latest session state).*

**Scope of this doc:** the library sharding implementation (ParallelBeam + the cone 
port + the remaining geometry ports + the retirement cascade).  The performance-regression
nightly harness + the metrics-visualization surface are a **separate track** — see
`performance_tracking_plan.md` and `sharding_status.md`.

---

## 0. Doc map (which doc is authoritative for what)

| Doc | Role |
|---|---|
| **this file (v3)** | current state + what's left + execution order — read first |
| `sharding_status.md` | living handoff log; TOP handoff = latest session |
| `sharding_implementation_plan_v2.md` | **archive**: placement architecture + `(g0,L)`/anchor/forward-accumulation design rationale |
| `sharding_implementation_plan.md` (v1) | completed Phases 0/A/B/F1/D/F2/C/E; **cross-cutting principles**; verified **hardware facts**; O1–O4 |
| `p6_increment_b_design.md` | **authoritative for the cone port** — increment-B staged plan + progress (B1–B5) |
| `p6_projector_rework_proposal.md` | projector-rework design; **§8a-design is canonical** (rest partly superseded) |
| `performance_tracking_plan.md` | nightly perf-regression harness + metrics — **separate track**, out of scope here |
| `.claude/lessons.md` | jax/GPU/placement/measurement playbook |
| `.claude/back_projection_overview.md` | projector internals (read before touching cone kernels) |
| `.claude/sinogram_sharding.md` | parked row-sharding-the-sinogram exploration |

*File:line references below are a 2026-06-17 snapshot and may drift; trust the symbol
name over the number.*

---

## 1. Status at a glance

| Area | State |
|---|---|
| **ParallelBeam** — filter, forward, back, qGGMRF, VCD, device-UX, inert padding | ✅ **fully sharded**, multi-GPU validated 1–4 GPU (8-GPU scaling at 1024³/1800³/2048³) |
| **Cone (P6)** — A, B1, B2, B3, B4, shared FBP/FDK filter, GPU n=1 back short-circuit | ✅ done, CPU-validated end-to-end + GPU-confirmed |
| **Cone B5** — exactly-inert slice padding | ✅ **done** (2026-06-18; CPU-validated — 4 deferred tests pass at 4 devices, full suite green at the default 4 CPU devices; GPU-confirmed in the nightly — the 4 non-dividing cone cells `513x449x385` flipped failing→ok on H100) |
| **Cone B4.5** — band kernel GPU cost | ⏸ open (deferred behind decision (a); see §4) |
| **C** — ParallelBeam on the `(g0,L)` template; FDK filter sharded; both cone back kernels kept | ✅ **substance done** (landed interspersed with B4/B5: polymorphic override-dispatch template, FDK filter per-view-shard, parallel overrides **kept** by decision 2026-06-18 — cheaper for parallel; no separate code phase) |
| **D** — translation + multiaxis ports | ⬜ pending implementation, tests, and tuning |
| **E** — retirement cascade (legacy single-device dispatch, `main_device`, `view_indices`, …) | ⬜ pending implementation, tests, and tuning (last — needs all geometries ported) |

---

## 2. Architecture in one screen

- **Uniform axes.**  Sinogram-like arrays shard **by view** (axis 0); recon-like arrays
  shard **by slice**.  The sinogram is **stationary**; the recon **moves** — forward
  **all-gathers** voxel cylinders to view-owners, back **reduce-scatters** partial
  cylinders to slice-owners.
- **Placements are always on.**  `recon_placement` / `sino_placement` (each a mesh +
  sharded axis) replace the old `main_device` / `sinogram_device` scalars.  Single-device
  is the **degenerate** case — a trivial 1-device `NamedSharding` mesh — so there is
  *one* code path.  `is_sharded` = `self.mesh is not None` (almost always True now).
  `_supports_sharding()` gates whether a geometry takes the placement path
  (**ParallelBeam + cone → True**; translation/multiaxis still False until ported).
- **Threading (ThreadPoolExecutor), not `shard_map`.**  Manual collectives over `addressable_shards`
  via the `run_per_device` threading helper and a single transfer primitive `move_shard`
  (empirical d2d-safe probe with a host-bounce fallback — see the L40S hazard in v1).
- **Output contract (P5).**  User-facing methods take an explicit `output_sharded=False`:
  default returns a plain array in the problem's **real** shape (gather + crop); `True`
  returns the internal device form.  Inputs are auto-detected.  Internal methods are
  sharded-only.  (The earlier "match-input" contract was retired at P5 Step 4.)
- **Exactly-inert divisibility padding.**  Auto uses *all* devices; a non-dividing axis is
  padded and the padding is **provably inert** — zero-fill at entry, validity masks on
  projector outputs, the qGGMRF interface mask, and the kernel global validity clip, so
  results are independent of device count and pad amount.  **Params answer "what is the
  problem?"** (`get_params` shapes are always real); **placements own the padded device
  shape** and each shard's global index range.  Every padding mask is the one predicate
  `k_global < real_count`.

---

## 3. ParallelBeam milestones (done)

The five components, each with its load-bearing technique.

### 3.1 Filter (F1) — view-sharded, zero cross-device comms
- **`apply_row_filter`** ([tomography_utils.py:14](mbirjax/tomography_utils.py:14)) is
  **row-batched**: a `lax.scan` over overlapping `ROW_FILTER_BATCH=1024` row windows,
  writing each window **in place** with `lax.dynamic_update_slice`, and clamping the last
  window start to `total_rows - batch` so the **tail batch overlaps** the previous one
  (safe — per-row filtering is idempotent).  Peak memory ≈ input + output (~2× a shard),
  independent of divisibility or device count.
- **`_apply_direct_recon_filter`** ([tomography_model.py:2621](mbirjax/tomography_model.py:2621))
  filters each device's own view-shard **on-device** via `run_per_device` + `assemble_sharded`
  — no cross-device movement.  The filter is folded into a tiny f32 array (no full-sinogram
  multiply; avoids `np.pi`-style f64 promotion).  **Shared** by FBP (parallel) and FDK
  (cone, with a cosine `row_weight`) — `ParallelBeamModel.fbp_filter`
  ([parallel_beam.py:442](mbirjax/parallel_beam.py:442)) delegates.

### 3.2 Back projection (D) — banded reduce-scatter
- View-sharded sino → slice-sharded recon.  Each view-owner back-projects its local views
  onto a **global slice band `[g0:g1)`**; `sum_band_to_owner`
  ([transfer.py:103](mbirjax/_sharding/transfer.py:103)) reduce-scatters the per-device
  partials onto the band's slice-owner; bands are concatenated into each owner's shard.
  Driver: `_sparse_back_project_sharded` ([tomography_model.py:1852](mbirjax/tomography_model.py:1852)),
  `_back_project_all_bands` (:1998).
- **Band sizing** `_slice_band_length` (:2127) = `min(`reduce-gather bound
  `slices_per_dev/n_dev`, compute cap `_BACK_PROJECT_MAX_BAND_WORK/num_pixels)`, floored —
  so even **n=1 streams**; `_balanced_slice_bounds` (:2186) tiles into the fewest balanced,
  non-overlapping bands.
- **ParallelBeam specialization:** a detector-**row crop** `[g0:g1)` (row r ↔ slice r),
  cheaper than the general banded kernel the base/cone uses.
- **GPU n=1 short-circuit (shared dispatch, cone-motivated):** a single-GPU mesh routes to the
  single-device kernel, wrapped as a 1-shard slice-sharded array; CPU keeps the band path.  The
  condition is geometry-agnostic, but for ParallelBeam it is ~neutral (its band path is already
  the cheap row-crop above) — the ~2.25× win is the **cone** case (§4).

### 3.3 Forward projection (C) — banded all-gather (the adjoint)
- Slice-sharded recon → view-sharded sino.  `broadcast_band_to_views`
  ([transfer.py:130](mbirjax/_sharding/transfer.py:130)) sends each slice band to all
  view-owners, each forward-projects its views, and the row-bands concatenate (parallel:
  row r ← slice r).  Driver: `_sparse_forward_project_sharded`
  ([tomography_model.py:1431](mbirjax/tomography_model.py:1431)), `_forward_project_all_bands` (:1647).
- **Correctness gate** = the forward/back **adjoint round-trip** `⟨Ax,y⟩ = ⟨x,Aᵀy⟩`, green
  at 1/2/4/8 devices.

### 3.4 qGGMRF prior (E) — per-owner local prior + halo exchange
- Each slice-owner computes the prior on its own shard; only the boundary slices ("halos")
  move.  **`stage_halos`** ([qggmrf.py:338](mbirjax/qggmrf.py:338)) extracts + pre-places
  halos **once per partition pass** (not per subset — per-subset halo reads were the
  multi-GPU scaling cap).  Reflected BC = `delta = 0` at a true edge.  Driver:
  `qggmrf_gradient_and_hessian_sharded` (:373), kernel `..._at_indices` (:71).
- **Inert padding:** an **interface-delta-mask** `interface_mask[j] = (g0 + j < S_real)`
  zeros padded interfaces (reflected BC relocated to the boundary), giving a padded-slice
  Hessian that stays positive and a gradient of exactly 0 — so `δ = −0/positive = −0.0`,
  no `jnp.where`.  Masks are built per layout and **cached**
  (`_qggmrf_interface_masks`, [tomography_model.py:847](mbirjax/tomography_model.py:847)),
  invalidated on recompile.

### 3.5 VCD loop (E) — on-device, donation, leak-free
- **Init chain stays sharded:** `direct_recon(output_sharded=True)` →
  `forward_project(output_sharded=True)`; no host gather into the loop.
- **In-place updates via buffer donation:** `update_recon`
  ([tomography_model.py:3618](mbirjax/tomography_model.py:3618)) is a donated-carry
  `.at[idx].add`; `update_error_sinogram` (:3623) is `@jax.jit donate_argnames` computing a
  fused `error - alpha*delta` (no separate scaled-delta transient).
- **The sharded-memory leak fix (load-bearing lesson):** jax keeps `NamedSharding` arrays in
  internal **reference cycles**, so they free on the **cyclic GC**, not refcount.  The old
  out-of-place per-subset error-sino update piled up one full view-sharded sinogram per
  subset until GC → peak scaled with subsets×passes.  Fix = the donated in-place FMA above
  + explicit `.delete()` of the eager-op transients (`delta_sinogram`, and
  `weighted_error_sinogram` for non-const weights) after a one-line `block_until_ready`.
  (Diagnosed empirically via a memjump trace; the recon side never leaked because it already
  donated — that asymmetry was the tell.)
- **Cross-mesh `alpha`:** forward scalars are reduced on the sino mesh, the line search runs
  on the recon mesh (replicate the scalar), then `alpha` is replicated back to the sino mesh
  to scale the sharded delta; `recon_indices` is replicated to keep pixel-axis gathers local.
  Driver: `vcd_subset_updater` (:3191), `vcd_partition_iterator` (:3104, stages halos once/pass).

### 3.6 Device-config UX + padding (P5)
- `configure_devices(None|int|list)` (auto = largest count dividing `num_slices`, skipping
  any count whose last shard would be fully padded); always-on "**N × PLATFORM (sharded)** +
  why-N" report at recon time.
- Exactly-inert divisibility padding (Stage 0 contract / Stage 1 views / Stage 2 slices) —
  ParallelBeam done & **GPU-validated** (1024×1023×1024 on 2×H100: NRMSE 2.6e-7, 2.18×,
  per-device peak halves, host-RSS delta 0).

### 3.7 Validation summary
- Full CPU suite green; multi-GPU 1–4 GPU validated; 8-GPU scaling at 1024³/1800³/2048³;
  VCD 1008³ 1d→4d **super-linear** (4.49×), memory shards cleanly, correctness vs prerelease
  ~8.8e-7.

---

## 4. Cone port (P6) — in progress

**The value is CAPACITY, not single-GPU speed.**  Cone projectors already scale ~N⁴ on GPU
with no cliff and fit 1024³ single-device; sharding is what lets *VCD* exceed one GPU at
1024³+ (the measured wall).  So cone is judged on correctness + memory-shards-1/n_dev +
no-single-device-regression.

### Done (CPU-validated; GPU-confirmed where noted)
- **A — channel-major horizontal fans.**  CPU win (~13× forward-horizontal at 256³),
  **GPU-neutral** (the cross-run "~2×" was variance).
- **B1 — banded cone kernels.**  `back_project_one_view_to_band` + banded vertical fans;
  **anchor rule** (z from `recon_shape` params + global `k = g0 + arange(L)`, not the input
  length); **global validity clip** `k_global < S_real` (the inert-padding hook).
- **B2 — single-device cone on the banded kernel.**  A *rolled* `lax.map` over
  `CONE_SLICE_BAND_SIZE=128` slice bands (compile size independent of slice count);
  `entries_per_cylinder_batch` **deleted**; memory- and time-neutral on both platforms.
- **B3 — de-closuring.**  Projector drivers lifted to module-level jitted functions, so the
  jit cache is **shared across model instances** (sweeps / the vcls sibling stop re-tracing).
  Root blocker fixed *at source*: `get_geometry_parameters` minted a fresh `namedtuple` class
  per call → distinct treedefs → cache misses; now `ParameterHandler.make_geometry_params`
  caches the class by field names.
- **B4 — sharded cone driver.**  Back = banded reduce-scatter (the B1 kernel via a
  geometry-neutral base hook; ParallelBeam overrides with its row-crop) **for n≥2 and CPU n=1**;
  at **GPU n=1 it short-circuits to the monolithic single-device kernel** (next bullet), so the
  banded reduce-scatter is never the GPU single-device path.  Forward = **decision C**:
  per-pixel-batch all-gather + monolithic (the banded/streamed forward measured 5–14× slower on
  CPU for ~13–23% memory).  `_supports_sharding()=True` for cone.  CPU-validated end to end at
  dividing counts; **GPU-confirmed** (B4.4: per-device peak ~1/n_dev + the 1024³ VCD capacity win
  — and the run that *surfaced* the band kernel's GPU back-time penalty → the short-circuit + B4.5).
- **Shared FBP/FDK filter** (cone delegates with a cosine `row_weight`); **GPU n=1 back
  short-circuit** (platform-divergent kernel; GPU-confirmed: 2.25–2.46× faster, back = main).

- **B5 — exactly-inert slice padding for cone.  ✅ DONE (2026-06-18, staged; CPU-validated).**
  The 4 deferred failures pass at 4 devices and the full suite is green at the default 4 CPU
  devices.  All four collapsed to ONE root cause: the cone sharded forward GATHERS the device-form
  (padded) cylinder and feeds the monolithic kernel, which anchors on / asserts the REAL slice
  count.  Fixes: **(1) load-bearing —** `_forward_project_to_view_shards` crops the gathered
  cylinder to `recon_placement.real_size` before the monolithic forward (padded slices are zero →
  EXACT).  **(2) test contract —** `sparse_back_project` STAYS device-form (the VCD loop /
  `output_sharded` need a shardable padded array; cropping to a non-dividing real count would break
  the sharding), so the geometry tests `verify_adjoint`/`verify_hessian` crop the device-form back
  projection to real slices — a no-op without padding.  This was a **latent, geometry-agnostic** test
  bug, not cone-specific: parallel fails identically at 3 devices (40→42), hidden only because 40
  divides 2/4.  **(3) `_supports_slice_padding()` hook DROPPED** (Greg): after B5 both sharding
  geometries support padding, so it would be all-True/dead.  **Bonus real bug** (found by the new
  helical padding test): `ConeBeamModel.helical_fdk_z_weight` built its per-slice z-weight at the
  REAL slice count and applied it to the device-form (padded) recon → broadcast crash; now built at
  the device-form length, anchored on real, padded slices forced to 0 weight (guards `0*inf→NaN`).
  Also `verify_adjoint` now zeros the random `y`'s padded views (the back projector needs padded
  views zero; a hand-built device-form `y` violated that → ~3% adjoint gap at view-padding counts).
  Tests: `tests/sharding/test_padding.py::TestPaddedSlicesCone` (circular+helical: projectors/
  Hessian/VCD sharded==single-device + forward/back device-form exact-zero); parallel exact-zero
  test extended to the forward direction.  GPU confirmation at a non-dividing slice count pending
  (CPU `MBIRJAX_NUM_CPU_DEVICES=4` is the proxy).  Details: `p6_increment_b_design.md` B5 entry.

### Open / next
- **B4.5 — band-kernel GPU cost (deferred).**  The band (reduce-scatter) kernel is **~2.25×
  slower than the pixel kernel ON GPU** (and the two are **platform-opposite**: CPU loves the
  band kernel, GPU loves the rolled-pixel kernel via the ×62 back-vertical cache cliff).  So
  multi-device back doesn't pay in *time* until ≥3 GPUs (crossover n≈2.25); VCD stays monotonic
  only because the forward parallelizes and masks it.  The real challenge is a kernel fast on
  *both*, else platform-specific selection (what the n=1 short-circuit already does).  Deferred
  behind **decision (a)** (2026-06-16: keep the simple GPU n=1 short-circuit rather than a
  memory-aware variant) — sharding is the capacity tool beyond 1024³, not a back-time lever.
  *Alternative axis to consider:* sharding the sinogram by **detector row** instead of by view,
  so the sino's sharded axis aligns with the recon's slice sharding — turning back projection
  into a mostly-local op (a geometry-driven footprint halo) rather than a view-reduce, which
  would sidestep the band-kernel reduce-scatter cost entirely.  Parked exploration; full
  analysis in `.claude/sinogram_sharding.md`.
- **FDK filter → sharded contract — ✅ DONE** (landed interspersed with B4/B5).
  `ConeBeamModel.fdk_filter` now uses the shared `_apply_direct_recon_filter` (the `fbp_filter`
  pattern: per-view-shard `run_per_device`, the FDK cosine pre-weight folded into `row_weight`),
  and `fdk_recon` stays sharded throughout (`_shard_sinogram` → `fdk_filter(output_sharded=True)`
  → `back_project(output_sharded=True)`) — no gather, no single-device init bottleneck.

---

## 5. Remaining work & execution order

1. **Cone B5** (inert padding) — ✅ **DONE 2026-06-18** (CPU-validated; cone correct at *all* device
   counts, the 4 tests pass, full suite green at the default 4 CPU devices; GPU-confirmed in the
   nightly — the 4 non-dividing cone cells `513x449x385` flipped failing→ok).  **D is now next**
   (C's substance landed interspersed with B4/B5 — see item 2).
2. **C — substance DONE, landed interspersed with B4/B5** (no separate code phase; the deviations
   that did parts of C happened alongside earlier items).  What C originally scoped, vs reality:
   - *ParallelBeam on the `(g0,L)` banded template* — done: the back/forward sharded drivers
     dispatch through polymorphic hooks (`_back_project_view_shard_to_band` /
     `_forward_project_to_view_shards`); the base is the geometry-neutral banded/gather path (cone),
     and ParallelBeam **overrides** with its row-crop back + banded-broadcast forward.  The
     transitional "geometry has banded kernels?" conditional is gone — it *became* this override
     dispatch in B4.
   - *ParallelBeam's overrides are **KEPT*** (decision 2026-06-18): the row-crop back-projects only
     detector rows `[g0:g1)` (= the slice band) and the banded forward never gathers the cylinder —
     both cheaper / more memory-bounded for parallel than the general path, so unifying onto the
     general template would be a parallel regression for pure code-simplicity.  The override pattern
     already gives "one template, one geometry-specific seam each."
   - *FDK-filter sharded contract* — done (see §4).
   - **Both cone back kernels stay**: the GPU n=1 back short-circuit (`_sparse_back_project_sharded`)
     routes single-GPU recons to `back_project_one_view_to_pixel_batch` (~2.25× faster on GPU than
     the band kernel; B4.5).  band = CPU + multi-device reduce-scatter, pixel = single-GPU —
     platform-complementary, so F1's "delete the loser" does **not** apply to cone back.  (E must
     likewise keep the single-device back driver/kernel.)
3. **D** — translation + multiaxis ports (same template; note `MultiAxisParallelModel` extends
   `TomographyModel` directly → it gets its own `_supports_sharding` flip).  **Settle the multiaxis
   FBP-filter fix with this port** — see §6 "Multiaxis FBP angular weighting" (decided in principle,
   implementation deferred).
4. **E — retirement cascade** (only after *all* geometries are ported): `main_device` /
   `sinogram_device`; the `is_sharded` else-branches + the legacy single-device *dispatch* —
   but **keep** the single-device back driver/kernel (`_sparse_back_project_single_device` /
   `back_project_one_view_to_pixel_batch`), which the GPU n=1 short-circuit calls; the
   `view_indices` machinery (incl. the test_projectors pin); `initialize_recon`'s early
   `device_put` block; `compute_hessian_diagonal`'s `output_device`; then a
   `grep -rn "RETIRE-*"` sweep (e.g., the trivial-mesh comparison tests retire).
5. **Post-P6** — the multi-GPU **user docs page** (use the word "sharding" sparingly);
   the choose-N-vs-communication policy (+ the CPU-cluster auto policy); **B4.5** if
   multi-device back *time* ever matters; revisit the prox-map prior under sharding if a
   PnP-at-scale need appears.

**Porting footgun for C/D (the lesson from the B5 cone port):** the place a new geometry breaks
under sharding is wherever a **per-slice or per-view operation assumes the problem's REAL count but
is handed the device (padded) form.**  That single shape is exactly what bit the two cone seams in
B5 — the gather-forward (the monolithic kernel asserts `recon_shape[2]`, the real slice count) and
`helical_fdk_z_weight` (a per-slice weight built at the real slice count, multiplied against the
padded device-form recon → broadcast crash).  **Front-run it when porting:** grep the geometry's
projector *and* its init/filter paths (FBP/FDK, any per-slice or per-view weighting) for
`recon_shape[2]` / `sinogram_shape[0]` (or `x.shape[axis]`) used as a **loop bound, a reshape
target, or an `arange`/weight length**, and reconcile each against the device-form length — crop to
the real count (forward gather), or build at the device-form length but anchor the physics on the
real count and mask the padded tail (the z-weight).  The padding *infrastructure* (entry zero-fill,
`_mask_padded_views` / `_mask_padded_slices`, the `k_global < real_count` clip) is already
geometry-agnostic, so the only new work per geometry is finding and fixing its own non-inert seam.
The strongest gate for "is it inert?" is the poison-the-padding test (forward of a recon with
*nonzero* padded slices must bit-match the clean-padded forward — `test_forward_inert_to_nonzero_recon_padding`),
not just "the padding stayed zero."

**Footgun to carry into E:** `is_sharded` and `n_devices > 1` have **decoupled** —
`is_sharded` is ~always True now, so every `is_sharded` site must be re-read for which question
it asks ("do I have a placement?" vs "do I have ≥2 physical devices?" = `len(shard_devices) > 1`).

---

## 6. Adjacent / tracked items (off the critical path)

- **Multiaxis FBP angular weighting — DECIDED in principle (option 1), implementation DEFERRED**
  (2026-06-18; Greg to discuss with others; land with or before the multiaxis port, increment D).
  - *Problem.*  `MultiAxisParallelModel.fbp_filter` builds a per-view directional 2-D ramp
    `|U·d_az + V·d_el|` whose orientation and magnitude come from `jnp.gradient(angles)` — the step
    between *list-neighbor* views.  But this geometry is **simultaneous fixed measurements** (multiple
    lasers/detectors), so there is **no acquisition trajectory and the `angles` list order is
    arbitrary** (laser-enumeration order).  An analytic inverse must depend only on the *set* of view
    directions, so any order-dependent weighting is wrong: here it makes the ramp's orientation and
    weight effectively arbitrary, and permuting the (arbitrary) list changes the recon.  The
    directions are a sparse, limited-range **2-D point set** on the sphere (azimuth × elevation), so
    the natural measure is a 2-D solid-angle element, not the 1-D arc-step `jnp.gradient` computes.
  - *Decision — option 1: uniform per-view weight + standard channel ramp* (treat it as stacked 2-D
    FBP; elevation approximated in the filter and corrected by MBIR).  Order-invariant, robust, and it
    reduces **exactly** to `ParallelBeamModel.fbp_filter` in the azimuth-only equally-spaced limit.
    The uniform weight is **`pi / num_views`** — the azimuthal angular measure, the *same* constant as
    parallel beam (no special 2-D/solid-angle constant enters, precisely because option 1 is
    2-D-FBP-per-slice; a solid-angle `dΩ` constant would appear only in the rejected option C).  Also
    fix the filter scaling to `1/(delta_voxel·delta_voxel_row)` (the student's `1/delta_voxel²`
    silently assumed `voxel_row_aspect == 1`).  Carries the same equispaced/limited-angle caveat as the
    other FBP/FDK paths (it is an MBIR initializer).
  - *Rejected — option C: order-invariant solid-angle density.*  The rigorous `dθ dφ` (weight each view
    by its local solid-angle area) is both over-engineered and numerically ill-conditioned for **few**
    directions (spherical Voronoi areas of a handful of points are boundary-dominated and meaningless),
    and the missing-data null space dominates the initializer error anyway.  Revisit only if standalone
    multiaxis FBP becomes a real deliverable.
  - *Gate to add with the implementation:* a **reorder-invariance** test (permute the view list, same
    set → recon unchanged to float tol) — it encodes the principle and currently FAILS; plus a
    reduce-to-`ParallelBeamModel` check in the azimuth-only equally-spaced limit.  Multiaxis FBP is
    currently **ungated** (not in `_geometry_types_for_tests`, no dedicated FBP/recon test).
  - *Note:* `fbp_filter` is still single-device (accepts `output_sharded`, no-ops it); the rewrite
    should take the per-view-shard `run_per_device` treatment when multiaxis is ported (D).
- **Settable view parameters — ✅ done.**  `view_params_array` is a traced projector arg;
  `set_view_parameters()` updates it with no recompile; `vcls` runs as a 1-view sibling.
  Deleting the old `view_indices` plumbing is the E task above.
- **Hybrid (recon-CPU / sino-GPU) — DROPPED** (2026-06-08; stitching/subsets + sharding cover
  the +19% it gave).  Removing `_transfer` + the `'sinograms'` mode is part of E.
- **Simplify the sparse-projector batching machinery** — separate tracked refactor (collapse the
  scan/map/vmap nest; remove the `lax.map` jax#27591 fragility; `apply_row_filter` is the
  template).  High blast-radius; do on its own with the projector suite as the gate.  B3
  de-closuring already touched this core.  *(v1 §Future project.)*
- **CPU-cluster auto-sharding policy** — `_auto_shard_cpu=True` is the default; remaining = real-
  cluster perf + a virtual-vs-real-CPU topology policy.  *(post-P6)*
- **Suite tidiness** — seed the remaining unseeded-`np.random` tests; pre-merge
  `import mbirjax`-before-`jax` sweep; public `shard_*` / `gather_*` wrappers.
- **Minor opens** — `configure_devices` / `use_gpu` unification; forward pixel-batch default.

---

## 7. Durable principles & facts (pointers, not copies)

- **Cross-cutting principles**, **verified hardware facts** (L40S silently zeros a
  device-resident cross-device copy; H100 d2d ok → the `move_shard` host-bounce fallback),
  and **O1–O4** resolutions: v1 §Cross-cutting / §Hardware findings / §Open questions.
- **`(g0,L)` slice-band interface + anchor rule + forward-banding-is-accumulation:** v2 §P3
  design note; `p6_projector_rework_proposal.md` §8a-design.
- **Project rule — exact equality is *never* the gate for computed floats:** tight `allclose`
  (1e-5 single-shot, 1e-4 iterated VCD); exact equality only for data-movement identities and
  constructed-zero invariants (e.g. padded entries == 0.0).  `.claude/lessons.md`.
- **jax/GPU playbook** — the reference-cycle + buffer-donation memory lesson; the
  platform-divergent back-projection kernel; the "never build a `namedtuple` class inside a
  jit-static-arg feeder" lesson; GPU run-to-run variance & thermal-throttle diagnosis:
  `.claude/lessons.md`.
