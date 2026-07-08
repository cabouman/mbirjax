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
| `increment_d_translation_design.md` | **authoritative for the translation port** — increment-D staged plan (T1–T5) |
| `increment_e_retirement_design.md` | **authoritative for increment E** — the retirement-cascade staged plan (E1–E5), verified architecture facts + open design decisions (DRAFT) |
| `p6_projector_rework_proposal.md` | projector-rework design; **§8a-design is canonical** (rest partly superseded) |
| `performance_tracking_plan.md` | nightly perf-regression harness + metrics — **separate track**, out of scope here |
| `.claude/lessons.md` | jax/GPU/placement/measurement playbook |
| `plans/sharding/back_projection_overview.md` | projector internals (read before touching cone kernels) |
| `plans/sharding/sinogram_sharding.md` | parked row-sharding-the-sinogram exploration |

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
| **D — translation** — T1 FDK filter, T2 banded back, T3 forward anchor, T4 `_supports_sharding`, T5 inert padding | ✅ **DONE 2026-06-19/20**, CPU + GPU validated (`increment_d_translation_design.md`).  Now on the always-on placement path; correct at all device counts incl. padding.  Merged to prerelease in PR #17 |
| **D — multiaxis** | ✅ **DONE 2026-06-20**, CPU-validated.  On the always-on placement path; correct at all device counts incl. padding.  Port (M0–M5): FBP angular-weighting fix (order-invariant channel ramp, §6); anisotropic voxels (row + slice aspect); banded back kernel + channel-major horizontal fans; forward anchor on `recon_shape[2]`; `_supports_sharding=True` (no driver overrides); inert padding (test-only).  Calibration anchored to ParallelBeam at el=0 (forward bit-exact).  GPU-confirmed (A100 + 4-GPU nightly); the `1/cos(elevation)` vertical path-length factor (§6) remains open.  Channel-major transpose also added to **translation** (forward ~1.8× on CPU) |
| **E** — retirement cascade (legacy single-device dispatch, `main_device`, `view_indices`, …) | ✅ **DONE 2026-06-22**, CPU-validated + committed (`increment_e_retirement_design.md` §4 — every row).  Retired `view_indices`→`owned_view_indices`, the no-mesh path + `_supports_sharding`, `output_device`, `main_device`/`sinogram_device`, `configure_sharding` (→ `configure_devices` only), the vacuous `if is_sharded` branches + the `is_sharded` property; merged the device-config into `_set_device_layout`+`_auto_device_pool`; retired `mesh` (kept `shard_devices` as the public accessor); sharded the denoiser (E3c); dropped redundant `initialize_recon` device-puts (P3).  Placements are the single source of device layout.  GPU-cluster confirmed (nightly) |
| **Post-E** — docs + utility sharding + Tk/cache fixes | ✅ **DONE 2026-06-24** (in PR #17): user/dev docs (`usr_multi_gpu`, `dev_sharding_overview`, `dev_api` rewrite); `gen_weights` / `gen_weights_mar` / the Shepp-Logan phantom / `generate_demo_data` made sharding-aware (+ phantom `max_block_gb` blocking and `target_max_attenuation`); `_pad_shard_on_axis` readability; `recon_shard_axis`/`sinogram_shard_axis` → classmethods; viewer/demo Tk-GC noise fix; **JAX per-fusion-autotune-cache disabled** (fixed the A100 `NOT_FOUND` cache errors) |
| **Prerelease PR (#17)** | ✅ **MERGED 2026-06-24** into `prerelease` (55 commits, +5032/-1933).  All four geometries + the QGGMRF denoiser sharded.  CPU suite green at 4 devices; A100 green for everything runnable.  Follow-ons (non-blocking, and likely to grow as wider use surfaces issues): cone 2048³/8 deadlock (§6), `mar`/preprocessing (#18), doc-xref cleanup |

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

## 4. Cone port (P6) — ✅ DONE (only the B4.5 band-kernel perf optimization is deferred, below)

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
  analysis in `plans/sharding/sinogram_sharding.md`.
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
3. **D** — translation + multiaxis ports (same template).
   - **Translation — ✅ DONE 2026-06-19/20** (CPU + GPU validated; full record in
     `increment_d_translation_design.md`).  T1–T5 landed exactly as planned (`TranslationModel` was
     `ConeBeamModel` *pre-port*, so the increments mapped ~1:1 and T5 needed zero production changes).
     Post-port refinements this session: a **scale-invariant sharded-test gate**
     (`conftest.assert_sharded_allclose`, surfaced by the translation Hessian + CPU reduction-order
     nondeterminism), and **dropping the redundant sharded VCD-recon test** for translation
     (`_PaddedReconMixin.RUN_SHARDED_VCD`; the sharded VCD loop is geometry-independent, gated on
     parallel + cone).  Merged to prerelease (PR #17).
     **GPU follow-ups (Greg's cluster, not blocking):** the GPU n=1 back short-circuit band-vs-pixel
     platform split for translation (decision 2b — active but UNMEASURED), and per-device memory/time
     scaling.  **Prereq:** `translation` baselines in mbirjax_metrics (separate session).
   - **Multiaxis — ✅ DONE 2026-06-20** (CPU-validated).  `MultiAxisParallelModel` extends
     `TomographyModel` directly and now runs the always-on placement path, correct at all device
     counts incl. padding.  Staged M0–M5 (gates folded into the existing `test_projectors` /
     `test_fbp` / `test_padding` harnesses, no new VCD gating — multiaxis is NOT in
     `_geometry_types_for_tests`):
       - **M0** adjoint/Hessian (via `anisotropic_multiaxis` in `test_projectors`).
       - **M1** FBP angular-weighting fix (§6 option 1): the directional 2-D ramp (order-DEPENDENT,
         from `jnp.gradient(angles)`) → the shared `_apply_direct_recon_filter` channel ramp +
         uniform `pi/num_views`.  Order-invariant; reduces exactly to `ParallelBeamModel.fbp_filter`.
       - **M-aniso** anisotropic voxels (`voxel_row_aspect` + `voxel_slice_aspect` through all four
         fans + `auto_set_recon_geometry` + `get_psf_radius`); horizontal fan adopts ParallelBeam's
         `footprint_xy` density.  Bit-identical to the prior kernels at unit aspects.
       - **M2** banded back kernel (`back_project_one_view_to_band`, global anchor + `k < S_real`
         clip) + rolled-`lax.map` rewire + **channel-major** horizontal fans; `entries_per_cylinder_batch`
         → module band/batch consts.
       - **M3** forward anchor on `recon_shape[2]` (global validity test; the inert-padding anchor).
       - **M4** `_supports_sharding=True` — ONE line, no driver overrides (geometry-neutral base
         hooks); `test_multiaxis_sharded`.
       - **M5** inert slice padding — **zero production changes** (`TestPaddedSlicesMultiAxis`).
     Calibration anchored to ParallelBeam/anisotropic_parallel at el=0 (forward **bit-exact**,
     `test_multiaxis_forward_reduces_to_parallel`).  **Open (not blocking):** GPU-cluster confirmation
     (channel-major, band kernel, sharding — the n=1 back band-vs-pixel split is geometry-agnostic and
     active but UNMEASURED for multiaxis); the `1/cos(elevation)` vertical path-length factor (§6).
   - **Translation channel-major** — the cone/parallel increment-A transpose was also applied to
     translation's two horizontal fans (it shared the cache-aliasing gap): forward ~1.8× faster on
     CPU at 512 channels (back neutral), value-preserving (forward rel ~7e-8, back bit-identical).
4. **E — retirement cascade** — ✅ **DONE 2026-06-22**, CPU-validated + committed.  Retired
   `main_device`/`sinogram_device`, the `is_sharded` else-branches *and* the `is_sharded`
   property, the legacy single-device *dispatch*, `view_indices` (→ `owned_view_indices`),
   `output_device` (public surface + `compute_hessian_diagonal`), `configure_sharding`
   (→ `configure_devices` only), and the redundant `initialize_recon` device-puts (P3); the
   RETIRE-marker sweep removed the trivial-mesh comparison tests.  **Kept** the single-device back
   driver/kernel (`_sparse_back_project_single_device` / `back_project_one_view_to_pixel_batch`),
   which the GPU n=1 short-circuit calls.  Also sharded the denoiser (E3c) and added
   `mjs.sharded_full`.  Full record: `increment_e_retirement_design.md`.  **Prerelease PR #17 MERGED
   (2026-06-24).**  Done since: `pixel_indices_worker` collapse (#22), the `mesh`/`shard_devices`
   revisit (`mesh` retired, `shard_devices` kept).  Remaining follow-up: `mar`/preprocessing sharding
   (#18 — see the `output_sharded` decision in §6).
5. **Post-P6** — the multi-GPU **user docs page** ✅ **DONE 2026-06-23**: `usr_multi_gpu.rst`
   (zero-effort path, device subsetting via `configure_devices`, efficiency tips, expectations,
   a gentle "sharding" overview) + `use_gpu`/`overview`/`advanced_features`/FAQ refresh, and the
   Tomography-Model "Device Configuration" section moved before "Saving and Loading" + refreshed.
   The prose `:meth:` cross-refs were fixed to **fully-qualified** `mbirjax.*` targets (an
   unqualified `:meth:`Class.method`` renders as plain text with **NO Sphinx warning**).
   - **Deferred doc-cleanup pass** (its own follow-up, NOT in the sharding PR): the remaining
     UNRESOLVED Sphinx py-xrefs that silently render as plain `<code>` (no warning). Detect by
     building the HTML and grepping for `<code class="xref py …">` **not** wrapped in `<a>`.
     Buckets: (a) `usr_api_overview` `.. autosummary::` table entries (unqualified) + wrong-target
     refs — `denoising.median_filter3d`→`mbirjax.median_filter3d`,
     `ParameterHandler.set_params`→`TomographyModel.set_params`,
     `generate_3d_shepp_logan_low_dynamic_range`→`mbirjax.utilities.*`; (b) docstring refs
     autodoc'd into `usr_denoising`/`usr_tomography_model`/`usr_utilities` whose targets aren't
     documented (`vcd_recon`, `get_recon_dict`; the `ParameterHandler` class has no `autoclass`);
     (c) external `jnp.median` (no intersphinx).  (The `sparse_forward_project`/`sparse_back_project`
     refs were removed as a side effect of the forward/back_project docstring rewrite.)
     Fix per case = qualify / correct-target / document-the-target (add `automethod`/`autoclass`) /
     downgrade-to-literal. The device-config private-helper docstring refs in `tomography_model.py`
     were already converted to literals as part of the docs work.
   - Then: the choose-N-vs-communication policy (+ the CPU-cluster auto policy); **B4.5** if
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

- **Multiaxis FBP angular weighting — ✅ DONE (option 1), 2026-06-20** (multiaxis port M1).
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
  - *Gates (added with M1):* a **reorder-invariance** test (permute the view list, same set → filter
    unchanged) — it went from rel-max ~8.5 (old directional ramp) to <1e-6 — plus a
    reduce-to-`ParallelBeamModel` check in the azimuth-only equally-spaced limit; both in
    `tests/sharding/test_fbp.py::TestMultiAxisFbpFilter`.  (Multiaxis is still NOT in
    `_geometry_types_for_tests` — no VCD-convergence/recon-NRMSE gating — but the projectors are now
    gated via `anisotropic_multiaxis` in `test_projectors`.)
  - *Done:* `fbp_filter` now delegates to the shared `_apply_direct_recon_filter` (per-view-shard
    `run_per_device` path, sharded once `_supports_sharding` is on — which it now is), so it is no
    longer single-device.
- **Multiaxis vertical-fan absolute magnitude (the `1/cos(elevation)` question) — OPEN, deferred.**
  - *What is verified.*  At elevation 0 the multiaxis forward projector reduces **bit-exactly** to
    `ParallelBeamModel` (isotropic) and to `anisotropic_parallel` (with `voxel_row_aspect ≠ 1`) — a
    permanent gate (`test_projectors.test_multiaxis_forward_reduces_to_parallel`).  So the in-plane
    density (incl. the channel-major fans and the M-aniso row-aspect) is calibrated against validated
    reference models, not merely adjoint-consistent.
  - *What is NOT pinned.*  The vertical (elevation) fan uses `scaling = 1.0` — no path-length
    normalization — so for **elevation ≠ 0** or **`voxel_slice_aspect ≠ 1`** the absolute magnitude is
    only *self-consistent* (forward/back adjoint holds, Hessian holds), not anchored to a reference; the
    adjoint is blind to a consistent scaling error, and parallel beam is not a valid reference once
    slices spread across rows.  Cone/translation carry a `1/cos(phi)` (cone-angle) path-length factor on
    their vertical fans; multiaxis has no analog.  But the geometry differs: the multiaxis detector is
    held **perpendicular to the (tilted) ray** (verified from the coordinate maps — the ray direction is
    `(−sinθ·cosφ, cosθ·cosφ, sinφ)` and the detector axes are both ⟂ to it), so there is **no
    detector-incidence obliquity** (unlike cone's fixed detector); the only tilt effect is the ray
    cutting obliquely through the axis-aligned voxel grid.  So the right factor must be **derived from
    that path length**, not copied from cone — it may not be a clean `1/cos(elevation)`.
  - *Decision.*  Left as `scaling = 1.0` through the M-aniso work (a path-length factor changes values
    even at unit aspects, so it cannot ride the reduce-to-isotropic gate).  Take it up as a **separate
    change** with its own **physical-fidelity gate** — forward-project a known object at elevation ≠ 0
    and compare to an analytic line integral (the adjoint cannot gate it).  Acceptable as-is for the
    intended use (an MBIR initializer; the iterative recon absorbs a global mis-scaling).
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
- **Auto device-count basis: recon-slices vs sino-views (OPEN, part of choose-N).**  `_auto_device_count`
  trims the device count on the **recon-slice** axis (`recon_shape[recon_shard_axis]`): it drops a count
  whose last *slice*-shard would be entirely padding.  But the projection compute lives on the
  **view-owners** (each view-owner forward/back-projects its own views), so the slice axis is the wrong
  proxy for "does this device do real work":
    - a device with an all-padding **slice**-shard can still own real **views** and do full projection
      (it projects bands into its views; the result reduces to the slice-owners) — the slice-based trim
      can drop a device that would still be useful (e.g. 5 slices / 100 views on 4 devices → trims to 3,
      though all 4 could each project 25 views);
    - a device with an all-padding **view**-shard does **no** projection even if it owns real slices, and
      the slice-based check never looks at views (e.g. 100 slices / 5 views on 4 devices → keeps 4, but
      the 4th device has no views to project).
  Revisit the basis (likely views, or both axes) as part of the **choose-N-vs-communication policy** (§5).
  Surfaced 2026-06-23 while writing the dev sharding-overview page; the page (option A) deliberately
  describes the trim as "a tunable heuristic" rather than enshrining the slice-based rule.  *(post-P6;
  may be pulled earlier — evaluate before the PR.)*
- **Suite tidiness** — seed the remaining unseeded-`np.random` tests; pre-merge
  `import mbirjax`-before-`jax` sweep; public `shard_*` / `gather_*` wrappers.
- **Minor opens** — `configure_devices` / `use_gpu` unification; forward pixel-batch default.
- **Cone-beam 8-GPU 2048³ recon hang — ✅ RESOLVED 2026-06-28.**  `partition_sequence=[4,6,7]` (skip the
  granularity-1 first partition) + the memory work (P2 init-sino free + host-transparency) let the **plain
  sharded `recon()`** run cone 2048³/8 at **peak 45.3 GB/shard of 77.6**, 3 iterations, healthy convergence
  (loss 1.73→0.86→0.56), completes + saves.  Better than the ~62–69 GB estimate (the per-subset transient
  scales sub-linearly with the full per-shard volume at fine granularity, so the cone-default OOM's ≥2.15×
  scaling overstated the [4,6,7] case).  Diagnosis history below.
- **Cone-beam 8-GPU 2048³ recon hang — DIAGNOSED 2026-06-24 as OUT OF MEMORY** (the NCCL "Acquire
  clique" timeout was a downstream symptom, not the cause; see the DIAGNOSED bullet below).
  A manual cone 2048³ recon on 8 H100s hung at the **first VCD subset update** (after FDK init +
  Hessian) with `Acquire clique … Expected 8 threads … not all arrived`; parallel beam at the same
  config worked.  Repro tooling (REMOVED 2026-06-28 once the cone OOM was resolved): `plans/experiments/sharding/cone_deadlock_repro/` (cone-vs-parallel ×
  size × device-count sweep, each config `timeout`-isolated, HLO dumped, build-ID printed).
  - **Mechanism:** no explicit collectives in mbirjax — projection is collective-free (thread pool
    + `make_array_from_single_device_arrays`).  The only multi-device collectives are XLA-GSPMD
    auto-inserted for the **VCD line-search scalar reductions** over the view-sharded sinogram
    (`get_forward_lin_quad`'s `jnp.sum`, the `alpha` sums) — an all-reduce on the sino mesh + a
    sino→recon cross-mesh reconcile.  Those are **shared with parallel beam**, so a cone-only hang
    points to cone-specific divergence or scale/resource (cone's whole-cylinder kernels are far
    heavier), NOT a structural bug.  (2048/8 divides evenly → not a padding-shape issue.)
  - **Sweep result (job 12776721, confirmed build, h015):** **ALL configs OK through 512³ on 8 GPUs**
    (parallel + cone, n=1/2/4/8).  So the sharded path is **structurally correct through 512³/8** —
    rules out a collective-participation bug.  The failure is either scale-only (the 2048³ regime:
    memory pressure / slow uneven per-device compile desyncing the all-reduce rendezvous) or a
    build/env artifact of the original run (which had build uncertainty — `__version__`/unsure build).
  - **2048³ attempt #1 (2026-06-23, job 12776973):** ALL configs `ERR` — but in the **repro script**,
    not the recon: `forward_project` defaulted to `output_sharded=False`, gathering the full 2048³
    sinogram (32 GiB) onto one GPU (on top of the 32 GiB phantom) → OOM in `_gather_to_host`, before
    recon started.  Geometry-independent (parallel + cone identical).  Fixed the repro to build the
    phantom **already-sharded** (`sharded_full`, per-shard -- avoids the slow/transient-heavy
    `gen_modified_3d_sl_phantom` and the 32 GiB host-random) and keep the forward projection AND the
    recon **device-sharded** (`output_sharded=True`) -- no single-device 32 GiB array anywhere.  (Side lesson: at 2048³ a gathered single-device output is itself 32 GiB — real runs must
    stay sharded.)
  - **DIAGNOSED (2026-06-24, job 12777612): the root cause is OUT OF MEMORY, not a collective deadlock.**
    Hardware: **80 GB H100s** (8 GPUs).  cone 2048³ TIMEOUT at n=4 and n=8; the cone n8 log is the BFC
    allocator failing a **4.02 GiB** allocation (`GPU_4_bfc`) on 5 of 8 GPUs.  HLO disproves the collective
    theory: cone has **67** all-reduces vs parallel's **139**, and **parallel n=8 (more collectives)
    completed OK** — so no unique/missing cone collective.  The original "Acquire clique … not all threads
    arrived" was a downstream symptom: a GPU stuck in the OOM-retry loop never reaches the rendezvous, so
    peers time out.  parallel n=4 ERR'd with a clean OOM.
    **Phase confirmed = the first VCD subset update.**  The cone n8 log reaches:
    forward-project → sinogram (552 s) → `Starting direct recon` (FDK init) → `Initializing error
    sinogram` → `Computing Hessian diagonal` → `Starting VCD iterations`, and the BFC failure fires
    *immediately after* the VCD-iterations banner.  So FDK init and the Hessian diagonal both fit; it is
    the **VCD subset-update working set** that overflows.
    **Conclusion:** cone's per-device VCD-update working set at 2048³/8 exceeds an 80 GB H100 (cone's
    whole-cylinder transients ≫ parallel's separable kernel).  **The collective-free-reduction candidate
    is OFF.**
  - **Next:** (1) phase is now pinned (first VCD subset update) — instrument the subset update to find the
    dominant transient (the per-subset forward/back of a slice band over its views, the band reduce, or the
    qGGMRF prox buffers).  (2) reduce cone's peak per-device VCD-update memory — the band slice count, the
    forward pixel-batch size, and the per-subset view working set are the levers — and/or use more GPUs.
    (3) Secondary UX: **PARTIALLY ADDRESSED 2026-06-25.**  Added a "use a finer partition_sequence
    (e.g. [4,6,7])" bullet to the GPU OOM guidance (`log_oom_guidance` in `_utils.py`), which
    `_handle_jax_error` already prints on a caught `JaxRuntimeError`.  This fires for **catchable** OOMs
    — single-device too-big recons (clean `RESOURCE_EXHAUSTED`) and the multi-device OOMs that surface
    cleanly (e.g. parallel n=4).  **STILL OPEN:** the multi-GPU OOM that *hangs* instead of raising (a
    GPU stuck in the BFC retry loop never reaches the NCCL rendezvous → "Acquire clique" timeout) — the
    exact 2048³/8 cone case — produces no exception, so the hint never prints there.  Converting that
    hang into a catchable error is a bigger, riskier change (allocator flags / collective timeouts) and
    the project deliberately does not pre-estimate fit (`set_devices` docstring); left as a separate
    follow-on.
  - **Memory-reduction plan (2026-06-25).**  Three independent levers, cheapest-first; the goal is to
    fit cone 2048³/8 by shrinking the first-VCD-update footprint.
    - **(P1) Partition granularity — PRIMARY, zero code.**  The OOM is on the FIRST VCD iteration
      because `partition_sequence` defaults to `[0, 2, 4, 6, 7]` (indices into
      `granularity = [1, 2, 4, 8, 16, 32, 64, 128, 256]`), so iteration 0 uses `granularity[0] = 1` — a
      single subset = **the entire recon**.  Hence `len(pixel_indices)` = all ~4.2M pixels and every
      subset-domain array (`prior_grad/hess`, `forward_grad/hess`, `delta`) is `(4.2M, 256/device)·4B ≈
      4.3 GiB` — exactly the failing 4.02 GiB allocation, ×5 co-resident, plus the whole-cylinder
      back-projection transient.  Finer first partition shrinks ALL of these proportionally.  Experiment
      (no code): `ct_model.set_params(partition_sequence=[2,4,6,7])` (first iter → 4 subsets) or
      `[4,6,7]` (→ 16 subsets, each subset array ~270 MiB).  Single-variable sweep; trade-off = a few
      extra iterations vs starting at granularity 1.
      **SWEEP RESULT (2026-06-25, 1024³, 2×H100, 2 GB/shard) — PARALLEL BEAM:** peak `bytes_in_use`/shard
      `[0,2,4,6,7]` = **32.5 GB**, `[2,4,6,7]` = **27.1 GB (−17%)**, `[4,6,7]` = **24.9 GB (−24%)** —
      and TIME was neutral/slightly faster (319.5 → 301 s; the granularity-1 first iteration was the
      slowest, not the cheapest).  Diminishing returns: dropping granularity 1 is the big win (−5.4 GB);
      4→16 subsets adds only −2.2 GB.  Steady-state floor = **8 GB/shard** (resident flat_recon +
      error_sinogram + fm_hessian + 1, each 2 GB) in ALL runs, so the transient ABOVE the floor falls
      24.5 → 16.9 GB.  That ~17 GB residual is the **projection working set** (delta_sinogram + the
      back/forward scratch), which granularity does NOT shrink (at 16 subsets the subset arrays are only
      ~130 MB).  Confirms the diagnosis numerically (the 2 GB/shard subset array here = 4.3 GB at 2048³/8
      = the failing 4.02 GiB alloc).
      **CONE BEAM (1024³, 2×H100, + P2), 2026-06-25 — granularity sweep:** peak/shard default `[0,2,4,6,7]`
      = **36.09 GB** (380 s), `[4,6,7]` = **28.67 GB** (388 s) → `[4,6,7]` cuts cone peak **−7.43 GB
      (−21%)**, nearly the same ABSOLUTE saving as parallel (~7.6 GB) — granularity shrinks the
      geometry-independent subset arrays, and the cone-specific whole-cylinder projection transient
      (**~8 GB/shard** above parallel) sits on top regardless of granularity.  Steady 2 GB; time neutral.
      **Extrapolation (CONE → 2048³/8, [4,6,7] + P2), anchored on cone's OWN OOM:** cone default OOM'd at
      2048³/8 (≥77.6 GB) and is 36.09 GB at 1024³/2, so the cone 1024³/2→2048³/8 scale is **≥ 77.6/36.09 =
      2.15×**.  Applying that to cone `[4,6,7]`'s 28.67 GB → **~62 GB** (×2.15), ~69 GB (×2.4); it only
      reaches the 77.6 ceiling near ×2.7.  So **`[4,6,7]` + P2 has a real shot at fitting cone 2048³/8**
      (best estimate ~62–69 GB), firmer than the earlier parallel-borrowed guess because it is anchored on
      cone's own scaling — but the margin is thin enough that a super-linear projection transient could
      reach the edge, so the 8-GPU cone run is still decisive.  If it does not fit, the lever is the
      **cone projector's per-call working set** (view-batch / band size — that ~8 GB transient), NOT P3
      (subset arrays are already ~130 MB at 16 subsets).
      **Caveat:** the sweep measured memory+time, not convergence; confirm `fm_rmse`/`prior_loss` land
      together across the three before adopting `[4,6,7]` as a large-problem default.
    - **(P1-design) Size-adaptive granularity sequence? — DECIDED 2026-06-25: defer; if ever, SIZE-only,
      NOT memory-adaptive.**  A memory/hardware-adaptive default (auto-pick the starting granularity from
      free GPU memory) would BREAK the sharding guarantee that **results are independent of device count**:
      the same problem would converge differently on different hardware (2048³ on 8×80 GB vs 16×40 GB →
      different schedule → different image), and that drift would surface in the cross-device/cross-platform
      correctness gating.  It also needs a peak predictor we do not have (geometry-dependent ~8 GB cone
      delta, unclear super-linear scaling — "sweep, don't guess").  The only reproducibility-safe form is
      **size-only**: derive the coarsest starting granularity deterministically from recon voxel count
      (e.g. subset count ≥ ceil(voxels / threshold), threshold calibrated from a size sweep), overridable —
      a function of the PROBLEM, identical on any hardware.  Even that is **gated on the convergence-quality
      data** from the 8-GPU run (unverified assumption: skipping granularity 1 does not hurt the final
      image).  Until then: keep the `[0,2,4,6,7]` default + the documented "use a finer sequence for large
      multi-device problems" guidance (now also surfaced in the GPU OOM message).
    - **(P2) Free the init-phase sinogram — DONE 2026-06-25 (`vcd_recon`).**  After
      `error_sinogram = sinogram - alpha*error_sinogram`, the placed sinogram's only remaining use was a
      dtype read (`_sino_ones_device_form`, now fed `error_sinogram`).  Free it then — ~4 GiB/shard
      reclaimed before the Hessian and the VCD loop — guarded by an **ownership test**: we delete only
      when to_sino allocated fresh buffers we own (host/numpy input, or a jax array on OTHER devices).
      An input already resident on the sino devices may be returned by to_sino as a no-copy reshard that
      **shares the caller's buffers** (or unchanged, e.g. `prepare_sino_for_devices`); deleting that
      corrupts the caller's array.  **Lesson learned the hard way:** the first attempt guarded on object
      identity (`sinogram is not _sino_in`) and broke 4 sweep tests with `Array has been deleted` — a
      no-copy reshard is a *different object* that *shares buffers*, so identity ≠ ownership; the
      device-disjoint test is the correct one.  CPU 4-device suite green (delete fires on numpy inputs;
      buffer-sharing reuse correctly skipped).
      **MEASURED (2026-06-25, 1024³, 2×H100, granularity [4,6,7]) — PARALLEL BEAM:** floor/shard
      **8 → 4 GB**, peak/shard **24.86 → 20.77 GB** — a −4 GB drop (better than the naive −2 GB sinogram
      estimate; the peak drop = the floor drop, so P2 lowered the whole baseline).  Likely mechanism: the
      explicit `.delete()` + `block_until_ready` clears the sinogram AND lets the gc-pending
      old-error_sinogram cycle reclaim (~2 + ~2); not fully pinned without a per-phase trace.  **Combined
      with P1 [4,6,7], parallel-beam peak is 32.5 → 20.8 GB (−36%) from the original default.**  (Cone is
      higher — see the CONE BEAM line under P1; cone is the geometry that drives the 2048³ OOM.)
    - **(P2b) `gen_weights` lands a host sinogram entirely on gpu0 — FIXED 2026-06-25 (host-preserving +
      opt-in shard).**  The bug: `gen_weights` used `jnp` ops unconditionally, so a **numpy/host**
      sinogram promoted the full result to one device (gpu0), and the CALLER held that 2 GB/shard array
      there for the whole recon → persistent inter-GPU asymmetry (Greg "had to be very careful" to avoid
      it).  Fix (Greg chose the interface): compute with the **input's own array module** — `xp = jnp if
      isinstance(sinogram, jax.Array) else np` — so a host sinogram yields **host** weights (recon
      streams them to shards; never a single-GPU full copy) and a (possibly sharded) jax sinogram yields
      jax weights inheriting its sharding (unchanged).  PLUS an optional **`ct_model=`** that places the
      sinogram in the model's view-sharded device form (`_shard_sinogram`, pad-aware) before weighting,
      returning recon-ready sharded weights with no single-device copy.  Strict improvement for every
      caller (`nsi.py`, demos).  Tests added: host-in→host-out, and `ct_model`-host-in→sharded-out (full
      sharding + preprocessing suites green).  Pairs with #18 (gen_weights_mar / preprocessing sharding).
    - **(P2c) `generate_demo_data` output type + gpu residue — FIXED 2026-06-25.**  Two issues: (1) the
      return annotation said `(np.ndarray, np.ndarray)` and the prose said "numpy", but it returns a
      3-tuple whose phantom was a single-device **jax** array (shepp-logan default) and whose
      `params['angles']` was always a gpu `jnp` array; (2) it left jax intermediates resident on the gpu.
      Fix: added **`output_sharded`** (matches `forward_project`/`recon` naming; default `None` →
      `devices is not None`).  `output_sharded=False` returns plain numpy (phantom + sinogram + params
      arrays) and explicitly `.delete()`s the device arrays we created (incl. the helical `phantom_core`)
      and `del`s the generation model — nothing left on the gpu; `True` returns device-form jax (sharded
      with `devices`, single-device otherwise).  Annotation → `tuple`; docs corrected.  Behavior change:
      the default (`devices=None`) phantom is now numpy (was single-device jax); demos/tests don't depend
      on it being jax.  Tests: default-all-numpy, `output_sharded=True` w/o devices → jax, and
      devices+`output_sharded=False` → gather-to-numpy matches the sharded values (suite green).
    - **(P2d) Should `gen_weights` / `gen_weights_mar` also get `output_sharded`? — DECIDED 2026-06-25.**
      **Design principle: `output_sharded` belongs only on functions that OWN a device layout** — model
      methods (`forward_project`/`recon`/…) and `generate_demo_data` (builds a model internally).
      - **`gen_weights`: NO.**  It is a model-less free function; with no devices of its own a bare
        `output_sharded=True` has nothing to shard across (it would only work *paired with* `ct_model` — a
        kwarg needing another kwarg).  Output form is already set by the **input's residence**
        (host→host, sharded→sharded) plus the optional **`ct_model=`** (the sharding source).
        `output_sharded=False` on a sharded input is a one-liner the caller can do.  Leave as-is.
      - **`gen_weights_mar`: YES, but FOLD INTO #18 (not a standalone add).**  It already takes `ct_model`
        (so it owns a layout → `output_sharded` fits, like `forward_project`), AND it still has the same
        host→gpu0 dump `gen_weights` had (the final `jnp.exp` promotes a host sinogram to one device).
        But a correct `output_sharded` also needs the **`init_recon` path** sharded: it calls
        `ct_model.forward_project(metal_mask)` with the DEFAULT gathered output (+ `multi_threshold_otsu`
        gathers), so `output_sharded=True` there would mix a sharded sinogram with a single-device
        `delta_metal` → sharding mismatch.  Making it work = host-preserving fix + `forward_project(...,
        output_sharded=True)` + sharded Otsu, which is exactly #18.  No in-repo callers → no urgency.
        (Option if wanted sooner: land just the host-preserving half — compute the final weights with the
        input's array module — to kill the gpu0 dump for the `init_recon=None` case, deferring
        `output_sharded` + the `init_recon` path to #18.  Not done now.)
    - **(P3) Free dead subset arrays in `vcd_subset_updater` — PLANNED, measure-first.**  `forward_grad/
      forward_hess` are dead after `delta` (line ~3019); `prior_grad/prior_hess` after `prior_quadratic`
      (~3027).  Freeing both drops co-resident subset arrays 5→~2 at the peak (~12 GiB/shard at
      granularity 1).  Open question is `del` (refcount, no sync — cheap) vs explicit `.delete()`
      (needs a per-subset `block_until_ready`, the cost the staged-halo work fought to avoid): some
      freshly-computed sharded arrays sit in jax ref cycles and don't free on refcount, others (assemble_
      sharded outputs) do.  Plan: land plain `del`, A/B the peak with `mbirjax.memory_stats.memory_report`
      (`peak_bytes_in_use`) on a mid-size cone case, escalate to `.delete()` only if `del` doesn't free
      and the peak reduction justifies the sync.  **Parked behind a GPU measurement.**
      **UPDATE (2026-06-25, after the P1 sweep): P3 is now largely redundant with fine granularity.**
      The sweep shows that at granularity 16 the subset arrays are only ~130 MB/shard (vs 2 GB at
      granularity 1) and the residual ~17 GB/shard transient is the PROJECTION working set, not the
      subset arrays.  So P3 only matters if we deliberately stay coarse.  If `[4,6,7]` does NOT get
      2048³/8 under the limit, the real next lever is that fixed projection transient — the cone
      projector's per-call working set (view-batch / band size), NOT the VCD subset arrays.
    - **Rejected: the `update_recon`-accumulation reorg.**  Folding the forward terms into the prior
      arrays via `update_recon(grad, pixel_indices, …)` does NOT work: (a) `update_recon` is
      `arr.at[idx].add` into the **full-recon-domain** array, but `prior_grad`/`forward_grad` are
      **subset-domain** `(len(pixel_indices), num_slices)` already row-aligned — combining them is the
      plain elementwise add the code already does; routing through a scatter writes the wrong rows; and
      (b) it destroys `prior_grad`/`prior_hess`, which the alpha line search needs (`prior_linear`,
      `prior_quadratic`).  Even adapted it wouldn't cut peak (the back-project output and the
      `fm_hessian` gather materialize either way; scatters add buffers).  P3 is the correct expression of
      the same instinct (drop dead arrays, don't re-route live ones).
  - **Unrelated cache aside (resolved 2026-06-24):** a *separate* GPU failure mode — `NOT_FOUND` on
    `<cache>/xla_gpu_per_fusion_autotune_cache_dir/tmp/...textproto` — surfaced when running the full
    suite on A100s after an env rebuild.  Root cause: jax defaults `jax_persistent_cache_enable_xla_caches`
    to the per-fusion autotune cache, whose temp-file writes fail on cluster NFS / a fresh cache dir.
    Fix: disable it when we set the compilation cache (keep the executable cache).  Not the cone hang.

- **`split_sino_recon` hardening (cone, no-z-shift split-recon) — IN PROGRESS (2026-06-25).**  Review
  found transients/contract gaps that undercut the function's "do half at a time on a fixed GPU set"
  purpose (revisits #16, which didn't cover the memory/host transients).  Ordered plan:
  - **S1 — host-side stitch (BIGGEST WIN; doing first).**  `stitch_arrays` is `jnp`-based, so stitching
    the two `device_get`'d halves re-uploads them to gpu0 and assembles the FULL volume (+ concat
    transients) on one device — and `split_sino_recon` returns that gpu0 array.  For a recon too big to
    fit whole on the GPUs (the reason to split) the stitch itself OOMs.  Fix: make `stitch_arrays`
    **host-preserving** (`xp = jnp if any-input-is-jax else np`, float32 blend weights so a float32 recon
    is not upcast), mirroring the `gen_weights`/`generate_demo_data` pattern.  Halves are already on host
    (`device_get`), so the full volume stays on host and is returned as numpy.  Blast radius: only
    `split_sino_recon` + a shape-only doctest.
  - **S2 — stop mutating the caller's weights (CORRECTNESS) — DONE 2026-06-25.**  `weights[:, lo:hi, :]`
    was a numpy VIEW and the in-place sine taper wrote through it (tapered the caller's `weights`; a
    jax/sharded weights would crash).  Fix: coerce `sino`/`weights` to host (`np.asarray`) at entry, build
    the weight halves as fresh `np.array` copies, and make the `sine_filter` **float32** (out-of-place it
    would upcast the float32 weights to f64).  Verified: caller's `weights` unmutated; jax `sino`/`weights`
    run without crashing and match the numpy path; recon stays float32; `test_split_sino` green.  (Folds
    in **S6** — the host-input contract is now coerced + documented in the docstring.)
  - **S3 — half-models inherit the parent device config — DONE 2026-06-25.**  `copy_ct_model` copies
    params but NOT the device layout, so an explicit `configure_devices(...)` on the parent was dropped
    and an auto half could pick a different count for its smaller recon.  Fix: after each half's
    recon_shape is set, `configure_devices(self.shard_devices)` on both halves (rebuilds each placement
    for its own shape; inert slice padding makes any count safe).  Done in `split_sino_recon` (targeted,
    not `copy_ct_model`, whose only caller this is).  Verified at 4 virtual devices: parent pinned to 2 →
    both halves inherit exactly those 2 (previously they ignored it); `test_split_sino` green.
  - **S4 — one half at a time — DONE 2026-06-25.**  Both halves' sino+weights+models were built upfront
    and held together.  Fix: extracted a nested `_recon_one_half(lo, hi, recon_shape, recon_slice_offset,
    taper_top)` that builds the half's model + sino slice + weights, recons, and `device_get`s — all
    heavy state local, so the top half is fully done and freed before the bottom is built (only one
    half's inputs resident at a time).  Pure restructure (same per-half params/order); `test_split_sino`
    green and S2/S3 re-verified (weights unmutated, both halves inherit the parent's 2-of-4 devices).
  - **S5 — `weights=None` overhead (low priority).**  Materializes full-half `np.ones` and forces
    non-constant-weights recons (a full-half weights array resident on GPU per half).  Inherent to the
    overlap taper given the `recon` API; note it, free the host ones promptly.
  - **S6 — document/enforce the host-input contract — DONE with S2** (`np.asarray` coercion + docstring note).
  - **Gate:** `tests/geometries/test_vcd.py::...test_split_sino` (split-vs-full nrmse) + the
    `stitch_arrays` doctest; values must be unchanged (S1 is a placement change, not a math change).

- **NSI production workflow (Lilly_recon.py; num_metal=0, cone) — observations + plan (2026-06-26).**
  Traced for memory/transients + sharding.  The recon is already the hardened path (num_metal=0 cone →
  `recon_plastic_metal` → `split_sino_recon`), so the work is in the script + preprocessing.
  **IMPLEMENTATION ORDER (2026-06-28):** (1) export host-side fix (#4) + its guard test — production
  blocker, small, host-residence proxy not an 18 GiB run; (2) two-stage split (#2) — debugging +
  clean-process recon, depends on (1) to finish end-to-end; (3) #18 preprocessing/MAR sharding — the
  time win, biggest, last (4a `gen_weights_mar`, then 4b view-shard the sino pipeline).  (1)+(2) are
  CPU-developable and independent of the 8-GPU cone gating run (which stays top overall priority).
  - **#1 — clip on the host — DONE 2026-06-26.**  `sino = jnp.maximum(sino, 0.0)` pushed the FULL
    sinogram onto gpu0, and `gen_weights` then added a second full-sino gpu0 array, only for
    `split_sino_recon` to gather both back to the host — a wasteful round-trip that can OOM one GPU
    before the recon starts.  Fixed to `np.maximum` (host); `gen_weights` (host-preserving) then yields
    host weights and each half-recon shards its own half from the host.  (`jnp` import now unused in that
    script — optional cleanup.)
  - **#2 — two-stage preprocess/recon split (PLANNED; Greg wants it regardless, for debugging).**
    **IMPLEMENTED 2026-06-28:** `mj.preprocess.save_preprocessing`/`load_preprocessing` added to
    `preprocess/utilities.py` (moved there from `utilities.py` — it is preprocessing-output I/O, co-located
    with the preprocessing pipeline; HDF5: sinogram + array params as datasets, scalars as one JSON attr;
    round-trip test in `tests/test_preprocessing.py` — exact sino/float32, tuple-restored `sinogram_shape`,
    numpy-scalar coercion, optional custom weights).  Two scripts in `mbirjax_applications/nsi/`:
    `Lilly_preprocess_to_disk.py`, `Lilly_recon_from_disk.py`, plus `Lilly_two_stage.sh` (runs stage 2 in a
    SEPARATE process for the clean allocator).  (Scripts live in the separate `mbirjax_applications` repo;
    only the `preprocess/utilities.py` helpers are the mbirjax change.)
    **TWO SEPARATE SCRIPTS (Greg's call 2026-06-28), not a `--stage` flag** — separation of concerns +
    honest naming (Lilly_recon currently does preprocessing too) AND it *guarantees* the fresh-process
    benefit (a `--stage both` one-shot runs in one process and loses the clean allocator):
    - **`Lilly_preprocess_to_disk.py`** — `compute_sino_and_params` → `auto_crop_sino_conebeam` →
      `np.maximum` → `mjp.save_preprocessing(out, sino, cone_beam_params, optional_params)`.
    - **`Lilly_recon_from_disk.py`** — `mjp.load_preprocessing(path)` → build `ConeBeamModel` + set_params
      → recompute weights (`gen_weights`, one cheap pass — NOT saved) → `recon_plastic_metal` →
      `export_recon_hdf5`.  (sino + the two param dicts is everything the recon needs; voxel pitch for the
      filename comes from the rebuilt model.)
    - **`mj.preprocess.save_preprocessing`/`load_preprocessing`** (the ONLY mbirjax core add) — HDF5 (matches
      `save_data_hdf5`): sino as a float32 dataset + the two param dicts (scalars as attrs, `angles` as a
      dataset).  Reusable by other NSI scripts.
    - **Orchestrator:** a shell script (two `python` invocations) and/or a Python orchestrator — but it
      MUST launch the recon as a SEPARATE process (shell sequence / `subprocess`), NOT import-and-call,
      or the clean-allocator benefit is lost.
    Rationale: (i) inspect/reuse the preprocessed sino while iterating on recon params (debugging — the
    primary motive); (ii) a FRESH recon process starts with a clean GPU BFC allocator + no leftover jit
    caches from preprocessing's batched gpu0 work — matters for memory-tight large recons.  **Fate of
    `Lilly_recon.py`:** the two split scripts become primary; retire the combined script eventually (it
    has the preprocessing-in-recon smell), keep it short-term until `test_script_no_metal.sh` is
    repointed.  **Confirm-first:** a `memory_report` at recon start, single-process (post-#1) vs a fresh
    process, quantifies the clean-allocator gain.  Composes with #18-4b: a view-sharded preprocessing just
    gathers at the disk boundary, so the on-disk contract stays host numpy (what recon-from-disk wants).
  - **#3 — shard the sino preprocessing (part of #18; benefit = TIME).**  The pipeline
    (`compute_sino_transmission`, `correct_det_rotation`, `correct_background_offset`, defective-pixel
    interpolation) batches over views on ONE device and gathers each batch to host — serialized.  It is
    per-view **independent** (no cross-view coupling), so view-sharding it is **embarrassingly parallel →
    near-linear time speedup** across devices (cleaner than the recon's band-coupled sharding).  Mechanism:
    distribute the raw `obj_scan` by view across devices (blank/dark means are small/replicated), run the
    per-view ops per shard (existing per-device pool / a NamedSharding over views), and EITHER gather to
    host (for the disk split) OR keep view-sharded to feed the recon's view-sharding with no gather.
    Caveat: `multi_threshold_otsu` (MAR, num_metal>0) is global and would gather — out of scope for the
    num_metal=0 path.  Real work; fold into #18.  Plan reference: this block.
  - **#4 — `export_recon_hdf5` OOMs on large recons (Charlie, 2026-06-28) — CODE DONE 2026-06-28
    (validation pending an env fix; see below).**  `export_recon_hdf5` `device_get`s to host (utilities.py:581) but then ships
    the FULL volume back to ONE GPU twice: `apply_cylindrical_mask` (remove_flash=True) is `jnp`
    throughout (`recon * circular_mask` is a full-volume jnp multiply), and `jnp.transpose` (utilities.py:586)
    needs another full-volume buffer → `RESOURCE_EXHAUSTED` at downsampling 1 (`f32[1370,1880,1880]` ≈ 18
    GiB; the transpose is the dying line, but the mask is also on-GPU).  Same pattern as the host-preserving
    campaign (gen_weights / stitch_arrays / generate_demo_data / split_sino stitch): post-recon
    whole-volume ops on one device don't scale.  **Fix (keep the whole export host-side):** (a)
    `apply_cylindrical_mask` → host-preserving (`xp = jnp if jax else np`; the 2D circular + 1D slice masks
    are small, so a host recon's multiplies stay on host), (b) `jnp.transpose` → `np.transpose` in
    `export_recon_hdf5`, (c) fix the stale "processes ... in batches to avoid GPU memory issues" docstring
    (it does whole-volume ops).  The report's minimal `np.transpose`-only fix is INCOMPLETE — the mask step
    still puts ~18 GiB on one GPU and can OOM there first.  Gate: a downsampling-1-scale export (or a check
    that no GPU array is created post-device_get).  (Reporter's install was non-editable — needs reinstall.)
    **DONE 2026-06-28 (validated, env now jax 0.10.1):** the REAL blocker for the mask was that
    `apply_cylindrical_mask` is **`@jax.jit`-decorated** — jit forces a device array and traces the input,
    so the `xp` host-preserving logic was INERT and the mask still shipped the whole volume to one GPU
    (the first `np.transpose`-only attempt would still have OOM'd at the mask).  Fix: **removed the jit
    decorator** (it's a one-shot post-recon mask, not a hot path; both callers — export + MAR
    segmentation — are non-hot) so the eager `xp` logic is genuinely host-preserving.  Plus `np.transpose`
    (host view) and docstrings.  Guard test `tests/test_utilities.py::TestExportReconHostResidence`
    (host-residence proxy: numpy-in→numpy-out for the mask, jax-in→jax-out, small export→import
    round-trip).  **Syntax-checked (py_compile); FULL validation PENDING an env fix** — the `mbirjax`
    conda env is broken (jax 0.4.21 vs jaxlib 0.10.1: the `pip install -e .` resolved an old jax against
    the loose `jax!=0.10.2` pin).  Fix = `pip install "jax==0.10.1"` (match jaxlib; ≠ 0.10.2 per the
    deliberate exclusion), then run the guard test + suite.  Consider tightening the pin (e.g.
    `jax>=0.10,!=0.10.2`) so `-e` can't pull a stale jax again.
  - **Problem 2 — benign `cuda_vmm` FABRIC warnings on ≥2 GPUs (Charlie) — no real plan change.**  XLA probes
    an NVLink-fabric memory handle the cluster disallows, prints `W… will retry with simpler handle types`,
    falls back to the standard peer path → no correctness/perf impact; ≥2 GPUs only; already in lessons.md
    (cuda_vmm_allocator).  Suppress with `TF_CPP_MIN_LOG_LEVEL=2` BEFORE `import jax` (already in
    tests/conftest.py); optionally set it in the NSI app scripts (caveat: hides ALL XLA warnings — unset
    when debugging).  Low priority, app-env-var level.
  - **#5 — `generate_demo_data` / shepp-logan generator: single `devices`, always host NumPy — DONE
    2026-06-28 (Greg's design).**  Was: `generate_demo_data` without `devices` built the phantom on
    `recon_placement.devices[0]` (ONE GPU) even though the model spanned 8 → the full 2048³ phantom
    (32 GB) materialized on gpu0 (+ a full-volume `* scale` ~64 GB transient + the reshard/gather) →
    OOM.  And the generator took a confusing `device`-vs-`devices` pair.  Now: (a)
    `generate_3d_shepp_logan_low_dynamic_range(phantom_shape, devices=None, ...)` — single `devices`
    (None → all available: GPUs else CPU); builds slice-sharded across them in parallel (or row-blocked
    on a single device), **gathers to host, frees the device arrays, returns NumPy** cropped to the real
    shape; `scale` folded per block/shard (no full-volume scale transient).  (b) `generate_demo_data`
    builds the phantom on the **model's** devices, forward-projects, and gathers **both phantom and
    sinogram to host NumPy** — `output_sharded` DROPPED (the phantom is a reference; recon prefers a host
    sinogram it shards itself).  So 2048³ generate_demo_data shards the phantom across all 8 GPUs (≈4
    GB/shard) instead of 32 GB on gpu0.  Tests updated (`TestShardedSheppLogan`/`TestGenerateDemoDataSharding`
    → host-numpy + value-match, 1-device vs N-device; the phantom is bit-identical 1-vs-N but only
    ~1e-5-close across two different shard counts — XLA float32 fusion varies by shard shape).  Also
    de-`device`-d the deprecated `gen_modified_3d_sl_phantom`.  **Follow-up fix 2026-06-28:** the sinogram
    must be gathered with `forward_project(..., output_sharded=True)` + `np.asarray` (shard-by-shard to the
    host).  The first version used `forward_project`'s default gather, which routes the whole sinogram
    through ONE device (`jnp.array(np.asarray(...))`) and OOM'd on the real 2048³/8 run (32 GiB on gpu0);
    `output_sharded=True` keeps it 4 GB/shard and `np.asarray` of a multi-device array assembles on the
    host.  (`save_preprocessing`/`load_preprocessing` also moved to `preprocess/utilities.py` →
    `mj.preprocess.*`.)
  - **#6 — host-safe `output_sharded=False` gather (the real fix for the single-device-gather footgun) —
    DONE 2026-06-28 (full suite green, 200 passed).**  Implemented all four sites: `_gather_to_host` →
    `np.asarray(x)` (host assembly, both shard counts); dropped the `jax.device_put(..., devices[0])`
    re-uploads at the `recon` and `prox_map` exits; `direct_recon` cylinder → host `np.zeros` + numpy
    scatter.  `output_sharded=False` (recon/forward/back/fbp/fdk/direct/hessian/prox/denoise) now returns
    **host NumPy** (docstrings updated; the misleading "single-device array" wording fixed).  Guard test
    `tests/sharding/test_vcd_sharded.py::TestGatherReturnsHostNumpy`.  One test fixup: removed a now-invalid
    `recon.block_until_ready()` in `test_vcd.py` (jax-only; the host gather already syncs) — the only place
    a jax method was called on a gathered output.  **PR note:** `output_sharded=False` returns host numpy
    (was single-device jax).  Original plan below.
  - **#6 (plan, now DONE above) — host-safe `output_sharded=False` gather.**  The default gather routes the whole volume through ONE device — the recurring footgun
    (hit at recon-exit, export, demo-data).  `_gather_to_host` does `jnp.array(np.asarray(x))` (re-upload
    to gpu0), AND callers materialize the full volume on one device: the `recon` exit (`jax.device_put
    (_gather_recon(recon), devices[0])`), the `prox_map` exit (same), and `direct_recon`'s cylinder
    scatter (`jnp.zeros(recon_shape)` + `.at[].set`).  **Plan (its own change + full suite):** (a)
    `_gather_to_host` → return host numpy (multi-shard `np.asarray(x)`; single-shard
    `np.asarray(shards[0].data)`); (b) drop the `jax.device_put(..., devices[0])` re-uploads at the recon
    and prox exits; (c) `direct_recon` cylinder → host `np.zeros` + numpy scatter; (d) verify
    `_gather_sinogram`/`_gather_recon` (slice-only), the denoiser, hessian, and the compute_prior_loss
    path; (e) docstrings + a host-residence test + full suite + a GPU memory_report confirm.  **Contract
    change:** `output_sharded=False` (recon / forward / back / fbp/fdk/direct / hessian / prox / denoise)
    returns host NumPy instead of a single-device jax array — matches the "plain array" docstrings, frees
    the device, host-safe at any size; PR-note it.  Risk low–moderate (no test asserts the jax return
    type; downstream use is numpy-safe).  Removes the need for per-call-site `output_sharded=True`.
  - **(revisit) `_generate_3d_shepp_logan_sharded` device loop — is it actually parallel?**  The loop
    DISPATCHES each device's band build inside `with jax.default_device(dev)` and relies on async dispatch
    to overlap.  Observed 2026-06-28: wall time is the same with or without a `piece.block_until_ready()`
    before each append — so either the dispatch is already overlapping (the block is a no-op on the
    critical path) OR the builds are not actually concurrent (and the block just isn't the bottleneck).
    Low stakes (the build is fast and gathered to host anyway), but worth confirming the loop overlaps as
    intended (a per-device timing trace) rather than serializing.

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
