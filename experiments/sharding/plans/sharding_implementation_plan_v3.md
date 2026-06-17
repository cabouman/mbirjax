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
| **Cone B5** — exactly-inert slice padding | 🚧 **next** (unblocks 4 deferred tests at non-dividing counts) |
| **Cone B4.5** — band kernel GPU cost | ⏸ open (deferred behind decision (a); see §4) |
| **C** — convert ParallelBeam to the `(g0,L)` template; delete monolithic cone kernel | ⬜ pending |
| **D** — translation + multiaxis ports | ⬜ pending |
| **E** — retirement cascade (legacy single-device paths, `main_device`, `view_indices`, …) | ⬜ pending (last — needs all geometries ported) |

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
- **Threading (Path G), not `shard_map`.**  Manual collectives over `addressable_shards`
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
- **GPU n=1 short-circuit:** a single-GPU mesh routes to the monolithic **pixel** kernel
  (~2.25× faster on GPU; the band kernel is platform-opposite — see §4), wrapped as a
  1-shard slice-sharded array; CPU keeps the band path.

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
  geometry-neutral base hook; ParallelBeam overrides with its row-crop).  Forward = **decision
  C**: per-pixel-batch all-gather + monolithic (the banded/streamed forward measured 5–14×
  slower on CPU for ~13–23% memory).  `_supports_sharding()=True` for cone.  CPU-validated end
  to end at dividing counts; **GPU-confirmed** (B4.4: per-device peak ~1/n_dev, the 1024³ VCD
  capacity win).
- **Shared FBP/FDK filter** (cone delegates with a cosine `row_weight`); **GPU n=1 back
  short-circuit** (platform-divergent kernel; GPU-confirmed: 2.25–2.46× faster, back = main).

### Open / next
- **B5 — exactly-inert slice padding for cone (NEXT).**  Unblocks the **4 deferred test
  failures** at non-dividing slice counts (≥4 devices auto-pad; the forward gather assembles
  the padded cylinder and the device-form shape leaks to tests asserting the real shape).
  Scope: crop the forward gather to the real slice count (padded slices are zero → exact);
  reconcile device-form-vs-real shape in the geometry tests / the internal
  `sparse_back_project` contract; consider a `_supports_slice_padding()` hook so "B4 =
  dividing counts only" is explicit (cone False until B5).  The masks + the B1 global clip are
  already geometry-agnostic.  (Failing tests reproduce on CPU with `MBIRJAX_NUM_CPU_DEVICES=4`.)
- **B4.5 — band-kernel GPU cost (deferred).**  The band (reduce-scatter) kernel is **~2.25×
  slower than the pixel kernel ON GPU** (and the two are **platform-opposite**: CPU loves the
  band kernel, GPU loves the rolled-pixel kernel via the ×62 back-vertical cache cliff).  So
  multi-device back doesn't pay in *time* until ≥3 GPUs (crossover n≈2.25); VCD stays monotonic
  only because the forward parallelizes and masks it.  The real challenge is a kernel fast on
  *both*, else platform-specific selection (what the n=1 short-circuit already does).  Deferred
  behind **decision (a)** — sharding is the capacity tool beyond 1024³, not a back-time lever.
  *Alternative axis to consider:* sharding the sinogram by **detector row** instead of by view,
  so the sino's sharded axis aligns with the recon's slice sharding — turning back projection
  into a mostly-local op (a geometry-driven footprint halo) rather than a view-reduce, which
  would sidestep the band-kernel reduce-scatter cost entirely.  Parked exploration; full
  analysis in `.claude/sinogram_sharding.md`.
- **FDK filter → sharded contract (cleanup).**  `ConeBeamModel.fdk_filter` still runs
  single-device, so a sharded cone `direct_recon`/`fdk_recon` init gathers → filters → re-shards
  (host round-trip + single-device bottleneck at init).  Convert to the `fbp_filter` pattern:
  dispatch the cone FDK weighting+ramp per view-shard via `run_per_device`, honor
  `output_sharded`.  Non-blocking (FDK = init only); likely lands with C.

---

## 5. Remaining work & execution order

1. **Cone B5** (inert padding) — finishes cone correctness at *all* device counts; unblocks the
   4 tests.
2. **C** — convert ParallelBeam to the `(g0,L)` banded template; delete the monolithic cone
   kernel + the transitional "geometry has banded kernels?" branch; fold in the FDK-filter
   cleanup.
3. **D** — translation + multiaxis ports (same template; note `MultiAxisParallelModel` extends
   `TomographyModel` directly → it gets its own `_supports_sharding` flip).
4. **E — retirement cascade** (only after *all* geometries are ported): `main_device` /
   `sinogram_device`; the `is_sharded` else-branches + legacy single-device bodies; the
   `view_indices` machinery (incl. the test_projectors pin); `initialize_recon`'s early
   `device_put` block; `compute_hessian_diagonal`'s `output_device`; then a
   `grep -rn "RETIRE-AFTER-SHARDING"` sweep (the trivial-mesh comparison tests retire).
5. **Post-P6** — the multi-GPU **user docs page** (use the word "sharding" sparingly);
   the choose-N-vs-communication policy (+ the CPU-cluster auto policy); **B4.5** if
   multi-device back *time* ever matters; revisit the prox-map prior under sharding if a
   PnP-at-scale need appears.

**Footgun to carry into E:** `is_sharded` and `n_devices > 1` have **decoupled** —
`is_sharded` is ~always True now, so every `is_sharded` site must be re-read for which question
it asks ("do I have a placement?" vs "do I have ≥2 physical devices?" = `len(shard_devices) > 1`).

---

## 6. Adjacent / tracked items (off the critical path)

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
