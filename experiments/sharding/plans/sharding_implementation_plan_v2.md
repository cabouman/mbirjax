# Sharding implementation plan v2 — placement architecture (beta `greg/parallel_sharding`)

*Created 2026-06-02.  This is the **current forward plan**.  It supersedes
`sharding_implementation_plan.md` for forward planning; that doc is retained as
the **completed-work record** (Phases 0/A/B/F1/D/F2 case studies) plus the
still-valid **cross-cutting principles, verified hardware facts, and resolved
open questions (O1–O4)** — read it for history and principles, this doc for
where we are going next.*

---

## Why this plan exists (direction change, 2026-06-02)

Phases F1 (FBP filter), D (back projection), and F2 (`direct_recon`, with the
match-input contract) are **done and GPU-validated**.  In designing the device
configuration UX we converged on a cleaner target architecture than the original
plan assumed, and decided to **re-open back projection** to build on it rather
than retrofit later.  The two ideas:

1. **Placements, not device scalars.**  Replace `main_device` / `sinogram_device`
   with `recon_placement` / `sino_placement` (each a `Sharding`).  Every device
   mode becomes a *placement pair*; single-device is a trivial 1-shard placement.
   Sharding is then **always on** (trivially when degenerate) — one code path.
2. **One movement interface.**  All recon↔compute data motion goes through an
   adjoint pair, `move_cylinders_to_sino` / `sum_cylinders_to_recon`, built on the
   existing `move_shard` primitive, with uniform **pixel-batched** streaming.

What does **not** change: F2's user-facing **match-input** contract and the
`direct_recon` scaling driver are at the public-API layer, independent of back
projection's internals — re-opening D does not disturb them (just re-run the
driver against the new internals).

---

## 0. Design summary

### 0a. Placements

- `recon_placement` and `sino_placement` replace `main_device` /
  `sinogram_device`; each is a `Sharding` (a mesh + shard devices; a single
  device is the trivial 1-shard case).
- Every mode is a placement pair:

  | mode | recon_placement | sino_placement |
  |---|---|---|
  | CPU-only (`none`) | trivial on CPU | trivial on CPU |
  | single-GPU (`full`) | trivial on GPU | trivial on GPU (same) |
  | hybrid (big recon) | trivial on CPU | trivial on GPU |
  | multi-GPU (`sharded`) | slice-sharded on GPU mesh | view-sharded on **same** GPU mesh |

- **Sharding is always on**; single-device is the degenerate trivial placement,
  so there is no separate non-sharded path (the trivial-sharding unification).
- **Scope.**  Support homogeneous multi-GPU and **recon-on-CPU / sino-on-GPU**
  (the big-recon case).  **Out of scope:** sino-on-CPU / recon-on-GPU (would only
  serve host-streaming a too-big *sinogram*; rare in CT, and Greg's call is we
  won't support it).  Multi-*node* is out of scope (single process only).
- The hybrid case uses **two separate single-device meshes** (CPU mesh for recon,
  GPU mesh for sino) — never one mixed mesh — which sidesteps the JAX
  heterogeneous-mesh fragility that O1 worried about.

### 0b. Movement interface (the one thing that crosses placements)

Only **voxel cylinders** move between placements; the sinogram is written
locally on its view-shard and never moves during projection.

- `move_cylinders_to_sino(cylinders)` — forward: gather the full cylinders (all
  slices) for a pixel batch onto each sino device.
- `sum_cylinders_to_recon(partials)` — back (adjoint): reduce per-sino-device
  partial cylinders over devices and scatter slice-ranges into the recon
  placement.
- **Take cylinders directly** (no pixel indexing inside): the pixel axis is
  unsharded, so the caller does `flat_recon[pix]` (a clean slice-sharded
  sub-array) and the write-back; the primitives deal only with the slice/device
  structure.  Keeps them pure and unit-testable.
- **Built on `move_shard`**, looping over the placements' shards: `N×N` for
  homogeneous multi-GPU, `1×1` for single-device/hybrid — **no mode branch**, the
  shard counts come from the placements.  `move_shard` to the same device must be
  a true **no-op** so the single-device path carries zero overhead.
- **Adjoint pair** (gather+concat ⊣ slice+sum) → keeps the projectors adjoint →
  the forward/back adjoint round-trip is the correctness gate.
- **As built (P2/P3):** the side-by-side measurement chose **slice-banding** over
  pixel-batched cylinders, so `move_cylinders_to_sino` / `sum_cylinders_to_recon`
  were **removed** and replaced by the banded adjoint pair `sum_band_to_owner`
  (reduce-scatter, back) / `broadcast_band_to_views` (all-gather, forward) in
  `_sharding/transfer.py`.  The bullets above describe the original cylinders-direct
  design (kept for rationale/history).

### 0c. Streaming = uniform pixel-batching

- The streaming unit is a fixed-size batch of **full cylinders** (a pixel batch),
  uniformly for both directions and both modes — the existing
  `transfer_pixel_batch_size` pattern, generalized.  `B_p` is the memory knob
  (bounds `B_p × num_slices` per device).
- The layer-2 driver is a single uniform pixel-batch loop with the geometry
  kernel injected; no mode-specific band axis.
- **Superseded — see P2's verdict (KEEP BAND).**  The side-by-side measurement
  chose slice-banding over pixel-batching (pixel's peak floored at ~1.7–2.7× band
  for only ~11–16% less time), so the as-built streaming unit is the **slice-band**,
  not a pixel batch.  `_slice_band_length` sizes it; a per-band compute cap
  (`_BACK_PROJECT_MAX_BAND_WORK`) also makes a single device stream.  The
  pixel-batching text above is retained for design rationale only.

### 0d. Vocabulary — user-facing vs internal

- **User-facing word is "devices."**  `configure_devices(devices=None)` is the
  control surface; "shard / mesh / placement" stay internal.  Long term, one
  concept: `configure_devices` subsumes `use_gpu`.
- **Internal** keeps `mesh`, `shard_devices`, `_sharding/`, `move_shard`,
  `recon_placement` / `sino_placement`, `move_cylinders_to_sino` /
  `sum_cylinders_to_recon`.

---

## Phase sequence

Build everything **next to the existing code on ParallelBeam first** (the proven
F1 pattern: parallel path → measure → promote winner → **delete loser**; do not
leave two paths lingering), then port to the other geometries.

### P1 — Placement foundation  ✅ DONE (committed)
- [x] `Placement` (`mbirjax/_sharding/placement.py`): device list + sharded axis
  + 1-D mesh; `shard_ranges()` / `shard_structure()` / `is_trivial`.
- [x] `move_cylinders_to_sino` / `sum_cylinders_to_recon` (cylinders-direct,
  `move_shard`-based, `N×N` / `1×1`, same-device no-op).  Adjoint identity tested.
- [x] Model builds `recon_placement` / `sino_placement` via `_set_placements()`
  (additive; coexists with `main_device`/`sinogram_device` + `mesh`, which the
  `TODO(P6)` retires).
- [x] Tests: `test_sharding_placement.py` (primitives + adjoint);
  `test_sharding_hooks.py::TestModelPlacements` (model construction + consistency).
- Deferred (intentionally): the broad `device_put → move` migration — it happens
  organically as each path is rewritten (P2/P3/P4), not as upfront churn.

### P2 — Back projection on placements (re-opened Phase D)  ✅ DONE (CPU + GPU)
- [x] `_sparse_back_project_pixel`: pixel-batched, uses `sum_cylinders_to_recon`;
  first consumer of `recon_placement` + the movement primitives.  Runs **alongside**
  the slice-banded `_sparse_back_project_sharded` via the temporary
  `_back_project_path` switch ('band'|'pixel'); shared setup in
  `_sharded_back_project_setup`.  `B_p` knob = `back_project_pixel_batch` /
  `_BACK_PROJECT_MAX_PIXEL_WORK`.
- [x] Tests: `TestPixelBackProject` (trivial bit-exact; matches band + single-device;
  Hessian; B_p sweep; match-input sharded-in).
- [x] Harness: `sparse_back_project_scaling.py` measures both paths side by side
  (+ topology/`dev2dev_safe`/per-GPU-throttle capture, `DEVICE_COUNTS`,
  `PIXEL_BATCH_SWEEP`).
- **Findings so far (clean H100×3 GPU run, 1008³):** time comparable (pixel edges
  ahead and scales slightly better at scale — 3.12× vs 2.75× on 3 dev); band uses
  **~2–2.7× less memory** (structural).  Tradeoff: pixel = simpler + scales
  as-well-or-better; band = memory-lean.
- **Decision rule:** the `B_p` sweep decides — if a smaller `B_p` brings pixel's
  memory to band's level without losing its time, **adopt pixel** (retire the band
  code: `_sparse_back_project_sharded`, `_slice_band_length`,
  `_balanced_slice_bounds`, `back_project_slice_band`; rename
  `_sparse_back_project_pixel` → `_sparse_back_project_sharded`; drop the switch);
  else keep band.  **VERDICT (2026-06-03): KEEP BAND.**  The B_p sweep showed
  pixel's peak memory plateaus at ~1.7–2.7× band (untunable — a floor below the
  per-batch transient) for only ~11–16% less time; memory/max-recon-per-GPU wins.
  The pixel path, its `_back_project_path` switch, and `back_project_pixel_batch`
  knob were removed; band remains `_sparse_back_project_sharded`.
  `_sharded_back_project_setup` (the factored shared setup) is kept.
- Anchors (met): trivial bit-exact vs prerelease; n>1 float-noise match.

### P3 — Forward projection on placements (Phase C)  ✅ DONE (CPU; standalone GPU forward-scaling sweep optional)
- [x] **Slice-banded all-gather** forward projection on ParallelBeam — the adjoint
  of the band back projector (chosen over the pixel-batched `move_cylinders_to_sino`
  for memory symmetry with back projection, same reasons band beat pixel in P2).
  New `broadcast_band_to_views` (`_sharding/transfer.py`), the adjoint of
  `sum_band_to_owner`; `_sparse_forward_project_sharded` + per-band helpers; the
  parallel-beam forward kernel sizes its output rows from the input slices.
- [x] **match-input** per the F2 contract (mirror of `back_project`).
- [x] Correctness gate: forward/back **adjoint round-trip** `<Ax,y>==<x,Aᵀy>` green
  at 1/2/4/8 devices; trivial bit-exact; prerelease forward baseline match (1.1e-5).
  Committed (`7461abb`).
- The superseded full-cylinder primitives `move_cylinders_to_sino` /
  `sum_cylinders_to_recon` were **removed** (the banded pair supersedes them; see the
  "as built" note under 0b/0c).
- The geometry-neutral `(g0, L)` slice-band interface was **designed** here (note
  below) but not yet built for parallel beam, which keeps its bit-exact external
  row-crop; the `(g0, L)` kernel lands when the geometries are ported.
- **Pending (optional):** standalone GPU forward-scaling sweep
  (`sparse_forward_project_scaling.py`); forward is already exercised on GPU inside
  the VCD scaling run.

### Design note — geometry-neutral slice-band projector interface (decided 2026-06-04)

Today the sharded back path streams the recon's slice axis in **bands** but
positions each band by **row-slicing the sinogram** (`data[:, g0:g1, :]`), which
only works for parallel beam (detector row `r` ↔ slice `r`).  The agreed
generalization:

- **Interface (B2): `(band_start g0, band_length L)` in *global slice-index*
  units**, passed to the per-view projector; the projector reads the full
  (resident) view and emits exactly slices `[g0:g0+L)`.  `g0` is a **dynamic
  (traced)** scalar; `L` is **static**.
  - *Why integer slice index, not a physical offset / `recon_slice_offset` (B1):*
    a per-geometry survey shows the z-shift is a *different* quantity in each
    geometry, so `recon_slice_offset` is **not** a uniform currency:

    | geometry | slice→row map | z-shift it uses | internal slice-batching |
    |---|---|---|---|
    | `parallel_beam` | positional, row = slice | *none* | *none* |
    | `cone_beam` | z-based vertical fan | `recon_slice_offset` (+ per-view helical) | `entries_per_cylinder_batch` |
    | `multiaxis_parallel` | z-based vertical fan | `recon_slice_offset` | `entries_per_cylinder_batch` |
    | `translation_model` | z-based vertical fan | `translation_vector[2]` (per view) | `entries_per_cylinder_batch` |

    The integer band *is* uniform.  Each geometry converts internally: parallel →
    rows `[g0:g0+L)` via `dynamic_slice` on the resident view; the z-based three →
    compute z for global indices `k = g0 + k_local` using their existing shift
    term.  `recon_slice_offset` stays each geometry's private parameter; **nothing
    is added to parallel or translation.**  This also keeps the door open for a
    future *row-sharded* parallel beam (a device just owns band
    `(g0=row_start, L=row_count)`, no reduce) — adding `recon_slice_offset` to
    parallel beam would have fought that.
- **Recompilation discipline:** `g0` **dynamic**, `L` **static** → ≤2 compiles
  (the balanced split yields ≤2 band lengths).  The per-band position must **not**
  ride through `recon_slice_offset`/`recon_shape`, which are closure/`static`
  constants of the projector — varying them per band rebuilds the projector
  (~`n_dev²` times).
  - *Rejected:* conveying `L` by passing an empty output cylinder — JAX keys the
    jit cache on input *shapes* exactly as on static-arg values, so it saves no
    compile and just adds a throwaway allocation.  Use a static `num_output_slices`
    arg.
  - *Assembly / in-place:* keep `concatenate` for the **multi-device threaded
    reduce-scatter** (an in-place `dynamic_update_slice` accumulator only goes
    in-place under `jit` + `donate_argnums`; in our Python band loop it copies
    per band and serializes the band-to-band streaming overlap).  In-place
    donation **is** the right assembly for the *jittable* single-device streaming
    (P6) and the forward accumulation (P3) — file it there.
- **Slice-batching subsumption (P6):** the z-based geometries' internal
  `entries_per_cylinder_batch` chunking *is* banding one layer down.  Once the
  kernel takes `(g0, L)`, the outer band loop subsumes it and that machinery is
  deleted from cone/multiaxis/translation — but this only fully lands when the
  **single-device path also drives band-looping** (the trivial-sharding
  unification), since today the internal chunking is what bounds single-device
  memory.
- **Adjoint test:** add the same `(g0, L)` to the forward projectors; "forward a
  band, back-project it, compare" at arbitrary `(g0, L)` becomes a strong
  per-geometry gate.

**Implementation note — why this is P3, not a quick parallel-beam change now.**
Inspecting the call stack: the per-view kernel is vmapped by the *shared* generic
layer (`projectors.py` `sparse_back_project_view_batch`, the
`jax.vmap(back_project_one_view_to_pixel_batch, ...)`), so adding a band argument
the kernel uses *internally* changes the generic projector API **and every
geometry's kernel signature at once** — there is no clean parallel-beam-only
version.  For parallel beam specifically the only *localized* way to band is the
**current external row-crop**, which is already bit-exact.  So: leave the external
row-crop as parallel beam's band mechanism for now, and build the `(g0, L)`
interface once in P3 across forward + back with the adjoint round-trip as the
gate.

### P4 — VCD on placements (Phase E)  ✅ DONE — multi-GPU validated (1–4 GPU); sharded-memory leak fixed & validated for all paths (const, non-const weights, positivity)
- [x] Entry/exit placements for sinogram (view) / recon (slice) / weights (view) via
  `to_sino`/`to_recon` in `vcd_recon` (replace the `device_put(..., main/sinogram_device)`
  gather hazards).
- [x] Halo exchange via `_extract_halos` in the new `_qggmrf_prior_sharded` — the
  recon-domain analogue of the sharded projectors (per-slice-owner local prior +
  `assemble_sharded`; halo-aware `qggmrf_*` ported from research, default `None` =
  reflected BC = bit-exact).
- [x] Default init: `direct_recon(sharded sinogram)` → slice-sharded init (match-input),
  `forward_project(init)` → view-sharded error sino; stays sharded into the loop.
- [x] No accidental gather/re-shard inside the loop body: recon-domain gathers use a
  mesh-replicated `recon_indices`; the cross-mesh line-search `alpha` is reduced to a
  host float (the two multi-device bugs the tests caught).  Audited by a test that
  `vcd_recon`'s pre-exit-gather return stays slice-sharded across all devices.
- Gate: full recon trivial **bit-exact**; 2/4/8-dev **NRMSE ~6e-7** vs single-device;
  prior trivial bit-exact + 2/4/8-dev float-match.  `tests/sharding/test_vcd_sharded.py`.
- [x] **GPU multi-device scaling validated, 1–4 GPU** (H100, `vcd_recon_gpu.yaml`, 10
  iters): 1008³ **1d→4d = 4.49× — super-linear** (bandwidth-bound op + smaller per-device
  working set; 2→4d = 2.16×); memory shards cleanly (1d 55.5 → 4d 12.7 GB/dev);
  correctness vs prerelease **8.79e-7**.  504³ is the size-floor (4d 1.92×, too small to
  amortize cross-NUMA).  The memory fix ENABLED the 1008³/1d measurement (was OOM).
  Prior-opt A (stage halos once/pass) + the on-device `alpha` replication (zero
  per-subset host syncs) remain in place.
- **Sharded-memory leak — DIAGNOSED & FIXED (the single-device / 1-device-mesh
  no-regression gate).**  Symptom: the sharded path's peak grew ~3.3× the no-mesh peak
  at 504³ (24.6 vs 7.4 GB) and OOM'd at 1008³, while the **default no-mesh path is
  fine** (equal-or-better than prerelease).
  - *Root cause:* **jax keeps sharded (`NamedSharding`) arrays in internal reference
    cycles** (an `ArrayImpl`'s `__dict__` holds its sharding/buffers and back-refs), so
    they are freed only by the **cyclic GC**, not by refcount.  The VCD subset loop
    updated the view-sharded **error sinogram out of place** every subset
    (`error_sinogram = error_sinogram - alpha*delta_sinogram`), allocating a fresh full
    sinogram each time; the stale ones (and the transient `delta_sinogram`) piled up —
    **one full view-sharded sinogram per subset** — until GC, so peak scaled with the
    number of subset-updates (subsets × passes).  Diagnosis was empirical: an
    instrumented memjump trace showed the `P('devices',None,None)` (view-sharded sino)
    count climbing 1/subset, `live_end` only dropping to 512 MB *after* an explicit
    `gc.collect()`, and `gc.get_referrers` naming the holding `ArrayImpl.__dict__`.
    The **recon side never leaked** because `update_recon` already updates `flat_recon`
    **in place via buffer donation** — that asymmetry was the tell.
  - *Fix (mesh-guarded; single-device path untouched):* new module-level
    `update_error_sinogram` does `error_sinogram - alpha*delta_sinogram` under
    `@partial(jax.jit, donate_argnames='error_sinogram')` so XLA updates the error
    sinogram **in place** (mirrors `update_recon`); the sharded branch of
    `vcd_subset_updater` releases the constant-weights alias (so the buffer is
    donatable), calls it, then **explicitly `.delete()`s the transient `delta_sinogram`**
    (also cycle-held) after a one-line `block_until_ready` for async safety.  Net:
    **peak flat at the `ns=1` floor (504³/5-iter: 25.8 → 6.9 GB), no per-subset growth,
    time unchanged (~47 s), no donation warning.**  Also lifts per-device memory at
    `n_dev>1` (raises max-recon-per-GPU), not just the degenerate 1-device mesh.
  - *Validated:* local correctness GREEN — `tests/sharding/test_vcd_sharded.py` 8/8
    (incl. trivial **bit-exact**; the alpha scale is kept eager so the donated update is a
    pure subtract — folding it in would emit an FMA and break bit-exactness), single-device
    `test_vcd` regression, single-device-vs-prerelease **5.3e-8 unchanged**.  **GPU 1-device
    (H100):** 504³/5-iter mesh peak **25.8 → 6.9 GB** (= no-mesh 7.8), 1008³/5-iter mesh
    **OOM → 56 GB, completes** (no-mesh 53) — **1-GPU no-regression gate MET**; `live_end`
    flat (no accumulation).  Minor: 1008³ mesh ~+6% over no-mesh (per-band/assemble
    transients; benign, shards away at n_dev>1).
  - *Consolidated cleanup (done):* the per-subset transient frees are now ONE cleanup
    section at the end of `vcd_subset_updater` (mesh only): after a single
    `block_until_ready` on the returned state, delete the scaled `delta_sinogram` and —
    for **non-constant weights** — the `weighted_error_sinogram` product.  Correctness
    coverage added (`test_vcd_sharded.py`, +4: non-const & positivity × trivial-bit-exact /
    2-4-8-dev NRMSE); full `tests/sharding/` + `tests/test_vcd.py` **77 passed**.
  - *Non-constant weights & positivity — VALIDATED (bounded).*  GPU (504³/5-iter/1-device-mesh):
    **non-const peak 7.9 GB**, **positivity peak 7.3 GB** — both ≈ const 6.9 GB, flat, no leak.
    The `weighted_error_sinogram.delete()` bounds non-const; positivity needs **no** extra
    delete (its `delta_sinogram` recompute and stranded original are forward-projection
    (`assemble_sharded`) outputs that free on refcount — the cycle is only in *eager-op*
    outputs).  (An earlier 33.4 GB "non-const leak" was a **stale GPU build** running the
    pre-fix binary — a fresh `pip install -e .` resolved it; ruler-before-code.  The
    temporary `_memprobe` localizer added to chase it has been removed.)
  - *Refactor (done):* `TomographyModel.is_sharded` property is the single source of
    truth for the sharded code path (replaced 21 `self.mesh is None/not None` checks; the
    body changes in one place at the mesh→placement migration, and this is what retires
    at unification).
- **Also pending:** hybrid timing for the `qggmrf_..._transfer` variant (deferred
  Q3); prox-map prior under sharding (untouched).

### P5 — Device-config UX (can land in parallel once P1 fixes what gets configured)
- `configure_devices(devices=None)` user-facing (None→auto, list→exact,
  int→count); sets the placements.  Keep an explicit-pin flag.
- **Automatic selection in `set_devices_and_batch_sizes`:** choose N = largest
  device count that divides both sharded axes *and* keeps per-device work above a
  floor (handles small-problem→fewer-devices + divisibility together); re-validate
  and warn on a geometry change that breaks the fit.
- `use_gpu='sharded'` as the incremental opt-in / internal state; `'automatic'`
  shards by default **after P4**.
- **Divisibility:** warn loudly *with fix instructions* (never silently idle
  GPUs); `prepare_sino_for_devices(sino, weights, n)` helper for the clean
  slice/row axis; view axis → choose a dividing N (uneven shards unavailable —
  JAX equal-shard restriction).
- **Always-on "hardware in use + why" report** at recon time.
- `auto_set_recon_geometry` / `scale_recon_shape` made device-aware (mainly a
  cone-beam lever; parallel-beam sharded axes are sinogram-side).

### P6 — Port + retire
- Port the sinogram transpose pattern from `back_project_one_view_to_pixel_batch` 
  and `forward_project_pixel_batch_to_one_view` in parallel beam to cone / 
  translation / multiaxis geometries.
- Port the placement/movement path to cone / translation / multiaxis geometries
  (each implements the `(g0, L)` slice-band projector interface from the P3 design
  note).
- Retire `main_device` / `sinogram_device` entirely (the trivial-sharding
  unification); remove the old single-device and slice-band code paths.
- With the single-device path unified onto band-looping, **delete the per-geometry
  `entries_per_cylinder_batch` slice-batching** (subsumed by the outer band loop)
  and switch the jittable single-device / forward assembly to an in-place
  `dynamic_update_slice` accumulator with `donate_argnums` (see the P3 design note).

**STATUS (2026-06-08): step 4 IMPLEMENTED as "2-Keep" — CPU-green, GPU re-validation pending.**
Change 1 (note 2: fold `alpha` into the donated FMA `update_error_sinogram`, drop the
`scaled_delta` transient + its `.delete()`; const-weights cleanup section now empty/skipped) and
Change 2 (ParallelBeam auto-defaults the *homogeneous* single-device case to a trivial 1-device
mesh via the new `_supports_sharding()` hook + `_sharding_configured` flag) both landed.  The
**heterogeneous recon-CPU/sino-GPU 'sinograms' mode stays on the legacy single-device branch**
(deliberate 2-Keep scope; a CPU+GPU pair is not one mesh), so `_transfer` is untouched and
note 3 / "2-Drop" is deferred pending the GPU `_transfer`-vs-band-streaming timing.  The flip
exposed + fixed a partial-view gap (`view_indices` routes to single-device on a trivial mesh,
raises only on a multi-device mesh).  No-mesh-ParallelBeam tests retired/updated.  Remaining:
GPU re-validation (Change 1 memory + Change 2 single-GPU no-regression), note 1 (baseline
tolerance → `allclose` 1e-4, now that MESH=False auto-meshes on beta ParallelBeam), then the
2-Drop decision.  See `sharding_status.md` HANDOFF (2026-06-08).

**Unification decision (the "step 4" trivial-placement question) — RESOLVED: Option B,
one always-on placement path.**  A trivial 1-device placement resolves to a 1-device
`NamedSharding` (not a `SingleDeviceSharding`), so the single-device case is just the
degenerate sharded case and the dual `is_sharded` branches collapse into one path.
Rationale: "no literal regression from prerelease" is the same over-strict invariant as
"bit-exact" — we accept ~1 ULP (iteratively amplified to ≤1e-4) differences, not byte
identity.  The per-subset `block_until_ready` + `.delete()` cleanup then runs single-
device too, but it only stalls *host dispatch*, which is in the noise on production-size
recons where projection compute dominates; and making the transient cleanup mandatory
everywhere means one memory strategy, with no single-device-only leak able to slip past.
Carry these into the implementation:
  1. **Swap the guard, don't drop it.**  Losing the "verbatim prerelease body" safety net
     means single-device correctness rides on the donation + eager-scale path, so replace
     the bit-exact-vs-prerelease check with a **tight `allclose`** single-device-vs-
     prerelease guard (repurpose `vcd_single_device_baseline.py` to compare at ~1e-4, not
     exact).
  2. **Unlocked simplification.**  The "keep alpha out of the donated jit to avoid an FMA"
     decision existed *only* to preserve trivial-mesh bit-exactness.  Once ~1 ULP is
     accepted, fold `error_sinogram ← error_sinogram − alpha·delta` into a **single
     donated FMA jit**, deleting the separate eager `scaled_delta` transient *and* its
     `.delete()` — the unified update gets simpler than either current path.
  3. **Heterogeneous recon-CPU / sino-GPU is the one behavior change.**  Option B routes
     the prior through `_qggmrf_prior_sharded` (two trivial placements) instead of the
     `qggmrf_..._transfer` fast path, whose *timing* is an existing deferred open question
     — measure before deleting `_transfer`.  **And: if this heterogeneous case becomes
     unwieldy** (it forces multiple code paths or other special cases) **we drop it.**  It
     is only a stop-gap for doing larger recons on a *single* GPU; the real target is large
     recons on *multiple* GPUs, and that path must not be complicated to keep the stop-gap
     alive.

**Retirement marker convention.**  Code, tests, or docs that exist *only* while the
legacy single-device path coexists with the sharded/placement path — and should be
removed once everything runs on placements — are tagged with the fixed, greppable
phrase **`RETIRE-AFTER-SHARDING`** (general stem `RETIRE-AFTER` for any milestone-
gated retirement).  Before doing the P6 retirement, `grep -rn "RETIRE-AFTER-SHARDING"`
to find every site.  Currently tagged:
  - the 7 `tests/sharding/` *trivial-mesh-vs-single-device* comparison tests
    (`test_trivial_*bit_exact*` in back/forward projection, fbp_recon, and VCD).
    These were exact-equality checks; **relaxed to a tight `allclose`** (single-shot
    `rtol/atol=1e-5`; iterative VCD recon `1e-4`, matching the GPU-proven multi-device
    sweep sibling) because the banded sharded path reorders non-associative FP sums,
    so on **GPU** the 1-device mesh differs from the monolithic single-device kernel
    by ~1 ULP (CPU compiles both identically and stays exact).  Once the legacy path
    is gone there is a single path and nothing to compare, so they retire.
  - the `is_sharded` property and its branch sites (already noted at its definition).

---

## Decisions & open questions

- **O1 (heterogeneous placement) — reframed/resolved.**  recon-CPU / sino-GPU via
  two *separate* trivial placements; the reverse (sino-CPU / recon-GPU) is **out
  of scope**.
- **O2 (return contract) — resolved (F2):** user-facing methods match input;
  internal methods are sharded-only.
- **O4 (divisibility):** warn-with-instructions + `prepare_sino_for_devices` +
  pick-N; pad/crop only on the clean slice/row axis.
- **Band-vs-pixel criterion (P2) — RESOLVED (2026-06-03): KEEP BAND.**  The
  side-by-side showed pixel's peak memory floored at ~1.7–2.7× band for only
  ~11–16% less time, so band stayed and the pixel path was removed.
- **Open:** exact `B_p` default / floor (set conservatively, refine by sweep);
  `configure_devices` precedence vs `use_gpu`; how `prepare_sino_for_devices`
  returns the crop spec to undo padding on the recon.

---

## What survives unchanged
- F2 **match-input** contract on `direct_recon` / `fbp_recon` / `back_project`.
- The `direct_recon` scaling driver + baseline (re-run against new internals).
- F1 FBP filter (`apply_row_filter`) — already view-sharded, no cylinder motion.
- Cross-cutting principles, hardware facts, and the completed-work record in
  `sharding_implementation_plan.md`.
