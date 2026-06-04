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

### 0c. Streaming = uniform pixel-batching

- The streaming unit is a fixed-size batch of **full cylinders** (a pixel batch),
  uniformly for both directions and both modes — the existing
  `transfer_pixel_batch_size` pattern, generalized.  `B_p` is the memory knob
  (bounds `B_p × num_slices` per device).
- The layer-2 driver is a single uniform pixel-batch loop with the geometry
  kernel injected; no mode-specific band axis.
- This **replaces Phase D's slice-banding** (pending the side-by-side
  measurement in P2).  Big-sino host-streaming being out of scope removes the
  only reason to keep slice-banding.

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

### P2 — Back projection on placements (re-opened Phase D)  ⏳ measuring
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

### P3 — Forward projection on placements (Phase C)  ← NEXT
- Pixel-batched `move_cylinders_to_sino` forward projection on ParallelBeam.
- **match-input** per the F2 contract.
- Correctness gate: forward/back **adjoint round-trip** against the validated P2
  back projector.
- **Design in P3: the geometry-neutral "project to a slice band" interface**
  (see the design note below).  P3 is where it should land, because it must be
  built for forward + back together so the adjoint round-trip gates it.

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

### P4 — VCD on placements (Phase E)
- Entry/exit placements for sinogram (view) / recon (slice) / weights (view).
- Halo exchange via `_extract_halos` in the VCD loop.
- Default init: `vcd_recon` computes `init_recon = direct_recon(sinogram)` — with
  match-input, pass it the already-sharded sinogram so the init stays sharded and
  flows into the loop with no round-trip.
- No accidental gather/re-shard inside the loop body.

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
- Port the placement/movement path to cone / translation / multiaxis geometries
  (each implements the `(g0, L)` slice-band projector interface from the P3 design
  note).
- Retire `main_device` / `sinogram_device` entirely (the trivial-sharding
  unification); remove the old single-device and slice-band code paths.
- With the single-device path unified onto band-looping, **delete the per-geometry
  `entries_per_cylinder_batch` slice-batching** (subsumed by the outer band loop)
  and switch the jittable single-device / forward assembly to an in-place
  `dynamic_update_slice` accumulator with `donate_argnums` (see the P3 design note).

---

## Decisions & open questions

- **O1 (heterogeneous placement) — reframed/resolved.**  recon-CPU / sino-GPU via
  two *separate* trivial placements; the reverse (sino-CPU / recon-GPU) is **out
  of scope**.
- **O2 (return contract) — resolved (F2):** user-facing methods match input;
  internal methods are sharded-only.
- **O4 (divisibility):** warn-with-instructions + `prepare_sino_for_devices` +
  pick-N; pad/crop only on the clean slice/row axis.
- **Band-vs-pixel criterion (P2):** adopt pixel-batching unless the side-by-side
  shows a clear memory/time regression on GPU.
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
