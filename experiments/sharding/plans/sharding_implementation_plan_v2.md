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

### P1 — Placement foundation
- Introduce `recon_placement` / `sino_placement` as `Sharding`s; trivial
  single-device placements; `move_shard` same-device no-op verified.
- Implement `move_cylinders_to_sino` / `sum_cylinders_to_recon` (cylinders-direct,
  `move_shard`-based, `N×N` / `1×1`).
- **Early low-risk migration:** replace raw `device_put` / `default_device` /
  `jnp.zeros(device=…)` with the placement/move helpers (behavior-preserving;
  no-op on same device) — validates the no-op property before the multi-device
  path depends on it, and starts the unification.
- Side-by-side scaffolding so the new path runs beside the existing code.
- Tests: trivial-placement bit-exact vs single-device; movement primitives
  unit-tested on a sharded cylinder array.

### P2 — Back projection on placements (re-opened Phase D)
- Pixel-batched `sum_cylinders_to_recon` back projection on ParallelBeam.
- **Side-by-side vs the slice-banded Phase D** on the existing GPU harness
  (memory peak / time / scaling at 256/512/1024³ on 1/2/4 GPUs).  Decision rule:
  if pixel-batching is within noise (expected — the Phase D ablation found the
  reduce-scatter ≤9% of time), **adopt it and retire the band code**; else keep
  band only if a clear regime needs it.
- Anchors: trivial bit-exact vs prerelease; n>1 float-noise match; the
  `sparse_back_project` correctness baseline.

### P3 — Forward projection on placements (Phase C)
- Pixel-batched `move_cylinders_to_sino` forward projection on ParallelBeam.
- **match-input** per the F2 contract.
- Correctness gate: forward/back **adjoint round-trip** against the validated P2
  back projector.

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
- Port the placement/movement path to cone / translation / multiaxis geometries.
- Retire `main_device` / `sinogram_device` entirely (the trivial-sharding
  unification); remove the old single-device and slice-band code paths.

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
