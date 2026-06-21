# Increment E — the retirement cascade: design note (DRAFT, 2026-06-20)

> **STATUS: DRAFT PLAN — not started.**  Written after a read-only survey of the current code (not the
> docs).  All geometries are now ported (ParallelBeam + cone + translation + multiaxis, all
> `_supports_sharding()=True`), which is the precondition E waited for.  This doc stages the cleanup and
> flags the design decisions that need Greg before coding.  Line numbers here are a 2026-06-20 snapshot;
> **trust the symbol name over the number.**  Sibling docs: `sharding_implementation_plan_v3.md` §5 item 4
> (the one-paragraph E sketch this expands), `increment_d_translation_design.md` (the staging template).

---

## 0. Verified architecture facts (the ground truth E rests on)

These were checked against the code, because the planning notes (and one survey pass) had a stale claim.

- **`is_sharded` is now GLOBALLY True.**  `is_sharded` ≡ `self.mesh is not None`.  `set_devices()` (the tail
  of every `__init__`) does `if not self._sharding_configured and self._supports_sharding(): _apply_mesh(...)`
  — and **all four concrete geometries override `_supports_sharding()` to return True** (ParallelBeam,
  cone, translation, multiaxis; the base default is False but is never the live value).  So every model
  gets a mesh — a trivial 1-device mesh when single — and `is_sharded` is True in every real instance.
  *(Correction to a survey pass that assumed cone is `_supports_sharding()=False`: it is True — see its
  docstring "the single-device case auto-defaults to a trivial 1-device mesh".)*
- **⇒ The no-mesh / `is_sharded==False` path is DEAD code** (reachable only by a hypothetical future
  geometry that does NOT override `_supports_sharding`).  This is the single biggest simplification in E:
  ~40 `is_sharded` sites in `tomography_model.py` are "do I have a placement?" tests whose `else` branch is
  now unreachable.
- **"Single-device" means two DIFFERENT things — keep them straight:**
  1. **The no-mesh DISPATCH** (`sparse_forward_project`/`sparse_back_project` routing to
     `_sparse_*_single_device` *when not `is_sharded`*) → **DEAD, retire.**
  2. **The single-device back DRIVER + KERNEL** (`_sparse_back_project_single_device`,
     `back_project_one_view_to_pixel_batch`) and the **forward `n_dev==1` one-shot** (inside
     `_forward_project_to_view_shards`) → **LIVE, KEEP.**  They are called *from inside the sharded path*
     by the **GPU n=1 back short-circuit** (`_sparse_back_project_sharded`: a single-GPU mesh routes to the
     pixel kernel — ~2.25× faster than the band kernel on GPU — and wraps the output as a 1-shard
     slice-sharded array) and the forward `n_dev==1` monolithic call.  Retiring (1) must NOT remove (2);
     `_sparse_back_project_single_device` simply stops being a dispatch target and becomes a helper the
     short-circuit calls.
- **`view_indices` is load-bearing for the SHARDED path, not just a legacy user feature.**  The sharded
  drivers pass `view_indices=view_ranges[view_owner]` so each view-owner's projector slices
  `view_params_array[view_indices]` and projects only *its* views against its local sinogram shard
  (`_forward_project_to_view_shards`, `_back_project_view_shard_to_band`; the slice happens in
  `projectors.py::_sparse_*_project`).  So retiring `view_indices` = remove the **user-facing view-subset
  kwarg + the multi-device guard + the single-device view-batching loop**, while **preserving per-owner
  view selection** by slicing `view_params_array` directly in the sharded driver.  (This is what the
  RETIRE note means by "the batching slices the traced `view_params_array` directly.")
- **`main_device`/`sinogram_device` are still partly live:** besides the dead single-device branches, they
  are the **default-device selector** — `set_devices` builds even the trivial 1-device mesh as
  `_apply_mesh([self.main_device], ...)` — and they seed `_set_placements`.  So they MORPH (a local device
  pick + placement accessors), not a pure delete.
- **The hybrid `'sinograms'` / `_transfer` mode is ALREADY gone** (0 refs; a comment in `set_devices`
  records the removal).  Not an E task.

---

## 1. Goal and non-goals

**Goal.**  Now that every geometry runs the always-on placement path, remove the legacy single-device
*dispatch* and its scaffolding so the code has ONE path, while keeping the platform-optimal single-device
back kernel the GPU n=1 short-circuit needs.  Net: delete dead branches, retire `view_indices` as a
user-facing feature, replace the `main_device`/`sinogram_device` scalars with placement accessors,
reconcile `output_device` with `output_sharded`, and retire the now-moot trivial-mesh-vs-legacy tests.

**Non-goals:** no behavior change for the supported (sharded/placement) path — every projector/recon
result stays bit-for-bit (or within the established float tol) what it is today; no new features; the
projector kernels and the band/pixel platform split are untouched (that is B4.5, separate).

---

## 2. The must-KEEP list (load-bearing — do NOT retire)

- `_sparse_back_project_single_device` + `back_project_one_view_to_pixel_batch` — called by the GPU n=1
  back short-circuit in `_sparse_back_project_sharded`.
- The forward `n_dev==1` one-shot inside `_forward_project_to_view_shards` (projects the full cylinder in
  one `sparse_forward_project` call — the memory-optimal single-GPU forward).
- `initialize_recon`'s early `device_put` guard (the `_committed_elsewhere` check) — prevents silent
  gathers of user-passed `NamedSharding` arrays.  MORPH its device refs to placement accessors; keep the
  guard.
- The per-owner view selection mechanism (today `view_indices=view_ranges[owner]`) — re-home, don't delete.
- The padding machinery, the `(g0,L)` band interface, the qGGMRF halo path — all geometry-neutral, all stay.

---

## 3. Design decisions to settle with Greg (before coding)

- **D1 — `view_indices` user-facing feature: remove entirely?**  vcls (the single-view sibling +
  `set_view_parameters`) already supersedes it and does NOT use `view_indices`.  Proposal: drop the
  user-facing view-subset kwarg from `sparse_forward_project`/`sparse_back_project` (+ the multi-device
  `NotImplementedError` guard + the single-device view-batching loop), and keep ONLY the internal
  per-owner slicing (renamed so it reads as an internal shard detail, e.g. the sharded driver passes a
  pre-sliced `view_params_array`).  *Confirm no supported user workflow still needs arbitrary view
  subsets outside vcls.*
- **D2 — the `_supports_sharding()` gate + the no-mesh path: delete or keep as a guard?**  Two options:
  (a) **delete the no-mesh path entirely** and flip the base default to True (or drop the gate) — cleanest,
  one code path, but removes the fallback for a future not-yet-ported geometry; (b) **keep the gate**
  (base default False) as the seam for a future geometry, and only delete the *currently-dead branches* it
  guards — safer/more conservative but leaves a dormant single-device path.  Proposal: **(a)** — the gate's
  whole purpose was the porting transition, which is done; a future geometry would re-introduce the seam
  deliberately.  Needs Greg's call (it is the most load-bearing decision).
- **D3 — `output_device` (on `sparse_*_project` + `compute_hessian_diagonal`): retire or keep?**  It is a
  single-device-only output-placement knob; `output_sharded` is the unified control.  Proposal: retire
  `output_device` from the public surface, route internal callers to `recon_placement.devices[0]` /
  `sino_placement.devices[0]`, and let `output_sharded` decide device-form-vs-gathered.  Confirm no caller
  needs to pin the gathered output to an arbitrary device.
- **D4 — staging granularity / commit boundaries** (see §4): one PR with staged commits, or smaller PRs?

---

## 4. Staged plan (E1–E5), each landing green for review

Hard gate throughout: **the full suite stays green at the default 4 CPU devices**, and the sharded-vs-…
correctness identities (adjoint, Hessian, VCD convergence, padding inertness) are unchanged — E is a
no-behavior-change cleanup for the supported path.  Order chosen so each stage shrinks the surface the
next one touches.

| Stage | Work | Key symbols / files | Gate |
|---|---|---|---|
| **E1 — retire `view_indices` (user-facing)** | Remove the kwarg from `sparse_forward_project` / `sparse_back_project` + their docstrings + the multi-device `NotImplementedError` guards; delete the single-device view-batching loop; **re-home per-owner selection** so the sharded driver slices `view_params_array` directly (no public kwarg).  Drop the projector-driver `view_indices` param in `projectors.py` once callers are converted.  Retire the `verify_view_batching` test + the `test_view_indices_not_supported` tests + the `test_projectors` `configure_devices(1)` pin. | `tomography_model.py` (sparse_*_project, the two RETIRE-AFTER notes, `_sharded_*_setup`, `_forward_project_to_view_shards`, `_back_project_view_shard_to_band`); `projectors.py` (`_jit_sparse_*`, `_sparse_*_project`); `tests/geometries/test_projectors.py`, `tests/sharding/test_{forward,back}_projection.py` | full suite green; adjoint/Hessian/VCD unchanged; vcls still runs |
| **E2 — delete the dead no-mesh path** | Remove the `is_sharded==False` `else` branches (now unreachable) across `tomography_model.py`: the no-mesh dispatch in `sparse_*_project`, the no-mesh placements branch in `_set_placements`, the `_extract_halos`/`_stage_halos`/`_shard_*`/`_gather_*` no-op early-returns, etc.  **Keep** `_sparse_back_project_single_device` + the pixel kernel + the forward `n_dev==1` one-shot (re-homed as short-circuit helpers).  Resolve **D2** (gate flip vs keep).  Re-read each of the ~40 `is_sharded` sites for the footgun: a site that really meant "≥2 physical devices" becomes `len(shard_devices) > 1` (the 4 dispatch sites at the `sparse_*_project` heads). | `tomography_model.py` (is_sharded sites, `_set_placements`, `set_devices` gate); `tests/sharding/test_hooks.py` (the two skipped no-mesh tests delete) | full suite green; the trivial-1-device path still works (it is now the only single-device path) |
| **E3 — retire `main_device`/`sinogram_device`** | Replace the scalars with placement accessors (`recon_placement.devices[0]` / `sino_placement.devices[0]`) at all ~75 use sites (incl. `vcd_utils.py`, `denoising.py`, `utilities.py`, `preprocess/mar.py`); MORPH `set_devices` to pick the default device locally instead of storing the attrs; keep `initialize_recon`'s early `device_put` guard (device refs → placement accessors).  Update `test_hooks` placement-vs-scalar assertions. | `tomography_model.py` (`set_devices`, `_set_placements`, `initialize_recon`, `vcd_*`, `compute_hessian_diagonal`); `vcd_utils.py`, `denoising.py`, `utilities.py`, `preprocess/mar.py`; `tests/sharding/test_hooks.py`, `tests/geometries/test_vcd.py` | full suite green; device placement unchanged |
| **E4 — reconcile `output_device` with `output_sharded`** | Per **D3**: retire `output_device` from the public surface (`sparse_*_project`, `compute_hessian_diagonal`); internal callers use placement accessors; `output_sharded` is the sole output-form control. | `tomography_model.py` (the three signatures + callers); any test passing `output_device` | full suite green |
| **E5 — the RETIRE-marker sweep + doc close-out** | `grep -rn "RETIRE"` and clear each: retire the trivial-mesh-vs-legacy comparison tests (`test_{back,forward}_projection.py`, `test_fbp_recon.py`, `test_vcd_sharded.py` — they compared the placement path to a legacy path that no longer exists; keep/relax to the sharded-vs-single-shard identities that remain meaningful); update the comments that referenced the retiring machinery; update `v3` §5 item 4 + `sharding_status.md`. | the 7 files with RETIRE markers; the plan docs | full suite green; no stale RETIRE markers |

De-scoped / already done: the `'sinograms'`/`_transfer` hybrid mode (gone); the projector-batching-machinery
refactor (separate tracked item, v3 §6); B4.5 (band-kernel GPU cost, separate).

---

## 5. Risks / footguns

1. **`view_indices` is the sharded per-owner mechanism (E1's crux).**  Removing the kwarg without
   re-homing the per-owner `view_params_array[owner_range]` slice would break every multi-device
   projection.  Gate E1 on the sharded tests (`test_*_sharded`, `test_padding`) AND the geometry adjoint
   tests, not just single-device.
2. **The is_sharded↔n_devices decoupling footgun** (v3 §5): when deleting the no-mesh branches (E2), a
   handful of sites really asked "≥2 physical devices?" (the dispatch heads + the GPU n=1 short-circuit
   condition).  Mis-collapsing one of these to "always sharded" would route a single-GPU recon down the
   slow band path (a 2.25× regression, not a correctness bug — so tests stay green; catch it by reasoning,
   and ideally by the nightly GPU back-time baseline).
3. **The GPU n=1 short-circuit must keep calling the single-device kernel** (must-keep list).  The CPU
   path deliberately does NOT short-circuit (the band kernel is ~8× faster on CPU).  Don't "simplify" the
   short-circuit away because it looks like dead single-device code.
4. **`compute_hessian_diagonal`'s default weights** are built under `jax.default_device(main_device)`; E3
   must move this to the sino placement device or the default-weights ones land on the wrong device.
5. **No GPU here, but the back-time regression in (2) is GPU-only** — flag any short-circuit/dispatch
   change for a nightly GPU check (the back-time + memory baselines).

---

## 6. RETIRE-marker inventory (the breadcrumbs, 2026-06-20)

- `tomography_model.py` ~1321, ~1750 — the two `view_indices` RETIRE-AFTER-SHARDING notes (E1).
- `tests/sharding/test_back_projection.py`, `test_forward_projection.py`, `test_fbp_recon.py`,
  `test_vcd_sharded.py` (×4) — "trivial-mesh-vs-legacy comparison, meaningful only while both paths
  coexist" (E5); `test_vcd_sharded` also has two relaxed-tolerance markers.
- `tests/sharding/test_hooks.py` ~70/78/89 — two `@unittest.skip("RETIRE-AFTER-SHARDING")` no-mesh tests
  (E2 delete).
- `tests/geometries/test_projectors.py` ~162 — the `configure_devices(1)` view-batching pin (E1).

**Net E surface (current snapshot):** `view_indices` ~75 refs, `main_device`/`sinogram_device` ~75,
`is_sharded` ~45 (mostly dead-`else` after E2), `output_device` ~36; 13 RETIRE markers across 7 files.
