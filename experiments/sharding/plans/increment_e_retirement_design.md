# Increment E — the retirement cascade: design note (DRAFT, 2026-06-20)

> **STATUS: INCREMENT E COMPLETE (E1, E2, E4, E3a, E3b, E3-cleanup, configure_sharding hard-delete, E3c, E3f, P3, E5 — all DONE, CPU-validated, committed).  Placements are the single source of device layout; main_device/sinogram_device/is_sharded/configure_sharding/the no-mesh path are gone.  NEXT = prerelease PR (+ follow-ups: pixel_indices_worker collapse, mar/preprocessing sharding; the mesh/shard_devices derived-property revisit).**  Written after a read-only
> survey of the current code (not the docs).  All geometries are now ported (ParallelBeam + cone +
> translation + multiaxis, all `_supports_sharding()=True`), the precondition E waited for.  The four
> design decisions (§3, D1–D4) are settled with Greg; staged plan in §4.  **Start condition:** begin E1
> once tomorrow's nightly confirms the multiaxis/translation GPU baselines (so E's no-behavior-change
> claim has a green GPU reference to regress against).  Line numbers here are a 2026-06-20 snapshot;
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

## 3. Design decisions — SETTLED with Greg (2026-06-20)

- **D1 — `view_indices`: drop the user-facing feature; keep + RENAME the internal mechanism.  RESOLVED.**
  Drop the user-facing view-subset kwarg from the *model* `sparse_forward_project`/`sparse_back_project`
  (+ the `None`-vs-subset dispatch, the multi-device `NotImplementedError` guard, and the single-device
  per-batch view loop).  vcls already supersedes it (1-view sibling + `set_view_parameters`, never
  `view_indices`).  Keep the **projector-level** param (Option A — the wrapper already has it, so zero new
  plumbing; Option B's pre-sliced-`view_params_array` override was the alternative, rejected as
  equal-code/higher-risk), and **RENAME it `owned_view_indices`** everywhere it survives (projector
  wrappers + jit drivers + gather; the sharded-driver `view_ranges[owner]` calls; ParallelBeam's
  `_back_project_view_shard_to_band` override) — it names the views a shard OWNS, not a user subset.  Do
  NOT rename the unrelated locals also called `view_indices` (`preprocess/zeiss_cb.py`, the zeiss/repro
  experiments — a different concept).  *Survey of public callers (2026-06-20):* the only GATED callers are
  three tests (retired in E1, §4); NO demo uses it; the un-gated experiments
  (`sandboxes/cone_beam_dev.py` — already stale vs the current API, `bugs_and_artifacts/...`, an archived
  ablation) are **left alone by decision** (housecleaning deferred, out of scope for E).
- **D2 — delete the no-mesh path entirely.  RESOLVED (option a).**  Remove the `is_sharded==False`
  branches and flip/drop the `_supports_sharding()` gate so there is ONE code path (the gate's purpose was
  the porting transition, now complete; a future unported geometry would re-introduce the seam
  deliberately).  Keep the single-device back driver/kernel + the forward `n_dev==1` one-shot (§2).
- **D3 — retire `output_device` entirely.  RESOLVED.**  Remove it from the public surface
  (`sparse_*_project`, `compute_hessian_diagonal`); internal callers use `recon_placement.devices[0]` /
  `sino_placement.devices[0]`; `output_sharded` is the sole output-form control.
- **D4 — one commit; no PR yet.  RESOLVED.**  Land E as a single staged commit (Greg commits from
  PyCharm); a PR waits until the sharding work has settled more fully.

---

## 4. Staged plan (E1–E5), each landing green for review

Hard gate throughout: **the full suite stays green at the default 4 CPU devices**, and the sharded-vs-…
correctness identities (adjoint, Hessian, VCD convergence, padding inertness) are unchanged — E is a
no-behavior-change cleanup for the supported path.  Order chosen so each stage shrinks the surface the
next one touches.

| Stage | Work | Key symbols / files | Gate |
|---|---|---|---|
| **E1 — ✅ DONE (2026-06-20)** — retire user-facing `view_indices`; rename the internal arg to `owned_view_indices` | Remove the kwarg from the *model* `sparse_forward_project`/`sparse_back_project` + docstrings + the `None`-vs-subset dispatch + the multi-device `NotImplementedError` guards + the single-device per-batch view loop (every call → the sharded path; the trivial mesh handles n=1).  **Option A:** the projector-level arg STAYS (the wrappers already have it) — **rename `view_indices` → `owned_view_indices`** in `projectors.py` (wrappers + jit drivers + the `view_params_array[...]` gather) and at the sharded-driver call sites (`view_ranges[owner]`) + ParallelBeam's `_back_project_view_shard_to_band` override; it names the views a shard OWNS.  Leave the unrelated `view_indices` locals (`zeiss_cb.py`, experiments) untouched.  Retire `verify_view_batching` + the two `test_view_indices_not_supported` tests + the `test_projectors` `configure_devices(1)` pin.  (No demo uses it; un-gated experiments left alone.) | `tomography_model.py` (sparse_*_project, the two RETIRE-AFTER notes, `_forward_project_to_view_shards`, `_back_project_view_shard_to_band`); `projectors.py` (`*_public`, `_jit_sparse_*`, `_sparse_*_project`); `parallel_beam.py`; `tests/geometries/test_projectors.py`, `tests/sharding/test_{forward,back}_projection.py` | full suite green; adjoint/Hessian/VCD unchanged; sharded n≥2 still selects per-owner views; vcls still runs |
| **E2 — ✅ DONE (2026-06-20)** — delete the dead no-mesh path | Dropped the `_supports_sharding` gate entirely (base method + its 3 uses: the `configure_sharding` reject block, `_auto_device_count`, the `set_devices` gate; + the 4 geometry overrides), so `set_devices` always auto-meshes and `is_sharded` is provably always True.  Removed the dead no-mesh `else` in `_set_placements`; made both `sparse_*_project` dispatches unconditionally sharded; deleted the orphaned `_sparse_forward_project_single_device`.  **Kept** `_sparse_back_project_single_device` + `back_project_one_view_to_pixel_batch` (GPU n=1 short-circuit) + the forward `n_dev==1` one-shot.  Retired `TestNoMeshNoOp`.  **DEFERRED to E3:** the broad `if is_sharded:` collapse — the remaining checks are vacuously True, and the ones gated against `device_put(main_device)` else-branches retire naturally when `main_device` does (E3).  (No "≥2 physical devices" `is_sharded` site remained after E1 — the device-count distinction is already via `len(...devices)` in the short-circuit — so the deferral carries no footgun.) | `tomography_model.py`, `cone_beam.py`, `parallel_beam.py`, `translation_model.py`, `multiaxis_parallel.py`, `tests/sharding/test_hooks.py` | full suite green (190p/2s); trivial-1-device path is now the only single-device path |
| **E4 — ✅ DONE (2026-06-20, ran BEFORE E3)** — retire `output_device` from the public projector surface | Removed `output_device` from `sparse_forward_project`, `sparse_back_project`, `compute_hessian_diagonal` (+ docstrings).  Collapsed the dead single-device fallthroughs in `forward_project`/`back_project` (they were the `output_device` callers).  Dropped the arg from the `compute_hessian_diagonal` + vcd `sparse_*_project` callers.  **Kept** `_sparse_back_project_single_device`'s `output_device` (the GPU n=1 short-circuit's placement mechanism) + the vcd-internal/`gen_set_of_pixel_partitions` params (E3).  *(Reordered before E3 because many `main_device` sites were `output_device=self.main_device` — removing `output_device` first eliminates them and avoids E3→E4 churn.)* | `tomography_model.py` (3 signatures + `forward_project`/`back_project` + the callers) | full suite green |
| **E3a — ✅ DONE (2026-06-22)** — device-config restructure + `mesh`/`shard_devices` retired to derived properties | **(a) Merged** `_apply_mesh` + `_set_placements` → one `_set_device_layout(devices, pinned)` (sets `dev2dev_safe` + `_sharding_configured`, invalidates the qGGMRF cache, builds both placements; no separate `self.mesh` field).  **(b)** `set_devices` is now Greg's 2-liner: `cur = recon_placement.devices if _sharding_configured else _auto_device_pool(); _set_device_layout(cur, pinned=_sharding_configured)`.  **(c)** Retired the `mesh`/`shard_devices` FIELDS → **derived read-only properties** off `recon_placement` (guarded: `None` before construction); the `NamedSharding(self.mesh,…)` sites + `is_sharded` read the properties unchanged.  ⚠ **Revisit (post-dust-settle, Greg):** properties are a transitional minimal-churn choice — later decide delete-and-rewrite-call-sites vs. keep-as-public-predicates.  **(e) Config-flow:** added `_auto_device_pool()` (one definition of "automatic", shared by `set_devices` + `configure_devices(None)`) — **Q1** robust GPU detect via `gpu_devices()` (not `.platform=='gpu'`), **Q2** CPU auto-shards consistently (dropped `if on_gpu else 1`); `configure_devices` is the sole real implementation (rewrote its docstring — **Q3**); **`configure_sharding` demoted to a thin back-compat alias** → `configure_devices` (the one no-arg `test_primitives` call switched to `configure_devices(1)`).  *(Alias hard-deleted in the E3-cleanup-2 row below.)*  KEPT `main_device`/`sinogram_device` + the `if is_sharded` checks → E3b. | `tomography_model.py` (device-config core); `tests/sharding/test_primitives.py` | full suite green (190p/2s) |
| **E3b — ✅ DONE (2026-06-22)** — retire `main_device`/`sinogram_device` | **(d)** Substituted the ~16 internal + ~6 external (`denoising`, `vcd_utils`, `mar`, `utilities`, `test_vcd`) array-creation/device-commit uses → `recon_placement.devices[0]` / `sino_placement.devices[0]` (this also FIXES the documented staleness: the scalar stayed a GPU after pinning CPU devices, the accessor tracks the layout).  Collapsed the 7 `main_device`-referencing branches (`_recon_devices` else, `to_sino`/`to_recon` ternaries, the qGGMRF-prior else, the `output_device` ternary → `None`, the 2 positivity `if not is_sharded` device-puts, the `recon_indices` else) since `is_sharded` ≡ True.  Removed the 2 fields + the `set_devices` assignment (+ the now-unused `on_gpu`/`cpus` locals) + reworded the stale `main_device` comments.  Updated `test_hooks` (placement-vs-scalar assertions → same-single-device).  (Local `main_device = cpu_devices()[0]` in `vcd_utils` streaming is NOT the model attr — left.) | `tomography_model.py`; `denoising.py`, `vcd_utils.py`, `preprocess/mar.py`, `utilities.py`; `tests/sharding/test_hooks.py`, `tests/geometries/test_vcd.py` | full suite green (190p/2s) |
| **E3-cleanup — ✅ DONE (2026-06-22)** — review-feedback follow-ups | Property docstrings note attribute-access; `is_sharded`/`mesh`/`shard_devices` marked RETIREMENT CANDIDATE (pending the revisit).  Deleted the dead `use_gpu == 'projections'` branch (already rejected by `verify_valid_params`).  Removed `output_device` from `get_forward_lin_quad` (always `None` → the two `device_put`s were no-ops).  **P1 + sweep:** added `sharded_full(placement, base_shape, fill_value, row_pad, dtype)` — a **static free function in `_sharding/placement.py`** (exported as `mjs.sharded_full`, since it depends only on a `Placement` and is reusable) that builds a `fill_value`-over-real / zero-pad array per-shard ON each device (the device-form analogue of `jnp.full`), no full single-device copy; used it for the default Hessian weights (fill 1), the constant init recon (fill = the int), and **unified** the duplicate `_sino_ones_device_form` (now a thin wrapper over it).  **P2:** `direct_recon`/`direct_filter` stubs honor `output_sharded` via the SAME helper (fill 0 for the sharded branch — the zeros analogue of the ones case).  **P4:** retired `_recon_devices` (≡ `shard_devices`).  **P5:** deleted the vestigial `test_vcd` sino device-puts (+ now-unused `import jax`).  **P6:** `'full'` is a deprecated synonym of `'automatic'` (stopped using it in `denoising.py`; updated docs/validation).  **P3 deferred** → task #20 (the `initialize_recon` partition/`_committed_elsewhere` single-device puts). | `tomography_model.py`, `denoising.py`, `_utils.py`; `tests/sharding/test_hooks.py`, `tests/geometries/test_vcd.py` | full suite green |
| **E3-cleanup-2 — ✅ DONE (2026-06-22, own commit)** — hard-delete `configure_sharding` | `configure_devices` is the sole device-config entry point.  Renamed the ~75 `configure_sharding(...)` callers across 11 `tests/sharding/` files → `configure_devices(...)` (all passed a device list, so identical; the one no-arg call was already `configure_devices(1)`), renamed the `TestConfigureSharding` class → `TestConfigureDevices`, deleted the alias method, and updated the mbirjax/ docstrings/comments + the OOM-guidance message (`_utils.py`).  Also renamed the ≈13 ungated `experiments/` callers (incl. the `hasattr(model, "configure_sharding")` prerelease-detection guard in `vcd_single_device_baseline.py` → `"configure_devices"`, which is likewise sharding-branch-only so the detection is preserved); all compile-checked.  `configure_sharding` now appears in no `.py` file. | `tomography_model.py`, `_sharding/__init__.py`, `_utils.py`; 11 `tests/sharding/*.py`; ≈13 `experiments/` files | full suite green; experiments compile |
| **E3c — ✅ DONE (2026-06-22)** — shard `denoising.py` | Dual-path (approved): **n_dev == 1** keeps the existing whole-sweep JIT (fast, no halos -- unchanged for every current use); **n_dev > 1** slice-shards `flat_image`/`flat_error_image` (`_shard_recon`) and runs a Python loop staging the qGGMRF halos once per pass (`_stage_halos`; `extract_halos` is host-side so it can't live in a JIT) with an eager per-subset updater using `_qggmrf_prior_sharded` + replicated `recon_indices` + buffer-donated in-place image/residual updates.  Identity forward model → only the recon mesh, so `alpha` needs no cross-mesh reconciliation (simpler than `vcd_recon`).  Factored the two paths into `_denoise_single_device` / `_denoise_sharded`.  The once-per-pass halo path differs early but **converges** to the single-device MAP estimate (measured n=2/4 vs n=1 max-rel-diff: ~9e-3 @6 iters → ~9e-4 @15 → ~3e-5 @30).  Gate test `tests/sharding/test_denoise_sharded.py` (sharded==single @20 iters, tol 3e-3, n=2/4).  Also extracted the shared run-logging header (`setup_logger` + version + `_device_report`) from `initialize_recon` into `ParameterHandler._log_run_header` (it lives with the logger it configures in `parameter_handler.py`; version/`_device_report` come from the TomographyModel subclass), now called by both recon and denoise (so the denoiser logs its sharded/device layout too); left the partition-gen / `_committed_elsewhere` / sino-validation overlap alone (genuinely different per path). | `denoising.py`, `tomography_model.py`, `parameter_handler.py`; `tests/sharding/test_denoise_sharded.py` | full suite green; sharded converges to single-device |
| **E3f — ✅ DONE (2026-06-22)** — collapse the vacuous `if is_sharded` branches | `is_sharded` ≡ True, so all internal conditional uses were dead/vacuous and were collapsed (~25 sites): the `if not self.is_sharded: return x` early-outs (`_shard_on_axis`, `_gather_to_host`, `_extract_halos`, `_stage_halos`), the `self.is_sharded and X` → `X` padding/halo guards (`_shard_sinogram`/`_gather_sinogram`/`_shard_recon`/`_gather_recon`, `_device_report`, `_qggmrf_interface_masks`, `pad_active`, `stage_per_pass`, `parallel_beam._sino_row_padding`), the `if self.is_sharded:` wrappers in `_sino_device_shape`/`_recon_device_shape` and the VCD forward-scalar replicate, the `_device_report` `' (sharded)'` suffix (always present now), and the dead single-device `else` branches in `_apply_direct_recon_filter` + the VCD error-sinogram update.  Then **deleted the `is_sharded` property entirely** (no internal uses left): removed the 4 `assertTrue(model.is_sharded)` test assertions (`test_primitives`, `test_hooks`) and replaced the `getattr(model, "is_sharded", False)` condition in the archived `cone_baseline_scaling.py` with `True`.  `is_sharded` now appears in no code (only plan docs + a couple of archived comments; `"is_sharded"` survives as an experiment OUTPUT-YAML field name).  Pure dead-code removal (verified no-op: recon/fbp/denoise + padded-shard round-trips unchanged; `hasattr(model,'is_sharded')` is now False). | `tomography_model.py` (~25 sites), `parallel_beam.py`; `tests/sharding/test_primitives.py`, `test_hooks.py`; `experiments/.../cone_baseline_scaling.py` | full suite green; no behavior change |
| **P3 — ✅ DONE (2026-06-22)** — `initialize_recon` single-device device-puts | Traced each value to where it's consumed and dropped two redundant single-device commits: (a) the `partitions = [jax.device_put(p, recon_placement.devices[0]) …]` list comp (gen_set_of_pixel_partitions already places on `output_device`); (b) the `_committed_elsewhere` guards for sinogram/weights/init_recon (vcd_recon's `to_sino`/`to_recon` → `_shard_on_axis` → `move_shard` place from any source device and never gather a prepared NamedSharding — so the guard only added a redundant hop).  Verified a no-op: a recon with a pre-committed-elsewhere sinogram is bit-identical to normal (n=1, n=2).  Also fixed a latent `get_2d_ror_mask` bug (the `use_ror_mask is False` branch returned `np.ones_like(recon_shape[:2])` → shape `(2,)`; now `np.ones(recon_shape[:2], dtype=bool)` — unreachable today but correct).  **Flagged (not done):** `pixel_indices_worker`/`partition_worker` is now an identical copy of `pixel_indices` (sino/recon share devices) — collapsing it touches the projector-call interface, a separate follow-up; and `gen_weights_mar` builds full sino/recon arrays on one device (a sharding opportunity, but MAR/preprocessing → task #18). | `tomography_model.py`, `vcd_utils.py` | full suite green; pre-committed input bit-identical |
| **E5 — ✅ DONE (2026-06-22)** — the RETIRE-marker sweep + doc close-out | Removed the **6 tautological** `test_trivial_*_bit_exact` tests (placement-recon-vs-placement-recon @1-device; `test_{back,forward}_projection.py`, `test_fbp_recon.py`, `test_vcd_sharded.py` ×3 recon).  **Kept + reworded** the 7th (`test_vcd_sharded::test_trivial_bit_exact`) — it cross-checks the sharded prior orchestrator @1-device vs the still-live standalone `qggmrf_gradient_and_hessian_at_indices` (the only n=1 orchestrator coverage), so NOT a dead-legacy comparison.  Reworded the stale comments that referenced retired machinery: the no-mesh refs (`_shard_on_axis`/`_gather_to_host` docstrings, the two "legacy single-device" comments, `parallel_beam`/`cone_beam` "no mesh configured"), and the translation/multiaxis "runs single-device until the placement port" `output_sharded` claims (they ARE sharded → "returns the view-sharded device form").  Doc close-out: `v3` §5 item 4 + status table row, `sharding_status.md` top handoff. | `tests/sharding/test_{back,forward}_projection,fbp_recon,vcd_sharded}.py`; `tomography_model.py`, `parallel_beam.py`, `cone_beam.py`, `translation_model.py`, `multiaxis_parallel.py`; the plan docs | full suite green (185p/2s); no stale RETIRE markers |
| **pixel_indices_worker collapse — ✅ DONE (2026-06-23)** — the P3 follow-up | `pixel_indices_worker` was "`pixel_indices` copied onto the worker device" — a hybrid-mode (CPU-recon/GPU-sino) artifact.  Since `sino_placement.devices[0]` == `recon_placement.devices[0]`, it was an identical copy.  Dropped the `partition_worker = device_put(partition, sino_placement.devices[0])` + `subset_worker` in `vcd_partition_iterator`, the `pixel_indices_worker` param of `vcd_subset_updater`, and routed the two projector calls (`sparse_back_project`/`sparse_forward_project`) through `pixel_indices`.  Pure no-op (the sharded projectors re-place the indices per view-owner anyway). | `tomography_model.py` | full suite green |

De-scoped / already done: the `'sinograms'`/`_transfer` hybrid mode (gone); the projector-batching-machinery
refactor (separate tracked item, v3 §6); B4.5 (band-kernel GPU cost, separate).

**Reviewed — no change needed:** `ConeBeamModel.split_sino_recon` (2026-06-22).  It reconstructs two
overlapping detector-row halves via fresh `copy_ct_model` sub-models and `.recon()` on each, then
`device_get` + host-side `stitch_arrays`.  Each sub-model runs `__init__`→`set_devices`, so each
half-recon auto-shards like any recon; verified n=1 and n=2 give identical split output (sharding is
transparent).  Nuance (pre-existing, not a bug): the sub-models inherit `use_gpu` (a param) but not an
explicit `configure_devices(n)` pin — so a pinned parent runs the halves on auto-selected devices.
Possible small enhancement (propagate the parent's device config to the sub-models); not required.

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
