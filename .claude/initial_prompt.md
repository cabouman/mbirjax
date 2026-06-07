

This is the mbirjax CT reconstruction project — multi-GPU/CPU sharding work, beta
worktree `mbirjax_sharding` on branch `greg/parallel_sharding`.  The sibling
`../mbirjax/` is the OLDER research branch `greg/parallel_tests` — don't run code
from there (stale).

Orient first by reading, in order:
1. `.claude/claude_prompt.md` — collaboration style + workflow (investigate and
   propose before editing; wait for approval on non-trivial changes; minimal,
   localized changes; no plan-phase/step references in code/comments/docstrings;
   measure honestly — suspect the ruler before the code; stay curious and challenge
   my assumptions).
2. `experiments/sharding/plans/sharding_status.md` — read the **▶ CURRENT FOCUS**
   block at the top of the latest handoff first (that's the live next-step), then
   skim the rest of that handoff.
3. `experiments/sharding/plans/sharding_implementation_plan_v2.md` — §0 design
   summary + the placement phases P1–P6 (forward plan; the old
   `sharding_implementation_plan.md` is the completed-work record + principles).
4. Skim `.claude/lessons.md` — the jax/GPU/placement/measurement playbook.  The
   newest, load-bearing section is **"Sharded VCD memory: jax reference cycles +
   buffer donation"** (directly relevant to the next task); the memory-ruler and
   benign-warning lessons also recur.
Verify claims against current code; memory/docs may lag.

Where we are: the placement architecture is live (`recon_placement`/`sino_placement`
alongside the legacy `main_device`/`sinogram_device`; the banded adjoint movement pair
`sum_band_to_owner`/`broadcast_band_to_views` in `mbirjax/_sharding/`).  DONE &
validated (CPU + GPU, 1–4 devices): sharded **back projection** (reduce-scatter,
slice-banded), **forward projection** (all-gather), and **P4 — VCD on placements**
(qGGMRF prior `_qggmrf_prior_sharded` halo-aware; prior-opt A `_stage_halos`; the
alpha host-sync fix via `_replicate_scalar`).

**The sharded-VCD memory leak is FIXED & validated (this was the big recent thread).**
Root cause: jax keeps sharded (`NamedSharding`) arrays in **internal reference cycles**,
so per-subset *out-of-place* updates of the view-sharded error sinogram leaked one full
sinogram/subset until GC (peak grew with subsets×passes; 1-device-mesh OOM'd at 1008³).
Fix (mesh-guarded; single-device untouched): `update_error_sinogram` does the subtract
under `@partial(jax.jit, donate_argnames='error_sinogram')` → **in-place** (alpha scaled
eagerly to keep trivial-mesh bit-exact), and a **single end-of-subset cleanup section**
`.delete()`s the eager-op transients (scaled delta; + `weighted_error_sinogram` for
non-const weights).  Forward-projection (`assemble_sharded`) outputs free on refcount, so
positivity needs no extra delete.  Validated GPU (504³/5-iter/1-device-mesh): const 6.9,
non-const 7.9, positivity 7.3 GB (all flat, no leak); 1008³ 1-device completes (was OOM);
1–4 GPU scaling 1008³ **1d→4d = 4.49× super-linear**, memory shards 1/n_dev; correctness
vs prerelease 8.79e-7; full suite (`tests/sharding/` + `tests/test_vcd.py`) **77 passed**.
Also landed: the **`is_sharded` property** (single source of truth replacing the
`self.mesh is None/not None` checks) and a size-sweep TIME-ideal curve fix (∝ voxels·views,
not voxels).

First task: **step 4 — mesh/no-mesh → placement unification** (v2 plan P6).  Fold the dual
`is_sharded`/no-mesh code paths into one always-on placement path: `update_error_sinogram`
becomes the single error-sinogram update (like `update_recon`), the `is_sharded` *guards*
retire — but the **transient-free cleanup section STAYS** (sharded-array reference cycles
are inherent to host-orchestrated arrays; donation + `.delete()` don't go away — see the
lessons section).  One design call to make: does a trivial 1-device placement resolve to a
plain `SingleDeviceSharding` (keeps single-device cycle-free, no `.delete()`/block there)
or a 1-device `NamedSharding` (uniform path, but then single-device also pays the cleanup)?
This is a broad, hot-path refactor → scope and propose before editing; suite is the gate.

Then / deferred: **P5** (device-config UX — `configure_devices`); **P6** (port the
placement/movement path to cone/translation/multiaxis, and retire
`main_device`/`sinogram_device`).  Deferred questions: the **hybrid** recon-CPU/sino-GPU
`qggmrf_..._transfer` timing; the **prox-map prior** under sharding (untouched); **B**
(parallelize the prior's per-shard loop — a CPU nicety; the GPU 4d behavior was size, not
the prior); a **qGGMRF scaling test + prerelease baseline** to round out the suite.
Speculative "many small subsets" structural ideas (row-shard parallel beam, sharded
momentum) are filed in chat history / not in scope.

Reminders:
- Commit workflow: I commit from PyCharm — **stage only / write DRAFT commit messages;
  never run `git commit`.**  Run tests with
  `source /Users/gbuzzard/miniforge3/etc/profile.d/conda.sh && conda activate mbirjax`;
  sharding tests in `tests/sharding/` (the sharded VCD file is `test_vcd_sharded.py`).
- I run the GPU/large recons on the cluster; flag GPU items for me.  **After editing
  mbirjax, the cluster needs a fresh `pip install -e .`** — a stale editable build once
  cost a whole diagnosis detour (a "33 GB leak" that was just the pre-fix binary).  When
  GPU behavior contradicts the local tests, suspect the build first.  Cluster cards can
  thermally throttle — pre-flight `nvidia-smi dmon`.  `results/` and `baselines/` are
  gitignored — record decision numbers in committed docs.
- **Sharded jax arrays sit in reference cycles** → freed only by cyclic GC, not refcount;
  single-device (`SingleDeviceSharding`) arrays free on refcount.  So update persistent
  sharded state **in place via buffer donation** and **`.delete()` eager-op transients**;
  don't fold the alpha scale into the donated jit (FMA breaks bit-exactness).  (Full
  detail in lessons.md.)
- Memory ruler: `peak_bytes_in_use` is the REAL live working set, preallocation-invariant
  (`PREALLOCATE=false` does NOT reveal the capacity floor; lower
  `XLA_PYTHON_CLIENT_MEM_FRACTION` to find the OOM threshold).  Don't extrapolate absolute
  MB across sizes.  Projection **time ∝ N⁴ (voxels×views), memory ∝ N³ (voxels)**.
- `CUDA_ERROR_NOT_PERMITTED` from `cuda_vmm_allocator.cc` is a BENIGN warning (silenced by
  default via `TF_CPP_MIN_LOG_LEVEL=2` in `mbirjax/_device_setup.py`).  Real errors are
  `E`/`F` or a Python traceback.
