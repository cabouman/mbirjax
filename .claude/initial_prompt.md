

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
2. `experiments/sharding/plans/sharding_status.md` — the latest HANDOFF (current
   state + NEXT).  Plans evolve through our dialog; they are not set in stone.
3. `experiments/sharding/plans/sharding_implementation_plan_v2.md` — forward plan.
   Read §0 design summary, the **§Migration state & branch-retirement map** (the
   two tables: device representations + what retires when), §P5/§P6, and
   §Adjacent tasks.  (`sharding_implementation_plan.md` = completed-work record +
   principles.)
4. Skim `.claude/lessons.md` — the jax/GPU/placement/measurement playbook
   ("Sharded VCD memory: reference cycles + buffer donation" is the load-bearing one).
Verify claims against current code; memory/docs may lag.

**Where we are (2026-06-08).**  The placement architecture is live, and **step 4
(Option B / "2-Keep") is DONE and GPU-validated** (8 GPUs to 2048³; single-GPU within
noise of the old path): ParallelBeam now runs the always-on placement path by default
(homogeneous single-device auto-defaults to a trivial 1-device mesh via
`_supports_sharding()` + `_sharding_configured`).  **Hybrid (recon-CPU/sino-GPU) is
DECIDED for drop** (the +19% N it buys is beaten by a 2nd GPU / slice-subset stitching;
analysis archived).  The dual `is_sharded` else-branches still exist — they retire at P6
(for the unported geometries + hybrid).

**Next (see status NEXT + v2 §P6/§P5/§Adjacent):**
- **P6** — port cone/translation/multiaxis to the placement/movement path, then the
  cascade of deletions (legacy single-device branches, `self.mesh`, `RETIRE-AFTER`
  tests, **hybrid drop**: `_transfer` + `'sinograms'` + the auto-default else-branch).
- **P5** (parallelizable) — `configure_devices`; make `set_devices_and_batch_sizes`
  device-count-aware (today it only inspects `gpus[0]`); divisibility (pick-N, or
  pad+mask — views easy via `weights=0`, slices hard; pad to the problem, never to N).
- **Adjacent / standalone** — settable view parameters (retire the `view_indices`
  hack; lift `view_params_array` from closure to a runtime arg + `set_view_parameters`);
  seed test RNGs to solidify the suite.
- **Carry into P6:** `is_sharded` and `n_devices > 1` have **decoupled** — re-read each
  `is_sharded` site for which question it asks (`len(shard_devices) > 1` for "≥2 devices").

Reminders:
- Commit workflow: I commit from PyCharm — **stage only / write DRAFT commit messages;
  never run `git commit`.**  Run tests with
  `source /Users/gbuzzard/miniforge3/etc/profile.d/conda.sh && conda activate mbirjax`;
  sharding tests in `tests/sharding/` (sharded VCD file is `test_vcd_sharded.py`).
- I run GPU/large recons on the cluster; **flag GPU items.**  After editing mbirjax the
  cluster needs a fresh `pip install -e .` — a stale editable build once impersonated a
  "leak"; when GPU behavior contradicts local tests, suspect the build first.  Cluster
  cards can thermally throttle — pre-flight `nvidia-smi dmon`.  `results/` and
  `baselines/` are gitignored — record decision numbers in committed docs.
- **Sharded jax arrays sit in reference cycles** (freed by cyclic GC, not refcount;
  `SingleDeviceSharding` frees on refcount).  Update persistent sharded state **in place
  via buffer donation**, `.delete()` eager-op transients.  This is permanent, not a
  step-4 scaffold (`update_error_sinogram` is now a donated FMA with `alpha` folded in).
- Correctness gate is **`allclose` ~1e-4**, not bit-exactness (the relaxation that
  underwrites the one-path direction).
- Memory ruler: `peak_bytes_in_use` is the REAL live set, preallocation-invariant
  (`PREALLOCATE=false` does NOT reveal the floor; lower `XLA_PYTHON_CLIENT_MEM_FRACTION`
  to find OOM).  Projection **time ∝ N⁴, memory ∝ N³**.  Utils in `mbirjax/memory_stats.py`.
- `CUDA_ERROR_NOT_PERMITTED` from `cuda_vmm_allocator.cc` is BENIGN (silenced by
  `TF_CPP_MIN_LOG_LEVEL=2` in `mbirjax/_device_setup.py`).  Real errors are `E`/`F` or a
  Python traceback.
