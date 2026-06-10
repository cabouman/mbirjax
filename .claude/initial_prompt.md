

This is the mbirjax CT reconstruction project — multi-GPU/CPU sharding work in the
`mbirjax` worktree on branch `greg/parallel_sharding`.  This is the **single** working
tree (the old dual-worktree setup was consolidated): the research branch
`greg/parallel_tests` is retired — its worktree was removed and its local branch
deleted (it still exists on the remote `cabouman/mbirjax` if ever needed).

Orient first by reading, in order:
1. `.claude/claude_prompt.md` — collaboration style + workflow (investigate and
   propose before editing; wait for approval on non-trivial changes; minimal,
   localized changes; no plan-phase/step references in code/comments/docstrings;
   measure honestly — suspect the ruler before the code; stay curious and challenge
   my assumptions).
2. `experiments/sharding/plans/sharding_status.md` — the latest HANDOFF (current
   state + NEXT).  Plans evolve through our dialog; they are not set in stone.
3. `experiments/sharding/plans/sharding_implementation_plan_v2.md` — forward plan.
   Read §0 design summary, the **§Migration state & branch-retirement map**, §P5/§P6,
   §Decisions, and §Adjacent tasks.  (`sharding_implementation_plan.md` = completed-work
   record + principles.)
4. Skim `.claude/lessons.md` — the jax/GPU/placement/measurement playbook
   ("Sharded VCD memory: reference cycles + buffer donation" is the load-bearing one).
Verify claims against current code; memory/docs may lag.

**Where we are (2026-06-10).**  The placement architecture is live and ParallelBeam runs
the always-on placement path.  This session: **hybrid (recon-CPU/sino-GPU 'sinograms') is
REMOVED** (selection tier + `_transfer` plumbing gone; auto is "GPU present → 'full', else
'none'", no pre-flight memory guess — an over-large recon OOMs-and-guides at recon time).
**P5 is mostly DONE & GPU-validated:** `configure_devices(None|int|list)`; **auto shards by
default across all dividing GPUs** (`_auto_device_count`, no floor; GPU-only — CPU is the
`_auto_shard_cpu` opt-in); an always-on `_device_report` ("N x PLATFORM [(sharded)]");
**order-independent divisibility** (config/geometry-change WARN, the hard clear error is at
the shard chokepoint `_shard_on_axis`).  **Band sizing RESOLVED** (GPU sweep: keep the
`n_dev²` default; budget-driven sizing rejected).  `set_devices_and_batch_sizes` renamed
`set_devices`.

**Next (see status NEXT + v2 §P5/§P6/§Adjacent):**
- **P5 Step 4 (the last P5 piece) — divisibility *padding*** so a non-dividing axis uses
  more than 1 device: **views** pad + `weights=0` mask (easy) + `prepare_sino_for_devices`;
  **slices** pick-N / problem-level pad + prior-aware mask (hard — likely defer).  Pad to a
  shape the problem owns, never to a multiple of N.  *Needs careful design — start fresh.*
- **P6** — port cone/translation/multiaxis to the placement/movement path, then the cascade
  of deletions (legacy `is_sharded` else-branches, `self.mesh`, `RETIRE-AFTER` tests,
  `pixel_indices_worker`/`partition_worker`).  NOTE: the hybrid drop is already DONE here
  (no longer a P6 item).  Also deferred to P6: device-aware `scale_recon_shape`/
  `auto_set_recon_geometry` (cone lever).
- **Adjacent / standalone** — settable view parameters (retire the `view_indices` hack; lift
  `view_params_array` from closure to a runtime arg + `set_view_parameters`); seed test RNGs;
  **CPU-cluster auto-sharding** (measure real-cluster perf; mature `_auto_shard_cpu`).
- **Carry into P6:** `is_sharded` and `n_devices > 1` are **decoupled** — re-read each
  `is_sharded` site for which question it asks (`len(shard_devices) > 1` for "≥2 devices").

Reminders:
- Commit workflow: I commit from PyCharm — **stage only / write DRAFT commit messages;
  never run `git commit`.**  Run tests with
  `source /Users/gbuzzard/miniforge3/etc/profile.d/conda.sh && conda activate mbirjax`;
  sharding tests in `tests/sharding/` (`MBIRJAX_NUM_CPU_DEVICES=4` to exercise multi-device
  on CPU; sharded VCD file is `test_vcd_sharded.py`).
- I run GPU/large recons on the cluster; **flag GPU items.**  Auto-sharding end-to-end only
  runs on real multi-GPU (CPU CI covers explicit `configure_sharding` + the `_auto_device_count`
  unit test + the `_auto_shard_cpu` opt-in).  After editing mbirjax the cluster needs a fresh
  `pip install -e .` — a stale build once impersonated a "leak".  Cards thermally throttle —
  pre-flight `nvidia-smi dmon`.  `results/`/`baselines/` are gitignored — record decisions in docs.
- **Sharded jax arrays sit in reference cycles** (freed by cyclic GC, not refcount;
  `SingleDeviceSharding` frees on refcount).  Update persistent sharded state **in place via
  buffer donation**, `.delete()` eager-op transients (`update_error_sinogram` is a donated FMA).
  This also surfaced a viewer bug: `slice_viewer` now `del`s + `gc.collect()`s so TkAgg's
  orphaned tk objects aren't finalized by a later GC during a sharded recon.
- Correctness gate is **`allclose` ~1e-4**, not bit-exactness.
- OOM/error handling: `is_oom` + `log_oom_guidance` in `mbirjax/_utils.py`; `_handle_jax_error`
  guides only on a real OOM and keys on the recon device platform (`_recon_devices`), not `use_gpu`.
- Memory ruler: `peak_bytes_in_use` is the REAL live set, preallocation-invariant (lower
  `XLA_PYTHON_CLIENT_MEM_FRACTION` to find OOM).  Projection **time ∝ N⁴, memory ∝ N³**.
- `CUDA_ERROR_NOT_PERMITTED` from `cuda_vmm_allocator.cc` is BENIGN (silenced by
  `TF_CPP_MIN_LOG_LEVEL=2` in `mbirjax/_device_setup.py`).  Real errors are `E`/`F` or a traceback.
