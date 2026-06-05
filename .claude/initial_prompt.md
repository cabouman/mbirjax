

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
4. Skim `.claude/lessons.md` — the jax/GPU/placement/measurement playbook (the
   memory-ruler, per-subset-sharding-size-floor, and benign-warning lessons are
   recent and load-bearing).
Verify claims against current code; memory/docs may lag.

Where we are: the placement architecture is live (`recon_placement`/`sino_placement`
replacing `main_device`/`sinogram_device`; the banded adjoint movement pair
`sum_band_to_owner`/`broadcast_band_to_views` in `mbirjax/_sharding/`).  DONE &
validated (CPU + GPU): sharded **back projection** (reduce-scatter, slice-banded),
**forward projection** (all-gather), and **P4 — VCD on placements**: the slice-sharded
**qGGMRF prior** (`_qggmrf_prior_sharded`, halo-aware), **prior-opt A** (stage halos
once per partition pass — `_stage_halos`), and the **alpha host-sync fix** (line search
stays on-device via `_replicate_scalar`; zero per-subset host syncs in the sharded
path).  GPU scaling is healthy: at 1008³, 2→4 devices = **2.20×** — the 4-device
flatness at ≤512³ was just SIZE (per-subset work too small to amortize cross-NUMA),
not a structural defect.  Correctness gates green (sharded VCD suite; single-device
regressions; correctness vs prerelease baseline 8.8e-7).

First task: the **single-device memory / no-regression** thread (see the CURRENT FOCUS
block for the full 4 steps).  The slice-**band** is really a per-device *memory* knob
but is currently sized only by device count (`~num_slices/n_dev²`), so at `n_dev=1`
it's full (no streaming); the 1-device-MESH path OOM'd at 1008³ where **prerelease
single-device fits 1008³** — we must not regress the 1-GPU capability.  Steps: (1)
prerelease single-device time+mem baseline — script ready,
`experiments/sharding/scaling_tests/vcd_single_device_baseline.py` (Greg runs on
prerelease AND beta); (2) verify beta's *no-mesh* path matches prerelease (should be
the verbatim prerelease body — all my VCD edits are mesh-guarded); (3) diagnose the
1-device-MESH heaviness (full band? no within-band pixel-batching like the
single-device body?); (4) make band sizing **memory-budget-driven**
(`band = min(reduce-scatter-optimal(n_dev), cap_from_byte_budget)`) so it streams on
1 device and uses finer bands for huge recons on N devices — this is the v2 plan's
deferred P6 single-device-streaming / trivial-sharding unification, pulled forward.
Steps 3–4 touch the projector hot path → scope and propose before editing.

Lower-priority / deferred: **B** (parallelize the prior's per-shard loop via
`run_per_device` — a CPU nicety; the GPU 4d cap turned out to be size, not the prior);
a **qGGMRF scaling test + prerelease baseline** to round out the suite; the **hybrid**
recon-CPU/sino-GPU `qggmrf_..._transfer` timing question; then **P5** (device-config
UX) and **P6** (port the placement path to cone/translation/multiaxis + retire
`main_device`/`sinogram_device`).  Speculative "many small subsets" structural ideas
(row-shard parallel beam, sharded momentum, etc.) are filed in the chat history /
not in scope now.

Reminders:
- Commit workflow: I commit from PyCharm — **stage only / write DRAFT commit
  messages; never run `git commit`.**  Run tests with
  `source /Users/gbuzzard/miniforge3/etc/profile.d/conda.sh && conda activate mbirjax`;
  sharding tests in `tests/sharding/` (`pytest tests/sharding/`).
- I run the GPU/large recons on the cluster; flag GPU items for me.  Cluster cards
  can thermally throttle — pre-flight `nvidia-smi dmon`.  `results/` and `baselines/`
  are gitignored — record decision numbers in committed docs.
- Memory ruler (settled this session): `peak_bytes_in_use` is the REAL live working
  set and is **preallocation-invariant** — `PREALLOCATE=false` does NOT reveal the
  capacity floor; to find it, lower `XLA_PYTHON_CLIENT_MEM_FRACTION` (hard cap) and
  find the OOM threshold.  Don't extrapolate absolute MB across sizes.
- `CUDA_ERROR_NOT_PERMITTED` from `cuda_vmm_allocator.cc` ("FABRIC+POSIX_FD … will
  retry") is a BENIGN warning; it's now silenced by default
  (`TF_CPP_MIN_LOG_LEVEL=2` set in `mbirjax/_device_setup.py`, overridable).  Real
  errors are `E`/`F` or a Python traceback.
