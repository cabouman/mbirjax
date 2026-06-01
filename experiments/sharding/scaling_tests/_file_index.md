# experiments/sharding/scaling_tests — file index

- `scaling_common.py` — shared engine, JAX-free to import (jax/mbirjax are
  imported lazily inside the functions that need them, so the orchestrator can
  use the pure helpers without spinning up a backend): device/platform detection,
  timing (warmup+trials), peak memory (GPU per-device / CPU RSS), correctness
  metric (percent of points above an fp32 threshold), YAML I/O, baseline npy/yaml
  save+load, speedup/mem-fraction annotators, subprocess orchestration
  (`run_worker`/`write_worker_result`), and matplotlib plots.  Problem-size sets
  live in each op driver, not here.
- `fbp_filter_scaling.py` — scaling + correctness driver for
  `ParallelBeamModel.fbp_filter`.  ISOLATED-SUBPROCESS HARNESS: an orchestrator
  (default, no args, touches no JAX) spawns fresh worker subprocesses — `--mode
  setup` (platform/devices + correctness) and `--mode measure --size` (one size,
  device counts DESCENDING so per-device peak memory reads correctly).  Measures
  one (size × device-count) grid, plotted two ways (device sweep: speedup &
  per-device memory; size sweep: time & memory).  Sizes/warmup/trials are
  top-of-file constants.  Run from the beta worktree root; the orchestrator forces
  the beta worktree onto each worker's PYTHONPATH and prints the resolved path.
- `fbp_filter_row_batch_sweep.py` — sweeps `ROW_FILTER_BATCH` for the
  `apply_row_filter` kernel (the F1 throughput-knee study; picked B=1024 on the
  GPU).  Reuses `fbp_filter_scaling`'s builders / `_beta_root` / `_OOM_MARKERS`,
  so it must sit alongside it.
- `fbp_filter_capture_baseline.py` — run ONCE from a prerelease checkout to
  capture the prerelease `fbp_filter` output for correctness comparison.  Writes
  a SINGLE platform-agnostic baseline (`baselines/fbp_filter.npy` for the array,
  `baselines/fbp_filter.yaml` for metadata); every beta run (CPU or GPU) compares
  against it, so a real CPU/GPU divergence is visible, not hidden.
- `sparse_back_project_scaling.py` — scaling + correctness driver for sharded
  back projection (`TomographyModel.sparse_back_project`).  Same
  isolated-subprocess harness as `fbp_filter_scaling.py`.  The timed unit is the
  internal `sparse_back_project` on a PRE-SHARDED sinogram, returning the
  slice-sharded recon-at-indices (no gather) — so it isolates the reduce-scatter
  + compute and the transient per-device partial buffer `(num_pixels, num_slices)`
  (the memory whose size drives the "do we need slice sub-tiling" decision).
  Back projection is much heavier per element than fbp_filter, so its top-of-file
  `SIZES` are smaller (tune freely).  Run from the beta worktree root.
- `sparse_back_project_capture_baseline.py` — run ONCE from a prerelease checkout
  to capture the prerelease `sparse_back_project` output (over full-FOV indices)
  for correctness comparison.  Writes `baselines/sparse_back_project.{npy,yaml}`.
  Beta's single-device path is the unchanged prerelease body, so this also gives
  the beta single-device run a near-zero correctness floor and catches future
  drift.
- `sparse_back_project_phase_ablation.py` — diagnostic that splits sharded back
  projection into Phase 1 (per-device compute) and Phase 2 (reduce-scatter) and
  times each across device counts, to discriminate comms-bound vs
  bandwidth-bound scaling.  Faithful to production (reconstructs the exact
  closures from `_sparse_back_project_sharded`, same kernel / `run_per_device` /
  `move_shard` calls); single-process (timing, not memory); self-resolves to the
  beta worktree via a `sys.path` prepend.  Finding (CPU): the cap is Phase 1
  (bandwidth-bound), Phase 2 is ≤9% — see `plans/sharding_status.md`.
- `results/` (gitignored) — generated YAML tables (timing + speedup + correctness
  metrics) and plots.
- `baselines/` (gitignored) — single prerelease reference: `<op>.npy` (array) +
  `<op>.yaml` (metadata: capture platform, seed, size, shape, dtype, timing).
