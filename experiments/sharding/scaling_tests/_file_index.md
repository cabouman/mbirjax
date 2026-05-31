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
- `fbp_filter_capture_baseline.py` — run ONCE from a prerelease checkout to
  capture the prerelease `fbp_filter` output for correctness comparison.  Writes
  a SINGLE platform-agnostic baseline (`baselines/fbp_filter.npy` for the array,
  `baselines/fbp_filter.yaml` for metadata); every beta run (CPU or GPU) compares
  against it, so a real CPU/GPU divergence is visible, not hidden.
- `results/` (gitignored) — generated YAML tables (timing + speedup + correctness
  metrics) and plots.
- `baselines/` (gitignored) — single prerelease reference: `<op>.npy` (array) +
  `<op>.yaml` (metadata: capture platform, seed, size, shape, dtype, timing).
