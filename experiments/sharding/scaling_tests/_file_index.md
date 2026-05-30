# experiments/sharding/scaling_tests — file index

- `scaling_common.py` — shared engine: mbirjax-path banner, device/platform
  detection, timing (warmup+trials), best-effort peak memory (GPU per-device /
  CPU RSS), correctness metric (percent of points above an fp32 threshold),
  YAML I/O, and matplotlib plots.  Size presets per platform (quick/medium/large).
- `fbp_filter_scaling.py` — scaling + correctness driver for
  `ParallelBeamModel.fbp_filter`: device sweep (strong scaling) and size sweep,
  vs a stored prerelease baseline.  Run from the beta worktree.
- `fbp_filter_capture_baseline.py` — run ONCE from a prerelease checkout to
  capture the prerelease `fbp_filter` output for correctness comparison.
- `results/` (gitignored) — generated YAML tables and plots.
- `baselines/` (gitignored) — captured prerelease reference outputs.
