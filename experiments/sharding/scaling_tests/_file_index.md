# experiments/sharding/scaling_tests — file index

- `scaling_common.py` — shared engine, JAX-free to import (jax/mbirjax are
  imported lazily inside the functions that need them, so the orchestrator can
  use the pure helpers without spinning up a backend): device/platform detection,
  timing (warmup+trials), peak memory (GPU per-device / CPU RSS), correctness
  metric (percent of points above an fp32 threshold), YAML I/O, baseline npy/yaml
  save+load, speedup/mem-fraction annotators, subprocess orchestration
  (`run_worker`/`write_worker_result`), matplotlib plots, and a best-effort GPU
  topology/UUID snapshot (`gpu_topology`, via `nvidia-smi`) so a run records which
  physical GPUs + interconnect/NUMA it got (the allocation-quality variable behind
  run-to-run multi-device scaling surprises), and a per-GPU clock/temp sample
  (`sample_gpu_state` / `throttled_gpus`) so a thermally-throttling card
  auto-flags itself instead of masquerading as a code regression.  Problem-size
  sets live in each op driver, not here.
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
  `SIZES` are smaller (tune freely).  Measures EACH path in `BACK_PROJECT_PATHS`
  ('band' = slice-banded, 'pixel' = pixel-batched) in its own fresh process per
  (size, path), writes a YAML + plots per path, and prints a band-vs-pixel
  time/memory comparison table (the head-to-head for the band-vs-pixel decision).
  Top-of-file knobs: `DEVICE_COUNTS` (e.g. [1,2,3] to skip a known-bad 4th GPU;
  GPU `SIZES` are divisible by 1/2/3/4 for this), and `PIXEL_BATCH_SWEEP` (sweep
  the pixel path's B_p to see whether it can match band's memory).  Each measure
  records a per-GPU clock/temp sample and marks a row `[THROTTLED]` if a GPU was
  throttling (that timing is unreliable).  Run from the beta worktree root.
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
- `sparse_back_project_memory_attribution.py` — GPU diagnostic at a fixed (size,
  device count).  Part A: exact per-device buffer breakdown (sino shard, recon
  output, full-cylinder partial, vmap buffer).  Part C: slice-band sweep —
  `back_project_slice_band` vs peak/shard + time, to find the memory/throughput
  knee.  (A "Part B" view/pixel batch sweep was removed — batch tuning leaves the
  peak flat.)  Isolated-subprocess harness.
- `sparse_back_project_single_device_sweep.py` — GPU single-device (n_dev=1)
  slice-band sweep: how far streaming cuts peak on ONE GPU (the "stretch the recon
  per GPU" measurement) and the sino+recon floor it can't beat without host
  streaming.  Sweeps band full→small, continues past OOM, reports peak / vs_full /
  peak-over-floor.
- `direct_recon_scaling.py` — scaling + correctness driver for the **Phase F2**
  end-to-end FBP pipeline (`direct_recon` / `fbp_recon`).  Same
  isolated-subprocess harness as `sparse_back_project_scaling.py`.  The timed
  unit is the on-device pipeline `fbp_filter -> sparse_back_project` on a
  PRE-SHARDED sinogram (slice-sharded out, NO gather), so it isolates the
  combined compute scaling and the combined transient memory (the filter FFT
  work area PLUS the back-projection partial buffer); the user-facing
  `direct_recon` additionally pays a non-scaling entry shard + exit gather, left
  out of the timed region.  Correctness (setup worker) compares the FULL
  single-device `direct_recon` volume vs the prerelease baseline.  Shares
  `SIZES` with the back-projection driver (the projector dominates).  Run from
  the beta worktree root.
- `direct_recon_capture_baseline.py` — run ONCE from a prerelease checkout to
  capture the prerelease `direct_recon` (FBP) volume for correctness comparison.
  Writes `baselines/direct_recon.{npy,yaml}`.  NOT a bit-exact gate: the
  back-projection single-device path is verbatim prerelease, but F1 rewrote the
  filter (per-view -> `apply_row_filter`), so beta-vs-prerelease is a FLOAT-NOISE
  comparison (~8e-8 on CPU, 0% above the 1e-4 threshold) that catches real
  divergence and CPU/GPU drift above that floor.
- `vcd_recon_scaling.py` — scaling + correctness driver for end-to-end sharded VCD
  (`TomographyModel.vcd_recon`).  Same isolated-subprocess harness as the projector
  drivers.  The timed unit is the INTERNAL `vcd_recon` on a PRE-SHARDED sinogram
  with FIXED partitions (built once, outside the timed region), returning the
  slice-sharded recon (NO exit gather) — so it isolates the iterative loop's
  forward/back projection + slice-sharded qGGMRF prior + cross-device movement and
  the peak transient of one full reconstruction.  Determinism: the timed callable
  seeds the GLOBAL numpy RNG each call (VCD shuffles subset order per iteration), so
  every trial does identical work and the result is reproducible across device
  counts.  Top-of-file knobs: `DEVICE_COUNTS`, `MAX_ITERATIONS`, per-platform
  `SIZES` (smaller than the projector drivers' — VCD is iterative).  Correctness
  (setup worker) runs the FULL single-device `recon` at a small size and compares
  the volume vs the prerelease baseline (FLOAT-NOISE gate; ~5e-8 on CPU).  Run from
  the beta worktree root.
- `vcd_recon_capture_baseline.py` — run ONCE from a prerelease checkout to capture
  the prerelease end-to-end `recon` volume.  Writes `baselines/vcd_recon.{npy,yaml}`.
  Seeds the global RNG once (partitions + per-iteration subset shuffles + the random
  sinogram all come from that one seeded stream) and forces exactly MAX_ITERATIONS
  (`stop_threshold_change_pct=0`) so capture and check are aligned.  Shares
  `run_reference_recon` with `vcd_recon_scaling.py` (one definition, one RNG
  sequence).  NOT bit-exact (iterative accumulation amplifies reduce-order
  differences; the qGGMRF prior's no-halo path is bit-exact with prerelease) — a
  float-noise gate that catches real divergence and CPU/GPU drift.
- `vcd_shard_vs_noshard.py` — demo-style head-to-head of the user-facing `recon`
  run no-shard (single device) vs shard (4 virtual CPU devices, or all usable GPUs)
  on the SAME Shepp-Logan data, reporting time / peak memory / NRMSE and the
  shard/no-shard ratios.  Isolated-subprocess harness (one fresh worker per mode;
  the shard worker's CPU virtual-device count is set via its env before JAX import).
  Runs the WHOLE recon (entry shard + loop + exit gather), so it reflects the real
  user call — for the pure sharded-loop peak use `vcd_recon_scaling.py`.  CPU memory
  is whole-process RSS (no per-device win on CPU); the per-device memory benefit
  shows on GPU.  Top-of-file knobs: size, `MAX_ITERATIONS`, `CPU_SHARD_DEVICES`.
- `results/` (gitignored) — generated YAML tables (timing + speedup + correctness
  metrics) and plots.
- `baselines/` (gitignored) — single prerelease reference: `<op>.npy` (array) +
  `<op>.yaml` (metadata: capture platform, seed, size, shape, dtype, timing).
