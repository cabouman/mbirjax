# experiments/sharding/scaling_tests — file index

Scripts to measure time/memory scaling of the sharded operators vs **device count**
and **problem size**, plus the correctness baselines they compare against.  Grouped
by purpose below.  Resolved one-off diagnostics live in `archive/` (see
`archive/_file_index.md`).

**Run-location contract (read first):**
- **Scaling drivers** and **live diagnostics** — run from the **beta worktree root**
  (the orchestrator forces the beta worktree onto each worker's PYTHONPATH and prints
  the resolved path).  No CLI args for the human: edit the top-of-file constants, then
  `python experiments/sharding/scaling_tests/<driver>.py`.
- **Capture-baseline scripts** — run **once from a *prerelease* checkout**, NOT the
  beta worktree.  Each writes a single platform-agnostic reference into beta's
  `baselines/`; every beta run (CPU or GPU) compares against it.
- **GPU hygiene** (multi-device runs): `nvidia-smi dmon` to pick/pin a cool card;
  device counts are measured DESCENDING so per-device peak memory reads correctly; a
  throttling card auto-flags its rows `[THROTTLED]`.

---

## Shared engine
- `scaling_common.py` — JAX-free-to-import engine: device/platform detection, timing
  (warmup+trials), peak memory (GPU per-device / CPU RSS), correctness metric, YAML +
  baseline I/O, speedup/mem-fraction annotators, subprocess orchestration
  (`run_worker`), the size/device-sweep matplotlib plots, and a GPU topology/clock/temp
  snapshot (so allocation quality and thermal throttling are recorded, not mistaken for
  code regressions).  It also hosts the **shared driver harness** every scaling driver
  builds on, so a driver is just config + op shims: `OOM_MARKERS`/`is_oom`, `beta_root`,
  `build_worker_env`, `build_setup_result` (worker side) + `print_setup_banner`
  (orchestrator side), and `run_measure_loop` (the device-count descent — OOM stop,
  throttle sampling, incremental publish, gc — driven by a per-op `build_and_time`
  callback).  Per-op problem-size sets live in each driver, not here.

## Scaling + correctness drivers (one per op)
Isolated-subprocess harness (in `scaling_common`): a JAX-free orchestrator spawns fresh
`--mode setup` (platform/devices + correctness) and `--mode measure` (one size, device
counts descending) workers, then writes a YAML grid + a device-sweep and a size-sweep
plot.  Each driver supplies only its op shims (`make_model` / `make_input` / `run_op` /
a correctness check) + a `build_and_time` callback; the orchestration and measurement
loop live in `scaling_common`.  Top-of-file knobs: `SIZES`, `DEVICE_COUNTS`,
`WARMUP`/`TRIALS`, correctness size/seed.
- `fbp_filter_scaling.py` — `ParallelBeamModel.fbp_filter` (view-sharded ramp filter).
- `sparse_back_project_scaling.py` — sharded back projection (reduce-scatter); times
  the internal op on a pre-sharded sinogram (slice-sharded out, no gather).  Heaviest
  per element, so `SIZES` are smaller; `BACK_PROJECT_PATHS`/`PIXEL_BATCH_SWEEP` knobs.
- `sparse_forward_project_scaling.py` — sharded forward projection (all-gather, the
  adjoint); correctness is the adjoint identity `<Ax,y> == <x,Aᵀy>`.  Single path
  (no band-vs-pixel), so one YAML + two plots.
- `direct_recon_scaling.py` — Phase-F2 FBP pipeline `fbp_filter → sparse_back_project`
  on a pre-sharded sinogram; shares `SIZES` with the back-projection driver.
- `vcd_recon_scaling.py` — end-to-end sharded VCD on a pre-sharded sinogram with fixed
  partitions (built outside the timed region); slice-sharded out, no gather.  Smaller
  `SIZES` (iterative); `MAX_ITERATIONS`, `MEM_*` knobs; seeds the global RNG per call
  for reproducibility.

## Capture-baseline scripts (run ONCE on a prerelease checkout)
Each writes `baselines/<op>.{npy,yaml}` (one platform-agnostic reference; CPU/GPU
divergence then shows up as a real difference).
- `fbp_filter_capture_baseline.py`
- `sparse_back_project_capture_baseline.py`
- `sparse_forward_project_capture_baseline.py`
- `direct_recon_capture_baseline.py` — float-noise gate (F1 rewrote the filter), not
  bit-exact.
- `vcd_recon_capture_baseline.py` — float-noise gate (iterative); shares
  `run_reference_recon` / the seeded RNG sequence with `vcd_recon_scaling.py`.

## Live diagnostics (not yet resolved)
- `vcd_single_device_baseline.py` — single-device `recon` time + peak-memory baseline
  for the "don't regress the 1-GPU path" check; `MESH` toggle runs the no-mesh path
  (prerelease-equivalent) or a trivial 1-device mesh (sharded path on one device).
  *To be repurposed for the step-4 Option B no-regression guard (tight allclose vs
  prerelease) — see `plans/sharding_implementation_plan_v2.md` §P6.*
- `sparse_back_project_single_device_sweep.py` — GPU single-device slice-band sweep:
  how far streaming cuts peak on ONE GPU and the sino+recon floor it can't beat.
  *Feeds the deferred P6 memory-budget band sizing.*

## Plotting
- `replot_from_yaml.py` — re-render the size-sweep + device-sweep plots from a saved
  results YAML without re-measuring (edit `YAML_NAME`, run from this dir).  JAX-free.

## Data directories (gitignored)
- `results/` — generated YAML grids (timing + speedup + correctness) and plots.
- `baselines/` — the single prerelease reference per op (`<op>.npy` + `<op>.yaml`).
- `logs/` — worker run logs.
- `archive/` — resolved one-off diagnostics (kept in git; see `archive/_file_index.md`).
