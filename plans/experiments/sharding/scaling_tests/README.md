# Performance / regression tracking

This directory holds the **performance-tracking toolchain** for the multi-device sharding work:
it measures correctness, time, and peak memory for the core operations across geometries and
device counts, tracks them over time, and gates on regressions.  Design + rationale:
`../plans/performance_tracking_plan.md`.

Older, per-operation scaling/baseline scripts (`*_scaling.py`, `*_capture_baseline.py`, the cone
ablations, etc.) have been superseded by `performance_tracking.py` and moved to `archive/`.

## The current scripts

| script | role |
|---|---|
| `scaling_common.py` | shared harness: isolated-subprocess workers, `time_op`, `peak_memory_mb`, throttle/topology sampling, YAML I/O.  Not run directly. |
| `performance_tracking.py` | the **engine** — sweeps geometry × op × size × device-count, writes a dated YAML, updates the record book, and runs the diff/gate.  Default run = the nightly config. |
| `run_performance_local.py` | **manual launcher** — params at the top; runs the current working tree, writes to an isolated `results/manual/<tag>/`, gate off.  For ad-hoc / in-progress measurement. |
| `capture_golden.py` | capture/refresh the **golden** reference the gate compares against (`results/golden/golden_<plat>.yaml`); `ONLY` for a selective refresh. |
| `capture_main_baseline.py` | capture the **`main`-branch single-device baseline** (run from a `main` worktree): time + peak memory + fingerprint per cell at the full sweep sizes -> `main_baseline_<plat>.yaml`, plus a small `.npy` correctness array per (geom, op).  The engine auto-discovers this and prints a soft "vs main (1 device)" note. |

Each measured cell is keyed `(geometry, op, size, n_dev)` and records `min_ms`, `mem_mb`,
`speedup`, structural flags, and a tolerant correctness `fingerprint`.

## Running it

Activate the env first:
```bash
source /Users/gbuzzard/miniforge3/etc/profile.d/conda.sh && conda activate mbirjax   # local
# on a cluster: module load cuda/conda as usual, then `conda activate mbirjax`
```

### CPU (local / Mac)
Runs out of the box — the engine creates virtual CPU devices (`MBIRJAX_NUM_CPU_DEVICES`, default 4)
and uses the `cpu` size set.
```bash
cd plans/experiments/sharding/scaling_tests
python performance_tracking.py          # full sweep -> results/regression/regression_cpu_<date>.yaml
python run_performance_local.py         # ad-hoc, edit the CONFIG block first -> results/manual/<tag>/
```
For step-through debugging set `INLINE = True` in `run_performance_local.py` (single process;
memory is then cumulative, so trust the default subprocess numbers for the memory ruler).

### GPU (cluster)
```bash
git pull && pip install -e .            # FRESH build — a stale build once impersonated a leak
nvidia-smi                              # GPUs visible, not occupied/throttling; run on ALLOCATED GPUs
cd plans/experiments/sharding/scaling_tests
python performance_tracking.py          # auto-detects GPU; uses the gpu size set (512/513/1024)
```
The engine sweeps each device count the hardware has (`[1,2,4]`, filtered to the GPU count) and
records per-device peak memory; an op that OOMs at a device count is recorded as a failure and the
descent stops there (smaller counts need more per-device memory).

### Golden + cross-version baseline
```bash
python capture_golden.py                # results/golden/golden_<plat>.yaml (drift/accept reference)
# capture the main-branch 1-device baseline (run from a main worktree so import mbirjax = main):
git worktree add ../mbirjax_main main
PYTHONPATH=../mbirjax_main python capture_main_baseline.py   # -> results/golden/main_baseline_<plat>.yaml + <geom>_<op>.npy
git worktree remove ../mbirjax_main
```

## Output layout (under `results/`, gitignored)
- `results/regression/regression_<plat>_<date>.yaml` — the dated time series (day-over-day diff).
- `results/regression/records_<plat>.yaml` — best-ever record book (per cell/metric + the commit).
- `results/manual/<tag>/` — isolated manual-run output (own dated files + record book).
- `results/golden/` — `golden_<plat>.yaml` (fingerprint reference) + the `.npy` deep-diff arrays.

## The gate (brief)
HARD (fails, exit non-zero): correctness fingerprint, structural changes, status `ok→fail`,
expected-but-absent cells, and — **GPU only** — memory growth.  SOFT (warn): speedup-ratio drop,
absolute time, CPU memory, sweep-set add/drop.  Every delta is reported with both its absolute and
percentage difference.  See the plan §10 for the full taxonomy.
