# Nightly regression harness

A standing, **fire-on-change** day-over-day check: it watches a few mbirjax branches and, whenever
one moves, measures every geometry × op × size × device-count (time + peak memory + a tolerant
correctness fingerprint), diffs against that branch's previous run + the golden, and flags
regressions. CPU runs on a Mac (launchd); GPU runs on the cluster (slurm/scrontab).

## How it's laid out (two repos)

- **`mbirjax_metrics`** (github.com/gbuzzard/mbirjax_metrics) — the **self-contained harness + data**,
  cloned on every node:
  ```
  tooling/scaling_tests/   engine: scaling_common.py, performance_tracking.py, run_nightly.py,
                           capture_golden.py, capture_main_baseline.py, run_performance_local.py
  tooling/regression/      this wrapper: run_regression.sh, regression.env, enable/disable,
                           com.mbirjax.regression.plist, cluster_preamble.sh.example, README.md
  golden/                  golden_<plat>.yaml, main_baseline_<plat>.yaml, <geom>_<op>.npy
  results/<plat>/<branch>/ regression_<plat>_<date>.yaml  (the time series) + tests_*.log
  state/<plat>/<branch>    last MEASURED commit per branch (fire-on-change)
  ```
  The harness is **authored in the mbirjax tree** (`experiments/sharding/scaling_tests/` +
  `dev_scripts/regression/`) and **deployed** into `tooling/` with `deploy_to_metrics.sh`.

- **`mbirjax`** — only the **library under test**. The nightly never uses a fixed checkout; it
  fresh-clones mbirjax and makes a throwaway worktree per changed branch.

## What a run does (`run_regression.sh`)

1. **Bootstrap**: source the node preamble (cluster proxy/modules), fresh-clone `mbirjax_metrics`
   into `$WORK_DIR/metrics`, and re-exec that clone's wrapper (so remote harness changes are picked up).
2. Activate the **dedicated** conda env; install the harness deps (matplotlib, ruamel).
3. For each tracked branch: `git ls-remote` its head; **skip if unchanged** (vs `state/`).
4. If anything changed: fresh-clone mbirjax once; per changed branch → worktree at that commit →
   `pip install -e "$WT[extras]"` → run its tests → run the perf engine (`run_nightly.py`, with
   `lib_root=$WT` so the library under test is selected + provenance recorded) → record the SHA.
5. Commit + push `results/` + `state/` to the metrics repo (**push failure is non-fatal**).
6. Exit non-zero **only** on a hard-gate perf regression → the cron/slurm mail is a real alert.

No per-node paths are baked in — only URLs + `$WORK_DIR` (under `$HOME` or `$SCRATCH`).

## One-time setup

**Both platforms**
1. Clone the metrics repo somewhere stable (this is the bootstrap/entrypoint clone).
2. Create the dedicated env: `conda create -n mbirjax_regression python=3.12` (the wrapper installs
   the rest each run). **Do not reuse your dev env** — the per-branch editable installs churn it.
3. Author/edit the harness in the mbirjax tree, then `bash dev_scripts/regression/deploy_to_metrics.sh`
   → review → commit + push the metrics clone. The nightly only sees **pushed** harness changes.
4. Edit `tooling/regression/regression.env`: `TRACKED_BRANCHES`, `POLL_SCHEDULE`, `NOTIFY`, `WORK_DIR`.

**Mac (launchd)**
```
bash tooling/regression/enable_nightly.sh     # fills + loads the launchd agent (runs at next wake if asleep)
bash tooling/regression/disable_nightly.sh    # unload + remove
```

**Cluster (scrontab) — P6, not yet written**
- Copy `cluster_preamble.sh.example` to a real path; set `PREAMBLE_FILE` to it (it `module load`s
  conda/cuda/cudnn and exports `HTTPS_PROXY`/`HTTP_PROXY` — git needs the proxy from a compute node).
- Put a fine-grained PAT (write access to `mbirjax_metrics` only) in a `chmod 600` file; set `TOKEN_FILE`.
- `nightly_regression.slurm` + an `#SCRON` entry will invoke `run_regression.sh` (still TODO).

## Validate before trusting it (Mac dry-run)
```
# with the dedicated env created and the harness deployed+pushed:
ENABLED=1  bash tooling/regression/run_regression.sh
```
Watch it: fresh-clone metrics → re-exec → (ls-remote) one changed branch → worktree → install →
tests → engine → push. Confirm a `results/<plat>/<branch>/regression_<plat>_<date>.yaml` appears and
the `state/<plat>/<branch>` SHA updates. A second immediate run should report "no tracked branch
changed" (fire-on-change working).

## Knobs (`regression.env`)
`ENABLED` (kill-switch) · `TRACKED_BRANCHES` · `POLL_SCHEDULE` · `METRICS_URL`/`MBIRJAX_URL` ·
`WORK_DIR` · `INSTALL_EXTRAS_{cpu,gpu}` (`cuda12,test` on GPU; `test` on CPU) · `CONDA_ENV` ·
`HARNESS_DEPS` · `RUN_TESTS`/`TEST_CPU_DEVICES` · `PREAMBLE_FILE` · `TOKEN_FILE` · `NOTIFY`.

## Notes / current limits
- `enable_nightly.sh` supports a **daily** `POLL_SCHEDULE` (`M H * * *`); richer cron specs would need
  more `StartCalendarInterval` entries.
- Per-branch **test** results are logged but **not yet gated/diffed** (the perf engine is the alert
  path); test diffing is a later increment.
- `scaling_common` imports matplotlib at module level (hence `HARNESS_DEPS`); making that import lazy
  would drop the dependency for the (non-plotting) nightly — optional cleanup.
- Golden/`main_baseline` are read from `metrics/golden/` via `REG_GOLDEN_DIR`; capture/refresh them
  with `capture_golden.py` / `capture_main_baseline.py` (a deliberate, human-triggered step).
