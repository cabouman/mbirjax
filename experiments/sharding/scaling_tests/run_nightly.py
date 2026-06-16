"""
experiments/sharding/scaling_tests/run_nightly.py
─────────────────────────────────────────────────
Nightly entry for the regression harness, driven by ``run_regression.sh`` via ENV VARS (no CLI
args, per project convention).  Sibling of ``run_performance_local.py`` but for the *nightly*:

  * measures a SPECIFIC library worktree (``REG_LIB_ROOT``) chosen by the wrapper — NOT the tree
    this file lives in.  The harness can therefore live in the metrics repo and measure main /
    prerelease / a dev branch, each in its own throwaway worktree.
  * writes the dated YAML into the metrics-repo results dir (``REG_OUT_DIR``);
  * GATES (non-zero exit on a HARD regression) so the cron / slurm job surfaces it as a real alert.

The library is selected by ``Config.lib_root`` -> ``build_worker_env(lib_root=...)`` sets each
worker's ``PYTHONPATH`` to that worktree and provenance is taken from it (the worktree's real SHA).
``lib_root`` defaults to ``beta_root()`` when unset, so ``run_performance_local`` is unchanged.

Env vars (set by the wrapper):
  REG_LIB_ROOT  (required)  absolute path to the library worktree under test
  REG_OUT_DIR   (required)  stable results dir, e.g. <metrics>/results/<plat>/<branch_slug>/
  REG_DATE      (optional)  YYYYMMDD, resolved ONCE by the wrapper (default: today)
  REG_GATE      (optional)  "1" (default) to set a non-zero exit on a hard-gate regression
  REG_RUN_TAG   (optional)  label recorded in the YAML (e.g. the branch name)
"""
import os
from datetime import datetime

import performance_tracking as pt   # module-level is JAX-free; workers import mbirjax lazily


def _require(name):
    val = os.environ.get(name)
    if not val:
        raise SystemExit(f"run_nightly: required env var {name} is not set")
    return val


def main():
    overrides = dict(
        lib_root=_require("REG_LIB_ROOT"),   # selects the library under test (PYTHONPATH + provenance)
        out_dir=_require("REG_OUT_DIR"),     # stable, OUTSIDE the throwaway worktree (day-over-day lives here)
        date=os.environ.get("REG_DATE") or datetime.now().strftime("%Y%m%d"),
        gate=os.environ.get("REG_GATE", "1") == "1",
        compare_to_prior=True,               # day-over-day vs the prior file in this out_dir (same branch)
    )
    run_tag = os.environ.get("REG_RUN_TAG")
    if run_tag:
        overrides["run_tag"] = run_tag
    golden_dir = os.environ.get("REG_GOLDEN_DIR")   # e.g. <metrics>/golden -> golden gate + vs-main
    if golden_dir:
        overrides["golden_dir"] = golden_dir

    if os.environ.get("REG_SMOKE") == "1":
        # Fast plumbing smoke (NOT a real measurement): a trivial 1-cell sweep to shake out the
        # wrapper end-to-end (clone -> worktree -> install -> engine -> results/state) in seconds.
        overrides.update(geometries=["parallel"], ops=["back"], device_counts=[1],
                         sizes={"cpu": [(40, 40, 48)], "gpu": [(40, 40, 48)]})

    config = pt.Config(**overrides)
    print("=" * 72)
    print("  performance_tracking — NIGHTLY run")
    print(f"  lib_root (under test): {config.lib_root}")
    print(f"  out_dir:               {config.out_dir}")
    print(f"  date / tag / gate:     {config.date} / {run_tag or '-'} / {config.gate}")
    print("=" * 72)

    result = pt.run(config)
    if config.gate and result and (result.get("gate") or {}).get("result") == "fail":
        raise SystemExit(1)   # HARD regression -> the wrapper turns this into a notification


if __name__ == "__main__":
    main()
