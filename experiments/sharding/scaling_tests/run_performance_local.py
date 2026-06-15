"""
experiments/sharding/scaling_tests/run_performance_local.py
───────────────────────────────────────────────────────────
Manual / ad-hoc launcher for the performance_tracking engine, for the CURRENT working tree.

Use this to measure in-progress changes WITHOUT touching the nightly regression results: it
runs against whatever code is in this checkout (no temp worktree) and forces the output into
``results/manual/<RUN_TAG>/`` with a timestamped filename, so repeated runs never overwrite
each other or the dated nightly files under ``results/regression/``.

    python experiments/sharding/scaling_tests/run_performance_local.py

Edit the CONFIG block below — it overrides a subset of performance_tracking.Config; every field
left unset keeps the engine default.  For step-through debugging set INLINE = True (single
process, fully debuggable in PyCharm; peak memory is then CUMULATIVE across the sweep, not
per-config — trust the default subprocess numbers for the memory ruler, INLINE for logic).
"""
import os
from datetime import datetime

import performance_tracking as pt
import scaling_common as sc


# ── CONFIG (edit here; a subset of performance_tracking.Config) ───────────────
GEOMETRIES = ["parallel", "cone"]
OPS = ["direct_filter", "forward", "back", "vcd_nonconst"]
DEVICE_COUNTS = [1, 2, 4]

# Per-platform SINOGRAM sizes (n_views, n_rows, n_channels), keyed 'cpu'/'gpu'.
# None -> use the engine's default sizes.  Override for a quick, small local run.
SIZES = None
# Example small override:
# SIZES = {"cpu": [(64, 56, 48)], "gpu": [(256, 224, 192)]}

INLINE = False          # True = single process, debuggable (cumulative memory)
RUN_TAG = "local"       # output -> results/manual/<RUN_TAG>/
VCD_ITERATIONS = 3


def main():
    overrides = dict(
        geometries=GEOMETRIES,
        ops=OPS,
        device_counts=DEVICE_COUNTS,
        inline=INLINE,
        run_tag=RUN_TAG,
        vcd_iterations=VCD_ITERATIONS,
        gate=False,             # informational diff only; a local run never fails the process

        # Isolated output: never the nightly results/regression/ dir.  Timestamped date so
        # repeated manual runs accumulate side by side instead of clobbering.
        out_dir=os.path.join(sc.RESULTS_DIR, "manual", RUN_TAG or "local"),
        date=datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    if SIZES is not None:
        overrides["sizes"] = SIZES
    config = pt.Config(**overrides)
    print("=" * 72)
    print("  performance_tracking — MANUAL local run (current working tree)")
    print(f"  out_dir: {config.out_dir}")
    print(f"  inline:  {config.inline}")
    print("=" * 72)
    pt.run(config)


if __name__ == "__main__":
    main()
