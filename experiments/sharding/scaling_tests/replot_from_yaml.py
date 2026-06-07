"""
experiments/sharding/scaling_tests/replot_from_yaml.py
──────────────────────────────────────────────────────
Regenerate the size-sweep and device-sweep plots from a saved results YAML, WITHOUT
re-running the (slow) measurement.  Useful after changing the plotting code — e.g.
the size-sweep TIME ideal curve now scales as voxels×views (projection cost), not
voxels (volume).

The grid in the YAML already has everything the plotters need (min_ms, mem_mb,
speedup, mem_frac per device count), so this is a pure re-render.  JAX-free.

Edit YAML_NAME, then run from this directory:
    python replot_from_yaml.py
"""
import os

import yaml

import scaling_common as sc


# ── config (edit here) ────────────────────────────────────────────────────────
YAML_NAME = "vcd_recon_gpu.yaml"   # a file in results/


def main():
    path = os.path.join(sc.RESULTS_DIR, YAML_NAME)
    with open(path) as f:
        r = yaml.safe_load(f)

    op = r["op"]
    grid = r["grid"]
    device_counts = r["device_counts"]
    sizes = r["sizes"]
    dev_label = r.get("device_label", r.get("platform", "?"))
    mem_kind = r.get("mem_kind", "n/a")

    base = os.path.splitext(path)[0]   # results/<op>_<plat>
    sc.plot_size_sweep(op, grid, device_counts, sizes, dev_label, mem_kind,
                       base + "_size_sweep.png")
    sc.plot_device_sweep(op, grid, device_counts, sizes, dev_label, mem_kind,
                         base + "_device_sweep.png")
    print(f"Re-rendered plots from {path}")


if __name__ == "__main__":
    main()
