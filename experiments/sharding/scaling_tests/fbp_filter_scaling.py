"""
experiments/sharding/scaling_tests/fbp_filter_scaling.py
─────────────────────────────────────────────────────────
Scaling + correctness driver for ParallelBeamModel.fbp_filter under sharding.

Measures one grid over (problem size × device count) and writes a text table,
YAML, and two plots to results/ — two views of the same grid:
  - device sweep: speedup & per-device memory vs device count, one curve per size;
  - size sweep:   time & memory vs size, one curve per device count.

ISOLATED-SUBPROCESS HARNESS
───────────────────────────
Memory (peak per-device) can only be measured reliably if each configuration
runs in a *fresh* process: JAX's peak_bytes_in_use / CPU RSS are cumulative
high-water marks that never reset within a process.  So this script is an
ORCHESTRATOR that spawns isolated WORKER subprocesses and never touches JAX
itself (otherwise it would hold GPU memory while a worker measures peak usage):

  - orchestrator (default, no args)  : spawns workers, collects YAML, plots.
  - worker --mode setup              : reports platform/devices + correctness.
  - worker --mode measure --size ... : times + measures memory for one size,
                                       running device counts DESCENDING (8→4→2→1)
                                       so per-device allocation is ascending in
                                       the fresh process and peak reads correctly.

You never pass --worker yourself; the orchestrator passes it to its own
subprocesses.  Run configuration lives in the labeled constants below (no CLI
args for the human) so runs are reproducible and PyCharm-friendly.

Run from the BETA worktree root (the orchestrator forces the beta worktree onto
each worker's PYTHONPATH, and the setup worker prints the resolved mbirjax path):

    python experiments/sharding/scaling_tests/fbp_filter_scaling.py
"""
import os
import sys
import argparse

import scaling_common as sc

import numpy as np

# NOTE: mbirjax (and therefore jax) is imported INSIDE the worker functions, not
# at the top level, on purpose.  This one file runs in two roles: the default
# (no --worker) is the ORCHESTRATOR, which only spawns worker subprocesses and
# collects their results and must stay completely JAX-free; the --worker
# invocations are WORKER subprocesses that actually touch JAX.  A top-level
# `import mbirjax` would run in both roles, pulling jax into the orchestrator —
# so the orchestrator could hold a (GPU) backend while a worker is measuring
# peak_bytes_in_use, contaminating exactly the per-device memory number the
# isolated-subprocess design exists to measure cleanly.  Importing mbirjax only
# in the worker keeps the orchestrator a pure, JAX-free supervisor.  (The import
# also has a load-bearing side effect — it runs mbirjax._device_setup, which sets
# the virtual-CPU-device XLA flag, and must precede jax backend init; see the
# import site in worker_measure.)


OP_NAME = "fbp_filter"

# Time-ideal slope for the size-sweep plot.  fbp_filter is a PER-VIEW filter whose
# cost is the sinogram size (∝ voxels, N³) -- unlike the projectors / VCD, which
# touch each voxel once per view (∝ voxels·views, N⁴, the plotter's default).
TIME_IDEAL = "voxels"

# ── Run configuration (edit here; no CLI args for the human) ──────────────────
# Problem sizes (n_views, n_rows, n_channels) PER OP — different ops want
# different sizes.  Both sweeps share these sizes and the device-count ladder
# (powers of two up to the available max, e.g. 1,2,4,8).  n_views must be
# divisible by every device count in the ladder, so the sizes below use powers
# of two that are multiples of the max (128/256/512 all divide 8/4/2/1).
# fbp_filter is cheap per element (a 1-device 256^3 run was ~0.2 s), so we use
# fairly large sizes to make the scaling visible.
# The GPU top size is 1624^3 (not 2048^3): 1624 ≈ 1024·4^(1/3), a ~4× volume
# step over 1024^3 rather than 8×, and 1624 is divisible by 8/4/2/1.
SIZES = {
    "cpu": [(128, 128, 128), (256, 256, 256), (512, 512, 512)],
    "gpu": [(512, 512, 512), (1024, 1024, 1024), (1624, 1624, 1624)],
}
WARMUP = 1
TRIALS = 3
CORRECTNESS_THRESHOLD = 1e-4

CORRECTNESS_SIZE = (64, 64, 64)   # small, fixed; comparison is size-independent
CORRECTNESS_SEED = 1234


# ── Op-specific builders (used by the worker) ─────────────────────────────────
def make_model(size, devices=None):
    """Build a ParallelBeamModel for the given (views, rows, channels).

    Returns None if the requested device count does not evenly divide the
    sharded axes (configure_sharding raises) — the caller skips that point.
    """
    import mbirjax
    n_views, n_rows, n_channels = size
    angles = np.linspace(0, np.pi, n_views, endpoint=False)
    model = mbirjax.ParallelBeamModel((n_views, n_rows, n_channels), angles)
    if devices is not None:
        try:
            model.configure_sharding(devices)
        except ValueError as e:
            print(f"    (skip: {len(devices)} devices incompatible with size "
                  f"{sc.size_label(size)}: {e})")
            return None
    return model


def make_input(size, seed=0):
    """Deterministic random sinogram of the given shape (numpy float32)."""
    rng = np.random.default_rng(seed)
    return rng.random(size, dtype=np.float32)


def run_fbp_filter(model, sino):
    """The timed op: filter a (pre-sharded if mesh) sinogram."""
    return model.fbp_filter(sino)


def parse_size_label(label):
    """'256x256x256' -> (256, 256, 256)."""
    return tuple(int(x) for x in label.split("x"))


# ── Worker side (runs in an isolated subprocess) ──────────────────────────────
def worker_setup(out_file):
    """Report platform, device count/label, and the single-device correctness
    check against the prerelease baseline.  One launch, before sizes are known."""
    import mbirjax  # device-setup-first: before any jax import (via sc below)
    plat, max_dev = sc.detect_platform()
    dev_label = sc.device_label()

    ref, meta = sc.load_baseline(OP_NAME)
    model = make_model(CORRECTNESS_SIZE, devices=None)
    sino = make_input(CORRECTNESS_SIZE, seed=CORRECTNESS_SEED)
    beta_out = np.asarray(run_fbp_filter(model, sino))

    if ref is None:
        corr = {"baseline_present": False}
        print("[setup] no prerelease baseline; correctness skipped "
              "(run fbp_filter_capture_baseline.py from a prerelease checkout)")
    else:
        captured_on = meta.get("captured_on_platform", "unknown")
        m = sc.correctness_metrics(ref, beta_out, threshold=CORRECTNESS_THRESHOLD)
        corr = {"baseline_present": True, "baseline_platform": captured_on,
                "current_platform": plat, "cross_platform": captured_on != plat, **m}
        print(f"[setup] correctness vs {captured_on} baseline: "
              f"max_abs_diff={m['max_abs_diff']:.3e}  "
              f"pct_above={m['pct_above_threshold']:.6f}%"
              + ("   <-- CROSS-PLATFORM" if captured_on != plat else ""))

    result = sc.build_setup_result(plat, max_dev, dev_label, corr)
    sc.write_worker_result(out_file, result)


def worker_measure(size_label, device_counts, warmup, trials, out_file):
    """Time + measure memory for one size; the shared descent does the rest."""
    # mbirjax's first import runs mbirjax._device_setup, which sets the virtual-CPU
    # device XLA flag from MBIRJAX_NUM_CPU_DEVICES; that must be in place BEFORE jax
    # initializes its backend.  scaling_common imports jax lazily, so the first
    # trigger is pick_devices() inside run_measure_loop -- import mbirjax here first.
    import mbirjax  # noqa: F401  (device-setup side effect; must precede jax init)
    size = parse_size_label(size_label)
    sino_np = make_input(size, seed=0)

    def build_and_time(n, devs):
        model = make_model(size, devices=devs)
        if model is None:
            return None
        # Pre-shard outside the timing loop: measure the op, not the scatter.
        sino = model._shard_sinogram(sino_np)
        stats, _ = sc.time_op(lambda: run_fbp_filter(model, sino), warmup, trials)
        mem_mb, mem_kind = sc.peak_memory_mb(devs)
        return stats, mem_mb, mem_kind

    sc.run_measure_loop(size_label, device_counts, out_file, build_and_time)


def run_worker(argv):
    """Dispatch a --worker invocation (internal; the orchestrator builds argv)."""
    p = argparse.ArgumentParser(description="fbp_filter scaling worker (internal)")
    p.add_argument("--worker", action="store_true")
    p.add_argument("--mode", choices=["setup", "measure"], required=True)
    p.add_argument("--size", default=None, help="LxRxC, for --mode measure")
    p.add_argument("--device-counts", type=int, nargs="+", default=None)
    p.add_argument("--warmup", type=int, default=WARMUP)
    p.add_argument("--trials", type=int, default=TRIALS)
    p.add_argument("--out-file", required=True)
    a = p.parse_args(argv)
    if a.mode == "setup":
        worker_setup(a.out_file)
    else:
        worker_measure(a.size, a.device_counts, a.warmup, a.trials, a.out_file)


# ── Orchestrator (default; touches no JAX) ────────────────────────────────────
def main():
    script = os.path.abspath(__file__)

    # Each worker inherits the beta worktree on PYTHONPATH + the JAX allocator
    # knobs (see sc.build_worker_env).
    worker_env = sc.build_worker_env()

    print("=" * 72)
    print("  fbp_filter scaling — isolated-subprocess harness (orchestrator)")
    print(f"  beta root: {sc.beta_root()}")
    print("=" * 72)

    # 1. Setup worker: platform, device count/label, correctness.
    setup, rc = sc.run_worker(
        script, ["--worker", "--mode", "setup"], extra_env=worker_env)
    if setup is None:
        print(f"  ERROR: setup worker produced no result (rc={rc}); aborting.")
        return
    plat, max_dev, dev_label, corr, mpath = sc.print_setup_banner(setup)

    sizes = SIZES[plat]
    size_labels = [sc.size_label(s) for s in sizes]
    device_counts = sc.default_device_counts(max_dev)   # e.g. [1, 2, 4, 8]
    # On CPU, pin the virtual device count so each worker matches this count.
    if plat == "cpu":
        worker_env["MBIRJAX_NUM_CPU_DEVICES"] = str(max_dev)
    print(f"  sizes: {size_labels}")
    print(f"  device counts: {device_counts}")

    # 2. One measure worker per size (fresh process each → clean peak memory).
    grid = {}
    failures_by_size = {}
    mem_kind = "n/a"
    for label in size_labels:
        args = ["--worker", "--mode", "measure", "--size", label,
                "--device-counts", *[str(n) for n in device_counts],
                "--warmup", str(WARMUP), "--trials", str(TRIALS)]
        res, rc = sc.run_worker(script, args, extra_env=worker_env)
        rows = (res or {}).get("rows") or []
        fails = (res or {}).get("failures") or []
        if fails:
            failures_by_size[label] = fails
            for fl in fails:
                tag = "OOM" if fl.get("oom") else "ERROR"
                print(f"  size {label}: n_devices={fl['n_devices']} {tag}")
        if not rows:
            print(f"  size {label}: worker returned no rows (rc={rc}); skipping")
            grid[label] = []
            continue
        mem_kind = res.get("mem_kind", mem_kind)
        sc.annotate_speedups(rows)        # speedup vs 1 device
        sc.annotate_mem_fraction(rows)    # memory relative to 1 device
        rows.sort(key=lambda r: r["n_devices"])   # ascending for readable YAML
        grid[label] = rows
        by_n = {r["n_devices"]: r for r in rows}
        summary = "  ".join(f"{n}d={by_n[n]['speedup']:.2f}x" for n in sorted(by_n))
        print(f"  size {label} speedup: {summary}")

    # 3. Persist results (YAML) and the two plots (two views of the one grid).
    results = {
        "op": OP_NAME,
        "platform": plat,
        "device_label": dev_label,
        "mbirjax_path": mpath,
        "warmup": WARMUP, "trials": TRIALS,
        "device_counts": device_counts,
        "sizes": size_labels,
        "mem_kind": mem_kind,
        "time_ideal": TIME_IDEAL,
        "correctness": corr,
        "grid": grid,
        "failures": failures_by_size,
    }
    sc.save_yaml(os.path.join(sc.RESULTS_DIR, f"{OP_NAME}_{plat}.yaml"), results)

    if any(grid.values()):
        sc.plot_device_sweep(
            OP_NAME, grid, device_counts, size_labels, dev_label, mem_kind,
            os.path.join(sc.RESULTS_DIR, f"{OP_NAME}_{plat}_device_sweep.png"))
        sc.plot_size_sweep(
            OP_NAME, grid, device_counts, size_labels, dev_label, mem_kind,
            os.path.join(sc.RESULTS_DIR, f"{OP_NAME}_{plat}_size_sweep.png"),
            time_ideal=TIME_IDEAL)

    print("\nDone.")


if __name__ == "__main__":
    if "--worker" in sys.argv:
        run_worker(sys.argv[1:])
    else:
        main()
