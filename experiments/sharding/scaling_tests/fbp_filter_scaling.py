"""
experiments/sharding/scaling_tests/fbp_filter_scaling.py
─────────────────────────────────────────────────────────
Scaling + correctness driver for ParallelBeamModel.fbp_filter under sharding.

Runs two sweeps and writes text tables, YAML, and plots to results/:
  1. by device count (strong scaling) at a fixed problem size;
  2. by problem size at a fixed device count.

Correctness is checked against a stored prerelease baseline (captured once with
fbp_filter_capture_baseline.py run from a prerelease checkout).  The correctness
size is small and fixed; the comparison is essentially size-independent because
the relevant differences come from geometry/pixel patterns, not array scale.

Run (from the BETA worktree root so beta code is loaded — the banner prints the
resolved mbirjax path so you can confirm):

    python experiments/sharding/scaling_tests/fbp_filter_scaling.py
    python experiments/sharding/scaling_tests/fbp_filter_scaling.py --size-set medium
    python experiments/sharding/scaling_tests/fbp_filter_scaling.py --device-counts 1 2 4
"""
import argparse
import os

import scaling_common as sc

import numpy as np


OP_NAME = "fbp_filter"
CORRECTNESS_SIZE = (64, 64, 64)   # small, fixed; comparison is size-independent
CORRECTNESS_SEED = 1234


# ── Op-specific builders ──────────────────────────────────────────────────────
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


# ── Sweeps ────────────────────────────────────────────────────────────────────
def sweep_by_devices(size, device_counts, warmup, trials):
    """Strong scaling: fixed problem size, vary device count."""
    print(f"\n{'─'*72}\nDEVICE SWEEP  (size {sc.size_label(size)})\n{'─'*72}")
    rows = []
    sino_np = make_input(size, seed=0)
    for n in device_counts:
        devs = sc.pick_devices(n)
        if devs is None:
            print(f"  n_devices={n}: not enough devices, skipping")
            continue
        model = make_model(size, devices=devs)
        if model is None:
            continue
        # Pre-shard outside the timing loop so we measure the op, not the scatter.
        sino = model._shard_sinogram(sino_np)
        stats, _ = sc.time_op(lambda: run_fbp_filter(model, sino), warmup, trials)
        mem_mb, mem_kind = sc.peak_memory_mb(devs)
        rows.append({"n_devices": n, **stats, "mem_mb": mem_mb})
        print(f"  n_devices={n:2d}  min={stats['min_ms']:8.2f} ms  "
              f"mean={stats['mean_ms']:8.2f} ms  mem={mem_mb:8.1f} MB ({mem_kind})")
    return rows, mem_kind if rows else "n/a"


def sweep_by_size(sizes, n_devices, warmup, trials):
    """Fixed device count, vary problem size."""
    print(f"\n{'─'*72}\nSIZE SWEEP  (n_devices {n_devices})\n{'─'*72}")
    rows = []
    devs = sc.pick_devices(n_devices)
    if devs is None:
        print(f"  n_devices={n_devices}: not enough devices; size sweep skipped")
        return rows, "n/a"
    for size in sizes:
        model = make_model(size, devices=devs)
        if model is None:
            continue
        sino = model._shard_sinogram(make_input(size, seed=0))
        stats, _ = sc.time_op(lambda: run_fbp_filter(model, sino), warmup, trials)
        mem_mb, mem_kind = sc.peak_memory_mb(devs)
        rows.append({"size": sc.size_label(size), **stats, "mem_mb": mem_mb})
        print(f"  size={sc.size_label(size):>14s}  min={stats['min_ms']:8.2f} ms  "
              f"mem={mem_mb:8.1f} MB ({mem_kind})")
    return rows, mem_kind if rows else "n/a"


def check_correctness(threshold):
    """Compare beta fbp_filter to the stored prerelease baseline (if present)."""
    print(f"\n{'─'*72}\nCORRECTNESS vs prerelease baseline\n{'─'*72}")
    plat, _ = sc.detect_platform()
    baseline_path = os.path.join(sc.BASELINES_DIR, f"{OP_NAME}_{plat}.yaml")
    # Always also compute the single-device beta result for a self-consistency
    # reference (useful even when no prerelease baseline exists yet).
    model = make_model(CORRECTNESS_SIZE, devices=None)
    sino = make_input(CORRECTNESS_SIZE, seed=CORRECTNESS_SEED)
    beta_out = np.asarray(run_fbp_filter(model, sino))

    if not os.path.exists(baseline_path):
        print(f"  no prerelease baseline at {baseline_path}")
        print(f"  (run fbp_filter_capture_baseline.py from a prerelease checkout)")
        return {"baseline_present": False}

    base = sc.load_yaml(baseline_path)
    ref = np.array(base["output"], dtype=np.float32).reshape(base["output_shape"])
    metrics = sc.correctness_metrics(ref, beta_out, threshold=threshold)
    print(f"  max_abs_diff      = {metrics['max_abs_diff']:.3e}")
    print(f"  pct_above_thresh  = {metrics['pct_above_threshold']:.6f}%  "
          f"({metrics['n_above']}/{metrics['n_total']}, threshold {threshold:.0e})")
    return {"baseline_present": True, **metrics}


def main():
    plat, max_dev = sc.detect_platform()
    parser = argparse.ArgumentParser(description="fbp_filter scaling + correctness")
    parser.add_argument("--size-set", choices=["quick", "medium", "large"],
                        default="quick")
    parser.add_argument("--device-counts", type=int, nargs="+", default=None,
                        help="device counts for the device sweep (default: ladder)")
    parser.add_argument("--sweep-device-count", type=int, default=None,
                        help="device count for the size sweep (default: max)")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=1e-4)
    args = parser.parse_args()

    sc.which_mbirjax()
    print(f"  platform: {plat}   max devices: {max_dev}")

    device_counts = (sorted(set(args.device_counts)) if args.device_counts
                     else sc.default_device_counts(max_dev))
    sizes = sc.SIZE_PRESETS[plat][args.size_set]
    # Device sweep uses the first preset size (kept fixed); size sweep uses all.
    device_sweep_size = sizes[0]
    sweep_dev_count = args.sweep_device_count or max_dev

    corr = check_correctness(args.threshold)
    dev_rows, dev_mem_kind = sweep_by_devices(
        device_sweep_size, device_counts, args.warmup, args.trials)
    size_rows, size_mem_kind = sweep_by_size(
        sizes, sweep_dev_count, args.warmup, args.trials)

    # Persist results (YAML) and plots.
    results = {
        "op": OP_NAME,
        "platform": plat,
        "mbirjax_path": os.path.dirname(__import__("mbirjax").__file__),
        "warmup": args.warmup, "trials": args.trials,
        "correctness": corr,
        "device_sweep": {"size": sc.size_label(device_sweep_size),
                         "mem_kind": dev_mem_kind, "rows": dev_rows},
        "size_sweep": {"n_devices": sweep_dev_count,
                       "mem_kind": size_mem_kind, "rows": size_rows},
    }
    sc.save_yaml(os.path.join(sc.RESULTS_DIR, f"{OP_NAME}_{plat}.yaml"), results)

    if dev_rows:
        sc.plot_device_sweep(
            OP_NAME, [r["n_devices"] for r in dev_rows],
            [r["min_ms"] for r in dev_rows], [r["mem_mb"] for r in dev_rows],
            dev_mem_kind,
            os.path.join(sc.RESULTS_DIR, f"{OP_NAME}_{plat}_device_sweep.png"))
    if size_rows:
        sc.plot_size_sweep(
            OP_NAME, [r["size"] for r in size_rows],
            [r["min_ms"] for r in size_rows], [r["mem_mb"] for r in size_rows],
            size_mem_kind,
            os.path.join(sc.RESULTS_DIR, f"{OP_NAME}_{plat}_size_sweep.png"))

    print("\nDone.")


if __name__ == "__main__":
    main()
