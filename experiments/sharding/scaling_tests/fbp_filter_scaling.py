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

  - orchestrator (default, no args)  : spawns workers, collects JSON, plots.
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
import gc
import sys
import argparse

import scaling_common as sc

import numpy as np


OP_NAME = "fbp_filter"

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

# Which fbp_filter kernel to measure: "per_view" | "flat" | "row_batch".  The
# worker applies this to mbirjax.parallel_beam._FBP_FILTER_KERNEL before timing,
# so the harness can compare kernels without editing the library.
FBP_KERNEL = "row_batch"

CORRECTNESS_SIZE = (64, 64, 64)   # small, fixed; comparison is size-independent
CORRECTNESS_SEED = 1234

# Substrings (upper-cased) that mark a caught failure as memory exhaustion.
# Beyond the clean allocator tokens, GPU FBP hits cuFFT OOM, which XLA surfaces
# as "INTERNAL: RET_CHECK ... Failed to create cuFFT batched plan with scratch
# allocator" / "Failed to allocate work area" — none of the usual OOM tokens.
# (Confirmed on H100 at 1624^3 / 1 device.)
_OOM_MARKERS = ("RESOURCE_EXHAUSTED", "OUT OF MEMORY", "OOM", "BAD_ALLOC",
                "FAILED TO ALLOCATE", "WORK AREA", "SCRATCH ALLOCATOR",
                "FAILED TO CREATE CUFFT")


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


def _select_kernel(kernel):
    """Apply the chosen fbp_filter kernel to the library before measuring."""
    import mbirjax.parallel_beam as pb
    pb._FBP_FILTER_KERNEL = kernel


# ── Worker side (runs in an isolated subprocess) ──────────────────────────────
def worker_setup(out_file, fbp_kernel):
    """Report platform, device count/label, and the single-device correctness
    check against the prerelease baseline.  One launch, before sizes are known."""
    import mbirjax  # device-setup-first: before any jax import (via sc below)
    _select_kernel(fbp_kernel)
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

    pkg_path = os.path.dirname(mbirjax.__file__)
    beta_state, branch = sc.beta_status(pkg_path)   # by git branch, not dir name
    result = {"platform": plat, "max_devices": max_dev, "device_label": dev_label,
              "mbirjax_path": pkg_path, "beta_state": beta_state, "branch": branch,
              "fbp_kernel": fbp_kernel, "correctness": corr}
    sc.write_worker_result(out_file, result)
    print(f"[setup] platform={plat}  max_devices={max_dev}  ({dev_label})")


def worker_measure(size_label, device_counts, warmup, trials, out_file, fbp_kernel):
    """Time + measure memory for one size, device counts DESCENDING.

    Descending order (8→4→2→1) makes per-device allocation ascending within this
    fresh process, so the cumulative peak_bytes_in_use equals each config's own
    allocation when read right after it.  It also means once a config OOMs, every
    later (fewer-device) config needs MORE per-device memory and would OOM too —
    so we catch the OOM, record it, and stop the descent.  Results are written
    incrementally so even a hard crash returns the completed configs.
    """
    import mbirjax  # device-setup-first
    _select_kernel(fbp_kernel)
    size = parse_size_label(size_label)
    desc = sorted(set(device_counts), reverse=True)
    print(f"\n[measure {size_label}]  device counts (descending): {desc}")
    sino_np = make_input(size, seed=0)
    rows = []
    failures = []
    mem_kind = "n/a"

    def _publish():
        sc.write_worker_result(out_file, {"size": size_label, "mem_kind": mem_kind,
                                          "rows": rows, "failures": failures})

    for n in desc:
        devs = sc.pick_devices(n)
        if devs is None:
            print(f"  n_devices={n}: not enough devices, skipping")
            continue
        model = make_model(size, devices=devs)
        if model is None:
            continue
        try:
            # Pre-shard outside the timing loop: measure the op, not the scatter.
            sino = model._shard_sinogram(sino_np)
            stats, _ = sc.time_op(lambda: run_fbp_filter(model, sino), warmup, trials)
            mem_mb, mem_kind = sc.peak_memory_mb(devs)
        except Exception as e:   # noqa: BLE001 — measurement harness: never abort the sweep
            msg = str(e).replace("\n", " ")
            is_oom = any(k in msg.upper() for k in _OOM_MARKERS)
            failures.append({"n_devices": n, "oom": is_oom, "error": msg[:300]})
            print(f"  n_devices={n:2d}  {'OOM' if is_oom else 'ERROR'}: {msg[:120]}")
            _publish()
            if is_oom:
                print(f"  stopping descent at {size_label}: fewer-device configs "
                      f"need more per-device memory and would also OOM")
                break
            continue
        rows.append({"n_devices": n, **stats, "mem_mb": mem_mb})
        print(f"  n_devices={n:2d}  min={stats['min_ms']:8.2f} ms  "
              f"mean={stats['mean_ms']:8.2f} ms  mem={mem_mb:8.1f} MB ({mem_kind})")
        # Publish partial progress and free this config before the next (larger)
        # one so peak_bytes_in_use reflects each config alone.
        _publish()
        del model, sino
        gc.collect()   # release device buffers before the next config allocates
    _publish()


def run_worker(argv):
    """Dispatch a --worker invocation (internal; the orchestrator builds argv)."""
    p = argparse.ArgumentParser(description="fbp_filter scaling worker (internal)")
    p.add_argument("--worker", action="store_true")
    p.add_argument("--mode", choices=["setup", "measure"], required=True)
    p.add_argument("--size", default=None, help="LxRxC, for --mode measure")
    p.add_argument("--device-counts", type=int, nargs="+", default=None)
    p.add_argument("--warmup", type=int, default=WARMUP)
    p.add_argument("--trials", type=int, default=TRIALS)
    p.add_argument("--fbp-kernel", default=FBP_KERNEL)
    p.add_argument("--out-file", required=True)
    a = p.parse_args(argv)
    if a.mode == "setup":
        worker_setup(a.out_file, a.fbp_kernel)
    else:
        worker_measure(a.size, a.device_counts, a.warmup, a.trials, a.out_file,
                       a.fbp_kernel)


# ── Orchestrator (default; touches no JAX) ────────────────────────────────────
def _beta_root():
    """Beta worktree root, derived from this file's location.

    This file is at <beta>/experiments/sharding/scaling_tests/, so the worktree
    root is three directories up from the file's directory.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        os.pardir, os.pardir, os.pardir))


def main():
    script = os.path.abspath(__file__)

    # Force the beta worktree onto each worker's PYTHONPATH so `import mbirjax`
    # resolves to beta regardless of how the orchestrator was launched (PyCharm
    # or CLI) — removes the sys.path footgun for the subprocesses.  Never
    # preallocate the whole GPU, so peak_bytes_in_use reflects true usage.
    beta_root = _beta_root()
    if not os.path.isdir(os.path.join(beta_root, "mbirjax")):
        print(f"  WARNING: no mbirjax/ under derived beta root {beta_root}")
    existing_pp = os.environ.get("PYTHONPATH", "")
    worker_env = {
        "PYTHONPATH": beta_root + (os.pathsep + existing_pp if existing_pp else ""),
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }

    print("=" * 72)
    print("  fbp_filter scaling — isolated-subprocess harness (orchestrator)")
    print(f"  beta root: {beta_root}")
    print("=" * 72)

    # 1. Setup worker: platform, device count/label, correctness.
    setup, rc = sc.run_worker(
        script, ["--worker", "--mode", "setup", "--fbp-kernel", FBP_KERNEL],
        extra_env=worker_env)
    if setup is None:
        print(f"  ERROR: setup worker produced no result (rc={rc}); aborting.")
        return
    plat = setup["platform"]
    max_dev = setup["max_devices"]
    dev_label = setup["device_label"]
    corr = setup["correctness"]
    mpath = setup.get("mbirjax_path", "?")
    beta_state = setup.get("beta_state", "unknown")
    branch = setup.get("branch")
    if beta_state == "beta":
        label = f"*** beta ***  (branch {branch})"
    elif beta_state == "not-beta":
        label = f"### NOT beta — branch {branch} — check PYTHONPATH ###"
    else:
        label = "(branch undetermined — verify path manually)"
    print(f"  mbirjax: {label}   {mpath}")
    print(f"  platform: {plat}   max devices: {max_dev}   ({dev_label})")
    print(f"  fbp kernel: {setup.get('fbp_kernel', '?')}")
    if corr.get("baseline_present"):
        print(f"  correctness: max_abs_diff={corr['max_abs_diff']:.3e}  "
              f"pct_above={corr['pct_above_threshold']:.6f}%"
              + ("   <-- CROSS-PLATFORM" if corr.get("cross_platform") else ""))
    else:
        print("  correctness: no baseline present")

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
                "--warmup", str(WARMUP), "--trials", str(TRIALS),
                "--fbp-kernel", FBP_KERNEL]
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
        "fbp_kernel": FBP_KERNEL,
        "mbirjax_path": mpath,
        "warmup": WARMUP, "trials": TRIALS,
        "device_counts": device_counts,
        "sizes": size_labels,
        "mem_kind": mem_kind,
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
            os.path.join(sc.RESULTS_DIR, f"{OP_NAME}_{plat}_size_sweep.png"))

    print("\nDone.")


if __name__ == "__main__":
    if "--worker" in sys.argv:
        run_worker(sys.argv[1:])
    else:
        main()
