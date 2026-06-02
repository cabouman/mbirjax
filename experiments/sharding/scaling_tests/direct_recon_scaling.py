"""
experiments/sharding/scaling_tests/direct_recon_scaling.py
──────────────────────────────────────────────────────────
Scaling + correctness driver for the sharded FBP pipeline (Phase F2) under the
view/slice sharding scheme.

``direct_recon`` / ``fbp_recon`` is the first usable end-to-end sharded pipeline:

    shard sinogram (view axis)
        -> fbp_filter        (internal: sharded view -> sharded view, per-view
                              ramp filter, zero cross-device comms)
        -> sparse_back_project (internal: reduce-scatter -> slice-sharded
                              recon-at-indices)
        -> gather recon      (user-facing boundary, at exit)

The **timed unit** here is the on-device pipeline ``fbp_filter ->
sparse_back_project`` on a PRE-SHARDED sinogram, returning the slice-sharded
recon-at-indices (NO gather) — exactly how fbp_filter_scaling and
sparse_back_project_scaling time their ops.  Pre-sharding (and not gathering)
isolates the combined COMPUTE scaling and, importantly, the combined **transient
per-device memory**: the FFT work area of the filter PLUS the back-projection
partial buffer ``(num_pixels, num_slices)``.  The user-facing ``direct_recon``
additionally pays a one-time entry shard (host->device scatter) and exit gather
(device->host); those are host transfers that do not scale with device count, so
they are deliberately excluded from the timed region (a separate end-to-end
measurement could add them, but they would only dilute the compute-scaling
signal this driver exists to show).

This mirrors sparse_back_project_scaling.py: same isolated-subprocess harness (an
orchestrator that touches no JAX spawns fresh worker subprocesses so per-device
peak memory reads correctly), same two plotted views of one (size × device-count)
grid, same top-of-file run configuration (no CLI args for the human).

  - orchestrator (default, no args)  : spawns workers, collects YAML, plots.
  - worker --mode setup              : reports platform/devices + single-device
                                       correctness of the full direct_recon vs
                                       the prerelease baseline.
  - worker --mode measure --size ... : times + measures memory for one size,
                                       device counts DESCENDING (8→4→2→1) so
                                       per-device allocation is ascending in the
                                       fresh process and peak reads correctly.

Run from the BETA worktree root:

    python experiments/sharding/scaling_tests/direct_recon_scaling.py
"""
import os
import gc
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


OP_NAME = "direct_recon"

# ── Run configuration (edit here; no CLI args for the human) ──────────────────
# Problem sizes (n_views, n_rows, n_channels).  The pipeline is dominated by the
# back-projection (the projector: cost ~ views × pixels × psf × slices), much
# heavier per element than the filter, so these match the back-projection sizes
# (smaller than the fbp_filter-only sizes) — tune freely.
# Divisibility: the view axis (n_views) and the slice axis (≈ n_rows for parallel
# beam) must both be divisible by every device count in the ladder; powers of two
# that are multiples of the max device count are safe (configure_sharding raises
# otherwise and the point is skipped).
SIZES = {
    "cpu": [(64, 64, 64), (128, 128, 128), (256, 256, 256), (400, 400, 400)],
    "gpu": [(256, 256, 256), (512, 512, 512), (1024, 1024, 1024)],
}
WARMUP = 1
TRIALS = 3
CORRECTNESS_THRESHOLD = 1e-4

CORRECTNESS_SIZE = (80, 48, 64)   # small, fixed, NON-symmetric (distinct views/
                                  # rows/channels) so shape/axis bugs surface;
                                  # comparison is size-independent
CORRECTNESS_SEED = 1234

# Substrings (upper-cased) that mark a caught failure as memory exhaustion.  The
# pipeline allocates via BOTH the projector / scatter-add path (back projection)
# and cuFFT (the filter), so keep the cuFFT-specific markers as well as the
# generic allocator tokens.
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
    """Deterministic random sinogram of the given shape (numpy float32).

    FBP (filter + adjoint back projection) is linear, so a random sinogram is a
    valid input for both timing and cross-mode/cross-platform correctness
    comparison.
    """
    rng = np.random.default_rng(seed)
    return rng.random(size, dtype=np.float32)


def make_indices(model):
    """Full field-of-view pixel indices for the model (deterministic per size)."""
    import mbirjax
    recon_shape = model.get_params('recon_shape')
    return mbirjax.gen_full_indices(recon_shape, use_ror_mask=model.use_ror_mask)


def run_pipeline(model, sino, pixel_indices):
    """The timed op: the on-device FBP pipeline on a PRE-SHARDED sinogram.

    fbp_filter (sharded view -> sharded view) -> sparse_back_project
    (reduce-scatter -> slice-sharded recon-at-indices).  No entry shard and no
    exit gather, so this measures the combined filter+back-projection compute and
    the combined transient per-device memory (FFT work area + partial buffer),
    not the host boundary transfers that the user-facing direct_recon adds.
    """
    filtered = model.fbp_filter(sino)
    return model.sparse_back_project(filtered, pixel_indices)


def run_direct_recon(model, sino):
    """The full user-facing recon (plain in -> plain out), used for correctness."""
    return model.direct_recon(sino)


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
    # Single-device (no mesh) == the unchanged prerelease filter + back-projection
    # (F2 only added the plain<->sharded boundary, a no-op without a mesh).  We
    # compare the FULL user-facing direct_recon volume — the real end-to-end
    # output — even though the measure path times the internal on-device pipeline.
    model = make_model(CORRECTNESS_SIZE, devices=None)
    sino = make_input(CORRECTNESS_SIZE, seed=CORRECTNESS_SEED)
    beta_out = np.asarray(run_direct_recon(model, sino))

    if ref is None:
        corr = {"baseline_present": False}
        print("[setup] no prerelease baseline; correctness skipped "
              "(run direct_recon_capture_baseline.py from a prerelease checkout)")
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
              "correctness": corr}
    sc.write_worker_result(out_file, result)
    print(f"[setup] platform={plat}  max_devices={max_dev}  ({dev_label})")


def worker_measure(size_label, device_counts, warmup, trials, out_file):
    """Time + measure memory for one size, device counts DESCENDING.

    Descending order (8→4→2→1) makes per-device allocation ascending within this
    fresh process, so the cumulative peak_bytes_in_use equals each config's own
    allocation when read right after it.  Once a config OOMs, every later
    (fewer-device) config needs MORE per-device memory and would OOM too — so we
    catch the OOM, record it, and stop the descent.  Results are written
    incrementally so even a hard crash returns the completed configs.
    """
    # Imported for its IMPORT SIDE EFFECT, not its API (mbirjax is not used by
    # name in this function; make_model imports it again where it's actually
    # called).  mbirjax's first import runs mbirjax._device_setup, which sets
    # XLA_FLAGS=--xla_force_host_platform_device_count from MBIRJAX_NUM_CPU_DEVICES
    # (the orchestrator sets that env for CPU runs).  That flag only takes effect
    # if it is in place BEFORE JAX initializes its backend, and scaling_common
    # imports jax LAZILY — so the first thing that would trigger backend init is
    # sc.pick_devices(...) just below.  Importing mbirjax here, ahead of that,
    # guarantees the virtual CPU device count is established; without it the
    # worker would come up with a single CPU device and every n>1 config would be
    # skipped (so the whole point of the sweep — multi-device scaling — is lost).
    import mbirjax  # noqa: F401  (device-setup side effect; must precede jax init)
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
            # Pre-shard the sinogram and precompute indices OUTSIDE the timing
            # loop: measure the on-device pipeline (filter + reduce-scatter), not
            # the entry scatter or the exit gather.
            sino = model._shard_sinogram(sino_np)
            idx = make_indices(model)
            stats, _ = sc.time_op(lambda: run_pipeline(model, sino, idx),
                                  warmup, trials)
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
    p = argparse.ArgumentParser(description="direct_recon scaling worker (internal)")
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
    # resolves to beta regardless of how the orchestrator was launched.
    # Preallocate the pool up front (no per-call growth → clean timing), with
    # MEM_FRACTION raised so the largest configs don't OOM on the cap;
    # peak_bytes_in_use still tracks in-use tensors, so memory stays accurate.
    beta_root = _beta_root()
    if not os.path.isdir(os.path.join(beta_root, "mbirjax")):
        print(f"  WARNING: no mbirjax/ under derived beta root {beta_root}")
    existing_pp = os.environ.get("PYTHONPATH", "")
    worker_env = {
        "PYTHONPATH": beta_root + (os.pathsep + existing_pp if existing_pp else ""),
        "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.9",
    }

    print("=" * 72)
    print("  direct_recon scaling — isolated-subprocess harness (orchestrator)")
    print(f"  beta root: {beta_root}")
    print("=" * 72)

    # 1. Setup worker: platform, device count/label, correctness.
    setup, rc = sc.run_worker(
        script, ["--worker", "--mode", "setup"], extra_env=worker_env)
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
