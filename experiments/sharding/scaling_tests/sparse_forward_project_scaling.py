"""
experiments/sharding/scaling_tests/sparse_forward_project_scaling.py
────────────────────────────────────────────────────────────────────
Scaling + correctness driver for sharded forward projection
(TomographyModel.sparse_forward_project) under the view/slice sharding scheme.

Forward projection is the all-gather **adjoint** of back projection's
reduce-scatter: the recon is slice-sharded, each slice-band is broadcast to every
view-owner, and each view-owner forward-projects ITS views from the band (no
reduce).  The timed unit is the INTERNAL ``sparse_forward_project`` on
PRE-SHARDED cylinders, returning the view-sharded sinogram (no gather) — the
analogue of how sparse_back_project is timed on a pre-sharded sinogram.  This
isolates the all-gather + compute and the transient per-view-owner band buffer.

This mirrors sparse_back_project_scaling.py: same isolated-subprocess harness (an
orchestrator that touches no JAX spawns fresh worker subprocesses so per-device
peak memory reads correctly), same two plotted views of one (size × device-count)
grid, same top-of-file run configuration (no CLI args for the human).  It is
simpler than the back driver: forward has a single sharded path (no band-vs-pixel
variants), so there is one YAML + two plots and no comparison table.

Correctness here is the **forward/back adjoint identity** ``<A x, y> == <x, A^T y>``
computed single-device on a small fixed size — self-contained (no captured
baseline needed) and exactly the gate that certifies the forward projector is the
adjoint of the validated back projector.

  - orchestrator (default, no args)  : spawns workers per size, collects
                                       YAMLs/plots.
  - worker --mode setup              : reports platform/devices + the adjoint
                                       identity correctness check.
  - worker --mode measure --size ... : times + measures memory for one size,
                                       device counts DESCENDING so per-device
                                       allocation is ascending in the fresh
                                       process and peak reads correctly.

Run from the BETA worktree root:

    python experiments/sharding/scaling_tests/sparse_forward_project_scaling.py
"""
import os
import gc
import sys
import argparse

import scaling_common as sc

import numpy as np

# NOTE: mbirjax (and therefore jax) is imported INSIDE the worker functions, not
# at the top level, on purpose -- see the long note in sparse_back_project_scaling.py.
# The orchestrator must stay JAX-free so it holds no device backend while a worker
# measures peak_bytes_in_use; the worker imports mbirjax for its device-setup side
# effect before any jax backend init.


OP_NAME = "sparse_forward_project"

# ── Run configuration (edit here; no CLI args for the human) ──────────────────
# Device-count ladder.  None → the automatic powers-of-two ladder.  Set
# explicitly to override — e.g. [1, 2, 3] to skip a known-bad 4th card.
DEVICE_COUNTS = [1, 2, 4, 8]  # [1, 2, 3]

# Problem sizes (n_views, n_rows, n_channels).  Same ladder as back projection:
# the view axis (n_views) and the slice axis (≈ n_rows for parallel beam) must
# both divide every device count used, else configure_sharding raises and the
# point is skipped.  Multiples of 12 admit a 3-device ladder (cooler first three
# GPUs); multiples of 16 are the power-of-two sizes.
SIZES_BY_12 = {
    "cpu": [(64, 64, 64), (128, 128, 128), (256, 256, 256), (400, 400, 400)],
    "gpu": [(252, 252, 252), (504, 504, 504), (1008, 1008, 1008)],
}
SIZES_BY_16 = {
    "cpu": [(64, 64, 64), (128, 128, 128), (256, 256, 256), (400, 400, 400)],
    "gpu": [(256, 256, 256), (512, 512, 512), (1024, 1024, 1024)],
}

SIZES = SIZES_BY_12 if 3 in DEVICE_COUNTS else SIZES_BY_16

WARMUP = 1
TRIALS = 3
CORRECTNESS_THRESHOLD = 1e-4   # rel error on the adjoint inner-product identity

CORRECTNESS_SIZE = (80, 48, 64)   # small, fixed, NON-symmetric (distinct views/
                                  # rows/channels) so shape/axis bugs surface
CORRECTNESS_SEED = 1234

# Substrings (upper-cased) that mark a caught failure as memory exhaustion.
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


def make_indices(model):
    """Full field-of-view pixel indices for the model (deterministic per size)."""
    import mbirjax
    recon_shape = model.get_params('recon_shape')
    return mbirjax.gen_full_indices(recon_shape, use_ror_mask=model.get_params('use_ror_mask'))


def make_cylinders(num_pixels, num_slices, seed=0):
    """Deterministic random recon cylinders (num_pixels, num_slices) float32.

    Forward projection is linear, so random cylinders are a valid input for both
    timing and the adjoint-identity correctness check.
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal((num_pixels, num_slices), dtype=np.float32)


def run_forward_project(model, cylinders, pixel_indices):
    """The timed op: all-gather forward projection of (pre-sharded) cylinders.

    Returns the view-sharded sinogram; no gather, so this measures the all-gather
    + compute, not the host transfer.
    """
    return model.sparse_forward_project(cylinders, pixel_indices)


def parse_size_label(label):
    """'256x256x256' -> (256, 256, 256)."""
    return tuple(int(x) for x in label.split("x"))


def _adjoint_rel_error(model, idx):
    """<A x, y> vs <x, A^T y> single-device; returns (lhs, rhs, rel_error).

    A = forward (sparse_forward_project), A^T = back (sparse_back_project).  Random
    cylinders x and sinogram y; the inner-product identity must hold to float
    precision, which certifies forward and back are exact adjoints.
    """
    num_slices = model.get_params('recon_shape')[2]
    num_pixels = len(idx)
    rng = np.random.default_rng(CORRECTNESS_SEED)
    x_cyl = rng.standard_normal((num_pixels, num_slices), dtype=np.float32)
    y_sino = rng.standard_normal(model.get_params('sinogram_shape'), dtype=np.float32)
    ax = np.asarray(run_forward_project(model, x_cyl, idx))
    aty = np.asarray(model.sparse_back_project(y_sino, idx))
    lhs = float(np.sum(ax * y_sino))
    rhs = float(np.sum(x_cyl * aty))
    rel = abs(lhs - rhs) / (abs(lhs) + 1e-30)
    return lhs, rhs, rel


# ── Worker side (runs in an isolated subprocess) ──────────────────────────────
def worker_setup(out_file):
    """Report platform, device count/label, and the adjoint-identity correctness
    check.  One launch, before sizes are known."""
    import mbirjax  # device-setup-first: before any jax import (via sc below)
    plat, max_dev = sc.detect_platform()
    dev_label = sc.device_label()

    # Single-device (no mesh) forward/back adjoint identity -- the forward
    # correctness gate, self-contained (no captured baseline needed).
    model = make_model(CORRECTNESS_SIZE, devices=None)
    idx = make_indices(model)
    lhs, rhs, rel = _adjoint_rel_error(model, idx)
    corr = {"baseline_present": True, "check": "adjoint_identity",
            "lhs": lhs, "rhs": rhs, "rel_error": rel,
            "max_abs_diff": abs(lhs - rhs),
            "pct_above_threshold": 0.0 if rel <= CORRECTNESS_THRESHOLD else 100.0,
            "cross_platform": False, "threshold": CORRECTNESS_THRESHOLD}
    print(f"[setup] adjoint identity <Ax,y>={lhs:.6f}  <x,A^Ty>={rhs:.6f}  "
          f"rel={rel:.3e}" + ("" if rel <= CORRECTNESS_THRESHOLD else "  <-- ABOVE THRESHOLD"))

    # Record physical GPUs / interconnect / NUMA and the host-bounce probe.
    topology = sc.gpu_topology() if plat == "gpu" else {}
    dev2dev_safe = None
    if plat == "gpu":
        try:
            import jax
            import mbirjax._sharding as mjs
            gpus = jax.devices("gpu")
            if len(gpus) > 1:
                dev2dev_safe = bool(mjs.is_dev2dev_safe(gpus))
        except Exception:   # noqa: BLE001 — best effort, never abort setup
            pass

    pkg_path = os.path.dirname(mbirjax.__file__)
    beta_state, branch = sc.beta_status(pkg_path)   # by git branch, not dir name
    result = {"platform": plat, "max_devices": max_dev, "device_label": dev_label,
              "mbirjax_path": pkg_path, "beta_state": beta_state, "branch": branch,
              "correctness": corr, "topology": topology, "dev2dev_safe": dev2dev_safe}
    sc.write_worker_result(out_file, result)
    print(f"[setup] platform={plat}  max_devices={max_dev}  ({dev_label})")
    if dev2dev_safe is not None:
        print(f"[setup] dev2dev_safe={dev2dev_safe}"
              + ("" if dev2dev_safe else "  <-- HOST-BOUNCE active (slow d2d!)"))
    if topology.get("devices"):
        print("[setup] GPUs:\n    " + topology["devices"].replace("\n", "\n    "))


def worker_measure(size_label, device_counts, warmup, trials, out_file):
    """Time + measure memory for one size, device counts DESCENDING.

    After each timed config a per-GPU clock/temp sample is recorded and a loud
    warning printed if any GPU was thermally throttling.  Descending order
    (8→4→2→1) makes per-device allocation ascending within this fresh process, so
    the cumulative peak_bytes_in_use equals each config's own allocation when read
    right after it.  An OOM stops the descent (fewer-device configs need more
    per-device memory).  Results are written incrementally.
    """
    import mbirjax  # noqa: F401  (device-setup side effect; must precede jax init)
    size = parse_size_label(size_label)
    desc = sorted(set(device_counts), reverse=True)
    print(f"\n[measure {size_label}]  device counts (descending): {desc}")

    # The cylinder input is the same for every device count of this size; build it
    # once from a throwaway no-device model (shapes only; outside the timing loop).
    ref_model = make_model(size, devices=None)
    idx_np = np.asarray(make_indices(ref_model))
    num_slices = ref_model.get_params('recon_shape')[2]
    cyl_np = make_cylinders(len(idx_np), num_slices, seed=0)
    del ref_model
    gc.collect()

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
            # Pre-shard the cylinders and precompute indices OUTSIDE the timing
            # loop: measure the all-gather + compute, not the entry scatter.
            cyl = model._shard_recon(cyl_np)
            idx = make_indices(model)
            stats, _ = sc.time_op(lambda: run_forward_project(model, cyl, idx),
                                  warmup, trials)
            mem_mb, mem_kind = sc.peak_memory_mb(devs)
            gpu_state = sc.sample_gpu_state()
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
        hot = sc.throttled_gpus(gpu_state)
        rows.append({"n_devices": n, **stats, "mem_mb": mem_mb,
                     "gpu_state": gpu_state, "throttled": bool(hot)})
        print(f"  n_devices={n:2d}  min={stats['min_ms']:8.2f} ms  "
              f"mean={stats['mean_ms']:8.2f} ms  mem={mem_mb:8.1f} MB ({mem_kind})")
        if hot:
            print("  !! THROTTLING — this timing is UNRELIABLE: "
                  + ", ".join(f"GPU{g['index']}={g['sm_mhz']}MHz@{g['temp_c']}C"
                              for g in hot))
        # Publish partial progress and free this config before the next (larger)
        # one so peak_bytes_in_use reflects each config alone.
        _publish()
        del model, cyl
        gc.collect()
    _publish()


def run_worker(argv):
    """Dispatch a --worker invocation (internal; the orchestrator builds argv)."""
    p = argparse.ArgumentParser(description="sparse_forward_project scaling worker (internal)")
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
    """Beta worktree root: three directories up from this file's directory."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__),
                                        os.pardir, os.pardir, os.pardir))


def main():
    script = os.path.abspath(__file__)

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
    print("  sparse_forward_project scaling — isolated-subprocess harness (orchestrator)")
    print(f"  beta root: {beta_root}")
    print("=" * 72)

    # 1. Setup worker: platform, device count/label, adjoint-identity correctness.
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
    topology = setup.get("topology") or {}
    dev2dev_safe = setup.get("dev2dev_safe")
    if beta_state == "beta":
        label = f"*** beta ***  (branch {branch})"
    elif beta_state == "not-beta":
        label = f"### NOT beta — branch {branch} — check PYTHONPATH ###"
    else:
        label = "(branch undetermined — verify path manually)"
    print(f"  mbirjax: {label}   {mpath}")
    print(f"  platform: {plat}   max devices: {max_dev}   ({dev_label})")
    print(f"  correctness (adjoint identity): rel_error={corr['rel_error']:.3e}  "
          f"max_abs_diff={corr['max_abs_diff']:.3e}"
          + ("" if corr['pct_above_threshold'] == 0.0 else "   <-- ABOVE THRESHOLD"))
    if dev2dev_safe is not None:
        print(f"  dev2dev_safe: {dev2dev_safe}"
              + ("" if dev2dev_safe else "  <-- HOST-BOUNCE active (slow d2d!)"))
    if topology.get("topo"):
        print("  GPU topology (nvidia-smi topo -m):")
        print("    " + topology["topo"].replace("\n", "\n    "))

    sizes = SIZES[plat]
    size_labels = [sc.size_label(s) for s in sizes]
    device_counts = [n for n in (DEVICE_COUNTS or sc.default_device_counts(max_dev))
                     if n <= max_dev]
    if plat == "cpu":
        worker_env["MBIRJAX_NUM_CPU_DEVICES"] = str(max_dev)
    print(f"  sizes: {size_labels}")
    print(f"  device counts: {device_counts}")

    # 2. One fresh worker per size (clean peak memory); collect the grid.
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
        rows.sort(key=lambda r: r["n_devices"])
        grid[label] = rows
        by_n = {r["n_devices"]: r for r in rows}
        summary = "  ".join(f"{n}d={by_n[n]['speedup']:.2f}x" for n in sorted(by_n))
        print(f"  size {label} speedup: {summary}")

    # 3. Persist results (one YAML) and two plots.
    results = {
        "op": OP_NAME, "platform": plat, "device_label": dev_label,
        "mbirjax_path": mpath, "warmup": WARMUP, "trials": TRIALS,
        "device_counts": device_counts, "sizes": size_labels,
        "mem_kind": mem_kind, "correctness": corr,
        "dev2dev_safe": dev2dev_safe, "topology": topology,
        "grid": grid, "failures": failures_by_size,
    }
    sc.save_yaml(os.path.join(sc.RESULTS_DIR, f"{OP_NAME}_{plat}.yaml"), results)
    if any(grid.values()):
        title = OP_NAME
        sc.plot_device_sweep(
            title, grid, device_counts, size_labels, dev_label, mem_kind,
            os.path.join(sc.RESULTS_DIR, f"{OP_NAME}_{plat}_device_sweep.png"))
        sc.plot_size_sweep(
            title, grid, device_counts, size_labels, dev_label, mem_kind,
            os.path.join(sc.RESULTS_DIR, f"{OP_NAME}_{plat}_size_sweep.png"))

    print("\nDone.")


if __name__ == "__main__":
    if "--worker" in sys.argv:
        run_worker(sys.argv[1:])
    else:
        main()
