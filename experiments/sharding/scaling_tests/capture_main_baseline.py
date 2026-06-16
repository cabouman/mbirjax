"""
experiments/sharding/scaling_tests/capture_main_baseline.py
───────────────────────────────────────────────────────────
Capture the **main-branch, single-device** baseline — the real pre-sharding 1-device reference:
  * time + peak memory per (geometry, op, size) at the FULL sweep sizes -> main_baseline_<plat>.yaml
  * a small `.npy` correctness array per (geometry, op) at one representative size

main has no sharding, so everything is single device.  The sharding branch is checked against this
within floating-point tolerance — NOT bit-exactly (a few lax.map/scatter outliers allowed).  The
engine auto-discovers main_baseline_<plat>.yaml and emits SOFT "vs main (1 device)" notes.

RUN FROM A main WORKTREE so `import mbirjax` resolves to main:

    git worktree add ../mbirjax_main main
    PYTHONPATH=../mbirjax_main python \
        <sharding>/experiments/sharding/scaling_tests/capture_main_baseline.py
    #   -> CONFIRM the printed branch is 'main'
    git worktree remove ../mbirjax_main

Architecture mirrors performance_tracking.py: the orchestrator stays JAX-free and spawns one fresh
worker per cell (so peak memory reads cleanly); workers inherit PYTHONPATH from the invocation, so
they load main.  We do NOT pre-skip sizes we expect to OOM — an OOM is recorded as a failure cell
(single device, so there is no descent).  Capture on CPU for cross-platform determinism, or on a
GPU node for the GPU 1-device numbers.
"""
import os
# Single device on both branches; must precede any mbirjax/jax import.  Workers inherit this.
os.environ.setdefault("MBIRJAX_NUM_CPU_DEVICES", "1")

import sys
import argparse
import traceback

import numpy as np

import scaling_common as sc
import performance_tracking as pt   # module-level is JAX-free; functions import mbirjax lazily


# ── CONFIG (edit here) ────────────────────────────────────────────────────────
# Run from the metrics repo's tooling -> writes <metrics>/golden/ (tracked; review + push).
# Run from the mbirjax checkout -> local results/golden/ (scratch).  REG_GOLDEN_DIR overrides.
GOLDEN_DIR = sc.golden_dir(__file__)
GEOMETRIES = ["parallel", "cone"]
OPS = ["direct_filter", "forward", "back", "vcd_nonconst"]
# Small representative size for the .npy correctness array (~1 MB); the timing/memory sweep uses
# the engine's full per-platform sizes.
NPY_SIZE = (40, 40, 48)


# ── Worker (one cell, single device; inherits PYTHONPATH -> main) ─────────────
def _place(arr):
    import jax
    p = jax.device_put(arr)        # default (single) device
    jax.block_until_ready(p)
    return p


def _op_runner(config, geometry, op, size):
    """Build the model + input single-device and return (run_fn, true_shape)."""
    model = pt.make_model(config, geometry, size)
    idx = pt.make_indices(model)
    sino = pt.make_sinogram(config, size)
    recon_shape = tuple(int(x) for x in model.get_params("recon_shape"))
    if op == "direct_filter":
        s = _place(sino)
        return (lambda: pt.run_filter(model, s)), tuple(size), recon_shape
    if op == "forward":
        cyl = _place(pt.make_cylinders(len(idx), recon_shape[2], config.input_seed))
        return (lambda: pt.run_forward(model, cyl, idx)), tuple(size), recon_shape
    if op == "back":
        s = _place(sino)
        return (lambda: pt.run_back(model, s, idx)), (len(idx), recon_shape[2]), recon_shape
    if op == "vcd_nonconst":
        w = pt.make_weights(config, size)
        parts, seq = pt.build_partitions(model, sino, w, config.vcd_iterations,
                                         seed=config.measure_seed)   # pinned (reproducible)
        model.setup_logger(print_logs=False)
        return (lambda: pt.run_vcd(model, sino, w, parts, seq, config.measure_seed)), \
               tuple(recon_shape), recon_shape
    raise ValueError(f"unknown op {op!r}")


def worker_measure(geometry, op, size_label, save_npy, out_file):
    import mbirjax  # noqa: F401  device-setup-first (main via inherited PYTHONPATH)
    import jax
    config = pt.Config()
    size = pt.parse_size_label(size_label)
    dev = jax.devices()[0]
    base = {"geometry": geometry, "op": op, "size": size_label, "n_devices": 1,
            "platform": dev.platform, "is_sharded": False}
    try:
        run_fn, true_shape, recon_shape = _op_runner(config, geometry, op, size)
        trials = 1 if size_label in config.single_trial_sizes else config.trials_by_op.get(op, 3)
        stats, result = sc.time_op(run_fn, config.warmup, trials)
        mem_mb, mem_kind = sc.peak_memory_mb([dev])
        fp = pt.fingerprint(result, true_shape)
        if save_npy:
            golden = os.environ.get("MAIN_BASELINE_GOLDEN", GOLDEN_DIR)
            os.makedirs(golden, exist_ok=True)
            np.save(os.path.join(golden, f"{geometry}_{op}.npy"), np.asarray(result))
        cell = {**base, "recon_shape": list(recon_shape), "trials": trials,
                **stats, "mem_mb": mem_mb, "fingerprint": fp}
    except Exception as e:   # noqa: BLE001 — single device: no descent, just record the failure
        tb = traceback.format_exc()
        cell = {**base, "failed": True, "oom": sc.is_oom(tb),
                "error": str(e).replace("\n", " ")[:300]}
    sc.write_worker_result(out_file, {"cell": cell})


def worker_setup(out_file):
    import mbirjax
    plat, _ = sc.detect_platform()
    path = os.path.dirname(mbirjax.__file__)
    probe = pt.make_model(pt.Config(), "parallel", NPY_SIZE)
    version = "sharding" if hasattr(probe, "configure_devices") else "main"   # sharding-only API
    sc.write_worker_result(out_file, {"platform": plat, "mbirjax_path": path,
                                      "branch": sc.mbirjax_git_branch(path),
                                      "mbirjax_version": sc.pyproject_version(os.path.dirname(path)),
                                      "version_marker": version, "device_label": sc.device_label()})


def run_worker(argv):
    p = argparse.ArgumentParser(description="main-baseline worker (internal)")
    p.add_argument("--worker", action="store_true")
    p.add_argument("--mode", choices=["setup", "measure"], required=True)
    p.add_argument("--geometry", default=None)
    p.add_argument("--op", default=None)
    p.add_argument("--size", default=None)
    p.add_argument("--save-npy", action="store_true")
    p.add_argument("--out-file", required=True)
    a = p.parse_args(argv)
    if a.mode == "setup":
        worker_setup(a.out_file)
    else:
        worker_measure(a.geometry, a.op, a.size, a.save_npy, a.out_file)


# ── Orchestrator (default; JAX-free) ──────────────────────────────────────────
def main():
    script = os.path.abspath(__file__)
    setup, rc = sc.run_worker(script, ["--worker", "--mode", "setup"], extra_env=None)
    if not setup:
        print(f"  ERROR: setup worker produced no result (rc={rc}); aborting.")
        return
    plat = setup["platform"]
    branch = setup.get("branch")
    print("=" * 72)
    print("  main-branch 1-device baseline")
    print(f"  mbirjax branch: {branch}   {setup.get('mbirjax_path')}")
    print(f"  platform: {plat}   ({setup.get('device_label')})")
    print("  (CONFIRM branch is 'main' — run from a main worktree)")
    print("=" * 72)

    config = pt.Config()
    npy_label = sc.size_label(NPY_SIZE)
    # NPY size first (small/fast + the .npy), then the full sweep sizes (timing/memory only).
    size_labels = [npy_label] + [sc.size_label(s) for s in config.sizes[plat]
                                 if sc.size_label(s) != npy_label]
    os.makedirs(GOLDEN_DIR, exist_ok=True)

    cells = []
    for geometry in GEOMETRIES:
        for op in OPS:
            for label in size_labels:
                args = ["--worker", "--mode", "measure", "--geometry", geometry,
                        "--op", op, "--size", label]
                if label == npy_label:
                    args.append("--save-npy")
                res, rc = sc.run_worker(script, args, extra_env={"MAIN_BASELINE_GOLDEN": GOLDEN_DIR})
                cell = (res or {}).get("cell")
                if not cell:
                    print(f"  {geometry} {op} {label}: no cell (rc={rc})")
                    continue
                cells.append(cell)
                if cell.get("failed"):
                    print(f"  {geometry:8s} {op:13s} {label:14s} "
                          f"[{'OOM' if cell.get('oom') else 'FAIL'}]")
                else:
                    print(f"  {geometry:8s} {op:13s} {label:14s} "
                          f"{cell['min_ms']:9.1f} ms  {cell['mem_mb']:8.0f} MB")

    result = {"kind": "main_baseline", "platform": plat, "branch": branch,
              "mbirjax_version": setup.get("mbirjax_version"),
              "mbirjax_path": setup.get("mbirjax_path"), "device_counts": [1],
              "npy_size": list(NPY_SIZE), "geometries": GEOMETRIES, "ops": OPS,
              "sizes": size_labels, "cells": cells}
    out = os.path.join(GOLDEN_DIR, f"main_baseline_{plat}.yaml")
    sc.save_yaml(out, result)
    nok = sum(1 for c in cells if not c.get("failed"))
    print(f"\nWrote {out}  ({nok} ok, {len(cells) - nok} failed/OOM)")
    print(f"  .npy correctness arrays at {npy_label} -> {GOLDEN_DIR}")
    print("Done.")


if __name__ == "__main__":
    if "--worker" in sys.argv:
        run_worker(sys.argv[1:])
    else:
        main()
