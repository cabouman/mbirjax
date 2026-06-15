"""
experiments/sharding/scaling_tests/capture_main_baseline.py
───────────────────────────────────────────────────────────
Capture the ``.npy`` correctness reference from the **main** branch (no sharding) for the
deep-diff baseline that ``compare_to_baseline`` (deferred) checks the sharding branch against.

main has no sharding, so this captures SINGLE-DEVICE outputs at one small representative size.
The sharding branch is meant to match these within floating-point tolerance — NOT bit-exactly —
with potentially a few outliers from the jax lax.map/scatter rounding bug.

It REUSES ``performance_tracking``'s input builders + run functions so the inputs/ops are
identical to the engine.  Those import mbirjax lazily, so when this runs from a main worktree
with main on PYTHONPATH, they use main's API:

    git worktree add ../mbirjax_main main
    PYTHONPATH=../mbirjax_main python \
        <sharding>/experiments/sharding/scaling_tests/capture_main_baseline.py
    #   -> CONFIRM the printed mbirjax path contains 'mbirjax_main'
    git worktree remove ../mbirjax_main

Single device is forced via MBIRJAX_NUM_CPU_DEVICES=1 before any import (a no-op on main, which
has no sharding), so a smoke run on the sharding branch also captures single-device.  Capture on
CPU for cross-platform determinism (set JAX_PLATFORMS=cpu if a GPU is visible and you want the CPU
reference).  Each op is captured independently: an op whose API differs on main is reported and
skipped, the rest still produce their .npy.
"""
import os

# Single device on both branches; must precede any mbirjax/jax import.
os.environ.setdefault("MBIRJAX_NUM_CPU_DEVICES", "1")

import numpy as np

import scaling_common as sc
import performance_tracking as pt   # module-level is JAX-free; its functions import mbirjax lazily


# ── CONFIG (edit here) ────────────────────────────────────────────────────────
GOLDEN_DIR = os.path.join(sc.RESULTS_DIR, "golden")    # later: <mbirjax_metrics>/golden/
GEOMETRIES = ["parallel", "cone"]
OPS = ["direct_filter", "forward", "back", "vcd_nonconst"]
# One small representative SINOGRAM size (~1 MB outputs).  Single-device, so no padding regardless
# of axis parity.  MUST match what compare_to_baseline re-runs the sharding branch at.
SIZE = (40, 40, 48)


def _run_op(config, geometry, op):
    """Build the model + deterministic input on the loaded mbirjax and run one op single-device."""
    model = pt.make_model(config, geometry, SIZE)
    idx = pt.make_indices(model)
    sino = pt.make_sinogram(config, SIZE)
    recon_shape = tuple(int(x) for x in model.get_params("recon_shape"))
    if op == "direct_filter":
        return pt.run_filter(model, sino)                    # TypeError-falls-back to plain call
    if op == "forward":
        cyl = pt.make_cylinders(len(idx), recon_shape[2], config.input_seed)
        return pt.run_forward(model, cyl, idx)
    if op == "back":
        return pt.run_back(model, sino, idx)
    if op == "vcd_nonconst":
        weights = pt.make_weights(config, SIZE)
        parts, seq = pt.build_partitions(model, sino, weights, config.vcd_iterations)
        model.setup_logger(print_logs=False)
        return pt.run_vcd(model, sino, weights, parts, seq, config.measure_seed)
    raise ValueError(f"unknown op {op!r}")


def main():
    import mbirjax
    path = os.path.dirname(mbirjax.__file__)
    branch = sc.mbirjax_git_branch(path)
    print("=" * 72)
    print(f"  mbirjax loaded from: {path}")
    print(f"  branch: {branch}  (CONFIRM this is 'main' — run from a main worktree)")
    print("=" * 72)

    config = pt.Config()   # default seeds / cone geometry / vcd_iterations
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    captured, failures = [], []
    for geometry in GEOMETRIES:
        for op in OPS:
            try:
                arr = np.asarray(_run_op(config, geometry, op))
                np.save(os.path.join(GOLDEN_DIR, f"{geometry}_{op}.npy"), arr)
                print(f"  {geometry:8s} {op:13s} -> shape {tuple(arr.shape)}  dtype {arr.dtype}")
                captured.append({"geometry": geometry, "op": op,
                                 "shape": list(arr.shape), "dtype": str(arr.dtype)})
            except Exception as e:   # noqa: BLE001 — a main API mismatch shouldn't kill the rest
                msg = str(e).replace("\n", " ")[:200]
                print(f"  {geometry:8s} {op:13s} -> FAILED: {msg}")
                failures.append({"geometry": geometry, "op": op, "error": msg})

    meta = {"kind": "main_baseline", "size": list(SIZE), "input_seed": config.input_seed,
            "weight_seed": config.weight_seed, "measure_seed": config.measure_seed,
            "vcd_iterations": config.vcd_iterations, "cone_sdd_over_channels": config.cone_sdd_over_channels,
            "single_device": True, "mbirjax_path": path, "branch": branch,
            "captured": captured, "failures": failures}
    sc.save_yaml(os.path.join(GOLDEN_DIR, "main_baseline_meta.yaml"), meta)
    print(f"\nWrote {len(captured)} .npy baseline(s) + meta to {GOLDEN_DIR}"
          + (f"  ({len(failures)} op(s) failed — see meta)" if failures else ""))


if __name__ == "__main__":
    main()
