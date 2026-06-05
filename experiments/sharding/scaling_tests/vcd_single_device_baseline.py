"""
experiments/sharding/scaling_tests/vcd_single_device_baseline.py
────────────────────────────────────────────────────────────────
Single-device (NO sharding configured) VCD ``recon`` time + peak-memory baseline,
for the "don't regress the 1-GPU path" check.

WHY: beta's no-mesh single-device path is supposed to be the verbatim prerelease
body, so a normal single-GPU user should be unaffected by the sharding work.  This
script measures the *user-facing* ``recon`` on one device at large sizes on BOTH
checkouts so we can confirm beta-no-mesh matches prerelease (no OOM, similar peak /
time) before we touch the 1-device *mesh* (sharded-path) memory.

It deliberately does NOT call ``configure_sharding`` — this is the plain
single-device path (``mesh is None``), exactly what prerelease has and what a normal
single-GPU user runs.  So the SAME script runs on prerelease and on beta; the banner
prints which mbirjax is loaded, and the output YAML is named by branch state.

Lean by design (big recons are slow): ONE run per size (no warmup/trials sweep —
peak memory is set by the early full-FOV ops + per-subset peak, not by iteration
count, and time here includes compile, which is fine for a regression comparison).
Each size is wrapped so a 1008³ OOM still records 504³ and logs the real traceback
(an OOM often surfaces as an unrelated error — we classify from the full stack).

Run on each checkout (from that checkout's root, or with PYTHONPATH set to it):

    # prerelease checkout:
    python <beta>/experiments/sharding/scaling_tests/vcd_single_device_baseline.py
    # beta checkout:
    python experiments/sharding/scaling_tests/vcd_single_device_baseline.py

Then compare results/vcd_single_device_<branchstate>.yaml across the two.
"""
import os
import gc
import time
import traceback

# Import mbirjax before jax (device-setup-first); this script is the process that
# touches JAX (not a JAX-free orchestrator), so a top-level import is correct here.
import mbirjax

import numpy as np

import scaling_common as sc


# ── Run configuration (edit here; no CLI args) ────────────────────────────────
# (n_views, n_rows, n_channels).  Ascending so each size's peak reads cleanly after
# the previous is freed.  504/1008 are the non-power-of-2 sizes used elsewhere.
SIZES = [(504, 504, 504), (1008, 1008, 1008)]
MAX_ITERATIONS = 5          # peak memory is reached early; a few iters suffice
SEED = 0
SHARPNESS = 1.0

_OOM_MARKERS = ("RESOURCE_EXHAUSTED", "OUT OF MEMORY", "OOM", "BAD_ALLOC",
                "FAILED TO ALLOCATE", "WORK AREA", "SCRATCH ALLOCATOR",
                "FAILED TO CREATE CUFFT")


def run_one(size):
    """Single-device recon of a fixed-seed random sinogram; return time + peak mem.

    No configure_sharding -> the plain single-device (mesh is None) path.  A random
    sinogram is fine: we are measuring the recon's time/peak (shape-driven), not
    accuracy.
    """
    n_views, n_rows, n_channels = size
    angles = np.linspace(0, np.pi, n_views, endpoint=False)
    model = mbirjax.ParallelBeamModel(size, angles)
    model.set_params(sharpness=SHARPNESS, verbose=0)

    devs = sc.pick_devices(1)                     # the single device memory is read on
    np.random.seed(SEED)
    sino = np.random.rand(*size).astype(np.float32)   # host array; recon shards nothing

    t0 = time.perf_counter()
    recon, _ = model.recon(sino, weights=None, max_iterations=MAX_ITERATIONS,
                           stop_threshold_change_pct=0.0, print_logs=False)
    import jax
    jax.block_until_ready(recon)
    elapsed_ms = (time.perf_counter() - t0) * 1e3

    mem_mb, mem_kind = sc.peak_memory_mb(devs if devs else [])
    del recon, sino
    gc.collect()
    return {"size": sc.size_label(size), "time_ms": elapsed_ms,
            "mem_mb": mem_mb, "mem_kind": mem_kind, "max_iterations": MAX_ITERATIONS}


def main():
    path = sc.which_mbirjax()                     # prints beta/not-beta banner + path
    plat, _ = sc.detect_platform()
    state, branch = sc.beta_status(path)
    print(f"  platform={plat}  branch={branch}  ({state})")
    print(f"  single-device recon (NO sharding) — sizes {[sc.size_label(s) for s in SIZES]}, "
          f"{MAX_ITERATIONS} iters, 1 run each")

    rows, failures = [], []
    for size in SIZES:
        label = sc.size_label(size)
        try:
            r = run_one(size)
            rows.append(r)
            print(f"  {label}:  time={r['time_ms']:9.0f} ms   peak={r['mem_mb']:9.1f} MB "
                  f"({r['mem_kind']})")
        except Exception as e:   # noqa: BLE001 — record, don't abort the sweep
            tb = traceback.format_exc()
            is_oom = any(k in tb.upper() for k in _OOM_MARKERS)
            failures.append({"size": label, "oom": is_oom,
                             "error": str(e).replace("\n", " ")[:300], "traceback": tb})
            print(f"  {label}:  {'OOM' if is_oom else 'ERROR'}: {str(e)[:120]}")
            if not is_oom:
                print(tb)
            # Larger sizes would also fail; stop here.
            break

    out = {"op": "vcd_single_device", "platform": plat, "branch": branch,
           "beta_state": state, "mbirjax_path": path, "max_iterations": MAX_ITERATIONS,
           "sizes": [sc.size_label(s) for s in SIZES], "rows": rows, "failures": failures}
    tag = state if state in ("beta", "not-beta") else "unknown"
    sc.save_yaml(os.path.join(sc.RESULTS_DIR, f"vcd_single_device_{tag}_{plat}.yaml"), out)
    print("\nDone.  Compare the beta vs prerelease YAMLs for time/peak parity (and no OOM).")


if __name__ == "__main__":
    main()
