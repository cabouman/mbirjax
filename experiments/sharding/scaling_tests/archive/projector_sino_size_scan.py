"""
experiments/sharding/scaling_tests/projector_sino_size_scan.py
───────────────────────────────────────────────────────────────
Single-device projector timing scan vs sinogram size.

Purpose: characterize forward and back projection time as each sinogram axis is
varied, to verify the **channel-major layout fix** (parallel_beam kernels) removed
the power-of-2 ``num_det_channels`` cache-aliasing penalty -- and to check it on
GPU as well as CPU.  Before the fix, a power-of-2 channel count ran several times
slower on CPU at large slice counts (the ``sinogram_view[:, n]`` column stride
aliased the cache); after the fix the time should be **flat** across power-of-2 vs
non-power-of-2 at the same work.

It times, on a **single device** (no mesh), four sweeps:
  - views    varied alone (rows, channels at BASE),
  - rows     varied alone (views, channels at BASE),
  - channels varied alone (views, rows at BASE)   <- the axis the fix targets,
  - uniform  (n, n, n),
each mixing powers of two (128/256/512/1024/2048) and non-powers of two
(192/252/384/504/1008/2016 ...).

A/B against the pre-fix kernel: ``git stash`` the parallel_beam.py layout change,
re-run, and diff the two YAMLs (the new one should be flat where the old spikes at
power-of-2 channels).

Auto-detects platform (GPU if present, else CPU) and uses the matching preset; the
GPU preset is meant to be scaled to the machine.  Run from the beta worktree root:

    python experiments/sharding/scaling_tests/projector_sino_size_scan.py
"""
import os
import gc
import time

import numpy as np
import yaml

# ── Run configuration (edit here) ─────────────────────────────────────────────
WARMUP = 1
TRIALS = 2

# Non-swept axes are held at BASE (kept modest, and non-power-of-2, so they don't
# confound the swept axis).  (views, rows, channels).
BASE = {
    "cpu": (48, 48, 96),
    "gpu": (192, 192, 384),
}

# Values the swept axis takes (mix of powers of two and non-powers of two).  Cost
# of forward/back ~ channels^2 * views * rows, so channel sweeps are the heaviest;
# the GPU lists go larger (real detectors are 1024/2048).  Tune the GPU lists to
# the machine.
SWEEPS = {
    "cpu": {
        "views":    [48, 64, 96, 128, 192, 256, 384, 512],
        "rows":     [48, 64, 96, 128, 192, 256, 384, 512],
        "channels": [64, 96, 128, 192, 252, 256, 384, 504, 512],
    },
    "gpu": {
        "views":    [192, 256, 384, 512, 768, 1024, 1536, 2048],
        "rows":     [192, 256, 384, 512, 768, 1024, 1536, 2048],
        "channels": [192, 256, 384, 512, 1008, 1024, 2016, 2048],
    },
}

# Uniform (n, n, n) scaling.  Cost ~ n^4, so keep the upper end modest on CPU.
UNIFORM = {
    "cpu": [64, 96, 128, 192, 252, 256, 320],
    "gpu": [256, 384, 512, 768, 1008, 1024],
}

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


# ── Timing ────────────────────────────────────────────────────────────────────
def _is_pow2(n):
    return (n & (n - 1)) == 0


def time_config(size, seed=0):
    """Single-device forward + back min time (ms) for one (views, rows, channels)."""
    import mbirjax
    import jax
    import jax.numpy as jnp
    nv, nr, nc = size
    angles = jnp.linspace(0, jnp.pi, nv, endpoint=False)
    model = mbirjax.ParallelBeamModel((nv, nr, nc), angles)
    idx = mbirjax.gen_full_indices(model.get_params('recon_shape'),
                                   use_ror_mask=model.get_params('use_ror_mask'))
    num_slices = model.get_params('recon_shape')[2]
    cyl = jnp.asarray(np.random.default_rng(seed).standard_normal(
        (len(idx), num_slices), dtype=np.float32))
    sino = jnp.asarray(np.random.default_rng(seed + 1).standard_normal(
        (nv, nr, nc), dtype=np.float32))

    def best(fn):
        out = fn(); jax.block_until_ready(out)             # warmup / compile
        ts = []
        for _ in range(TRIALS):
            t0 = time.perf_counter()
            out = fn(); jax.block_until_ready(out)
            ts.append((time.perf_counter() - t0) * 1000)
        return min(ts)

    fwd = best(lambda: model.sparse_forward_project(cyl, idx))
    bp = best(lambda: model.sparse_back_project(sino, idx))
    npix = len(idx)
    del model, cyl, sino, idx
    gc.collect()
    return fwd, bp, npix


def build_configs(plat):
    """List of (sweep_name, swept_value, size) covering the four sweeps."""
    base = BASE[plat]
    cfgs = []
    for name, axis in (("views", 0), ("rows", 1), ("channels", 2)):
        for v in SWEEPS[plat][name]:
            size = list(base); size[axis] = v
            cfgs.append((name, v, tuple(size)))
    for n in UNIFORM[plat]:
        cfgs.append(("uniform", n, (n, n, n)))
    return cfgs


def main():
    # Import mbirjax first (device-setup side effect), then probe the platform.
    import mbirjax  # noqa: F401
    import jax
    try:
        plat = "gpu" if jax.devices("gpu") else "cpu"
    except RuntimeError:
        plat = "cpu"
    dev = jax.devices()[0]
    print("=" * 78)
    print(f"  projector sino-size scan  —  platform={plat}  device={dev}")
    print(f"  BASE={BASE[plat]}  WARMUP={WARMUP} TRIALS={TRIALS}")
    print("=" * 78)

    configs = build_configs(plat)
    rows = []
    current_group = None
    for name, val, size in configs:
        if name != current_group:
            current_group = name
            held = {"views": f"rows={BASE[plat][1]}, channels={BASE[plat][2]}",
                    "rows": f"views={BASE[plat][0]}, channels={BASE[plat][2]}",
                    "channels": f"views={BASE[plat][0]}, rows={BASE[plat][1]}",
                    "uniform": "(n,n,n)"}[name]
            print(f"\n== {name} sweep  ({held}) ==")
        try:
            fwd, bp, npix = time_config(size)
        except Exception as e:   # noqa: BLE001 — keep the scan going past an OOM
            print(f"  {name}={val:<5} size={size}  ERROR: {str(e)[:80]}")
            rows.append({"sweep": name, "value": val, "size": list(size),
                         "error": str(e)[:200]})
            continue
        nv, nr, nch = size
        flags = (f"ch_pow2={_is_pow2(nch)!s:<5}"
                 f" v_pow2={_is_pow2(nv)!s:<5} r_pow2={_is_pow2(nr)!s:<5}")
        print(f"  {name}={val:<5} size=({nv},{nr},{nch})  "
              f"fwd={fwd:9.1f} ms  back={bp:9.1f} ms  npix={npix:>8}  {flags}")
        rows.append({"sweep": name, "value": val, "size": list(size),
                     "forward_ms": fwd, "back_ms": bp, "num_pixels": npix,
                     "channels_pow2": _is_pow2(nch),
                     "views_pow2": _is_pow2(nv), "rows_pow2": _is_pow2(nr)})

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"projector_sino_size_scan_{plat}.yaml")
    with open(out, "w") as f:
        yaml.safe_dump({"platform": plat, "device": str(dev),
                        "base": list(BASE[plat]), "warmup": WARMUP,
                        "trials": TRIALS, "rows": rows}, f, sort_keys=False)
    print(f"\nWrote {out}")
    print("Done.")


if __name__ == "__main__":
    main()
