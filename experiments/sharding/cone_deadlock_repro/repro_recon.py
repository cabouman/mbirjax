#!/usr/bin/env python
"""Minimal cone-vs-parallel sharded-recon repro for the multi-GPU NCCL "clique" deadlock.

Run ONE (geometry, size) configuration per process.  The number of devices is set by the
driver via ``CUDA_VISIBLE_DEVICES`` (so MBIRJAX's automatic selection shards across exactly
those GPUs -- the same path as a real run), not via an in-process flag.  The script prints
timestamped progress markers that mirror the production log
("Starting VCD iterations", ...), so a hang is pinned to the exact phase.  It is meant to be
wrapped in ``timeout`` by run_sweep.sh: a deadlocked config is killed and recorded, the rest
keep going.

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python repro_recon.py --geometry cone --size 256 --iterations 3
"""
import argparse
import sys
import time

_T0 = time.time()


def log(msg):
    print("[{:7.1f}s] {}".format(time.time() - _T0, msg), flush=True)


def build_model(geometry, sino_shape, num_views):
    import numpy as np
    import mbirjax as mj
    if geometry == "parallel":
        angles = np.linspace(0, np.pi, num_views, endpoint=False)
        return mj.ParallelBeamModel(sino_shape, angles)
    # cone: pick a benign magnification ~2 (source-to-detector = 2 x source-to-iso)
    angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)
    source_iso_dist = 4.0 * sino_shape[2]
    source_detector_dist = 2.0 * source_iso_dist
    return mj.ConeBeamModel(sino_shape, angles,
                            source_detector_dist=source_detector_dist,
                            source_iso_dist=source_iso_dist)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", choices=["parallel", "cone"], required=True)
    ap.add_argument("--size", type=int, default=256,
                    help="cubic problem: views = det_rows = det_channels = recon_rows = ... = size")
    ap.add_argument("--views", type=int, default=None, help="override number of views (default = size)")
    ap.add_argument("--iterations", type=int, default=3,
                    help="VCD max_iterations (>=1 reaches the first subset update where the hang occurred)")
    args = ap.parse_args()

    N = args.size
    num_views = args.views or N
    sino_shape = (num_views, N, N)  # (views, det_rows, det_channels)

    log("importing jax + mbirjax (geometry={}, size={}, views={})".format(args.geometry, N, num_views))
    import numpy as np
    import jax
    import mbirjax as mj
    log("mbirjax {} ; jax backend sees {} device(s): {}".format(
        getattr(mj, "__version__", "?"), len(jax.devices()), jax.devices()))

    log("building {} model, sino_shape={}".format(args.geometry, sino_shape))
    model = build_model(args.geometry, sino_shape, num_views)
    model.set_params(verbose=1)
    log("model built; device_summary = {}".format(model.device_summary))

    # A valid sinogram: forward-project a phantom (fall back to a random volume if needed).
    try:
        log("generating phantom")
        phantom = model.gen_modified_3d_sl_phantom()
    except Exception as e:  # noqa: BLE001 - any phantom issue, just use random data
        log("phantom generation failed ({}); using a random recon-shaped volume".format(e))
        recon_shape = model.get_params("recon_shape")
        phantom = np.asarray(np.random.RandomState(0).rand(*recon_shape), dtype=np.float32)
    log("forward-projecting phantom -> sinogram")
    sinogram = model.forward_project(phantom)
    jax.block_until_ready(sinogram)
    log("sinogram ready, shape {}".format(tuple(sinogram.shape)))

    log("=== starting recon (max_iterations={}) -- the hang occurred at the first VCD subset update ==="
        .format(args.iterations))
    t_recon = time.time()
    out = model.recon(sinogram, max_iterations=args.iterations)
    recon = out[0] if isinstance(out, (tuple, list)) else out
    jax.block_until_ready(recon)
    log("recon COMPLETED in {:.1f}s; recon shape {}".format(time.time() - t_recon, tuple(recon.shape)))
    log("SUCCESS")


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - surface the full traceback in the per-config log
        import traceback
        log("FAILED with exception:")
        traceback.print_exc()
        sys.exit(1)
