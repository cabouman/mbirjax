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

    # Build the input WITHOUT ever materializing a full single-device array.  At 2048^3 a phantom is
    # 32 GiB; both gen_modified_3d_sl_phantom (analytic ellipsoids) and host np.random are slow there
    # and spike memory.  sharded_full builds a constant volume PER SHARD directly on each device
    # (~4 GiB/device at 2048^3/8, no 32 GiB transient).  The content is irrelevant here -- memory and
    # the collective structure are shape-driven, and a uniform block still forward-projects to a
    # non-trivial sinogram that drives the full VCD loop.
    import mbirjax._sharding as mjs
    recon_shape = model.get_params("recon_shape")
    log("building a sharded constant phantom (per-shard, no single-device transient)")
    try:
        phantom = mjs.sharded_full(model.recon_placement, recon_shape, 1.0)
    except Exception as e:  # noqa: BLE001 - fall back to a single-device phantom (fine at small sizes)
        log("sharded_full failed ({}); falling back to a single-device phantom".format(e))
        phantom = model.gen_modified_3d_sl_phantom()
    # Keep everything DEVICE-SHARDED: output_sharded=True avoids gathering the full sinogram (and
    # later the recon) onto a single device.  At large sizes a gathered array is huge (a 2048^3
    # float32 sinogram is 32 GiB) and would OOM one GPU -- which is a property of the gather, not of
    # the sharded recon we are trying to exercise.
    log("forward-projecting phantom -> sinogram (device-sharded, no host gather)")
    sinogram = model.forward_project(phantom, output_sharded=True)
    jax.block_until_ready(sinogram)
    del phantom   # free the single-device phantom (~32 GiB at 2048^3) before recon
    log("sinogram ready, shape {}".format(tuple(sinogram.shape)))

    log("=== starting recon (max_iterations={}, output_sharded) -- the hang occurred at the first "
        "VCD subset update ===".format(args.iterations))
    t_recon = time.time()
    out = model.recon(sinogram, max_iterations=args.iterations, output_sharded=True)
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
