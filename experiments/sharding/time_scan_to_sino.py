#!/usr/bin/env python
"""AD HOC / EPHEMERAL -- isolate the scan_to_sino (sharded compute) wall time from the fixed
radiograph read, to measure the real multi-device speedup of the preprocessing main loop.

`bash time` on collect_nsi_golden.py / Lilly_recon.py measures the WHOLE script (Samba read +
preprocess + save), so the sharded-compute speedup is masked by the fixed read.  This loads+crops
ONCE, then times scan_to_sino alone at 1 device vs all visible devices (with warm-up + a
byte-identical check).

Run on the cluster.  Control the device count with CUDA_VISIBLE_DEVICES, and prefer a realistic
config (low downsample, many views) where compute -- not the per-TIFF read -- dominates:

  CUDA_VISIBLE_DEVICES=0          python time_scan_to_sino.py --data_path ... --downsampling 1 --subsample_view_factor 1
  CUDA_VISIBLE_DEVICES=0,1,2,3    python time_scan_to_sino.py --data_path ... --downsampling 1 --subsample_view_factor 1
"""
import argparse
import time

import numpy as np
import jax
import mbirjax.preprocess as mjp
from mbirjax.preprocess import nsi, utilities as U


def _time(fn, repeats=3):
    """Best-of-`repeats` wall time (after a warm-up call), plus the result of the first timed run."""
    fn()  # warm up: compile put_in_slice / dm_pix.rotate, prime transfers
    best, out = float("inf"), None
    for _ in range(repeats):
        t = time.time()
        out = fn()
        np.asarray(out)  # force the gather so the timing includes the device->host transfer
        best = min(best, time.time() - t)
    return best, out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--downsampling", type=int, default=1)
    p.add_argument("--subsample_view_factor", type=int, default=1)
    args = p.parse_args()
    ds = (args.downsampling, args.downsampling)

    print("visible devices:", jax.devices())

    # ---- Fixed cost: load + crop ONCE (NOT timed as part of the sharded compute) ----
    t0 = time.time()
    obj, blank, dark, nsi_params, defective = nsi.load_scans_and_params(
        args.data_path, subsample_view_factor=args.subsample_view_factor, verbose=0)
    crop = nsi_params["max_crop"]
    cone_beam_params, optional_params = nsi.convert_nsi_to_mbirjax_params(
        nsi_params, downsample_factor=ds, crop_pixels_sides=crop, crop_pixels_top=crop, crop_pixels_bottom=crop)
    obj, blank, dark, defective = mjp.crop_view_data(
        obj, blank, dark, crop_pixels_sides=crop, crop_pixels_top=crop, crop_pixels_bottom=crop,
        defective_pixel_array=defective)
    det_rotation = optional_params["det_rotation"]
    print(f"load+crop (fixed, not the sharded part): {time.time() - t0:.2f}s ; obj_scan {obj.shape}")

    devs = jax.devices()

    def run(devices):
        # blank/dark are modified in place by downsampling, so hand each run fresh copies; obj is read-only.
        return U.scan_to_sino(obj, blank.copy(), dark.copy(), defective,
                              downsample_factor=ds, det_rotation=det_rotation, devices=devices)

    t1, s1 = _time(lambda: run(devs[:1]))
    print(f"scan_to_sino  1 device : {t1:.3f}s")
    if len(devs) > 1:
        tN, sN = _time(lambda: run(devs))
        print(f"scan_to_sino {len(devs)} devices: {tN:.3f}s   speedup {t1 / tN:.2f}x")
        print("byte-identical (1 vs N):", np.array_equal(np.asarray(s1), np.asarray(sN)))
