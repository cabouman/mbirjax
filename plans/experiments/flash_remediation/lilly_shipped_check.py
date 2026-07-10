"""Confirming variant for the version check: the SHIPPED split_sino_recon at the current
branch (taper and all) on Lilly -- expected CLEAN, matching Greg\x27s experience that adding
the sine taper removed the stripes.  Compares against recon_ref_15_head.npy (the unsplit
15-iter reference computed at the same library version).  All knobs below (no CLI args).
"""
import os
import numpy as np
import mbirjax as mj
import mbirjax.preprocess as mjp
from lilly_split_ablations import DATASET_DIR, OUT_DIR, DOWNSAMPLE_FACTOR, SUBSAMPLE_VIEW_FACTOR

MAX_ITERATIONS = 15
SEAM_VIEW_HALF_WIDTH = 16
INTERIOR_RADIUS_FRAC = 0.85

if __name__ == "__main__":
    print("mbirjax install:", mj.__file__, flush=True)
    out = f"{OUT_DIR}/recon_shipped_15_head.npy"
    sino, cone_params, optional_params = mjp.nsi.compute_sino_and_params(
        DATASET_DIR, downsample_factor=DOWNSAMPLE_FACTOR,
        subsample_view_factor=SUBSAMPLE_VIEW_FACTOR)
    sino = np.asarray(sino)
    ct_model = mj.ConeBeamModel(**cone_params)
    ct_model.set_params(**optional_params)
    weights = np.asarray(mj.gen_weights(sino, "transmission_root"))
    if not os.path.exists(out):
        np.random.seed(0)
        recon, _ = ct_model.split_sino_recon(sino, weights=weights, half_overlap=5,
                                             max_iterations=MAX_ITERATIONS,
                                             stop_threshold_change_pct=0.2)
        np.save(out, np.asarray(recon))

    ref = np.load(f"{OUT_DIR}/recon_ref_15_head.npy")
    split = np.load(out)
    shape = ref.shape
    slice_off = ct_model.get_params("recon_slice_offset")
    dslice = ct_model.get_params("voxel_slice_aspect") * ct_model.get_params("delta_voxel")
    split_index = int(np.round((shape[2] - 1) / 2.0 - slice_off / dslice))
    i = np.arange(shape[0], dtype=np.float32)[:, None] - (shape[0] - 1) / 2.0
    j = np.arange(shape[1], dtype=np.float32)[None, :] - (shape[1] - 1) / 2.0
    disk = np.sqrt(i**2 + j**2) < INTERIOR_RADIUS_FRAC * (min(shape[:2]) / 2.0)
    rms = np.sqrt(np.mean((split - ref)[disk] ** 2, axis=0))
    lo = max(0, split_index - SEAM_VIEW_HALF_WIDTH)
    hi = min(shape[2], split_index + SEAM_VIEW_HALF_WIDTH + 1)
    bg = np.median(np.concatenate([rms[:lo], rms[hi:]]))
    print(f"split at slice {split_index}; background median RMS {bg:.3e}", flush=True)
    for s in range(lo, hi):
        marker = "  <-- split" if s == split_index else ""
        print(f"  slice {s:4d}: RMS {rms[s]:.3e}  ({rms[s]/bg:6.1f}x bg){marker}", flush=True)
    print(f"seam max/background: {rms[lo:hi].max()/bg:.1f}x", flush=True)
    print(f"seam max RMS: {rms[lo:hi].max():.3e}  (568 no-taper was ~8e-3; 568 taper_15 was 6.5e-4)", flush=True)
    print("done: lilly_shipped_check", flush=True)
