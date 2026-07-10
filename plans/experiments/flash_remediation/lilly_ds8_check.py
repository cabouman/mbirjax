"""Do the Lilly seam stripes persist at 8x downsampling (views AND detector)?

If yes, the 8x problem becomes the fast-turnaround workhorse for the seam investigation;
if no, stay at 4x.  Mechanism prediction: PERSIST -- the unexplained extension depth
h_sino*(1+R/SID)*pitch - h_recon is a RATIO of like-scaled quantities (R/SID physical,
pitch ratio ~1 at any NSI downsampling), so ~1.05 slices at 8x as at 4x.

Runs the unsplit reference plus the no_taper and taper split variants (h=5/5, 15 fixed
iterations, transmission_root weights) via the local parameterized split, and prints the
per-slice seam RMS verdict for both variants.  All knobs below (no CLI args).
"""
import os
import numpy as np
import mbirjax as mj
import mbirjax.preprocess as mjp
from lilly_split_ablations import split_recon, DATASET_DIR, OUT_DIR, H_DEFAULT

DOWNSAMPLE_FACTOR = (8, 8)
SUBSAMPLE_VIEW_FACTOR = 8
MAX_ITERATIONS = 15
SEAM_VIEW_HALF_WIDTH = 12
INTERIOR_RADIUS_FRAC = 0.85

if __name__ == "__main__":
    print("mbirjax install:", mj.__file__, flush=True)
    sino, cone_params, optional_params = mjp.nsi.compute_sino_and_params(
        DATASET_DIR, downsample_factor=DOWNSAMPLE_FACTOR,
        subsample_view_factor=SUBSAMPLE_VIEW_FACTOR)
    sino = np.asarray(sino)
    print(f"sino shape {sino.shape}", flush=True)
    ct_model = mj.ConeBeamModel(**cone_params)
    ct_model.set_params(**optional_params)
    print({k: ct_model.get_params(k) for k in
           ["delta_det_row", "det_row_offset", "recon_shape", "recon_slice_offset"]},
          flush=True)
    weights = np.asarray(mj.gen_weights(sino, "transmission_root"))
    ct_model.auto_set_regularization_params(sino)
    ct_model.set_params(auto_regularize_flag=False)

    jobs = [
        ("ref_15_ds8", None),
        ("no_taper_15_ds8", dict(h_sino=H_DEFAULT, h_recon=H_DEFAULT, taper=False,
                                 num_iterations=MAX_ITERATIONS)),
        ("taper_15_ds8", dict(h_sino=H_DEFAULT, h_recon=H_DEFAULT, taper=True,
                              num_iterations=MAX_ITERATIONS)),
    ]
    for name, kw in jobs:
        out = f"{OUT_DIR}/recon_{name}.npy"
        if os.path.exists(out):
            print(f"--- {name}: exists, skipped", flush=True)
            continue
        print(f"--- {name}", flush=True)
        if kw is None:
            np.random.seed(0)
            vol, _ = ct_model.recon(sino, weights=weights, max_iterations=MAX_ITERATIONS,
                                    stop_threshold_change_pct=1e-9, print_logs=False)
            vol = np.asarray(vol)
        else:
            vol = split_recon(ct_model, sino, weights, **kw)
        np.save(out, vol)

    ref = np.load(f"{OUT_DIR}/recon_ref_15_ds8.npy")
    shape = ref.shape
    slice_off = ct_model.get_params("recon_slice_offset")
    dslice = ct_model.get_params("voxel_slice_aspect") * ct_model.get_params("delta_voxel")
    split_index = int(np.round((shape[2] - 1) / 2.0 - slice_off / dslice))
    i = np.arange(shape[0], dtype=np.float32)[:, None] - (shape[0] - 1) / 2.0
    j = np.arange(shape[1], dtype=np.float32)[None, :] - (shape[1] - 1) / 2.0
    disk = np.sqrt(i**2 + j**2) < INTERIOR_RADIUS_FRAC * (min(shape[:2]) / 2.0)
    lo = max(0, split_index - SEAM_VIEW_HALF_WIDTH)
    hi = min(shape[2], split_index + SEAM_VIEW_HALF_WIDTH + 1)
    for name in ["no_taper_15_ds8", "taper_15_ds8"]:
        split = np.load(f"{OUT_DIR}/recon_{name}.npy")
        rms = np.sqrt(np.mean((split - ref)[disk] ** 2, axis=0))
        bg = np.median(np.concatenate([rms[:lo], rms[hi:]]))
        print(f"\n=== {name}: split at slice {split_index}; background median RMS {bg:.3e}",
              flush=True)
        for s in range(lo, hi):
            marker = "  <-- split" if s == split_index else ""
            print(f"  slice {s:4d}: RMS {rms[s]:.3e}  ({rms[s]/bg:6.1f}x bg){marker}",
                  flush=True)
        print(f"{name} seam max RMS: {rms[lo:hi].max():.3e}  ({rms[lo:hi].max()/bg:.1f}x bg)",
              flush=True)
    print("done: lilly_ds8_check", flush=True)
