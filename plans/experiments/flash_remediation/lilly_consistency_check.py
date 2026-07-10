"""The reverse ablation: is model-data INCONSISTENCY the missing stripe ingredient?

Eleven Lilly-matched synthetic conditions (structure, weights, noise, segmentation,
row binning linear and in transmission space) all fail to stripe, while the real 8x
no-taper split stripes at 324x background.  This experiment ablates in the OTHER
direction: the exact Lilly 8x problem -- same model, same geometry offsets, same
transmission_root weights as the striped case -- with ONLY the data replaced by the
forward projection of the unsplit reference recon.  That sinogram is perfectly
consistent with the model by construction but carries the real object structure and
scale.  Single variable = consistency.

  Stripes VANISH  -> real-data inconsistency drives the stripes (dose it back in next
                     via proj(ref) + alpha*(real - proj(ref))).
  Stripes PERSIST -> the object/geometry combination is the driver and the synthetic
                     phantoms simply have not captured it.

All knobs below (no CLI args).
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
    sino_real, cone_params, optional_params = mjp.nsi.compute_sino_and_params(
        DATASET_DIR, downsample_factor=DOWNSAMPLE_FACTOR,
        subsample_view_factor=SUBSAMPLE_VIEW_FACTOR)
    sino_real = np.asarray(sino_real)
    ct_model = mj.ConeBeamModel(**cone_params)
    ct_model.set_params(**optional_params)
    # Weights EXACTLY as in the striped case (from the real sino) -- not a variable here.
    weights = np.asarray(mj.gen_weights(sino_real, "transmission_root"))

    ref_real = np.load(f"{OUT_DIR}/recon_ref_15_ds8.npy")
    sino_cons = np.asarray(ct_model.forward_project(ref_real))
    resid = sino_real - sino_cons
    print(f"real-vs-consistent residual: RMS {np.sqrt(np.mean(resid**2)):.4f}, "
          f"p99 |resid| {np.percentile(np.abs(resid), 99):.4f}, "
          f"sino_real RMS {np.sqrt(np.mean(sino_real**2)):.4f}", flush=True)

    ct_model.auto_set_regularization_params(sino_cons)
    ct_model.set_params(auto_regularize_flag=False)

    jobs = [("ref_cons_ds8", None),
            ("no_taper_cons_ds8", dict(h_sino=H_DEFAULT, h_recon=H_DEFAULT, taper=False,
                                       num_iterations=MAX_ITERATIONS))]
    for name, kw in jobs:
        out = f"{OUT_DIR}/recon_{name}.npy"
        if os.path.exists(out):
            print(f"--- {name}: exists, skipped", flush=True)
            continue
        print(f"--- {name}", flush=True)
        if kw is None:
            np.random.seed(0)
            vol, _ = ct_model.recon(sino_cons, weights=weights,
                                    max_iterations=MAX_ITERATIONS,
                                    stop_threshold_change_pct=1e-9, print_logs=False)
            vol = np.asarray(vol)
        else:
            vol = split_recon(ct_model, sino_cons, weights, **kw)
        np.save(out, vol)

    ref = np.load(f"{OUT_DIR}/recon_ref_cons_ds8.npy")
    split = np.load(f"{OUT_DIR}/recon_no_taper_cons_ds8.npy")
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
    print(f"seam max RMS: {rms[lo:hi].max():.3e}  ({rms[lo:hi].max()/bg:.1f}x bg)  "
          f"[real no-taper was 7.9e-3]", flush=True)
    print("done: lilly_consistency_check", flush=True)
