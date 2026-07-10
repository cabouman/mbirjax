"""Consistency-ablation round 2: WHICH ingredient of the consistent-data case drives the
stripes -- the weights, or the fractional axial alignment?

lilly_consistency_check.py showed the stripes need NO data inconsistency: the forward
projection of the reference recon through the real model stripes at 7.7e-3 (298x bg),
matching the real data.  The remaining ingredients are the object (shared), the weights,
and the model geometry.  The geometry suspect is specific: det_row_offset = -1.95 rows +
recon_slice_offset = 1.95 slices put the SPLIT PLANE 0.45 slices off the true iso plane
(near worst case), while the clean synthetics had exact alignment.

  cons_unitw     consistent sino, real offsets kept, UNIT weights     -> weights test
  cons_nooffset  same object trimmed to 235 slices (odd -> centered grid has
                 split_offset EXACTLY 0), projected through a model with the axial
                 offsets zeroed; transmission_root weights of that sino -> alignment test

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

def seam_report(name, split_vol, ref, model):
    shape = ref.shape
    slice_off = model.get_params("recon_slice_offset")
    dslice = model.get_params("voxel_slice_aspect") * model.get_params("delta_voxel")
    split_index = int(np.round((shape[2] - 1) / 2.0 - slice_off / dslice))
    i = np.arange(shape[0], dtype=np.float32)[:, None] - (shape[0] - 1) / 2.0
    j = np.arange(shape[1], dtype=np.float32)[None, :] - (shape[1] - 1) / 2.0
    disk = np.sqrt(i**2 + j**2) < INTERIOR_RADIUS_FRAC * (min(shape[:2]) / 2.0)
    rms = np.sqrt(np.mean((split_vol - ref)[disk] ** 2, axis=0))
    lo = max(0, split_index - SEAM_VIEW_HALF_WIDTH)
    hi = min(shape[2], split_index + SEAM_VIEW_HALF_WIDTH + 1)
    bg = np.median(np.concatenate([rms[:lo], rms[hi:]]))
    print(f"{name}: split at {split_index}; bg {bg:.3e}; "
          f"seam max RMS {rms[lo:hi].max():.3e} ({rms[lo:hi].max()/bg:.1f}x bg)",
          flush=True)

def run_pair(tag, model, sino, weights):
    model.auto_set_regularization_params(sino)
    model.set_params(auto_regularize_flag=False)
    ref_path = f"{OUT_DIR}/recon_ref_{tag}.npy"
    if not os.path.exists(ref_path):
        np.random.seed(0)
        vol, _ = model.recon(sino, weights=weights, max_iterations=MAX_ITERATIONS,
                             stop_threshold_change_pct=1e-9, print_logs=False)
        np.save(ref_path, np.asarray(vol))
    split_path = f"{OUT_DIR}/recon_no_taper_{tag}.npy"
    if not os.path.exists(split_path):
        w = weights if weights is not None else np.ones_like(sino)
        vol = split_recon(model, sino, w, h_sino=H_DEFAULT, h_recon=H_DEFAULT,
                          taper=False, num_iterations=MAX_ITERATIONS)
        np.save(split_path, vol)
    seam_report(tag, np.load(split_path), np.load(ref_path), model)

if __name__ == "__main__":
    print("mbirjax install:", mj.__file__, flush=True)
    sino_real, cone_params, optional_params = mjp.nsi.compute_sino_and_params(
        DATASET_DIR, downsample_factor=DOWNSAMPLE_FACTOR,
        subsample_view_factor=SUBSAMPLE_VIEW_FACTOR)
    sino_real = np.asarray(sino_real)
    print("optional_params:", optional_params, flush=True)
    ref_real = np.load(f"{OUT_DIR}/recon_ref_15_ds8.npy")

    # --- cons_unitw: real offsets, consistent sino, unit weights ---
    model_a = mj.ConeBeamModel(**cone_params)
    model_a.set_params(**optional_params)
    sino_cons = np.asarray(model_a.forward_project(ref_real))
    run_pair("cons_unitw_ds8", model_a, sino_cons, None)

    # --- cons_nooffset: axial offsets zeroed, odd slice count -> split_offset 0 ---
    model_b = mj.ConeBeamModel(**cone_params)
    opt_b = dict(optional_params)
    opt_b["det_row_offset"] = 0.0
    opt_b["recon_slice_offset"] = 0.0
    model_b.set_params(**opt_b)
    shape_b = list(model_b.get_params("recon_shape"))
    shape_b[2] = 235  # odd -> centered grid, split_offset exactly 0
    model_b.set_params(recon_shape=tuple(shape_b))
    obj_b = ref_real[:, :, :235]
    sino_b = np.asarray(model_b.forward_project(obj_b))
    weights_b = np.asarray(mj.gen_weights(sino_b, "transmission_root"))
    run_pair("cons_nooffset_ds8", model_b, sino_b, weights_b)
    print("done: lilly_cons2", flush=True)
