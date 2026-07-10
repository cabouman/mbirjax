"""Lilly seam-stripe VERSION check: does the CURRENT library reproduce the 568f6b7 stripes?

All prior Lilly runs (repro + ablations) used the editable install pinned at 568f6b7
(2026-06-26); all synthetic P2c probes used greg/kernel_investigation (post projector-
kernel campaign).  Library version is therefore CONFOUNDED across the real-vs-synthetic
comparison.  This script re-runs the exact 6/26 no-taper split conditions (ds (4,4)/ss 2,
transmission_root weights, h=5/5, 15 fixed iterations) against whatever mbirjax install is
on sys.path -- submit it under the current-branch env.  Outputs recon_*_head.npy alongside
the 568 volumes, and prints the per-slice seam profile so the log itself gives the verdict.

  CLEAN here  -> the stripes are specific to the old projector (kernel campaign changed
                 rounding/gather behavior); the h_recon proposal needs rethinking.
  STRIPED     -> version ruled out; the synthetic-reproduction gap is about real-data
                 properties (object structure, weights).

All knobs below (no CLI args).
"""
import os
import numpy as np
import mbirjax as mj
import mbirjax.preprocess as mjp
from lilly_split_ablations import (split_recon, DATASET_DIR, OUT_DIR,
                                   DOWNSAMPLE_FACTOR, SUBSAMPLE_VIEW_FACTOR, H_DEFAULT)

MAX_ITERATIONS = 15
SEAM_VIEW_HALF_WIDTH = 16     # slices printed around the split
INTERIOR_RADIUS_FRAC = 0.85

if __name__ == "__main__":
    print("mbirjax install:", mj.__file__, flush=True)
    print("loading NSI scan...", flush=True)
    sino, cone_params, optional_params = mjp.nsi.compute_sino_and_params(
        DATASET_DIR, downsample_factor=DOWNSAMPLE_FACTOR,
        subsample_view_factor=SUBSAMPLE_VIEW_FACTOR)
    sino = np.asarray(sino)
    ct_model = mj.ConeBeamModel(**cone_params)
    ct_model.set_params(**optional_params)
    weights = np.asarray(mj.gen_weights(sino, "transmission_root"))
    ct_model.auto_set_regularization_params(sino)
    ct_model.set_params(auto_regularize_flag=False)

    jobs = [
        ("ref_15_head", None),
        ("no_taper_15_head", dict(h_sino=H_DEFAULT, h_recon=H_DEFAULT, taper=False,
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

    # Inline seam verdict: per-slice RMS of (split - unsplit ref) over the interior disk,
    # exactly the lilly_seam_analysis.py metric.
    ref = np.load(f"{OUT_DIR}/recon_ref_15_head.npy")
    split = np.load(f"{OUT_DIR}/recon_no_taper_15_head.npy")
    shape = ref.shape
    slice_off = ct_model.get_params("recon_slice_offset")
    dslice = ct_model.get_params("voxel_slice_aspect") * ct_model.get_params("delta_voxel")
    split_index = int(np.round((shape[2] - 1) / 2.0 - slice_off / dslice))
    i = np.arange(shape[0], dtype=np.float32)[:, None] - (shape[0] - 1) / 2.0
    j = np.arange(shape[1], dtype=np.float32)[None, :] - (shape[1] - 1) / 2.0
    disk = np.sqrt(i**2 + j**2) < INTERIOR_RADIUS_FRAC * (min(shape[:2]) / 2.0)
    diff = split - ref
    rms = np.sqrt(np.mean(diff[disk] ** 2, axis=0))
    lo = max(0, split_index - SEAM_VIEW_HALF_WIDTH)
    hi = min(shape[2], split_index + SEAM_VIEW_HALF_WIDTH + 1)
    bg = np.median(np.concatenate([rms[:lo], rms[hi:]]))
    print(f"split at slice {split_index}; background median RMS {bg:.3e}", flush=True)
    for s in range(lo, hi):
        marker = "  <-- split" if s == split_index else ""
        print(f"  slice {s:4d}: RMS {rms[s]:.3e}  ({rms[s]/bg:6.1f}x bg){marker}", flush=True)
    print(f"seam max/background: {rms[lo:hi].max()/bg:.1f}x", flush=True)
    print("done: lilly_head_check", flush=True)
