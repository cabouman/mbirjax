"""Phase 3 / P2c follow-up: reproduce the split_sino_recon seam stripes on real data.

Greg's report (2026-07-08): the Lilly D01788 NSI scan, reconstructed with
split_sino_recon AS OF COMMIT 568f6b7 (2026-06-26 -- which has NO sine taper; the taper
was added later, presumably in response to these stripes), shows stripes near the seam.
That contradicts the P2c synthetic result (no-taper split == unsplit to float noise), so
something real is missing from the synthetic: candidates are per-half independent stopping
(each half stops on its own 0.2% change metric), real transmission weights, and
fractional iso-row / pitch-ratio geometry.

This script runs on a machine where the editable mbirjax install points at 568f6b7 (or
any commit under test) and reproduces the 6/26 run: NSI load, transmission_root weights,
split_sino_recon with defaults, plus an UNSPLIT reference recon of the same sinogram for
comparison.  Volumes are saved for seam analysis (figures built separately).

All knobs below (no CLI args).
"""

import numpy as np
import mbirjax as mj
import mbirjax.preprocess as mjp

# #### Paths (cluster: gautschi scratch)
DATASET_DIR = '/scratch/gautschi/buzzard/flash_lilly/D01788'
OUT_DIR = '/scratch/gautschi/buzzard/flash_lilly'

# #### Load settings (downsampled for a fast first reproduction; escalate if no stripes)
DOWNSAMPLE_FACTOR = (4, 4)
SUBSAMPLE_VIEW_FACTOR = 2

# #### Recon settings: the split_sino_recon DEFAULTS as of 568f6b7
HALF_OVERLAP = 5
MAX_ITERATIONS = 15
STOP_THRESHOLD = 0.2

if __name__ == '__main__':
    print('loading NSI scan...', flush=True)
    sino, cone_params, optional_params = mjp.nsi.compute_sino_and_params(
        DATASET_DIR, downsample_factor=DOWNSAMPLE_FACTOR,
        subsample_view_factor=SUBSAMPLE_VIEW_FACTOR)
    sino = np.asarray(sino)
    print(f'sino shape {sino.shape}', flush=True)

    ct_model = mj.ConeBeamModel(**cone_params)
    ct_model.set_params(**optional_params)
    print({k: ct_model.get_params(k) for k in
           ['delta_det_row', 'det_row_offset', 'delta_voxel', 'voxel_slice_aspect',
            'recon_shape', 'recon_slice_offset']}, flush=True)
    weights = np.asarray(mj.gen_weights(sino, 'transmission_root'))

    print('split_sino_recon (568f6b7 defaults)...', flush=True)
    recon_split, split_dict = ct_model.split_sino_recon(
        sino, weights=weights, half_overlap=HALF_OVERLAP,
        max_iterations=MAX_ITERATIONS, stop_threshold_change_pct=STOP_THRESHOLD)
    np.save(f'{OUT_DIR}/recon_split.npy', np.asarray(recon_split))
    for side in ['top', 'bottom']:
        log = split_dict.get(f'recon_log_{side}', '')
        print(f'--- {side} half recon log tail:', flush=True)
        print(str(log)[-400:], flush=True)

    print('unsplit reference recon...', flush=True)
    recon_full, _ = ct_model.recon(sino, weights=weights,
                                   max_iterations=MAX_ITERATIONS,
                                   stop_threshold_change_pct=STOP_THRESHOLD)
    np.save(f'{OUT_DIR}/recon_full.npy', np.asarray(recon_full))
    print('done', flush=True)
