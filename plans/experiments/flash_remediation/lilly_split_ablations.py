"""Lilly seam-stripe ablations: transient vs structural vs taper.

The 568f6b7 reproduction (lilly_split_repro.py) showed +/-40% alternating stripes within
~10 slices of the seam, with ALL recons run to the full 15 iterations (change% still
1.1-1.5%, far above the stop) -- so everything is under-converged, and per-half stopping
is ruled out as the cause.  Three hypotheses remain:

  T (transient)   the stripes are the convergence transient of the halves' small
                  end-of-extension inconsistency (each half is axially truncated ~1 slice
                  deep at its extension end at this geometry) -> more iterations fade them;
  S (structural)  the inconsistency itself is the problem at any iteration count ->
                  a deeper recon extension fixes it, iterations alone do not;
  W (taper)       the shipped sine taper suppresses the driver -> taper at 15 iterations
                  is clean while no-taper at 15 is striped (reproduces why the fix worked).

Variants (parameterized local split, equal fixed iterations per half; real transmission
weights, taper multiplied in):
  ref60                unsplit reference, 60 iterations
  no_taper_60          h=5/5, no taper, 60 iterations          -> tests T
  taper_15             h=5/5, sine taper, 15 iterations        -> tests W
  no_taper_deep_15     h_sino=5, h_recon=12, no taper, 15 it   -> tests S

(The 15-iteration no-taper split and 15-iteration unsplit reference already exist from
lilly_split_repro.py.)  All knobs below (no CLI args).
"""

import numpy as np
import mbirjax as mj
import mbirjax.preprocess as mjp

DATASET_DIR = '/scratch/gautschi/buzzard/flash_lilly/D01788'
OUT_DIR = '/scratch/gautschi/buzzard/flash_lilly'
DOWNSAMPLE_FACTOR = (4, 4)
SUBSAMPLE_VIEW_FACTOR = 2
H_DEFAULT = 5
H_DEEP = 12


def split_recon(full_model, sino, weights, h_sino, h_recon, taper, num_iterations):
    """Parameterized split (mirrors split_sino_recon geometry; taper multiplies weights)."""
    sino = np.asarray(sino)
    num_views, full_rows, num_cols = sino.shape
    delta_det_row = full_model.get_params('delta_det_row')
    full_det_row_offset = full_model.get_params('det_row_offset')
    delta_voxel = full_model.get_params('delta_voxel')
    delta_slice = full_model.get_params('voxel_slice_aspect') * delta_voxel
    full_shape = full_model.get_params('recon_shape')
    full_slice_offset = full_model.get_params('recon_slice_offset')

    full_slices = full_shape[2]
    iso_slice_float = (full_slices - 1) / 2.0 - full_slice_offset / delta_slice
    split_index = int(np.round(iso_slice_float))
    top_num_slices = split_index + 1
    split_offset = split_index - iso_slice_float

    det_iso_float = (full_rows - 1) / 2.0 + full_det_row_offset / delta_det_row
    det_iso = int(np.round(det_iso_float))
    full_det_center = (full_rows - 1) / 2.0

    top_shape = (full_shape[0], full_shape[1], top_num_slices + h_recon)
    bot_shape = (full_shape[0], full_shape[1], (full_slices - top_num_slices) + h_recon)
    top_off = (+h_recon - (top_shape[2] - 1) / 2 + 0 + split_offset) * delta_slice
    bot_off = (-h_recon + (bot_shape[2] - 1) / 2 + 1 + split_offset) * delta_slice

    if h_sino > 0:
        ramp = np.sin((np.pi / 2) * np.linspace(0, 1, h_sino, endpoint=False)).astype(np.float32)

    def one_half(lo, hi, shape, slice_offset, taper_top):
        num_rows = hi - lo
        det_center = (num_rows - 1) / 2.0
        det_off = full_det_row_offset + (full_det_center - (det_center + lo)) * delta_det_row
        model = mj.copy_ct_model(full_model, new_num_det_rows=num_rows)
        model.set_params(det_row_offset=det_off, auto_regularize_flag=False,
                         recon_shape=shape, recon_slice_offset=slice_offset, verbose=0)
        w = np.array(weights[:, lo:hi, :])
        if taper and h_sino > 0:
            if taper_top:
                w[:, -h_sino:, :] *= ramp[None, ::-1, None]
            else:
                w[:, :h_sino, :] *= ramp[None, :, None]
        recon, _ = model.recon(sino[:, lo:hi, :], weights=w,
                               max_iterations=num_iterations,
                               stop_threshold_change_pct=1e-9, print_logs=False)
        return np.asarray(recon)

    np.random.seed(0)
    top = one_half(0, min(det_iso + h_sino, full_rows), top_shape, top_off, taper_top=True)
    np.random.seed(0)
    bot = one_half(max(det_iso - h_sino, 0), full_rows, bot_shape, bot_off, taper_top=False)

    ramp_overlap = min(4, h_recon)
    ramp_overlap -= ramp_overlap % 2
    return np.asarray(mj.stitch_arrays([top, bot], axis=2, overlap=2 * h_recon,
                                       ramp_overlap=ramp_overlap))


if __name__ == '__main__':
    import os
    print('loading NSI scan...', flush=True)
    sino, cone_params, optional_params = mjp.nsi.compute_sino_and_params(
        DATASET_DIR, downsample_factor=DOWNSAMPLE_FACTOR,
        subsample_view_factor=SUBSAMPLE_VIEW_FACTOR)
    sino = np.asarray(sino)
    ct_model = mj.ConeBeamModel(**cone_params)
    ct_model.set_params(**optional_params)
    weights = np.asarray(mj.gen_weights(sino, 'transmission_root'))
    ct_model.auto_set_regularization_params(sino)
    ct_model.set_params(auto_regularize_flag=False)

    jobs = [
        ('ref60', None),
        ('no_taper_60', dict(h_sino=H_DEFAULT, h_recon=H_DEFAULT, taper=False,
                             num_iterations=60)),
        ('taper_15', dict(h_sino=H_DEFAULT, h_recon=H_DEFAULT, taper=True,
                          num_iterations=15)),
        ('no_taper_deep_15', dict(h_sino=H_DEFAULT, h_recon=H_DEEP, taper=False,
                                  num_iterations=15)),
    ]
    for name, kw in jobs:
        out = f'{OUT_DIR}/recon_{name}.npy'
        if os.path.exists(out):
            print(f'--- {name}: exists, skipped', flush=True)
            continue
        print(f'--- {name}', flush=True)
        if kw is None:
            np.random.seed(0)
            vol, _ = ct_model.recon(sino, weights=weights, max_iterations=60,
                                    stop_threshold_change_pct=1e-9, print_logs=False)
            vol = np.asarray(vol)
        else:
            vol = split_recon(ct_model, sino, weights, **kw)
        np.save(out, vol)
    print('ablations done', flush=True)
