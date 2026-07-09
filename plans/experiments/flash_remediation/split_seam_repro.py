"""P2c: the split_sino_recon seam A/B -- which split design produces the cleanest seam?

split_sino_recon halves memory by reconstructing the detector-row halves separately and
stitching.  The mechanism analysis (plan doc, "the split_sino_recon seam...") frames the
design: the forward model is ~separable at the iso-row split (source in the iso plane, so
no ray crosses it; coupling = PSF/rounding width only), but the qGGMRF prior couples slices
-- the extended (past-split) recon slices act as PRIOR BOUNDARY CONDITIONS and must be
data-accurate.  This script A/Bs the design space on an object FULLY inside the FoV and
slab (no physical truncation, so ALL seam error is split-induced), judged against an
UNSPLIT reference recon at the same iteration count:

  current        sino overlap + sine taper + recon overlap (the shipped design, h=5)
  no_taper       identical geometry, taper OFF          -> isolates the taper's effect
  no_taper_deep  taper OFF, recon overlap deepened so the extended rows' contributions
                 are fully explainable (h_recon = ceil(h_sino*(1+R/SID)*pitch) + psf)
                 -> separates "taper vs no taper" from extension depth
  truncate       sino CUT at the split row, recon overlap kept -> the prior-extrapolation
                 variant (predicted to tie on smooth objects, lose when structure crosses
                 the split)

Two phantoms: STRUCTURED (laminate layers crossing the split + a sphere straddling it --
the hard case) and SMOOTH (no laminate) as the control.  The split logic is a local
parameterized reimplementation of cone_beam.split_sino_recon's geometry (library
untouched); regularization comes from the full sinogram and is copied to the halves,
exactly as the library does.

Run inside the mbirjax conda env (CPU or GPU); all knobs below (no CLI args).  Results in
results/p2c_*; set make_figures=True (run_phantoms=[] to skip recons) to build figures.
"""

import os
import numpy as np
import mbirjax as mj  # noqa: F401 -- must precede anything that touches jax
import truncation_common as tc

# ---------------------------------------------------------------------------
# Run control
# ---------------------------------------------------------------------------
run_phantoms = ['structured', 'smooth', 'structured_widefan', 'widefan_noise15']
make_figures = True
skip_existing = True

# ---------------------------------------------------------------------------
# Problem definition (object CONTAINED in FoV and slab -- no physical truncation)
# ---------------------------------------------------------------------------
NUM_VIEWS = 128
NUM_DET_ROWS = 64
NUM_DET_CHANNELS = 96
RADIUS_FRAC = 0.75
Z_LO_FRAC, Z_HI_FRAC = -0.85, 0.85
TARGET_LINE_INTEGRAL = 2.0
# Runs: phantom structure x fan angle x real-data conditions.  Motivated by Greg's
# real-data challenge (2026-07-09; Lilly D01788 shows seam stripes the original synthetic
# missed).  structured_widefan matches Lilly's R/SID (0.2 -> unexplained coupling ~1 slice
# at h=5/5, like Lilly's 1.06) but came back CLEAN with unit weights at 40 iterations --
# fan angle alone does not reproduce the stripes; widefan_noise15 adds the remaining
# real-data conditions (photon noise, transmission weights, 15 iterations).
RUNS = {
    'structured': dict(laminate=3, sdd_factor=4.0, sid_factor=2.0),
    'smooth': dict(laminate=0, sdd_factor=4.0, sid_factor=2.0),
    'structured_widefan': dict(laminate=3, sdd_factor=2.5, sid_factor=1.25),
    # Match the REAL-data conditions that the plain widefan run lacked: photon noise +
    # transmission weights + the 6/26 default of 15 iterations.
    'widefan_noise15': dict(laminate=3, sdd_factor=2.5, sid_factor=1.25, noise=True,
                            iters=15),
}

H_DEFAULT = 5                    # the shipped half_overlap default
PSF_MARGIN = 2
NUM_ITERATIONS = 40
SEAM_HALF_WIDTH = 4              # seam metric region: split +/- this many slices
INTERIOR_RADIUS_FRAC = 0.85

# label -> (h_sino, h_recon or 'deep', taper)
VARIANTS = {
    'current': (H_DEFAULT, H_DEFAULT, True),
    'no_taper': (H_DEFAULT, H_DEFAULT, False),
    'no_taper_deep': (H_DEFAULT, 'deep', False),
    'truncate': (0, H_DEFAULT, False),
}

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(OUT_DIR, 'figures')
RES_DIR = os.path.join(OUT_DIR, 'results')


def split_recon(full_model, sino, h_sino, h_recon, taper, num_iterations, base_weights=None):
    """Parameterized reimplementation of split_sino_recon's split (circular orbit only).

    Mirrors the library's geometry math (iso row/slice, per-half detector and recon
    offsets, sine taper on the overlap rows, stitch with a small ramp), with h_sino and
    h_recon independent so the variants above are expressible.  h_sino = 0 cuts the
    sinogram exactly at the iso row.  Returns the stitched host volume.
    """
    sino = np.asarray(sino)
    num_views, full_rows, num_cols = sino.shape
    delta_det_row = full_model.get_params('delta_det_row')
    full_det_row_offset = full_model.get_params('det_row_offset')
    delta_voxel = full_model.get_params('delta_voxel')
    delta_slice = full_model.get_params('voxel_slice_aspect') * delta_voxel
    full_shape = full_model.get_params('recon_shape')
    full_slice_offset = full_model.get_params('recon_slice_offset')
    mag = full_model.get_magnification()

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
        weights = (np.ones((num_views, num_rows, num_cols), dtype=np.float32)
                   if base_weights is None else np.array(base_weights[:, lo:hi, :]))
        if taper and h_sino > 0:
            if taper_top:
                weights[:, -h_sino:, :] *= ramp[None, ::-1, None]
            else:
                weights[:, :h_sino, :] *= ramp[None, :, None]
        recon, _ = model.recon(sino[:, lo:hi, :], weights=weights,
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


def seam_mask(shape, split_index, interior_radius_frac, half_width):
    rows, cols, slices = shape
    i = np.arange(rows, dtype=np.float32)[:, None] - (rows - 1) / 2.0
    j = np.arange(cols, dtype=np.float32)[None, :] - (cols - 1) / 2.0
    interior2d = np.sqrt(i ** 2 + j ** 2) < interior_radius_frac * (min(rows, cols) / 2.0)
    band = np.zeros(slices, dtype=bool)
    band[max(0, split_index - half_width):split_index + half_width + 1] = True
    return interior2d[:, :, None] & band[None, None, :]


def run_phantom(tag):
    print(f'\n================ {tag} ================', flush=True)
    sinogram_shape = (NUM_VIEWS, NUM_DET_ROWS, NUM_DET_CHANNELS)
    angles = np.linspace(0, 2 * np.pi, NUM_VIEWS, endpoint=False)
    model = mj.ConeBeamModel(sinogram_shape, angles,
                             source_detector_dist=RUNS[tag]['sdd_factor'] * NUM_DET_CHANNELS,
                             source_iso_dist=RUNS[tag]['sid_factor'] * NUM_DET_CHANNELS)
    model.set_params(verbose=0)
    shape = model.get_params('recon_shape')
    delta_voxel = model.get_params('delta_voxel')

    phantom = tc.build_phantom(shape, shape, delta_voxel, RADIUS_FRAC, Z_LO_FRAC, Z_HI_FRAC,
                               TARGET_LINE_INTEGRAL, laminate_period=RUNS[tag]['laminate'])
    sino = np.asarray(model.forward_project(phantom))
    # Optional real-data conditions: photon noise + transmission weights (weights are
    # sliced per half inside split_recon, exactly as split_sino_recon slices user weights).
    if RUNS[tag].get('noise', False):
        sino, base_weights = tc.add_transmission_noise(sino, i0=1e4, seed=1)
    else:
        base_weights = None
    num_iterations = RUNS[tag].get('iters', NUM_ITERATIONS)
    # Regularization from the FULL sinogram, as the library does; halves copy it.
    model.auto_set_regularization_params(sino)

    # The deep recon overlap: the extended rows reach h_sino*delta_row past iso at the
    # detector, i.e. (h_sino*delta_row/mag)*(1 + R/SID) at the far FoV edge -> slices.
    geo = tc.cone_axial_geometry(model, psf_margin=0)
    r_over_sid = geo['fov_radius'] / (RUNS[tag]['sid_factor'] * NUM_DET_CHANNELS)
    mag = model.get_magnification()
    delta_slice = model.get_params('voxel_slice_aspect') * delta_voxel
    h_deep = int(np.ceil(H_DEFAULT * (model.get_params('delta_det_row') / mag)
                         * (1 + r_over_sid) / delta_slice)) + PSF_MARGIN
    print(f'shape {shape}; deep recon overlap = {h_deep} slices (default {H_DEFAULT})',
          flush=True)

    np.save(os.path.join(RES_DIR, f'p2c_{tag}_truth.npy'), phantom)
    ref_path = os.path.join(RES_DIR, f'p2c_{tag}_reference.npy')
    if not (skip_existing and os.path.exists(ref_path)):
        np.random.seed(0)
        reference, _ = model.recon(sino, weights=base_weights,
                                   max_iterations=num_iterations,
                                   stop_threshold_change_pct=1e-9, print_logs=False)
        np.save(ref_path, np.asarray(reference))
        print('reference (unsplit) done', flush=True)

    for label, (h_sino, h_recon, taper) in VARIANTS.items():
        out_path = os.path.join(RES_DIR, f'p2c_{tag}_{label}.npy')
        if skip_existing and os.path.exists(out_path):
            print(f'--- {tag}/{label}: exists, skipped', flush=True)
            continue
        h_r = h_deep if h_recon == 'deep' else h_recon
        print(f'--- {tag}/{label}: h_sino {h_sino}, h_recon {h_r}, taper {taper}',
              flush=True)
        stitched = split_recon(model, sino, h_sino, h_r, taper, num_iterations,
                               base_weights=base_weights)
        np.save(out_path, stitched)


def make_all_figures():
    for tag in RUNS:
        truth_path = os.path.join(RES_DIR, f'p2c_{tag}_truth.npy')
        if not os.path.exists(truth_path):
            print(f'  (no results for {tag}; skipped)')
            continue
        truth = np.load(truth_path)
        reference = np.load(os.path.join(RES_DIR, f'p2c_{tag}_reference.npy'))
        shape = truth.shape
        split_index = int(np.round((shape[2] - 1) / 2.0))
        seam = seam_mask(shape, split_index, INTERIOR_RADIUS_FRAC, SEAM_HALF_WIDTH)
        # Normalize like the other pages: by the ground-truth RMS over the RoR cylinder.
        ror = tc.make_masks(shape, INTERIOR_RADIUS_FRAC, 4)['ror']
        norm = float(np.sqrt(np.mean(truth[ror] ** 2)))

        display, table = {}, []
        for label in VARIANTS:
            path = os.path.join(RES_DIR, f'p2c_{tag}_{label}.npy')
            if not os.path.exists(path):
                continue
            vol = np.load(path)
            seam_ref = float(np.sqrt(np.mean((vol - reference)[seam] ** 2)) / norm)
            seam_truth = float(np.sqrt(np.mean((vol - truth)[seam] ** 2)) / norm)
            table.append((label, seam_ref, seam_truth))
            display[f'recon: {label}\nseam-vs-ref NRMSE {seam_ref:.3f}'] = vol
        if not display:
            continue

        tc.save_slice_montage(reference, display, axis=1, index=shape[1] // 2,
                              title=f'P2c {tag}: x-z sections vs the unsplit reference '
                                    f'(split at slice {split_index})',
                              path=os.path.join(FIG_DIR, f'p2c_{tag}_xz.png'),
                              region_mask=seam, region_label='dashed = seam-NRMSE region',
                              reference_label='unsplit reference')
        masks = tc.make_masks(shape, INTERIOR_RADIUS_FRAC, 4)
        profiles = {label: np.load(os.path.join(RES_DIR, f'p2c_{tag}_{label}.npy'))
                    for label, *_ in table}
        tc.plot_z_profile(reference, profiles, masks,
                          f'P2c {tag}: z profile across the split (interior-disk mean)',
                          os.path.join(FIG_DIR, f'p2c_{tag}_z_profile.png'),
                          xlim=(split_index - 12, split_index + 12),
                          truth_label='unsplit reference')
        print(f'\n{tag}: seam NRMSE vs reference / vs ground truth '
              f'(reference itself vs truth in seam: '
              f'{np.sqrt(np.mean((reference - truth)[seam] ** 2)) / norm:.4f})')
        for label, seam_ref, seam_truth in table:
            print(f'  {label:>14}: {seam_ref:.4f} / {seam_truth:.4f}')
    print(f'Figures in {FIG_DIR} (p2c_*)')


if __name__ == '__main__':
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)
    for tag in run_phantoms:
        run_phantom(tag)
    if make_figures:
        make_all_figures()
    print('\nDone: phantoms ' + (', '.join(run_phantoms) if run_phantoms else '(none)'))
