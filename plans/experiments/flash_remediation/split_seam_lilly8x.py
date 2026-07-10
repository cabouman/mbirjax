"""Synthetic reproduction of the Lilly seam stripes -- the Lilly-matched 8x screening.

Motivated by the figure clue that the Lilly stripes localize at the dense vertical pins
(~10x body attenuation, crossing the seam axially), plus the fact that transmission_root
weights through such pins give the overlap rows strongly structured weighting that unit
weights never had.  This screen isolates those two ingredients, one at a time, on a
synthetic problem matched to the Lilly 8x workhorse geometry (R/SID 0.21, 225 views,
235x187 detector, pitch ratio 1, h_sino = h_recon = 5, 15 iterations; object fully inside
the FoV and slab so ALL seam error is split-induced).

Per Greg (2026-07-09): run ONLY the unsplit reference + the no_taper split per condition
-- if no_taper does not stripe there is no point running the other variants.  Success
signature (from the real data): peak per-slice seam RMS ~1e-3 or worse with an
every-other-slice zigzag; the taper/deep variants come later, only for conditions that
stripe.

Conditions (single-variable ladder; RUN_CONDITIONS below selects which run):
  Round 1 (2026-07-09) -- ALL CLEAN (seam max 1e-5..1e-4, <2x bg; Lilly 8x = 7.9e-3, 324x):
  pins_unit         SOLID dense axial pins crossing the split; unit weights
  body_troot        smooth body only; transmission_root weights from the NOISELESS sino
  pins_troot        both, still noise-free
  pins_troot_noise  both + photon noise (weights from the noisy sino, like real data)
  Round 1 also exonerated weight texture by measurement: Lilly's line integrals are MILD
  (p99 ~0.6, max ~1.2 -> transmission_root weights only reach ~0.55) -- the synthetic's
  texture already exceeded the real one.

  Round 2: the real pins are SEGMENTED at the 1-3 slice scale (axial profile swings
  0.2 -> ~0 -> 0.2 slice-to-slice; chains of dense blobs with near-void gaps crossing the
  seam) -- voxel-scale axial structure the solid pins lacked, at exactly the stripes'
  every-other-slice frequency:
  segpins_unit      pins segmented with a regular 2-on/2-off slice period; unit weights
  segpins_troot     the same + transmission_root weights
  ragpins_unit      irregular (seeded) segment lengths 1-3, like the real chains
  Round 2 (2026-07-09) -- ALL CLEAN too (3-5e-5, <1.7x bg).

  Round 3: DETECTOR-ROW BINNING as the data-inconsistency source.  The real 8x sinogram
  averages 8 raw rows per bin; each raw row has its own cone angle, so the binned row is
  a mixture the point-row model cannot represent -- a structured inconsistency in every
  row.  The halves share the overlap rows but resolve the inconsistency with DIFFERENT
  (truncated) support near the split; the stitch exposes their disagreement.  Fits every
  observation: consistent synthetic data never stripes regardless of structure/weights;
  the deep extension fixes it (support restored -> same resolution of the same data);
  the taper works at 4x but fails at 8x (bin width doubles -> intra-bin cone-angle
  spread doubles).  Emulated by projecting at BIN_FACTOR-finer row pitch and averaging
  groups of BIN_FACTOR fine rows:
  binrow_segpins_unit  binned-row sino of the segmented-pin phantom; unit weights
  binrow_body_unit     binned-row sino of the smooth body (control: binning alone)
  Round 3 (2026-07-09) -- CLEAN, and NECESSARILY so: averaging LINE INTEGRALS is linear,
  so the binned sino is exactly the projection of an axially smoothed object -- still
  perfectly consistent data.  The flaw was the emulation, not the hypothesis.

  Round 4: bin in TRANSMISSION space, as real preprocessing does (counts are averaged,
  THEN -log): p_bin = -log(mean(exp(-p_fine))).  The Jensen gap makes the binned data
  genuinely inconsistent wherever p varies within a bin (segment edges, interfaces), and
  it grows ~quadratically with bin width -- matching the taper's 4x-works / 8x-fails
  regime behavior:
  binexp_segpins_unit  transmission-binned sino, segmented pins, unit weights
  binexp_body_unit     transmission-binned sino, smooth body (control)
  Round 4 (2026-07-09) -- CLEAN (Jensen gap ~1e-3 at this contrast: too weak).

  Round 5: THE ANSWER, from the reverse (real-data) ablation in lilly_consistency_check
  / lilly_cons2: the stripes need NO inconsistency (consistent projection of the ref
  recon stripes at 7.7e-3) and NO weights (unit weights 8.5e-3), but ZEROING THE AXIAL
  OFFSETS kills them 740x (1.1e-5).  The governing variable is the RELATIVE fractional
  misalignment between the sino cut row and the recon split slice (Lilly: cut 0.05 rows
  off iso, split 0.45 slices off -> mismatch ~0.4).  Default synthetic models LOCK the
  two grids together (even the 0.5-offset widefan case had both off by 0.5 in the same
  direction -> relative mismatch 0), which is why every prior condition was clean.
  Dose-response confirmation, fully synthetic (det_row_offset in ROW units; recon grid
  centered, 235 odd -> split_offset 0, so the relative mismatch = the row offset):
  offpins_015 / offpins_030 / offpins_045   segpins, unit weights, offset 0.15/0.30/0.45
  offbody_045                               smooth body at 0.45 (does structure matter?)

All knobs below (no CLI args).  Needs split_seam_repro.py (split_recon) and
truncation_common.py (noise helper) importable from the same directory.
"""

import os
import numpy as np
import mbirjax as mj
import truncation_common as tc
from split_seam_repro import split_recon

# ---------------------------------------------------------------------------
# Lilly-matched geometry (8x workhorse scale).  R/SID = 1/(2*sdd_factor) = 0.21; the
# magnification is NOT matched (2.0 vs Lilly's 4.69) -- P2a-R established that fractional
# axial quantities are governed by R/SID alone.
# ---------------------------------------------------------------------------
NUM_VIEWS = 225
NUM_DET_ROWS = 235
NUM_DET_CHANNELS = 187
SDD_FACTOR = 2.38
SID_FACTOR = 1.19

# Phantom: body cylinder + dense axial pins crossing the split (Lilly-like contrast).
BODY_RADIUS_FRAC = 0.75          # of the FoV radius
Z_LO_FRAC, Z_HI_FRAC = -0.85, 0.85
BODY_LINE_INTEGRAL = 3.0         # body attenuation scaled so a central ray integrates to this
PIN_CONTRAST = 10.0              # pin density / body density
PIN_RADIUS_VOX = 1.6
PIN_R_FRAC = 0.35                # pin center radius, as a fraction of the FoV radius
N_PINS = 3

H_DEFAULT = 5
NUM_ITERATIONS = 15
NOISE_I0 = 1e4
SEAM_VIEW_HALF_WIDTH = 12
INTERIOR_RADIUS_FRAC = 0.85

SEG_PERIOD_ON, SEG_PERIOD_OFF = 2, 2   # regular segmentation: slices dense / void
RAG_SEED = 7                           # irregular segmentation RNG seed
BIN_FACTOR = 8                         # fine rows averaged per detector bin (matches ds8)

CONDITIONS = {
    'pins_unit': dict(phantom='pins', weights='unit'),
    'body_troot': dict(phantom='body', weights='troot'),
    'pins_troot': dict(phantom='pins', weights='troot'),
    'pins_troot_noise': dict(phantom='pins', weights='troot_noise'),
    'segpins_unit': dict(phantom='segpins', weights='unit'),
    'segpins_troot': dict(phantom='segpins', weights='troot'),
    'ragpins_unit': dict(phantom='ragpins', weights='unit'),
    'binrow_segpins_unit': dict(phantom='segpins', weights='unit', bin_rows='linear'),
    'binrow_body_unit': dict(phantom='body', weights='unit', bin_rows='linear'),
    'binexp_segpins_unit': dict(phantom='segpins', weights='unit', bin_rows='counts'),
    'binexp_body_unit': dict(phantom='body', weights='unit', bin_rows='counts'),
    'offpins_015': dict(phantom='segpins', weights='unit', row_offset_frac=0.15),
    'offpins_030': dict(phantom='segpins', weights='unit', row_offset_frac=0.30),
    'offpins_045': dict(phantom='segpins', weights='unit', row_offset_frac=0.45),
    'offbody_045': dict(phantom='body', weights='unit', row_offset_frac=0.45),
}
RUN_CONDITIONS = ['offpins_015', 'offpins_030', 'offpins_045', 'offbody_045']

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(OUT_DIR, 'results')


def pin_segment_mask(slices, mode):
    """Axial density mask for the pins: 1 = dense segment, 0 = gap (body density)."""
    if mode == 'solid':
        return np.ones(slices, dtype=np.float32)
    if mode == 'seg':
        return (np.arange(slices) % (SEG_PERIOD_ON + SEG_PERIOD_OFF)
                < SEG_PERIOD_ON).astype(np.float32)
    if mode == 'rag':
        rng = np.random.RandomState(RAG_SEED)
        mask, on = [], True
        while len(mask) < slices:
            mask.extend([1.0 if on else 0.0] * rng.randint(1, 4))
            on = not on
        return np.array(mask[:slices], dtype=np.float32)
    raise ValueError(mode)


def make_phantom(recon_shape, fov_radius_vox, pin_mode):
    """Body cylinder; pin_mode None = body only, else 'solid'/'seg'/'rag' axial pins
    crossing the whole slab (dense segments at PIN_CONTRAST x body, gaps at body)."""
    rows, cols, slices = recon_shape
    i = np.arange(rows, dtype=np.float32)[:, None] - (rows - 1) / 2.0
    j = np.arange(cols, dtype=np.float32)[None, :] - (cols - 1) / 2.0
    r = np.sqrt(i ** 2 + j ** 2)
    body2d = r < BODY_RADIUS_FRAC * fov_radius_vox
    body_density = BODY_LINE_INTEGRAL / (2 * BODY_RADIUS_FRAC * fov_radius_vox)
    z = np.arange(slices, dtype=np.float32) - (slices - 1) / 2.0
    zmask = ((z > Z_LO_FRAC * (slices - 1) / 2.0) &
             (z < Z_HI_FRAC * (slices - 1) / 2.0)).astype(np.float32)
    phantom = (body2d.astype(np.float32) * body_density)[:, :, None] * zmask[None, None, :]
    if pin_mode is not None:
        pin2d = np.zeros((rows, cols), dtype=bool)
        for k in range(N_PINS):
            ang = 2 * np.pi * k / N_PINS
            ci = PIN_R_FRAC * fov_radius_vox * np.cos(ang)
            cj = PIN_R_FRAC * fov_radius_vox * np.sin(ang)
            pin2d |= np.sqrt((i - ci) ** 2 + (j - cj) ** 2) < PIN_RADIUS_VOX
        seg = pin_segment_mask(slices, pin_mode) * zmask
        pin_density = body_density * (1 + (PIN_CONTRAST - 1) * seg)  # per-slice
        phantom[pin2d] = pin_density[None, :] * zmask[None, :]
    return phantom


def seam_report(name, split_vol, ref, split_index):
    shape = ref.shape
    i = np.arange(shape[0], dtype=np.float32)[:, None] - (shape[0] - 1) / 2.0
    j = np.arange(shape[1], dtype=np.float32)[None, :] - (shape[1] - 1) / 2.0
    disk = np.sqrt(i ** 2 + j ** 2) < INTERIOR_RADIUS_FRAC * (min(shape[:2]) / 2.0)
    rms = np.sqrt(np.mean((split_vol - ref)[disk] ** 2, axis=0))
    lo = max(0, split_index - SEAM_VIEW_HALF_WIDTH)
    hi = min(shape[2], split_index + SEAM_VIEW_HALF_WIDTH + 1)
    bg = np.median(np.concatenate([rms[:lo], rms[hi:]]))
    print(f'\n=== {name}: split at slice {split_index}; background median RMS {bg:.3e}',
          flush=True)
    for s in range(lo, hi):
        marker = '  <-- split' if s == split_index else ''
        print(f'  slice {s:4d}: RMS {rms[s]:.3e}  ({rms[s]/bg:6.1f}x bg){marker}',
              flush=True)
    print(f'{name} seam max RMS: {rms[lo:hi].max():.3e}  ({rms[lo:hi].max()/bg:.1f}x bg)',
          flush=True)


def run_condition(tag):
    print(f'\n================ {tag} ================', flush=True)
    spec = CONDITIONS[tag]
    sinogram_shape = (NUM_VIEWS, NUM_DET_ROWS, NUM_DET_CHANNELS)
    angles = np.linspace(0, 2 * np.pi, NUM_VIEWS, endpoint=False)
    model = mj.ConeBeamModel(sinogram_shape, angles,
                             source_detector_dist=SDD_FACTOR * NUM_DET_CHANNELS,
                             source_iso_dist=SID_FACTOR * NUM_DET_CHANNELS)
    model.set_params(verbose=0)
    # Relative cut-vs-split misalignment: shift the DETECTOR by a fraction of a row
    # (recon grid stays centered with an odd slice count -> split_offset 0, so the
    # relative mismatch equals this row offset).  Applied before projection, so the
    # data stays perfectly consistent with the offset model.
    off_frac = spec.get('row_offset_frac', 0.0)
    if off_frac:
        model.set_params(det_row_offset=off_frac * model.get_params('delta_det_row'))
        print(f'det_row_offset = {off_frac} rows', flush=True)
    shape = model.get_params('recon_shape')
    delta_voxel = model.get_params('delta_voxel')
    fov_radius_vox = (NUM_DET_CHANNELS / 2.0) / model.get_magnification() / delta_voxel
    r_over_sid = (fov_radius_vox * delta_voxel) / (SID_FACTOR * NUM_DET_CHANNELS)
    print(f'recon shape {shape}; R/SID {r_over_sid:.3f}', flush=True)

    pin_mode = {'body': None, 'pins': 'solid', 'segpins': 'seg',
                'ragpins': 'rag'}[spec['phantom']]
    phantom = make_phantom(shape, fov_radius_vox, pin_mode)
    bin_mode = spec.get('bin_rows', None)
    if bin_mode is not None:
        # Project at BIN_FACTOR-finer row pitch over the same physical span, then bin
        # groups of BIN_FACTOR fine rows.  'linear' averages the line integrals (round 3:
        # provably still-consistent data).  'counts' averages TRANSMISSION then takes
        # -log, as real preprocessing does -- the Jensen gap makes the data genuinely
        # inconsistent wherever p varies within a bin.
        fine_shape = (NUM_VIEWS, NUM_DET_ROWS * BIN_FACTOR, NUM_DET_CHANNELS)
        fine_model = mj.ConeBeamModel(fine_shape, angles,
                                      source_detector_dist=SDD_FACTOR * NUM_DET_CHANNELS,
                                      source_iso_dist=SID_FACTOR * NUM_DET_CHANNELS)
        fine_model.set_params(verbose=0, delta_det_row=1.0 / BIN_FACTOR,
                              recon_shape=shape, delta_voxel=delta_voxel)
        fine_sino = np.asarray(fine_model.forward_project(phantom))
        grouped = fine_sino.reshape(NUM_VIEWS, NUM_DET_ROWS, BIN_FACTOR,
                                    NUM_DET_CHANNELS)
        if bin_mode == 'linear':
            sino = grouped.mean(axis=2)
        else:
            sino = -np.log(np.exp(-grouped.astype(np.float64)).mean(axis=2)
                           ).astype(np.float32)
        del fine_sino, grouped
    else:
        sino = np.asarray(model.forward_project(phantom))
    print(f'sino max line integral {sino.max():.2f}', flush=True)

    if spec['weights'] == 'unit':
        weights = None
    elif spec['weights'] == 'troot':
        weights = np.asarray(mj.gen_weights(sino, 'transmission_root'))
    elif spec['weights'] == 'troot_noise':
        sino, _ = tc.add_transmission_noise(sino, i0=NOISE_I0, seed=1)
        weights = np.asarray(mj.gen_weights(sino, 'transmission_root'))
    model.auto_set_regularization_params(sino)
    model.set_params(auto_regularize_flag=False)

    ref_path = os.path.join(RES_DIR, f'p2c8_{tag}_reference.npy')
    if not os.path.exists(ref_path):
        np.random.seed(0)
        ref, _ = model.recon(sino, weights=weights, max_iterations=NUM_ITERATIONS,
                             stop_threshold_change_pct=1e-9, print_logs=False)
        np.save(ref_path, np.asarray(ref))
    ref = np.load(ref_path)

    split_path = os.path.join(RES_DIR, f'p2c8_{tag}_no_taper.npy')
    if not os.path.exists(split_path):
        vol = split_recon(model, sino, h_sino=H_DEFAULT, h_recon=H_DEFAULT, taper=False,
                          num_iterations=NUM_ITERATIONS, base_weights=weights)
        np.save(split_path, vol)
    vol = np.load(split_path)

    slice_off = model.get_params('recon_slice_offset')
    dslice = model.get_params('voxel_slice_aspect') * delta_voxel
    split_index = int(np.round((shape[2] - 1) / 2.0 - slice_off / dslice))
    np.save(os.path.join(RES_DIR, f'p2c8_{tag}_truth.npy'), phantom)
    seam_report(f'{tag}/no_taper', vol, ref, split_index)


if __name__ == '__main__':
    os.makedirs(RES_DIR, exist_ok=True)
    for tag in RUN_CONDITIONS:
        run_condition(tag)
    print('\ndone: split_seam_lilly8x screening', flush=True)
