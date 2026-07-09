"""P2b: radial (channel) truncation -- padding-scale sweeps across overshoots and regimes.

Design (plan doc "Phase 2 plan" P2b + the 2026-07-08 design discussion): unlike z, the
radial case has NO visibility bound (any material, however far out, is measured in the
views where it lies near the source-axis line), so the padding knee is empirical and the
policy must track the object's overshoot.  The only hard stop is MECHANICAL: the rotating
object must clear source and detector, R_obj < min(SID, SDD-SID) -- 8x the FoV radius in
the base geometry -- which the 'extreme' section approaches deliberately.

Sections (each independently runnable; a cluster job runs a subset by editing
run_sections below):
  core          overshoot 1.25x, default regime: scales {1.0, 1.1, 1.2, 1.35~cover, 1.5}
                + channel-taper-only (falsification control) + pad1.2+taper (combo check)
  overshoot_1p1 overshoot 1.1x:  scales {1.0, 1.05, 1.2~cover}
  overshoot_1p5 overshoot 1.5x:  scales {1.0, 1.25, 1.6~cover}
  extreme       overshoot 4.0x (half the rotation bound): scales {1.0, 2.5, 4.1~cover}
  widefan       overshoot 1.25x, SID/SDD shortened at fixed magnification (R/SID
                0.125->0.2): scales {1.0, 1.2, 1.35}
  sharp         overshoot 1.25x, sharpness 2 / snr_db 35, 160 iterations (the axial
                convergence lesson): scales {1.0, 1.2, 1.35}
  noise         overshoot 1.25x, photon noise (i0 1e4) + transmission weights at default
                regularization: scales {1.0, 1.2, 1.35}
Regime riders are INDEPENDENT single-variable probes (don't stack noise on changed
regularization -- Greg).

Run inside the mbirjax conda env (CPU or GPU); all knobs below (no CLI args).  Results land
in results/p2b_*; set make_figures=True (with run_sections=[]) to build figures from
whatever results are present.
"""

import os
import numpy as np
import mbirjax as mj  # noqa: F401 -- must precede anything that touches jax
import matplotlib.pyplot as plt
import truncation_common as tc

# ---------------------------------------------------------------------------
# Run control (cluster jobs sed run_sections/make_figures per job)
# ---------------------------------------------------------------------------
run_sections = ['core', 'overshoot_1p1', 'overshoot_1p5', 'extreme', 'widefan',
                'sharp', 'noise']
make_figures = True
skip_existing = True              # skip variants whose results/ files already exist

# ---------------------------------------------------------------------------
# Shared problem definition (the Phase 1 lateral setup)
# ---------------------------------------------------------------------------
NUM_VIEWS = 128
NUM_DET_ROWS = 32
NUM_DET_CHANNELS = 128
SID_FACTOR = 2.0                  # SID = 2.0 * channels
Z_LO_FRAC, Z_HI_FRAC = -0.75, 0.75    # contained in z: radial truncation only
TARGET_LINE_INTEGRAL = 2.0
INTERIOR_RADIUS_FRAC = 0.85
END_SLICE_MARGIN = 4
CHANNEL_TAPER_WIDTH = 16          # quarter-sine over this many channels, EACH side
NOISE_I0 = 1e4
NOISE_SEED = 1

# Sections: regime parameters + (variant label, pad scale, channel taper on/off).
SECTIONS = {
    'core': dict(rf=1.25, sdd_factor=4.0, sharpness=0.0, snr_db=None, noise=False,
                 iters=40, variants=[('pad_1.00', 1.00, False), ('pad_1.10', 1.10, False),
                                     ('pad_1.20', 1.20, False), ('pad_1.35', 1.35, False),
                                     ('pad_1.50', 1.50, False), ('taper_only', 1.00, True),
                                     ('pad_1.20+taper', 1.20, True)]),
    # Each overshoot's scale set brackets its own "cover" (~overshoot + a small margin)
    # AND goes past it, so every knee curve can show the descent, the knee, and the
    # plateau/reversal beyond it (Greg, 2026-07-08).
    'overshoot_1p1': dict(rf=1.10, sdd_factor=4.0, sharpness=0.0, snr_db=None, noise=False,
                          iters=40, variants=[('pad_1.00', 1.00, False),
                                              ('pad_1.05', 1.05, False),
                                              ('pad_1.12', 1.12, False),
                                              ('pad_1.20', 1.20, False),
                                              ('pad_1.35', 1.35, False)]),
    'overshoot_1p5': dict(rf=1.50, sdd_factor=4.0, sharpness=0.0, snr_db=None, noise=False,
                          iters=40, variants=[('pad_1.00', 1.00, False),
                                              ('pad_1.25', 1.25, False),
                                              ('pad_1.45', 1.45, False),
                                              ('pad_1.60', 1.60, False),
                                              ('pad_1.85', 1.85, False),
                                              ('pad_2.10', 2.10, False),
                                              ('pad_2.40', 2.40, False),
                                              ('pad_2.80', 2.80, False)]),
    'extreme': dict(rf=4.00, sdd_factor=4.0, sharpness=0.0, snr_db=None, noise=False,
                    iters=40, variants=[('pad_1.00', 1.00, False),
                                        ('pad_2.50', 2.50, False),
                                        ('pad_4.10', 4.10, False),
                                        ('pad_4.60', 4.60, False)]),
    # Wide fan via SHORTER distances at the same magnification (SID 2.0C->1.25C, SDD
    # 4C->2.5C: R/SID 0.125->0.2).  Shrinking only SDD fails the rotation bound: with
    # SID 2C, SDD 2.5C the detector clearance is 0.5C and a 1.25x object EXACTLY touches
    # the detector -- the bound check caught it (job p2b_c, 2026-07-08).
    'widefan': dict(rf=1.25, sdd_factor=2.5, sid_factor=1.25, sharpness=0.0, snr_db=None,
                    noise=False, iters=40, variants=[('pad_1.00', 1.00, False),
                                                     ('pad_1.20', 1.20, False),
                                                     ('pad_1.35', 1.35, False),
                                                     ('pad_1.50', 1.50, False)]),
    'sharp': dict(rf=1.25, sdd_factor=4.0, sharpness=2.0, snr_db=35.0, noise=False,
                  iters=160, variants=[('pad_1.00', 1.00, False),
                                       ('pad_1.20', 1.20, False),
                                       ('pad_1.35', 1.35, False),
                                       ('pad_1.50', 1.50, False)]),
    'noise': dict(rf=1.25, sdd_factor=4.0, sharpness=0.0, snr_db=None, noise=True,
                  iters=40, variants=[('pad_1.00', 1.00, False),
                                      ('pad_1.20', 1.20, False),
                                      ('pad_1.35', 1.35, False),
                                      ('pad_1.50', 1.50, False)]),
}

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(OUT_DIR, 'figures')
RES_DIR = os.path.join(OUT_DIR, 'results')


def run_section(name, cfg):
    """Build the section's models/phantom once, then run its variants."""
    print(f'\n================ {name} (overshoot {cfg["rf"]}x) ================', flush=True)
    sinogram_shape = (NUM_VIEWS, NUM_DET_ROWS, NUM_DET_CHANNELS)
    angles = np.linspace(0, 2 * np.pi, NUM_VIEWS, endpoint=False)
    sdd = cfg['sdd_factor'] * NUM_DET_CHANNELS
    sid = cfg.get('sid_factor', SID_FACTOR) * NUM_DET_CHANNELS

    # Physical rotation bound check: the phantom must clear source and detector.
    probe = mj.ConeBeamModel(sinogram_shape, angles, source_detector_dist=sdd,
                             source_iso_dist=sid)
    small_shape = probe.get_params('recon_shape')
    delta_voxel = probe.get_params('delta_voxel')
    fov_radius_phys = min(small_shape[0], small_shape[1]) * delta_voxel / 2.0
    obj_radius_phys = cfg['rf'] * fov_radius_phys
    rot_bound = min(sid, sdd - sid)
    print(f'object radius {obj_radius_phys:.0f} vs rotation bound {rot_bound:.0f} '
          f'(max overshoot {rot_bound / fov_radius_phys:.1f}x FoV)', flush=True)
    assert obj_radius_phys < rot_bound, 'phantom would not clear source/detector'

    lateral_margin = int(np.ceil((cfg['rf'] - 1.0) * small_shape[1] / 2.0)) + 6
    recon_model, truth_model = tc.make_cone_models(
        sinogram_shape, angles, sdd, sid, lateral_margin=lateral_margin, slice_margin=0,
        sharpness=cfg['sharpness'])
    if cfg['snr_db'] is not None:
        recon_model.set_params(snr_db=cfg['snr_db'])
    small_shape = recon_model.get_params('recon_shape')
    big_shape = truth_model.get_params('recon_shape')

    phantom_big = tc.build_phantom(big_shape, small_shape, delta_voxel, cfg['rf'],
                                   Z_LO_FRAC, Z_HI_FRAC, TARGET_LINE_INTEGRAL)
    truth_small = tc.center_crop(phantom_big, small_shape)
    print(f'small {small_shape}, ground truth phantom {big_shape}; forward-projecting...',
          flush=True)
    sinogram = np.asarray(truth_model.forward_project(phantom_big))

    if cfg['noise']:
        sino_used, base_weights = tc.add_transmission_noise(sinogram, i0=NOISE_I0,
                                                            seed=NOISE_SEED)
    else:
        sino_used, base_weights = sinogram, None

    masks = tc.make_masks(small_shape, INTERIOR_RADIUS_FRAC, END_SLICE_MARGIN)
    np.save(os.path.join(RES_DIR, f'p2b_{name}_truth.npy'), truth_small)

    for label, scale, taper in cfg['variants']:
        metrics_path = os.path.join(RES_DIR,
                                    f'p2b_{name}_{label.replace("+", "_")}_metrics.npz')
        if skip_existing and os.path.exists(metrics_path):
            print(f'--- {name}/{label}: results exist, skipped', flush=True)
            continue
        model = (recon_model if scale == 1.0
                 else tc.make_padded_model(recon_model, pad_scale_lateral=scale))
        weights = base_weights
        if taper:
            ch_taper = tc.make_channel_taper_weights(sinogram_shape,
                                                     k_each_side=CHANNEL_TAPER_WIDTH)
            weights = ch_taper if weights is None else weights * ch_taper
        print(f'--- {name}/{label}: recon shape {model.get_params("recon_shape")}',
              flush=True)
        metrics, snaps = tc.run_tracked_recon(model, sino_used, truth_small, masks,
                                              cfg['iters'], label=label, weights=weights)
        np.savez(os.path.join(RES_DIR, f'p2b_{name}_{label.replace("+", "_")}_metrics.npz'),
                 **{key: np.array(vals) for key, vals in metrics.items()})
        np.save(os.path.join(RES_DIR, f'p2b_{name}_{label.replace("+", "_")}_final.npy'),
                snaps[cfg['iters'] - 1])


def load_final_metrics(name, label):
    """(ring, interior) final NRMSE for one variant, or None if not yet run."""
    path = os.path.join(RES_DIR, f'p2b_{name}_{label.replace("+", "_")}_metrics.npz')
    if not os.path.exists(path):
        return None
    saved = np.load(path)
    return float(saved['nrmse_ring'][-1]), float(saved['nrmse_interior'][-1])


def make_all_figures():
    """Knee curves + montages from whatever section results are present."""
    # --- knee curves: one figure for the overshoot axis, one for the regimes ---
    def knee_figure(groups, title, path):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for group_label, name, variants in groups:
            scales, rings, interiors = [], [], []
            for label, scale, taper in variants:
                if taper:
                    continue  # taper variants annotate montages, not knee curves
                vals = load_final_metrics(name, label)
                if vals is None:
                    continue
                scales.append(scale); rings.append(vals[0]); interiors.append(vals[1])
            if not scales:
                print(f'  (no results yet for {name}; skipped in {title})')
                continue
            axes[0].plot(scales, rings, 'o-', label=group_label)
            axes[1].plot(scales, interiors, 'o-', label=group_label)
        for ax, ylabel in zip(axes, ['ring NRMSE', 'interior NRMSE']):
            ax.set_xlabel('padding scale'); ax.set_ylabel(ylabel)
            ax.set_yscale('log'); ax.grid(True, alpha=0.3); ax.legend()
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    knee_figure([(f'overshoot {SECTIONS[n]["rf"]}x', n, SECTIONS[n]['variants'])
                 for n in ['overshoot_1p1', 'core', 'overshoot_1p5', 'extreme']],
                'P2b knee curves by overshoot (default regime)',
                os.path.join(FIG_DIR, 'p2b_knee_overshoot.png'))
    knee_figure([('default', 'core', SECTIONS['core']['variants']),
                 ('wide fan', 'widefan', SECTIONS['widefan']['variants']),
                 ('sharp (160 it)', 'sharp', SECTIONS['sharp']['variants']),
                 ('noise + weights', 'noise', SECTIONS['noise']['variants'])],
                'P2b knee curves by regime (overshoot 1.25x)',
                os.path.join(FIG_DIR, 'p2b_knee_regimes.png'))

    # --- montages: core and extreme sections, center slice ---
    for name, picks in [('core', ['pad_1.00', 'pad_1.20', 'pad_1.35', 'taper_only']),
                        ('extreme', ['pad_1.00', 'pad_2.50', 'pad_4.10'])]:
        truth_path = os.path.join(RES_DIR, f'p2b_{name}_truth.npy')
        if not os.path.exists(truth_path):
            print(f'  (no results yet for {name}; montage skipped)')
            continue
        truth_small = np.load(truth_path)
        masks = tc.make_masks(truth_small.shape, INTERIOR_RADIUS_FRAC, END_SLICE_MARGIN)
        display = {}
        for label in picks:
            vals = load_final_metrics(name, label)
            fin = os.path.join(RES_DIR, f'p2b_{name}_{label.replace("+", "_")}_final.npy')
            if vals is None or not os.path.exists(fin):
                continue
            display[f'recon: {label}\nring NRMSE {vals[0]:.3f}'] = np.load(fin)
        if display:
            tc.save_slice_montage(truth_small, display, axis=2,
                                  index=truth_small.shape[2] // 2,
                                  title=f'P2b {name}: center slice',
                                  path=os.path.join(FIG_DIR, f'p2b_{name}_center.png'),
                                  region_mask=masks['ring'],
                                  region_label='dashed = ring-NRMSE region')
    print(f'Figures in {FIG_DIR} (p2b_*)')


if __name__ == '__main__':
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)
    for name in run_sections:
        run_section(name, SECTIONS[name])
    if make_figures:
        make_all_figures()
    print('\nDone: sections ' + (', '.join(run_sections) if run_sections else '(none)'))
