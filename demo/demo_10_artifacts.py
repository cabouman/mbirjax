# -*- coding: utf-8 -*-
"""**MBIRJAX: Artifacts demo — unmodeled physics, what it does, and which knob fixes it**

This script simulates a CT scan of a plastic slab holding a layer of metal
balls, with selectable unmodeled physics -- lateral or axial field-of-view
truncation, beam hardening (a polychromatic source), or combinations -- then
reconstructs with the standard linear model and displays ground truth and
reconstructions side by side.  Each regime prints what to look for and which
parameter remedies it (lateral/axial padding, the plastic-metal MAR
correction :func:`mbirjax.preprocess.recon_plastic_metal`).

Pick a regime by name below.  Tags compose left to right (later tags win),
and any individual value can be overridden through ``custom``:

    regime = ('beam_hardening_streaks', 'small', 'sparse')
    regime = ('cylinder_streaks', 'medium')
    regime = ('bga_like', 'large')
    custom = {'recon': {'lateral_pad_scale': 1.5}}   # turn a remedy on

Artifact tags (choose one or more; they compose):
    cylinder_streaks       object wider than the field of view + aggressive
                           regularization: fine vertical z-coherent striping.
    beam_hardening_cupping one large metal ball, polychromatic source: the
                           classic darkened-center / cupped profile.
    beam_hardening_streaks a grid of small metal balls: dark bands and streaks
                           between the balls.  ('beam_hardening' = alias.)
    lateral_fov            object wider than the field of view at default
                           settings: the bright boundary ring + interior bias.
    axial_fov              object taller than the axial coverage: flash and
                           ringing at the top and bottom slices.
    bga_like               lateral_fov + hardening + a dense grid: the
                           uncorrected real-scan situation.

Size tags (choose one):     small (64), medium (128, default), large (256).
Density tags (optional):    sparse (few big balls), dense (many small balls).

Every parameter is documented inline in ``BASE_PARAMS`` at the bottom of this
file; the regime tags are small overlays on it (``TAGS``), so what each tag
changes is directly readable there.

Notes on reading the beam-hardening results:
  - The MAR correction reinserts the metal as reconstructed, so it does not
    (and does not try to) restore the true metal values; the whole-volume
    NRMSE is metal-dominated.  The plastic-region NRMSE and the slice viewer
    are the meaningful comparisons.
  - The plastic/metal split is identified by the rays that miss the metal:
    dense ball grids (few metal-free rays) make the fit harder, and extra
    correction-reconstruction alternations can amplify segmentation errors on
    synthetic data -- both are interesting regimes to explore.

No command line arguments: edit the two lines below and run.
"""

import pprint

import numpy as np
import mbirjax as mj
import mbirjax.preprocess as mjp

# ----------------------------- user choice --------------------------------------
regime = ('beam_hardening_streaks', 'medium')  # tags: see the docstring above
custom = {}          # optional overrides, e.g. {'recon': {'sharpness': 2.0}}
# --------------------------------------------------------------------------------


# ---- A compact polychromatic model: Kramers spectrum + two-basis attenuation ----
def klein_nishina(e_kev):
    k = np.asarray(e_kev, dtype=np.float64) / 511.0
    return ((1 + k) / k ** 2 * (2 * (1 + k) / (1 + 2 * k) - np.log(1 + 2 * k) / k)
            + np.log(1 + 2 * k) / (2 * k) - (1 + 3 * k) / (1 + 2 * k) ** 2)


def make_spectrum(kvp, filtration_mm_al, n_bins=24, e_min=20.0, e_ref=60.0):
    edges = np.linspace(e_min, kvp, n_bins + 1)
    e = 0.5 * (edges[:-1] + edges[1:])
    mu_al = 0.075 * (0.35 * (e / e_ref) ** -3
                     + 0.65 * klein_nishina(e) / klein_nishina(e_ref))
    s = np.maximum(kvp - e, 0.0) / e * np.exp(-mu_al * filtration_mm_al)
    return e, s / s.sum()


def poly_sinogram(path_sinos, w_pe_list, energies, weights, severity, e_ref=60.0):
    """y = -ln sum_k w_k exp(-sum_i r_i(E_k) t_i); each material's ratio curve
    is normalized so the linear model is its tangent at t -> 0 (the severity
    dial bends curvature only).  severity 0 returns the linear sinogram."""
    if severity == 0.0:
        return sum(path_sinos).astype(np.float32)
    ratios = []
    for w_pe in w_pe_list:
        r = w_pe * (energies / e_ref) ** -3 \
            + (1 - w_pe) * klein_nishina(energies) / klein_nishina(e_ref)
        rs = (1.0 - severity) + severity * r
        ratios.append(rs / np.sum(weights * rs))
    total = np.zeros_like(path_sinos[0], dtype=np.float64)
    for k, wk in enumerate(weights):
        expo = sum(float(rs[k]) * t for t, rs in zip(path_sinos, ratios))
        total += wk * np.exp(-expo)
    return (-np.log(total)).astype(np.float32)


# ---- Phantom: slab + one layer of metal balls, one map per material -------------
def ball_grid_materials(recon_shape, base_n, d):
    """Material maps on the GENERATION grid.  Ball sizes are fractions of the
    base image size (base_n) so they do not change when the grid is enlarged;
    the slab width is a fraction of the generation grid (that is what makes it
    overflow the base field of view when data.gen_lateral_scale > 1)."""
    rows, cols, slices = recon_shape
    slab = np.zeros(recon_shape, dtype=np.float32)
    metals = [np.zeros(recon_shape, dtype=np.float32) for _ in d['metal_values']]
    rc, cc, zc = (rows - 1) / 2.0, (cols - 1) / 2.0, (slices - 1) / 2.0
    half = 0.5 * d['slab_xy_frac'] * min(rows, cols)
    half_z = 0.5 * d['slab_z_frac'] * slices
    slab[int(rc - half):int(rc + half) + 1, int(cc - half):int(cc + half) + 1,
         int(zc - half_z):int(zc + half_z) + 1] = d['slab_value']

    pitch = d['ball_pitch_frac'] * base_n
    radius = d['ball_radius_frac'] * base_n
    k = int((half - 2 * radius - 2) // pitch)
    centers = pitch * np.arange(-k, k + 1)
    print(f'Phantom: {len(centers) ** 2} ball(s) of radius {radius:.1f} voxels, '
          f'{len(d["metal_values"])} metal type(s)')
    zball = (slices - 1) * d['ball_layer_z_frac']
    rad_i = int(np.ceil(radius)) + 1
    for i, br in enumerate(centers + rc):
        for j, bc in enumerate(centers + cc):
            m = (i + j) % len(d['metal_values'])
            rr = np.arange(int(br) - rad_i, int(br) + rad_i + 1)
            cw = np.arange(int(bc) - rad_i, int(bc) + rad_i + 1)
            zw = np.arange(max(0, int(zball) - rad_i),
                           min(slices, int(zball) + rad_i + 1))
            sphere = ((rr - br)[:, None, None] ** 2 + (cw - bc)[None, :, None] ** 2
                      + (zw - zball)[None, None, :] ** 2) <= radius ** 2
            window = slab[np.ix_(rr, cw, zw)]
            window[sphere] = 0.0
            slab[np.ix_(rr, cw, zw)] = window
            window = metals[m][np.ix_(rr, cw, zw)]
            window[sphere] = d['metal_values'][m]
            metals[m][np.ix_(rr, cw, zw)] = window
    return slab, metals


def make_model(g, lateral_scale=1.0, axial_pad=0.0):
    """A cone or parallel model, optionally laterally enlarged / axially padded."""
    sinogram_shape = (g['num_views'], g['num_det_rows'], g['num_det_channels'])
    if g['model_type'] == 'cone':
        angles = np.linspace(0, 2 * np.pi, g['num_views'], endpoint=False)
        sdd = g['sdd_factor'] * g['num_det_channels']
        m = mj.ConeBeamModel(sinogram_shape, angles,
                             source_detector_dist=sdd, source_iso_dist=sdd / 2)
        if np.max(axial_pad) > 0:
            m.set_params(axial_pad_fraction=axial_pad)
    else:
        angles = np.linspace(0, np.pi, g['num_views'], endpoint=False)
        m = mj.ParallelBeamModel(sinogram_shape, angles)
    if lateral_scale > 1.0:
        m.scale_recon_shape(lateral_scale, lateral_scale)
    return m


def central_crop_pair(a, b):
    """Central crops of two volumes to their common shape (for error metrics)."""
    shape = tuple(min(sa, sb) for sa, sb in zip(a.shape, b.shape))
    def crop(v):
        starts = [(sv - s) // 2 for sv, s in zip(v.shape, shape)]
        return v[tuple(slice(st, st + s) for st, s in zip(starts, shape))]
    return crop(a), crop(b)


# ============================== the pipeline =====================================
def main():
    p = get_params(*regime, custom=custom)
    print('Resolved parameters:')
    pprint.pprint(p, sort_dicts=False)

    g, d, h, r = p['geometry'], p['data'], p['hardening'], p['recon']

    # Generation model: the grid the TRUE object lives on (possibly beyond the
    # base field of view laterally and/or axially).
    gen_model = make_model(g, lateral_scale=d['gen_lateral_scale'],
                           axial_pad=(1.0 if d['gen_axial_overflow'] else 0.0))
    gen_shape = gen_model.get_params('recon_shape')
    print(f"Generation grid {gen_shape}; building the phantom and forward "
          f"projecting each material...")
    slab_map, metal_maps = ball_grid_materials(gen_shape, g['num_det_channels'], d)
    path_sinos = [np.asarray(gen_model.forward_project(m), dtype=np.float64)
                  for m in [slab_map] + metal_maps]
    scale = d['target_max_attenuation'] / float(sum(path_sinos).max())
    path_sinos = [t * scale for t in path_sinos]
    ground_truth = (slab_map + sum(metal_maps)) * scale

    print('Applying the polychromatic (beam hardening) model...')
    energies, spec_w = make_spectrum(h['kvp'], h['filtration_mm_al'])
    sino = poly_sinogram(path_sinos, [h['plastic_w_pe']] + list(h['metal_w_pe']),
                         energies, spec_w, h['severity'])
    if p['noise']['i0'] is not None:
        rng = np.random.default_rng(0)
        counts = rng.poisson(p['noise']['i0'] * np.exp(-np.float64(sino)))
        sino = (-np.log(np.maximum(counts, 1) / p['noise']['i0'])).astype(np.float32)
    weights = mj.gen_weights(sino, weight_type='transmission_root')

    if p['display']['show_sinogram']:
        # Nonblocking: this window stays open beside the reconstruction viewer
        # below and becomes fully interactive once that (blocking) viewer opens.
        mj.slice_viewer(sino, slice_axis=0, slice_label='View', block=False,
                        title=f"Simulated sinogram {regime} "
                              f"({g['model_type']} beam)")

    # Reconstruction model: the base grid, plus any remedies the user enabled.
    ct_model = make_model(g, lateral_scale=r['lateral_pad_scale'],
                          axial_pad=r['axial_pad_fraction'])
    ct_model.set_params(sharpness=r['sharpness'], snr_db=r['snr_db'])
    print(f"Reconstruction grid {ct_model.get_params('recon_shape')}; "
          "standard reconstruction (linear model)...")
    recon_std, _ = ct_model.recon(sino, weights=weights,
                                  max_iterations=r['max_iterations'])
    recons = [ground_truth, np.asarray(recon_std)]
    labels = ['ground truth', 'standard recon']

    m = p['mar']
    if m['use_mar']:
        print('MAR reconstruction (plastic-metal beam hardening correction)...')
        recon_mar = mjp.recon_plastic_metal(
            ct_model, sino, weights, num_BH_iterations=m['num_bh_iterations'],
            num_metal=m['num_metal'], order=m['order'], alpha=m['alpha'],
            beta=m['beta'], gamma=m['gamma'],
            max_iterations=r['max_iterations'])
        recons.append(np.asarray(recon_mar))
        labels.append('MAR recon')

    # Error metrics on the overlap of each recon with the ground truth.
    metal_floor = 0.5 * max(d['metal_values']) * scale
    for rec, lab in zip(recons[1:], labels[1:]):
        gt_c, rec_c = central_crop_pair(ground_truth, rec)
        err = np.linalg.norm(rec_c - gt_c) / np.linalg.norm(gt_c)
        plastic = (gt_c > 0) & (gt_c < metal_floor)
        perr = (np.linalg.norm((rec_c - gt_c)[plastic])
                / np.linalg.norm(gt_c[plastic]))
        print(f'  {lab}: NRMSE vs ground truth = {err:.4f} '
              f'(plastic region only: {perr:.4f})')

    # The what-to-look-for guidance goes into the viewer title (and the
    # terminal), wrapped to keep the figure header readable.
    import textwrap
    guidance = []
    for tag in regime:
        if tag in WHAT_TO_LOOK_FOR:
            print(f'[{tag}] {WHAT_TO_LOOK_FOR[tag]}')
            guidance.extend(textwrap.wrap(f'{tag}: {WHAT_TO_LOOK_FOR[tag]}',
                                          width=110))
    title = '\n'.join([f"Artifacts demo {regime} ({g['model_type']} beam)"]
                      + guidance)
    mj.slice_viewer(*recons, slice_label=labels, title=title,
                    vmin=0.0, vmax=float(ground_truth.max()))


# =========================== regimes and parameters ==============================
def _deep_update(dst, src):
    for key, val in src.items():
        if isinstance(val, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], val)
        else:
            dst[key] = val
    return dst


BASE_PARAMS = {
    'geometry': dict(
        model_type='cone',          # 'cone' or 'parallel'
        num_views=100,              # projection views (cone: over 2*pi)
        num_det_rows=128,           # detector rows (axial direction)
        num_det_channels=128,       # detector channels (lateral direction)
        sdd_factor=2.5),            # cone: SDD in units of num_det_channels
    'data': dict(
        gen_lateral_scale=1.0,      # >1: the true object overflows the lateral FoV
        gen_axial_overflow=False,   # True: the object extends beyond axial coverage
        slab_xy_frac=0.6,           # slab width, fraction of the GENERATION grid
        slab_z_frac=0.5,            # slab height, fraction of the generation slices
        slab_value=1.0,             # plastic attenuation at the reference energy
        metal_values=(6.0, 4.5),    # attenuation per metal type (one ball class each)
        ball_pitch_frac=0.12,       # ball lattice pitch, fraction of the base size
        ball_radius_frac=0.055,     # ball radius, fraction of the base size
        ball_layer_z_frac=0.5,      # ball layer height, fraction of the volume
        target_max_attenuation=5.0),  # scale so the ideal sinogram peaks near this
    'hardening': dict(
        severity=0.0,               # 0 = linear data ... 1 = strong beam hardening
        plastic_w_pe=0.05,          # photoelectric weight (hardening factor) ...
        metal_w_pe=(0.8, 0.55),     # ... per material at the reference energy
        kvp=140.0,                  # source spectrum endpoint (keV)
        filtration_mm_al=2.0),      # aluminum filtration (mm)
    'noise': dict(
        i0=1.0e4),                  # photons/ray Poisson noise; None = noiseless
    'recon': dict(
        sharpness=1.0,              # regularization sharpness
        snr_db=30.0,                # regularization snr_db
        max_iterations=15,          # MBIR iterations
        lateral_pad_scale=1.0,      # >1 = the lateral-truncation remedy (padding)
        axial_pad_fraction=0.0),    # 0..1 or (top, bottom): the axial remedy
    'mar': dict(
        use_mar=False,              # also run recon_plastic_metal
        num_metal=2,                # metal classes to segment and correct
        order=3,                    # beam-hardening polynomial degree
        num_bh_iterations=1,        # correction <-> reconstruction alternations
        alpha=1.0, beta=0.002, gamma=0.1),  # see recon_plastic_metal
    'display': dict(
        show_sinogram=True),        # open a (nonblocking) sinogram viewer too
}

TAGS = {
    # ---- artifact tags (one or more; they compose) ----
    'cylinder_streaks': {
        'data': dict(gen_lateral_scale=1.5, slab_xy_frac=0.72),
        'recon': dict(sharpness=1.5, snr_db=35.0),
    },
    'beam_hardening_cupping': {     # one LARGE ball: the classic cupped profile
        'data': dict(ball_pitch_frac=2.0, ball_radius_frac=0.12,
                     metal_values=(6.0,)),
        'hardening': dict(severity=0.7, metal_w_pe=(0.8,)),
        'mar': dict(use_mar=True, num_metal=1),
    },
    'beam_hardening_streaks': {     # many small balls: interball dark bands
        'data': dict(ball_pitch_frac=0.10, ball_radius_frac=0.035),
        'hardening': dict(severity=0.7),
        'mar': dict(use_mar=True),
    },
    'lateral_fov': {
        'data': dict(gen_lateral_scale=1.5, slab_xy_frac=0.72),
    },
    'axial_fov': {
        'data': dict(gen_axial_overflow=True, slab_z_frac=0.9),
    },
    'bga_like': {   # the real-scan mimic: truncated + hardened + dense grid
        'data': dict(gen_lateral_scale=1.5, slab_xy_frac=0.72,
                     ball_pitch_frac=0.12, ball_radius_frac=0.055),
        'hardening': dict(severity=0.5),
        'recon': dict(sharpness=1.5, snr_db=35.0),
        'mar': dict(use_mar=True),
    },
    # ---- size tags (pick one; default medium) ----
    'small': {'geometry': dict(num_views=40, num_det_rows=64,
                               num_det_channels=64),
              'recon': dict(max_iterations=10)},
    'medium': {},
    'large': {'geometry': dict(num_views=200, num_det_rows=256,
                               num_det_channels=256)},
    # ---- ball-density tags (pick one; regimes carry their own default) ----
    'sparse': {'data': dict(ball_pitch_frac=0.16, ball_radius_frac=0.05)},
    'dense': {'data': dict(ball_pitch_frac=0.10, ball_radius_frac=0.04)},
}
TAGS['beam_hardening'] = TAGS['beam_hardening_streaks']   # alias

WHAT_TO_LOOK_FOR = {
    'cylinder_streaks': ('transpose to the (x,z) view: fine vertical striping '
                         'across the slab plus bright flash at the lateral edges.'
                         "  Remedy: custom = {'recon': {'lateral_pad_scale': 1.5}}."),
    'beam_hardening_cupping': ('axial view at the ball layer: the ball is darker '
                               'toward its center and the slab darkens around it. '
                               'Narrow the intensity range to see it clearly.  The MAR '
                               'recon restores the slab level; plastic-region NRMSE is '
                               'the fair number (the metal is reinserted as '
                               'reconstructed).'),
    'beam_hardening_streaks': ('axial view at the ball layer: dark bands between '
                               'balls and a darkened slab.  Compare the MAR recon; '
                               'plastic-region NRMSE is the fair number.'),
    'lateral_fov': ('axial view: a bright ring at the reconstruction boundary '
                    'and an elevated interior.'
                    "  Remedy: custom = {'recon': {'lateral_pad_scale': 1.5}}."),
    'axial_fov': ('transpose to (x,z): bright flash and ringing at the top and '
                  'bottom slices.'
                  "  Remedy: custom = {'recon': {'axial_pad_fraction': 1.0}}."),
    'bga_like': ('truncation + hardening at once -- the uncorrected real-scan '
                 'situation.  Explore the remedies one at a time.'),
}
WHAT_TO_LOOK_FOR['beam_hardening'] = WHAT_TO_LOOK_FOR['beam_hardening_streaks']


def get_params(*tags, custom=None):
    """Return the parameter dict for one or more regime tags.

    Tags compose left to right (later tags win); ``custom`` (a possibly-nested
    dict of the same shape) is applied last.

    Artifact tags:   cylinder_streaks, beam_hardening_cupping,
                     beam_hardening_streaks (alias: beam_hardening),
                     lateral_fov, axial_fov, bga_like
    Size tags:       small, medium (default), large
    Density tags:    sparse, dense
    """
    import copy
    params = copy.deepcopy(BASE_PARAMS)
    for tag in tags:
        if tag not in TAGS:
            raise ValueError(f'unknown tag {tag!r}; valid tags: {sorted(TAGS)}')
        _deep_update(params, copy.deepcopy(TAGS[tag]))
    if custom:
        _deep_update(params, custom)
    return params


if __name__ == '__main__':
    main()
