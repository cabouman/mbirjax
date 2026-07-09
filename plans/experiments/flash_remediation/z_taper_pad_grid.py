"""P2a: axial (SiC-like) case -- row taper vs z-padding vs combinations.

Same one-sided z-truncation setup as z_truncation_repro.py (laminated cylinder extending
past the top of the covered slab).  Seven variants ("variants", not "cells"/"arms" -- Greg):

  padding {none, partial, full, overfull} x row taper {off, on}

with the geometry-derived quantities from truncation_common.cone_axial_geometry:
- "full" padding uses the derived slice scale: for a circular orbit no measured ray
  reaches |z| > h_max*(SID+R)/SDD, so full z-padding is GEOMETRY-BOUNDED (scale <=
  (SID+R)/SID + psf margin) -- there is no "object too long to pad" case in z.
- the taper width is the number of edge rows whose rays exit the (that variant's) slab.
- "overfull" (a much larger scale) is a control: it should TIE with "full" if the bound
  is right.  "full + taper" forces the none-variant's taper width onto a slab that needs
  none, testing whether tapering when unnecessary costs quality.

The plan's original "far-overshoot" recon case collapses under the same bound: an object
extending far past max_visible_z produces the IDENTICAL sinogram to one cut at the bound,
so it is checked here as a cheap forward-projection identity instead of a recon campaign.

Hypotheses under test (plan doc "Phase 2 plan", P2a):
  H1 taper-alone softens the flash but blurs real end-slice structure (laminate);
  H2 with full padding the taper is unnecessary;
  H3 the taper+pad combo wins only at partial padding.

Run inside the mbirjax conda env; all knobs below (no CLI args).  Outputs land in
figures/ (PNGs, prefix p2a_) and results/ next to this script.
"""

import os
import numpy as np
import mbirjax as mj  # noqa: F401 -- must precede anything that touches jax
import truncation_common as tc

if __name__ == '__main__':

    # #### Problem size (identical to z_truncation_repro.py)
    num_views = 128
    num_det_rows = 64
    num_det_channels = 96
    source_detector_dist = 4.0 * num_det_channels
    source_iso_dist = 2.0 * num_det_channels   # magnification 2

    # #### Phantom (identical to z_truncation_repro.py: one-sided axial truncation)
    radius_frac = 0.75
    z_lo_frac, z_hi_frac = -0.80, 1.60
    laminate_period = 3
    target_line_integral = 2.0
    z_hi_far = 4.0                 # the far-overshoot identity check (forward-only)

    # #### Recon / metrics
    num_iterations = 40
    snapshot_iters = (9, 19)
    interior_radius_frac = 0.85
    end_slice_margin = 6
    psf_margin = 2                 # slices/rows of safety on the geometry-derived widths
    figures_only = False           # True: skip the recons, rebuild figures from results/p2a_*

    out_dir = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(out_dir, 'figures')
    res_dir = os.path.join(out_dir, 'results')
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    # ---- Base models and phantom (as in Phase 1) ----
    sinogram_shape = (num_views, num_det_rows, num_det_channels)
    angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)

    probe = mj.ConeBeamModel(sinogram_shape, angles, source_detector_dist=source_detector_dist,
                             source_iso_dist=source_iso_dist)
    small_slices = probe.get_params('recon_shape')[2]
    half_slab_vox = small_slices / 2.0
    slice_margin = int(np.ceil((max(abs(z_lo_frac), abs(z_hi_far)) - 1.0) * half_slab_vox)) + 6

    recon_model, truth_model = tc.make_cone_models(
        sinogram_shape, angles, source_detector_dist, source_iso_dist,
        lateral_margin=0, slice_margin=slice_margin)
    small_shape = recon_model.get_params('recon_shape')
    big_shape = truth_model.get_params('recon_shape')
    delta_voxel = recon_model.get_params('delta_voxel')

    geo = tc.cone_axial_geometry(recon_model, psf_margin=psf_margin)
    print(f'small (default) recon shape: {small_shape}, ground truth phantom: {big_shape}')
    print(f'axial geometry: half_slab {geo["half_slab"]:.1f}, max visible |z| '
          f'{geo["max_visible_z"]:.1f} (ratio {geo["max_visible_z"]/geo["half_slab"]:.3f}), '
          f'full pad scale {geo["full_pad_scale"]:.3f}, taper rows {geo["taper_rows"]}')

    phantom_big = tc.build_phantom(big_shape, small_shape, delta_voxel, radius_frac,
                                   z_lo_frac, z_hi_frac, target_line_integral,
                                   laminate_period=laminate_period)
    truth_small = tc.center_crop(phantom_big, small_shape)
    print('Forward-projecting the ground truth phantom...')
    sinogram = np.asarray(truth_model.forward_project(phantom_big))

    # ---- Far-overshoot identity check (forward-only; replaces the far recon case) ----
    # Material beyond max_visible_z never projects, so extending the object from z_hi to
    # z_hi_far must leave the sinogram unchanged (up to the projector's footprint tails).
    phantom_far = tc.build_phantom(big_shape, small_shape, delta_voxel, radius_frac,
                                   z_lo_frac, z_hi_far, target_line_integral,
                                   laminate_period=laminate_period)
    sinogram_far = np.asarray(truth_model.forward_project(phantom_far))
    far_diff = float(np.max(np.abs(sinogram_far - sinogram)))
    print(f'far-overshoot identity: max |sino(z_hi={z_hi_far}) - sino(z_hi={z_hi_frac})| '
          f'= {far_diff:.2e} (bound predicts ~0; sino max {sinogram.max():.2f})')
    del phantom_far, sinogram_far

    # ---- Variant grid ----
    # The truncated side is the HIGH-z side; measured in Phase 1: detector row -1 views
    # high z here, so tapers apply to the LAST k rows.
    masks = tc.make_masks(small_shape, interior_radius_frac, end_slice_margin)
    partial_pad_scale = 1.0 + 0.5 * (geo['full_pad_scale'] - 1.0)

    def padded(scale):
        model = tc.make_padded_model(recon_model, pad_scale_slices=scale)
        return model

    variants = {}  # label -> (model, taper_rows or 0)
    variants['none'] = (recon_model, 0)
    variants['taper'] = (recon_model, geo['taper_rows'])
    partial_model = padded(partial_pad_scale)
    partial_geo = tc.cone_axial_geometry(partial_model, psf_margin=psf_margin)
    variants['pad_partial'] = (partial_model, 0)
    variants['pad_partial+taper'] = (partial_model, partial_geo['taper_rows'])
    full_model = padded(geo['full_pad_scale'])
    full_geo = tc.cone_axial_geometry(full_model, psf_margin=psf_margin)
    variants['pad_full'] = (full_model, 0)
    # The full slab needs no taper by geometry; force the none-variant's width to test
    # whether tapering when unnecessary costs quality (hypothesis H2).
    variants['pad_full+taper'] = (full_model, geo['taper_rows'])
    variants['pad_overfull'] = (padded(1.7), 0)

    print(f'partial pad scale {partial_pad_scale:.3f} (taper rows there: '
          f'{partial_geo["taper_rows"]}); full-slab geometric taper rows: '
          f'{full_geo["taper_rows"]} (0 expected; forcing {geo["taper_rows"]} in '
          f'pad_full+taper as the H2 probe)')

    if figures_only:
        # Rebuild figures from a prior full run's saved results (no recons).  Label-prefix
        # matching is safe here: 'pad_partial_' does not prefix 'pad_partial+taper_...'.
        saved = np.load(os.path.join(res_dir, 'p2a_metrics.npz'))
        metrics_by_variant = {}
        for label in variants:
            prefix = label + '_'
            metrics_by_variant[label] = {key[len(prefix):]: list(saved[key])
                                         for key in saved.files if key.startswith(prefix)}
        final_by_variant = {
            label: np.load(os.path.join(res_dir, f'p2a_final_{label.replace("+", "_")}.npy'))
            for label in variants}
    else:
        metrics_by_variant = {}
        final_by_variant = {}
        for label, (model, k_taper) in variants.items():
            shape = model.get_params('recon_shape')
            weights = tc.make_row_taper_weights(sinogram_shape, k_last=k_taper) if k_taper else None
            print(f'--- {label}: recon shape {shape}, taper rows {k_taper}')
            metrics, snaps = tc.run_tracked_recon(model, sinogram, truth_small, masks,
                                                  num_iterations, snapshot_iters, label=label,
                                                  weights=weights)
            metrics_by_variant[label] = metrics
            final_by_variant[label] = snaps[num_iterations - 1]

        # ---- Persist ----
        np.savez(os.path.join(res_dir, 'p2a_metrics.npz'),
                 **{f'{label}_{key}': np.array(vals)
                    for label, mm in metrics_by_variant.items() for key, vals in mm.items()})
        for label, recon in final_by_variant.items():
            np.save(os.path.join(res_dir, f'p2a_final_{label.replace("+", "_")}.npy'), recon)

    # ---- Figures ----
    # Montage display labels carry the taper width and the truncated-end NRMSE, so the
    # figures are self-contained; the dashed box on the truth panel shows WHERE that NRMSE
    # is measured (the end_top region: interior disk x last end_slice_margin slices).
    end_vals = {label: mm['nrmse_end_top'][-1] for label, mm in metrics_by_variant.items()}
    taper_widths = {label: k for label, (_model, k) in variants.items()}

    def display(label):
        name = f'{label} ({taper_widths[label]} rows)' if taper_widths[label] else label
        return f'recon: {name}\nend NRMSE {end_vals[label]:.3f}'

    center_col = small_shape[1] // 2
    region_kwargs = dict(region_mask=masks['end_top'],
                         region_label='dashed box = end-NRMSE region')
    group_a = {display(k): final_by_variant[k] for k in ['none', 'taper']}
    tc.save_slice_montage(truth_small, group_a, axis=1, index=center_col,
                          title='P2a: no remediation vs row taper (x-z recons; truncated side = top)',
                          path=os.path.join(fig_dir, 'p2a_xz_taper.png'), **region_kwargs)
    group_b = {display(k): final_by_variant[k] for k in
               ['pad_partial', 'pad_partial+taper', 'pad_full', 'pad_overfull']}
    tc.save_slice_montage(truth_small, group_b, axis=1, index=center_col,
                          title='P2a: padding levels (x-z recons; truncated side = top)',
                          path=os.path.join(fig_dir, 'p2a_xz_padding.png'), **region_kwargs)

    # Two z-profiles with distinct questions: the first compares one representative per
    # remediation FAMILY (does each family fix the end?); the second zooms into the padded
    # family alone (level sweep + does adding a taper hurt).  pad_full appears in both as
    # the shared reference.
    family_picks = {'none': final_by_variant['none'],
                    f'taper ({taper_widths["taper"]} rows)': final_by_variant['taper'],
                    'pad_full': final_by_variant['pad_full']}
    tc.plot_z_profile(truth_small, family_picks, masks,
                      'P2a: one representative per family (z profile, interior-disk mean)',
                      os.path.join(fig_dir, 'p2a_z_profile.png'))
    pad_family = {'pad_partial': final_by_variant['pad_partial'],
                  f'pad_partial+taper ({taper_widths["pad_partial+taper"]} rows)':
                      final_by_variant['pad_partial+taper'],
                  'pad_full': final_by_variant['pad_full'],
                  f'pad_full+taper (forced {taper_widths["pad_full+taper"]} rows)':
                      final_by_variant['pad_full+taper'],
                  'pad_overfull': final_by_variant['pad_overfull']}
    tc.plot_z_profile(truth_small, pad_family, masks,
                      'P2a: the padded family, zoomed to the truncated end',
                      os.path.join(fig_dir, 'p2a_z_profile_controls.png'),
                      xlim=(44, small_shape[2] - 1))
    tc.plot_convergence(metrics_by_variant,
                        ['nrmse_end_top', 'nrmse_interior', 'change_pct'],
                        'P2a axial variants: convergence by region',
                        os.path.join(fig_dir, 'p2a_convergence.png'))

    print('\n=== Final-iteration summary (iter {}) ==='.format(num_iterations))
    print(f'{"variant":>18}  {"end_top":>8}  {"end_bot":>8}  {"interior":>8}  {"change%":>8}')
    for label, mm in metrics_by_variant.items():
        print(f'{label:>18}  {mm["nrmse_end_top"][-1]:8.4f}  {mm["nrmse_end_bot"][-1]:8.4f}  '
              f'{mm["nrmse_interior"][-1]:8.4f}  {mm["change_pct"][-1]:8.3f}')
    print(f'Figures in {fig_dir} (p2a_*)')
