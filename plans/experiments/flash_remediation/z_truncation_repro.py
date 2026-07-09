"""Phase 1 repro: AXIAL truncation (object taller than the covered slab, ONE-sided) ->
end-slice flash + z-ringing, the SiC-like case.

A laterally-contained laminated cylinder extends past the top of the covered slab (like
the SiC composite, which stays inside the FoV in channels but leaves the detector in
rows).  The top detector rows therefore collect attenuation from slices the model cannot
represent.  Variants: the default recon, and optionally a padded-support recon
(scale_recon_shape in z) as the mechanism check.  See
plans/flash_remediation/flash_remediation_plan.md.

Run inside the mbirjax conda env; all knobs are below (no CLI args).  Outputs land in
figures/ (PNGs) and results/ (npz/npy) next to this script.
"""

import os
import numpy as np
import mbirjax as mj  # noqa: F401 -- must precede anything that touches jax
import truncation_common as tc

if __name__ == '__main__':

    # #### Problem size
    num_views = 128
    num_det_rows = 64              # z is the axis under study, so give it resolution
    num_det_channels = 96
    source_detector_dist = 4.0 * num_det_channels
    source_iso_dist = 2.0 * num_det_channels   # magnification 2

    # #### Phantom (sized relative to the small model's FoV / slab)
    radius_frac = 0.75             # inside the FoV laterally: NO lateral truncation here
    z_lo_frac, z_hi_frac = -0.80, 1.60   # bottom inside the slab, top 60% of a half-slab BEYOND it
    laminate_period = 3            # alternating-contrast z layers: makes axial ringing visible
    target_line_integral = 2.0

    # #### Variants
    run_padded_variant = True
    pad_scale_slices = 1.7         # padded half-slab 0.85 * slab > 1.6 half-slab phantom top

    # #### Recon / metrics
    num_iterations = 40
    snapshot_iters = (4, 9, 19)
    interior_radius_frac = 0.85
    end_slice_margin = 6           # end-slab region for the axial-flash metric

    out_dir = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(out_dir, 'figures')
    res_dir = os.path.join(out_dir, 'results')
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    # ---- Models and phantom ----
    sinogram_shape = (num_views, num_det_rows, num_det_channels)
    angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)

    # The slice margin must contain the phantom's overshoot past the slab (plus spare);
    # computed from a probe model so it tracks the auto slice count.
    probe = mj.ConeBeamModel(sinogram_shape, angles, source_detector_dist=source_detector_dist,
                             source_iso_dist=source_iso_dist)
    small_slices = probe.get_params('recon_shape')[2]
    half_slab_vox = small_slices / 2.0
    slice_margin = int(np.ceil((max(abs(z_lo_frac), abs(z_hi_frac)) - 1.0) * half_slab_vox)) + 6

    recon_model, truth_model = tc.make_cone_models(
        sinogram_shape, angles, source_detector_dist, source_iso_dist,
        lateral_margin=0, slice_margin=slice_margin)
    small_shape = recon_model.get_params('recon_shape')
    big_shape = truth_model.get_params('recon_shape')
    delta_voxel = recon_model.get_params('delta_voxel')
    print(f'small (default) recon shape: {small_shape}, ground truth phantom: {big_shape}, '
          f'delta_voxel: {delta_voxel:.3f}')

    phantom_big = tc.build_phantom(big_shape, small_shape, delta_voxel, radius_frac,
                                   z_lo_frac, z_hi_frac, target_line_integral,
                                   laminate_period=laminate_period)
    truth_small = tc.center_crop(phantom_big, small_shape)

    print('Forward-projecting the ground truth phantom through the real detector...')
    sinogram = np.asarray(truth_model.forward_project(phantom_big))
    # One-sidedness check: the phantom's contained end stops short of the slab edge, so
    # the detector rows viewing that side read ~0; the extended end's rows read the
    # truncated attenuation.  (Row index 0 views the LOW-z side here.)
    print(f'sinogram range: [{sinogram.min():.3f}, {sinogram.max():.3f}], '
          f'row-0 mean: {sinogram[:, 0, :].mean():.3f} vs row-(-1) mean: '
          f'{sinogram[:, -1, :].mean():.3f} (one ~0, one large = one-sided, as intended)')

    # ---- Tracked recons ----
    masks = tc.make_masks(small_shape, interior_radius_frac, end_slice_margin)
    metrics_by_variant = {}
    final_by_variant = {}
    snapshots_by_variant = {}

    print('Variant 1: default recon shape (slab = detector z-coverage)')
    metrics, snaps = tc.run_tracked_recon(recon_model, sinogram, truth_small, masks,
                                          num_iterations, snapshot_iters, label='default')
    metrics_by_variant['default'] = metrics
    snapshots_by_variant['default'] = snaps
    final_by_variant['default'] = snaps[num_iterations - 1]

    if run_padded_variant:
        print(f'Variant 2: padded support (scale_recon_shape x{pad_scale_slices} in z)')
        padded_model = tc.make_padded_model(recon_model, pad_scale_slices=pad_scale_slices)
        print(f'  padded recon shape: {padded_model.get_params("recon_shape")}')
        metrics, snaps = tc.run_tracked_recon(padded_model, sinogram, truth_small, masks,
                                              num_iterations, snapshot_iters, label='padded')
        metrics_by_variant['padded'] = metrics
        snapshots_by_variant['padded'] = snaps
        final_by_variant['padded'] = snaps[num_iterations - 1]

    # ---- Figures ----
    # Montage labels carry each recon's truncated-end NRMSE; the dashed contour on the
    # truth panel shows WHERE it is measured (interior disk x top end slices).
    end_vals = {label: mm['nrmse_end_top'] for label, mm in metrics_by_variant.items()}
    center_col = small_shape[1] // 2
    region_kwargs = dict(region_mask=masks['end_top'],
                         region_label='dashed = end-NRMSE region')
    finals = {f'recon: {label}\nend NRMSE {vals[-1]:.3f}': final_by_variant[label]
              for label, vals in end_vals.items()}
    tc.save_slice_montage(truth_small, finals, axis=1, index=center_col,
                          title=f'Axial truncation: x-z section (iter {num_iterations}; '
                                f'truncated side = top)',
                          path=os.path.join(fig_dir, 'z_xz_section.png'),
                          **region_kwargs)
    early = {f'recon: default, iter {it + 1}\nend NRMSE {end_vals["default"][it]:.3f}':
                 snapshots_by_variant['default'][it] for it in snapshot_iters}
    tc.save_slice_montage(truth_small, early, axis=1, index=center_col,
                          title='Axial truncation: artifact buildup (default variant, x-z)',
                          path=os.path.join(fig_dir, 'z_buildup.png'),
                          **region_kwargs)
    tc.plot_z_profile(truth_small, final_by_variant, masks,
                      'Axial truncation: z profile (mean over interior disk)',
                      os.path.join(fig_dir, 'z_profile.png'))
    tc.plot_convergence(metrics_by_variant,
                        ['nrmse_interior', 'nrmse_end_top', 'excess_end_top', 'change_pct'],
                        'Axial truncation (one-sided): convergence by region',
                        os.path.join(fig_dir, 'z_convergence.png'))

    # ---- Persist numbers ----
    np.savez(os.path.join(res_dir, 'z_metrics.npz'),
             **{f'{variant}_{key}': np.array(vals)
                for variant, mm in metrics_by_variant.items() for key, vals in mm.items()})
    for variant, recon in final_by_variant.items():
        np.save(os.path.join(res_dir, f'z_final_{variant}.npy'), recon)
    np.save(os.path.join(res_dir, 'z_truth.npy'), truth_small)

    print('\n=== Final-iteration summary ===')
    for variant, mm in metrics_by_variant.items():
        print(f'{variant:>8}: interior NRMSE {mm["nrmse_interior"][-1]:.4f}, '
              f'end_top NRMSE {mm["nrmse_end_top"][-1]:.4f} (truncated side), '
              f'end_bot NRMSE {mm["nrmse_end_bot"][-1]:.4f} (contained side), '
              f'change% {mm["change_pct"][-1]:.3f}')
    print(f'Figures in {fig_dir}')
