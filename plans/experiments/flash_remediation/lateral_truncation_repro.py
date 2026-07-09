"""Phase 1 repro: LATERAL truncation (object wider than the FoV) -> the radial flash ring.

A wide, z-contained cylinder phantom (radius > the small model's FoV radius) is forward-
projected through the real detector via an enlarged ground-truth phantom, then reconstructed with
the default-shape model.  Variants: the default recon, and optionally a padded-support recon
(scale_recon_shape) as the mechanism check -- if padding absorbs the ring, the model-
support explanation is confirmed.  See plans/flash_remediation/flash_remediation_plan.md.

Run inside the mbirjax conda env; all knobs are below (no CLI args).  Outputs land in
figures/ (PNGs) and results/ (npz/npy) next to this script.
"""

import os
import numpy as np
import mbirjax as mj  # noqa: F401 -- must precede anything that touches jax
import truncation_common as tc

if __name__ == '__main__':

    # #### Problem size (small = fast CPU iteration; the failure mode is size-independent)
    num_views = 128
    num_det_rows = 32              # thin slab: the radial ring is per-slice, rows just cost time
    num_det_channels = 128
    source_detector_dist = 4.0 * num_det_channels
    source_iso_dist = 2.0 * num_det_channels   # magnification 2

    # #### Phantom (sized relative to the small model's FoV)
    radius_frac = 1.25             # cylinder radius / FoV radius: > 1 = lateral truncation
    z_lo_frac, z_hi_frac = -0.75, 0.75   # contained in the slab: NO axial truncation here
    target_line_integral = 2.0

    # #### Variants
    run_padded_variant = True
    pad_scale_lateral = 1.35       # padded support radius 1.35 * FoV > 1.25 phantom

    # #### Recon / metrics
    num_iterations = 40            # past the default 15 cap, to see the convergence tail
    snapshot_iters = (4, 9, 19)
    interior_radius_frac = 0.85    # interior/ring split (the partition study: ring ~ outer 5-15%)
    end_slice_margin = 4

    out_dir = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(out_dir, 'figures')
    res_dir = os.path.join(out_dir, 'results')
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    # ---- Models and phantom ----
    sinogram_shape = (num_views, num_det_rows, num_det_channels)
    angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)
    fov_radius_vox = num_det_channels / 2.0
    lateral_margin = int(np.ceil((radius_frac - 1.0) * fov_radius_vox)) + 6

    recon_model, truth_model = tc.make_cone_models(
        sinogram_shape, angles, source_detector_dist, source_iso_dist,
        lateral_margin=lateral_margin, slice_margin=0)
    small_shape = recon_model.get_params('recon_shape')
    big_shape = truth_model.get_params('recon_shape')
    delta_voxel = recon_model.get_params('delta_voxel')
    print(f'small (default) recon shape: {small_shape}, ground truth phantom: {big_shape}, '
          f'delta_voxel: {delta_voxel:.3f}')

    phantom_big = tc.build_phantom(big_shape, small_shape, delta_voxel, radius_frac,
                                   z_lo_frac, z_hi_frac, target_line_integral)
    truth_small = tc.center_crop(phantom_big, small_shape)

    print('Forward-projecting the ground truth phantom through the real detector...')
    sinogram = np.asarray(truth_model.forward_project(phantom_big))
    print(f'sinogram range: [{sinogram.min():.3f}, {sinogram.max():.3f}], '
          f'edge-channel mean: {sinogram[:, :, [0, -1]].mean():.3f} '
          f'(nonzero = truncated, as intended)')

    # ---- Tracked recons ----
    masks = tc.make_masks(small_shape, interior_radius_frac, end_slice_margin)
    metrics_by_variant = {}
    final_by_variant = {}
    snapshots_by_variant = {}

    print('Variant 1: default recon shape (zero lateral margin)')
    metrics, snaps = tc.run_tracked_recon(recon_model, sinogram, truth_small, masks,
                                          num_iterations, snapshot_iters, label='default')
    metrics_by_variant['default'] = metrics
    snapshots_by_variant['default'] = snaps
    final_by_variant['default'] = snaps[num_iterations - 1]

    if run_padded_variant:
        print(f'Variant 2: padded support (scale_recon_shape x{pad_scale_lateral} laterally)')
        padded_model = tc.make_padded_model(recon_model, pad_scale_lateral=pad_scale_lateral)
        print(f'  padded recon shape: {padded_model.get_params("recon_shape")}')
        metrics, snaps = tc.run_tracked_recon(padded_model, sinogram, truth_small, masks,
                                              num_iterations, snapshot_iters, label='padded')
        metrics_by_variant['padded'] = metrics
        snapshots_by_variant['padded'] = snaps
        final_by_variant['padded'] = snaps[num_iterations - 1]

    # ---- Figures ----
    # Montage labels carry each recon's ring NRMSE; the dashed contour on the truth panel
    # shows WHERE it is measured (the RoR-edge annulus, central slices).
    ring_vals = {label: mm['nrmse_ring'] for label, mm in metrics_by_variant.items()}
    center_slice = small_shape[2] // 2
    region_kwargs = dict(region_mask=masks['ring'],
                         region_label='dashed = ring-NRMSE region')
    finals = {f'recon: {label}\nring NRMSE {vals[-1]:.3f}': final_by_variant[label]
              for label, vals in ring_vals.items()}
    tc.save_slice_montage(truth_small, finals, axis=2, index=center_slice,
                          title=f'Lateral truncation: center slice (iter {num_iterations})',
                          path=os.path.join(fig_dir, 'lateral_center_slice.png'),
                          **region_kwargs)
    # An early-iteration montage shows how the ring builds up over iterations.
    early = {f'recon: default, iter {it + 1}\nring NRMSE {ring_vals["default"][it]:.3f}':
                 snapshots_by_variant['default'][it] for it in snapshot_iters}
    tc.save_slice_montage(truth_small, early, axis=2, index=center_slice,
                          title='Lateral truncation: ring buildup (default variant)',
                          path=os.path.join(fig_dir, 'lateral_ring_buildup.png'),
                          **region_kwargs)
    tc.plot_radial_profile(truth_small, final_by_variant, small_shape, end_slice_margin,
                           'Lateral truncation: radial profile (central slices)',
                           os.path.join(fig_dir, 'lateral_radial_profile.png'))
    tc.plot_convergence(metrics_by_variant,
                        ['nrmse_interior', 'nrmse_ring', 'excess_ring', 'change_pct'],
                        'Lateral truncation: convergence by region',
                        os.path.join(fig_dir, 'lateral_convergence.png'))

    # ---- Persist numbers ----
    np.savez(os.path.join(res_dir, 'lateral_metrics.npz'),
             **{f'{variant}_{key}': np.array(vals)
                for variant, mm in metrics_by_variant.items() for key, vals in mm.items()})
    for variant, recon in final_by_variant.items():
        np.save(os.path.join(res_dir, f'lateral_final_{variant}.npy'), recon)
    np.save(os.path.join(res_dir, 'lateral_truth.npy'), truth_small)

    print('\n=== Final-iteration summary ===')
    for variant, mm in metrics_by_variant.items():
        print(f'{variant:>8}: interior NRMSE {mm["nrmse_interior"][-1]:.4f}, '
              f'ring NRMSE {mm["nrmse_ring"][-1]:.4f}, '
              f'ring excess {mm["excess_ring"][-1]:+.4f}, '
              f'change% {mm["change_pct"][-1]:.3f}')
    print(f'Figures in {fig_dir}')
