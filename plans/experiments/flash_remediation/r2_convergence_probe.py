"""P2a-R follow-up: is R2's pad_full under-converged at 40 iterations, and do the vertical
streaks fade with more iterations?

Greg's hypothesis (2026-07-08): the sharp-regularization regime converges slower, and the
vertical streaks in the R2 montage are ALGORITHM artifacts -- VCD's update unit is an (x,y)
pixel column, so column-wise patterning is its fingerprint -- which should fade at true
convergence unless the problem is genuinely underdetermined.

This probe reruns ONLY the R2 pad_full variant, for 4x the iterations (160), snapshotting at
40/80/160, and tracks a streak-amplitude index alongside the usual region metrics:
    streak_amp = std over x of (central-z column means - their smoothed version)
computed on the center-y x-z section (captures the vertical-streak energy specifically).

Run inside the mbirjax conda env; all knobs below (no CLI args).  Outputs land in
figures/ (p2ar_r2probe_*) and results/ next to this script.
"""

import os
import numpy as np
import mbirjax as mj  # noqa: F401 -- must precede anything that touches jax
import truncation_common as tc

if __name__ == '__main__':

    # #### R2 configuration (identical to z_robustness_check.py R2_sharp)
    num_views = 128
    num_det_rows = 64
    num_det_channels = 96
    source_detector_dist = 4.0 * num_det_channels
    source_iso_dist = 2.0 * num_det_channels
    sharpness = 2.0
    snr_db = 35.0
    radius_frac = 0.75
    z_lo_frac, z_hi_frac = -0.80, 1.60
    laminate_period = 3
    target_line_integral = 2.0
    end_slice_margin = 6
    interior_radius_frac = 0.85
    psf_margin = 2

    # #### Probe knobs
    num_iterations = 160
    snapshot_iters = (39, 79)      # + the final iteration; montage shows all three

    out_dir = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(out_dir, 'figures')
    res_dir = os.path.join(out_dir, 'results')

    # ---- Same construction as the robustness script's R2 ----
    sinogram_shape = (num_views, num_det_rows, num_det_channels)
    angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)
    probe = mj.ConeBeamModel(sinogram_shape, angles, source_detector_dist=source_detector_dist,
                             source_iso_dist=source_iso_dist)
    half_slab_vox = probe.get_params('recon_shape')[2] / 2.0
    slice_margin = int(np.ceil((max(abs(z_lo_frac), abs(z_hi_frac)) - 1.0) * half_slab_vox)) + 6

    recon_model, truth_model = tc.make_cone_models(
        sinogram_shape, angles, source_detector_dist, source_iso_dist,
        lateral_margin=0, slice_margin=slice_margin, sharpness=sharpness)
    recon_model.set_params(snr_db=snr_db)
    small_shape = recon_model.get_params('recon_shape')
    big_shape = truth_model.get_params('recon_shape')
    delta_voxel = recon_model.get_params('delta_voxel')
    geo = tc.cone_axial_geometry(recon_model, psf_margin=psf_margin)

    phantom_big = tc.build_phantom(big_shape, small_shape, delta_voxel, radius_frac,
                                   z_lo_frac, z_hi_frac, target_line_integral,
                                   laminate_period=laminate_period)
    truth_small = tc.center_crop(phantom_big, small_shape)
    sinogram = np.asarray(truth_model.forward_project(phantom_big))

    masks = tc.make_masks(small_shape, interior_radius_frac, end_slice_margin)
    full_model = tc.make_padded_model(recon_model, pad_scale_slices=geo['full_pad_scale'])

    print(f'R2 pad_full extended run: {num_iterations} iterations, recon shape '
          f'{full_model.get_params("recon_shape")}')
    metrics, snaps = tc.run_tracked_recon(full_model, sinogram, truth_small, masks,
                                          num_iterations, snapshot_iters, label='pad_full')

    # ---- Streak-amplitude index on each snapshot ----
    def streak_amp(volume):
        """Std over x of the de-trended central-z column means of the center-y section."""
        section = volume[:, small_shape[1] // 2, :]              # (x, z)
        central = section[:, end_slice_margin:-end_slice_margin]
        col_means = central.mean(axis=1)
        smooth = np.convolve(col_means, np.ones(7) / 7, mode='same')
        interior = slice(8, -8)                                   # avoid edge effects
        return float(np.std((col_means - smooth)[interior]))

    print('\niter  end_top  interior  change%   streak_amp')
    for it in sorted(snaps):
        print(f'{it + 1:4d}  {metrics["nrmse_end_top"][it]:.4f}   '
              f'{metrics["nrmse_interior"][it]:.4f}    '
              f'{metrics["change_pct"][it] if it > 0 else float("nan"):.3f}     '
              f'{streak_amp(snaps[it]):.5f}')

    # ---- Figures: montage of the three snapshots + convergence vs the default regime ----
    center_col = small_shape[1] // 2
    display = {f'recon: pad_full, iter {it + 1}\nend NRMSE {metrics["nrmse_end_top"][it]:.3f}':
                   snaps[it] for it in sorted(snaps)}
    tc.save_slice_montage(truth_small, display, axis=1, index=center_col,
                          title='R2 pad_full: does more iteration fade the streaks? '
                                '(x-z recons; truncated side = top)',
                          path=os.path.join(fig_dir, 'p2ar_r2probe_xz.png'),
                          region_mask=masks['end_top'],
                          region_label='dashed = end-NRMSE region')

    p2a = np.load(os.path.join(res_dir, 'p2a_metrics.npz'))
    compare = {'R2 pad_full (sharp, 160 it)': metrics,
               'P2a pad_full (default reg, 40 it)': {
                   key: list(p2a[f'pad_full_{key}'])
                   for key in ['nrmse_end_top', 'nrmse_interior', 'change_pct']}}
    tc.plot_convergence(compare, ['nrmse_end_top', 'nrmse_interior', 'change_pct'],
                        'R2 vs default regularization: pad_full convergence',
                        os.path.join(fig_dir, 'p2ar_r2probe_convergence.png'))

    np.savez(os.path.join(res_dir, 'p2ar_r2probe_metrics.npz'),
             **{key: np.array(vals) for key, vals in metrics.items()})
    for it in sorted(snaps):
        np.save(os.path.join(res_dir, f'p2ar_r2probe_iter{it + 1}.npy'), snaps[it])
    print(f'Figures in {fig_dir} (p2ar_r2probe_*)')
