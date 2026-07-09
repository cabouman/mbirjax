"""P2a-R: robustness rider for the P2a axial recommendation (pad in z, don't taper).

P2a ran at one geometry, default regularization, and no noise.  This script re-checks the
RANKING {none, taper, pad_full} -- not all seven variants -- in three regimes chosen to probe
where the conclusion could plausibly bend (plan doc, P2a-R):

- R1_widecone: SDD 4C -> 2.5C and 64 -> 128 detector rows (a wide-CONE configuration).  The governing ratio analysis
  says every FRACTIONAL quantity (visibility bound, exit-row band, end wedge) is set by
  the FAN half-angle R/SID = C*delta_c/(2*SDD) alone -- more rows change nothing
  fractionally, and magnification per se never enters.  R1 widens R/SID from 0.125 to 0.2
  and doubles the rows, and re-runs the far-overshoot bit-exact identity, so it validates
  (or falsifies) those formulas well away from the P2a operating point.
- R2_sharp: sharpness 2.0, snr_db 35 (the BGA center-slice-artifact regime) -- a very
  different prior-to-data balance, hence a different screening length for the
  prior-extrapolation failure mode.  No noise.
- R3_noise: photon-count noise (i0 = 1e4) with matching 'transmission' weights, at DEFAULT
  sharpness/snr_db -- noise and regularization are probed as INDEPENDENT single-variable
  axes (stacking them would confound which one moved the result; Greg).  The taper variant
  multiplies its ramp into the transmission weights, as a user would.

Success criterion: pad_full < taper < none on truncated-end NRMSE in every regime, and
R1's measured geometry quantities match the cone_axial_geometry predictions.

Run inside the mbirjax conda env; all knobs below (no CLI args).  Outputs land in
figures/ (p2ar_*) and results/ next to this script.
"""

import os
import numpy as np
import mbirjax as mj  # noqa: F401 -- must precede anything that touches jax
import truncation_common as tc

if __name__ == '__main__':

    # #### Shared problem size (channels and views as in P2a; rows/SDD vary per config)
    num_views = 128
    num_det_channels = 96
    sid_factor = 2.0               # SID = 2.0 * channels, as in P2a

    # #### Phantom (identical fractions to P2a: one-sided axial truncation)
    radius_frac = 0.75
    z_lo_frac, z_hi_frac = -0.80, 1.60
    laminate_period = 3
    target_line_integral = 2.0
    z_hi_far = 4.0                 # far-overshoot identity check (R1 only)

    # #### Configurations under test
    CONFIGS = {
        'R1_widecone': dict(sdd_factor=2.5, num_det_rows=128, sharpness=0.0, snr_db=None,
                           noise=False, end_slice_margin=12),
        'R2_sharp':   dict(sdd_factor=4.0, num_det_rows=64, sharpness=2.0, snr_db=35.0,
                           noise=False, end_slice_margin=6),
        'R3_noise':   dict(sdd_factor=4.0, num_det_rows=64, sharpness=0.0, snr_db=None,
                           noise=True, end_slice_margin=6),
    }
    noise_i0 = 1e4                 # photon count at zero attenuation (R3)
    noise_seed = 1

    # #### Recon / metrics
    num_iterations = 40
    interior_radius_frac = 0.85
    psf_margin = 2
    figures_only = False           # True: skip the recons, rebuild figures from results/p2ar_*

    out_dir = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(out_dir, 'figures')
    res_dir = os.path.join(out_dir, 'results')
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    summary = {}
    for cfg_name, cfg in CONFIGS.items():
        print(f'\n================ {cfg_name} ================')
        sinogram_shape = (num_views, cfg['num_det_rows'], num_det_channels)
        angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)
        source_detector_dist = cfg['sdd_factor'] * num_det_channels
        source_iso_dist = sid_factor * num_det_channels

        # Ground-truth-phantom slice margin: R1 also builds the far phantom, so size for z_hi_far.
        probe = mj.ConeBeamModel(sinogram_shape, angles,
                                 source_detector_dist=source_detector_dist,
                                 source_iso_dist=source_iso_dist)
        half_slab_vox = probe.get_params('recon_shape')[2] / 2.0
        z_hi_margin = z_hi_far if cfg_name == 'R1_widecone' else z_hi_frac
        slice_margin = int(np.ceil((max(abs(z_lo_frac), abs(z_hi_margin)) - 1.0)
                                   * half_slab_vox)) + 6

        recon_model, truth_model = tc.make_cone_models(
            sinogram_shape, angles, source_detector_dist, source_iso_dist,
            lateral_margin=0, slice_margin=slice_margin, sharpness=cfg['sharpness'])
        if cfg['snr_db'] is not None:
            recon_model.set_params(snr_db=cfg['snr_db'])
        small_shape = recon_model.get_params('recon_shape')
        big_shape = truth_model.get_params('recon_shape')
        delta_voxel = recon_model.get_params('delta_voxel')

        geo = tc.cone_axial_geometry(recon_model, psf_margin=psf_margin)
        r_over_sid = geo['fov_radius'] / source_iso_dist
        print(f'recon shape {small_shape}, ground truth phantom {big_shape}; R/SID {r_over_sid:.3f} '
              f'-> predicted bound ratio {1 + r_over_sid:.3f} (measured '
              f'{geo["max_visible_z"]/geo["half_slab"]:.3f}), full pad scale '
              f'{geo["full_pad_scale"]:.3f}, taper rows {geo["taper_rows"]} '
              f'of {cfg["num_det_rows"]}')

        phantom_big = tc.build_phantom(big_shape, small_shape, delta_voxel, radius_frac,
                                       z_lo_frac, z_hi_frac, target_line_integral,
                                       laminate_period=laminate_period)
        truth_small = tc.center_crop(phantom_big, small_shape)
        sinogram = np.asarray(truth_model.forward_project(phantom_big))

        if cfg_name == 'R1_widecone':
            phantom_far = tc.build_phantom(big_shape, small_shape, delta_voxel, radius_frac,
                                           z_lo_frac, z_hi_far, target_line_integral,
                                           laminate_period=laminate_period)
            far_diff = float(np.max(np.abs(
                np.asarray(truth_model.forward_project(phantom_far)) - sinogram)))
            print(f'far-overshoot identity at R/SID {r_over_sid:.3f}: max |diff| '
                  f'= {far_diff:.2e} (bound predicts ~0)')
            del phantom_far

        # Noise + statistical weights (R3): all variants recon the SAME noisy sinogram
        # with transmission weights; the taper multiplies its ramp into them.
        if cfg['noise']:
            sino_used, base_weights = tc.add_transmission_noise(sinogram, i0=noise_i0,
                                                                seed=noise_seed)
            print(f'noise added: i0 {noise_i0:.0e}, sino max {sino_used.max():.2f}, '
                  f'weights range [{base_weights.min():.3f}, {base_weights.max():.3f}]')
        else:
            sino_used, base_weights = sinogram, None

        masks = tc.make_masks(small_shape, interior_radius_frac, cfg['end_slice_margin'])
        full_model = tc.make_padded_model(recon_model, pad_scale_slices=geo['full_pad_scale'])
        taper = tc.make_row_taper_weights(sinogram_shape, k_last=geo['taper_rows'])
        variants = {  # label -> (model, weights)
            'none': (recon_model, base_weights),
            f'taper ({geo["taper_rows"]} rows)':
                (recon_model, taper if base_weights is None else base_weights * taper),
            'pad_full': (full_model, base_weights),
        }

        metrics_by_variant = {}
        final_by_variant = {}
        if figures_only:
            saved = np.load(os.path.join(res_dir, f'p2ar_{cfg_name}_metrics.npz'))
            for label in variants:
                prefix = label + '_'
                metrics_by_variant[label] = {key[len(prefix):]: list(saved[key])
                                             for key in saved.files if key.startswith(prefix)}
                fname = f'p2ar_{cfg_name}_final_{label.split(" ")[0]}.npy'
                final_by_variant[label] = np.load(os.path.join(res_dir, fname))
        else:
            for label, (model, weights) in variants.items():
                print(f'--- {cfg_name} / {label}')
                metrics, snaps = tc.run_tracked_recon(model, sino_used, truth_small, masks,
                                                      num_iterations, label=label,
                                                      weights=weights)
                metrics_by_variant[label] = metrics
                final_by_variant[label] = snaps[num_iterations - 1]
            np.savez(os.path.join(res_dir, f'p2ar_{cfg_name}_metrics.npz'),
                     **{f'{label}_{key}': np.array(vals)
                        for label, mm in metrics_by_variant.items()
                        for key, vals in mm.items()})
            for label, recon in final_by_variant.items():
                fname = f'p2ar_{cfg_name}_final_{label.split(" ")[0]}.npy'
                np.save(os.path.join(res_dir, fname), recon)

        # Figures: one labeled recon montage + one z-profile per configuration.
        end_vals = {label: mm['nrmse_end_top'][-1]
                    for label, mm in metrics_by_variant.items()}
        display = {f'recon: {label}\nend NRMSE {end_vals[label]:.3f}': vol
                   for label, vol in final_by_variant.items()}
        center_col = small_shape[1] // 2
        tc.save_slice_montage(truth_small, display, axis=1, index=center_col,
                              title=f'P2a-R {cfg_name}: x-z recons (truncated side = top)',
                              path=os.path.join(fig_dir, f'p2ar_{cfg_name}_xz.png'),
                              region_mask=masks['end_top'],
                              region_label='dashed = end-NRMSE region')
        tc.plot_z_profile(truth_small, final_by_variant, masks,
                          f'P2a-R {cfg_name}: z profile (interior-disk mean)',
                          os.path.join(fig_dir, f'p2ar_{cfg_name}_z_profile.png'))
        summary[cfg_name] = {label: (end_vals[label],
                                     metrics_by_variant[label]['nrmse_interior'][-1])
                             for label in variants}

    print('\n=== P2a-R summary (iter 40): end_top NRMSE / interior NRMSE ===')
    for cfg_name, rows in summary.items():
        parts = [f'{label}: {end_v:.3f} / {int_v:.3f}' for label, (end_v, int_v) in rows.items()]
        ordered = sorted(rows.items(), key=lambda kv: kv[1][0])
        ranking_ok = (ordered[0][0] == 'pad_full' and ordered[-1][0] == 'none')
        print(f'{cfg_name:>12}:  ' + '  |  '.join(parts) +
              ('   [ranking OK]' if ranking_ok else '   [RANKING CHANGED]'))
    print(f'Figures in {fig_dir} (p2ar_*)')
