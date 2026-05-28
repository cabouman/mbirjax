import gc
import os
import time
import h5py
import jax.numpy as jnp
import mbirjax as mj
import mbirjax.preprocess as mjp
import numpy as np
from jax import clear_caches
import bh_curve_fit_utils as but


if __name__ == '__main__':

    #####################
    # Configuration
    #####################
    dataset_url = '/depot/bouman/data/ORNL/fit_bh_curve_nozzle_data.tgz'
    dataset_url_scan = (
        'https://www.datadepot.rcac.purdue.edu/bouman/data/hfn_scan.tgz')
    download_dir = './demo_data/'
    output_dir = './results/nozzle_bh_curve_fit/'
    os.makedirs(output_dir, exist_ok=True)

    N_bh_params = 6
    cheb_degree = 10
    input_vmin = 0.0
    input_vmax = 5.0
    use_measured_range = True

    num_recon_views = 200
    num_iterations = 18
    correction_batch_size = 16

    ###############
    # Load data
    ###############
    dataset_dir = mj.download_and_extract(dataset_url, download_dir)

    print('Loading training beam-hardened projection')
    training_beam_hardened_projection = np.load(os.path.join(
        dataset_dir, 'training_beam_hardened_projection.npy'))
    print('  shape: {}'.format(training_beam_hardened_projection.shape))

    print('Loading training linear projection')
    training_linear_projection = np.load(os.path.join(
        dataset_dir, 'training_linear_projection.npy'))
    print('  shape: {}'.format(training_linear_projection.shape))

    #####################
    # Fit forward BH curve
    #####################
    linear_projection = training_linear_projection.ravel()
    target_projection = training_beam_hardened_projection.ravel()

    print('\nTraining forward BH curve')
    best_bh_params = mj.fit_beam_hardening_curve(
        linear_projection, target_projection, N_bh_params)
    print('Best BH parameters: {}'.format(best_bh_params))
    np.save(os.path.join(output_dir, 'forward_bh_params.npy'), best_bh_params)

    #####################
    # Plot forward BH curve
    #####################
    length = max(1.0, float(np.max(training_linear_projection)) * 1.05)
    x_path_length_plot = np.linspace(0.0, length, 500)
    but.plot_bh_correction_curve(
        x_path_length_plot,
        best_bh_params,
        output_path=os.path.join(output_dir, 'forward_bh_curve.png')
    )

    #####################
    # Fit Chebyshev inverse curve
    #####################
    measured_vmin = float(np.nanmin(training_beam_hardened_projection))
    measured_vmax = float(np.nanmax(training_beam_hardened_projection))
    if use_measured_range:
        inverse_vmin = measured_vmin
        inverse_vmax = measured_vmax
    else:
        inverse_vmin = input_vmin
        inverse_vmax = input_vmax

    print(
        '\nChebyshev inverse range: [{:.6g}, {:.6g}]'.format(
            inverse_vmin, inverse_vmax))
    print(
        '  measured range: [{:.6g}, {:.6g}], input default: [{:.6g}, {:.6g}]'
        .format(measured_vmin, measured_vmax, input_vmin, input_vmax))

    cheb_coeffs, y_domain = mj.fit_inverse_beam_hardening_curve(
        best_bh_params,
        vmin=inverse_vmin,
        vmax=inverse_vmax,
        degree=cheb_degree,
        num_samples=4000,
    )
    print('Chebyshev degree: {}'.format(cheb_degree))
    print('Chebyshev coeffs: {}'.format(cheb_coeffs))
    print('Chebyshev y-domain: {}'.format(y_domain))

    np.savez(
        os.path.join(output_dir, 'chebyshev_inverse_fit.npz'),
        cheb_coeffs=cheb_coeffs,
        y_domain=np.asarray(y_domain),
    )

    #####################
    # Load full ORNL sinogram, geometry, and baseline correction
    #####################
    dataset_dir_scan = mj.download_and_extract(
        dataset_url_scan, download_dir)

    hdf5_files = sorted(
        f for f in os.listdir(dataset_dir_scan)
        if f.lower().endswith(('.h5', '.hdf5')))
    filename = os.path.join(dataset_dir_scan, hdf5_files[0])

    print('\nLoading ORNL sinogram and geometry')
    full_sino, cone_beam_params, optional_params = (
        mjp.pymbir.compute_sino_and_params(filename, bh_correction=False))
    print('Full measured sinogram shape: {}'.format(full_sino.shape))

    print('\nLoading baseline-corrected ORNL sinogram')
    baseline_sino, _, _ = (
        mjp.pymbir.compute_sino_and_params(filename, bh_correction=True))

    with h5py.File(filename, 'r') as h5_file:
        BHCN_params = h5_file.attrs['BHC_params']

    #####################
    # Plot inverse correction curve
    #####################
    y_inverse_eval = np.linspace(y_domain[0], y_domain[1], 500)
    baseline_corrected_projection = mjp.pymbir.apply_bh_correction(
        y_inverse_eval, BHCN_params)
    cheb_corrected_projection = (
        mj.apply_fitted_inverse_beam_hardening_curve(
            y_inverse_eval, cheb_coeffs, y_domain)
    )

    but.plot_inverse_correction_curve(
        y_inverse_eval,
        baseline_corrected_projection,
        cheb_corrected_projection,
        output_path=os.path.join(output_dir, 'inverse_correction_curve.png'),
    )

    #####################
    # Uniform short-scan subset
    #####################
    angle_candidates = cone_beam_params['angles']
    num_det_channels = cone_beam_params['sinogram_shape'][2]
    source_detector_dist = cone_beam_params['source_detector_dist']
    detector_cone_angle = 2 * np.arctan2(
        num_det_channels / 2, source_detector_dist)

    candidates_normalized = np.abs(angle_candidates - angle_candidates[0])
    end_index = np.where(
        candidates_normalized < np.pi + detector_cone_angle)[0][-1]
    uniform_index_list = but.create_uniform_index(
        angle_candidates, end_index, num_recon_views)
    print(
        'Using {} uniformly sampled views from {} short-scan views.'.format(
            len(uniform_index_list), end_index + 1))

    uniform_angles = angle_candidates[uniform_index_list]
    recon_sino_no_corr = full_sino[uniform_index_list].astype(
        np.float32, copy=False)
    del full_sino

    #####################
    # Apply baseline and proposed corrections
    #####################
    print('\nSelecting baseline correction')
    recon_sino_baseline = baseline_sino[uniform_index_list].astype(
        np.float32, copy=False)
    del baseline_sino

    print('\nApplying Chebyshev inverse correction')
    recon_sino_cheb = np.empty_like(recon_sino_no_corr, dtype=np.float32)
    for start in range(0, recon_sino_no_corr.shape[0], correction_batch_size):
        stop = min(start + correction_batch_size, recon_sino_no_corr.shape[0])
        recon_sino_cheb[start:stop] = (
            mj.apply_fitted_inverse_beam_hardening_curve(
                recon_sino_no_corr[start:stop],
                cheb_coeffs,
                y_domain,
                clip=True)
        ).astype(np.float32)

    #####################
    # Build CT model for the selected views
    #####################
    ct_model = mj.ConeBeamModel(**cone_beam_params)
    ct_model.set_params(**optional_params)
    ct_model_uniform = mj.copy_ct_model(ct_model, uniform_angles)

    #####################
    # Reconstruct the three sinograms
    #####################
    sinograms_to_recon = {
        'No BH correction': recon_sino_no_corr,
        'Baseline correction': recon_sino_baseline,
        'Chebyshev inverse correction': recon_sino_cheb,
    }

    recons = {}
    for name, sino_recon in sinograms_to_recon.items():
        print('\nReconstructing: {}'.format(name))
        sino_sub = jnp.asarray(sino_recon)

        t0 = time.time()
        recon, _ = ct_model_uniform.recon(
            sino_sub, weights=None, max_iterations=num_iterations)
        recon.block_until_ready()
        print('  {}: {:.1f} s'.format(name, time.time() - t0))

        recon_np = np.asarray(recon)
        recons[name] = recon_np
        del recon, sino_sub
        gc.collect()
        clear_caches()

    #####################
    # View only the three final reconstructions
    #####################
    print('\nLaunching slice_viewer for recon comparison...')
    mj.slice_viewer(
        recons['No BH correction'],
        recons['Baseline correction'],
        recons['Chebyshev inverse correction'],
        vmin=-0.05,
        vmax=0.10,
        slice_label=[
            'No BH correction',
            'Baseline correction',
            'Chebyshev inverse correction'],
    )
