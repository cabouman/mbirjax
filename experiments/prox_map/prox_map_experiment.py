import numpy as np
import time
import jax.numpy as jnp
import mbirjax as mj

if __name__ == "__main__":
    """
    This is a script to develop, debug, and tune the parallel beam mbirjax code.
    """
    # Choose the geometry type
    geometry_type = 'parallel'

    print('Using {} geometry.'.format(geometry_type))

    # Set parameters
    num_views = 64
    view_angle = np.pi / 2
    num_det_rows = 40
    num_det_channels = 64
    sharpness = 4.0
    
    # These can be adjusted to describe the geometry in the cone beam case.
    # np.Inf is an allowable value, in which case this is essentially parallel beam
    source_detector_dist = 4 * num_det_channels
    source_iso_dist = source_detector_dist

    detector_cone_angle = 2 * np.arctan2(num_det_channels / 2, source_detector_dist)

    start_angle = view_angle
    end_angle = view_angle + (np.pi + detector_cone_angle)

    # Initialize sinogram
    sinogram_shape = (num_views, num_det_rows, num_det_channels)
    angles = jnp.linspace(start_angle, end_angle, num_views, endpoint=False)

    # Set up the model
    ct_model = mj.ConeBeamModel(sinogram_shape, angles, source_detector_dist=source_detector_dist, source_iso_dist=source_iso_dist)

    # Generate 3D Shepp Logan phantom
    print('Creating phantom')
    phantom = ct_model.gen_modified_3d_sl_phantom()

    # Generate synthetic sinogram data
    print('Creating sinogram')
    sinogram = ct_model.forward_project(phantom)

    # View sinogram
    # mj.slice_viewer(sinogram, title='Original sinogram', slice_axis=0, slice_label='View')

    # Generate weights array - for an initial reconstruction, use weights = None, then modify as desired.
    weights = None
    # weights = ct_model.gen_weights(sinogram / sinogram.max(), weight_type='transmission_root')

    # Set reconstruction parameter values
    ct_model.set_params(sharpness=sharpness, verbose=1)

    # Print out model parameters
    ct_model.print_params()

    # ##########################
    # Perform VCD reconstruction
    prox_input = phantom + 0.1 * (phantom > 0)  # jnp.zeros_like(phantom)

    ct_model.set_params(sharpness=6, snr_db=60)
    recon, recon_dict = ct_model.recon(sinogram, weights=weights, max_iterations=20, first_iteration=0,
                                       stop_threshold_change_pct=0.01)

    # Do a prox map with small sigma_prox to return very nearly the prox_input
    prox_recon0, prox_recon_dict0 = ct_model.prox_map(prox_input, sinogram, weights=weights, sigma_prox=1e-6, max_iterations=20,
                                                      first_iteration=0, init_recon=prox_input)

    # Do a prox map with large sigma_prox to return very nearly the same as a recon with small prior
    prox_recon1, prox_recon_dict1 = ct_model.prox_map(prox_input, sinogram, weights=weights, sigma_prox=1e6, max_iterations=20,
                                                      first_iteration=0, init_recon=recon)

    print(np.linalg.norm(prox_recon0 - prox_input) / np.linalg.norm(prox_input))
    print(np.linalg.norm(prox_recon1 - recon) / np.linalg.norm(recon))
    print(ct_model.get_params('sigma_prox'))
    # ##########################

    # Display results
    mj.slice_viewer(prox_input, prox_recon0, recon, prox_recon1, data_dicts=[None, recon_dict, prox_recon_dict0, prox_recon_dict1],
                         title='prox input, prox: 0.0001, recon, prox: 10000', vmin=-0.1, vmax=1.2)


