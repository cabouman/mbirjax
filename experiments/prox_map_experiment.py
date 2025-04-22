import numpy as np
import time
import jax.numpy as jnp
import mbirjax.parallel_beam

if __name__ == "__main__":
    """
    This is a script to develop, debug, and tune the parallel beam mbirjax code.
    """
    # Choose the geometry type
    geometry_type = 'cone'

    print('Using {} geometry.'.format(geometry_type))

    # Set parameters
    num_views = 1
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
    ct_model = mbirjax.ConeBeamModel(sinogram_shape, angles, source_detector_dist=source_detector_dist, source_iso_dist=source_iso_dist)

    # Generate 3D Shepp Logan phantom
    print('Creating phantom')
    phantom = ct_model.gen_modified_3d_sl_phantom()

    # Generate synthetic sinogram data
    print('Creating sinogram')
    sinogram = ct_model.forward_project(phantom)

    # View sinogram
    mbirjax.slice_viewer(sinogram, title='Original sinogram', slice_axis=0, slice_label='View')

    # Generate weights array - for an initial reconstruction, use weights = None, then modify as desired.
    weights = None
    # weights = ct_model.gen_weights(sinogram / sinogram.max(), weight_type='transmission_root')

    # Set reconstruction parameter values
    ct_model.set_params(sharpness=sharpness, verbose=1)

    # Print out model parameters
    ct_model.print_params()

    # ##########################
    # Perform VCD reconstruction
    prox_input = phantom  # jnp.zeros_like(phantom)
    time0 = time.time()
    recon, recon_params = ct_model.prox_map(prox_input / 2, sinogram, weights=weights, max_iterations=10, init_recon=prox_input, first_iteration=3)

    recon.block_until_ready()
    elapsed = time.time() - time0
    print('Elapsed time for recon is {:.3f} seconds'.format(elapsed))
    # ##########################

    # Display results
    mbirjax.slice_viewer(phantom-recon, title='Phantom (left) vs VCD Recon (right)', vmin=-0.1, vmax=0.5)


