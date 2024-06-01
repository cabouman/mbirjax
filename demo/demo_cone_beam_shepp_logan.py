import numpy as np
import time
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mbirjax.plot_utils as pu
import mbirjax.parallel_beam

if __name__ == "__main__":
    """
    This is a script to develop, debug, and tune the vcd reconstruction with a parallel beam projector
    """
    # Set parameters
    num_views = 256
    num_det_rows = 20
    num_det_channels = 256
    source_detector_dist = 4 * num_det_channels
    source_iso_dist = source_detector_dist

    start_angle = -np.pi*(1/2)
    end_angle = np.pi*(1/2)
    sharpness = 0.0

    # Initialize sinogram
    sinogram = jnp.zeros((num_views, num_det_rows, num_det_channels))
    angles = jnp.linspace(start_angle, end_angle, num_views, endpoint=False)

    # Set up parallel beam model
    cone_model = mbirjax.ConeBeamModel(sinogram.shape, angles, source_detector_dist=source_detector_dist, source_iso_dist=source_iso_dist)

    # Generate 3D Shepp Logan phantom
    print('Creating phantom')
    phantom = cone_model.gen_modified_3d_sl_phantom()

    # Generate synthetic sinogram data
    print('Creating sinogram')
    sinogram = cone_model.forward_project(phantom)

    # View sinogram
    pu.slice_viewer(sinogram.transpose((1, 2, 0)), title='Original sinogram')

    # Generate weights array
    weights = cone_model.gen_weights(sinogram / sinogram.max(), weight_type='transmission_root')

    # Set reconstruction parameter values
    cone_model.set_params(sharpness=sharpness, verbose=1)

    # Print out model parameters
    cone_model.print_params()

    # ##########################
    # Perform VCD reconstruction
    print('Starting recon')
    time0 = time.time()
    recon, recon_params = cone_model.recon(sinogram, weights=weights)

    recon.block_until_ready()
    elapsed = time.time() - time0
    print('Elapsed time for recon is {:.3f} seconds'.format(elapsed))
    # ##########################

    # Print out parameters used in recon
    print('Recon sigma_y: {}'.format(recon_params.auto_regularization_parameters.sigma_y))
    print('Recon sigma_x: {}'.format(recon_params.auto_regularization_parameters.sigma_x))
    print('Recon params: {}'.format(recon_params))

    # Display results
    pu.slice_viewer(phantom, recon, title='Phantom (left) vs VCD Recon (right)')

