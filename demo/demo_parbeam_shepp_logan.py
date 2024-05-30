import numpy as np
import time
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mbirjax.plot_utils as pu
import mbirjax.parallel_beam

if __name__ == "__main__":
    """
    This is a script to develop, debug, and tune the parallel beam mbirjax code.
    """
    # Set parameters
    num_views = 256
    num_det_rows = 20
    num_det_channels = 256
    start_angle = -np.pi*(1/2)
    end_angle = np.pi*(1/2)
    sharpness = 0.0

    # Initialize sinogram
    sinogram = jnp.zeros((num_views, num_det_rows, num_det_channels))
    angles = jnp.linspace(start_angle, end_angle, num_views, endpoint=False)

    # Set up parallel beam model
    parallel_model = mbirjax.ParallelBeamModel(sinogram.shape, angles)

    # Generate 3D Shepp Logan phantom
    print('Creating phantom')
    phantom = parallel_model.gen_modified_3d_sl_phantom()

    # Generate synthetic sinogram data
    print('Creating sinogram')
    sinogram = parallel_model.forward_project(phantom)

    # View sinogram
    pu.slice_viewer(sinogram.transpose((1, 2, 0)), title='Original sinogram')

    # Generate weights array
    weights = parallel_model.gen_weights(sinogram / sinogram.max(), weight_type='transmission_root')

    # Set reconstruction parameter values
    parallel_model.set_params(sharpness=sharpness, verbose=1)

    # Print out model parameters
    parallel_model.print_params()

    # ##########################
    # Perform VCD reconstruction
    time0 = time.time()
    recon, recon_params = parallel_model.recon(sinogram, weights=weights)

    recon.block_until_ready()
    elapsed = time.time() - time0
    print('Elapsed time for recon is {:.3f} seconds'.format(elapsed))
    # ##########################

    # Display results
    pu.slice_viewer(phantom, recon, title='Phantom (left) vs VCD Recon (right)')

