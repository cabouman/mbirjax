import numpy as np
import time
import jax.numpy as jnp
import mbirjax
import mbirjax.parallel_beam
import mbirjax.plot_utils as pu

if __name__ == "__main__":
    """
    This is a script to develop, debug, and tune the vcd reconstruction with a parallel beam projector
    """
    # Set parameters
    num_iters = 10
    num_views = 256
    num_det_rows = 5
    num_det_channels = 256
    start_angle = 0
    end_angle = np.pi
    sharpness = 0.0

    # Initialize sinogram
    sinogram = jnp.zeros((num_views, num_det_rows, num_det_channels))
    angles = jnp.linspace(start_angle, np.pi, num_views, endpoint=False)

    # Set up parallel beam model
    parallel_model = mbirjax.parallel_beam.ParallelBeamModel(angles, sinogram.shape)

    # Generate 3D Shepp Logan phantom
    phantom = parallel_model.gen_3d_sl_phantom()

    # Generate synthetic sinogram data
    full_indices = parallel_model.gen_full_indices()
    voxel_values = parallel_model.get_voxels_at_indices(phantom, full_indices)
    sinogram = parallel_model.forward_project(voxel_values, full_indices)

    # Generate weights array
    weights = parallel_model.gen_weights(sinogram / sinogram.max(), weight_type='transmission_root')

    # Set reconstruction parameter values
    parallel_model.set_params(sharpness=sharpness, verbose=1)
    # parallel_model.set_params(positivity_flag=True)

    # Print out model parameters
    parallel_model.print_params()

    # ##########################
    # Perform VCD reconstruction
    time0 = time.time()
    recon, fm_rmse = parallel_model.recon(sinogram, weights=weights)

    recon.block_until_ready()
    elapsed = time.time() - time0
    print('Elapsed time for recon is {:.3f} seconds'.format(elapsed))
    # ##########################

    # Reshape recon into 3D form
    recon_3d = parallel_model.reshape_recon(recon)

    pu.slice_viewer(recon_3d)
    # Display results
    # pu.display_slices(phantom, sinogram, recon_3d)
