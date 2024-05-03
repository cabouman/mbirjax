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
    # ##########################
    # Test batch size feature
    view_batch_size = 100
    voxel_batch_size = 10000

    # Set parameters
    num_iters = 10
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
    parallel_model = mbirjax.ParallelBeamModel(angles, sinogram.shape)

    # Here are other things you might want to do
    parallel_model.set_params(view_batch_size=view_batch_size, voxel_batch_size=voxel_batch_size)
    #parallel_model.set_params(num_recon_rows=256//4)    # You can make the recon rectangular
    #parallel_model.set_params(delta_pixel_recon=1.0)    # You can change the pixel pitch
    #parallel_model.set_params(det_channel_offset=10.5)    # You can change the center-of-rotation in the sinogram
    #parallel_model.set_params(granularity=[1, 8, 64, 256], partition_sequence=[0, 1, 2, 3, 2, 3, 2, 3, 3, 3, 3, 3, 3], num_iterations=13) # You can change the iterations

    # Generate 3D Shepp Logan phantom
    phantom = parallel_model.gen_3d_sl_phantom()

    # Generate synthetic sinogram data
    sinogram = parallel_model.forward_project(phantom)

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

    # Display results
    pu.slice_viewer(phantom, recon_3d, title='Phantom (left) vs VCD Recon (right)')

    # You can also display individual slides with the sinogram
    #pu.display_slices(phantom, sinogram, recon_3d)
