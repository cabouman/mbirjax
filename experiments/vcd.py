import numpy as np
import time
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mbirjax


def display_slices( phantom, sinogram, recon ) :
    num_recon_slices = phantom.shape[2]
    vmin = 0.0
    vmax = phantom.max()
    vsinomax = sinogram.max()

    for slice_index in range(num_recon_slices) :
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
        fig.suptitle('Demo of VCD reconstruction - Slice {}'.format(slice_index))

        # Display original phantom slice
        a0 = ax[0].imshow(phantom[:, :, slice_index], vmin=vmin, vmax=vmax, cmap='gray')
        plt.colorbar(a0, ax=ax[0])
        ax[0].set_title('Original Phantom')

        # Display sinogram slice
        a1 = ax[1].imshow(sinogram[:, slice_index, :], vmin=vmin, vmax=vsinomax, cmap='gray')
        plt.colorbar(a1, ax=ax[1])
        ax[1].set_title('Sinogram')

        # Display reconstructed slice
        a2 = ax[2].imshow(recon[:, :, slice_index], vmin=vmin, vmax=vmax, cmap='gray')
        plt.colorbar(a2, ax=ax[2])
        ax[2].set_title('VCD Reconstruction')

        plt.show(block=False)
        input("Press Enter to continue to the next slice or type 'exit' to quit: ").strip().lower()
        plt.close(fig)
        if input() == 'exit':
            break


if __name__ == "__main__":
    """
    This is a script to develop, debug, and tune the vcd reconstruction with a parallel beam projector
    """
    # Set parameters
    num_iters = 10
    num_views = 256
    num_det_rows = 10
    num_det_channels = 256
    start_angle = 0
    end_angle = np.pi
    sharpness = 0.0

    # Best outcome so far
    granularity = np.array([1, 8, 64, 256])  # Desired granularity of each partition

    # Best outcome so far
    granularity = np.array([1, 8, 64, 256])  # Desired granularity of each partition
    sequence = mbirjax.gen_partition_sequence([0, 1, 2, 3, 1, 2, 3, 2, 3, 3], num_iters)  # MG-VCD

    # Initialize sinogram
    sinogram = jnp.zeros((num_views, num_det_rows, num_det_channels))
    angles = jnp.linspace(start_angle, np.pi, num_views, endpoint=False)

    # Set up parallel beam model
    parallel_model = mbirjax.ParallelBeamModel(angles, sinogram.shape)

    # Generate phantom
    num_recon_rows, num_recon_cols, num_recon_slices = (
        parallel_model.get_params(['num_recon_rows', 'num_recon_cols', 'num_recon_slices']))
    phantom = mbirjax.generate_3d_shepp_logan(num_recon_rows, num_recon_cols, num_recon_slices)

    # Generate set of voxel partitions
    partitions = mbirjax.gen_set_of_voxel_partitions(num_recon_rows, num_recon_cols, granularity)

    # Generate sinogram data
    full_indices = partitions[0]
    voxel_values = phantom.reshape((-1, num_recon_slices))[full_indices]
    sinogram = parallel_model.forward_project(voxel_values[0], full_indices[0])

    # Generate weights array
    weights = parallel_model.calc_weights(sinogram/sinogram.max(), weight_type='transmission_root')

    # Autoset reconstruction parameter values
    # Bug?: After setting a meta parameter like sharpness, we need to run auto_set_regularization
    parallel_model.set_params(sharpness=sharpness,verbose=1)
    parallel_model.auto_set_regularization_params(sinogram, weights=weights)

    # Print out model parameters
    parallel_model.print_params()

    # ##########################
    # Warm up projectors for accurate timing
    error_sinogram = sinogram
    recon = jnp.zeros((num_recon_rows, num_recon_cols, num_recon_slices))
    hessian = parallel_model.compute_hessian_diagonal(weights=weights, angles=angles)

    # Run each partition once to finish compiling
    print('Compiling ...')
    for i in range(len(granularity)):
        error_sinogram, recon = parallel_model.vcd_iteration(error_sinogram, recon, partitions[i], hessian,
                                                             weights=weights)
    print('Done compiling.')
    # ##########################


    # ##########################
    # Multi-Granularity VCD (MGVCD) reconstruction section
    time0 = time.time()
    recon, fm_rmse = parallel_model.vcd_recon(sinogram, partitions, sequence, weights=weights)

    elapsed = time.time() - time0
    print('Elapsed post-compile time for {} iterations with {}x{}x{} recon is {:.3f} '
          'seconds'.format(num_iters, num_recon_rows, num_recon_cols, num_recon_slices, elapsed))
    recon = recon.reshape((num_recon_rows, num_recon_cols, num_recon_slices))
    # ##########################

    # Plot granularity sequence
    labels = ['Vectorized Coordinate Descent',]
    granularity_sequences = [granularity[sequence],]
    loss_sequences = [fm_rmse,]
    mbirjax.plot_granularity_and_loss(granularity_sequences=granularity_sequences, losses=loss_sequences, labels=labels, granularity_ylim=(0, 256), loss_ylim=(0.1, 15))

    # Display results
    display_slices(phantom, sinogram, recon)
