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

    # Initialize sinogram
    sinogram = jnp.zeros((num_views, num_det_rows, num_det_channels))
    angles = jnp.linspace(start_angle, np.pi, num_views, endpoint=False)

    # Set up parallel beam model
    parallel_model = mbirjax.ParallelBeamModel(angles, sinogram.shape)

    # Generate 3D Shepp Logan phantom
    phantom = parallel_model.generate_3d_shepp_logan()

    # Generate set of voxel partitions
    partitions = parallel_model.gen_set_of_voxel_partitions()

    # Generate sequence of partitions to use
    partition_sequence = parallel_model.gen_partition_sequence()

    # Generate sinogram data
    full_indices = parallel_model.gen_full_partitions()
    voxel_values = phantom.reshape((-1, num_det_rows))[full_indices]
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
    # Perform VCD reconstruction
    time0 = time.time()
    recon, fm_rmse = parallel_model.vcd_recon(sinogram, partitions, partition_sequence, weights=weights)

    elapsed = time.time() - time0
    print('Elapsed post-compile time recon is {:.3f} seconds'.format(elapsed))
    # ##########################

    # Reshape recon into 3D form
    recon = parallel_model.reshape_recon(recon)

    # Display results
    display_slices(phantom, sinogram, recon)
