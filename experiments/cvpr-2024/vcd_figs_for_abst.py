import numpy as np
import time
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mbirjax


def combined_display(granularity_sequence, loss, phantom, recon, granularity_ylim=None, loss_ylim=None):
    # Set global font size
    plt.rcParams.update({'font.size': 18})  # Adjust font size here

    # Calculate the center slice index
    center_slice_index = phantom.shape[2] // 2

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    fig.suptitle('Combined Display of Granularity-Loss and Center Image Slices', fontsize=16)  # Larger title

    # Granularity Plot as a stem plot
    ax1 = axes[0, 0]
    ax1.stem(granularity_sequence, linefmt='b-', markerfmt='bo', basefmt=' ', label='Granularity Sequence')
    ax1.set_ylabel('Granularity', color='b')
    ax1.legend(loc='upper left')
    if granularity_ylim:
        ax1.set_ylim(granularity_ylim)

    # Loss Plot on the same row but different column
    ax2 = axes[0, 1]
    ax2.plot(loss, label='Loss', color='r')
    ax2.set_ylabel('Loss', color='r')
    ax2.set_yscale('log')
    ax2.legend(loc='upper right')
    if loss_ylim:
        ax2.set_ylim(loss_ylim)

    # Phantom center slice display side by side with the reconstruction
    ax_phantom = axes[1, 0]
    img_phantom = ax_phantom.imshow(phantom[:, :, center_slice_index], cmap='gray')
    plt.colorbar(img_phantom, ax=ax_phantom)
    ax_phantom.set_title('Phantom - Center Slice')

    # Reconstruction center slice display next to the phantom
    ax_recon = axes[1, 1]
    img_recon = ax_recon.imshow(recon[:, :, center_slice_index], cmap='gray')
    plt.colorbar(img_recon, ax=ax_recon)
    ax_recon.set_title('Reconstruction - Center Slice')

    plt.tight_layout()
    plt.show()

# Example usage:
# combined_display(granularity_sequence, loss, phantom, recon, granularity_ylim=None, loss_ylim=None)


def display_slices_for_abstract( recon1, recon2, recon3, labels) :
    # Set global font size
    plt.rcParams.update({'font.size': 15})  # Adjust font size here

    vmin = 0.0
    vmax = phantom.max()
    slice_index = recon1.shape[2] // 2

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

    a0 = ax[0].imshow(recon1[:, :, slice_index], vmin=vmin, vmax=vmax, cmap='gray')
    #plt.colorbar(a0, ax=ax[0])
    ax[0].set_title(labels[0])

    a0 = ax[1].imshow(recon2[:, :, slice_index], vmin=vmin, vmax=vmax, cmap='gray')
    #plt.colorbar(a0, ax=ax[1])
    ax[1].set_title(labels[1])

    a2 = ax[2].imshow(recon3[:, :, slice_index], vmin=vmin, vmax=vmax, cmap='gray')
    #plt.colorbar(a2, ax=ax[2])
    ax[2].set_title(labels[2])

    plt.show()

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

    # Generate pictures for Figure 1
    num_recon_rows_fig = 32
    num_recon_cols_fig = 32
    granularity_fig = np.array([1, 2, 4, 8])
    partitions_fig = mbirjax.gen_set_of_voxel_partitions(num_recon_rows_fig, num_recon_cols_fig, granularity_fig)

    # Plot the set of partitions
    mbirjax.debug_plot_partitions(partitions_fig, num_recon_rows_fig, num_recon_cols_fig)




    # Best outcome so far
    granularity = np.array([1, 8, 64, 256])  # Desired granularity of each partition
    sequence_gd = mbirjax.gen_partition_sequence([0], num_iters)  # Pure Gradient Descent
    sequence_cd = mbirjax.gen_partition_sequence([3], num_iters)  # Pure Coordinate Descent
    sequence_vcd = mbirjax.gen_partition_sequence([0, 1, 2, 3, 1, 2, 3, 2, 3, 3], num_iters)  # MG-VCD

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
        error_sinogram, recon = parallel_model.vcd_partition_iteration(error_sinogram, recon, partitions[i], hessian,
                                                                       weights=weights)
    print('Done compiling.')
    # ##########################


    # ##########################
    # Multi-Granula VCD (MGVCD) reconstruction section

    # do Gradient Descent
    recon_gd, fm_rmse_gd = parallel_model.vcd_recon(sinogram, partitions, sequence_gd, weights=weights)
    recon_gd = recon_gd.reshape((num_recon_rows, num_recon_cols, num_recon_slices))
    print('Done Gradient Descent Reconstruction')

    # do Coordinate Descent
    recon_cd, fm_rmse_cd = parallel_model.vcd_recon(sinogram, partitions, sequence_cd, weights=weights)
    recon_cd = recon_cd.reshape((num_recon_rows, num_recon_cols, num_recon_slices))
    print('Done Coordinate Descent Reconstruction')

    # do Vector Coordinate Descent
    recon_vcd, fm_rmse_vcd = parallel_model.vcd_recon(sinogram, partitions, sequence_vcd, weights=weights)
    recon_vcd = recon_vcd.reshape((num_recon_rows, num_recon_cols, num_recon_slices))
    print('Done Vector Coordinate Descent Reconstruction')
    # ##########################

    # Display reconstructions
    labels = ['Gradient Descent', 'Coordinate Descent', 'Vectorized Coordinate Descent']
    display_slices_for_abstract(recon_gd, recon_cd, recon_vcd, labels)

    # Example usage:
    granularity_sequences = [granularity[sequence_gd], granularity[sequence_cd], granularity[sequence_vcd]]
    losses = [fm_rmse_gd, fm_rmse_cd, fm_rmse_vcd]
    labels = ['Gradient Descent', 'Coordinate Descent', 'Vectorized Coordinate Descent']
    fig_title = 'Granularity and Loss vs. Iteration Number for Different Reconstruction Methods'
    mbirjax.plot_granularity_and_loss(granularity_sequences, losses, labels, granularity_ylim=(0, 256), loss_ylim=(0.1, 15))
