import numpy as np
import time
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mbirjax
import mbirjax.parallel_beam
import mbirjax.plot_utils as pu


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
    parallel_model = mbirjax.parallel_beam.ParallelBeamModel(angles, sinogram.shape)

    # Generate 3D Shepp Logan phantom
    phantom = parallel_model.gen_3d_shepp_logan_phantom()

    # Generate synthetic sinogram data
    full_indices = parallel_model.gen_full_indices()
    voxel_values = parallel_model.get_voxels_at_indices(phantom, full_indices)
    sinogram = parallel_model.forward_project(voxel_values, full_indices)

    # Generate weights array
    weights = parallel_model.calc_weights(sinogram/sinogram.max(), weight_type='transmission_root')

    # Set reconstruction parameter values
    parallel_model.set_params(sharpness=sharpness, verbose=1)
    #parallel_model.set_params(positivity_flag=True)

    # Print out model parameters
    parallel_model.print_params()

    # ##########################
    # Perform VCD reconstruction
    recon_vcd, fm_rmse_vcd = parallel_model.recon(sinogram, weights=weights)
    granularity = np.array(parallel_model.get_params('granularity'))
    partition_sequence = parallel_model.gen_partition_sequence()
    granularity_sequence_vcd = granularity[partition_sequence]

    # Perform GD reconstruction
    parallel_model.set_params(partition_sequence=[0,])
    recon_gd, fm_rmse_gd = parallel_model.recon(sinogram, weights=weights)
    granularity = np.array(parallel_model.get_params('granularity'))
    partition_sequence = parallel_model.gen_partition_sequence()
    granularity_sequence_gd = granularity[partition_sequence]

    # Perform CD reconstruction
    parallel_model.set_params(partition_sequence=[3,])
    recon_cd, fm_rmse_cd = parallel_model.recon(sinogram, weights=weights)
    granularity = np.array(parallel_model.get_params('granularity'))
    partition_sequence = parallel_model.gen_partition_sequence()
    granularity_sequence_cd = granularity[partition_sequence]
    # ##########################


    # Display reconstructions
    labels = ['Gradient Descent', 'Vectorized Coordinate Descent', 'Coordinate Descent']
    display_slices_for_abstract(recon_gd, recon_vcd, recon_cd, labels)

    # Display granularity plots:
    granularity_sequences = [granularity_sequence_gd, granularity_sequence_vcd, granularity_sequence_cd]
    losses = [fm_rmse_gd, fm_rmse_vcd, fm_rmse_cd]
    labels = ['Gradient Descent', 'Vectorized Coordinate Descent', 'Coordinate Descent']
    pu.plot_granularity_and_loss(granularity_sequences, losses, labels, granularity_ylim=(0, 256), loss_ylim=(0.1, 15))


    # Generate sequence of partition images for Figure 1
    parallel_model.set_params(num_recon_rows=32, num_recon_cols=32, granularity=[1, 2, 4, 8, 16])
    partitions_fig = parallel_model.gen_set_of_voxel_partitions()

    # Plot the set of partitions
    pu.debug_plot_partitions(partitions=partitions_fig, num_recon_rows=32, num_recon_cols=32)
