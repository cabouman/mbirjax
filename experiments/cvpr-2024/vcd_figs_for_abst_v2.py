import numpy as np
import time
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mbirjax
import mbirjax.parallel_beam


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
    # Choose the geometry type
    geometry_type = 'cone'  # 'cone' or 'parallel'

    print('Using {} geometry.'.format(geometry_type))

    # Set parameters
    num_views = 256
    num_det_rows = 10
    num_det_channels = 256
    start_angle = 0
    end_angle = np.pi
    sharpness = 0.0

    # These can be adjusted to describe the geometry in the cone beam case.
    # np.Inf is an allowable value, in which case this is essentially parallel beam
    source_detector_dist = 4 * num_det_channels
    source_iso_dist = source_detector_dist

    if geometry_type == 'cone':
        detector_cone_angle = 2 * np.arctan2(num_det_channels / 2, source_detector_dist)
    else:
        detector_cone_angle = 0
    start_angle = -(np.pi + detector_cone_angle) * (1 / 2)
    end_angle = (np.pi + detector_cone_angle) * (1 / 2)

    # Initialize sinogram
    sinogram = jnp.zeros((num_views, num_det_rows, num_det_channels))
    angles = jnp.linspace(start_angle, end_angle, num_views, endpoint=False)

    # Set up the model
    if geometry_type == 'cone':
        ct_model = mbirjax.ConeBeamModel(sinogram.shape, angles, source_detector_dist=source_detector_dist,
                                         source_iso_dist=source_iso_dist)
    elif geometry_type == 'parallel':
        ct_model = mbirjax.ParallelBeamModel(sinogram.shape, angles)
    else:
        raise ValueError('Invalid geometry type.  Expected cone or parallel, got {}'.format(geometry_type))

    # Generate 3D Shepp Logan phantom
    phantom = ct_model.gen_modified_3d_sl_phantom()

    # Generate synthetic sinogram data
    sinogram = ct_model.forward_project(phantom)

    # Generate weights array
    weights = ct_model.gen_weights(sinogram / sinogram.max(), weight_type='transmission_root')

    # Set reconstruction parameter values
    ct_model.set_params(sharpness=sharpness, verbose=1)

    # Print out model parameters
    ct_model.print_params()

    # 'granularity': {'val': [1, 4, 64, 128], 'recompile_flag': False},
    # 'partition_sequence': {'val': [0, 1, 2, 2, 2], 'recompile_flag': False},

    granularity_alt_1 = [4, 48, 96, 128]
    partition_sequence_alt_1 = [0, 1, 2, 2, 2, 2, 3]

    granularity_alt_2 = [2, 48, 96, 128]
    partition_sequence_alt_2 = [0, 1, 2, 2, 2, 2, 3]

    # ##########################
    # Perform default VCD reconstruction
    print('Starting default sequence')
    num_iterations = 8
    recon_default, recon_params_default = ct_model.recon(sinogram, weights=weights, num_iterations=num_iterations,
                                                       compute_prior_loss=True)
    fm_rmse_default = recon_params_default.fm_rmse
    prior_loss_default = recon_params_default.prior_loss
    partition_sequence = recon_params_default.partition_sequence
    granularity = np.array(recon_params_default.granularity)
    granularity_sequence_default = granularity[partition_sequence]
    label_default = 'Default: ' + str(granularity_sequence_default)

    # Perform alt_1 default reconstruction
    print('Starting alt_1 sequence')
    ct_model.set_params(partition_sequence=partition_sequence_alt_1)
    granularity = np.array(granularity_alt_1)
    ct_model.set_params(granularity=granularity)
    recon_alt_1, recon_params_alt_1 = ct_model.recon(sinogram, weights=weights, num_iterations=num_iterations,
                                                       compute_prior_loss=True)
    fm_rmse_alt_1 = recon_params_alt_1.fm_rmse
    prior_loss_alt_1 = recon_params_alt_1.prior_loss
    partition_sequence = recon_params_alt_1.partition_sequence
    granularity = np.array(recon_params_alt_1.granularity)
    granularity_sequence_alt_1 = granularity[partition_sequence]
    label_alt_1 = 'alt_1: ' + str(granularity_sequence_alt_1)

    # Perform alt_2 reconstruction
    print('Starting alt_2 sequence')
    ct_model.set_params(partition_sequence=partition_sequence_alt_2)
    granularity = np.array(granularity_alt_2)
    ct_model.set_params(granularity=granularity)
    recon_alt_2, recon_params_alt_2 = ct_model.recon(sinogram, weights=weights, num_iterations=num_iterations,
                                                       compute_prior_loss=True)
    fm_rmse_alt_2 = recon_params_alt_2.fm_rmse
    prior_loss_alt_2 = recon_params_alt_2.prior_loss
    partition_sequence = recon_params_alt_2.partition_sequence
    granularity = np.array(recon_params_alt_2.granularity)
    granularity_sequence_alt_2 = granularity[partition_sequence]
    label_alt_2 = 'alt_2: ' + str(granularity_sequence_alt_2)
    # ##########################

    # Display reconstructions
    labels = [label_alt_1, label_default, label_alt_2]
    # display_slices_for_abstract(recon_alt_1, recon_default, recon_alt_2, labels)

    # Display granularity plots:
    granularity_sequences = [granularity_sequence_alt_1, granularity_sequence_default, granularity_sequence_alt_2]
    fm_losses = [fm_rmse_alt_1, fm_rmse_default, fm_rmse_alt_2]
    prior_losses = [prior_loss_alt_1, prior_loss_default, prior_loss_alt_2]
    # labels = ['Gradient Descent', 'Vectorized Coordinate Descent', 'Coordinate Descent']
    mbirjax.plot_granularity_and_loss(granularity_sequences, fm_losses, prior_losses, labels, granularity_ylim=(0, 256), loss_ylim=(0.1, 15))

    # Generate sequence of partition images for Figure 1
    recon_shape = (32, 32, 1)
    partitions_fig = mbirjax.gen_set_of_pixel_partitions(recon_shape=recon_shape, granularity=[1, 4, 16, 64, 256])

    # Plot the set of partitions
    mbirjax.debug_plot_partitions(partitions=partitions_fig, recon_shape=recon_shape)
