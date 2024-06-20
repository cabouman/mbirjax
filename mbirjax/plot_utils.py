import matplotlib.pyplot as plt
import numpy
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import gridspec
from matplotlib.widgets import RangeSlider, Slider


def slice_viewer(data, data2=None, title='', vmin=None, vmax=None, slice_label='Slice', slice_label2=None,
                 slice_axis=2, slice_axis2=None, cmap='gray'):
    """
    Display slices of one or two 3D image volumes with a consistent grayscale across slices.
    Allows interactive selection of slices and intensity window. If two images are provided,
    they are displayed side by side for comparative purposes.  If the two images have the same shape, then zoom
    and pan will be applied to each image simultaneously.  If not pan and zoom are applied to one image at a time.

    Args:
        data (ndarray or jax array): 3D image volume with shape (height, width, depth) or 2D (height, width).
        data2 (numpy array or jax array, optional): Second 2D or 3D image volume with the same constraints.
        title (string, optional, default=''): Figure super title
        vmin (float, optional): minimum for displayed intensity
        vmax (float, optional): maximum for displayed intensity
        slice_label (str, optional): Text label to be used for a given slice.  Defaults to 'Slice'
        slice_label2 (str, optional): Text label to be used for a given slice for data2.  Defaults to slice_label.
        slice_axis (int, optional): The dimension of data to use for the slice index.  That is, if slice_axis=1, then the
            displayed images will be data[:, slice_index, :]
        slice_axis2 (int, optional): The dimension of data to use for the slice index for data2.

    Example:
        .. code-block:: python

            data1 = np.random.rand(100, 100, 50)  # Random 3D volume
            data2 = np.random.rand(100, 100, 50)  # Another random 3D volume
            slice_viewer(data1, data2, slice_axis=2, title='Slice Demo', slice_label='Current slice')  # View slices of both volumes side by side
    """
    if data.ndim < 2 or data.ndim > 3:  # or (data2 is not None and data.shape[slice_axis] != data2.shape[slice_axis]):
        error_msg = 'The input data must be a 2D or 3D array'  #, and if data2 is provided, then data.shape[slice_axis] '
        # error_msg += 'must equal data2.shape[slice_axis])'
        raise ValueError(error_msg)
    elif data.ndim == 2:
        data = data.reshape(data.shape + (1,))
    if data2 is not None:
        if data2.ndim < 2 or data2.ndim > 3:
            error_msg = 'The input data2 must be a 2D or 3D array'
            raise ValueError(error_msg)
        elif data2.ndim == 2:
            data2 = data2.reshape(data2.shape + (1,))

    # Move the specified slice axis into the last position
    data = numpy.moveaxis(data, slice_axis, 2)
    if data2 is not None:
        if slice_axis2 is None:
            slice_axis2 = slice_axis
        data2 = numpy.moveaxis(data2, slice_axis2, 2)
        if slice_label2 is None:
            slice_label2 = slice_label

    # Define min and max grayscale values for consistent coloring across slices
    if vmin is None:
        vmin = min(data.min(), data2.min()) if data2 is not None else data.min()
    if vmax is None:
        vmax = max(data.max(), data2.max()) if data2 is not None else data.max()
    if vmin > vmax:
        raise ValueError('vmin must be less than or equal to vmax')
    if vmin == vmax:
        eps = 1e-6  # This is not floating point epsilon, just a small number for display purposes.
        scale = np.clip(eps * np.abs(vmax), a_min=eps, a_max=None)
        vmin = vmin - scale
        vmax = vmax + scale

    # Set up the plot
    figwidth = 6 if data2 is None else 10
    fig = plt.figure(figsize=(figwidth, 6))
    fig.suptitle(title)

    # Set up the subplots. One or two images in the first row along with colorbars.
    gs = gridspec.GridSpec(nrows=3, ncols=2, height_ratios=[10, 1, 1])
    ax_data, ax_data2 = None, None
    if data2 is None:
        ax_data = fig.add_subplot(gs[0, :])
    else:
        ax_data = fig.add_subplot(gs[0, 0])
        share_axis = ax_data if data.shape == data2.shape else None
        ax_data2 = fig.add_subplot(gs[0, 1], sharex=share_axis, sharey=share_axis)

    # Show the initial slice
    slice_index = data.shape[2] // 2
    im = ax_data.imshow(data[:, :, slice_index], cmap=cmap, vmin=vmin, vmax=vmax)
    im2 = None
    if data2 is not None:
        slice_index2 = data2.shape[2] // 2
        im2 = ax_data2.imshow(data2[:, :, slice_index2], cmap=cmap, vmin=vmin, vmax=vmax)
        ax_data2.set_title(f'{slice_label2} {slice_index2}')

    # Set up a colorbar next to the rightmost image, but add extra space to both to make them the same size.
    divider = make_axes_locatable(ax_data)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    if data2 is not None:
        divider2 = make_axes_locatable(ax_data2)
        cax.axis('off')
        cax = divider2.append_axes('right', size='5%', pad=0.05)

    fig.colorbar(im, cax=cax, orientation='vertical')

    # Then add the slice slider
    ax_slice_slider = fig.add_subplot(gs[1, :])
    slice_slider = Slider(ax=ax_slice_slider, label=slice_label, valmin=0, valmax=data.shape[2] - 1,
                          valinit=slice_index, valfmt='%0.0f')

    # Then the intensity slider
    ax_intensity_slider = fig.add_subplot(gs[2, :])
    log_intensity_range = np.log10(vmax - vmin)
    num_digits = max(- int(np.round(log_intensity_range)) + 2, 0)
    valfmt = '%0.' + str(num_digits) + 'f'
    valinit = (vmin, vmax)
    intensity_slider = RangeSlider(ax_intensity_slider, "Intensity\nrange", vmin, vmax,
                                   valinit=valinit, valfmt=valfmt)

    ax_data.set_title(f'{slice_label} {slice_index}')

    fig.text(0.01, 0.95, 'Close plot \nto continue')

    # Set up the callback functions for the sliders
    def update_intensity(val):
        # The val passed to a callback by the RangeSlider will
        # be a tuple of (min, max), which are used to set vmin and vmax for both images.

        # Update the image's colormap
        im.norm.vmin = val[0]
        im.norm.vmax = val[1]

        if data2 is not None:
            im2.norm.vmin = val[0]
            im2.norm.vmax = val[1]

        # Redraw the figure to ensure it updates
        fig.canvas.draw_idle()

    def update_slice(val):
        # The val passed to a callback by the Slider is a single float.  We cast it to an int to index the slice.
        cur_slice = int(np.round(val))
        im.set_data(data[:, :, cur_slice])
        ax_data.set_title(f'{slice_label} {cur_slice}')
        if data2 is not None:
            slice_fraction = cur_slice / data.shape[2]
            cur_slice2 = int(np.round(slice_fraction * data2.shape[2]))
            im2.set_data(data2[:, :, cur_slice2])
            ax_data2.set_title(f'{slice_label2} {cur_slice2}')

        # Redraw the figure to ensure it updates
        fig.canvas.draw_idle()

    # Connect the sliders to the callback functions
    intensity_slider.on_changed(update_intensity)
    slice_slider.on_changed(update_slice)

    plt.tight_layout()
    plt.show()


def debug_plot_partitions(partitions, recon_shape):
    """
    Visualizes a set of partitions as color images in a single row, where each partition is represented by a different color.

    Parameters:
        partitions (tuple of arrays): A tuple where each element is a 2D numpy array representing a partition.
        recon_shape (tuple): Shape of phantom in (rows, columns, slices).
    """
    num_recon_rows, recon_shape = recon_shape[:2]
    plt.rcParams.update({'font.size': 24})  # Adjust font size here
    num_partitions = len(partitions)
    fig, axes = plt.subplots(nrows=1, ncols=num_partitions, figsize=(5 * num_partitions, 5))

    for i, partition in enumerate(partitions):
        # Create an empty image array to fill with subset colors
        image = np.zeros((num_recon_rows * recon_shape), dtype=int)

        # Assign a unique color (integer label) to each subset
        for subset_index, indices in enumerate(partition):
            image[indices.flatten()] = subset_index + 1  # Color code starts from 1 upwards

        # Reshape the image array back to 2D format
        image = image.reshape((num_recon_rows, recon_shape))

        # Plotting
        if num_partitions == 1:
            ax = axes
        else:
            ax = axes[i]

        cax = ax.imshow(image, cmap='nipy_spectral', interpolation='nearest')
        ax.set_title(f'{len(partition)} Subsets')
        ax.axis('off')  # Turn off axis labels

    plt.tight_layout()
    plt.show()


def debug_plot_indices(num_recon_rows, num_recon_cols, indices, recon_at_indices=None, num_recon_slices=1,
                       title='Debug Plot'):
    """
    Visualizes indices on a reconstruction grid and optionally displays reconstruction data at these indices.

    Parameters:
        num_recon_rows (int): Number of rows in the reconstruction grid.
        num_recon_cols (int): Number of columns in the reconstruction grid.
        indices (array): Flat indices in the reconstruction grid to be highlighted or modified.
        recon_at_indices (array, optional): Values to set at specified indices in the reconstruction grid. If provided,
                                            displays the reconstruction at these indices across slices.
        num_recon_slices (int): Number of slices in the reconstruction grid, default is 1.
        title (str): Title for the plot.

    Usage:
        When recon_at_indices is not provided, the function visualizes the indices on a 2D grid.
        When recon_at_indices is provided, it also shows the reconstructed values at these indices in 3D.

    Example:
        debug_plot_indices_or_reconstruction(100, 100, [5050, 10001], recon_at_indices=[1, -1], num_recon_slices=5, title='Recon Visualization')
    """
    # Create an empty grid
    recon = np.zeros((num_recon_rows * num_recon_cols, num_recon_slices))

    # Create a mask for indices
    mask = np.zeros(num_recon_rows * num_recon_cols)
    mask[indices] = 1  # Highlight indices
    mask = mask.reshape((num_recon_rows, num_recon_cols))

    # If reconstruction data is provided, add it to the recon grid
    if recon_at_indices is not None:
        for i, idx in enumerate(indices):
            recon[idx, :] += recon_at_indices[i]

    # Visualization
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    # Display the mask of indices
    im_mask = ax[0].imshow(mask, cmap='viridis')
    ax[0].set_title('Mask of Indices')
    plt.colorbar(im_mask, ax=ax[0])

    if recon_at_indices is not None:
        # Display the reconstructed values if provided
        im_recon = ax[1].imshow(recon[:, 0].reshape((num_recon_rows, num_recon_cols)), cmap='viridis')
        ax[1].set_title(f'{title} at Indices')
        plt.colorbar(im_recon, ax=ax[1])
    else:
        ax[1].axis('off')  # Turn off the second subplot if no reconstruction data is provided

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_granularity_and_loss(granularity_sequences, fm_losses, prior_losses, labels, granularity_ylim=None, loss_ylim=None,
                              fig_title=None):
    """
    Plots multiple granularity and loss data sets on a single figure.

    Args:
        granularity_sequences (list of lists): A list containing different granularity sequences.
        losses (list of lists): A list containing different loss data corresponding to the granularity sequences.
        labels (list of str): Labels for each subplot to distinguish between different data sets.
        granularity_ylim (tuple, optional): Limits for the granularity axis (y-limits), applied to all plots.
        loss_ylim (tuple, optional): Limits for the loss axis (y-limits), applied to all plots.
    """
    num_plots = len(granularity_sequences)
    fig, axes = plt.subplots(nrows=1, ncols=num_plots, figsize=(5 * num_plots, 4), sharey='row')
    fig.suptitle(fig_title)

    if num_plots == 1:
        axes = [axes]  # Make it iterable for a single subplot scenario

    for ax, granularity_sequence, fm_loss, prior_loss, label in zip(axes, granularity_sequences, fm_losses, prior_losses, labels):
        index = list(1 + np.arange(len(granularity_sequence)))

        # Plot granularity sequence on the first y-axis
        ax1 = ax
        ax1.stem(index, granularity_sequence, label='Granularity Sequence', basefmt=" ", linefmt='b', markerfmt='bo')
        ax1.set_ylabel('Granularity', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        if granularity_ylim:
            ax1.set_ylim(granularity_ylim)  # Apply fixed y-limit for granularity

        # Create a second y-axis for the loss
        ax2 = ax1.twinx()
        ax2.plot(index, fm_loss, label='Data loss', color='r')
        ax2.plot(index, prior_loss, label='Prior loss', color='g')
        ax2.set_ylabel('Loss', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.set_yscale('log')
        if loss_ylim:
            ax2.set_ylim(loss_ylim)  # Apply fixed y-limit for loss, ensure log scale is considered

        # Set labels and legends
        ax1.set_xlabel('Iteration Number')
        ax.set_title(label)

        # Add legends
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

    plt.tight_layout()
    plt.show()
