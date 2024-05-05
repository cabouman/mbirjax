import matplotlib.pyplot as plt
import numpy as np


def display_slices(phantom, sinogram, recon):
    """
    Displays slices of phantom, sinogram, and reconstructed image side by side for comparison.

    This function iterates over each slice allowing the user to manually advance through slices by pressing Enter.

    Parameters:
        phantom (numpy array): The phantom in (row, column, slice) format.
        sinogram (numpy array): The sinogram in (view, row, column) format.
        recon (numpy array): The recon in (row, column, slice) format.
    """
    num_recon_slices = phantom.shape[2]
    vmin = 0.0
    vmax = phantom.max()
    vsinomax = sinogram.max()

    for slice_index in range(num_recon_slices):
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


global slice_index, ax, fig, cbar, img, vertical_line


from mpl_toolkits.axes_grid1 import make_axes_locatable


def slice_viewer(data, data2=None, title=''):
    """
    Display slices of one or two 3D image volumes with a consistent grayscale across slices.
    Allows interactive selection of slices via a draggable line on a colorbar-like axis. If two images are provided,
    they are displayed side by side for comparative purposes.

    Args:
        data (numpy.ndarray or jax.numpy.DeviceArray): 3D image volume with shape (height, width, depth).
        data2 (numpy.ndarray or jax.numpy.DeviceArray, optional): Second 3D image volume with the same shape as the first.
        title (string, optional, default=''): Figure super title

    The function sets up a matplotlib figure with interactive controls to view different slices
    by clicking and dragging on a custom colorbar. Each slice is displayed using the same grayscale range
    determined by the global min and max of the entire volume.
    """
    global slice_index, cbar, vertical_line
    slice_index = data.shape[2] // 2  # Initial slice index

    # Define min and max grayscale values for consistent coloring across slices
    vmin = min(data.min(), data2.min() if data2 is not None else data.min())
    vmax = max(data.max(), data2.max() if data2 is not None else data.max())

    def update_slice(x):
        """Update the displayed slice based on the position of the mouse click or drag on the colorbar axis."""
        global slice_index, vertical_line
        slice_index = int(0.5 + x / ax_colorbar.get_xlim()[1] * data.shape[2])
        vertical_line.set_xdata([slice_index, slice_index])
        redraw_fig()

    def redraw_fig(show_colorbar=False):
        """Redraw the figure to update the slice and its display.
        The colorbar is drawn only at initialization.
        """
        ax.clear()
        if data2 is not None:
            image_divider = vmax * np.ones((data.shape[0], 5))
            im = ax.imshow(np.concatenate((data[:, :, slice_index], image_divider, data2[:, :, slice_index]), axis=1), cmap='gray',
                      vmin=vmin, vmax=vmax)
            ax.set_title(f'Slice {slice_index} Comparison')
        else:
            im = ax.imshow(data[:, :, slice_index], cmap='gray', vmin=vmin, vmax=vmax)
            ax.set_title(f'Slice {slice_index}')
        if show_colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')
        fig.canvas.draw_idle()

    def on_press(event):
        """Handle mouse press events for interactive slice selection."""
        if event.inaxes == ax_colorbar:
            update_slice(event.xdata)

    def on_motion(event):
        """Handle mouse motion events for continuous slice selection while dragging."""
        if event.inaxes == ax_colorbar and event.button == 1:
            update_slice(event.xdata)

    # Setup the plot
    plt.ion()  # Turn on interactive mode
    fig, axes = plt.subplots(3, 1, gridspec_kw={'height_ratios': [10, 1, 0.5]})
    fig.suptitle(title)
    ax, ax_colorbar, ax_instruction = axes

    # Setup the interactive vertical line in the slider
    ax_colorbar.set_xlim(0, data.shape[2])
    ax_colorbar.set_ylim(0, 1)
    vertical_line = ax_colorbar.axvline(slice_index, color='black', linewidth=4)  # Movable line
    ax_colorbar.set_facecolor('white')
    ax_colorbar.set_yticks([])
    ax_colorbar.set_xticks([])

    # Add a label below the slider
    ax_instruction.text(0.5, 0.5, 'Click and drag to change slice', ha='center', va='center', fontsize=10)
    ax_instruction.set_axis_off()

    # Connect events
    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)

    # Initial drawing
    redraw_fig(show_colorbar=True)  # Call redraw to handle initial display for single or dual images
    plt.show()

    plt.pause(0.1)  # Delay to ensure window stays open
    input("Press any key to close ")

    plt.ioff()  # Turn off interactive mode
    plt.close()


# Example usage:
# data1 = np.random.rand(100, 100, 50)  # Random 3D volume
# data2 = np.random.rand(100, 100, 50)  # Another random 3D volume
# slice_viewer(data1, data2)  # View slices of both volumes side by side


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
        ax.set_title(f'{len(partition)} Partitions')
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


def plot_granularity_and_loss(granularity_sequences, losses, labels, granularity_ylim=None, loss_ylim=None,
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

    for ax, granularity_sequence, loss, label in zip(axes, granularity_sequences, losses, labels):
        index = list(range(len(granularity_sequence)))

        # Plot granularity sequence on the first y-axis
        ax1 = ax
        ax1.stem(index, granularity_sequence, label='Granularity Sequence', basefmt=" ", linefmt='b', markerfmt='bo')
        ax1.set_ylabel('Granularity', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        if granularity_ylim:
            ax1.set_ylim(granularity_ylim)  # Apply fixed y-limit for granularity

        # Create a second y-axis for the loss
        ax2 = ax1.twinx()
        ax2.plot(index, loss, label='Loss', color='r')
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
