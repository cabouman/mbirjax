import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

global slice_index, slice_line, vmin_cur, vmax_cur, vmin_line, vmax_line, intensity_line, ax, fig, cur_fig, img


def slice_viewer(data, data2=None, title='', vmin=None, vmax=None, slice_label='Slice'):
    """
    Display slices of one or two 3D image volumes with a consistent grayscale across slices.
    Allows interactive selection of slices via a draggable line on a colorbar-like axis. If two images are provided,
    they are displayed side by side for comparative purposes.

    Args:
        data (numpy.ndarray or jax.numpy.DeviceArray): 3D image volume with shape (height, width, depth).
        data2 (numpy.ndarray or jax.numpy.DeviceArray, optional): Second 3D image volume with the same shape as the first.
        title (string, optional, default=''): Figure super title
        vmin (float): minimum for displayed intensity
        vmax (float): maximum for displayed intensity
        slice_label (str): Text label to be used for a given slice.  Defaults to 'Slice'

    The function sets up a matplotlib figure with interactive controls to view different slices
    by clicking and dragging on a custom colorbar. Each slice is displayed using the same grayscale range
    determined by the global min and max of the entire volume.
    """
    global slice_index, slice_line, vmin_cur, vmax_cur, vmin_line, vmax_line, intensity_line, cur_fig

    slice_index = data.shape[2] // 2  # Initial slice index

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

    vmin_cur, vmax_cur = vmin, vmax
    cur_fig = None

    def update_slice(x):
        """Update the displayed slice based on the position of the mouse click or drag on the colorbar axis."""
        global slice_index, slice_line, cur_fig
        slice_index = int(0.5 + x / ax_slice_slider.get_xlim()[1] * (data.shape[2] - 1))
        slice_line.set_xdata([slice_index, slice_index])
        redraw_fig(cur_fig)

    def update_intensity(x):
        global vmin_cur, vmax_cur, vmin_line, vmax_line, intensity_line, cur_fig
        # Determine whether x is closer to vmin_cur or vmax_cur and set the line appropriately
        if x - vmin_cur < vmax_cur - x:
            vmin_cur = x
            vmin_line.set_xdata([vmin_cur, vmin_cur])
        else:
            vmax_cur = x
            vmax_line.set_xdata([vmax_cur, vmax_cur])

        xmin_cur = (vmin_cur - vmin) / (vmax - vmin)
        xmax_cur = (vmax_cur - vmin) / (vmax - vmin)

        intensity_line.set_xdata([xmin_cur, xmax_cur])
        redraw_fig(cur_fig)

    def redraw_fig(fig, first_pass=False):
        """Redraw the figure to update the slice and its display.
        The colorbar is drawn only at initialization.
        """
        ax.clear()
        if data2 is not None:
            image_divider = vmax * np.ones((data.shape[0], 5))
            cur_data = np.concatenate((data[:, :, slice_index], image_divider, data2[:, :, slice_index]), axis=1)
            ax.set_title(f'{slice_label} {slice_index} Comparison')
        else:
            cur_data = data[:, :, slice_index]
            ax.set_title(f'{slice_label} {slice_index}')
        im = ax.imshow(np.clip(cur_data, vmin_cur, vmax_cur), cmap='gray', vmin=vmin_cur, vmax=vmax_cur)
        if not first_pass:
            fig.axes[5].remove()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)

        fig.colorbar(im, cax=cax, orientation='vertical')
        fig.canvas.draw_idle()

    def on_press(event):
        """Handle mouse press events for interactive slice selection."""
        if event.inaxes == ax_slice_slider:
            update_slice(event.xdata)
        if event.inaxes == ax_intensity_slider:
            update_intensity(event.xdata)

    def on_motion(event):
        """Handle mouse motion events for continuous slice selection while dragging."""
        if event.inaxes == ax_slice_slider and event.button == 1:
            update_slice(event.xdata)
        if event.inaxes == ax_intensity_slider and event.button == 1:
            update_intensity(event.xdata)

    # Setup the plot
    plt.ion()  # Turn on interactive mode
    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(6, 6), gridspec_kw={'height_ratios': [10, 1, 0.5, 1, 0.5]})
    fig.suptitle(title)
    ax = axes[0]
    ax_slice_slider, ax_slice_instruction = axes[1], axes[2]
    ax_intensity_slider, ax_intensity_instructions = axes[3], axes[4]

    # Setup the interactive vertical line in the slice slider
    ax_slice_slider.set_xlim(0, data.shape[2] - 1)
    ax_slice_slider.set_ylim(0, 1)
    slice_line = ax_slice_slider.axvline(slice_index, color='black', linewidth=4)  # Movable line
    ax_slice_slider.set_facecolor('white')
    ax_slice_slider.set_yticks([])
    ax_slice_slider.set_xticks([])

    # Add a label below the slider
    ax_slice_instruction.text(0.5, 0.5, f'Click and drag to change {slice_label.lower()}', ha='center', va='center', fontsize=10)
    ax_slice_instruction.set_axis_off()

    # Setup the interactive window in the intensity slider
    ax_intensity_slider.set_xlim(vmin, vmax)
    ax_intensity_slider.set_ylim(0, 1)
    intensity_line = ax_intensity_slider.axhline(y=0.5, color='red', linewidth=4)
    ax_intensity_slider.set_facecolor('white')
    ax_intensity_slider.set_yticks([])
    ax_intensity_slider.set_xticks([vmin, vmax])
    vmin_line = ax_intensity_slider.axvline(vmin, color='blue', linewidth=4)  # Movable line
    vmax_line = ax_intensity_slider.axvline(vmax, color='red', linewidth=4)

    # Add a label below the slider
    ax_intensity_instructions.text(0.5, 0.5, 'Click and drag to change intensity window', ha='center', va='center', fontsize=10)
    ax_intensity_instructions.set_axis_off()

    # Connect events
    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)

    # Initial drawing
    cur_fig = fig
    redraw_fig(fig, first_pass=True)  # Call redraw to handle initial display for single or dual images

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
