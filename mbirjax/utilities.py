import os, sys

# === Core scientific/plotting libraries ===
import h5py
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import gridspec
from matplotlib.widgets import RangeSlider, Slider, RadioButtons, CheckButtons

# === Project-specific imports ===
import mbirjax as mj
import webview
import urllib.request
import tarfile
import warnings
from urllib.parse import urlparse
import shutil


# === Persistent state for viewer interaction ===
class SliceViewerState:
    def __init__(self):
        self.circles = []              # Circular ROI overlays
        self.text_boxes = []           # Text annotations for stats
        self.cur_slices = []           # Current slice index per volume
        self.axes = []                 # Matplotlib axes per volume
        self.images = []               # Image handles for display
        self.is_drawing = False        # Flag to track drawing mode
        self.data = []                 # View-oriented data arrays
        self.original_data = []        # Original unmodified arrays
        self.labels = []               # Axis titles per dataset
        self.slice_axes = []           # Active slice axis per volume

viewer_state = SliceViewerState()

# === Clear circle and text overlays from the viewer ===
def remove_existing_graphics():
    for item in viewer_state.circles + viewer_state.text_boxes:
        if item is not None:
            item.remove()
    viewer_state.circles = [None] * len(viewer_state.axes)
    viewer_state.text_boxes = [None] * len(viewer_state.axes)

# === Main interactive slice viewer ===
def slice_viewer(*datasets, title='', vmin=None, vmax=None, slice_label=None, slice_axis=None,
                 cmap='gray', show_instructions=True):
    """
    Launch an interactive viewer for one or more 3D image volumes.

    This viewer allows users to scroll through slices of 3D volumes, adjust
    intensity ranges, draw circular regions of interest (ROIs) to view local
    statistics, and change the slicing axis interactively. When multiple volumes
    are displayed, slice axes can be coupled or decoupled interactively.

    Args:
        *datasets (ndarray): One or more 2D or 3D arrays representing image volumes.
            Each volume must be at least 2D, and will be reshaped if needed.
        title (str, optional): Title to display above the figure.
        vmin (float, optional): Minimum intensity value for the color scale.
        vmax (float, optional): Maximum intensity value for the color scale.
        slice_label (str or list of str, optional): Label(s) to display above each image volume.
            If a string is provided, it is broadcast to all datasets. Defaults to 'Slice'.
        slice_axis (int or list of int, optional): The axis along which to slice each volume.
            Can be a single int applied to all datasets, or a list specifying an axis per dataset.
            Defaults to 2 if not specified.
        cmap (str, optional): Colormap to use for displaying images. Default is 'gray'.
        show_instructions (bool, optional): Whether to show usage instructions in the figure.

    Example:
        >>> data1 = np.random.rand(100, 100, 50)
        >>> data2 = np.random.rand(100, 100, 60)
        >>> slice_viewer(data1, data2, slice_axis=[2, 2], title='Two volumes', slice_label='Axial')
    """
    state = viewer_state
    n_volumes = len(datasets)

    # Normalize input formats for slice_axis and slice_label
    if isinstance(slice_axis, int) or slice_axis is None:
        slice_axes = [2 if slice_axis is None else slice_axis] * n_volumes
    else:
        slice_axes = list(slice_axis)

    if isinstance(slice_label, str) or slice_label is None:
        slice_labels = [f"Slice {i+1}" if slice_label is None else slice_label for i in range(n_volumes)]
    else:
        slice_labels = list(slice_label)

    all_same_axis = all(ax == slice_axes[0] for ax in slice_axes)

    # Preprocess data: move specified slice axis to axis 2 for consistent viewing
    state.original_data = list(datasets)
    state.slice_axes = slice_axes.copy()
    state.data = []
    for i, data in enumerate(datasets):
        if data.ndim == 2:
            data = data[..., np.newaxis]
        elif data.ndim != 3:
            raise ValueError("Each input data must be a 2D or 3D array")
        data = np.moveaxis(data, slice_axes[i], 2)
        state.data.append(data)

    # Determine default vmin/vmax intensity range across all datasets
    all_data = np.concatenate([d.ravel() for d in state.data])
    if vmin is None:
        vmin = np.min(all_data)
    if vmax is None:
        vmax = np.max(all_data)
    if vmin > vmax:
        raise ValueError("vmin must be less than or equal to vmax")
    if vmin == vmax:
        eps = 1e-6
        scale = np.clip(eps * np.abs(vmax), a_min=eps, a_max=None)
        vmin -= scale
        vmax += scale

    # === Set up layout ===
    figwidth = 6 * n_volumes
    fig = plt.figure(figsize=(figwidth, 8))
    fig.suptitle(title)
    gs = gridspec.GridSpec(nrows=5, ncols=n_volumes, height_ratios=[10, 1, 1, 1, 1])

    # === Main image display ===
    state.axes = []
    state.images = []
    state.circles = [None] * n_volumes
    state.text_boxes = [None] * n_volumes
    state.cur_slices = [d.shape[2] // 2 for d in state.data]
    state.labels = slice_labels

    for i, d in enumerate(state.data):
        ax = fig.add_subplot(gs[0, i])
        img = ax.imshow(d[:, :, state.cur_slices[i]], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"{slice_labels[i]} {state.cur_slices[i]}\nShape: {datasets[i].shape}")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(img, cax=cax, orientation='vertical')
        state.axes.append(ax)
        state.images.append(img)

    # === Shared slice index slider ===
    ax_slice_slider = fig.add_subplot(gs[1, :])
    slice_slider = Slider(ax_slice_slider, label="Slice", valmin=0,
                          valmax=max(d.shape[2] for d in state.data) - 1,
                          valinit=state.cur_slices[0], valfmt='%0.0f')

    # === Intensity range slider ===
    ax_intensity_slider = fig.add_subplot(gs[2, :])
    log_range = np.log10(vmax - vmin)
    digits = max(-int(np.round(log_range)) + 2, 0)
    valfmt = '%0.' + str(digits) + 'f'
    intensity_slider = RangeSlider(ax=ax_intensity_slider, label="Intensity range",
                                   valmin=vmin, valmax=vmax, valinit=(vmin, vmax), valfmt=valfmt)

    # === Functions and controls for changing slice axis ===
    def update_axis(i, new_axis):
        if new_axis != state.slice_axes[i]:
            orig = state.original_data[i]
            state.slice_axes[i] = new_axis
            new_data = np.moveaxis(orig, new_axis, 2)
            state.data[i] = new_data
            state.cur_slices[i] = new_data.shape[2] // 2
            state.images[i].set_data(new_data[:, :, state.cur_slices[i]])
            state.axes[i].set_title(f"{state.labels[i]} {state.cur_slices[i]}\nShape: {orig.shape}")
            fig.canvas.draw_idle()

    axis_buttons = []

    def setup_axis_controls(decoupled):
        nonlocal axis_buttons
        for b in axis_buttons:
            b.ax.clear()
        axis_buttons.clear()
        if not decoupled:
            # Shared control
            ax_shared = fig.add_subplot(gs[3, :])
            buttons = RadioButtons(ax_shared, labels=["0", "1", "2"])
            for label in buttons.labels: label.set_fontsize(8)
            buttons.set_active(state.slice_axes[0])
            def update_all_axes(label):
                new_axis = int(label)
                for i in range(n_volumes):
                    update_axis(i, new_axis)
            buttons.on_clicked(update_all_axes)
            ax_shared.set_title("Slice axis")
            axis_buttons.append(buttons)
        else:
            # Individual controls
            for i in range(n_volumes):
                ax_radio = fig.add_subplot(gs[3, i])
                buttons = RadioButtons(ax_radio, labels=["0", "1", "2"])
                for label in buttons.labels: label.set_fontsize(8)
                buttons.set_active(state.slice_axes[i])
                buttons.on_clicked(lambda label, i=i: update_axis(i, int(label)))
                axis_buttons.append(buttons)

    # === Create decoupling toggle and initial setup ===
    if n_volumes == 1:
        setup_axis_controls(False)
    else:
        ax_checkbox = fig.add_subplot(gs[4, 0])
        cb = CheckButtons(ax_checkbox, labels=["Decouple slice axes"], actives=[not all_same_axis])
        def toggle_decoupled(label):
            for b in axis_buttons:
                b.ax.remove()
            axis_buttons.clear()
            fig.canvas.draw_idle()
            setup_axis_controls(cb.get_status()[0])
        cb.on_clicked(toggle_decoupled)
        setup_axis_controls(cb.get_status()[0])

    # === Intensity slider callback ===
    def update_intensity(val):
        for img in state.images:
            img.set_clim(val[0], val[1])
        fig.canvas.draw_idle()

    # === Slice slider callback ===
    def update_slice(val):
        new_slice = int(round(val))
        for i, d in enumerate(state.data):
            frac = new_slice / d.shape[2]
            slice_idx = int(round(frac * d.shape[2]))
            slice_idx = np.clip(slice_idx, 0, d.shape[2] - 1)
            state.cur_slices[i] = slice_idx
            state.images[i].set_data(d[:, :, slice_idx])
            state.axes[i].set_title(f"{state.labels[i]} {slice_idx}\nShape: {state.original_data[i].shape}")
        display_mean()
        fig.canvas.draw_idle()

    # === ROI statistics ===
    def get_mask(data, x, y, r):
        ny, nx = data.shape[:2]
        xv, yv = np.meshgrid(np.arange(nx), np.arange(ny))
        return (xv - x)**2 + (yv - y)**2 <= r**2

    def display_mean():
        if all(c is None for c in state.circles):
            return
        for i, circle in enumerate(state.circles):
            if circle is None:
                continue
            x, y = circle.center
            r = circle.get_radius()
            mask = get_mask(state.data[i][:, :, state.cur_slices[i]], x, y, r)
            values = state.data[i][:, :, state.cur_slices[i]][mask]
            if values.size:
                mean, std = np.mean(values), np.std(values)
                state.text_boxes[i] = state.axes[i].text(0.05, 0.95,
                    f"Mean: {mean:.3g}\nStd Dev: {std:.3g}",
                    transform=state.axes[i].transAxes,
                    fontsize=12, va='top', bbox=dict(facecolor='white', alpha=1.0))
        fig.canvas.draw_idle()

    # === Mouse interaction for ROI drawing ===
    def on_press(event):
        if event.inaxes not in state.axes:
            return
        if hasattr(fig.canvas, 'toolbar') and getattr(fig.canvas.toolbar, 'mode', ''):
            return
        state.is_drawing = True
        remove_existing_graphics()
        for i, ax in enumerate(state.axes):
            circ = plt.Circle((event.xdata, event.ydata), 0, color='red', lw=2, fill=False, alpha=1.0)
            state.circles[i] = circ
            ax.add_patch(circ)
        fig.canvas.draw_idle()

    def on_motion(event):
        if not state.is_drawing or event.inaxes not in state.axes:
            return
        if hasattr(fig.canvas, 'toolbar') and getattr(fig.canvas.toolbar, 'mode', ''):
            return
        if event.xdata is not None and event.ydata is not None:
            dx = event.xdata - state.circles[0].center[0]
            dy = event.ydata - state.circles[0].center[1]
            r = np.sqrt(dx**2 + dy**2)
            for circ in state.circles:
                if circ: circ.set_radius(r)
        fig.canvas.draw_idle()

    def on_release(event):
        if event.inaxes not in state.axes:
            return
        if hasattr(fig.canvas, 'toolbar') and getattr(fig.canvas.toolbar, 'mode', ''):
            return
        state.is_drawing = False
        display_mean()

    def on_key(event):
        if event.key == 'escape':
            remove_existing_graphics()
            fig.canvas.draw_idle()

    # === Register events ===
    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('key_press_event', on_key)

    slice_slider.on_changed(update_slice)
    intensity_slider.on_changed(update_intensity)

    if show_instructions:
        fig.text(0.01, 0.85, 'Close plot\nto continue', rotation='vertical')
        fig.text(0.01, 0.3, 'Click and drag to select; esc to deselect.', rotation='vertical')

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
                              fig_title='granularity'):
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
    fig, axes = plt.subplots(nrows=1, ncols=num_plots, figsize=(6 * num_plots, 7), sharey='row')
    fig.suptitle(fig_title)

    if num_plots == 1:
        axes = [axes]  # Make it iterable for a single subplot scenario

    for ax, granularity_sequence, fm_loss, prior_loss, label in zip(axes, granularity_sequences, fm_losses, prior_losses, labels):
        index = list(1 + np.arange(len(granularity_sequence)))

        # Plot granularity sequence on the first y-axis
        ax1 = ax
        ax1.stem(index, granularity_sequence, label='Number of subsets', basefmt=" ", linefmt='b', markerfmt='bo')
        ax1.set_ylabel('Number of subsets', color='b')
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
        ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    plt.tight_layout()
    plt.show()

    figure_folder_name = mj.make_figure_folder()
    os.makedirs(figure_folder_name, exist_ok=True)
    fig.savefig(os.path.join(figure_folder_name, fig_title + '_plots.png'), bbox_inches='tight')

def make_figure_folder(fig_folder_name=None):
    if fig_folder_name is None:
        fig_folder_name = 'figs'
    os.makedirs(fig_folder_name, exist_ok=True)
    return fig_folder_name


def export_recon_to_hdf5(filename, recon, recon_description="", delta_pixel_image=1.0, alu_description="", data_dict=None):
    """
    Writes a reconstructed image and its metadata to an HDF5 file. Allows for parameterized inputs or a single data dictionary.

    Args:
        filename (str): Path to save the HDF5 file.
        recon (ndarray): 3D reconstructed image data.
        recon_description (str, optional): Description of the CT reconstruction. Ignored if data_dict is provided.
        delta_pixel_image (float, optional): Pixel spacing in arbitrary length units. Ignored if data_dict is provided.
        alu_description (str, optional): Description of the arbitrary length units. Ignored if data_dict is provided.
        data_dict (dict, optional): Dictionary containing all data fields. Overrides individual parameters if provided. Expected keys:
            - 'recon' (ndarray): 3D reconstructed image data.
            - 'recon_description' (str): Description of the CT reconstruction.
            - 'delta_pixel_image' (float): Pixel spacing in arbitrary length units.
            - 'alu_description' (str): Description of the arbitrary length units.

    Example usage:
        >>> # Using individual parameters
        >>> export_recon_to_hdf5('recon.h5', recon=my_recon_array, recon_description="CT Scan 001")

        >>> # Using a data dictionary
        >>> data = {
        ...     'recon': my_recon_array,
        ...     'recon_description': "CT Scan 001",
        ...     'delta_pixel_image': 0.5,
        ...     'alu_description': "1 ALU = 0.5 mm"
        ... }
        >>> export_recon_to_hdf5('recon.h5', data_dict=data)
    """
    if data_dict:
        recon = data_dict.get('recon', recon)
        recon_description = data_dict.get('recon_description', recon_description)
        delta_pixel_image = data_dict.get('delta_pixel_image', delta_pixel_image)
        alu_description = data_dict.get('alu_description', alu_description)

    with h5py.File(filename, "w") as f:
        # Write data
        f.create_dataset("recon", data=recon)

        # Write attributes
        f.attrs["recon_description"] = recon_description
        f.attrs["delta_pixel_image"] = delta_pixel_image
        f.attrs["alu_description"] = alu_description

    print("Export complete: {}".format(filename))


def import_recon_from_hdf5(filename):
    """
    Reads a reconstructed image and its metadata from an HDF5 file created by export_recon_to_hdf5().

    Args:
        filename (string): Fully specified path to the HDF5 file.

    Returns:
        dict: A dictionary containing the reconstruction data and its metadata with the following keys:
            - 'recon' (ndarray): 3D reconstructed image.
            - 'recon_description' (string): Description of the CT reconstruction.
            - 'delta_pixel_image' (float): Image pixel spacing in arbitrary length units.
            - 'alu_description' (string): Description of the arbitrary length units for pixel spacing.

    Example usage:
        >>> # Assuming 'recon.h5' was created using export_recon_to_hdf5()
        >>> data = import_recon_from_hdf5('recon.h5')
        >>> print(data['recon'].shape)
        >>> print(data['recon_description'])
        >>> print(data['delta_pixel_image'])
        >>> print(data['alu_description'])
    """
    with h5py.File(filename, "r") as f:
        # Extract the reconstruction dataset
        recon = np.array(f["recon"])

        # Extract metadata attributes
        recon_description = f.attrs.get("recon_description", "")
        alu_description = f.attrs.get("alu_description", "")
        delta_pixel_image = f.attrs.get("delta_pixel_image", 1.0)

    # Return as a dictionary to allow for future extension
    return {
        'recon': recon,
        'recon_description': recon_description,
        'delta_pixel_image': delta_pixel_image,
        'alu_description': alu_description
    }


def download_and_extract_tar(download_url, save_dir):
    """
    Download or copy a .tar file from a URL or local file path, extract it to the specified directory, and return the path to the extracted top-level directory.

    If the tarball already exists in the save directory, the user will be prompted to decide whether to overwrite it.

    Parameters
    ----------
    download_url : str
        URL or local file path to the tarball. If a URL, it must be public.
    save_dir : str
        Path to the directory where the tarball will be saved/copied and extracted.

    Returns
    -------
    extracted_file_name : str
        The path to the extracted top-level directory.

    Example
    -------
    >>> extracted_dir = download_and_extract_tar("https://example.com/data.tar.gz", "./data")
    >>> print(f"Extracted data is in: {extracted_dir}")

    >>> extracted_dir = download_and_extract_tar("/path/to/local/data.tar.gz", "./data")
    >>> print(f"Extracted data is in: {extracted_dir}")
    """
    is_download = True
    parsed = urlparse(download_url)
    is_url = parsed.scheme in ('http', 'https')

    tarball_name = os.path.basename(parsed.path if is_url else download_url)
    tarball_path = os.path.join(save_dir, tarball_name)

    if os.path.exists(tarball_path):
        is_download = query_yes_no(f"\nData named {tarball_path} already exists.\nDo you still want to download/copy and overwrite the file?")

    if is_download:
        os.makedirs(os.path.dirname(tarball_path), exist_ok=True)
        if is_url:
            print("Downloading file ...")
            try:
                urllib.request.urlretrieve(download_url, tarball_path)
            except urllib.error.HTTPError as e:
                if e.code == 401:
                    raise RuntimeError(f'HTTP {e.code}: authentication failed!')
                elif e.code == 403:
                    raise RuntimeError(f'HTTP {e.code}: URL forbidden!')
                elif e.code == 404:
                    raise RuntimeError(f'HTTP {e.code}: URL not found!')
                else:
                    raise RuntimeError(f'HTTP {e.code}: {e.reason}')
            except urllib.error.URLError as e:
                raise RuntimeError('URLError raised! Check internet connection.')
            print(f"Download successful! Tarball file saved to {tarball_path}")
        else:
            print(f"Copying local file from {download_url} to {tarball_path} ...")
            if not os.path.isfile(download_url):
                raise RuntimeError(f"Provided file path does not exist: {download_url}")
            shutil.copy2(download_url, tarball_path)
            print(f"Copy successful! Tarball file saved to {tarball_path}")

        print(f"Extracting tarball file to {save_dir} ...")
        with tarfile.open(tarball_path, 'r') as tar_file:
            tar_file.extractall(save_dir)
        print(f"Extraction successful!")

    top_level_dir = get_top_level_tar_dir(tarball_path)
    extracted_file_name = os.path.join(save_dir, top_level_dir)

    return extracted_file_name

def get_top_level_tar_dir(tar_path, max_entries=10):
    """
    Determine the top-level directory inside a tarball file by sampling up to max_entries members.

    Parameters
    ----------
    tar_path : str
        Path to the tarball file.
    max_entries : int
        Maximum number of entries to sample.

    Returns
    -------
    dir_name : str
        The name of the top-level directory.
    """
    top_levels = set()

    with tarfile.open(tar_path, 'r') as tar:
        for i, member in enumerate(tar):
            if not member.name.strip():
                continue
            top_dir = member.name.split('/')[0]
            top_levels.add(top_dir)

            if len(top_levels) > 1 or i + 1 >= max_entries:
                break
    if len(top_levels) == 1:
        dir_name = top_levels.pop()
    else:
        raise ValueError("No top level directory found in {}".format(tar_path))
    return dir_name

def query_yes_no(question, default="n"):
    """
    Ask a yes/no question via input() and return the answer.

    Parameters
    ----------
    question : str
        The question presented to the user.
    default : str
        The default answer if the user just presses Enter ("y" or "n").

    Returns
    -------
    bool
        True for "yes" or Enter, False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    prompt = f" [y/n, default={default}] "
    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' (or 'y' or 'n').\n")
    return
