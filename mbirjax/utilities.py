import os, sys
from enum import Enum
from typing import Union

import jax
# === Core scientific/plotting libraries ===
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from jax import lax
from PIL import ImageFont, Image, ImageDraw
from jax import numpy as jnp

# === Project-specific imports ===
import mbirjax as mj
import urllib.request
import urllib.error
import tarfile
from urllib.parse import urlparse
import shutil
import h5py
import re
import warnings
import subprocess


def load_data_hdf5(file_path):
    """
    Load a numpy array from an HDF5 file.

    This function loads an array stored in an HDF5 file using :func:`save_data_hdf5`.
    It also loads any associated attributes and returns them as a dict.

    Args:
        file_path (str): Path to the HDF5 file containing the reconstructed volume.

    Returns:
        tuple: (array, data_dict)
            - array (ndarray): The array saved by :func:`save_data_hdf5`
            - data_dict (dict): A dict with the attributes for the data array.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If more than one dataset is not found in the file.

    Example:
        >>> import mbirjax as mj
        >>> recon, recon_dict = mj.load_data_hdf5("output/recon_volume.h5")
        >>> recon.shape
        (64, 256, 256)
    """
    with h5py.File(file_path, "r") as f:
        array_names = [key for key in f.keys()] # If this h5 file was created with save_data_hdf5, then there will be only one key
        if len(array_names) > 1:
            raise ValueError('More than one array found in {}. Unable to load.'.format(file_path))
        data_name = array_names[0]
        array = f[data_name][()]
        data_dict = dict()
        for name in f[data_name].attrs.keys():
            data_dict[name] = f[data_name].attrs[name]

        return array, data_dict


def save_volume_as_gif(volume, filename, vmin=0, vmax=1):
    """
    Save a 3D volume as a GIF, iterating over axis 0 (row-wise).

    Args:
        volume (np.ndarray): 3D array to save as a movie.
        filename (str): Output path for the GIF file.
        vmin (float): Min pixel value for display normalization.
        vmax (float): Max pixel value for display normalization.
    """
    try:
        import imageio.v2 as imageio
    except ImportError:
        print("The 'imageio' package is not installed. Please install it using:\n    pip install imageio")
        return

    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    images = []
    for i in range(volume.shape[0]):
        fig, ax = plt.subplots()
        canvas = FigureCanvas(fig)
        ax.imshow(volume[i, :, :].T, cmap='gray', vmin=vmin, vmax=vmax)
        ax.axis('off')
        # Convert canvas to image using RGBA buffer, then drop alpha channel
        canvas.draw()
        buf = canvas.get_renderer().buffer_rgba()
        image = np.frombuffer(buf, dtype=np.uint8).reshape(canvas.get_width_height()[::-1] + (4,))
        image = image[..., :3]  # Drop alpha channel
        images.append(image)
        plt.close(fig)

    imageio.mimsave(filename, images, fps=5)  # 5 frames per second


def save_4d_volume_as_gif(volume, filename, slice_axis=1, slice_index=None, vmin=0, vmax=1,
                          titles=None, fps=5):
    """
    Save one spatial slice of a 4D volume as a GIF stepping through time.

    A 4D volume has shape (num_times, nx, ny, nz), so time is always axis 0 and the movie always
    plays over it.  What varies is which spatial plane is shown: ``slice_axis`` picks the spatial
    axis to hold fixed and ``slice_index`` picks the position along it.  The default shows the
    middle x slice, that is the YZ plane through the center of the volume.

    This is deliberately separate from :func:`save_volume_as_gif`, which iterates over axis 0 of a
    3D volume -- a spatial axis, shown transposed.  Here axis 0 means time and is never a display
    axis, and the image is shown in its natural orientation, so the two have different meanings
    for the same array and should not be conflated.

    Args:
        volume (np.ndarray): 4D array of shape (num_times, nx, ny, nz).
        filename (str): Output path for the GIF file.
        slice_axis (int, optional): Spatial axis to slice: 1, 2 or 3 for x, y or z.  Defaults to
            1, giving the YZ plane.
        slice_index (int, optional): Position along ``slice_axis``.  Defaults to None, the middle
            of that axis.
        vmin (float): Min pixel value for display normalization.
        vmax (float): Max pixel value for display normalization.
        titles (str or sequence of str, optional): Title drawn above each frame.  Defaults to
            None, which names the fixed slice and numbers the time frame.  A single string is
            formatted with the frame index as ``titles.format(t)``; a sequence supplies one title
            per time frame; ``''`` draws no title.
        fps (float, optional): Frames per second in the saved GIF.  Defaults to 5.

    Raises:
        ValueError: If ``volume`` is not 4D, ``slice_axis`` is not 1, 2 or 3, ``slice_index`` is
            out of range, or a title sequence does not have one entry per time frame.

    Example:
        >>> # The middle x slice of a 4D reconstruction, playing over time.
        >>> mj.save_4d_volume_as_gif(recon_4d, 'recon_4d.gif', vmax=0.06)
        >>> # A specific z slice instead, one frame every 200 ms.
        >>> mj.save_4d_volume_as_gif(recon_4d, 'recon_4d_z.gif', slice_axis=3, slice_index=40, fps=5)
    """
    volume = np.asarray(volume)
    if volume.ndim != 4:
        raise ValueError('volume must be 4D with shape (num_times, nx, ny, nz); '
                         'got shape {}.'.format(volume.shape))
    if slice_axis not in (1, 2, 3):
        raise ValueError('slice_axis must be 1, 2 or 3 (x, y or z); axis 0 is time and is always '
                         'the movie axis.  Got {}.'.format(slice_axis))

    num_times = volume.shape[0]
    axis_length = volume.shape[slice_axis]
    if slice_index is None:
        slice_index = axis_length // 2
    if not -axis_length <= slice_index < axis_length:
        raise ValueError('slice_index {} is out of range for axis {} of length {}.'.format(
            slice_index, slice_axis, axis_length))

    if titles is None:
        axis_name = {1: 'x', 2: 'y', 3: 'z'}[slice_axis]
        frame_titles = ['{} slice = {}, t = {}'.format(axis_name, slice_index % axis_length, t)
                        for t in range(num_times)]
    elif isinstance(titles, str):
        frame_titles = [titles.format(t) for t in range(num_times)]
    else:
        frame_titles = list(titles)
        if len(frame_titles) != num_times:
            raise ValueError('titles has {} entries but the volume has {} time frames.'.format(
                len(frame_titles), num_times))

    try:
        import imageio.v2 as imageio
    except ImportError:
        print("The 'imageio' package is not installed. Please install it using:\n    pip install imageio")
        return

    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    # One small 2D image per time frame; the full volume is never copied.
    planes = np.take(volume, slice_index, axis=slice_axis)
    images = []
    for t in range(num_times):
        fig, ax = plt.subplots()
        canvas = FigureCanvas(fig)
        ax.imshow(planes[t], cmap='gray', vmin=vmin, vmax=vmax)
        ax.axis('off')
        if frame_titles[t]:
            ax.set_title(frame_titles[t])
        # Convert canvas to image using RGBA buffer, then drop alpha channel
        canvas.draw()
        buf = canvas.get_renderer().buffer_rgba()
        image = np.frombuffer(buf, dtype=np.uint8).reshape(canvas.get_width_height()[::-1] + (4,))
        image = image[..., :3]  # Drop alpha channel
        images.append(image)
        plt.close(fig)

    imageio.mimsave(filename, images, fps=fps)


def _write_hdf5_streaming(file_path, array_name, out_shape, dtype, produce_slab, attributes_dict=None):
    """Create an HDF5 dataset of out_shape/dtype and fill it slab-by-slab along axis 0.

    produce_slab(i0, i1) returns the contiguous slab written to dset[i0:i1].  Only one slab is
    held at a time, so a large or strided source is never fully copied.
    """
    mj.makedirs(file_path)
    with h5py.File(file_path, 'w') as f:
        dset = f.create_dataset(array_name, shape=out_shape, dtype=dtype)
        if len(out_shape) == 0:
            dset[...] = produce_slab(0, 0)
        else:
            row_bytes = np.dtype(dtype).itemsize * int(np.prod(out_shape[1:], dtype=np.int64))
            slab = max(1, (1 << 30) // max(row_bytes, 1))   # ~1 GiB per write
            for i in range(0, out_shape[0], slab):
                dset[i:i + slab] = produce_slab(i, min(i + slab, out_shape[0]))
        if isinstance(attributes_dict, dict):
            attributes_dict = mj.TomographyModel.convert_subdicts_to_strings(attributes_dict)
            for key, value in attributes_dict.items():
                dset.attrs[key] = value


def save_data_hdf5(file_path, array, array_name='array', attributes_dict=None):
    """
    Save a NumPy or JAX array to an HDF5 file, optionally including metadata as attributes.
    The resulting structure has a single dataset with one array and associated text attributes.
    These can be retrieved using :func:`load_data_hdf5`.

    Args:
        file_path (str): Full path to the output HDF5 file. Directories will be created if they do not exist.
        array (ndarray or jax.Array): The volume data to save.
        array_name (str): Name of the dataset within the HDF5 file. Defaults to 'array'.
        attributes_dict (dict, optional): Dictionary of attributes to store as metadata in the dataset.
            Keys must be strings, and values should be serializable as HDF5 attributes.

    Returns:
        None

    Example:
        >>> import numpy as np
        >>> volume = np.random.rand(64, 64, 64)
        >>> attrs = {'voxel_size': '1.0mm', 'modality': 'CT'}
        >>> save_data_hdf5('output/recon.h5', volume, array_name='recon', attributes_dict=attrs)
        Nothing

    Example:
        >>> recon, recon_dict = ct_model.recon(sinogram)
        >>> recon_info = {'ALU units': '0.3mm', 'sinogram name': 'test part 038'}
        >>> file_path = './output/test_part_038.yaml'
        >>> mj.save_data_hdf5(file_path, recon, recon_info)
    """
    # Stream the array to disk slab-by-slab (no full contiguous copy, even for a strided view).
    def produce_slab(i0, i1):
        return np.asarray(array) if array.ndim == 0 else np.ascontiguousarray(array[i0:i1])

    _write_hdf5_streaming(file_path, array_name, array.shape, array.dtype, produce_slab, attributes_dict)


def display_translation_vectors(translation_vectors, recon_shape):
    """Display the x and z components of translation vectors using a scatter plot,
    and overlay a box representing the reconstruction volume in (column, slice) space.

    Args:
        translation_vectors (np.ndarray): Array of shape (N, 3) containing [dx, dy, dz] vectors.
        recon_shape (tuple[int, int, int]): Shape of the reconstruction volume (rows, columns, slices).
    """
    dx = translation_vectors[:, 0]
    dz = translation_vectors[:, 2]

    plt.figure(figsize=(6, 6))
    plt.scatter(dx, dz, c='blue', marker='o', label='Translations')

    # Get col and slice dimensions (horizontal and vertical axes in view)
    num_cols = recon_shape[1]
    num_slices = recon_shape[2]

    # Compute box boundaries centered around origin
    half_width = num_cols / 2
    half_height = num_slices / 2

    box_x = [-half_width, half_width, half_width, -half_width, -half_width]
    box_z = [-half_height, -half_height, half_height, half_height, -half_height]

    plt.plot(box_x, box_z, 'r--', linewidth=2, label='Reconstruction Region')

    plt.title("Translation Grid Points with Recon Outline")
    plt.xlabel("Horizontal Translation in ALU")
    plt.ylabel("Vertical Translation in ALU")
    plt.axis('equal')
    plt.legend(loc='upper right')
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


def plot_granularity_and_loss(granularity_sequences, fm_losses, prior_losses, labels, granularity_ylim=None,
                              loss_ylim=None,
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

    for ax, granularity_sequence, fm_loss, prior_loss, label in zip(axes, granularity_sequences, fm_losses,
                                                                    prior_losses, labels):
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


def makedirs(filepath):
    save_dir = os.path.dirname(filepath)
    if save_dir:
        try:
            os.makedirs(save_dir, exist_ok=True)
        except Exception as e:
            raise Exception(f"Could not create save directory '{save_dir}': {e}")


def merge_log_files(merged_path, labeled_paths):
    """Merge temp log files into one file, each under a section header, and remove the temps.

    Missing temps are skipped; if none exist, no file is written.

    Args:
        merged_path (str): Path of the merged output file.
        labeled_paths (iterable): (label, path) pairs in the order they should appear.
    """
    labeled_paths = [(label, path) for label, path in labeled_paths
                     if path is not None and os.path.exists(path)]
    if not labeled_paths:
        return
    with open(merged_path, 'w') as merged:
        for label, path in labeled_paths:
            merged.write('======== {} ========\n'.format(label))
            with open(path, 'r') as f:
                merged.write(f.read())
            os.remove(path)


def download_and_extract(download_url, save_dir):
    """
    Download or copy a file from a URL or local file path. If the file is a tarball (.tar, .tar.gz, etc.), extract it
    into the specified directory. Supports Google Drive links, standard HTTP/HTTPS URLs, and local paths.

    If the file already exists in the save directory, it will not be re-downloaded or copied.

    Args:
        download_url (str): URL or local file path to the file. Supported formats include:
            - Google Drive shared links
            - HTTP/HTTPS URLs
            - Local file paths
        save_dir (str): Directory where the file will be saved and extracted (if applicable).

    Returns:
        str:
            - For tar files: Path to the extracted top-level directory.
            - For other files: Path to the downloaded or copied file.

    Raises:
        RuntimeError: If the file cannot be downloaded, copied, or extracted.
        ValueError: If the Google Drive URL is invalid or tar file has no top-level directory.

    Examples:
        >>> extracted_dir = download_and_extract("https://example.com/data.tar.gz", "./data")
        >>> file_path = download_and_extract("https://drive.google.com/file/d/1ABC123/view", "./data")
        >>> result = download_and_extract("/path/to/local/data.tar.gz", "./data")
    """

    def is_google_drive_url(url):
        """Check if URL is a Google Drive link"""
        return "drive.google.com" in url

    def is_tar_file(filename):
        """Check if file is a tar archive based on extension"""
        tar_extensions = ['.tar', '.tar.gz', '.tgz', '.tar.bz2', '.tbz2', '.tar.xz', '.txz']
        return any(filename.lower().endswith(ext) for ext in tar_extensions)

    def extract_google_drive_id(url):
        """Extract Google Drive file ID from URL"""
        pattern = r"(?:https?:\/\/)?(?:www\.)?drive\.google\.com\/(?:file\/d\/|open\?id=)([a-zA-Z0-9_-]+)"
        match = re.search(pattern, url)
        if match:
            return match.group(1)
        else:
            raise ValueError("Invalid Google Drive URL format")

    parsed = urlparse(download_url)
    is_url = parsed.scheme in ('http', 'https')
    is_google_drive = is_url and is_google_drive_url(download_url)

    if is_google_drive:
        file_id = extract_google_drive_id(download_url)
        marker_file = os.path.join(save_dir, f".gdrive_{file_id}")

        if os.path.exists(marker_file):
            with open(marker_file, 'r') as f:
                actual_filename = f.read().strip()
            file_path = os.path.join(save_dir, actual_filename)
            filename = actual_filename

            if os.path.exists(file_path):
                is_download = False
            else:
                is_download = True

        else:
            filename = f"gdrive_{file_id}"
            is_download = True
    else:
        filename = os.path.basename(parsed.path if is_url else download_url)
        file_path = os.path.join(save_dir, filename)
        if os.path.exists(file_path):
            is_download = False
        else:
            is_download = True

    if is_download:
        os.makedirs(save_dir, exist_ok=True)

        if is_url:
            if is_google_drive:
                print("Downloading file from Google Drive...")
                import gdown
                try:
                    gdrive_url = f"https://drive.google.com/uc?id={file_id}"

                    downloaded_path = gdown.download(gdrive_url, output=None, quiet=False)
                    if downloaded_path and os.path.isfile(downloaded_path):
                        actual_filename = os.path.basename(downloaded_path)
                        target_path = os.path.join(save_dir, actual_filename)
                        shutil.move(downloaded_path, target_path)
                        file_path = target_path
                        filename = actual_filename

                        with open(marker_file, 'w') as f:
                            f.write(actual_filename)
                    else:
                        raise RuntimeError("Google Drive download failed or returned invalid path")

                    print(f"Download successful! File saved to {file_path}")
                except Exception as e:
                    raise RuntimeError(f"Google Drive download failed: {str(e)}")
            else:
                print("Downloading file...")
                try:
                    urllib.request.urlretrieve(download_url, file_path)
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
                    res = subprocess.run(
                        ["curl", "-L", "--fail", "-o", file_path, download_url],
                        capture_output=True, text=True
                    )
                    if res.returncode != 0:
                        raise RuntimeError(f"Download failed with curl: {res.stderr.strip() or res.stdout.strip()}")
                print(f"Download successful! File saved to {file_path}")
        else:
            print(f"Copying local file from {download_url} to {file_path}...")
            if not os.path.isfile(download_url):
                raise RuntimeError(f"Provided file path does not exist: {download_url}")
            shutil.copy2(download_url, file_path)
            print(f"Copy successful! File saved to {file_path}")

        if is_tar_file(filename):
            print(f"Extracting tarball file to {save_dir}...")
            try:
                with tarfile.open(file_path, 'r') as tar_file:
                    tar_file.extractall(save_dir)
                print(f"Extraction successful!")

                top_level_dir = get_top_level_tar_dir(file_path)
                extracted_path = os.path.join(save_dir, top_level_dir)
                return extracted_path
            except Exception as e:
                raise RuntimeError(f"Failed to extract tar file: {str(e)}")
        else:
            return file_path

    if is_google_drive and not is_download:
        try:
            with open(marker_file, 'r') as f:
                actual_filename = f.read().strip()
            file_path = os.path.join(save_dir, actual_filename)
            filename = actual_filename
        except:
            file_path = os.path.join(save_dir, filename)

    if is_tar_file(filename):
        top_level_dir = get_top_level_tar_dir(file_path)
        file_path = os.path.join(save_dir, top_level_dir)

    return file_path


# Deprecated alias for backward compatibility
def download_and_extract_tar(download_url, save_dir):
    """
    Deprecated alias for download_and_extract().

    This function exists for backward compatibility and will be removed in a future release.
    """
    warnings.warn("'download_and_extract_tar' is deprecated and will be removed in a future release. Please use 'download_and_extract' instead.")
    return download_and_extract(download_url, save_dir)


def get_top_level_tar_dir(tar_path, max_entries=1):
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



def export_recon_hdf5(file_path, recon, recon_dict=None, remove_flash=False, radial_margin=10, top_margin=10, bottom_margin=10):
    """
    Export a 3D reconstruction volume to an HDF5 file with optional post-processing.

    This function works with either numpy or jax arrays or sharded jax arrays.
    The function also transposes the reconstruction to right-hand coordinates (slice, col, row),
    and writes the reconstruction and optional metadata to an HDF5 file.

    Args:
        file_path (str): Full path to the output HDF5 file. Parent directories will be created if they do not exist.
        recon (Union[np.ndarray, jax.Array]): 3D volume in (row, col, slice) order. Will be converted to NumPy before writing.
        recon_dict (dict, optional): Dictionary of attributes to store as metadata in the dataset.
        remove_flash (bool, optional): Whether to apply a cylindrical mask to remove peripheral and top/bottom slices. Defaults to False.
        radial_margin (int, optional): Margin in pixels to subtract from the cylinder radius. Defaults to 10.
        top_margin (int, optional): Number of top slices to set to zero along the Z-axis. Defaults to 10.
        bottom_margin (int, optional): Number of bottom slices to set to zero along the Z-axis. Defaults to 10.

    Example:
        >>> from mbirjax.utilities import export_recon_hdf5
        >>> import jax.numpy as jnp
        >>> recon = jnp.ones((128, 128, 64))  # (row, col, slice) order
        >>> export_recon_hdf5("output/recon_volume.h5", recon, recon_dict={"scan_id": "sample1"})
    """

    # Move the input to the host (NumPy) first so numpy/jax/sharded all collapse to one host case.
    recon = jax.device_get(recon)

    if not remove_flash:
        # Transposed view; save_data_hdf5 streams it slab-by-slab, so no full copy is made.
        save_data_hdf5(file_path, np.transpose(recon, (2, 1, 0)), 'recon', recon_dict)
        return

    # remove_flash: mask + transpose + write one slab at a time, so no full masked volume is built.
    # Slabbing along the slice axis keeps full (rows, cols), so apply_cylindrical_mask gives the
    # identical circular mask per slab; we just map the global top/bottom margins to each slab.
    num_rows, num_cols, num_slices = recon.shape

    def produce_slab(s0, s1):
        ds = s1 - s0
        local_top = min(max(top_margin - s0, 0), ds)                       # global top slices in this slab
        local_bottom = min(max(s1 - (num_slices - bottom_margin), 0), ds)  # global bottom slices in this slab
        block = mj.preprocess.apply_cylindrical_mask(recon[:, :, s0:s1], radial_margin, local_top, local_bottom)
        return np.ascontiguousarray(np.transpose(block, (2, 1, 0)))        # (ds, C, R)

    _write_hdf5_streaming(file_path, 'recon', (num_slices, num_cols, num_rows), recon.dtype,
                          produce_slab, recon_dict)


def import_recon_hdf5(file_path):
    """
    Import a 3D reconstruction volume from an HDF5 file.

    This function loads a reconstruction volume and associated metadata from an HDF5 file,
    and reorders the volume axes from the file's (slice, col, row) layout to (row, col, slice)
    to match MBIRJAX conventions, so a volume written by export_recon_hdf5 is recovered unchanged.

    Args:
        file_path (str): Path to the HDF5 file containing the reconstruction volume.

    Returns:
        Tuple[np.ndarray, dict]: A tuple containing:
            - recon (np.ndarray): The reconstructed 3D volume in (row, col, slice) order.
            - recon_dict (dict): Dictionary containing metadata associated with the reconstruction.

    Example:
        >>> from mbirjax.utilities import import_recon_hdf5
        >>> recon, recon_dict = import_recon_hdf5("output/recon_volume.h5")
        >>> print(recon.shape)
        (128, 128, 64)
    """
    recon, recon_dict = load_data_hdf5(file_path=file_path)

    recon = np.transpose(recon, axes=(2, 1, 0))

    return recon, recon_dict


def generate_3d_shepp_logan_reference(phantom_shape):
    """
    Generate a 3D Shepp Logan phantom based on below reference.

    Kak AC, Slaney M. Principles of computerized tomographic imaging. Page.102. IEEE Press, New York, 1988. https://engineering.purdue.edu/~malcolm/pct/CTI_Ch03.pdf

    Args:
        phantom_shape (tuple or list of ints): num_rows, num_cols, num_slices

    Return:
        out_image: 3D array, num_slices*num_rows*num_cols

    Note:
        This function produces 6 intermediate arrays that each have shape phantom_shape, so if phantom_shape is
        large, then this will use a lot of peak memory.
    """

    # The function describing the phantom is defined as the sum of 10 ellipsoids inside a 2×2×2 cube:
    sl3d_paras = [
        {'x0': 0.0, 'y0': 0.0, 'z0': 0.0, 'a': 0.69, 'b': 0.92, 'c': 0.9, 'gamma': 0, 'gray_level': 2.0},
        {'x0': 0.0, 'y0': 0.0, 'z0': 0.0, 'a': 0.6624, 'b': 0.874, 'c': 0.88, 'gamma': 0, 'gray_level': -0.98},
        {'x0': -0.22, 'y0': 0.0, 'z0': -0.25, 'a': 0.41, 'b': 0.16, 'c': 0.21, 'gamma': 108, 'gray_level': -0.02},
        {'x0': 0.22, 'y0': 0.0, 'z0': -0.25, 'a': 0.31, 'b': 0.11, 'c': 0.22, 'gamma': 72, 'gray_level': -0.02},
        {'x0': 0.0, 'y0': 0.35, 'z0': -0.25, 'a': 0.21, 'b': 0.25, 'c': 0.5, 'gamma': 0, 'gray_level': 0.02},
        {'x0': 0.0, 'y0': 0.1, 'z0': -0.25, 'a': 0.046, 'b': 0.046, 'c': 0.046, 'gamma': 0, 'gray_level': 0.02},
        {'x0': -0.08, 'y0': -0.65, 'z0': -0.25, 'a': 0.046, 'b': 0.023, 'c': 0.02, 'gamma': 0, 'gray_level': 0.01},
        {'x0': 0.06, 'y0': -0.65, 'z0': -0.25, 'a': 0.046, 'b': 0.023, 'c': 0.02, 'gamma': 90, 'gray_level': 0.01},
        {'x0': 0.06, 'y0': -0.105, 'z0': 0.625, 'a': 0.056, 'b': 0.04, 'c': 0.1, 'gamma': 90, 'gray_level': 0.02},
        {'x0': 0.0, 'y0': 0.1, 'z0': 0.625, 'a': 0.056, 'b': 0.056, 'c': 0.1, 'gamma': 0, 'gray_level': -0.02}
    ]

    num_rows, num_cols, num_slices = phantom_shape
    axis_x = np.linspace(-1.0, 1.0, num_cols)
    axis_y = np.linspace(1.0, -1.0, num_rows)
    axis_z = np.linspace(-1.0, 1.0, num_slices)

    x_grid, y_grid, z_grid = np.meshgrid(axis_x, axis_y, axis_z)
    image = x_grid * 0.0

    for el_paras in sl3d_paras:
        image += _gen_ellipsoid(x_grid=x_grid, y_grid=y_grid, z_grid=z_grid, x0=el_paras['x0'], y0=el_paras['y0'],
                               z0=el_paras['z0'],
                               a=el_paras['a'], b=el_paras['b'], c=el_paras['c'],
                               gamma=el_paras['gamma'] / 180.0 * np.pi,
                               gray_level=el_paras['gray_level'])

    return image.transpose((1, 0, 2))


def generate_3d_shepp_logan_low_dynamic_range(phantom_shape, devices=None, max_block_gb=4.0,
                                              target_max_attenuation=None):
    """
    Generates a 3D Shepp-Logan phantom with specified dimensions.

    The phantom is a reference object, so it is always returned as a host NumPy array: the build is
    distributed across ``devices`` (slice-sharded, in parallel) so a large phantom is never
    materialized whole on a single device, then it is gathered to the host and the device arrays are
    freed.

    Args:
        phantom_shape (tuple): Phantom shape in (rows, columns, slices).
        devices (sequence of jax devices, optional): Devices to build the phantom across.  Defaults to
            None, which uses all available devices (the GPUs when a GPU backend is present, else the
            CPU devices) -- the same set a reconstruction would shard over.  With more than one device
            the phantom is built **slice-sharded** (each device builds its own band of slices, no
            inter-device communication); with a single device it is built row-blocked to bound peak
            memory.  Either way the result is gathered to the host.
        max_block_gb (float, optional): Rough upper bound (GB) on the temporary memory used by the
            single-device (row-blocked) build.  Defaults to 4.0.  Ignored for a multi-device build
            (each device's slice band is already small).
        target_max_attenuation (float, optional): If given, scale the phantom so that the peak line
            integral through it (its forward projection) is roughly this value, **independent of the
            array shape**.  Without it, the sinogram grows linearly with the array size (a ray
            crosses more voxels), which is unrealistic -- real -log-attenuation sinograms sit around
            0 to 6-8.  The scale is analytic (from the main ellipsoid's extent along the longest
            axis) and ASSUMES ``delta_voxel ~= 1``, since the phantom cannot see the projector's
            voxel spacing (the sinogram scales linearly with ``delta_voxel``).  Default None leaves
            the phantom unscaled (the historical behavior).

    Returns:
        numpy.ndarray: A 3D host array of shape ``phantom_shape`` with the voxel intensities of the
        phantom.

    Note:
        The phantom is independent across voxels, so the multi-device build splits slices and the
        single-device build blocks rows -- neither needs inter-device communication.
    """
    scale = 1.0 if target_max_attenuation is None \
        else _shepp_logan_attenuation_scale(phantom_shape, target_max_attenuation)
    if devices is None:
        devices = jax.devices()        # all available devices (GPUs if a GPU backend is present, else CPU)
    if len(devices) > 1:
        phantom = _generate_3d_shepp_logan_sharded(phantom_shape, devices, scale)  # slice-sharded (device form)
    else:
        with jax.default_device(devices[0]):
            phantom = _generate_3d_shepp_logan_blocked(phantom_shape, max_block_gb, scale)
    # The phantom is a reference: gather to the host, drop any device-form slice padding (the sharded
    # build pads the slice axis up to the device count), free the device array(s), and return NumPy.
    n_rows, n_cols, n_slices = phantom_shape
    host = np.asarray(phantom)[:n_rows, :n_cols, :n_slices]
    if isinstance(phantom, jax.Array):
        phantom.delete()
    return host


# Semi-axes (rows, cols, slices) of the MAIN Shepp-Logan ellipsoid -- the largest structure, which
# dominates the longest line integral.  Must match the first ellipsoid in _add_shepp_logan_ellipsoids.
_MAIN_ELLIPSOID_SEMI_AXES = (0.69, 0.92, 0.9)


def _shepp_logan_attenuation_scale(phantom_shape, target_max_attenuation):
    """Intensity scale so the peak forward projection of the phantom is ~``target_max_attenuation``.

    The longest line integral runs along the array axis with the largest ``semi_axis_k * shape_k``: a
    ray through the center of the main ellipsoid crosses ~ ``semi_axis_k * shape_k`` voxels (the
    ellipsoid spans ``[-semi, semi]`` of the normalized ``[-1, 1]`` axis, i.e. ``semi`` of the
    ``shape_k`` half-axis voxels on each side).  With intensity 1 and ``delta_voxel = 1`` the peak
    sinogram value is ~ that voxel count, so scaling the phantom by
    ``target / max_k(semi_k * shape_k)`` puts the peak near ``target_max_attenuation``.  ASSUMES
    ``delta_voxel ~= 1`` (the phantom cannot see the projector's voxel spacing; the sinogram scales
    linearly with ``delta_voxel``).
    """
    longest_path_voxels = max(s * n for s, n in zip(_MAIN_ELLIPSOID_SEMI_AXES, phantom_shape))
    interior_intensity = 0.28  # The approximate average intensity along the center of the main ellipse
    return (target_max_attenuation / longest_path_voxels) / interior_intensity


def _add_shepp_logan_ellipsoids(phantom, grids, z_locations):
    """Add the nine standard low-dynamic-range Shepp-Logan ellipsoids to ``phantom``.

    ``phantom`` is ``(rows, cols, num_slices)`` and ``z_locations`` holds the z coordinate of each of
    its slices -- so this works on a full volume or on a contiguous band of slices (the sharded
    build passes one band per device).  Shared by the single-device and sharded paths so the
    ellipsoid definitions live in one place.
    """
    phantom = add_ellipsoid(phantom, grids, z_locations, 0, 0, 0, 0.69, 0.92, 0.9, intensity=1)
    # Smaller ellipsoids and other structures
    phantom = add_ellipsoid(phantom, grids, z_locations, 0, 0.0184, 0, 0.6624, 0.874, 0.88, intensity=-0.8)
    phantom = add_ellipsoid(phantom, grids, z_locations, 0.22, 0, 0, 0.41, 0.16, 0.21, angle=108, intensity=-0.2)
    phantom = add_ellipsoid(phantom, grids, z_locations, -0.22, 0, 0, 0.31, 0.11, 0.22, angle=72, intensity=-0.2)
    phantom = add_ellipsoid(phantom, grids, z_locations, 0, 0.35, 0, 0.21, 0.25, 0.5, intensity=0.1)
    phantom = add_ellipsoid(phantom, grids, z_locations, 0, 0.1, 0, 0.046, 0.046, 0.046, intensity=0.1)
    phantom = add_ellipsoid(phantom, grids, z_locations, 0, -0.1, 0, 0.046, 0.046, 0.046, intensity=0.1)
    phantom = add_ellipsoid(phantom, grids, z_locations, -0.08, -0.605, 0, 0.046, 0.023, 0.02, angle=0, intensity=0.1)
    phantom = add_ellipsoid(phantom, grids, z_locations, 0, -0.605, 0, 0.023, 0.023, 0.02, angle=0, intensity=0.1)
    return phantom


def _generate_3d_shepp_logan_blocked(phantom_shape, max_block_gb, scale=1.0):
    """Single-device Shepp-Logan build, blocked over ROWS with ``lax.map`` to bound peak memory.

    Every voxel of the phantom is independent, so the rows are split into fixed-size blocks and
    built one at a time: ``lax.map`` keeps only the current block's transients live (XLA scans the
    blocks) and the per-block results are concatenated.  Without this, the per-slice ``vmap`` inside
    each of the nine ``add_ellipsoid`` calls materializes several full-volume transients at once,
    which can exceed device memory at large sizes (e.g. 2048^3).  Bit-identical to the unblocked
    build (the per-voxel formula is unchanged; only the loop structure differs).  ``scale`` (default
    1.0) multiplies the result -- the optional attenuation rescaling.
    """
    N, M, P = phantom_shape
    # Block over ROWS (axis 0), not slices: lax.map stacks its results on a new LEADING axis, so
    # blocking the leading axis lets the blocks reassemble by a cheap contiguous reshape (concatenate
    # over axis 0).  Blocking slices would instead need a full-volume transpose to move the block axis
    # back to axis 2, which would defeat the memory bound.  Every voxel is independent, so the axis is
    # free and the result is identical to the slice-sharded build -- the sharded path blocks slices
    # only because that is the recon-by-slice device layout, a constraint this memory blocking lacks.
    # Choose a row-block count so one block's transients stay near max_block_gb.  A block holds a few
    # (block_rows, M, P) arrays at once (the ellipsoid comparison + the running sum), so budget ~4x.
    bytes_per_full = N * M * P * 4
    num_blocks = max(1, int(np.ceil(4 * bytes_per_full / (max_block_gb * 1024 ** 3))))
    block_rows = -(-N // num_blocks)          # ceil(N / num_blocks): rows per block
    n_blocks = -(-N // block_rows)            # ceil(N / block_rows): number of blocks
    padded_N = n_blocks * block_rows          # pad rows up to a whole number of fixed-size blocks

    # In-plane grids (rows x cols), padded on the row axis so every block is the same fixed size; the
    # pad rows are cropped off the final result.  z coordinates are shared across all blocks.
    x_grid, y_grid = jnp.meshgrid(jnp.linspace(-1, 1, N), jnp.linspace(-1, 1, M), indexing='ij')
    x_grid = jnp.pad(x_grid, ((0, padded_N - N), (0, 0)), mode='edge')
    y_grid = jnp.pad(y_grid, ((0, padded_N - N), (0, 0)), mode='edge')
    z_locations = jnp.linspace(-1, 1, P)

    def build_row_block(i):
        # Build the phantom for one fixed-size band of rows [i*block_rows : (i+1)*block_rows).
        r0 = i * block_rows
        xb = lax.dynamic_slice(x_grid, (r0, 0), (block_rows, M))
        yb = lax.dynamic_slice(y_grid, (r0, 0), (block_rows, M))
        grids = (xb, yb, None, None)          # add_ellipsoid uses only the x/y grids
        block = _add_shepp_logan_ellipsoids(jnp.zeros((block_rows, M, P)), grids, z_locations)
        return block if scale == 1.0 else block * scale       # scale per block -> no full-volume transient

    blocks = lax.map(build_row_block, jnp.arange(n_blocks))   # (n_blocks, block_rows, M, P)
    phantom = jnp.concatenate(blocks, axis=0)[:N]             # (N, M, P): merge blocks, crop padded rows
    return phantom


def _generate_3d_shepp_logan_sharded(phantom_shape, devices, scale=1.0):
    """Build the Shepp-Logan phantom slice-sharded across ``devices``.

    The phantom is embarrassingly parallel along slices -- each z-slice depends only on the shared
    in-plane grid and its own z coordinate -- so every device builds its own contiguous band of
    slices entirely on-device: no halos, no cross-device communication, and the full volume never
    exists on one device.  The result is a slice-sharded ``jax.Array`` padded to the device form
    (the tail slices are zero, exactly like a sharded recon), and matches the single-device phantom
    on the real slices.  ``scale`` (default 1.0) multiplies each shard -- the optional attenuation
    rescaling (the zero padding stays zero).
    """
    import mbirjax._sharding as mjs
    N, M, P = phantom_shape
    # Shard the phantom on the recon shard axis (the slice axis -- last axis for a (rows, cols,
    # slices) volume), so it matches how a model shards a recon.  The slice axis is split into one
    # contiguous band per device, and when P does not divide len(devices) it is padded up to the
    # next multiple (placement.padded_size), with the extra slices zero-filled and inert.
    placement = mjs.Placement(devices, axis=mj.TomographyModel.recon_shard_axis(), real_size=P)

    # Build one shard per device.  We just LOOP -- no ThreadPoolExecutor -- because each piece is
    # pure JAX: the `with jax.default_device(dev)` block DISPATCHES that device's build and returns
    # immediately (JAX dispatch is asynchronous), so the devices compute concurrently and
    # assemble_sharded joins them.  A thread pool is only needed when a per-device step BLOCKS (a
    # host transfer, block_until_ready, or reading addressable_shards) and would otherwise serialize
    # the loop -- which this fully independent build never does.
    pieces = []
    for dev, (start, end), n_real in placement.padded_shard_ranges():
        block = end - start                       # this device's band length on the padded slice axis
        with jax.default_device(dev):
            parts = []
            if n_real > 0:
                # This device owns real slices [start, start + n_real).  Rebuild the shared in-plane
                # grids locally (cheap, (N, M)) so they live on this device, then build just this
                # band's slices using the same z coordinates as the single-device phantom.
                x_grid, y_grid = jnp.meshgrid(jnp.linspace(-1, 1, N), jnp.linspace(-1, 1, M), indexing='ij')
                i_grid, j_grid = jnp.meshgrid(jnp.arange(N), jnp.arange(M), indexing='ij')
                grids = (x_grid, y_grid, i_grid, j_grid)
                z_band = jnp.linspace(-1, 1, P)[start:start + n_real]
                parts.append(_add_shepp_logan_ellipsoids(jnp.zeros((N, M, n_real)), grids, z_band))
            if block - n_real > 0:
                # Device-form zero padding at the end of the slice axis (kept inert downstream).
                parts.append(jnp.zeros((N, M, block - n_real)))
            # A shard is the real band, the zero pad, or -- on the one boundary device that straddles
            # real/padding -- both joined along the slice axis.
            piece = parts[0] if len(parts) == 1 else jnp.concatenate(parts, axis=2)
            if scale != 1.0:
                piece = piece * scale          # zero padding stays zero
        pieces.append(piece)

    # Wrap the per-device shards (each already resident on its device) as one slice-sharded array.
    return mjs.assemble_sharded(pieces, (N, M, placement.padded_size), placement.shard_structure(3))


def gen_translation_phantom(recon_shape, option, text, fill_rate=0.05, font_size=20, text_row_indices=None,
                            horizontal_offset=0, vertical_offset=0, voxel_slice_aspect=1.0):
    """
    Generate a synthetic ground truth phantom based on the selected option.

    Args:
        recon_shape (tuple[int, int, int]): Shape of the reconstruction volume.
        option (str): Phantom type to generate. Options are 'dots' or 'text'.
        text (list[str]): List of ASCII text strings to render.
        fill_rate (float, optional): Fill rate of the reconstruction volume. Default is 0.05.
        font_size (int, optional): Font size of the ASCII words. Default is 20.
        text_row_indices (list[int], optional): List of row indices where each text string should be placed. Default is None.
                                           If None, words are automatically distributed evenly across the first dimension.
                                           Must have the same length as 'words' if provided.
        horizontal_offset (int, optional): Horizontal offset of the text to be rendered. Positive value shifts the phantom right. Default is 0.
        vertical_offset (int, optional): Vertical offset of the text to be rendered. Positive value shifts the phantom up. Default is 0.
        voxel_slice_aspect (float, optional): Ratio between slice voxel spacing and column voxel spacing. Default is 1.0.

    Returns:
        np.ndarray: Generated phantom volume.
    """
    if option == 'dots':
        return gen_dot_phantom(recon_shape, fill_rate)
    elif option == 'text':
        return gen_text_phantom(recon_shape, text, font_size, text_row_indices, horizontal_offset, vertical_offset,
                                voxel_slice_aspect=voxel_slice_aspect)
    else:
        raise ValueError(f"Unsupported phantom option: {option}")


def gen_dot_phantom(recon_shape, fill_rate):
    """
    Generate a synthetic ground truth reconstruction volume.

    Args:
        recon_shape (tuple[int, int, int]): Shape of the reconstruction volume.
        fill_rate (float): Fill rate of the reconstruction volume.

    Returns:
        np.ndarray: Ground truth reconstruction volume with sparse binary features.
    """
    np.random.seed(42)
    gt_recon = np.zeros(recon_shape, dtype=np.float32)

    y_pad = recon_shape[0] // 6
    central_start = y_pad
    central_end = recon_shape[0] - y_pad

    row_size = recon_shape[1] * recon_shape[2]
    num_ones_per_row = int(row_size * fill_rate)

    for row_idx in range(central_start, central_end):
        flat_row = gt_recon[row_idx].flatten()
        positions_ones = np.random.choice(row_size, num_ones_per_row, replace=False)
        flat_row[positions_ones] = 1.0
        gt_recon[row_idx] = flat_row.reshape(recon_shape[1:])

    return gt_recon


def gen_text_phantom(recon_shape, words, font_size, row_indices=None, horizontal_offset=0,
                     vertical_offset=0, voxel_slice_aspect=1.0, font_path="DejaVuSans.ttf"):
    """
    Generate a 3D text phantom with binary word patterns embedded in specific slices.

    Args:
        recon_shape (tuple[int, int, int]): Shape of the phantom volume (num_rows, num_cols, num_slices).
        words (list[str]): List of ASCII words to render.
        font_size (int): Font size of ASCII words.
        row_indices (list[int], optional): List of row indices where each word should be placed. Default is None.
                                           If None, words are automatically distributed evenly across the first dimension.
                                           Must have the same length as 'words' if provided.
        horizontal_offset (int, optional): Horizontal offset of the text to be rendered. Positive value shifts the phantom right. Default is 0.
        vertical_offset (int, optional): Vertical offset of the text to be rendered. Positive value shifts the phantom up. Default is 0.
        voxel_slice_aspect (float, optional): Ratio between slice voxel spacing and column voxel spacing. The rendered
            text is corrected so it has the same physical aspect ratio when slices are anisotropic. Default is 1.0.
        font_path (str, optional): Path to the TrueType font file. Default is "DejaVuSans.ttf".

    Returns:
        np.ndarray: A 3D numpy array of shape `recon_shape` containing the text phantom.
    """
    if voxel_slice_aspect <= 0:
        raise ValueError(f"voxel_slice_aspect must be positive. Got {voxel_slice_aspect}.")

    if row_indices is not None:
        if len(row_indices) != len(words):
            raise ValueError(
                f"Length of row_indices ({len(row_indices)}) must match length of words ({len(words)})")

        for idx in row_indices:
            if not (0 <= idx < recon_shape[0]):
                raise ValueError(f"Row index {idx} is out of bounds for first dimension of size {recon_shape[0]}")

        positions = []
        for row_idx in row_indices:
            col_pos = recon_shape[1] // 2 + horizontal_offset
            slice_pos = recon_shape[2] // 2 - vertical_offset
            positions.append((row_idx, col_pos, slice_pos))
    else:
        positions = []
        row_positions = np.linspace(0, recon_shape[0] - 1, len(words) + 2)[1:-1]
        for r in row_positions:
            col_pos = recon_shape[1] // 2 + horizontal_offset
            slice_pos = recon_shape[2] // 2 - vertical_offset
            positions.append((int(round(r)), col_pos, slice_pos))

    array_size = int(np.minimum(recon_shape[1], recon_shape[2]))
    array_num_cols = array_size
    array_num_slices = int(round(array_size / voxel_slice_aspect))
    array_num_slices = min(max(array_num_slices, 1), recon_shape[2])

    phantom = np.zeros(recon_shape, dtype=np.float32)
    try:
        font = ImageFont.truetype(font_path, size=font_size)
    except OSError:
        from pathlib import Path
        fallback_paths = [
            "/System/Library/Fonts/Supplemental/Arial.ttf",  # macOS fallback
            "/Library/Fonts/Arial.ttf",  # Additional macOS path
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux fallback
        ]
        for fallback in fallback_paths:
            if Path(fallback).exists():
                font = ImageFont.truetype(fallback, size=font_size)
                break
        else:
            raise FileNotFoundError(
                f"Could not find a usable font. Tried the following paths:\n"
                + "\n".join(fallback_paths)
                + "\nPlease install one of these fonts or specify a valid font_path."
            )

    for word, (r, c, s) in zip(words, positions):
        img = Image.new('L', (array_size, array_size), 0)
        draw = ImageDraw.Draw(img)

        text_box = draw.textbbox((0, 0), word, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]

        x = (array_size - text_width) // 2 - text_box[0]
        y = (array_size - text_height) // 2 - text_box[1]
        draw.text((x, y), word, fill=1, font=font)

        word_array = np.array(img.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT))
        if array_num_slices != array_size:
            word_img = Image.fromarray(word_array)
            nearest_resampling = getattr(getattr(Image, 'Resampling', Image), 'NEAREST')
            word_img = word_img.resize((array_num_slices, array_num_cols), resample=nearest_resampling)
            word_array = np.array(word_img)
        word_array = (word_array > 0).astype(np.float32)

        # Crop or pad word_array to fit in the recon volume
        r_start, r_end = r, r + 1
        c_start = c - array_num_cols // 2
        c_end = c_start + array_num_cols
        s_start = s - array_num_slices // 2
        s_end = s_start + array_num_slices

        c_start_valid = max(c_start, 0)
        c_end_valid = min(c_end, recon_shape[1])
        s_start_valid = max(s_start, 0)
        s_end_valid = min(s_end, recon_shape[2])

        word_c_start = c_start_valid - c_start
        word_c_end = word_c_start + (c_end_valid - c_start_valid)
        word_s_start = s_start_valid - s_start
        word_s_end = word_s_start + (s_end_valid - s_start_valid)

        # Place cropped word_array into phantom
        word_crop = word_array[word_c_start:word_c_end, word_s_start:word_s_end]
        phantom[r_start:r_end, c_start_valid:c_end_valid, s_start_valid:s_end_valid] = word_crop

    return phantom


def _gen_ellipsoid(x_grid, y_grid, z_grid, x0, y0, z0, a, b, c, gray_level, alpha=0, beta=0, gamma=0):
    """
    Return an image with a 3D ellipsoid in a 3D plane with a center of [x0,y0,z0] and ...

    Args:
        x_grid(jax array): 3D grid of X coordinate values.
        y_grid(jax array): 3D grid of Y coordinate values.
        z_grid(jax array): 3D grid of Z coordinate values.
        x0(float): horizontal center of ellipsoid.
        y0(float): vertical center of ellipsoid.
        z0(float): normal center of ellipsoid.
        a(float): X-axis radius.
        b(float): Y-axis radius.
        c(float): Z-axis radius.
        gray_level(float): Gray level for the ellipse.
        alpha(float): [Default=0.0] counter-clockwise angle of rotation by X-axis in radians.
        beta(float): [Default=0.0] counter-clockwise angle of rotation by Y-axis in radians.
        gamma(float): [Default=0.0] counter-clockwise angle of rotation by Z-axis in radians.

    Return:
        ndarray: 3D array with the same shape as x_grid, y_grid, and z_grid

    """
    # Generate Rotation Matrix.
    rx = np.array([[1, 0, 0], [0, np.cos(-alpha), -np.sin(-alpha)], [0, np.sin(-alpha), np.cos(-alpha)]])
    ry = np.array([[np.cos(-beta), 0, np.sin(-beta)], [0, 1, 0], [-np.sin(-beta), 0, np.cos(-beta)]])
    rz = np.array([[np.cos(-gamma), -np.sin(-gamma), 0], [np.sin(-gamma), np.cos(-gamma), 0], [0, 0, 1]])
    r = np.dot(rx, np.dot(ry, rz))

    cor = np.array([x_grid.flatten() - x0, y_grid.flatten() - y0, z_grid.flatten() - z0])

    image = ((np.dot(r[0], cor)) ** 2 / a ** 2 + (np.dot(r[1], cor)) ** 2 / b ** 2 + (
        np.dot(r[2], cor)) ** 2 / c ** 2 <= 1.0) * gray_level

    return image.reshape(x_grid.shape)


@jax.jit
def add_ellipsoid(current_volume, grids, z_locations, x0, y0, z0, a, b, c, angle=0, intensity=1.0):
    """
    Add an ellipsoid to an existing jax array.  This is done using lax.scan over the z slices to avoid
    using really large arrays when the volume is large.

    Args:
        current_volume (jax array): 3D volume
        grids (tuple):  A tuple of x_grid, y_grid, i_grid, j_grid obtained as in generate_3d_shepp_logan_low_dynamic_range
        z_locations (jax array): A 1D array of z coordinates of the volume
        x0 (float): x center for the ellipsoid
        y0 (float): y center for the ellipsoid
        z0 (float): z center for the ellipsoid
        a (float): x radius
        b (float): y radius
        c (float): z radius
        angle (float): angle of rotation of the ellipsoid in the xy plane around (x0, y0)
        intensity (float): The constant value of the ellipsoid to be added.

    Returns:
        3D jax array: current_volume + ellipsoid
    """

    # Unpack the grids and determine the xy locations for this angle
    x_grid, y_grid, i_grid, j_grid = grids
    cos_angle = jnp.cos(jnp.deg2rad(angle))
    sin_angle = jnp.sin(jnp.deg2rad(angle))
    Xr = cos_angle * (x_grid - x0) + sin_angle * (y_grid - y0)
    Yr = -sin_angle * (x_grid - x0) + cos_angle * (y_grid - y0)

    # Determine which xy locations will be updated for this ellipsoid
    xy_norm = Xr**2 / a**2 + Yr**2 / b**2

    def add_slice_vmap(volume_slice, z):
        return volume_slice + intensity * ((xy_norm + (z - z0)**2 / c**2) <= 1).astype(float)

    volume_map = jax.vmap(add_slice_vmap, in_axes=(2, 0), out_axes=2)
    current_volume = volume_map(current_volume, z_locations)

    return current_volume


class ObjectType(str, Enum):
    SHEPP_LOGAN = 'shepp-logan'
    CUBE = 'cube'


class ModelType(str, Enum):
    PARALLEL = 'parallel'
    CONE = 'cone'
    TRANSLATION = 'translation'


def get_helical_half_rotation_slice_range(
    ct_model,
    helical_pitch,
    helical_z_shifts,
):
    """
    Return the contiguous slice range whose z positions are visible for at least
    half a rotation.

    Assumes helical_z_shifts are monotone and uniformly sampled.
    
    Returns:
        start_slice, stop_slice, valid_slice_mask
        where stop_slice is exclusive.
    """
    recon_shape = ct_model.get_params('recon_shape')
    delta_voxel, voxel_slice_aspect, recon_slice_offset = ct_model.get_params(
        ['delta_voxel', 'voxel_slice_aspect', 'recon_slice_offset']
    )
    sinogram_shape = ct_model.get_params('sinogram_shape')
    delta_det_row = ct_model.get_params('delta_det_row')
    magnification = ct_model.get_magnification()

    num_slices = recon_shape[2]
    num_det_rows = sinogram_shape[1]

    delta_voxel_slice = voxel_slice_aspect * delta_voxel

    # Slice-center z locations in reconstruction coordinates.
    k = jnp.arange(num_slices)
    z_k = delta_voxel_slice * (k - (num_slices - 1) / 2.0) + recon_slice_offset

    # Detector height mapped to isocenter.
    det_height_iso = num_det_rows * delta_det_row / magnification

    # Table travel range.
    z_shift_min = jnp.min(helical_z_shifts)
    z_shift_max = jnp.max(helical_z_shifts)

    # Extra interior trim needed when pitch > 1.
    # For pitch <= 1, the table-travel endpoints already have at least
    # half-rotation visibility, so no trim is needed.
    trim = 0.5 * det_height_iso * jnp.maximum(float(helical_pitch) - 1.0, 0.0)

    z_min = z_shift_min + trim
    z_max = z_shift_max - trim

    valid_slice_mask = (z_k >= z_min) & (z_k <= z_max)

    valid_indices = np.where(np.asarray(valid_slice_mask))[0]
    if len(valid_indices) == 0:
        raise ValueError(
            "No slices are visible for at least half a rotation. "
            "Check helical_pitch, helical_z_range, num_views, and detector height."
        )

    start_slice = int(valid_indices[0])
    stop_slice = int(valid_indices[-1] + 1)

    return start_slice, stop_slice


def generate_demo_data(
    object_type: Union[ObjectType, str] = ObjectType.SHEPP_LOGAN,
    model_type: Union[ModelType, str] = ModelType.CONE,
    num_views: int = 64,
    num_det_rows: int = 96,
    delta_det_row: float = 1,
    num_det_channels: int = 128,
    delta_det_channel: float = 1,
    num_x_translations: int = 7,
    num_z_translations: int = 7,
    x_spacing: float = 22,
    z_spacing: float = 22,
    use_helical: bool = False,
    helical_pitch: float | None = None,
    helical_z_range: float | None = None,
    helical_z_center: float = 0.0,
    use_curved_detector: bool = False,
    voxel_row_aspect: float = 1.0,
    voxel_slice_aspect: float = 1.0,
    target_max_attenuation: float | None = None,
    devices: list | tuple | None = None,
) -> tuple:
    """
    Create a simple object and a sinogram for demonstration purposes.

    This function will create a 3D volume (aka object or phantom) of the specified type, then use the model type and
    parameters to create a simulated sinogram.  The object type 'shepp-logan' gives a simplified version of the
    classic Shepp-Logan test phantom, and type 'cube' gives a simple cube object.

    The phantom and the sinogram are built distributed across the model's devices (in parallel) so a
    large problem is never materialized whole on one device, then gathered to the host: both are
    returned on the host.  The output sinogram has shape (num_views, num_det_rows,
    num_det_channels); each 2D array sinogram[view_index] is a simulated image from the detector, with
    num_det_rows indicating the vertical size and num_det_channels the horizontal size.

    Args:
        object_type (str, optional): One of 'shepp-logan' or 'cube'.  Defaults to 'shepp-logan'.
        model_type (str, optional): One of 'parallel', 'cone', or 'translation'.  Defaults to 'cone'.
        num_views (int, optional):  Number of views in the output sinogram.  Defaults to 64. Ignored when model_type is 'translation'
        num_det_rows (int, optional): Number of rows (vertical) in the output sinogram.  Defaults to 96.
        num_det_channels (int, optional): Number of channels (horizontal) in the output sinogram.  Defaults to 128.
        num_x_translations (int, optional): Number of horizontal translations for translation mode.  Defaults to 7.
        num_z_translations (int, optional): Number of vertical translations for translation mode.  Defaults to 7.
        x_spacing (float, optional): Horizontal spacing between translations in ALU.  Defaults to 22.
        z_spacing (float, optional): Vertical spacing between translations in ALU.  Defaults to 22.
        use_helical (bool, optional):
            If True and model_type == 'cone', generate a helical cone-beam trajectory by
            supplying per-view z_shifts to ConeBeamModel. Defaults to False.
        helical_pitch (float, optional):
            Helical pitch (dimensionless) for helical mode.
            pitch = (table travel per rotation) / (det height at iso).  This is the fraction of the detector height at iso traveled per rotation.
        helical_z_range (float, optional): Total axial travel over the scan in ALU for helical mode.
        helical_z_center (float, optional): Midpoint of axial travel over the scan in ALU for helical mode.
        use_curved_detector (bool, optional): (cone beam geometry parameter)
        voxel_row_aspect (float, optional): Aspect ratio for recon rows relative to columns.  Defaults to 1.0.
        voxel_slice_aspect (float, optional): Aspect ratio for recon slices relative to rows.  Defaults to 1.0.
        target_max_attenuation (float, optional): Target max sinogram attenuation for Shepp-Logan phantom.  Defaults to None, for which each voxel is in the range [0, 1].  May not be accurate if any detector or voxel dimensions are not 1.
        devices (sequence of jax devices, optional): Devices to distribute the generation across.
            Defaults to None, which uses the model's automatic selection (all available GPUs, else the
            CPU devices).  The phantom and sinogram are built across these devices in parallel and then
            gathered to the host -- this only affects where the work runs, not the result.

    Returns:
        tuple: (object, sinogram, params)
            - object: the phantom volume, shape recon_shape = (num_rows, num_cols, num_slices).
            - sinogram: shape (num_views, num_det_rows, num_det_channels).
            - params (dict): contains 'angles' and, for 'cone', also 'source_detector_dist' and 'source_iso_dist'.

        sinogram is always a host NumPy array (what ``recon`` prefers -- it shards it across
        devices itself), and the arrays in ``params`` are NumPy as well.  object is host NumPy
        for 'shepp-logan' but a JAX array for 'cube'.
    """
    # Coerce types to Enum
    object_type = ObjectType(object_type)
    model_type = ModelType(model_type)

    start_angle = -np.pi
    end_angle = np.pi

    # Initialize model

    if model_type == ModelType.PARALLEL:
        start_angle = 0
        sinogram_shape = (num_views, num_det_rows, num_det_channels)
        angles = jnp.linspace(start_angle, end_angle, num_views, endpoint=False)
        ct_model_for_generation = mj.ParallelBeamModel(sinogram_shape, angles)
        ct_model_for_generation.set_params(voxel_row_aspect=voxel_row_aspect)
        ct_model_for_generation.set_params(voxel_slice_aspect=voxel_slice_aspect)
        ct_model_for_generation.auto_set_recon_geometry()
        params = {'angles': angles, 'voxel_row_aspect': voxel_row_aspect, 'voxel_slice_aspect': voxel_slice_aspect}
    elif model_type == ModelType.CONE:
        # For cone beam geometry, we need to describe the distances source to detector and source to rotation axis.
        # np.Inf is an allowable value, in which case this is essentially parallel beam
        source_detector_dist = 4 * num_det_channels
        source_iso_dist = source_detector_dist/2
        sinogram_shape = (num_views, num_det_rows, num_det_channels)
        if not use_helical:
            angles = jnp.linspace(start_angle, end_angle, num_views, endpoint=False)
            ct_model_for_generation = mj.ConeBeamModel(sinogram_shape, angles, source_detector_dist=source_detector_dist,
                                                       source_iso_dist=source_iso_dist, use_curved_detector=use_curved_detector)
            ct_model_for_generation.set_params(voxel_row_aspect=voxel_row_aspect)
            ct_model_for_generation.set_params(voxel_slice_aspect=voxel_slice_aspect)
            ct_model_for_generation.auto_set_recon_geometry()
            params = {'angles': angles, 'source_detector_dist': source_detector_dist, 'source_iso_dist': source_iso_dist,
                      'use_curved_detector': use_curved_detector, 'voxel_row_aspect': voxel_row_aspect, 'voxel_slice_aspect': voxel_slice_aspect}
        else:
            # Require both helical_pitch and helical_z_range
            if helical_pitch is None or helical_z_range is None:
                raise ValueError("Helical trajectory requires both helical_pitch and helical_z_range.")

            # Compute magnification
            if jnp.isinf(source_detector_dist):
                magnification = 1
            else:
                magnification = source_detector_dist / source_iso_dist

            # detector height mapped to iso, in ALU
            det_height_iso = float(num_det_rows) * (delta_det_row / magnification)

            # Travel per rotation (ALU) and derived rotations/views-per-rotation
            z_per_rot = float(helical_pitch) * det_height_iso
            if z_per_rot <= 0:
                raise ValueError(f"helical_pitch must be > 0 (got {helical_pitch}).")
            if float(helical_z_range) < 0:
                raise ValueError(f"helical_z_range must be >= 0 (got {helical_z_range}).")

            # Derived number of rotations and views per rotation
            if float(helical_z_range) == 0.0: # circular reconstruction
                num_rotations = 1.0
                views_per_rotation = float(num_views)
            else:
                num_rotations = float(helical_z_range) / z_per_rot
                if num_rotations <= 0:
                    raise ValueError("Derived num_rotations <= 0; check pitch/z_range.")
                views_per_rotation = float(num_views) / num_rotations

            # Angles: advance by 2*pi/views_per_rotation each view
            angle_step = (2.0 * np.pi) / views_per_rotation
            angles = start_angle + angle_step * jnp.arange(num_views)

            # z_shifts: span z_range across scan, centered at z_center
            z0 = float(helical_z_center) - 0.5 * float(helical_z_range)
            z1 = float(helical_z_center) + 0.5 * float(helical_z_range)
            helical_z_shifts = jnp.linspace(z0, z1, num_views, endpoint=True)

            ct_model_for_generation = mj.ConeBeamModel(
                sinogram_shape,
                angles,
                source_detector_dist=source_detector_dist,
                source_iso_dist=source_iso_dist,
                helical_z_shifts=helical_z_shifts,
                use_curved_detector=use_curved_detector
            )
            ct_model_for_generation.set_params(voxel_row_aspect=voxel_row_aspect)
            ct_model_for_generation.set_params(voxel_slice_aspect=voxel_slice_aspect)
            ct_model_for_generation.auto_set_recon_geometry()

            params = {
                'angles': angles,
                'source_detector_dist': source_detector_dist,
                'source_iso_dist': source_iso_dist,
                'helical_z_shifts': helical_z_shifts,
                'use_curved_detector': use_curved_detector,
                'voxel_row_aspect': voxel_row_aspect,
                'voxel_slice_aspect': voxel_slice_aspect
            }
    elif model_type == ModelType.TRANSLATION:
        source_iso_dist = min(num_det_rows, num_det_channels) / 2
        source_detector_dist = source_iso_dist
        translation_vectors = gen_translation_vectors(num_x_translations, num_z_translations, x_spacing, z_spacing)
        num_views = translation_vectors.shape[0]
        sinogram_shape = (num_views, num_det_rows, num_det_channels)
        ct_model_for_generation = mj.TranslationModel(sinogram_shape, translation_vectors, source_detector_dist=source_detector_dist,
                                                      source_iso_dist=source_iso_dist)
        params = {'translation_vectors': translation_vectors}
    else:
        raise ValueError(f'Invalid model type. Expected one of {[m.value for m in ModelType]}, got {model_type}')

    # Pin the generation model to the requested devices so the phantom, the forward projection, and
    # the returned sinogram all share one layout.  (Without this the model auto-selects devices, so a
    # device subset would reshard the phantom and return the sinogram on the auto devices instead.)
    # None leaves the automatic selection in place.
    if devices is not None:
        ct_model_for_generation.configure_devices(devices)

    # Generate the phantom on the MODEL's devices (slice-sharded across all of them when multi-device,
    # so a large phantom is never built whole on one device).  generate_3d_shepp_logan_low_dynamic_range
    # gathers it to the host and returns NumPy; gen_cube_phantom returns a JAX array.
    print('Creating phantom')
    recon_shape = ct_model_for_generation.get_params('recon_shape')
    model_devices = ct_model_for_generation.shard_devices   # None -> the generator uses all available
    phantom_shape = recon_shape
    embed_slice_start = 0
    embed_slice_stop = recon_shape[2]
    if model_type == ModelType.CONE and use_helical:
        embed_slice_start, embed_slice_stop = get_helical_half_rotation_slice_range(
            ct_model_for_generation,
            helical_pitch,
            helical_z_shifts
        )
        phantom_shape = (
            recon_shape[0],
            recon_shape[1],
            embed_slice_stop - embed_slice_start,
        )
    if object_type == ObjectType.SHEPP_LOGAN:
        phantom_core = generate_3d_shepp_logan_low_dynamic_range(
            phantom_shape, devices=model_devices, target_max_attenuation=target_max_attenuation)
    elif object_type == ObjectType.CUBE:
        phantom_core = gen_cube_phantom(phantom_shape)
    else:
        raise ValueError(f'Invalid object type. Expected one of {[o.value for o in ObjectType]}, got {object_type}')
    if model_type == ModelType.CONE and use_helical:
        # Embed the partial-slice phantom into the full recon volume (host arrays throughout --
        # phantom_core is already host NumPy).
        phantom = np.zeros(recon_shape, dtype=np.float32)
        phantom[:, :, embed_slice_start:embed_slice_stop] = phantom_core
    else:
        phantom = phantom_core

    # Forward project across the model's devices, then gather the sinogram to the host.  Keep the
    # sinogram SHARDED out of forward_project (output_sharded=True) so the whole sinogram is never
    # routed through a single device (which OOMs at large sizes -- e.g. a 32 GiB 2048^3 sinogram on one
    # GPU), then gather it on a separate line.  _gather_sinogram is _gather_to_host (assemble on the
    # host shard-by-shard) PLUS a crop of any inert view/detector-row padding back to the real shape --
    # bare _gather_to_host would keep that padding.  recon prefers a host sinogram (it shards it itself)
    # and the shepp-logan phantom is already host (the cube phantom stays a JAX array); the params
    # arrays are NumPy too; nothing large is left resident on a device.
    print('Creating sinogram')
    sinogram_sharded = ct_model_for_generation.forward_project(phantom, output_sharded=True)
    sinogram = ct_model_for_generation._gather_sinogram(sinogram_sharded)
    params = {k: (np.asarray(v) if isinstance(v, jax.Array) else v) for k, v in params.items()}

    del ct_model_for_generation
    return phantom, sinogram, params


def gen_translation_vectors(num_x_translations, num_z_translations, x_spacing, z_spacing):
    """
    Generate translation vectors for lateral (x) and axial (z) displacements.

    Args:
        num_x_translations (int): Number of x-direction translations
        num_z_translations (int): Number of z-direction translations
        x_spacing (float): Spacing between x translations in ALU
        z_spacing (float): Spacing between z translations in ALU

    Returns:
        np.ndarray: Array of shape (num_views, 3) with translation vectors [dx, dy, dz]
    """
    num_views = num_x_translations * num_z_translations
    translation_vectors = np.zeros((num_views, 3))

    x_center = (num_x_translations - 1) / 2
    z_center = (num_z_translations - 1) / 2

    idx = 0
    for row in range(num_z_translations):
        for col in range(num_x_translations):
            dx = (col - x_center) * x_spacing
            dz = (row - z_center) * z_spacing
            dy = 0
            translation_vectors[idx] = [dx, dy, dz]
            idx += 1

    return translation_vectors


def gen_cube_phantom(recon_shape, device=None):
    """Code to generate a simple phantom """
    # Compute phantom height and width
    num_recon_rows, num_recon_cols, num_recon_slices = recon_shape[:3]
    phantom_rows = num_recon_rows // 4  # Phantom height
    phantom_cols = num_recon_cols // 4  # Phantom width

    # Allocate phantom memory
    phantom = np.zeros((num_recon_rows, num_recon_cols, num_recon_slices))

    # Compute start and end locations
    start_rows = (num_recon_rows - phantom_rows) // 2
    stop_rows = (num_recon_rows + phantom_rows) // 2
    start_cols = (num_recon_cols - phantom_cols) // 2
    stop_cols = (num_recon_cols + phantom_cols) // 2
    for slice_index in np.arange(num_recon_slices):
        shift_cols = int(slice_index * phantom_cols / num_recon_slices)
        phantom[start_rows:stop_rows, (shift_cols + start_cols):(shift_cols + stop_cols), slice_index] = 1.0 / max(
            phantom_rows, phantom_cols)

    return jnp.array(phantom, device=device)


def stitch_arrays(array_list, overlap, axis=2, ramp_overlap=None):
    """
    Concatenate JAX arrays along one axis while linearly blending a fixed overlap
    between adjacent arrays.

    This behaves like `jnp.concatenate` except that for each adjacent pair, the
    first `overlap_length` elements of the second array and the last
    `overlap_length` elements of the current result are combined by a piece-wise linear cross‑fade.

    All non‑`axis` dimensions must match across inputs.

    Args:
        array_list (list of ndarray or jax.Array): Sequence of 2+ arrays to stitch.  The result is
            built on the inputs' own array module, so host (NumPy) inputs stitch on the host (no gather
            to a single device) and jax inputs stitch on-device.
        overlap (int): Number of elements overlapped between arrays.
            Must be `>= 1` and not exceed the length of any input along `axis`.
        axis (int, optional): Axis along which to stitch. Defaults to 2.
        ramp_overlap (int, optional): Target number of blended (0 < w < 1) elements. Defaults to None.

    Returns:
        ndarray or jax.Array: Stitched array, on the inputs' own array module (host NumPy in -> host
        out, jax in -> on-device out). Its shape equals the input shape with the
        length along `axis` equal to:

            sum(len_k) - (len(array_list) - 1) * overlap_length

        where `len_k` are the lengths of each input along `axis`.

    Raises:
        ValueError: If fewer than two arrays are provided, if non‑`axis`
            dimensions differ, or if any array is shorter than
            `overlap_length` along `axis`.

    Example:
        >>> import jax.numpy as jnp
        >>> a0 = jnp.arange(2*2*5).reshape(2, 2, 5)
        >>> a1 = jnp.arange(2*2*6).reshape(2, 2, 6)
        >>> out = stitch_arrays([a0, a1], overlap=3, axis=2)
        >>> out.shape
        (2, 2, 8)

        # 8 comes from 5 + 6 - 3 (one overlap between two arrays).
    """
    # Check for valid input
    if not isinstance(array_list, list) or len(array_list) < 2:
        raise ValueError('array_list must be a list of 2 or more jax arrays.')
    for dim in range(array_list[0].ndim):
        lengths = [array.shape[dim] for array in array_list]
        if dim != axis:
            if np.amax(lengths) != np.amin(lengths):
                raise ValueError('The shapes of the arrays in array_list must be the same except in the dimension specified by axis.')
        if dim == axis:
            if np.amin(lengths) < overlap:
                raise ValueError('Each array must have length at least overlap in the dimension specified by axis.')

    # Create weights for blending two arrays
    # ramp_overlap is the target number of blended (0 < w < 1) pixels
    # However, if ramp_overlap and overlap have different parities, then ramp_overlap is decremented to match parity.
    if ramp_overlap is None:
        ramp_overlap = overlap // 2  # default: ramp over ~half the overlap
    ramp_overlap = min(ramp_overlap, overlap)
    ramp_overlap -= (overlap - ramp_overlap) % 2  # match overlap's parity -> symmetric plateaus
    ramp_overlap = max(ramp_overlap, overlap % 2)  # floor at 0 (even overlap) or 1 (odd overlap)
    flat_pad = (overlap - ramp_overlap) // 2  # equal plateau on each side
    # Build the blend weights and assemble on the inputs' OWN array module so the result stays where the
    # inputs live: host (NumPy) arrays stitch on the HOST (no gather to a single device), device/jax
    # arrays stitch on-device.  split_sino_recon relies on this -- it passes host halves, so the full
    # volume is never reassembled on one GPU (which would defeat the half-at-a-time memory saving and
    # OOM for a recon too large to fit whole).  float32 weights avoid upcasting a float32 recon to f64.
    xp = jnp if any(isinstance(a, jax.Array) for a in array_list) else np
    ramp = (xp.arange(ramp_overlap, dtype=xp.float32) + 1) / (ramp_overlap + 1)  # strictly between 0 and 1
    weights = xp.concatenate([xp.zeros(flat_pad, dtype=xp.float32), ramp, xp.ones(flat_pad, dtype=xp.float32)])

    # Broadcast weights to match array dimensions
    weights_shape = np.ones(array_list[0].ndim, dtype=int)
    weights_shape[0] = len(weights)
    weights = weights.reshape(weights_shape)

    # Start with the first array in the list
    stitched = xp.swapaxes(array_list[0], 0, axis)

    # Iterate through each subsequent array in the list
    for next_array in array_list[1:]:
        # Extract the overlap from the current end of the stitched array and the beginning of the next array
        overlap_current = stitched[-overlap:]
        next_array = xp.swapaxes(next_array, 0, axis)
        overlap_next = next_array[:overlap]

        # Weighted average for the overlapping part
        weighted_overlap = (1 - weights) * overlap_current + weights * overlap_next

        # Replace the overlap in the stitched array
        stitched = xp.concatenate([stitched[:-overlap], weighted_overlap], axis=0)

        # Append the non-overlapping remainder of the next array
        stitched = xp.concatenate([stitched, next_array[overlap:]], axis=0)

    return xp.swapaxes(stitched, 0, axis)


def get_ct_model(geometry_type, sinogram_shape, angles, source_detector_dist=None, source_iso_dist=None, helical_z_shifts=None):
    """
    Create an instance of TomographyModel with the given parameters

    Args:
        geometry_type (str): 'parallel' or 'cone'
        sinogram_shape (tuple list of int): (num_views, num_rows, num_channels)
        angles (ndarray of float): 1D vector of projection angles in radians
        source_detector_dist (float or None, optional): Distance in ALU from source to detector.  Defaults to None for geometries that don't need this.
        source_iso_dist (float or None, optional): Distance in ALU from source to iso.  Defaults to None for geometries that don't need this.
        helical_z_shifts (numpy or jax array, optional):
            Per-view axial shifts (ALU), same length as angles.
            Required when use_helical=True.

    Returns:
        An instance of ConeBeamModel or ParallelBeam model
    """
    if geometry_type == 'cone':
        model = mj.ConeBeamModel(sinogram_shape, angles, source_detector_dist=source_detector_dist,
                                 source_iso_dist=source_iso_dist, helical_z_shifts=helical_z_shifts)
    elif geometry_type == 'parallel':
        if helical_z_shifts is not None:
            warnings.warn("Helical mode (helical_z_shifts) is only supported for geometry_type='cone'; ignoring z_shifts.", UserWarning)
        model = mj.ParallelBeamModel(sinogram_shape, angles)
    else:
        raise ValueError('Invalid geometry type.  Expected cone or parallel, got {}'.format(geometry_type))

    return model


_geometry_class_by_type_str = None


def _resolve_geometry_class(geometry_type):
    """Resolve a geometry model class from the ``geometry_type`` string that
    :meth:`~mbirjax.TomographyModel.get_all_params` records (``str(type(model))``)."""
    global _geometry_class_by_type_str
    if _geometry_class_by_type_str is None:
        _geometry_class_by_type_str = {
            str(cls): cls for cls in (mj.ConeBeamModel, mj.ParallelBeamModel,
                                      mj.TranslationModel, mj.MultiAxisParallelModel)}
    try:
        return _geometry_class_by_type_str[geometry_type]
    except KeyError:
        raise ValueError("build_model: unrecognized geometry_type {!r}; expected one of {}."
                         .format(geometry_type, list(_geometry_class_by_type_str)))


def build_model(required_params, optional_params=None, regularization=None):
    """
    Construct a model from parameter dicts and compute its reconstruction geometry.

    The single place the ``construct -> set_params -> auto_set_recon_geometry`` sequence lives, so a
    caller never forgets the final ``auto_set_recon_geometry`` (which would leave the reconstruction
    grid sized with default detector pitches).  Because ``required_params`` carries ``geometry_type``
    (see :meth:`~mbirjax.TomographyModel.get_all_params`), the correct model class is resolved here
    and ``(required_params, optional_params)`` is a self-contained model description -- calling this
    reads like calling the constructor through the new interface.

    Args:
        required_params (dict): The model constructor's arguments, including ``geometry_type`` (as
            returned in the first element of ``get_all_params``).
        optional_params (dict, optional): Additional parameters applied with ``set_params`` (detector
            pitches, offsets, ``delta_voxel``, ``recon_shape``, ...).  Defaults to None.
        regularization (dict, optional): Recon-time regularization parameters applied with
            ``set_params``.  Defaults to None (the model's default regularization).

    Returns:
        TomographyModel: the constructed model, with ``auto_set_recon_geometry`` applied.
    """
    required_params = dict(required_params)
    model_class = _resolve_geometry_class(required_params.pop('geometry_type'))
    model = model_class(**required_params)

    optional_params = dict(optional_params) if optional_params else {}
    # A pinned recon_shape must be applied AFTER auto_set_recon_geometry, or the automatic pass would
    # overwrite it (the translation reader pins recon_shape; a faithful save/load round-trip relies
    # on this ordering).
    pinned_recon_shape = optional_params.pop('recon_shape', None)
    # Apply the structural/optional params WITH name validation, so a typo'd key still raises; then
    # apply the regularization knobs with no_warning to suppress the "directly setting regularization"
    # advisory (this is a faithful rebuild, not a user hand-setting sigma_x).
    if optional_params:
        model.set_params(**optional_params)
    if regularization:
        model.set_params(no_warning=True, **regularization)
    model.auto_set_recon_geometry()
    if pinned_recon_shape is not None:
        model.set_params(no_warning=True, recon_shape=pinned_recon_shape)
    return model


def copy_ct_model(ct_model, new_angles=None, new_helical_z_shifts=None, new_num_det_rows=None, new_num_det_cols=None):
    """
    Create a TomographyModel with the same type and parameters as the given ct_model except with the new input angles
    and a corresponding sinogram shape.  Restricted to ParallelBeam and ConeBeam models.

    Args:
        ct_model (TomographyModel): The model to copy.
        new_angles (ndarray of float, optional): 1D vector of projection angles in radians.
            If None, then use the angles in ct_model. Defaults to None.
        new_helical_z_shifts (ndarray of float, optional): 1D vector of per-view axial shifts in ALU for ConeBeamModel.
            Defaults to None.
        new_num_det_rows (int, optional): Number of detector rows in the new model.
            If None, then use the num_det_rows in ct_model. Defaults to None.
        new_num_det_cols (int, optional): Number of detector columns in the new model.
            If None, then use the num_det_cols in ct_model. Defaults to None.

    Returns:
        An instance of ConeBeamModel or ParallelBeam model
    """
    if str(type(ct_model)).find('ConeBeamModel') > 0:
        is_cone = True
    elif str(type(ct_model)).find('ParallelBeamModel') > 0:
        is_cone = False
    else:
        raise TypeError('copy_ct_model() is restricted to ConeBeam and ParallelBeam Models')

    # get_all_params is the single source of truth for reading the params back out: it gives the
    # constructor args with the view components already unpacked (angles + helical_z_shifts for cone)
    # and geometry_type in required, so build_model can reconstruct the class.
    required, optional, regularization = ct_model.get_all_params()

    old_angles = required['angles']
    new_shape = list(required['sinogram_shape'])

    if is_cone:
        old_helical_z_shifts = required['helical_z_shifts']
        if new_angles is None and new_helical_z_shifts is None:
            new_helical_z_shifts = old_helical_z_shifts
        elif new_angles is not None and new_helical_z_shifts is None:
            if np.any(np.asarray(old_helical_z_shifts) != 0):
                raise ValueError('copy_ct_model: new_helical_z_shifts must be specified when changing angles for a helical scan.')
            new_helical_z_shifts = np.zeros_like(new_angles)
        elif new_angles is not None and new_helical_z_shifts is not None:
            if len(new_angles) != len(new_helical_z_shifts):
                raise ValueError('copy_ct_model: new_angles and new_helical_z_shifts must have the same length.')
        elif new_angles is None and new_helical_z_shifts is not None:
            if len(old_helical_z_shifts) != len(new_helical_z_shifts):
                raise ValueError('copy_ct_model: new_helical_z_shifts must have the same length as the existing angles.')
        required['helical_z_shifts'] = new_helical_z_shifts

    if new_angles is None:
        new_angles = old_angles
    new_shape[0] = len(new_angles)
    if new_num_det_rows is not None:
        new_shape[1] = new_num_det_rows
    if new_num_det_cols is not None:
        new_shape[2] = new_num_det_cols
    required['angles'] = new_angles
    required['sinogram_shape'] = tuple(new_shape)

    # The sinogram shape changed, so drop recon_shape and let build_model's auto pass recompute it.
    optional.pop('recon_shape', None)
    return build_model(required, optional, regularization)


def construct_time_frame_models(model, frames_per_rotation=6, frame_overlap_factor=2.0):
    """
    Split a full rotation into overlapping fixed-size time frames and build one model per frame.

    This is the model-only primitive: it needs no data, so it can be used to set up a 4D
    reconstruction or to forward project a time-varying phantom frame by frame before any
    sinogram exists.  Use :func:`construct_time_frames` to split a sinogram at the same time.

    The number of views per frame and the stride between frames are derived from the model's
    angle spacing, so they stay correct under view subsampling.  Trailing views that cannot
    form a full frame are discarded.

    Args:
        model (mbirjax.TomographyModel): Fully-built ConeBeamModel or ParallelBeamModel for the
            full scan.
        frames_per_rotation (int, optional): Number of time frames per full 360 degree rotation.
            Defaults to 6 (one frame every 60 degrees).
        frame_overlap_factor (float, optional): Number of frames that share any given view.  Each
            frame spans frame_overlap_factor * (360 / frames_per_rotation) degrees.  Defaults to
            2.0 (each frame spans 120 degrees).

    Returns:
        (model_frames, view_slices): one model and one view slice per time frame.
            - model_frames (list of mbirjax.TomographyModel): per-frame models built with
              :func:`copy_ct_model` on the frame's angles.
            - view_slices (list of slice): the views of the full sinogram belonging to each frame.

    Raises:
        ValueError: If the model angles have zero spacing, if the frame parameters give a frame
            span or stride smaller than one view, or if a frame would be longer than the scan.

    Example:
        >>> model_frames, view_slices = mj.construct_time_frame_models(ct_model)
        >>> sino_frames = [sinogram[view_slice] for view_slice in view_slices]
    """
    # Internal angular quantities in radians.
    angle_stride = 2.0 * np.pi / frames_per_rotation
    angle_span_per_frame = frame_overlap_factor * angle_stride

    required_params, _, _ = model.get_all_params()
    angles = required_params['angles']
    num_views = len(angles)

    # Angle step in radians from the model's actual view spacing.
    angle_step = float(np.median(np.abs(np.diff(angles))))
    if angle_step <= 0:
        raise ValueError('Model angles must have nonzero spacing.')
    views_per_frame = int(round(angle_span_per_frame / angle_step))
    stride = int(round(angle_stride / angle_step))

    if views_per_frame <= 0:
        raise ValueError('frame_overlap_factor gives a frame span smaller than one view.')
    if stride <= 0:
        raise ValueError('frames_per_rotation gives a stride smaller than one view.')
    if views_per_frame > num_views:
        raise ValueError('frame span cannot exceed the full scan.')

    # Each copy re-derives the recon geometry, which reports the axial padding at verbose >= 1.
    # Only the angles differ between frames, so that report is identical every time; print it
    # once for the first frame and build the rest quietly rather than repeating it nt times.
    # The source model's verbosity is restored before returning, and each frame gets it too.
    verbose = model.get_params('verbose')
    model_frames = []
    view_slices = []
    try:
        for start in range(0, num_views - views_per_frame + 1, stride):
            view_slice = slice(start, start + views_per_frame)
            view_slices.append(view_slice)
            frame = copy_ct_model(model, new_angles=angles[view_slice])
            model.set_params(no_warning=True, verbose=0)
            frame.set_params(no_warning=True, verbose=verbose)
            model_frames.append(frame)
    finally:
        model.set_params(no_warning=True, verbose=verbose)
    return model_frames, view_slices


def construct_time_frames(sinogram, model, frames_per_rotation=6, frame_overlap_factor=2.0):
    """
    Split a full sinogram into overlapping fixed-size time frames, with one model per frame.

    Frames are defined by :func:`construct_time_frame_models`; this wrapper additionally slices
    the sinogram.  The sinogram frames are views into ``sinogram``, so no data is copied.

    Args:
        sinogram (ndarray): Full sinogram, shape (num_views, num_det_rows, num_det_channels).
        model (mbirjax.TomographyModel): Fully-built ConeBeamModel or ParallelBeamModel for the
            full scan.
        frames_per_rotation (int, optional): Number of time frames per full 360 degree rotation.
            Defaults to 6 (one frame every 60 degrees).
        frame_overlap_factor (float, optional): Number of frames that share any given view.  Each
            frame spans frame_overlap_factor * (360 / frames_per_rotation) degrees.  Defaults to
            2.0 (each frame spans 120 degrees).

    Returns:
        (sino_frames, model_frames): one sinogram and one model per time frame.
            - sino_frames (list of ndarray): per-frame sinograms, each a view into ``sinogram``.
              Trailing views that cannot form a full frame are discarded.
            - model_frames (list of mbirjax.TomographyModel): per-frame models built with
              :func:`copy_ct_model`.

    Example:
        >>> sino_frames, model_frames = mj.construct_time_frames(sinogram, ct_model)
        >>> recon_frame_0, _ = model_frames[0].recon(sino_frames[0])
    """
    model_frames, view_slices = construct_time_frame_models(
        model, frames_per_rotation=frames_per_rotation,
        frame_overlap_factor=frame_overlap_factor)
    sino_frames = [sinogram[view_slice] for view_slice in view_slices]
    return sino_frames, model_frames


def calc_tct_recon_params(source_det_dist, source_iso_dist, delta_det_row, delta_det_channel, sinogram_shape, translation_vectors, voxel_row_aspect=1.0, voxel_slice_aspect=1.0):
    """
    Calculate the translation geometry parameters: recon_shape, delta_voxel, voxel_row_aspect

    Args:
        source_det_dist (float): distance from the X-ray source to the detector (in ALU)
        source_iso_dist (float): distance from the X-ray source to the isocenter (in ALU)
        delta_det_row (float): the spacing between detector rows (in ALU)
        delta_det_channel (float): the spacing between detector channels (in ALU)
        sinogram_shape (tuple): Shape of the sinogram as (num_views, num_det_rows, num_det_channels)
        translation_vectors (numpy array): A (num_views, 3) array of translations (x, y, z) in ALU
        voxel_row_aspect (float): the aspect ratio between delta_voxel_row and delta_voxel. Defaults to 1.0
        voxel_slice_aspect (float): the aspect ratio between delta_voxel_slice and delta_voxel. Defaults to 1.0

    Returns:
        recon_shape (tuple): Shape of the reconstruction shape as (num_recon_rows, num_recon_cols, num_recon_slices)
        delta_voxel (float): the voxel pitch at isocenter (in ALU)
        voxel_row_aspect (float): the aspect ratio between delta_voxel_row and delta_voxel
    """
    # Get parameters
    num_views, num_det_rows, num_det_channels = sinogram_shape

    # Calculate magnification
    magnification = source_det_dist / source_iso_dist

    # Calculate the width and height of the detector in ALU
    detect_box = jnp.array([delta_det_channel * num_det_channels, delta_det_row * num_det_rows])

    # Compute avg_view_slope = tan(cone_angle/2) along the x and z directions
    # This is the average slope of a view that a pixel at iso sees.
    # detect_box/4 = distance from the (center of the detector) to (halfway to the edge of the detector).
    # Using the average seems to be better than using the maximum.
    avg_view_slope = (detect_box / 4) / source_det_dist

    # Compute detector pixel pitch at iso
    # Note that this may differ from delta_voxel
    # However, we will use det_pixel_pitch_iso to calculate both the number rows and their pitch
    det_pixel_pitch_iso_vec = jnp.array([delta_det_row, delta_det_channel]) / magnification
    det_pixel_pitch_iso = jnp.max(det_pixel_pitch_iso_vec)

    # Set delta_voxel
    delta_voxel = float(det_pixel_pitch_iso)

    # Compute delta_voxel in slice dimension
    delta_voxel_slice = voxel_slice_aspect * delta_voxel

    ######### Compute the row pitch based on a heuristic #########
    # ToDo: There will be problems if avg_view_slope is small or zero. Discuss with Greg.
    # The following code will result in an isotropic voxel when the avg_view_slope > 76 deg.
    nominal_row_pitch = 4.0 * det_pixel_pitch_iso_vec / avg_view_slope
    nominal_row_pitch = jnp.max(nominal_row_pitch)  # Take the maximum of the nominal pitches along x and z
    delta_recon_row = jnp.maximum(nominal_row_pitch, det_pixel_pitch_iso)  # Ensure that the row resolution is not higher than the (x,z) detector resolution
    delta_recon_row = float(delta_recon_row)

    ##### Compute voxel row aspect
    # In translation geometry, anisotropic row spacing is usually needed for good reconstruction results.
    #
    # If voxel_row_aspect == 1.0 (default value), assume the user did not explicitly specify
    # a row aspect ratio, and automatically compute it using the current TCT row-pitch heuristic.
    #
    # Otherwise, use the user-defined voxel_row_aspect to determine delta_recon_row.
    if voxel_row_aspect == 1.0:
        voxel_row_aspect = delta_recon_row / delta_voxel
    else:
        delta_recon_row = voxel_row_aspect * delta_voxel

    # Compute cube = (width, depth, height) of the scanned region in ALU
    max_translation = jnp.amax(translation_vectors, axis=0)  # Translate object right/up when positive
    min_translation = jnp.amin(translation_vectors, axis=0)  # Translate object left/down when negative
    cube = max_translation - min_translation

    # Compute recon_box = (num_recon_cols, num_recon_slices) of the reconstruction volume.
    # The reconstruction box size is determined using:
    #   delta_voxel for the column direction
    #   delta_voxel_slice for the slice direction
    recon_box = jnp.ceil(jnp.array([cube[0], cube[2]]) / jnp.array([delta_voxel, delta_voxel_slice]))

    # ************ Use a heuristic to determine a reasonable number of rows *************
    # Compute the number of unknown pixels per view
    num_pixels_per_view = ((recon_box[0] + num_det_rows) * (recon_box[1] + num_det_channels)) / num_views
    num_measurements_per_view = num_det_channels * num_det_rows
    # Select the number of rows so that (number of unknowns) = 2*(the number of measurements)
    num_recon_rows = 2*jnp.ceil(num_measurements_per_view / num_pixels_per_view)

    # Make sure the object extends no further than halfway to the source
    max_recon_rows = jnp.floor((source_iso_dist - cube[1]) / delta_recon_row)
    if max_recon_rows < 1:
        print(f"[Error] Computed max_recon_rows = {max_recon_rows} < 1. This suggests the object extends beyond the source.")
    num_recon_rows = jnp.minimum(num_recon_rows, max_recon_rows)

    # Set the parameters to their computed values
    num_recon_cols, num_recon_slices = recon_box
    num_recon_cols = int(num_recon_cols)
    num_recon_rows = int(num_recon_rows)
    num_recon_slices = int(num_recon_slices)
    recon_shape = (num_recon_rows, num_recon_cols, num_recon_slices)

    return recon_shape, delta_voxel, voxel_row_aspect


def estimate_background_cluster_boundaries(sinogram):
    """
    Estimate background cluster left and right boundaries from the sinogram histogram.
    This function assumes that the background takes on values near zero.

    Args:
        sinogram (ndarray): 3D jax array containing sinogram with shape (num_views, num_det_rows, num_det_channels).

    Returns:
        left_boundary (float): value of the left boundary to the background cluster
        right_boundary (float): value of the right boundary to the background cluster
    """
    # Compute histogram of sinogram values
    hist, edges = np.histogram(sinogram.ravel(), bins=400)

    centers = 0.5 * (edges[:-1] + edges[1:])

    # Find all local peaks in the histogram
    peak_indices = []

    if len(hist) > 1 and hist[0] > hist[1]:
        peak_indices.append(0)

    for i in range(1, len(hist) - 1):
        if hist[i] >= hist[i - 1] and hist[i] > hist[i + 1]:
            peak_indices.append(i)

    # Choose the peak closest to intensity 0 (background peak)
    if len(peak_indices) == 0:
        peak_idx = int(np.argmin(np.abs(centers - 0.0)))
    else:
        peak_idx = min(peak_indices, key=lambda i: abs(centers[i] - 0.0))

    # Define background width cutoff level (10% of peak height)
    peak_height = hist[peak_idx]
    cutoff = 0.1 * peak_height

    # Find left boundary of background cluster
    left_boundary_idx = peak_idx
    while left_boundary_idx > 0 and hist[left_boundary_idx] > cutoff:
        left_boundary_idx -= 1

    # Find right boundary of background cluster
    right_boundary_idx = peak_idx
    while right_boundary_idx < len(hist) - 1 and hist[right_boundary_idx] > cutoff:
        right_boundary_idx += 1

    # Compute background cluster width = right_boundary - left_boundary
    left_boundary = centers[left_boundary_idx]
    right_boundary = centers[right_boundary_idx]

    return left_boundary, right_boundary