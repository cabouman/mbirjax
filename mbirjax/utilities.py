import os, sys
from enum import Enum
from typing import Union

import jax
# === Core scientific/plotting libraries ===
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
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
    # Ensure output directory exists
    mj.makedirs(file_path)

    # Open HDF5 file for writing
    with h5py.File(file_path, 'w') as f:
        # Save reconstruction array
        arr = np.array(array)
        volume_data = f.create_dataset(array_name, data=arr)

        # Save reconstruction parameters as attributes
        if isinstance(attributes_dict, dict):
            # Convert subdicts to strings
            attributes_dict = mj.TomographyModel.convert_subdicts_to_strings(attributes_dict)
            for key, value in attributes_dict.items():
                volume_data.attrs[key] = value


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

    This function transposes the input volume to right-hand coordinates (slice, row, col),
    optionally applies a cylindrical mask to remove peripheral and top/bottom slices (referred to as `flash`),
    and writes the volume and optional metadata to an HDF5 file.

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

    @jax.jit
    def process_recon(local_recon):

        if remove_flash:
            local_recon = mj.preprocess.apply_cylindrical_mask(local_recon, radial_margin, top_margin, bottom_margin)

        # Convert to right-handed coordinate system
        local_recon = jnp.transpose(local_recon, (2, 1, 0))
        local_recon = jax.device_get(local_recon)
        return local_recon

    recon = jax.device_get(recon)
    recon = process_recon(recon)
    save_data_hdf5(file_path, recon, 'recon', recon_dict)


def import_recon_hdf5(file_path):
    """
    Import a 3D reconstruction volume from an HDF5 file.

    This function loads a reconstruction volume and associated metadata from an HDF5 file,
    and reorders the volume axes from (slice, row, col) to (row, col, slice) to match
    MBIRJAX conventions.

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

    recon = recon[::-1, :, :]
    recon = np.transpose(recon, axes=(1, 2, 0))

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


def generate_3d_shepp_logan_low_dynamic_range(phantom_shape, device=None):
    """
    Generates a 3D Shepp-Logan phantom with specified dimensions.

    Args:
        phantom_shape (tuple): Phantom shape in (rows, columns, slices).
        device (jax device): Device on which to place the output phantom.

    Returns:
        ndarray: A 3D numpy array of shape phantom_shape representing the voxel intensities of the phantom.

    Note:
        This function uses a memory-efficient approach to generating large phantoms.
    """
    # Get space for the result and set up the grids for add_ellipsoid
    with jax.default_device(device):
        phantom = jnp.zeros(phantom_shape, device=device)
    N, M, P = phantom_shape
    x_locations = jnp.linspace(-1, 1, N)
    y_locations = jnp.linspace(-1, 1, M)
    z_locations = jnp.linspace(-1, 1, P)
    x_grid, y_grid = jnp.meshgrid(x_locations, y_locations, indexing='ij')
    i_grid, j_grid = jnp.meshgrid(jnp.arange(N), jnp.arange(M), indexing='ij')
    grids = (x_grid, y_grid, i_grid, j_grid)

    # Main ellipsoid
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


def gen_translation_phantom(recon_shape, option, text, fill_rate=0.05, font_size=20, text_row_indices=None, horizontal_offset=0, vertical_offset=0):
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

    Returns:
        np.ndarray: Generated phantom volume.
    """
    if option == 'dots':
        return gen_dot_phantom(recon_shape, fill_rate)
    elif option == 'text':
        return gen_text_phantom(recon_shape, text, font_size, text_row_indices, horizontal_offset, vertical_offset)
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
                     vertical_offset=0, font_path="DejaVuSans.ttf"):
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
        font_path (str, optional): Path to the TrueType font file. Default is "DejaVuSans.ttf".

    Returns:
        np.ndarray: A 3D numpy array of shape `recon_shape` containing the text phantom.
    """
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

    array_size = np.minimum(recon_shape[1], recon_shape[2])

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
        word_array = (word_array > 0).astype(np.float32)

        # Crop or pad word_array to fit in the recon volume
        r_start, r_end = r, r + 1
        c_start = c - array_size // 2
        c_end = c_start + array_size
        s_start = s - array_size // 2
        s_end = s_start + array_size

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


def generate_demo_data(
    object_type: Union[ObjectType, str] = ObjectType.SHEPP_LOGAN,
    model_type: Union[ModelType, str] = ModelType.CONE,
    num_views: int = 64,
    num_det_rows: int = 96,
    num_det_channels: int = 128,
    num_x_translations: int = 7,
    num_z_translations: int = 7,
    x_spacing: float = 22,
    z_spacing: float = 22
) -> (np.ndarray, np.ndarray):
    """
    Create a simple object and a sinogram for demonstration purposes.

    This function will create a 3D volume (aka object or phantom) of the specified type, then use the model type and
    parameters to create a simulated sinogram.  The object type 'shepp-logan' gives a simplified version of the
    classic Shepp-Logan test phantom, and type 'cube' gives a simple cube object.

    The output sinogram is a 3D numpy array with shape (num_views, num_det_rows, num_det_channels).  Each 2D array
    sinogram[view_index] is a simulated image from the detector, with num_det_rows indicating the vertical size
    and num_det_channels representing the horizontal size.

    Args:
        object_type (str, optional): One of 'shepp-logan' or 'cube'.  Defaults to 'shepp-logan'.
        model_type (str, optional): One of 'parallel', 'cone', or 'translation'.  Defaults to 'cone'.
        num_views (int, optional):  Number of views in the output sinogram.  Defaults to 64. Ignored when model_type is 'translation'
        num_det_rows (int, optional): Number of rows (vertical) in the output sinogram.  Defaults to 40.
        num_det_channels (int, optional): Number of channels (horizontal) in the output sinogram.  Defaults to 128.
        num_x_translations (int, optional): Number of horizontal translations for translation mode.  Defaults to 7.
        num_z_translations (int, optional): Number of vertical translations for translation mode.  Defaults to 7.
        x_spacing (float, optional): Horizontal spacing between translations in ALU.  Defaults to 22.
        z_spacing (float, optional): Vertical spacing between translations in ALU.  Defaults to 22.

    Returns:
        tuple: (object, sinogram, params)
            - object (np.ndarray): a volume with shape (num_det_channels, num_det_channels, num_det_rows)
            - sinogram (np.ndarray): a sinogram with shape (num_views, num_det_rows, num_det_channels)
            - params (dict): a dict containing 'angles' and, if model_type is 'cone', then also 'source_detector_dist' and 'source_iso_dist'
    """
    # Coerce types to Enum
    object_type = ObjectType(object_type)
    model_type = ModelType(model_type)

    start_angle = -np.pi
    end_angle = np.pi

    # Initialize model

    if model_type == ModelType.PARALLEL:
        sinogram_shape = (num_views, num_det_rows, num_det_channels)
        angles = jnp.linspace(start_angle, end_angle, num_views, endpoint=False)
        ct_model_for_generation = mj.ParallelBeamModel(sinogram_shape, angles)
        params = {'angles': angles}
    elif model_type == ModelType.CONE:
        # For cone beam geometry, we need to describe the distances source to detector and source to rotation axis.
        # np.Inf is an allowable value, in which case this is essentially parallel beam
        source_detector_dist = 4 * num_det_channels
        source_iso_dist = source_detector_dist
        sinogram_shape = (num_views, num_det_rows, num_det_channels)
        angles = jnp.linspace(start_angle, end_angle, num_views, endpoint=False)
        ct_model_for_generation = mj.ConeBeamModel(sinogram_shape, angles, source_detector_dist=source_detector_dist,
                                                   source_iso_dist=source_iso_dist)
        params = {'angles': angles, 'source_detector_dist': source_detector_dist, 'source_iso_dist': source_iso_dist}
    elif model_type == ModelType.TRANSLATION:
        source_iso_dist = np.min(num_det_rows, num_det_channels) / 2
        source_detector_dist = source_iso_dist
        translation_vectors = gen_translation_vectors(num_x_translations, num_z_translations, x_spacing, z_spacing)
        num_views = translation_vectors.shape[0]
        sinogram_shape = (num_views, num_det_rows, num_det_channels)
        ct_model_for_generation = mj.TranslationModel(sinogram_shape, translation_vectors, source_detector_dist=source_detector_dist,
                                                      source_iso_dist=source_iso_dist)
        params = {'translation_vectors': translation_vectors}
    else:
        raise ValueError(f'Invalid model type. Expected one of {[m.value for m in ModelType]}, got {model_type}')

    # Generate phantom
    print('Creating phantom')
    recon_shape = ct_model_for_generation.get_params('recon_shape')
    device = ct_model_for_generation.main_device
    if object_type == ObjectType.SHEPP_LOGAN:
        phantom = generate_3d_shepp_logan_low_dynamic_range(recon_shape, device=device)
    elif object_type == ObjectType.CUBE:
        phantom = gen_cube_phantom(recon_shape, device=device)
    else:
        raise ValueError(f'Invalid object type. Expected one of {[o.value for o in ObjectType]}, got {object_type}')

    # Generate synthetic sinogram data
    print('Creating sinogram')
    sinogram = ct_model_for_generation.forward_project(phantom)
    sinogram = np.asarray(sinogram)

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


def stitch_arrays(array_list, overlap, axis=2):
    """
    Concatenate JAX arrays along one axis while linearly blending a fixed overlap
    between adjacent arrays.

    This behaves like `jnp.concatenate` except that for each adjacent pair, the
    first `overlap_length` elements of the second array and the last
    `overlap_length` elements of the current result are combined by a piece-wise linear cross‑fade.

    All non‑`axis` dimensions must match across inputs.

    Args:
        array_list (list[jax.Array]): Sequence of 2+ JAX arrays to stitch.
        overlap (int): Number of elements to blend between each adjacent pair.
            Must be `>= 1` and not exceed the length of any input along `axis`.
        axis (int, optional): Axis along which to stitch. Defaults to 2.

    Returns:
        jax.Array: Stitched array. Its shape equals the input shape with the
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

    # Create a piecewise linear weight array:
    # 0 for first 25%, linear ramp 0→1 over middle 50%, 1 for final 25%.
    t = jnp.linspace(0, 1, overlap)
    weights = jnp.clip((t - 0.25) / 0.5, 0.0, 1.0)
    weights_shape = np.ones(array_list[0].ndim, dtype=int)
    weights_shape[0] = len(weights)
    weights = weights.reshape(weights_shape)

    # Start with the first array in the list
    stitched = jnp.swapaxes(array_list[0], 0, axis)

    # Iterate through each subsequent array in the list
    for next_array in array_list[1:]:
        # Extract the overlap from the current end of the stitched array and the beginning of the next array
        overlap_current = stitched[-overlap:]
        next_array = jnp.swapaxes(next_array, 0, axis)
        overlap_next = next_array[:overlap]

        # Weighted average for the overlapping part
        weighted_overlap = (1 - weights) * overlap_current + weights * overlap_next

        # Replace the overlap in the stitched array
        stitched = jnp.concatenate([stitched[:-overlap], weighted_overlap], axis=0)

        # Append the non-overlapping remainder of the next array
        stitched = jnp.concatenate([stitched, next_array[overlap:]], axis=0)

    return jnp.swapaxes(stitched, 0, axis)


def get_ct_model(geometry_type, sinogram_shape, angles, source_detector_dist=None, source_iso_dist=None):
    """
    Create an instance of TomographyModel with the given parameters

    Args:
        geometry_type (str): 'parallel' or 'cone'
        sinogram_shape (tuple list of int): (num_views, num_rows, num_channels)
        angles (ndarray of float): 1D vector of projection angles in radians
        source_detector_dist (float or None, optional): Distance in ALU from source to detector.  Defaults to None for geometries that don't need this.
        source_iso_dist (float or None, optional): Distance in ALU from source to iso.  Defaults to None for geometries that don't need this.

    Returns:
        An instance of ConeBeamModel or ParallelBeam model
    """
    if geometry_type == 'cone':
        model = mj.ConeBeamModel(sinogram_shape, angles, source_detector_dist=source_detector_dist,
                                 source_iso_dist=source_iso_dist)
    elif geometry_type == 'parallel':
        model = mj.ParallelBeamModel(sinogram_shape, angles)
    else:
        raise ValueError('Invalid geometry type.  Expected cone or parallel, got {}'.format(geometry_type))

    return model


def copy_ct_model(ct_model, new_angles=None, new_num_det_rows=None, new_num_det_cols=None):
    """
    Create a TomographyModel with the same type and parameters as the given ct_model except with the new input angles
    and a corresponding sinogram shape.  Restricted to ParallelBeam and ConeBeam models.

    Args:
        ct_model (TomographyModel): The model to copy.
        new_angles (ndarray of float, optional): 1D vector of projection angles in radians.  If None, then use the angles in ct_model. Defaults to None.
        new_num_det_rows (int, optional): Number of detector rows in the new model.  If None, then use the num_det_rows in ct_model. Defaults to None.
        new_num_det_cols (int, optional): Number of detector columns in the new model.  If None, then use the num_det_cols in ct_model. Defaults to None.

    Returns:
        An instance of ConeBeamModel or ParallelBeam model
    """
    required_param_names = ct_model.get_required_param_names()
    required_params, other_params = ct_model.get_required_params_from_dict(ct_model.params,
                                                                           required_param_names=required_param_names,
                                                                           values_only=True)

    #  Get the shape of the old sinogram
    new_shape = list(ct_model.get_params('sinogram_shape'))
    try:
        old_angles = ct_model.get_params('angles')
    except NameError as e:
        raise 'copy_ct_model() is restricted to ConeBeam and ParallelBeam Models.'
    if new_angles is None:
        new_angles = old_angles
    new_shape[0] = len(new_angles)

    if new_num_det_rows is not None:
        new_shape[1] = new_num_det_rows

    if new_num_det_cols is not None:
        new_shape[2] = new_num_det_cols

    # Set the new sinogram shape and angles
    required_params['sinogram_shape'] = tuple(new_shape)
    required_params['angles'] = new_angles
    new_model = type(ct_model)(**required_params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        del other_params['recon_shape']  # This should be set automatically by the constructor
        new_model.set_params(**other_params)

    return new_model
