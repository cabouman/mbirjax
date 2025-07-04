import os, sys

# === Core scientific/plotting libraries ===
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

# === Project-specific imports ===
import mbirjax as mj
import urllib.request
import urllib.error
import tarfile
from urllib.parse import urlparse
import shutil
import h5py
import re
import gdown
import warnings
from ruamel.yaml import YAML


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
        >>> recon, recon_dict = mbirjax.utilities.load_data_hdf5("output/recon_volume.h5")
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
        >>> mbirjax.utilities.save_data_hdf5(file_path, recon, recon_info)
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
            is_download = False
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
                    raise RuntimeError('URLError raised! Check internet connection.')
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



def export_recon_hdf5(file_path, recon, recon_dict=None, remove_flash=True, radial_margin=10, top_margin=10, bottom_margin=10):
    """
    Export a 3D reconstruction volume to an HDF5 file with optional post-processing.

    This function transposes the input volume to right-hand coordinates (slice, row, col),
    optionally applies a cylindrical mask to remove peripheral and top/bottom slices (referred to as `flash`),
    and writes the volume and optional metadata to an HDF5 file.

    Args:
        file_path (str): Full path to the output HDF5 file. Parent directories will be created if they do not exist.
        recon (Union[np.ndarray, jax.Array]): 3D volume in (row, col, slice) order. Will be converted to NumPy before writing.
        recon_dict (dict, optional): Dictionary of attributes to store as metadata in the dataset.
        remove_flash (bool, optional): Whether to apply a cylindrical mask to remove peripheral and top/bottom slices. Defaults to True.
        radial_margin (int, optional): Margin in pixels to subtract from the cylinder radius. Defaults to 10.
        top_margin (int, optional): Number of top slices to set to zero along the Z-axis. Defaults to 10.
        bottom_margin (int, optional): Number of bottom slices to set to zero along the Z-axis. Defaults to 10.

    Example:
        >>> from mbirjax.utilities import export_recon_hdf5
        >>> import jax.numpy as jnp
        >>> recon = jnp.ones((128, 128, 64))  # (row, col, slice) order
        >>> export_recon_hdf5("output/recon_volume.h5", recon, recon_dict={"scan_id": "sample1"})
    """

    recon = jnp.asarray(recon)

    if remove_flash:
        recon = mj.preprocess.apply_cylindrical_mask(recon, radial_margin, top_margin, bottom_margin)

    recon = jnp.transpose(recon, (2, 0, 1))
    # Flip the slice axis to convert to right-handed coordinate system
    recon = recon[::-1, :, :]
    recon = np.array(recon)

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

