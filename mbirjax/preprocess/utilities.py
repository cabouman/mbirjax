import numpy as np
import warnings
import math
import scipy
import tifffile
import glob
import os
import h5py
import jax.numpy as jnp
import jax
import dm_pix
import tqdm


def compute_sino_transmission(obj_scan, blank_scan, dark_scan, defective_pixel_array=(), batch_size=90):
    """
    Compute sinogram from object, blank, and dark scans.

    This function computes sinogram by taking the negative log of the attenuation estimate.
    It can also take in a list of defective pixels and correct those pixel values.
    The invalid sinogram entries are the union of defective pixel entries and sinogram entries with values of inf or Nan.

    Args:
        obj_scan (ndarray, float): 3D object scan with shape (num_views, num_det_rows, num_det_channels).
        blank_scan (ndarray, float): [Default=None] 3D blank scan with shape (num_blank_scans, num_det_rows, num_det_channels). When num_blank_scans>1, the pixel-wise mean will be used as the blank scan.
        dark_scan (ndarray, float): [Default=None] 3D dark scan with shape (num_dark_scans, num_det_rows, num_det_channels). When num_dark_scans>1, the pixel-wise mean will be used as the dark scan.
        defective_pixel_array (optional, ndarray): A list of tuples containing indices of invalid sinogram pixels, with the format (view_idx, row_idx, channel_idx) or (detector_row_idx, detector_channel_idx).
            If None, then the invalid pixels will be identified as sino entries with inf or Nan values.
        batch_size (int): Size of view batch to use in passing data to gpu.

    Returns:
        - **sino** (*ndarray, float*): Sinogram data with shape (num_views, num_det_rows, num_det_channels).
    """
    # Compute mean for blank and dark scans and move them to GPU if available
    blank_scan_mean = jnp.array(np.mean(blank_scan, axis=0, keepdims=True))
    dark_scan_mean = jnp.array(np.mean(dark_scan, axis=0, keepdims=True))

    sino_batches_list = []  # Initialize a list to store sinogram batches

    num_views = obj_scan.shape[0]  # Total number of views

    num_defective_pixels = len(defective_pixel_array)
    if num_defective_pixels > 0:
        defective_pixel_array_jax = jnp.array(defective_pixel_array)
        flat_indices = jnp.ravel_multi_index(defective_pixel_array_jax.T, obj_scan.shape[1:])
    else:
        defective_pixel_array_jax = ()
        flat_indices = None

    # Process obj_scan in batches
    for i in tqdm.tqdm(range(0, num_views, batch_size)):

        obj_scan_batch = obj_scan[i:min(i + batch_size, num_views)]
        obj_scan_batch = jnp.array(obj_scan_batch)

        obj_scan_batch = jnp.abs(obj_scan_batch - dark_scan_mean)
        blank_scan_batch = jnp.abs(blank_scan_mean - dark_scan_mean)

        # We use jnp.nan here because we'll later use np.nanmedian to get rid of nans and other defective pixels.
        sino_batch = -jnp.log(jnp.where(obj_scan_batch / blank_scan_batch > 0, obj_scan_batch / blank_scan_batch, jnp.nan))

        # We'll set all defective pixels to NaN to be able to use jnp.nanmedian in interpolate_defective_pixels
        num_defective_pixels = len(defective_pixel_array)
        if num_defective_pixels > 0:
            # For obj_scan, we need to set the bad pixels at every view to 0, so we can't use put directly.
            sino_batch = put_in_slice(sino_batch, flat_indices, jnp.nan)

        sino_batch = interpolate_defective_pixels(sino_batch, defective_pixel_array_jax)
        sino_batches_list.append(np.array(sino_batch))

    sino = np.concatenate(sino_batches_list, axis=0)
    print("Sinogram computation complete.")

    return sino


def interpolate_defective_pixels(sino, defective_pixel_array=()):
    """
    Interpolates defective sinogram entries with the mean of neighboring pixels.

    Args:
        sino (jax array, float): Sinogram data with 3D shape (num_views, num_det_rows, num_det_channels).
        defective_pixel_array (jax array): A list of tuples containing indices of invalid sinogram pixels, with the format (detector_row_idx, detector_channel_idx) or (view_idx, detector_row_idx, detector_channel_idx).

    Returns:
        2-element tuple containing:
        - **sino** (*jax array, float*): Corrected sinogram data with shape (num_views, num_det_rows, num_det_channels).
        - **defective_pixel_list** (*list(tuple)*): Updated defective_pixel_list with the format (detector_row_idx, detector_channel_idx) or (view_idx, detector_row_idx, detector_channel_idx).
    """

    sino_shape = sino.shape
    sino = jnp.nan_to_num(sino, copy=False, nan=jnp.nan, posinf=jnp.nan, neginf=jnp.nan)

    # First handle the defective_pixel_array, each of which goes across all views
    # For each nan entry, we take the 3x3 neighbors in the same view and apply nanmedian.
    sino = sino.reshape((sino_shape[0], -1))
    neighbor_radius = 1

    # Generate all i_offset and j_offset combinations.  num_nbrs_1d = 2 * neighbor_radius + 1
    offsets = jnp.arange(-neighbor_radius, neighbor_radius + 1)
    i_offsets, j_offsets = jnp.meshgrid(offsets, offsets, indexing='ij')  # Shape (num_nbrs_1d, num_nbrs_1d)
    i_offsets = i_offsets.ravel()  # (num_nbrs_1d^2,)
    j_offsets = j_offsets.ravel()  # (num_nbrs_1d^2,)
    offsets_expanded = jnp.stack((i_offsets, j_offsets), axis=1)[None, :, :]  # (1, num_nbrs_1d^2, 2)

    if len(defective_pixel_array) > 0:
        # Broadcast defective_pixel_array to match offset shapes
        defective_pixel_expanded = defective_pixel_array[:, None, :]  # (num_defective_pixels, 1, 2)

        # Add the indices and offsets to get valid neighbors, then convert to indices.  We use clip mode
        # for raveling to stay in bounds.  If a neighbor is out of bounds, clipping will yield a neighbor.
        neighbor_coords = defective_pixel_expanded + offsets_expanded  # (num_defective_pixels, num_nbrs_1d^2, 2)
        flat_indices = jnp.ravel_multi_index(neighbor_coords.transpose(2, 0, 1),
                                             sino_shape[1:], mode='clip')  # (num_defective_pixels, num_nbrs_1d^2)

        # Gather neighbor values for all views and use nanmedian to replace the values in sino
        neighbor_values_flat = sino[:, flat_indices]  # (num_views, num_defective_pixels, num_nbrs_1d^2)
        median_values = jnp.nanmedian(neighbor_values_flat, axis=2)
        flat_indices = jnp.ravel_multi_index(defective_pixel_array.T, sino_shape[1:])
        sino = put_in_slice(sino, flat_indices, median_values)

    # Repeat on individual nans until there are no more.  Each index now has 3 components.
    sino = sino.reshape(sino_shape)
    nan_indices = jnp.argwhere(jnp.isnan(sino))
    offsets_expanded = jnp.stack((0*i_offsets, i_offsets, j_offsets), axis=1)[None, :, :]  # (1, num_nbrs_1d^2, 3)
    num_nans = nan_indices.shape[0]
    while num_nans > 0:
        sino = sino.flatten()

        nan_inds_expanded = nan_indices[:, None, :]  # (num_nans, 1, 3)
        neighbor_coords = nan_inds_expanded + offsets_expanded  # (num_nans, num_nbrs_1d^2, 2)
        flat_indices = jnp.ravel_multi_index(neighbor_coords.transpose(2, 0, 1),
                                             sino_shape, mode='clip')  # (num_nans, num_nbrs_1d^2)

        # Gather neighbor values for all views and replace the values, ignoring any nan values
        neighbor_values_flat = sino[flat_indices]  # (num_nans, num_nbrs_1d^2)
        median_values = jnp.nanmedian(neighbor_values_flat, axis=1)
        flat_indices = jnp.ravel_multi_index(nan_indices.T, sino_shape)
        sino = sino.at[flat_indices].set(median_values)

        sino = sino.reshape(sino_shape)
        nan_indices = jnp.argwhere(np.isnan(sino))
        new_num_nans = nan_indices.shape[0]
        if new_num_nans >= num_nans:
            raise ValueError('Unable to remove all defective pixels from sinogram.')
        else:
            num_nans = new_num_nans

    return sino


def correct_det_rotation(sino, weights=None, det_rotation=0.0):
    """
    Correct sinogram data and weights to account for detector rotation.

    This function can be used to rotate sinogram views when the axis of rotation is not exactly aligned with the detector columns.

    Args:
        sino (float, ndarray): Sinogram data with 3D shape (num_views, num_det_rows, num_det_channels).
        weights (float, ndarray): Sinogram weights, with the same array shape as ``sino``.
        det_rotation (optional, float): tilt angle between the rotation axis and the detector columns in unit of radians.

    Returns:
        - A numpy array containing the corrected sinogram data if weights is None.
        - A tuple (sino, weights) if weights is not None
    """
    sino = scipy.ndimage.rotate(sino, np.rad2deg(det_rotation), axes=(1,2), reshape=False, order=3)
    # weights not provided
    if weights is None:
        return sino
    # weights provided
    print("correct_det_rotation: weights provided by the user. Please note that zero weight entries might become non-zero after tilt angle correction.")
    weights = scipy.ndimage.rotate(weights, np.rad2deg(det_rotation), axes=(1,2), reshape=False, order=3)
    return sino, weights


def correct_det_rotation_and_background(sino, det_rotation=0.0, background_offset=0.0, batch_size=30):
    """
    Correct sinogram data to account for detector rotation, using JAX for batch processing and GPU acceleration.
    Weights are not modified.

    Args:
        sino (numpy.ndarray): Sinogram data with 3D shape (num_views, num_det_rows, num_det_channels).
        det_rotation (optional, float): tilt angle between the rotation axis and the detector columns in radians.
        background_offset (optional, float): background offset subtracted from sinogram data before correction.
        batch_size (int): Number of views to process in each batch to avoid memory overload.

    Returns:
        - A numpy.ndarray containing the corrected sinogram data if weights is None.
        - A tuple (sino_corrected, weights) if weights is not None.
    """

    num_views = sino.shape[0]  # Total number of views
    sino_batches_list = []  # Initialize a list to store sinogram batches

    # Process in batches with looping and progress printing
    for i in tqdm.tqdm(range(0, num_views, batch_size)):

        # Get the current batch (from i to i + batch_size)
        sino_batch = jnp.array(sino[i:min(i + batch_size, num_views)]) - background_offset

        # Apply the rotation on this batch
        sino_batch = sino_batch.transpose(1, 2, 0)
        sino_batch = dm_pix.rotate(sino_batch, det_rotation, order=1, mode='constant', cval=0.0)
        sino_batch = sino_batch.transpose(2, 0, 1)

        # Append the rotated batch to the list
        sino_batches_list.append(np.array(sino_batch))

    sino_rotated = np.concatenate(sino_batches_list, axis=0)

    return sino_rotated


def estimate_background_offset(sino, edge_width=9):
    """
    Estimate background offset of a sinogram using JAX for GPU acceleration.

    Args:
        sino (numpy.ndarray): Sinogram data with shape (num_views, num_det_rows, num_det_channels).
        edge_width (int, optional): Width of the edge regions in pixels. Must be an integer >= 1.  Defaults to 9.

    Returns:
        offset (float): Background offset value.
    """

    if edge_width < 1:
        edge_width = 1
        warnings.warn("edge_width of background regions should be >= 1! Setting edge_width to 1.")

    _, _, num_det_channels = sino.shape

    # Extract edge regions from the sinogram (top, left, right)
    sino_edge_left = sino[:, :, :edge_width].flatten()
    sino_edge_right = sino[:, :, num_det_channels-edge_width:].flatten()
    sino_edge_top = sino[:, :edge_width, :].flatten()
    offset = np.median(np.concatenate((sino_edge_left, sino_edge_right, sino_edge_top)))

    return offset



# ####### subroutines for image cropping and down-sampling
def downsample_scans(obj_scan, blank_scan, dark_scan,
                     downsample_factor, defective_pixel_array=(), batch_size=90):
    """Performs Down-sampling to the scan images in the detector plane.

    Args:
        obj_scan (numpy array of floats): A stack of sinograms. 3D numpy array, (num_views, num_det_rows, num_det_channels).
        blank_scan (numpy array of floats): A blank scan. 2D numpy array, (num_det_rows, num_det_channels).
        dark_scan (numpy array of floats): A dark scan. 3D numpy array, (num_det_rows, num_det_channels).
        downsample_factor ([int, int]): Default=[1,1]] Two numbers to define down-sample factor.
        defective_pixel_array (ndarray): Array of shape (num_defective_pixels, 2)
        batch_size (int): Number of views to include in one jax batch.

    Returns:
        Downsampled scans
        - **obj_scan** (*ndarray, float*): A stack of sinograms. 3D numpy array, (num_views, num_det_rows, num_det_channels).
        - **blank_scan** (*ndarray, float*): A blank scan. 3D numpy array, (num_det_rows, num_det_channels).
        - **dark_scan** (*ndarray, float*): A dark scan. 3D numpy array, (num_det_rows, num_det_channels).
    """

    assert len(downsample_factor) == 2, 'factor({}) needs to be of len 2'.format(downsample_factor)
    assert (downsample_factor[0] >= 1 and downsample_factor[1] >= 1), 'factor({}) along each dimension should be greater or equal to 1'.format(downsample_factor)

    # Set defective pixels to nan for use with nanmean
    if len(defective_pixel_array) > 0:
        # Set defective pixels to 0
        flat_indices = np.ravel_multi_index(defective_pixel_array.T, blank_scan.shape[1:])
        np.put(blank_scan[0], flat_indices, np.nan)
        np.put(dark_scan[0], flat_indices, np.nan)
    else:
        flat_indices = None

    # crop the scan if the size is not divisible by downsample_factor.
    new_size1 = downsample_factor[0] * (obj_scan.shape[1] // downsample_factor[0])
    new_size2 = downsample_factor[1] * (obj_scan.shape[2] // downsample_factor[1])

    blank_scan = blank_scan[:, 0:new_size1, 0:new_size2]
    dark_scan = dark_scan[:, 0:new_size1, 0:new_size2]

    # Reshape into blocks specified by the downsampling factor and then use nanmean to average over the blocks.
    block_shape = (blank_scan.shape[1] // downsample_factor[0], downsample_factor[0],
                   blank_scan.shape[2] // downsample_factor[1], downsample_factor[1])
    # Take the mean over blocks, ignoring nans.  Any blocks with all nans will yield a nan.
    blank_scan = blank_scan.reshape((blank_scan.shape[0],) + block_shape)
    blank_scan = np.nanmean(blank_scan, axis=(2, 4))
    dark_scan = dark_scan.reshape((dark_scan.shape[0],) + block_shape)
    dark_scan = np.nanmean(dark_scan, axis=(2, 4))

    # For obj_scan, we'll batch over the views.
    num_views = obj_scan.shape[0]  # Total number of views
    obj_scan_list = []  # Initialize a list to store sinogram batches

    # Process in batches using jax with looping and progress printing
    if flat_indices is not None:
        flat_indices = jnp.array(flat_indices)
    for i in tqdm.tqdm(range(0, num_views, batch_size)):        # Get the current batch (from i to i + batch_size)
        obj_scan_batch = jnp.array(obj_scan[i:min(i + batch_size, num_views)])  # Send to the gpu if there is one
        if flat_indices is not None:
            obj_scan_batch = put_in_slice(obj_scan_batch, flat_indices, jnp.nan)
        # Crop and reshape into blocks
        obj_scan_batch = obj_scan_batch[:, 0:new_size1, 0:new_size2]
        obj_scan_batch = obj_scan_batch.reshape((obj_scan_batch.shape[0],) + block_shape)

        # Compute block mean and append this batch to the list back on the cpu
        obj_scan_batch = jnp.nanmean(obj_scan_batch, axis=(2, 4))
        obj_scan_list.append(np.array(obj_scan_batch))

    obj_scan = np.concatenate(obj_scan_list, axis=0)

    # new defective pixel list = {indices of pixels where the downsampling block contains all bad pixels}
    defective_pixel_array = np.argwhere(np.isnan(blank_scan[0]))
    if len(defective_pixel_array) == 0:
        defective_pixel_array = ()

    return obj_scan, blank_scan, dark_scan, defective_pixel_array


def crop_scans(obj_scan, blank_scan, dark_scan, crop_pixels_sides=0, crop_pixels_top=0, crop_pixels_bottom=0,
               defective_pixel_array=()):
    """Crop obj_scan, blank_scan, and dark_scan images by decimal factors, and update defective_pixel_list accordingly.
    Args:
        obj_scan (ndarray): A stack of sinograms. 3D numpy array, (num_views, num_det_rows, num_det_channels).
        blank_scan (ndarray) : A blank scan. 3D numpy array, (1, num_det_rows, num_det_channels).
        dark_scan (ndarray): A dark scan. 3D numpy array, (1, num_det_rows, num_det_channels).
        crop_pixels_sides (int, optional): The number of pixels to crop from each side of the sinogram. Defaults to 0.
        crop_pixels_top (int, optional): The number of pixels to crop from top of the sinogram. Defaults to 0.
        crop_pixels_bottom (int, optional): The number of pixels to crop from bottom of the sinogram. Defaults to 0.

            The scan images will be cropped using the following algorithm:
                obj_scan <- obj_scan[:, crop_pixels_top:-crop_pixels_bottom, crop_pixels_sides:-crop_pixels_sides]

        defective_pixel_array (ndarray):

    Returns:
        Cropped scans
        - **obj_scan** (*ndarray, float*): A stack of sinograms. 3D numpy array, (num_views, num_det_rows, num_det_channels).
        - **blank_scan** (*ndarray, float*): A blank scan. 3D numpy array, (1, num_det_rows, num_det_channels).
        - **dark_scan** (*ndarray, float*): A dark scan. 3D numpy array, (1, num_det_rows, num_det_channels).
    """

    assert (0 <= crop_pixels_sides < obj_scan.shape[2] // 2 and
            0 <= crop_pixels_top and 0 <= crop_pixels_bottom and crop_pixels_top + crop_pixels_bottom < obj_scan.shape[1]), \
        ('crop_pixels should be nonnegative integers so that crop_pixels_top + crop_pixels_bottom < view height and'
         ' 2*crop_pixels_sides < view width')

    Nr_lo = crop_pixels_top
    Nr_hi = obj_scan.shape[1] - crop_pixels_bottom

    Nc_lo = crop_pixels_sides
    Nc_hi = obj_scan.shape[2] - crop_pixels_sides

    obj_scan = obj_scan[:, Nr_lo:Nr_hi, Nc_lo:Nc_hi]
    blank_scan = blank_scan[:, Nr_lo:Nr_hi, Nc_lo:Nc_hi]
    dark_scan = dark_scan[:, Nr_lo:Nr_hi, Nc_lo:Nc_hi]

    # Remove any defective pixels that are outside the new cropped region
    if len(defective_pixel_array) > 0:
        in_bounds = (defective_pixel_array[:, 0] >= Nr_lo) & (defective_pixel_array[:, 0] < Nr_hi) & \
                    (defective_pixel_array[:, 1] >= Nc_lo) & (defective_pixel_array[:, 1] < Nc_hi)
        defective_pixel_array = defective_pixel_array[in_bounds]
        defective_pixel_array -= np.array([Nr_lo, Nc_lo]).reshape(1, 2)

    return obj_scan, blank_scan, dark_scan, defective_pixel_array
# ####### END subroutines for image cropping and down-sampling


# ####### subroutines for loading scan images
def read_scan_img(img_path):
    """Reads a single scan image from an image path. This function is a subroutine to the function `read_scan_dir`.

    Args:
        img_path (string): Path object or file object pointing to an image.
            The image type must be compatible with `PIL.Image.open()`. See `https://pillow.readthedocs.io/en/stable/reference/Image.html` for more details.
    Returns:
        ndarray (float): 2D numpy array. A single scan image.
    """
    img = tifffile.imread(img_path)

    if np.issubdtype(img.dtype, np.integer):
        # make float and normalize integer types
        maxval = np.iinfo(img.dtype).max
        img = img.astype(np.float32) / maxval

    return img.astype(np.float32)


def read_scan_dir(scan_dir, view_ids=None):
    """Reads a stack of scan images from a directory. This function is a subroutine to `load_scans_and_params`.

    Args:
        scan_dir (string): Path to a ConeBeam Scan directory.
            Example: "<absolute_path_to_dataset>/Radiographs"
        view_ids (ndarray of ints, optional, default=None): List of view indices to specify which scans to read.
    Returns:
        ndarray (float): 3D numpy array, (num_views, num_det_rows, num_det_channels). A stack of scan images.
    """

    # Get the files that are views and check that we have as many as we need
    img_path_list = sorted(glob.glob(os.path.join(scan_dir, '*[0-9].tif')))
    # Set the view ids if none given or check that we have enough.  This assumes that all the views are in the
    # directory and are labeled sequentially.
    if view_ids is None:
        view_ids = np.arange(len(img_path_list))
    else:
        max_view_id = np.amax(view_ids)
        if max_view_id >= len(img_path_list):
            raise FileNotFoundError('The max view index was given as {}, but there are only {} views in {}'.format(max_view_id, len(img_path_list), scan_dir))
    img_path_list = [img_path_list[idx] for idx in view_ids]

    output_views = tifffile.imread(img_path_list, ioworkers=48, maxworkers=8)

    # return shape = num_views x num_det_rows x num_det_channels
    return output_views
# ####### END subroutines for loading scan images


def unit_vector(v):
    """ Normalize v. Returns v/||v|| """
    return v / np.linalg.norm(v)


def project_vector_to_vector(u1, u2):
    """ Projects the vector u1 onto the vector u2. Returns the vector <u1|u2>.
    """
    u2 = unit_vector(u2)
    u1_proj = np.dot(u1, u2)*u2
    return u1_proj


# ####### Multi-threshold Otsu's method
def multi_threshold_otsu(image, classes=2):
    """
    Segments an image into several different classes using Otsu's method.

    Parameters
    ----------
    image : ndarray
        Input image in ndarray of float type.
    classes : int, optional
        Number of classes to threshold (i.e., number of resulting regions). Default is 2.

    Returns
    -------
    list
        List of threshold values that divide the image into the specified number of classes.
    """
    if classes < 2:
        raise ValueError("Number of classes must be at least 2")

    # Compute the histogram of the image
    hist, bin_edges = np.histogram(image, bins=256, range=(np.min(image), np.max(image)))

    # Find the optimal thresholds using a recursive approach
    thresholds = _recursive_otsu(hist, classes - 1)

    # Convert histogram bin indices to original image values
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    scaled_thresholds = [bin_centers[t] for t in thresholds]

    return scaled_thresholds


def _recursive_otsu(hist, num_thresholds):
    """
    Recursively applies Otsu's method to find the best thresholds for multiple classes.

    Parameters
    ----------
    hist : ndarray
        Histogram of the image.
    num_thresholds : int
        Number of thresholds to find.

    Returns
    -------
    list
        List of thresholds that divide the histogram into the specified number of classes.
    """
    # Base case: no thresholds needed
    if num_thresholds == 0:
        return []

    # Base case: single threshold needed
    if num_thresholds == 1:
        return [_binary_threshold_otsu(hist)]

    best_thresholds = []
    best_variance = float('inf')

    # Iterate through possible thresholds
    for t in range(1, len(hist) - 1):
        # Split histogram at the threshold
        left_hist = hist[:t]
        right_hist = hist[t:]

        # Recursively find thresholds for left and right segments
        left_thresholds = _recursive_otsu(left_hist, num_thresholds // 2)
        right_thresholds = _recursive_otsu(right_hist, num_thresholds - len(left_thresholds) - 1)

        # Combine thresholds
        thresholds = left_thresholds + [t] + [x + t for x in right_thresholds]

        # Compute the total within-class variance
        total_variance = _compute_within_class_variance(hist, thresholds)

        # Update the best thresholds if the current variance is lower
        if total_variance < best_variance:
            best_variance = total_variance
            best_thresholds = thresholds

    return best_thresholds


def _binary_threshold_otsu(hist):
    """
    Finds the best threshold for binary segmentation using Otsu's method.

    Parameters
    ----------
    hist : ndarray
        Histogram of the image.

    Returns
    -------
    int
        Best threshold for binary segmentation.
    """
    total = np.sum(hist)
    current_max, threshold = 0, 0
    sum_total, sum_foreground, weight_foreground, weight_background = 0, 0, 0, 0

    # Compute the sum of pixel values
    for i in range(len(hist)):
        sum_total += i * hist[i]

    # Iterate through possible thresholds
    for i in range(len(hist)):
        weight_foreground += hist[i]
        if weight_foreground == 0:
            continue
        weight_background = total - weight_foreground
        if weight_background == 0:
            break

        sum_foreground += i * hist[i]
        mean_foreground = sum_foreground / weight_foreground
        mean_background = (sum_total - sum_foreground) / weight_background

        # Compute between-class variance
        between_class_variance = weight_foreground * weight_background * (mean_foreground - mean_background) ** 2
        if between_class_variance > current_max:
            current_max = between_class_variance
            threshold = i

    return threshold


def _compute_within_class_variance(hist, thresholds):
    """
    Computes the total within-class variance given a set of thresholds.

    Parameters
    ----------
    hist : ndarray
        Histogram of the image.
    thresholds : list
        List of thresholds that divide the histogram into multiple classes.

    Returns
    -------
    float
        Total within-class variance.
    """
    total_variance = 0
    thresholds = [0] + thresholds + [len(hist)]

    # Iterate through each segment defined by the thresholds
    for i in range(len(thresholds) - 1):
        class_hist = hist[thresholds[i]:thresholds[i+1]]
        class_prob = np.sum(class_hist)
        if class_prob == 0:
            continue
        class_mean = np.sum(class_hist * np.arange(thresholds[i], thresholds[i+1])) / class_prob
        class_variance = np.sum(((np.arange(thresholds[i], thresholds[i+1]) - class_mean) ** 2) * class_hist) / class_prob
        total_variance += class_variance * class_prob

    return total_variance
# ####### END Multi-threshold Otsu's method


def export_recon_to_hdf5(recon, filename, recon_description="", delta_pixel_image=1.0, alu_description =""):
    """
    This function writes a reconstructed image to an HDF5 file.

    Optimal parameters can be used to store a description of the reconstruction and the pixels spacing.

    Args:
        recon (float, ndarray): 3D reconstructed reconstruction to be saved.
        filename (string): Fully specified path to save the HDF5 file.
        recon_description (string, optional) [Default=""]: Description of CT reconstruction.
        delta_pixel_image (float, optional) [Default=1.0]:  Image pixel spacing in arbitrary length units.
        alu_description (string, optional) [Default=""]: Description of the arbitrary length units for pixel spacing. Example: "1 ALU = 5 mm".
    """
    f = h5py.File(filename, "w")
    # voxel values
    f.create_dataset("recon", data=recon)
    # recon shape
    f.attrs["recon_description"] = recon_description
    f.attrs["alu_description"] = alu_description
    f.attrs["delta_pixel_image"] = delta_pixel_image

    print("Attributes of HDF5 file: ")
    for k in f.attrs.keys():
        print(f"{k}: ", f.attrs[k])

    return


# Normally, this function would be too simple to jit.  However, by using jit, we may be able to
# prevent jax from some extra memory use due to assignemt and/or reshaping.
@jax.jit
def put_in_slice(array, flat_indices, value):
    """
    Similar to numpy.put(array, flat_indices, value), which would produce array.flat[flat_indices] = value.
    However, this function requires that array have an extra leading dimension, and that the value for a given
    index is copied across that dimension.  Roughly, array[:, flat_indices] = value

    Args:
        array (jax array): Numpy array of dimension n+1
        flat_indices (jax array of int): Indices obtained using ravel_multi_index using array.shape[1:]
        value (float or jax array): Values to be copied in.  Must be able to broadcast to array.shape[1:]

    Returns:
        ndarray
    """
    array_shape = array.shape
    array = array.reshape(array_shape[0], -1)
    array = array.at[:, flat_indices].set(value)
    array = array.reshape(array_shape)
    return array
