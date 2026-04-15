import numpy as np
import warnings
import glob
import os
import jax.numpy as jnp
import jax
import cv2
import mbirjax as mj

def compute_sino_transmission(obj_scan, blank_scan, dark_scan, defective_pixel_array=(), batch_size=90):
    """
    Compute sinogram from object, blank, and dark scans.

    This function computes a sinogram by taking the negative logarithm of the normalized transmission image:
    `-log((obj - dark) / (blank - dark))`. It supports correction for defective pixels.

    The invalid sinogram entries are defined as:
    - Any values resulting in `inf` or `NaN`
    - Any indices listed in the `defective_pixel_array` (if provided)

    Args:
        obj_scan (ndarray): 
            A 3D object scan of shape (num_views, num_det_rows, num_det_channels).
        blank_scan (ndarray, optional): 
            A 3D blank scan of shape (num_blank_scans, num_det_rows, num_det_channels). 
            If `num_blank_scans > 1`, a pixel-wise mean will be computed.
        dark_scan (ndarray, optional): 
            A 3D dark scan of shape (num_dark_scans, num_det_rows, num_det_channels). 
            If `num_dark_scans > 1`, a pixel-wise mean will be computed.
        defective_pixel_array (ndarray, optional): 
            An array of defective pixel indices. Format can be either 
            (view_idx, row_idx, channel_idx) or (row_idx, channel_idx), if shared across views.
            If `None`, invalid pixels are inferred from `NaN` or `inf` values.
        batch_size (int): 
            Number of views to process in each GPU batch.

    Returns:
        ndarray: 
            The computed sinogram, with shape (num_views, num_det_rows, num_det_channels).
    """    # Compute mean for blank and dark scans and move them to GPU if available
    import tqdm
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



def correct_det_rotation(sino, det_rotation=0.0, batch_size=30):
    """
    Correct sinogram data to account for detector rotation, using JAX for batch processing and GPU acceleration.
    Weights are not modified.

    Args:
        sino (numpy.ndarray): Sinogram data with 3D shape (num_views, num_det_rows, num_det_channels).
        det_rotation (optional, float): tilt angle between the rotation axis and the detector columns in radians.
        batch_size (int): Number of views to process in each batch to avoid memory overload.

    Returns:
        - A numpy.ndarray containing the corrected sinogram data if weights is None.
        - A tuple (sino_corrected, weights) if weights is not None.
    """

    import dm_pix
    import tqdm
    num_views = sino.shape[0]  # Total number of views
    sino_batches_list = []  # Initialize a list to store sinogram batches

    # Process in batches with looping and progress printing
    for i in tqdm.tqdm(range(0, num_views, batch_size)):

        # Get the current batch (from i to i + batch_size)
        sino_batch = jnp.array(sino[i:min(i + batch_size, num_views)])

        # Apply the rotation on this batch
        sino_batch = sino_batch.transpose(1, 2, 0)
        sino_batch = dm_pix.rotate(sino_batch, det_rotation, order=1, mode='constant', cval=0.0)
        sino_batch = sino_batch.transpose(2, 0, 1)

        # Append the rotated batch to the list
        sino_batches_list.append(np.array(sino_batch))

    sino_rotated = np.concatenate(sino_batches_list, axis=0)

    return sino_rotated


def correct_background_offset(sino, edge_width=9, option='global'):
    """
    Correct background offset in a sinogram.

    Args:
        sino (numpy.ndarray): Sinogram data with shape (num_views, num_det_rows, num_det_channels).
        edge_width (int, optional): Width of the edge regions in pixels. Must be an integer >= 1.  Defaults to 9.
        option (str): "global" or "per_view". Defaults to 'global'.

    Returns:
        sino_corrected (numpy.ndarray)
    """

    if edge_width < 1:
        edge_width = 1
        warnings.warn("edge_width of background regions should be >= 1! Setting edge_width to 1.")

    if option == "global":
        _, _, num_det_channels = sino.shape

        # Extract edge regions from the sinogram (top, left, right)
        sino_edge_left  = sino[:, :, :edge_width].flatten()
        sino_edge_right = sino[:, :, num_det_channels-edge_width:].flatten()
        sino_edge_top   = sino[:, :edge_width, :].flatten()

        med_left = np.median(sino_edge_left)
        med_right = np.median(sino_edge_right)
        med_top = np.median(sino_edge_top)

        offset = np.median([med_left, med_right, med_top])

        sino_corrected = sino - offset

    elif option == "per_view":
        num_views, _, num_det_channels = sino.shape

        sino_edge_left  = sino[:, :, :edge_width].reshape(num_views, -1)
        sino_edge_right = sino[:, :, num_det_channels-edge_width:].reshape(num_views, -1)
        sino_edge_top   = sino[:, :edge_width, :].reshape(num_views, -1)

        med_left  = np.median(sino_edge_left, axis=1)
        med_right = np.median(sino_edge_right, axis=1)
        med_top   = np.median(sino_edge_top, axis=1)

        edge_medians = np.stack([med_left, med_right, med_top], axis=1)
        offset = np.median(edge_medians, axis=1)   # (num_views,)

        sino_corrected = sino - offset[:, None, None]

    else:
        raise ValueError("option must be 'global' or 'per_view'")

    return sino_corrected


# ####### subroutines for image cropping and down-sampling
def downsample_view_data(obj_scan, blank_scan, dark_scan, downsample_factor, defective_pixel_array=(), batch_size=90):
    """
    Performs down-sampling of the scan images in the detector plane.
    This is done for the object, blank_scan, and dark_scan data,
    and the defective_pixel_array is updated to reflect the new pixel grid.

    Args:
        obj_scan (ndarray): A stack of sinograms. 3D NumPy array of shape (num_views, num_det_rows, num_det_channels).
        blank_scan (ndarray): Blank scan(s). 3D NumPy array of shape (num_blank_views, num_det_rows, num_det_channels).
        dark_scan (ndarray): Dark scan(s). 3D NumPy array of shape (num_dark_views, num_det_rows, num_det_channels).
        downsample_factor (tuple of int): Two integers defining the down-sample factor. Must be ≥ 1 in each dimension.
        defective_pixel_array (ndarray): Array of shape (num_defective_pixels, 2) indicating defective pixel coordinates.
        batch_size (int): Number of views to include in one JAX batch. Controls memory usage.

    Notes:
        This function supports both singleton blank/dark scans (shape (1, H, W)) and multi-view scans
        (shape (N, H, W), where N > 1). Downsampling is applied independently to each view.

    Returns:
        tuple:
        - **obj_scan** (ndarray): Downsampled object scan. Shape (num_views, new_rows, new_cols).
        - **blank_scan** (ndarray): Downsampled blank scan(s). Shape (num_blank_views, new_rows, new_cols).
        - **dark_scan** (ndarray): Downsampled dark scan(s). Shape (num_dark_views, new_rows, new_cols).
        - **defective_pixel_array** (ndarray): Updated defective pixel coordinates. Shape (N_def, 2).
    """
    import tqdm
    assert len(downsample_factor) == 2, 'factor({}) needs to be of len 2'.format(downsample_factor)
    assert (downsample_factor[0] >= 1 and downsample_factor[1] >= 1), 'factor({}) along each dimension should be greater or equal to 1'.format(downsample_factor)

    # Set defective pixels to nan for use with nanmean
    if len(defective_pixel_array) > 0:
        # Set defective pixels to 0
        flat_indices = np.ravel_multi_index(defective_pixel_array.T, blank_scan.shape[1:])
        for i in range(blank_scan.shape[0]):
            np.put(blank_scan[i], flat_indices, np.nan)
            np.put(dark_scan[i], flat_indices, np.nan)
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
    blank_scan = np.stack([
        np.nanmean(scan.reshape(block_shape), axis=(1, 3))
        for scan in blank_scan
    ], axis=0)

    dark_scan = np.stack([
        np.nanmean(scan.reshape(block_shape), axis=(1, 3))
        for scan in dark_scan
    ], axis=0)

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
    nan_mask = np.isnan(blank_scan).any(axis=0)  # Combine across all views
    defective_pixel_array = np.argwhere(nan_mask)
    if len(defective_pixel_array) == 0:
        defective_pixel_array = ()

    return obj_scan, blank_scan, dark_scan, defective_pixel_array



def crop_view_data(obj_scan, blank_scan, dark_scan, crop_pixels_sides=0, crop_pixels_top=0, crop_pixels_bottom=0, defective_pixel_array=()):
    """
    Crop `obj_scan`, `blank_scan`, and `dark_scan` by the specified pixel amounts and update `defective_pixel_array`.

    The same number of pixels is cropped from the left and right sides (via `crop_pixels_sides`) to
    preserve the detector center/rotation axis. Top and bottom cropping are controlled independently by
    `crop_pixels_top` and `crop_pixels_bottom`. Any defective pixels that fall outside the cropped region
    are removed; remaining coordinates are shifted to the new origin of the cropped images.

    Args:
        obj_scan (np.ndarray):
            Sinogram stack of shape `(num_views, num_det_rows, num_det_channels)`.
        blank_scan (np.ndarray):
            Blank scan(s) of shape `(num_blank_views, num_det_rows, num_det_channels)`.
        dark_scan (np.ndarray):
            Dark scan(s) of shape `(num_dark_views, num_det_rows, num_det_channels)`.
        crop_pixels_sides (int, optional):
            Number of pixels to remove from **each** side (left and right) of the detector channels.
            Defaults to `0`.
        crop_pixels_top (int, optional):
            Number of pixels to remove from the top (small row indices). Defaults to `0`.
        crop_pixels_bottom (int, optional):
            Number of pixels to remove from the bottom (large row indices). Defaults to `0`.
        defective_pixel_array (np.ndarray | tuple, optional):
            Array of shape `(num_defective_pixels, 2)` containing `(row, col)` pixel coordinates that are
            known to be defective **in detector coordinates shared across views**. May be an empty tuple
            `()` if no defects are provided. Defaults to `()`.

    Returns:
        tuple:
            A 4-tuple `(obj_scan, blank_scan, dark_scan, defective_pixel_array)` where

            * **obj_scan** (*np.ndarray*): Cropped object scan of shape `(num_views, new_rows, new_cols)`.
            * **blank_scan** (*np.ndarray*): Cropped blank scan(s) of shape `(num_blank_views, new_rows, new_cols)`.
            * **dark_scan** (*np.ndarray*): Cropped dark scan(s) of shape `(num_dark_views, new_rows, new_cols)`.
            * **defective_pixel_array** (*np.ndarray | tuple*): Updated defective-pixel coordinates in the
              cropped detector grid (shape `(N_def, 2)`), or `()` if no defects remain.

    Raises:
        AssertionError: If any crop amount is negative, or if
            `crop_pixels_top + crop_pixels_bottom >= num_det_rows`, or if
            `2 * crop_pixels_sides >= num_det_channels`.

    Notes:
        This function supports both singleton and multi-view `blank_scan`/`dark_scan`. Cropping is applied
        identically across all views.
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

def _normalize_to_float32(img: np.ndarray) -> np.ndarray:
    """
    Convert image to float32 and normalize if it is an integer dtype.

    - If `imgs.dtype` is an integer type, cast to float32 and divide by the max value for that dtype.
    - Otherwise, cast to float32 without scaling.

    Args:
        img (np.ndarray): Input image array.

    Returns:
        np.ndarray: float32 array, normalized to [0, 1] if input was integer.
    """
    if np.issubdtype(img.dtype, np.integer):
        maxval = np.iinfo(img.dtype).max
        return img.astype(np.float32) / maxval
    return img.astype(np.float32)

def read_scan_img(img_path):
    """
    Reads a scan image from a TIFF file. Supports both 2D and 3D TIFFs.

    This function loads a TIFF image using `tifffile.imread()`, then calls _normalize_to_float32() to normalizes it to float32 format if the
    input is of integer type. If the image has more than two dimensions (e.g., 3D volumes or RGB channels),
    the returned array preserves that shape.

    Args:
        img_path (str): Path to the image file. The file must be readable by `tifffile`.

    Returns:
        np.ndarray: Image data as a float32 NumPy array. Can be 2D or higher dimensional depending on the input.
    """
    import tifffile
    img = tifffile.imread(img_path)
    img = _normalize_to_float32(img)
    return img


def read_scan_dir(scan_dir, view_ids=None):
    """Reads a stack of scan images from a directory. This function is a subroutine to `load_scans_and_params`.

    Args:
        scan_dir (string): Path to a ConeBeam Scan directory.
            Example: "<absolute_path_to_dataset>/Radiographs"
        view_ids (ndarray of ints, optional, default=None): List of view indices to specify which scans to read.
    Returns:
        ndarray (float): 3D numpy array, (num_views, num_det_rows, num_det_channels). A stack of scan images.
    """

    import tifffile
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
    output_views = _normalize_to_float32(output_views)

    # return shape = num_views x num_det_rows x num_det_channels
    return output_views
# ####### END subroutines for loading scan images


def compute_scaling_factor(target_vect: jnp.ndarray, vect_to_scale: jnp.ndarray) -> float:
    """
    Approximate the optimal scalar α that minimizes the squared error ‖target_vect – α vect_to_scale‖².
    This is computed as <target_vect, vect_to_scale> / (<vect_to_scale, vect_to_scale> + epsilon) to
    avoid division by 0, hence is only approximate for vect_to_scale near 0.

    Args:
        target_vect (jnp.ndarray):
            Target reconstruction vector or array of shape (N,) or higher-dimensional.
        vect_to_scale (jnp.ndarray):
            Vector or array of same shape as `target_vect`.

    Returns:
        float:
            Scalar α minimizing ‖target_vect – α vect_to_scale‖².

    Example:
        >>> v = jnp.array([1.0, 2.0, 3.0])
        >>> u = jnp.array([0.5, 1.0, 1.5])
        >>> alpha = compute_scaling_factor(v,u)
    """
    target_vect = jnp.asarray(target_vect)
    vect_to_scale = jnp.asarray(vect_to_scale)

    numerator = jnp.sum(vect_to_scale * target_vect)
    denominator = jnp.sum(vect_to_scale * vect_to_scale)
    epsilon = 1e-8
    return float(numerator / (denominator + epsilon))


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


def unit_vector(v):
    """ Normalize v. Returns v/||v|| """
    return v / np.linalg.norm(v)


def project_vector_to_vector(u1, u2):
    """ Projects the vector u1 onto the vector u2. Returns the vector <u1|u2>.
    """
    u2 = unit_vector(u2)
    u1_proj = np.dot(u1, u2)*u2
    return u1_proj

from functools import partial
@partial(jax.jit, static_argnames=['top_margin', 'bottom_margin'])
def apply_cylindrical_mask(recon, radial_margin=0, top_margin=0, bottom_margin=0):
    """
    Applies a cylindrical mask to a 3D reconstruction volume.

    This function zeros out all voxels outside a centered cylindrical region
    in the (row, col) plane and also zeroes a specified number of slices from
    the top and bottom along the Z-axis (slice axis).

    This function is useful for removing `flash` that typically accumulates on the boundaries of an MBIR reconstruction volume.

    Note:
        This function may need to be converted to batch over slices for very large recons.

    Args:
        recon (jnp.ndarray): 3D volume with shape (num_rows, num_cols, num_slices).
        radial_margin (int): Margin to subtract from the cylinder radius in pixels.
        top_margin (int): Number of top slices to set to zero along the Z-axis.
        bottom_margin (int): Number of bottom slices to set to zero along the Z-axis.

    Returns:
        jnp.ndarray: Masked 3D volume of the same shape as `recon`.

    Example:
        >>> import jax.numpy as jnp
        >>> vol = jnp.ones((128, 128, 64))
        >>> masked_vol = apply_cylindrical_mask(vol,radial_margin=10,top_margin=4,bottom_margin=4)
        >>> masked_vol.shape
        (128, 128, 64)
    """
    num_recon_rows, num_recon_cols, num_slices = recon.shape
    row_center = (num_recon_rows - 1) / 2
    col_center = (num_recon_cols - 1) / 2

    base_radius = max(row_center, col_center)
    radius = base_radius - radial_margin

    # Create circular mask in (row, col) plane
    row_coords, col_coords = jnp.meshgrid(jnp.arange(num_recon_rows), jnp.arange(num_recon_cols), indexing='ij')
    dist_sq = (row_coords - row_center) ** 2 + (col_coords - col_center) ** 2
    circular_mask = (dist_sq <= radius ** 2).astype(recon.dtype)

    # Apply cylindrical mask to all slices
    recon = recon * circular_mask[:, :, None]

    # Apply a mask to the top and bottom margins
    slice_mask = jnp.ones((num_slices, ))
    if top_margin > 0:
        slice_mask = slice_mask.at[:top_margin].set(0)
    if bottom_margin > 0:
        slice_mask = slice_mask.at[-bottom_margin:].set(0)
    recon = recon * slice_mask[None, None, :]

    return recon

def est_crop_width(sino, safety_buffer=20):
    """Estimate crop widths for removing blank margins in a 3D sinogram.

    Args:
        sino (np.ndarray): Input sinogram array .
        safety_buffer (int, optional): Safety buffer (in pixels) to keep around the
            detected object region on each boundary. Defaults to 20.

    Returns:
        crop_top (int): Number of detector rows to crop from the top.
        crop_bottom (int): Number of detector rows to crop from the bottom.
        crop_left (int): Number of detector channels to crop from the left.
        crop_right (int): Number of detector channels to crop from the right.

    """
    sino_indicator_mask = mj.TomographyModel._get_sino_indicator(sino)

    union_mask = np.any(sino_indicator_mask, axis=0)

    rows = np.any(union_mask, axis=1)
    cols = np.any(union_mask, axis=0)

    # argmax of the binary returns the first 1's index
    top_width = np.argmax(rows)
    bottom_width = np.argmax(rows[::-1])
    left_width = np.argmax(cols)
    right_width = np.argmax(cols[::-1])

    # Include a margin to save some empty region on each boundary
    crop_pixels_top = max(top_width - safety_buffer, 0)
    crop_pixels_bottom = max(bottom_width - safety_buffer, 0)
    crop_pixels_left = max(left_width - safety_buffer, 0)
    crop_pixels_right = max(right_width - safety_buffer, 0)

    return crop_pixels_top, crop_pixels_bottom, crop_pixels_left, crop_pixels_right


def auto_crop_sino_conebeam(sino, cone_beam_params, optional_params, safety_buffer=20):
    """
    Automatically crop unused sinogram margins and update cone-beam geometry parameters.

    This reduces the reconstruction volume by removing blank detector margins in the sinogram and
    updating the corresponding geometry offsets so the physical coordinate system remains consistent.

    Args:
        sino (np.ndarray): Input sinogram array with shape (num_views, num_det_rows, num_det_channels).
        cone_beam_params (dict): Cone-beam geometry parameters that can be passed to the model constructor.
        optional_params (dict): Optional geometry parameters set after the model is constructed.
        safety_buffer (int, optional): Safety buffer (in pixels) to keep around the detected object region.
            Defaults to 20.

    Returns:
        tuple:
            A 3-tuple ``(sino, cone_beam_params, optional_params)`` where:

            * **sino** (*np.ndarray*): Cropped sinogram with updated shape.
            * **cone_beam_params** (*dict*): Updated parameters with adjusted ``'sinogram_shape'``.
            * **optional_params** (*dict*): Updated parameters with adjusted ``'det_row_offset'``,
              ``'det_channel_offset'``, and ``'recon_slice_offset'``.
    """
    crop_pixels_top, crop_pixels_bottom, crop_pixels_left, crop_pixels_right = est_crop_width(sino, safety_buffer)

    Nr_lo = crop_pixels_top
    Nr_hi = sino.shape[1] - crop_pixels_bottom

    Nc_lo = crop_pixels_left
    Nc_hi = sino.shape[2] - crop_pixels_right

    sino = sino[:, Nr_lo:Nr_hi, Nc_lo:Nc_hi]

    # Correct geometry parameters det_row_offset and det_channel_offset after cropping
    cone_beam_params['sinogram_shape'] = sino.shape
    delta_det_row, delta_det_channel = optional_params['delta_det_row'], optional_params['delta_det_channel']
    optional_params['det_row_offset'] += (crop_pixels_bottom - crop_pixels_top)/2 * delta_det_row
    optional_params['det_channel_offset'] += (crop_pixels_right - crop_pixels_left)/2 * delta_det_channel

    # Correct geometry parameter recon_slice_offset
    recon_slice_offset = optional_params['recon_slice_offset']
    source_detector_dist = cone_beam_params["source_detector_dist"]
    source_iso_dist = cone_beam_params["source_iso_dist"]
    magnification = source_detector_dist / source_iso_dist

    # Sign convention: positive recon_slice_offset reconstructs below the iso, vice versa
    recon_slice_offset -= (crop_pixels_bottom - crop_pixels_top)/2 * delta_det_row / magnification
    optional_params['recon_slice_offset'] = recon_slice_offset

    return sino, cone_beam_params, optional_params


def estimate_sino_view_offset(ct_model, sino, direct_recon):
    """
    Estimate per-view 2D shifts for a sinogram.

    This function estimate the shifts in three steps:
    1. Forward project the preliminary reconstruction using the CT model.
    2. Apply high-pass filtering to both the sinogram and the
        forward projection of the preliminary reconstruction.
    3. For each view, estimate a 2D shift that aligns the sinogram view
        to the corresponding forward-projected view using an image alignment method from OpenCV

    Args:
        ct_model (mj.TomographyModel): A CT model object that defined the CT geometry.
        sino (numpy array or jax array): 3D sinogram data with shape (num_views, num_det_rows, num_det_channels).
        direct_recon (numpy array or jax array): A preliminary 3D reconstruction of the sinogram.

    Returns:
        estimated_shifts (numpy.array or jax array): A (num_views, 2) array of per-view shift (y, x) in pixels.
            Each shift specified how much the corresponding sinogram slice should be shifted to match forward projection.
            Positive x shifts the view right. Positive y shifts the view down.
    """
    # Verify the input recon shape
    recon_shape = ct_model.get_params('recon_shape')
    if direct_recon.shape != recon_shape:
        raise ValueError("Input recon shape does not match ct_model's recon shape.")

    # Forward project the reconstruction
    sino_from_recon = ct_model.forward_project(direct_recon)

    # Apply a high-pass filter to sinogram and forward projection of the reconstruction
    filtered_sino = sino_high_pass_filtering(sino)
    filtered_sino_from_recon = sino_high_pass_filtering(sino_from_recon)

    # Estimate the shift between original sinogram and forward projected recon
    num_slices, num_rows, num_channels = sino.shape
    estimated_shifts = np.zeros((num_slices, 2))

    warp_matrix = np.eye(2, 3, dtype=np.float32)
    for slice_index in range(num_slices):
        sino_from_recon_view = np.asarray(filtered_sino_from_recon[slice_index, :, :], dtype=sino_from_recon.dtype)
        sino_view = np.asarray(filtered_sino[slice_index, :, :], dtype=sino.dtype)
        cc, warp_matrix = cv2.findTransformECC(sino_from_recon_view, sino_view, warp_matrix,
                                               cv2.MOTION_TRANSLATION)
        estimated_shifts[slice_index, 0] = -warp_matrix[1, 2]
        estimated_shifts[slice_index, 1] = -warp_matrix[0, 2]

    return estimated_shifts


def sino_high_pass_filtering(sino, sigma_row=3.0, sigma_col=15.0, subtract_view_mean=True):
    """
    High-pass filter for 3D cone-beam sinogram.

    Args:
        sino (numpy array or jax array): 3D sinogram data with shape (num_views, num_det_rows, num_det_channels).
        sigma_row (float, optional): Gaussian sigma along detector rows (vertical). Use smaller value than sigma_col.
        Defaults to 3.0.
        sigma_col (float, optional): Gaussian sigma along detector channels (horizontal). Defaults to 15.0.
        subtract_view_mean (bool, optional): If True, subtract per-view mean (DC offset removal). Defaults to True.

    Returns:
        filtered_sino (numpy array): High-pass filtered sinogram, same shape as input.
    """
    sino_np = np.asarray(sino)
    if sino_np.ndim != 3:
        raise ValueError(f"Expected shape (num_views, num_det_rows, num_det_channels), got {sino_np.shape}")

    num_views, num_det_rows, num_det_channels = sino_np.shape
    filtered_sino = np.empty_like(sino_np)

    for view in range(num_views):
        single_view = sino_np[view]

        # Subtract per-view mean
        if subtract_view_mean:
            single_view = single_view - single_view.mean()

        # Estimate low frequency component for each view
        loss_pass_estimate = cv2.GaussianBlur(
            single_view,
            ksize=(0, 0),
            sigmaX=sigma_col,
            sigmaY=sigma_row,
            borderType=cv2.BORDER_REFLECT,
        )

        filtered_sino[view] = single_view - loss_pass_estimate

    return filtered_sino


def align_sino_views(ct_model, sino, direct_recon):
    """
    Align each sinogram view using estimated per-view shifts.

    This function performs sinogram alignment in two steps:
    1. Estimate a 2D shift for each sinogram view.
    2. Align each sinogram view using the estimated shift with the forward projected reconstruction.

    The alignment helps correct small per-view misalignments between the
    measured sinogram and the forward projection of a preliminary reconstruction.

    Args:
        ct_model (mj.TomographyModel): A CT model object that defined the CT geometry.
        sino (numpy array or jax array): 3D sinogram data with shape (num_views, num_det_rows, num_det_channels).
        direct_recon (numpy array or jax array): A preliminary 3D reconstruction of the sinogram.

    Returns:
        jax array: Aligned sinogram with the same shape as the input sinogram (num_views, num_det_rows, num_det_channels).
    """
    # Estimate per-view shift of the sinogram
    estimated_shifts = estimate_sino_view_offset(ct_model, sino, direct_recon)

    # Align each view of the sinogram using estimated shifts
    def shift_view(sino_view, shift):
        dy, dx = shift
        shifted_view = jax.image.scale_and_translate(sino_view,
                                                      shape=sino_view.shape,
                                                      spatial_dims=(0, 1),
                                                      scale=jnp.array([1.0, 1.0]),
                                                      translation=jnp.array([dy, dx]),
                                                      method="linear",
                                                      antialias=False)
        return shifted_view

    return jax.vmap(shift_view, in_axes=(0, 0))(sino, estimated_shifts)

def convert_full_scan_to_short_scan_conebeam(sino, cone_beam_params, optional_params):
    """
    Convert a full scan sinogram into a short scan sinogram by selecting
    a contiguous angular range of approximately pi + horizontal cone angle.

    Args:
        sino (np.ndarray): ndarray of shape (num_views, num_det_rows, num_det_channels)
        cone_beam_params (dict): Cone-beam geometry parameters that can be passed to the model constructor.
        optional_params (dict): Optional geometry parameters set after the model is constructed.

    Returns:
        sino_short: ndarray
        cone_beam_params_short: dict
    """

    num_views, num_det_rows, num_det_channels = sino.shape

    angles = cone_beam_params["angles"]
    source_detector_dist = cone_beam_params["source_detector_dist"]

    delta_det_channel = optional_params["delta_det_channel"]
    det_channel_offset = optional_params["det_channel_offset"]

    half_detector_width = delta_det_channel * num_det_channels / 2 + np.abs(det_channel_offset)
    half_horiz_cone_angle = np.arctan2(half_detector_width, source_detector_dist)
    angle_range = np.pi + half_horiz_cone_angle

    # This function selects the short scan starting from the first view.
    view_start = 0
    # Need np.unwrap so that if the scan is taken counter-clockwise,
    # 0, 359, 358, ... becomes 0, -1, -2, ..., which is easier for
    # finding the ending index of the short-scan view selection.
    angles_unwrap = np.unwrap(angles)
    candidate_idx = np.argwhere(np.abs(angles_unwrap) >= angle_range)
    if candidate_idx.size == 0:
        warnings.warn("Input scan does not contain enough angular coverage for short scan, returning the original scan")
        view_end = num_views
    else:
        view_end = int(candidate_idx[0, 0]) + 1
    sino_short = sino[view_start:view_end]
    angles_short = angles[view_start:view_end]

    cone_beam_params_short = dict(cone_beam_params)

    cone_beam_params_short['sinogram_shape'] = sino_short.shape
    cone_beam_params_short['angles'] = angles_short

    return sino_short, cone_beam_params_short