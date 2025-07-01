import numpy as np
import warnings
import tifffile
import glob
import os
import jax.numpy as jnp
import jax
import dm_pix
import tqdm


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
    Crop obj_scan, blank_scan, and dark_scan images by an integer number of pixels, and update defective_pixel_array accordingly.
    The left and right side pixels are cropped the same amount in order to preserve the center of rotation.

    Args:
        obj_scan (ndarray): Sinogram. 3D numpy array of shape (num_views, num_det_rows, num_det_channels).
        blank_scan (ndarray): Blank scan(s). 3D numpy array of shape (num_blank_views, num_det_rows, num_det_channels).
        dark_scan (ndarray): Dark scan(s). 3D numpy array of shape (num_dark_views, num_det_rows, num_det_channels).
        crop_pixels_sides (int, optional): Number of pixels to crop from each side of the sinogram. Defaults to 0.
        crop_pixels_top (int, optional): Number of pixels to crop from the top of the sinogram. Defaults to 0.
        crop_pixels_bottom (int, optional): Number of pixels to crop from the bottom of the sinogram. Defaults to 0.
        defective_pixel_array (ndarray): Array of shape (num_defective_pixels, 2) containing (row, col) coordinates.

    Notes:
        This function supports both singleton blank/dark scans (with shape (1, H, W)) and multi-view scans
        (with shape (N, H, W), where N > 1). Cropping is applied consistently across all views.

    Returns:
        tuple:
        - **obj_scan** (*ndarray, float*): Cropped stack of sinograms. 3D numpy array of shape (num_views, new_rows, new_cols).
        - **blank_scan** (*ndarray, float*): Cropped blank scan(s). 3D numpy array of shape (num_blank_views, new_rows, new_cols).
        - **dark_scan** (*ndarray, float*): Cropped dark scan(s). 3D numpy array of shape (num_dark_views, new_rows, new_cols).
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


def apply_cylindrical_mask(recon, radial_margin=0, top_margin=0, bottom_margin=0):
    """
    Applies a cylindrical mask to a 3D reconstruction volume.

    This function zeros out all voxels outside a centered cylindrical region
    in the (row, col) plane and also zeroes a specified number of slices from
    the top and bottom along the Z-axis (slice axis).

    This function is useful for removing `flash` that typically accumulates on the boundaries of an MBIR reconstruction volume.

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

    # Zero out top and bottom slices along Z
    if top_margin > 0:
        recon = recon.at[:, :, :top_margin].set(0)
    if bottom_margin > 0:
        recon = recon.at[:, :, -bottom_margin:].set(0)

    return recon

