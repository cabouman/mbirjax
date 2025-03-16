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


def compute_sino_transmission(obj_scan, blank_scan, dark_scan, defective_pixel_list=None, correct_defective_pixels=True):
    """
    Compute sinogram from object, blank, and dark scans.

    This function computes sinogram by taking the negative log of the attenuation estimate.
    It can also take in a list of defective pixels and correct those pixel values.
    The invalid sinogram entries are the union of defective pixel entries and sinogram entries with values of inf or Nan.

    Args:
        obj_scan (ndarray, float): 3D object scan with shape (num_views, num_det_rows, num_det_channels).
        blank_scan (ndarray, float): [Default=None] 3D blank scan with shape (num_blank_scans, num_det_rows, num_det_channels). When num_blank_scans>1, the pixel-wise mean will be used as the blank scan.
        dark_scan (ndarray, float): [Default=None] 3D dark scan with shape (num_dark_scans, num_det_rows, num_det_channels). When num_dark_scans>1, the pixel-wise mean will be used as the dark scan.
        defective_pixel_list (optional, list(tuple)): A list of tuples containing indices of invalid sinogram pixels, with the format (view_idx, row_idx, channel_idx) or (detector_row_idx, detector_channel_idx).
            If None, then the invalid pixels will be identified as sino entries with inf or Nan values.
        correct_defective_pixels (optioonal, boolean): [Default=True] If true, the defective sinogram entries will be automatically corrected with `mbirjax.preprocess.interpolate_defective_pixels()`.

    Returns:
        2-element tuple containing:
        - **sino** (*ndarray, float*): Sinogram data with shape (num_views, num_det_rows, num_det_channels).
        - **defective_pixel_list** (list(tuple)): A list of tuples containing indices of invalid sinogram pixels, with the format (view_idx, row_idx, channel_idx) or (detector_row_idx, detector_channel_idx).

    """
    # take average of multiple blank/dark scans, and expand the dimension to be the same as obj_scan.
    blank_scan = 0 * obj_scan + np.mean(blank_scan, axis=0, keepdims=True)
    dark_scan = 0 * obj_scan + np.mean(dark_scan, axis=0, keepdims=True)

    obj_scan = obj_scan - dark_scan
    blank_scan = blank_scan - dark_scan

    #### compute the sinogram.
    # suppress warnings in np.log(), since the defective sino entries will be corrected.
    with np.errstate(divide='ignore', invalid='ignore'):
        sino = -np.log(obj_scan / blank_scan)

    # set the sino pixels corresponding to the provided defective list to 0.0
    if defective_pixel_list is None:
        defective_pixel_list = []
    else:    # if provided list is not None
        for defective_pixel_idx in defective_pixel_list:
            if len(defective_pixel_idx) == 2:
                (r,c) = defective_pixel_idx
                sino[:,r,c] = 0.0
            elif len(defective_pixel_idx) == 3:
                (v,r,c) = defective_pixel_idx
                sino[v,r,c] = 0.0
            else:
                raise Exception("compute_sino_transmission: index information in defective_pixel_list cannot be parsed.")

    # set NaN sino pixels to 0.0
    nan_pixel_list = list(map(tuple, np.argwhere(np.isnan(sino)) ))
    for (v,r,c) in nan_pixel_list:
        sino[v,r,c] = 0.0

    # set Inf sino pixels to 0.0
    inf_pixel_list = list(map(tuple, np.argwhere(np.isinf(sino)) ))
    for (v,r,c) in inf_pixel_list:
        sino[v,r,c] = 0.0

    # defective_pixel_list = union{input_defective_pixel_list, nan_pixel_list, inf_pixel_list}
    defective_pixel_list = list(set().union(defective_pixel_list,nan_pixel_list,inf_pixel_list))

    if correct_defective_pixels:
        print("Interpolate invalid sinogram entries.")
        sino, defective_pixel_list = interpolate_defective_pixels(sino, defective_pixel_list)
    else:
        if defective_pixel_list:
            print("Invalid sino entries detected! Please correct then manually or with function `mbirjax.preprocess.interpolate_defective_pixels()`.")
    return sino, defective_pixel_list


def compute_sino_transmission_jax(obj_scan, blank_scan, dark_scan, defective_pixel_array=(), batch_size=90):
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

    Returns:
        - **sino** (*ndarray, float*): Sinogram data with shape (num_views, num_det_rows, num_det_channels).
    """
    # Compute mean for blank and dark scans and move them to GPU if available
    blank_scan_mean = jnp.array(np.mean(blank_scan, axis=0, keepdims=True))
    dark_scan_mean = jnp.array(np.mean(dark_scan, axis=0, keepdims=True))

    sino_batches_list = []  # Initialize a list to store sinogram batches

    num_views = obj_scan.shape[0]  # Total number of views

    # Process obj_scan in batches
    for i in tqdm.tqdm(range(0, num_views, batch_size)):

        obj_scan_batch = obj_scan[i:min(i + batch_size, num_views)]
        obj_scan_batch = jnp.array(obj_scan_batch)

        obj_scan_batch = jnp.abs(obj_scan_batch - dark_scan_mean)
        blank_scan_batch = jnp.abs(blank_scan_mean - dark_scan_mean)

        # We use jnp.nan here because we'll later use np.nanmedian to get rid of nans and other defective pixels.
        sino_batch = -jnp.log(jnp.where(obj_scan_batch / blank_scan_batch > 0, obj_scan_batch / blank_scan_batch, jnp.nan))
        sino_batches_list.append(np.array(sino_batch))

    del sino_batch, obj_scan, blank_scan, dark_scan, obj_scan_batch, blank_scan_batch, dark_scan_mean, blank_scan_mean
    sino = np.concatenate(sino_batches_list, axis=0)
    del sino_batches_list
    print("Sinogram computation complete.")

    print("Interpolating invalid sinogram entries.")
    sino = interpolate_defective_pixels(sino, defective_pixel_array)

    return sino


def interpolate_defective_pixels(sino, defective_pixel_array=()):
    """
    Interpolates defective sinogram entries with the mean of neighboring pixels.

    Args:
        sino (ndarray, float): Sinogram data with 3D shape (num_views, num_det_rows, num_det_channels).
        defective_pixel_array (ndarray): A list of tuples containing indices of invalid sinogram pixels, with the format (detector_row_idx, detector_channel_idx) or (view_idx, detector_row_idx, detector_channel_idx).

    Returns:
        2-element tuple containing:
        - **sino** (*ndarray, float*): Corrected sinogram data with shape (num_views, num_det_rows, num_det_channels).
        - **defective_pixel_list** (*list(tuple)*): Updated defective_pixel_list with the format (detector_row_idx, detector_channel_idx) or (view_idx, detector_row_idx, detector_channel_idx).
    """
    # We'll set all defective pixels to NaN to be able to use np.nanmedian
    num_defective_pixels = len(defective_pixel_array)
    if num_defective_pixels > 0:
        flat_indices = np.ravel_multi_index(defective_pixel_array.T, sino.shape[1:])
        # For obj_scan, we need to set the bad pixels at every view to 0, so we can't use put directly.
        put_in_slice(sino, flat_indices, np.nan)
    else:
        defective_pixel_array = ()
    num_defective_pixels = len(defective_pixel_array)

    sino_shape = sino.shape
    num_views, num_rows, num_channels = sino_shape
    sino = np.nan_to_num(sino, copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan)

    # First handle the defective_pixel_array, each of which goes across all views
    # For each nan entry, we take the 3x3 neighbors in the same view and apply nanmedian.
    sino = sino.reshape((sino_shape[0], -1))
    neighbor_radius = 1

    # Generate all i_offset and j_offset combinations.  num_nbrs_1d = 2 * neighbor_radius + 1
    offsets = np.arange(-neighbor_radius, neighbor_radius + 1)
    i_offsets, j_offsets = np.meshgrid(offsets, offsets, indexing='ij')  # Shape (num_nbrs_1d, num_nbrs_1d)
    i_offsets = i_offsets.ravel()  # (num_nbrs_1d^2,)
    j_offsets = j_offsets.ravel()  # (num_nbrs_1d^2,)
    offsets_expanded = np.stack((i_offsets, j_offsets), axis=1)[None, :, :]  # (1, num_nbrs_1d^2, 2)

    if num_defective_pixels > 0:
        # Broadcast defective_pixel_array to match offset shapes
        defective_pixel_expanded = defective_pixel_array[:, None, :]  # (num_defective_pixels, 1, 2)

        # Add the indices and offsets to get valid neighbors, then convert to indices.  We use clip mode
        # for raveling to stay in bounds.  If a neighbor is out of bounds, clipping will yield a neighbor.
        neighbor_coords = defective_pixel_expanded + offsets_expanded  # (num_defective_pixels, num_nbrs_1d^2, 2)
        flat_indices = np.ravel_multi_index(neighbor_coords.transpose(2, 0, 1),
                                            sino_shape[1:], mode='clip')  # (num_defective_pixels, num_nbrs_1d^2)

        # Gather neighbor values for all views and use nanmedian to replace the values in sino
        neighbor_values_flat = sino[:, flat_indices]  # (num_views, num_defective_pixels, num_nbrs_1d^2)
        median_values = np.nanmedian(neighbor_values_flat, axis=2)
        flat_indices = np.ravel_multi_index(defective_pixel_array.T, sino_shape[1:])
        put_in_slice(sino, flat_indices, median_values)

    # Repeat on individual nans until there are no more.  Each index now has 3 components.
    sino = sino.reshape(sino_shape)
    nan_indices = np.argwhere(np.isnan(sino))
    offsets_expanded = np.stack((0*i_offsets, i_offsets, j_offsets), axis=1)[None, :, :]  # (1, num_nbrs_1d^2, 3)
    num_nans = nan_indices.shape[0]
    while num_nans > 0:
        sino = sino.flatten()

        nan_inds_expanded = nan_indices[:, None, :]  # (num_nans, 1, 3)
        neighbor_coords = nan_inds_expanded + offsets_expanded  # (num_nans, num_nbrs_1d^2, 2)
        flat_indices = np.ravel_multi_index(neighbor_coords.transpose(2, 0, 1),
                                            sino_shape, mode='clip')  # (num_nans, num_nbrs_1d^2)

        # Gather neighbor values for all views and replace the values, ignoring any nan values
        neighbor_values_flat = sino[flat_indices]  # (num_nans, num_nbrs_1d^2)
        median_values = np.nanmedian(neighbor_values_flat, axis=1)
        flat_indices = np.ravel_multi_index(nan_indices.T, sino_shape)
        sino[flat_indices] = median_values

        sino = sino.reshape(sino_shape)
        nan_indices = np.argwhere(np.isnan(sino))
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


def correct_det_rotation_batch_pix(sino, det_rotation=0.0, batch_size=30):
    """
    Correct sinogram data to account for detector rotation, using JAX for batch processing and GPU acceleration.
    Weights are not modified.

    Args:
        sino (jax.numpy.ndarray): Sinogram data with 3D shape (num_views, num_det_rows, num_det_channels).
        det_rotation (optional, float): tilt angle between the rotation axis and the detector columns in radians.
        batch_size (int): Number of views to process in each batch to avoid memory overload.

    Returns:
        - A jax.numpy.ndarray containing the corrected sinogram data if weights is None.
        - A tuple (sino_corrected, weights) if weights is not None.
    """

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


def estimate_background_offset(sino, option=0, edge_width=9):
    """
    Estimate background offset of a sinogram from the edge pixels.

    This function estimates the background offset when no object is present by computing a robust centroid estimate using `edge_width` pixels along the edge of the sinogram across views.
    Typically, this estimate is subtracted from the sinogram so that air is reconstructed as approximately 0.

    Args:
        sino (float, ndarray): Sinogram data with 3D shape (num_views, num_det_rows, num_det_channels).
        option (int, optional): [Default=0] Option of algorithm used to calculate the background offset.
        edge_width(int, optional): [Default=9] Width of the edge regions in pixels. It must be an odd integer >= 3.
    Returns:
        offset (float): Background offset value.
    """

    # Check validity of edge_width value
    assert(isinstance(edge_width, int)), "edge_width must be an integer!"
    if (edge_width % 2 == 0):
        edge_width = edge_width+1
        warnings.warn(f"edge_width of background regions should be an odd number! Setting edge_width to {edge_width}.")

    if (edge_width < 3):
        warnings.warn("edge_width of background regions should be >= 3! Setting edge_width to 3.")
        edge_width = 3

    _, _, num_det_channels = sino.shape

    # calculate mean sinogram
    sino_median=np.median(sino, axis=0)

    # offset value of the top edge region.
    # Calculated as median([median value of each horizontal line in top edge region])
    median_top = np.median(np.median(sino_median[:edge_width], axis=1))

    # offset value of the left edge region.
    # Calculated as median([median value of each vertical line in left edge region])
    median_left = np.median(np.median(sino_median[:, :edge_width], axis=0))

    # offset value of the right edge region.
    # Calculated as median([median value of each vertical line in right edge region])
    median_right = np.median(np.median(sino_median[:, num_det_channels-edge_width:], axis=0))

    # offset = median of three offset values from top, left, right edge regions.
    offset = np.median([median_top, median_left, median_right])
    return offset


def estimate_background_offset_jax(sino, edge_width=9):
    """
    Estimate background offset of a sinogram using JAX for GPU acceleration.

    Args:
        sino (jax.numpy.ndarray): Sinogram data with shape (num_views, num_det_rows, num_det_channels).
        edge_width (int, optional): Width of the edge regions in pixels. Must be an odd integer >= 3.

    Returns:
        offset (float): Background offset value.
    """
    if edge_width % 2 == 0:
        edge_width += 1
        warnings.warn(f"edge_width of background regions should be an odd number! Setting edge_width to {edge_width}.")
    if edge_width < 3:
        edge_width = 3
        warnings.warn("edge_width of background regions should be >= 3! Setting edge_width to 3.")

    _, _, num_det_channels = sino.shape
    # Extract edge regions directly from the sinogram (without computing a full median or transferring full sino to gpu)
    sino_edge = jnp.asarray(sino[:, :edge_width, :])
    median_top = jnp.median(jnp.median(jnp.median(sino_edge, axis=0), axis=1))  # Top edge

    sino_edge = jnp.asarray(sino[:, :, :edge_width])
    median_left = jnp.median(jnp.median(jnp.median(sino_edge, axis=0), axis=0))  # Left edge

    sino_edge = jnp.asarray(sino[:, :, num_det_channels-edge_width:])
    median_right = jnp.median(jnp.median(jnp.median(sino_edge, axis=0), axis=0))  # Right edge

    # Compute final offset as median of the three regions
    offset = jnp.median(jnp.array([median_top, median_left, median_right]))

    return offset


# ####### subroutines for image cropping and down-sampling
def downsample_scans(obj_scan, blank_scan, dark_scan,
                     downsample_factor, defective_pixel_array=()):
    """Performs Down-sampling to the scan images in the detector plane.

    Args:
        obj_scan (numpy array of floats): A stack of sinograms. 3D numpy array, (num_views, num_det_rows, num_det_channels).
        blank_scan (numpy array of floats): A blank scan. 2D numpy array, (num_det_rows, num_det_channels).
        dark_scan (numpy array of floats): A dark scan. 3D numpy array, (num_det_rows, num_det_channels).
        downsample_factor ([int, int]): Default=[1,1]] Two numbers to define down-sample factor.
        defective_pixel_array (ndarray):

    Returns:
        Downsampled scans
        - **obj_scan** (*ndarray, float*): A stack of sinograms. 3D numpy array, (num_views, num_det_rows, num_det_channels).
        - **blank_scan** (*ndarray, float*): A blank scan. 3D numpy array, (num_det_rows, num_det_channels).
        - **dark_scan** (*ndarray, float*): A dark scan. 3D numpy array, (num_det_rows, num_det_channels).
    """

    assert len(downsample_factor) == 2, 'factor({}) needs to be of len 2'.format(downsample_factor)
    assert (downsample_factor[0] >= 1 and downsample_factor[1] >= 1), 'factor({}) along each dimension should be greater or equal to 1'.format(downsample_factor)

    # Create a mask of good pixels
    good_pixel_mask = np.ones(blank_scan.shape[1:], dtype=np.uint8)
    if len(defective_pixel_array) > 0:
        # Set defective pixels to 0
        flat_indices = np.ravel_multi_index(defective_pixel_array.T, good_pixel_mask.shape)
        np.put(good_pixel_mask, flat_indices, 0)
        np.put(blank_scan[0], flat_indices, 0)
        np.put(dark_scan[0], flat_indices, 0)
        # For obj_scan, we need to set the bad pixels at every view to 0, so we can't use put directly.
        put_in_slice(obj_scan, flat_indices, 0.0)

    # crop the scan if the size is not divisible by downsample_factor.
    new_size1 = downsample_factor[0] * (obj_scan.shape[1] // downsample_factor[0])
    new_size2 = downsample_factor[1] * (obj_scan.shape[2] // downsample_factor[1])

    obj_scan = obj_scan[:, 0:new_size1, 0:new_size2]
    blank_scan = blank_scan[:, 0:new_size1, 0:new_size2]
    dark_scan = dark_scan[:, 0:new_size1, 0:new_size2]
    good_pixel_mask = good_pixel_mask[0:new_size1, 0:new_size2]

    # Compute block sum of the high res scan images. Defective pixels are excluded.
    # We do this by reshaping using the downsampling factor and then taking the average
    # over blocks, weighting by the block average of good_pixel_mask.
    block_shape = (good_pixel_mask.shape[0] // downsample_factor[0], downsample_factor[0],
                   good_pixel_mask.shape[1] // downsample_factor[1], downsample_factor[1])
    # filter out defective pixels
    good_pixel_mask = good_pixel_mask.reshape((1,) + block_shape)
    obj_scan = obj_scan.reshape((obj_scan.shape[0],) + block_shape)
    blank_scan = blank_scan.reshape((blank_scan.shape[0],) + block_shape)
    dark_scan = dark_scan.reshape((dark_scan.shape[0],) + block_shape)

    # compute block sum
    obj_scan = obj_scan.sum((2,4))
    blank_scan = blank_scan.sum((2, 4))
    dark_scan = dark_scan.sum((2, 4))
    # number of good pixels in each down-sampling block
    good_pixel_count = good_pixel_mask.sum((2, 4)).astype(np.uint32)

    # new defective pixel list = {indices of pixels where the downsampling block contains all bad pixels}
    defective_pixel_list = np.argwhere(good_pixel_count[0] == 0)
    if len(defective_pixel_list) > 0:
        defective_pixel_array = np.array(defective_pixel_list)
    else:
        defective_pixel_array = ()
    good_pixel_count = good_pixel_count.astype(np.float32)

    # compute block averaging by dividing block sum with number of good pixels in the block
    # The division by good_pixel_count could give inf or nan, but only at defective pixels.
    obj_scan = obj_scan / good_pixel_count
    blank_scan = blank_scan / good_pixel_count
    dark_scan = dark_scan / good_pixel_count

    return obj_scan, blank_scan, dark_scan, defective_pixel_array


def crop_scans(obj_scan, blank_scan, dark_scan,
                crop_region=((0, 1), (0, 1)), defective_pixel_array=()):
    """Crop obj_scan, blank_scan, and dark_scan images by decimal factors, and update defective_pixel_list accordingly.
    Args:
        obj_scan (float): A stack of sinograms. 3D numpy array, (num_views, num_det_rows, num_det_channels).
        blank_scan (float) : A blank scan. 3D numpy array, (1, num_det_rows, num_det_channels).
        dark_scan (float): A dark scan. 3D numpy array, (1, num_det_rows, num_det_channels).
        crop_region ([(float, float),(float, float)] or [float, float, float, float]):
            [Default=[(0, 1), (0, 1)]] Two points to define the bounding box. Sequence of [(row0, row1), (col0, col1)] or
            [row0, row1, col0, col1], where 0<=row0 <= row1<=1 and 0<=col0 <= col1<=1.

            The scan images will be cropped using the following algorithm:
                obj_scan <- obj_scan[:,Nr_lo:Nr_hi, Nc_lo:Nc_hi], where
                    - Nr_lo = round(row0 * obj_scan.shape[1])
                    - Nr_hi = round(row1 * obj_scan.shape[1])
                    - Nc_lo = round(col0 * obj_scan.shape[2])
                    - Nc_hi = round(col1 * obj_scan.shape[2])
        defective_pixel_array (ndarray):

    Returns:
        Cropped scans
        - **obj_scan** (*ndarray, float*): A stack of sinograms. 3D numpy array, (num_views, num_det_rows, num_det_channels).
        - **blank_scan** (*ndarray, float*): A blank scan. 3D numpy array, (1, num_det_rows, num_det_channels).
        - **dark_scan** (*ndarray, float*): A dark scan. 3D numpy array, (1, num_det_rows, num_det_channels).
    """
    if isinstance(crop_region[0], (list, tuple)):
        (row0, row1), (col0, col1) = crop_region
    else:
        row0, row1, col0, col1 = crop_region

    assert 0 <= row0 <= row1 <= 1 and 0 <= col0 <= col1 <= 1, 'crop_region should be sequence of [(row0, row1), (col0, col1)] ' \
                                                      'or [row0, row1, col0, col1], where 1>=row1 >= row0>=0 and 1>=col1 >= col0>=0.'
    assert math.isclose(col0, 1 - col1), 'horizontal crop limits must be symmetric'

    Nr_lo = round(row0 * obj_scan.shape[1])
    Nc_lo = round(col0 * obj_scan.shape[2])

    Nr_hi = round(row1 * obj_scan.shape[1])
    Nc_hi = round(col1 * obj_scan.shape[2])

    obj_scan = obj_scan[:, Nr_lo:Nr_hi, Nc_lo:Nc_hi]
    blank_scan = blank_scan[:, Nr_lo:Nr_hi, Nc_lo:Nc_hi]
    dark_scan = dark_scan[:, Nr_lo:Nr_hi, Nc_lo:Nc_hi]

    # Remove any defective pixels that are outside the new cropped region
    if len(defective_pixel_array) > 0:
        in_bounds = (defective_pixel_array[:, 0] >= Nr_lo) & (defective_pixel_array[:, 0] < Nr_hi) & \
                    (defective_pixel_array[:, 1] >= Nc_lo) & (defective_pixel_array[:, 1] < Nc_hi)
        defective_pixel_array = defective_pixel_array[in_bounds]

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


def put_in_slice(array, flat_indices, value):
    """
    Similar to numpy.put(array, flat_indices, value), which would produce array.flat[flat_indices] = value.
    However, this function requires that array have an extra leading dimension, and that the value for a given
    index is copied across that dimension.  With abuse of notation, array[:, flat_indices] = value

    Args:
        array (ndarray): Numpy array of dimension n+1
        flat_indices (ndarray of int): Indices obtained using ravel_multi_index using array.shape[1:]
        value (float or ndarray): Values to be copied in.  Must be able to broadcast to array.shape[1:]

    Returns:
        ndarray
    """
    array_shape = array.shape
    array = array.reshape(array_shape[0], -1)
    array[:, flat_indices] = value
    array = array.reshape(array_shape)
    return array
