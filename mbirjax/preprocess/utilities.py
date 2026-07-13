import numpy as np
import warnings
import glob
import os
import jax.numpy as jnp
import jax
import cv2
import h5py
import mbirjax as mj
import scipy

from . import pipeline

def _transmission_kernel(obj_batch, blank_minus_dark, dark_scan_mean, flat_indices, defective_pixel_array_jax):
    """Per-view-batch transmission kernel (pure device-array op).

    Computes ``-log`` of the dark-corrected, blank-normalized object scan, sets shared defective
    pixels to NaN, and interpolates the NaNs.  ``blank_minus_dark = |blank_mean - dark_mean|`` is
    precomputed by the caller (it does not vary across batches).
    """
    obj_batch = jnp.abs(obj_batch - dark_scan_mean)
    # jnp.nan for non-positive ratios so nanmedian later removes them along with defective pixels.
    sino_batch = -jnp.log(jnp.where(obj_batch / blank_minus_dark > 0, obj_batch / blank_minus_dark, jnp.nan))
    if flat_indices is not None:
        # Shared defective pixels (same in every view) -> NaN for nanmedian interpolation.
        sino_batch = put_in_slice(sino_batch, flat_indices, jnp.nan)
    return interpolate_defective_pixels(sino_batch, defective_pixel_array_jax)


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
            An array of defective pixel indices, one (row_idx, channel_idx) pair per row;
            these pixels are treated as defective in every view.
            If `None`, invalid pixels are inferred from `NaN` or `inf` values.
        batch_size (int): 
            Number of views to process in each GPU batch.

    Returns:
        ndarray: 
            The computed sinogram, with shape (num_views, num_det_rows, num_det_channels).
    """
    # Blank/dark means (device); blank_minus_dark does not vary across batches, so precompute it once.
    blank_scan_mean = jnp.array(np.mean(blank_scan, axis=0, keepdims=True))
    dark_scan_mean = jnp.array(np.mean(dark_scan, axis=0, keepdims=True))
    blank_minus_dark = jnp.abs(blank_scan_mean - dark_scan_mean)

    if len(defective_pixel_array) > 0:
        defective_pixel_array_jax = jnp.array(defective_pixel_array)
        flat_indices = jnp.ravel_multi_index(defective_pixel_array_jax.T, obj_scan.shape[1:])
    else:
        defective_pixel_array_jax = ()
        flat_indices = None

    sino = pipeline.map_view_batches(
        obj_scan,
        lambda obj_batch: _transmission_kernel(obj_batch, blank_minus_dark, dark_scan_mean,
                                               flat_indices, defective_pixel_array_jax),
        batch_size)
    print("Sinogram computation complete.")
    return sino


def _warn_unfilled(num_residual):
    """Host-side warning hook (invoked via ``jax.debug.callback``) when some pixels could not be filled.

    ``jax.debug.callback`` runs this on the host even from inside a jitted/traced kernel, so the dense
    NaN-fill below can stay fully jittable while still surfacing the rare "cluster too big" case."""
    num_residual = int(num_residual)
    if num_residual > 0:
        warnings.warn(f"NaN-fill: {num_residual} pixel(s) left after the fill passes (defective/zinger "
                      f"clusters wider than the fill reach); setting them to 0. Increase num_passes if "
                      f"this is unexpected.")


def _box3x3_sum(x):
    """Sum over each 3x3 (row, channel) window, per view, zero-padded at the detector edges.

    ``x`` has shape (num_views, num_det_rows, num_det_channels); the window spans only the trailing two
    (detector) axes.  Uses ``reduce_window`` -- a fused sliding-window sum -- so it does NOT materialize
    nine shifted copies of the array (O(N) memory, unlike stack-the-9-neighbors-and-reduce)."""
    return jax.lax.reduce_window(x, 0.0, jax.lax.add,
                                 window_dimensions=(1, 3, 3),
                                 window_strides=(1, 1, 1),
                                 padding='SAME')


def _interpolate_fill_pass(sino):
    """One dense pass: replace each NaN pixel with the MEAN of its finite 3x3 in-view neighbors.

    Finite pixels are unchanged.  A NaN with no finite neighbor stays NaN (it gets filled on a later
    pass, as the surrounding NaN region shrinks inward).  Computed densely with two box sums (values and
    a validity mask), so it is jittable and memory-light -- no per-NaN gather, no neighbor stack."""
    is_nan = jnp.isnan(sino)
    valid = (~is_nan).astype(sino.dtype)
    filled = jnp.where(is_nan, jnp.zeros((), sino.dtype), sino)        # NaNs contribute 0 to the sum
    neighbor_sum = _box3x3_sum(filled)
    neighbor_count = _box3x3_sum(valid)
    neighbor_mean = neighbor_sum / jnp.maximum(neighbor_count, 1.0)    # avoid 0/0 where no valid neighbor
    return jnp.where(is_nan & (neighbor_count > 0), neighbor_mean, sino)


def _fill_nan_pixels(sino, num_passes=3):
    """Fill every NaN pixel with the mean of its finite 3x3 in-view neighbors, in ``num_passes`` dense
    passes (each fills the NaN frontier inward by one pixel), then warn (host-side) about and zero any
    pixel still NaN (a defective/zinger cluster wider than the fill reach).  Pure + fixed-shape ->
    jittable.  Shared by ``interpolate_defective_pixels`` and the zinger correction -- they differ only
    in HOW pixels are flagged NaN before calling this."""
    for _ in range(num_passes):
        sino = _interpolate_fill_pass(sino)
    # ordered=True ties the callback to the data dependency so the warning fires before the result is
    # consumed (rather than firing asynchronously and possibly being missed).
    jax.debug.callback(_warn_unfilled, jnp.sum(jnp.isnan(sino)), ordered=True)
    return jnp.nan_to_num(sino, nan=0.0)


def _zinger_fill(sino, zinger_threshold, num_passes=3):
    """Mark zinger pixels (anomalously negative: ``value < zinger_threshold``) and non-finite values NaN,
    then fill them from their finite 3x3 in-view neighbors via :func:`_fill_nan_pixels`.  Pure +
    fixed-shape -> jittable (per view-batch); ``zinger_threshold`` is a precomputed scalar."""
    sino = jnp.where(jnp.isfinite(sino), sino, jnp.nan)        # +-inf / NaN -> NaN
    sino = jnp.where(sino < zinger_threshold, jnp.nan, sino)   # flag zingers
    return _fill_nan_pixels(sino, num_passes)


def interpolate_defective_pixels(sino, defective_pixel_array=(), num_passes=3):
    """Replace defective / non-finite sinogram pixels with the mean of their finite 3x3 in-view neighbors.

    Invalid pixels are the non-finite entries (NaN / +-inf -- e.g. the transmission's ``obj/blank <= 0``
    pixels) plus any pixel listed in ``defective_pixel_array`` (the same detector coords in every view).
    They are marked NaN and then filled by ``num_passes`` dense neighborhood-mean passes: each pass
    replaces every NaN that has >=1 finite neighbor with the mean of its finite 3x3 in-view neighbors, so
    a NaN region shrinks inward by one pixel per pass.  ``num_passes`` thus bounds the largest fillable
    dead-pixel cluster (radius ``num_passes``); any pixel still NaN afterward triggers a warning and is
    set to 0.

    This is a fully **jittable, fixed-shape** computation (dense ``reduce_window`` passes -- no
    ``argwhere``, no data-dependent ``while`` loop), so the whole scan->sinogram kernel can be wrapped in
    a single ``jax.jit``.  The fill uses the neighbor MEAN (not median): for dead pixels surrounded by
    valid data the two are nearly identical, and they affect <1% of pixels.

    Args:
        sino (jax array, float): (num_views, num_det_rows, num_det_channels).
        defective_pixel_array (array or tuple): shared (det_row, det_channel) coords of defective pixels,
            or () for none.  (The transmission caller already NaN-marks these; re-marking here is cheap
            and keeps this function correct when called on its own.)
        num_passes (int): number of dense fill passes = max fillable dead-pixel cluster radius.

    Returns:
        jax array, float: sinogram with invalid pixels interpolated (any unfillable ones set to 0).
    """
    num_views, num_rows, num_channels = sino.shape

    # Mark every invalid pixel NaN: non-finite values, plus the known shared defective pixels.
    sino = jnp.where(jnp.isfinite(sino), sino, jnp.nan)
    if len(defective_pixel_array) > 0:
        # mode='clip' (a no-op for these in-bounds coords) keeps ravel_multi_index jittable -- the
        # default mode='raise' forces a concrete bounds check that fails under jit.
        defective_flat = jnp.ravel_multi_index(jnp.asarray(defective_pixel_array).T,
                                               (num_rows, num_channels), mode='clip')  # (num_defective,)
        sino = sino.reshape(num_views, -1).at[:, defective_flat].set(jnp.nan).reshape(
            num_views, num_rows, num_channels)

    # Dense neighborhood-mean fill, a fixed number of passes (shared with the zinger correction).
    return _fill_nan_pixels(sino, num_passes)



def _rotation_kernel(sino_batch, det_rotation):
    """Per-view-batch detector-rotation kernel (pure device-array op): rotate each view's
    (row, channel) plane by ``det_rotation`` radians with bilinear interpolation, zero outside.

    This is a direct 2-D bilinear rotation that mirrors ``dm_pix.rotate``'s geometry (same rotation
    matrix, rotation center, and constant zero-fill boundary) but computes the rotated sampling grid
    ONCE for the (row, channel) plane and reuses it across all views.  ``dm_pix.rotate`` instead treats
    the whole (rows, cols, views) block as a single 3-D image: it builds a full 3-D coordinate grid (the
    2-D grid replicated across every view) and does ``2**3 = 8``-corner linear interpolation, which
    transiently allocates tens of GB for a large view batch -- the dominant preprocessing GPU peak.
    Because the view axis is an identity dimension there (each output view samples only its own input
    view), the per-view result is plain 4-corner 2-D bilinear interpolation, which we compute directly:
    gather the 4 neighbors for every view in one shot with shared 2-D weights.  ~10x less memory and
    faster, matching dm_pix to floating-point tolerance (not bit-exact: different op ordering).

    Args:
        sino_batch (jax array): (num_views, num_det_rows, num_det_channels).
        det_rotation (float): rotation angle in radians (dm_pix sign convention).
    """
    num_views, num_rows, num_cols = sino_batch.shape
    cos_a = jnp.cos(det_rotation)
    sin_a = jnp.sin(det_rotation)
    center_row = (num_rows - 1) / 2.0
    center_col = (num_cols - 1) / 2.0

    # For each OUTPUT pixel (i, j), find the INPUT location it samples, using the same affine map dm_pix
    # applies: coord = R @ pixel + offset, with offset = center - R @ center (rotation about the center).
    grid_i, grid_j = jnp.meshgrid(jnp.arange(num_rows), jnp.arange(num_cols), indexing='ij')  # (rows, cols)
    offset_row = center_row - (cos_a * center_row + sin_a * center_col)
    offset_col = center_col - (-sin_a * center_row + cos_a * center_col)
    src_row = cos_a * grid_i + sin_a * grid_j + offset_row   # (num_rows, num_cols)
    src_col = -sin_a * grid_i + cos_a * grid_j + offset_col

    # Bilinear neighbors, matching dm_pix: lower = floor, upper = ceil, weight = coord - floor; the
    # gather indices are clipped into range and out-of-image samples are zeroed afterwards via the mask.
    lower_row = jnp.floor(src_row)
    lower_col = jnp.floor(src_col)
    frac_row = src_row - lower_row     # (num_rows, num_cols)
    frac_col = src_col - lower_col
    r0 = jnp.clip(lower_row.astype(jnp.int32), 0, num_rows - 1)
    r1 = jnp.clip(jnp.ceil(src_row).astype(jnp.int32), 0, num_rows - 1)
    c0 = jnp.clip(lower_col.astype(jnp.int32), 0, num_cols - 1)
    c1 = jnp.clip(jnp.ceil(src_col).astype(jnp.int32), 0, num_cols - 1)

    # Gather the 4 neighbors for EVERY view at once; the (num_rows, num_cols) weights broadcast over the
    # leading view axis.  sino_batch[:, r, c] has shape (num_views, num_rows, num_cols).
    rotated = (((1.0 - frac_row) * (1.0 - frac_col)) * sino_batch[:, r0, c0]
               + ((1.0 - frac_row) * frac_col) * sino_batch[:, r0, c1]
               + (frac_row * (1.0 - frac_col)) * sino_batch[:, r1, c0]
               + (frac_row * frac_col) * sino_batch[:, r1, c1])

    # Zero any output pixel whose sample fell outside the original image (mode='constant', cval=0).
    in_bounds = (src_row >= 0) & (src_row <= num_rows - 1) & (src_col >= 0) & (src_col <= num_cols - 1)
    return rotated * in_bounds


# Jitted rotation: XLA fuses the 4-corner gather + weighted sum + boundary mask into a single kernel,
# reusing buffers (lower peak than eager op-by-op, which materializes every intermediate) and avoiding
# per-op dispatch (faster, which matters most for multi-device throughput).  ``det_rotation`` flows as a
# traced scalar, so one compile per (num_views, num_rows, num_cols) shape is reused across angles.  Run
# under ``jax.default_device(device)`` (as the preprocessing workers are), the computation -- and its
# arange/meshgrid constants -- lands on that worker's device.
_rotation_kernel_jit = jax.jit(_rotation_kernel)


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

    return pipeline.map_view_batches(sino, lambda b: _rotation_kernel_jit(b, det_rotation), batch_size)


def correct_background_offset(sino, edge_width=9, option='global'):
    """
    Correct background offset in a sinogram.

    Args:
        sino (numpy.ndarray): Sinogram data with shape (num_views, num_det_rows, num_det_channels).
        edge_width (int, optional): Width of the edge regions in pixels. Must be an integer >= 1.  Defaults to 9.
        option (str or None): One of:
            - None: No correction; return the input sinogram unchanged.
            - "global": Estimate one scalar offset from edge regions across all views.
            - "per_view": Estimate one offset per view from edge regions.
            Defaults to 'global'.

    Returns:
        sino_corrected (numpy.ndarray)
    """

    # No-op option: return the original sinogram without modification.
    if option is None:
        return sino

    if edge_width < 1:
        edge_width = 1
        warnings.warn("edge_width of background regions should be >= 1! Setting edge_width to 1.")

    num_views, _, num_det_channels = sino.shape

    sino_edge_left  = sino[:, :, :edge_width].reshape(num_views, -1)
    sino_edge_right = sino[:, :, num_det_channels-edge_width:].reshape(num_views, -1)
    sino_edge_top   = sino[:, :edge_width, :].reshape(num_views, -1)

    med_left  = np.median(sino_edge_left, axis=1)
    med_right = np.median(sino_edge_right, axis=1)
    med_top   = np.median(sino_edge_top, axis=1)

    edge_medians = np.stack([med_left, med_right, med_top], axis=1)
    offset = np.median(edge_medians, axis=1)   # (num_views,)

    if option == "global":
        # Estimate one scalar offset from edge regions across all views
        percentile = 10
        offset = np.percentile(offset, percentile)
        sino_corrected = sino - offset

    elif option == "per_view":
        sino_corrected = sino - offset[:, None, None]

    else:
        raise ValueError("option must be None, 'global' or 'per_view'")

    return sino_corrected


# ####### subroutines for image cropping and down-sampling
def _downsample_obj_kernel(obj_batch, flat_indices, new_size1, new_size2, block_shape):
    """Per-view-batch object-scan downsample kernel (pure device-array op): set defective pixels to
    NaN, crop to a block-divisible size, and block-average with nanmean."""
    if flat_indices is not None:
        obj_batch = put_in_slice(obj_batch, flat_indices, jnp.nan)
    obj_batch = obj_batch[:, 0:new_size1, 0:new_size2]
    obj_batch = obj_batch.reshape((obj_batch.shape[0],) + block_shape)
    return jnp.nanmean(obj_batch, axis=(2, 4))


def _downsample_blank_dark(blank_scan, dark_scan, downsample_factor, defective_pixel_array=()):
    """Host-side part of view-data downsampling: NaN-mask defective pixels, crop to a block-divisible
    size, and block-average the blank and dark scans.  Returns the downsampled blank/dark, the NEW
    (downsampled-grid) defective array, and the parameters :func:`_downsample_obj_kernel` needs for the
    object scan.  (blank/dark share the object scan's detector dimensions, so its block params are
    computed here.)

    Returns:
        (blank_scan, dark_scan, defective_pixel_array, obj_flat_indices, new_size1, new_size2, block_shape)
    """
    # Set defective pixels to NaN for use with nanmean.  blank_scan and dark_scan may have different
    # numbers of views, so loop each over its OWN leading dimension (a single shared loop over
    # blank_scan.shape[0] would index out of bounds on a singleton dark_scan, or vice versa).
    if len(defective_pixel_array) > 0:
        flat_indices = np.ravel_multi_index(defective_pixel_array.T, blank_scan.shape[1:])
        for i in range(blank_scan.shape[0]):
            np.put(blank_scan[i], flat_indices, np.nan)
        for i in range(dark_scan.shape[0]):
            np.put(dark_scan[i], flat_indices, np.nan)
    else:
        flat_indices = None

    # Crop the scan if the size is not divisible by downsample_factor (blank/dark share the object
    # scan's detector dimensions).
    new_size1 = downsample_factor[0] * (blank_scan.shape[1] // downsample_factor[0])
    new_size2 = downsample_factor[1] * (blank_scan.shape[2] // downsample_factor[1])

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

    # new defective pixel list = {indices of pixels where the downsampling block contains all bad pixels}
    nan_mask = np.isnan(blank_scan).any(axis=0)  # Combine across all views
    defective_pixel_array = np.argwhere(nan_mask)
    if len(defective_pixel_array) == 0:
        defective_pixel_array = ()

    # flat_indices stays a HOST array so the per-view object-scan kernel is device-agnostic (it
    # auto-promotes to each batch's device); this matters for the multi-device fused path.
    return blank_scan, dark_scan, defective_pixel_array, flat_indices, new_size1, new_size2, block_shape


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

    blank_scan, dark_scan, defective_pixel_array, obj_flat_indices, new_size1, new_size2, block_shape = \
        _downsample_blank_dark(blank_scan, dark_scan, downsample_factor, defective_pixel_array)

    # Object scan: batch over views through the shared driver, block-averaging each view-batch.
    obj_scan = pipeline.map_view_batches(
        obj_scan,
        lambda b: _downsample_obj_kernel(b, obj_flat_indices, new_size1, new_size2, block_shape),
        batch_size)

    return obj_scan, blank_scan, dark_scan, defective_pixel_array


def scan_to_sino(obj_scan, blank_scan, dark_scan, defective_pixel_array=(),
                 downsample_factor=(1, 1), det_rotation=0.0, zinger_pixel_ratio=None,
                 batch_size=90, devices=None, max_views_to_use=20):
    """Fused scan -> sinogram for cropped scan data, view-sharded across devices.

    Runs (optional downsample) -> transmission -> (optional detector rotation) -> (optional zinger
    correction) as a single on-device pass per view-batch, so the object scan is uploaded once and the
    sinogram gathered once -- instead of a separate host round-trip for each stage.  Equivalent to
    calling ``downsample_view_data`` (when ``downsample_factor`` exceeds 1), then
    ``compute_sino_transmission``, then ``correct_det_rotation`` (and ``interpolate_zinger_pixels`` when
    ``zinger_pixel_ratio`` is set), but without materializing the intermediates on the host.
    Background-offset correction is left to the caller (a cheap host pass).

    Every stage here is **per-view** (the defective/zinger interpolation uses within-view neighbors), so
    the views are split into contiguous shards across ``devices`` and processed concurrently with no
    cross-device communication; the result is identical regardless of device count (and to the
    single-device sequential path).  Detector rotation is skipped entirely when ``det_rotation == 0``.

    Zinger correction (``zinger_pixel_ratio`` not None) is folded in here so it costs no extra host
    round-trip, and it runs **before** the caller's offset/shift passes -- correct, since a zinger should
    be removed before a sub-pixel detector shift could interpolate it into its neighbors.  Its threshold
    is ``-ratio * RMS(sino over support)`` estimated from a cheap pre-pass over a ~``max_views_to_use``
    view subsample (the threshold is statistical; detection then runs on every view in the main pass).

    Args:
        obj_scan, blank_scan, dark_scan (ndarray): cropped scans (object batched along axis 0).
        defective_pixel_array (ndarray or tuple): shared defective-pixel (row, col) coords, or ().
        downsample_factor (tuple[int, int]): detector row/channel downsample; (1, 1) skips downsampling.
        det_rotation (float): detector rotation in radians; 0 skips the rotation.
        zinger_pixel_ratio (float or None): if set, fold in zinger correction with this ratio; None skips it.
        batch_size (int): number of views per on-device batch.
        devices (sequence or None): devices to spread the views over; ``None`` uses ``jax.devices()``.
        max_views_to_use (int): views sampled for the zinger threshold estimate (when zinger is enabled).

    Returns:
        numpy.ndarray: the sinogram, shape (num_views, num_det_rows, num_det_channels).
    """
    if devices is None:
        devices = jax.devices()

    obj_flat_indices = new_size1 = new_size2 = block_shape = None
    do_downsample = downsample_factor[0] * downsample_factor[1] > 1
    if do_downsample:
        blank_scan, dark_scan, defective_pixel_array, obj_flat_indices, new_size1, new_size2, block_shape = \
            _downsample_blank_dark(blank_scan, dark_scan, downsample_factor, defective_pixel_array)

    # Transmission constants kept as HOST NumPy so the fused kernel is device-agnostic: each value
    # auto-promotes to its batch's device, which is what lets the SAME kernel run on every view shard.
    # mean/abs are the identical IEEE float32 ops whether evaluated here on the host or on a device, so
    # this matches the single-device (jnp-constant) result bit-for-bit.  blank_minus_dark does not vary
    # across batches, so precompute it once.
    blank_scan_mean = np.mean(blank_scan, axis=0, keepdims=True)
    dark_scan_mean = np.mean(dark_scan, axis=0, keepdims=True)
    blank_minus_dark = np.abs(blank_scan_mean - dark_scan_mean)

    # Defective-pixel indices for the transmission kernel, raveled against the detector grid the kernel
    # sees (the downsampled grid when downsampling; blank/dark share the object scan's detector dims).
    trans_det_shape = blank_scan.shape[1:]
    if len(defective_pixel_array) > 0:
        defective_pixel_array = np.asarray(defective_pixel_array)
        trans_flat_indices = np.ravel_multi_index(defective_pixel_array.T, trans_det_shape)
    else:
        defective_pixel_array = ()
        trans_flat_indices = None

    do_rotation = det_rotation != 0.0
    do_zinger = zinger_pixel_ratio is not None

    # The whole per-batch kernel -- (downsample) -> transmission -> interpolate -> (rotation) ->
    # (zinger) -- is jittable (interpolate/zinger use a fixed-iteration dense fill instead of
    # argwhere/while), so we wrap it in a single jax.jit.  XLA then fuses the stages, reuses buffers, and
    # dispatches one kernel per batch instead of dozens of eager ops (lower memory + faster, esp.
    # multi-device).  The host constants (blank_minus_dark, etc.), the do_* flags, det_rotation, and the
    # zinger threshold are captured as compile constants.  Run under each worker's jax.default_device,
    # the kernel compiles per device with its constants on that device (one compile per shape per device,
    # reused per batch).  ``zinger_threshold`` is built fresh per call so we can compile a no-zinger
    # variant for the threshold pre-pass below.
    def build_fused_kernel(zinger_threshold):
        @jax.jit
        def fused_kernel(obj_batch):
            if do_downsample:
                obj_batch = _downsample_obj_kernel(obj_batch, obj_flat_indices, new_size1, new_size2, block_shape)
            sino_batch = _transmission_kernel(obj_batch, blank_minus_dark, dark_scan_mean,
                                              trans_flat_indices, defective_pixel_array)
            if do_rotation:
                sino_batch = _rotation_kernel(sino_batch, det_rotation)
            if zinger_threshold is not None:
                sino_batch = _zinger_fill(sino_batch, zinger_threshold)
            return sino_batch
        return fused_kernel

    # Zinger threshold pre-pass: the threshold is the RMS over the sinogram support, which we estimate by
    # running the kernel (without zinger) on a ~max_views_to_use even view subsample -- cheap, single
    # device.  This avoids a second full pass over the sinogram (the standalone interpolate_zinger_pixels
    # path) by folding the correction into the main pass below.
    zinger_threshold = None
    if do_zinger:
        obj_sub = mj.TomographyModel.subsample_views(obj_scan, max_views_to_use)
        sino_sub = pipeline.map_view_batches(obj_sub, build_fused_kernel(None),
                                             batch_size, devices=[devices[0]])
        zinger_threshold = _zinger_threshold(sino_sub, zinger_pixel_ratio, max_views_to_use)

    sino = pipeline.map_view_batches(obj_scan, build_fused_kernel(zinger_threshold), batch_size, devices=devices)
    print("Sinogram computation complete.")
    return sino



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


def read_tif_img(img_path):
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


def read_tif_stack_dir(scan_dir, view_ids=None):
    """Reads a tif stack of scan images from a directory. This function is a subroutine to `load_scans_and_params`.

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
    if len(img_path_list) == 0:
        img_path_list = sorted(glob.glob(os.path.join(scan_dir, '*[0-9].tiff')))  # Assume files are '.tif' but check '.tiff' if not

    # if no views are found, raise an error
    if len(img_path_list) == 0:
        raise FileNotFoundError('No scan images found in directory: {}'.format(scan_dir))

    # Set view_idx to be an array corresponding to the views that should be read.
    # This assumes that all the views are labeled sequentially.
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


# Not jitted, so a numpy recon stays on the host (jit would force the whole volume onto one GPU).
def apply_cylindrical_mask(recon, radial_margin=0, top_margin=0, bottom_margin=0, num_real_slices=None):
    """
    Applies a cylindrical mask to a 3D reconstruction volume.

    This function zeros out all voxels outside a centered cylindrical region
    in the (row, col) plane and also zeroes a specified number of slices from
    the top and bottom along the Z-axis (slice axis).

    This function is useful for removing `flash` that typically accumulates on the boundaries of an MBIR reconstruction volume.

    Note:
        A numpy recon is masked on the host and a jax recon on-device, so a large host volume is
        never shipped onto a single device (which would OOM for big recons).

    Args:
        recon (np.ndarray or jax.Array): 3D volume with shape (num_rows, num_cols, num_slices).
        radial_margin (int): Margin to subtract from the cylinder radius in pixels.
        top_margin (int): Number of top slices to set to zero along the Z-axis.
        bottom_margin (int): Number of bottom slices to set to zero along the Z-axis.
        num_real_slices (int or None): Number of REAL slices when ``recon`` is a device-form volume whose
            slice axis is zero-padded (see the recon placement).  The bottom margin is applied at the end
            of the real slices, not the padded end.  None (default) means all slices are real.

    Returns:
        np.ndarray or jax.Array: Masked 3D volume of the same shape and array module as `recon`.

    Example:
        >>> import jax.numpy as jnp
        >>> vol = jnp.ones((128, 128, 64))
        >>> masked_vol = apply_cylindrical_mask(vol,radial_margin=10,top_margin=4,bottom_margin=4)
        >>> masked_vol.shape
        (128, 128, 64)
    """
    # Use recon's own array module so a numpy recon stays on the host and a jax recon stays on-device.
    xp = jnp if isinstance(recon, jax.Array) else np

    num_recon_rows, num_recon_cols, num_slices = recon.shape
    row_center = (num_recon_rows - 1) / 2
    col_center = (num_recon_cols - 1) / 2

    base_radius = max(row_center, col_center)
    radius = base_radius - radial_margin

    # Create circular mask in (row, col) plane (small; built on recon's module).
    row_coords, col_coords = xp.meshgrid(xp.arange(num_recon_rows), xp.arange(num_recon_cols), indexing='ij')
    dist_sq = (row_coords - row_center) ** 2 + (col_coords - col_center) ** 2
    circular_mask = (dist_sq <= radius ** 2).astype(recon.dtype)

    # Circular mask: this multiply allocates the one new array we return; input is untouched.
    recon = recon * circular_mask[:, :, None]

    # Zero top/bottom margins in place on that new array (no second full-volume copy -> 2x not 3x).
    # numpy is truly in place; jax is immutable so use .at[].set (on-device, not the big path).
    # The bottom margin ends at the REAL slice count: on a slice-padded device-form volume, [-b:] would
    # zero the (already-zero) padding instead of the real bottom slices.
    num_real_slices = recon.shape[2] if num_real_slices is None else num_real_slices
    if xp is np:
        if top_margin > 0:
            recon[:, :, :top_margin] = 0
        if bottom_margin > 0:
            recon[:, :, num_real_slices - bottom_margin:num_real_slices] = 0
    else:
        if top_margin > 0:
            recon = recon.at[:, :, :top_margin].set(0)
        if bottom_margin > 0:
            recon = recon.at[:, :, num_real_slices - bottom_margin:num_real_slices].set(0)

    return recon

def est_crop_width(sino, safety_buffer=20, max_views_to_use=20):
    """Estimate crop widths for removing blank margins in a 3D sinogram.

    Args:
        sino (np.ndarray): Input sinogram array .
        safety_buffer (int, optional): Safety buffer (in pixels) to keep around the
            detected object region on each boundary. Defaults to 20.
        max_views_to_use (int, optional): Cap on the number of views sampled for the support
            estimate. Defaults to 20.

    Returns:
        crop_top (int): Number of detector rows to crop from the top.
        crop_bottom (int): Number of detector rows to crop from the bottom.
        crop_left (int): Number of detector channels to crop from the left.
        crop_right (int): Number of detector channels to crop from the right.

    """
    # The crop boundaries need full detector row/column resolution, but support detection is
    # statistical across views -- so subsample VIEWS (axis 0), exactly as auto_set_regularization_params
    # does, to keep this fast on large sinograms.  _get_sino_indicator on the full sinogram is several
    # full-array host passes (finiteness check, histogram, median of the object subset, the full int8
    # mask); on ~max_views_to_use views it is cheap.  The safety_buffer absorbs the small approximation
    # from sampling fewer views (the object's row/column extent is stable across the rotation).
    sino = mj.TomographyModel.subsample_views(sino, max_views_to_use)

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
            * **optional_params** (*dict*): Updated parameters with adjusted ``'det_row_offset'``
              and ``'det_channel_offset'``.  (Any other entries pass through untouched; call
              ``auto_set_recon_geometry()`` after applying the parameters so the recon grid
              tracks the adjusted offsets.)
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


def fit_beam_hardening_curve(linear_projection, target_projection, num_parameters=5, zero_offset_normalized=True):
    """
    Fit a parametric beam-hardening function from paired samples.

    Args:
        linear_projection (np.ndarray): Ideal linear projection or path-length
            samples.
        target_projection (np.ndarray): Target beam-hardened projection
            samples paired with ``linear_projection``.
        num_parameters (int, optional): Total number of fitted parameters.
            Defaults to 5.
        zero_offset_normalized (bool, optional): If True, use the normalized
            forward model with ``h(0) = 0``. If False, use the
            unnormalized log-sum-exp form. Defaults to True.

    Returns:
        ndarray: Optimized parameter vector
            ``[theta_0, theta_1, ..., theta_{num_parameters-1}]``.

    Example:
        >>> linear_projection = sinogram.ravel()
        >>> target_projection = sinogram_nonlinear.ravel()
        >>> fitted_params = fit_beam_hardening_curve(
        ...     linear_projection, target_projection, num_parameters=5)
        >>> y_pred = apply_beam_hardening_curve(
        ...     sinogram_test, fitted_params)
    """
    num_parameters = int(num_parameters)
    if num_parameters < 2:
        raise ValueError(
            'fit_beam_hardening_curve: num_parameters must be at least 2.')

    linear_projection = np.asarray(linear_projection, dtype=np.float64)
    target_projection = np.asarray(target_projection, dtype=np.float64)

    if linear_projection.size != target_projection.size:
        raise ValueError(
            'fit_beam_hardening_curve: Input and target projection arrays must contain the same number of samples.')

    linear_projection = linear_projection.ravel()
    target_projection = target_projection.ravel()

    valid_mask = (
        np.isfinite(linear_projection) & np.isfinite(target_projection)
        & (linear_projection > 1e-6) & (target_projection > 1e-6)
    )
    linear_projection = linear_projection[valid_mask]
    target_projection = target_projection[valid_mask]

    if linear_projection.size == 0:
        raise ValueError(
            'fit_beam_hardening_curve: No valid training samples remain.')

    initial_params = np.zeros(num_parameters, dtype=np.float64)
    initial_params[0] = 1.0

    max_nfev = 20 * num_parameters
    solver_options = dict(
        loss="linear",
        method="trf",
        max_nfev=max_nfev,
        ftol=1e-5,
        xtol=1e-5,
        gtol=1e-5,
        verbose=1,
    )

    optimization_result = scipy.optimize.least_squares(
        _beam_hardening_curve_residuals,
        initial_params,
        args=(linear_projection, target_projection, zero_offset_normalized),
        **solver_options,
    )

    if not optimization_result.success:
        warnings.warn(
            f'fit_beam_hardening_curve: Beam-hardening curve fit did not converge: {optimization_result.message}',
            RuntimeWarning)

    return optimization_result.x


def apply_beam_hardening_curve(linear_projection, params, zero_offset_normalized=True):
    """
    Apply a fitted parametric beam-hardening function.

    If ``zero_offset_normalized`` is True, this uses

        f(p) = log(sum_i exp(theta_i))
               - log(sum_i exp(theta_i - i * theta_0 * p)),

    which forces ``f(0) = 0``. If False, this uses the form

        f(p) = -log(sum_i exp(theta_i - i * theta_0 * p)).

    Args:
        linear_projection (np.ndarray): Linear projection values.
        params (np.ndarray): Parameter vector
            ``[theta_0, theta_1, ..., theta_N]``.
        zero_offset_normalized (bool, optional): Select the zero-normalized
            forward model. Defaults to True.

    Returns:
        ndarray: Beam-hardened projection values with the same shape as
            ``linear_projection``.
    """
    linear_projection = np.asarray(linear_projection, dtype=np.float64)
    params = np.asarray(params, dtype=np.float64).reshape(-1)

    if params.size < 2:
        raise ValueError(
            'Expected at least 2 parameters: theta_0 and one log-weight.')

    theta_0 = params[0]
    theta_rest = params[1:]

    log_sum_exp_p = np.full_like(linear_projection, -np.inf, dtype=np.float64)
    for i, theta_i in enumerate(theta_rest, start=1):
        exponent = theta_i - i * theta_0 * linear_projection
        log_sum_exp_p = np.logaddexp(log_sum_exp_p, exponent)

    if not zero_offset_normalized:
        return -log_sum_exp_p

    log_sum_exp_0 = -np.inf
    for theta_i in theta_rest:
        log_sum_exp_0 = np.logaddexp(log_sum_exp_0, theta_i)

    return log_sum_exp_0 - log_sum_exp_p


def _beam_hardening_curve_residuals(params, linear_projection, target_projection, zero_offset_normalized):
    """
    Return fitted-minus-target residuals for nonlinear least-squares fitting.

    Args:
        params (np.ndarray): Current beam-hardening
            model parameters.
        linear_projection (np.ndarray): Filtered linear projection samples.
        target_projection (np.ndarray): Filtered target beam-hardened samples.
        zero_offset_normalized (bool, optional): Select the zero-normalized forward model.

    Returns:
        ndarray: One-dimensional residual vector used by
            :func:`scipy.optimize.least_squares`.
    """
    fitted_projection = apply_beam_hardening_curve(
        linear_projection, params,
        zero_offset_normalized=zero_offset_normalized)

    return fitted_projection.ravel() - target_projection.ravel()


def fit_inverse_beam_hardening_curve(forward_params, vmin=0.0, vmax=5.0, degree=10, num_samples=2000, zero_offset_normalized=True):
    """
    Fit a Chebyshev inverse that linearizes beam-hardened projections.

    Args:
        forward_params (np.ndarray): Forward beam-hardening parameters from
            :func:`fit_beam_hardening_curve`.
        vmin (float, optional): Minimum input projection value to correct.
        vmax (float, optional): Maximum input projection value to correct.
        degree (int, optional): Chebyshev polynomial degree. Defaults to 10.
        num_samples (int, optional): Number of fitting samples.
        zero_offset_normalized (bool, optional): Match the forward model
            normalization used to fit ``forward_params``.

    Returns:
        tuple: ``(cheb_coeffs, y_domain)`` where ``cheb_coeffs`` is an
            ndarray of length ``degree + 1`` and ``y_domain`` is ``(vmin,
            vmax)`` for later inverse evaluation.

    Example:
        >>> forward_params = fit_beam_hardening_curve(
        ...     sinogram.ravel(), sinogram_nonlinear.ravel(),
        ...     num_parameters=5)
        >>> cheb_coeffs, y_domain = fit_inverse_beam_hardening_curve(
        ...     forward_params,
        ...     vmin=0.0,
        ...     vmax=float(sinogram_nonlinear.max()),
        ...     degree=10)
        >>> sinogram_linearized = apply_inverse_beam_hardening_curve(
        ...     sinogram_nonlinear, cheb_coeffs, y_domain)
    """
    forward_params = np.asarray(forward_params, dtype=np.float64).reshape(-1)
    vmin = float(vmin)
    vmax = float(vmax)
    degree = int(degree)
    num_samples = int(num_samples)

    if not (np.isfinite(vmin) and np.isfinite(vmax)) or vmax <= vmin:
        raise ValueError(
            'fit_inverse_beam_hardening_curve: require finite vmin < vmax.')
    if degree < 1:
        raise ValueError(
            'fit_inverse_beam_hardening_curve: degree must be at least 1.')
    if num_samples < degree + 1:
        raise ValueError(
            'fit_inverse_beam_hardening_curve: num_samples must be at least '
            'degree + 1.')

    # estimate effective attenuation (h'(0))
    epsilon = 1e-6
    forward_at_zero = apply_beam_hardening_curve(
        0.0, forward_params,
        zero_offset_normalized=zero_offset_normalized)

    forward_at_epsilon = apply_beam_hardening_curve(
        epsilon, forward_params,
        zero_offset_normalized=zero_offset_normalized)

    effective_attenuation = float(
        (forward_at_epsilon - forward_at_zero) / epsilon)

    path_min = 0.0
    path_max = max(path_min + 1.0, abs(vmax) + 1.0)
    # Each pass doubles path_max; this guard prevents an infinite expansion loop.
    max_expand_iterations = 64
    for _ in range(max_expand_iterations):
        y_at_path_max = apply_beam_hardening_curve(
            path_max, forward_params,
            zero_offset_normalized=zero_offset_normalized)
        if np.isfinite(y_at_path_max) and y_at_path_max >= vmax:
            break
        path_max = path_min + 2.0 * (path_max - path_min)
    else:
        raise ValueError(
            'fit_inverse_beam_hardening_curve: could not expand the path '
            'length grid enough to cover vmax.')

    p_grid = np.linspace(
        path_min, path_max, 4 * num_samples, dtype=np.float64)
    y_grid = apply_beam_hardening_curve(
        p_grid, forward_params,
        zero_offset_normalized=zero_offset_normalized)

    valid_mask = np.isfinite(p_grid) & np.isfinite(y_grid)
    p_grid = p_grid[valid_mask]
    y_grid = y_grid[valid_mask]
    if p_grid.size < degree + 1:
        raise ValueError(
            'fit_inverse_beam_hardening_curve: too few finite forward '
            'samples.')

    sort_idx = np.argsort(y_grid)
    y_sorted = y_grid[sort_idx]
    p_sorted = p_grid[sort_idx]
    y_unique, unique_idx = np.unique(y_sorted, return_index=True)
    p_unique = p_sorted[unique_idx]

    if y_unique.size < degree + 1 or y_unique[-1] <= y_unique[0]:
        raise ValueError(
            'fit_inverse_beam_hardening_curve: forward model is not '
            'invertible on the sampled grid.')
    if vmax > y_unique[-1]:
        raise ValueError(
            'fit_inverse_beam_hardening_curve: sampled forward '
            f'model only reaches {y_unique[-1]:.6g}, below vmax={vmax:.6g}.')
    if vmin < y_unique[0]:
        warnings.warn(
            'fit_inverse_beam_hardening_curve: vmin is below the forward '
            'value at zero path length; low-end inverse samples will be '
            'clamped to zero.',
            RuntimeWarning)

    y_samples = np.linspace(vmin, vmax, num_samples, dtype=np.float64)
    p_samples = np.interp(
        y_samples, y_unique, p_unique,
        left=p_unique[0], right=p_unique[-1])
    linearized_projection_samples = p_samples * effective_attenuation

    y_scaled = 2.0 * (y_samples - vmin) / (vmax - vmin) - 1.0
    cheb_coeffs = np.polynomial.chebyshev.chebfit(
        y_scaled, linearized_projection_samples, deg=degree)

    return cheb_coeffs, (vmin, vmax)


def apply_inverse_beam_hardening_curve(beam_hardened_projection, cheb_coeffs, y_domain, clip=False):
    """
    Apply a fitted Chebyshev inverse to linearize projection values.

    Args:
        beam_hardened_projection (np.ndarray): Beam-hardened projection
            values. Arrays of any shape are accepted and the output preserves
            that shape.
        cheb_coeffs (np.ndarray): Coefficients returned by
            :func:`fit_inverse_beam_hardening_curve`.
        y_domain (tuple): ``(vmin, vmax)`` projection range used for fitting.
        clip (bool, optional): If True, clip input values into ``y_domain``
            before evaluation. If False, warn when extrapolating. Defaults to
            False.

    Returns:
        ndarray: Linearized projection values with the same shape as
            ``beam_hardened_projection``.
    """
    beam_hardened_projection = np.asarray(
        beam_hardened_projection, dtype=np.float64)
    cheb_coeffs = np.asarray(cheb_coeffs, dtype=np.float64).reshape(-1)
    y_min, y_max = float(y_domain[0]), float(y_domain[1])

    if y_max <= y_min:
        raise ValueError(
            'apply_inverse_beam_hardening_curve: y_domain must '
            'satisfy y_max > y_min.')

    if clip:
        y_eval = np.clip(beam_hardened_projection, y_min, y_max)
    else:
        if (np.any(beam_hardened_projection < y_min)
                or np.any(beam_hardened_projection > y_max)):
            warnings.warn(
                'apply_inverse_beam_hardening_curve: inputs lie '
                'outside the fitted y_domain; extrapolated values may be '
                'unreliable.',
                RuntimeWarning)
        y_eval = beam_hardened_projection

    y_scaled = 2.0 * (y_eval - y_min) / (y_max - y_min) - 1.0
    return np.polynomial.chebyshev.chebval(y_scaled, cheb_coeffs)


def _zinger_threshold(sino, zinger_pixel_ratio, max_views_to_use=20):
    """Zinger-detection threshold = ``-zinger_pixel_ratio * RMS(sino over its support)``.

    The threshold is a STATISTICAL quantity, so it is estimated from a view subsample (via
    ``TomographyModel.subsample_views``) -- pass the FULL sinogram; the subsampling happens here, which
    avoids running ``_get_sino_indicator`` + ``sino**2`` over the whole (possibly ~20 GB) sinogram.
    Returns a Python float (negative; zingers are anomalously negative)."""
    sino_sub = mj.TomographyModel.subsample_views(sino, max_views_to_use)
    sino_indicator = mj.TomographyModel._get_sino_indicator(sino_sub)
    typical_sino_value = float(np.average(sino_sub ** 2, None, sino_indicator) ** 0.5)
    return -zinger_pixel_ratio * typical_sino_value


def detect_zinger_pixels(sino, zinger_pixel_ratio=0.1, max_views_to_use=20):
    """
    Detect zinger pixels from sinogram.

    Zinger pixels are identified as unusually negative values. A pixel is
    classified as a zinger if:

        value < -zinger_pixel_ratio * typical_sino_value

    Args:
        sino (numpy.ndarray): A 3D sinogram of shape (num_views, num_det_rows, num_det_channels).
        zinger_pixel_ratio (float, optional): Ratio used for zinger pixels detection. Defaults to 0.1.
        max_views_to_use (int, optional): Cap on the number of views sampled for the typical-value
            (threshold) estimate. Defaults to 20.

    Returns:
        ndarray:
            Array of zinger pixel indices with shape (num_zinger_pixels, 3). Format in (view_idx, row_idx, channel_idx).
    """
    # Estimate the threshold from a view subsample (statistical; done inside _zinger_threshold), then
    # DETECT across ALL views (a zinger can occur in any view).
    zinger_threshold = _zinger_threshold(sino, zinger_pixel_ratio, max_views_to_use)
    zinger_pixel_array = np.argwhere(np.asarray(sino) < zinger_threshold).astype(int)

    return zinger_pixel_array.reshape((-1, 3))


def interpolate_zinger_pixels(sino, zinger_pixel_ratio=0.1, num_passes=3, batch_size=90, devices=None,
                              max_views_to_use=20):
    """
    Detect and interpolate zinger sinogram entries (anomalously negative values) with the MEAN of their
    finite 3x3 in-view neighbors.

    A pixel is a zinger if ``value < -zinger_pixel_ratio * RMS(sino over support)`` (the threshold is
    estimated once from a cheap view subsample).  Detection + fill run **per view-batch through the
    shared pipeline driver** -- memory-bounded and optionally multi-device -- reusing the same
    :func:`_zinger_fill` / :func:`_fill_nan_pixels` machinery as :func:`interpolate_defective_pixels`
    (dense ``reduce_window`` passes; no ``argwhere``, no data-dependent loop).  This is a thin wrapper:
    the same zinger correction is also available folded into :func:`scan_to_sino` via its
    ``zinger_pixel_ratio`` argument (one pass, no extra host round-trip).

    Args:
        sino (numpy or jax array): A 3D sinogram of shape (num_views, num_det_rows, num_det_channels).
        zinger_pixel_ratio (float, optional): Ratio used for zinger detection. Defaults to 0.1.
        num_passes (int, optional): Dense fill passes = max fillable zinger-cluster radius. Defaults to 3.
        batch_size (int, optional): Views per on-device batch. Defaults to 90.
        devices (sequence or None, optional): Devices to spread the views over; None = single device.
        max_views_to_use (int, optional): Views sampled for the threshold estimate. Defaults to 20.

    Returns:
        numpy.ndarray: Corrected 3D sinogram; any pixel still NaN after ``num_passes`` is set to 0 (with
        a warning).
    """
    zinger_threshold = _zinger_threshold(sino, zinger_pixel_ratio, max_views_to_use)
    kernel = jax.jit(lambda b: _zinger_fill(b, zinger_threshold, num_passes))
    return pipeline.map_view_batches(sino, kernel, batch_size, devices=devices)


def save_preprocessing(file_path, sinogram, cone_beam_params, optional_params, weights=None):
    """Save a preprocessed sinogram and its geometry parameters for a two-stage (preprocess -> recon)
    workflow.

    This lets preprocessing run once and write its result to disk, so reconstruction can be launched as
    a separate process -- useful for debugging (inspect/reuse the preprocessed sinogram) and so the
    memory-tight recon starts with a clean GPU allocator.  Reload with :func:`load_preprocessing` and
    rebuild the model with ``ConeBeamModel(**cone_beam_params)`` + ``set_params(**optional_params)``.

    Only the GEOMETRY/preprocessing parameters are saved (the two dicts above); regularization
    parameters (sharpness, sigma_x, snr_db, ...) are deliberately NOT saved -- they are a recon-time
    choice and are set in the recon stage (iterating on them without re-preprocessing is the point of
    the split).  For full model persistence (geometry + regularization), use the parameter-handler
    save/load instead.

    The sinogram, any array-valued parameters (e.g. ``angles``), and ``weights`` (if given) are stored
    as HDF5 datasets; the remaining scalar parameters are stored as one JSON attribute.  The sinogram
    (and weights) are written as float32.

    Args:
        file_path (str): Output HDF5 path (parent directories are created).
        sinogram (ndarray or jax.Array): The preprocessed sinogram (gathered to the host and cast to
            float32 before writing).
        cone_beam_params (dict): Parameters for the model constructor (as returned by, e.g.,
            ``mbirjax.preprocess.nsi.compute_sino_and_params``).
        optional_params (dict): Parameters applied via ``set_params`` after construction.
        weights (ndarray or jax.Array, optional): Custom reconstruction weights to save alongside the
            sinogram.  Defaults to None -- omit them when the recon will regenerate standard weights with
            ``gen_weights`` (cheaper to recompute than to store a second full-size array); pass them only
            when they are custom and worth preserving.

    Returns:
        None
    """
    import json
    mj.makedirs(file_path)

    def _is_array(v):
        # numpy/jax arrays (and other ndim>0 array-likes) -> datasets; scalars/tuples/str -> JSON.
        return hasattr(v, 'ndim') and getattr(v, 'ndim', 0) > 0

    scalar_params = {'cone_beam_params': {}, 'optional_params': {}}
    with h5py.File(file_path, 'w') as f:
        f.create_dataset('sinogram', data=np.asarray(sinogram, dtype=np.float32))
        if weights is not None:
            f.create_dataset('weights', data=np.asarray(weights, dtype=np.float32))
        for dname, d in (('cone_beam_params', cone_beam_params), ('optional_params', optional_params)):
            for key, value in d.items():
                if _is_array(value):
                    f.create_dataset('{}__{}'.format(dname, key), data=np.asarray(value))
                else:
                    scalar_params[dname][key] = value
        # numpy scalars -> Python via .item(); tuples -> JSON lists (sinogram_shape restored on load).
        f.attrs['params'] = json.dumps(scalar_params, default=lambda o: o.item())
        f.attrs['format'] = 'mbirjax_preprocessing_v1'


def load_preprocessing(file_path):
    """Load a sinogram and geometry parameters saved by :func:`save_preprocessing`.

    Args:
        file_path (str): Path to the HDF5 file written by :func:`save_preprocessing`.

    Returns:
        tuple: ``(sinogram, cone_beam_params, optional_params, weights)`` -- a host NumPy sinogram, the
        two parameter dicts (ready for ``ConeBeamModel(**cone_beam_params)`` +
        ``set_params(**optional_params)`` + ``auto_set_recon_geometry()``), and ``weights`` (a host
        NumPy array if custom weights were saved, else ``None`` -- in which case the recon should
        regenerate them with ``gen_weights``).
    """
    import json
    params = {'cone_beam_params': {}, 'optional_params': {}}
    with h5py.File(file_path, 'r') as f:
        sinogram = f['sinogram'][()]
        weights = f['weights'][()] if 'weights' in f else None
        scalar_params = json.loads(f.attrs['params'])
        for dname in params:
            params[dname].update(scalar_params.get(dname, {}))
            prefix = dname + '__'
            for ds_name in f:
                if ds_name.startswith(prefix):
                    params[dname][ds_name[len(prefix):]] = f[ds_name][()]
    cone_beam_params, optional_params = params['cone_beam_params'], params['optional_params']
    # JSON turns the sinogram_shape tuple into a list; restore the tuple the constructor expects.
    if 'sinogram_shape' in cone_beam_params:
        cone_beam_params['sinogram_shape'] = tuple(int(x) for x in cone_beam_params['sinogram_shape'])
    return sinogram, cone_beam_params, optional_params, weights