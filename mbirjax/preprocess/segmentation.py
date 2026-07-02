import functools
import numpy as np
import jax
import jax.numpy as jnp
import mbirjax.preprocess as mjp


def _masked_histogram(image, valid_mask, num_bins, xp):
    """Histogram of the valid entries of ``image``, matching ``histogram(image, num_bins,
    range=(min, max))`` computed over the valid entries only.

    The range comes from masked min/max, and invalid entries are pushed to a finite sentinel ABOVE the
    range, which ``histogram``'s range semantics then drop -- so the bin EDGES and counts are exactly
    those of the valid entries (no post-hoc count correction needed).  ``xp`` is the array module:
    ``np`` runs eagerly on the host; ``jnp`` runs on-device (see :func:`_sharded_histogram` for the
    jitted wrapper).  The infinity/one constants are typed to ``image.dtype`` so the ``where`` cannot
    silently upcast (a float64 scalar would double the full-size temporaries).
    """
    inf = xp.asarray(xp.inf, dtype=image.dtype)
    lo = xp.min(xp.where(valid_mask, image, inf))
    hi = xp.max(xp.where(valid_mask, image, -inf))
    sentinel = hi + xp.maximum(xp.abs(hi), xp.asarray(1.0, dtype=image.dtype))  # finite, strictly > hi
    return xp.histogram(xp.where(valid_mask, image, sentinel), bins=num_bins, range=(lo, hi))


@functools.partial(jax.jit, static_argnums=2)   # num_bins sets the output shape -> compile constant
def _sharded_histogram(image, valid_mask, num_bins=1024):
    """Jitted :func:`_masked_histogram` for a (possibly view/slice-sharded) jax volume.

    A histogram is sum-decomposable (``hist(x) = sum over shards of hist(x_shard)``, integer counts), so
    on a sharded array XLA compiles this to per-shard partial histograms + an all-reduce of the tiny
    ``(num_bins,)`` output -- verified 0 all-gathers.  min/max are likewise exact all-reduces (order
    insensitive), so the result is bit-identical to the host computation on the valid entries.

    Returns (hist, bin_edges) as device arrays (tiny; callers convert to host).
    """
    return _masked_histogram(image, valid_mask, num_bins, jnp)


def multi_threshold_otsu(image, classes=2, num_bins=1024, valid_mask=None):
    """
    Segment an image into multiple intensity classes using Otsu's method.

    This function computes optimal threshold values that divide an image into the specified
    number of classes by minimizing the intra-class variance. It returns `classes - 1` thresholds
    that can be used to partition the image intensity range into `classes` distinct segments.

    A NumPy image is histogrammed on the host; a JAX image is histogrammed on its own device(s) --
    including a sharded volume, whose per-shard partial histograms combine in a cross-device reduction
    (bit-identical counts; no gather of the volume).  Only the tiny histogram itself comes to the host
    for the threshold search.

    Args:
        image (np.ndarray or jax.Array):
            Input image of floating-point values.
        classes (int, optional):
            Number of classes to divide the image into. Must be ≥ 2. Defaults to 2.
        num_bins (int, optional):
            Number of bins to use when constructing the image histogram. Defaults to 256.
        valid_mask (array or None, optional):
            Broadcastable boolean mask, True on the entries to include (applied uniformly for numpy and
            jax inputs).  Used e.g. to exclude the zero-padded entries of a device-form (sharded) volume
            so the histogram range and counts match the unpadded volume exactly.  None includes
            everything.

    Returns:
        list of float:
            A list of `classes - 1` threshold values, given in increasing order. These thresholds
            can be used to separate the image into `classes` distinct intensity regions.

    Example:
        >>> thresholds = multi_threshold_otsu(image, classes=4)
        >>> # Resulting thresholds will split image into 4 intensity regions
    """
    if classes < 2:
        raise ValueError("Number of classes must be at least 2")

    if num_bins < classes:
        raise ValueError("Number of bins must be at least equal to number of classes")

    # Compute the histogram of the valid entries: on-device (sharded-safe) for a jax image, host for
    # numpy -- the same masked semantics either way.
    if isinstance(image, jax.Array):
        if valid_mask is None:
            valid_mask = jnp.ones((1,) * image.ndim, dtype=bool)
        hist, bin_edges = _sharded_histogram(image, valid_mask, num_bins)
        hist, bin_edges = np.array(hist), np.array(bin_edges)   # tiny (num_bins,) transfers
    elif valid_mask is not None:
        hist, bin_edges = _masked_histogram(image, np.asarray(valid_mask), num_bins, np)
    else:
        hist, bin_edges = np.histogram(image, bins=num_bins, range=(np.min(image), np.max(image)))

    # Find the optimal thresholds using a recursive approach
    thresholds = _recursive_otsu(hist, classes - 1)

    # Convert histogram bin indices to original image values
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    scaled_thresholds = [bin_centers[t] for t in thresholds]
    # print(scaled_thresholds)

    # import matplotlib.pyplot as plt
    # plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), edgecolor="black", align="edge")
    # plt.show(block=True)
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


def segment_plastic_metal(recon, num_metal, radial_margin=10, top_margin=10, bottom_margin=10,
                          valid_mask=None, num_real_slices=None):
    """
    Segment a reconstruction into plastic and multiple metal masks using multi-threshold Otsu.

    ``recon`` may be a host array, a single-device jax array, or a **sharded** device-form volume; the
    segmentation runs on whatever devices hold it (the Otsu histogram reduces across shards without
    gathering the volume).  For a device-form volume whose slice axis is zero-padded, pass
    ``valid_mask`` / ``num_real_slices`` so the padded slices are excluded from the histogram, the
    bottom margin lands on the real bottom slices, and the class masks are zero on padding (otherwise a
    threshold interval spanning 0 would include the padded voxels and bias the scaling factors).

    Args:
        recon (np.ndarray or jax.Array): Reconstructed volume array (host, single-device, or sharded).
        num_metal (int): Number of metal materials to segment.
        radial_margin (int, optional): Margin in pixels to subtract from the cylindrical mask radius.
        top_margin (int, optional): Number of slices to mask out from the top of the volume.
        bottom_margin (int, optional): Number of slices to mask out from the bottom of the volume.
        valid_mask (jax array or None, optional): Broadcastable boolean mask, True on real voxels.
        num_real_slices (int or None, optional): Real slice count (see ``apply_cylindrical_mask``).

    Returns:
        Tuple[jnp.ndarray, List[jnp.ndarray], float, List[float]]:
            - plastic_mask (jnp.ndarray): Binary mask for plastic regions.
            - metal_masks (List[jnp.ndarray]): List of binary masks for each metal region.
            - plastic_scale (float): Scaling factor for plastic region.
            - metal_scales (List[float]): List of scaling factors for each metal region.
    """
    if num_metal <= 0:
        raise ValueError("num_metal must be positive")

    # Remove any flash from the boundary of the recon
    recon = mjp.apply_cylindrical_mask(recon, radial_margin=radial_margin, top_margin=top_margin,
                                       bottom_margin=bottom_margin, num_real_slices=num_real_slices)

    # Compute thresholds using multi-threshold Otsu (padded voxels excluded via valid_mask)
    thresholds = multi_threshold_otsu(recon, classes=num_metal + 2, valid_mask=valid_mask)

    # Plastic: lowest class
    plastic_low_threshold = thresholds[0]
    plastic_metal_threshold = thresholds[1]

    # Masks are 1 inside the class interval AND on a real voxel: padded voxels are exactly 0, so a class
    # interval spanning 0 would otherwise mark them, biasing compute_scaling_factor's denominator.
    def class_mask(lower, upper):
        in_class = (recon > lower) & (recon <= upper)
        if valid_mask is not None:
            in_class = in_class & valid_mask
        return jnp.where(in_class, 1.0, 0.0)

    plastic_mask = class_mask(plastic_low_threshold, plastic_metal_threshold)
    plastic_scale = mjp.compute_scaling_factor(recon, plastic_mask)

    # Metal masks and scaling
    metal_masks = []
    metal_scales = []
    for i in range(1, num_metal + 1):  # start from index 1
        lower = thresholds[i]
        upper = thresholds[i + 1] if i + 1 < len(thresholds) else jnp.inf
        metal_mask = class_mask(lower, upper)
        metal_masks.append(metal_mask)
        metal_scales.append(mjp.compute_scaling_factor(recon, metal_mask))

    return plastic_mask, metal_masks, plastic_scale, metal_scales



