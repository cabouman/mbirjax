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
            Number of bins to use when constructing the image histogram. Defaults to 1024.
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

    # Find the optimal thresholds (half-open class-boundary bin indices) by dynamic programming
    thresholds = _otsu_thresholds_dp(hist, classes - 1)

    # Convert boundary indices to image values.  bin_edges[t] is the exact cut for boundary t: values
    # below it fall precisely in bins < t (the lower classes), matching the histogram split.
    scaled_thresholds = [bin_edges[t] for t in thresholds]

    return scaled_thresholds


def _otsu_thresholds_dp(hist, num_thresholds):
    """
    Multi-threshold Otsu via dynamic programming.

    Otsu's criterion minimizes the total within-class variance: for thresholds t_1 < ... < t_k, class
    ``c`` spans the bin interval ``[t_c, t_{c+1})`` (with t_0 = 0 and t_{k+1} = num_bins) and the
    objective is

        sum over classes of   sum_{i in class} (i - class mean)^2 * hist[i].

    The objective is separable over the classes, so the optimal thresholds solve the classic 1-D
    segmentation DP

        D[c][b] = min over s < b of  D[c-1][s] + cost(s, b),

    where ``cost(a, b)`` is the within-class term of a single class spanning bins ``[a, b)``.  The cost
    is O(1) from prefix sums of the histogram's zeroth/first/second moments, each DP stage is one
    vectorized (B+1)^2 min-reduction, and the thresholds come from an argmin backtrack: O(k B^2)
    float64 NumPy with no recursion and no per-bin Python loops, exact for every ``num_thresholds``.

    Threshold convention: returned values are half-open class boundaries -- threshold ``t`` means bin
    ``t`` starts the next class.  The consistent threshold VALUE is therefore the left bin edge
    ``bin_edges[t]``: values below it fall exactly in bins < t (the lower classes).

    Args:
        hist (ndarray): Histogram of the image (counts; any nonnegative dtype).
        num_thresholds (int): Number of thresholds to find (k = classes - 1).

    Returns:
        list of int: strictly increasing boundary indices in ``[1, len(hist) - 1]``.
    """
    if num_thresholds == 0:
        return []

    hist = np.asarray(hist, dtype=np.float64)
    num_bins = len(hist)
    # Bin coordinates centered at the histogram midpoint: the within-class variance is shift-invariant,
    # and centering shrinks the moment magnitudes (~4x on the second moment), reducing cancellation in
    # the prefix-difference arithmetic below.  float64 is required regardless: the raw second moment
    # reaches ~bin^2 * count ~ 1e15 for a 1e9-voxel volume, far beyond float32/int32.
    bin_coord = np.arange(num_bins, dtype=np.float64) - (num_bins - 1) / 2.0

    # Moment prefix sums, each with a leading 0 so that P[j] = (sum over bins i < j) and the moment of
    # any bin interval [a, b) is P[b] - P[a].  m0 = counts, m1 = first moment, m2 = second moment.
    m0 = np.concatenate(([0.0], np.cumsum(hist)))
    m1 = np.concatenate(([0.0], np.cumsum(bin_coord * hist)))
    m2 = np.concatenate(([0.0], np.cumsum(bin_coord * bin_coord * hist)))

    # Moments of every candidate class interval at once: outer differences, entry [a, b] = P[b] - P[a]
    # = the moment of bins [a, b).
    int_m0 = m0[None, :] - m0[:, None]
    int_m1 = m1[None, :] - m1[:, None]
    int_m2 = m2[None, :] - m2[:, None]

    # Within-class cost of the interval [a, b): expanding sum (i - mean)^2 h_i with mean = M1/M0 gives
    # M2 - M1^2/M0.  Empty (zero-count) intervals cost 0 (an empty class contributes no variance);
    # structurally invalid entries (a >= b) are +inf so the argmin can never produce non-increasing
    # boundaries.
    mean_sq_term = np.divide(int_m1 ** 2, int_m0, out=np.zeros_like(int_m0), where=int_m0 > 0)
    cost = np.maximum(int_m2 - mean_sq_term, 0.0)      # clip tiny negative rounding residue
    invalid = ~np.triu(np.ones((num_bins + 1, num_bins + 1), dtype=bool), k=1)   # a >= b
    cost[invalid] = np.inf

    # DP stages: best[b] = minimal cost of covering bins [0, b) with the current number of classes.
    # Each stage adds one class: total[s, b] = (best cover of [0, s) so far) + (one new class [s, b));
    # the argmin over s is recorded per b for the backtrack.
    best = cost[0, :].copy()                           # one class: [0, b)
    split_of = np.zeros((num_thresholds, num_bins + 1), dtype=np.int64)
    for stage in range(num_thresholds):
        total = best[:, None] + cost                   # total[s, b]
        split_of[stage] = np.argmin(total, axis=0)
        best = np.min(total, axis=0)

    # Backtrack from the full range [0, num_bins): the last stage's argmin at b = num_bins is the last
    # threshold; each recovered threshold then indexes the previous stage's argmin row.
    boundaries = []
    b = num_bins
    for stage in range(num_thresholds - 1, -1, -1):
        b = int(split_of[stage][b])
        boundaries.append(b)
    return boundaries[::-1]


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



