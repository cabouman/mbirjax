import functools
import math
import numpy as np
import jax
import jax.numpy as jnp
import mbirjax.preprocess as mjp


def _masked_histogram(image, valid_mask, num_bins, xp):
    """Histogram of the valid entries of ``image``, matching ``histogram(image, num_bins,
    range=(min, max))`` computed over the valid entries only.

    The range comes from masked min/max, and invalid entries are pushed to a finite sentinel ABOVE the
    range, which ``histogram``'s range semantics then drop -- so the bin EDGES and counts are exactly
    those of the valid entries (no post-hoc count correction needed).  ``xp`` is the array module
    (currently always ``np``, on the host).  The infinity/one constants are typed to ``image.dtype``
    so the ``where`` cannot
    silently upcast (a float64 scalar would double the full-size temporaries).
    """
    inf = xp.asarray(xp.inf, dtype=image.dtype)
    lo = xp.min(xp.where(valid_mask, image, inf))
    hi = xp.max(xp.where(valid_mask, image, -inf))
    sentinel = hi + xp.maximum(xp.abs(hi), xp.asarray(1.0, dtype=image.dtype))  # finite, strictly > hi
    # Python-float range endpoints: numpy computes the bin edges in the RANGE's dtype, so float
    # endpoints give the f64-computed, image-dtype-cast edges -- the same edges the sharded path
    # computes -- whereas f32 endpoints would shift some edges by a few ULP.
    return xp.histogram(xp.where(valid_mask, image, sentinel), bins=num_bins, range=(float(lo), float(hi)))


@jax.jit
def _masked_min_max(image, valid_mask):
    """Masked min/max of a (possibly sharded) volume -- pure reductions (cross-device all-reduce,
    no gathers).  Constants typed to the image dtype so the where cannot upcast."""
    inf = jnp.asarray(jnp.inf, dtype=image.dtype)
    lo = jnp.min(jnp.where(valid_mask, image, inf))
    hi = jnp.max(jnp.where(valid_mask, image, -inf))
    return lo, hi


@functools.partial(jax.jit, static_argnums=4)
def _local_bucketize_histogram(image, valid_mask, lo, hi, num_bins):
    """Histogram of one single-device slab by explicit bucketize + scatter-add (see _sharded_histogram).

    Semantics match ``np.histogram`` over the valid entries with ``range=(lo, hi)``: in-range values
    map to bin ``floor(num_bins * (x - lo) / span)``, ``hi`` itself lands in the last bin (numpy's
    closed last edge), and invalid or out-of-range entries are dropped.

    Sizes are bounded by construction: callers pass slabs of at most ``_HISTOGRAM_SLAB_ELEMENTS``
    elements, so the int32 counts cannot wrap (a bin cannot exceed the slab size < 2^31) and the
    flattened index below is a short, single-device axis -- the >2^31 flat-index hazards do not apply.
    The exact int64 total across slabs is accumulated on the host by the caller.

    Args:
        image (jax array): one slab of the volume, resident on a single device.
        valid_mask (array): boolean mask, broadcastable against ``image``; True on entries to count.
        lo, hi (scalars): histogram range, typed to ``image.dtype`` by the caller.
        num_bins (int): number of bins (static -- it sets the output shape).

    Returns:
        jax array: (num_bins,) int32 counts, on the slab's device.
    """
    # Bin index for every entry: map [lo, hi] linearly onto [0, num_bins], then clip so that x == hi
    # falls in the last bin.  Out-of-range values also clip into [0, num_bins - 1]; their (arbitrary)
    # index is harmless because they contribute 0 below.  The tiny floor on span guards lo == hi
    # (a constant slab).
    span = jnp.maximum(hi - lo, jnp.asarray(jnp.finfo(image.dtype).tiny, dtype=image.dtype))
    bin_index = jnp.clip((num_bins * (image - lo) / span).astype(jnp.int32), 0, num_bins - 1)

    # Each entry contributes 1 if it is valid and inside [lo, hi], else 0.
    in_range = (image >= lo) & (image <= hi)
    contribution = (valid_mask & in_range).astype(jnp.int32)

    # Scatter-add the contributions into the per-bin counts.
    counts = jnp.zeros(num_bins, dtype=jnp.int32)
    return counts.at[bin_index.reshape(-1)].add(contribution.reshape(-1))


# Per-slab element budget for the on-device histogram: bounds the bucketize's int32 index/contribution
# temporaries (~2 GB at 2^28) and, more importantly, guarantees every slab's int32 counts are EXACT
# (a slab bin cannot exceed the slab size < 2^31).
_HISTOGRAM_SLAB_ELEMENTS = 2 ** 28


def _iter_local_blocks(image, valid_mask):
    """Yield ``(block, mask_block)``: each device's LOCAL shard of ``image`` (device-resident jax
    array; no cross-device movement) with the matching broadcastable piece of ``valid_mask``.

    Shards are deduplicated by their global index so a replicated (or single-device) array yields its
    data exactly once -- nothing is double counted."""
    mask = np.asarray(valid_mask)
    seen = set()
    for shard in image.addressable_shards:
        key = tuple((s.start, s.stop, s.step) for s in shard.index)
        if key in seen:
            continue
        seen.add(key)
        # The mask follows the shard's global slice only on axes where it has real extent; its size-1
        # (broadcast) axes are kept whole.
        sel = tuple(idx if mask.shape[i] != 1 else slice(None) for i, idx in enumerate(shard.index))
        yield shard.data, mask[sel]


def _iter_slabs(block, mask_block):
    """Split a local block into leading-axis slabs of at most ``_HISTOGRAM_SLAB_ELEMENTS`` elements
    (the mask slabs along too, unless broadcast on that axis)."""
    per_row = max(1, math.prod(block.shape[1:]))   # exact Python ints (np.prod can wrap on Windows)
    rows_per_slab = max(1, _HISTOGRAM_SLAB_ELEMENTS // per_row)
    for j in range(0, block.shape[0], rows_per_slab):
        mask_slab = mask_block if mask_block.shape[0] == 1 else mask_block[j:j + rows_per_slab]
        yield block[j:j + rows_per_slab], mask_slab


def _sharded_histogram(image, valid_mask, num_bins=1024):
    """Masked histogram of a (possibly view/slice-sharded) jax volume, matching
    ``np.histogram(valid entries, num_bins, range=(min, max))``, with EXACT int64 counts.

    A histogram is sum-decomposable (``hist(x) = sum over pieces of hist(piece)``, integer counts), and
    the decomposition is enforced explicitly: each device's LOCAL shard block is histogrammed on its own
    device in slabs (int32, exact within a slab), and the tiny ``(num_bins,)`` partials are combined on
    the HOST in int64 -- exact at any volume size, with no cross-device collectives at all (the masked
    min/max combine on the host too; min/max are order-free).  All slabs are dispatched asynchronously
    before any result is read, so the devices overlap without threads.

    Two approaches are deliberately NOT used: (1) ``jnp.histogram`` or a global ``.at[idx].add`` scatter
    on the sharded array -- GSPMD does not partition scatter and lowers both with all-gathers of the
    IMAGE-SIZED index/update arrays onto every device (observed as a 47 GiB allocation on an ~18 GiB
    sharded recon); (2) ``shard_map`` -- it invokes XLA's SPMD partitioner, whose lowering has bitten
    this codebase before (see ``experiments/sharding/parallel_performance/fbp_parallel_options.md``);
    the per-device local pattern used here matches the fbp filter and the preprocessing driver.

    Semantics vs ``np.histogram``: range = masked min/max; invalid/out-of-range entries dropped; ``hi``
    lands in the last (closed) bin.  A value exactly on an interior bin edge can differ from numpy's
    edge arithmetic by one bin (float rounding of the scaled index) -- irrelevant at Otsu's bin
    granularity.

    Returns:
        (hist, bin_edges): host numpy arrays -- int64 counts and float64 edges (as ``np.histogram``).
    """
    blocks = list(_iter_local_blocks(image, valid_mask))

    # Pass 1: masked min/max per slab; dispatch everything, then read; combine exactly on the host.
    pending = [_masked_min_max(slab, mask_slab)
               for block, mask_block in blocks for slab, mask_slab in _iter_slabs(block, mask_block)]
    lo = min(float(lo_hi[0]) for lo_hi in pending)
    hi = max(float(lo_hi[1]) for lo_hi in pending)

    # Pass 2: bucketize + scatter per slab (int32, exact within a slab); accumulate on the host in
    # int64.  lo/hi are passed as image-dtype scalars so the kernel's arithmetic stays in that dtype.
    lo_dev = np.asarray(lo, dtype=image.dtype)
    hi_dev = np.asarray(hi, dtype=image.dtype)
    pending = [_local_bucketize_histogram(slab, mask_slab, lo_dev, hi_dev, num_bins)
               for block, mask_block in blocks for slab, mask_slab in _iter_slabs(block, mask_block)]
    hist = np.zeros(num_bins, dtype=np.int64)
    for partial in pending:
        hist += np.asarray(partial, dtype=np.int64)

    # Edges computed on the host exactly as np.histogram does -- a linspace in the image's dtype
    # (np.histogram's bin_type is result_type(range, data)) -- so the np and jax paths of
    # multi_threshold_otsu produce identical edges for identical data.
    bin_edges = np.linspace(lo, hi, num_bins + 1, dtype=image.dtype)
    return hist, bin_edges


def multi_threshold_otsu(image, classes=2, num_bins=1024, valid_mask=None):
    """
    Segment an image into multiple intensity classes using Otsu's method.

    This function computes optimal threshold values that divide an image into the specified
    number of classes by minimizing the intra-class variance. It returns `classes - 1` thresholds
    that can be used to partition the image intensity range into `classes` distinct segments.

    A NumPy image is histogrammed on the host; a JAX image is histogrammed on its own device(s) --
    including a sharded volume, whose per-shard partial histograms are summed on the host
    (bit-identical counts; no gather of the volume).  Only the tiny per-shard histograms come to the
    host for the threshold search.

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

    # Compute the histogram of the valid entries: on-device (sharded-safe, exact int64 counts) for a
    # jax image, host for numpy -- the same masked semantics either way.
    if isinstance(image, jax.Array):
        if valid_mask is None:
            valid_mask = np.ones((1,) * image.ndim, dtype=bool)
        hist, bin_edges = _sharded_histogram(image, valid_mask, num_bins)   # host numpy results
    elif valid_mask is not None:
        hist, bin_edges = _masked_histogram(image, np.asarray(valid_mask), num_bins, np)
    else:
        # Python-float range endpoints for the same reason as in _masked_histogram (edge consistency
        # across the numpy and sharded paths).
        hist, bin_edges = np.histogram(image, bins=num_bins, range=(float(np.min(image)), float(np.max(image))))

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
    # Bin coordinates centered and scaled to [-1, 1].  The within-class variance is shift-invariant,
    # and a uniform scale multiplies EVERY interval cost by the same factor, so the DP's argmin -- and
    # hence the returned thresholds -- is unchanged in exact arithmetic.  Centering removes the
    # mean-offset cancellation in the prefix-difference arithmetic below; scaling keeps the moment
    # prefix sums O(total count) instead of O(count * bins^2) (~1e15 for a 1e9-voxel volume), so all
    # magnitudes stay comfortably conditioned in float64.
    half_span = max((num_bins - 1) / 2.0, 1.0)     # max() guards the degenerate 1-bin histogram
    bin_coord = (np.arange(num_bins, dtype=np.float64) - (num_bins - 1) / 2.0) / half_span

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



