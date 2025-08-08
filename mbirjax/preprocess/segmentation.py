import numpy as np
import jax.numpy as jnp
import mbirjax.preprocess as mjp


def multi_threshold_otsu(image, classes=2, num_bins=1024):
    """
    Segment an image into multiple intensity classes using Otsu's method.

    This function computes optimal threshold values that divide an image into the specified
    number of classes by minimizing the intra-class variance. It returns `classes - 1` thresholds
    that can be used to partition the image intensity range into `classes` distinct segments.

    Args:
        image (np.ndarray):
            Input image as a NumPy array of floating-point values.
        classes (int, optional):
            Number of classes to divide the image into. Must be ≥ 2. Defaults to 2.
        num_bins (int, optional):
            Number of bins to use when constructing the image histogram. Defaults to 256.

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

    # Compute the histogram of the image
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


def segment_plastic_metal(recon, num_metal, radial_margin=10, top_margin=10, bottom_margin=10):
    """
    Non-binary (soft) segmentation using multi-threshold Otsu + Gaussian soft labels.

    Returns:
        plastic_mask  : fractional mask for plastic (same shape as recon)
        metal_masks   : list of fractional masks for each metal k=0..num_metal-1
        plastic_scale : weighted scaling factor for plastic
        metal_scales  : list of weighted scaling factors for each metal
    """
    # Remove any flash from the boundary of the recon
    recon = mjp.apply_cylindrical_mask(recon, radial_margin=radial_margin, top_margin=top_margin,
                                       bottom_margin=bottom_margin)

    # Compute thresholds using multi-threshold Otsu
    # Classes: [background, plastic, metal_0, metal_1, ...]
    num_classes = num_metal + 2
    thresholds = multi_threshold_otsu(recon, classes=num_classes)  # length K-1, ascending

    # Assigns each voxel in `recon` to a bin index based on `thresholds`:
    #   0            if value <= thresholds[0]
    #   1            if thresholds[0] < value <= thresholds[1]
    #   ...
    #   K-1          if value > thresholds[-1]
    cls = jnp.digitize(recon, thresholds)

    # Per-class mean/var (from hard labels) for the Gaussian weights
    eps = 1e-6
    means = []
    variances = []

    for class_idx in range(num_classes):
        class_mask = (cls == class_idx)  # Boolean mask for this class
        voxel_count = class_mask.sum().astype(recon.dtype)  # Number of voxels in the class

        class_mean = jnp.sum(recon * class_mask) / (voxel_count + eps)
        class_variance = jnp.sum(class_mask * (recon - class_mean) ** 2) / (voxel_count + eps)

        means.append(class_mean)
        variances.append(class_variance)

    means = jnp.asarray(means)
    variances = jnp.asarray(variances) + eps  # keep strictly positive

    # Soft weights: only two nonzero per voxel (classes i and i+1 inside each bin)
    weights = jnp.zeros((num_classes,) + recon.shape, dtype=recon.dtype)

    # Below first threshold -> class 0 is 1
    below = (cls == 0) & (recon <= thresholds[0])
    weights = weights.at[0].set(jnp.where(below, 1.0, 0.0))

    # Above last threshold -> class K-1 is 1
    above = (cls == num_classes - 1) & (recon > thresholds[-1])
    weights = weights.at[num_classes - 1].set(jnp.where(above, 1.0, 0.0))

    # Inside each interior bin (t_i, t_{i+1}] -> mix class i and i+1
    for i in range(num_classes - 1):
        in_bin = (cls == i) & (recon > (thresholds[i - 1] if i > 0 else -jnp.inf)) & (
            recon <= (thresholds[i] if i < num_classes - 1 else jnp.inf)
        )
        # Gaussian-like unnormalized likelihoods
        wi = jnp.exp(-0.5 * (recon - means[i]) ** 2 / variances[i])
        wj = jnp.exp(-0.5 * (recon - means[i + 1]) ** 2 / variances[i + 1])
        s = wi + wj + eps
        wi, wj = wi / s, wj / s

        # Write only on voxels in this bin
        weights = weights.at[i].set(jnp.where(in_bin, wi, weights[i]))
        weights = weights.at[i + 1].set(jnp.where(in_bin, wj, weights[i + 1]))

    # Extract plastic and metal masks
    plastic_mask = weights[1]
    metal_masks = [weights[2 + k] for k in range(num_metal)]

    # Weighted scaling (uses mask as weights).

    plastic_scale = mjp.compute_scaling_factor(recon, plastic_mask)
    metal_scales = [mjp.compute_scaling_factor(recon, m) for m in metal_masks]

    return plastic_mask, metal_masks, plastic_scale, metal_scales
