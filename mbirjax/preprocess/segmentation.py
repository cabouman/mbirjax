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

    Args:
        hist (ndarray): Histogram of the image.
        num_thresholds (int): Number of thresholds to find.

    Returns:
        list: Threshold values that divide the histogram into the specified number of classes.
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

    Args:
        hist (ndarray): Histogram of the image.

    Returns:
        int: Best threshold value for binary segmentation.
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

    Args:
        hist (ndarray): Histogram of the image.
        thresholds (list): Threshold values that divide the histogram into multiple classes.

    Returns:
        float: Total within-class variance.
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


def _compute_class_mean_var(recon, num_classes, thresholds):
    """
    Compute per-class mean and variance of voxel intensities in `recon`, based on hard class assignments.

    Args:
        recon (jnp.ndarray): The reconstructed volume array.
        num_classes (int): Number of discrete classes.
        thresholds (Sequence[float]): Thresholds defining the bin edges.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]:
            - means: Array of per-class means.
            - variances: Array of per-class variances.
    """

    # Assigns each voxel in `recon` to a bin index based on `thresholds`:
    #   0            if value <= thresholds[0]
    #   1            if thresholds[0] < value <= thresholds[1]
    #   ...
    #   K-1          if value > thresholds[-1]
    cls = jnp.digitize(recon, jnp.array(thresholds))

    # Per-class mean/var (from hard labels) for the Gaussian weights
    means = []
    variances = []

    for class_idx in range(num_classes):
        class_mask = (cls == class_idx)  # Boolean mask for this class
        voxel_count = class_mask.sum().astype(recon.dtype)  # Number of voxels in the class

        class_mean = jnp.sum(recon * class_mask) / (voxel_count)
        class_variance = jnp.sum(class_mask * (recon - class_mean) ** 2) / (voxel_count)

        means.append(class_mean)
        variances.append(class_variance)

    means = jnp.asarray(means)
    variances = jnp.asarray(variances)

    return means, variances


def segment_plastic_metal(recon, num_metal, soft_frac=0.1, sharpness=1.0, radial_margin=10, top_margin=10, bottom_margin=10):
    """
    Non-binary (soft) segmentation using multi-threshold Otsu + Gaussian soft labels.
    Args:
        recon (jnp.ndarray): Reconstructed volume array.
        num_metal (int): Number of metal materials to segment.
        sharpness (float, optional): Controls the steepness of Gaussian weighting between adjacent
            classes.
            - sharpness = 1.0 → default smoothness.
            - sharpness > 1.0 → steeper transitions, masks closer to binary.
            - sharpness < 1.0 → flatter transitions, more blended masks.
                    soft_frac (float): Fraction of windowspixels to assign soft labels.
        soft_frac (float): Fraction of pixels to assign soft labels (recommended < 1.0 to avoid touching bin edges).
        radial_margin (int, optional): Margin in pixels to subtract from the cylindrical mask radius.
        top_margin (int, optional): Number of slices to mask out from the top of the volume.
        bottom_margin (int, optional): Number of slices to mask out from the bottom of the volume.

    Returns:
        Tuple[jnp.ndarray, List[jnp.ndarray], float, List[float]]:
            plastic_mask (jnp.ndarray): fractional mask for plastic (same shape as recon)
            metal_masks (List[jnp.ndarray]): list of fractional masks for each metal
            plastic_scale (float): weighted scaling factor for plastic
            metal_scales (List[float]): list of weighted scaling factors for each metal
    """
    # Remove any flash from the boundary of the recon
    recon = mjp.apply_cylindrical_mask(recon, radial_margin=radial_margin, top_margin=top_margin,
                                       bottom_margin=bottom_margin)

    # Compute thresholds using multi-threshold Otsu
    # Classes: [background, plastic, metal_0, metal_1, ...]
    num_classes = num_metal + 2
    thresholds = multi_threshold_otsu(recon, classes=num_classes)  # length K-1, ascending

    means, variances = _compute_class_mean_var(recon, num_classes, thresholds)

    # Soft weights: only two nonzero per voxel (classes i and i+1 inside each bin)
    weights = jnp.zeros((num_classes,) + recon.shape, dtype=recon.dtype)

    # Compute binary masks
    for i in range(num_classes):
        if i == 0:
            # First class: recon <= thresholds[0]
            in_class = recon <= thresholds[0]
        elif i == num_classes - 1:
            # Last class: recon > thresholds[-1]
            in_class = recon > thresholds[-1]
        else:
            # Interior classes: thresholds[i-1] < recon <= thresholds[i]
            in_class = (recon > thresholds[i - 1]) & (recon <= thresholds[i])

        weights = weights.at[i].set(jnp.where(in_class, 1.0, 0.0))

    # Inside each interior bin (t_i, t_{i+1}] -> assign soft labels
    for i in range(0, num_classes - 1):
        lower_range = thresholds[i] - soft_frac * (thresholds[i] - means[i]) if i > 0 else thresholds[0]
        upper_range = thresholds[i] + soft_frac * (means[i+1] - means[i]) if i < num_classes - 1 else thresholds[-1]
        in_bin = (recon > lower_range) & (recon <= upper_range)
        # Gaussian-like unnormalized likelihoods
        wi = jnp.exp(-0.5 * (recon - means[i]) ** 2 / (variances[i] / sharpness))
        wj = jnp.exp(-0.5 * (recon - means[i + 1]) ** 2 / (variances[i + 1] / sharpness))
        s = wi + wj
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
