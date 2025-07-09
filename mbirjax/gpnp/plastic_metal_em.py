import jax
import jax.numpy as jnp
import mbirjax as mj
import mbirjax.preprocess as mjp
import matplotlib.pyplot as plt


def plastic_metal_prox(ct_model, x, sino, bh_coeffs, pm_segmentation, weights=None,
                       init_recon=None, max_iterations=3, first_iteration=0):
    """
    Compute the proximal map for plastic-metal segmentation.
    Note that sigma_y and sigma_prox are set automatically or with manual override using set_params().

    Args:
        ct_model (TomographModel):
        x:
        sino:
        bh_coeffs:
        pm_segmentation (tuple of arrays):
        weights (float or None or jax array or numpy array):
        init_recon (jax array):
        max_iterations (int):
        first_iteration (int):

    Returns:

    """
    plastic_seg, metal_seg = pm_segmentation
    cross_bh_coeffs, metal_bh_coeffs = bh_coeffs  # Note that metal_bh_coeffs[0] = 0

    sino_m_ideal = ct_model.forward_project(x * metal_seg)  # Or possibly just metal_seg
    sino_m_hardened = apply_polynomial(sino_m_ideal, metal_bh_coeffs)
    sino_residual = sino - sino_m_hardened
    del sino_m_hardened

    epsilon = 1e-6
    linear_plastic_coef = apply_polynomial(sino_m_ideal, cross_bh_coeffs)
    linear_plastic_coef = jnp.clip(linear_plastic_coef, min=epsilon, max=None)

    # Compute BH corrected version of plastic sinogram with metal removed and scale plastic to match measured sino
    corrected_sino = sino_residual / linear_plastic_coef
    plastic_scale = mjp.compute_scaling_factor(sino_residual[sino_m_ideal < epsilon], corrected_sino[sino_m_ideal < epsilon])
    corrected_sino = plastic_scale * corrected_sino + sino_m_ideal

    prox_output = ct_model.prox_map(x, corrected_sino, init_recon=init_recon, weights=weights,
                                    max_iterations=max_iterations, first_iteration=first_iteration)
    return prox_output


def estimate_em_params(ct_model, x_samples, sino, bh_coeffs, pm_segmentation, sigma_threshold=0.0):
    """
    Estimate the plastic-metal segmentation and beam hardening coefficients given a set of reconstructions
    and the associated sinogram.

    Args:
        ct_model (TomographyModel):
        x_samples (iterable): List or tuple of sample reconstructions
        sino (jax array):  Sinogram for these reconstructions
        bh_coeffs (tuple or list): Two vectors of polynomial coefficients.  First for c[j]*p*m**j, second for d[j]*m**j with d[0] = 0.
        pm_segmentation (tuple or list):  Two arrays (p, m), each the size of a reconstruction, with each entry of p, m, p+m in the range [0, 1]
        sigma_threshold (float):  Same as sigma_threshold of segment_plastic_metal

    Returns:
        tuple, tuple:  bh_coeffs_new, pm_seg_new, each with the structure of the input values
    """

    # Set order of models and do initialization
    cross_bh_coeffs, metal_bh_coeffs = bh_coeffs
    cross_len, metal_len = len(cross_bh_coeffs), len(metal_bh_coeffs)
    HtH = 0
    Hty = 0
    phi_p = 0
    phi_m = 0
    num_samples = len(x_samples)

    # Estimate segmentation and beam hardening
    for x in x_samples:
        # Get the segmentation for this sample
        phi_p_x, phi_m_x = segment_plastic_metal(x, sigma_threshold=sigma_threshold)
        phi_p = phi_p + phi_p_x / num_samples
        phi_m = phi_m + phi_m_x / num_samples

        # Make the ideal plastic and metal sinograms - we normalize by the maximum, so there is
        # no need to scale the sinograms based on data or segmentation.
        sino_p_ideal = ct_model.forward_project(phi_p_x)
        sino_p_ideal /= jnp.amax(sino_p_ideal)
        sino_m_ideal = ct_model.forward_project(phi_m_x)
        sino_m_ideal /= jnp.amax(sino_m_ideal)

        # Build H matrix for this sample and accumulate over samples
        H_x = [sino_p_ideal * sino_m_ideal ** i for i in range(cross_len)]  # p, p*m, ..., p*m^(cross_len-1)
        H_x += [sino_m_ideal ** i for i in range(1, metal_len)]  # m, m^2, ..., m^(metal_len-1)
        H_x = jnp.stack(H_x, axis=0)

        Hty = Hty + H_x @ sino
        HtH = HtH + H_x.T @ H_x

    # Regularize and solve for least square value of theta that minimizes || y - H theta ||^2
    sigma_max = jnp.linalg.norm(HtH, ord=2)
    epsilon = 2e-4
    HtH_reg = HtH + (epsilon ** 2) * sigma_max * jnp.eye(HtH.shape[0])
    theta = jnp.linalg.solve(HtH_reg, Hty)

    # Set up the return values
    bh_coeffs_new = (theta[0:cross_len], jnp.concatenate([jnp.zeros(1), theta[cross_len:]]))
    pm_seg_new = (phi_p, phi_m)

    new_weight = 0.5
    bh_coeffs_new = [(1 - new_weight) * bh_c + new_weight * bh_c_n for bh_c, bh_c_n in zip(bh_coeffs, bh_coeffs_new)]
    pm_seg_new = [(1 - new_weight) * pm_s + new_weight * pm_s_n for pm_s, pm_s_n in zip(pm_segmentation, pm_seg_new)]

    return bh_coeffs_new, pm_seg_new


def segment_plastic_metal(recon, apply_median=True, sigma_threshold=0.0):
    """
    Estimate the fraction of plastic and metal in each voxel of the given recon.  This segmentation is done
    using one pass of multi-threshold Otsu to estimate a threshold between background and plastic, then clipping
    of metal to a multiple of plastic, then a second round of multi-threshold Otsu to separate plastic and metal.

    Args:
        recon (jax array):  Recon to segment.
        apply_median (bool, optional):  If True, then apply a 3x3x3 median filter before segmentation. Defaults to True.
        sigma_threshold (float, optional): If positive, then each voxel is mapped to a component using N(v; T, sigma),
            where v is the voxel value, T is the threshold, and sigma=sigma_threshold.  Defaults to 0.

    Returns:
        tuple: Two arrays (p, m), each of the same shape as recon, with each entry of p, m, p+m in the range [0, 1].
    """
    cur_recon = mj.preprocess.apply_cylindrical_mask(recon, radial_margin=5, top_margin=5, bottom_margin=5)

    if apply_median:
        cur_recon = mj.median_filter3d(cur_recon)
    counts, bins = jnp.histogram(cur_recon.flatten(), bins=1000)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0

    # Separate the background from the plastic and metal
    thresholds = mj.preprocess.multi_threshold_otsu(cur_recon, classes=3)
    plastic_low_threshold = thresholds[0]
    plastic_low_ind = jnp.where(bin_centers > plastic_low_threshold)[0][0]  # The index of the center just to the right of plastic_low_threshold
    plastic_peak_ind = jnp.argmax(counts[plastic_low_ind:]) + plastic_low_ind
    plastic_peak_value = bin_centers[plastic_peak_ind]

    # Restrict to plastic and metal, then clip the outlier metals to a multiple of plastic
    recon_vals_plastic_metal = cur_recon[cur_recon >= plastic_low_threshold]
    recon_vals_plastic_metal = jnp.clip(recon_vals_plastic_metal, None, 2.5 * plastic_peak_value)
    thresholds_pm = mj.preprocess.multi_threshold_otsu(recon_vals_plastic_metal, classes=2)
    plastic_high_threshold = thresholds_pm[0]
    plastic_high_ind = jnp.where(bin_centers < plastic_high_threshold)[0][-1]  # The index just to the left of threshold
    print('Plastic thresholds = {:.3g}, {:.3g}'.format(plastic_low_threshold, plastic_high_threshold))

    # counts_clipped, _ = np.histogram(np.clip(cur_recon.flatten(), None, 3 * plastic_peak_value), bins=bins)
    plt.semilogy(bin_centers, counts)
    plt.plot([bin_centers[plastic_low_ind], bin_centers[plastic_high_ind]], [counts[plastic_low_ind], counts[plastic_high_ind]], '*')
    plt.title('Histogram of counts plus threshold values')

    # plastic_mask = (cur_recon > plastic_low_threshold) * (cur_recon < plastic_high_threshold)
    # metal_mask = (cur_recon >= plastic_high_threshold)

    metal_seg = jax.scipy.stats.norm.cdf(cur_recon - plastic_high_threshold, scale=sigma_threshold)
    plastic_seg = jax.scipy.stats.norm.cdf(cur_recon - plastic_low_threshold, scale=sigma_threshold) - metal_seg

    mj.slice_viewer(recon, cur_recon * plastic_seg, cur_recon * metal_seg, vmin=0, vmax=0.05,
                    title='Input recon and result after median filter and segmentation')

    return plastic_seg, metal_seg


def apply_polynomial(input_array, coeffs):
    """
    Apply a polynomial to each element of an input array.  This can be used for beam hardening correction.

    The output is computed as:

        output_array = coeffs[0] + coeffs[1] * input_array + coeffs[2] * input_array**2 + ...

    Args:
        input_array (jax array or numpy array):  Input to polynomial
        coeffs (list or array of floats): Coefficients for the polynomial. The k-th term corresponds to input_array**k.

    Returns:
        jax array of same shape as input_array

    Example:
        >>> import mbirjax.preprocess as mjp
        >>> coeffs = [0.0, 1.0, -0.2, 0.1]  # Correction: input_array - 0.2 * input_array^2 + 0.1 * input_array^3
        >>> output_array = mjp.apply_polynomial(input_array, coeffs)
    """
    # Ensure inputs are JAX arrays
    input_array = jnp.asarray(input_array)
    coeffs = jnp.asarray(coeffs)
    array_shape = input_array.shape

    if len(coeffs) == 0:
        return jnp.zeros(array_shape)

    def eval_poly(value, jnp_coeffs):
        # We assume this is for small polynomials, so we use an explicit for loop.
        # c[0] + c[1] * x + c[2] * x^2 = (c[2] * x + c[1]) * x + c[0]
        out_value = jnp_coeffs[-1]  # Get the top coefficient
        for coeff in jnp_coeffs[:-1][::-1]:  # Get the others in order from top to bottom
            out_value *= value
            out_value += coeff
        return out_value

    # The coefficient array gets copied with each value assigned to a worker, but for small
    # polynomials, this shouldn't lead to excessive memory use.
    output_array = jax.vmap(eval_poly, in_axes=(0, None))(input_array.flatten(), coeffs)
    output_array = output_array.reshape(array_shape)
    return output_array

