import jax
import jax.numpy as jnp
import numpy as np
import mbirjax as mj
import mbirjax.preprocess as mjp
import matplotlib.pyplot as plt

def gpnp_em(ct_model, sino):
    """

    Args:
        ct_model (mbirjax TomographyModel):
        sino (numpy array):

    Returns:
        recon, pm_segmentation
    """
    num_outer_steps = 10
    steps_before_param_estimation = 2
    sigma_max = 0.4
    sigma_min = 0.01
    schedule_factor = (sigma_min / sigma_max) ** (1 / num_outer_steps)
    sigma_sample = sigma_max
    beta_sample = 0.25
    r_sample = 1.2  # Higher values of r give more regularization

    # Create an initial reconstruction
    ct_model.view_batch_size_for_vmap = 128
    weights = ct_model.gen_weights(sino, 'transmission_root')

    print('Initial recon')
    x_sample, recon_dict = ct_model.recon(sino, weights)
    x_sample = jnp.clip(x_sample, 0, None)
    x_prox = x_sample.copy()

    print('Initial parameter estimation')
    ideal_metal_sino, pm_segmentation = estimate_em_params(ct_model, [x_sample], sino,
                                                           ideal_metal_sino=None, pm_segmentation=None)

    denoise_sigma = sigma_sample * np.percentile(x_sample, 98)
    denoiser = mj.QGGMRFDenoiser(x_sample.shape)
    rand = np.random.randn(*x_sample.shape)
    x_sample = x_sample + denoise_sigma * rand
    initialize_prox = True

    for j in range(num_outer_steps):
        # Denoise the current image
        # mj.slice_viewer(x_sample, slice_axis=0, title='Noisy sample')

        print('Denoising')
        x_denoised, recon_dict = denoiser.denoise(x_sample, 2*denoise_sigma)
        x_denoised = jnp.clip(x_denoised, 0, None)
        mj.slice_viewer(x_denoised, slice_axis=0, title='Denoised sample')

        # Get a random image and sample from the prior generator
        rand = np.random.randn(*x_sample.shape)
        scale = np.sqrt(beta_sample) * denoise_sigma
        x_sample = (1 - r_sample * beta_sample) * x_sample + r_sample * beta_sample * x_denoised + scale * rand

        # Get the deterministic proximal map update, then sample from the forward generator
        print('Proximal map')
        sigma_prox = scale
        x_prox, recon_dict = plastic_metal_prox(ct_model, x_sample, sino, ideal_metal_sino, pm_segmentation, sigma_prox, weights=weights,
                                                init_recon=x_prox, max_iterations=15, first_iteration=0, initialize_prox=initialize_prox)
        initialize_prox = False
        mj.slice_viewer(jnp.clip(x_prox, 0, None), slice_axis=0, title='Output of prox map, sigma={}'.format(denoise_sigma))
        rand = np.random.randn(*x_sample.shape)
        x_sample = x_prox + scale * rand

        # Update the segmentation estimate
        if j % steps_before_param_estimation == 0:
            print('Estimating parameters')
            ideal_metal_sino, pm_segmentation = estimate_em_params(ct_model, [x_sample], sino,
                                                                   ideal_metal_sino, pm_segmentation)
            mj.slice_viewer(ideal_metal_sino, pm_segmentation[0], pm_segmentation[1], slice_axis=0, title='ideal metal sino, plastic, metal')

        denoise_sigma *= schedule_factor

    print('Estimating parameters')
    ideal_metal_sino, pm_segmentation = estimate_em_params(ct_model, [x_prox], sino,
                                                           ideal_metal_sino, pm_segmentation)

    x_prox = jnp.clip(x_prox, 0, None)
    mj.slice_viewer(x_prox, slice_axis=0, title='Final output')
    return x_prox, recon_dict, pm_segmentation


def plastic_metal_prox(ct_model, x, sino, ideal_metal_sino, pm_segmentation, sigma_prox, weights=None,
                       init_recon=None, max_iterations=3, first_iteration=0, initialize_prox=True):
    """
    Compute the proximal map for plastic-metal segmentation.
    Note that sigma_y and sigma_prox are set automatically or with manual override using set_params().

    Args:
        ct_model (TomographModel):
        x:
        sino:
        ideal_metal_sino:
        pm_segmentation (tuple of arrays):
        sigma_prox (float):
        weights (float or None or jax array or numpy array):
        init_recon (jax array):
        max_iterations (int):
        first_iteration (int):
        initialize_prox (bool):

    Returns:

    """
    epsilon = 1e-6
    hardening_fraction = 0.5
    background_threshold_percentile = 80

    plastic_mask, metal_mask = pm_segmentation
    y_m_mono = ct_model.forward_project(metal_mask * x)
    y_p_mono = ct_model.forward_project(plastic_mask)

    no_metal = jnp.abs(y_m_mono) < epsilon
    no_plastic = jnp.abs(y_p_mono) < epsilon
    b_only = no_plastic * no_metal
    b_only_clipped = b_only * (sino < jnp.percentile(sino[b_only], background_threshold_percentile))

    # Background estimation: single value per view
    y_back = jnp.sum(sino * b_only_clipped, axis=(1,2), keepdims=True) / jnp.sum(b_only_clipped, axis=(1,2), keepdims=True)

    # Match the plastic-only projection to the measured sinogram minus background in the places of plastic only
    alpha_p = mjp.compute_scaling_factor((sino - y_back) * (1 - no_plastic) * no_metal,
                                         y_p_mono * (1 - no_plastic) * no_metal)
    y_p_mono = alpha_p * y_p_mono

    y_m_hardened = jnp.clip((sino - y_back - y_p_mono) * (1 - no_metal), min=0, max=None)
    y_m_corrected = (1 - hardening_fraction) * ideal_metal_sino + hardening_fraction * y_m_hardened

    y_p_corrected = jnp.clip(sino - y_back - y_m_hardened, min=0, max=None)
    y_corrected = y_p_corrected + y_m_corrected

    prox_output = ct_model.prox_map(x, y_corrected, sigma_prox, weights=weights, init_recon=init_recon, do_initialization=initialize_prox,
                                    max_iterations=max_iterations, first_iteration=first_iteration)

    return prox_output


def estimate_em_params(ct_model, x_samples, sino, ideal_metal_sino=None, pm_segmentation=None, mask_margin=10):
    """
    Estimate the plastic-metal segmentation and mean metal sinogram given a set of reconstructions
    and the associated sinogram.

    Args:
        ct_model (TomographyModel):
        x_samples (iterable): List or tuple of sample reconstructions
        sino (jax array):  Sinogram for these reconstructions
        ideal_metal_sino (jax array or None, optional): Any existing metal sino if available, which is averaged with the new metal sino.  Defaults to None.
        pm_segmentation (tuple or list or None, optional):  Two arrays (p, m), each the size of a reconstruction, with each entry of p, m, p+m in the range [0, 1]. These are averaged with the new segmentations.  Defaults to None.
        mask_margin (int): Number of pixels to zero out from the reconstruction.

    Returns:
        tuple, tuple:  bh_coeffs_new, pm_seg_new, each with the structure of the input values
    """

    # Set order of models and do initialization
    phi_p = 0
    phi_m = 0
    ideal_metal_sino_new = jnp.zeros_like(sino)
    num_samples = len(x_samples)

    # Estimate segmentation and beam hardening
    for x in x_samples:
        # Get the segmentation for this sample
        x = mjp.apply_cylindrical_mask(x, radial_margin=mask_margin, top_margin=mask_margin, bottom_margin=mask_margin)
        plastic_mask, metal_mask, _, _ = segment_plastic_metal(x, apply_med_filter=True, apply_cylindrical_mask=False)

        phi_p = phi_p + plastic_mask / num_samples
        phi_m = phi_m + metal_mask / num_samples

        ideal_metal_sino_new = ideal_metal_sino_new + ct_model.forward_project(metal_mask * x) / num_samples

    # Set up the return values
    pm_seg_new = (phi_p, phi_m)

    new_weight = 0.5
    if ideal_metal_sino is not None:
        ideal_metal_sino_new = (1 - new_weight) * ideal_metal_sino_new + new_weight * ideal_metal_sino

    if pm_segmentation is not None:
        pm_seg_new = [(1 - new_weight) * pm_s + new_weight * pm_s_n for pm_s, pm_s_n in zip(pm_segmentation, pm_seg_new)]

    return ideal_metal_sino_new, pm_seg_new


def segment_plastic_metal(recon, radial_margin=10, top_margin=10, bottom_margin=10, apply_med_filter=True,
                          apply_cylindrical_mask=True):
    """
    Segment a reconstruction into plastic and metal masks using multi-threshold Otsu.

    This function uses multi-threshold Otsu segmentation to classify the input
    reconstruction into several classes and returns binary masks for the plastic
    and metal components. It also returns a scaling factor representing the average
    value of the reconstruction in each segmented region. A cylindrical mask is applied
    to the volume prior to thresholding.

    Args:
        recon (jnp.ndarray): Reconstructed volume array.
        radial_margin (int, optional): Margin in pixels to subtract from the cylindrical mask radius. Defaults to 10.
        top_margin (int, optional): Number of slices to mask out from the top of the volume. Defaults to 10.
        bottom_margin (int, optional): Number of slices to mask out from the bottom of the volume. Defaults to 10.
        apply_med_filter (bool, optional): If True, then apply a median filter to the reconstruction. Defaults to True.
        apply_cylindrical_mask (bool, optional): If True, then apply a cylindrical mask to the reconstruction. Defaults to True.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray, float, float]: A tuple containing:
            - plastic_mask (jnp.ndarray): Binary mask for plastic regions.
            - metal_mask (jnp.ndarray): Binary mask for metal regions.
            - plastic_scale (float): Scaling factor for the plastic mask.
            - metal_scale (float): Scaling factor for the metal mask.

    Example:
        >>> import mbirjax as mj
        >>> import mbirjax.preprocess as mjp
        >>> plastic_mask, metal_mask, plastic_scale, metal_scale = mjp.segment_plastic_metal(recon)
        >>> mj.slice_viewer(plastic_mask, metal_mask, vmin=0, vmax=1.0,
        ...                 slice_label=['Plastic', 'Metal'],
        ...                 title='Plastic and Metal Masks')
    """
    # Determine class thresholds based on the 3-classes
    # Remove any flash from the boundary of the recon
    if apply_cylindrical_mask:
        recon = mjp.apply_cylindrical_mask(recon, radial_margin=radial_margin, top_margin=top_margin, bottom_margin=bottom_margin)
    if apply_med_filter:
        cur_recon = mj.median_filter3d(recon)
        print('Done median')
    else:
        cur_recon = recon
    counts, bins = np.histogram(cur_recon.flatten(), bins=1000)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0

    # Separate the background from the plastic and metal
    thresholds = mj.preprocess.multi_threshold_otsu(cur_recon, classes=3)
    plastic_low_threshold = thresholds[0]
    plastic_low_ind = np.where(bin_centers > plastic_low_threshold)[0][0]  # The index of the center just to the right of plastic_low_threshold
    plastic_peak_ind = np.argmax(counts[plastic_low_ind:]) + plastic_low_ind
    plastic_peak_value = bin_centers[plastic_peak_ind]

    # Restrict to plastic and metal, then clip the outlier metals to a multiple of plastic
    plastic_metal_factor = 3.5
    recon_vals_plastic_metal = cur_recon[cur_recon >= plastic_low_threshold]
    recon_vals_plastic_metal = np.clip(recon_vals_plastic_metal, None, plastic_metal_factor * plastic_peak_value)
    thresholds_pm = mj.preprocess.multi_threshold_otsu(recon_vals_plastic_metal, classes=2)
    plastic_metal_threshold = thresholds_pm[0]

    # Create masks
    plastic_mask = (cur_recon > plastic_low_threshold) * (cur_recon < plastic_metal_threshold)
    metal_mask = (cur_recon >= plastic_metal_threshold)

    # Scale factors that match the unitary masks to the reconstruction
    plastic_scale = mjp.compute_scaling_factor(recon, plastic_mask)
    metal_scale = mjp.compute_scaling_factor(recon, metal_mask)

    # Debug code
    # print('Plastic thresholds = {:.3g}, {:.3g}'.format(plastic_low_threshold, plastic_metal_threshold))
    # plastic_high_ind = jnp.where(bin_centers < plastic_metal_threshold)[0][-1]  # The index just to the left of threshold
    # plt.semilogy(bin_centers, counts)
    # plt.plot([bin_centers[plastic_low_ind], bin_centers[plastic_high_ind]], [counts[plastic_low_ind], counts[plastic_high_ind]], '*')
    # plt.title('Histogram of counts plus threshold values')
    # plt.show(block=True)
    # End debug

    return plastic_mask, metal_mask, plastic_scale, metal_scale


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

