import jax
import jax.numpy as jnp
import mbirjax as mj
import mbirjax.preprocess as mjp
import random


def gen_huber_weights(weights, sino_error, T=1.0, delta=1.0, epsilon=1e-6):
    """
    This function generates generalized Huber weights based on the method described in the referenced notes.
    It adds robustness by treating any element where ``|sino_error / weights| > T`` as an outlier,
    down-weighting it according to the generalized Huber function.

    The function returns new `ghuber_weights`.

    Typically, to obtain the final robust weights, the `ghuber_weights` should be multiplied by the original `weights`:

        final_weights = weights * ghuber_weights

    Args:
        weights: jnp.ndarray of shape (views, rows, cols):
            Initial weights, typically derived from inverse variance estimates.
        sino_error: jnp.ndarray of shape (views, rows, cols):
            Sinogram error array representing deviations from the model.
        T: float, optional (default=1.0):
            Threshold parameter; values greater than T are treated as outliers.
        delta: float, optional (default=1.0):
            Controls the strength of the generalized Huber function (delta=1 corresponds to the conventional Huber).
        epsilon: float, optional (default=1e-6):
            Small number to avoid division by zero.

    Returns:
        huber_weights: jnp.ndarray of shape (views, rows, cols)
            The computed generalized Huber weights.

    Notes:
        The generalized Huber function used in this function is based on:
        Venkatakrishnan, S. V., Drummy, L. F., Jackson, M., De Graef, M., Simmons, J. P., and Bouman, C. A.,
        "Model-Based Iterative Reconstruction for Bright-Field Electron Tomography,"
        IEEE Transactions on Computational Imaging, vol. 1, no. 1, pp. 1–15, 2015. DOI: 10.1109/TCI.2014.2371751

    Example:
        >>> from mbirjax import gen_huber_weights
        >>> huber_weights = gen_huber_weights(weights, sino_error)
        >>> final_weights = weights * huber_weights
    """
    if not (0.0 <= delta <= 1.0):
        raise ValueError("delta must be between 0 and 1.")

    weights = jnp.asarray(weights)
    sino_error = jnp.asarray(sino_error)

    # Compute std and global alpha
    std = 1.0 / jnp.maximum(jnp.sqrt(weights), epsilon)
    alpha = jnp.linalg.norm(sino_error) / (jnp.linalg.norm(std) + epsilon)
    std_norm = alpha * std

    # Compute normalized error
    normalized_error = sino_error / std_norm
    abs_norm_error = jnp.abs(normalized_error)

    # Apply generalized Huber function
    huber_weights = jnp.where(abs_norm_error <= T, 1.0, (delta * T) / (abs_norm_error + epsilon))

    return huber_weights


def BH_correction(sino, alpha, batch_size=64):
    """
    Apply a polynomial beam hardening correction to a sinogram.

    This function applies a polynomial correction to each view of the sinogram
    by evaluating powers of the sinogram values and weighting them by the coefficients in `alpha`,
    while also including the original linear term (the sinogram itself).

    The corrected sinogram is computed as:

        corrected_sino = sino + alpha[0] * sino**2 + alpha[1] * sino**3 + ...

    It processes the sinogram in batches of views for memory efficiency.

    Args:
        sino (jnp.ndarray or np.ndarray of shape (views, rows, cols)):
            Input sinogram to correct.
        alpha (list or array of floats):
            Coefficients for the polynomial correction. The k-th term corresponds to sino^(k+1).
        batch_size (int, optional, default=16):
            Number of views to process in a single batch.

    Returns:
        corrected_sino: jnp.ndarray of shape (views, rows, cols)
            Beam hardening corrected sinogram.

    Example:
        >>> import mbirjax.preprocess as mjp
        >>> alpha = [1.0, 0.2, 0.1]  # Correction: sino + 0.2 * sino^2 + 0.1 * sino^3
        >>> corrected_sino = mjp.BH_correction(sino, alpha)
    """
    # Ensure inputs are JAX arrays
    sino = jnp.asarray(sino)
    alpha = jnp.asarray(alpha)

    views, rows, cols = sino.shape
    corrected = []

    for i in range(0, views, batch_size):
        sino_batch = sino[i:i+batch_size]

        # Initialize corrected batch to the linear term (sino_batch)
        corrected_batch = jnp.array(sino_batch)

        # Apply polynomial terms
        for k in range(len(alpha)):
            corrected_batch += alpha[k] * jnp.power(sino_batch, k + 1)

        corrected.append(corrected_batch)

    corrected_sino = jnp.concatenate(corrected, axis=0)

    return corrected_sino


def _generate_polynomial_combinations(num_terms, max_order):
    """
    Generate all combinations of polynomial powers such that the total degree
    (sum of exponents) is <= max_order, excluding the all-zero combination.
    The combinations are sorted in increasing order of total degree.

    Args:
        num_terms (int): Number of variables (e.g., 2 for x, y).
        max_order (int): Maximum total degree of the polynomial.

    Returns:
        list[tuple[int]]: List of exponent tuples representing valid terms.
    """
    combinations = []

    def generate_recursive(current_combination, remaining_terms):
        if remaining_terms == 0:
            total_degree = sum(current_combination)
            if 0 < total_degree <= max_order:
                combinations.append(tuple(current_combination))
            return

        for power in range(max_order + 1):
            generate_recursive(current_combination + [power], remaining_terms - 1)

    generate_recursive([], num_terms)

    # Sort by total degree (sum of powers)
    combinations.sort(key=lambda x: sum(x))
    return combinations


def correct_BH_plastic_metal(ct_model, measured_sino, recon, num_metal=1, order=3, alpha=1, beta=0.005):
    """
    Perform beam hardening correction for CT sinograms with plastic and multiple metal components
    using a polynomial fitting model with regularization.

    Args:
        ct_model: CT model object with a `forward_project` method and a `main_device` attribute.
        measured_sino (jnp.ndarray): Raw measured sinogram.
        recon (jnp.ndarray): Reconstructed 3D volume used for segmentation of plastic and metal regions.
        num_metal (int, optional): Number of metal materials to segment and correct for. Defaults to 1.
        order (int, optional): Maximum total degree of the beam hardening correction polynomial. Defaults to 3.
        alpha (float, optional): Degree-dependent scaling factor for regularization weights. Higher values penalize
            higher-order terms more strongly. Defaults to 1.
        beta (float, optional): Regularization strength for ridge regression. Defaults to 0.005.

    Returns:
        jnp.ndarray: Beam-hardening corrected sinogram of the same shape as `measured_sino`.
    """
    metal_exponent_list = _generate_polynomial_combinations(num_metal, order)
    device = ct_model.main_device
    y = measured_sino.reshape(-1)

    # --- Segment ---
    plastic_mask, metal_masks, plastic_scale, metal_scales = mjp.segment_plastic_metal(recon, num_metal=num_metal)

    # --- Project ---
    ideal_plastic_sino = plastic_scale * ct_model.forward_project(jax.device_put(plastic_mask, device)).reshape(-1)
    del plastic_mask
    p_normalization = jnp.max(jnp.abs(ideal_plastic_sino))
    p = ideal_plastic_sino / p_normalization

    metal_sinos = []
    metals = []
    for mask, scale in zip(metal_masks, metal_scales):
        sino = scale * ct_model.forward_project(jax.device_put(mask, device)).reshape(-1)
        norm = jnp.max(jnp.abs(sino))
        metal_sinos.append(sino)
        metals.append(sino / norm)
    del metal_masks

    metal_terms = []  # e.g., [m1**1 * m2**0, m1**0 * m2**1, m1**1 * m2**1, ...]
    for exponents in metal_exponent_list:
        term = jnp.ones_like(metals[0])
        for m, exp in zip(metals, exponents):
            if exp > 0:
                term *= m ** exp
        metal_terms.append(term)
    metal_terms = jnp.stack(metal_terms, axis=1)

    num_metal_terms = len(metal_exponent_list)
    num_cross_terms = len(_generate_polynomial_combinations(num_metal, order-1))

    H = jnp.concatenate([p.reshape(-1, 1), p[:, None] * metal_terms[:, :num_cross_terms], metal_terms], axis=1)

    # Compute total degree for each cross term and metal term
    cross_exponent_list = _generate_polynomial_combinations(num_metal, order-1)
    cross_degree = [1 + sum(exponents) for exponents in cross_exponent_list]
    metal_degree = [sum(exponents) for exponents in metal_exponent_list]

    # Construct diagonal regularization weights: higher-degree terms are penalized more.
    # This applies stronger regularization to higher-order terms when alpha > 0.
    weights = jnp.asarray([1] + cross_degree + metal_degree)
    weight_matrix = jnp.diag(1 + weights ** alpha)

    Hty = jnp.dot(H.T, y)
    HtH = jnp.dot(H.T, H)

    # --- Solve for theta ---
    scaling_const = jnp.trace(HtH) / jnp.trace(weight_matrix)
    lambda_reg = beta * scaling_const
    HtH_reg = HtH + lambda_reg * weight_matrix
    theta = jnp.linalg.solve(HtH_reg, Hty)
    print(f'theta = {theta}')

    # --- Build correction ---
    # Term 0: p
    linear_plastic_coef = theta[0] * jnp.ones_like(p) + sum(theta[n + 1] * metal_terms[:, n] for n in range(num_cross_terms))

    # Metal-only sinogram
    metal_sino = jnp.zeros_like(y)
    metal_sino = sum(theta[n + 1 + num_cross_terms] * metal_terms[:,n] for n in range(num_metal_terms))

    # Regularize
    denom_floor = 1e-6 * jnp.linalg.norm(linear_plastic_coef)
    linear_plastic_coef = jnp.where(jnp.abs(linear_plastic_coef) > denom_floor, linear_plastic_coef, denom_floor)

    # Compute corrected plastic sinogram
    corrected_plastic_sino = p_normalization * (y - metal_sino) / linear_plastic_coef

    # Compute a scaling factor by performing least-squares fitting between the corrected plastic sinogram
    # and the measured sinogram at plastic-only locations (i.e., where plastic is present and all metals are absent)
    metal_absent = jnp.ones_like(p)
    for metal in metal_sinos:
        metal_absent = metal_absent & (metal == 0)
    condition = (p != 0) & metal_absent
    plastic_only_indices = jnp.where(condition)[0]

    plastic_scale = mjp.compute_scaling_factor(y[plastic_only_indices], corrected_plastic_sino[plastic_only_indices])
    scaled_corrected_plastic_sino = plastic_scale * corrected_plastic_sino

    # Combine all scaled components
    corrected_sino_flat = scaled_corrected_plastic_sino + sum(metal_sinos)

    corrected_sino = corrected_sino_flat.reshape(measured_sino.shape)
    return corrected_sino


def recon_BH_plastic_metal(ct_model, sino, weights, num_BH_iterations=3, stop_threshold_pct=0.5,
                           num_metal=1, order=3, alpha=1, beta=0.005, verbose=0):
    """
    Perform iterative metal artifact reduction using plastic-metal beam hardening correction.

    This function alternates between beam hardening correction (via `correct_BH_plastic_metal`)
    and reconstruction, refining the image over several iterations to suppress metal-induced artifacts.

    Args:
        ct_model: MBIRJAX cone beam model instance with `direct_recon` and `recon` methods.
        sino (jnp.ndarray):  Input sinogram data to be corrected.
        weights (jnp.ndarray): Transmission weights used in the reconstruction algorithm.
        num_BH_iterations (int, optional): Number of correction-reconstruction iterations. Defaults to 3.
        stop_threshold_pct (float, optional): Relative change threshold (%) for early stopping in MBIR. Defaults to 0.5.
        num_metal (int, optional): Number of metal materials to segment and correct for. Defaults to 1.
        order (int, optional): Maximum total degree of the beam hardening correction polynomial. Defaults to 3.
        alpha (float, optional): Degree-dependent scaling factor for regularization weights. Higher values penalize
            higher-order terms more strongly. Defaults to 1.
        beta (float, optional): Regularization strength for ridge regression. Defaults to 0.005.
        verbose (int, optional): Verbosity level for printing intermediate information. Defaults to 0.

    Returns:
         jnp.ndarray: The final corrected reconstruction after iterative beam hardening correction.

    Example:
        >>> recon = recon_BH_plastic_metal(
        ...     ct_model, sino, weights,
        ...     num_BH_iterations=3,
        ...     stop_threshold_pct=0.5,
        ...     num_metal=1,
        ...     order=3,
        ...     alpha=1,
        ...     beta=0.005,
        ...     verbose=1
        ... )
        >>> mj.slice_viewer(recon)
    """
    if verbose > 0:
        print("\n********* Perform initial FDK reconstruction **********")
    recon = ct_model.direct_recon(sino)

    for i in range(num_BH_iterations):
        # Estimate Corrected Sinogram
        corrected_sinogram = correct_BH_plastic_metal(ct_model, sino, recon, num_metal=num_metal, order=order, alpha=alpha, beta=beta)

        # Reconstruct Corrected Sinogram
        recon, _ = ct_model.recon(corrected_sinogram, weights=weights, init_recon=recon, stop_threshold_change_pct=stop_threshold_pct)

        if verbose > 0:
            print(f"\n************ BH Iteration {i + 1}: Display plastic and metal mask **************")
            plastic_mask, metal_masks, plastic_scale, metal_scales = mjp.segment_plastic_metal(recon, num_metal)
            labels = ['Plastic Mask'] + [f'Metal {j + 1} Mask' for j in range(len(metal_masks))]
            mj.slice_viewer(plastic_mask, *metal_masks, vmin=0, vmax=1.0,
                            slice_label=labels,
                            title=f'Iteration {i + 1}: Comparison of Plastic and Metal Masks')

    return recon
