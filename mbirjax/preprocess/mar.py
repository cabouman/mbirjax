import jax
import jax.numpy as jnp
import mbirjax as mj
import mbirjax.preprocess as mjp
import random
import warnings


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
        corrected_batch = jnp.zeros_like(sino_batch)

        # Apply polynomial terms
        for k in range(len(alpha)):
            corrected_batch += alpha[k] * jnp.power(sino_batch, k + 1)

        corrected.append(corrected_batch)

    corrected_sino = jnp.concatenate(corrected, axis=0)

    return corrected_sino


def _generate_metal_combinations(num_metal, max_order):
    """
    Generate all combinations of polynomial powers such that the total degree
    (sum of exponents) is <= max_order, excluding the all-zero combination.
    The combinations are sorted in increasing order of total degree.

    Args:
        num_metal (int): Number of metals.
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

    generate_recursive([], num_metal)

    # Sort by total degree (sum of powers)
    combinations.sort(key=lambda x: sum(x))
    return combinations


def _generate_basis_vectors(recon, num_metal, ct_model, device):
    """
    Segment plastic and metal regions from a reconstruction, project them,
    and return the unnormalized basis vectors for beam hardening modeling.

    Args:
        recon (jnp.ndarray): Reconstructed image.
        num_metal (int): Number of metal types to segment.
        ct_model: Forward projection model with a `.forward_project()` method.
        device: JAX device to put the masks on for projection.

    Returns:
        p (jnp.ndarray): Unnormalized plastic basis vector (flattened sinogram).
        metals (list of jnp.ndarray): List of unnormalized metal basis vectors.
    """
    # --- Segment plastic and metal regions in the reconstruction ---
    # plastic_mask: Mask for plastic regions.
    # metal_masks: List of masks for each metal.
    # plastic_scale: Scaling factor for the plastic region.
    # metal_scales: List of scaling factors for each metal region.
    plastic_mask, metal_masks, plastic_scale, metal_scales = mjp.segment_plastic_metal(recon, num_metal=num_metal)

    # --- Forward project, scale and vectorize plastic ---
    p = plastic_scale * ct_model.forward_project(jax.device_put(plastic_mask, device)).reshape(-1)

    # --- Forward project, scale and vectorize each metal in the metals list ---
    metals = []
    for mask, scale in zip(metal_masks, metal_scales):
        m = scale * ct_model.forward_project(jax.device_put(mask, device)).reshape(-1)
        metals.append(m)

    return p, metals


def _get_column_H(col_index, p, metal_basis, H_exponent_list):
    """
    Compute the col_index-th column of the basis matrix H.

    The column is constructed as a monomial of the form:
        H[:, col_index] = p^e0 * m_0^e1 * m_1^e2 * ... * m_{n-1}^en

    where (e0, e1, ..., en) = H_exponent_list[col_index].

    Args:
        col_index (int): Index of the column to compute.
        p (jnp.ndarray): Normalized plastic basis vector.
        metal_basis (list of jnp.ndarray): Normalized metal basis vectors [m_0, m_1, ..., m_{n-1}].
        H_exponent_list (list of tuple): List of exponent tuples defining each column of H.

    Returns:
        jnp.ndarray: The computed column of H (same shape as p and m_i).
    """
    exponents = H_exponent_list[col_index]
    assert len(exponents) == 1 + len(metal_basis), "Mismatch between exponent tuple and number of basis vectors."

    col = p ** exponents[0]
    for metal, exp in zip(metal_basis, exponents[1:]):
        col *= metal ** exp

    return col

def _estimate_BH_model_params(p, metal_basis, y, H_exponent_list, num_cross_terms, alpha, beta):
    """
    Estimate polynomial beam hardening model parameters by solving a regularized least squares system.

    This function avoids constructing the full H matrix by computing HtH and Hty using only the columns of H
    as needed via `_get_column_H()`.

    Args:
        p (jnp.ndarray): Normalized plastic basis vector.
        metal_basis (list of jnp.ndarray): List of normalized metal basis vectors.
        y (jnp.ndarray): Measured sinogram.
        H_exponent_list (list of tuple[int]): List of exponent tuples defining each column of the basis matrix H.
        num_cross_terms (int): Number of cross terms (plastic × metal); remaining terms are metal-only.
        alpha (float): Regularization exponent; higher alpha penalizes higher-degree terms more.
        beta (float): Regularization strength scaling factor.

    Returns:
        theta (jnp.ndarray): Estimated model parameters corresponding to each column in H.
    """
    num_cols = len(H_exponent_list)

    HtH = jnp.zeros((num_cols, num_cols))
    Hty = jnp.zeros(num_cols)

    # Compute the upper triangle of HtH and mirror it.
    for i in range(num_cols):
        h_i = _get_column_H(i, p, metal_basis, H_exponent_list)
        Hty = Hty.at[i].set(jnp.dot(h_i, y))
        for j in range(i, num_cols):
            h_j = _get_column_H(j, p, metal_basis, H_exponent_list)
            dot_ij = jnp.dot(h_i, h_j)
            HtH = HtH.at[i, j].set(dot_ij)
            if i != j:
                HtH = HtH.at[j, i].set(dot_ij)

    # Compute total degree for each cross term and metal term
    cross_degree = [sum(exponent) for exponent in H_exponent_list[0:1+num_cross_terms]]
    metal_degree = [sum(exponent) for exponent in H_exponent_list[1+num_cross_terms:]]

    # Construct diagonal regularization weights: higher-degree terms are penalized more.
    # This applies stronger regularization to higher-order terms when alpha > 0.
    # Add 1 to the beginning to represent the weight for the linear plastic term (p^1).
    weights = jnp.asarray(cross_degree + metal_degree)
    weight_matrix = jnp.diag(1 + weights ** alpha)

    # --- Solve for theta ---
    scaling_const = jnp.trace(HtH) / jnp.trace(weight_matrix)
    lambda_reg = beta * scaling_const
    HtH_reg = HtH + lambda_reg * weight_matrix
    theta = jnp.linalg.solve(HtH_reg, Hty)

    return theta

def _correct_plastic_sinogram(y, p, metal_basis, theta, H_exponent_list, num_cross_terms, num_metal_terms, p_normalization, gamma):
    """
    Perform beam hardening correction on the plastic sinogram.

    This function subtracts the metal-only contributions from the measured sinogram
    and normalizes the result using the linear plastic component, yielding a corrected
    sinogram that approximates the plastic-only contribution.

    The correction is based on a polynomial basis matrix H whose columns correspond to:
        - Plastic term: p
        - Cross terms: p*m, p*m^2, ...
        - Metal-only terms: m, m^2, m^3, ...

    The H matrix looks like: [p, p*m, p*m^2, m, m^2, m^3]
    The correction is applied as:
        corrected_plastic = p_normalization * max(y - H_metal·θ_m, 0) / (max(H_plastic·θ_p, γ * median(H_plastic·θ_p))
    The stabilization term involving γ prevents division by near-zero or negative values, reducing streaks
    and numerical instability.

    Args:
        y (jnp.ndarray): Measured sinogram.
        p (jnp.ndarray): Normalized plastic-only sinogram.
        metal_basis (list of jnp.ndarray): List of sinograms of different metals.
        theta (jnp.ndarray): Estimated coefficients for the polynomial terms in H.
        H_exponent_list (list of tuple): Exponent tuples defining each column of H.
        num_cross_terms (int): Number of cross terms involving both p and metal.
        num_metal_terms (int): Number of metal-only terms in H.
        p_normalization (float): Normalization factor applied to p.
        gamma (float, optional): Stabilization factor.

    Returns:
        corrected_plastic_sino (jnp.ndarray): Beam-hardening-corrected plastic sinogram.
    """

    # Compute the denominator (linear plastic + cross terms) from the first (1 + num_cross_terms) columns of H
    linear_plastic_coef = jnp.zeros_like(y)
    for i in range(0, 1 + num_cross_terms):
        # Use a dummy input of ones to extract the structure of the i-th basis column (i.e., coefficient of p)
        linear_plastic_coef += theta[i] * _get_column_H(i, jnp.ones_like(y), metal_basis, H_exponent_list)

    y_minus_metal = y
    # Subtract metal-only terms (from H columns after the cross terms)
    for j in range(1 + num_cross_terms, 1 + num_cross_terms + num_metal_terms):
        y_minus_metal -= theta[j] * _get_column_H(j, p, metal_basis, H_exponent_list)

    # Enforce non-negativity on the residual sinogram (plastic + cross terms)
    y_minus_metal = jnp.maximum(y_minus_metal, 0)

    # Compute median of plastic coefficients (used to define a stabilization floor)
    median_plastic_coef = jnp.median(linear_plastic_coef)
    min_plastic_coef = gamma * median_plastic_coef

    # A negative median would be non-physical and may indicate instability in the algorithm
    # In that case, issue a runtime warning to flag the potential problem
    if float(median_plastic_coef) <= 0:
        warnings.warn("Median of linear_plastic_coef is negative", RuntimeWarning)

    # Clamp linear_plastic_coef at min_plastic_coef to prevent division by very small or negative values
    clamped_plastic_coef = jnp.maximum(linear_plastic_coef, min_plastic_coef)
    corrected_plastic_sino = p_normalization * y_minus_metal / clamped_plastic_coef

    return corrected_plastic_sino


def correct_BH_plastic_metal(ct_model, measured_sino, recon, num_metal=1, order=3, alpha=1, beta=0.02, gamma=0.4):
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
        beta (float, optional): Regularization strength for ridge regression. Defaults to 0.02.
        gamma (float, optional): Stabilization factor.

    Returns:
        jnp.ndarray: Beam-hardening corrected sinogram of the same shape as `measured_sino`.
    """
    # Construct the exponent list of the metal sinograms.
    metal_exponent_list = _generate_metal_combinations(num_metal, order)
    cross_exponent_list = _generate_metal_combinations(num_metal, order - 1)
    num_metal_terms = len(metal_exponent_list)
    num_cross_terms = len(cross_exponent_list)

    # Construct the exponent list for each column of the basis matrix H.
    # Each entry in H_exponent_list is a tuple representing the exponents of (p, m_0, m_1, ..., m_{num_metal-1}).
    # - Linear plastic term: (1, 0, 0, ...)
    # - Cross terms: The leading 1 indicates the presence of a linear p term.
    # - Metal-only terms: The leading 0 indicates there is no p in the term.
    # - Total number of columns: 1 + num_cross_terms + num_metal_terms.
    H_exponent_list = (
            [(1,) + (0,) * num_metal] +
            [(1, *t) for t in cross_exponent_list] +
            [(0, *t) for t in metal_exponent_list])

    device = ct_model.main_device
    y = measured_sino.reshape(-1)

    # Get normalized basis vectors p and [m_0, m_1, ...]
    p, metal_basis = _generate_basis_vectors(recon, num_metal, ct_model, device)
    p_normalization = jnp.max(jnp.abs(p))
    metals_normalization = [jnp.max(jnp.abs(arr)) for arr in metal_basis]
    p = p / p_normalization
    metal_basis = [arr / norm for arr, norm in zip(metal_basis, metals_normalization)]

    # Estimate beam hardening model parameters theta
    theta = _estimate_BH_model_params(p, metal_basis, y, H_exponent_list, num_cross_terms, alpha, beta)
    # print(f'theta = {theta}')

    # Compute the corrected plastic sinogram
    corrected_plastic_sino = _correct_plastic_sinogram(y, p, metal_basis, theta,H_exponent_list,
                                                       num_cross_terms, num_metal_terms, p_normalization, gamma)

    # Compute a scaling factor by performing least-squares fitting between the corrected plastic sinogram
    # and the measured sinogram at plastic-only locations (i.e., where plastic is present and all metals are absent)
    metal_absent = jnp.ones_like(p, dtype=bool)
    for metal in metal_basis:
        metal_absent = metal_absent & (metal == 0)
    condition = (p != 0) & metal_absent
    plastic_only_indices = jnp.where(condition)[0]

    plastic_scale = mjp.compute_scaling_factor(y[plastic_only_indices], corrected_plastic_sino[plastic_only_indices])
    scaled_corrected_plastic_sino = plastic_scale * corrected_plastic_sino

    # Combine all scaled components, denormalize the metals
    corrected_sino_flat = scaled_corrected_plastic_sino + sum(arr * norm for arr, norm in zip(metal_basis, metals_normalization))

    corrected_sino = corrected_sino_flat.reshape(measured_sino.shape)
    return corrected_sino


def recon_BH_plastic_metal(ct_model, sino, weights, num_BH_iterations=3, stop_threshold_change_pct=0.5,
                           num_metal=1, order=3, alpha=1, beta=0.02, gamma=0.4, verbose=0):
    """
    Perform iterative metal artifact reduction using plastic-metal beam hardening correction.

    This function alternates between beam hardening correction (via `correct_BH_plastic_metal`)
    and reconstruction, refining the image over several iterations to suppress metal-induced artifacts.

    Args:
        ct_model: MBIRJAX cone beam model instance with `direct_recon` and `recon` methods.
        sino (jnp.ndarray):  Input sinogram data to be corrected.
        weights (jnp.ndarray): Transmission weights used in the reconstruction algorithm.
        num_BH_iterations (int, optional): Number of correction-reconstruction iterations. Defaults to 3.
        stop_threshold_change_pct (float, optional): Relative change threshold (%) for early stopping in MBIR. Defaults to 0.5.
        num_metal (int, optional): Number of metal materials to segment and correct for. Defaults to 1.
        order (int, optional): Maximum total degree of the beam hardening correction polynomial. Defaults to 3.
        alpha (float, optional): Degree-dependent scaling factor for regularization weights. Higher values penalize
            higher-order terms more strongly. Defaults to 1.
        beta (float, optional): Regularization strength for ridge regression. Defaults to 0.02.
        gamma (float, optional): Stabilization factor used in plastic correction. Multiplies the median of `s_p`
            to set a positive floor in the denominator, preventing division by near-zero or negative values. Defaults to 0.4.
        verbose (int, optional): Verbosity level for printing intermediate information. Defaults to 0.

    Returns:
         jnp.ndarray: The final corrected reconstruction after iterative beam hardening correction.

    Example:
        >>> recon = recon_BH_plastic_metal(
        ...     ct_model, sino, weights,
        ...     num_BH_iterations=3,
        ...     stop_threshold_change_pct=0.5,
        ...     num_metal=1,
        ...     order=3,
        ...     alpha=1,
        ...     beta=0.005,
        ...     verbose=1
        ... )
        >>> mj.slice_viewer(recon)
    """
    if verbose >= 1:
        print("\n************ Perform initial FDK reconstruction  **************")
    recon = ct_model.direct_recon(sino)

    for i in range(num_BH_iterations):
        # Estimate Corrected Sinogram
        corrected_sinogram = correct_BH_plastic_metal(ct_model, sino, recon, num_metal=num_metal, order=order, alpha=alpha, beta=beta, gamma=gamma)

        # Reconstruct Corrected Sinogram
        if verbose >= 1:
            print(f"\n************ Perform MBIR reconstruction {i + 1} **************")
        recon, _ = ct_model.recon(corrected_sinogram, weights=weights, init_recon=recon, stop_threshold_change_pct=stop_threshold_change_pct)

        if verbose >= 2:
            print(f"\n************ BH Iteration {i + 1}: Display plastic and metal mask **************")
            plastic_mask, metal_masks, plastic_scale, metal_scales = mjp.segment_plastic_metal(recon, num_metal)
            labels = ['Plastic Mask'] + [f'Metal {j + 1} Mask' for j in range(len(metal_masks))]
            mj.slice_viewer(plastic_mask, *metal_masks, vmin=0, vmax=1.0,
                            slice_label=labels,
                            title=f'Iteration {i + 1}: Comparison of Plastic and Metal Masks')

    return recon
