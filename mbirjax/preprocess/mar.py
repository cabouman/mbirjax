import jax
import jax.numpy as jnp
import mbirjax as mj
import mbirjax.preprocess as mjp


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
        >>> from mbirjax.preprocess import BH_correction
        >>> alpha = [1.0, 0.2, 0.1]  # Correction: sino + 0.2 * sino^2 + 0.1 * sino^3
        >>> corrected_sino = BH_correction(sino, alpha)
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


def correct_BH_plastic_metal(ct_model, measured_sino, recon, epsilon=2e-4, num_metal=1, order=(3, 4), include_const=False):
    """
    Beam-hardening correction for plastic and multiple metal components.

    Args:
        ct_model: Object with forward_project method and main_device attribute.
        measured_sino (jnp.ndarray): Raw sinogram.
        recon (jnp.ndarray): Reconstructed volume.
        epsilon (float): Regularization strength.
        num_metal (int): Number of metal materials to segment.
        order (Tuple[int, int]): (cross_order, metal_order)
        include_const (bool): Whether to include a constant term.

    Returns:
        corrected_sino (jnp.ndarray): Beam-hardening corrected sinogram.
    """
    cross_order, metal_order = order
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
        sino = ct_model.forward_project(jax.device_put(mask*recon, device)).reshape(-1)
        norm = jnp.max(jnp.abs(sino))
        metal_sinos.append(sino)
        metals.append(sino / norm)
    del metal_masks

    H_cols = []

    # Term 0: p
    H_cols.append(p)

    # Terms 1 to N*cross_order: p * m_i^j
    for m in metals:
        for j in range(1, cross_order):
            H_cols.append(p * (m ** j))

    # Joint cross terms: p * prod(m_i^j) for j = 1 to cross_order - 1
    if num_metal > 1:
        for j in range(1, cross_order):
            joint = p * jnp.prod(jnp.stack([m**j for m in metals]), axis=0)
            H_cols.append(joint)

    # Metal-only terms: m_i^j for j = 1 to metal_order - 1
    for m in metals:
        for j in range(1, metal_order):
            H_cols.append(m ** j)

    if include_const:
        H_cols.append(jnp.ones_like(p))

    order_total = len(H_cols)
    HtH = jnp.zeros((order_total, order_total))
    Hty = jnp.zeros(order_total)
    # Compute H^T y and H^T H
    for i in range(order_total):
        Hty = Hty.at[i].set(jnp.dot(H_cols[i], y))
        for j in range(order_total):
            HtH = HtH.at[i, j].set(jnp.dot(H_cols[i], H_cols[j]))

    # --- Solve for theta ---
    sigma_max = jnp.linalg.norm(HtH, ord=2)
    HtH_reg = HtH + (epsilon ** 2) * sigma_max * jnp.eye(len(Hty))
    theta = jnp.linalg.solve(HtH_reg, Hty)

    # --- Build correction ---
    idx = 0
    # Term 0: p
    linear_plastic_coef = theta[idx] * jnp.ones_like(p)
    idx += 1

    # p * m_i^j terms
    for m in metals:
        for j in range(1, cross_order):
            linear_plastic_coef += theta[idx] * (m ** j)
            idx += 1

    # joint p * m1^j * ... * mn^j terms
    if num_metal > 1:
        for j in range(1, cross_order):
            joint = jnp.prod(jnp.stack([m ** j for m in metals]), axis=0)
            linear_plastic_coef += theta[idx] * joint
            idx += 1

    # Metal-only sinogram
    metal_sino = jnp.zeros_like(y)
    for m in metals:
        for j in range(1, metal_order):
            metal_sino += theta[idx] * (m ** j)
            idx += 1

    # constant
    if include_const:
        metal_sino += theta[-1]

    # Regularize
    denom_floor = 1e-6 * jnp.linalg.norm(linear_plastic_coef)
    linear_plastic_coef = jnp.where(jnp.abs(linear_plastic_coef) > denom_floor, linear_plastic_coef, denom_floor)

    # Compute corrected plastic sinogram
    corrected_plastic_sino = p_normalization * (y - metal_sino) / linear_plastic_coef

    corrected_sino_flat = corrected_plastic_sino + sum(metal_sinos)
    corrected_sino = corrected_sino_flat.reshape(measured_sino.shape)
    return corrected_sino




def recon_BH_plastic_metal(ct_model, sino, weights, num_BH_iterations=3, stop_threshold_pct=0.5, verbose=0,
                           num_metal=1, order=(3, 4), include_const=False):
    """
    Perform iterative metal artifact reduction using plastic-metal beam hardening correction.

    This function repeatedly applies `BHC_plastic_metal()` and reconstructs from the corrected
    sinogram to iteratively refine the reconstruction and reduce metal artifacts.

    Args:
        ct_model: MBIRJAX cone beam model instance used for reconstruction.
        sino (jnp.ndarray): Input sinogram data to be corrected.
        weights (jnp.ndarray): Transmission weights used in the reconstruction algorithm.
        num_BH_iterations (int, optional): Number of beam hardening correction and reconstruction iterations to perform. Defaults to 3.
        stop_threshold_pct (float, optional): Threshold for stopping reconstruction iterations based on relative change in reconstruction. Defaults to 0.5.
        verbose (int, optional): Verbosity level for printing intermediate information. Defaults to 0.
        num_metal (int): Number of metal materials to segment.
        order (list, optional):
            List of two integers specifying the order of polynomial terms for plastic and metal components respectively.
            Defaults to [3, 4].
        include_const (bool, optional):
            Whether to include a constant term in the model. Defaults to False.

    Returns:
        jnp.ndarray: The final corrected reconstruction after iterative beam hardening correction.

    Example:
        >>> recon = recon_BH_plastic_metal(ct_model, sino, weights, num_BH_iterations=3, verbose=1, order=[3,4], include_const=False)
        >>> mj.slice_viewer(recon)
    """
    if verbose > 0:
        print("\n********* Perform initial FDK reconstruction **********")
    recon = ct_model.direct_recon(sino)

    for i in range(num_BH_iterations):
        # Estimate Corrected Sinogram
        corrected_sinogram = correct_BH_plastic_metal(ct_model, sino, recon, num_metal=num_metal, order=order, include_const=include_const)

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
