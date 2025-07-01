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


def correct_BH_plastic_metal(ct_model, measured_sino, recon, epsilon=2e-4, order=(3, 4), include_const=False):
    """
    Beam-hardening correction for objects containing a combination of plastic and metal.

    The function takes the measured sinogram and initial reconstruction as input, and it returns a corrected sinogram.
    It is designed to reduce metal artifacts for scans of objects made from a combination of plastic and metal material.
    The metal and plastic materials are each assumed to be composed of a single material.
    However, it should work fine for a combination of different plastics as long as their optical density properites do not vary too much.

    Note:
        The corrected sinogram should result in a more accurate reconstruction of the plastic, but may not accurately reconstruct the metal portion.

    Args:
        ct_model:
            Object with `forward_project` method and `main_device` attribute.
        measured_sino (jnp.ndarray):
            Raw sinogram data of shape (views, rows, cols).
        recon (jnp.ndarray):
            Reconstructed volume array corresponding to `measured_sino`.
        epsilon (float, optional):
            Tolerance for regularization.

    Returns:
        corrected_sino (jnp.ndarray):
            Beam-hardening corrected sinogram, same shape as `measured_sino`.

    Example:
        >>> corrected = correct_BH_plastic_metal(ct_model, measured_sino, recon)
    """
    # Segment recon into plastic and metal regions
    plastic_mask, metal_mask, plastic_scale, metal_scale = mjp.segment_plastic_metal(recon)

    # Forward project idealized plastic and metal components
    device = ct_model.main_device
    ideal_plastic_sino = plastic_scale * ct_model.forward_project(jax.device_put(plastic_mask, device)).reshape(-1)
    ideal_metal_sino = metal_scale * ct_model.forward_project(jax.device_put(metal_mask, device)).reshape(-1)
    y = measured_sino.reshape(-1)

    # Compute normalized plastic and metal sinograms with max amplitude = 1
    p_normalization = jnp.max(jnp.abs(ideal_plastic_sino))
    m_normalization = jnp.max(jnp.abs(ideal_metal_sino))
    p = ideal_plastic_sino / p_normalization
    m = ideal_metal_sino / m_normalization

    # Set order of models
    cross_order, metal_order = order

    # Build H matrix
    H = [p * m ** i for i in range(cross_order)]  # p, p*m, ..., p*m^(cross_order-1)
    H += [m ** i for i in range(1, metal_order)]  # m, m^2, ..., m^(metal_order-1)

    # Include constant if desired
    if include_const:
        H.append(jnp.ones_like(p))  # constant term at the end

    # Compute H^t H and H^t y
    order_total = len(H)
    HtH = jnp.zeros((order_total, order_total))
    Hty = jnp.zeros(order_total)

    for i in range(order_total):
        Hty = Hty.at[i].set(jnp.dot(H[i], y))
        for j in range(order_total):
            HtH = HtH.at[i, j].set(jnp.dot(H[i], H[j]))

    # Regularize and solve for least square value of theta that minimizes || y - H theta ||^2
    sigma_max = jnp.linalg.norm(HtH, ord=2)
    HtH_reg = HtH + (epsilon ** 2) * sigma_max * jnp.eye(order_total)
    theta = jnp.linalg.solve(HtH_reg, Hty)

    # Separate metal terms
    metal_start_idx = cross_order
    metal_sino = jnp.zeros_like(y)
    for idx in range(metal_start_idx, order_total):
        metal_sino += theta[idx] * H[idx]

    # Build linear plastic scaling denominator
    linear_plastic_coef = jnp.zeros_like(p)
    for idx in range(cross_order):
        linear_plastic_coef += theta[idx] * (m ** idx)

    denom_floor = 1e-6 * jnp.linalg.norm(linear_plastic_coef)
    linear_plastic_coef = jnp.where(jnp.abs(linear_plastic_coef) > denom_floor, linear_plastic_coef, denom_floor)

    # Numerator: subtract metal + constant if included
    numerator = y - metal_sino
    if include_const:
        numerator -= theta[-1]

    # Compute BH corrected version of plastic sinogram with metal removed
    corrected_plastic_sino = p_normalization * numerator / linear_plastic_coef

    # Combine corrected plastic and metal sinogram and reshape
    corrected_sino_flat = corrected_plastic_sino + ideal_metal_sino
    corrected_sino = corrected_sino_flat.reshape(measured_sino.shape)

    return corrected_sino



def recon_BH_plastic_metal(ct_model, sino, weights, num_BH_iterations=3, stop_threshold_pct=0.5, verbose=0,
                           order=(3, 4), include_const=False):
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
        corrected_sinogram = correct_BH_plastic_metal(ct_model, sino, recon, order=order, include_const=include_const)

        # Reconstruct Corrected Sinogram
        recon, _ = ct_model.recon(corrected_sinogram, weights=weights, init_recon=recon, stop_threshold_change_pct=stop_threshold_pct)

        if verbose > 0:
            print(f"\n************ BH Iteration {i + 1}: Display plastic and metal mask **************")
            plastic_mask, metal_mask, plastic_scale, metal_scale = mjp.segment_plastic_metal(recon)
            mj.slice_viewer(plastic_mask, metal_mask, vmin=0, vmax=1.0,
                            slice_label=['Plastic Mask', 'Metal Mask'],
                            title=f'Iteration {i + 1}: Comparison of Plastic and Metal Masks')

    return recon
