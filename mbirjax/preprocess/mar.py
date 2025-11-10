import jax
import jax.numpy as jnp
import mbirjax as mj
import mbirjax.preprocess as mjp
import random
import warnings
from jaxopt import OSQP


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


def _generate_metal_exponent_list(num_metal, max_order):
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


def _est_plastic_metal_sinos_from_recon(recon, num_metal, ct_model, device):
    """
    Segment plastic and metal regions from a reconstruction, project them,
    and return the unnormalized sinogram p, m0, m1, ... for beam hardening modeling.

    Args:
        recon (jnp.ndarray): Reconstructed image.
        num_metal (int): Number of metal types to segment.
        ct_model: Forward projection model with a `.forward_project()` method.
        device: JAX device to put the masks on for projection.

    Returns:
        plastic_sino_est (jnp.ndarray): Unnormalized plastic sino estimation.
        metal_sino_est (list of jnp.ndarray): List of unnormalized metal sino estimation.
    """
    # --- Segment plastic and metal regions in the reconstruction ---
    # plastic_mask: Mask for plastic regions.
    # metal_masks: List of masks for each metal.
    # plastic_scale: Scaling factor for the plastic region.
    # metal_scales: List of scaling factors for each metal region.
    plastic_mask, metal_masks, plastic_scale, metal_scales = mjp.segment_plastic_metal(recon, num_metal=num_metal)

    # --- Forward project, scale and vectorize plastic ---
    plastic_sino_est = plastic_scale * ct_model.forward_project(jax.device_put(plastic_mask, device)).reshape(-1)

    # --- Forward project the masked out metal regions ---
    metal_sino_est = []
    for mask, scale in zip(metal_masks, metal_scales):
        m = ct_model.forward_project(jax.device_put(mask * recon, device)).reshape(-1)
        metal_sino_est.append(m)

    return plastic_sino_est, metal_sino_est


def _get_column_H(col_index, plastic_sino_est, metal_sino_est, H_exponent_list):
    """
    Compute the col_index-th column of the matrix H.

    The column is constructed as a monomial of the form:
        H[:, col_index] = p^e0 * m_0^e1 * m_1^e2 * ... * m_{n-1}^en

    where (e0, e1, ..., en) = H_exponent_list[col_index].

    Args:
        col_index (int): Index of the column to compute.
        plastic_sino_est (jnp.ndarray): Normalized plastic sinogram estimation.
        metal_sino_est (list of jnp.ndarray): Normalized metal sinogram estimation [m_0, m_1, ..., m_{n-1}].
        H_exponent_list (list of tuple): List of exponent tuples defining each column of H.

    Returns:
        jnp.ndarray: The computed column of H (same shape as p and m_i).
    """
    exponents = H_exponent_list[col_index]
    assert len(exponents) == 1 + len(metal_sino_est), "Mismatch between exponent tuple and number of sinograms."

    col = plastic_sino_est ** exponents[0]
    for metal, exp in zip(metal_sino_est, exponents[1:]):
        col *= metal ** exp

    return col

def _get_row_H(row_index, plastic_sino_est, metal_sino_est, H_exponent_list):
    """
    Compute the row_index-th column of the matrix H.

    Args:
        row_index (int): Index of the row to compute.
        plastic_sino_est (jnp.ndarray): Normalized plastic sinogram estimation.
        metal_sino_est (list of jnp.ndarray): Normalized metal sinogram estimation [m_0, m_1, ..., m_{n-1}].
        H_exponent_list (list of tuple): List of exponent tuples defining each column of H.

    Returns:
        jnp.ndarray: The computed row of H.
    """
    pi = plastic_sino_est[row_index]
    mi = [m[row_index] for m in metal_sino_est]
    row_vals = []
    for exps in H_exponent_list:
        val = (pi ** exps[0])
        for mk, ek in zip(mi, exps[1:]):
            val = val * (mk ** ek)
        row_vals.append(val)
    return jnp.asarray(row_vals)


def _find_most_violated_constraints(measured_sino, plastic_sino_est, metal_sino_est, theta, H_exponent_list, num_cross_terms):
    """
    Compute the most violated constraints for the beam hardening model.

    The BH model enforces two types of inequality constraints:
        1. Plastic positivity:        H_p[i,:] θ_p ≥ 0
       2. Residual positivity:       y[i] − H_m[i,:] θ_m ≥ 0

    This function evaluates the indices and values of the entries that most violate
    the constraints.

    Returns:
        i_min_Sp (int): Index of smallest Sp entry.
       v_min_Sp (float): Value of Sp[i_min_Sp].
       i_min_residual (int): Index of smallest (y − Sm) entry.
       v_min_residual (float): Value of (y − Sm)[i_min_residual].
    """
    num_cols = len(H_exponent_list)
    Sp = jnp.zeros_like(measured_sino)
    for i in range(0, 1 + num_cross_terms):
        # Use a dummy input of ones to extract the structure of the i-th column (i.e., coefficient of p)
        Sp = Sp + theta[i] * _get_column_H(i, jnp.ones_like(plastic_sino_est), metal_sino_est, H_exponent_list)

    # y_minus_Sm = y - metal-only
    y_minus_Sm = measured_sino
    # Subtract metal-only terms (from H columns after the cross terms)
    for j in range(1 + num_cross_terms, num_cols):
        y_minus_Sm = y_minus_Sm - theta[j] * _get_column_H(j, plastic_sino_est, metal_sino_est, H_exponent_list)

    # Lower-bound violator: minimize Sp and y-Sm
    i_min_Sp = int(jnp.argmin(Sp))
    i_min_residual = int(jnp.argmin(y_minus_Sm))

    v_min_Sp = Sp[i_min_Sp]
    v_min_residual = y_minus_Sm[i_min_residual]

    return i_min_Sp, v_min_Sp, i_min_residual, v_min_residual



def _estimate_BH_model_params_using_OSQP(Q, c, G, h):
    """
    Solve the constrained quadratic optimization problem:

        minimize_θ   0.5 * θᵀ Q θ + cᵀ θ
        subject to   G θ ≤ h

    The problem is solved using the JAXOpt `OSQP` solver when constraints are provided.
    If `G` or `h` is `None`, an unconstrained least-squares solution is computed directly.

    Args:
        Q (jnp.ndarray): Quadratic term matrix.
        c (jnp.ndarray): Linear term vector.
        G (jnp.ndarray): Inequality constraint matrix.
        h (jnp.ndarray): Right-hand side vector for the inequality constraints.

    Returns:
        jnp.ndarray: Solution vector θ.
    """
    if G is None or h is None:
        # No constraints - solve unconstrained QP directly
        theta = jnp.linalg.solve(Q, -c)
    else:
        solver = OSQP()
        sol = solver.run(params_obj=(Q, c), params_eq=None, params_ineq=(G, h)).params
        theta = sol.primal
    return theta

def _compute_entry_for_OSQP(plastic_sino_est, metal_sino_est, measured_sino, H_exponent_list, num_cross_terms, alpha, beta):
    """Compute entry for OSQP quadratic programming solver."""
    num_cols = len(H_exponent_list)

    HtH = jnp.zeros((num_cols, num_cols))
    Hty = jnp.zeros(num_cols)

    # Compute the upper triangle of HtH and mirror it.
    for i in range(num_cols):
        h_i = _get_column_H(i, plastic_sino_est, metal_sino_est, H_exponent_list)
        Hty = Hty.at[i].set(jnp.dot(h_i, measured_sino))
        for j in range(i, num_cols):
            h_j = _get_column_H(j, plastic_sino_est, metal_sino_est, H_exponent_list)
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

    Q = HtH + lambda_reg * weight_matrix
    c = -Hty

    return Q, c

def _estimate_BH_model_params(plastic_sino_est, metal_sino_est, measured_sino, H_exponent_list, num_cross_terms, alpha, beta, num_constraint_update_iter=10, tolerance=-1e-5):
    """
    Estimate polynomial beam hardening model parameters with iterative constraints search.

    This function solves a regularized least squares problem with inequality constraints to
    enforce nonnegativity on the plastic and residual sinograms. The optimization problem is:

        minimize_θ   0.5‖Hθ − y‖² + 0.5λ‖θ‖²_Λ
        subject to   H_p[i,:] θ_p ≥ 0 and y[i] − H_m[i,:] θ_m ≥ 0

    where:
        - H_p contains the plastic and plastic–metal cross-term columns.
        - H_m contains the metal-only columns.

    The function uses an iterative active constraint selection method:
        1. Start from the unconstrained least squares estimate.
        2. Identify indices where the constraints are violated.
        3. Add the most violated constraints to the set.
        4. Re-solve the quadratic program (QP) using OSQP.
        5. Repeat until all constraints are satisfied or `num_constraint_update_iter` is reached.

    Args:
        plastic_sino_est (jnp.ndarray): Normalized plastic sinogram estimation.
        metal_sino_est (list of jnp.ndarray): List of normalized metal sino estimation.
        measured_sino (jnp.ndarray): Measured sinogram.
        H_exponent_list (list of tuple[int]): List of exponent tuples defining each column of the matrix H.
        num_cross_terms (int): Number of cross terms (plastic × metal); remaining terms are metal-only.
        alpha (float): Regularization exponent; higher alpha penalizes higher-degree terms more.
        beta (float): Regularization strength scaling factor.
        num_constraint_update_iter (int): Number of iterations for updating constraints.
        tolerance (float): Tolerance for stopping criteria.

    Returns:
        theta (jnp.ndarray): Estimated model parameters corresponding to each column in H.

    """
    num_cols = len(H_exponent_list)
    dp = 1 + num_cross_terms

    # Lists that store the indices of the points that most violate the constraints
    C_p = []
    C_m = []

    # Construct the entries Q, c, G and h of OSQP for solving the constraint optimization
    Q, c = _compute_entry_for_OSQP(plastic_sino_est, metal_sino_est, measured_sino, H_exponent_list, num_cross_terms, alpha, beta)
    G = jnp.zeros((0, num_cols))  # no active constraints yet
    h = jnp.zeros((0,))

    # Initial θ solved without constraint
    theta = _estimate_BH_model_params_using_OSQP(Q, c, G=None, h=None)

    for iter in range(num_constraint_update_iter):
        # Find the indices and values of the points that most violate each constraint
        i_min_Sp, v_min_Sp, i_min_residual, v_min_residual = _find_most_violated_constraints(measured_sino, plastic_sino_est, metal_sino_est, theta, H_exponent_list, num_cross_terms)

        # (1) Hp θp ≥ 0  ->  (-Hp) θ ≤ 0
        if v_min_Sp < tolerance and (i_min_Sp not in C_p):
            row_p = _get_row_H(i_min_Sp, jnp.ones_like(measured_sino), metal_sino_est, H_exponent_list)
            # Negative row_p[:dp] to ensure Hpθp >= 0
            g_p = jnp.concatenate([-row_p[:dp], jnp.zeros((num_cols - dp,))])
            h_p = jnp.array([0.0])
            G = jnp.vstack([G, g_p[None, :]])
            h = jnp.concatenate([h, h_p])
            C_p.append(i_min_Sp)

        # (2) y − Hm θm ≥ 0  ->  (Hm) θ ≤ y
        if v_min_residual < tolerance and (i_min_residual not in C_m):
            row_m = _get_row_H(i_min_residual, plastic_sino_est, metal_sino_est, H_exponent_list)
            # Positive row_m[dp:] to ensure y-Hmθm >= 0
            g_m = jnp.concatenate([jnp.zeros(dp), row_m[dp:]])
            h_m = jnp.array([measured_sino[i_min_residual]])
            G = jnp.vstack([G, g_m[None, :]])
            h = jnp.concatenate([h, h_m])
            C_m.append(i_min_residual)

        # Early exit if both constraints are satisfied (within tolerances)
        if (v_min_Sp >= tolerance) and (v_min_residual >= tolerance):
            break
        theta = _estimate_BH_model_params_using_OSQP(Q, c, G, h)
    return theta


def _correct_plastic_sinogram(measured_sino, plastic_sino_est, metal_sino_est, theta, H_exponent_list, num_cross_terms, num_metal_terms, p_normalization, gamma):
    """
    Perform beam hardening correction on the plastic sinogram.

    This function subtracts the metal-only contributions from the measured sinogram
    and normalizes the result using the linear plastic component, yielding a corrected
    sinogram that approximates the plastic-only contribution.

    The correction is based on a polynomial matrix H whose columns correspond to:
        - Plastic term: p
        - Cross terms: p*m, p*m^2, ...
        - Metal-only terms: m, m^2, m^3, ...

    The H matrix looks like: [p, p*m, p*m^2, m, m^2, m^3]
    The correction is applied as:
        corrected_plastic = p_normalization * max(y - H_metal·θ_m, 0) / (max(H_plastic·θ_p, γ * median(H_plastic·θ_p))
    The stabilization term involving γ prevents division by near-zero or negative values, reducing streaks
    and numerical instability.

    Args:
        measured_sino (jnp.ndarray): Measured sinogram.
        plastic_sino_est (jnp.ndarray): Normalized plastic sino estimation.
        metal_sino_est (list of jnp.ndarray): List of normalized metal sino estimation..
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
    Sp = jnp.zeros_like(measured_sino)
    for i in range(0, 1 + num_cross_terms):
        # Use a dummy input of ones to extract the structure of the i-th column (i.e., coefficient of p)
        Sp += theta[i] * _get_column_H(i, jnp.ones_like(measured_sino), metal_sino_est, H_exponent_list)

    y_minus_Sm = measured_sino
    # Subtract metal-only terms (from H columns after the cross terms)
    for j in range(1 + num_cross_terms, 1 + num_cross_terms + num_metal_terms):
        y_minus_Sm -= theta[j] * _get_column_H(j, plastic_sino_est, metal_sino_est, H_exponent_list)

    # Enforce non-negativity on the residual sinogram (plastic + cross terms)
    y_minus_Sm = jnp.maximum(y_minus_Sm, 0)

    # Compute median of plastic coefficients (used to define a stabilization floor)
    median_plastic_coef = jnp.median(Sp)
    Sp_floor = gamma * median_plastic_coef

    # A negative median would be non-physical and may indicate instability in the algorithm
    # In that case, issue a runtime warning to flag the potential problem
    if float(median_plastic_coef) <= 0:
        warnings.warn("Median of Sp is negative", RuntimeWarning)

    # Clamp Sp at Sp_floor to prevent division by very small or negative values
    clamped_plastic_coef = jnp.maximum(Sp, Sp_floor)
    corrected_plastic_sino = p_normalization * y_minus_Sm / clamped_plastic_coef

    return corrected_plastic_sino

def _estimate_plastic_scaling(plastic_sino_est, metal_sino_est, measured_sino, plastic_sino_corrected):
    # Compute a scaling factor by performing least-squares fitting between the corrected plastic sinogram
    # and the measured sinogram at plastic-only locations (i.e., where plastic is present and all metals are absent)
    metal_absent = jnp.ones_like(plastic_sino_est, dtype=bool)
    for metal in metal_sino_est:
        metal_absent = metal_absent & (metal == 0)

    # Find the metal-absent indices
    condition = (plastic_sino_est != 0) & metal_absent

    plastic_sino_scale = mjp.compute_scaling_factor(measured_sino[condition], plastic_sino_corrected[condition])
    return plastic_sino_scale

def correct_sino_plastic_metal(ct_model, measured_sino, recon, num_metal=1, order=3, alpha=1, beta=0.002, gamma=0.1, num_constraint_update_iter=10):
    """
    This function corrects the measured sinogram of an object with plastic and multiple metal components by fitting a
    beam hardening model to the sinogram and removing the metal contributions.

    Args:
        ct_model: CT model object with a `forward_project` method and a `main_device` attribute.
        measured_sino (jnp.ndarray): Raw measured sinogram.
        recon (jnp.ndarray): Reconstructed 3D volume used for segmentation of plastic and metal regions.
        num_metal (int, optional): Number of metal materials to segment and correct for. Defaults to 1.
        order (int, optional): Maximum total degree of the beam hardening correction polynomial. Defaults to 3.
        alpha (float, optional): Degree-dependent scaling factor for regularization weights. Higher values penalize
            higher-order terms more strongly. Defaults to 1.
        beta (float, optional): Regularization strength for ridge regression. Defaults to 0.002.
        gamma (float, optional): Stabilization factor. Defaults to 0.1.
        num_constraint_update_iter (int, optional): Number of iterations for updating constraints. Defaults to 10.

    Returns:
        jnp.ndarray: Beam-hardening corrected sinogram of the same shape as `measured_sino`.
    """
    # Construct the exponent list of the metal sinograms.
    metal_exponent_list = _generate_metal_exponent_list(num_metal, order)
    cross_exponent_list = _generate_metal_exponent_list(num_metal, order - 1)
    num_metal_terms = len(metal_exponent_list)
    num_cross_terms = len(cross_exponent_list)

    # Construct the exponent list for each column of the matrix H.
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
    sino_shape = measured_sino.shape
    measured_sino = measured_sino.reshape(-1)

    # Get normalized sinogram p and [m_0, m_1, ...]
    plastic_sino_est, metal_sino_est = _est_plastic_metal_sinos_from_recon(recon, num_metal, ct_model, device)
    plastic_sino_scale = jnp.max(jnp.abs(plastic_sino_est))
    metal_sino_scale = [jnp.max(jnp.abs(arr)) for arr in metal_sino_est]
    plastic_sino_est = plastic_sino_est / plastic_sino_scale
    metal_sino_est = [arr / norm for arr, norm in zip(metal_sino_est, metal_sino_scale)]

    # Estimate beam hardening model parameters theta
    theta = _estimate_BH_model_params(plastic_sino_est, metal_sino_est, measured_sino, H_exponent_list, num_cross_terms, alpha, beta, num_constraint_update_iter)
    # print(f'theta = {theta}')

    # Compute the corrected plastic sinogram
    plastic_sino_corrected = _correct_plastic_sinogram(measured_sino, plastic_sino_est, metal_sino_est, theta, H_exponent_list,
                                                       num_cross_terms, num_metal_terms, plastic_sino_scale, gamma)

    # Compute and apply the scaling of the corrected plastic sino
    plastic_sino_corrected_scale = _estimate_plastic_scaling(plastic_sino_est, metal_sino_est, measured_sino, plastic_sino_corrected)
    scaled_corrected_plastic_sino = plastic_sino_corrected_scale * plastic_sino_corrected

    # Combine the scaled corrected plastic sino and the metal sinos
    corrected_sino_flat = scaled_corrected_plastic_sino + sum(arr * norm for arr, norm in zip(metal_sino_est, metal_sino_scale))
    corrected_sino = corrected_sino_flat.reshape(sino_shape)

    return corrected_sino


def recon_plastic_metal(ct_model, sino, weights, num_BH_iterations=3, num_constraint_update_iter=10, stop_threshold_change_pct=0.5,
                        num_metal=1, order=3, alpha=1, beta=0.002, gamma=0.1, verbose=0):
    """
    Perform iterative metal artifact reduction for object with plastic and metal components.

    This function alternates between adaptive beam hardening correction (via `correct_sino_plastic_metal`)
    and reconstruction, refining the image over several iterations to suppress metal-induced artifacts.

    Args:
        ct_model: MBIRJAX cone beam model instance with `direct_recon` and `recon` methods.
        sino (jnp.ndarray):  Input sinogram data to be corrected.
        weights (jnp.ndarray): Transmission weights used in the reconstruction algorithm.
        num_BH_iterations (int, optional): Number of correction-reconstruction iterations. Defaults to 3.
        num_constraint_update_iter (int, optional): Number of iterations for updating constraints.
            At each iteration, the most violated constraints are activated and the quadratic program is re-solved via OSQP.
        stop_threshold_change_pct (float, optional): Relative change threshold (%) for early stopping in MBIR. Defaults to 0.5.
        num_metal (int, optional): Number of metal materials to segment and correct for. Defaults to 1.
        order (int, optional): Maximum total degree of the beam hardening correction polynomial. Defaults to 3.
        alpha (float, optional): Degree-dependent scaling factor for regularization weights. Higher values penalize
            higher-order terms more strongly. Defaults to 1.
        beta (float, optional): Regularization strength for ridge regression. Defaults to 0.002.
        gamma (float, optional): Stabilization factor used in plastic correction. Multiplies the median of `s_p`
            to set a positive floor in the denominator, preventing division by near-zero or negative values. Defaults to 0.1.
        verbose (int, optional): Verbosity level for printing intermediate information. Defaults to 0.

    Returns:
         jnp.ndarray: The final corrected reconstruction after iterative beam hardening correction.

    Example:
        >>> recon = recon_plastic_metal(
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
        corrected_sinogram = correct_sino_plastic_metal(ct_model, sino, recon, num_metal=num_metal, order=order, alpha=alpha, beta=beta, gamma=gamma, num_constraint_update_iter=num_constraint_update_iter)

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
