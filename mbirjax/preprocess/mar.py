import jax
import jax.numpy as jnp
import numpy as np
import functools
import mbirjax as mj
import mbirjax.preprocess as mjp
from . import pipeline
import random
import warnings
from scipy.sparse import csc_matrix
import osqp

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

    return _gen_huber_weights_kernel(jnp.asarray(weights), jnp.asarray(sino_error), T, delta, epsilon)


@jax.jit
def _gen_huber_weights_kernel(weights, sino_error, T, delta, epsilon):
    """Jitted body of :func:`gen_huber_weights`.

    Eagerly this chain materialized ~5 full-sinogram temporaries (std, the two norms' squares,
    normalized_error, abs, where) and the two ``jnp.linalg.norm`` reductions each ran as separate
    dispatches (with their own cross-device collectives on sharded inputs).  Jitted, XLA fuses the
    elementwise chain into the reductions and the final ``where`` -- no full-size temporaries, one
    executable.  (``jnp.linalg.norm`` with default ord is a pure XLA sum-of-squares reduction; it does
    not call cuSolver.)"""
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


def BH_correction(sino, alpha, batch_size=64, devices=None):
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
    alpha = jnp.asarray(alpha)

    # Per-view-batch polynomial evaluation, driven through the shared pipeline driver: jitted so XLA
    # fuses the powers + weighted sum (buffer reuse, bounded memory), and memory-bounded / optionally
    # multi-device via map_view_batches.  The correction is per-pixel, so batching is exact.
    @jax.jit
    def kernel(sino_batch):
        corrected = jnp.zeros_like(sino_batch)
        for k in range(len(alpha)):
            corrected = corrected + alpha[k] * jnp.power(sino_batch, k + 1)
        return corrected

    return pipeline.map_view_batches(sino, kernel, batch_size, devices=devices)


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


def _est_plastic_metal_sinos_from_recon(recon, num_metal, ct_model):
    """
    Segment plastic and metal regions from a reconstruction, project them,
    and return the unnormalized sinogram p, m0, m1, ... for beam hardening modeling.

    Args:
        recon (jnp.ndarray): Reconstructed image.
        num_metal (int): Number of metal types to segment.
        ct_model: Forward projection model with a `.forward_project()` method.

    Returns:
        plastic_sino_est (jnp.ndarray): Unnormalized plastic sino estimation.
        metal_sino_est (list of jnp.ndarray): List of unnormalized metal sino estimation.
    """
    # Shard the recon once at entry (a no-op if it is already sharded, e.g. from a recon with
    # output_sharded=True).  The segmentation then runs on-device across the shards -- including the
    # Otsu histogram, which reduces per-shard partial histograms without gathering the volume -- and the
    # 1+num_metal forward projections below consume the SAME sharded recon instead of re-uploading a
    # recon-sized array per mask.
    recon = ct_model._shard_recon(recon)

    # Slice-padding info: the sharded recon may be zero-padded on the slice axis (to divide evenly
    # across devices).  valid_mask (True on real slices; broadcastable, tiny; None when unpadded) and
    # num_real_slices let the segmentation exclude the padded slices from its statistics and masks.
    pl = ct_model.recon_placement
    valid_mask = pl.real_mask(recon.ndim)

    # --- Segment plastic and metal regions in the reconstruction ---
    # plastic_mask: Mask for plastic regions.
    # metal_masks: List of masks for each metal.
    # plastic_scale: Scaling factor for the plastic region.
    # metal_scales: List of scaling factors for each metal region.
    plastic_mask, metal_masks, plastic_scale, metal_scales = mjp.segment_plastic_metal(
        recon, num_metal=num_metal, valid_mask=valid_mask, num_real_slices=pl.real_size)

    # --- Forward project and scale plastic ---
    # The masks/recon are already sharded, so forward_project consumes them with no further movement.
    # Keep the OUTPUT view-sharded (output_sharded=True) and do NOT flatten: the whole correction below
    # runs on these sharded 3-D sinograms so no full-length sino vector ever lands on a single device.
    plastic_sino_est = plastic_scale * ct_model.forward_project(plastic_mask, output_sharded=True)

    # --- Forward project the masked out metal regions ---
    metal_sino_est = []
    for mask, scale in zip(metal_masks, metal_scales):
        m = ct_model.forward_project(mask * recon, output_sharded=True)
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

    # Most exponents are 0 or 1 (the exponent tuples are sparse), so skip the no-op factors instead of
    # materializing full-sinogram-sized ones/copies: x**0 == 1 exactly (skip the factor) and x**1 == x
    # exactly (use the array directly, no power op).  Byte-identical to the dense product.
    col = None
    for arr, exp in zip([plastic_sino_est] + list(metal_sino_est), exponents):
        if exp == 0:
            continue
        term = arr if exp == 1 else arr ** exp
        col = term if col is None else col * term
    if col is None:
        # All-zero exponent tuple (the constant column) -- excluded by construction from H, but handle
        # it correctly if a caller ever asks.
        col = jnp.ones_like(plastic_sino_est)
    return col

def _get_row_H(row_index, plastic_sino_est, metal_sino_est, H_exponent_list):
    """
    Compute the row_index-th row of the matrix H.

    Args:
        row_index (int): Index of the row to compute.
        plastic_sino_est (jnp.ndarray): Normalized plastic sinogram estimation.
        metal_sino_est (list of jnp.ndarray): Normalized metal sinogram estimation [m_0, m_1, ..., m_{n-1}].
        H_exponent_list (list of tuple): List of exponent tuples defining each column of H.

    Returns:
        jnp.ndarray: The computed row of H.
    """
    # row_index is a FLAT pixel index (from argmin over the flattened sinogram); the sinos are now 3-D
    # (view-sharded), so index the flattened view (reshape(-1) preserves the sharding; this is a
    # single-element gather).
    pi = plastic_sino_est.reshape(-1)[row_index]
    mi = [m.reshape(-1)[row_index] for m in metal_sino_est]
    row_vals = []
    for exps in H_exponent_list:
        val = (pi ** exps[0])
        for mk, ek in zip(mi, exps[1:]):
            val = val * (mk ** ek)
        row_vals.append(val)
    return jnp.asarray(row_vals)


def _find_most_violated_constraints(measured_sino, plastic_sino_est, metal_sino_est, theta, H_exponent_list, num_cross_terms, view_mask=None):
    """
    Compute the most violated constraints for the beam hardening model.

    The BH model enforces two types of inequality constraints:
        1. Plastic positivity:        H_p[i,:] θ_p ≥ 0
        2. Residual positivity:       y[i] − H_m[i,:] θ_m ≥ 0

    This function evaluates the indices and values of the entries that most violate
    the constraints.  When the sinograms are device-form (view-sharded, possibly zero-padded on the view
    axis), ``view_mask`` (1 on real views, 0 on padded, broadcasting over the sinogram) excludes the
    padded views from the argmin so a padded entry is never selected as a constraint.

    Returns:
        i_min_Sp (int): FLAT index of smallest Sp entry (into the flattened sinogram).
        v_min_Sp (float): Value of Sp at that entry.
        i_min_residual (int): FLAT index of smallest (y − Sm) entry.
        v_min_residual (float): Value of (y − Sm) at that entry.
    """
    num_cols = len(H_exponent_list)
    # The coefficient of p in column i is the column with its p factor removed, so zero the p exponent
    # (the sparse _get_column_H then SKIPS that factor) rather than passing a dummy full-sinogram ones
    # array -- eagerly, the ones would be reallocated (and multiplied) on every loop iteration.
    p_coeff_exponents = [(0,) + exps[1:] for exps in H_exponent_list]
    Sp = jnp.zeros_like(measured_sino)
    for i in range(0, 1 + num_cross_terms):
        Sp = Sp + theta[i] * _get_column_H(i, plastic_sino_est, metal_sino_est, p_coeff_exponents)

    # y_minus_Sm = y - metal-only
    y_minus_Sm = measured_sino
    # Subtract metal-only terms (from H columns after the cross terms)
    for j in range(1 + num_cross_terms, num_cols):
        y_minus_Sm = y_minus_Sm - theta[j] * _get_column_H(j, plastic_sino_est, metal_sino_est, H_exponent_list)

    # Lower-bound violator: minimize Sp and y-Sm over the REAL views (padded views set to +inf so they
    # can't win the argmin).  argmin over the N-D sinogram returns a FLAT index; read the values back via
    # the flattened view (reshape(-1) preserves the sharding).
    Sp_masked = Sp if view_mask is None else jnp.where(view_mask, Sp, jnp.inf)
    ymSm_masked = y_minus_Sm if view_mask is None else jnp.where(view_mask, y_minus_Sm, jnp.inf)
    i_min_Sp = int(jnp.argmin(Sp_masked))
    i_min_residual = int(jnp.argmin(ymSm_masked))

    v_min_Sp = Sp.reshape(-1)[i_min_Sp]
    v_min_residual = y_minus_Sm.reshape(-1)[i_min_residual]

    return i_min_Sp, v_min_Sp, i_min_residual, v_min_residual



def _estimate_BH_model_params_using_OSQP(P, q, A, u):
    """
    Solve the constrained quadratic optimization problem:

        minimize_θ   0.5 * θᵀ P θ + qᵀ θ
        subject to   A θ ≤ u

    The problem is solved using the OSQP solver when constraints are provided.
    If `A` or `u` is `None`, an unconstrained least-squares solution is computed directly.

    Args:
        P (jnp.ndarray): Quadratic term matrix.
        q (jnp.ndarray): Linear term vector.
        A (jnp.ndarray): Inequality constraint matrix.
        u (jnp.ndarray): Right-hand side vector for the inequality constraints.

    Returns:
        jnp.ndarray: Solution vector θ.
    """
    # Convert to numpy for linalg.solve and OSQP
    P_numpy = np.asarray(P)
    q_numpy = np.asarray(q)

    if A is None or u is None:
        # No constraints - solve unconstrained QP directly.  The system is tiny (num_cols x num_cols),
        # so solve on the HOST with numpy, like the OSQP branch below: jnp.linalg.solve dispatches to
        # cuSolver on GPU, whose handle/workspace allocation lives OUTSIDE XLA's memory pool and fails
        # when the pool has reserved nearly all device memory (XLA_PYTHON_CLIENT_MEM_FRACTION) --
        # an async failure that only surfaces when theta is first read.
        theta = np.linalg.solve(P_numpy, -q_numpy)
        return jnp.asarray(theta, dtype=jnp.float32)

    # Convert arrays as required by OSQP. These matrices are small.
    A_numpy = np.asarray(A)
    u_numpy = np.asarray(u)

    P_sparse = csc_matrix(P_numpy)
    A_sparse = csc_matrix(A_numpy)

    solver = osqp.OSQP()
    solver.setup(P=P_sparse, q=q_numpy, A=A_sparse, l=None, u=u_numpy, alpha=1.0, verbose=0)
    result = solver.solve()
    theta = jnp.asarray(result.x, dtype=jnp.float32)

    return theta

def _compute_entry_for_OSQP(plastic_sino_est, metal_sino_est, measured_sino, H_exponent_list, num_cross_terms, alpha, beta):
    """Compute entries for OSQP quadratic programming solver."""
    num_cols = len(H_exponent_list)

    HtH = jnp.zeros((num_cols, num_cols))
    Hty = jnp.zeros(num_cols)

    # Compute the upper triangle of HtH and mirror it.  Use jnp.sum(a*b) rather than jnp.dot so the
    # inner products work on the N-D view-sharded sinograms (a cross-device all-reduce).  Padded views
    # contribute 0 (their h_i are built from plastic/metal, which are 0 on padded views), so no mask is
    # needed here.
    for i in range(num_cols):
        h_i = _get_column_H(i, plastic_sino_est, metal_sino_est, H_exponent_list)
        Hty = Hty.at[i].set(jnp.sum(h_i * measured_sino))
        for j in range(i, num_cols):
            h_j = _get_column_H(j, plastic_sino_est, metal_sino_est, H_exponent_list)
            dot_ij = jnp.sum(h_i * h_j)
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

    P = HtH + lambda_reg * weight_matrix
    q = -Hty

    return P, q

def _estimate_BH_model_params(plastic_sino_est, metal_sino_est, measured_sino, H_exponent_list, num_cross_terms, alpha, beta, num_constraint_update_iter=10, tolerance=-1e-5, view_mask=None):
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

    # Construct the entries P, q, A and u of OSQP for solving the constraint optimization
    P, q = _compute_entry_for_OSQP(plastic_sino_est, metal_sino_est, measured_sino, H_exponent_list, num_cross_terms, alpha, beta)
    A = jnp.zeros((0, num_cols))  # no active constraints yet
    u = jnp.zeros((0,))

    # Initial θ solved without constraint
    theta = _estimate_BH_model_params_using_OSQP(P, q, A=None, u=None)

    for iter in range(num_constraint_update_iter):
        # Find the indices and values of the points that most violate each constraint
        i_min_Sp, v_min_Sp, i_min_residual, v_min_residual = _find_most_violated_constraints(measured_sino, plastic_sino_est, metal_sino_est, theta, H_exponent_list, num_cross_terms, view_mask=view_mask)

        # (1) Hp θp ≥ 0  ->  (-Hp) θ ≤ 0
        if v_min_Sp < tolerance and (i_min_Sp not in C_p):
            # Coefficient-of-p row: zero the p exponent (pi**0 == 1 exactly) instead of allocating a
            # full-sinogram dummy ones array just to read its one pixel.
            p_coeff_exponents = [(0,) + exps[1:] for exps in H_exponent_list]
            row_p = _get_row_H(i_min_Sp, plastic_sino_est, metal_sino_est, p_coeff_exponents)
            # Negative row_p[:dp] to ensure Hpθp >= 0
            A_p = jnp.concatenate([-row_p[:dp], jnp.zeros((num_cols - dp,))])
            u_p = jnp.array([0.0])
            A = jnp.vstack([A, A_p[None, :]])
            u = jnp.concatenate([u, u_p])
            C_p.append(i_min_Sp)

        # (2) y − Hm θm ≥ 0  ->  (Hm) θ ≤ y
        if v_min_residual < tolerance and (i_min_residual not in C_m):
            row_m = _get_row_H(i_min_residual, plastic_sino_est, metal_sino_est, H_exponent_list)
            # Positive row_m[dp:] to ensure y-Hmθm >= 0
            A_m = jnp.concatenate([jnp.zeros(dp), row_m[dp:]])
            # i_min_residual is a FLAT pixel index; the sinogram is 3-D (view-sharded), so read the one
            # pixel through the flattened view (reshape preserves the sharding) -- integer-indexing the
            # 3-D array would silently grab a whole VIEW along axis 0 instead.
            u_m = jnp.array([measured_sino.reshape(-1)[i_min_residual]])
            A = jnp.vstack([A, A_m[None, :]])
            u = jnp.concatenate([u, u_m])
            C_m.append(i_min_residual)

        # Early exit if both constraints are satisfied (within tolerances)
        if (v_min_Sp >= tolerance) and (v_min_residual >= tolerance):
            break
        theta = _estimate_BH_model_params_using_OSQP(P, q, A, u)
    return theta


def _correct_plastic_sinogram(measured_sino, plastic_sino_est, metal_sino_est, theta, H_exponent_list, num_cross_terms, num_metal_terms, p_normalization, gamma, view_mask=None, num_real_pixels=None):
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
        corrected_plastic = p_normalization * max(y - H_metal·θ_m, 0) / (max(H_plastic·θ_p, γ * mean(H_plastic·θ_p))
    The stabilization term involving γ prevents division by near-zero or negative values, reducing streaks
    and numerical instability.  (``view_mask`` / ``num_real_pixels`` restrict the mean to the real views
    when the sinogram is device-form and zero-padded.)

    Args:
        measured_sino (jnp.ndarray): Measured sinogram.
        plastic_sino_est (jnp.ndarray): Normalized plastic sino estimation.
        metal_sino_est (list of jnp.ndarray): List of normalized metal sino estimation.
        theta (jnp.ndarray): Estimated coefficients for the polynomial terms in H.
        H_exponent_list (list of tuple): Exponent tuples defining each column of H.
        num_cross_terms (int): Number of cross terms involving both p and metal.
        num_metal_terms (int): Number of metal-only terms in H.
        p_normalization (float): Normalization factor applied to p.
        gamma (float, optional): Stabilization factor.

    Returns:
        corrected_plastic_sino (jnp.ndarray): Beam-hardening-corrected plastic sinogram.
    """

    # Compute the denominator (linear plastic + cross terms) from the first (1 + num_cross_terms) columns
    # of H.  The coefficient of p in column i is the column with its p factor removed, so zero the p
    # exponent (the sparse _get_column_H then SKIPS that factor) rather than passing a dummy
    # full-sinogram ones array -- eagerly, the ones would be reallocated on every loop iteration.
    p_coeff_exponents = [(0,) + exps[1:] for exps in H_exponent_list]
    Sp = jnp.zeros_like(measured_sino)
    for i in range(0, 1 + num_cross_terms):
        Sp += theta[i] * _get_column_H(i, plastic_sino_est, metal_sino_est, p_coeff_exponents)

    y_minus_Sm = measured_sino
    # Subtract metal-only terms (from H columns after the cross terms)
    for j in range(1 + num_cross_terms, 1 + num_cross_terms + num_metal_terms):
        y_minus_Sm -= theta[j] * _get_column_H(j, plastic_sino_est, metal_sino_est, H_exponent_list)

    # Enforce non-negativity on the residual sinogram (plastic + cross terms)
    y_minus_Sm = jnp.maximum(y_minus_Sm, 0)

    # Central plastic coefficient, used to define a stabilization floor.  We use the MEAN rather than the
    # median so it is a cross-device all-reduce (median needs a global sort, which would gather the
    # sharded sinogram back to one device); over the sinogram support the two are close, and this only
    # sets a floor.  When the sinogram is view-sharded and zero-padded, exclude the padded views via
    # view_mask so they don't drag the mean toward 0.
    if view_mask is None:
        mean_plastic_coef = jnp.mean(Sp)
    else:
        mean_plastic_coef = jnp.sum(Sp * view_mask) / num_real_pixels
    Sp_floor = gamma * mean_plastic_coef

    # A negative mean would be non-physical and may indicate instability in the algorithm
    # In that case, issue a runtime warning to flag the potential problem
    if float(mean_plastic_coef) <= 0:
        warnings.warn("Mean of Sp is negative", RuntimeWarning)

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

    # Plastic-only locations.  Instead of boolean-index selection (measured[condition]) -- which has a
    # data-dependent shape and would gather the view-sharded sinogram to one device -- zero out the other
    # locations and let compute_scaling_factor's inner products (sum(a*b)/sum(b*b)) all-reduce over the
    # shards.  This also drops padded views (plastic==0 there, so condition is False), so no separate
    # real-view mask is needed here.
    condition = (plastic_sino_est != 0) & metal_absent

    plastic_sino_scale = mjp.compute_scaling_factor(jnp.where(condition, measured_sino, 0.0),
                                                    jnp.where(condition, plastic_sino_corrected, 0.0))
    return plastic_sino_scale

def correct_sino_plastic_metal(ct_model, measured_sino, recon, num_metal=1, order=3, alpha=1, beta=0.002, gamma=0.1, num_constraint_update_iter=10):
    """
    This function corrects the measured sinogram of an object with plastic and multiple metal components by fitting a
    beam hardening model to the sinogram and removing the metal contributions.

    Args:
        ct_model: CT model object with a `forward_project` method and recon_placement / sino_placement.
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

    # Shard the measured sinogram to the model's devices (view-sharded, zero-padded on the view axis to
    # divide evenly).  Keep it as a 3-D view-sharded array -- do NOT flatten to one device -- so the whole
    # correction runs on view-sharded sinograms and no full-length sino vector lands on a single device
    # (critical for large, e.g. 20+ GB, sinograms).  The estimates below stay view-sharded too
    # (output_sharded=True), so every step is a per-shard elementwise op or a cross-device reduction.
    measured_sino = ct_model.prepare_sino_for_devices(measured_sino)

    # Real-view mask (True on real views, False on padded; None when unpadded) broadcasting over the
    # shard axis, plus the real pixel count -- used to exclude padded views from the statistical
    # reductions (the Sp mean floor and the constraint argmins).  A negligible padded_size-length
    # vector, not a full-sino mask.
    pl = ct_model.sino_placement
    view_mask = pl.real_mask(measured_sino.ndim)
    ax = ct_model.sinogram_shard_axis() % measured_sino.ndim
    num_real_pixels = pl.real_size * (measured_sino.size // measured_sino.shape[ax])

    # Get normalized sinogram p and [m_0, m_1, ...] (view-sharded; forward_project handles placement).
    plastic_sino_est, metal_sino_est = _est_plastic_metal_sinos_from_recon(recon, num_metal, ct_model)
    plastic_sino_scale = jnp.max(jnp.abs(plastic_sino_est))   # max over padded 0s is unaffected
    metal_sino_scale = [jnp.max(jnp.abs(arr)) for arr in metal_sino_est]
    # JAX division does not raise on 0/0 -- an empty (all-zero) plastic or metal estimate would silently
    # fill the normalized sinogram with NaNs and fail far downstream.  Check the scales explicitly (a
    # scalar sync each, once per correction) and fail fast with an actionable message.  ``not > 0`` also
    # catches a NaN scale (e.g. a NaN in the recon).
    if not float(plastic_sino_scale) > 0:
        raise ValueError(
            "The estimated plastic sinogram is empty (the plastic segmentation class contains no "
            "voxels).  Check the input reconstruction, num_metal, and the cylindrical-mask margins.")
    for metal_index, scale in enumerate(metal_sino_scale):
        if not float(scale) > 0:
            raise ValueError(
                f"The estimated sinogram for metal {metal_index} is empty (its segmentation class "
                f"contains no voxels).  num_metal={num_metal} may be too large for this object.")
    plastic_sino_est = plastic_sino_est / plastic_sino_scale
    metal_sino_est = [arr / norm for arr, norm in zip(metal_sino_est, metal_sino_scale)]

    # Estimate beam hardening model parameters theta
    theta = _estimate_BH_model_params(plastic_sino_est, metal_sino_est, measured_sino, H_exponent_list, num_cross_terms, alpha, beta, num_constraint_update_iter, view_mask=view_mask)
    # print(f'theta = {theta}')

    # Compute the corrected plastic sinogram
    plastic_sino_corrected = _correct_plastic_sinogram(measured_sino, plastic_sino_est, metal_sino_est, theta, H_exponent_list,
                                                       num_cross_terms, num_metal_terms, plastic_sino_scale, gamma,
                                                       view_mask=view_mask, num_real_pixels=num_real_pixels)

    # Compute and apply the scaling of the corrected plastic sino
    plastic_sino_corrected_scale = _estimate_plastic_scaling(plastic_sino_est, metal_sino_est, measured_sino, plastic_sino_corrected)
    scaled_corrected_plastic_sino = plastic_sino_corrected_scale * plastic_sino_corrected

    # Combine the scaled corrected plastic sino and the metal sinos (all view-sharded), then gather to a
    # host sinogram cropped to the real views (dropping padding) for the downstream recon.
    corrected_sino = scaled_corrected_plastic_sino + sum(arr * norm for arr, norm in zip(metal_sino_est, metal_sino_scale))
    return ct_model._gather_sinogram(corrected_sino)


def recon_plastic_metal(ct_model, sino, weights, num_BH_iterations=3, num_constraint_update_iter=10, stop_threshold_change_pct=0.5,
                        num_metal=1, order=3, alpha=1, beta=0.002, gamma=0.1, verbose=0, output_sharded=False):
    """
    Perform iterative metal artifact reduction using plastic-metal beam hardening correction.  If num_metal is 0,
    then this performs a standard MBIR recon.

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
        output_sharded (bool, optional): Choose the form of the returned reconstruction.  If False
            (default), return an ordinary host NumPy array.  If True, leave it distributed
            (slice-sharded) across the model's devices for a following on-device step.

    Returns:
         numpy or jax array: The final corrected reconstruction after iterative beam hardening
         correction -- a host NumPy array by default, or slice-sharded if ``output_sharded=True``.

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
    # Check for nonnegative num_metals
    if num_metal < 0:
        raise ValueError("num_metal must be >= 0")

    # Use split sino recon by default for cone beam.  recon() keeps its output SHARDED (the next
    # correction consumes it on-device with no gather/re-upload); split_sino_recon returns a host recon
    # BY DESIGN (it splits on the host so the full sinogram is never device-resident) -- the next
    # correction re-shards it at entry (one upload; device-side stitching is a possible follow-up).
    # Either form converges at correct_sino_plastic_metal / the exit adapter below.
    if 'ConeBeamModel' in ct_model.get_params('geometry_type'):
        recon_function = ct_model.split_sino_recon
    else:
        recon_function = functools.partial(ct_model.recon, output_sharded=True)

    # Deliver the user-requested output form: _shard_recon / _gather_recon are each a no-op when the
    # loop's final recon is already in that form.
    def to_output_form(r):
        return ct_model._shard_recon(r) if output_sharded else ct_model._gather_recon(r)

    # Do a regular recon if num_metal == 0
    if num_metal == 0:
        recon, _ = recon_function(sino, weights=weights, stop_threshold_change_pct=stop_threshold_change_pct)
        return to_output_form(recon)

    # Continue with beam hardening and segmentation
    if verbose >= 1:
        print("\n************ Perform initial FDK reconstruction  **************")
    recon = ct_model.direct_recon(sino, output_sharded=True)

    for i in range(num_BH_iterations):
        # Estimate Corrected Sinogram
        if verbose >= 1:
            print(f"\n************ Correct sino plastic metal {i + 1}  **************")
        corrected_sinogram = correct_sino_plastic_metal(ct_model, sino, recon, num_metal=num_metal, order=order, alpha=alpha, beta=beta, gamma=gamma, num_constraint_update_iter=num_constraint_update_iter)

        # Reconstruct Corrected Sinogram
        if verbose >= 1:
            print(f"\n************ Perform MBIR reconstruction {i + 1} **************")
        recon, _ = recon_function(corrected_sinogram, weights=weights, init_recon=recon, stop_threshold_change_pct=stop_threshold_change_pct)

        if verbose >= 2:
            print(f"\n************ BH Iteration {i + 1}: Display plastic and metal mask **************")
            plastic_mask, metal_masks, plastic_scale, metal_scales = mjp.segment_plastic_metal(recon, num_metal)
            labels = ['Plastic Mask'] + [f'Metal {j + 1} Mask' for j in range(len(metal_masks))]
            mj.slice_viewer(plastic_mask, *metal_masks, vmin=0, vmax=1.0,
                            slice_label=labels,
                            title=f'Iteration {i + 1}: Comparison of Plastic and Metal Masks')

    return to_output_form(recon)
