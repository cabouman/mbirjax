from functools import partial

import jax
from jax import numpy as jnp


@partial(jax.jit, static_argnames=['sigma_prox'])
def prox_gradient_at_indices(recon, prox_input, pixel_indices, sigma_prox):
    """
    Calculate the gradient and hessian at each index location in a reconstructed image using the qGGMRF prior.

    Args:
        recon (jax.array): 2D reconstructed image array with shape (num_recon_rows x num_recon_cols, num_recon_slices).
        prox_input (jax.array): 2D reconstructed image array with shape (num_recon_rows x num_recon_cols, num_recon_slices).
        pixel_indices (int array): Array of shape (N_indices, num_recon_slices) representing the indices of voxels in a flattened array to be updated.
        sigma_prox (float): Standard deviation parameter of the proximal map.

    Returns:
        first_derivative of shape (N_indices, num_recon_slices) representing the gradient of the prox term at specified indices.
    """

    # Compute the prior model gradient at all voxels
    cur_diff = recon[pixel_indices] - prox_input[pixel_indices]
    pm_gradient = (1.0 / (sigma_prox ** 2.0)) * cur_diff

    # Shape of pm_gradient is (num indices)x(num slices)
    return pm_gradient


@partial(jax.jit, static_argnames='qggmrf_params')
def qggmrf_loss(full_recon, qggmrf_params):
    """
    Computes the loss for the qGGMRF prior for a given recon.  This is meant only for relatively small recons
    for debugging and demo purposes.

    Args:

    Returns:
        float
    """
    # Get the parameters
    b, sigma_x, p, q, T = qggmrf_params

    # Normalize b to sum to 1, then get the per-axis b.
    b_per_axis = [(b[j] + b[j + 1]) / (2 * sum(b)) for j in [0, 2, 4]]

    def rho_ref(delta):
        # Compute rho from Table 8.1 in FCI
        a_min = T * sigma_x * jnp.finfo(jnp.float32).eps
        abs_delta = jnp.clip(abs(delta), a_min, None)
        delta_scale = abs_delta / (T * sigma_x)  # delta_scale has a min of eps
        ds_q_minus_p = (delta_scale ** (q - p))
        numerator = (abs_delta ** p) / (p * sigma_x ** p)
        numerator *= ds_q_minus_p
        rho_value = numerator / (1 + ds_q_minus_p)
        return rho_value

    # Add rho over all the neighbor differences
    loss = 0
    for axis in [0, 1, 2]:
        cur_delta = jnp.diff(full_recon, axis=axis)
        loss += jnp.sum(b_per_axis[axis] * rho_ref(cur_delta))

    return loss


@partial(jax.jit, static_argnames=['recon_shape', 'qggmrf_params'])
def qggmrf_gradient_and_hessian_at_indices(flat_recon, recon_shape, pixel_indices, qggmrf_params):
    """
    Calculate the gradient and hessian at each index location in a reconstructed image using the surrogate function for
    the qGGMRF prior.
    Calculations taken from Figure 8.5 (page 119) of FCI for the qGGMRF prior model.

    Args:
        flat_recon (jax.array): 2D reconstructed image array with shape (num_recon_rows x num_recon_cols, num_recon_slices).
        recon_shape (tuple of ints): shape of the original recon:  (num_recon_rows, num_recon_cols, num_recon_slices).
        pixel_indices (int array): Array of shape (N_indices, num_recon_slices) representing the indices of voxels in a flattened array to be updated.
        qggmrf_params (tuple): The parameters b, sigma_x, p, q, T

    Returns:
        tuple of two arrays and a float (first_derivative, second_derivative, loss).  The first two entries have shape
        (N_indices, num_recon_slices) representing the gradient and Hessian values at specified indices. loss is 1x1.
    """
    # Initialize the neighborhood weights for averaging surrounding pixel values.
    # Order is [row+1, row-1, col+1, col-1, slice+1, slice-1] - see definition in _utils.py
    b, sigma_x, p, q, T = qggmrf_params

    # First work on cylinders - determine the contributions from neighbors in the voxel cylinder
    qggmrf_params = (b, sigma_x, p, q, T)
    cylinder_map = jax.vmap(qggmrf_grad_and_hessian_per_cylinder, in_axes=(0, None))
    gradient, hessian = cylinder_map(flat_recon[pixel_indices], qggmrf_params)

    # Then work on slices - add in the contributions from neighbors in the same slice
    slice_map = jax.vmap(qggmrf_grad_and_hessian_per_slice, in_axes=(1, None, None, None, 1, 1), out_axes=1)
    gradient, hessian = slice_map(flat_recon, recon_shape, pixel_indices, qggmrf_params, gradient, hessian)

    return gradient, hessian


@partial(jax.jit, static_argnames='qggmrf_params')
def qggmrf_grad_and_hessian_per_cylinder(voxel_cylinder, qggmrf_params):
    """
    Compute the qggmrf gradient and diagonal Hessian at each voxel of a voxel_cylinder.

    Args:
        voxel_cylinder (jax array): 1D array of voxel values
        qggmrf_params (tuple): The parameters b, sigma_x, p, q, T

    Returns:
        tuple of gradient and Hessian, each of which is a 1D jax array of the same length as voxel_cylinder
    """
    b, sigma_x, p, q, T = qggmrf_params
    b_at_slice_plus_one, b_at_slice_minus_one = b[4:6]

    # Voxel cylinder is 1D.
    # Compute the differences delta[j] = voxel_cylinder[j+1] - voxel_cylinder[j], then add 0s at both ends to
    # represent reflected boundaries at 0 and the end.
    # Get v[0]-v[-1], v[1]-v[0], v[2]-v[1], ..., v[n]-v[n-1]  (where v[-1] = v[0], v[n]=v[n-1] in this case).
    zero = jnp.zeros(1)
    delta = jnp.concatenate((zero, jnp.diff(voxel_cylinder), zero))

    # Compute the primary quantity used for the gradient and Hessian
    # Use b_for_delta = 1 here and scale by b_slice below.
    b_tilde_times_2 = get_2_b_tilde(delta, 1, qggmrf_params)
    b_tilde_times_2_times_delta = b_tilde_times_2 * delta

    # The gradient_cylinder gets a term from each neighbor, slice+1 and slice-1
    # First do the gradient and Hessian for slice+1.  Here delta[1:] has v[1]-v[0], v[2]-v[1], so we need to use
    # -delta since delta is supposed to be xs - xr, where xs is the current point of interest.
    gradient_cylinder = -b_at_slice_plus_one * b_tilde_times_2_times_delta[1:]
    hessian_cylinder = b_at_slice_plus_one * b_tilde_times_2[1:]

    # For slice-1, we use delta[0:], which has v[0]-v[-1], v[1]-v[0] and hence has the correct sign.
    gradient_cylinder += b_at_slice_minus_one * b_tilde_times_2_times_delta[:-1]
    hessian_cylinder += b_at_slice_minus_one * b_tilde_times_2[:-1]

    return gradient_cylinder, hessian_cylinder


@partial(jax.jit, static_argnames=['recon_shape', 'qggmrf_params'])
def qggmrf_grad_and_hessian_per_slice(flat_recon_slice, recon_shape, pixel_indices, qggmrf_params,
                                      initial_gradient_slice, initial_hessian_slice):
    """
    Compute the qggmrf gradient and diagonal Hessian at each voxel of a slice of voxels.
    The results are added to initial_gradient_slice, initial_hessian_slice.

    Args:
        flat_recon_slice (jax array): 1D array of voxels in a single slice of a recon.
        The locations of flat_recon_slice[pixel_indices] within the recon are given by
        row_index, col_index = jnp.unravel_index(pixel_indices, shape=(num_rows, num_cols))
        recon_shape (tuple of ints): shape of the original recon:  (num_recon_rows, num_recon_cols, num_recon_slices).
        pixel_indices (int array): Array of shape (N_indices, num_recon_slices) representing the indices of voxels in a
        flattened recon.
        qggmrf_params: b, sigma_x, p, q, T
        initial_gradient_slice (jax array): Array of the same shape as flat_recon_slice
        initial_hessian_slice (jax array): Array of the same shape as flat_recon_slice

    Returns:
        tuple of gradient, hessian
    """
    # Get the parameters
    b, sigma_x, p, q, T = qggmrf_params

    # Extract the shape of the reconstruction array.
    num_rows, num_cols, num_slices = recon_shape[:3]

    # Convert flat indices to 2D indices for row and column access.
    row_index, col_index = jnp.unravel_index(pixel_indices, shape=(num_rows, num_cols))

    # Define relative positions for accessing neighborhood voxels.
    row_plus_one, row_minus_one = [1, 0], [-1, 0]
    col_plus_one, col_minus_one = [0, 1], [0, -1]
    b_row_plus_one, b_row_minus_one = b[0:2]
    b_col_plus_one, b_col_minus_one = b[2:4]

    offsets = [row_plus_one, row_minus_one, col_plus_one, col_minus_one]
    b_values = [b_row_plus_one, b_row_minus_one, b_col_plus_one, b_col_minus_one]

    # Access the central voxels' values at the given pixel_indices. Length of xs0 is (num indices)
    xs0 = flat_recon_slice[pixel_indices]

    # Loop over the offsets and accumulate the gradients and Hessian diagonal entries.
    loss = jnp.zeros(1)
    for offset, b_value in zip(offsets, b_values):
        offset_indices = jnp.ravel_multi_index([row_index + offset[0], col_index + offset[1]],
                                               dims=(num_rows, num_cols), mode='clip')
        delta = xs0 - flat_recon_slice[offset_indices]

        # Compute the primary quantity used for the gradient and Hessian
        b_tilde_times_2 = get_2_b_tilde(delta, b_value, qggmrf_params)

        # Update the gradient and Hessian for this delta
        b_tilde_times_2_times_delta = b_tilde_times_2 * delta
        initial_gradient_slice += b_tilde_times_2_times_delta
        initial_hessian_slice += b_tilde_times_2

    return initial_gradient_slice, initial_hessian_slice


@partial(jax.jit, static_argnames='qggmrf_params')
def get_2_b_tilde(delta, b_for_delta, qggmrf_params):
    """
    Compute rho'(delta) / delta using a slightly modifided form of page 153 of FCI for the qGGMRF prior model.

    Args:
        delta (float or jax array): (batch_size, P) array of pixel differences between center and each of P neighboring
        pixels.
        b_for_delta (float): The value of b associated with this delta.
        qggmrf_params (tuple): Parameters in the form (b, sigma_x, p, q, T)

    Returns:
        float or jax array: rho'(delta) / (2 delta)
    """
    b, sigma_x, p, q, T = qggmrf_params

    # Scale by T * sigma_x and get powers
    # Note that |delta|^r = T^r sigma_x^r |delta / (T sigma_x)|^r for any r, so we use a scaled version of delta_prime
    eps_float32 = jnp.finfo(jnp.float32).eps  # Smallest single precision float
    scaled_delta = abs(delta) / (T * sigma_x) + eps_float32  # Avoid delta=0 in delta**(q-p) since q < p
    delta_q_minus_2 = scaled_delta ** (q - 2.0)
    delta_q_minus_p = scaled_delta ** (q - p)

    numerator = delta_q_minus_2 * ((q / p) + delta_q_minus_p)
    denominator = (1 + delta_q_minus_p) ** 2
    scale = (T ** (p - 2)) / (sigma_x * sigma_x)

    rho_prime_over_delta = scale * numerator / denominator
    b_tilde_times_2 = b_for_delta * rho_prime_over_delta
    return b_tilde_times_2


def b_tilde_by_definition(delta, sigma_x, p, q, T):
    # This is a reference implementation to compute b_tilde = rho'(delta) / (2 delta) from Table 8.1 in FCI
    a_min = T * sigma_x * jnp.finfo(jnp.float32).eps
    abs_delta = jnp.clip(abs(delta), a_min, None)
    delta_scale = abs_delta / (T * sigma_x)  # delta_scale has a min of eps

    ds_q_minus_p = (delta_scale ** (q - p))
    ds_p_minus_2 = (abs_delta ** (p - 2))

    numerator = ds_p_minus_2 / (2 * sigma_x ** p)
    numerator *= ds_q_minus_p * ((q / p) + ds_q_minus_p)
    b_tilde_value = numerator / (1 + ds_q_minus_p) ** 2
    return b_tilde_value


@partial(jax.jit, static_argnames='qggmrf_params')
def compute_surrogate_and_grad(x, x_prime, qggmrf_params):
    # This is a reference implementation to compute the value and full gradient of the surrogate
    # in the form Q(x; x') and \nabla_x Q(x; x')
    # x and x_prime must be arrays in the shape of a full 3D reconstruction.
    # The surrogate is (1/2) \sum_(s,r) b_tilde_(s,r) (x_s - x_r)^2, where the sum is over ordered pairs, and b_tilde
    # is evaluated at x_prime.  The partial with respect to x_s is \sum_r 2 b_tilde_(s,r) (x_s - x_r), where the
    # extra factor of 2 comes from the fact that x_s appears in both first position and second position.

    # Get the parameters
    b, sigma_x, p, q, T = qggmrf_params

    # Normalize b to sum to 1, then get the per-axis b.
    b_per_axis = [(b[j] + b[j + 1]) / (2 * sum(b)) for j in [0, 2, 4]]

    grad = jnp.zeros_like(x_prime)
    surrogate_value = 0
    # Get b_tilde at all the deltas
    for axis in [0, 1, 2]:
        cur_delta_prime = jnp.diff(x_prime, axis=axis)
        cur_delta = jnp.diff(x, axis=axis)
        new_shape = list(x.shape)
        new_shape[axis] = 1
        zero = jnp.zeros(new_shape)

        # Evaluate b_tilde over all differences in this axis
        cur_delta_prime = jnp.concatenate((zero, cur_delta_prime, zero),
                                          axis=axis)  # Include 0 differences for reflected boundaries
        cur_delta = jnp.concatenate((zero, cur_delta, zero),
                                    axis=axis)
        cur_b_tilde = b_tilde_by_definition(cur_delta_prime, sigma_x, p, q, T)

        # Sum over forward and backward differences
        num_points = cur_b_tilde.shape[axis]
        b_tilde_plus = jax.lax.slice_in_dim(cur_b_tilde, 1, num_points, axis=axis)
        b_tilde_minus = jax.lax.slice_in_dim(cur_b_tilde, 0, num_points - 1, axis=axis)

        cur_delta_plus = jax.lax.slice_in_dim(cur_delta, 1, num_points, axis=axis)
        cur_delta_minus = jax.lax.slice_in_dim(cur_delta, 0, num_points - 1, axis=axis)

        surrogate_value += (1 / 2) * b_per_axis[axis] * jnp.sum(b_tilde_plus * cur_delta_plus * cur_delta_plus +
                                                                b_tilde_minus * cur_delta_minus * cur_delta_minus)
        grad += 2 * b_per_axis[axis] * (- b_tilde_plus * cur_delta_plus + b_tilde_minus * cur_delta_minus)

    return surrogate_value, grad


def compute_qggmrf_grad_and_hessian(full_recon, qggmrf_params):
    # This is a reference implementation to compute the full gradient and Hessian diagonal of the surrogate
    # in the form \nabla Q(x; x'=\hat{x}), evaluated at x=\hat{x}=full_recon
    # The output here should equal the output of
    # qggmrf_gradient_and_hessian_at_indices(flat_recon, recon_shape, pixel_indices, qggmrf_params) when
    #    recon_shape = full_recon.shape
    #    flat_recon = full_recon.reshape((recon_shape[0] * recon_shape[1], recon_shape[2})
    #    pixel_indices = np.arange(flat_recon.shape[0])
    # This function is not optimized for time or memory efficiency.

    # Get the parameters
    b, sigma_x, p, q, T = qggmrf_params

    # Normalize b to sum to 1, then get the per-axis b.
    b_per_axis = [(b[j] + b[j + 1]) / (2 * sum(b)) for j in [0, 2, 4]]

    hess = jnp.zeros_like(full_recon)
    grad = jnp.zeros_like(full_recon)

    # Sum over all the neighbor differences
    for axis in [0, 1, 2]:
        cur_delta = jnp.diff(full_recon, axis=axis)
        new_shape = list(full_recon.shape)
        new_shape[axis] = 1
        zero = jnp.zeros(new_shape)

        # Evaluate b_tilde over all differences in this axis
        cur_delta = jnp.concatenate((zero, cur_delta, zero),
                                    axis=axis)  # Include 0 differences for reflected boundaries
        cur_b_tilde = b_tilde_by_definition(cur_delta, sigma_x, p, q, T)

        # Sum b_tilde evaluated over forward and backward differences
        num_points = cur_b_tilde.shape[axis]
        b_tilde_plus = jax.lax.slice_in_dim(cur_b_tilde, 1, num_points, axis=axis)
        b_tilde_minus = jax.lax.slice_in_dim(cur_b_tilde, 0, num_points - 1, axis=axis)
        hess += 2 * b_per_axis[axis] * (b_tilde_plus + b_tilde_minus)

        cur_delta_plus = jax.lax.slice_in_dim(cur_delta, 1, num_points, axis=axis)
        cur_delta_minus = jax.lax.slice_in_dim(cur_delta, 0, num_points - 1, axis=axis)
        grad += 2 * b_per_axis[axis] * (- b_tilde_plus * cur_delta_plus + b_tilde_minus * cur_delta_minus)

    return grad, hess
