from functools import partial

import jax
from jax import numpy as jnp
import numpy as np

import mbirjax._sharding as mjs


@partial(jax.jit, static_argnames=['sigma_prox'])
def prox_gradient_at_indices(recon, prox_input, pixel_indices, sigma_prox):
    """
    Calculate the gradient at each pixel index locations in a reconstructed image using the proximal map prior.

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
def qggmrf_gradient_and_hessian_at_indices(flat_recon, recon_shape, pixel_indices, qggmrf_params,
                                           left_halo=None, right_halo=None, interface_mask=None):
    """
    Calculate the gradient and hessian at each index location in a reconstructed image using the surrogate function for
    the qGGMRF prior.
    Calculations taken from Figure 8.5 (page 119) of FCI for the qGGMRF prior model.

    Args:
        flat_recon (jax.array): 2D reconstructed image array with shape (num_recon_rows x num_recon_cols, num_recon_slices).
            When operating on a single slice-shard, num_recon_slices is the local (per-shard) slice count.
        recon_shape (tuple of ints): shape of the original recon:  (num_recon_rows, num_recon_cols, num_recon_slices).
            num_recon_rows and num_recon_cols are always the full (unsharded) values; the slice count entry should match
            ``flat_recon``'s (local) slice count -- only the in-slice term uses recon_shape, and it ignores the slice count.
        pixel_indices (int array): Array of shape (N_indices, num_recon_slices) representing the indices of voxels in a flattened array to be updated.
        qggmrf_params (tuple): The parameters b, sigma_x, p, q, T
        left_halo (jax.array or None): 1D array of shape (num_recon_rows x num_recon_cols,) holding the slice
            immediately to the left (lower global slice index) of this shard, used to compute the inter-slice prior
            term at the shard's left boundary.  Pass None at the true left edge of the full recon, which mirrors the
            local boundary slice (reflected BC) and reproduces the single-device result exactly.
        right_halo (jax.array or None): 1D array of shape (num_recon_rows x num_recon_cols,) holding the slice
            immediately to the right (higher global slice index) of this shard.  Pass None at the true right edge of
            the full recon (reflected BC).
        interface_mask (jax.array or None): optional 1D float array of length num_recon_slices+1 (the LOCAL slice
            count plus one) multiplying the inter-slice differences; shared by every cylinder of this shard.  Entry
            j corresponds to the interface between local slices j-1 and j; a 0 entry decouples that pair (reflected
            BC at that interface).  Used when the slice axis is padded for sharding: mask every interface whose
            higher-index GLOBAL slice is padded (the predicate g0 + j < num_real_slices over the shard's global
            range).  None applies no masking -- the unpadded/single-device path is unchanged.

    Returns:
        tuple of two arrays (first_derivative, second_derivative), each of shape
        (N_indices, num_recon_slices) representing the gradient and Hessian values at specified indices.
    """
    # Initialize the neighborhood weights for averaging surrounding pixel values.
    # Order is [row+1, row-1, col+1, col-1, slice+1, slice-1] - see definition in _utils.py
    b, sigma_x, p, q, T = qggmrf_params

    # Resolve the per-cylinder boundary values for the inter-slice (cylinder) prior term.
    # With no halo (a true edge or single-device run), mirror the local boundary slice: passing
    # voxel_cylinder[0] as left_val and voxel_cylinder[-1] as right_val makes both boundary deltas
    # zero, exactly reproducing the reflected-BC single-device behavior.  With a halo (a shard
    # interior boundary), use the actual neighboring-shard slice values so the inter-slice gradient
    # is correct at the boundary -- and no temporary extended-cylinder array is allocated.
    lh_vals = (left_halo[pixel_indices]  if left_halo  is not None
               else flat_recon[pixel_indices, 0])   # (N_indices,)
    rh_vals = (right_halo[pixel_indices] if right_halo is not None
               else flat_recon[pixel_indices, -1])  # (N_indices,)

    # First work on cylinders - determine the contributions from neighbors in the voxel cylinder.
    # lh_vals / rh_vals are vmapped as per-cylinder scalars (in_axes 0); the interface mask is the
    # same for every cylinder of the shard (the slice structure is shared), so it broadcasts
    # (in_axes None) rather than being replicated per cylinder.
    qggmrf_params = (b, sigma_x, p, q, T)
    cylinder_map = jax.vmap(qggmrf_grad_and_hessian_per_cylinder, in_axes=(0, None, 0, 0, None))
    gradient, hessian = cylinder_map(flat_recon[pixel_indices], qggmrf_params, lh_vals, rh_vals,
                                     interface_mask)

    # Then work on slices - add in the contributions from neighbors in the same slice (fully local, no halos)
    slice_map = jax.vmap(qggmrf_grad_and_hessian_per_slice, in_axes=(1, None, None, None, 1, 1), out_axes=1)
    gradient, hessian = slice_map(flat_recon, recon_shape, pixel_indices, qggmrf_params, gradient, hessian)

    return gradient, hessian


@partial(jax.jit, static_argnames='qggmrf_params')
def qggmrf_grad_and_hessian_per_cylinder(voxel_cylinder, qggmrf_params, left_val, right_val,
                                         interface_mask=None):
    """
    Compute the qggmrf gradient and diagonal Hessian at each voxel of a voxel_cylinder.

    Args:
        voxel_cylinder (jax array): 1D array of voxel values for one (row, col) location, spanning all slices local
            to this shard.
        qggmrf_params (tuple): The parameters b, sigma_x, p, q, T
        left_val (scalar): Value of the voxel in the slice immediately before this cylinder (global slice index -1
            relative to the cylinder).  Pass voxel_cylinder[0] for a reflected (zero-delta) boundary at the true
            left edge of the recon.
        right_val (scalar): Value of the voxel in the slice immediately after this cylinder (global slice index n
            relative to the cylinder).  Pass voxel_cylinder[-1] for a reflected (zero-delta) boundary at the true
            right edge of the recon.
        interface_mask (jax array or None): optional 1D float array of length len(voxel_cylinder)+1 multiplying
            the slice-to-slice differences.  Entry j corresponds to the interface between local slices j-1 and j
            (entries 0 and n are the left/right boundary interfaces), so a 0 entry decouples the two slices it
            joins exactly as the reflected boundary condition does at a true edge (the boundary difference is
            forced to zero while the Hessian keeps its b_tilde(0) term).  Used when the slice axis is padded for
            multi-device sharding: zeroing every interface whose higher-index slice is padded reproduces the
            reflected boundary at the last REAL slice -- even mid-shard -- and keeps the padded slices' gradient
            exactly zero.  None (the default) applies no masking.

    Returns:
        tuple of gradient and Hessian, each of which is a 1D jax array of the same length as voxel_cylinder
    """
    b, sigma_x, p, q, T = qggmrf_params
    b_at_slice_plus_one, b_at_slice_minus_one = b[4:6]

    # Voxel cylinder is 1D.
    # Build delta[j] = voxel_cylinder[j] - voxel_cylinder[j-1] for interior positions, with explicit boundary
    # deltas at each end derived from the neighbor values.  Result is
    #   v[0]-left_val, v[1]-v[0], v[2]-v[1], ..., v[n-1]-v[n-2], right_val-v[n-1].
    # Passing left_val = voxel_cylinder[0] makes delta[0] = 0 and right_val = voxel_cylinder[-1] makes delta[n] = 0,
    # which is the reflected boundary condition used on a single device; a true neighbor slice (a shard halo) gives
    # the correct cross-boundary delta instead.
    left_delta = voxel_cylinder[:1] - left_val    # shape (1,)
    right_delta = right_val - voxel_cylinder[-1:]  # shape (1,)
    delta = jnp.concatenate((left_delta, jnp.diff(voxel_cylinder), right_delta))

    # Force masked interfaces to a zero difference.  Reflected BC at a true edge IS a zero boundary
    # delta, so this is the same boundary condition applied at an arbitrary interface; the Hessian
    # still receives the b_tilde(0) term from a masked interface, exactly as it does at a true edge.
    if interface_mask is not None:
        delta = delta * interface_mask

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
    Compute rho'(delta) / delta using a slightly modified form on page 117 of FCI for the qGGMRF prior model.

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


# ─────────────────────────────────────────────────────────────────────────────
# Sharded-prior orchestration (slice-sharded recon).
#
# The qGGMRF prior couples each slice to its slice-axis neighbors, so on a
# slice-sharded recon each shard needs one boundary slice from each adjacent
# shard (a HALO), and -- when the slice axis is zero-padded for sharding -- an
# interface MASK that reproduces the reflected boundary condition at the last
# real slice (see qggmrf_grad_and_hessian_per_cylinder).  These functions take
# everything explicitly (the recon, the placement, the staged halos/masks) so
# they are unit-testable without a TomographyModel; the model owns thin wrappers
# that supply its placement state.
# ─────────────────────────────────────────────────────────────────────────────

def extract_halos(flat_recon, slice_axis=-1):
    """Return per-shard boundary slices for the qGGMRF inter-slice prior term.

    ``flat_recon`` is sharded along ``slice_axis`` (the last axis), so each
    device holds ``(num_pixels, local_slices)``.  The prior couples a slice to
    its neighbors, so each shard needs one boundary slice from each adjacent
    shard:

      left_halo[i]  = last slice of shard i-1  (None for the first shard)
      right_halo[i] = first slice of shard i+1 (None for the last shard)

    Reads ``2*(n_shards-1)`` slices to host -- negligible vs. compute.  With a
    single shard both lists are ``[None]`` (the true-edge reflected BC).

    Args:
        flat_recon (jax array): a slice-sharded flat recon ``(num_pixels, slices)``.
        slice_axis (int): the sharded slice axis (may be negative).

    Returns:
        (left_halos, right_halos): two lists, one entry per shard in slice-start
        order; each entry is a numpy array of shape ``(num_pixels,)`` or None at
        a true edge.
    """
    slice_axis = slice_axis % flat_recon.ndim
    # Order shards by their start index along the sharded (slice) axis so the
    # device sequence is deterministic.
    shards = sorted(flat_recon.addressable_shards,
                    key=lambda s: s.index[slice_axis].start)
    left_halos = [None] + [np.asarray(s.data[..., -1]) for s in shards[:-1]]
    right_halos = [np.asarray(s.data[..., 0]) for s in shards[1:]] + [None]
    return left_halos, right_halos


def stage_halos(flat_recon, slice_axis=-1):
    """Extract the qGGMRF boundary halos once and pre-place each on its shard's device.

    :func:`extract_halos` reads the boundary slices to host; this wrapper then
    ``device_put``s each onto the device of the shard that will use it, so a
    caller can stage the halos ONCE (e.g. per VCD partition pass) and hand the
    result to :func:`qggmrf_gradient_and_hessian_sharded` for every subset in
    that pass -- turning a per-subset host round-trip into a per-pass one (the
    per-subset host round-trips are what cap VCD's multi-GPU scaling).

    The per-shard ordering is by slice-start, matching the shard sort in
    :func:`qggmrf_gradient_and_hessian_sharded`; the recon's sharding is constant
    across a pass, so a halo staged on ``shards[i].device`` here lines up with
    shard ``i`` there even though the recon array itself is replaced each subset.

    Args:
        flat_recon (jax array): the (slice-sharded) recon to read boundaries from.
        slice_axis (int): the sharded slice axis (may be negative).

    Returns:
        (staged_left, staged_right): per-shard lists (slice-start order); each
        entry is an on-device halo slice ``(num_pixels,)`` or ``None`` at a true
        recon edge.
    """
    left_halos, right_halos = extract_halos(flat_recon, slice_axis)
    slice_axis = slice_axis % flat_recon.ndim
    shards = sorted(flat_recon.addressable_shards,
                    key=lambda s: s.index[slice_axis].start)
    staged_left = [None if h is None else jax.device_put(h, s.device)
                   for h, s in zip(left_halos, shards)]
    staged_right = [None if h is None else jax.device_put(h, s.device)
                    for h, s in zip(right_halos, shards)]
    return staged_left, staged_right


def qggmrf_gradient_and_hessian_sharded(flat_recon, pixel_indices, qggmrf_params,
                                        num_rows, num_cols, recon_placement,
                                        staged_halos=None, interface_masks=None):
    """Compute the qGGMRF prior gradient and Hessian on a slice-sharded recon.

    This is the recon-domain analogue of the sharded projectors: the recon is
    sharded by slice, so each slice-owner computes the prior on its own
    slice-shard **locally** and the results are assembled (with no data
    movement) into one slice-sharded array.  An interior shard boundary uses a
    halo slice from the adjacent shard; a zero-padded slice axis uses the
    per-shard interface masks (reflected BC at the last real slice; the padded
    slices' gradient is exactly zero and their Hessian stays positive).  At the
    true recon edges the halo is None and the boundary slice is mirrored
    (reflected BC) -- matching the single-device result exactly.

    The in-slice (row/col) prior term is fully local and uses only the shard's
    own slices, so passing the *local* slice count in the kernel's recon_shape is
    correct (and identical across equal shards, so the jitted prior compiles once).

    No gather is performed: the inputs are slice-sharded and the outputs are
    returned slice-sharded (matching ``recon_placement``).

    Args:
        flat_recon (jax array): slice-sharded recon ``(num_pixels, num_slices_device)``
            (a 1-device mesh is the trivial 1-shard case; the slice axis may be
            padded -- the device form).
        pixel_indices (jax array): 1D indices into the flattened (rows, cols)
            identifying the subset of cylinders to evaluate.
        qggmrf_params (tuple): the prior parameters ``(b, sigma_x, p, q, T)``.
        num_rows, num_cols (int): the recon's in-plane shape (problem-owned; the
            pixel axis is never sharded or padded).
        recon_placement: the slice-axis Placement (supplies the shard axis and
            the NamedSharding used to reassemble the outputs).
        staged_halos (tuple or None): ``(staged_left, staged_right)`` from
            :func:`stage_halos`, pre-placed on the shard devices.  Pass these to
            avoid re-reading the halos every subset (the VCD loop stages once per
            pass).  When ``None``, the halos are extracted+staged here from
            ``flat_recon`` (the self-contained path, e.g. for tests).
        interface_masks (dict or None): device -> ``(local_slices+1,)`` float32
            interface mask, when the slice axis is padded (see the model's
            ``_qggmrf_interface_masks``); None when nothing is padded.

    Returns:
        (gradient, hessian): each a slice-sharded array of shape
        ``(len(pixel_indices), num_slices_device)``.
    """
    slice_axis = recon_placement.axis % flat_recon.ndim
    # The DEVICE-FORM slice count, from the array itself: padded when the slice
    # axis does not divide the device count.
    num_slices_device = flat_recon.shape[slice_axis]
    num_indices = len(pixel_indices)

    # Boundary slices each shard needs from its neighbors (None at the true edges),
    # pre-placed on the shard devices.  Stage here if the caller did not.
    if staged_halos is None:
        staged_left, staged_right = stage_halos(flat_recon, slice_axis)
    else:
        staged_left, staged_right = staged_halos

    # Order the shards by their start index along the sharded (slice) axis so the
    # device sequence matches the staged-halo order and recon_placement's device
    # order (used to reassemble below).
    shards = sorted(flat_recon.addressable_shards,
                    key=lambda s: s.index[slice_axis].start)

    grad_owned, hess_owned = [], []
    for i, shard in enumerate(shards):
        device = shard.device
        local = shard.data                       # (num_pixels, local_slices) on this device
        local_slices = local.shape[slice_axis]
        recon_shape_local = (num_rows, num_cols, local_slices)
        # Indices must be resident on this shard's device; the halos/masks already are.
        local_indices = jax.device_put(pixel_indices, device)
        g, h = qggmrf_gradient_and_hessian_at_indices(
            local, recon_shape_local, local_indices, qggmrf_params,
            left_halo=staged_left[i], right_halo=staged_right[i],
            interface_mask=None if interface_masks is None else interface_masks[device])
        grad_owned.append(g)
        hess_owned.append(h)

    # Wrap the per-shard pieces into one slice-sharded array (no data movement).
    structure = recon_placement.shard_structure(2)
    gradient = mjs.assemble_sharded(grad_owned, (num_indices, num_slices_device), structure)
    hessian = mjs.assemble_sharded(hess_owned, (num_indices, num_slices_device), structure)
    return gradient, hessian


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

    grad = 0 * x_prime
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


def get_b_from_nbr_wts(qggmrf_nbr_wts):

    # Convert the 3-element list of neighbor weights to a 6-element array of weights in each direction
    b = np.array([qggmrf_nbr_wts, qggmrf_nbr_wts])
    b = b.T.flatten()
    b = b / np.sum(b)
    b = tuple(b)
    return b
