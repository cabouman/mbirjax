"""FROZEN pre-change batching reference for A/B measurement -- NOT production code.

mbirjax/projectors.py keeps a single batching implementation (no side-by-side versions in
the package); the 'old' side of driver-level A/B comparisons lives here instead.  This file
freezes sum_function_in_batches as it stood at greg/shard_profiling -- the form that scans
over a PRE-RESHAPED copy of the batched input, which on GPU materializes a full copy of that
input (the view-shard-sized temp reservation in back projection; see
projector_batching_characterization.md section 4) -- plus reference drivers that use it.
The reference drivers reuse the LIVE concatenate_function_in_batches (unchanged by the
patch) and the live geometry kernels, so a reference-vs-live comparison isolates exactly
the one changed mechanism: windowed vs reshaped batch reads on the sum axes.

Used by driver_ab_probe.py and transient_memory_probe.py.  If mbirjax/projectors.py's
driver signatures change, update the copies here to match.
"""
import jax
import jax.numpy as jnp

from mbirjax.projectors import concatenate_function_in_batches, ensure_tuple


def sum_function_in_batches_reference(function_to_sum, data_to_batch, batch_size, extra_args=()):
    """Frozen copy of the pre-windowed-read sum_function_in_batches (reshape + scan-over-xs)."""
    data_to_batch = ensure_tuple(data_to_batch)
    extra_args = ensure_tuple(extra_args)

    def add_one_batch(summed_and_fixed_data, batched_data):
        summed_data = summed_and_fixed_data[0]
        fixed_data = summed_and_fixed_data[1:]
        output_to_add = function_to_sum(*batched_data, *fixed_data)
        summed_data = jax.tree_util.tree_map(jnp.add, *(summed_data, output_to_add))
        return [summed_data, *fixed_data], None

    num_input_points = data_to_batch[0].shape[0]
    batch_size = num_input_points if batch_size is None else batch_size
    num_remaining = num_input_points % batch_size
    initial_batch_size = batch_size if num_remaining == 0 else num_remaining

    initial_batch = [data[:initial_batch_size] for data in data_to_batch]
    summed_output = function_to_sum(*initial_batch, *extra_args)

    if batch_size < num_input_points:
        num_batches = (num_input_points - initial_batch_size) // batch_size
        output_shape = (num_batches, batch_size,)
        data_batched = [jnp.reshape(data[initial_batch_size:], output_shape + data.shape[1:])
                        for data in data_to_batch]
        initial_carry = [summed_output, *extra_args]
        final_carry, _ = jax.lax.scan(add_one_batch, initial_carry, data_batched)
        summed_output = final_carry[0]

    return summed_output


# ----------------------------------------------------------------------------------
# Reference drivers: copies of the live drivers with ONLY the sum helper swapped for the
# frozen reference above (the concatenate helper is identical to live).
# ----------------------------------------------------------------------------------
def _sparse_forward_project_reference(view_params_array, voxel_values, pixel_indices,
                                      fwd_kernel, projector_params, pixel_batch_size,
                                      view_batch_size, owned_view_indices=()):
    cur_view_params_array = view_params_array
    if len(owned_view_indices) > 0:
        cur_view_params_array = view_params_array[owned_view_indices]

    def forward_project_pixel_batch(local_values, local_pix_indices):
        def forward_project_single_view(single_view_params):
            return fwd_kernel(local_values, local_pix_indices, single_view_params, projector_params)

        def forward_project_view_batch(view_params_batch):
            return jax.vmap(forward_project_single_view)(view_params_batch)

        return concatenate_function_in_batches(forward_project_view_batch, cur_view_params_array,
                                               view_batch_size)

    return sum_function_in_batches_reference(forward_project_pixel_batch,
                                             (voxel_values, pixel_indices), pixel_batch_size)


_jit_sparse_forward_project_reference = jax.jit(
    _sparse_forward_project_reference,
    static_argnames=['fwd_kernel', 'projector_params', 'pixel_batch_size', 'view_batch_size'])


def _sparse_back_project_reference(view_params_array, sinogram, pixel_indices,
                                   back_kernel, projector_params, pixel_batch_size,
                                   view_batch_size, coeff_power=1, owned_view_indices=()):
    cur_view_params_array = view_params_array
    if len(owned_view_indices) > 0:
        cur_view_params_array = view_params_array[owned_view_indices]

    def back_project_view_batch(local_view_batch, local_view_params_batch, local_pixel_indices):
        def back_project_pixel_batch(pixel_indices_batch):
            bp_vmap = jax.vmap(back_kernel, in_axes=(0, None, 0, None, None))
            per_view_voxel_values_batch = bp_vmap(local_view_batch, pixel_indices_batch,
                                                  local_view_params_batch, projector_params,
                                                  coeff_power)
            with jax.named_scope("projector/back/view_reduce"):
                return jnp.sum(per_view_voxel_values_batch, axis=0)

        return concatenate_function_in_batches(back_project_pixel_batch, local_pixel_indices,
                                               pixel_batch_size)

    return sum_function_in_batches_reference(back_project_view_batch,
                                             (sinogram, cur_view_params_array),
                                             view_batch_size, pixel_indices)


_jit_sparse_back_project_reference = jax.jit(
    _sparse_back_project_reference,
    static_argnames=['back_kernel', 'projector_params', 'pixel_batch_size', 'view_batch_size',
                     'coeff_power'])


def _sparse_back_project_band_reference(view_params_array, sinogram, pixel_indices, g0,
                                        num_band_slices, back_band_kernel, projector_params,
                                        pixel_batch_size, view_batch_size, coeff_power=1,
                                        owned_view_indices=()):
    cur_view_params_array = view_params_array
    if len(owned_view_indices) > 0:
        cur_view_params_array = view_params_array[owned_view_indices]

    def back_project_view_batch(local_view_batch, local_view_params_batch, local_pixel_indices):
        def back_project_pixel_batch(pixel_indices_batch):
            bp_vmap = jax.vmap(back_band_kernel, in_axes=(0, None, 0, None, None, None, None))
            per_view_band = bp_vmap(local_view_batch, pixel_indices_batch, local_view_params_batch,
                                    projector_params, g0, num_band_slices, coeff_power)
            with jax.named_scope("projector/back/view_reduce"):
                return jnp.sum(per_view_band, axis=0)

        return concatenate_function_in_batches(back_project_pixel_batch, local_pixel_indices,
                                               pixel_batch_size)

    return sum_function_in_batches_reference(back_project_view_batch,
                                             (sinogram, cur_view_params_array),
                                             view_batch_size, pixel_indices)


_jit_sparse_back_project_band_reference = jax.jit(
    _sparse_back_project_band_reference,
    static_argnames=['num_band_slices', 'back_band_kernel', 'projector_params',
                     'pixel_batch_size', 'view_batch_size', 'coeff_power'])
