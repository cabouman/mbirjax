from collections import namedtuple
import jax
import jax.numpy as jnp
import mbirjax
from functools import partial


# The ProjectorParams namedtuple TYPE is defined ONCE here (module level), not rebuilt per
# instance.  jax registers namedtuples as pytrees and keys the jit static-argument cache on the
# pytree treedef -- which includes the namedtuple CLASS -- so rebuilding the class per call (as
# namedtuple() inside a function does) would give each instance a distinct pytree type and defeat
# the shared module-level projector jit cache.  geometry_params gets the same treatment at its
# source (ParameterHandler.make_geometry_params).
ProjectorParams = namedtuple('ProjectorParams', ['sinogram_shape', 'recon_shape', 'geometry_params'])


class Projectors:

    def __init__(self, tomography_model):

        self.tomography_model = tomography_model
        self.sparse_forward_project, self.sparse_back_project = None, None
        self.sparse_back_project_band = None   # set in create_projectors only for banded geometries
        self.create_projectors(tomography_model)

    def create_projectors(self, tomography_model):
        """
        Compute the forward and back projectors for this geometry and current view parameters

        Args:
            tomography_model (mbirjax.TomographyModel): An instance describing the current geometry and implementing the following 2 functions:

                * forward_project_pixel_batch_to_one_view (callable): jit-compilable function implementing :meth:`TomographyModel.forward_project_pixel_batch_to_one_view`
                * back_project_one_view_to_pixel_batch (callable): jit-compilable function implementing :meth:`TomographyModel.back_project_one_view_to_pixel_batch`

        Returns:
            Nothing, but the class variables `sparse_forward_project` and `sparse_back_project` are set to callable
            functions.  These are used to implement the following methods:

            * `sparse_forward_project`: :meth:`TomographyModel.sparse_forward_project`
            * `sparse_back_project`: :meth:`TomographyModel.sparse_back_project`

        Note:
            The returned functions will be jit compiled each time they are called with a new shape of input.  If
            called multiple times with the same shape of input, then the cached version will be used, which will
            give reduced execution time relative to the initial call.

            This method requires geometry-specific implementations of
            :meth:`TomographyModel.forward_project_pixel_batch_to_one_view` and
            :meth:`TomographyModel.back_project_one_view_to_pixel_batch`.

        """
        forward_project_pixel_batch_to_one_view = tomography_model.forward_project_pixel_batch_to_one_view
        back_project_one_view_to_pixel_batch = tomography_model.back_project_one_view_to_pixel_batch
        # Geometries whose back projection bleeds a slice across a RANGE of detector rows provide a
        # BANDED back kernel (cone; later translation/multiaxis), used by the sharded slice-band
        # path; parallel beam has none (it crops detector rows instead).
        back_project_one_view_to_band = getattr(tomography_model, 'back_project_one_view_to_band', None)

        # geometry_params already uses a shared namedtuple class (make_geometry_params); combine it
        # with the shapes into the module-level ProjectorParams type so the module-level projector
        # jit cache is shared across instances (see the note on ProjectorParams at the top).
        geometry_params = self.tomography_model.get_geometry_parameters()
        sinogram_shape, recon_shape = self.tomography_model.get_params(['sinogram_shape', 'recon_shape'])
        projector_params = ProjectorParams(sinogram_shape, recon_shape, geometry_params)

        view_params_name = self.tomography_model.get_params('view_params_name')
        # The view parameters are a RUNTIME input to the jitted projectors, not a baked
        # (closure-captured) constant: the current array is stored on this Projectors
        # object and passed as a traced argument on every call.  Because only the VALUES
        # are traced (the shape is static), TomographyModel.set_view_parameters can change
        # the angles/translations with NO recompile; a view-COUNT change is a geometry
        # change and rebuilds the projectors through set_params as before.
        self.view_params_array = jnp.asarray(self.tomography_model.get_params(view_params_name))
        pixel_batch_size = self.tomography_model.pixel_batch_size_for_vmap
        view_batch_size = self.tomography_model.view_batch_size_for_vmap

        # The jitted drivers are MODULE-LEVEL functions (defined at the end of this file),
        # not per-instance closures, so their jit cache is SHARED across model instances: a
        # second model with the same geometry reuses the first model's compiled program
        # instead of re-tracing.  The geometry-specific pieces that used to be captured by
        # closure -- the per-view kernel, projector_params, and the two batch sizes -- are
        # passed as STATIC arguments (hashable; projector_params is already static in every
        # per-view kernel).  The view-parameter array stays a TRACED argument, so
        # set_view_parameters changes its values with no recompile.
        #
        # Keep references to the module-level jitted functions on this object so existing
        # introspection (e.g. _jit_sparse_forward_project._cache_size()) still works; the
        # cache they expose is now the one shared across instances.
        self._jit_sparse_forward_project = _jit_sparse_forward_project
        self._jit_sparse_back_project = _jit_sparse_back_project

        # Both batching versions of each driver stay available side by side: v1 is the
        # original fixed-batch machinery, v2 the balanced windowed batching (see the module
        # comment on the v2 helpers).  _driver_version() reads the model attribute at CALL
        # time -- like the view-parameter late binding above -- so flipping
        # tomography_model.projector_batching_version switches versions on the next call
        # with no rebuild, and a v1-vs-v2 comparison runs on one model with identical
        # view params and placements.
        def _driver_version(v1_driver, v2_driver):
            version = getattr(self.tomography_model, 'projector_batching_version', 1)
            return v2_driver if version == 2 else v1_driver

        # Public entry points keep the original signatures; they read the CURRENT
        # view-parameter array off this object at call time (late binding), so
        # set_view_parameters takes effect on the next call with no recompile.
        def sparse_forward_project_public(voxel_values, pixel_indices, owned_view_indices=()):
            driver = _driver_version(_jit_sparse_forward_project, _jit_sparse_forward_project_v2)
            return driver(
                self.view_params_array, voxel_values, pixel_indices,
                fwd_kernel=forward_project_pixel_batch_to_one_view,
                projector_params=projector_params,
                pixel_batch_size=pixel_batch_size,
                view_batch_size=view_batch_size,
                owned_view_indices=owned_view_indices)

        def sparse_back_project_public(sinogram, pixel_indices, coeff_power=1, owned_view_indices=()):
            driver = _driver_version(_jit_sparse_back_project, _jit_sparse_back_project_v2)
            return driver(
                self.view_params_array, sinogram, pixel_indices,
                back_kernel=back_project_one_view_to_pixel_batch,
                projector_params=projector_params,
                pixel_batch_size=pixel_batch_size,
                view_batch_size=view_batch_size,
                coeff_power=coeff_power,
                owned_view_indices=owned_view_indices)

        self.sparse_forward_project = sparse_forward_project_public
        self.sparse_back_project = sparse_back_project_public

        # Banded back projector (only for geometries with a banded kernel -- cone): projects a
        # view-owner's full views onto ONE global slice band [g0, g0 + num_band_slices), batched
        # and summed exactly like sparse_back_project so memory stays bounded by the batch sizes,
        # not by the view count.  Used by the sharded reduce-scatter (see
        # TomographyModel._back_project_view_shard_to_band).  g0 is traced (per band),
        # num_band_slices is static.
        if back_project_one_view_to_band is not None:
            def sparse_back_project_band_public(sinogram, pixel_indices, g0, num_band_slices,
                                                owned_view_indices=(), coeff_power=1):
                driver = _driver_version(_jit_sparse_back_project_band,
                                         _jit_sparse_back_project_band_v2)
                return driver(
                    self.view_params_array, sinogram, pixel_indices, g0, num_band_slices,
                    back_band_kernel=back_project_one_view_to_band,
                    projector_params=projector_params,
                    pixel_batch_size=pixel_batch_size,
                    view_batch_size=view_batch_size,
                    coeff_power=coeff_power,
                    owned_view_indices=owned_view_indices)

            self.sparse_back_project_band = sparse_back_project_band_public
            self._jit_sparse_back_project_band = _jit_sparse_back_project_band


def concatenate_function_in_batches(function, data_to_batch, batch_size):
    """
    Apply a given function to a set of data, batching over the first index, concatenating the results along axis=0.
    The function should operate on subsets of the form data_to_batch[start:start+batch_size] when function takes a
    single input or on analogous subsets of each element data_to_batch[j] when function takes multiple
    inputs.  The output of function should be an array or tuple of arrays.  These output are concatenated along
    the leading axis.

    The shape of each array is determined by the output(s) of function, which
    is concatenated along the leading axis.  If function returns a fixed shape output, then the result has
    size given by the total number batches (full or partial) times the length of the leading axis of the output.

    Args:
        function (callable): A function to be mapped over batches of the input data.  This will be called on
        the unpacked elements of data_to_batch (after batching) using a call of the form
        output_data = function(batch) when data_to_batch is a single array or
        output_data = function(*batch) when data_to_batch is a tuple.
        data_to_batch (jax array or tuple of arrays): An array of data to be batched and sent to function. If a tuple,
        then each element should have the same size leading axis.
        batch_size (int): The maximum number of entries to process at one time.

    Returns:
        An array or tuple of arrays.
    """
    data_to_batch = ensure_tuple(data_to_batch)

    # Apply the batch projector directly to an initial batch
    num_input_points = data_to_batch[0].shape[0]
    batch_size = num_input_points if batch_size is None else batch_size
    num_remaining = num_input_points % batch_size

    # If the input is a multiple of batch_size, then we'll do a full batch, otherwise just the excess.
    initial_batch_size = batch_size if num_remaining == 0 else num_remaining

    initial_batch = [data[:initial_batch_size] for data in data_to_batch]
    output_data = function(*initial_batch)

    # Then deal with the batches if there are any
    if batch_size < num_input_points:
        def wrapped_function(arg_list):
            return function(*arg_list)

        num_batches = (num_input_points - initial_batch_size) // batch_size
        output_shape = (num_batches, batch_size,)
        data_batched = [jnp.reshape(data[initial_batch_size:], output_shape + data.shape[1:])
                        for data in data_to_batch]

        # Apply the function in batches
        output_data_batched = jax.lax.map(wrapped_function, data_batched)

        # The output data may be a single array or a tuple or list of arrays
        # First unbatch the data by reshaping the first 2 dims to be the number of points in all the batches.
        # Using tree_map, this can be done on either a single array or a tuple of arrays and get either a
        # single array or a tuple back
        output_unbatched = jax.tree_util.tree_map(unbatch, output_data_batched)

        # Now stack the first partial batch with this unbatched result
        output_data = jax.tree_util.tree_map(concatenate_arrays, *(output_data, output_unbatched))

    return output_data


def unbatch(array):
    """
    Reshape a jax array from (n0, n1, ...) to (n0*n1, ...)

    Args:
        array (jax array): array to be reshaped

    Returns:
        jax array
    """
    return jax.numpy.reshape(array, (array.shape[0] * array.shape[1],) + array.shape[2:])


def concatenate_arrays(*arrays):
    """
    Helper function to concatenate a list or tuple of arrays along the leading axis.

    Args:
        *arrays: list of arrays, with compatible dimensions arrays[j].shape[1:]

    Returns:
        array of shape (n, ) + arrays[0].shape[1:], where n is the sum over j of arrays[j].shape[0]
    """
    return jax.numpy.concatenate(arrays, axis=0)


def sum_function_in_batches(function_to_sum, data_to_batch, batch_size, extra_args=()):
    """
    Apply a given function to a set of data, batching over the first index, summing the results.
    The function should operate on subsets of the form data_to_batch[start:start+batch_size] when function takes a
    single input or on analogous subsets of each element data_to_batch[j] when function takes multiple
    inputs.  The output of function should be a scalar or fixed size array.

    Args:
        function_to_sum (callable): A function to be mapped over batches of the input data.  This will be called on
        the unpacked elements of data_to_batch (after batching) and extra_args using a call of the form
        summed_data += function_to_sum(batched_data, *fixed_data) when data_to_batch is a single array or
        summed_data += function_to_sum(*batched_data, *fixed_data) when data_to_batch is a tuple.
        data_to_batch (jax array or tuple of arrays): The data to be processed in batches.  If a tuple, then each element
        should have the same size leading axis.
        batch_size (int): The maximum batch size.
        extra_args (tuple): Any additional arguments needed by function_to_sum

    Returns:
        jax array or scalar output of function_to_sum, summed over all the elements in data_to_batch.
    """
    data_to_batch = ensure_tuple(data_to_batch)
    extra_args = ensure_tuple(extra_args)

    def add_one_batch(summed_and_fixed_data, batched_data):
        """
        Apply the externally defined function function_to_sum to the data in the tuple batched_data
        and add the result to an existing result.  The existing result is the first element in the tuple
        summed_and_fixed_data.  Any remaining elements of summed_and_fixed_data are for additional arguments
        to function_to_sum.  batched_data and fixed_data are unpacked before calling function_to sum. The
         primary functionality is summed_data += function_to_sum(*batched_data, *fixed_data)

        Args:
            summed_and_fixed_data (tuple or list): The first element is an array of the shape returned by
            function_to_sum.  This shape should not depend on batched_data.  The remaining elements are
            extra arguments to be sent to function_to_sum.
            batched_data (tuple or list):  The data for use in function_to_sum.

        Returns:
            tuple of ([summed_data, *fixed_data], None)
        """
        summed_data = summed_and_fixed_data[0]
        fixed_data = summed_and_fixed_data[1:]
        output_to_add = function_to_sum(*batched_data, *fixed_data)
        summed_data = jax.tree_util.tree_map(jnp.add, *(summed_data, output_to_add))

        return [summed_data, *fixed_data], None

    # Apply the batch projector directly to an initial batch to get the initial output
    num_input_points = data_to_batch[0].shape[0]
    batch_size = num_input_points if batch_size is None else batch_size
    num_remaining = num_input_points % batch_size
    # If the input is a multiple of batch_size, then we'll do a full batch, otherwise just the excess.
    initial_batch_size = batch_size if num_remaining == 0 else num_remaining

    initial_batch = [data[:initial_batch_size] for data in data_to_batch]
    summed_output = function_to_sum(*initial_batch, *extra_args)

    # Then deal with the batches if there are any
    if batch_size < num_input_points:
        num_batches = (num_input_points - initial_batch_size) // batch_size
        output_shape = (num_batches, batch_size,)
        data_batched = [jnp.reshape(data[initial_batch_size:], output_shape + data.shape[1:])
                        for data in data_to_batch]

        # Set up a scan over the batches.
        initial_carry = [summed_output, *extra_args]
        final_carry, _ = jax.lax.scan(add_one_batch, initial_carry, data_batched)

        summed_output = final_carry[0]

    return summed_output


def ensure_tuple(var_args):
    """
    Convert a singleton to a one-element tuple if needed, and convert a list to a tuple
    Args:
        var_args: singleton or list or tuple

    Returns:
        tuple
    """
    # Check if var_args is already a tuple
    if isinstance(var_args, tuple):
        return var_args
    # Check if var_args is a list
    elif isinstance(var_args, list):
        return tuple(var_args)
    # Assume var_args is a single item if it's neither a list nor a tuple
    else:
        return (var_args, )


# ──────────────────────────────────────────────────────────────────────────────
# v2 batching helpers: balanced windowed batching
#
# These are the v2 counterparts of concatenate_function_in_batches / sum_function_in_batches,
# with the same call signatures and output contracts.  Two mechanical changes (see
# experiments/projector_batching/batching_refactor_design.md for the full rationale and the
# measurements behind it):
#
#   1. BALANCED batch sizes.  Instead of a fixed batch with a ragged remainder (which can be
#      arbitrarily small), the batch size is reduced to the smallest value that covers the
#      input in the same number of batches: num_batches = ceil(n / batch_size), then
#      balanced_size = ceil(n / num_batches).  All full batches are equal, and the one
#      partial batch (when the input does not divide evenly) is NEAR-FULL-SIZE rather than
#      arbitrarily small.  Measured on H100: per-item projector time is flat in batch size
#      down to half the cap, so balancing is free in time and makes ANY cap shape-safe.
#
#   2. WINDOWED reads via lax.dynamic_slice instead of jnp.reshape -- on the SUM axes only
#      (sum_in_balanced_batches).  The v1 reshape of the batched input materializes a full
#      copy of that input on GPU (measured: one extra view-shard-sized temp in the back
#      projector); slicing one window per scan step reads the original array in place.  The
#      CONCATENATE axes (map_in_balanced_batches) keep v1's reshape + lax.map mechanics --
#      measured ~1.6x faster on CPU than the windowed form, and their batched inputs are
#      small index/parameter arrays, so the reshape copy is negligible there (see that
#      function's docstring).
#
# Like v1, the partial batch (if any) runs FIRST as a separate call, then lax.map / lax.scan
# covers the equal full batches.  In the sum helper, an evenly divisible input skips the
# partial call entirely -- the scan carry starts at zeros (shaped via jax.eval_shape, which
# traces without compiling), one fewer inlined kernel than v1.
# ──────────────────────────────────────────────────────────────────────────────
def balanced_batch(num_input_points, batch_size):
    """Compute a balanced batching of ``num_input_points`` items with batches <= ``batch_size``.

    Uses the fewest batches that respect the cap, with all full batches equal:
    ``num_batches = ceil(n / batch_size)``, ``balanced_size = ceil(n / num_batches)``.
    The same policy as TomographyModel._balanced_slice_bounds, in batch-size form.

    Args:
        num_input_points (int): Total number of items (n > 0).
        batch_size (int or None): Maximum batch size; None means one batch of everything.

    Returns:
        tuple (balanced_size, num_batches, residual): ``residual = num_batches *
        balanced_size - num_input_points`` satisfies ``0 <= residual < balanced_size``; when
        residual > 0 the caller runs one partial batch of ``balanced_size - residual`` items
        (which is >= 1) plus ``num_batches - 1`` full batches.
    """
    if batch_size is None:
        batch_size = num_input_points
    batch_size = min(batch_size, num_input_points)
    num_batches = -(-num_input_points // batch_size)          # ceil division
    balanced_size = -(-num_input_points // num_batches)
    residual = num_batches * balanced_size - num_input_points
    return balanced_size, num_batches, residual


def map_in_balanced_batches(function, data_to_batch, batch_size):
    """v2 of :func:`concatenate_function_in_batches`: same contract and mechanics, with
    BALANCED batch sizes.

    Applies ``function`` to leading-axis batches of ``data_to_batch`` and concatenates the
    per-batch outputs along the leading axis (which must match the input batch size, as in
    v1).  Only the SIZING differs from v1: the near-full-size partial batch runs first, then
    ``lax.map`` covers the equal full batches.

    The mechanics deliberately stay v1's reshape + ``lax.map`` + concatenate, NOT a windowed
    scan: measured on CPU (cone forward, N=128, identical batch width 128), a scan writing
    per-window results into a carry via dynamic_update_slice was ~1.6x SLOWER than lax.map --
    the map's stacked output fuses with the kernel where the carry update does not.  The
    input reshape that the windowed form would have avoided is harmless HERE by construction:
    every concatenate-axis input in the projector drivers is a small index/parameter array
    (view params, pixel indices) -- the large arrays are closed over, not batched, on these
    axes.  The large-array reshape copy lives on the SUM axes, where
    :func:`sum_in_balanced_batches` does use the windowed form.

    Args:
        function (callable): Called as function(*batch) on leading-axis batches of the data.
        data_to_batch (jax array or tuple of arrays): Same-size leading axes.
        batch_size (int or None): Maximum batch size (None = single call).

    Returns:
        An array or tuple of arrays, as returned by function, with leading axis equal to the
        total number of input points.
    """
    data_to_batch = ensure_tuple(data_to_batch)
    num_input_points = data_to_batch[0].shape[0]
    balanced_size, num_batches, residual = balanced_batch(num_input_points, batch_size)
    if num_batches == 1:
        return function(*data_to_batch)

    # The initial batch always runs inline first, exactly as in v1 (a full batch when the
    # input divides evenly, else the near-full-size partial batch) and lax.map covers the
    # remaining full batches.
    initial_size = balanced_size if residual == 0 else balanced_size - residual
    output_data = function(*[data[:initial_size] for data in data_to_batch])

    def wrapped_function(arg_list):
        return function(*arg_list)

    num_mapped_batches = (num_input_points - initial_size) // balanced_size
    output_shape = (num_mapped_batches, balanced_size)
    data_batched = [jnp.reshape(data[initial_size:], output_shape + data.shape[1:])
                    for data in data_to_batch]
    output_data_batched = jax.lax.map(wrapped_function, data_batched)
    output_unbatched = jax.tree_util.tree_map(unbatch, output_data_batched)

    output_data = jax.tree_util.tree_map(concatenate_arrays,
                                         *(output_data, output_unbatched))
    return output_data


def sum_in_balanced_batches(function_to_sum, data_to_batch, batch_size, extra_args=()):
    """v2 of :func:`sum_function_in_batches`: same contract, balanced windowed batches.

    Applies ``function_to_sum`` to leading-axis batches of ``data_to_batch`` and SUMS the
    per-batch outputs (which must have a batch-independent shape, as in v1).  The windows are
    read with ``dynamic_slice`` (no input reshape/copy); the partial batch (if any) runs
    first, mirroring v1's initial batch.

    Args:
        function_to_sum (callable): Called as function_to_sum(*batch, *extra_args).
        data_to_batch (jax array or tuple of arrays): Same-size leading axes.
        batch_size (int or None): Maximum window size (None = single call).
        extra_args (tuple): Additional non-batched arguments for function_to_sum.

    Returns:
        The output of function_to_sum summed over all batches.
    """
    data_to_batch = ensure_tuple(data_to_batch)
    extra_args = ensure_tuple(extra_args)
    num_input_points = data_to_batch[0].shape[0]
    balanced_size, num_batches, residual = balanced_batch(num_input_points, batch_size)
    if num_batches == 1:
        return function_to_sum(*data_to_batch, *extra_args)

    num_full_batches = num_batches if residual == 0 else num_batches - 1
    partial_size = num_input_points - num_full_batches * balanced_size

    if partial_size > 0:
        # Initialize the running sum from the near-full-size partial batch (v1's structure).
        summed_output = function_to_sum(*[data[:partial_size] for data in data_to_batch],
                                        *extra_args)
    else:
        # Evenly divisible: start the sum at zeros so the kernel appears ONCE (in the scan
        # body).  eval_shape supplies the output structure without compiling anything.
        window0 = [jax.lax.dynamic_slice_in_dim(data, 0, balanced_size, axis=0)
                   for data in data_to_batch]
        out_struct = jax.eval_shape(lambda *w: function_to_sum(*w, *extra_args), *window0)
        summed_output = jax.tree_util.tree_map(
            lambda s: jnp.zeros(s.shape, dtype=s.dtype), out_struct)

    def add_one_batch(summed_and_fixed_data, k):
        # Carry layout mirrors v1: [running sum, *extra_args].
        summed_data = summed_and_fixed_data[0]
        fixed_data = summed_and_fixed_data[1:]
        start = partial_size + k * balanced_size
        window = [jax.lax.dynamic_slice_in_dim(data, start, balanced_size, axis=0)
                  for data in data_to_batch]
        output_to_add = function_to_sum(*window, *fixed_data)
        summed_data = jax.tree_util.tree_map(jnp.add, summed_data, output_to_add)
        return [summed_data, *fixed_data], None

    initial_carry = [summed_output, *extra_args]
    final_carry, _ = jax.lax.scan(add_one_batch, initial_carry, jnp.arange(num_full_batches))
    return final_carry[0]


# ──────────────────────────────────────────────────────────────────────────────
# Module-level projector drivers (shared jit cache across model instances)
#
# These were per-instance closures inside Projectors.create_projectors.  Lifting them to
# module level -- with the geometry-specific per-view kernel, projector_params, and batch
# sizes as STATIC arguments and the view-parameter array as a TRACED argument -- lets
# jax.jit key its cache on those values, so two models with the same geometry SHARE one
# compiled program instead of each re-tracing (tracing dominates the per-model first-call
# cost).  Only the two top-level signatures (Projectors.sparse_forward_project /
# .sparse_back_project) are part of the public interface; these drivers are internal.
# ──────────────────────────────────────────────────────────────────────────────
def _sparse_forward_project(view_params_array, voxel_values, pixel_indices,
                            fwd_kernel, projector_params, pixel_batch_size, view_batch_size,
                            owned_view_indices=()):
    """Forward project voxels to a sinogram, batching over pixels then views.

    Batches over pixels (sum_function_in_batches); within each pixel batch, maps the
    geometry's per-view forward kernel over batches of views (concatenate_function_in_batches)
    and sums the pixel-batch contributions.  fwd_kernel, projector_params, pixel_batch_size and
    view_batch_size are static; view_params_array / voxel_values / pixel_indices are traced.
    """
    cur_view_params_array = view_params_array
    if len(owned_view_indices) > 0:
        cur_view_params_array = view_params_array[owned_view_indices]

    def forward_project_pixel_batch(local_values, local_pix_indices):
        def forward_project_single_view(single_view_params):
            return fwd_kernel(local_values, local_pix_indices, single_view_params, projector_params)

        def forward_project_view_batch(view_params_batch):
            # vmap (not lax.map) over views: the per-view kernel reuses the same voxel batch,
            # so parallelizing over views trades a little memory for speed.
            return jax.vmap(forward_project_single_view)(view_params_batch)

        return concatenate_function_in_batches(forward_project_view_batch, cur_view_params_array,
                                               view_batch_size)

    return sum_function_in_batches(forward_project_pixel_batch, (voxel_values, pixel_indices),
                                   pixel_batch_size)


_jit_sparse_forward_project = jax.jit(
    _sparse_forward_project,
    static_argnames=['fwd_kernel', 'projector_params', 'pixel_batch_size', 'view_batch_size'])


def _sparse_back_project(view_params_array, sinogram, pixel_indices,
                         back_kernel, projector_params, pixel_batch_size, view_batch_size,
                         coeff_power=1, owned_view_indices=()):
    """Back project a sinogram to voxels, batching over views (summing) then pixels.

    Batches over views (sum_function_in_batches); within each view batch, maps the geometry's
    per-view back kernel over the views (vmap) and sums them, then batches over pixels
    (concatenate_function_in_batches).  back_kernel, projector_params, pixel_batch_size,
    view_batch_size and coeff_power are static; view_params_array / sinogram / pixel_indices
    are traced.
    """
    cur_view_params_array = view_params_array
    if len(owned_view_indices) > 0:
        cur_view_params_array = view_params_array[owned_view_indices]

    def back_project_view_batch(local_view_batch, local_view_params_batch, local_pixel_indices):
        def back_project_pixel_batch(pixel_indices_batch):
            # Map the per-view back kernel over the views in this batch, then sum over views.
            bp_vmap = jax.vmap(back_kernel, in_axes=(0, None, 0, None, None))
            per_view_voxel_values_batch = bp_vmap(local_view_batch, pixel_indices_batch,
                                                  local_view_params_batch, projector_params, coeff_power)
            # Driver-level reduce (shared across geometries, not the geometry kernel) -- give it its
            # own named_scope so profilers attribute it as a distinct region instead of leaving it
            # unscoped/unmapped.
            with jax.named_scope("projector/back/view_reduce"):
                return jnp.sum(per_view_voxel_values_batch, axis=0)

        return concatenate_function_in_batches(back_project_pixel_batch, local_pixel_indices,
                                               pixel_batch_size)

    return sum_function_in_batches(back_project_view_batch, (sinogram, cur_view_params_array),
                                   view_batch_size, pixel_indices)


_jit_sparse_back_project = jax.jit(
    _sparse_back_project,
    static_argnames=['back_kernel', 'projector_params', 'pixel_batch_size', 'view_batch_size',
                     'coeff_power'])


def _sparse_back_project_band(view_params_array, sinogram, pixel_indices, g0, num_band_slices,
                              back_band_kernel, projector_params, pixel_batch_size, view_batch_size,
                              coeff_power=1, owned_view_indices=()):
    """Back project a sinogram onto the GLOBAL recon slice band [g0, g0 + num_band_slices).

    The banded analogue of _sparse_back_project: it batches over views (summing) then pixels in
    exactly the same way -- so the per-view-batch vmap stack and the per-pixel-batch transients are
    bounded by view_batch_size / pixel_batch_size, NOT by the number of views -- but calls the
    geometry's BANDED back kernel, which produces only the L = num_band_slices slices [g0, g0 + L)
    from the FULL view (no detector-row crop; for geometries whose back projection draws a slice
    from a RANGE of rows).  g0 is TRACED (a band's global start, so the bands of a slice axis do
    not retrace); num_band_slices is STATIC (it sets the output slice count); back_band_kernel /
    projector_params / batch sizes / coeff_power are static.
    """
    cur_view_params_array = view_params_array
    if len(owned_view_indices) > 0:
        cur_view_params_array = view_params_array[owned_view_indices]

    def back_project_view_batch(local_view_batch, local_view_params_batch, local_pixel_indices):
        def back_project_pixel_batch(pixel_indices_batch):
            # Map the per-view banded back kernel over the views in this batch, then sum over views.
            bp_vmap = jax.vmap(back_band_kernel, in_axes=(0, None, 0, None, None, None, None))
            per_view_band = bp_vmap(local_view_batch, pixel_indices_batch, local_view_params_batch,
                                    projector_params, g0, num_band_slices, coeff_power)
            with jax.named_scope("projector/back/view_reduce"):   # driver reduce; see _sparse_back_project
                return jnp.sum(per_view_band, axis=0)

        return concatenate_function_in_batches(back_project_pixel_batch, local_pixel_indices,
                                               pixel_batch_size)

    return sum_function_in_batches(back_project_view_batch, (sinogram, cur_view_params_array),
                                   view_batch_size, pixel_indices)


_jit_sparse_back_project_band = jax.jit(
    _sparse_back_project_band,
    static_argnames=['num_band_slices', 'back_band_kernel', 'projector_params',
                     'pixel_batch_size', 'view_batch_size', 'coeff_power'])


# ──────────────────────────────────────────────────────────────────────────────
# v2 drivers: line-for-line mirrors of the v1 drivers above with ONLY the batching
# helpers swapped (concatenate_function_in_batches -> map_in_balanced_batches,
# sum_function_in_batches -> sum_in_balanced_batches).  Kept structurally identical to v1
# on purpose: stepping through the two side by side shows exactly one mechanism changing,
# and any v1-vs-v2 difference is attributable to the batching alone.  Selected per model
# via TomographyModel.projector_batching_version (see Projectors.create_projectors).
# ──────────────────────────────────────────────────────────────────────────────
def _sparse_forward_project_v2(view_params_array, voxel_values, pixel_indices,
                               fwd_kernel, projector_params, pixel_batch_size, view_batch_size,
                               owned_view_indices=()):
    """v2 of :func:`_sparse_forward_project` (balanced windowed batching; same contract)."""
    cur_view_params_array = view_params_array
    if len(owned_view_indices) > 0:
        cur_view_params_array = view_params_array[owned_view_indices]

    def forward_project_pixel_batch(local_values, local_pix_indices):
        def forward_project_single_view(single_view_params):
            return fwd_kernel(local_values, local_pix_indices, single_view_params, projector_params)

        def forward_project_view_batch(view_params_batch):
            # vmap (not a sequential map) over views: the per-view kernel reuses the same
            # voxel batch, so parallelizing over views trades a little memory for speed.
            return jax.vmap(forward_project_single_view)(view_params_batch)

        return map_in_balanced_batches(forward_project_view_batch, cur_view_params_array,
                                       view_batch_size)

    return sum_in_balanced_batches(forward_project_pixel_batch, (voxel_values, pixel_indices),
                                   pixel_batch_size)


_jit_sparse_forward_project_v2 = jax.jit(
    _sparse_forward_project_v2,
    static_argnames=['fwd_kernel', 'projector_params', 'pixel_batch_size', 'view_batch_size'])


def _sparse_back_project_v2(view_params_array, sinogram, pixel_indices,
                            back_kernel, projector_params, pixel_batch_size, view_batch_size,
                            coeff_power=1, owned_view_indices=()):
    """v2 of :func:`_sparse_back_project` (balanced windowed batching; same contract)."""
    cur_view_params_array = view_params_array
    if len(owned_view_indices) > 0:
        cur_view_params_array = view_params_array[owned_view_indices]

    def back_project_view_batch(local_view_batch, local_view_params_batch, local_pixel_indices):
        def back_project_pixel_batch(pixel_indices_batch):
            # Map the per-view back kernel over the views in this batch, then sum over views.
            bp_vmap = jax.vmap(back_kernel, in_axes=(0, None, 0, None, None))
            per_view_voxel_values_batch = bp_vmap(local_view_batch, pixel_indices_batch,
                                                  local_view_params_batch, projector_params, coeff_power)
            with jax.named_scope("projector/back/view_reduce"):   # driver reduce; see v1
                return jnp.sum(per_view_voxel_values_batch, axis=0)

        return map_in_balanced_batches(back_project_pixel_batch, local_pixel_indices,
                                       pixel_batch_size)

    return sum_in_balanced_batches(back_project_view_batch, (sinogram, cur_view_params_array),
                                   view_batch_size, pixel_indices)


_jit_sparse_back_project_v2 = jax.jit(
    _sparse_back_project_v2,
    static_argnames=['back_kernel', 'projector_params', 'pixel_batch_size', 'view_batch_size',
                     'coeff_power'])


def _sparse_back_project_band_v2(view_params_array, sinogram, pixel_indices, g0, num_band_slices,
                                 back_band_kernel, projector_params, pixel_batch_size, view_batch_size,
                                 coeff_power=1, owned_view_indices=()):
    """v2 of :func:`_sparse_back_project_band` (balanced windowed batching; same contract)."""
    cur_view_params_array = view_params_array
    if len(owned_view_indices) > 0:
        cur_view_params_array = view_params_array[owned_view_indices]

    def back_project_view_batch(local_view_batch, local_view_params_batch, local_pixel_indices):
        def back_project_pixel_batch(pixel_indices_batch):
            # Map the per-view banded back kernel over the views in this batch, then sum over views.
            bp_vmap = jax.vmap(back_band_kernel, in_axes=(0, None, 0, None, None, None, None))
            per_view_band = bp_vmap(local_view_batch, pixel_indices_batch, local_view_params_batch,
                                    projector_params, g0, num_band_slices, coeff_power)
            with jax.named_scope("projector/back/view_reduce"):   # driver reduce; see v1
                return jnp.sum(per_view_band, axis=0)

        return map_in_balanced_batches(back_project_pixel_batch, local_pixel_indices,
                                       pixel_batch_size)

    return sum_in_balanced_batches(back_project_view_batch, (sinogram, cur_view_params_array),
                                   view_batch_size, pixel_indices)


_jit_sparse_back_project_band_v2 = jax.jit(
    _sparse_back_project_band_v2,
    static_argnames=['num_band_slices', 'back_band_kernel', 'projector_params',
                     'pixel_batch_size', 'view_batch_size', 'coeff_power'])
