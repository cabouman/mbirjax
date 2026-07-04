"""v1-vs-v2 equivalence tests for the balanced windowed projector batching.

The v2 drivers (projectors.py: map_in_balanced_batches / sum_in_balanced_batches and the
_v2 drivers) must match v1 to float noise for every batching branch.  The gate is the
scale-invariant rel-max (conftest.assert_sharded_allclose) at 1e-5, NEVER exact equality:
v1 and v2 are different executables, and the sum-axis batch boundaries legitimately change
the reduction order.

The axis-count matrix is chosen to hit every static branch of balanced_batch on both axes:
n < B (single call), n == B, B | n (no partial batch), and two ragged cases including
n = B + 1 (the worst case, B* ~ B/2).  Design and measurements:
experiments/projector_batching/batching_refactor_design.md.
"""
import numpy as np
import pytest

import mbirjax
import jax.numpy as jnp
from mbirjax.projectors import (
    ProjectorParams, balanced_batch,
    _jit_sparse_forward_project, _jit_sparse_forward_project_v2,
    _jit_sparse_back_project, _jit_sparse_back_project_v2,
    _jit_sparse_back_project_band, _jit_sparse_back_project_band_v2)

from conftest import assert_sharded_allclose


# ----------------------------------------------------------------------------------
# balanced_batch: the pure host helper
# ----------------------------------------------------------------------------------
def test_balanced_batch_invariants():
    """Exhaustive small sweep: the balanced tiling must cover n exactly with equal full
    batches <= batch_size and a partial batch that is never empty."""
    for n in range(1, 300):
        for batch_size in list(range(1, 20)) + [n, n + 7, None]:
            balanced_size, num_batches, residual = balanced_batch(n, batch_size)
            cap = n if batch_size is None else min(batch_size, n)
            assert 1 <= balanced_size <= cap
            assert num_batches == -(-n // cap)               # fewest batches under the cap
            assert 0 <= residual < balanced_size             # partial batch is never empty
            num_full = num_batches if residual == 0 else num_batches - 1
            partial = n - num_full * balanced_size
            assert partial == (0 if residual == 0 else balanced_size - residual)
            assert partial + num_full * balanced_size == n   # exact tiling, no overlap


# ----------------------------------------------------------------------------------
# Driver equivalence
# ----------------------------------------------------------------------------------
# Batch caps for the tests (small, so the small models below exercise real batching).
VIEW_B = 8
PIXEL_B = 16

# Axis counts hitting every balanced_batch branch: below/at/multiple-of/just-above/ragged.
VIEW_COUNTS = [5, 8, 24, 9, 21]      # vs VIEW_B=8:  n<B, n==B, B|n, n=B+1, ragged
PIXEL_COUNTS = [10, 16, 48, 17, 43]  # vs PIXEL_B=16: same branches


def make_cone_model():
    num_views = max(VIEW_COUNTS)
    angles = jnp.linspace(0, np.pi, num_views, endpoint=False)
    model = mbirjax.ConeBeamModel((num_views, 24, 24), angles,
                                  source_detector_dist=96.0, source_iso_dist=48.0)
    return model


def make_parallel_model():
    num_views = max(VIEW_COUNTS)
    angles = jnp.linspace(0, np.pi, num_views, endpoint=False)
    return mbirjax.ParallelBeamModel((num_views, 24, 24), angles)


def driver_args(model):
    """Assemble the traced/static driver arguments from a model, as create_projectors does."""
    sinogram_shape, recon_shape = model.get_params(['sinogram_shape', 'recon_shape'])
    projector_params = ProjectorParams(tuple(sinogram_shape), tuple(recon_shape),
                                       model.get_geometry_parameters())
    view_params = model.projector_functions.view_params_array
    return sinogram_shape, recon_shape, projector_params, view_params


def make_inputs(model, num_pixels, seed):
    """Seeded voxel cylinders, pixel indices, and a sinogram sized to the model."""
    rng = np.random.default_rng(seed)
    sinogram_shape, recon_shape, _, _ = driver_args(model)
    max_index = recon_shape[0] * recon_shape[1]
    pixel_indices = jnp.array(rng.choice(max_index, size=num_pixels, replace=False),
                              dtype=jnp.int32)
    voxel_values = jnp.array(rng.standard_normal((num_pixels, recon_shape[2])),
                             dtype=jnp.float32)
    sinogram = jnp.array(rng.standard_normal(tuple(sinogram_shape)), dtype=jnp.float32)
    return voxel_values, pixel_indices, sinogram


@pytest.fixture(scope='module')
def cone_model():
    return make_cone_model()


@pytest.fixture(scope='module')
def parallel_model():
    return make_parallel_model()


def _model(request, which):
    return request.getfixturevalue(which)


@pytest.mark.parametrize('which_model', ['cone_model', 'parallel_model'])
@pytest.mark.parametrize('num_views', VIEW_COUNTS)
def test_forward_v1_v2_view_axis(request, which_model, num_views):
    """Forward driver, sweeping the view axis branches at a fixed ragged pixel count."""
    model = _model(request, which_model)
    _, _, projector_params, view_params = driver_args(model)
    voxel_values, pixel_indices, _ = make_inputs(model, num_pixels=43, seed=0)
    owned = jnp.arange(num_views)      # a view-shard index range, as the sharded paths pass
    kwargs = dict(fwd_kernel=model.forward_project_pixel_batch_to_one_view,
                  projector_params=projector_params,
                  pixel_batch_size=PIXEL_B, view_batch_size=VIEW_B,
                  owned_view_indices=owned)
    out_v1 = _jit_sparse_forward_project(view_params, voxel_values, pixel_indices, **kwargs)
    out_v2 = _jit_sparse_forward_project_v2(view_params, voxel_values, pixel_indices, **kwargs)
    assert_sharded_allclose(out_v2, out_v1, msg=f'forward {which_model} views={num_views}')


@pytest.mark.parametrize('which_model', ['cone_model', 'parallel_model'])
@pytest.mark.parametrize('num_pixels', PIXEL_COUNTS)
def test_forward_v1_v2_pixel_axis(request, which_model, num_pixels):
    """Forward driver, sweeping the pixel (sum) axis branches at a ragged view count."""
    model = _model(request, which_model)
    _, _, projector_params, view_params = driver_args(model)
    voxel_values, pixel_indices, _ = make_inputs(model, num_pixels=num_pixels, seed=1)
    owned = jnp.arange(21)
    kwargs = dict(fwd_kernel=model.forward_project_pixel_batch_to_one_view,
                  projector_params=projector_params,
                  pixel_batch_size=PIXEL_B, view_batch_size=VIEW_B,
                  owned_view_indices=owned)
    out_v1 = _jit_sparse_forward_project(view_params, voxel_values, pixel_indices, **kwargs)
    out_v2 = _jit_sparse_forward_project_v2(view_params, voxel_values, pixel_indices, **kwargs)
    assert_sharded_allclose(out_v2, out_v1, msg=f'forward {which_model} pixels={num_pixels}')


@pytest.mark.parametrize('which_model', ['cone_model', 'parallel_model'])
@pytest.mark.parametrize('coeff_power', [1, 2])
@pytest.mark.parametrize('num_views', VIEW_COUNTS)
def test_back_v1_v2_view_axis(request, which_model, coeff_power, num_views):
    """Back driver, sweeping the view (sum) axis branches; coeff_power=2 is the Hessian path."""
    model = _model(request, which_model)
    _, _, projector_params, view_params = driver_args(model)
    _, pixel_indices, sinogram = make_inputs(model, num_pixels=43, seed=2)
    owned = jnp.arange(num_views)
    local_sino = sinogram[owned]       # the view shard this owner holds
    kwargs = dict(back_kernel=model.back_project_one_view_to_pixel_batch,
                  projector_params=projector_params,
                  pixel_batch_size=PIXEL_B, view_batch_size=VIEW_B,
                  coeff_power=coeff_power, owned_view_indices=owned)
    out_v1 = _jit_sparse_back_project(view_params, local_sino, pixel_indices, **kwargs)
    out_v2 = _jit_sparse_back_project_v2(view_params, local_sino, pixel_indices, **kwargs)
    assert_sharded_allclose(out_v2, out_v1,
                            msg=f'back {which_model} views={num_views} p={coeff_power}')


@pytest.mark.parametrize('which_model', ['cone_model', 'parallel_model'])
@pytest.mark.parametrize('num_pixels', PIXEL_COUNTS)
def test_back_v1_v2_pixel_axis(request, which_model, num_pixels):
    """Back driver, sweeping the pixel (concatenate) axis branches."""
    model = _model(request, which_model)
    _, _, projector_params, view_params = driver_args(model)
    _, pixel_indices, sinogram = make_inputs(model, num_pixels=num_pixels, seed=3)
    owned = jnp.arange(21)
    local_sino = sinogram[owned]
    kwargs = dict(back_kernel=model.back_project_one_view_to_pixel_batch,
                  projector_params=projector_params,
                  pixel_batch_size=PIXEL_B, view_batch_size=VIEW_B,
                  coeff_power=1, owned_view_indices=owned)
    out_v1 = _jit_sparse_back_project(view_params, local_sino, pixel_indices, **kwargs)
    out_v2 = _jit_sparse_back_project_v2(view_params, local_sino, pixel_indices, **kwargs)
    assert_sharded_allclose(out_v2, out_v1, msg=f'back {which_model} pixels={num_pixels}')


@pytest.mark.parametrize('num_views', [9, 21])
@pytest.mark.parametrize('g0_band', [(0, 6), (5, 7)])
def test_back_band_v1_v2(cone_model, num_views, g0_band):
    """Banded back driver (cone only), ragged on both axes, two band positions."""
    model = cone_model
    _, recon_shape, projector_params, view_params = driver_args(model)
    _, pixel_indices, sinogram = make_inputs(model, num_pixels=43, seed=4)
    g0, num_band_slices = g0_band
    assert g0 + num_band_slices <= recon_shape[2]
    owned = jnp.arange(num_views)
    local_sino = sinogram[owned]
    kwargs = dict(back_band_kernel=model.back_project_one_view_to_band,
                  projector_params=projector_params,
                  pixel_batch_size=PIXEL_B, view_batch_size=VIEW_B,
                  coeff_power=1, owned_view_indices=owned)
    out_v1 = _jit_sparse_back_project_band(view_params, local_sino, pixel_indices,
                                           g0, num_band_slices, **kwargs)
    out_v2 = _jit_sparse_back_project_band_v2(view_params, local_sino, pixel_indices,
                                              g0, num_band_slices, **kwargs)
    assert_sharded_allclose(out_v2, out_v1, msg=f'band views={num_views} band={g0_band}')


def test_model_level_version_flip(cone_model):
    """The projector_batching_version attribute switches versions on a LIVE model with no
    rebuild, and the two versions agree through the full model-level projector path."""
    model = cone_model
    recon_shape = model.get_params('recon_shape')
    rng = np.random.default_rng(5)
    num_pixels = 43
    pixel_indices = jnp.array(rng.choice(recon_shape[0] * recon_shape[1], size=num_pixels,
                                         replace=False), dtype=jnp.int32)
    voxel_values = jnp.array(rng.standard_normal((num_pixels, recon_shape[2])),
                             dtype=jnp.float32)

    # Set the version explicitly on both calls (the default may itself be overridden by
    # MBIRJAX_PROJECTOR_BATCHING_VERSION when the suite is run with v2 forced).
    original_version = model.projector_batching_version
    try:
        model.projector_batching_version = 1
        sino_v1 = np.asarray(model.sparse_forward_project(voxel_values, pixel_indices))
        model.projector_batching_version = 2
        sino_v2 = np.asarray(model.sparse_forward_project(voxel_values, pixel_indices))
    finally:
        model.projector_batching_version = original_version    # module-scoped fixture: restore
    assert_sharded_allclose(sino_v2, sino_v1, msg='model-level forward, version flip')
