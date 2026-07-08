"""Tests for the concrete scatter centers (projectors._jit_compute_scatter_centers
and the wrapper-level machinery around it).

The projector wrappers make the horizontal fans' integer channel centers CONCRETE inputs to the projector
programs, removing the round-inside-vmap/map/scatter precondition of the known XLA rounding
bug (see experiments/bugs_and_artifacts/jax rounding bug/phase_d_design.md).  These
tests pin: the centers computation against a direct round of the geometry coordinate (both
layouts), the chunked wrapper path against the single-call path (the hybrid must only choose
WHERE the view loop lives, never values), and the concreteness guard (the wrappers must
refuse to run inside an outer jit, where the centers would silently become tracers).
"""
import unittest

import numpy as np
import jax
import jax.numpy as jnp

import mbirjax
import mbirjax.projectors as projectors
from mbirjax.projectors import ProjectorParams, _jit_compute_scatter_centers


def make_model(num_views=24, num_rows=40, num_channels=32):
    angles = np.linspace(0, np.pi, num_views, endpoint=False)
    model = mbirjax.ParallelBeamModel((num_views, num_rows, num_channels), angles)
    model.configure_devices(1)
    return model


class TestScatterCenters(unittest.TestCase):

    def test_centers_match_rounded_coordinate_both_layouts(self):
        model = make_model()
        pp = ProjectorParams(tuple(model.get_params('sinogram_shape')),
                             tuple(model.get_params('recon_shape')),
                             model.get_geometry_parameters())
        recon_shape = model.get_params('recon_shape')
        idx = mbirjax.gen_full_indices(recon_shape, use_ror_mask=False)
        vp = model.projector_functions.view_params_array
        ref = np.stack([np.asarray(jnp.round(
            model.compute_channel_coordinate(idx, v, pp)).astype(jnp.int32)) for v in vp])
        vp_major = _jit_compute_scatter_centers(vp, idx, channel_coord_fn=model.compute_channel_coordinate,
                                                projector_params=pp, pixels_major=False)
        px_major = _jit_compute_scatter_centers(vp, idx, channel_coord_fn=model.compute_channel_coordinate,
                                                projector_params=pp, pixels_major=True)
        self.assertEqual(vp_major.dtype, jnp.int32)
        np.testing.assert_array_equal(np.asarray(vp_major), ref)          # (V, P)
        np.testing.assert_array_equal(np.asarray(px_major), ref.T)        # (P, V)

    def test_chunked_wrapper_matches_single_call(self):
        # Force the chunked path with a tiny threshold: results must match the single-call
        # path EXACTLY for forward (same per-chunk programs, concatenated) and to summation
        # order for back (the chunk accumulation reorders the view sum).
        model = make_model()
        recon_shape = model.get_params('recon_shape')
        idx = mbirjax.gen_full_indices(recon_shape, use_ror_mask=False)
        rng = np.random.default_rng(2)
        vox = rng.random((len(idx), recon_shape[2]), dtype=np.float32)
        sino = rng.random(model.get_params('sinogram_shape'), dtype=np.float32)
        pf = model.projector_functions      # the wrapper level, where the hybrid lives
        fwd_single = np.asarray(pf.sparse_forward_project(vox, idx))
        back_single = np.asarray(pf.sparse_back_project(sino, idx))
        saved = projectors.N_PC_SINGLE_CALL_MAX_BYTES
        try:
            projectors.N_PC_SINGLE_CALL_MAX_BYTES = 1     # everything chunks
            fwd_chunked = np.asarray(pf.sparse_forward_project(vox, idx))
            back_chunked = np.asarray(pf.sparse_back_project(sino, idx))
        finally:
            projectors.N_PC_SINGLE_CALL_MAX_BYTES = saved
        np.testing.assert_array_equal(fwd_chunked, fwd_single)
        np.testing.assert_allclose(back_chunked, back_single, rtol=1e-5,
                                   atol=1e-5 * np.abs(back_single).max())
        # Also with owned views (the sharded path's calling convention): an odd-sized subset
        # exercises the ragged tail chunk.
        owned = tuple(range(3, 20))
        fwd_o = np.asarray(pf.sparse_forward_project(vox, idx, owned_view_indices=owned))
        projectors.N_PC_SINGLE_CALL_MAX_BYTES = 1
        try:
            fwd_o_chunked = np.asarray(pf.sparse_forward_project(vox, idx, owned_view_indices=owned))
        finally:
            projectors.N_PC_SINGLE_CALL_MAX_BYTES = saved
        self.assertEqual(fwd_o.shape[0], len(owned))
        np.testing.assert_array_equal(fwd_o_chunked, fwd_o)

    def test_wrapper_refuses_outer_jit(self):
        # The CONCRETENESS contract: inside an outer jit the centers would become tracers
        # and the rounding-bug precondition would silently return -- the wrapper must fail
        # loudly instead.
        model = make_model()
        recon_shape = model.get_params('recon_shape')
        idx = mbirjax.gen_full_indices(recon_shape, use_ror_mask=False)
        vox = np.zeros((len(idx), recon_shape[2]), dtype=np.float32)

        def wrapped(v):
            return model.projector_functions.sparse_forward_project(v, idx)

        with self.assertRaises(AssertionError):
            jax.jit(wrapped)(vox)


if __name__ == '__main__':
    unittest.main()
