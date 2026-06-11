"""
Tests for TomographyModel.set_view_parameters (settable view parameters).

The view-parameter array (angles / translation vectors) is a RUNTIME input to the
jitted projectors, not a closure-baked constant, so set_view_parameters changes the
values with no recompile.  The contract tested here:

  - a model whose parameters are CHANGED to B must project the same as a model
    BUILT with B, to tight float tolerance.  NOT bit-exact: the two models compile
    SEPARATE executables for the same program, and on GPU two compilations of
    identical HLO can differ by ~1 ULP (autotuning / reduction order) -- the
    established suite convention is exact equality only for same-executable
    identities, tight allclose across compilations.  A stale baked value would
    fail at O(1), so the tolerance still proves nothing is baked;
  - the A -> B -> A round trip on ONE model restores the original projection to
    tight tolerance (exact equality is never the gate for computed floats: even
    one executable can reorder GPU scatter-add summation run to run);
  - changing values does not add jit cache entries (the no-recompile guarantee);
  - a shape (view-count) change is rejected -- that is a geometry change for
    set_params;
  - works for multi-component view parameters (cone: angle + z-shift) and on the
    auto-sharded path (a bare ParallelBeam model on a multi-device host).

Runs single-device or auto-sharded depending on the host's device count; the
setter-vs-fresh comparison is valid either way (both models share one config).
"""
import unittest

# Import mbirjax before jax (device-setup-first ordering).
import mbirjax

import numpy as np
import jax
import jax.numpy as jnp


def _assert_roundtrip_equal(testcase, restored, original, msg):
    """Same-executable restoration check, at tight float tolerance.

    Exact equality is never the right gate for COMPUTED floats (project rule):
    even one executable's run-to-run results can differ on GPU (scatter-add
    atomics reorder summation).  Tight closeness still proves the restoration --
    a stale or wrong value would miss by orders of magnitude."""
    np.testing.assert_allclose(restored, original, rtol=1e-6, atol=1e-6, err_msg=msg)


class TestSetViewParameters(unittest.TestCase):

    SINO_SHAPE = (8, 8, 32)

    def _angles(self, seed):
        rng = np.random.default_rng(seed)
        return jnp.asarray(np.sort(rng.uniform(0, np.pi, self.SINO_SHAPE[0])).astype(np.float32))

    def test_setter_matches_fresh_model_parallel(self):
        angles_a, angles_b = self._angles(0), self._angles(1)
        model = mbirjax.ParallelBeamModel(self.SINO_SHAPE, angles_a)
        fresh = mbirjax.ParallelBeamModel(self.SINO_SHAPE, angles_b)

        rng = np.random.default_rng(2)
        recon_shape = model.get_params('recon_shape')
        recon = jnp.asarray(rng.standard_normal(recon_shape, dtype=np.float32))
        sino = jnp.asarray(rng.standard_normal(self.SINO_SHAPE, dtype=np.float32))

        # Prime the jit caches at angles A (the values must not be baked into them).
        fwd_at_a = np.asarray(model.forward_project(recon))

        model.set_view_parameters(angles_b)
        # The stored parameter updated too (save/load + any later recompile see B).
        np.testing.assert_array_equal(np.asarray(model.get_params('angles')),
                                      np.asarray(angles_b))

        # Cross-compilation comparison: tight allclose (see the module docstring; a
        # stale baked value would fail at O(1), so this still proves the lift).
        fwd_set, fwd_fresh = model.forward_project(recon), fresh.forward_project(recon)
        np.testing.assert_allclose(np.asarray(fwd_set), np.asarray(fwd_fresh),
                                   rtol=1e-5, atol=1e-5,
                                   err_msg="forward projection after set_view_parameters != fresh model")
        back_set, back_fresh = model.back_project(sino), fresh.back_project(sino)
        np.testing.assert_allclose(np.asarray(back_set), np.asarray(back_fresh),
                                   rtol=1e-5, atol=1e-5,
                                   err_msg="back projection after set_view_parameters != fresh model")

        # Restoring A reproduces the original projection (same executable; tight
        # tolerance -- see _assert_roundtrip_equal).
        model.set_view_parameters(angles_a)
        fwd_back_at_a = np.asarray(model.forward_project(recon))
        _assert_roundtrip_equal(self, fwd_back_at_a, fwd_at_a,
                                "A -> B -> A round trip did not restore the original projection")

    def test_no_recompile_on_value_change(self):
        model = mbirjax.ParallelBeamModel(self.SINO_SHAPE, self._angles(3))
        rng = np.random.default_rng(4)
        recon = jnp.asarray(rng.standard_normal(model.get_params('recon_shape'),
                                                dtype=np.float32))
        _ = model.forward_project(recon)   # compile at the first values
        pf = model.projector_functions
        n_fwd = pf._jit_sparse_forward_project._cache_size()
        for seed in (5, 6, 7):
            model.set_view_parameters(self._angles(seed))
            _ = model.forward_project(recon)
        self.assertEqual(pf._jit_sparse_forward_project._cache_size(), n_fwd,
                         msg="set_view_parameters triggered a recompile")

    def test_view_count_change_raises(self):
        model = mbirjax.ParallelBeamModel(self.SINO_SHAPE, self._angles(8))
        with self.assertRaises(ValueError) as ctx:
            model.set_view_parameters(jnp.zeros(self.SINO_SHAPE[0] + 1))
        self.assertIn('set_params', str(ctx.exception))

    def test_setter_matches_fresh_model_cone(self):
        # Multi-component view params (angle, z-shift) on an unported (legacy
        # single-device) geometry: nothing baked depends on the values
        # (psf_radius / magnification / slice_range_length are distance-derived).
        angles_a, angles_b = self._angles(9), self._angles(10)
        kwargs = dict(source_detector_dist=4 * self.SINO_SHAPE[2],
                      source_iso_dist=2 * self.SINO_SHAPE[2])
        model = mbirjax.ConeBeamModel(self.SINO_SHAPE, angles_a, **kwargs)
        fresh = mbirjax.ConeBeamModel(self.SINO_SHAPE, angles_b, **kwargs)

        rng = np.random.default_rng(11)
        recon = jnp.asarray(rng.standard_normal(model.get_params('recon_shape'),
                                                dtype=np.float32))
        params_a = model.get_params('view_params_array')
        fwd_at_a = np.asarray(model.forward_project(recon))   # prime at A

        # Cone stacks (angle, z) into view_params_array; replace with fresh's array.
        # Cross-compilation comparison -> tight allclose (see the module docstring).
        model.set_view_parameters(fresh.get_params('view_params_array'))
        fwd_set, fwd_fresh = model.forward_project(recon), fresh.forward_project(recon)
        np.testing.assert_allclose(np.asarray(fwd_set), np.asarray(fwd_fresh),
                                   rtol=1e-5, atol=1e-5,
                                   err_msg="cone forward projection after set_view_parameters != fresh model")

        # Same-executable round trip restores the original projection.
        model.set_view_parameters(params_a)
        fwd_back_at_a = np.asarray(model.forward_project(recon))
        _assert_roundtrip_equal(self, fwd_back_at_a, fwd_at_a,
                                "cone A -> B -> A round trip did not restore the original projection")


if __name__ == "__main__":
    unittest.main()
