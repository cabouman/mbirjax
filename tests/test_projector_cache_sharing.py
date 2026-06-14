"""
tests/test_projector_cache_sharing.py
──────────────────────────────────────
B3 (de-closuring) gate: the projector drivers are MODULE-LEVEL jitted functions
(mbirjax.projectors._jit_sparse_forward_project / _jit_sparse_back_project), so two
DIFFERENT model instances with the same geometry SHARE one compiled program instead of
each re-tracing (tracing dominates the per-model first-call cost).

The test drives the Projectors object directly (model.projector_functions), which is the
single-device path that cone/translation/multiaxis use in production.  It checks that the
FIRST model's projection traces exactly one program into each module-level cache, and that
a SECOND fresh model with the same geometry adds NONE (it reuses the first's program) -- and
that the two instances agree.

Before B3 each instance owned its own jitted closure, so there was no shared module-level
cache to reuse; this test would not even find the module-level handles.

A deliberately unusual (prime) sinogram shape is used so the module caches are cold for it
at the start of the test, making the "+1 then +0" counts meaningful.
"""
import unittest

import numpy as np
import jax.numpy as jnp

import mbirjax
import mbirjax.projectors as projectors


class TestProjectorCacheSharing(unittest.TestCase):

    SINO_SHAPE = (13, 11, 17)   # prime, non-symmetric -> caches are cold for it

    def _make_model(self):
        angles = jnp.asarray(np.linspace(0, np.pi, self.SINO_SHAPE[0], endpoint=False))
        model = mbirjax.ParallelBeamModel(self.SINO_SHAPE, angles)
        model.set_params(verbose=0)
        return model

    def test_two_instances_share_compiled_programs(self):
        rng = np.random.default_rng(0)

        # First model + inputs.  Building the model does NOT call the projectors (no trace yet).
        m1 = self._make_model()
        recon_shape = tuple(int(x) for x in m1.get_params('recon_shape'))
        idx = jnp.asarray(mbirjax.gen_full_indices(recon_shape,
                                                   use_ror_mask=m1.get_params('use_ror_mask')))
        num_pixels = int(idx.shape[0])
        flat = jnp.asarray(rng.standard_normal((num_pixels, recon_shape[2]), dtype=np.float32))
        sino = jnp.asarray(rng.standard_normal(self.SINO_SHAPE, dtype=np.float32))

        fwd_before = projectors._jit_sparse_forward_project._cache_size()
        bak_before = projectors._jit_sparse_back_project._cache_size()

        proj1 = m1.projector_functions
        f1 = np.asarray(proj1.sparse_forward_project(flat, idx))
        b1 = np.asarray(proj1.sparse_back_project(sino, idx))

        fwd_after_1 = projectors._jit_sparse_forward_project._cache_size()
        bak_after_1 = projectors._jit_sparse_back_project._cache_size()
        # The module-level cache grew by exactly one program -> create_projectors really
        # routes through the shared module-level jit (not a per-instance closure).
        self.assertEqual(fwd_after_1, fwd_before + 1, msg="forward projector did not trace one shared program")
        self.assertEqual(bak_after_1, bak_before + 1, msg="back projector did not trace one shared program")

        # Second, independent model with the SAME geometry.
        m2 = self._make_model()
        proj2 = m2.projector_functions
        self.assertIsNot(proj1, proj2)
        f2 = np.asarray(proj2.sparse_forward_project(flat, idx))
        b2 = np.asarray(proj2.sparse_back_project(sino, idx))

        # It must REUSE the first model's compiled programs (no new cache entries).
        self.assertEqual(projectors._jit_sparse_forward_project._cache_size(), fwd_after_1,
                         msg="second model re-traced the forward projector (cache not shared)")
        self.assertEqual(projectors._jit_sparse_back_project._cache_size(), bak_after_1,
                         msg="second model re-traced the back projector (cache not shared)")

        # And the two instances agree (computed floats -> tight allclose).
        np.testing.assert_allclose(f1, f2, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(b1, b2, rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
    unittest.main()
