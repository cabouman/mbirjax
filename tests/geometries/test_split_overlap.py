import unittest
import warnings

import numpy as np
import jax.numpy as jnp
import mbirjax as mj


class TestSplitOverlap(unittest.TestCase):
    """split_sino_recon's geometry-derived recon overlap, align_split_grid, and fallback.

    These gate the STRUCTURAL step-B behavior (overlap formula, alignment bookkeeping,
    feasibility fallback, no weight mutation) on tiny fast models; the mode-vs-mode value
    agreement with recon() is gated at production size in test_vcd.test_split_sino, and
    the real-data seam behavior is validated on the Lilly scans (plans/flash_remediation).
    """

    V, N, C = 16, 24, 32

    def _model(self, num_det_rows=None, **set_params):
        num_det_rows = self.N if num_det_rows is None else num_det_rows
        sdd = 4.0 * self.C
        angles = jnp.linspace(0, jnp.pi, self.V, endpoint=False)
        model = mj.ConeBeamModel((self.V, num_det_rows, self.C), angles,
                                 source_detector_dist=sdd, source_iso_dist=sdd / 2.0)
        if set_params:
            model.set_params(**set_params)
            model.auto_set_recon_geometry()
        model.set_params(verbose=0)
        return model

    def _sino(self, model, seed=3):
        rng = np.random.default_rng(seed)
        return rng.uniform(0.0, 1.0, tuple(int(x) for x in
                                           model.get_params('sinogram_shape'))).astype(np.float32)

    def _split(self, model, sino, **kwargs):
        np.random.seed(0)
        return model.split_sino_recon(sino, max_iterations=2, stop_threshold_change_pct=0.0,
                                      print_logs=False, **kwargs)

    def test_recon_overlap_formula(self):
        # Isotropic: R = 0.5*32*0.5 = 8, SID = 64 -> (1 + R/SID) = 1.125; rho = 1;
        # h_sino = 5 -> h_recon = ceil(5 * 1.125) + 2 = 8.
        model = self._model()
        recon, info = self._split(model, self._sino(model))
        sp = info['split_params']
        self.assertEqual(sp['half_overlap_sino'], 5)
        self.assertEqual(sp['half_overlap_recon'], 8)
        # The stitched output covers exactly the model's recon grid.
        self.assertEqual(recon.shape, tuple(int(x) for x in model.get_params('recon_shape')))

    def test_recon_overlap_formula_coarse_slices(self):
        # voxel_slice_aspect = 2.9 (slices coarser than iso-mapped rows): the SINO overlap
        # scales up to span ~half_overlap slices (round(5 * 2.9) = 14), and the recon overlap
        # comes back down through rho = 1/2.9: ceil(14 * 1.125 / 2.9) + 2 = 8.  A taller
        # detector (48 rows) keeps both halves thicker than the overlap (no fallback).
        model = self._model(num_det_rows=48, voxel_slice_aspect=2.9)
        recon, info = self._split(model, self._sino(model))
        sp = info['split_params']
        self.assertEqual(sp['half_overlap_sino'], 14)
        self.assertEqual(sp['half_overlap_recon'], 8)

    def test_align_split_grid(self):
        # A fractional det_row_offset misaligns the cut row and the split slice; without
        # alignment the mismatch is reported (about 0.45 slices here), with alignment the
        # residual is ~0 and the output grid shifted by at most half a slice.
        model = self._model(det_row_offset=0.45)
        sino = self._sino(model)
        _, info = self._split(model, sino)
        self.assertGreater(abs(info['split_params']['split_cut_mismatch_slices']), 0.3)
        self.assertEqual(info['split_params']['grid_shift_alu'], 0.0)

        recon, info = self._split(model, sino, align_split_grid=True)
        sp = info['split_params']
        self.assertLess(abs(sp['split_cut_mismatch_slices']), 1e-6)
        delta_voxel_slice = (model.get_params('voxel_slice_aspect')
                             * model.get_params('delta_voxel'))
        self.assertLessEqual(abs(sp['grid_shift_alu']), 0.5 * delta_voxel_slice + 1e-9)
        self.assertEqual(recon.shape, tuple(int(x) for x in model.get_params('recon_shape')))

    def test_fallback_when_halves_too_thin(self):
        # Fewer kept slices than the recon overlap on one side -> fall back to plain recon
        # (warned), returning the standard recon dict (no split_params).
        model = self._model()
        rs = model.get_params('recon_shape')
        model.set_params(recon_shape=(rs[0], rs[1], 12))   # halves ~6 < h_recon 8
        sino = self._sino(model)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            recon, info = self._split(model, sino)
        self.assertTrue(any('falling back' in str(c.message) for c in caught))
        self.assertNotIn('split_params', info)
        self.assertEqual(recon.shape, (int(rs[0]), int(rs[1]), 12))

    def test_caller_weights_not_mutated(self):
        # The halves now use host VIEWS of the caller's weights (the taper that used to
        # write into copies is retired) -- so nothing may mutate the caller's array.
        model = self._model()
        sino = self._sino(model)
        rng = np.random.default_rng(7)
        weights = rng.uniform(0.5, 1.5, sino.shape).astype(np.float32)
        weights_before = weights.copy()
        self._split(model, sino, weights=weights)
        np.testing.assert_array_equal(weights, weights_before)


if __name__ == '__main__':
    unittest.main()
