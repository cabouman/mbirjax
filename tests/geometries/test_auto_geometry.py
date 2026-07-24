import unittest
import numpy as np
import jax.numpy as jnp
import mbirjax as mj


class TestSupportRadius(unittest.TestCase):
    """Unit tests for mj.get_support_radius (vcd_utils): the maximum axis distance, to the
    voxel outer edge, of any pixel the projectors can update."""

    def test_ellipse_mask_uses_max_half_width(self):
        # Default RoR mask = the inscribed ellipse: farthest pixels sit at the ends of the
        # longer physical axis, so the radius is half the larger grid width.
        self.assertAlmostEqual(mj.get_support_radius((32, 32, 5), 0.5, 0.5), 8.0)
        # Rectangular grid / anisotropic pitches: rows 40 * 1.0 = 40 wide, cols 32 * 0.5 = 16.
        self.assertAlmostEqual(mj.get_support_radius((40, 32), 1.0, 0.5), 20.0)
        self.assertAlmostEqual(mj.get_support_radius((32, 40), 0.5, 1.0), 20.0)

    def test_no_mask_uses_half_diagonal(self):
        # With the mask disabled, corner pixels are updated, so the bound is the half-diagonal.
        self.assertAlmostEqual(mj.get_support_radius((32, 32), 0.5, 0.5, use_ror_mask=False),
                               float(np.hypot(8.0, 8.0)))

    def test_custom_mask_uses_half_diagonal(self):
        # A custom mask also gets the conservative half-diagonal (over-estimating only
        # over-pads; under-estimating would reintroduce truncation artifacts).
        mask = np.zeros((32, 32), dtype=np.int8)
        mask[10:20, 10:20] = 1
        self.assertAlmostEqual(mj.get_support_radius((32, 32), 0.5, 0.5, use_ror_mask=mask),
                               float(np.hypot(8.0, 8.0)))


class TestConeAxialExtension(unittest.TestCase):
    """The per-end axial extension in ConeBeamModel.auto_set_recon_geometry.

    Every test uses the same small geometry so the expectations can be hand-derived:
    8 views, 7 detector rows, 32 channels, unit detector pitch, sdd = 4*32 = 128,
    sid = 64 (magnification 2).  Then delta_voxel = 0.5, recon rows = cols = 32,
    support radius R = 0.5 * 32 * 0.5 = 8, and the far-side conversion from detector
    height v to iso z is 1/mag + R/sdd = 0.5 + 8/128 = 0.5625 (= (sid + R)/sdd).
    The detector maps to iso height H_iso = 7 * (1/2) = 3.5, so with no offset each
    detector edge (v = +/-3.5) reaches |z| = 3.5 * 0.5625 = 1.96875: an excess of
    0.21875 over the base half-slab 1.75, i.e. one extra 0.5-thick slice per end.
    """

    V, N, C = 8, 7, 32

    def _model(self, num_det_rows=None, sdd=None, **set_params):
        num_det_rows = self.N if num_det_rows is None else num_det_rows
        sdd = 4.0 * self.C if sdd is None else sdd
        angles = jnp.linspace(0, jnp.pi, self.V, endpoint=False)
        helical = set_params.pop('helical_z_shifts', None)
        model = mj.ConeBeamModel((self.V, num_det_rows, self.C), angles,
                                 source_detector_dist=sdd, source_iso_dist=sdd / 2.0,
                                 helical_z_shifts=helical)
        # Pin the fraction so these tests are independent of the package default.
        set_params.setdefault('axial_pad_fraction', 1.0)
        # verbose=0 keeps the padding printout (verbose >= 1) out of the test output;
        # the printout itself is pinned by test_pad_fraction_printout.
        model.set_params(verbose=0, **set_params)
        model.auto_set_recon_geometry()
        return model

    def _check(self, model, expected_slices, expected_offset):
        self.assertEqual(int(model.get_params('recon_shape')[2]), expected_slices)
        self.assertAlmostEqual(float(model.get_params('recon_slice_offset')), expected_offset,
                               places=6)

    def test_circular_symmetric(self):
        # Base 7 slices + 1 per end (excess 0.21875 each); symmetric, so no recentering.
        model = self._model()
        self._check(model, 7 + 2, 0.0)
        # Lateral shape is untouched by the axial extension.
        self.assertEqual(tuple(model.get_params('recon_shape')[:2]), (32, 32))

    def test_auto_geometry_is_idempotent(self):
        # Re-running auto_set_recon_geometry must recompute from scratch, not accumulate.
        model = self._model()
        first = (model.get_params('recon_shape'), float(model.get_params('recon_slice_offset')))
        model.auto_set_recon_geometry()
        second = (model.get_params('recon_shape'), float(model.get_params('recon_slice_offset')))
        self.assertEqual(first, second)

    def test_offset_detector_extends_per_end(self):
        # det_row_offset = -1.0 shifts the detector up one row (library convention:
        # v = (m - center) * delta_det_row - det_row_offset), so v_top = 4.5, v_bot = -2.5:
        # excess_top = 4.5 * 0.5625 - 1.75 = 0.78125 -> 2 slices; excess_bot < 0 -> 0 slices.
        # The slab recenters by (2 - 0)/2 slices = +0.5 ALU toward the extended end.
        model = self._model(det_row_offset=-1.0)
        self._check(model, 7 + 2 + 0, +0.5)

    def test_large_offset_clamps_unmeasured_end(self):
        # det_row_offset = -5.0 exceeds the half height (3.5): both edges are on the same
        # side (v_top = 8.5, v_bot = 1.5), so the near end gets NO extension (the clamp)
        # and the far end grows by ceil((8.5 * 0.5625 - 1.75)/0.5) = 7 slices.
        model = self._model(det_row_offset=-5.0)
        self._check(model, 7 + 7 + 0, +0.5 * 7 * 0.5)

    def test_helical_extends_at_travel_ends(self):
        # z travel of 2 ALU: base = ceil((3.5 + 2)/0.5) = 11 slices centered at mid-travel;
        # the per-end excess is the same as the circular case (the base slab already ends at
        # z_max + H_iso/2 and z_min - H_iso/2), so +1 slice per end and no recentering.
        model = self._model(helical_z_shifts=np.linspace(-1.0, 1.0, self.V))
        self._check(model, 11 + 2, 0.0)

    def test_inf_sdd_reduces_to_offset_compensation(self):
        # SDD = inf: rays are parallel in z (mag = 1, delta_voxel = 1), so with no offset
        # there is NO extension, and with an offset the extension is exactly the offset.
        model = self._model(sdd=jnp.inf)
        self._check(model, 7, 0.0)
        model = self._model(sdd=jnp.inf, det_row_offset=-1.0)
        self._check(model, 7 + 1, +0.5)

    def test_ror_mask_selects_support_radius(self):
        # A wider fan (sdd = 1.5*C, sid = 0.75*C) and a taller detector (15 rows) separate
        # the two radii: with the ellipse mask R = 8 gives 3 slices per end
        # (excess = 7.5 * (0.5 + 8/48) - 3.75 = 1.25); with use_ror_mask=False the
        # half-diagonal R = 8*sqrt(2) gives 4 (excess = 1.768).
        sdd = 1.5 * self.C
        model = self._model(num_det_rows=15, sdd=sdd)
        self._check(model, 15 + 6, 0.0)
        model = self._model(num_det_rows=15, sdd=sdd, use_ror_mask=False)
        self._check(model, 15 + 8, 0.0)

    def test_anisotropic_voxels(self):
        # voxel_slice_aspect = 2.9 coarsens the slice pitch to 1.45: base = ceil(3.5/1.45)
        # = 3 slices, and the same 0.21875 excess still ceils to 1 slice per end.
        model = self._model(voxel_slice_aspect=2.9)
        self._check(model, 3 + 2, 0.0)
        # voxel_row_aspect = 1.9 changes the recon row count (round(32/0.95/2) = 17) and
        # hence R = 0.5*max(17*0.95, 32*0.5) = 8.075: excess = 3.5*(0.5 + 8.075/128) - 1.75
        # = 0.2208 -> still 1 slice per end.
        model = self._model(voxel_row_aspect=1.9)
        self._check(model, 7 + 2, 0.0)

    def test_pad_fraction_zero_is_unpadded(self):
        # fraction 0 adds no padding: base 7 slices, and the offset is EXACTLY the helix
        # center (0 for circular) even with an offset detector, since with no added
        # slices the recentering is a no-op.
        self._check(self._model(axial_pad_fraction=0.0), 7, 0.0)
        model = self._model(axial_pad_fraction=0.0, det_row_offset=-1.0)
        self._check(model, 7, 0.0)
        self.assertEqual(float(model.get_params('recon_slice_offset')), 0.0)   # exact, not almost

    def test_pad_fraction_scales_and_is_monotone(self):
        # The fraction scales each end's excess BEFORE the ceil: at 0.5 the scaled excess
        # 0.109 still ceils to one slice (ceil stickiness near zero), at 3.0 it reaches
        # ceil(0.65625/0.5) = 2 per end.
        for fraction, expected in [(0.5, 9), (1.0, 9), (3.0, 11)]:
            self._check(self._model(axial_pad_fraction=fraction), expected, 0.0)

    def test_pad_fraction_tuple_sets_ends_independently(self):
        # A (top, bottom) pair acts per end: keeping only one end of the symmetric
        # 1-slice-per-end case gives 8 slices, recentered a half-slice (+/- 0.25 ALU)
        # toward the padded end.  Top = low slice indexes (z points down), so top-only
        # padding moves the offset NEGATIVE.  A list works the same (JSON round-trips
        # tuples to lists).
        self._check(self._model(axial_pad_fraction=(1.0, 0.0)), 8, -0.25)
        self._check(self._model(axial_pad_fraction=(0.0, 1.0)), 8, +0.25)
        self._check(self._model(axial_pad_fraction=[1.0, 0.0]), 8, -0.25)
        # Per-end counts compose additively: (1,0) and (0,1) together account for exactly
        # the full extension over the fraction-0 base.
        n = lambda apf: int(self._model(axial_pad_fraction=apf).get_params('recon_shape')[2])
        self.assertEqual(n((1.0, 0.0)) + n((0.0, 1.0)), n(1.0) + n(0.0))

    def test_pad_fraction_zero_helical(self):
        # fraction 0 on the helical case: just the base ceil((3.5 + 2)/0.5) = 11 slices,
        # centered on the (symmetric) travel.
        model = self._model(axial_pad_fraction=0.0,
                            helical_z_shifts=np.linspace(-1.0, 1.0, self.V))
        self._check(model, 11, 0.0)

    def test_pad_fraction_roundtrip_build_model(self):
        # The knob rides get_all_params -> build_model: it lands in optional_params, is
        # applied BEFORE the auto pass, and the rebuilt model reproduces the reduced
        # shape, the offset, and the knob itself.
        model = self._model(axial_pad_fraction=(1.0, 0.0))
        required, optional, regularization = model.get_all_params()
        self.assertEqual(tuple(optional['axial_pad_fraction']), (1.0, 0.0))
        rebuilt = mj.build_model(required, optional, regularization)
        self._check(rebuilt, 8, -0.25)
        self.assertEqual(tuple(rebuilt.get_params('axial_pad_fraction')), (1.0, 0.0))

    def test_pad_fraction_validation(self):
        # Negative fractions and wrong-length tuples fail loudly at the auto pass.
        for bad in (-0.1, (1.0, -0.1), (1.0, 1.0, 1.0)):
            with self.assertRaises(ValueError):
                self._model(axial_pad_fraction=bad)

    def test_pad_fraction_printout(self):
        # The padding line prints at verbose >= 1 on an explicit auto pass, not at verbose 0.
        import io, contextlib
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            model = self._model(axial_pad_fraction=1.0)    # helper runs at verbose 0: silent
        self.assertNotIn('Axial padding', buffer.getvalue())
        model.set_params(verbose=1)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            model.auto_set_recon_geometry()                # explicit call at verbose 1: prints
        self.assertIn('Axial padding: +1/+1 slices', buffer.getvalue())
        self.assertIn('axial_pad_fraction=1.0', buffer.getvalue())

    def test_helical_fdk_and_recon_finite_with_extension(self):
        # The padding slices at the travel ends have zero central-ray coverage in the helical
        # FDK z-weight (only diverging edge rays graze them).
        # Regression: when the padding first landed, that produced num_views/0 = inf in
        # helical_fdk_z_weight and NaN-poisoned the FDK initializer and every recon built on
        # it; the weight must zero the uncovered slices instead.
        model = self._model(helical_z_shifts=np.linspace(-1.0, 1.0, self.V))
        rng = np.random.default_rng(4)
        sino = jnp.asarray(rng.standard_normal(model.get_params('sinogram_shape'),
                                               dtype=np.float32))
        fdk = np.asarray(model.fdk_recon(sino))
        self.assertTrue(np.all(np.isfinite(fdk)))
        model.set_params(verbose=0)
        np.random.seed(0)
        recon, _ = model.recon(sino, max_iterations=2, stop_threshold_change_pct=0.0,
                               print_logs=False)
        self.assertTrue(np.all(np.isfinite(np.asarray(recon))))

    def test_beyond_bound_projects_to_exactly_zero(self):
        # Voxels beyond the illuminated extent intersect no measured rays,
        # so their forward projection is EXACTLY zero -- a constructed-zero invariant (no
        # contribution is ever generated), for which exact equality is the correct gate.
        # Enlarge the slice axis past the auto shape and fill only the added end slices.
        model = self._model()
        auto_shape = model.get_params('recon_shape')
        extra = 4
        model.set_params(recon_shape=(auto_shape[0], auto_shape[1], auto_shape[2] + 2 * extra))
        recon = np.zeros(model.get_params('recon_shape'), dtype=np.float32)
        recon[:, :, :extra] = 1.0
        recon[:, :, -extra:] = 1.0
        sino = np.asarray(model.forward_project(recon))
        self.assertEqual(float(np.abs(sino).max()), 0.0)

    def test_end_slices_are_reached(self):
        # Tightness of the ceil'd bound: the outermost slices of the AUTO shape must receive
        # nonzero back projection (they exist to absorb real measured rays, not as padding).
        model = self._model()
        back = np.asarray(model.back_project(np.ones((self.V, self.N, self.C),
                                                     dtype=np.float32)))
        self.assertGreater(float(np.abs(back[:, :, 0]).max()), 0.0)
        self.assertGreater(float(np.abs(back[:, :, -1]).max()), 0.0)


if __name__ == '__main__':
    unittest.main()
