"""
Tests for view-axis padding.

When the view count does not divide the device count, the view axis is
zero-padded to the next multiple of the device count and the padding is kept
EXACTLY INERT:

  - entry placement (`_shard_sinogram` / `prepare_sino_for_devices`) zero-fills
    the padded tail shard-by-shard (no padded host copy is ever created);
  - the sharded forward projection zeroes its padded views post-assembly
    (`_mask_padded_views`), so padded views of every sinogram-domain array are
    identically zero, always;
  - padded view indices are clamped onto the last real view's parameters (their
    values are masked, so the angle never matters);
  - the FBP filter scale (pi/num_views) and the loss normalizations use the REAL
    view count from the params.

The result must be independent of the padding: every operation on a non-dividing
view count must match the single-device result to the same tolerances as the
dividing case, and padded entries must be exactly zero.

num_views=7 (prime) guarantees VIEW padding at every device count > 1;
num_det_rows=8 keeps the slice axis dividing for those tests, isolating the view
machinery.  SLICE padding is tested separately below (TestQggmrfInterfaceMask for
the kernel; TestPaddedSlicesParallel / TestPaddedSlicesCone end-to-end with a prime
slice count): forced-zero padded slices (entry zero-fill), the back-projector
output mask (_mask_padded_slices, the postcondition mirror of the forward view
mask), detector rows padding with the slices (parallel beam row r <-> slice r),
and the qGGMRF interface mask reproducing reflected BC at the last real slice.

Runs on whatever devices conftest provides (real GPUs on a cluster, virtual CPU
devices otherwise).
"""
import unittest

# Import mbirjax before jax (device-setup-first ordering).
import mbirjax

import numpy as np
import jax
import jax.numpy as jnp

from conftest import preferred_devices, assert_sharded_allclose


NUM_VIEWS = 7          # prime: never divides a device count > 1
NUM_ROWS = 8           # -> num_slices = 8, divisible by 2/4/8
NUM_CHANNELS = 32


def _make_model():
    """Parallel-beam model with a PRIME view count (always padded when sharded >1).

    Pins a single device so the bare model is a deterministic single-device
    REFERENCE regardless of GPU count; sharded tests override with their own
    configure_devices(devs).
    """
    angles = jnp.linspace(0, jnp.pi, NUM_VIEWS, endpoint=False)
    model = mbirjax.ParallelBeamModel((NUM_VIEWS, NUM_ROWS, NUM_CHANNELS), angles)
    model.configure_devices(1)
    return model


def _random_sino(model, seed=0):
    shape = model.get_params('sinogram_shape')
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape, dtype=np.float32)


def _random_recon(model, seed=0):
    recon_shape = model.get_params('recon_shape')
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.standard_normal(recon_shape, dtype=np.float32))


def _padded_views(n_views, n_dev):
    return ((n_views + n_dev - 1) // n_dev) * n_dev


class TestEntryPadding(unittest.TestCase):
    """The pad-aware entry placement: layout, zero tail, and the public helper."""

    def setUp(self):
        self.devs = preferred_devices(2)
        if self.devs is None:
            self.skipTest("need >= 2 devices")
        self.model = _make_model()
        self.model.configure_devices(self.devs)
        self.v_pad = _padded_views(NUM_VIEWS, 2)

    def test_pad_shard_layout_and_zero_tail(self):
        sino = _random_sino(self.model)
        sharded = self.model._shard_sinogram(sino)
        # Device form: padded view axis, view-sharded, equal blocks.
        self.assertEqual(sharded.shape, (self.v_pad, NUM_ROWS, NUM_CHANNELS))
        self.assertIsInstance(sharded.sharding, jax.sharding.NamedSharding)
        self.assertEqual(sharded.sharding.spec[0], 'devices')
        blocks = sorted(s.data.shape[0] for s in sharded.addressable_shards)
        self.assertEqual(blocks, [self.v_pad // 2] * 2)
        # Values: real views preserved exactly, padded tail exactly zero.
        gathered = np.asarray(sharded)
        np.testing.assert_array_equal(gathered[:NUM_VIEWS], sino)
        np.testing.assert_array_equal(gathered[NUM_VIEWS:], 0.0)

    def test_gather_crops_padding(self):
        sino = _random_sino(self.model)
        sharded = self.model._shard_sinogram(sino)
        back = self.model._gather_sinogram(sharded)
        self.assertEqual(tuple(back.shape), (NUM_VIEWS, NUM_ROWS, NUM_CHANNELS))
        np.testing.assert_array_equal(np.asarray(back), sino)

    def test_prepare_sino_for_devices(self):
        sino = _random_sino(self.model)
        weights = np.abs(_random_sino(self.model, seed=1)) + 0.5
        prepared_sino, prepared_weights = self.model.prepare_sino_for_devices(sino, weights)
        for prepared, source in ((prepared_sino, sino), (prepared_weights, weights)):
            self.assertEqual(prepared.shape[0], self.v_pad)
            gathered = np.asarray(prepared)
            np.testing.assert_array_equal(gathered[:NUM_VIEWS], source)
            np.testing.assert_array_equal(gathered[NUM_VIEWS:], 0.0)
        # A prepared array passes through the entry placement untouched (no movement).
        again = self.model._shard_sinogram(prepared_sino)
        self.assertIs(again, prepared_sino)
        # Without weights, the helper returns just the sinogram.
        only_sino = self.model.prepare_sino_for_devices(sino)
        self.assertEqual(only_sino.shape[0], self.v_pad)

    def test_wrong_view_count_raises_with_guidance(self):
        bad = np.zeros((NUM_VIEWS - 1, NUM_ROWS, NUM_CHANNELS), dtype=np.float32)
        with self.assertRaises(ValueError) as ctx:
            self.model._shard_sinogram(bad)
        self.assertIn("prepare_sino_for_devices", str(ctx.exception))

    def test_device_summary_mentions_padding(self):
        # device_summary is the public, read-only form of the resolved device report.
        self.assertIn('views padded {}->{}'.format(NUM_VIEWS, self.v_pad),
                      self.model.device_summary)


class TestPaddedProjectors(unittest.TestCase):
    """Forward/back/FBP on a padded view axis match the single-device reference,
    and the forward output's padded views are identically zero (the invariant)."""

    def setUp(self):
        self.devs = preferred_devices(2)
        if self.devs is None:
            self.skipTest("need >= 2 devices")

    def _sharded_model(self, n=2):
        devs = preferred_devices(n)
        if devs is None:
            return None
        model = _make_model()
        model.configure_devices(devs)
        return model

    def test_forward_masked_and_cropped(self):
        ref_model = _make_model()
        recon = _random_recon(ref_model)
        ref = np.asarray(ref_model.forward_project(recon))

        model = self._sharded_model()
        # Device form: padded view axis, padded views EXACTLY zero (the invariant).
        out_dev = model.forward_project(recon, output_sharded=True)
        self.assertEqual(out_dev.shape[0], _padded_views(NUM_VIEWS, 2))
        out_np = np.asarray(out_dev)
        np.testing.assert_array_equal(out_np[NUM_VIEWS:], 0.0)
        assert_sharded_allclose(out_np[:NUM_VIEWS], ref)
        # Default: plain output cropped to the real view count.
        out_plain = model.forward_project(recon)
        self.assertEqual(tuple(out_plain.shape), tuple(ref.shape))
        assert_sharded_allclose(np.asarray(out_plain), ref)

    def test_back_project_matches(self):
        ref_model = _make_model()
        sino = _random_sino(ref_model)
        ref = np.asarray(ref_model.back_project(sino))
        for n in (2, 4):
            model = self._sharded_model(n)
            if model is None:
                continue
            out = model.back_project(sino)
            assert_sharded_allclose(np.asarray(out), ref, msg=f"padded back_project mismatch at n_dev={n}")

    def test_fbp_recon_matches(self):
        """Covers the filter (pi over the REAL view count) + back projection chain."""
        ref_model = _make_model()
        sino = _random_sino(ref_model)
        ref = np.asarray(ref_model.fbp_recon(sino))
        for n in (2, 4):
            model = self._sharded_model(n)
            if model is None:
                continue
            for fn_name in ('fbp_recon', 'direct_recon'):
                out = getattr(model, fn_name)(sino)
                assert_sharded_allclose(np.asarray(out), ref, msg=f"padded {fn_name} mismatch at n_dev={n}")

    def test_hessian_diagonal_matches(self):
        """Constant-weights Hessian: the device-form ones must have ZERO padded views
        (a ones-padded tail would back-project spurious contributions)."""
        ref_model = _make_model()
        ref = np.asarray(ref_model.compute_hessian_diagonal())
        model = self._sharded_model()
        out = np.asarray(model.compute_hessian_diagonal())
        assert_sharded_allclose(out, ref)

    def test_adjoint_round_trip_padded(self):
        """<A x, y> == <x, A^T y> with padding: the mask keeps the forward/back pair
        exact adjoints on the REAL subspace."""
        ref_model = _make_model()
        recon_shape = ref_model.get_params('recon_shape')
        idx = mbirjax.gen_full_indices(recon_shape,
                                       use_ror_mask=ref_model.get_params('use_ror_mask'))
        rng = np.random.default_rng(3)
        x_cyl = jnp.asarray(rng.standard_normal((len(idx), recon_shape[2]), dtype=np.float32))
        y_sino = _random_sino(ref_model, seed=4)
        for n in (2, 4):
            devs = preferred_devices(n)
            if devs is None:
                continue
            m = _make_model()
            m.configure_devices(devs)
            ax = np.asarray(m.sparse_forward_project(m._shard_recon(x_cyl), idx))
            aty = np.asarray(m.sparse_back_project(m._shard_sinogram(y_sino), idx))
            # ax's padded views are zero, so summing against the real y over the
            # real range is the full inner product.
            lhs = float(np.sum(ax[:NUM_VIEWS] * y_sino))
            rhs = float(np.sum(np.asarray(x_cyl) * aty))
            np.testing.assert_allclose(lhs, rhs, rtol=1e-4,
                                       err_msg=f"padded adjoint mismatch at n_dev={n}")


class TestPaddedVcdRecon(unittest.TestCase):
    """End-to-end VCD recon on a prime view count: the padded sharded recon must
    match the single-device recon to the same tolerance as the dividing case."""

    # Mode-vs-mode comparison: discriminates from iteration 1, and fewer iterations
    # accumulate less FP-reorder divergence (gate-safening as well as faster).
    MAX_ITERS = 3

    def _recon(self, model, sino, weights=None, prepared=False):
        np.random.seed(0)  # fix partitions + subset order so modes are comparable
        if model.shard_devices is not None:
            model._vcd_halo_per_subset = True   # exact prior path (the tight gate)
        if prepared:
            if weights is not None:
                sino, weights = model.prepare_sino_for_devices(sino, weights)
            else:
                sino = model.prepare_sino_for_devices(sino)
        model.set_params(verbose=0)  # Silence warnings about background
        recon, _ = model.recon(sino, weights=weights, max_iterations=self.MAX_ITERS,
                               stop_threshold_change_pct=0.0, print_logs=False)
        return np.asarray(recon)

    def test_recon_matches_single_device(self):
        sino = _random_sino(_make_model())
        ref = self._recon(_make_model(), sino)
        ran_multi = False
        for n in (2, 4):
            devs = preferred_devices(n)
            if devs is None:
                continue
            model = _make_model()
            model.configure_devices(devs)
            out = self._recon(model, sino)
            assert_sharded_allclose(out, ref, msg=f"padded recon mismatch at n_dev={n}", tol=1e-4)
            ran_multi = True
        if not ran_multi:
            self.skipTest("no usable device count > 1")

    def test_fm_rmse_independent_of_padding(self):
        """The REPORTED forward-model loss (fm_rmse) must match the single-device value on a
        padded run with DEFAULT weights.

        Regression test for the scalar-weights normalization bug: vcd_recon represents default
        weights as the Python scalar 1, which fell into get_forward_model_loss's padded
        weights-ARRAY branch, where jnp.sum(scalar) is the scalar (not element_count * scalar)
        -- making avg_weight ~ 1/num_real_elements and inflating the reported loss by
        ~sqrt(num_real_elements) whenever padding was active.  The recon values were
        unaffected, so only a loss assertion catches a regression here."""
        devs = preferred_devices(2)
        if devs is None:
            self.skipTest("need >= 2 devices")
        sino = _random_sino(_make_model())

        def fm_rmse(model):
            np.random.seed(0)   # fix partitions + subset order so the runs are comparable
            if model.shard_devices is not None:
                model._vcd_halo_per_subset = True
            model.set_params(verbose=0)
            _, recon_dict = model.recon(sino, max_iterations=self.MAX_ITERS,
                                        stop_threshold_change_pct=0.0, print_logs=False)
            return np.asarray(recon_dict['recon_params']['fm_rmse'])

        ref = fm_rmse(_make_model())          # single device: 7 views, unpadded
        model = _make_model()
        model.configure_devices(devs)         # 7 views on 2+ devices: view axis padded
        out = fm_rmse(model)
        # Same per-iteration values up to float reordering; the bug was a ~20x blowup at this
        # sinogram size (sqrt of 7*8*32 elements), so the tolerance discriminates decisively.
        np.testing.assert_allclose(out, ref, rtol=1e-3,
                                   err_msg="padded-run fm_rmse diverged from single-device")

    def test_recon_matches_nonconst_weights(self):
        """Non-constant weights: the zero-padded weights tail must keep the padded
        views out of the weighted error, the line search, and the Hessian."""
        m = _make_model()
        sino = _random_sino(m)
        rng = np.random.default_rng(7)
        weights = rng.uniform(0.5, 1.5, m.get_params('sinogram_shape')).astype(np.float32)
        ref = self._recon(_make_model(), sino, weights=weights)
        devs = preferred_devices(2)
        if devs is None:
            self.skipTest("need >= 2 devices")
        model = _make_model()
        model.configure_devices(devs)
        out = self._recon(model, sino, weights=weights)
        assert_sharded_allclose(out, ref, msg="padded non-const-weights recon mismatch", tol=1e-4)

    def test_prepared_input_recon_matches(self):
        """recon() accepts a prepare_sino_for_devices result (no silent gather, no
        re-pad) and matches the plain-input run on the same devices."""
        devs = preferred_devices(2)
        if devs is None:
            self.skipTest("need >= 2 devices")
        sino = _random_sino(_make_model())

        model_plain = _make_model()
        model_plain.configure_devices(devs)
        ref = self._recon(model_plain, sino)

        model_prep = _make_model()
        model_prep.configure_devices(devs)
        out = self._recon(model_prep, sino, prepared=True)
        assert_sharded_allclose(out, ref, msg="prepared-input recon diverged from plain-input recon")


class _PaddedReconMixin:
    """Shared end-to-end SLICE-padding checks for any geometry whose recon shards by slice.

    A prime slice count pads at every device count > 1; results must be independent of the
    padding (sharded == single-device at the dividing-case tolerances), the device-form
    padded entries must be EXACTLY zero (the forced-zero invariant -- entry zero-fill + the
    back-projector output mask + the qGGMRF interface mask, with no division guard: the
    padded VCD update is -0/positive = 0 by construction), and the forward must be inert even
    to NONZERO values in the recon padding.

    All real sizes are read from params, so the checks are geometry-agnostic: the recon slice
    count equals the detector rows for parallel and circular cone but NOT for helical cone.
    Subclasses set VARIANTS + PADS_ROWS and implement _make_model(variant)."""

    VARIANTS = ()          # model descriptors handed to _make_model (e.g. a helical flag)
    PADS_ROWS = False      # True when detector rows pad with the slices (parallel: row r <-> slice r)
    NUM_CHANNELS = 32
    # Run the sharded VCD-recon checks (the iterated-loop paths).  The sharded VCD LOOP is
    # geometry-independent, so it is gated once per beam family (parallel + cone); a geometry that
    # only adds projectors/filters (translation, multiaxis) sets this False -- its projector-level
    # padding is still gated by the single-shot checks below.  See the NOTE in the translation
    # subclass / test_translation_sharded.py.
    RUN_SHARDED_VCD = True
    # Mode-vs-mode comparison: discriminates from iteration 1, and fewer iterations
    # accumulate less FP-reorder divergence (gate-safening as well as faster).
    MAX_ITERS = 3
    PROJ_TOL = 1e-5        # single-shot projector / Hessian tolerance
    RECON_TOL = 1e-4       # iterated VCD tolerance (per-step FP-reorder accumulates)

    # ---- subclass hooks ----
    def _make_model(self, variant):
        raise NotImplementedError

    def _label(self, variant):
        return f"variant={variant}"

    # ---- shared helpers ----
    def _sino(self, model, seed=0):
        rng = np.random.default_rng(seed)
        return jnp.asarray(rng.standard_normal(
            model.get_params('sinogram_shape'), dtype=np.float32))

    def _recon_array(self, model, seed=2):
        rs = tuple(int(x) for x in model.get_params('recon_shape'))
        rng = np.random.default_rng(seed)
        return jnp.asarray(rng.standard_normal(rs, dtype=np.float32))

    def _sharded_models(self, variant):
        """Fresh sharded models at every available device count > 1 (each PADS the slice
        axis, since the geometry's slice count is prime)."""
        for n in (2, 3, 4):
            devs = preferred_devices(n)
            if devs is None:
                continue
            model = self._make_model(variant)
            model.configure_devices(devs)
            yield n, model

    def _recon(self, model, sino, weights=None, seed=0):
        np.random.seed(seed)  # fix partitions + subset order so modes are comparable
        if model.shard_devices is not None:
            # Force the EXACT per-subset halo path: the default stages halos once per
            # partition pass, which is exact except at gen_pixel_partition's few replicated
            # pixels (a documented ~2e-3 approximation, tested separately in
            # test_vcd_sharded).  This test gates the PADDING machinery at RECON_TOL, so it
            # must not absorb that unrelated approximation.
            model._vcd_halo_per_subset = True
        model.set_params(verbose=0)  # Silence warnings about background
        recon, _ = model.recon(sino, weights=weights, max_iterations=self.MAX_ITERS,
                               stop_threshold_change_pct=0.0, print_logs=False)
        return np.asarray(recon)

    # ---- shared tests ----
    def test_projectors_and_hessian_match_single_device(self):
        """back / forward / Hessian-diagonal at a padded slice count match the
        single-device reference (the padding is inert)."""
        for variant in self.VARIANTS:
            with self.subTest(self._label(variant)):
                ref_model = self._make_model(variant)
                sino = self._sino(ref_model)
                recon = self._recon_array(ref_model)
                ref_back = np.asarray(ref_model.back_project(sino))
                ref_fwd = np.asarray(ref_model.forward_project(recon))
                ref_hess = np.asarray(ref_model.compute_hessian_diagonal())
                ran = False
                for n, model in self._sharded_models(variant):
                    assert_sharded_allclose(np.asarray(model.back_project(sino)), ref_back, msg=f"back mismatch {self._label(variant)} n_dev={n}", tol=self.PROJ_TOL)
                    assert_sharded_allclose(np.asarray(model.forward_project(recon)), ref_fwd, msg=f"forward mismatch {self._label(variant)} n_dev={n}", tol=self.PROJ_TOL)
                    assert_sharded_allclose(np.asarray(model.compute_hessian_diagonal()), ref_hess, msg=f"hessian mismatch {self._label(variant)} n_dev={n}", tol=self.PROJ_TOL)
                    ran = True
                if not ran:
                    self.skipTest("no usable device count > 1")

    def test_vcd_recon_matches_single_device(self):
        """A short VCD recon (const and non-const weights) at a padded slice count matches
        the single-device recon: projectors + the qGGMRF interface mask are all
        padding-correct, so the result is independent of the padding."""
        if not self.RUN_SHARDED_VCD:
            self.skipTest("sharded VCD loop is geometry-independent (gated on parallel + cone)")
        for variant in self.VARIANTS:
            with self.subTest(self._label(variant)):
                ref_model = self._make_model(variant)
                sino = self._sino(ref_model, seed=4)
                rng = np.random.default_rng(5)
                weights = jnp.asarray(rng.uniform(
                    0.5, 1.5, ref_model.get_params('sinogram_shape')).astype(np.float32))
                ref_const = self._recon(self._make_model(variant), sino)
                ref_wts = self._recon(self._make_model(variant), sino, weights=weights)
                self.assertTrue(np.all(np.isfinite(ref_const)))
                ran = False
                for n, model in self._sharded_models(variant):
                    out = self._recon(model, sino)
                    self.assertTrue(np.all(np.isfinite(out)),
                                    msg=f"NaN/inf {self._label(variant)} n_dev={n}")
                    assert_sharded_allclose(out, ref_const, msg=f"const recon mismatch {self._label(variant)} n_dev={n}", tol=self.RECON_TOL)
                    model_w = self._make_model(variant)
                    model_w.configure_devices(preferred_devices(n))
                    out_w = self._recon(model_w, sino, weights=weights)
                    assert_sharded_allclose(out_w, ref_wts, msg=f"weighted recon mismatch {self._label(variant)} n_dev={n}", tol=self.RECON_TOL)
                    ran = True
                if not ran:
                    self.skipTest("no usable device count > 1")

    def test_padded_entries_exactly_zero_in_device_form(self):
        """Forced-zero invariant at the device-form exits, BOTH directions: padded SLICES of
        the back projection and of a VCD recon are exactly zero; padded VIEWS (and, for
        geometries that pad rows with slices, padded detector ROWS) of the forward are exactly
        zero.  Real sizes from params (helical cone's slice count != detector rows)."""
        ran = False
        for variant in self.VARIANTS:
            for n, model in self._sharded_models(variant):
                real_views, real_rows = (int(s) for s in model.get_params('sinogram_shape')[:2])
                real_slices = int(model.get_params('recon_shape')[2])
                sino = self._sino(model, seed=6)
                recon = self._recon_array(model, seed=7)
                # BACK device form: padded slices (last axis) exactly zero.
                back = np.asarray(model.back_project(sino, output_sharded=True))
                if back.shape[-1] != real_slices:
                    self.assertTrue(np.all(back[..., real_slices:] == 0.0),
                                    msg=f"back padded slices not zero {self._label(variant)} n_dev={n}")
                # FORWARD device form: padded views exactly zero; detector rows either pad with
                # the slices (parallel) and are zero there, or stay real (cone).
                fwd = np.asarray(model.forward_project(recon, output_sharded=True))
                if fwd.shape[0] != real_views:
                    self.assertTrue(np.all(fwd[real_views:] == 0.0),
                                    msg=f"forward padded views not zero {self._label(variant)} n_dev={n}")
                if self.PADS_ROWS:
                    if fwd.shape[1] != real_rows:
                        self.assertTrue(np.all(fwd[:, real_rows:] == 0.0),
                                        msg=f"forward padded rows not zero {self._label(variant)} n_dev={n}")
                else:
                    self.assertEqual(fwd.shape[1], real_rows,
                                     msg=f"{self._label(variant)}: forward must keep real detector rows")
                # RECON device form: padded slices stay exactly zero through the VCD loop.  The
                # iterated-loop padding inertness is geometry-independent (gated on parallel +
                # cone); skip it where RUN_SHARDED_VCD is off -- the back/forward forced-zero above
                # is the geometry-specific part and always runs.
                if self.RUN_SHARDED_VCD:
                    np.random.seed(0)
                    model.set_params(verbose=0)  # Silence warnings about background
                    rec = np.asarray(model.recon(sino, max_iterations=self.MAX_ITERS,
                                                 stop_threshold_change_pct=0.0, print_logs=False,
                                                 output_sharded=True)[0])
                    if rec.shape[-1] != real_slices:
                        self.assertTrue(np.all(rec[..., real_slices:] == 0.0),
                                        msg=f"recon padded slices not zero {self._label(variant)} n_dev={n}")
                    self.assertTrue(np.all(np.isfinite(rec)))
                ran = True
        if not ran:
            self.skipTest("no padded device count available")

    def test_forward_inert_to_nonzero_recon_padding(self):
        """Inertness tested DIRECTLY: the forward result depends ONLY on the real slices, not
        on the VALUES in the recon's padded slices (the stronger claim than 'the padding stays
        zero').  Forward-project a device-form recon twice on the SAME sharded model -- once
        zero-filled, once with the padded slices poisoned with a large constant -- and require
        the results to MATCH.  The poison reaches only cropped-away entries: the base gather-forward
        crops the padded slices before the kernel; parallel projects them solely to padded detector
        rows, which the gather drops.  On CPU this is bit-exact even for a 1e6 poison, so the padding
        truly is inert; but forward_project is run-to-run NONDETERMINISTIC on GPU (scatter-add atomics
        reorder summation between two separate calls), so the two results differ by ~1 ULP regardless
        of the poison.  So gate on a scale-invariant peak-relative max: a real leak is ~O(1) relative
        (the 1e3 poison), while the GPU run-to-run noise is ~1e-6."""
        for variant in self.VARIANTS:
            with self.subTest(self._label(variant)):
                model0 = self._make_model(variant)
                real_recon = self._recon_array(model0, seed=9)
                real_slices = int(model0.get_params('recon_shape')[2])
                ran = False
                for n, model in self._sharded_models(variant):
                    padded = model._shard_recon(real_recon)     # device form, zero-filled tail
                    if padded.shape[-1] == real_slices:
                        continue                                # this count did not pad
                    clean = np.asarray(model.forward_project(padded))
                    poisoned = padded.at[..., real_slices:].set(1.0e3)
                    out = np.asarray(model.forward_project(poisoned))
                    # Inline scale-invariant gate (peak-relative max): dependency-free, so this is
                    # byte-identical on conebeam_sharding (no conftest.assert_sharded_allclose) and
                    # sharding_extensions -> the rebase/merge into prerelease does not conflict here.
                    denom = float(np.max(np.abs(clean))) or 1.0
                    rel = float(np.max(np.abs(out - clean)) / denom)
                    self.assertLess(rel, 1e-3, msg=f"forward not inert to padded values "
                                    f"{self._label(variant)} n_dev={n} (rel={rel:.2e})")
                    ran = True
                if not ran:
                    self.skipTest("no padded device count available")


class TestPaddedSlicesParallel(_PaddedReconMixin, unittest.TestCase):
    """Parallel beam: recon slices == detector rows, so the sinogram's row axis pads with the
    slices (row r <-> slice r).  A prime row/slice count (7) pads at every device count > 1."""

    VARIANTS = ('parallel',)
    PADS_ROWS = True
    NUM_VIEWS = 8
    NUM_ROWS = 7           # prime -> num_slices = 7 pads at every device count > 1

    def _make_model(self, variant='parallel'):
        angles = jnp.linspace(0, jnp.pi, self.NUM_VIEWS, endpoint=False)
        model = mbirjax.ParallelBeamModel(
            (self.NUM_VIEWS, self.NUM_ROWS, self.NUM_CHANNELS), angles)
        model.configure_devices(1)
        return model

    def _label(self, variant):
        return "parallel"

    def test_fbp_recon_matches_single_device(self):
        """Parallel-only direct recon (filter + adjoint back projection) under slice padding."""
        sino = self._sino(self._make_model(), seed=3)
        ref = np.asarray(self._make_model().fbp_recon(sino))
        ran = False
        for n, model in self._sharded_models('parallel'):
            assert_sharded_allclose(np.asarray(model.fbp_recon(sino)), ref, msg=f"fbp_recon mismatch at n_dev={n}", tol=self.PROJ_TOL)
            ran = True
        if not ran:
            self.skipTest("no usable device count > 1")


class TestPaddedSlicesCone(_PaddedReconMixin, unittest.TestCase):
    """Cone beam: detector rows are independent of slices (no row padding); the sharded
    forward GATHERS + CROPS the device-form cylinder before the monolithic kernel (which
    anchors its slice->detector-row geometry on the REAL slice count).  Variants: circular
    and helical (helical's z-range gives a slice count != detector rows, so the real slice
    count is read from params, not assumed to be num_det_rows).  num_slices=7 is prime, so
    every device count > 1 pads."""

    VARIANTS = (False, True)   # circular, helical
    PADS_ROWS = False
    NUM_VIEWS = 8
    NUM_DET_ROWS = 7           # isotropic cone -> num_slices = 7 (prime: pads at every count > 1)

    def _make_model(self, helical=False, curved=False):
        angles = jnp.linspace(0, jnp.pi, self.NUM_VIEWS, endpoint=False)
        sdd = 4.0 * self.NUM_CHANNELS
        kwargs = dict(source_detector_dist=sdd, source_iso_dist=sdd / 2.0,
                      use_curved_detector=curved)
        if helical:
            kwargs['helical_z_shifts'] = np.linspace(-1.0, 1.0, self.NUM_VIEWS)
        model = mbirjax.ConeBeamModel(
            (self.NUM_VIEWS, self.NUM_DET_ROWS, self.NUM_CHANNELS), angles, **kwargs)
        model.configure_devices(1)   # deterministic single-device reference; sharded tests override
        return model

    def _label(self, helical):
        return "helical" if helical else "circular"

    def test_curved_detector_projectors_padding(self):
        """A few fast projector checks with use_curved_detector=True under slice padding.
        Detector curvature lives in the channel axis, so the slice<->row map (and hence the
        slice cropping/banding) must be curvature-agnostic; this guards that combination,
        which the other tests (flat detector) do not reach."""
        ref_model = self._make_model(curved=True)
        sino = self._sino(ref_model)
        recon = self._recon_array(ref_model)
        ref_back = np.asarray(ref_model.back_project(sino))
        ref_fwd = np.asarray(ref_model.forward_project(recon))
        ran = False
        for n in (2, 3, 4):
            devs = preferred_devices(n)
            if devs is None:
                continue
            model = self._make_model(curved=True)
            model.configure_devices(devs)
            assert_sharded_allclose(np.asarray(model.back_project(sino)), ref_back, msg=f"curved back mismatch n_dev={n}", tol=self.PROJ_TOL)
            assert_sharded_allclose(np.asarray(model.forward_project(recon)), ref_fwd, msg=f"curved forward mismatch n_dev={n}", tol=self.PROJ_TOL)
            ran = True
        if not ran:
            self.skipTest("no usable device count > 1")

    def test_fully_padded_trailing_shard(self):
        """A tiny 3-slice cone on 4 devices makes the LAST shard entirely padding (n_valid
        == 0): exercises the _mask_padded_slices / _mask_padded_views n_valid<=0 branch and
        a gather that concatenates a fully-zero shard -- which auto-config normally avoids
        (it skips a count whose last shard is all padding), so only an explicit configure
        reaches it.  Projector-level, so it stays fast."""
        devs = preferred_devices(4)
        if devs is None:
            self.skipTest("need 4 devices")
        angles = jnp.linspace(0, jnp.pi, self.NUM_VIEWS, endpoint=False)
        sdd = 4.0 * self.NUM_CHANNELS

        def tiny():
            m = mbirjax.ConeBeamModel((self.NUM_VIEWS, 3, self.NUM_CHANNELS), angles,
                                      source_detector_dist=sdd, source_iso_dist=sdd / 2.0)
            m.configure_devices(1)
            return m

        ref_model = tiny()
        # 3 slices over 4 devices -> shards of 1, so the last shard is entirely padding.
        self.assertEqual(int(ref_model.get_params('recon_shape')[2]), 3)
        sino = self._sino(ref_model)
        recon = self._recon_array(ref_model)
        ref_back = np.asarray(ref_model.back_project(sino))
        ref_fwd = np.asarray(ref_model.forward_project(recon))
        model = tiny()
        model.configure_devices(devs)
        assert_sharded_allclose(np.asarray(model.back_project(sino)), ref_back, msg="fully-padded-shard back mismatch", tol=self.PROJ_TOL)
        assert_sharded_allclose(np.asarray(model.forward_project(recon)), ref_fwd, msg="fully-padded-shard forward mismatch", tol=self.PROJ_TOL)
        back_dev = np.asarray(model.back_project(sino, output_sharded=True))
        self.assertTrue(np.all(back_dev[..., 3:] == 0.0),
                        msg="fully-padded trailing shard not exactly zero")


class TestPaddedSlicesTranslation(_PaddedReconMixin, unittest.TestCase):
    """Translation: like cone, the recon slice count is independent of the detector rows (it is
    auto-sized from the z-translation extent), so the sharded forward GATHERS + CROPS the
    device-form cylinder before the monolithic kernel and the detector rows do NOT pad with the
    slices (PADS_ROWS=False).  Variants: isotropic and anisotropic (voxel_slice_aspect=2.9, the
    suite's anisotropic_translation aspect -> a slice count that is NOT num_det_rows).  Both
    z-ranges are tuned to a prime slice count (7), so every device count > 1 pads.

    RUN_SHARDED_VCD=False: the sharded VCD loop is geometry-independent (gated on parallel + cone);
    translation's projector-level padding is gated by the single-shot checks (projectors/Hessian,
    forward/back device-form exact-zero, forward-inert).  This also keeps the tiny translation
    recon out of the suite, so no partition-granularity warning."""

    VARIANTS = ('isotropic', 'anisotropic')
    RUN_SHARDED_VCD = False
    PADS_ROWS = False
    NUM_VIEWS = 8
    NUM_DET_ROWS = 32
    # z-translation half-range per variant, tuned so auto_set_recon_geometry lands on 7 slices
    # (prime).  recon_shape is computed deterministically from params (no GPU reordering), so
    # these are platform-stable; the surrounding plateau is wide enough that the exact value is
    # not knife-edge.
    _ZRANGE = {'isotropic': 1.65, 'anisotropic': 4.6}

    def _make_model(self, variant='isotropic'):
        nv, ndr, ndc = self.NUM_VIEWS, self.NUM_DET_ROWS, self.NUM_CHANNELS
        zr = self._ZRANGE[variant]
        tv = np.zeros((nv, 3))
        tv[:, 0] = np.linspace(-8.0, 8.0, nv)        # x translations (set the recon row extent)
        tv[:, 2] = np.linspace(-zr, zr, nv)          # z translations (set the slice extent)
        sdd = 4.0 * ndc
        model = mbirjax.TranslationModel((nv, ndr, ndc), jnp.asarray(tv),
                                         source_detector_dist=sdd, source_iso_dist=sdd / 2.0)
        if variant == 'anisotropic':
            model.set_params(voxel_row_aspect=1.9)
            model.set_params(voxel_slice_aspect=2.9)
            model.auto_set_recon_geometry()
        model.configure_devices(1)   # deterministic single-device reference; sharded tests override
        return model

    def _label(self, variant):
        return variant

    def test_prime_slice_count(self):
        """Guard the tuned geometry: both variants must auto-size to the prime slice count the
        padding coverage relies on (a drift in auto_set_recon_geometry would silently stop
        exercising the padded path)."""
        for variant in self.VARIANTS:
            with self.subTest(variant=variant):
                self.assertEqual(int(self._make_model(variant).get_params('recon_shape')[2]), 7)

    def test_fully_padded_trailing_shard(self):
        """A tiny 3-slice translation on 4 devices makes the LAST shard entirely padding
        (n_valid == 0): exercises the _mask_padded_* n_valid<=0 branch + a gather that
        concatenates a fully-zero shard (which auto-config avoids by skipping such a count, so
        only an explicit configure reaches it).  Projector-level, so it stays fast."""
        devs = preferred_devices(4)
        if devs is None:
            self.skipTest("need 4 devices")
        nv, ndr, ndc = self.NUM_VIEWS, self.NUM_DET_ROWS, self.NUM_CHANNELS
        sdd = 4.0 * ndc

        def tiny():
            tv = np.zeros((nv, 3))
            tv[:, 0] = np.linspace(-8.0, 8.0, nv)
            tv[:, 2] = np.linspace(-0.65, 0.65, nv)   # short z-range -> 3 slices
            m = mbirjax.TranslationModel((nv, ndr, ndc), jnp.asarray(tv),
                                         source_detector_dist=sdd, source_iso_dist=sdd / 2.0)
            m.configure_devices(1)
            return m

        ref_model = tiny()
        # 3 slices over 4 devices -> shards of 1, so the last shard is entirely padding.
        self.assertEqual(int(ref_model.get_params('recon_shape')[2]), 3)
        sino = self._sino(ref_model)
        recon = self._recon_array(ref_model)
        ref_back = np.asarray(ref_model.back_project(sino))
        ref_fwd = np.asarray(ref_model.forward_project(recon))
        model = tiny()
        model.configure_devices(devs)
        assert_sharded_allclose(np.asarray(model.back_project(sino)), ref_back, msg="fully-padded-shard back mismatch", tol=self.PROJ_TOL)
        assert_sharded_allclose(np.asarray(model.forward_project(recon)), ref_fwd, msg="fully-padded-shard forward mismatch", tol=self.PROJ_TOL)
        back_dev = np.asarray(model.back_project(sino, output_sharded=True))
        self.assertTrue(np.all(back_dev[..., 3:] == 0.0),
                        msg="fully-padded trailing shard not exactly zero")


class TestPaddedSlicesMultiAxis(_PaddedReconMixin, unittest.TestCase):
    """Multiaxis parallel beam: with a nonzero elevation a slice maps to a RANGE of detector
    rows, so detector rows do NOT pad with slices (PADS_ROWS=False, like cone/translation).  The
    geometry is tuned (per-variant num_det_rows) so auto_set_recon_geometry lands on a prime
    7-slice recon, which pads at every device count > 1.  RUN_SHARDED_VCD=False: the sharded VCD
    loop is geometry-independent and gated by parallel + cone (see test_multiaxis_sharded.py); the
    single-shot projector/forward-inertness checks are the geometry-specific padding gates."""

    VARIANTS = ('isotropic', 'anisotropic')
    RUN_SHARDED_VCD = False
    PADS_ROWS = False
    NUM_VIEWS = 8
    # Per-variant num_det_rows tuned so auto_set_recon_geometry lands on 7 (prime) slices: the
    # isotropic slice count tracks the detector rows, while the anisotropic slice pitch (2.9)
    # needs a proportionally taller detector.  recon_shape is computed deterministically from
    # params, so these are platform-stable; guarded by test_prime_slice_count.
    _NDR = {'isotropic': 7, 'anisotropic': 22}

    @staticmethod
    def _angles(nv):
        az = np.linspace(0.0, np.pi, nv, endpoint=False)
        el = np.deg2rad(np.linspace(-5.0, 5.0, nv))   # modest tilt: slices spread across rows
        return jnp.asarray(np.stack([az, el], axis=1))

    def _make_model(self, variant='isotropic'):
        nv, ndr, ndc = self.NUM_VIEWS, self._NDR[variant], self.NUM_CHANNELS
        model = mbirjax.MultiAxisParallelModel((nv, ndr, ndc), self._angles(nv))
        if variant == 'anisotropic':
            model.set_params(voxel_row_aspect=1.9, voxel_slice_aspect=2.9)
        model.auto_set_recon_geometry()
        model.configure_devices(1)   # deterministic single-device reference; sharded tests override
        return model

    def _label(self, variant):
        return variant

    def test_prime_slice_count(self):
        """Guard the tuned geometry: both variants must auto-size to the prime 7-slice count the
        padding coverage relies on (a drift in auto_set_recon_geometry would silently stop
        exercising the padded path)."""
        for variant in self.VARIANTS:
            with self.subTest(variant=variant):
                self.assertEqual(int(self._make_model(variant).get_params('recon_shape')[2]), 7)

    def test_fully_padded_trailing_shard(self):
        """A tiny 3-slice multiaxis on 4 devices makes the LAST shard entirely padding
        (n_valid == 0): exercises the _mask_padded_* n_valid<=0 branch + a gather that
        concatenates a fully-zero shard (which auto-config avoids by skipping such a count, so
        only an explicit configure reaches it).  Projector-level, so it stays fast."""
        devs = preferred_devices(4)
        if devs is None:
            self.skipTest("need 4 devices")
        nv, ndc = self.NUM_VIEWS, self.NUM_CHANNELS

        def tiny():
            m = mbirjax.MultiAxisParallelModel((nv, 3, ndc), self._angles(nv))  # 3 det rows -> 3 slices
            m.auto_set_recon_geometry()
            m.configure_devices(1)
            return m

        ref_model = tiny()
        # 3 slices over 4 devices -> shards of 1, so the last shard is entirely padding.
        self.assertEqual(int(ref_model.get_params('recon_shape')[2]), 3)
        sino = self._sino(ref_model)
        recon = self._recon_array(ref_model)
        ref_back = np.asarray(ref_model.back_project(sino))
        ref_fwd = np.asarray(ref_model.forward_project(recon))
        model = tiny()
        model.configure_devices(devs)
        assert_sharded_allclose(np.asarray(model.back_project(sino)), ref_back, msg="fully-padded-shard back mismatch", tol=self.PROJ_TOL)
        assert_sharded_allclose(np.asarray(model.forward_project(recon)), ref_fwd, msg="fully-padded-shard forward mismatch", tol=self.PROJ_TOL)
        back_dev = np.asarray(model.back_project(sino, output_sharded=True))
        self.assertTrue(np.all(back_dev[..., 3:] == 0.0),
                        msg="fully-padded trailing shard not exactly zero")


class TestQggmrfInterfaceMask(unittest.TestCase):
    """Kernel-level slice-padding mask, with NO mesh.

    The qGGMRF inter-slice term builds delta[j] = difference across the interface
    between local slices j-1 and j (j = 0..L are the L+1 interfaces of an L-slice
    cylinder, including both boundary interfaces).  Reflected BC at a true edge is
    implemented as a ZERO boundary delta, so a multiplicative interface mask IS the
    reflected boundary condition relocated to an arbitrary interface: masking every
    interface whose higher-index global slice is padded (g0 + j < num_real_slices)
    reproduces reflected BC at the last REAL slice -- even mid-shard -- makes the
    padded slices' gradient exactly zero, and leaves their Hessian positive (the
    b_tilde(0) terms), so the VCD denominator never forms 0/0.
    """

    def _qggmrf_params(self):
        model = _make_model()
        qggmrf_nbr_wts, sigma_x, p, q, T = model.get_params(
            ['qggmrf_nbr_wts', 'sigma_x', 'p', 'q', 'T'])
        b = mbirjax.get_b_from_nbr_wts(qggmrf_nbr_wts)
        return (b, sigma_x, p, q, T)

    def test_masked_padded_cylinder_matches_truncated_real(self):
        """One cylinder, boundary MID-shard: the masked kernel on the zero-padded
        cylinder must equal the unmasked kernel on the truncated real cylinder on
        the real slices (mathematically identical op chains; compared at float
        noise -- the two shapes compile separate executables, and exact equality
        is never the gate for computed floats), with exactly-zero gradient and
        finite positive Hessian on the padded slices."""
        params = self._qggmrf_params()
        rng = np.random.default_rng(3)
        L, k = 10, 6                       # local slices, real slices (pad = 4, mid-shard boundary)
        v_real = jnp.asarray(rng.standard_normal(k).astype(np.float32))
        v_pad = jnp.concatenate([v_real, jnp.zeros(L - k, dtype=jnp.float32)])
        # Single shard at g0 = 0: valid iff the interface's higher-index slice is real.
        mask = jnp.asarray((np.arange(L + 1) < k).astype(np.float32))

        # Reference: the unpadded cylinder with reflected BC at both true edges.
        g_ref, h_ref = mbirjax.qggmrf_grad_and_hessian_per_cylinder(
            v_real, params, v_real[0], v_real[-1])
        # Masked padded cylinder (right_val is the reflected padded tail; masked anyway).
        g_pad, h_pad = mbirjax.qggmrf_grad_and_hessian_per_cylinder(
            v_pad, params, v_pad[0], v_pad[-1], interface_mask=mask)

        # Real slices: the same per-element computation, gated at float noise.
        np.testing.assert_allclose(np.asarray(g_pad[:k]), np.asarray(g_ref),
                                   rtol=1e-6, atol=1e-6,
                                   err_msg="real-slice gradient changed under padding+mask")
        np.testing.assert_allclose(np.asarray(h_pad[:k]), np.asarray(h_ref),
                                   rtol=1e-6, atol=1e-6,
                                   err_msg="real-slice Hessian changed under padding+mask")
        # Padded slices: gradient exactly zero; Hessian finite and strictly positive
        # (b_tilde(0) terms -- this is what keeps the VCD division well-posed with no guard).
        self.assertTrue(np.all(np.asarray(g_pad[k:]) == 0.0),
                        msg="padded-slice gradient not exactly zero")
        h_tail = np.asarray(h_pad[k:])
        self.assertTrue(np.all(np.isfinite(h_tail)) and np.all(h_tail > 0.0),
                        msg="padded-slice Hessian not finite-positive")

    def test_masked_shards_with_halos_match_unpadded_reference(self):
        """Two shards with halos, boundary mid-LAST-shard: per-shard masked results
        must reproduce the full unpadded reference on the real slices (the
        at_indices level -- cylinder term + in-slice term + halos + mask together)."""
        params = self._qggmrf_params()
        num_rows = num_cols = 4
        P = num_rows * num_cols
        L, S_real = 5, 8                   # 2 shards x 5 slices = 10 padded, boundary at shard1 local 3
        rng = np.random.default_rng(5)
        flat_real = jnp.asarray(rng.standard_normal((P, S_real), dtype=np.float32))
        flat_pad = jnp.concatenate([flat_real, jnp.zeros((P, 2 * L - S_real), dtype=jnp.float32)], axis=1)
        idx = jnp.arange(P)

        # Full unpadded reference (reflected BC at both true edges).
        g_full, h_full = mbirjax.qggmrf_gradient_and_hessian_at_indices(
            flat_real, (num_rows, num_cols, S_real), idx, params)

        shard0, shard1 = flat_pad[:, :L], flat_pad[:, L:]
        # The one predicate: interface j of a shard starting at g0 is valid iff g0 + j < S_real.
        mask0 = jnp.asarray(((0 + np.arange(L + 1)) < S_real).astype(np.float32))  # all-ones (fully real)
        mask1 = jnp.asarray(((L + np.arange(L + 1)) < S_real).astype(np.float32))

        # Shard 0: true left edge, interior right boundary (halo = first slice of shard 1, real).
        g0_, h0_ = mbirjax.qggmrf_gradient_and_hessian_at_indices(
            shard0, (num_rows, num_cols, L), idx, params,
            left_halo=None, right_halo=flat_pad[:, L], interface_mask=mask0)
        # Shard 1: interior left boundary (halo = last slice of shard 0), true right edge.
        g1_, h1_ = mbirjax.qggmrf_gradient_and_hessian_at_indices(
            shard1, (num_rows, num_cols, L), idx, params,
            left_halo=flat_pad[:, L - 1], right_halo=None, interface_mask=mask1)

        k1 = S_real - L                    # real slices local to shard 1
        np.testing.assert_allclose(np.asarray(g0_), np.asarray(g_full[:, :L]), rtol=1e-6, atol=1e-6,
                                   err_msg="shard 0 gradient diverged from unpadded reference")
        np.testing.assert_allclose(np.asarray(h0_), np.asarray(h_full[:, :L]), rtol=1e-6, atol=1e-6,
                                   err_msg="shard 0 Hessian diverged from unpadded reference")
        np.testing.assert_allclose(np.asarray(g1_[:, :k1]), np.asarray(g_full[:, L:]), rtol=1e-6, atol=1e-6,
                                   err_msg="shard 1 real-slice gradient diverged from unpadded reference")
        np.testing.assert_allclose(np.asarray(h1_[:, :k1]), np.asarray(h_full[:, L:]), rtol=1e-6, atol=1e-6,
                                   err_msg="shard 1 real-slice Hessian diverged from unpadded reference")
        # Padded columns: gradient exactly zero, Hessian finite-positive.
        self.assertTrue(np.all(np.asarray(g1_[:, k1:]) == 0.0))
        h_tail = np.asarray(h1_[:, k1:])
        self.assertTrue(np.all(np.isfinite(h_tail)) and np.all(h_tail > 0.0))

    def test_all_ones_mask_is_identity(self):
        """An all-ones mask must equal no mask at float noise (the uniform-trace
        form used for fully-real shards when the slice axis is padded; multiply by
        1.0 is mathematically the identity, but mask/no-mask compile separate
        executables, and exact equality is never the gate for computed floats)."""
        params = self._qggmrf_params()
        rng = np.random.default_rng(7)
        v = jnp.asarray(rng.standard_normal(6).astype(np.float32))
        ones = jnp.ones(7, dtype=jnp.float32)
        g0_, h0_ = mbirjax.qggmrf_grad_and_hessian_per_cylinder(v, params, v[0], v[-1])
        g1_, h1_ = mbirjax.qggmrf_grad_and_hessian_per_cylinder(v, params, v[0], v[-1],
                                                                interface_mask=ones)
        np.testing.assert_allclose(np.asarray(g0_), np.asarray(g1_), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(np.asarray(h0_), np.asarray(h1_), rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
