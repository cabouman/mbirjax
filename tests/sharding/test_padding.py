"""
Tests for view-axis padding (P5 Step 4 Stage 1).

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
machinery.  SLICE padding (P5 Step 4 Stage 2) is tested separately below
(TestQggmrfInterfaceMask for the kernel; TestPaddedSlices end-to-end with a prime
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

from conftest import preferred_devices


NUM_VIEWS = 7          # prime: never divides a device count > 1
NUM_ROWS = 8           # -> num_slices = 8, divisible by 2/4/8
NUM_CHANNELS = 32


def _make_model():
    """Parallel-beam model with a PRIME view count (always padded when sharded >1).

    Pins a single device so the bare model is a deterministic single-device
    REFERENCE regardless of GPU count; sharded tests override with their own
    configure_sharding(devs).
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
        self.model.configure_sharding(self.devs)
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
        model.configure_sharding(devs)
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
        np.testing.assert_allclose(out_np[:NUM_VIEWS], ref, rtol=1e-5, atol=1e-5)
        # Default: plain output cropped to the real view count.
        out_plain = model.forward_project(recon)
        self.assertEqual(tuple(out_plain.shape), tuple(ref.shape))
        np.testing.assert_allclose(np.asarray(out_plain), ref, rtol=1e-5, atol=1e-5)

    def test_back_project_matches(self):
        ref_model = _make_model()
        sino = _random_sino(ref_model)
        ref = np.asarray(ref_model.back_project(sino))
        for n in (2, 4):
            model = self._sharded_model(n)
            if model is None:
                continue
            out = model.back_project(sino)
            np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-5, atol=1e-5,
                                       err_msg=f"padded back_project mismatch at n_dev={n}")

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
                np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-5, atol=1e-5,
                                           err_msg=f"padded {fn_name} mismatch at n_dev={n}")

    def test_hessian_diagonal_matches(self):
        """Constant-weights Hessian: the device-form ones must have ZERO padded views
        (a ones-padded tail would back-project spurious contributions)."""
        ref_model = _make_model()
        ref = np.asarray(ref_model.compute_hessian_diagonal())
        model = self._sharded_model()
        out = np.asarray(model.compute_hessian_diagonal())
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)

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
            m.configure_sharding(devs)
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

    MAX_ITERS = 6

    def _recon(self, model, sino, weights=None, prepared=False):
        np.random.seed(0)  # fix partitions + subset order so modes are comparable
        if model.mesh is not None:
            model._vcd_halo_per_subset = True   # exact prior path (the tight gate)
        if prepared:
            if weights is not None:
                sino, weights = model.prepare_sino_for_devices(sino, weights)
            else:
                sino = model.prepare_sino_for_devices(sino)
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
            model.configure_sharding(devs)
            out = self._recon(model, sino)
            np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4,
                                       err_msg=f"padded recon mismatch at n_dev={n}")
            ran_multi = True
        if not ran_multi:
            self.skipTest("no usable device count > 1")

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
        model.configure_sharding(devs)
        out = self._recon(model, sino, weights=weights)
        np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4,
                                   err_msg="padded non-const-weights recon mismatch")

    def test_prepared_input_recon_matches(self):
        """recon() accepts a prepare_sino_for_devices result (no silent gather, no
        re-pad) and matches the plain-input run on the same devices."""
        devs = preferred_devices(2)
        if devs is None:
            self.skipTest("need >= 2 devices")
        sino = _random_sino(_make_model())

        model_plain = _make_model()
        model_plain.configure_sharding(devs)
        ref = self._recon(model_plain, sino)

        model_prep = _make_model()
        model_prep.configure_sharding(devs)
        out = self._recon(model_prep, sino, prepared=True)
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5,
                                   err_msg="prepared-input recon diverged from plain-input recon")


class TestPaddedSlices(unittest.TestCase):
    """End-to-end SLICE padding (P5 Step 4 Stage 2): a prime slice count (7) pads at
    every device count > 1, and detector rows pad with the slices.  The results must
    be independent of the padding (match the single-device reference at the same
    tolerances as the dividing case) and the device-form padded slices must be
    EXACTLY zero (the forced-zero invariant -- established by entry zero-fill +
    the back-projector output mask + the qGGMRF interface mask, with no division
    guard anywhere: the padded VCD update is -0/positive = 0 by construction)."""

    NUM_VIEWS = 8
    NUM_ROWS = 7        # prime -> num_slices = 7 pads at every device count > 1
    NUM_CHANNELS = 32
    MAX_ITERS = 6

    def _make_model(self):
        angles = jnp.linspace(0, jnp.pi, self.NUM_VIEWS, endpoint=False)
        model = mbirjax.ParallelBeamModel(
            (self.NUM_VIEWS, self.NUM_ROWS, self.NUM_CHANNELS), angles)
        model.configure_devices(1)
        return model

    def _sino(self, seed=0):
        rng = np.random.default_rng(seed)
        return jnp.asarray(rng.standard_normal(
            (self.NUM_VIEWS, self.NUM_ROWS, self.NUM_CHANNELS), dtype=np.float32))

    def _sharded_models(self, max_n=4):
        for n in (2, 3, 4):
            if n > max_n:
                continue
            devs = preferred_devices(n)
            if devs is None:
                continue
            model = self._make_model()
            model.configure_sharding(devs)
            yield n, model

    def test_projectors_and_hessian_match_single_device(self):
        ref_model = self._make_model()
        sino = self._sino()
        ref_back = np.asarray(ref_model.back_project(sino))
        rng = np.random.default_rng(2)
        recon_shape = ref_model.get_params('recon_shape')
        recon = jnp.asarray(rng.standard_normal(recon_shape, dtype=np.float32))
        ref_fwd = np.asarray(ref_model.forward_project(recon))
        ref_hess = np.asarray(ref_model.compute_hessian_diagonal())
        ran = False
        for n, model in self._sharded_models():
            back = np.asarray(model.back_project(sino))
            np.testing.assert_allclose(back, ref_back, rtol=1e-5, atol=1e-5,
                                       err_msg=f"back_project mismatch at n_dev={n}")
            fwd = np.asarray(model.forward_project(recon))
            np.testing.assert_allclose(fwd, ref_fwd, rtol=1e-5, atol=1e-5,
                                       err_msg=f"forward_project mismatch at n_dev={n}")
            hess = np.asarray(model.compute_hessian_diagonal())
            np.testing.assert_allclose(hess, ref_hess, rtol=1e-5, atol=1e-5,
                                       err_msg=f"hessian mismatch at n_dev={n}")
            ran = True
        if not ran:
            self.skipTest("no usable device count > 1")

    def test_fbp_recon_matches_single_device(self):
        sino = self._sino(seed=3)
        ref = np.asarray(self._make_model().fbp_recon(sino))
        ran = False
        for n, model in self._sharded_models():
            out = np.asarray(model.fbp_recon(sino))
            np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5,
                                       err_msg=f"fbp_recon mismatch at n_dev={n}")
            ran = True
        if not ran:
            self.skipTest("no usable device count > 1")

    def _recon(self, model, sino, weights=None, seed=0):
        np.random.seed(seed)  # fix partitions + subset order so modes are comparable
        if model.mesh is not None:
            # Force the EXACT per-subset halo path: the default stages halos once per
            # partition pass, which is exact except at gen_pixel_partition's few replicated
            # pixels (a documented ~2e-3 approximation, tested separately in
            # test_vcd_sharded).  This test gates the PADDING machinery at 1e-4, so it
            # must not absorb that unrelated approximation.
            model._vcd_halo_per_subset = True
        recon, _ = model.recon(sino, weights=weights, max_iterations=self.MAX_ITERS,
                               stop_threshold_change_pct=0.0, print_logs=False)
        return np.asarray(recon)

    def test_vcd_recon_matches_single_device(self):
        """Padded-slice VCD (const and non-const weights) matches the single-device
        recon: the qGGMRF interface mask reproduces reflected BC at the last real
        slice, so the result is independent of the padding."""
        sino = self._sino(seed=4)
        rng = np.random.default_rng(5)
        weights = jnp.asarray(rng.uniform(
            0.5, 1.5, (self.NUM_VIEWS, self.NUM_ROWS, self.NUM_CHANNELS)).astype(np.float32))
        ref_const = self._recon(self._make_model(), sino)
        ref_wts = self._recon(self._make_model(), sino, weights=weights)
        self.assertTrue(np.all(np.isfinite(ref_const)))
        ran = False
        for n, model in self._sharded_models():
            out = self._recon(model, sino)
            self.assertTrue(np.all(np.isfinite(out)), msg=f"NaN/inf at n_dev={n}")
            np.testing.assert_allclose(out, ref_const, rtol=1e-4, atol=1e-4,
                                       err_msg=f"const-weights recon mismatch at n_dev={n}")
            model_w = next(m for k, m in self._sharded_models(max_n=n) if k == n)
            out_w = self._recon(model_w, sino, weights=weights)
            np.testing.assert_allclose(out_w, ref_wts, rtol=1e-4, atol=1e-4,
                                       err_msg=f"weighted recon mismatch at n_dev={n}")
            ran = True
        if not ran:
            self.skipTest("no usable device count > 1")

    def test_padded_slices_exactly_zero_in_device_form(self):
        """The forced-zero invariant, verified at the exits that expose the device
        form: padded slices of the back projection and of a full VCD recon are
        EXACTLY zero (no drift -- every update adds exact zeros there)."""
        sino = self._sino(seed=6)
        ran = False
        for n, model in self._sharded_models():
            back = model.back_project(sino, output_sharded=True)
            if back.shape[-1] == self.NUM_ROWS:
                continue   # this count happens not to pad; nothing to check
            back_np = np.asarray(back)
            self.assertTrue(np.all(back_np[..., self.NUM_ROWS:] == 0.0),
                            msg=f"back-projection padded slices not exactly zero at n_dev={n}")
            np.random.seed(0)
            recon, _ = model.recon(sino, max_iterations=3, stop_threshold_change_pct=0.0,
                                   print_logs=False, output_sharded=True)
            recon_np = np.asarray(recon)
            self.assertTrue(np.all(recon_np[..., self.NUM_ROWS:] == 0.0),
                            msg=f"recon padded slices not exactly zero at n_dev={n}")
            self.assertTrue(np.all(np.isfinite(recon_np)))
            ran = True
        if not ran:
            self.skipTest("no padded device count available")


class TestQggmrfInterfaceMask(unittest.TestCase):
    """Kernel-level slice-padding mask (P5 Step 4 Stage 2), with NO mesh.

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
        cylinder must equal the unmasked kernel on the truncated real cylinder
        (bit-exact on the real slices), with exactly-zero gradient and finite
        positive Hessian on the padded slices."""
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

        # Real slices: identical values, element for element (same elementwise op chain).
        self.assertTrue(np.array_equal(np.asarray(g_pad[:k]), np.asarray(g_ref)),
                        msg="real-slice gradient changed under padding+mask")
        self.assertTrue(np.array_equal(np.asarray(h_pad[:k]), np.asarray(h_ref)),
                        msg="real-slice Hessian changed under padding+mask")
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
        """An all-ones mask must be bit-identical to no mask (the uniform-trace form
        used for fully-real shards when the slice axis is padded)."""
        params = self._qggmrf_params()
        rng = np.random.default_rng(7)
        v = jnp.asarray(rng.standard_normal(6).astype(np.float32))
        ones = jnp.ones(7, dtype=jnp.float32)
        g0_, h0_ = mbirjax.qggmrf_grad_and_hessian_per_cylinder(v, params, v[0], v[-1])
        g1_, h1_ = mbirjax.qggmrf_grad_and_hessian_per_cylinder(v, params, v[0], v[-1],
                                                                interface_mask=ones)
        self.assertTrue(np.array_equal(np.asarray(g0_), np.asarray(g1_)))
        self.assertTrue(np.array_equal(np.asarray(h0_), np.asarray(h1_)))


if __name__ == "__main__":
    unittest.main()
