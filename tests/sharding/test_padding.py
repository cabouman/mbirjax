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

num_views=7 (prime) guarantees padding at every device count > 1; num_det_rows=8
keeps the slice axis divisible by 2/4/8 (slices must still divide -- slice
padding is a later step).

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


if __name__ == "__main__":
    unittest.main()
