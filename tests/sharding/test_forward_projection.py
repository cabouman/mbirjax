"""
Tests for sharded forward projection (ParallelBeamModel / TomographyModel).

Forward projection is the all-gather **adjoint** of back projection's
reduce-scatter: the recon is slice-sharded, each slice-band is broadcast to every
view-owner (``broadcast_band_to_views`` -- the adjoint of ``sum_band_to_owner``),
and each view-owner forward-projects its own views from the band (no reduce, since
each detector row has a single producer).  The headline correctness gate is the
forward/back **adjoint round-trip** ``<A x, y> == <x, A^T y>`` against the
validated band back projector.

These tests check:
  - single-device (no-mesh) forward projection is plain-in / plain-out;
  - trivial 1-device sharding is BIT-EXACT vs the single-device path;
  - 2/4/8-device sharding matches single-device to float noise;
  - match-input: a sharded recon returns a view-sharded sinogram (no gather);
  - sparse_forward_project keeps the result view-sharded (no gather inside);
  - the forward/back adjoint identity holds across device counts;
  - a view subset (view_indices != None) is rejected in sharded mode.

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


def _make_model(num_views=8, num_rows=8, num_channels=32):
    """Small parallel-beam model; num_views and num_rows divisible by 2/4/8."""
    angles = jnp.linspace(0, jnp.pi, num_views, endpoint=False)
    return mbirjax.ParallelBeamModel((num_views, num_rows, num_channels), angles)


def _indices(model):
    recon_shape = model.get_params('recon_shape')
    return mbirjax.gen_full_indices(recon_shape, use_ror_mask=model.get_params('use_ror_mask'))


def _random_recon(model, seed=0):
    """A reproducible random 3-D recon volume of the model's recon shape."""
    recon_shape = model.get_params('recon_shape')
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.standard_normal(recon_shape, dtype=np.float32))


def _random_cylinders(model, seed=0):
    """Random slice cylinders (num_pixels, num_slices) over the model's FOV.

    Forward projection is linear, so random cylinders are a valid input for a
    cross-mode comparison; physical fidelity is irrelevant here.
    """
    num_pixels = len(_indices(model))
    num_slices = model.get_params('recon_shape')[2]
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.standard_normal((num_pixels, num_slices), dtype=np.float32))


def _random_sino(model, seed=0):
    shape = model.get_params('sinogram_shape')
    return jnp.asarray(np.random.default_rng(seed).standard_normal(shape, dtype=np.float32))


class TestForwardProjectSingleDevice(unittest.TestCase):
    """No mesh configured: plain-in / plain-out, shape correct."""

    def test_runs_and_shape(self):
        model = _make_model()
        out = model.forward_project(_random_recon(model))
        self.assertEqual(tuple(out.shape), tuple(model.get_params('sinogram_shape')))
        self.assertNotIsInstance(getattr(out, 'sharding', None),
                                 jax.sharding.NamedSharding)


class TestForwardProjectSharded(unittest.TestCase):

    def setUp(self):
        self.devs = preferred_devices(2)
        if self.devs is None:
            self.skipTest("need >= 2 devices")

    @staticmethod
    def _divisible(model, n):
        """True if the model's sharded sinogram and recon axes both divide n."""
        sino_shape = model.get_params('sinogram_shape')
        recon_shape = model.get_params('recon_shape')
        sino_axis = model.sinogram_shard_axis() % len(sino_shape)
        recon_axis = model.recon_shard_axis() % len(recon_shape)
        return sino_shape[sino_axis] % n == 0 and recon_shape[recon_axis] % n == 0

    def _check_divisible(self, model, n):
        if not self._divisible(model, n):
            self.skipTest(f"sharded axes not divisible by {n}")

    def test_trivial_sharding_bit_exact(self):
        """1-device mesh matches the unconfigured single-device path to tight float
        tolerance.  (Name kept for history; was an exact-equality check.)

        RETIRE-AFTER-SHARDING: trivial-mesh-vs-legacy comparison, meaningful only
        while both paths coexist; once every geometry runs on placements there is
        one path and nothing to compare.  Relaxed from exact equality because the
        banded sharded path reorders non-associative FP sums -> ~1 ULP GPU
        difference (CPU compiles both identically and stays exact).  A tight
        tolerance still trips on any real algorithmic drift.
        """
        single_dev = preferred_devices(1)
        if single_dev is None:
            self.skipTest("need >= 1 device")
        model = _make_model()
        recon = _random_recon(model)
        ref = np.asarray(model.forward_project(recon))

        shard_model = _make_model()
        shard_model.configure_sharding(single_dev)
        out = np.asarray(shard_model.forward_project(recon))
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5,
                                   err_msg="trivial sharding diverged beyond float noise")

    def test_sharded_matches_single_device(self):
        """2-device public forward_project matches single-device to float noise and
        returns a plain (gathered) array."""
        model = _make_model()
        self._check_divisible(model, 2)
        recon = _random_recon(model)
        ref = np.asarray(model.forward_project(recon))

        shard_model = _make_model()
        shard_model.configure_sharding(self.devs)
        out = shard_model.forward_project(recon)
        # User-facing, PLAIN input: gathered (plain) output.
        self.assertNotIsInstance(getattr(out, 'sharding', None),
                                 jax.sharding.NamedSharding)
        np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-5, atol=1e-5)

    def test_sharded_input_returns_sharded_sino(self):
        """Match-input contract: forward_project of a SHARDED recon returns a
        view-sharded sinogram (no gather) matching the plain single-device sinogram."""
        model = _make_model()
        self._check_divisible(model, 2)
        recon = _random_recon(model)
        ref = np.asarray(model.forward_project(recon))     # single-device plain

        shard_model = _make_model()
        shard_model.configure_sharding(self.devs)
        sharded_recon = shard_model._shard_recon(recon)
        out = shard_model.forward_project(sharded_recon)

        # Sharded in -> sharded out: a view-sharded sinogram.
        self.assertIsInstance(out.sharding, jax.sharding.NamedSharding)
        self.assertEqual(tuple(out.shape),
                         tuple(shard_model.get_params('sinogram_shape')))
        view_axis = shard_model.sinogram_shard_axis() % out.ndim
        self.assertEqual(out.sharding.spec[view_axis], 'devices')
        for ax in range(out.ndim):
            if ax != view_axis:
                self.assertIsNone(out.sharding.spec[ax])
        np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-5, atol=1e-5)

    def test_internal_returns_view_sharded(self):
        """sparse_forward_project keeps the result view-sharded (no gather inside)."""
        model = _make_model()
        self._check_divisible(model, 2)
        model.configure_sharding(self.devs)
        idx = _indices(model)
        cyl = model._shard_recon(_random_cylinders(model))
        out = model.sparse_forward_project(cyl, idx)
        self.assertIsInstance(out.sharding, jax.sharding.NamedSharding)
        view_axis = model.sinogram_shard_axis() % out.ndim
        self.assertEqual(out.sharding.spec[view_axis], 'devices')
        for ax in range(out.ndim):
            if ax != view_axis:
                self.assertIsNone(out.sharding.spec[ax])

    def test_view_indices_not_supported(self):
        """A view subset is rejected in sharded mode (full view-sharded output only)."""
        model = _make_model()
        self._check_divisible(model, 2)
        model.configure_sharding(self.devs)
        idx = _indices(model)
        cyl = model._shard_recon(_random_cylinders(model))
        num_views = model.get_params('sinogram_shape')[0]
        with self.assertRaises(NotImplementedError):
            model.sparse_forward_project(cyl, idx, view_indices=jnp.arange(num_views))

    def test_device_count_sweep(self):
        """The all-gather is correct across device counts, not just two."""
        ref_model = _make_model()
        recon = _random_recon(ref_model)
        ref = np.asarray(ref_model.forward_project(recon))
        ran_multi = False
        for n in (2, 4, 8):
            devs = preferred_devices(n)
            if devs is None or not self._divisible(ref_model, n):
                continue
            m = _make_model()
            m.configure_sharding(devs)
            out = np.asarray(m.forward_project(recon))
            np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)
            ran_multi = True
        if not ran_multi:
            self.skipTest("no usable multi-device configuration")

    def test_adjoint_round_trip(self):
        """The P3 gate: <A x, y> == <x, A^T y> for the sharded forward/back pair.

        x is random slice cylinders, y a random sinogram.  A = forward (all-gather),
        A^T = back (reduce-scatter).  The inner-product identity must hold to float
        precision at every device count -- that is what certifies the two sharded
        projectors are exact adjoints.
        """
        ref_model = _make_model()
        x_cyl = _random_cylinders(ref_model, seed=1)
        y_sino = _random_sino(ref_model, seed=2)
        idx = _indices(ref_model)
        for n in (1, 2, 4, 8):
            devs = preferred_devices(n)
            if devs is None or (n > 1 and not self._divisible(ref_model, n)):
                continue
            m = _make_model()
            m.configure_sharding(devs)
            ax = np.asarray(m.sparse_forward_project(m._shard_recon(x_cyl), idx))
            aty = np.asarray(m.sparse_back_project(m._shard_sinogram(y_sino), idx))
            lhs = float(np.sum(ax * np.asarray(y_sino)))
            rhs = float(np.sum(np.asarray(x_cyl) * aty))
            np.testing.assert_allclose(lhs, rhs, rtol=1e-4,
                                       err_msg=f"adjoint mismatch at n={n}")


if __name__ == '__main__':
    unittest.main()
