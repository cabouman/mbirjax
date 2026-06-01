"""
Tests for sharded back projection (ParallelBeamModel / TomographyModel).

Sharding scheme: the sinogram is sharded by **view** and the recon by **slice**.
Back projection sums each voxel's contribution over all views, and the views are
split across devices, so it is a *reduce-scatter*: each device back-projects its
own views onto the full voxel cylinders (Phase 1), then for each slice-owner the
per-device partials are summed over that owner's slice band (Phase 2).

  * Internal ``sparse_back_project`` is a sharded-contract method: given a
    view-sharded sinogram it returns a slice-sharded recon-at-indices (no gather).
  * User-facing ``back_project`` shards at entry and gathers at exit (plain out).

These tests check:
  - single-device (no-mesh) back projection is plain-in / plain-out;
  - trivial 1-device sharding is BIT-EXACT vs the single-device path (the
    prerelease regression gate);
  - 2-device sharding matches single-device to float noise, for both the public
    back_project and the coeff_power=2 (Hessian-diagonal) path;
  - the result scales correctly across device counts (2/4/8 where available) —
    the reduce-scatter is not hardcoded to two devices;
  - the internal sparse_back_project returns a slice-sharded array (no gather);
  - a view subset (view_indices != None) is rejected in sharded mode for now.

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


def _random_sino(model, seed=0):
    """A reproducible random sinogram of the model's sinogram shape.

    Back projection is linear, so a random sinogram is a valid input for
    cross-mode comparison; physical fidelity is irrelevant here.
    """
    shape = model.get_params('sinogram_shape')
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.standard_normal(shape, dtype=np.float32))


class TestBackProjectSingleDevice(unittest.TestCase):
    """No mesh configured: plain-in / plain-out, shape correct."""

    def test_runs_and_shape(self):
        model = _make_model()
        out = model.back_project(_random_sino(model))
        self.assertEqual(tuple(out.shape), tuple(model.get_params('recon_shape')))
        self.assertNotIsInstance(getattr(out, 'sharding', None),
                                 jax.sharding.NamedSharding)


class TestBackProjectSharded(unittest.TestCase):

    def setUp(self):
        self.devs = preferred_devices(2)
        if self.devs is None:
            self.skipTest("need >= 2 devices")

    @staticmethod
    def _divisible(model, n):
        """True if the model's sharded sinogram and recon axes both divide n.

        Reads the axis-declaration hooks (the single source of truth that
        configure_sharding itself uses) rather than hardcoding the view/slice
        axes, so a geometry that overrides the axes is checked consistently.
        """
        sino_shape = model.get_params('sinogram_shape')
        recon_shape = model.get_params('recon_shape')
        sino_axis = model.sinogram_shard_axis() % len(sino_shape)
        recon_axis = model.recon_shard_axis() % len(recon_shape)
        return sino_shape[sino_axis] % n == 0 and recon_shape[recon_axis] % n == 0

    def _check_divisible(self, model, n):
        if not self._divisible(model, n):
            self.skipTest(f"sharded axes not divisible by {n}")

    def test_trivial_sharding_bit_exact(self):
        """1-device mesh must be bit-exact to the unconfigured single-device path."""
        single_dev = preferred_devices(1)
        if single_dev is None:
            self.skipTest("need >= 1 device")
        model = _make_model()
        sino = _random_sino(model)
        ref = np.asarray(model.back_project(sino))

        shard_model = _make_model()
        shard_model.configure_sharding(single_dev)
        out = np.asarray(shard_model.back_project(sino))
        self.assertTrue(np.array_equal(out, ref),
                        msg=f"trivial sharding not bit-exact; max|diff|={np.max(np.abs(out-ref))}")

    def test_sharded_matches_single_device(self):
        """2-device public back_project matches single-device to float noise and
        returns a plain (gathered) array."""
        model = _make_model()
        self._check_divisible(model, 2)
        sino = _random_sino(model)
        ref = np.asarray(model.back_project(sino))

        shard_model = _make_model()
        shard_model.configure_sharding(self.devs)
        out = shard_model.back_project(sino)
        # User-facing: gathered (plain) output.
        self.assertNotIsInstance(getattr(out, 'sharding', None),
                                 jax.sharding.NamedSharding)
        np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-5, atol=1e-5)

    def test_internal_returns_slice_sharded(self):
        """sparse_back_project keeps the result slice-sharded (no gather inside)."""
        model = _make_model()
        self._check_divisible(model, 2)
        model.configure_sharding(self.devs)
        recon_shape = model.get_params('recon_shape')
        idx = mbirjax.gen_full_indices(recon_shape, use_ror_mask=model.use_ror_mask)
        sharded_in = model._shard_sinogram(_random_sino(model))
        out = model.sparse_back_project(sharded_in, idx)
        self.assertIsInstance(out.sharding, jax.sharding.NamedSharding)
        # The sharded axis is the one declared by recon_shard_axis() (the slice
        # axis), normalized to the rank of the (num_pixels, num_slices) result;
        # every other axis must be unsharded.
        slice_axis = model.recon_shard_axis() % out.ndim
        self.assertEqual(out.sharding.spec[slice_axis], 'devices')
        for ax in range(out.ndim):
            if ax != slice_axis:
                self.assertIsNone(out.sharding.spec[ax])

    def test_hessian_diagonal_sharded_matches(self):
        """coeff_power=2 (Hessian diagonal) matches single-device to float noise."""
        model = _make_model()
        self._check_divisible(model, 2)
        weights = _random_sino(model, seed=7)
        # Random weights are fine for a linear-operator cross-mode comparison.
        ref = np.asarray(model.compute_hessian_diagonal(weights))

        shard_model = _make_model()
        shard_model.configure_sharding(self.devs)
        out = np.asarray(shard_model.compute_hessian_diagonal(weights))
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)

    def test_view_indices_not_supported(self):
        """A view subset is rejected in sharded mode (full view-sharded input only)."""
        model = _make_model()
        self._check_divisible(model, 2)
        model.configure_sharding(self.devs)
        recon_shape = model.get_params('recon_shape')
        idx = mbirjax.gen_full_indices(recon_shape, use_ror_mask=model.use_ror_mask)
        sharded_in = model._shard_sinogram(_random_sino(model))
        num_views = model.get_params('sinogram_shape')[0]
        with self.assertRaises(NotImplementedError):
            model.sparse_back_project(sharded_in, idx, view_indices=jnp.arange(num_views))

    def test_device_count_sweep(self):
        """The reduce-scatter is correct across device counts, not just two."""
        ref_model = _make_model()
        sino = _random_sino(ref_model)
        ref = np.asarray(ref_model.back_project(sino))
        ran_multi = False
        for n in (2, 4, 8):
            devs = preferred_devices(n)
            if devs is None:
                continue
            model = _make_model()
            if not self._divisible(model, n):
                continue
            model.configure_sharding(devs)
            out = np.asarray(model.back_project(sino))
            np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5,
                                       err_msg=f"mismatch at n_dev={n}")
            ran_multi = True
        if not ran_multi:
            self.skipTest("no usable device count > 1")


if __name__ == "__main__":
    unittest.main()
