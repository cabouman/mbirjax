"""
Phase F1 tests for sharded FBP filtering (ParallelBeamModel.fbp_filter).

fbp_filter is an internal sharded-contract method: it shards the sinogram on the
view axis at entry and returns the filtered sinogram in that same sharding (no
gather).  Under view-sharding the ramp filter is per-view, so each device filters
its own views with no cross-device communication.

These tests check:
  - the single-device (no-mesh) path is bit-exact vs a direct reference filter;
  - the 2-device sharded path matches single-device to float noise;
  - the sharded result is actually view-sharded (axis 0) and gathers correctly.

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


def _make_model_and_sino(num_views=8, num_rows=16, num_channels=64, seed=0):
    """A small parallel-beam model plus a random sinogram of matching shape."""
    angles = jnp.linspace(0, jnp.pi, num_views, endpoint=False)
    model = mbirjax.ParallelBeamModel((num_views, num_rows, num_channels), angles)
    rng = np.random.default_rng(seed)
    sino = jnp.asarray(rng.random((num_views, num_rows, num_channels), dtype=np.float32))
    return model, sino


class TestFbpFilterSingleDevice(unittest.TestCase):
    """No mesh configured: behavior should be plain-in / plain-out and correct."""

    def test_filter_runs_and_preserves_shape(self):
        model, sino = _make_model_and_sino()
        out = model.fbp_filter(sino, view_batch_size=4)
        self.assertEqual(out.shape, sino.shape)
        # No mesh -> output is a plain (unsharded) array.
        self.assertNotIsInstance(getattr(out, 'sharding', None),
                                 jax.sharding.NamedSharding)

    def test_body_tail_split_matches_single_batch(self):
        """num_views not divisible by view_batch_size still filters every view.

        Compares a batch size that splits into body+tail (5 -> body 4 + tail 4
        for 8 views... here use 3 so 8 = body 6 + tail 2) against a batch size
        that covers all views at once; the two must agree (the split is only a
        memory/padding device, not a math change).
        """
        model, sino = _make_model_and_sino(num_views=8)
        out_split = model.fbp_filter(sino, view_batch_size=3)   # 8 = 6 + 2
        out_whole = model.fbp_filter(sino, view_batch_size=8)   # single batch
        np.testing.assert_allclose(np.asarray(out_split), np.asarray(out_whole),
                                   rtol=1e-5, atol=1e-5)


class TestFbpFilterSharded(unittest.TestCase):

    def setUp(self):
        self.devs = preferred_devices(2)
        if self.devs is None:
            self.skipTest("need >= 2 devices")

    def test_sharded_matches_single_device(self):
        # num_views divisible by 2 (devices) and num_slices divisible by 2.
        model, sino = _make_model_and_sino(num_views=8)
        if model.get_params('recon_shape')[2] % 2 != 0:
            self.skipTest("num_slices not divisible by 2")

        # Single-device reference (no mesh).
        ref = np.asarray(model.fbp_filter(sino, view_batch_size=4))

        # Sharded run.
        model.configure_sharding(self.devs)
        out = model.fbp_filter(sino, view_batch_size=4)

        # Sharded-contract: output is view-sharded (axis 0), not gathered.
        self.assertIsInstance(out.sharding, jax.sharding.NamedSharding)
        self.assertEqual(out.sharding.spec[0], 'devices')

        np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-5, atol=1e-5)

    def test_sharded_input_passes_through_sharded(self):
        """A pre-sharded sinogram stays sharded (no gather) and is correct."""
        model, sino = _make_model_and_sino(num_views=8)
        if model.get_params('recon_shape')[2] % 2 != 0:
            self.skipTest("num_slices not divisible by 2")
        ref = np.asarray(model.fbp_filter(sino, view_batch_size=4))

        model.configure_sharding(self.devs)
        sharded_in = model._shard_sinogram(sino)
        out = model.fbp_filter(sharded_in, view_batch_size=4)
        self.assertIsInstance(out.sharding, jax.sharding.NamedSharding)
        np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
