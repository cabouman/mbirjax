"""
Tests for the placement foundation (mbirjax._sharding).

`Placement` is the unit that replaces the scalar main_device / sinogram_device
fields: how a recon-like or sino-like array is distributed across devices.  The
data that crosses the recon↔sino boundary (voxel-cylinder slice-bands) is moved
by the banded adjoint pair ``sum_band_to_owner`` (back / reduce) and
``broadcast_band_to_views`` (forward / broadcast), built on ``move_shard``.

These tests check, with no TomographyModel:
  - Placement.shard_ranges / shard_structure / is_trivial, including the
    divisibility error;
  - move_shard to the same device preserves values (the zero-overhead trivial
    path);
  - sum_band_to_owner reduces band partials onto the slice-owner;
  - broadcast_band_to_views copies a band to every view-owner;
  - the adjoint identity <broadcast(x), y> == <x, sum(y)>.

Runs on whatever devices conftest provides (real GPUs on a cluster, virtual CPU
devices otherwise).
"""
import unittest

# Import mbirjax before jax (device-setup-first ordering).
import mbirjax
import mbirjax._sharding as mjs

import numpy as np
import jax

from conftest import preferred_devices


class TestPlacement(unittest.TestCase):
    """Pure-Python Placement behavior (one device is enough)."""

    def setUp(self):
        self.devs = preferred_devices(1)
        if self.devs is None:
            self.skipTest("need >= 1 device")

    def test_trivial_flags(self):
        p = mjs.Placement(self.devs[:1], axis=-1)
        self.assertTrue(p.is_trivial)
        self.assertEqual(p.n_devices, 1)

    def test_shard_ranges_equal_blocks(self):
        two = preferred_devices(2)
        if two is None:
            self.skipTest("need >= 2 devices")
        p = mjs.Placement(two, axis=-1)
        blocks = p.shard_ranges(8)
        self.assertEqual([rng for _, rng in blocks], [(0, 4), (4, 8)])
        self.assertEqual([d for d, _ in blocks], list(two))

    def test_shard_ranges_divisibility_error(self):
        two = preferred_devices(2)
        if two is None:
            self.skipTest("need >= 2 devices")
        p = mjs.Placement(two, axis=-1)
        with self.assertRaises(ValueError):
            p.shard_ranges(7)

    def test_shard_structure_spec(self):
        p = mjs.Placement(self.devs[:1], axis=-1)
        sh = p.shard_structure(2)   # (batch, num_slices): slice axis = last
        self.assertEqual(sh.spec[1], "devices")
        self.assertIsNone(sh.spec[0])

    def test_move_shard_same_device_preserves_values(self):
        """move_shard to the device an array already lives on keeps its values
        (device_put is a no-op there — the zero-overhead single-device path)."""
        dev = self.devs[0]
        x = jax.device_put(np.arange(12, dtype=np.float32).reshape(3, 4), dev)
        y = mjs.move_shard(x, dev, dev2dev_safe=True)
        np.testing.assert_array_equal(np.asarray(y), np.asarray(x))


class TestBandMovement(unittest.TestCase):
    """The banded adjoint pair: sum_band_to_owner / broadcast_band_to_views."""

    def test_sum_band_to_owner(self):
        """Per-device band partials are summed onto the owner."""
        n = 2
        devs = preferred_devices(n)
        if devs is None:
            self.skipTest("need >= 2 devices")
        batch, band = 5, 4
        rng = np.random.default_rng(0)
        parts_np = [rng.standard_normal((batch, band), dtype=np.float32) for _ in range(n)]
        partials = [jax.device_put(parts_np[i], devs[i]) for i in range(n)]
        out = mjs.sum_band_to_owner(partials, devs[0], dev2dev_safe=True)
        np.testing.assert_allclose(np.asarray(out), np.sum(parts_np, axis=0),
                                   rtol=1e-6, atol=1e-6)

    def test_broadcast_band_to_views(self):
        """A band on its slice-owner is copied to every view-owner."""
        n = 2
        devs = preferred_devices(n)
        if devs is None:
            self.skipTest("need >= 2 devices")
        batch, band = 5, 4
        band_np = np.random.default_rng(1).standard_normal((batch, band), dtype=np.float32)
        on_owner = jax.device_put(band_np, devs[0])
        full = mjs.broadcast_band_to_views(on_owner, devs, dev2dev_safe=True)
        self.assertEqual(set(full.keys()), set(devs))
        for dev in devs:
            np.testing.assert_allclose(np.asarray(full[dev]), band_np,
                                       rtol=1e-6, atol=1e-6)

    def test_adjoint_identity(self):
        """<broadcast(x), y> == <x, sum(y)>.

        broadcast copies a band to every view-owner; its adjoint sums the
        per-device band partials onto the slice-owner.  The inner products must
        agree (the property forward/back projection relies on)."""
        n = 2
        devs = preferred_devices(n)
        if devs is None:
            self.skipTest("need >= 2 devices")
        batch, band = 5, 4
        rng = np.random.default_rng(2)
        x_np = rng.standard_normal((batch, band), dtype=np.float32)
        y_np = [rng.standard_normal((batch, band), dtype=np.float32) for _ in range(n)]

        x = jax.device_put(x_np, devs[0])
        full = mjs.broadcast_band_to_views(x, devs, dev2dev_safe=True)
        lhs = sum(float(np.sum(np.asarray(full[devs[i]]) * y_np[i])) for i in range(n))

        y = [jax.device_put(y_np[i], devs[i]) for i in range(n)]
        summed = mjs.sum_band_to_owner(y, devs[0], dev2dev_safe=True)
        rhs = float(np.sum(x_np * np.asarray(summed)))

        self.assertAlmostEqual(lhs, rhs, places=4)


if __name__ == "__main__":
    unittest.main()
