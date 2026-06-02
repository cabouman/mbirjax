"""
Tests for the placement foundation (mbirjax._sharding.placement).

`Placement` is the unit that will replace the scalar main_device /
sinogram_device fields: how a recon-like or sino-like array is distributed
across devices.  `move_cylinders_to_sino` / `sum_cylinders_to_recon` are the
adjoint pair that moves voxel cylinders between a recon placement and a sino
placement (forward all-gather / back reduce-scatter), built on `move_shard`.

These tests check, with no TomographyModel:
  - Placement.shard_ranges / shard_structure / is_trivial, including the
    divisibility error;
  - move_shard to the same device preserves values (the zero-overhead trivial
    path);
  - move_cylinders_to_sino assembles the full cylinders on each sino device, for
    N×N (multi-device) and 1×1 (trivial);
  - sum_cylinders_to_recon reduces over sino devices and returns a slice-sharded
    result, for N×N and 1×1;
  - the adjoint identity <move(x), y> == <x, sum(y)>.

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


def _slice_sharded_cylinders(np_cyl, recon_placement):
    """Place a (batch, num_slices) numpy array slice-sharded on recon_placement."""
    return jax.device_put(np_cyl, recon_placement.shard_structure(np_cyl.ndim))


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


class TestCylinderMovement(unittest.TestCase):

    def _run_move(self, n):
        devs = preferred_devices(n)
        if devs is None:
            self.skipTest(f"need >= {n} devices")
        batch, num_slices = 5, 4 * n          # divisible by n
        rng = np.random.default_rng(0)
        np_cyl = rng.standard_normal((batch, num_slices), dtype=np.float32)

        recon_pl = mjs.Placement(devs, axis=-1)   # slices
        sino_pl = mjs.Placement(devs, axis=0)     # views (only .devices used here)
        cyl = _slice_sharded_cylinders(np_cyl, recon_pl)

        full = mjs.move_cylinders_to_sino(cyl, sino_pl, dev2dev_safe=True)
        # Every sino device must hold the FULL cylinders, equal to the original.
        self.assertEqual(set(full.keys()), set(devs))
        for dev in devs:
            np.testing.assert_allclose(np.asarray(full[dev]), np_cyl,
                                       rtol=1e-6, atol=1e-6)

    def test_move_trivial(self):
        self._run_move(1)

    def test_move_multidevice(self):
        self._run_move(2)

    def _run_sum(self, n):
        devs = preferred_devices(n)
        if devs is None:
            self.skipTest(f"need >= {n} devices")
        batch, num_slices = 5, 4 * n
        rng = np.random.default_rng(1)
        # One distinct full cylinder per sino device, each placed on its device.
        np_parts = [rng.standard_normal((batch, num_slices), dtype=np.float32)
                    for _ in range(n)]
        partials = {devs[i]: jax.device_put(np_parts[i], devs[i]) for i in range(n)}

        recon_pl = mjs.Placement(devs, axis=-1)
        out = mjs.sum_cylinders_to_recon(partials, recon_pl, dev2dev_safe=True)

        # Slice-sharded result equal to the elementwise sum over sino devices.
        self.assertIsInstance(out.sharding, jax.sharding.NamedSharding)
        self.assertEqual(out.sharding.spec[1], "devices")
        np.testing.assert_allclose(np.asarray(out), np.sum(np_parts, axis=0),
                                   rtol=1e-6, atol=1e-6)

    def test_sum_trivial(self):
        self._run_sum(1)

    def test_sum_multidevice(self):
        self._run_sum(2)

    def test_adjoint_identity(self):
        """<move_cylinders_to_sino(x), y> == <x, sum_cylinders_to_recon(y)>.

        move replicates the full cylinders to each sino device; its adjoint sums
        the per-device partials and re-shards by slice.  The inner products must
        agree (the property forward/back projection relies on)."""
        n = 2
        devs = preferred_devices(n)
        if devs is None:
            self.skipTest("need >= 2 devices")
        batch, num_slices = 5, 4 * n
        rng = np.random.default_rng(2)
        np_x = rng.standard_normal((batch, num_slices), dtype=np.float32)
        np_y = [rng.standard_normal((batch, num_slices), dtype=np.float32)
                for _ in range(n)]

        recon_pl = mjs.Placement(devs, axis=-1)
        sino_pl = mjs.Placement(devs, axis=0)

        x = _slice_sharded_cylinders(np_x, recon_pl)
        full = mjs.move_cylinders_to_sino(x, sino_pl, dev2dev_safe=True)
        lhs = sum(float(np.sum(np.asarray(full[devs[i]]) * np_y[i]))
                  for i in range(n))

        y = {devs[i]: jax.device_put(np_y[i], devs[i]) for i in range(n)}
        summed = mjs.sum_cylinders_to_recon(y, recon_pl, dev2dev_safe=True)
        rhs = float(np.sum(np_x * np.asarray(summed)))

        self.assertAlmostEqual(lhs, rhs, places=4)


if __name__ == "__main__":
    unittest.main()
