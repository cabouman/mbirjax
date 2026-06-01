"""
Tests for the geometry sharding hooks on TomographyModel.

Covers the axis-declaration hooks, the shard/gather round-trips for sinogram and
recon (both 3-D and flat), the _extract_halos boundary-slice logic, and the
no-op behavior when no mesh is configured.  Runs on whatever devices are present
(real GPUs on a cluster, virtual CPU devices otherwise, via conftest).
"""
import unittest

# Import mbirjax before jax (device-setup-first ordering).
import mbirjax

import numpy as np
import jax

from conftest import preferred_devices


def _make_model(num_views=8):
    """ParallelBeamModel with a small geometry; num_views divisible by 1 and 2."""
    return mbirjax.ParallelBeamModel(
        (num_views, 4, 16), angles=np.linspace(0, np.pi, num_views, endpoint=False)
    )


class TestAxisHooks(unittest.TestCase):

    def test_default_axes(self):
        model = _make_model()
        self.assertEqual(model.sinogram_shard_axis(), 0)   # views
        self.assertEqual(model.recon_shard_axis(), -1)     # last axis = slices


class TestNoMeshNoOp(unittest.TestCase):
    """Without configure_sharding, all hooks are no-ops (mesh is None)."""

    def test_shard_and_gather_are_noops(self):
        model = _make_model()
        self.assertIsNone(model.mesh)
        sino = np.ones((8, 4, 16), dtype=np.float32)
        # _shard_* returns the input unchanged (same object) when no mesh.
        self.assertIs(model._shard_sinogram(sino), sino)
        self.assertIs(model._gather_sinogram(sino), sino)
        self.assertIs(model._shard_recon(sino), sino)
        self.assertIs(model._gather_recon(sino), sino)

    def test_extract_halos_no_mesh(self):
        model = _make_model()
        left, right = model._extract_halos(np.ones((6, 16), dtype=np.float32))
        self.assertEqual(left, [None])
        self.assertEqual(right, [None])


class TestShardGatherRoundTrip(unittest.TestCase):

    def setUp(self):
        self.devs = preferred_devices(2)
        if self.devs is None:
            self.skipTest("need >= 2 devices")
        self.model = _make_model()
        # num_slices must be divisible by 2 for the recon shard.
        self.num_slices = self.model.get_params('recon_shape')[2]
        if self.num_slices % 2 != 0:
            self.skipTest(f"num_slices {self.num_slices} not divisible by 2")
        self.model.configure_sharding(self.devs)

    def test_sinogram_shard_axis_and_roundtrip(self):
        sino = np.arange(8 * 4 * 16, dtype=np.float32).reshape(8, 4, 16)
        sharded = self.model._shard_sinogram(sino)
        # Partitioned on axis 0 (views): spec is ('devices', None, None).
        self.assertEqual(sharded.sharding.spec[0], 'devices')
        self.assertIsNone(sharded.sharding.spec[1])
        self.assertIsNone(sharded.sharding.spec[2])
        gathered = self.model._gather_sinogram(sharded)
        np.testing.assert_array_equal(np.asarray(gathered), sino)

    def test_flat_recon_shard_axis_and_roundtrip(self):
        flat = np.arange(6 * self.num_slices, dtype=np.float32).reshape(6, self.num_slices)
        sharded = self.model._shard_recon(flat)
        # Partitioned on the last axis (slices) of a 2-D flat recon: (None, 'devices').
        self.assertIsNone(sharded.sharding.spec[0])
        self.assertEqual(sharded.sharding.spec[1], 'devices')
        gathered = self.model._gather_recon(sharded)
        np.testing.assert_array_equal(np.asarray(gathered), flat)

    def test_3d_recon_shard_axis_and_roundtrip(self):
        rs = self.model.get_params('recon_shape')
        r3 = np.arange(int(np.prod(rs)), dtype=np.float32).reshape(rs)
        sharded = self.model._shard_recon(r3)
        # Partitioned on the last axis (slices) of a 3-D recon: (None, None, 'devices').
        self.assertIsNone(sharded.sharding.spec[0])
        self.assertIsNone(sharded.sharding.spec[1])
        self.assertEqual(sharded.sharding.spec[2], 'devices')
        gathered = self.model._gather_recon(sharded)
        np.testing.assert_array_equal(np.asarray(gathered), r3)

    def test_reshard_is_noop(self):
        """Sharding an already-correctly-sharded array returns it unchanged."""
        sino = np.ones((8, 4, 16), dtype=np.float32)
        sharded = self.model._shard_sinogram(sino)
        again = self.model._shard_sinogram(sharded)
        self.assertIs(again, sharded)


class TestExtractHalos(unittest.TestCase):

    def setUp(self):
        self.devs = preferred_devices(2)
        if self.devs is None:
            self.skipTest("need >= 2 devices")
        self.model = _make_model()
        self.num_slices = self.model.get_params('recon_shape')[2]
        if self.num_slices % 2 != 0:
            self.skipTest(f"num_slices {self.num_slices} not divisible by 2")
        self.model.configure_sharding(self.devs)

    def test_halos_match_boundary_slices(self):
        num_pixels = 6
        flat = np.arange(num_pixels * self.num_slices, dtype=np.float32).reshape(
            num_pixels, self.num_slices)
        sharded = self.model._shard_recon(flat)
        left, right = self.model._extract_halos(sharded)

        # Two devices: device 0 holds slices [0:h), device 1 holds [h:2h).
        h = self.num_slices // 2
        self.assertEqual(len(left), 2)
        self.assertEqual(len(right), 2)
        # Boundary devices get None on the outer side.
        self.assertIsNone(left[0])
        self.assertIsNone(right[-1])
        # left[1] = last slice of device 0 = column h-1 of the global array.
        np.testing.assert_array_equal(left[1], flat[:, h - 1])
        # right[0] = first slice of device 1 = column h of the global array.
        np.testing.assert_array_equal(right[0], flat[:, h])


if __name__ == "__main__":
    unittest.main()
