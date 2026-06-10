"""
Tests for the geometry sharding hooks on TomographyModel.

Covers the axis-declaration hooks, the shard/gather round-trips for sinogram and
recon (both 3-D and flat), the _extract_halos boundary-slice logic, the no-op
behavior when no mesh is configured, and the recon/sino placement construction.
Runs on whatever devices are present (real GPUs on a cluster, virtual CPU devices
otherwise, via conftest).
"""
import unittest
import math

# Import mbirjax before jax (device-setup-first ordering).
import mbirjax

import numpy as np
import jax

from conftest import preferred_devices


def _make_model(num_views=8):
    """ParallelBeamModel with a small geometry; num_views divisible by 1 and 2.

    Pins a single device so the bare model is a deterministic single-device REFERENCE regardless of
    how many GPUs are present (auto-sharding now uses all available GPUs by default); tests that
    exercise multi-device sharding override this with their own configure_sharding(devs).
    """
    model = mbirjax.ParallelBeamModel(
        (num_views, 4, 16), angles=np.linspace(0, np.pi, num_views, endpoint=False)
    )
    model.configure_devices(1)
    return model


class TestAxisHooks(unittest.TestCase):

    def test_default_axes(self):
        model = _make_model()
        self.assertEqual(model.sinogram_shard_axis(), 0)   # views
        self.assertEqual(model.recon_shard_axis(), -1)     # last axis = slices


class TestNoMeshNoOp(unittest.TestCase):
    """Without a mesh, the base-class hooks are no-ops (mesh is None).

    RETIRE-AFTER-SHARDING: ParallelBeam now auto-defaults to a trivial 1-device mesh
    (Option B), so it no longer exercises the no-mesh branch.  That branch survives only
    for the not-yet-ported geometries (cone / translation / multiaxis) and retires at P6.
    These tests used ParallelBeam as a convenient concrete model, so they are skipped until
    the no-mesh path either gets a non-ParallelBeam home or is removed at P6.
    """

    @unittest.skip("RETIRE-AFTER-SHARDING: ParallelBeam auto-meshes; no-mesh path is non-PB only (P6).")
    def test_shard_and_gather_are_noops(self):
        model = _make_model()
        self.assertIsNone(model.mesh)
        sino = np.ones((8, 4, 16), dtype=np.float32)
        # _shard_* returns the input unchanged (same object) when no mesh.
        self.assertIs(model._shard_sinogram(sino), sino)
        self.assertIs(model._gather_sinogram(sino), sino)
        self.assertIs(model._shard_recon(sino), sino)
        self.assertIs(model._gather_recon(sino), sino)

    @unittest.skip("RETIRE-AFTER-SHARDING: ParallelBeam auto-meshes; no-mesh path is non-PB only (P6).")
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


class TestModelPlacements(unittest.TestCase):
    """The model builds recon_placement / sino_placement from its device config."""

    def test_single_device_trivial_placements(self):
        model = _make_model()   # _make_model pins a single device
        # ParallelBeam runs the always-on placement path (Option B), so even a single device is
        # "sharded" over one device with trivial (1-shard) placements on that device.
        self.assertTrue(model.is_sharded)
        self.assertEqual(len(model.shard_devices), 1)
        for pl, axis in ((model.recon_placement, model.recon_shard_axis()),
                         (model.sino_placement, model.sinogram_shard_axis())):
            self.assertIsNotNone(pl)
            self.assertTrue(pl.is_trivial)
            self.assertEqual(pl.n_devices, 1)
            self.assertEqual(pl.axis, axis)
        # The trivial placements sit on the configured single devices.
        self.assertEqual(model.recon_placement.devices, [model.main_device])
        self.assertEqual(model.sino_placement.devices, [model.sinogram_device])

    def test_auto_device_count_picks_largest_dividing(self):
        # Auto multi-GPU selection: _auto_device_count(k) is the largest n <= k that divides BOTH
        # sharded axes (i.e. divides gcd(num_views, num_slices)).  CPU-testable (pure arithmetic),
        # so it covers the GPU-only auto-sharding selection without needing real GPUs.
        model = _make_model()
        num_views = model.get_params('sinogram_shape')[0]
        num_slices = model.get_params('recon_shape')[2]
        g = math.gcd(int(num_views), int(num_slices))
        for k in (1, 2, 3, 4, 8):
            expected = max((n for n in range(1, k + 1) if g % n == 0), default=1)
            self.assertEqual(model._auto_device_count(k), expected)
        self.assertEqual(model._auto_device_count(0), 1)   # no devices -> 1

    def test_auto_shards_cpu_when_enabled(self):
        # The _auto_shard_cpu opt-in lets AUTOMATIC selection shard across CPU devices (it is GPU-only
        # by default).  A bare model is single-device on CPU; with the flag set it auto-shards across
        # the CPU devices, and the sharded back projection matches the single-device reference.  Note:
        # this builds bare models directly (not _make_model, which pins a single device).
        if preferred_devices(2) is None:
            self.skipTest("need >= 2 devices")
        try:
            if len(jax.devices('gpu')) > 0:
                self.skipTest("GPU present: auto uses GPUs; this exercises the CPU opt-in")
        except RuntimeError:
            pass
        angles = np.linspace(0, np.pi, 8, endpoint=False)
        idx_shape = (8, 8, 16)   # num_views = num_slices = 8 -> gcd 8, shardable across CPU devices

        # Single-device reference.
        ref_model = mbirjax.ParallelBeamModel(idx_shape, angles)
        ref_model.configure_devices(1)
        sino = np.random.default_rng(0).random(idx_shape, dtype=np.float32)
        idx = mbirjax.gen_full_indices(ref_model.get_params('recon_shape'),
                                       use_ror_mask=ref_model.get_params('use_ror_mask'))
        ref = np.asarray(ref_model.sparse_back_project(sino, idx))

        # Bare model: single-device on CPU by default; opt in and re-select.
        model = mbirjax.ParallelBeamModel(idx_shape, angles)
        self.assertEqual(len(model.shard_devices), 1)
        model._auto_shard_cpu = True
        model.set_devices()
        n_cpu = len(jax.devices('cpu'))
        self.assertEqual(len(model.shard_devices), model._auto_device_count(n_cpu))
        self.assertGreater(len(model.shard_devices), 1)
        self.assertEqual(model.use_gpu, 'sharded')
        self.assertEqual(model._platform_label(model.shard_devices[0]), 'CPU')

        out = np.asarray(model.sparse_back_project(sino, idx))
        np.testing.assert_allclose(
            out, ref, rtol=1e-5, atol=1e-5,
            err_msg="auto CPU-sharded back projection diverged from single device")

    def test_sharded_placements_over_mesh(self):
        devs = preferred_devices(2)
        if devs is None:
            self.skipTest("need >= 2 devices")
        model = _make_model()
        if model.get_params('recon_shape')[2] % 2 != 0:
            self.skipTest("num_slices not divisible by 2")
        model.configure_sharding(devs)
        self.assertEqual(model.recon_placement.devices, list(devs))
        self.assertEqual(model.sino_placement.devices, list(devs))
        self.assertEqual(model.recon_placement.axis, model.recon_shard_axis())
        self.assertEqual(model.sino_placement.axis, model.sinogram_shard_axis())

    def test_placements_match_existing_sharding(self):
        """Placement.shard_structure matches the sharding _shard_recon /
        _shard_sinogram produce, so the placements describe the same distribution
        as the existing hooks."""
        devs = preferred_devices(2)
        if devs is None:
            self.skipTest("need >= 2 devices")
        model = _make_model()
        num_slices = model.get_params('recon_shape')[2]
        if num_slices % 2 != 0:
            self.skipTest("num_slices not divisible by 2")
        model.configure_sharding(devs)
        flat = np.ones((6, num_slices), dtype=np.float32)
        self.assertEqual(model._shard_recon(flat).sharding,
                         model.recon_placement.shard_structure(2))
        sino = np.ones((8, 4, 16), dtype=np.float32)
        self.assertEqual(model._shard_sinogram(sino).sharding,
                         model.sino_placement.shard_structure(3))


if __name__ == "__main__":
    unittest.main()
