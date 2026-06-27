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


class TestShardedSheppLogan(unittest.TestCase):
    """generate_3d_shepp_logan_low_dynamic_range(devices=...) builds the same phantom as the
    single-device path, slice-sharded across devices with inert zero padding."""

    def setUp(self):
        self.devs = preferred_devices(2)
        if self.devs is None:
            self.skipTest("need >= 2 devices")

    def test_sharded_matches_single_device_dividing(self):
        # slices divisible by the device count -> no padding, device-form == real shape.
        shape = (16, 12, 8)
        single = np.asarray(mbirjax.generate_3d_shepp_logan_low_dynamic_range(shape))
        sharded = mbirjax.generate_3d_shepp_logan_low_dynamic_range(shape, devices=self.devs)
        self.assertEqual(len(sharded.addressable_shards), len(self.devs))
        self.assertEqual(tuple(sharded.shape), shape)
        # Independent per-device build (no reduction) -> bit-identical to single-device.
        np.testing.assert_array_equal(np.asarray(sharded), single)

    def test_sharded_matches_single_device_padded(self):
        # slices NOT divisible by the device count -> padded to the next multiple, tail is zero.
        n = len(self.devs)
        real_slices = 2 * n + 1                 # not a multiple of n
        padded_slices = ((real_slices + n - 1) // n) * n
        shape = (16, 12, real_slices)
        single = np.asarray(mbirjax.generate_3d_shepp_logan_low_dynamic_range(shape))
        sharded = mbirjax.generate_3d_shepp_logan_low_dynamic_range(shape, devices=self.devs)
        arr = np.asarray(sharded)
        self.assertEqual(arr.shape, (16, 12, padded_slices))     # device form on the slice axis
        np.testing.assert_array_equal(arr[:, :, :real_slices], single)   # real region matches
        np.testing.assert_array_equal(arr[:, :, real_slices:], 0)        # padding is exactly inert

    def test_sharded_target_attenuation_matches_single(self):
        # The opt-in attenuation scale is applied identically in the sharded and single-device builds.
        shape = (16, 12, 8)
        single = np.asarray(mbirjax.generate_3d_shepp_logan_low_dynamic_range(shape, target_max_attenuation=6.0))
        sharded = mbirjax.generate_3d_shepp_logan_low_dynamic_range(shape, devices=self.devs, target_max_attenuation=6.0)
        np.testing.assert_array_equal(np.asarray(sharded), single)


class TestSheppLoganAttenuationScale(unittest.TestCase):
    """The opt-in target_max_attenuation applies the analytic main-ellipsoid scale (single device)."""

    def setUp(self):
        self.devs = preferred_devices(1)
        if self.devs is None:
            self.skipTest("need >= 1 device")

    def test_target_applies_analytic_scale(self):
        # target_max_attenuation uniformly scales the phantom by the function's own analytic factor.
        # Compare against that factor directly (not a hand-copied formula) so the test stays correct
        # if the scale calibration is retuned.
        from mbirjax.utilities import _shepp_logan_attenuation_scale
        shape = (20, 14, 9)
        target = 6.0
        base = np.asarray(mbirjax.generate_3d_shepp_logan_low_dynamic_range(shape))
        scaled = np.asarray(
            mbirjax.generate_3d_shepp_logan_low_dynamic_range(shape, target_max_attenuation=target))
        np.testing.assert_allclose(scaled, base * _shepp_logan_attenuation_scale(shape, target),
                                   rtol=1e-6, atol=1e-6)
        self.assertEqual(float(base.max()), 1.0)     # default (target None) is unscaled


class TestGenWeightsSharding(unittest.TestCase):
    """gen_weights is element-wise, so a view-sharded sinogram gives view-sharded weights with the
    same sharding (no gather, no collective), and the values match the per-element formulas."""

    @staticmethod
    def _refs(s):
        return {'unweighted': np.ones_like(s), 'transmission': np.exp(-s),
                'transmission_root': np.exp(-s / 2), 'emission': 1.0 / (np.abs(s) + 0.1)}

    def test_values_plain(self):
        s = (np.random.RandomState(0).rand(6, 5, 7).astype(np.float32) * 3)
        for wt, ref in self._refs(s).items():
            np.testing.assert_allclose(np.asarray(mbirjax.gen_weights(s, wt)), ref, rtol=1e-5, atol=1e-6)

    def test_host_input_stays_on_host(self):
        # A numpy sinogram yields numpy weights (host-preserving): nothing is landed on a device, so a
        # large host sinogram is never copied whole onto one GPU before recon streams it to shards.
        s = (np.random.RandomState(1).rand(6, 5, 7).astype(np.float32) * 3)
        for wt in self._refs(s):
            w = mbirjax.gen_weights(s, wt)
            self.assertIsInstance(w, np.ndarray)        # host in -> host out (not a single-device jax array)
            self.assertEqual(w.dtype, np.float32)       # element-wise op preserves dtype

    def test_ct_model_shards_host_input(self):
        # ct_model= distributes a host sinogram into the model's view-sharded device form and weights it
        # per shard, so the result is sharded across all devices (no single-device copy of the input).
        devs = preferred_devices(2)
        if devs is None:
            self.skipTest("need >= 2 devices")
        # views (sharded) and det-rows (parallel beam pads rows to the recon-slice count) both
        # divisible by the device count -> device form == real shape, so no inert padding to mask.
        num_views, num_rows, num_channels = 4 * len(devs), 2 * len(devs), 6
        s = (np.random.RandomState(2).rand(num_views, num_rows, num_channels).astype(np.float32) * 3)
        angles = np.linspace(0, np.pi, num_views, endpoint=False)
        model = mbirjax.ParallelBeamModel((num_views, num_rows, num_channels), angles)
        model.configure_devices(devs)
        for wt, ref in self._refs(s).items():
            w = mbirjax.gen_weights(s, wt, ct_model=model)
            self.assertEqual(len(w.addressable_shards), len(devs))   # host in + ct_model -> sharded out
            np.testing.assert_allclose(np.asarray(w), ref, rtol=1e-5, atol=1e-6)

    def test_preserves_sharding(self):
        devs = preferred_devices(2)
        if devs is None:
            self.skipTest("need >= 2 devices")
        s = (np.random.RandomState(0).rand(8, 5, 6).astype(np.float32) * 3)
        mesh = jax.sharding.Mesh(np.array(devs), ('d',))
        shd = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('d', None, None))
        sino = jax.device_put(s, shd)
        for wt, ref in self._refs(s).items():
            w = mbirjax.gen_weights(sino, wt)
            self.assertEqual(len(w.addressable_shards), len(devs))     # sharded in -> sharded out
            np.testing.assert_allclose(np.asarray(w), ref, rtol=1e-5, atol=1e-6)


class TestGenerateDemoDataSharding(unittest.TestCase):
    """generate_demo_data returns numpy by default and device-sharded data when devices is given."""

    def test_default_returns_numpy(self):
        # Default (no devices): BOTH object and sinogram are plain numpy, and params arrays too -- no
        # device residue.  (Previously the shepp-logan object came back as a single-device jax array.)
        phantom, sino, params = mbirjax.generate_demo_data(model_type='parallel', num_views=12,
                                                           num_det_rows=10, num_det_channels=14)
        self.assertIsInstance(sino, np.ndarray)
        self.assertIsInstance(phantom, np.ndarray)
        self.assertIsInstance(params['angles'], np.ndarray)

    def test_output_sharded_true_without_devices_is_jax(self):
        # output_sharded=True overrides the default and returns device-form jax arrays even with no
        # devices (single-device / trivial 1-shard).
        phantom, sino, _ = mbirjax.generate_demo_data(model_type='parallel', num_views=12,
                                                       num_det_rows=10, num_det_channels=14,
                                                       output_sharded=True)
        self.assertIsInstance(sino, jax.Array)
        self.assertIsInstance(phantom, jax.Array)

    def test_devices_returns_sharded(self):
        devs = preferred_devices(2)
        if devs is None:
            self.skipTest("need >= 2 devices")
        phantom, sino, _ = mbirjax.generate_demo_data(model_type='parallel', num_views=20,
                                                      num_det_rows=16, num_det_channels=24, devices=devs)
        self.assertEqual(tuple(sino.shape), (20, 16, 24))
        self.assertEqual(len(sino.addressable_shards), len(devs))      # view-sharded sinogram
        self.assertEqual(len(phantom.addressable_shards), len(devs))   # slice-sharded phantom

    def test_devices_with_output_sharded_false_gathers_to_numpy(self):
        # devices given but output_sharded=False: compute sharded, then gather to numpy and free the
        # device arrays.  Values match the sharded path.
        devs = preferred_devices(2)
        if devs is None:
            self.skipTest("need >= 2 devices")
        kw = dict(model_type='parallel', num_views=20, num_det_rows=16, num_det_channels=24)
        ph_s, sino_s, _ = mbirjax.generate_demo_data(devices=devs, **kw)
        ph_h, sino_h, _ = mbirjax.generate_demo_data(devices=devs, output_sharded=False, **kw)
        self.assertIsInstance(sino_h, np.ndarray)
        self.assertIsInstance(ph_h, np.ndarray)
        np.testing.assert_allclose(sino_h, np.asarray(sino_s), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(ph_h, np.asarray(ph_s), rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
