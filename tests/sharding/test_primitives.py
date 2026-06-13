"""
Tests for the sharding primitives.

Covers the model-agnostic helpers in mbirjax._sharding (transfer +
thread_execution) and TomographyModel.configure_sharding.  Runs on whatever
devices are present: real GPUs on a cluster, virtual CPU devices on a laptop/CI
(set up by conftest).
"""
import unittest

# Import mbirjax before jax so mbirjax._device_setup configures the virtual
# device count before JAX initializes its backends.  (conftest also sets this up
# for the test process, but we follow the same ordering users should follow.)
import mbirjax
import mbirjax._sharding as mjs   # internal sharding primitives, prefixed

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from conftest import preferred_devices


class TestTransfer(unittest.TestCase):

    def test_is_dev2dev_safe_single_device(self):
        """A single device has no cross-device copy, so it is trivially safe."""
        self.assertTrue(mjs.is_dev2dev_safe(jax.devices()[:1]))

    def test_is_dev2dev_safe_two_devices(self):
        """On 2 devices the probe runs; on CPU/virtual it should report safe."""
        devs = preferred_devices(2)
        if devs is None:
            self.skipTest("need >= 2 devices")
        # Virtual CPU devices and H100 are safe; L40S would return False.  We
        # don't assert a fixed value on unknown GPU hardware, only that the
        # probe returns a bool and round-trips a value when it claims safe.
        result = mjs.is_dev2dev_safe(devs)
        self.assertIsInstance(result, bool)

    def test_move_shard_direct_roundtrip(self):
        """move_shard with dev2dev_safe=True places data correctly."""
        devs = preferred_devices(2)
        if devs is None:
            self.skipTest("need >= 2 devices")
        if not mjs.is_dev2dev_safe(devs):
            self.skipTest("direct d2d not safe on this hardware")
        x = jax.device_put(jnp.arange(4.0), devs[0])
        y = mjs.move_shard(x, devs[1], dev2dev_safe=True)
        self.assertEqual(list(y.devices())[0], devs[1])
        np.testing.assert_array_equal(np.asarray(y), np.arange(4.0))

    def test_move_shard_host_bounce_roundtrip(self):
        """move_shard with dev2dev_safe=False (host bounce) is always correct."""
        devs = preferred_devices(2)
        if devs is None:
            self.skipTest("need >= 2 devices")
        x = jax.device_put(jnp.arange(4.0), devs[0])
        with self.assertWarns(UserWarning):
            # First host-bounce in the process warns once.
            y = mjs.move_shard(x, devs[1], dev2dev_safe=False)
        self.assertEqual(list(y.devices())[0], devs[1])
        np.testing.assert_array_equal(np.asarray(y), np.arange(4.0))


class TestExecution(unittest.TestCase):

    def test_run_per_device_order_and_values(self):
        """run_per_device returns one result per device, in device order."""
        devs = preferred_devices(2)
        if devs is None:
            self.skipTest("need >= 2 devices")

        def worker_process(i, device):
            return jnp.asarray(i * 10.0) + 1.0

        results = mjs.run_per_device(devs, worker_process)
        self.assertEqual(len(results), len(devs))
        self.assertEqual([float(r) for r in results], [1.0, 11.0])

    def test_run_per_device_results_on_correct_devices(self):
        """Each worker's result lands on its own device."""
        devs = preferred_devices(2)
        if devs is None:
            self.skipTest("need >= 2 devices")

        def worker_process(i, device):
            return jnp.ones(3)

        results = mjs.run_per_device(devs, worker_process)
        for i, r in enumerate(results):
            self.assertEqual(list(r.devices())[0], devs[i])

    def test_device_pool_reuse_matches_per_call(self):
        """A reused device_pool gives the same results, in device order, as the
        default per-call pool -- across several calls (the streaming use case)."""
        devs = preferred_devices(2)
        if devs is None:
            self.skipTest("need >= 2 devices")

        def worker_process(i, device, k=0):
            return jnp.asarray(i * 10.0) + k

        # Per-call pools (executor=None) as the reference, over several "bands".
        ref = [mjs.run_per_device(devs, lambda i, d, k=k: worker_process(i, d, k))
               for k in range(3)]
        # Same calls through one reused pool.
        with mjs.device_pool(len(devs)) as pool:
            got = [mjs.run_per_device(devs, lambda i, d, k=k: worker_process(i, d, k),
                                      executor=pool)
                   for k in range(3)]
        self.assertEqual([[float(x) for x in row] for row in got],
                         [[float(x) for x in row] for row in ref])
        self.assertEqual([[float(x) for x in row] for row in ref],
                         [[0.0, 10.0], [1.0, 11.0], [2.0, 12.0]])

    def test_assemble_sharded_matches_reference(self):
        """Assembling per-device row-shards reproduces the global array."""
        devs = preferred_devices(2)
        if devs is None:
            self.skipTest("need >= 2 devices")
        from jax.sharding import Mesh
        mesh = Mesh(np.array(devs), ('devices',))
        sharding = NamedSharding(mesh, P('devices', None))

        global_arr = np.arange(8.0).reshape(2, 4)  # row i -> device i
        parts = [jax.device_put(global_arr[i:i + 1], devs[i]) for i in range(2)]
        assembled = mjs.assemble_sharded(parts, global_arr.shape, sharding)
        np.testing.assert_array_equal(np.asarray(assembled), global_arr)


class TestConfigureSharding(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Geometry chosen so num_views and num_slices are both divisible by 1
        # and 2 (the device counts tested here).
        cls.sinogram_shape = (8, 4, 16)   # num_views = 8
        # recon_shape[2] (num_slices) follows from auto geometry; we check it.

    def _make_model(self):
        return mbirjax.ParallelBeamModel(
            self.sinogram_shape, angles=np.linspace(0, np.pi, 8, endpoint=False)
        )

    def test_default_single_device_trivial_mesh(self):
        model = self._make_model()
        model.configure_sharding()  # devices=None -> 1-device trivial mesh
        self.assertIsNotNone(model.mesh)
        self.assertEqual(model.mesh.devices.size, 1)
        self.assertTrue(model.dev2dev_safe)
        self.assertTrue(model.is_sharded)

    def test_two_device_mesh(self):
        devs = preferred_devices(2)
        if devs is None:
            self.skipTest("need >= 2 devices")
        model = self._make_model()
        num_slices = model.get_params('recon_shape')[2]
        if num_slices % 2 != 0:
            self.skipTest(f"num_slices {num_slices} not divisible by 2")
        model.configure_sharding(devs)
        self.assertEqual(model.mesh.devices.size, 2)
        self.assertIsInstance(model.dev2dev_safe, bool)

    def test_non_dividing_axes_pad_instead_of_raising(self):
        """Neither sharded axis constrains the device count anymore (P5 Step 4 Stage 2):
        a non-dividing view count AND a non-dividing slice count are zero-padded to the
        device form, with the tails exactly zero.  num_views=8 over 3 devices pads to 9;
        num_slices=4 over 3 devices pads to 6 (an explicitly configured count may even
        leave the last shard fully padded -- correct, just wasteful; AUTO skips those).
        For parallel beam the detector rows pad with the slices (row r <-> slice r), so
        the sino device form is (9, 6, channels).
        """
        devs = preferred_devices(3)
        if devs is None:
            self.skipTest("need >= 3 devices")
        model = self._make_model()
        num_slices = model.get_params('recon_shape')[2]
        num_rows = model.get_params('sinogram_shape')[1]
        self.assertNotEqual(num_slices % 3, 0)
        model.configure_sharding(devs)   # no warning, no raise: padding handles both axes
        # Views pad 8 -> 9 and rows pad with the slices; all padded entries exactly zero.
        sino = np.ones(model.get_params('sinogram_shape'), dtype=np.float32)
        sharded = model._shard_sinogram(sino)
        self.assertEqual(sharded.shape, (9, 6, sino.shape[2]))
        sharded_np = np.asarray(sharded)
        np.testing.assert_array_equal(sharded_np[8:], 0.0)
        np.testing.assert_array_equal(sharded_np[:, num_rows:, :], 0.0)
        np.testing.assert_array_equal(np.asarray(model._gather_sinogram(sharded)), sino)
        # Slices pad 4 -> 6 with a zero tail, and the exit gather crops back.
        rows, cols, _ = model.get_params('recon_shape')
        flat = np.ones((rows * cols, num_slices), dtype=np.float32)
        sharded_recon = model._shard_recon(flat)
        self.assertEqual(sharded_recon.shape, (rows * cols, 6))
        np.testing.assert_array_equal(np.asarray(sharded_recon)[:, num_slices:], 0.0)
        np.testing.assert_array_equal(np.asarray(model._gather_recon(sharded_recon)), flat)

    def test_shape_change_rederives_padding(self):
        """Order-independence under padding: a user-selected device count is kept across
        shape changes, and the pad metadata is re-derived from the new shapes on every
        recompile -- a non-dividing slice count pads, a dividing one carries no padding.
        Also: a STALE device-form array from the previous (padded) layout is rejected
        with a clear error once the new layout no longer pads, even though its size
        happens to divide the device count (the silent-wrong-results corner from
        Stage 1, closed by the entry shape check).
        """
        devs = preferred_devices(2)
        if devs is None:
            self.skipTest("need >= 2 devices")
        model = self._make_model()                          # num_views = 8 (divisible by 2)
        rows, cols, _ = model.get_params('recon_shape')
        model.set_params(recon_shape=(rows, cols, 5))        # 5 slices over 2 devices: pads to 6
        model.configure_sharding(devs)
        out = model._shard_recon(np.zeros((rows * cols, 5), dtype=np.float32))
        self.assertEqual(out.shape, (rows * cols, 6))
        stale_device_form = np.zeros((rows * cols, 6), dtype=np.float32)
        # Change to a dividing slice count: same devices, no padding, real shapes only.
        model.set_params(recon_shape=(rows, cols, 4))
        self.assertEqual(len(model.shard_devices), 2)
        out = model._shard_recon(np.zeros((rows * cols, 4), dtype=np.float32))
        self.assertEqual(out.shape, (rows * cols, 4))
        # The stale 6-slice device form (6 divides 2!) must NOT shard silently.
        with self.assertRaises(ValueError):
            model._shard_recon(stale_device_form)


if __name__ == "__main__":
    unittest.main()
