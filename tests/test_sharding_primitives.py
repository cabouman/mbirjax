"""
Phase A unit tests for the sharding primitives.

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
        self.assertEqual(model.use_gpu, 'sharded')

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

    def test_divisibility_error(self):
        """A device count that doesn't divide a sharded axis raises clearly.

        With num_views=8 (sinogram view axis) and 3 devices, the sinogram-axis
        divisibility check fires first; the message names the sharded axis.
        """
        devs = preferred_devices(3)
        if devs is None:
            self.skipTest("need >= 3 devices")
        model = self._make_model()  # num_views = 8, not divisible by 3
        with self.assertRaises(ValueError) as ctx:
            model.configure_sharding(devs)
        msg = str(ctx.exception)
        self.assertIn("divisible", msg)
        self.assertIn("sinogram axis", msg)


if __name__ == "__main__":
    unittest.main()
