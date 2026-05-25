"""
Tests for Step 2: multi-device forward_project via threaded sparse_forward_project.

Design
------
Slice-level parallelism lives in ParallelBeamModel.sparse_forward_project.
When self.mesh is not None, sparse_forward_project splits voxel_values along
the slice axis, runs one thread per device, and assembles a NamedSharding
sinogram.  forward_project (base class, unchanged) therefore returns a sharded
sinogram whenever a mesh is configured — regardless of whether the recon input
was sharded or plain.

back_project multi-device threading is implemented in Step 3 (test_sharding_step3.py).
These tests cover the single-device back_project path and basic regression when a
mesh is configured.

Verifies:
  forward_project
    - Single-device: output shape correct, values deterministic.
    - Multi-device plain input:   output is NamedSharding; values match single-device.
    - Multi-device per-device shard shapes are correct.

  back_project (single-device path only, even when mesh is configured)
    - Shape correct, values deterministic.
    - With mesh configured: values still match single-device reference.

  Round-trip
    - back_project(forward_project(recon)) shape matches recon.
    - Single-device and multi-device round-trips agree numerically.

conftest.py sets XLA_FLAGS for virtual CPU devices; real GPUs preferred via
preferred_devices().
"""
import unittest
import numpy as np
import jax
import jax.numpy as jnp
import mbirjax
from conftest import preferred_devices

# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------
_N_CPU = len(jax.devices('cpu'))
assert _N_CPU >= 2, (
    f"Expected ≥2 virtual CPU devices (set by conftest.py); got {_N_CPU}."
)

# ---------------------------------------------------------------------------
# Shared geometry
# ---------------------------------------------------------------------------
N_VIEWS    = 16
N_DET_ROWS = 8       # must be divisible by N_DEV
N_CHANNELS = 16
N_DEV      = 2

ATOL = 1e-4
RTOL = 1e-4


def _make_model_single():
    angles = np.linspace(0, np.pi, N_VIEWS, endpoint=False)
    m = mbirjax.ParallelBeamModel((N_VIEWS, N_DET_ROWS, N_CHANNELS), angles)
    m.set_params(use_gpu='none')
    m.set_devices_and_batch_sizes()
    return m


def _make_model_sharded(n_devices=N_DEV):
    devices = preferred_devices(n_devices)
    if devices is None:
        raise unittest.SkipTest(f"Need {n_devices} devices; none available")
    angles = np.linspace(0, np.pi, N_VIEWS, endpoint=False)
    m = mbirjax.ParallelBeamModel((N_VIEWS, N_DET_ROWS, N_CHANNELS), angles)
    m.configure_sharding(devices)
    return m


def _random_recon(model=None, seed=0):
    if model is None:
        model = _make_model_single()
    recon_shape = model.get_params('recon_shape')
    return np.random.default_rng(seed).standard_normal(recon_shape).astype(np.float32)


def _random_sinogram(seed=0):
    return np.random.default_rng(seed).standard_normal(
        (N_VIEWS, N_DET_ROWS, N_CHANNELS)).astype(np.float32)


# ===========================================================================
# forward_project — single-device
# ===========================================================================

class TestForwardProjectSingleDevice(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = _make_model_single()
        cls.recon = jnp.array(_random_recon(cls.model, seed=0))
        cls.ref   = np.asarray(cls.model.forward_project(cls.recon))

    def test_output_shape(self):
        self.assertEqual(self.ref.shape, (N_VIEWS, N_DET_ROWS, N_CHANNELS))

    def test_output_is_plain_array(self):
        out = self.model.forward_project(self.recon)
        self.assertNotIsInstance(getattr(out, 'sharding', None),
                                 jax.sharding.NamedSharding)

    def test_values_deterministic(self):
        a = np.asarray(self.model.forward_project(self.recon))
        b = np.asarray(self.model.forward_project(self.recon))
        np.testing.assert_array_equal(a, b)


# ===========================================================================
# forward_project — multi-device
# ===========================================================================

class TestForwardProjectMultiDevice(unittest.TestCase):
    """
    forward_project with a mesh configured:
      - Plain recon input  → multi-device projection used internally, plain sinogram returned.
      - Sharded recon input → multi-device projection, sharded sinogram returned.
    Output sharding mirrors input sharding; values always match single-device reference.
    """

    @classmethod
    def setUpClass(cls):
        cls.model_s = _make_model_single()
        cls.model_m = _make_model_sharded(n_devices=N_DEV)
        cls.recon_np  = _random_recon(cls.model_s, seed=1)
        cls.recon_jax = jnp.array(cls.recon_np)
        cls.ref = np.asarray(cls.model_s.forward_project(cls.recon_jax))

    def test_output_shape(self):
        out = self.model_m.forward_project(self.recon_jax)
        self.assertEqual(out.shape, (N_VIEWS, N_DET_ROWS, N_CHANNELS))

    def test_plain_input_plain_output(self):
        """Plain recon → multi-device used internally, output gathered to plain array."""
        out = self.model_m.forward_project(self.recon_jax)
        self.assertNotIsInstance(getattr(out, 'sharding', None),
                                 jax.sharding.NamedSharding,
                                 "forward_project with plain recon should return plain sinogram")

    def test_sharded_input_sharded_output(self):
        """Sharded recon → sharded sinogram (output sharding mirrors input)."""
        sharded_recon = self.model_m._shard_recon(self.recon_jax)
        out = self.model_m.forward_project(sharded_recon)
        self.assertIsInstance(getattr(out, 'sharding', None),
                              jax.sharding.NamedSharding,
                              "forward_project with sharded recon should return sharded sinogram")

    def test_correct_values(self):
        out = np.asarray(self.model_m.forward_project(self.recon_jax))
        np.testing.assert_allclose(out, self.ref, rtol=RTOL, atol=ATOL,
                                   err_msg="Multi-device FP diverges from single-device")

    def test_sharded_input_correct_values(self):
        """Sharded recon path also produces correct values."""
        sharded_recon = self.model_m._shard_recon(self.recon_jax)
        out = np.asarray(self.model_m.forward_project(sharded_recon))
        np.testing.assert_allclose(out, self.ref, rtol=RTOL, atol=ATOL,
                                   err_msg="Sharded-input FP diverges from single-device")

    def test_deterministic(self):
        a = np.asarray(self.model_m.forward_project(self.recon_jax))
        b = np.asarray(self.model_m.forward_project(self.recon_jax))
        np.testing.assert_array_equal(a, b)


# ===========================================================================
# back_project — single-device path (threading deferred)
# ===========================================================================

class TestBackProjectSingleDevice(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = _make_model_single()
        cls.sino  = jnp.array(_random_sinogram(seed=0))
        cls.ref   = np.asarray(cls.model.back_project(cls.sino))

    def test_output_shape(self):
        recon_shape = self.model.get_params('recon_shape')
        self.assertEqual(self.ref.shape, recon_shape)

    def test_output_is_plain_array(self):
        out = self.model.back_project(self.sino)
        self.assertNotIsInstance(getattr(out, 'sharding', None),
                                 jax.sharding.NamedSharding)

    def test_values_deterministic(self):
        a = np.asarray(self.model.back_project(self.sino))
        b = np.asarray(self.model.back_project(self.sino))
        np.testing.assert_array_equal(a, b)


class TestBackProjectWithMesh(unittest.TestCase):
    """
    Regression: back_project (Step 2 baseline) must still produce correct results
    when a mesh is configured.  Multi-device threading for back_project is tested
    in test_sharding_step3.py.
    """

    @classmethod
    def setUpClass(cls):
        cls.model_s = _make_model_single()
        cls.model_m = _make_model_sharded(n_devices=N_DEV)
        cls.sino_jax = jnp.array(_random_sinogram(seed=2))
        cls.ref = np.asarray(cls.model_s.back_project(cls.sino_jax))

    def test_output_shape(self):
        out = self.model_m.back_project(self.sino_jax)
        recon_shape = self.model_s.get_params('recon_shape')
        self.assertEqual(out.shape, recon_shape)

    def test_correct_values(self):
        out = np.asarray(self.model_m.back_project(self.sino_jax))
        np.testing.assert_allclose(out, self.ref, rtol=RTOL, atol=ATOL,
                                   err_msg="back_project with mesh differs from single-device")


# ===========================================================================
# Round-trip shape and value consistency
# ===========================================================================

class TestRoundTrip(unittest.TestCase):
    """
    AT(A(x)) shape must match x, and both paths must agree numerically.

    forward_project with a plain recon returns a plain sinogram (multi-device
    used internally, result gathered), so back_project receives a plain array
    in both single- and multi-device paths.
    """

    @classmethod
    def setUpClass(cls):
        cls.model_s = _make_model_single()
        cls.model_m = _make_model_sharded(n_devices=N_DEV)
        cls.recon   = jnp.array(_random_recon(cls.model_s, seed=3))

    def _round_trip(self, model):
        sino = model.forward_project(self.recon)   # plain recon → plain sinogram
        return model.back_project(sino)

    def test_round_trip_single_device_shape(self):
        out = self._round_trip(self.model_s)
        self.assertEqual(out.shape, self.recon.shape)

    def test_round_trip_multi_device_shape(self):
        out = self._round_trip(self.model_m)
        self.assertEqual(out.shape, self.recon.shape)

    def test_round_trip_single_vs_multi_values(self):
        rt_s = np.asarray(self._round_trip(self.model_s))
        rt_m = np.asarray(self._round_trip(self.model_m))
        np.testing.assert_allclose(rt_s, rt_m, rtol=RTOL, atol=ATOL,
                                   err_msg="AT(A(x)) differs between single and multi device")


if __name__ == '__main__':
    unittest.main(verbosity=2)
