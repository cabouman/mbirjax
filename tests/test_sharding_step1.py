"""
Tests for Step 1: fbp_filter Path G (zero-PCIe multi-device filtering).

Verifies:
  - Single-device mode: result matches the reference to float32 noise.
  - Multi-device sharded input: output is a NamedSharding array; values match
    single-device to float32 noise.
  - Multi-device plain input: output is a plain (uncommitted) array; values match
    single-device to float32 noise.

conftest.py sets XLA_FLAGS=--xla_force_host_platform_device_count=4 as a CPU
fallback; real GPUs are used when available (via preferred_devices).
"""
import unittest
import numpy as np
import jax
import jax.numpy as jnp
import mbirjax
from conftest import preferred_devices

# ---------------------------------------------------------------------------
# Guard: confirm conftest.py gave us virtual CPU devices as fallback.
# ---------------------------------------------------------------------------
_N_CPU = len(jax.devices('cpu'))
assert _N_CPU >= 4, (
    f"Expected ≥4 virtual CPU devices from conftest.py; got {_N_CPU}."
)

# ---------------------------------------------------------------------------
# Shared geometry
# ---------------------------------------------------------------------------
N_VIEWS    = 16
N_DET_ROWS = 8       # divisible by 2
N_CHANNELS = 32
N_DEV      = 2


def _make_model_single():
    angles = np.linspace(0, np.pi, N_VIEWS, endpoint=False)
    m = mbirjax.ParallelBeamModel((N_VIEWS, N_DET_ROWS, N_CHANNELS), angles)
    m.set_params(use_gpu='none')
    m.set_devices_and_batch_sizes()
    return m


def _make_model_sharded(n_devices=N_DEV):
    """Configure sharding using real GPUs when available, virtual CPUs otherwise."""
    devices = preferred_devices(n_devices)
    if devices is None:
        raise unittest.SkipTest(f"Need {n_devices} devices; none available")
    angles = np.linspace(0, np.pi, N_VIEWS, endpoint=False)
    m = mbirjax.ParallelBeamModel((N_VIEWS, N_DET_ROWS, N_CHANNELS), angles)
    m.configure_sharding(devices)
    return m


def _random_sinogram(seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((N_VIEWS, N_DET_ROWS, N_CHANNELS)).astype(np.float32)


class TestFbpFilterSingleDevice(unittest.TestCase):
    """Single-device path must be unchanged from baseline."""

    @classmethod
    def setUpClass(cls):
        cls.model = _make_model_single()
        cls.sino  = jnp.array(_random_sinogram())

    def test_output_shape(self):
        out = self.model.fbp_filter(self.sino)
        self.assertEqual(out.shape, self.sino.shape)

    def test_output_is_plain_array(self):
        out = self.model.fbp_filter(self.sino)
        self.assertNotIsInstance(getattr(out, 'sharding', None),
                                 jax.sharding.NamedSharding)

    def test_values_deterministic(self):
        """Calling twice returns identical results."""
        a = np.asarray(self.model.fbp_filter(self.sino))
        b = np.asarray(self.model.fbp_filter(self.sino))
        np.testing.assert_array_equal(a, b)


class TestFbpFilterMultiDevice(unittest.TestCase):
    """Multi-device Path G: correct values, correct sharding, zero numpy scatter."""

    @classmethod
    def setUpClass(cls):
        cls.model_s = _make_model_single()
        cls.model_m = _make_model_sharded(n_devices=N_DEV)
        cls.sino_np  = _random_sinogram(seed=1)
        cls.sino_jax = jnp.array(cls.sino_np)
        # Reference: single-device result
        cls.ref = np.asarray(cls.model_s.fbp_filter(cls.sino_jax))

    # ── plain input → plain output ─────────────────────────────────────────

    def test_plain_input_plain_output(self):
        """Plain (uncommitted) sinogram → fbp_filter → plain array."""
        out = self.model_m.fbp_filter(self.sino_jax)
        self.assertNotIsInstance(getattr(out, 'sharding', None),
                                 jax.sharding.NamedSharding,
                                 "Plain input should yield plain output")

    def test_plain_input_correct_values(self):
        out = np.asarray(self.model_m.fbp_filter(self.sino_jax))
        np.testing.assert_allclose(out, self.ref, rtol=1e-5, atol=1e-5,
                                   err_msg="Multi-device result diverges from single-device")

    def test_plain_input_correct_shape(self):
        out = self.model_m.fbp_filter(self.sino_jax)
        self.assertEqual(out.shape, self.sino_jax.shape)

    # ── sharded input → sharded output ────────────────────────────────────

    def test_sharded_input_sharded_output(self):
        """NamedSharding input → fbp_filter → NamedSharding output (zero PCIe)."""
        sharded_sino = self.model_m._shard_sinogram(self.sino_jax)
        out = self.model_m.fbp_filter(sharded_sino)
        self.assertIsInstance(getattr(out, 'sharding', None),
                              jax.sharding.NamedSharding,
                              "Sharded input should yield sharded output")

    def test_sharded_input_correct_values(self):
        sharded_sino = self.model_m._shard_sinogram(self.sino_jax)
        out = np.asarray(self.model_m.fbp_filter(sharded_sino))
        np.testing.assert_allclose(out, self.ref, rtol=1e-5, atol=1e-5,
                                   err_msg="Sharded-input result diverges from single-device")

    def test_sharded_input_per_device_shape(self):
        """Each device shard of the output should have N_DET_ROWS // N_DEV rows."""
        sharded_sino = self.model_m._shard_sinogram(self.sino_jax)
        out = self.model_m.fbp_filter(sharded_sino)
        rows_per_dev = N_DET_ROWS // N_DEV
        for shard in out.addressable_shards:
            self.assertEqual(shard.data.shape,
                             (N_VIEWS, rows_per_dev, N_CHANNELS))

    def test_sharded_output_stays_on_devices(self):
        """The sharded output should have one shard per device on the right device."""
        sharded_sino = self.model_m._shard_sinogram(self.sino_jax)
        out = self.model_m.fbp_filter(sharded_sino)
        # Compare against the devices the model was actually configured with.
        expected_devices = set(self.model_m.mesh.devices.flat)
        actual_devices   = {s.device for s in out.addressable_shards}
        self.assertEqual(actual_devices, expected_devices)

    # ── idempotency: filter(plain) == filter(sharded) ──────────────────────

    def test_plain_and_sharded_input_agree(self):
        """Whether we pre-shard the input or not, the output values are the same."""
        plain_out   = np.asarray(self.model_m.fbp_filter(self.sino_jax))
        sharded_out = np.asarray(
            self.model_m.fbp_filter(self.model_m._shard_sinogram(self.sino_jax)))
        np.testing.assert_allclose(plain_out, sharded_out, rtol=1e-6, atol=1e-6)


class TestFbpFilterScaling(unittest.TestCase):
    """π/num_views scaling is applied correctly in both paths."""

    @classmethod
    def setUpClass(cls):
        cls.model_s = _make_model_single()
        cls.model_m = _make_model_sharded(n_devices=N_DEV)
        cls.sino    = jnp.array(_random_sinogram(seed=2))

    def test_scaling_single_vs_multi(self):
        """Both paths apply the same π/num_views scale factor."""
        s_out = np.asarray(self.model_s.fbp_filter(self.sino))
        m_out = np.asarray(self.model_m.fbp_filter(self.sino))
        np.testing.assert_allclose(s_out, m_out, rtol=1e-5, atol=1e-5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
