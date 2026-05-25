"""
Tests for Step 3: multi-device back_project via threaded sparse_back_project.

Design
------
Slice-level parallelism is added to ParallelBeamModel via two new methods:

  _prepare_sinogram_for_backprojection(sinogram)
      Hook called by TomographyModel.back_project before sparse_back_project.
      Shards sinogram along det_rows (axis 1) when a mesh is configured.
      Base class is a no-op.

  sparse_back_project(sinogram, pixel_indices, ...)
      ParallelBeamModel override.  When sinogram is NamedSharding, runs one
      thread per device on the local sinogram shard and assembles a sharded
      flat_recon.  Falls through to super() for plain inputs.

TomographyModel.back_project is updated to:
  - call _prepare_sinogram_for_backprojection before sparse_back_project
  - call _gather_recon on the result (no-op if plain; gathers if sharded)
  - re-shard the 3-D recon if the input sinogram was sharded

Output sharding of back_project mirrors input sharding:
  plain sinogram  → plain recon   (multi-device used internally, result gathered)
  sharded sinogram → sharded recon

Verifies:
  sparse_back_project
    - Plain input falls through to single-device path.
    - Sharded input takes threading path, returns NamedSharding flat_recon.
    - Values match single-device reference.

  back_project
    - Single-device: shape correct, values deterministic (regression check).
    - Multi-device plain input:  correct values, plain recon returned.
    - Multi-device sharded input: correct values, sharded recon returned.

  Round-trip (forward + back)
    - Shape matches recon.
    - Single- and multi-device paths agree numerically.

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
# sparse_back_project — plain input falls through to base class
# ===========================================================================

class TestSparseBackProjectPlain(unittest.TestCase):
    """Plain sinogram → base-class single-device path, plain flat_recon returned."""

    @classmethod
    def setUpClass(cls):
        cls.model_s = _make_model_single()
        cls.model_m = _make_model_sharded(n_devices=N_DEV)
        cls.sino_np  = _random_sinogram(seed=0)
        cls.sino_jax = jnp.array(cls.sino_np)
        recon_shape  = cls.model_s.get_params('recon_shape')
        cls.pixel_indices = mbirjax.gen_full_indices(
            recon_shape, use_ror_mask=cls.model_s.use_ror_mask)
        cls.ref = np.asarray(
            cls.model_s.sparse_back_project(cls.sino_jax, cls.pixel_indices))

    def test_output_shape(self):
        recon_shape = self.model_m.get_params('recon_shape')
        num_pixels = len(self.pixel_indices)
        num_slices = recon_shape[2]
        self.assertEqual(self.ref.shape, (num_pixels, num_slices))

    def test_plain_input_plain_output(self):
        out = self.model_m.sparse_back_project(self.sino_jax, self.pixel_indices)
        self.assertNotIsInstance(getattr(out, 'sharding', None),
                                 jax.sharding.NamedSharding,
                                 "plain sinogram should return plain flat_recon")

    def test_correct_values(self):
        out = np.asarray(self.model_m.sparse_back_project(self.sino_jax, self.pixel_indices))
        np.testing.assert_allclose(out, self.ref, rtol=RTOL, atol=ATOL)


# ===========================================================================
# sparse_back_project — sharded input takes threading path
# ===========================================================================

class TestSparseBackProjectSharded(unittest.TestCase):
    """Sharded sinogram → threading path, sharded flat_recon returned."""

    @classmethod
    def setUpClass(cls):
        cls.model_s = _make_model_single()
        cls.model_m = _make_model_sharded(n_devices=N_DEV)
        cls.sino_np  = _random_sinogram(seed=1)
        cls.sino_jax = jnp.array(cls.sino_np)
        recon_shape  = cls.model_s.get_params('recon_shape')
        cls.pixel_indices = mbirjax.gen_full_indices(
            recon_shape, use_ror_mask=cls.model_s.use_ror_mask)
        cls.ref = np.asarray(
            cls.model_s.sparse_back_project(cls.sino_jax, cls.pixel_indices))
        cls.sharded_sino = cls.model_m._shard_sinogram(cls.sino_jax)

    def test_sharded_input_sharded_output(self):
        out = self.model_m.sparse_back_project(self.sharded_sino, self.pixel_indices)
        self.assertIsInstance(getattr(out, 'sharding', None),
                              jax.sharding.NamedSharding,
                              "sharded sinogram should return sharded flat_recon")

    def test_output_shape(self):
        out = self.model_m.sparse_back_project(self.sharded_sino, self.pixel_indices)
        recon_shape = self.model_m.get_params('recon_shape')
        num_pixels = len(self.pixel_indices)
        num_slices = recon_shape[2]
        self.assertEqual(out.shape, (num_pixels, num_slices))

    def test_correct_values(self):
        out = np.asarray(
            self.model_m.sparse_back_project(self.sharded_sino, self.pixel_indices))
        np.testing.assert_allclose(out, self.ref, rtol=RTOL, atol=ATOL,
                                   err_msg="Sharded sparse_back_project diverges from single-device")

    def test_deterministic(self):
        a = np.asarray(self.model_m.sparse_back_project(self.sharded_sino, self.pixel_indices))
        b = np.asarray(self.model_m.sparse_back_project(self.sharded_sino, self.pixel_indices))
        np.testing.assert_array_equal(a, b)


# ===========================================================================
# back_project — single-device (regression check)
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


# ===========================================================================
# back_project — multi-device
# ===========================================================================

class TestBackProjectMultiDevice(unittest.TestCase):
    """
    back_project with a mesh configured:
      - Plain sinogram input  → multi-device used internally, plain recon returned.
      - Sharded sinogram input → multi-device, sharded recon returned.
    Output sharding mirrors input sharding; values always match single-device reference.
    """

    @classmethod
    def setUpClass(cls):
        cls.model_s = _make_model_single()
        cls.model_m = _make_model_sharded(n_devices=N_DEV)
        cls.sino_np  = _random_sinogram(seed=2)
        cls.sino_jax = jnp.array(cls.sino_np)
        cls.ref = np.asarray(cls.model_s.back_project(cls.sino_jax))

    def test_output_shape(self):
        out = self.model_m.back_project(self.sino_jax)
        recon_shape = self.model_s.get_params('recon_shape')
        self.assertEqual(out.shape, recon_shape)

    def test_plain_input_plain_output(self):
        """Plain sinogram → multi-device used internally, output gathered to plain recon."""
        out = self.model_m.back_project(self.sino_jax)
        self.assertNotIsInstance(getattr(out, 'sharding', None),
                                 jax.sharding.NamedSharding,
                                 "back_project with plain sinogram should return plain recon")

    def test_sharded_input_sharded_output(self):
        """Sharded sinogram → sharded recon (output sharding mirrors input)."""
        sharded_sino = self.model_m._shard_sinogram(self.sino_jax)
        out = self.model_m.back_project(sharded_sino)
        self.assertIsInstance(getattr(out, 'sharding', None),
                              jax.sharding.NamedSharding,
                              "back_project with sharded sinogram should return sharded recon")

    def test_correct_values(self):
        out = np.asarray(self.model_m.back_project(self.sino_jax))
        np.testing.assert_allclose(out, self.ref, rtol=RTOL, atol=ATOL,
                                   err_msg="Multi-device back_project diverges from single-device")

    def test_sharded_input_correct_values(self):
        """Sharded sinogram path also produces correct values."""
        sharded_sino = self.model_m._shard_sinogram(self.sino_jax)
        out = np.asarray(self.model_m.back_project(sharded_sino))
        np.testing.assert_allclose(out, self.ref, rtol=RTOL, atol=ATOL,
                                   err_msg="Sharded-input back_project diverges from single-device")

    def test_deterministic(self):
        a = np.asarray(self.model_m.back_project(self.sino_jax))
        b = np.asarray(self.model_m.back_project(self.sino_jax))
        np.testing.assert_array_equal(a, b)


# ===========================================================================
# Round-trip shape and value consistency
# ===========================================================================

class TestRoundTrip(unittest.TestCase):
    """
    AT(A(x)) shape must match x, and single- and multi-device paths must agree.

    forward_project with plain recon → plain sinogram (multi-device internally).
    back_project with plain sinogram → plain recon (multi-device internally).
    Both directions now use threading when a mesh is configured.
    """

    @classmethod
    def setUpClass(cls):
        cls.model_s = _make_model_single()
        cls.model_m = _make_model_sharded(n_devices=N_DEV)
        cls.recon   = jnp.array(_random_recon(cls.model_s, seed=3))

    def _round_trip(self, model):
        sino = model.forward_project(self.recon)   # plain recon → plain sinogram
        return model.back_project(sino)             # plain sinogram → plain recon

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
