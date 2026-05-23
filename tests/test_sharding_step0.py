"""
Tests for Step 0 of the multi-GPU sharding infrastructure.

Covers:
  0a. _maybe_gather — no-op when mesh is None; true gather when mesh is set
  0b. Geometry-specific hook methods — no-op / correct values in single-device and
      multi-device modes; NotImplementedError in base class when mesh is set
  0c. set_devices_and_batch_sizes integration — 'sharded' mode, geometry-change
      warning, no circular calls
  0d. Docstrings for sparse_forward_project / sparse_back_project contain the
      multi-device sharding note

Run with::

    python -m pytest tests/test_sharding_step0.py -v

conftest.py sets XLA_FLAGS=--xla_force_host_platform_device_count=4 (CPU fallback)
and provides preferred_devices() which selects real GPUs when available.
"""
import warnings
import unittest
import numpy as np
import jax
import jax.numpy as jnp
import mbirjax
import mbirjax.tomography_model as tm
from conftest import preferred_devices

# ---------------------------------------------------------------------------
# Verify that conftest.py took effect.
# On any machine, conftest.py guarantees ≥4 virtual CPU devices via XLA_FLAGS.
# Real GPUs (if present) are also available and will be preferred by
# preferred_devices(), but the CPU count assertion is the safest fallback guard.
# ---------------------------------------------------------------------------
_N_VIRTUAL_CPU_DEVICES = len(jax.devices('cpu'))
assert _N_VIRTUAL_CPU_DEVICES >= 4, (
    f"Expected ≥4 virtual CPU devices (set by conftest.py via XLA_FLAGS), "
    f"but JAX reports only {_N_VIRTUAL_CPU_DEVICES}. "
    f"Run this file in isolation or ensure conftest.py is loaded before JAX."
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
N_VIEWS       = 12   # small; sharding tests care about rows/slices, not views
N_DET_ROWS    = 8    # divisible by 2 (and 4) for clean per-device splits
N_DET_CHANS   = 16
N_RECON_ROWS  = 8
N_RECON_COLS  = 8
N_SLICES      = 8    # == N_DET_ROWS for parallel beam consistency


def _make_model(use_gpu='none'):
    """Return a small ParallelBeamModel in single-device mode."""
    angles = np.linspace(0, np.pi, N_VIEWS, endpoint=False)
    model = mbirjax.ParallelBeamModel(
        (N_VIEWS, N_DET_ROWS, N_DET_CHANS), angles)
    model.set_params(use_gpu=use_gpu)
    model.set_devices_and_batch_sizes()
    return model


def _make_sharded_model(n_devices=2):
    """Return a ParallelBeamModel configured with n_devices (GPU if available)."""
    devices = preferred_devices(n_devices)
    if devices is None:
        raise unittest.SkipTest(f"Need {n_devices} devices; none available")
    angles = np.linspace(0, np.pi, N_VIEWS, endpoint=False)
    model = mbirjax.ParallelBeamModel(
        (N_VIEWS, N_DET_ROWS, N_DET_CHANS), angles)
    model.configure_sharding(devices)
    return model, devices


# ---------------------------------------------------------------------------
# 0a  _maybe_gather
# ---------------------------------------------------------------------------
class TestMaybeGatherSingleDevice(unittest.TestCase):
    """_maybe_gather is a no-op when mesh is None."""

    def test_returns_same_object(self):
        model = _make_model()
        x = jnp.ones((4, 8))
        self.assertIs(model._maybe_gather(x), x)


class TestMaybeGatherMultiDevice(unittest.TestCase):
    """_maybe_gather correctly collects a sharded array (GPU if available, else CPU)."""

    @classmethod
    def setUpClass(cls):
        cls.model, cls.devices = _make_sharded_model(n_devices=2)

    def test_gather_produces_uncommitted_array(self):
        sino = jnp.ones((N_VIEWS, N_DET_ROWS, N_DET_CHANS))
        sharded = self.model._shard_sinogram(sino)
        gathered = self.model._maybe_gather(sharded)
        self.assertNotIsInstance(
            getattr(gathered, 'sharding', None), jax.sharding.NamedSharding,
            "_maybe_gather result should not be NamedSharding")

    def test_gather_preserves_values(self):
        rng = np.random.default_rng(0)
        data = rng.standard_normal((N_VIEWS, N_DET_ROWS, N_DET_CHANS)).astype(np.float32)
        sino = jnp.array(data)
        sharded = self.model._shard_sinogram(sino)
        gathered = self.model._maybe_gather(sharded)
        np.testing.assert_array_equal(np.asarray(gathered), data)


# ---------------------------------------------------------------------------
# 0b  Hook methods — single-device mode
# ---------------------------------------------------------------------------
class TestHookMethodsSingleDevice(unittest.TestCase):
    """All hook methods are no-ops (return the same object) in single-device mode."""

    @classmethod
    def setUpClass(cls):
        cls.model    = _make_model()
        cls.sino     = jnp.zeros((N_VIEWS, N_DET_ROWS, N_DET_CHANS))
        cls.recon_3d = jnp.zeros((N_RECON_ROWS, N_RECON_COLS, N_SLICES))
        cls.flat_rec = jnp.zeros((N_RECON_ROWS * N_RECON_COLS, N_SLICES))

    def test_shard_sinogram_noop(self):
        self.assertIs(self.model._shard_sinogram(self.sino), self.sino)

    def test_gather_sinogram_noop(self):
        self.assertIs(self.model._gather_sinogram(self.sino), self.sino)

    def test_shard_recon_3d_noop(self):
        self.assertIs(self.model._shard_recon(self.recon_3d), self.recon_3d)

    def test_shard_recon_flat_noop(self):
        self.assertIs(self.model._shard_recon(self.flat_rec), self.flat_rec)

    def test_gather_recon_noop(self):
        self.assertIs(self.model._gather_recon(self.recon_3d), self.recon_3d)

    def test_extract_halos_returns_none_sentinel(self):
        lh, rh = self.model._extract_halos(self.flat_rec)
        self.assertEqual(lh, [None])
        self.assertEqual(rh, [None])


# ---------------------------------------------------------------------------
# 0b  Hook methods — multi-device mode
# ---------------------------------------------------------------------------
class TestHookMethodsMultiDevice(unittest.TestCase):
    """Hook methods shard / gather correctly with real (virtual) devices."""

    N_DEV = 2

    @classmethod
    def setUpClass(cls):
        cls.model, cls.devices = _make_sharded_model(n_devices=cls.N_DEV)
        cls.rows_per_dev = N_DET_ROWS // cls.N_DEV   # 4
        cls.slices_per_dev = N_SLICES // cls.N_DEV   # 4

    # -- sinogram sharding ---------------------------------------------------

    def test_shard_sinogram_is_named_sharding(self):
        sino = jnp.ones((N_VIEWS, N_DET_ROWS, N_DET_CHANS))
        sharded = self.model._shard_sinogram(sino)
        self.assertIsInstance(getattr(sharded, 'sharding', None),
                              jax.sharding.NamedSharding)

    def test_shard_sinogram_per_device_shape(self):
        sino = jnp.ones((N_VIEWS, N_DET_ROWS, N_DET_CHANS))
        sharded = self.model._shard_sinogram(sino)
        for shard in sharded.addressable_shards:
            self.assertEqual(shard.data.shape,
                             (N_VIEWS, self.rows_per_dev, N_DET_CHANS))

    def test_shard_sinogram_preserves_values(self):
        data = np.arange(N_VIEWS * N_DET_ROWS * N_DET_CHANS,
                         dtype=np.float32).reshape(N_VIEWS, N_DET_ROWS, N_DET_CHANS)
        sharded = self.model._shard_sinogram(jnp.array(data))
        np.testing.assert_array_equal(np.asarray(sharded), data)

    def test_shard_sinogram_idempotent(self):
        """Sharding an already-sharded array is a no-op (no re-scatter)."""
        sino = jnp.ones((N_VIEWS, N_DET_ROWS, N_DET_CHANS))
        s1 = self.model._shard_sinogram(sino)
        s2 = self.model._shard_sinogram(s1)
        self.assertIs(s2, s1)

    # -- 3-D recon sharding --------------------------------------------------

    def test_shard_recon_3d_is_named_sharding(self):
        recon = jnp.ones((N_RECON_ROWS, N_RECON_COLS, N_SLICES))
        sharded = self.model._shard_recon(recon)
        self.assertIsInstance(getattr(sharded, 'sharding', None),
                              jax.sharding.NamedSharding)

    def test_shard_recon_3d_per_device_shape(self):
        recon = jnp.ones((N_RECON_ROWS, N_RECON_COLS, N_SLICES))
        sharded = self.model._shard_recon(recon)
        for shard in sharded.addressable_shards:
            self.assertEqual(shard.data.shape,
                             (N_RECON_ROWS, N_RECON_COLS, self.slices_per_dev))

    # -- flat recon sharding -------------------------------------------------

    def test_shard_recon_flat_per_device_shape(self):
        flat = jnp.ones((N_RECON_ROWS * N_RECON_COLS, N_SLICES))
        sharded = self.model._shard_recon(flat)
        for shard in sharded.addressable_shards:
            self.assertEqual(shard.data.shape,
                             (N_RECON_ROWS * N_RECON_COLS, self.slices_per_dev))

    # -- gather roundtrip ----------------------------------------------------

    def test_gather_sinogram_roundtrip(self):
        data = np.random.default_rng(1).standard_normal(
            (N_VIEWS, N_DET_ROWS, N_DET_CHANS)).astype(np.float32)
        sino = jnp.array(data)
        gathered = self.model._gather_sinogram(self.model._shard_sinogram(sino))
        self.assertNotIsInstance(getattr(gathered, 'sharding', None),
                                 jax.sharding.NamedSharding)
        np.testing.assert_array_equal(np.asarray(gathered), data)

    def test_gather_recon_roundtrip(self):
        data = np.random.default_rng(2).standard_normal(
            (N_RECON_ROWS, N_RECON_COLS, N_SLICES)).astype(np.float32)
        recon = jnp.array(data)
        gathered = self.model._gather_recon(self.model._shard_recon(recon))
        self.assertNotIsInstance(getattr(gathered, 'sharding', None),
                                 jax.sharding.NamedSharding)
        np.testing.assert_array_equal(np.asarray(gathered), data)

    # -- halo extraction -----------------------------------------------------

    def test_extract_halos_list_lengths(self):
        flat = jnp.ones((N_RECON_ROWS * N_RECON_COLS, N_SLICES))
        sharded = self.model._shard_recon(flat)
        lh, rh = self.model._extract_halos(sharded)
        self.assertEqual(len(lh), self.N_DEV)
        self.assertEqual(len(rh), self.N_DEV)

    def test_extract_halos_boundary_entries_are_none(self):
        """First left halo and last right halo must be None (reflected BC)."""
        flat = jnp.ones((N_RECON_ROWS * N_RECON_COLS, N_SLICES))
        sharded = self.model._shard_recon(flat)
        lh, rh = self.model._extract_halos(sharded)
        self.assertIsNone(lh[0],  "left_halos[0] should be None (no left neighbour)")
        self.assertIsNone(rh[-1], "right_halos[-1] should be None (no right neighbour)")

    def test_extract_halos_values_correct(self):
        """Halo slices must match the actual boundary columns of adjacent shards.

        With N_SLICES=8 and N_DEV=2:
          device 0 holds global slices 0..3  (local columns 0..3)
          device 1 holds global slices 4..7  (local columns 0..3)

        Halo semantics (ghost slices for qggmrf):
          right_halos[0] = ghost to the RIGHT of device 0
                         = first slice of device 1 = global slice 4, value 4.0
          left_halos[1]  = ghost to the LEFT of device 1
                         = last slice of device 0  = global slice 3, value 3.0
        """
        n_pixels = N_RECON_ROWS * N_RECON_COLS
        # flat_recon value at column j = float(j) for all pixels
        data = np.broadcast_to(
            np.arange(N_SLICES, dtype=np.float32),
            (n_pixels, N_SLICES)).copy()
        flat = jnp.array(data)
        sharded = self.model._shard_recon(flat)
        lh, rh = self.model._extract_halos(sharded)

        # right_halos[0]: first slice of device 1 → global index = slices_per_dev = 4
        expected_right = np.full(n_pixels, float(self.slices_per_dev), dtype=np.float32)
        np.testing.assert_array_equal(
            rh[0], expected_right,
            err_msg="right_halos[0] should be the first slice of device 1's shard "
                    f"(global slice {self.slices_per_dev})")

        # left_halos[1]: last slice of device 0 → global index = slices_per_dev - 1 = 3
        expected_left = np.full(n_pixels, float(self.slices_per_dev - 1), dtype=np.float32)
        np.testing.assert_array_equal(
            lh[1], expected_left,
            err_msg="left_halos[1] should be the last slice of device 0's shard "
                    f"(global slice {self.slices_per_dev - 1})")

    def test_extract_halos_inner_halos_are_numpy(self):
        """Interior halos must be numpy arrays (not JAX), ready for per-device upload."""
        flat = jnp.ones((N_RECON_ROWS * N_RECON_COLS, N_SLICES))
        sharded = self.model._shard_recon(flat)
        lh, rh = self.model._extract_halos(sharded)
        for i, h in enumerate(lh):
            if h is not None:
                self.assertIsInstance(h, np.ndarray, f"left_halos[{i}] should be numpy")
        for i, h in enumerate(rh):
            if h is not None:
                self.assertIsInstance(h, np.ndarray, f"right_halos[{i}] should be numpy")


# ---------------------------------------------------------------------------
# 0b  Base class hooks raise NotImplementedError when mesh is set
# ---------------------------------------------------------------------------
class TestBaseClassHooksRaiseWhenMeshSet(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.base = tm.TomographyModel.__new__(tm.TomographyModel)
        cls.base.mesh = object()   # non-None sentinel; bypasses device setup
        cls.sino     = jnp.zeros((4, 8, 8))
        cls.recon    = jnp.zeros((8, 8, 8))
        cls.flat_rec = jnp.zeros((64, 8))

    def test_shard_sinogram_raises(self):
        with self.assertRaises(NotImplementedError):
            tm.TomographyModel._shard_sinogram(self.base, self.sino)

    def test_shard_recon_raises(self):
        with self.assertRaises(NotImplementedError):
            tm.TomographyModel._shard_recon(self.base, self.recon)

    def test_extract_halos_raises(self):
        with self.assertRaises(NotImplementedError):
            tm.TomographyModel._extract_halos(self.base, self.flat_rec)


# ---------------------------------------------------------------------------
# 0c  set_devices_and_batch_sizes / configure_sharding
# ---------------------------------------------------------------------------
class TestShardedModeSelection(unittest.TestCase):

    def test_verify_valid_params_accepts_sharded(self):
        # set_params(use_gpu='sharded') without configure_sharding() triggers the
        # GPU auto-detect path, which correctly warns that no GPUs were found.
        # Suppress those expected warnings so they don't pollute test output.
        model = _make_model()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            model.set_params(use_gpu='sharded')
        try:
            model.verify_valid_params()
        except Exception as e:
            self.fail(f"verify_valid_params raised for use_gpu='sharded': {e}")

    def test_geometry_change_warning(self):
        """Switching from a sharded state to single-device must warn."""
        model = _make_model(use_gpu='none')
        # Inject a non-None mesh *after* set_params so the warning fires inside
        # our capture context rather than inside set_params itself.
        model.mesh = object()
        model._explicit_shard_devices = None
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            model.set_devices_and_batch_sizes()
        matched = [x for x in w if 'switched from multi-device' in str(x.message)]
        self.assertTrue(matched,
                        "Expected 'switched from multi-device' warning; "
                        "got: " + str([str(x.message) for x in w]))

    def test_configure_sharding_none_no_infinite_recursion(self):
        model = _make_model()
        try:
            model.configure_sharding(None)
        except RecursionError:
            self.fail("configure_sharding(None) caused infinite recursion")

    def test_explicit_shard_devices_initialised_to_none(self):
        model = _make_model()
        self.assertIsNone(model._explicit_shard_devices)


class TestConfigureShadingWithVirtualDevices(unittest.TestCase):
    """configure_sharding sets mesh, clears device attributes, selects 'sharded'."""

    N_DEV = 2

    @classmethod
    def setUpClass(cls):
        cls.model, cls.devices = _make_sharded_model(n_devices=cls.N_DEV)

    def test_mesh_is_set(self):
        self.assertIsNotNone(self.model.mesh)

    def test_main_device_is_none(self):
        self.assertIsNone(self.model.main_device)

    def test_sinogram_device_is_none(self):
        self.assertIsNone(self.model.sinogram_device)

    def test_use_gpu_is_sharded(self):
        # self.use_gpu is the *actual* mode set by set_devices_and_batch_sizes;
        # get_params('use_gpu') returns the user-requested value in the params dict
        # (e.g. 'automatic'), which may differ.  The active mode is what matters here.
        self.assertEqual(self.model.use_gpu, 'sharded')

    def test_mesh_device_count(self):
        self.assertEqual(self.model.mesh.devices.size, self.N_DEV)

    def test_explicit_devices_stored(self):
        self.assertEqual(list(self.model._explicit_shard_devices),
                         list(self.devices))


# ---------------------------------------------------------------------------
# 0d  Docstring notes
# ---------------------------------------------------------------------------
class TestDocstringNotes(unittest.TestCase):

    def test_sparse_forward_project_has_multidevice_note(self):
        doc = mbirjax.TomographyModel.sparse_forward_project.__doc__ or ''
        self.assertIn('multi-device', doc.lower())
        self.assertIn('NamedSharding', doc)

    def test_sparse_back_project_has_multidevice_note(self):
        doc = mbirjax.TomographyModel.sparse_back_project.__doc__ or ''
        self.assertIn('multi-device', doc.lower())
        self.assertIn('NamedSharding', doc)


if __name__ == '__main__':
    unittest.main(verbosity=2)
