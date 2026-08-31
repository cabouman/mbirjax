"""
Fast CPU tests for the 4D MACE model.  No GPU required.

Run from the repo root:  python -m pytest tests/test_mace4d.py
"""
import csv
import os
import tempfile
import unittest
import warnings

import jax
import jax.numpy as jnp
import numpy as np
from scipy.fft import dct

import mbirjax as mj
from mbirjax.mace4d import (
    _DENOISE_COST_PER_PLANE,
    _DENOISE_MAX_ITERATIONS,
    _DENOISE_STOP_THRESHOLD_PCT,
    _assign_tasks,
    _batched_hyperplane_denoise,
    _configure_denoiser,
    _dejitter_4d_dct,
    _denoise_constants,
    _denoiser_wrapper,
    _get_qggmrf_denoiser,
    _normalize_prior_weights,
    _resolve_devices,
)

NUM_VIEWS = 24          # 24 views over 360 degrees -> 15 degrees per view
DET_ROWS = 8
DET_COLS = 10


def _small_model():
    """Tiny cone-beam model with evenly spaced angles over 360 degrees."""
    angles = np.radians(360.0 / NUM_VIEWS) * np.arange(NUM_VIEWS)
    return mj.ConeBeamModel(sinogram_shape=(NUM_VIEWS, DET_ROWS, DET_COLS), angles=angles,
                            source_detector_dist=100.0, source_iso_dist=50.0)


def _smooth_sino(num_views=NUM_VIEWS, rows=DET_ROWS, cols=DET_COLS):
    """A smooth, positive synthetic sinogram.

    A random sinogram reconstructs into a volume with extreme values on which the qGGMRF line
    search can hit 0/0 for some randomly drawn pixel partitions.
    """
    v = np.linspace(-1.0, 1.0, rows)
    c = np.linspace(-1.0, 1.0, cols)
    base = np.exp(-(v[:, None] ** 2 + c[None, :] ** 2))
    return np.stack([base * (1.0 + 0.1 * np.sin(0.5 * a))
                     for a in range(num_views)]).astype(np.float32)


class TestDejitter(unittest.TestCase):
    """DCT-I temporal dejitter."""

    def test_dejitter_zeroes_target_bands_only(self):
        print('Testing that dejitter zeroes the jitter bands and leaves the rest untouched')
        rng = np.random.default_rng(1)
        num_frames, period, band_width = 49, 6, 1
        vol = rng.normal(size=(num_frames, 3, 4, 5)).astype(np.float32)
        out = _dejitter_4d_dct(vol, period=period, verbose=False)

        coefs_in = dct(vol.reshape(num_frames, -1), type=1, norm='ortho', axis=0)
        coefs_out = dct(out.reshape(num_frames, -1), type=1, norm='ortho', axis=0)
        zeroed = set()
        for h in range(1, period // 2 + 1):
            k0 = int(round(2 * (num_frames - 1) / (period / h)))
            zeroed.update(range(max(0, k0 - band_width), min(num_frames, k0 + band_width + 1)))
        kept = sorted(set(range(num_frames)) - zeroed)
        self.assertLess(np.abs(coefs_out[sorted(zeroed)]).max(), 1e-5)
        self.assertTrue(np.allclose(coefs_out[kept], coefs_in[kept], atol=1e-4))

    def test_dejitter_chunked_equals_whole(self):
        print('Testing that chunked dejitter matches the single-pass result')
        rng = np.random.default_rng(2)
        vol = rng.normal(size=(30, 4, 5, 6)).astype(np.float32)
        whole = _dejitter_4d_dct(vol, period=6, verbose=False)
        chunked = _dejitter_4d_dct(vol, period=6, chunk_size=2, verbose=False)
        self.assertTrue(np.allclose(whole, chunked, atol=1e-5))


class TestPriorWeights(unittest.TestCase):
    """Agent weight normalization."""

    def test_scalar_and_list(self):
        print('Testing scalar and per-orientation prior weights')
        self.assertTrue(np.allclose(_normalize_prior_weights(0.5),
                                    [0.5, 0.5 / 3, 0.5 / 3, 0.5 / 3]))
        self.assertTrue(np.allclose(_normalize_prior_weights([0.1, 0.2, 0.3]),
                                    [0.4, 0.1, 0.2, 0.3]))

    def test_invalid_raises(self):
        print('Testing that out-of-range prior weights raise')
        for bad in (1.5, -0.1, [0.5, 0.5, 0.5], [0.1, 0.2]):
            with self.assertRaises(ValueError):
                _normalize_prior_weights(bad)


class TestTaskAssignment(unittest.TestCase):
    """Static least-loaded task-to-device assignment."""

    def test_assign_tasks_balance(self):
        print('Testing that tasks are balanced across four devices')
        plane_counts = [192, 65, 65]
        frame_device, orient_device = _assign_tasks(25, plane_counts, 4)
        self.assertEqual(len(frame_device), 25)
        self.assertEqual(len(orient_device), 3)
        self.assertTrue(set(frame_device) | set(orient_device) <= {0, 1, 2, 3})
        # The three denoise tasks land on three different devices.
        self.assertEqual(len(set(orient_device)), 3)
        # Device loads (prox = 1, denoise = cost) end up within one prox task.
        loads = [0.0] * 4
        for k, d in enumerate(orient_device):
            loads[d] += _DENOISE_COST_PER_PLANE * plane_counts[k]
        for d in frame_device:
            loads[d] += 1.0
        self.assertLessEqual(max(loads) - min(loads), 1.0 + 1e-9)

    def test_assign_tasks_single_device(self):
        print('Testing that a single device takes every task')
        frame_device, orient_device = _assign_tasks(5, [8, 6, 7], 1)
        self.assertEqual(frame_device, [0] * 5)
        self.assertEqual(orient_device, [0] * 3)


class TestDeviceResolution(unittest.TestCase):
    """configure_devices argument forms."""

    def test_resolve_none_int_and_list(self):
        print('Testing device resolution for None, int and explicit device list')
        automatic = _resolve_devices(None)
        self.assertGreaterEqual(len(automatic), 1)

        self.assertEqual(len(_resolve_devices(1)), 1)

        explicit = jax.devices('cpu')[:1]
        self.assertEqual(_resolve_devices(explicit), list(explicit))
        self.assertEqual(_resolve_devices([0]), [jax.devices()[0]])
        self.assertEqual(_resolve_devices('cpu'), list(jax.devices('cpu')))

    def test_resolve_invalid_raises(self):
        print('Testing that an impossible device request raises')
        with self.assertRaises(ValueError):
            _resolve_devices(len(jax.devices()) + 1)
        with self.assertRaises(ValueError):
            _resolve_devices('tpu')

    def test_configure_devices_pins_the_choice(self):
        print('Testing that configure_devices pins the device list on the model')
        mace = mj.MACE4DModel(_small_model())
        self.assertEqual(len(mace.devices), len(_resolve_devices(None)))   # automatic by default
        mace.configure_devices(1)
        self.assertEqual(len(mace.devices), 1)


class TestConstruction(unittest.TestCase):
    """Frame structure available straight from the constructor."""

    def setUp(self):
        self.ct_model = _small_model()
        self.sinogram = _smooth_sino()

    def test_frames_from_model_alone(self):
        print('Testing that the constructor derives the frame structure from the model')
        mace = mj.MACE4DModel(self.ct_model, frames_per_rotation=6, frame_overlap_factor=2.0)
        # 120 degree frames advancing 60 degrees: views 0-7, 4-11, 8-15, 12-19, 16-23.
        self.assertEqual(mace.nt, 5)
        self.assertEqual(len(mace.model_list), 5)
        self.assertEqual(mace.view_slices[1], slice(4, 12))

    def test_view_slices_match_the_standalone_wrapper(self):
        """The model's frames must be the frames construct_time_frames would hand a user."""
        print('Testing that model view slices agree with construct_time_frames')
        mace = mj.MACE4DModel(self.ct_model, frames_per_rotation=6, frame_overlap_factor=2.0)
        sino_frames, _ = mj.construct_time_frames(self.sinogram, self.ct_model,
                                                  frames_per_rotation=6, frame_overlap_factor=2.0)
        self.assertEqual(len(sino_frames), mace.nt)
        for view_slice, sino_frame in zip(mace.view_slices, sino_frames):
            self.assertTrue(np.array_equal(self.sinogram[view_slice], sino_frame))

    def test_num_frames_truncates(self):
        print('Testing that num_frames keeps only the first frames')
        mace = mj.MACE4DModel(self.ct_model, num_frames=3)
        self.assertEqual(mace.nt, 3)
        self.assertEqual(len(mace.model_list), 3)
        self.assertEqual(mace.view_slices, [slice(0, 8), slice(4, 12), slice(8, 16)])

    def test_num_frames_above_the_total_uses_all(self):
        print('Testing that an oversized num_frames uses every frame')
        self.assertEqual(mj.MACE4DModel(self.ct_model, num_frames=99).nt, 5)

    def test_wrong_sinogram_shape_raises(self):
        print('Testing that a mismatched sinogram is rejected before any work is done')
        mace = mj.MACE4DModel(self.ct_model, num_frames=1)
        with self.assertRaises(ValueError):
            mace.recon(np.zeros((NUM_VIEWS, DET_ROWS, DET_COLS + 1), dtype=np.float32))
        with self.assertRaises(ValueError):
            mace.recon(self.sinogram, weights=np.ones((NUM_VIEWS, DET_ROWS, DET_COLS + 1)))


class TestParameters(unittest.TestCase):
    """Reconstruction parameters through set_params."""

    def setUp(self):
        self.mace = mj.MACE4DModel(_small_model(), num_frames=2)

    def test_defaults_and_updates(self):
        print('Testing MACE parameter defaults and updates')
        self.assertEqual(self.mace.get_params('mace_prior_weight'), 0.5)
        self.assertEqual(self.mace.get_params('rho_mann'), 0.5)
        self.assertEqual(self.mace.get_params('prox_num_iterations'), 3)
        self.assertEqual(self.mace.get_params('prox_stop_threshold'), 0.02)
        self.assertTrue(self.mace.get_params('dejitter'))
        self.assertIsNone(self.mace.get_params('sigma_prox'))

        self.mace.set_params(rho_mann=0.25, dejitter=False)
        self.assertEqual(self.mace.get_params('rho_mann'), 0.25)
        self.assertFalse(self.mace.get_params('dejitter'))

    def test_bad_prior_weight_rejected_at_set_time(self):
        print('Testing that an invalid prior weight is rejected by set_params')
        with self.assertRaises(ValueError):
            self.mace.set_params(mace_prior_weight=1.5)

    def test_unknown_parameter_rejected(self):
        print('Testing that an unknown parameter name is rejected')
        with self.assertRaises(ValueError):
            self.mace.set_params(not_a_parameter=1)

    def test_sigma_prox_does_not_warn_about_auto_regularization(self):
        """This model has no regularization of its own, so that base-class warning is noise."""
        print('Testing that setting sigma_prox is warning-free on the MACE model')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            self.mace.set_params(sigma_prox=0.1)
        self.assertEqual([w for w in caught if 'auto-regularization' in str(w.message)], [])
        self.assertEqual(self.mace.get_params('sigma_prox'), 0.1)


class TestReconEndToEnd(unittest.TestCase):
    """Serial reconstruction on one device."""

    def setUp(self):
        np.random.seed(0)   # the pixel partitions are drawn from numpy's global generator
        self.ct_model = _small_model()
        self.sinogram = _smooth_sino()
        # dejitter=False: a period-6 filter on a 3-frame time axis would zero the whole
        # temporal spectrum of this tiny test problem.
        self.mace = mj.MACE4DModel(self.ct_model, num_frames=3)
        self.mace.set_params(dejitter=False, verbose=0)
        self.mace.configure_devices(1)

    def test_recon_and_logs(self):
        print('Testing an end-to-end serial reconstruction with logging')
        with tempfile.TemporaryDirectory() as tmp_dir:
            init_dir = os.path.join(tmp_dir, 'init')
            log_dir = os.path.join(tmp_dir, 'logs')
            recon, recon_dict = self.mace.recon(self.sinogram, max_iterations=1,
                                                stop_threshold_change_pct=0,
                                                init_dir=init_dir, log_dir=log_dir)
            self.assertEqual(recon.shape, (3,) + self.mace.recon_shape)
            self.assertTrue(np.all(np.isfinite(recon)))
            for name in ('run_info.txt', 'timing_log.csv', 'task_log.csv'):
                self.assertTrue(os.path.isfile(os.path.join(log_dir, name)))
            self.assertEqual(sorted(recon_dict),
                             ['model_params', 'notes', 'recon_params', 'timing'])
            self.assertEqual(len(recon_dict['timing']), 1)
            self.assertEqual(recon_dict['recon_params']['iterations completed'], 1)
            self.assertEqual(recon_dict['recon_params']['weights'], 'unit (weights=None)')

            # The initialization is cached, so a second run must find it.
            self.assertTrue(os.path.isfile(os.path.join(init_dir, 'init_recon.npy')))
            recon_again, _ = self.mace.recon(self.sinogram, max_iterations=1,
                                             stop_threshold_change_pct=0, init_dir=init_dir)
            self.assertEqual(recon_again.shape, recon.shape)

    def test_weights_are_used_per_frame(self):
        """Explicit weights must reach the frames and change the result."""
        print('Testing explicit weights against the unit-weight default')
        weights = mj.gen_weights(self.sinogram, weight_type='transmission_root')
        with tempfile.TemporaryDirectory() as tmp_dir:
            init_dir = os.path.join(tmp_dir, 'init')   # shared init isolates the weight effect
            unweighted, _ = self.mace.recon(self.sinogram, max_iterations=1,
                                            stop_threshold_change_pct=0, init_dir=init_dir)
            weighted, recon_dict = self.mace.recon(self.sinogram, weights=weights,
                                                   max_iterations=1, stop_threshold_change_pct=0,
                                                   init_dir=init_dir)
        self.assertEqual(recon_dict['recon_params']['weights'], 'supplied by caller')
        self.assertTrue(np.all(np.isfinite(weighted)))
        self.assertFalse(np.allclose(weighted, unweighted))

    def test_stop_threshold_ends_the_loop_early(self):
        print('Testing that the consensus stopping threshold ends the outer loop')
        _, recon_dict = self.mace.recon(self.sinogram, max_iterations=5,
                                        stop_threshold_change_pct=1e9)
        self.assertEqual(len(recon_dict['timing']), 1)
        self.assertEqual(recon_dict['recon_params']['iterations completed'], 1)

    def test_init_recon_is_validated_and_used(self):
        print('Testing that a supplied initial image is validated and used')
        shape = (3,) + self.mace.recon_shape
        supplied = np.linspace(0.0, 0.1, int(np.prod(shape)), dtype=np.float32).reshape(shape)
        _, recon_dict = self.mace.recon(self.sinogram, init_recon=supplied, max_iterations=1,
                                        stop_threshold_change_pct=0)
        self.assertEqual(recon_dict['recon_params']['init source'], 'provided by caller')
        with self.assertRaises(ValueError):
            self.mace.recon(self.sinogram, init_recon=np.zeros((1, 2, 3, 4)))

    def test_constant_init_recon_reports_the_cause(self):
        """A constant initial image leaves nothing to estimate the denoiser sigma from."""
        print('Testing that a constant initial image gives an explanatory error')
        constant = np.zeros((3,) + self.mace.recon_shape, dtype=np.float32)
        with self.assertRaises(ValueError) as caught:
            self.mace.recon(self.sinogram, init_recon=constant, max_iterations=1)
        self.assertIn('constant', str(caught.exception))


class TestReconMultiDevice(unittest.TestCase):
    """The threaded path, exercised on virtual CPU devices when no GPU is present."""

    def test_recon_across_two_devices(self):
        print('Testing the threaded multi-device path')
        devices = jax.devices()[:2]
        if len(devices) < 2:
            self.skipTest('needs at least two devices')

        np.random.seed(0)
        mace = mj.MACE4DModel(_small_model(), num_frames=4)
        mace.set_params(dejitter=False, verbose=0)
        mace.configure_devices(devices)

        with tempfile.TemporaryDirectory() as tmp_dir:
            log_dir = os.path.join(tmp_dir, 'logs')
            recon, _ = mace.recon(_smooth_sino(), max_iterations=1,
                                  stop_threshold_change_pct=0, log_dir=log_dir)
            self.assertEqual(recon.shape, (4,) + mace.recon_shape)
            self.assertTrue(np.all(np.isfinite(recon)))
            with open(os.path.join(log_dir, 'task_log.csv')) as f:
                rows = list(csv.DictReader(f))
        # Both devices must have run tasks, or the concurrency under test never happened.
        self.assertEqual({row['device'] for row in rows}, {'0', '1'})


class TestInitCache(unittest.TestCase):
    """The cached initialization image."""

    def setUp(self):
        self.mace = mj.MACE4DModel(_small_model())
        self.mace.set_params(verbose=0)

    def test_validate_init_recon(self):
        print('Testing initial-image validation')
        expected = (self.mace.nt,) + self.mace.recon_shape
        out = self.mace._validate_init_recon(np.zeros(expected, dtype=np.float64))
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(out.shape, expected)
        with self.assertRaises(ValueError):
            self.mace._validate_init_recon(np.zeros((1, 2, 3, 4)))

    def test_cache_absent_invalid_then_valid(self):
        print('Testing the init cache for a missing, invalid and valid file')
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Absent: silent None.
            self.assertIsNone(self.mace._load_cached_init(tmp_dir))

            # Present but wrong shape: warns and returns None.
            np.save(os.path.join(tmp_dir, 'init_recon.npy'),
                    np.zeros((1, 2, 3, 4), dtype=np.float32))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                self.assertIsNone(self.mace._load_cached_init(tmp_dir))
            self.assertEqual(len(caught), 1)
            self.assertIn('invalid', str(caught[0].message))

            # Valid: loaded as float32.
            good = np.zeros((self.mace.nt,) + self.mace.recon_shape, dtype=np.float32)
            np.save(os.path.join(tmp_dir, 'init_recon.npy'), good)
            loaded = self.mace._load_cached_init(tmp_dir)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.shape, good.shape)


class TestBatchedDenoise(unittest.TestCase):
    """The vmapped hyperplane denoiser."""

    def test_batched_denoise_equals_serial(self):
        """The vmapped batch must reproduce per-volume denoising with the same constants,
        including lanes that converge at different iterations."""
        print('Testing that batched denoising matches per-volume denoising')
        np.random.seed(0)
        rng = np.random.default_rng(3)
        x = (np.linspace(0, 1, 8 * 10 * 12).reshape(8, 10, 12)[None]
             + 0.05 * rng.normal(size=(6, 8, 10, 12))).astype(np.float32)
        device = jax.devices()[0]
        denoiser = _get_qggmrf_denoiser(x.shape[1:], device)
        _configure_denoiser(denoiser, sigma=0.05, image_for_stats=x.reshape(-1, 10, 12))

        y_batched = _batched_hyperplane_denoise(x, denoiser, device)

        partition, fm_constant, qggmrf_params, image_shape = _denoise_constants(denoiser)
        stop_thresh = _DENOISE_STOP_THRESHOLD_PCT / 100.0
        y_serial = np.empty_like(x)
        for i in range(x.shape[0]):
            flat = jnp.asarray(x[i].reshape(-1, x.shape[3]))
            out, _, _, _ = denoiser._denoise_single_device(
                flat, jnp.zeros_like(flat), partition, fm_constant, qggmrf_params,
                image_shape, _DENOISE_MAX_ITERATIONS, stop_thresh, 0)
            y_serial[i] = np.asarray(out).reshape(x.shape[1:])

        self.assertTrue(np.allclose(y_batched, y_serial, atol=1e-5))
        self.assertFalse(np.allclose(y_batched, x))   # denoising actually changed the volumes

    def test_denoiser_wrapper_shape_and_axes(self):
        print('Testing that the hyperplane permutation round-trips to the original shape')
        np.random.seed(0)
        rng = np.random.default_rng(4)
        x = rng.normal(size=(5, 6, 7, 8)).astype(np.float32)
        y = _denoiser_wrapper(x, permute_vector=(3, 0, 1, 2), sigma=0.1,
                              device=jax.devices()[0])
        self.assertEqual(y.shape, x.shape)


if __name__ == '__main__':
    unittest.main()
