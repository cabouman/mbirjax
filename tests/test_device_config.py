"""Device-configuration surface: configure_devices platform strings + the use_gpu deprecation.

configure_devices is THE device control surface; use_gpu is deprecated (device selection is an
execution-environment choice, not a model parameter -- a persisted value silently follows a
saved model to a different machine).  During the deprecation window set_params(use_gpu=...)
warns and is HONORED by forwarding to configure_devices -- which also fixes the old precedence
bug where use_gpu='none' was silently ignored once configure_devices had pinned devices.

These tests run on CPU (>= 2 virtual devices from _device_setup), so the platform FLIP itself
cannot be exercised here; what is pinned instead: the string resolutions, the warning, the
forwarding's effect on the layout (including over a previous pin), and that internal paths
(construction, print_params) neither warn nor disturb the layout.
"""
import unittest
import warnings

# Import mbirjax before jax (device-setup-first ordering).
import mbirjax

import numpy as np
import jax


def make_model(num_views=32):
    angles = np.linspace(0, np.pi, num_views, endpoint=False)
    return mbirjax.ParallelBeamModel((num_views, 40, 32), angles)


class TestConfigureDevicesPlatformStrings(unittest.TestCase):

    def test_cpu_string_pins_cpu_pool(self):
        model = make_model()
        model.configure_devices('cpu')
        self.assertTrue(all(d.platform == 'cpu' for d in model.shard_devices))
        self.assertTrue(model._sharding_configured)
        # Same trimming as automatic selection: never more devices than the auto count allows.
        self.assertEqual(len(model.shard_devices),
                         model._auto_device_count(len(jax.devices('cpu'))))

    def test_gpu_string_raises_without_gpu_backend(self):
        try:
            jax.devices('gpu')
            self.skipTest('GPU backend present; the no-GPU error path is not reachable')
        except RuntimeError:
            pass
        model = make_model()
        with self.assertRaisesRegex(ValueError, 'no GPU backend'):
            model.configure_devices('gpu')

    def test_unknown_string_raises(self):
        model = make_model()
        with self.assertRaisesRegex(ValueError, "'cpu' or 'gpu'"):
            model.configure_devices('tpu-ish')


class TestUseGpuDeprecation(unittest.TestCase):

    def test_set_params_use_gpu_warns_and_is_honored(self):
        model = make_model()
        with self.assertWarns(DeprecationWarning):
            model.set_params(use_gpu='none')
        self.assertTrue(all(d.platform == 'cpu' for d in model.shard_devices))
        self.assertTrue(model._sharding_configured)     # 'none' forwards to a CPU pin

    def test_use_gpu_takes_effect_over_a_previous_pin(self):
        # The old precedence bug: after configure_devices, set_params(use_gpu=...) was silently
        # ignored (the pinned branch of set_devices never consults use_gpu).  The deprecation
        # forwarding makes the LATEST instruction win: pin one device, then request automatic,
        # and the layout must return to the auto pool.
        model = make_model()
        auto_count = len(model.shard_devices)           # construction = automatic selection
        if auto_count < 2:
            self.skipTest('need >= 2 devices to distinguish a 1-device pin from the auto pool')
        model.configure_devices(1)
        self.assertEqual(len(model.shard_devices), 1)
        with self.assertWarns(DeprecationWarning):
            model.set_params(use_gpu='automatic')
        self.assertEqual(len(model.shard_devices), auto_count)
        self.assertFalse(model._sharding_configured)    # automatic tracking restored

    def test_internal_paths_do_not_warn(self):
        # Construction and print_params must neither emit the deprecation warning nor disturb
        # the device layout (print_params formerly re-set use_gpu, forcing a recompile per
        # print and -- under the forwarding -- it would have silently unpinned a configured
        # model).
        with warnings.catch_warnings():
            warnings.simplefilter('error', DeprecationWarning)
            model = make_model()
            model.configure_devices(1)
            devices_before = tuple(model.shard_devices)
            model.set_params(verbose=0)
            model.print_params()
        self.assertEqual(tuple(model.shard_devices), devices_before)
        self.assertTrue(model._sharding_configured)

    def test_denoiser_constructs_without_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter('error', DeprecationWarning)
            mbirjax.QGGMRFDenoiser((8, 16, 16))


if __name__ == '__main__':
    unittest.main()
