"""
Tests for the end-to-end sharded FBP pipeline (ParallelBeamModel.fbp_recon /
direct_recon)

fbp_recon is **user-facing**: the input may be plain or sharded (a plain
sinogram is sharded on the view axis once at entry), and the OUTPUT form is
chosen by the ``output_sharded`` kwarg.  Under a mesh it runs the on-device
pipeline:

    fbp_filter(output_sharded=True) (sharded view -> sharded view)
        -> back_project(output_sharded=True) (sharded sino -> slice-sharded recon)

The data stays resident on its devices throughout (zero intermediate host
transfer).  By default the recon is gathered to a plain array at exit; with
output_sharded=True it is returned slice-sharded (no host round-trip).
direct_recon simply delegates to fbp_recon.

These tests check:
  - single-device (no-mesh) fbp_recon / direct_recon is plain-in / plain-out;
  - a trivial 1-device mesh is BIT-EXACT vs the unconfigured single-device path
    (the prerelease regression gate);
  - 2/4/8-device sharding matches single-device to float noise; the default
    returns a plain (gathered) recon, output_sharded=True returns a
    slice-sharded recon;
  - the no-intermediate-gather invariant: the sinogram handed to back_project
    inside fbp_recon is view-sharded (axis 0), i.e. the filter output never
    round-trips through the host.

Runs on whatever devices conftest provides (real GPUs on a cluster, virtual CPU
devices otherwise).
"""
import unittest

# Import mbirjax before jax (device-setup-first ordering).
import mbirjax

import numpy as np
import jax
import jax.numpy as jnp

from conftest import preferred_devices


def _make_model(num_views=8, num_rows=8, num_channels=32):
    """Small parallel-beam model; num_views and num_rows divisible by 2/4/8."""
    angles = jnp.linspace(0, jnp.pi, num_views, endpoint=False)
    # Pin a single device so the bare model is a deterministic single-device REFERENCE regardless of
    # how many GPUs are present (auto-sharding now uses all available GPUs by default); tests that
    # exercise multi-device sharding override this with their own configure_sharding(devs).
    model = mbirjax.ParallelBeamModel((num_views, num_rows, num_channels), angles)
    model.configure_devices(1)
    return model


def _random_sino(model, seed=0):
    """A reproducible random sinogram of the model's sinogram shape.

    FBP (filter + adjoint back projection) is linear, so a random sinogram is a
    valid input for cross-mode comparison; physical fidelity is irrelevant here.
    """
    shape = model.get_params('sinogram_shape')
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.standard_normal(shape, dtype=np.float32))


def _divisible(model, n):
    """True if the model's sharded sinogram and recon axes both divide n."""
    sino_shape = model.get_params('sinogram_shape')
    recon_shape = model.get_params('recon_shape')
    sino_axis = model.sinogram_shard_axis() % len(sino_shape)
    recon_axis = model.recon_shard_axis() % len(recon_shape)
    return sino_shape[sino_axis] % n == 0 and recon_shape[recon_axis] % n == 0


class TestFbpReconSingleDevice(unittest.TestCase):
    """No mesh configured: plain-in / plain-out, shape correct."""

    def test_runs_and_shape(self):
        model = _make_model()
        out = model.fbp_recon(_random_sino(model))
        self.assertEqual(tuple(out.shape), tuple(model.get_params('recon_shape')))
        self.assertNotIsInstance(getattr(out, 'sharding', None),
                                 jax.sharding.NamedSharding)

    def test_direct_recon_matches_fbp_recon(self):
        """direct_recon is just a thin delegate to fbp_recon."""
        model = _make_model()
        sino = _random_sino(model)
        a = np.asarray(model.fbp_recon(sino))
        b = np.asarray(model.direct_recon(sino))
        self.assertTrue(np.array_equal(a, b))


class TestFbpReconSharded(unittest.TestCase):

    def setUp(self):
        self.devs = preferred_devices(2)
        if self.devs is None:
            self.skipTest("need >= 2 devices")

    def _check_divisible(self, model, n):
        if not _divisible(model, n):
            self.skipTest(f"sharded axes not divisible by {n}")

    def test_trivial_sharding_bit_exact(self):
        """1-device mesh matches the unconfigured single-device path to tight float
        tolerance.  (Name kept for history; was an exact-equality check.)

        RETIRE-AFTER-SHARDING: trivial-mesh-vs-legacy comparison, meaningful only
        while both paths coexist; once every geometry runs on placements there is
        one path and nothing to compare.  Relaxed from exact equality because the
        banded sharded path reorders non-associative FP sums -> ~1 ULP GPU
        difference (CPU compiles both identically and stays exact).  A tight
        tolerance still trips on any real algorithmic drift.
        """
        single_dev = preferred_devices(1)
        if single_dev is None:
            self.skipTest("need >= 1 device")
        model = _make_model()
        sino = _random_sino(model)
        ref = np.asarray(model.fbp_recon(sino))

        shard_model = _make_model()
        shard_model.configure_sharding(single_dev)
        out = np.asarray(shard_model.fbp_recon(sino))
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5,
                                   err_msg="trivial sharding diverged beyond float noise")

    def test_sharded_matches_single_device(self):
        """2-device fbp_recon matches single-device to float noise and returns a
        plain (gathered) recon."""
        model = _make_model()
        self._check_divisible(model, 2)
        sino = _random_sino(model)
        ref = np.asarray(model.fbp_recon(sino))

        shard_model = _make_model()
        shard_model.configure_sharding(self.devs)
        out = shard_model.fbp_recon(sino)
        # Match-input, PLAIN input: gathered (plain) output.
        self.assertNotIsInstance(getattr(out, 'sharding', None),
                                 jax.sharding.NamedSharding)
        np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-5, atol=1e-5)

    def test_output_sharded_returns_sharded_recon(self):
        """output_sharded=True on fbp_recon / direct_recon returns a slice-sharded
        recon (no gather) matching the plain single-device recon to float noise --
        even for a PLAIN input (the kwarg, not the input placement, decides)."""
        model = _make_model()
        self._check_divisible(model, 2)
        sino = _random_sino(model)
        ref = np.asarray(model.fbp_recon(sino))        # single-device plain

        shard_model = _make_model()
        shard_model.configure_sharding(self.devs)
        recon_axis = shard_model.recon_shard_axis()
        for fn_name in ('fbp_recon', 'direct_recon'):
            out = getattr(shard_model, fn_name)(sino, output_sharded=True)
            self.assertIsInstance(
                out.sharding, jax.sharding.NamedSharding,
                msg=f"{fn_name} should return a sharded recon for output_sharded=True")
            ax = recon_axis % out.ndim
            self.assertEqual(out.sharding.spec[ax], 'devices')
            np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-5, atol=1e-5,
                                       err_msg=f"{fn_name} output_sharded mismatch")
        # And the inverse: a SHARDED input with the default still returns plain.
        sharded_in = shard_model._shard_sinogram(sino)
        out = shard_model.fbp_recon(sharded_in)
        self.assertNotIsInstance(getattr(out, 'sharding', None),
                                 jax.sharding.NamedSharding)
        np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-5, atol=1e-5)

    def test_no_intermediate_gather(self):
        """The sinogram fbp_recon hands to back_project must still be view-sharded
        -- the filter output stays on-device (no host round-trip between filter
        and back projection)."""
        model = _make_model()
        self._check_divisible(model, 2)
        model.configure_sharding(self.devs)

        captured = {}
        original_back_project = model.back_project

        def spy_back_project(sinogram, *args, **kwargs):
            captured['sharding'] = getattr(sinogram, 'sharding', None)
            return original_back_project(sinogram, *args, **kwargs)

        model.back_project = spy_back_project
        model.fbp_recon(_random_sino(model))

        sharding = captured.get('sharding')
        self.assertIsInstance(sharding, jax.sharding.NamedSharding)
        # View axis (axis 0) is the one distributed across devices.
        view_axis = model.sinogram_shard_axis() % 3
        self.assertEqual(sharding.spec[view_axis], 'devices')

    def test_device_count_sweep(self):
        """The pipeline is correct across device counts, not just two."""
        ref_model = _make_model()
        sino = _random_sino(ref_model)
        ref = np.asarray(ref_model.fbp_recon(sino))
        ran_multi = False
        for n in (2, 4, 8):
            devs = preferred_devices(n)
            if devs is None:
                continue
            model = _make_model()
            if not _divisible(model, n):
                continue
            model.configure_sharding(devs)
            out = np.asarray(model.fbp_recon(sino))
            np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5,
                                       err_msg=f"mismatch at n_dev={n}")
            ran_multi = True
        if not ran_multi:
            self.skipTest("no usable device count > 1")


if __name__ == "__main__":
    unittest.main()
