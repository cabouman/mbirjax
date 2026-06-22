"""
Tests for sharded QGGMRF denoising (QGGMRFDenoiser).

The denoiser is a TomographyModel whose forward model is the IDENTITY (the
"error sinogram" is the residual image), so there is no projection and no view
sharding -- only the recon mesh.  The image flattens to (num_pixels, num_slices)
and slice-shards on the last axis, exactly like the recon.

Two paths (see QGGMRFDenoiser.denoise):
  - one device -> the whole sweep runs in a single JIT (no halos: a single shard
    uses the reflected BC).  This is the path every current single-device use
    takes and is exercised by tests/test_denoiser.py.
  - multiple devices -> the flat arrays slice-shard and a Python loop stages the
    qGGMRF halos once per pass (host-side, so it cannot live in a JIT), mirroring
    vcd_recon.

Because the forward model is the identity, the qGGMRF prior (and hence the
once-per-pass halo approximation) drives the whole update rather than being
washed out by a back projection.  So the sharded path follows a slightly
different EARLY iteration path than the single-device exact prior -- but both
converge to the same MAP estimate (measured single-vs-multi max-rel-diff: ~9e-3
at 6 iters, ~9e-4 at 15, ~3e-5 at 30).  The test therefore runs enough iterations
to be well converged and compares with a tolerance that comfortably covers the
residual path difference while still tripping on any real algorithmic drift.

Runs on whatever devices conftest provides (real GPUs on a cluster, virtual CPU
devices otherwise).
"""
import unittest

# Import mbirjax before jax (device-setup-first ordering).
import mbirjax as mj

import numpy as np

from conftest import preferred_devices, assert_sharded_allclose

# Enough iterations that the sharded and single-device denoisers are well
# converged (the once-per-pass halo path difference has shrunk to << TOL).
ITERATIONS = 20
TOL = 3e-3


def _noisy_image(shape=(8, 16, 16), sigma=0.1, seed=0):
    """A noisy Shepp-Logan; the slice axis (last, here 16) divides 2/4/8 so it
    shards cleanly across those device counts."""
    rng = np.random.RandomState(seed)
    clean = np.asarray(mj.generate_3d_shepp_logan_low_dynamic_range(shape), dtype=np.float32)
    return clean + sigma * rng.randn(*shape).astype(np.float32), sigma


def _denoise(devices, image, sigma):
    """Denoise `image` on the given devices (an int count or a device list),
    forcing exactly ITERATIONS iterations so the comparison is at a fixed,
    converged point regardless of the stop threshold."""
    denoiser = mj.QGGMRFDenoiser(image.shape)
    denoiser.configure_devices(devices)
    out, _ = denoiser.denoise(image, sigma_noise=sigma, max_iterations=ITERATIONS,
                              stop_threshold_change_pct=0.0, print_logs=False)
    return np.asarray(out)


class TestShardedDenoise(unittest.TestCase):

    def test_sharded_denoise_matches_single_device(self):
        """The multi-device (slice-sharded) denoiser converges to the single-device
        result.  Single device = the JITed whole-sweep path; multi-device = the
        Python-loop + once-per-pass-halo path."""
        image, sigma = _noisy_image()
        ref = _denoise(1, image, sigma)   # single-device reference (the JIT whole-sweep path)
        self.assertTrue(np.isfinite(ref).all())

        ran_multi = False
        for n in (2, 4):
            devs = preferred_devices(n)
            if devs is None:
                continue
            num_slices = image.shape[2]
            if num_slices % n != 0:
                # Padding would also work, but keep the gate on the clean (no-pad) case.
                continue
            out = _denoise(devs, image, sigma)
            self.assertEqual(out.shape, ref.shape)
            assert_sharded_allclose(out, ref, msg=f"sharded denoise mismatch at n_dev={n}", tol=TOL)
            ran_multi = True
        if not ran_multi:
            self.skipTest("no usable device count > 1")


if __name__ == "__main__":
    unittest.main()
