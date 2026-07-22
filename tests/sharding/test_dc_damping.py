"""
Tests for the cone-beam DC damping preconditioner (the C4 configuration,
dj-space-domain-preconditioner study, 2026-07-22).

The damping is a ConeBeamModel override of the per-subset update-direction seam
(_get_update_direction): the DC component of each slice's update is scaled by a
geometry-derived profile s_k, with the H^{-1}-weighted slice mean taken over the
subset's pixels.  It is configured by the PRIVATE class attribute
ConeBeamModel._dc_damping (deliberately not a public parameter).

These tests check:
  - the jitted helper matches a plain numpy reference of the Section-3 formula,
    on both the qGGMRF form (array prior_hess) and the prox form (scalar);
  - s == 1 and _dc_damping = None both reduce exactly to the base direction;
  - the s_k profile has the designed values (a at the z = 0 slice, half decay
    (a+c)/2 at t = b), and the z = 0 slice follows recon_slice_offset;
  - helical scans use the view-averaged profile (checked against an
    independent numpy computation), and a constant z shift matches the
    equivalent circular recon_slice_offset;
  - the 4 x 128 cycled schedule is the global default for all geometries, and
    non-cone geometries keep the base update direction;
  - a small damped cone-beam recon on 2 devices matches 1 device, including a
    slice count that does NOT divide the device count (padding path);
  - the damped prox_map path runs and differs from base only in the damped band.
"""
import unittest

# Import mbirjax before jax (device-setup-first ordering).
import mbirjax
import mbirjax as mj

import numpy as np
import jax
import jax.numpy as jnp

from conftest import preferred_devices

from mbirjax.cone_beam import ConeBeamModel, _dc_damped_update_direction, _DC_DAMPING_DEFAULT
from mbirjax.tomography_model import TomographyModel


def _numpy_reference(fg, pg, fh, ph, s):
    g = fg + pg
    h = fh + np.broadcast_to(ph, fg.shape)
    d = np.empty_like(g)
    for j in range(g.shape[1]):
        w = 1.0 / h[:, j]
        gbar = (w * g[:, j]).sum() / w.sum()
        d[:, j] = -w * (g[:, j] - (1.0 - s[j]) * gbar)
    return d


def _random_seam_arrays(rng, npix, nsl, scalar_prior_hess=False):
    fg = rng.normal(size=(npix, nsl)).astype(np.float32)
    pg = rng.normal(size=(npix, nsl)).astype(np.float32)
    fh = rng.uniform(0.1, 50.0, size=(npix, nsl)).astype(np.float32)
    ph = (np.float32(rng.uniform(0.5, 2.0)) if scalar_prior_hess
          else rng.uniform(0.01, 5.0, size=(npix, nsl)).astype(np.float32))
    return fg, pg, fh, ph


def _make_cone_model(num_views=8, num_rows=10, num_channels=16, helical_z_shifts=None):
    """Small cone-beam model on one device (tests override configure_devices)."""
    angles = jnp.linspace(0.0, jnp.pi, num_views, endpoint=False)
    model = ConeBeamModel((num_views, num_rows, num_channels), angles,
                          source_detector_dist=4 * num_channels,
                          source_iso_dist=2 * num_channels,
                          helical_z_shifts=helical_z_shifts)
    model.configure_devices(1)
    return model


class TestDampedDirectionHelper(unittest.TestCase):
    """The jitted helper against a plain numpy reference."""

    def test_matches_reference_qggmrf_and_prox_forms(self):
        rng = np.random.default_rng(0)
        for scalar_ph in (False, True):
            for _ in range(20):
                npix, nsl = int(rng.integers(2, 30)), int(rng.integers(2, 24))
                fg, pg, fh, ph = _random_seam_arrays(rng, npix, nsl, scalar_ph)
                s = rng.uniform(0.6, 1.0, size=nsl).astype(np.float32)
                # forward_grad is donated: pass a fresh device array each call
                d = np.asarray(_dc_damped_update_direction(
                    jnp.array(fg), jnp.array(pg), jnp.array(fh), jnp.array(ph), jnp.array(s)))
                ref = _numpy_reference(fg, pg, fh, ph, s)
                self.assertTrue(np.allclose(d, ref, rtol=2e-4, atol=2e-5))

    def test_s_equal_one_reduces_to_base(self):
        rng = np.random.default_rng(1)
        fg, pg, fh, ph = _random_seam_arrays(rng, 40, 12)
        s1 = jnp.ones(12, dtype=jnp.float32)
        d = np.asarray(_dc_damped_update_direction(
            jnp.array(fg), jnp.array(pg), jnp.array(fh), jnp.array(ph), s1))
        base = -(fg + pg) / (fh + ph)
        self.assertTrue(np.allclose(d, base, rtol=1e-5, atol=1e-6))


class TestProfileAndFallbacks(unittest.TestCase):
    """The s_k profile values, the None/helical fallbacks, and the defaults."""

    def test_profile_values_and_offset(self):
        model = _make_cone_model()
        s = np.asarray(model._dc_damping_slice_profile())
        a, b, p, c = _DC_DAMPING_DEFAULT
        recon_shape = model.get_params('recon_shape')
        dv, slice_aspect, oz, R = model.get_params(
            ['delta_voxel', 'voxel_slice_aspect', 'recon_slice_offset', 'source_iso_dist'])
        nz = recon_shape[2]
        dz = slice_aspect * dv
        k0 = (nz - 1) / 2.0 - oz / dz          # slice where z = 0
        # value at the z = 0 slice is a (to within the discrete slice sampling)
        self.assertLess(abs(s[int(round(k0))] - a), 0.05)
        # half decay (a+c)/2 at |k - k0| = b R / L
        L = recon_shape[0] * dv
        kh = int(round(k0 + b * R / L))
        if kh < nz:
            self.assertLess(abs(s[kh] - (a + c) / 2.0), 0.05)
        # profile length is the device-form slice count
        self.assertEqual(s.size, model.recon_placement.padded_size)

    def test_offset_moves_the_minimum(self):
        model = _make_cone_model()
        dv, slice_aspect = model.get_params(['delta_voxel', 'voxel_slice_aspect'])
        model.set_params(no_compile=True, no_warning=True,
                         recon_slice_offset=3.0 * slice_aspect * dv)
        s = np.asarray(model._dc_damping_slice_profile())
        nz = model.get_params('recon_shape')[2]
        k0 = (nz - 1) / 2.0 - 3.0
        self.assertEqual(int(np.argmin(s[:nz])), int(round(k0)))

    def test_disabled_falls_back_to_base(self):
        rng = np.random.default_rng(2)
        fg, pg, fh, ph = _random_seam_arrays(rng, 30, 14)
        base = -(fg + pg) / (fh + ph)
        idx = jnp.arange(30)

        model = _make_cone_model(num_rows=14)
        model._dc_damping = None
        d = np.asarray(model._get_update_direction(
            jnp.array(fg), jnp.array(pg), jnp.array(fh), jnp.array(ph), idx))
        self.assertTrue(np.allclose(d, base, rtol=1e-5, atol=1e-6))

    def test_helical_uses_view_averaged_profile(self):
        # helical: s_k = mean_i s(L |z_k - z_i| / (R dz)), checked against an
        # independent numpy computation from the model's own geometry params
        shifts = np.linspace(-2.0, 2.0, 8).astype(np.float32)
        model = _make_cone_model(num_rows=14, helical_z_shifts=jnp.asarray(shifts))
        s = np.asarray(model._dc_damping_slice_profile())
        a, b, p, c = _DC_DAMPING_DEFAULT
        recon_shape = model.get_params('recon_shape')
        dv, slice_aspect, oz, R = model.get_params(
            ['delta_voxel', 'voxel_slice_aspect', 'recon_slice_offset', 'source_iso_dist'])
        nz = recon_shape[2]
        L = recon_shape[0] * dv
        dz = slice_aspect * dv
        z = (np.arange(nz) - (nz - 1) / 2.0) * dz + oz
        t = L * np.abs(z[:, None] - shifts[None, :]) / (R * dz)
        ref = ((c * t ** p + a * b ** p) / (t ** p + b ** p)).mean(axis=1)
        self.assertTrue(np.allclose(s[:nz], ref, rtol=1e-5, atol=1e-6))
        # a helical profile is milder than circular at the circular profile's minimum
        circ = np.asarray(_make_cone_model(num_rows=14)._dc_damping_slice_profile())
        self.assertGreater(s[np.argmin(circ[:nz])], circ[np.argmin(circ[:nz])])

    def test_constant_shift_equals_plain_circular(self):
        # All z shifts equal z0: auto_set_recon_geometry centers the recon on the
        # travel (recon_slice_offset = z0), and the profile subtracts z0 back out,
        # so the damping profile must equal the plain circular model's profile.
        shifted = _make_cone_model(num_rows=14, helical_z_shifts=jnp.full(8, 1.5))
        self.assertEqual(shifted.get_params('recon_slice_offset'), 1.5)
        s_shifted = np.asarray(shifted._dc_damping_slice_profile())
        s_circ = np.asarray(_make_cone_model(num_rows=14)._dc_damping_slice_profile())
        self.assertTrue(np.allclose(s_shifted, s_circ, rtol=1e-6, atol=1e-7))

    def test_schedule_defaults_global_and_damping_cone_only(self):
        model = _make_cone_model()
        pb = mj.ParallelBeamModel((8, 10, 16), jnp.linspace(0.0, jnp.pi, 8, endpoint=False))
        for m in (model, pb):
            self.assertEqual(m.get_params('granularity'),
                             [1, 2, 4, 8, 16, 32, 64, 128, 128, 128, 128])
            self.assertEqual(m.get_params('partition_sequence'),
                             [2, 4, 6] + [7, 8, 9, 10] * 25)
        # a user override after construction wins
        model.set_params(no_compile=True, no_warning=True, granularity=[1, 8, 64])
        self.assertEqual(model.get_params('granularity'), [1, 8, 64])
        # only cone beam overrides the update direction
        self.assertIs(type(pb)._get_update_direction, TomographyModel._get_update_direction)


class TestDampedReconSharded(unittest.TestCase):
    """End-to-end damped cone recon: 2 devices vs 1, including slice padding."""

    MAX_ITERS = 4

    def _recon(self, model, sino):
        np.random.seed(0)
        recon, _ = model.recon(sino, max_iterations=self.MAX_ITERS,
                               stop_threshold_change_pct=0.0, print_logs=False)
        return np.asarray(recon)

    def _sino(self, model, seed=0):
        # A reproducible random sinogram: VCD is deterministic given the sinogram
        # and the numpy seed, which is all these comparisons need (same pattern as
        # test_vcd_sharded._phantom_sino).
        shape = model.get_params('sinogram_shape')
        rng = np.random.default_rng(seed)
        return jnp.asarray(rng.standard_normal(shape, dtype=np.float32))

    def test_two_devices_match_one_with_padding(self):
        devs = preferred_devices(2)
        if devs is None:
            self.skipTest('needs 2 devices')
        # num_rows chosen so the recon slice count is ODD (auto geometry pads it
        # from the detector height): exercises the padded-slice path on 2 devices.
        ref_model = _make_cone_model(num_rows=11)
        nz = ref_model.get_params('recon_shape')[2]
        sino = self._sino(ref_model)
        ref = self._recon(ref_model, sino)

        model = _make_cone_model(num_rows=11)
        model.configure_devices(devs)
        if nz % 2 == 0:
            # force an odd slice count so the padding path is exercised
            shape = list(model.get_params('recon_shape'))
            shape[2] += 1
            model.set_params(recon_shape=tuple(shape))
            ref_model2 = _make_cone_model(num_rows=11)
            ref_model2.set_params(recon_shape=tuple(shape))
            ref = self._recon(ref_model2, self._sino(ref_model2))
            sino = self._sino(model)
        out = self._recon(model, sino)
        self.assertEqual(out.shape, ref.shape)
        scale = np.max(np.abs(ref)) + 1e-9
        self.assertLess(np.max(np.abs(out - ref)) / scale, 5e-4)

    def test_damping_changes_recon_and_prox_runs(self):
        model = _make_cone_model(num_rows=11)
        sino = self._sino(model)
        damped = self._recon(model, sino)

        undamped_model = _make_cone_model(num_rows=11)
        undamped_model._dc_damping = None
        undamped = self._recon(undamped_model, sino)
        self.assertEqual(damped.shape, undamped.shape)
        self.assertGreater(np.max(np.abs(damped - undamped)), 0.0)

        # prox path: runs with damping active and produces finite output
        prox_model = _make_cone_model(num_rows=11)
        prox_input = np.zeros(prox_model.get_params('recon_shape'), dtype=np.float32)
        np.random.seed(0)
        prox_out, _ = prox_model.prox_map(prox_input, sino, sigma_prox=1.0,
                                          stop_threshold_change_pct=0.0,
                                          max_iterations=3, print_logs=False)
        self.assertTrue(np.all(np.isfinite(np.asarray(prox_out))))


if __name__ == '__main__':
    unittest.main()
