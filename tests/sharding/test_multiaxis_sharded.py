"""
Tests for sharded MULTIAXIS-PARALLEL projection.

MultiAxisParallelModel shards the recon by **slice** and the sinogram by **view**, like cone
beam and translation, using the same geometry-neutral placement path (it adds no projector/
driver overrides):

  * BACK is a reduce-scatter on the BANDED back kernel -- with a nonzero elevation a slice is
    drawn from a RANGE of detector rows, so rows cannot be cropped; each view-owner banded-back-
    projects its views onto a slice band and the partials are summed onto the slice-owner.
  * FORWARD GATHERS the full slice cylinder onto each view-owner (per pixel-batch, to bound
    memory) and runs the MONOLITHIC forward (decision C); it does not band the forward.

Even the single-device case runs on the placement path
(a trivial 1-device mesh), so the per-geometry multiaxis gates in tests/geometries already
exercise the n_dev=1 path.  These tests add the MULTI-DEVICE gate: the sharded path must match
the single-device path to float noise, for both an isotropic and an anisotropic
(voxel_row_aspect / voxel_slice_aspect != 1) geometry, on back / forward / Hessian-diagonal.

There is deliberately NO sharded VCD-recon test here: the sharded VCD loop is geometry-independent
and gated by the cone + parallel VCD tests (multiaxis adds no loop overrides); see the NOTE at the
bottom.

Runs on whatever devices conftest provides (real GPUs on a cluster, virtual CPU devices
otherwise); skips device counts the small geometry's axes do not divide.  The non-dividing
(padded) case is covered separately by the padding tests (tests/sharding).
"""
import unittest

# Import mbirjax before jax (device-setup-first ordering).
import mbirjax

import numpy as np
import jax
import jax.numpy as jnp

from conftest import preferred_devices, assert_sharded_allclose


def _make_multiaxis_model(anisotropic=False, num_views=8, num_det_rows=32, num_det_channels=32):
    """Small multiaxis model with a deterministic +-8 deg elevation spread (nonzero tilt, so the
    vertical fan genuinely spreads slices across detector rows).  num_det_rows=32 auto-sizes the
    recon to 32 slices isotropic / 16 slices anisotropic (slice aspect 2), so both the view axis
    (8) and the slice axis stay divisible by 2 and 4.  Pinned to a single device so the bare model
    is a deterministic single-device REFERENCE; the multi-device tests override with their own
    configure_devices(devs)."""
    az = np.linspace(0.0, np.pi, num_views, endpoint=False)
    el = np.deg2rad(np.linspace(-8.0, 8.0, num_views))
    angles = jnp.asarray(np.stack([az, el], axis=1))
    model = mbirjax.MultiAxisParallelModel((num_views, num_det_rows, num_det_channels), angles)
    if anisotropic:
        model.set_params(voxel_row_aspect=1.9, voxel_slice_aspect=2.0)
    model.auto_set_recon_geometry()
    model.configure_devices(1)
    return model


def _random_sino(model, seed=0):
    shape = model.get_params('sinogram_shape')
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.standard_normal(shape, dtype=np.float32))


def _random_recon(model, seed=1):
    shape = tuple(int(x) for x in model.get_params('recon_shape'))
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.standard_normal(shape, dtype=np.float32))


def _usable_device_counts(model):
    """Available device counts > 1 (from conftest) that divide both sharded axes."""
    sino_shape = model.get_params('sinogram_shape')
    recon_shape = model.get_params('recon_shape')
    sino_axis = model.sinogram_shard_axis() % len(sino_shape)
    recon_axis = model.recon_shard_axis() % len(recon_shape)
    counts = []
    for n in (2, 4):
        devs = preferred_devices(n)
        if devs is None:
            continue
        if sino_shape[sino_axis] % n == 0 and recon_shape[recon_axis] % n == 0:
            counts.append((n, devs))
    return counts


class TestMultiAxisShardedProjectors(unittest.TestCase):
    """Single-shot projector comparisons: sharded == single-device to float noise."""

    GEOMETRIES = (False, True)   # isotropic, anisotropic
    TOL = 1e-5

    def test_geometry_axes_divisible(self):
        """Guard against auto-sizer drift: the view (8) and slice axes must divide 4, so the
        n=2,4 sharded comparisons actually run rather than silently skip on a divisibility miss."""
        for anisotropic in self.GEOMETRIES:
            with self.subTest(anisotropic=anisotropic):
                m = _make_multiaxis_model(anisotropic=anisotropic)
                nv = int(m.get_params('sinogram_shape')[0])
                ns = int(m.get_params('recon_shape')[2])
                self.assertEqual((nv % 4, ns % 4), (0, 0),
                                 f"anisotropic={anisotropic}: views={nv} slices={ns} not divisible by 4")

    def _sweep(self, anisotropic, ref_fn, shard_fn, label):
        ref_model = _make_multiaxis_model(anisotropic=anisotropic)
        counts = _usable_device_counts(ref_model)
        if not counts:
            self.skipTest("no usable device count > 1 (need >= 2 devices and divisible axes)")
        ref = np.asarray(ref_fn(ref_model))
        for n, devs in counts:
            model = _make_multiaxis_model(anisotropic=anisotropic)
            model.configure_devices(devs)
            out = np.asarray(shard_fn(model))
            assert_sharded_allclose(out, ref, tol=self.TOL,
                                    msg=f"{label} mismatch: anisotropic={anisotropic} n_dev={n}")

    def test_back_matches_single_device(self):
        for anisotropic in self.GEOMETRIES:
            with self.subTest(anisotropic=anisotropic):
                sino = _random_sino(_make_multiaxis_model(anisotropic=anisotropic))
                self._sweep(anisotropic, lambda m: m.back_project(sino),
                            lambda m: m.back_project(sino), "back")

    def test_forward_matches_single_device(self):
        for anisotropic in self.GEOMETRIES:
            with self.subTest(anisotropic=anisotropic):
                recon = _random_recon(_make_multiaxis_model(anisotropic=anisotropic))
                self._sweep(anisotropic, lambda m: m.forward_project(recon),
                            lambda m: m.forward_project(recon), "forward")

    def test_hessian_diagonal_matches_single_device(self):
        """coeff_power=2 (Hessian diagonal) sharded == single-device, via the scale-invariant
        gate (conftest.assert_sharded_allclose): squaring the projection coefficients gives the
        Hessian a large dynamic range, so the reduce-scatter summation-reorder noise (unbiased,
        ~1e-7 of the peak) lands on near-zero diagonal entries as a large RELATIVE diff that a
        fixed atol would false-fail.  back/forward have no such squaring but use the same gate."""
        for anisotropic in self.GEOMETRIES:
            with self.subTest(anisotropic=anisotropic):
                weights = _random_sino(_make_multiaxis_model(anisotropic=anisotropic), seed=7)
                self._sweep(anisotropic, lambda m: m.compute_hessian_diagonal(weights),
                            lambda m: m.compute_hessian_diagonal(weights), "hessian")


# NOTE: there is deliberately NO sharded VCD-recon test for multiaxis.  The sharded VCD LOOP
# (reduce-scatter back / gather forward / halo-exchanged qGGMRF prior / partitioning / donation) is
# geometry-INDEPENDENT and is gated sharded by the cone + parallel VCD tests; multiaxis adds no loop
# overrides, and its sharded projectors are gated above (back/forward/Hessian == single).  A
# per-geometry sharded recon would only re-run that shared machinery (same rationale as
# tests/geometries/test_vcd.py gating full convergence on parallel + cone only).


if __name__ == "__main__":
    unittest.main()
