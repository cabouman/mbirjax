"""
Tests for sharded TRANSLATION projection and reconstruction.

Translation shards the recon by **slice** and the sinogram by **view**, like cone beam,
using the same geometry-neutral placement path (TranslationModel adds no projector/driver
overrides beyond _supports_sharding=True):

  * BACK is a reduce-scatter on the BANDED back kernel -- a slice is drawn from a RANGE of
    detector rows, so rows cannot be cropped; each view-owner banded-back-projects its views
    onto a slice band and the partials are summed onto the slice-owner.
  * FORWARD GATHERS the full slice cylinder onto each view-owner (per pixel-batch, to bound
    memory) and runs the MONOLITHIC forward (decision C); it does not band the forward.

Once ``_supports_sharding()`` is True for translation, even the single-device case runs on the
placement path (a trivial 1-device mesh), so the per-geometry translation gates in
tests/geometries already exercise the n_dev=1 path.  These tests add the MULTI-DEVICE gate: the
sharded path must match the single-device path to float noise, for both an isotropic and an
anisotropic (voxel_slice_aspect != 1) translation geometry:

  - back / forward / Hessian-diagonal (coeff_power=2) single-shot projector comparisons;
  - a short VCD reconstruction (the full pipeline incl. the qGGMRF prior) sharded vs single.

Runs on whatever devices conftest provides (real GPUs on a cluster, virtual CPU devices
otherwise); skips device counts that the small geometry's axes do not divide.  The
non-dividing (padded) case is covered separately by the padding tests (tests/sharding).
"""
import unittest

# Import mbirjax before jax (device-setup-first ordering).
import mbirjax

import numpy as np
import jax
import jax.numpy as jnp

from conftest import preferred_devices


def _make_translation_model(anisotropic=False, num_views=8, num_det_rows=32, num_det_channels=32):
    """Small translation model.  num_views=8, num_det_rows=32 with a +-4 z translation range
    auto-sizes the recon to 16 slices isotropic / 8 slices anisotropic (aspect 2), so both the
    view axis (8) and the slice axis stay divisible by 2 and 4.  Pinned to a single device so the
    bare model is a deterministic single-device REFERENCE; the multi-device tests override with
    their own configure_sharding(devs)."""
    tv = np.zeros((num_views, 3))
    tv[:, 0] = np.linspace(-8.0, 8.0, num_views)
    tv[:, 2] = np.linspace(-4.0, 4.0, num_views)
    sdd = 4.0 * num_det_channels
    model = mbirjax.TranslationModel((num_views, num_det_rows, num_det_channels), jnp.asarray(tv),
                                     source_detector_dist=sdd, source_iso_dist=sdd / 2.0)
    if anisotropic:
        model.set_params(voxel_row_aspect=1.9)
        model.set_params(voxel_slice_aspect=2.0)
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


class TestTranslationShardedProjectors(unittest.TestCase):
    """Single-shot projector comparisons: sharded == single-device to float noise."""

    GEOMETRIES = (False, True)   # isotropic, anisotropic
    TOL = 1e-5

    def _sweep(self, anisotropic, ref_fn, shard_fn, label, atol_floor_from_peak=False):
        ref_model = _make_translation_model(anisotropic=anisotropic)
        counts = _usable_device_counts(ref_model)
        if not counts:
            self.skipTest("no usable device count > 1 (need >= 2 devices and divisible axes)")
        ref = np.asarray(ref_fn(ref_model))
        # The Hessian (coeff_power=2) squares the projection coefficients, giving the
        # back-projection a large dynamic range.  The reduce-scatter summation-reorder
        # difference between the sharded and single-device sums is unbiased and ~1e-7 of the
        # PEAK magnitude (measured), but on the near-zero diagonal entries that absolute noise
        # is a large RELATIVE diff -> a pure rtol gate false-fails.  Anchor the absolute floor
        # to the peak (atol = TOL * max|ref|, ~100x the measured noise) so small entries are
        # judged against the noise floor.  back/forward have no such squaring and keep the
        # plain tight TOL on both rtol and atol.
        atol = self.TOL * float(np.max(np.abs(ref))) if atol_floor_from_peak else self.TOL
        for n, devs in counts:
            model = _make_translation_model(anisotropic=anisotropic)
            model.configure_sharding(devs)
            out = np.asarray(shard_fn(model))
            np.testing.assert_allclose(
                out, ref, rtol=self.TOL, atol=atol,
                err_msg=f"{label} mismatch: anisotropic={anisotropic} n_dev={n}")

    def test_back_matches_single_device(self):
        for anisotropic in self.GEOMETRIES:
            with self.subTest(anisotropic=anisotropic):
                sino = _random_sino(_make_translation_model(anisotropic=anisotropic))
                self._sweep(anisotropic, lambda m: m.back_project(sino),
                            lambda m: m.back_project(sino), "back")

    def test_forward_matches_single_device(self):
        for anisotropic in self.GEOMETRIES:
            with self.subTest(anisotropic=anisotropic):
                recon = _random_recon(_make_translation_model(anisotropic=anisotropic))
                self._sweep(anisotropic, lambda m: m.forward_project(recon),
                            lambda m: m.forward_project(recon), "forward")

    def test_hessian_diagonal_matches_single_device(self):
        for anisotropic in self.GEOMETRIES:
            with self.subTest(anisotropic=anisotropic):
                weights = _random_sino(_make_translation_model(anisotropic=anisotropic), seed=7)
                self._sweep(anisotropic, lambda m: m.compute_hessian_diagonal(weights),
                            lambda m: m.compute_hessian_diagonal(weights), "hessian",
                            atol_floor_from_peak=True)


class TestTranslationShardedRecon(unittest.TestCase):
    """The full VCD recon (projectors + qGGMRF prior) sharded == single-device."""

    MAX_ITERS = 3
    TOL = 1e-4   # iterated: per-step FP-reorder differences accumulate (matches the cone sweep)

    def _recon(self, model, sino):
        np.random.seed(0)  # fix partitions + subset order so modes are comparable
        if model.mesh is not None:
            # Re-extract halos every subset -> the exact prior path (reproduces single-device).
            model._vcd_halo_per_subset = True
        model.set_params(verbose=0)
        recon, _ = model.recon(sino, max_iterations=self.MAX_ITERS,
                               stop_threshold_change_pct=0.0, print_logs=False)
        return np.asarray(recon)

    def test_vcd_recon_matches_single_device(self):
        for anisotropic in (False, True):
            with self.subTest(anisotropic=anisotropic):
                ref_model = _make_translation_model(anisotropic=anisotropic)
                counts = _usable_device_counts(ref_model)
                if not counts:
                    self.skipTest("no usable device count > 1")
                sino = _random_sino(ref_model, seed=2)
                ref = self._recon(_make_translation_model(anisotropic=anisotropic), sino)
                for n, devs in counts:
                    model = _make_translation_model(anisotropic=anisotropic)
                    model.configure_sharding(devs)
                    out = self._recon(model, sino)
                    np.testing.assert_allclose(
                        out, ref, rtol=self.TOL, atol=self.TOL,
                        err_msg=f"VCD recon mismatch: anisotropic={anisotropic} n_dev={n}")


if __name__ == "__main__":
    unittest.main()
