"""
Merged tests for the sharded GEOMETRY-NEUTRAL projectors: cone, multiaxis-parallel,
and translation.

All three geometries shard the recon by **slice** and the sinogram by **view** using
the same geometry-neutral placement path (translation and multiaxis add no projector/
driver overrides; cone drives it via the banded back / gather-then-monolithic forward):

  * BACK is a reduce-scatter on the BANDED back kernel -- a slice is drawn from a RANGE
    of detector rows, so rows cannot be cropped; each view-owner banded-back-projects its
    views onto a slice band and the partials are summed onto the slice-owner.
  * FORWARD GATHERS the full slice cylinder onto each view-owner (per pixel-batch, to
    bound memory) and runs the MONOLITHIC forward (decision C); it does not band forward.

Once ``_supports_sharding()`` is True, even the single-device case runs on the placement
path (a trivial 1-device mesh), so the per-geometry gates in tests/geometries already
exercise the n_dev=1 path.  These tests add the MULTI-DEVICE gate: the sharded path must
match the single-device path to float noise, on back / forward / Hessian-diagonal
(coeff_power=2), for every geometry and its variant (cone: circular/helical; multiaxis
and translation: isotropic/anisotropic).

This module MERGES the former per-geometry files (test_cone_sharded.py,
test_multiaxis_sharded.py, test_translation_sharded.py), which shared byte-identical
helpers and identical test bodies -- only the model factory and variant label differed.
The identical helpers (_random_sino, _random_recon, _usable_device_counts) now live in
tests/sharding/conftest.py; the geometry-specific model factories and the (near-identical)
device sweep stay module-local and are driven by GEOMETRY_CONFIGS below.

Runs on whatever devices conftest provides (real GPUs on a cluster, virtual CPU devices
otherwise); self-skips device counts the small geometry's axes do not divide.  The
non-dividing (padded) case is covered separately by the padding tests (tests/sharding).

Notes preserved from the originals:
  * There is deliberately NO sharded VCD-recon test for multiaxis or translation.  The
    sharded VCD loop (reduce-scatter back / gather forward / halo-exchanged qGGMRF prior /
    partitioning / donation) is geometry-INDEPENDENT and is gated sharded by the cone +
    parallel VCD tests; those geometries add no loop overrides.  The cone VCD-loop gate is
    kept here (TestConeShardedRecon) exactly as before, cone-only.
"""
import unittest

# Import mbirjax before jax (device-setup-first ordering).
import mbirjax

import numpy as np
import jax
import jax.numpy as jnp

from conftest import (assert_sharded_allclose, _random_sino, _random_recon,
                      _usable_device_counts)


# ---------------------------------------------------------------------------
# Geometry-specific model factories (kept module-local -- they genuinely differ).
# In every factory the VARIANT flag is the first positional argument, so a config can
# build a model with ``factory(variant)`` regardless of geometry.
# ---------------------------------------------------------------------------
def _make_cone_model(helical=False, num_views=8, num_det_rows=24, num_det_channels=32):
    """Small cone model.  At magnification 2 the auto-sized recon has num_det_rows slices, so
    num_views=8 and num_det_rows=24 keep both sharded axes divisible by 2 and 4.  Pinned to a
    single device so the bare model is a deterministic single-device REFERENCE; the multi-device
    tests override with their own configure_devices(devs)."""
    angles = jnp.linspace(0, jnp.pi, num_views, endpoint=False)
    sdd = 4.0 * num_det_channels
    kwargs = dict(source_detector_dist=sdd, source_iso_dist=sdd / 2.0)
    if helical:
        kwargs['helical_z_shifts'] = np.linspace(-1.0, 1.0, num_views)
    model = mbirjax.ConeBeamModel((num_views, num_det_rows, num_det_channels), angles, **kwargs)
    model.configure_devices(1)
    return model


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


def _make_translation_model(anisotropic=False, num_views=8, num_det_rows=32, num_det_channels=32):
    """Small translation model.  num_views=8, num_det_rows=32 with a +-4 z translation range
    auto-sizes the recon to 16 slices isotropic / 8 slices anisotropic (aspect 2), so both the
    view axis (8) and the slice axis stay divisible by 2 and 4.  Pinned to a single device so the
    bare model is a deterministic single-device REFERENCE; the multi-device tests override with
    their own configure_devices(devs)."""
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


# Each geometry x its two variants.  ``variant_kw`` is only the label used in assertion
# messages (matching the original per-geometry files: cone said "helical=", the other two
# said "anisotropic=").  ``factory(variant)`` builds the model.
GEOMETRY_CONFIGS = (
    ("cone",        _make_cone_model,        "helical",     (False, True)),
    ("multiaxis",   _make_multiaxis_model,   "anisotropic", (False, True)),
    ("translation", _make_translation_model, "anisotropic", (False, True)),
)


def _sweep(test, factory, variant, variant_kw, ref_fn, shard_fn, label, tol):
    """Generalized version of the (near-identical) per-geometry ``_sweep``: build a
    single-device reference, then for each usable device count > 1 build a fresh sharded
    model and assert the sharded result matches the reference via the scale-invariant gate.
    """
    ref_model = factory(variant)
    counts = _usable_device_counts(ref_model)
    if not counts:
        test.skipTest("no usable device count > 1 (need >= 2 devices and divisible axes)")
    ref = np.asarray(ref_fn(ref_model))
    for n, devs in counts:
        model = factory(variant)
        model.configure_devices(devs)
        out = np.asarray(shard_fn(model))
        assert_sharded_allclose(out, ref, tol=tol,
                                msg=f"{label} mismatch: {variant_kw}={variant} n_dev={n}")


class TestGeometryShardedProjectors(unittest.TestCase):
    """Single-shot projector comparisons across cone/multiaxis/translation and their variants:
    sharded == single-device to float noise."""

    TOL = 1e-5

    def test_geometry_axes_divisible(self):
        """Guard against auto-sizer drift: the sharded view axis and slice axis must divide 4, so
        the n=2,4 sharded comparisons actually run rather than silently skip on a divisibility miss.
        (Folds in the former multiaxis-only test_geometry_axes_divisible, generalized to every
        geometry via the model's own shard-axis accessors -- the same axes _usable_device_counts
        uses.)"""
        for name, factory, variant_kw, variants in GEOMETRY_CONFIGS:
            for variant in variants:
                with self.subTest(geometry=name, **{variant_kw: variant}):
                    m = factory(variant)
                    sino_shape = m.get_params('sinogram_shape')
                    recon_shape = m.get_params('recon_shape')
                    sino_axis = m.sinogram_shard_axis() % len(sino_shape)
                    recon_axis = m.recon_shard_axis() % len(recon_shape)
                    nv = int(sino_shape[sino_axis])
                    ns = int(recon_shape[recon_axis])
                    self.assertEqual(
                        (nv % 4, ns % 4), (0, 0),
                        f"{name} {variant_kw}={variant}: sino_axis_len={nv} recon_axis_len={ns} "
                        f"not divisible by 4")

    def test_back_matches_single_device(self):
        for name, factory, variant_kw, variants in GEOMETRY_CONFIGS:
            for variant in variants:
                with self.subTest(geometry=name, **{variant_kw: variant}):
                    sino = _random_sino(factory(variant))
                    _sweep(self, factory, variant, variant_kw,
                           lambda m: m.back_project(sino),
                           lambda m: m.back_project(sino), "back", self.TOL)

    def test_forward_matches_single_device(self):
        for name, factory, variant_kw, variants in GEOMETRY_CONFIGS:
            for variant in variants:
                with self.subTest(geometry=name, **{variant_kw: variant}):
                    recon = _random_recon(factory(variant))
                    _sweep(self, factory, variant, variant_kw,
                           lambda m: m.forward_project(recon),
                           lambda m: m.forward_project(recon), "forward", self.TOL)

    def test_hessian_diagonal_matches_single_device(self):
        """coeff_power=2 (Hessian diagonal) sharded == single-device, via the scale-invariant gate
        (conftest.assert_sharded_allclose).  Squaring the projection coefficients gives the Hessian
        a large dynamic range, so the reduce-scatter summation-reorder noise (unbiased, ~1e-7 of the
        peak) lands on near-zero diagonal entries as a large RELATIVE diff that a fixed atol would
        false-fail/flake -- this is the case that motivated the scale-invariant ruler.  back/forward
        have no such squaring but use the same gate."""
        for name, factory, variant_kw, variants in GEOMETRY_CONFIGS:
            for variant in variants:
                with self.subTest(geometry=name, **{variant_kw: variant}):
                    weights = _random_sino(factory(variant), seed=7)
                    _sweep(self, factory, variant, variant_kw,
                           lambda m: m.compute_hessian_diagonal(weights),
                           lambda m: m.compute_hessian_diagonal(weights), "hessian", self.TOL)


class TestConeShardedRecon(unittest.TestCase):
    """The full VCD recon (projectors + qGGMRF prior) sharded == single-device.

    Kept CONE-ONLY (per the NOTE in the former per-geometry files): the sharded VCD loop is
    geometry-INDEPENDENT and is gated sharded by the cone + parallel VCD tests; multiaxis and
    translation add no loop overrides, so re-running it per geometry would only re-exercise the
    same shared machinery.
    """

    MAX_ITERS = 3
    TOL = 1e-4   # iterated: per-step FP-reorder differences accumulate (matches parallel sweep)

    def _recon(self, model, sino):
        np.random.seed(0)  # fix partitions + subset order so modes are comparable
        if model.shard_devices is not None:
            # Re-extract halos every subset -> the exact prior path (reproduces single-device).
            model._vcd_halo_per_subset = True
        model.set_params(verbose=0)  # Silence warnings about background
        recon, _ = model.recon(sino, max_iterations=self.MAX_ITERS,
                               stop_threshold_change_pct=0.0, print_logs=False)
        return np.asarray(recon)

    def test_vcd_recon_matches_single_device(self):
        for helical in (False, True):
            with self.subTest(helical=helical):
                ref_model = _make_cone_model(helical=helical)
                counts = _usable_device_counts(ref_model)
                if not counts:
                    self.skipTest("no usable device count > 1")
                sino = _random_sino(ref_model, seed=2)
                ref = self._recon(_make_cone_model(helical=helical), sino)
                for n, devs in counts:
                    model = _make_cone_model(helical=helical)
                    model.configure_devices(devs)
                    out = self._recon(model, sino)
                    assert_sharded_allclose(out, ref, tol=self.TOL,
                                            msg=f"VCD recon mismatch: helical={helical} n_dev={n}")


if __name__ == "__main__":
    unittest.main()
