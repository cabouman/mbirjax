"""
Tests for sharded FBP filtering (ParallelBeamModel).

Both ``fbp_filter`` and its thin alias ``direct_filter`` are **user-facing**:
the input may be plain or view-sharded (a plain input is sharded on the view
axis at entry), and the OUTPUT form is chosen by the ``output_sharded`` kwarg —
a plain (gathered) array by default, the view-sharded device form when True.
Pipelined internal callers (``fbp_recon`` / ``direct_recon``) pass
``output_sharded=True`` so the filter output never round-trips through the host.

Under view-sharding the ramp filter is per-view, so each device filters its own
views with no cross-device communication.

These tests check:
  - the single-device path filters and preserves shape, returning a plain array;
  - ``output_sharded=True`` returns a view-sharded result (axis 0) with no
    gather, matching the single-device result to float noise, for both a plain
    and a pre-sharded input; the default returns a plain array;
  - ``direct_filter`` is a thin alias for ``fbp_filter`` (same result).

Runs on whatever devices conftest provides (real GPUs on a cluster, virtual CPU
devices otherwise).
"""
import unittest

# Import mbirjax before jax (device-setup-first ordering).
import mbirjax

import numpy as np
import jax
import jax.numpy as jnp

from conftest import preferred_devices, assert_sharded_allclose


def _make_model_and_sino(num_views=8, num_rows=16, num_channels=64, seed=0):
    """A small parallel-beam model plus a random sinogram of matching shape."""
    angles = jnp.linspace(0, jnp.pi, num_views, endpoint=False)
    model = mbirjax.ParallelBeamModel((num_views, num_rows, num_channels), angles)
    # Pin a single device so the bare model is a deterministic single-device reference regardless of
    # GPU count (auto-sharding now uses all GPUs by default); the sharded tests configure their own.
    model.configure_devices(1)
    rng = np.random.default_rng(seed)
    sino = jnp.asarray(rng.random((num_views, num_rows, num_channels), dtype=np.float32))
    return model, sino


class TestMultiAxisFbpFilter(unittest.TestCase):
    """Correctness gates for the MultiAxisParallelModel FBP filter.

    The old fbp_filter built a per-view directional 2-D ramp from jnp.gradient(angles) --
    the step between LIST-NEIGHBOR views.  Because this geometry is simultaneous fixed
    measurements with no acquisition trajectory, the angles list order is arbitrary, so
    that filter was order-dependent and ill-defined.  The fix uses the standard 1-D channel
    ramp + a uniform pi/num_views weight (the shared _apply_direct_recon_filter path), which
    is order-INVARIANT and reduces exactly to ParallelBeamModel.fbp_filter in the
    azimuth-only, zero-elevation, equally-spaced limit."""

    def test_fbp_filter_reorder_invariant(self):
        """Permuting the arbitrary view list (set unchanged) must not change the filtered
        result -- the defining property the old directional ramp violated (it gave
        rel-max ~8.5)."""
        rng = np.random.default_rng(7)
        num_views, num_rows, num_channels = 16, 32, 32
        az = np.linspace(0.0, np.pi, num_views, endpoint=False)
        el = np.deg2rad(rng.uniform(-15, 15, num_views))
        angles = np.stack([az, el], axis=1)
        perm = rng.permutation(num_views)

        m = mbirjax.MultiAxisParallelModel((num_views, num_rows, num_channels), jnp.asarray(angles))
        m_perm = mbirjax.MultiAxisParallelModel((num_views, num_rows, num_channels),
                                                jnp.asarray(angles[perm]))
        for model in (m, m_perm):
            model.configure_devices(1)
            model.auto_set_recon_geometry()

        y = jnp.asarray(rng.standard_normal((num_views, num_rows, num_channels), dtype=np.float32))
        f = np.asarray(m.fbp_filter(y))
        f_perm_aligned = np.asarray(m_perm.fbp_filter(y[perm]))[np.argsort(perm)]
        rel = float(np.max(np.abs(f - f_perm_aligned)) / np.max(np.abs(f)))
        self.assertLess(rel, 1e-6, f"fbp_filter not order-invariant: rel-max diff {rel:.3e}")

    def test_fbp_filter_reduces_to_parallel(self):
        """Zero elevation + equally-spaced azimuths: the multiaxis FBP filter must equal
        ParallelBeamModel.fbp_filter (same shared kernel, same scaling)."""
        num_views, num_rows, num_channels = 16, 32, 32
        az = np.linspace(0.0, np.pi, num_views, endpoint=False)
        m_ma = mbirjax.MultiAxisParallelModel((num_views, num_rows, num_channels),
                                              jnp.asarray(np.stack([az, np.zeros_like(az)], axis=1)))
        m_pb = mbirjax.ParallelBeamModel((num_views, num_rows, num_channels), jnp.asarray(az))
        # Pin identical voxel scaling so the only thing under test is the filter itself.
        for model in (m_ma, m_pb):
            model.configure_devices(1)
            model.set_params(delta_voxel=1.0, voxel_row_aspect=1.0)

        rng = np.random.default_rng(3)
        y = jnp.asarray(rng.random((num_views, num_rows, num_channels), dtype=np.float32))
        assert_sharded_allclose(np.asarray(m_ma.fbp_filter(y)), np.asarray(m_pb.fbp_filter(y)))


class TestFbpFilterSingleDevice(unittest.TestCase):
    """Single-device (pinned 1-device mesh) behavior: a plain input is filtered and shape-preserved."""

    def test_filter_runs_and_preserves_shape(self):
        model, sino = _make_model_and_sino()
        out = model.fbp_filter(sino)
        self.assertEqual(out.shape, sino.shape)
        # Default output_sharded=False: a plain (gathered) array, regardless of the
        # trivial 1-device mesh the model runs on.  The device form is checked in
        # TestFbpFilterSharded via output_sharded=True.
        self.assertNotIsInstance(getattr(out, 'sharding', None),
                                 jax.sharding.NamedSharding)

    def test_row_batch_non_divisible_matches(self):
        """A row count not divisible by the row-filter batch still filters every
        row correctly.

        The kernel (tomography_utils.apply_row_filter) scans overlapping B-row
        windows; when B does not divide views*rows the last window overlaps and
        recomputes a few rows (idempotent for a per-row filter).  A tiny,
        non-dividing batch must therefore match the default single-window result.
        """
        import mbirjax.tomography_utils as tu
        model, sino = _make_model_and_sino(num_views=8)
        ref = np.asarray(model.fbp_filter(sino))     # default batch (one window here)
        saved = tu.ROW_FILTER_BATCH
        try:
            tu.ROW_FILTER_BATCH = 7                   # won't divide views*rows
            jax.clear_caches()                        # force a re-trace at the new batch
            out = np.asarray(model.fbp_filter(sino))
        finally:
            tu.ROW_FILTER_BATCH = saved
            jax.clear_caches()
        assert_sharded_allclose(out, ref)


class TestFbpFilterSharded(unittest.TestCase):

    def setUp(self):
        self.devs = preferred_devices(2)
        if self.devs is None:
            self.skipTest("need >= 2 devices")

    def test_sharded_matches_single_device(self):
        """A plain sinogram into fbp_filter under a mesh is sharded at entry and
        matches the single-device result; the default returns a plain array and
        output_sharded=True returns the view-sharded device form."""
        model, sino = _make_model_and_sino(num_views=8)
        if model.get_params('recon_shape')[2] % 2 != 0:
            self.skipTest("num_slices not divisible by 2")

        # Single-device reference.
        ref = np.asarray(model.fbp_filter(sino))

        model.configure_devices(self.devs)

        # Default: plain (gathered) output.
        out_plain = model.fbp_filter(sino)
        self.assertNotIsInstance(getattr(out_plain, 'sharding', None),
                                 jax.sharding.NamedSharding)
        assert_sharded_allclose(np.asarray(out_plain), ref)

        # output_sharded=True: the device form, view-sharded on axis 0.
        out = model.fbp_filter(sino, output_sharded=True)
        self.assertIsInstance(out.sharding, jax.sharding.NamedSharding)
        self.assertEqual(out.sharding.spec[0], 'devices')
        assert_sharded_allclose(np.asarray(out), ref)

    def test_direct_filter_is_fbp_filter_alias(self):
        """direct_filter is a thin alias for fbp_filter: same contract
        (output form chosen by output_sharded), same values."""
        model, sino = _make_model_and_sino(num_views=8)
        if model.get_params('recon_shape')[2] % 2 != 0:
            self.skipTest("num_slices not divisible by 2")

        model.configure_devices(self.devs)
        sharded_in = model._shard_sinogram(sino)
        via_fbp = model.fbp_filter(sharded_in, output_sharded=True)
        via_direct = model.direct_filter(sharded_in, output_sharded=True)

        # Same device-form output (view-sharded, no gather) and same values at the
        # suite's single-shot 1e-5 gate (exact equality is never the gate for computed
        # floats: GPU kernels can reorder summation even between two calls of one
        # executable).
        self.assertIsInstance(via_direct.sharding, jax.sharding.NamedSharding)
        self.assertEqual(via_direct.sharding.spec[0], 'devices')
        assert_sharded_allclose(np.asarray(via_direct), np.asarray(via_fbp))

    def test_sharded_input_output_sharded_stays_sharded(self):
        """A pre-sharded sinogram with output_sharded=True stays sharded (no
        gather) and is correct; with the default it is gathered to plain."""
        model, sino = _make_model_and_sino(num_views=8)
        if model.get_params('recon_shape')[2] % 2 != 0:
            self.skipTest("num_slices not divisible by 2")
        ref = np.asarray(model.fbp_filter(sino))

        model.configure_devices(self.devs)
        sharded_in = model._shard_sinogram(sino)
        out = model.fbp_filter(sharded_in, output_sharded=True)
        self.assertIsInstance(out.sharding, jax.sharding.NamedSharding)
        assert_sharded_allclose(np.asarray(out), ref)

        out_plain = model.fbp_filter(sharded_in)
        self.assertNotIsInstance(getattr(out_plain, 'sharding', None),
                                 jax.sharding.NamedSharding)
        assert_sharded_allclose(np.asarray(out_plain), ref)


if __name__ == "__main__":
    unittest.main()
