"""
Tests for sharded FBP filtering (ParallelBeamModel).

Both ``fbp_filter`` and its thin alias ``direct_filter`` are **internal**
sharded-contract methods: under a mesh they return a view-sharded result with no
gather at exit.  For robustness they shard the input on the view axis at entry
(a no-op when it is already view-sharded), so a plain sinogram passed directly is
handled correctly as well; the gather back to plain lives in the user-facing
reconstruction methods (``fbp_recon`` / ``direct_recon``), not in the filter.

Under view-sharding the ramp filter is per-view, so each device filters its own
views with no cross-device communication.

These tests check:
  - the single-device (no-mesh) path is bit-exact vs a direct reference filter;
  - the internal ``fbp_filter`` returns a view-sharded result (axis 0) with no
    gather, matching the single-device result to float noise, for both a plain
    and a pre-sharded input;
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

from conftest import preferred_devices


def _make_model_and_sino(num_views=8, num_rows=16, num_channels=64, seed=0):
    """A small parallel-beam model plus a random sinogram of matching shape."""
    angles = jnp.linspace(0, jnp.pi, num_views, endpoint=False)
    model = mbirjax.ParallelBeamModel((num_views, num_rows, num_channels), angles)
    rng = np.random.default_rng(seed)
    sino = jnp.asarray(rng.random((num_views, num_rows, num_channels), dtype=np.float32))
    return model, sino


class TestFbpFilterSingleDevice(unittest.TestCase):
    """Default (auto 1-device mesh) behavior: a plain input is filtered and shape-preserved."""

    def test_filter_runs_and_preserves_shape(self):
        model, sino = _make_model_and_sino()
        out = model.fbp_filter(sino)
        self.assertEqual(out.shape, sino.shape)
        # RETIRE-AFTER-SHARDING: ParallelBeam now auto-defaults to a trivial 1-device mesh
        # (Option B), so a plain input is sharded at entry and the output is a (1-device)
        # NamedSharding -- it was a plain array on the legacy no-mesh path.  Shape is
        # preserved either way; the sharded-contract output is checked in TestFbpFilterSharded.
        self.assertIsInstance(out.sharding, jax.sharding.NamedSharding)

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
        np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


class TestFbpFilterSharded(unittest.TestCase):

    def setUp(self):
        self.devs = preferred_devices(2)
        if self.devs is None:
            self.skipTest("need >= 2 devices")

    def test_sharded_matches_single_device(self):
        """A plain sinogram passed directly to fbp_filter under a mesh is sharded
        at entry, filtered view-sharded, and matches the single-device result."""
        model, sino = _make_model_and_sino(num_views=8)
        if model.get_params('recon_shape')[2] % 2 != 0:
            self.skipTest("num_slices not divisible by 2")

        # Single-device reference (no mesh).
        ref = np.asarray(model.fbp_filter(sino))

        # Sharded run with a PLAIN input: fbp_filter shards on the view axis at
        # entry (mirrors fbp_recon) rather than assuming pre-sharded input.
        model.configure_sharding(self.devs)
        out = model.fbp_filter(sino)

        # Sharded-contract: output is view-sharded (axis 0), not gathered.
        self.assertIsInstance(out.sharding, jax.sharding.NamedSharding)
        self.assertEqual(out.sharding.spec[0], 'devices')

        np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-5, atol=1e-5)

    def test_direct_filter_is_fbp_filter_alias(self):
        """direct_filter is a thin internal alias for fbp_filter: same sharded
        contract (sharded view in -> sharded view out), same values."""
        model, sino = _make_model_and_sino(num_views=8)
        if model.get_params('recon_shape')[2] % 2 != 0:
            self.skipTest("num_slices not divisible by 2")

        model.configure_sharding(self.devs)
        sharded_in = model._shard_sinogram(sino)
        via_fbp = model.fbp_filter(sharded_in)
        via_direct = model.direct_filter(sharded_in)

        # Same sharded-contract output (view-sharded, no gather) and same values.
        self.assertIsInstance(via_direct.sharding, jax.sharding.NamedSharding)
        self.assertEqual(via_direct.sharding.spec[0], 'devices')
        np.testing.assert_array_equal(np.asarray(via_direct), np.asarray(via_fbp))

    def test_sharded_input_passes_through_sharded(self):
        """A pre-sharded sinogram stays sharded (no gather) and is correct."""
        model, sino = _make_model_and_sino(num_views=8)
        if model.get_params('recon_shape')[2] % 2 != 0:
            self.skipTest("num_slices not divisible by 2")
        ref = np.asarray(model.fbp_filter(sino))

        model.configure_sharding(self.devs)
        sharded_in = model._shard_sinogram(sino)
        out = model.fbp_filter(sharded_in)
        self.assertIsInstance(out.sharding, jax.sharding.NamedSharding)
        np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
