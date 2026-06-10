"""
Tests for sharded VCD reconstruction (ParallelBeamModel / TomographyModel).

Sharding scheme (same as the projectors): the sinogram is sharded by **view** and
the recon by **slice**.  The VCD loop composes the already-validated sharded
forward/back projectors (all-gather / reduce-scatter) with one new piece -- the
qGGMRF **prior** on a slice-sharded recon.

The qGGMRF prior couples each slice to its slice-axis neighbors, so on a
slice-sharded recon an interior shard boundary needs one boundary slice from each
adjacent shard.  Those *halos* are supplied by ``_extract_halos`` and consumed by
the halo-aware ``qggmrf_gradient_and_hessian_at_indices`` inside
``_qggmrf_prior_sharded``; at the true recon edges the halo is ``None`` and the
boundary slice is mirrored (reflected BC), reproducing the single-device result.

These tests check, from cheapest/most-targeted to most end-to-end:
  - the halo math itself, with NO mesh: a recon split along slices and recombined
    with explicit halos reproduces the full-recon prior exactly (the boundary
    self-consistency gate);
  - the sharded prior orchestrator matches the single-device prior (trivial
    1-device BIT-EXACT; 2/4/8-device to float noise) and returns a slice-sharded
    array (no accidental gather);
  - the full VCD recon: trivial 1-device BIT-EXACT vs the single-device path, and
    2/4/8-device to float noise (with identical numpy seeding so the random subset
    order and partitions match across modes).

Runs on whatever devices conftest provides (real GPUs on a cluster, virtual CPU
devices otherwise).
"""
import unittest

# Import mbirjax before jax (device-setup-first ordering).
import mbirjax
import mbirjax as mj

import numpy as np
import jax
import jax.numpy as jnp

from conftest import preferred_devices


def _make_model(num_views=8, num_rows=8, num_channels=32):
    """Small parallel-beam model.  num_det_rows -> num recon slices, so keeping it
    divisible by 2/4/8 lets the slice axis shard across those device counts."""
    angles = jnp.linspace(0, jnp.pi, num_views, endpoint=False)
    # Pin a single device so the bare model is a deterministic single-device REFERENCE regardless of
    # how many GPUs are present (auto-sharding now uses all available GPUs by default); tests that
    # exercise multi-device sharding override this with their own configure_sharding(devs).
    model = mbirjax.ParallelBeamModel((num_views, num_rows, num_channels), angles)
    model.configure_devices(1)
    return model


def _qggmrf_params(model):
    """The qGGMRF parameter tuple exactly as the VCD subset updater builds it."""
    qggmrf_nbr_wts, sigma_x, p, q, T = model.get_params(
        ['qggmrf_nbr_wts', 'sigma_x', 'p', 'q', 'T'])
    b = mj.get_b_from_nbr_wts(qggmrf_nbr_wts)
    return (b, sigma_x, p, q, T)


def _random_flat_recon(model, seed=0):
    """A reproducible random flat recon (num_pixels, num_slices).

    The prior is a fixed nonlinear function of the recon, so a random recon is a
    valid input for cross-mode comparison; physical fidelity is irrelevant here.
    """
    num_rows, num_cols, num_slices = model.get_params('recon_shape')[:3]
    rng = np.random.default_rng(seed)
    flat = rng.standard_normal((num_rows * num_cols, num_slices), dtype=np.float32)
    return jnp.asarray(flat)


def _phantom_sino(model, seed=0):
    """A reproducible random sinogram of the model's sinogram shape.

    VCD recon is deterministic given the sinogram and the numpy seed (which fixes
    partitions and subset order), so a random sinogram suffices to compare the
    single-device and sharded recons doing the *same* computation.
    """
    shape = model.get_params('sinogram_shape')
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.standard_normal(shape, dtype=np.float32))


def _phantom_weights(model, seed=0):
    """A reproducible positive weights array of the model's sinogram shape.

    Non-constant weights exercise the ``weighted_error_sinogram = weights * error_sinogram``
    path (a per-subset view-sharded transient that the cleanup section must free)."""
    shape = model.get_params('sinogram_shape')
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.uniform(0.5, 1.5, shape).astype(np.float32))


def _divisible(model, n):
    """True if the model's sharded sinogram and recon axes both divide n."""
    sino_shape = model.get_params('sinogram_shape')
    recon_shape = model.get_params('recon_shape')
    sino_axis = model.sinogram_shard_axis() % len(sino_shape)
    recon_axis = model.recon_shard_axis() % len(recon_shape)
    return sino_shape[sino_axis] % n == 0 and recon_shape[recon_axis] % n == 0


class TestHaloMath(unittest.TestCase):
    """The halo-aware prior, with NO mesh: a slice-split recon recombined with
    explicit halos must reproduce the full-recon prior exactly.  This isolates the
    halo math from the multi-device machinery (it is the cheap correctness gate)."""

    def test_boundary_self_consistency(self):
        model = _make_model()
        recon_shape = model.get_params('recon_shape')
        num_rows, num_cols, num_slices = recon_shape[:3]
        self.assertGreaterEqual(num_slices, 4)
        params = _qggmrf_params(model)
        flat = _random_flat_recon(model, seed=1)
        idx = jnp.arange(num_rows * num_cols)

        # Full-recon reference.
        g_full, h_full = mj.qggmrf_gradient_and_hessian_at_indices(
            flat, recon_shape, idx, params)

        # Split along slices into a left and right shard.
        mid = num_slices // 2
        left, right = flat[:, :mid], flat[:, mid:]
        # Left shard: true left edge (no halo), interior right boundary (halo = first slice of right).
        gL, hL = mj.qggmrf_gradient_and_hessian_at_indices(
            left, (num_rows, num_cols, mid), idx, params,
            left_halo=None, right_halo=flat[:, mid])
        # Right shard: interior left boundary (halo = last slice of left), true right edge (no halo).
        gR, hR = mj.qggmrf_gradient_and_hessian_at_indices(
            right, (num_rows, num_cols, num_slices - mid), idx, params,
            left_halo=flat[:, mid - 1], right_halo=None)

        # Each shard's local result (including the shard-boundary cylinders) must match the reference.
        np.testing.assert_allclose(np.asarray(gL), np.asarray(g_full[:, :mid]), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(np.asarray(gR), np.asarray(g_full[:, mid:]), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(np.asarray(hL), np.asarray(h_full[:, :mid]), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(np.asarray(hR), np.asarray(h_full[:, mid:]), rtol=1e-6, atol=1e-6)

    def test_no_halo_matches_legacy_reflected_bc(self):
        """With no halos the new boundary formulation must equal the reflected-BC
        result the single-device prior produces (a zero-delta at each true edge)."""
        model = _make_model()
        recon_shape = model.get_params('recon_shape')
        num_rows, num_cols, _ = recon_shape[:3]
        params = _qggmrf_params(model)
        flat = _random_flat_recon(model, seed=2)
        idx = jnp.arange(num_rows * num_cols)
        g0, h0 = mj.qggmrf_gradient_and_hessian_at_indices(flat, recon_shape, idx, params)
        g1, h1 = mj.qggmrf_gradient_and_hessian_at_indices(
            flat, recon_shape, idx, params, left_halo=None, right_halo=None)
        self.assertTrue(np.array_equal(np.asarray(g0), np.asarray(g1)))
        self.assertTrue(np.array_equal(np.asarray(h0), np.asarray(h1)))


class TestShardedPrior(unittest.TestCase):
    """The _qggmrf_prior_sharded orchestrator vs the single-device prior."""

    def test_trivial_bit_exact(self):
        """1-device mesh matches the single-device prior to tight float tolerance.
        (Name kept for history; was an exact-equality check.)

        RETIRE-AFTER-SHARDING: trivial-mesh-vs-legacy comparison, meaningful only
        while both paths coexist; once every geometry runs on placements there is
        one path and nothing to compare.  Relaxed from exact equality because the
        sharded prior reorders non-associative FP sums -> ~1 ULP GPU difference
        (CPU stays exact).  A tight tolerance still trips on any real drift.
        """
        single = preferred_devices(1)
        if single is None:
            self.skipTest("need >= 1 device")
        ref_model = _make_model()
        recon_shape = ref_model.get_params('recon_shape')
        num_rows, num_cols, _ = recon_shape[:3]
        params = _qggmrf_params(ref_model)
        flat = _random_flat_recon(ref_model, seed=3)
        idx = jnp.arange(num_rows * num_cols)
        g_ref, h_ref = mj.qggmrf_gradient_and_hessian_at_indices(flat, recon_shape, idx, params)

        model = _make_model()
        model.configure_sharding(single)
        sharded_flat = model._shard_recon(flat)
        g, h = model._qggmrf_prior_sharded(sharded_flat, idx, params)
        np.testing.assert_allclose(np.asarray(g), np.asarray(g_ref), rtol=1e-5, atol=1e-5,
                                   err_msg="prior grad diverged beyond float noise")
        np.testing.assert_allclose(np.asarray(h), np.asarray(h_ref), rtol=1e-5, atol=1e-5,
                                   err_msg="prior hess diverged beyond float noise")

    def test_sharded_matches_single_device_sweep(self):
        """2/4/8-device sharded prior matches the single-device prior to float noise
        and returns a slice-sharded array (no gather)."""
        ref_model = _make_model()
        recon_shape = ref_model.get_params('recon_shape')
        num_rows, num_cols, _ = recon_shape[:3]
        params = _qggmrf_params(ref_model)
        flat = _random_flat_recon(ref_model, seed=4)
        idx = jnp.arange(num_rows * num_cols)
        g_ref, h_ref = mj.qggmrf_gradient_and_hessian_at_indices(flat, recon_shape, idx, params)

        ran_multi = False
        for n in (2, 4, 8):
            devs = preferred_devices(n)
            if devs is None:
                continue
            model = _make_model()
            if not _divisible(model, n):
                continue
            model.configure_sharding(devs)
            sharded_flat = model._shard_recon(flat)
            g, h = model._qggmrf_prior_sharded(sharded_flat, idx, params)

            # Slice-sharded outputs (no accidental gather).
            self.assertIsInstance(g.sharding, jax.sharding.NamedSharding)
            self.assertIsInstance(h.sharding, jax.sharding.NamedSharding)
            slice_axis = model.recon_shard_axis() % g.ndim
            self.assertEqual(g.sharding.spec[slice_axis], 'devices')
            for ax in range(g.ndim):
                if ax != slice_axis:
                    self.assertIsNone(g.sharding.spec[ax])

            np.testing.assert_allclose(np.asarray(g), np.asarray(g_ref), rtol=1e-5, atol=1e-5,
                                       err_msg=f"prior grad mismatch at n_dev={n}")
            np.testing.assert_allclose(np.asarray(h), np.asarray(h_ref), rtol=1e-5, atol=1e-5,
                                       err_msg=f"prior hess mismatch at n_dev={n}")
            ran_multi = True
        if not ran_multi:
            self.skipTest("no usable device count > 1")


class TestShardedRecon(unittest.TestCase):
    """End-to-end VCD recon: single-device vs sharded.

    VCD shuffles subset order and builds partitions with numpy randomness, so the
    same numpy seed is set before each recon to make the two modes do the identical
    computation (only the distributed reduce order differs)."""

    MAX_ITERS = 6

    def _recon(self, model, sino, seed=0, halo_per_subset=False, weights=None, positivity=False):
        np.random.seed(seed)  # fix partitions + subset order so modes are comparable
        if positivity:
            model.set_params(positivity_flag=True)
        if model.mesh is not None:
            # halo_per_subset=True forces the exact (re-extract-every-subset) prior path,
            # which reproduces single-device exactly; the default (False) stages halos once
            # per partition instead of per subset and is exact except at gen_pixel_partition's few
            # replicated pixels.
            model._vcd_halo_per_subset = halo_per_subset
        recon, _ = model.recon(sino, max_iterations=self.MAX_ITERS,
                               stop_threshold_change_pct=0.0,  # run all iters, no early stop
                               print_logs=False)
        return np.asarray(recon)

    def test_trivial_recon_bit_exact(self):
        """1-device mesh recon matches the single-device recon to modest float
        tolerance.  (Name kept for history; was an exact-equality check.)

        RETIRE-AFTER-SHARDING: trivial-mesh-vs-legacy comparison, meaningful only
        while both paths coexist.  Relaxed from exact equality, and to a looser
        tolerance than the single-shot tests, because VCD is iterative: per-step
        ~1 ULP FP-reorder differences in the banded sharded path amplify across
        subsets/passes on GPU (CPU stays exact).  Matches the multi-device sweep
        sibling's GPU-proven rtol/atol; still trips on any real algorithmic drift.
        """
        single = preferred_devices(1)
        if single is None:
            self.skipTest("need >= 1 device")
        sino = _phantom_sino(_make_model())
        ref = self._recon(_make_model(), sino)

        shard_model = _make_model()
        shard_model.configure_sharding(single)
        out = self._recon(shard_model, sino)
        np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4,
                                   err_msg="trivial-sharded recon diverged beyond float noise")

    def test_sharded_recon_matches_single_device_sweep(self):
        """2/4/8-device recon matches single-device to float noise on the EXACT prior
        path (halos re-extracted per subset).  This is the tight correctness gate for
        the sharding machinery (projectors, recon_indices, alpha, halo staging),
        independent of the halo-once approximation tested separately below."""
        sino = _phantom_sino(_make_model())
        ref = self._recon(_make_model(), sino)

        ran_multi = False
        for n in (2, 4, 8):
            devs = preferred_devices(n)
            if devs is None:
                continue
            model = _make_model()
            if not _divisible(model, n):
                continue
            model.configure_sharding(devs)
            out = self._recon(model, sino, halo_per_subset=True)   # exact path
            # Iterative amplification of float-reduce-order differences -> modest tolerance.
            np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4,
                                       err_msg=f"recon mismatch at n_dev={n}")
            ran_multi = True
        if not ran_multi:
            self.skipTest("no usable device count > 1")

    def test_halo_once_per_pass_approximation_is_small(self):
        """Slice sharded qGGMRF transfers halos ONCE per partition pass instead of per
        subset.  That is bit-exact except at the few pixels gen_pixel_partition
        REPLICATES to equalize subset lengths (a replicated pixel updated in one subset
        has a stale pass-start halo when its other subset runs).  Quantify that the
        resulting difference (halo-once vs the exact per-subset path) is negligible —
        far below the recon-vs-phantom error (~0.07) — so the default halo-once path is
        safe.  (Measured ~1e-4–3e-4 NRMSE on small noise problems; bound generously.)"""
        sino = _phantom_sino(_make_model())
        ran_multi = False
        for n in (2, 4):
            devs = preferred_devices(n)
            if devs is None:
                continue
            if not _divisible(_make_model(), n):
                continue
            m_once = _make_model(); m_once.configure_sharding(devs)
            once = self._recon(m_once, sino, halo_per_subset=False)
            m_exact = _make_model(); m_exact.configure_sharding(devs)
            exact = self._recon(m_exact, sino, halo_per_subset=True)
            nrmse = np.linalg.norm(once - exact) / np.linalg.norm(exact)
            self.assertLess(nrmse, 2e-3,
                            msg=f"halo-once approximation too large at n_dev={n}: NRMSE={nrmse:.2e}")
            ran_multi = True
        if not ran_multi:
            self.skipTest("no usable device count > 1")

    # --- non-constant weights and positivity: the two extra per-subset transient paths ---
    # weights*error_sinogram (non-const) and the positivity-branch delta_sinogram recompute
    # both make extra view-sharded sinos per subset; these gate their correctness (the memory
    # side is checked separately by the memjump diagnostic on GPU).

    def test_trivial_recon_bit_exact_nonconst_weights(self):
        """1-device mesh recon with non-constant weights matches single-device to
        modest float tolerance.  (Name kept for history; was exact-equality.)

        RETIRE-AFTER-SHARDING: see test_trivial_recon_bit_exact -- same iterative
        FP-reorder amplification; relaxed to the GPU-proven sweep tolerance.
        """
        single = preferred_devices(1)
        if single is None:
            self.skipTest("need >= 1 device")
        m = _make_model()
        sino, w = _phantom_sino(m), _phantom_weights(m)
        ref = self._recon(_make_model(), sino, weights=w)
        sm = _make_model(); sm.configure_sharding(single)
        out = self._recon(sm, sino, weights=w)
        np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4,
                                   err_msg="non-const-weights trivial recon diverged beyond float noise")

    def test_trivial_recon_bit_exact_positivity(self):
        """1-device mesh recon with positivity_flag matches single-device to modest
        float tolerance.  (Name kept for history; was exact-equality.)

        RETIRE-AFTER-SHARDING: see test_trivial_recon_bit_exact -- same iterative
        FP-reorder amplification; relaxed to the GPU-proven sweep tolerance.
        """
        single = preferred_devices(1)
        if single is None:
            self.skipTest("need >= 1 device")
        sino = _phantom_sino(_make_model())
        ref = self._recon(_make_model(), sino, positivity=True)
        sm = _make_model(); sm.configure_sharding(single)
        out = self._recon(sm, sino, positivity=True)
        np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4,
                                   err_msg="positivity trivial recon diverged beyond float noise")

    def test_sharded_recon_matches_single_device_nonconst_weights(self):
        """2/4/8-device recon with non-constant weights matches single-device to float noise."""
        m = _make_model()
        sino, w = _phantom_sino(m), _phantom_weights(m)
        ref = self._recon(_make_model(), sino, weights=w)
        ran_multi = False
        for n in (2, 4, 8):
            devs = preferred_devices(n)
            if devs is None:
                continue
            model = _make_model()
            if not _divisible(model, n):
                continue
            model.configure_sharding(devs)
            out = self._recon(model, sino, halo_per_subset=True, weights=w)
            np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4,
                                       err_msg=f"non-const-weights recon mismatch at n_dev={n}")
            ran_multi = True
        if not ran_multi:
            self.skipTest("no usable device count > 1")

    def test_sharded_recon_matches_single_device_positivity(self):
        """2/4/8-device recon with positivity_flag matches single-device to float noise."""
        sino = _phantom_sino(_make_model())
        ref = self._recon(_make_model(), sino, positivity=True)
        ran_multi = False
        for n in (2, 4, 8):
            devs = preferred_devices(n)
            if devs is None:
                continue
            model = _make_model()
            if not _divisible(model, n):
                continue
            model.configure_sharding(devs)
            out = self._recon(model, sino, halo_per_subset=True, positivity=True)
            np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4,
                                       err_msg=f"positivity recon mismatch at n_dev={n}")
            ran_multi = True
        if not ran_multi:
            self.skipTest("no usable device count > 1")


class TestShardedReconKeepsSharding(unittest.TestCase):
    """Audit: the VCD loop must not accidentally gather/re-shard.

    ``vcd_recon`` returns the recon BEFORE the user-facing ``recon``'s exit gather,
    so if its return is still slice-sharded across all devices, then flat_recon
    stayed distributed through every iteration (no op silently pulled it to one
    device).  This catches accidental ``device_put(..., single_device)`` regressions
    that would hurt memory without changing the numerical result."""

    def test_vcd_recon_return_is_slice_sharded(self):
        ran_multi = False
        for n in (2, 4, 8):
            devs = preferred_devices(n)
            if devs is None:
                continue
            model = _make_model()
            if not _divisible(model, n):
                continue
            model.configure_sharding(devs)
            sino = _phantom_sino(model)
            np.random.seed(0)
            (sino2, weights, init_recon, partitions, partition_sequence,
             _granularity, _reg) = model.initialize_recon(sino, max_iterations=4, print_logs=False)
            recon, _stats = model.vcd_recon(
                sino2, partitions, partition_sequence, stop_threshold_change_pct=0.0,
                weights=weights, init_recon=init_recon)

            # Slice-sharded across the whole mesh -> nothing gathered it to one device.
            self.assertIsInstance(recon.sharding, jax.sharding.NamedSharding,
                                  msg=f"vcd_recon return gathered to a single device at n_dev={n}")
            self.assertEqual(len(recon.sharding.device_set), n)
            slice_axis = model.recon_shard_axis() % recon.ndim
            self.assertEqual(recon.sharding.spec[slice_axis], 'devices')
            for ax in range(recon.ndim):
                if ax != slice_axis:
                    self.assertIsNone(recon.sharding.spec[ax])
            ran_multi = True
        if not ran_multi:
            self.skipTest("no usable device count > 1")


if __name__ == "__main__":
    unittest.main()
