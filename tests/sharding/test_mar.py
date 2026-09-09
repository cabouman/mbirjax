"""
Sharded MAR tests (``preprocess/mar.py``): device-count consistency of the beam-hardening
correction (with view AND slice padding), the constraint-update path, the empty-class guards, the
per-axis argmin, and the ``recon_plastic_metal`` output contract.

These exist because two real bugs lived in branches no test exercised: a flat pixel index applied
to the 3-D view-sharded sinogram silently grabbed a whole VIEW (only the residual-constraint branch
hit it -- forced here via ``tolerance``), and ``jnp.argmin``'s int32 index labels WRAP on >2^31
arrays (only full-size data hits it; these tests pin the small-scale behavior the per-axis
``_argmin_3d`` rewrite must preserve).
"""
import unittest
import numpy as np
import jax.numpy as jnp
import mbirjax as mj
import mbirjax.preprocess as mjp
from mbirjax.preprocess.mar import (_argmin_3d, _correct_plastic_sinogram,
                                    _est_plastic_metal_sinos_from_recon,
                                    _estimate_BH_model_params, _estimate_BH_model_params_using_OSQP,
                                    _find_most_violated_constraints, _generate_metal_exponent_list,
                                    _METAL_SUPPORT_FLOOR)
from conftest import preferred_devices, assert_sharded_allclose

# 31 views and 25 detector rows -> 25 recon slices: BOTH the view and slice axes pad at 3 devices,
# so the device-form (padded) code paths are exercised, not just the dividing case.
SINO_SHAPE = (31, 25, 28)


def _make_model(devices=None):
    angles = np.linspace(0, np.pi, SINO_SHAPE[0], endpoint=False)
    model = mj.ParallelBeamModel(SINO_SHAPE, angles)
    if devices is not None:
        model.configure_devices(devices)
    return model


def _make_phantom(model, with_metal=True):
    """Plastic cylinder (0.3 + small noise) with an optional dense metal blob (3.0)."""
    recon_shape = model.get_params('recon_shape')
    num_rows, num_cols, _ = recon_shape
    rng = np.random.default_rng(0)
    recon = np.zeros(recon_shape, np.float32)
    yy, xx = np.mgrid[0:num_rows, 0:num_cols]
    cylinder = ((yy - num_rows / 2) ** 2 + (xx - num_cols / 2) ** 2) < (min(num_rows, num_cols) / 2 - 3) ** 2
    recon[cylinder] = 0.3
    recon += (rng.normal(0, 0.01, recon_shape) * cylinder[:, :, None]).astype(np.float32)
    if with_metal:
        recon[num_rows // 2 - 2:num_rows // 2 + 2, num_cols // 2 - 2:num_cols // 2 + 2, :] = 3.0
    return recon


def _h_exponent_list(num_metal, order):
    metal_exp = _generate_metal_exponent_list(num_metal, order)
    cross_exp = _generate_metal_exponent_list(num_metal, order - 1)
    return ([(1,) + (0,) * num_metal] + [(1, *t) for t in cross_exp] + [(0, *t) for t in metal_exp],
            len(cross_exp))


class TestArgmin3d(unittest.TestCase):

    def test_matches_flat_argmin_including_ties(self):
        """_argmin_3d == unraveled flat argmin (index AND value), including tie cases: a duplicate
        minimum in a later view, a duplicate later in the same view, and an all-equal plateau
        (row-major first-occurrence tie-breaking)."""
        rng = np.random.default_rng(0)
        for trial in range(60):
            x = rng.normal(size=(7, 6, 5)).astype(np.float32)
            if trial % 4 == 1:
                pos = np.unravel_index(np.argmin(x), x.shape)
                x[(pos[0] + 2) % 7, pos[1], pos[2]] = x[pos]
            elif trial % 4 == 2:
                pos = np.unravel_index(np.argmin(x), x.shape)
                x[pos[0], (pos[1] + 1) % 6, (pos[2] + 2) % 5] = x[pos]
            elif trial % 4 == 3:
                x[:] = 1.0
            idx, val = _argmin_3d(jnp.asarray(x))
            flat = int(np.argmin(x))
            self.assertEqual(np.ravel_multi_index(idx, x.shape), flat)
            self.assertEqual(float(val), float(x.reshape(-1)[flat]))


class TestCorrectPlasticSinogramBigCounts(unittest.TestCase):

    def test_num_real_pixels_beyond_int32(self):
        """_correct_plastic_sinogram must accept a real-pixel count above 2^31 (a full-size
        sinogram, e.g. 1600x1617x1422 = 3.7e9): a Python int crossing into the jitted division
        is cast to int32 by jax and raised OverflowError before the float() fix.  The overflow
        depends on the COUNT's value, not the array size, so tiny arrays pin it."""
        shape = (4, 3, 5)
        plastic = jnp.full(shape, 0.5, jnp.float32)
        metal = [jnp.full(shape, 0.2, jnp.float32)]
        measured = jnp.full(shape, 0.7, jnp.float32)
        # One linear plastic column + one metal-only column (no cross terms).
        h_exponents = [(1, 0), (0, 1)]
        theta = jnp.array([1.0, 0.5], jnp.float32)
        view_mask = jnp.ones((shape[0], 1, 1), jnp.float32)
        corrected = _correct_plastic_sinogram(
            measured, plastic, metal, theta, h_exponents, num_cross_terms=0,
            num_metal_terms=1, p_normalization=1.0, gamma=0.05,
            view_mask=view_mask, num_real_pixels=2 ** 31 + 100)
        self.assertTrue(bool(jnp.all(jnp.isfinite(corrected))))


class TestCorrectSinoPlasticMetal(unittest.TestCase):

    def test_multi_device_consistency(self):
        """The corrected sinogram at 3 devices (padded views AND slices) matches the 1-device
        result.  Tolerance 1e-3, looser than the projector 1e-5: the BH fit's constraint SELECTION
        (argmin pixels) is discretely sensitive to reduce-order float noise, and theta differences
        then spread across the sinogram (measured rel_max ~2e-4 on this phantom)."""
        single = preferred_devices(1)
        multi = preferred_devices(3)
        if single is None or multi is None:
            self.skipTest("need >= 3 devices")
        model1 = _make_model(single)
        phantom = _make_phantom(model1)
        measured = np.array(model1.forward_project(phantom))

        ref = np.asarray(mjp.mar.correct_sino_plastic_metal(model1, measured, phantom, num_metal=1))
        model3 = _make_model(multi)
        out = np.asarray(mjp.mar.correct_sino_plastic_metal(model3, measured, phantom, num_metal=1))

        self.assertTrue(np.isfinite(ref).all() and np.isfinite(out).all())
        self.assertEqual(out.shape, measured.shape)      # padding cropped on return
        assert_sharded_allclose(out, ref, msg="corrected sinogram diverged across device counts",
                                tol=1e-3)

    def test_constraint_update_path(self):
        """Force BOTH constraint branches every iteration (tolerance=1e10) on a padded sharded
        model: exercises _get_row_H, the u_m pixel read, the A/u stacking, and the OSQP solve --
        the path where the flat-index bugs lived."""
        multi = preferred_devices(3)
        if multi is None:
            self.skipTest("need >= 3 devices")
        model = _make_model(multi)
        phantom = _make_phantom(model)
        measured = model.prepare_sino_for_devices(np.array(model.forward_project(phantom)))

        num_metal = 1
        h_exponents, num_cross = _h_exponent_list(num_metal, order=3)
        plastic, metals = _est_plastic_metal_sinos_from_recon(phantom, num_metal, model)
        plastic = plastic / jnp.max(jnp.abs(plastic))
        metals = [m / jnp.max(jnp.abs(m)) for m in metals]
        view_mask = model.sino_placement.real_mask(measured.ndim)

        theta = _estimate_BH_model_params(plastic, metals, measured, h_exponents, num_cross,
                                          alpha=1, beta=0.002, num_constraint_update_iter=3,
                                          tolerance=1e10, view_mask=view_mask)
        theta = np.asarray(theta)
        self.assertEqual(theta.shape, (len(h_exponents),))
        self.assertTrue(np.isfinite(theta).all())

    def test_empty_plastic_raises(self):
        """A recon whose Otsu plastic class is empty (two-valued: background 0 and metal 3.0 only)
        must fail fast with the actionable ValueError, not propagate NaNs from a 0/0 normalize."""
        single = preferred_devices(1)
        if single is None:
            self.skipTest("need >= 1 device")
        model = _make_model(single)
        phantom = _make_phantom(model, with_metal=True)
        phantom[(phantom > 0) & (phantom < 1.0)] = 0.0       # remove the plastic class entirely
        measured = np.array(model.forward_project(phantom))
        with self.assertRaisesRegex(ValueError, "plastic"):
            mjp.mar.correct_sino_plastic_metal(model, measured, phantom, num_metal=1)


class TestConstraintSelectionRobustness(unittest.TestCase):
    """Noisy measured sinograms have NEGATIVE pixels on air rays (log-domain noise).  Selecting one
    as a residual-positivity constraint where the metal estimates are zero poses ``0 <= y < 0`` --
    structurally infeasible -- and OSQP's infeasibility sentinel x (2143289344.0: the float32-NaN
    bit pattern as a value, which IS finite) then silently replaced theta and collapsed the
    corrected plastic to ~0 (the demo_10_artifacts 'beam_hardening_cupping' num_metal=1 collapse).
    These tests pin the three fix layers: support-restricted constraint selection, the RHS clamp
    at 0, and the OSQP solver-status guard."""

    @staticmethod
    def _fit_inputs():
        """Tiny fit inputs (p, [m], y) with y = 2 p + 1.5 m exactly, metal support on two pixels,
        one sub-floor metal pixel, and the most negative y pixel OFF the metal support."""
        shape = (4, 5, 6)
        rng = np.random.default_rng(3)
        p = rng.uniform(0.1, 1.0, shape).astype(np.float32)
        m = np.zeros(shape, np.float32)
        m[2, 1, 1], m[2, 1, 2] = 1.0, 0.5
        m[0, 0, 1] = 0.1 * _METAL_SUPPORT_FLOOR          # sub-floor: not eligible support
        y = (2.0 * p + 1.5 * m).astype(np.float32)
        y[0, 0, 0] = -0.5                                # air-ray noise pixel; m == 0 there
        return p, m, y

    def test_residual_argmin_restricted_to_metal_support(self):
        """The residual-constraint argmin must land on a pixel with real metal support, even when
        the globally most negative residual sits off support (the infeasible-constraint trap)."""
        p, m, y = self._fit_inputs()
        h_exponents, num_cross = _h_exponent_list(1, order=3)
        theta = jnp.zeros(len(h_exponents), jnp.float32)     # Sm == 0, so y - Sm == y
        _, _, idx_res, v_res = _find_most_violated_constraints(
            jnp.asarray(y), jnp.asarray(p), [jnp.asarray(m)], theta, h_exponents, num_cross)
        support = m > _METAL_SUPPORT_FLOOR
        self.assertLess(y.min(), 0)                                              # the trap exists...
        self.assertFalse(support[np.unravel_index(y.argmin(), y.shape)])         # ...and is off support
        self.assertTrue(support[idx_res])
        self.assertEqual(float(v_res), float(y[support].min()))

    def test_osqp_infeasible_returns_none(self):
        """An infeasible QP (0 * theta <= -1) must yield None, not OSQP's finite sentinel vector."""
        theta = _estimate_BH_model_params_using_OSQP(
            jnp.eye(2), jnp.zeros(2), jnp.zeros((1, 2)), jnp.array([-1.0]))
        self.assertIsNone(theta)

    def test_negative_off_support_pixel_does_not_poison_theta(self):
        """End to end through _estimate_BH_model_params: with the most negative measured pixel off
        the metal support, theta must come out finite and O(1).  Before the fix this exact setup
        made OSQP declare the QP primal infeasible and theta became the 2.14e9 sentinel."""
        p, m, y = self._fit_inputs()
        h_exponents, num_cross = _h_exponent_list(1, order=3)
        theta = np.asarray(_estimate_BH_model_params(
            jnp.asarray(p), [jnp.asarray(m)], jnp.asarray(y), h_exponents, num_cross,
            alpha=1, beta=0.002, num_constraint_update_iter=5))
        self.assertTrue(np.isfinite(theta).all())
        self.assertLess(np.max(np.abs(theta)), 100.0)

    def test_negative_measurement_on_support_cannot_explode_theta(self):
        """A negative measured pixel ON thin metal support (m = 0.02): unclamped, the constraint
        H_m theta <= y < 0 demands theta_3 <= y/m ~ -25 and the metal polynomial explodes; with the
        RHS clamped at 0 it only asks the polynomial to vanish there, so theta stays O(1)."""
        p, m, y = self._fit_inputs()
        m[0, 0, 0] = 0.02                                # thin but eligible support at y = -0.5
        h_exponents, num_cross = _h_exponent_list(1, order=3)
        theta = np.asarray(_estimate_BH_model_params(
            jnp.asarray(p), [jnp.asarray(m)], jnp.asarray(y), h_exponents, num_cross,
            alpha=1, beta=0.002, num_constraint_update_iter=5))
        self.assertTrue(np.isfinite(theta).all())
        self.assertLess(np.max(np.abs(theta)), 10.0)


class TestReconPlasticMetalContract(unittest.TestCase):

    def test_output_sharded_contract(self):
        """The output_sharded FORM contract: the default returns a host ndarray at the problem's real
        shape; output_sharded=True returns the slice-sharded device form (possibly slice-padded) with
        no gather; and the flag changes only the output form, not the computation.

        The form assertions are strict (deterministic).  The value comparison is deliberately LOOSE
        (tol=1e-3): the two calls are INDEPENDENT runs of the full pipeline (FDK + BH fit + VCD), and
        GPU run-to-run noise for that pipeline is context-dependent -- measured 6e-7..1.5e-5 across 15
        seeded same/cross-flag pairs on 3xH100, but 1.1e-4 in a pytest-process context -- so a tight
        gate here would be a flaky GPU-reproducibility test, not a contract test.  A real contract bug
        (wrong data, padded-form leak, missing gather) fails the strict asserts or shows as O(1).
        Flag-equivalence evidence: interleaved same/cross-flag runs on one model showed cross-flag
        diffs (max 1.5e-5) statistically identical to within-flag run-to-run noise (max 9.6e-6).
        The VCD partition RNG is seeded before each call so both draw identical partitions."""
        multi = preferred_devices(3)
        if multi is None:
            self.skipTest("need >= 3 devices")
        model = _make_model(multi)
        phantom = _make_phantom(model)
        measured = np.array(model.forward_project(phantom))
        real_shape = tuple(model.get_params('recon_shape'))

        np.random.seed(7)
        recon_host = mjp.recon_plastic_metal(model, measured, None, num_BH_iterations=1,
                                             num_metal=1, stop_threshold_change_pct=5.0)
        np.random.seed(7)
        recon_dev = mjp.recon_plastic_metal(model, measured, None, num_BH_iterations=1,
                                            num_metal=1, stop_threshold_change_pct=5.0,
                                            output_sharded=True)

        # Strict form contract: host default.
        self.assertIsInstance(recon_host, np.ndarray)
        self.assertEqual(recon_host.shape, real_shape)
        self.assertTrue(np.isfinite(recon_host).all())

        # Strict form contract: device form.  A jax array (not gathered), distributed over the
        # model's devices, slice axis at the device-form (padded) length.
        self.assertNotIsInstance(recon_dev, np.ndarray)
        self.assertEqual(set(recon_dev.devices()), set(multi))
        self.assertEqual(recon_dev.shape[-1], model.recon_placement.padded_size)
        gathered = np.asarray(model._gather_recon(recon_dev))
        self.assertEqual(gathered.shape, real_shape)         # gather crops the padding
        self.assertTrue(np.isfinite(gathered).all())

        # Loose same-computation sanity check (see docstring for the measured tolerance basis).
        assert_sharded_allclose(gathered, recon_host,
                                msg="output_sharded form diverged from the host default", tol=1e-3)


if __name__ == '__main__':
    unittest.main()
