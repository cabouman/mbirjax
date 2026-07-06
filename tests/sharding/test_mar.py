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
from mbirjax.preprocess.mar import (_argmin_3d, _est_plastic_metal_sinos_from_recon,
                                    _estimate_BH_model_params, _generate_metal_exponent_list)
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


class TestReconPlasticMetalContract(unittest.TestCase):

    def test_output_sharded_contract(self):
        """recon_plastic_metal returns a host ndarray by default and the sharded device form with
        output_sharded=True, and the two agree.  The VCD partition RNG is seeded before each call
        so the two runs draw identical partitions (comparing unseeded runs compares noise)."""
        multi = preferred_devices(3)
        if multi is None:
            self.skipTest("need >= 3 devices")
        model = _make_model(multi)
        phantom = _make_phantom(model)
        measured = np.array(model.forward_project(phantom))

        np.random.seed(7)
        recon_host = mjp.recon_plastic_metal(model, measured, None, num_BH_iterations=1,
                                             num_metal=1, stop_threshold_change_pct=5.0)
        np.random.seed(7)
        recon_dev = mjp.recon_plastic_metal(model, measured, None, num_BH_iterations=1,
                                            num_metal=1, stop_threshold_change_pct=5.0,
                                            output_sharded=True)

        self.assertIsInstance(recon_host, np.ndarray)
        self.assertEqual(recon_host.shape, tuple(model.get_params('recon_shape')))
        self.assertNotIsInstance(recon_dev, np.ndarray)      # device form
        assert_sharded_allclose(np.asarray(model._gather_recon(recon_dev)), recon_host,
                                msg="output_sharded form diverged from the host default", tol=1e-4)


if __name__ == '__main__':
    unittest.main()
