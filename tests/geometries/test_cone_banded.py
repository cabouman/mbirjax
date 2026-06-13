"""
tests/geometries/test_cone_banded.py
─────────────────────────────────────
Unit tests for the cone banded projector kernels:
``ConeBeamModel.back_project_one_view_to_band`` / ``forward_project_band_to_one_view``.

These are the strong per-geometry correctness gates for the banded interface, run
BEFORE the kernels are wired into the projectors/sharding.  They check:

  * band-decomposition: the monolithic per-view projector equals the sum (forward) /
    concatenation (back) of the banded projector over a tiling of the slice axis --
    because the bands tile disjointly, so each contribution is counted exactly once;
  * full-band == monolithic: a single band (g0=0, L=S) reproduces the monolithic
    kernel (the anchor is faithful on full cylinders);
  * adjoint-at-(g0, L): <A_band x, y> == <x, A_bandᵀ y> for an arbitrary band
    (forward_band and back_band are exact adjoints);
  * the Hessian path (coeff_power=2) decomposes the same way.

Every check runs on a CIRCULAR and a HELICAL cone geometry, so the anchor's
helical-z-shift term is exercised (a helical view has a nonzero per-view z shift).

Computed floats -> tight allclose (project rule: never exact equality for computed
floats).  Band decomposition only reorders summation, so CPU agreement is ~1e-6;
the gates use 1e-4 to stay robust on GPU (projector scatter-add noise ~8e-6 rel).

RETIRE: the band-decomposition, full-band-equals-monolithic, and Hessian-decomposition
tests COMPARE the banded kernels against the monolithic
forward_project_pixel_batch_to_one_view / back_project_one_view_to_pixel_batch; they
retire when those monolithic kernels are deleted (after the projectors are switched
to the banded path).  The adjoint-at-(g0, L) test is self-contained (it needs no
monolithic reference) and stays as a permanent gate.
"""
import unittest
from collections import namedtuple

import numpy as np
import jax
import jax.numpy as jnp

import mbirjax as mj
from mbirjax import ConeBeamModel


def make_projector_params(model):
    """Rebuild the ProjectorParams namedtuple exactly as projectors.py does, so the
    cone per-view kernels can be called directly."""
    gp = model.get_geometry_parameters()
    sinogram_shape, recon_shape = model.get_params(['sinogram_shape', 'recon_shape'])
    PP = namedtuple('ProjectorParams', ['sinogram_shape', 'recon_shape', 'geometry_params'])
    return PP(sinogram_shape, recon_shape, gp)


def band_bounds(num_slices):
    """A non-uniform tiling of [0, num_slices) into 3 bands (stresses arbitrary
    (g0, L), including unequal lengths)."""
    a = num_slices // 3
    b = 2 * num_slices // 3
    return [(0, a), (a, b), (b, num_slices)]


def make_config(num_views, num_det_rows, num_det_channels, helical):
    """Build a small, NON-symmetric cone geometry (distinct views/rows/channels so
    shape/axis bugs surface) at magnification 2 (the test-suite convention).  When
    ``helical`` is True the views carry nonzero z shifts, so the anchor's
    helical-z-shift term is exercised.  Returns a dict of everything the kernels need.
    """
    angles = np.linspace(0, np.pi, num_views, endpoint=False)
    sdd = 4.0 * num_det_channels
    kwargs = dict(source_detector_dist=sdd, source_iso_dist=sdd / 2.0)
    if helical:
        # A modest helix so the recon (auto-sized to cover the z travel) stays small.
        kwargs['helical_z_shifts'] = np.linspace(-1.0, 1.0, num_views)
    model = ConeBeamModel((num_views, num_det_rows, num_det_channels), angles, **kwargs)
    model.set_params(verbose=0)
    recon_shape = tuple(int(x) for x in model.get_params('recon_shape'))
    idx = jnp.asarray(mj.gen_full_indices(recon_shape, use_ror_mask=model.get_params('use_ror_mask')))
    view_params = jnp.asarray(model.get_params('view_params_array'))
    # Use the LAST view for helical (largest |z shift|), view 0 for circular.
    svp = view_params[-1] if helical else view_params[0]
    return {
        'name': 'helical' if helical else 'circular',
        'pp': make_projector_params(model), 'svp': svp, 'idx': idx,
        'S': recon_shape[2], 'num_pixels': int(idx.shape[0]),
        'num_det_rows': num_det_rows, 'num_det_channels': num_det_channels,
    }


class TestConeBandedProjector(unittest.TestCase):

    RTOL, ATOL = 1e-4, 1e-4

    @classmethod
    def setUpClass(cls):
        cls.configs = [make_config(9, 24, 32, helical=False),
                       make_config(9, 24, 32, helical=True)]
        cls.rng = np.random.default_rng(1234)

    # ── helpers (operate on a config dict) ──────────────────────────────────────
    def _back_monolithic(self, c, view, coeff_power=1):
        return np.asarray(ConeBeamModel.back_project_one_view_to_pixel_batch(
            view, c['idx'], c['svp'], c['pp'], coeff_power=coeff_power))

    def _back_banded_concat(self, c, view, coeff_power=1):
        bands = [np.asarray(ConeBeamModel.back_project_one_view_to_band(
            view, c['idx'], c['svp'], c['pp'], g0, g1 - g0, coeff_power=coeff_power))
            for (g0, g1) in band_bounds(c['S'])]
        return np.concatenate(bands, axis=1)   # (num_pixels, S)

    def _forward_monolithic(self, c, voxels):
        return np.asarray(ConeBeamModel.forward_project_pixel_batch_to_one_view(
            voxels, c['idx'], c['svp'], c['pp']))

    def _forward_banded_sum(self, c, voxels):
        acc = np.zeros((c['num_det_rows'], c['num_det_channels']), np.float64)
        for (g0, g1) in band_bounds(c['S']):
            acc += np.asarray(ConeBeamModel.forward_project_band_to_one_view(
                voxels[:, g0:g1], c['idx'], c['svp'], c['pp'], g0))
        return acc

    def _rand_view(self, c):
        return jnp.asarray(self.rng.standard_normal((c['num_det_rows'], c['num_det_channels']), np.float32))

    def _rand_voxels(self, c):
        return jnp.asarray(self.rng.standard_normal((c['num_pixels'], c['S']), np.float32))

    # ── band-decomposition (RETIRE with the monolithic kernels) ─────────────────
    def test_back_band_decomposition(self):
        """RETIRE-with-monolithic: monolithic back == concat of banded back."""
        for c in self.configs:
            with self.subTest(geometry=c['name']):
                view = self._rand_view(c)
                np.testing.assert_allclose(self._back_banded_concat(c, view),
                                           self._back_monolithic(c, view),
                                           rtol=self.RTOL, atol=self.ATOL)

    def test_back_full_band_equals_monolithic(self):
        """RETIRE-with-monolithic: a single full band (g0=0, L=S) == monolithic back."""
        for c in self.configs:
            with self.subTest(geometry=c['name']):
                view = self._rand_view(c)
                full = np.asarray(ConeBeamModel.back_project_one_view_to_band(
                    view, c['idx'], c['svp'], c['pp'], 0, c['S'], coeff_power=1))
                np.testing.assert_allclose(full, self._back_monolithic(c, view),
                                           rtol=self.RTOL, atol=self.ATOL)

    def test_back_band_decomposition_hessian(self):
        """RETIRE-with-monolithic: Hessian path (coeff_power=2) decomposes the same way."""
        for c in self.configs:
            with self.subTest(geometry=c['name']):
                view = self._rand_view(c)
                np.testing.assert_allclose(self._back_banded_concat(c, view, coeff_power=2),
                                           self._back_monolithic(c, view, coeff_power=2),
                                           rtol=self.RTOL, atol=self.ATOL)

    def test_forward_band_decomposition(self):
        """RETIRE-with-monolithic: monolithic forward == sum of banded forward."""
        for c in self.configs:
            with self.subTest(geometry=c['name']):
                voxels = self._rand_voxels(c)
                np.testing.assert_allclose(self._forward_banded_sum(c, voxels),
                                           self._forward_monolithic(c, voxels),
                                           rtol=self.RTOL, atol=self.ATOL)

    def test_forward_full_band_equals_monolithic(self):
        """RETIRE-with-monolithic: a single full band (g0=0, L=S) == monolithic forward."""
        for c in self.configs:
            with self.subTest(geometry=c['name']):
                voxels = self._rand_voxels(c)
                full = np.asarray(ConeBeamModel.forward_project_band_to_one_view(
                    voxels, c['idx'], c['svp'], c['pp'], 0))
                np.testing.assert_allclose(full, self._forward_monolithic(c, voxels),
                                           rtol=self.RTOL, atol=self.ATOL)

    # ── adjoint at an arbitrary band (PERMANENT -- no monolithic reference) ──────
    def test_band_adjoint(self):
        """<A_band x, y> == <x, A_bandᵀ y> for an arbitrary interior band (forward_band
        and back_band are exact adjoints; coeff_power=1)."""
        for c in self.configs:
            with self.subTest(geometry=c['name']):
                g0, g1 = band_bounds(c['S'])[1]        # middle band (g0 > 0, interior)
                L = g1 - g0
                x_band = jnp.asarray(self.rng.standard_normal((c['num_pixels'], L), np.float32))
                y_view = self._rand_view(c)
                ax = np.asarray(ConeBeamModel.forward_project_band_to_one_view(
                    x_band, c['idx'], c['svp'], c['pp'], g0))
                aty = np.asarray(ConeBeamModel.back_project_one_view_to_band(
                    y_view, c['idx'], c['svp'], c['pp'], g0, L, coeff_power=1))
                lhs = float(np.sum(ax * np.asarray(y_view)))
                rhs = float(np.sum(np.asarray(x_band) * aty))
                rel = abs(lhs - rhs) / (abs(lhs) + abs(rhs) + 1e-30)
                self.assertLess(rel, 1e-4,
                                f"[{c['name']}] band adjoint mismatch: <Ax,y>={lhs:.6e} "
                                f"<x,Aᵀy>={rhs:.6e} rel={rel:.2e}")


if __name__ == '__main__':
    unittest.main()
