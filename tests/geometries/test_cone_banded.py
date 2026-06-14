"""
tests/geometries/test_cone_banded.py
─────────────────────────────────────
Unit tests for the cone banded BACK projector kernel
``ConeBeamModel.back_project_one_view_to_band`` and the production back projector
``ConeBeamModel.back_project_one_view_to_pixel_batch`` that is now built on it.

These are the strong per-geometry correctness gates for the banded interface.  They
check:

  * band-decomposition: the monolithic per-view BACK projector equals the
    concatenation of the banded back projector over a tiling of the slice axis --
    because the bands tile disjointly, so each contribution is counted exactly once;
  * full-band == monolithic: a single band (g0=0, L=S) reproduces the monolithic
    back kernel (the anchor is faithful on full cylinders);
  * the Hessian path (coeff_power=2) decomposes the same way;
  * driver A/B: the production back_project_one_view_to_pixel_batch (horizontal fan
    once + a rolled jax.lax.map over slice bands + reshape/crop) reproduces the
    monolithic back projector across band sizes -- validating the lax.map reassembly,
    not just the band kernel.

Every check runs on a CIRCULAR and a HELICAL cone geometry, so the anchor's
helical-z-shift term is exercised (a helical view has a nonzero per-view z shift).

Computed floats -> tight allclose (project rule: never exact equality for computed
floats).  Band decomposition only reorders summation, so CPU agreement is ~1e-6;
the gates use 1e-4 to stay robust on GPU (projector scatter-add noise ~8e-6 rel).

RETIRE: every test here COMPARES the banded back kernel against the monolithic
back_project_one_view_to_pixel_batch, so they all retire when that monolithic kernel
is deleted (after the back projector is switched to the banded path).  At that point
the wired-path adjoint identity in test_projectors becomes the banded back kernel's
permanent gate -- the banded back is a correct restriction of the monolithic back
(band-decomposition), which test_projectors already gates for adjointness.

(The forward projection does not band -- the multi-device forward gathers the slice
cylinder per pixel-batch and runs the monolithic forward -- so there is no banded
forward kernel to test here.)
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
        # The pre-banded monolithic back projector: horizontal fan once -> monolithic
        # vertical fan (entries_per_cylinder_batch slice chunking).  This is the A/B
        # reference for both the band-decomposition tests and the banded production
        # back_project_one_view_to_pixel_batch.  RETIRE with the monolithic vertical fan.
        det_cyl = ConeBeamModel.back_horizontal_fan_one_view_to_pixel_batch(
            view, c['idx'], c['svp'], c['pp'], coeff_power=coeff_power)
        return np.asarray(ConeBeamModel.back_vertical_fan_one_view_to_pixel_batch(
            det_cyl, c['idx'], c['svp'], c['pp'], coeff_power=coeff_power))

    def _back_banded_concat(self, c, view, coeff_power=1):
        bands = [np.asarray(ConeBeamModel.back_project_one_view_to_band(
            view, c['idx'], c['svp'], c['pp'], g0, g1 - g0, coeff_power=coeff_power))
            for (g0, g1) in band_bounds(c['S'])]
        return np.concatenate(bands, axis=1)   # (num_pixels, S)

    def _rand_view(self, c):
        return jnp.asarray(self.rng.standard_normal((c['num_det_rows'], c['num_det_channels']), np.float32))

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

    # ── driver A/B: the production banded back projector vs the monolithic ───────
    def test_back_driver_matches_monolithic(self):
        """RETIRE-with-monolithic: the production back_project_one_view_to_pixel_batch
        (horizontal fan once + rolled lax.map over slice bands + reshape/crop) reproduces
        the monolithic back projector.  Unlike the band-decomposition test (which uses an
        explicit np.concatenate), this exercises the production lax.map REASSEMBLY -- so a
        transpose/reshape/crop bug would surface here.  Several band sizes are run: one that
        does NOT divide the slice count (the padded-last-band crop), a small one (several
        bands), and one >= S (a single band, clamped).  Covers coeff_power 1 and 2."""
        for c in self.configs:
            with self.subTest(geometry=c['name']):
                view = self._rand_view(c)
                S = c['S']
                # Non-divisor (remainder -> crop), small (several bands), and a single band.
                band_sizes = sorted({max(1, S // 3), max(2, S // 2 + 1), S})
                for coeff_power in (1, 2):
                    ref = self._back_monolithic(c, view, coeff_power=coeff_power)
                    for bs in band_sizes:
                        prod = np.asarray(ConeBeamModel.back_project_one_view_to_pixel_batch(
                            view, c['idx'], c['svp'], c['pp'],
                            coeff_power=coeff_power, slice_band_size=bs))
                        np.testing.assert_allclose(
                            prod, ref, rtol=self.RTOL, atol=self.ATOL,
                            err_msg=f"[{c['name']}] coeff_power={coeff_power} band_size={bs}")


if __name__ == '__main__':
    unittest.main()
