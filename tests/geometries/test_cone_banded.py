"""
tests/geometries/test_cone_banded.py
─────────────────────────────────────
Unit tests for the cone banded BACK projector: the per-band kernel
``ConeBeamModel.back_project_one_view_to_band`` and the production back projector
``ConeBeamModel.back_project_one_view_to_pixel_batch`` that is built on it.

The production back projector hoists the horizontal fan and walks the recon slice axis
in UNIFORM bands with a rolled jax.lax.map (+ reshape/crop).  This test checks that it
reproduces an INDEPENDENT banded assembly -- an explicit np.concatenate of
back_project_one_view_to_band over a NON-uniform tiling (which recomputes the horizontal
fan per band) -- across several band sizes (including one that does not divide the slice
count, exercising the padded-last-band crop) and coeff_power 1 (projection) and 2
(Hessian).  Agreement validates the horizontal-hoist, the lax.map reassembly, and the
banded vertical fan's tiling-invariance.

Both a CIRCULAR and a HELICAL cone geometry are exercised, so the anchor's
helical-z-shift term is covered (a helical view has a nonzero per-view z shift).

The banded vertical fan's own physical correctness (anchor, footprint, helical z shift)
is gated INDEPENDENTLY by the cone adjoint identity in test_projectors and the cone
convergence gate in test_vcd, both of which run the production banded back projector.

Computed floats -> tight allclose (project rule: never exact equality for computed
floats).  Band reassembly only reorders summation, so CPU agreement is ~1e-6; the gates
use 1e-4 to stay robust on GPU (projector scatter-add noise ~8e-6 rel).
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
    def _back_banded_concat(self, c, view, coeff_power=1):
        # Independent reference: explicit np.concatenate of the per-band back projector
        # over a NON-uniform tiling (each call recomputes its own horizontal fan).
        bands = [np.asarray(ConeBeamModel.back_project_one_view_to_band(
            view, c['idx'], c['svp'], c['pp'], g0, g1 - g0, coeff_power=coeff_power))
            for (g0, g1) in band_bounds(c['S'])]
        return np.concatenate(bands, axis=1)   # (num_pixels, S)

    def _rand_view(self, c):
        return jnp.asarray(self.rng.standard_normal((c['num_det_rows'], c['num_det_channels']), np.float32))

    # ── production back projector vs an independent banded assembly ──────────────
    def test_back_production_matches_band_concat(self):
        """The production back_project_one_view_to_pixel_batch (horizontal fan once + a
        rolled jax.lax.map over uniform slice bands + reshape/crop) reproduces an explicit
        np.concatenate of back_project_one_view_to_band over a non-uniform tiling.  Run
        across band sizes -- one that does NOT divide the slice count (padded-last-band
        crop), a small one (several bands), and one >= S (single band, clamped) -- and
        coeff_power 1 and 2.  This validates the horizontal-hoist, the lax.map reassembly,
        and tiling-invariance; the banded vertical fan's physical correctness is gated by
        test_projectors (adjoint identity) and test_vcd (convergence)."""
        for c in self.configs:
            with self.subTest(geometry=c['name']):
                view = self._rand_view(c)
                S = c['S']
                # Non-divisor (remainder -> crop), small (several bands), and a single band.
                band_sizes = sorted({max(1, S // 3), max(2, S // 2 + 1), S})
                for coeff_power in (1, 2):
                    ref = self._back_banded_concat(c, view, coeff_power=coeff_power)
                    for bs in band_sizes:
                        prod = np.asarray(ConeBeamModel.back_project_one_view_to_pixel_batch(
                            view, c['idx'], c['svp'], c['pp'],
                            coeff_power=coeff_power, slice_band_size=bs))
                        np.testing.assert_allclose(
                            prod, ref, rtol=self.RTOL, atol=self.ATOL,
                            err_msg=f"[{c['name']}] coeff_power={coeff_power} band_size={bs}")


if __name__ == '__main__':
    unittest.main()
