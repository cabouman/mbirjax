import unittest
import warnings

import numpy as np
import jax.numpy as jnp
import mbirjax as mj

TRUNCATION_MATCH = 'Lateral FoV truncation'


def _sino(shape, edge_to_edge):
    """A synthetic sinogram with a clean background/object split: an object block over the
    central rows, spanning ALL channels (edge_to_edge=True, the truncated signature) or only
    the central half (False, contained)."""
    sino = np.zeros(shape, dtype=np.float32)
    ch = slice(None) if edge_to_edge else slice(shape[2] // 4, -shape[2] // 4)
    sino[:, shape[1] // 4: -shape[1] // 4, ch] = 1.0
    return sino


def _caught_truncation(model, sino):
    """Run auto_set_regularization_params and return the caught truncation warnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model.auto_set_regularization_params(jnp.asarray(sino))
    return [c for c in caught if TRUNCATION_MATCH in str(c.message)]


class TestLateralTruncationWarning(unittest.TestCase):
    """The lateral detect-and-warn hook (_check_lateral_truncation): fires on edge-touching
    support for the rotating geometries, stays silent for contained scans, the all-ones
    indicator fallback, verbose=0, and the geometries that override it (translation,
    denoiser)."""

    V, N, C = 8, 16, 24

    def _cone(self):
        sdd = 4.0 * self.C
        return mj.ConeBeamModel((self.V, self.N, self.C),
                                jnp.linspace(0, jnp.pi, self.V, endpoint=False),
                                source_detector_dist=sdd, source_iso_dist=sdd / 2.0)

    def test_cone_warns_on_edge_support(self):
        model = self._cone()
        caught = _caught_truncation(model, _sino((self.V, self.N, self.C), edge_to_edge=True))
        self.assertEqual(len(caught), 1)
        # The message names the remedy.
        self.assertIn('scale_recon_shape', str(caught[0].message))

    def test_cone_silent_when_contained(self):
        model = self._cone()
        caught = _caught_truncation(model, _sino((self.V, self.N, self.C), edge_to_edge=False))
        self.assertEqual(caught, [])

    def test_silent_on_all_ones_indicator_fallback(self):
        # An all-zero sinogram takes the no-positive-values fallback (all-ones indicator,
        # its own warning): the truncation check must skip rather than fire spuriously.
        model = self._cone()
        caught = _caught_truncation(model, np.zeros((self.V, self.N, self.C), dtype=np.float32))
        self.assertEqual(caught, [])

    def test_silent_at_verbose_zero(self):
        model = self._cone()
        model.set_params(verbose=0)
        caught = _caught_truncation(model, _sino((self.V, self.N, self.C), edge_to_edge=True))
        self.assertEqual(caught, [])

    def test_parallel_warns_on_edge_support(self):
        # The hook lives in TomographyModel, so every rotating geometry inherits it.
        model = mj.ParallelBeamModel((self.V, self.N, self.C),
                                     jnp.linspace(0, jnp.pi, self.V, endpoint=False))
        caught = _caught_truncation(model, _sino((self.V, self.N, self.C), edge_to_edge=True))
        self.assertEqual(len(caught), 1)

    def test_translation_override_is_silent(self):
        # Translation tomography: a plate spanning the FoV is the NORMAL condition.
        tv = np.zeros((self.V, 3))
        tv[:, 0] = np.linspace(-8.0, 8.0, self.V)
        tv[:, 2] = np.linspace(-1.65, 1.65, self.V)
        sdd = 4.0 * self.C
        model = mj.TranslationModel((self.V, self.N, self.C), jnp.asarray(tv),
                                    source_detector_dist=sdd, source_iso_dist=sdd / 2.0)
        caught = _caught_truncation(model, _sino((self.V, self.N, self.C), edge_to_edge=True))
        self.assertEqual(caught, [])

    def test_denoiser_override_is_silent(self):
        # The denoiser's 'sinogram' is an image; content at the frame edge is normal.  Its
        # denoise() flow sets sigma_noise before auto_set_regularization_params (and even runs
        # it at verbose=0); mirror the sigma_noise setup but keep verbose=1 so this test
        # exercises the override rather than the verbose gate.
        denoiser = mj.QGGMRFDenoiser((self.V, self.N, self.C))
        denoiser.set_params(sigma_noise=0.05)
        image = _sino((self.V, self.N, self.C), edge_to_edge=True)
        rng = np.random.default_rng(0)
        image = image + 0.05 * rng.standard_normal(image.shape).astype(np.float32)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            denoiser.auto_set_regularization_params(jnp.asarray(image))
        self.assertEqual([c for c in caught if TRUNCATION_MATCH in str(c.message)], [])


if __name__ == '__main__':
    unittest.main()
