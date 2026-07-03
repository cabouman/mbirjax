"""
tests/geometries/test_multiaxis_elevation.py
────────────────────────────────────────────
Physics gates specific to the MultiAxisParallelModel elevation geometry -- the pieces the
existing adjoint / reduce-to-parallel tests (test_projectors.py) and the banded-projector test
(test_multiaxis_banded.py) do NOT cover:

  * MASS CONSERVATION across elevation.  A parallel-beam projection is a line integral, so the
    total projected mass of a fixed voxel must equal voxel_volume / detector_pixel_area,
    INDEPENDENT of azimuth and elevation.  Gates the elevation pathlength correction (the old
    projector let mass fall off as cos(el)) and the footprint-floor removal (mass stays finite
    and conserved up to grazing elevation, since the max-of-edges footprint never collapses).

  * det_channel_offset SIGN.  At zero elevation a nonzero det_channel_offset must shift the
    detector the same way as ParallelBeamModel (the +offset convention).  Gates the sign fix
    (the old code used -det_channel_offset).

  * FBP PERMUTATION INVARIANCE.  The views are SIMULTANEOUS measurements with no acquisition
    trajectory, so the arbitrary view order must not affect the direct reconstruction.  Gates the
    order-invariant filter (the old jnp.gradient directional ramp failed this).

Small, CPU-runnable shapes.  Computed floats -> tolerance gates (project rule: never exact
equality for computed floats); GPU scatter-add noise is ~1e-6 rel.
"""
import unittest

import numpy as np
import jax.numpy as jnp

import mbirjax as mj


class TestMultiAxisElevationPhysics(unittest.TestCase):

    def test_mass_conservation_across_elevation(self):
        """Single-voxel projected mass == dv*dvr*dvs/(ddr*ddc), finite and angle-independent."""
        N = 25
        for (vra, vsa, ddr, ddc) in [(1.0, 1.0, 1.0, 1.0), (1.5, 0.7, 1.2, 0.9)]:
            dv = 1.0
            expected = dv * (vra * dv) * (vsa * dv) / (ddr * ddc)
            for (azd, eld) in [(0, 0), (37, 25), (63, 44), (20, 70), (10, 89)]:
                with self.subTest(row_aspect=vra, slice_aspect=vsa, az=azd, el=eld):
                    ang = jnp.array([[np.deg2rad(azd), np.deg2rad(eld)]])
                    m = mj.MultiAxisParallelModel((1, 81, 81), ang)
                    m.set_params(verbose=0, recon_shape=(N, N, N), delta_voxel=dv,
                                 delta_det_row=ddr, delta_det_channel=ddc,
                                 voxel_row_aspect=vra, voxel_slice_aspect=vsa)
                    vol = np.zeros((N, N, N), np.float32)
                    vol[N // 2, N // 2, N // 2] = 1.0
                    total = float(jnp.sum(m.forward_project(jnp.asarray(vol))))
                    self.assertTrue(np.isfinite(total),
                                    f"non-finite projected mass at az={azd}, el={eld}")
                    rel = abs(total - expected) / expected
                    self.assertLess(rel, 2e-3,
                                    f"mass not conserved: got {total:.5f}, expected {expected:.5f} "
                                    f"(rel {rel:.2e}) at az={azd}, el={eld}")

    def test_det_channel_offset_matches_parallel(self):
        """At el=0, a nonzero det_channel_offset must match ParallelBeamModel (same +sign)."""
        nv, ndr, ndc, N = 12, 24, 41, 24
        az = jnp.linspace(0.0, jnp.pi, nv, endpoint=False) + 1e-4
        ma_angles = jnp.stack([az, jnp.zeros(nv)], axis=1)
        phantom = jnp.asarray(np.random.default_rng(1).random((N, N, N), dtype=np.float32))
        for off in (0.0, 3.5, -2.0):
            with self.subTest(det_channel_offset=off):
                ma = mj.MultiAxisParallelModel((nv, ndr, ndc), ma_angles)
                ma.set_params(verbose=0, recon_shape=(N, N, N), delta_voxel=1.0,
                              delta_det_channel=1.0, delta_det_row=1.0, det_channel_offset=off)
                pb = mj.ParallelBeamModel((nv, ndr, ndc), az)
                pb.set_params(verbose=0, recon_shape=(N, N, N), delta_voxel=1.0,
                              delta_det_channel=1.0, det_channel_offset=off)
                s_ma = np.asarray(ma.forward_project(phantom))
                s_pb = np.asarray(pb.forward_project(phantom))
                rel = float(np.max(np.abs(s_ma - s_pb)) / np.max(np.abs(s_pb)))
                self.assertLess(rel, 1e-6,
                                f"multiaxis(el=0) forward != parallel with det_channel_offset={off}: "
                                f"rel-max {rel:.3e}")

    def test_fbp_permutation_invariance(self):
        """Permuting the (arbitrary) view order must leave the FBP reconstruction unchanged."""
        nv, ndr, ndc, N = 10, 24, 33, 24
        rng = np.random.default_rng(2)
        az = np.linspace(0.0, np.pi, nv, endpoint=False).astype(np.float32)
        el = (np.deg2rad(15.0) * np.cos(2.0 * az)).astype(np.float32)
        angles = jnp.asarray(np.stack([az, el], axis=1))
        sino = jnp.asarray(rng.random((nv, ndr, ndc), dtype=np.float32))

        def recon(angs, s):
            m = mj.MultiAxisParallelModel((nv, ndr, ndc), angs)
            m.set_params(verbose=0, recon_shape=(N, N, N), delta_voxel=1.0,
                         delta_det_row=1.0, delta_det_channel=1.0)
            return np.asarray(m.fbp_recon(s))

        rec = recon(angles, sino)
        perm = rng.permutation(nv)
        rec_perm = recon(angles[perm], sino[perm])
        rel = float(np.max(np.abs(rec - rec_perm)) / (np.max(np.abs(rec)) + 1e-12))
        self.assertLess(rel, 1e-5, f"FBP is not permutation-invariant: rel-max {rel:.3e}")


if __name__ == "__main__":
    unittest.main()
