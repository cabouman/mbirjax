import tempfile
import unittest
import numpy as np
import jax
import jax.numpy as jnp
import mbirjax as mj


# Geometries gated by the adjoint/Hessian projector tests.  Multiaxis is gated HERE but
# intentionally kept OUT of mj._utils._geometry_types_for_tests, so it is NOT pulled into the
# VCD-convergence net (test_vcd) or the recon-NRMSE net (test_fbp_fdk): multiaxis is a
# limited-angle geometry whose direct recon is only an MBIR initializer, and (as with
# translation) its iterated VCD loop is geometry-independent and already gated by parallel +
# cone.  Only the anisotropic variant is listed: it subsumes the isotropic case (aspects ==
# 1), so a separate isotropic entry would be redundant.  (Anisotropic voxels are not yet wired
# into the multiaxis kernels, so this currently behaves isotropically; once they are wired,
# this same entry exercises the anisotropic path with no test change.)
_PROJECTOR_GEOMETRY_TYPES = list(mj._utils._geometry_types_for_tests) + ['anisotropic_multiaxis']


class TestProjectors(unittest.TestCase):
    """
    Test the adjoint property of the forward and back projectors, both the full versions and the sparse voxel version.
    This means if x is an image, and y is a sinogram, then <y, Ax> = <Aty, x>.
    The code below verifies this for the full forward and back projectors and the versions that specify a
    subset of voxels in x.
    This code also verifies that first applying the full back projector and selecting the voxels to get (Aty)[ss]
    is the same as using the subset back projector with the specified set of voxels.
    """

    def setUp(self):
        """Set up before each test method."""
        # Choose the geometry type (adds multiaxis to the shared list; see
        # _PROJECTOR_GEOMETRY_TYPES at module top for why it is not in the shared list).
        self.geometry_types = _PROJECTOR_GEOMETRY_TYPES

        # Set parameters
        self.num_views = 64
        self.num_det_rows = 40
        self.num_det_channels = 128
        self.sharpness = 0.0
        
        # These can be adjusted to scale voxel aspect ratios for the anisotropic cases
        self.voxel_row_aspect = 1.9
        self.voxel_slice_aspect = 2.9 # Only for cone beam and translation

        # These can be adjusted to describe the geometry in the cone beam case.
        # np.Inf is an allowable value, in which case this is essentially parallel beam
        self.source_detector_dist = 4 * self.num_det_channels
        self.source_iso_dist = self.source_detector_dist / 2
        
        # These can be adjusted to describe the geometry in the helical cone beam case.
        self.helical_pitch = 0.5
        self.helical_z_range = 80.0
        self.helical_z_center = 40.0

        # Initialize sinogram
        self.sinogram_shape = (self.num_views, self.num_det_rows, self.num_det_channels)
        self.angles = None
        self.helical_z_shifts = None
        self.translation_vector = None

    def tearDown(self):
        """Clean up after each test method."""
        pass

    def set_view_params(self, geometry_type):
        if geometry_type == 'cone':
            detector_cone_angle = 2 * np.arctan2(self.num_det_channels / 2, self.source_detector_dist)
        elif geometry_type == 'anisotropic_cone':
            detector_cone_angle = 2 * np.arctan2(self.num_det_channels / 2, self.source_detector_dist)
        elif geometry_type == 'helical_cone':
            detector_cone_angle = 2 * np.arctan2(self.num_det_channels / 2, self.source_detector_dist)
            
            magnification = self.source_detector_dist / self.source_iso_dist
            det_height_iso = self.num_det_rows / magnification
            z_per_rot = self.helical_pitch * det_height_iso
            dz_per_view = z_per_rot / self.num_views
            view_offsets = jnp.arange(self.num_views) - (self.num_views - 1) / 2
            self.helical_z_shifts = self.helical_z_center + dz_per_view * view_offsets
        else:
            detector_cone_angle = 0
        start_angle = -(np.pi + detector_cone_angle) * (1 / 2)
        end_angle = (np.pi + detector_cone_angle) * (1 / 2)
        # Add a small offset to the angles to avoid the jax rounding bug
        # See plans/experiments/bugs_and_artifacts/jax rounding bug/jax_rounding_bug.md
        self.angles = jnp.linspace(start_angle, end_angle, self.num_views, endpoint=False) + 1e-4

        # Multiaxis takes (num_views, 2) = [azimuth, elevation]: reuse the azimuth sweep above
        # and add a deterministic spread of elevations (~ +/-17 deg) so the vertical (tilt) fan
        # is exercised, not just the azimuth-only (parallel-equivalent) limit.
        self.multiaxis_angles = jnp.stack(
            [self.angles, jnp.linspace(-0.30, 0.30, self.num_views)], axis=1)

    def set_translation_vectors(self, geometry_type):
        if geometry_type in ('translation', 'anisotropic_translation'):
            self.translation_vectors = np.zeros((self.num_views, 3))
            self.translation_vectors[:, 0] = np.random.uniform(-10, 10, self.num_views)
            self.translation_vectors[:, 1] = 0.0
            self.translation_vectors[:, 2] = np.random.uniform(-10, 10, self.num_views)
            self.translation_vectors = jnp.array(self.translation_vectors)
        else:
            self.translation_vectors = None

    def get_model(self, geometry_type):
        if geometry_type == 'cone':
            ct_model = mj.ConeBeamModel(self.sinogram_shape, self.angles,
                                             source_detector_dist=self.source_detector_dist,
                                             source_iso_dist=self.source_iso_dist)
        elif geometry_type == 'anisotropic_cone':
            ct_model = mj.ConeBeamModel(self.sinogram_shape, self.angles,
                                             source_detector_dist=self.source_detector_dist,
                                             source_iso_dist=self.source_iso_dist)
            ct_model.set_params(voxel_row_aspect=self.voxel_row_aspect)
            ct_model.set_params(voxel_slice_aspect=self.voxel_slice_aspect)
            ct_model.auto_set_recon_geometry()
        elif geometry_type == 'helical_cone':
            ct_model = mj.ConeBeamModel(self.sinogram_shape, self.angles, helical_z_shifts=self.helical_z_shifts,
                                             source_detector_dist=self.source_detector_dist,
                                             source_iso_dist=self.source_iso_dist)
        elif geometry_type == 'parallel':
            ct_model = mj.ParallelBeamModel(self.sinogram_shape, self.angles)
        elif geometry_type == 'anisotropic_parallel':
            ct_model = mj.ParallelBeamModel(self.sinogram_shape, self.angles)
            ct_model.set_params(voxel_row_aspect=self.voxel_row_aspect)
            ct_model.auto_set_recon_geometry()
        elif geometry_type == 'translation':
            ct_model = mj.TranslationModel(self.sinogram_shape, self.translation_vectors,
                                                source_detector_dist=self.source_detector_dist,
                                                source_iso_dist=self.source_iso_dist)
        elif geometry_type == 'anisotropic_translation':
            ct_model = mj.TranslationModel(self.sinogram_shape, self.translation_vectors,
                                           source_detector_dist=self.source_detector_dist,
                                           source_iso_dist=self.source_iso_dist)
            ct_model.set_params(voxel_row_aspect=self.voxel_row_aspect)
            ct_model.set_params(voxel_slice_aspect=self.voxel_slice_aspect)
            ct_model.auto_set_recon_geometry()
        elif geometry_type == 'anisotropic_multiaxis':
            ct_model = mj.MultiAxisParallelModel(self.sinogram_shape, self.multiaxis_angles)
            ct_model.set_params(voxel_row_aspect=self.voxel_row_aspect)
            ct_model.set_params(voxel_slice_aspect=self.voxel_slice_aspect)
            ct_model.auto_set_recon_geometry()
        else:
            raise ValueError('Invalid geometry type.  Expected cone or parallel, got {}'.format(geometry_type))

        return ct_model

    # One test method per (operation, geometry) pair is generated at the bottom of this
    # file (test_adjoint_parallel, test_hessian_cone, ...) instead of three tests looping
    # subTests: pytest can then report, select (-k cone), and distribute (pytest-xdist)
    # them individually.

    def verify_adjoint(self, geometry_type):
        """
        Verify the adjoint property of the projectors:
        Choose a random phantom, x, and a random sinogram, y, and verify that <y, Ax> = <Aty, x>.
        """
        self.set_view_params(geometry_type)
        self.set_translation_vectors(geometry_type)
        ct_model = self.get_model(geometry_type)

        # Initialize a random key
        seed_value = np.random.randint(1000000)
        key = jax.random.PRNGKey(seed_value)

        # Generate phantom
        recon_shape = ct_model.get_params('recon_shape')
        num_recon_rows, num_recon_cols, num_recon_slices = recon_shape[:3]
        phantom_shape = recon_shape
        embed_slice_start = 0
        embed_slice_stop = recon_shape[2]
        if geometry_type == 'helical_cone':
            embed_slice_start, embed_slice_stop = mj.get_helical_half_rotation_slice_range(
                ct_model,
                self.helical_pitch,
                self.helical_z_shifts
            )
            phantom_shape = (
                recon_shape[0],
                recon_shape[1],
                embed_slice_stop - embed_slice_start,
            )
        phantom_core = mj.gen_cube_phantom(phantom_shape)
        if geometry_type == 'helical_cone':
            phantom = jnp.zeros(recon_shape)
            phantom = phantom.at[:, :, embed_slice_start:embed_slice_stop].set(phantom_core)
        else:
            phantom = phantom_core

        # Generate indices of pixels
        num_subsets = 1
        use_ror_mask = ct_model.get_params('use_ror_mask')
        full_indices = mj.gen_pixel_partition(recon_shape, num_subsets=num_subsets, use_ror_mask=use_ror_mask)

        # Generate sinogram data
        voxel_values = phantom.reshape((-1,) + recon_shape[2:])[full_indices]

        # Compute forward projection
        sinogram = ct_model.sparse_forward_project(voxel_values[0], full_indices[0])

        # Get the vector of indices
        indices = jnp.arange(num_recon_rows * num_recon_cols)
        num_trials = 3
        indices = jnp.mod(np.arange(num_trials, dtype=int).reshape((-1, 1)) + indices.reshape((1, -1)),
                          num_recon_rows * num_recon_cols)

        # Convert to jax arrays
        sinogram = jnp.array(sinogram)
        indices = jnp.array(indices)

        # Run once to finish compiling and get backprojection shape.
        # On a sharded model whose slice count does not divide the device count, the slice
        # axis is zero-padded to the device form and sparse_back_project (a sharded-contract
        # internal method) returns that device form; crop the inert zero padding so the
        # adjoint identity is checked on the problem's real slices.  A no-op without padding.
        bp = ct_model.sparse_back_project(sinogram, indices[0])[:, :num_recon_slices]

        # ##########################
        # Test the adjoint property
        # Get a random 3D phantom to test the adjoint property
        key, subkey = jax.random.split(key)
        x = jax.random.uniform(subkey, shape=bp.shape)
        key, subkey = jax.random.split(key)
        y = jax.random.uniform(subkey, shape=sinogram.shape)
        # When the view count does not divide the device count, ``sinogram`` is the device form
        # with a zero-padded view tail; the back projector relies on those padded views being
        # zero (production zero-fills them at entry), so back-projecting a nonzero random tail
        # would contaminate A^T y at the clamped padding angle.  Zero the padded views to respect
        # that contract.  A no-op without view padding (num_real_views == sinogram.shape[0]).
        num_real_views = ct_model.get_params('sinogram_shape')[0]
        y = y.at[num_real_views:].set(0.0)

        # Do a forward projection, then a backprojection (crop the device-form padded
        # slices, as above, so <Aty, x> is taken on the real slices).
        voxel_values = x.reshape((-1, num_recon_slices))[indices[0]]
        Ax = ct_model.sparse_forward_project(voxel_values, indices[0])
        Aty = ct_model.sparse_back_project(y, indices[0])[:, :num_recon_slices]

        # Calculate <Aty, x> and <y, Ax>
        Aty_x = jnp.sum(Aty * x)
        y_Ax = jnp.sum(y * Ax)

        # Determine if property holds
        adjoint_test_result = np.allclose(Aty_x, y_Ax, rtol=1e-4)
        diff = Aty_x - y_Ax
        abs_diff = jnp.abs(diff)
        rel_diff = abs_diff / jnp.maximum(jnp.abs(Aty_x), jnp.abs(y_Ax))
        print("Aty_x =", Aty_x)
        print("y_Ax =", y_Ax)
        print("absolute difference =", diff)
        print("relative difference =", rel_diff)
        self.assertTrue(adjoint_test_result)

    def test_multiaxis_forward_reduces_to_parallel(self):
        """At zero elevation the multiaxis forward projector must reduce EXACTLY to
        ParallelBeamModel (isotropic) and to anisotropic_parallel (with a row aspect).

        This is a strong ABSOLUTE-magnitude anchor that the adjoint identity cannot provide:
        the adjoint passes for any consistent-but-mis-scaled forward/back pair, whereas
        matching a validated reference model pins the scaling.  voxel_slice_aspect is left at 1
        here because once slices spread across detector rows (any aspect or elevation) parallel
        beam is no longer a valid reference (it is slice-independent).
        """
        nv, ndr, ndc = 16, 32, 32
        az = jnp.linspace(0.0, jnp.pi, nv, endpoint=False) + 1e-4
        ma_angles = jnp.stack([az, jnp.zeros(nv)], axis=1)     # zero elevation
        rng = np.random.default_rng(0)
        for row_aspect in (1.0, 1.9):
            with self.subTest(voxel_row_aspect=row_aspect):
                ma = mj.MultiAxisParallelModel((nv, ndr, ndc), ma_angles)
                ma.set_params(verbose=0, voxel_row_aspect=row_aspect)
                ma.auto_set_recon_geometry()
                recon_shape = tuple(int(v) for v in ma.get_params('recon_shape'))
                # ParallelBeam reference with the SAME recon geometry + row aspect.
                pb = mj.ParallelBeamModel((nv, ndr, ndc), az)
                pb.set_params(verbose=0, voxel_row_aspect=row_aspect, recon_shape=recon_shape,
                              delta_voxel=float(ma.get_params('delta_voxel')))
                phantom = jnp.asarray(rng.random(recon_shape, dtype=np.float32))
                s_ma = np.asarray(ma.forward_project(phantom))
                s_pb = np.asarray(pb.forward_project(phantom))
                # Scale-invariant gate (project rule: never exact equality for computed floats);
                # the value is ~0 on CPU, the 1e-6 margin absorbs any GPU reduction reordering.
                rel = float(np.max(np.abs(s_ma - s_pb)) / np.max(np.abs(s_pb)))
                self.assertLess(rel, 1e-6,
                                f"multiaxis(el=0) forward != parallel: rel-max {rel:.3e} "
                                f"(voxel_row_aspect={row_aspect})")

    def verify_hessian(self, geometry_type):
        """
        Verify the hessian property of the back projector:
        Choose a random pixel, set it to epsilon, apply A^T A and compare to the value from compute_hessian_diagaonal.
        """
        self.set_view_params(geometry_type)
        self.set_translation_vectors(geometry_type)
        ct_model = self.get_model(geometry_type)

        # ## Test the hessian against a finite difference approximation ## #
        hessian = ct_model.compute_hessian_diagonal()

        # Initialize a random key
        seed_value = np.random.randint(1000000)
        key = jax.random.PRNGKey(seed_value)

        recon_shape = ct_model.get_params('recon_shape')
        num_recon_rows, num_recon_cols, num_recon_slices = recon_shape[:3]
        x = jnp.zeros(recon_shape)
        key, subkey = jax.random.split(key)
        i = jax.random.randint(subkey, shape=(), minval=0, maxval=num_recon_rows)
        key, subkey = jax.random.split(key)
        j = jax.random.randint(subkey, shape=(), minval=0, maxval=num_recon_cols)
        key, subkey = jax.random.split(key)
        k = jax.random.randint(subkey, shape=(), minval=0, maxval=num_recon_slices)

        # Get the vector of indices
        indices = jnp.arange(num_recon_rows * num_recon_cols)
        # num_trials = 3
        # indices = jnp.mod(np.arange(num_trials, dtype=int).reshape((-1, 1)) + indices.reshape((1, -1)),
        #                   num_recon_rows * num_recon_cols)

        eps = 0.01
        x = x.at[i, j, k].set(eps)
        voxel_values = x.reshape((-1, num_recon_slices))[indices]
        Ax = ct_model.sparse_forward_project(voxel_values, indices)
        # Crop the device-form padded slices (sharded, non-dividing slice count) before
        # reshaping to the real recon shape; the padded slices are inert zeros.  No-op
        # without padding.  The tested pixel (i, j, k) has k < num_recon_slices (real).
        AtAx = ct_model.sparse_back_project(Ax, indices)[:, :num_recon_slices].reshape(x.shape)
        finite_diff_hessian = AtAx[i, j, k] / eps

        # Determine if property holds
        hessian_test_result = jnp.allclose(hessian.reshape(x.shape)[i, j, k], finite_diff_hessian)
        self.assertTrue(hessian_test_result)


def _add_per_geometry_projector_tests():
    """Generate one test_<operation>_<geometry> method per pair (see note in TestProjectors)."""
    operations = ('adjoint', 'hessian')
    for geometry_type in _PROJECTOR_GEOMETRY_TYPES:
        for operation in operations:
            def test(self, geometry_type=geometry_type, operation=operation):
                print('Testing {} with {}'.format(operation, geometry_type))
                getattr(self, 'verify_' + operation)(geometry_type)
            test.__name__ = 'test_{}_{}'.format(operation, geometry_type)
            test.__doc__ = '{} check for the {} geometry.'.format(operation, geometry_type)
            setattr(TestProjectors, test.__name__, test)


_add_per_geometry_projector_tests()


if __name__ == '__main__':
    unittest.main()