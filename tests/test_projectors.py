import tempfile
import unittest
import numpy as np
import jax
import jax.numpy as jnp
import mbirjax


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
        # Choose the geometry type
        self.geometry_types = mbirjax._utils._geometry_types_for_tests

        # Set parameters
        self.num_views = 64
        self.num_det_rows = 40
        self.num_det_channels = 128
        self.sharpness = 0.0

        # These can be adjusted to describe the geometry in the cone beam case.
        # np.Inf is an allowable value, in which case this is essentially parallel beam
        self.source_detector_dist = 4 * self.num_det_channels
        self.source_iso_dist = self.source_detector_dist

        # Initialize sinogram
        self.sinogram_shape = (self.num_views, self.num_det_rows, self.num_det_channels)
        self.angles = None
        self.translation_vector = None

    def tearDown(self):
        """Clean up after each test method."""
        pass

    def set_angles(self, geometry_type):
        if geometry_type == 'cone':
            detector_cone_angle = 2 * np.arctan2(self.num_det_channels / 2, self.source_detector_dist)
        else:
            detector_cone_angle = 0
        start_angle = -(np.pi + detector_cone_angle) * (1 / 2)
        end_angle = (np.pi + detector_cone_angle) * (1 / 2)
        self.angles = jnp.linspace(start_angle, end_angle, self.num_views, endpoint=False)

    def set_translation_vectors(self, geometry_type):
        if geometry_type == 'translation':
            self.translation_vectors = np.zeros((self.num_views, 3))
            self.translation_vectors[:, 0] = np.random.uniform(-10, 10, self.num_views)
            self.translation_vectors[:, 1] = 0.0
            self.translation_vectors[:, 2] = np.random.uniform(-10, 10, self.num_views)
            self.translation_vectors = jnp.array(self.translation_vectors)
        else:
            self.translation_vectors = None

    def get_model(self, geometry_type):
        if geometry_type == 'cone':
            ct_model = mbirjax.ConeBeamModel(self.sinogram_shape, self.angles,
                                             source_detector_dist=self.source_detector_dist,
                                             source_iso_dist=self.source_iso_dist)
        elif geometry_type == 'parallel':
            ct_model = mbirjax.ParallelBeamModel(self.sinogram_shape, self.angles)
        elif geometry_type == 'translation':
            ct_model = mbirjax.TranslationModel(self.sinogram_shape, self.translation_vectors,
                                                source_detector_dist=self.source_detector_dist,
                                                source_iso_dist=self.source_iso_dist)
        else:
            raise ValueError('Invalid geometry type.  Expected cone or parallel, got {}'.format(geometry_type))

        return ct_model

    def test_all_adjoints(self):
        for geometry_type in self.geometry_types:
            with self.subTest(geometry_type=geometry_type):
                print("Testing adjoint with", geometry_type)
                self.verify_adjoint(geometry_type)

    def test_all_hessians(self):
        for geometry_type in self.geometry_types:
            with self.subTest(geometry_type=geometry_type):
                print("Testing Hessian with", geometry_type)
                self.verify_hessian(geometry_type)

    def test_save_load(self):
        for geometry_type in self.geometry_types:
            with self.subTest(geometry_type=geometry_type):
                print("Testing save/load with", geometry_type)
                self.verify_save_load(geometry_type)

    def verify_save_load(self, geometry_type):
        """
        Verify the adjoint property of the projectors:
        Choose a random phantom, x, and a random sinogram, y, and verify that <y, Ax> = <Aty, x>.
        """
        self.set_angles(geometry_type)
        self.set_translation_vectors(geometry_type)
        ct_model = self.get_model(geometry_type)

        # Generate phantom
        recon_shape = ct_model.get_params('recon_shape')
        num_recon_rows, num_recon_cols, num_recon_slices = recon_shape[:3]

        # Get the vector of indices
        indices = jnp.arange(num_recon_rows * num_recon_cols)

        # ##########################
        # Do a forward and back projection from a single pixel
        i, j = num_recon_rows // 4, num_recon_cols // 3
        x = jnp.zeros(recon_shape)
        x = x.at[i, j, :].set(1)
        voxel_values = x.reshape((-1, num_recon_slices))[indices]

        Ax = ct_model.sparse_forward_project(voxel_values, indices)
        Aty = ct_model.sparse_back_project(Ax, indices)
        Aty = ct_model.reshape_recon(Aty)

        # Save the model
        with tempfile.NamedTemporaryFile('w', suffix='.yaml') as file:
            filename = file.name
            ct_model.to_file(filename)

            # Load the model
            new_model = self.get_model(geometry_type)
            new_model = new_model.from_file(filename)

        # Compare parameters
        same_params = mbirjax.ParameterHandler.compare_parameter_handlers(ct_model, new_model)
        assert same_params

        # Do a forward and back projection with loaded model
        Ax_new = new_model.sparse_forward_project(voxel_values, indices)
        Aty_new = new_model.sparse_back_project(Ax_new, indices)
        Aty_new = new_model.reshape_recon(Aty_new)

        # Compare to original
        assert(np.allclose(Aty, Aty_new, atol=1e-4))

    def test_view_batching(self):
        for geometry_type in self.geometry_types:
            with self.subTest(geometry_type=geometry_type):
                print("Testing view batching with", geometry_type)
                self.verify_view_batching(geometry_type)

    def verify_view_batching(self, geometry_type):
        self.set_angles(geometry_type)
        self.set_translation_vectors(geometry_type)
        ct_model = self.get_model(geometry_type)

        # Generate phantom
        recon_shape = ct_model.get_params('recon_shape')
        phantom = mbirjax.gen_cube_phantom(recon_shape)

        # Generate indices of pixels and get the voxel cylinders
        full_indices = mbirjax.gen_pixel_partition(recon_shape, num_subsets=1)[0]
        voxel_values = phantom.reshape((-1,) + recon_shape[2:])[full_indices]

        # Compute forward projection with all the views at once
        sinogram = ct_model.sparse_forward_project(voxel_values, full_indices)

        # Then compute the sinogram over multiple batches and reassemble them
        num_views = sinogram.shape[0]
        num_subsets = np.random.randint(2, 8)
        view_subsets = [jnp.arange(j, num_views, num_subsets) for j in range(num_subsets)]  # We don't use array_split because we want the entries to be interleaved for testing.
        sinogram_batched = [ct_model.sparse_forward_project(voxel_values, full_indices, view_indices=view_subsets[j]) for j in range(num_subsets)]
        sinogram_stitched = np.zeros_like(sinogram)
        for j in range(sinogram.shape[0]):
            sinogram_stitched[j] = sinogram_batched[j % num_subsets][j // num_subsets]

        forward_view_batch_test_result = np.allclose(sinogram, sinogram_stitched)
        self.assertTrue(forward_view_batch_test_result)

        # Then repeat for back projection
        back_projection = ct_model.sparse_back_project(sinogram, full_indices)
        back_projection_batched = [ct_model.sparse_back_project(sinogram, full_indices, view_indices=view_subsets[j]) for j in range(num_subsets)]
        back_projection_batched = np.stack(back_projection_batched, axis=0)
        back_projection_stitched = np.sum(back_projection_batched, axis=0)
        proj_diff = np.abs(back_projection_stitched - back_projection)
        # # The following is designed to highlight the bug associated with rounding in jax.
        # if np.sum(proj_diff > 1e-4) > 10:
        #     print('Num above threshold = {}, max diff = {}'.format(np.sum(proj_diff > 1e-4), np.amax(proj_diff)))
        #     row_index0, col_index0 = jnp.unravel_index(full_indices, recon_shape[:2])
        #     recon0 = jnp.zeros(recon_shape)
        #     recon0 = recon0.at[row_index0, col_index0].set(back_projection)
        #     recon1 = jnp.zeros(recon_shape)
        #     recon1 = recon1.at[row_index0, col_index0].set(back_projection_stitched)
        #     title = 'Standard backprojection (left) and \nabs diff with back projection via multiple view subsets (right)'
        #     title += '\nDifferences are due to inconsistent choices of rounding in jax.  See experiments/bugs'
        #     mbirjax.slice_viewer(recon0, recon1-recon0, slice_axis=2, vmax=0.2, title=title)
        back_view_batch_test_result = np.sum(proj_diff > 1e-4) < 1000 and np.amax(proj_diff) < 0.2
        self.assertTrue(back_view_batch_test_result)

    def verify_adjoint(self, geometry_type):
        """
        Verify the adjoint property of the projectors:
        Choose a random phantom, x, and a random sinogram, y, and verify that <y, Ax> = <Aty, x>.
        """
        self.set_angles(geometry_type)
        self.set_translation_vectors(geometry_type)
        ct_model = self.get_model(geometry_type)

        # Initialize a random key
        seed_value = np.random.randint(1000000)
        key = jax.random.PRNGKey(seed_value)

        # Generate phantom
        recon_shape = ct_model.get_params('recon_shape')
        num_recon_rows, num_recon_cols, num_recon_slices = recon_shape[:3]
        phantom = mbirjax.gen_cube_phantom(recon_shape)

        # Generate indices of pixels
        num_subsets = 1
        full_indices = mbirjax.gen_pixel_partition(recon_shape, num_subsets)

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

        # Run once to finish compiling and get backprojection shape
        bp = ct_model.sparse_back_project(sinogram, indices[0])

        # ##########################
        # Test the adjoint property
        # Get a random 3D phantom to test the adjoint property
        key, subkey = jax.random.split(key)
        x = jax.random.uniform(subkey, shape=bp.shape)
        key, subkey = jax.random.split(key)
        y = jax.random.uniform(subkey, shape=sinogram.shape)

        # Do a forward projection, then a backprojection
        voxel_values = x.reshape((-1, num_recon_slices))[indices[0]]
        Ax = ct_model.sparse_forward_project(voxel_values, indices[0])
        Aty = ct_model.sparse_back_project(y, indices[0])

        # Calculate <Aty, x> and <y, Ax>
        Aty_x = jnp.sum(Aty * x)
        y_Ax = jnp.sum(y * Ax)

        # Determine if property holds
        adjoint_test_result = np.allclose(Aty_x, y_Ax, rtol=1e-4)
        print("maximum difference = ", np.max(Aty_x - y_Ax))
        print("minimum difference = ", np.min(Aty_x - y_Ax))
        self.assertTrue(adjoint_test_result)

    def verify_hessian(self, geometry_type):
        """
        Verify the hessian property of the back projector:
        Choose a random pixel, set it to epsilon, apply A^T A and compare to the value from compute_hessian_diagaonal.
        """
        self.set_angles(geometry_type)
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
        i, j = jax.random.randint(subkey, shape=(2,), minval=0, maxval=num_recon_rows)
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
        AtAx = ct_model.sparse_back_project(Ax, indices).reshape(x.shape)
        finite_diff_hessian = AtAx[i, j, k] / eps

        # Determine if property holds
        hessian_test_result = jnp.allclose(hessian.reshape(x.shape)[i, j, k], finite_diff_hessian)
        self.assertTrue(hessian_test_result)


if __name__ == '__main__':
    unittest.main()
