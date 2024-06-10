import numpy as np
import jax
import jax.numpy as jnp
import mbirjax
import unittest


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
        self.geometry_types = ['parallel', 'cone']

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

    def get_model(self, geometry_type):
        if geometry_type == 'cone':
            ct_model = mbirjax.ConeBeamModel(self.sinogram_shape, self.angles,
                                             source_detector_dist=self.source_detector_dist,
                                             source_iso_dist=self.source_iso_dist)
        elif geometry_type == 'parallel':
            ct_model = mbirjax.ParallelBeamModel(self.sinogram_shape, self.angles)
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

    def verify_adjoint(self, geometry_type):
        """
        Verify the adjoint property of the projectors:
        Choose a random phantom, x, and a random sinogram, y, and verify that <y, Ax> = <Aty, x>.
        """
        self.set_angles(geometry_type)
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
        adjoint_test_result = np.allclose(Aty_x, y_Ax)
        self.assertTrue(adjoint_test_result)

    def verify_hessian(self, geometry_type):
        """
        Verify the hessian property of the back projector:
        Choose a random pixel, set it to epsilon, apply A^T A and compare to the value from compute_hessian_diagaonal.
        """
        self.set_angles(geometry_type)
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
