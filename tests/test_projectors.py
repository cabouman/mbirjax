# test_projectors.py

import types
import numpy as np
import yaml
import warnings
import gc
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
        pass

    def tearDown(self):
        """Clean up after each test method."""
        pass


    def test_parallel_beam_adjoint_property(self):
        """
        Verify the adjoint property of ParallelBeamModel projector:
        Choose a random phantom, x, and a random sinogram, y, and verify that <y, Ax> = <Aty, x>.
        Verify this both for the full projector and the subset projector.
        """
        # Initialize sinogram parameters
        num_views = 64
        num_det_rows = 5
        num_det_channels = 64
        start_angle = -np.pi * (1 / 2)
        end_angle = np.pi * (1 / 2)

        # Initialize sinogram
        sinogram = jnp.zeros((num_views, num_det_rows, num_det_channels))
        angles = jnp.linspace(start_angle, end_angle, num_views, endpoint=False)

        # Set up parallel beam model
        parallel_model = mbirjax.ParallelBeamModel(sinogram.shape, angles)

        # Initialize a random key
        seed_value = np.random.randint(1000000)
        key = jax.random.PRNGKey(seed_value)

        # Generate phantom
        recon_shape = parallel_model.get_params('recon_shape')
        num_recon_rows, num_recon_cols, num_recon_slices = recon_shape[:3]
        phantom = mbirjax.gen_cube_phantom(recon_shape)

        # Generate indices of pixels
        num_subsets = 1
        full_indices = mbirjax.gen_pixel_partition(recon_shape, num_subsets)
        num_subsets = 5
        subset_indices = mbirjax.gen_pixel_partition(recon_shape, num_subsets)

        # Generate sinogram data
        voxel_values = phantom.reshape((-1,) + recon_shape[2:])[full_indices]

        # Compute forward projection
        sinogram = parallel_model.sparse_forward_project(voxel_values[0], full_indices[0])

        # Get the vector of indices
        indices = jnp.arange(num_recon_rows * num_recon_cols)
        num_trials = 3
        indices = jnp.mod(np.arange(num_trials, dtype=int).reshape((-1, 1)) + indices.reshape((1, -1)),
                          num_recon_rows * num_recon_cols)

        # Convert to jax arrays
        sinogram = jnp.array(sinogram)
        indices = jnp.array(indices)

        # Run once to finish compiling and get backprojection shape
        bp = parallel_model.sparse_back_project(sinogram, indices[0])

        # ##########################
        # Test the adjoint property
        # Get a random 3D phantom to test the adjoint property
        key, subkey = jax.random.split(key)
        x = jax.random.uniform(subkey, shape=bp.shape)
        key, subkey = jax.random.split(key)
        y = jax.random.uniform(subkey, shape=sinogram.shape)

        # Do a forward projection, then a backprojection
        voxel_values = x.reshape((-1, num_recon_slices))[indices[0]]
        Ax = parallel_model.sparse_forward_project(voxel_values, indices[0])
        Aty = parallel_model.sparse_back_project(y, indices[0])

        # Calculate <Aty, x> and <y, Ax>
        Aty_x = jnp.sum(Aty * x)
        y_Ax = jnp.sum(y * Ax)

        # Determine if property holds
        adjoint_test_result = np.allclose(Aty_x, y_Ax)
        self.assertTrue(adjoint_test_result)


    def test_parallel_beam_hessian_property(self):
        """
        Verify the hessian property of ParallelBeamModel projector:
        Choose a random phantom, x, and a random sinogram, y, and verify that <y, Ax> = <Aty, x>.
        Verify this both for the full projector and the subset projector.
        """
        # Initialize sinogram parameters
        num_views = 64
        num_det_rows = 5
        num_det_channels = 64
        start_angle = -np.pi * (1 / 2)
        end_angle = np.pi * (1 / 2)

        # Initialize sinogram
        sinogram = jnp.zeros((num_views, num_det_rows, num_det_channels))
        angles = jnp.linspace(start_angle, end_angle, num_views, endpoint=False)

        # Set up parallel beam model
        parallel_model = mbirjax.ParallelBeamModel(sinogram.shape, angles)

        # Initialize a random key
        seed_value = np.random.randint(1000000)
        key = jax.random.PRNGKey(seed_value)

        # Generate phantom
        recon_shape = parallel_model.get_params('recon_shape')
        num_recon_rows, num_recon_cols, num_recon_slices = recon_shape[:3]
        phantom = mbirjax.gen_cube_phantom(recon_shape)

        # Generate indices of pixels
        num_subsets = 1
        full_indices = mbirjax.gen_pixel_partition(recon_shape, num_subsets)
        num_subsets = 5
        subset_indices = mbirjax.gen_pixel_partition(recon_shape, num_subsets)

        # Generate sinogram data
        voxel_values = phantom.reshape((-1,) + recon_shape[2:])[full_indices]

        # Compute forward projection
        sinogram = parallel_model.sparse_forward_project(voxel_values[0], full_indices[0])

        # Get the vector of indices
        indices = jnp.arange(num_recon_rows * num_recon_cols)
        num_trials = 3
        indices = jnp.mod(np.arange(num_trials, dtype=int).reshape((-1, 1)) + indices.reshape((1, -1)),
                          num_recon_rows * num_recon_cols)

        # Convert to jax arrays
        sinogram = jnp.array(sinogram)
        indices = jnp.array(indices)

        # Run once to finish compiling and get backprojection shape
        bp = parallel_model.sparse_back_project(sinogram, indices[0])

        # ##########################
        # ## Test the hessian against a finite difference approximation ## #
        hessian = parallel_model.compute_hessian_diagonal()

        x = jnp.zeros(recon_shape)
        key, subkey = jax.random.split(key)
        i, j = jax.random.randint(subkey, shape=(2,), minval=0, maxval=num_recon_rows)
        key, subkey = jax.random.split(key)
        k = jax.random.randint(subkey, shape=(), minval=0, maxval=num_recon_slices)

        eps = 0.01
        x = x.at[i, j, k].set(eps)
        voxel_values = x.reshape((-1, num_recon_slices))[indices[0]]
        Ax = parallel_model.sparse_forward_project(voxel_values, indices[0])
        AtAx = parallel_model.sparse_back_project(Ax, indices[0]).reshape(x.shape)
        finite_diff_hessian = AtAx[i, j, k] / eps

        # Determine if property holds
        hessian_test_result = jnp.allclose(hessian.reshape(x.shape)[i, j, k], finite_diff_hessian)
        self.assertTrue(hessian_test_result)
        #print('Hessian matches finite difference: {}'.format(hessian_test_result))


if __name__ == '__main__':
    unittest.main()
