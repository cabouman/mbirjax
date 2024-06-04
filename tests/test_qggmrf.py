# test_qggmrf.py

import numpy as np
import jax
import jax.numpy as jnp
import mbirjax
import unittest


class TestQGGMRF(unittest.TestCase):
    """
    Test components of the qggmrf prior model
    """

    def setUp(self):
        """Set up before each test method."""
        pass

    def tearDown(self):
        """Clean up after each test method."""
        pass

    def test_delta(self):
        recon_shape = (60, 101, 47)
        voxel_values = np.zeros(recon_shape)

        # Choose a pixel from the middle third of the recon shape.
        row_index = np.random.randint(recon_shape[0] // 3) + recon_shape[0] // 3
        col_index = np.random.randint(recon_shape[1] // 3) + recon_shape[1] // 3

        # Set up a single pixel cylinder with a linearly increasing set of values down the cylinder
        cylinder_values = np.random.rand(recon_shape[2])
        voxel_values[row_index, col_index] = cylinder_values
        voxel_values = voxel_values.reshape((-1, recon_shape[2]))
        # Set up to calculate the 8 nearest neighbors of the center pixel
        row_inds = [row_index - 1, row_index, row_index + 1]
        col_inds = [col_index - 1, col_index, col_index + 1]
        pixel_indices = [jnp.ravel_multi_index((r, c), recon_shape[0:2]) for r in row_inds for c in col_inds]
        pixel_indices = jnp.array(pixel_indices)
        # Get delta - comes back with shape (num_pixel_indices, num_slices, num_neighbors=6)
        delta = mbirjax.tomography_model.get_delta(voxel_values, recon_shape, pixel_indices)

        corner_inds = (0, 2, 6, 8)
        corners = np.array([delta[corner_ind] for corner_ind in corner_inds])

        # The corners should all be 0
        eps = 1e-7
        assert(jnp.allclose(corners, 0, atol=eps))

        # The 4 side points should have negative the center values
        sides = [np.array(delta[1, :, 0]) + cylinder_values]
        sides.append(np.array(delta[3, :, 2]) + cylinder_values)
        sides.append(np.array(delta[5, :, 3]) + cylinder_values)
        sides.append(np.array(delta[7, :, 1]) + cylinder_values)
        sides = np.array(sides)
        assert(jnp.allclose(sides, 0, atol=eps))

        # The center point should have the center values in the first 4 columns (for side points)
        assert(jnp.allclose(delta[4, :, 0:4] - cylinder_values.reshape((-1, 1)), 0, atol=eps))
        # Then the difference values in the next two columns (for up and down)
        assert(jnp.allclose(delta[4, 0:-1, 4] + jnp.diff(cylinder_values), 0, atol=eps))
        assert(jnp.allclose(delta[4, 1:, 5] - jnp.diff(cylinder_values), 0, atol=eps))


if __name__ == '__main__':
    unittest.main()
