import numpy as np
import jax
import jax.numpy as jnp
import mbirjax
import unittest


class TestProx(unittest.TestCase):
    """
    Test the proximal map function for the special case when the input and initial condition are both the ground truth.
    """

    def setUp(self):
        """Set up before each test method."""
        np.random.seed(0)  # Set a seed to avoid variations due to partition creation.

        # Set parameters
        self.num_views = 32
        self.num_det_rows = 40
        self.num_det_channels = 128
        self.sharpness = 0.0

        # Initialize sinogram
        self.sinogram_shape = (self.num_views, self.num_det_rows, self.num_det_channels)
        self.angles = None

    def tearDown(self):
        """Clean up after each test method."""
        pass

    def test_prox(self):
        """
        Verify that the proximal map function works for a simple corner case.
        """
        start_angle = 0.0
        end_angle = np.pi

        self.angles = jnp.linspace(start_angle, end_angle, self.num_views, endpoint=False)
        ct_model = mbirjax.ParallelBeamModel(self.sinogram_shape, self.angles)

        # Generate 3D Shepp Logan phantom
        print('  Creating phantom')
        phantom = ct_model.gen_modified_3d_sl_phantom()

        # Generate synthetic sinogram data
        print('  Creating sinogram')
        sinogram = ct_model.forward_project(phantom)

        # Set reconstruction parameter values
        ct_model.set_params(verbose=0)

        # ##########################
        # Evaluating the proximal map
        print('  Starting proximal map')
        prox_input = phantom
        recon, recon_params = ct_model.prox_map(prox_input, sinogram, max_iterations=5, init_recon=prox_input)
        recon.block_until_ready()

        max_diff = np.amax(np.abs(phantom - recon))
        tolerance = 1e-5
        self.assertTrue(max_diff < tolerance)


if __name__ == '__main__':
    unittest.main()
