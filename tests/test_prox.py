import numpy as np
import jax
import jax.numpy as jnp
import mbirjax as mj
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
        ct_model = mj.ParallelBeamModel(self.sinogram_shape, self.angles)

        # Generate 3D Shepp Logan phantom
        print('  Creating phantom')
        phantom_shape = ct_model.get_params('recon_shape')
        phantom = mj.generate_3d_shepp_logan_low_dynamic_range(phantom_shape)

        # Generate synthetic sinogram data
        print('  Creating sinogram')
        sinogram = ct_model.forward_project(phantom)

        # Set reconstruction parameter values
        ct_model.set_params(verbose=0)

        # ##########################
        # Evaluating the proximal map
        print('  Starting proximal map')
        prox_input = phantom + 0.1 * (phantom > 0)

        ct_model.set_params(sharpness=6, snr_db=60)
        recon, recon_dict = ct_model.recon(sinogram, max_iterations=20, first_iteration=0,
                                           stop_threshold_change_pct=0.01)

        # Do a prox map with small sigma_prox to return very nearly the prox_input
        prox_recon0, prox_recon_dict0 = ct_model.prox_map(prox_input, sinogram, sigma_prox=1e-6,
                                                          max_iterations=20,
                                                          first_iteration=0, init_recon=prox_input)

        # Do a prox map with large sigma_prox to return very nearly the same as a recon with small prior
        prox_recon1, prox_recon_dict1 = ct_model.prox_map(prox_input, sinogram, sigma_prox=1e6,
                                                          max_iterations=20,
                                                          first_iteration=0, init_recon=recon)

        small_sigma_nrmse = np.linalg.norm(prox_recon0 - prox_input) / np.linalg.norm(prox_input)
        large_sigma_nrmse = np.linalg.norm(prox_recon1 - recon) / np.linalg.norm(recon)
        tolerance = 2e-5
        self.assertTrue(small_sigma_nrmse < tolerance)
        tolerance = 3e-4
        self.assertTrue(large_sigma_nrmse < tolerance)


if __name__ == '__main__':
    unittest.main()
