import numpy as np
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

    def test_qggmrf_denoiser(self):
        """
        Verify that the denoiser works.
        """
        for image_shape in [(1, 100, 100), (100, 100, 100)]:

            max_iterations = 20
            stop_threshold_change_pct = 0.1
            sigma_noise = 0.1
            sharpness = 0.0

            # Get some noisy data
            phantom = np.zeros(image_shape)
            phantom[:, image_shape[1]//3:2*image_shape[1]//3, image_shape[2]//3:2*image_shape[2]//3] = 1
            phantom_noisy = phantom + sigma_noise * np.random.randn(*image_shape)

            denoiser = mj.QGGMRFDenoiser(phantom.shape)
            denoiser.set_params(sharpness=sharpness)
            sigma_noise = 0.1
            phantom_denoised, recon_dict = denoiser.denoise(phantom_noisy, sigma_noise,
                                                            max_iterations=max_iterations,
                                                            stop_threshold_change_pct=stop_threshold_change_pct)
            nrmse = np.linalg.norm(phantom_denoised - phantom) / np.linalg.norm(phantom)
            tolerance = 0.2
            self.assertTrue(nrmse < tolerance)

    def test_median_filter_3d(self):
        a = np.arange(3 * 3 * 3).reshape((3, 3, 3))

        b = mj.median_filter3d(a, return_min_max=True)
        b0 = np.array([[[3., 3., 4.],
                        [6., 6., 7.],
                        [6., 7., 8.]],
                       [[10., 11., 11.],
                        [12., 13., 14.],
                        [15., 15., 16.]],
                       [[18., 19., 20.],
                        [19., 20., 20.],
                        [22., 23., 23.]]])
        b1 = np.array([[[0., 0., 1.],
                        [0., 0., 1.],
                        [3., 3., 4.]],

                       [[0., 0., 1.],
                        [0., 0., 1.],
                        [3., 3., 4.]],

                       [[9., 9., 10.],
                        [9., 9., 10.],
                        [12., 12., 13.]]])
        b2 = np.array([[[13., 14., 14.],
                        [16., 17., 17.],
                        [16., 17., 17.]],

                       [[22., 23., 23.],
                        [25., 26., 26.],
                        [25., 26., 26.]],

                       [[22., 23., 23.],
                        [25., 26., 26.],
                        [25., 26., 26.]]])
        assert np.allclose(b[0], b0)
        assert np.allclose(b[1], b1)
        assert np.allclose(b[2], b2)


if __name__ == '__main__':
    unittest.main()
