import unittest
import numpy as np
import jax.numpy as jnp
import mbirjax

import mbirjax.preprocess.utilities as preprocess


class TestNSIPreprocessing(unittest.TestCase):
    """
    Unit tests for NSI dataset preprocessing functions.
    Tests:
    Sinogram computation from scans using JAX.
    """
    @staticmethod
    def generate_dark_scan(shape, mean=0, stddev=1, clip_negative=True, seed=None):
        """
        Generate a random dark scan with Gaussian noise.

        Parameters:
        - shape (tuple): Shape of the dark scan (e.g., (height, width)).
        - mean (float or jax Array): Mean of the Gaussian noise (default: 0.0).
        - stddev (float): Standard deviation of the Gaussian noise (default: 0.01).
        - clip_negative (bool): Whether to clip negative values to zero (default: True).

        Returns:
        - dark_scan (numpy.ndarray): Simulated dark scan.
        """
        if seed is not None:
            np.random.seed(seed)
        # Generate Gaussian noise
        dark_scan = np.random.normal(mean, stddev, size=shape)

        # We should get no negative values.  If we do, then we take the absolute value instead of
        # clipping so that the values are still random.
        if clip_negative:
            dark_scan = np.abs(dark_scan)

        return dark_scan

    def setUp(self):
        """Set up parameters and initialize models before each test."""
        # Sinogram parameters
        self.num_views = 40
        self.num_det_rows = 64
        self.num_det_channels = 128
        self.sinogram_shape = (self.num_views, self.num_det_rows, self.num_det_channels)

        # Geometry parameters - FDK
        self.source_detector_dist = 4 * self.num_det_channels
        self.source_iso_dist = self.source_detector_dist / 2
        start_angle = -jnp.pi  # For testing purposes, we use a full 360 degrees
        end_angle = jnp.pi
        self.angles = jnp.linspace(start_angle, end_angle, self.num_views, endpoint=False)
        self.maximum_intensity = 4.0

        # Initialize CT model
        self.cone_model = mbirjax.ConeBeamModel(self.sinogram_shape,
                                                self.angles,
                                                source_detector_dist=self.source_detector_dist,
                                                source_iso_dist=self.source_iso_dist)

        # Generate 3D Shepp-Logan phantom and sinogram
        self.phantom = self.cone_model.gen_modified_3d_sl_phantom()
        self.sino_gt = self.cone_model.forward_project(self.phantom)
        # Normalize the sinogram
        self.sino_gt = self.sino_gt / jnp.percentile(self.sino_gt, 98)
        self.ideal_obj_scan = self.maximum_intensity * jnp.exp(-self.sino_gt)
        # Set the mean and standard deviation for the dark scan
        # These values are estimated empirically from NSI data.
        dark_mean = 0.02
        dark_stddev = 0.001

        # Create blank scan using the maximum intensity plus a realization of the dark scan.
        # Then repeat for the object scan, using the noise-free scan plus a new dark scan.
        self.blank_scan = self.maximum_intensity + self.generate_dark_scan(self.sinogram_shape, mean=dark_mean, stddev=dark_stddev, seed=42)
        self.obj_scan = self.ideal_obj_scan + self.generate_dark_scan(self.sinogram_shape[1:], mean=dark_mean, stddev=dark_stddev, seed=43)

        # Randomly generate two defective pixel coordinates with length 2 and 3 to test the function's ability to handle different coordinate lengths
        np.random.seed(25)
        self.defective_pixel_list = [(np.random.randint(0, self.num_det_rows-1), np.random.randint(0, self.num_det_channels-1)), (np.random.randint(0, self.num_views-1), np.random.randint(0, self.num_det_rows-1), np.random.randint(0, self.num_det_channels-1))]
        # Randomly set pixel to negative value to test the function's ability to handle negative object scan values
        self.obj_scan = self.obj_scan.at[np.random.randint(0, self.num_views-1), np.random.randint(0, self.num_det_rows-1), np.random.randint(0, self.num_det_channels-1)].set(-1)

        self.dark_scan = self.generate_dark_scan(self.sinogram_shape, mean=dark_mean, stddev=dark_stddev, seed=44)

        self.compute_sino_tolerance = {'atol': 1.17, 'mean_tol': 0.0012}
        self.estimate_bg_tolerance = {'atol': 1.17, 'mean_tol': 0.0012}


    def test_background_offset_correction(self):
        """Test if background offset correction is consistent between JAX and GDT implementations."""
        sino_computed, _ = preprocess.compute_sino_transmission_jax(self.obj_scan, self.blank_scan, self.dark_scan, defective_pixel_list=self.defective_pixel_list)

        # Compute background offsets
        background_offset = preprocess.estimate_background_offset_jax(sino_computed, edge_width=3)
        print("background_offset = ", background_offset)
        sino_computed = sino_computed - background_offset

        max_diff = np.max(np.abs(sino_computed - self.sino_gt))
        mean_diff = np.mean(np.abs(sino_computed - self.sino_gt))
        tolerance = self.estimate_bg_tolerance['atol']
        tolerance_mean = self.estimate_bg_tolerance['mean_tol']
        import matplotlib.pyplot as plt
        mbirjax.slice_viewer(sino_computed, slice_axis=0)
        plt.show(block=True)

        # Check if differences are within tolerance
        self.assertTrue(
            max_diff < tolerance and mean_diff < tolerance_mean,
            f"Sinograms differ more than the tolerance. "
            f"Max diff: {max_diff} (tolerance: {tolerance}), Mean diff: {mean_diff} (tolerance: {tolerance_mean})"
        )
        self.assertFalse(np.isnan(sino_computed).any(), "Error: sino_computed contains NaN values!")

if __name__ == '__main__':
    unittest.main()

