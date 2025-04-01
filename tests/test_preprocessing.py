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
    def generate_dark_scan(shape, mean=0.0, stddev=1.1, clip_negative=True, seed=None):
        """
        Generate a random dark scan with Gaussian noise.

        Parameters:
        - shape (tuple): Shape of the dark scan (e.g., (height, width)).
        - mean (float or ndarray or jax.array): Mean of the Gaussian noise (default: 0.0).
        - stddev (float or ndarray or jax.array): Standard deviation of the Gaussian noise (default: 0.01).
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
        # Define the crop region to test cropping and defective pixels.  We'll mask the phantom so that there is
        # sufficient border to do background estimation.  We'll include (0, 0) and the opposite corners as defective
        # pixels.  The crop region will exclude (0, 0) but include the opposite pixel.  In the crop region,
        # we have (row_frac0, row_frac1), (col_frac0, col_frac1), so we need the form (row_frac0, 1), (0, 1) to meet these conditions.
        self.crop_pixels_sides = 1
        self.crop_pixels_top = 5
        self.crop_pixels_bottom = 0
        self.edge_width = 4
        row0 = round(self.crop_pixels_top)
        border_width = self.edge_width

        self.phantom = self.cone_model.gen_modified_3d_sl_phantom()
        sino_gt = self.cone_model.forward_project(self.phantom)

        # Mask the borders as needed
        sino_gt = np.array(sino_gt)
        sino_gt[:, :border_width + row0, :] = 0.0  # Top
        sino_gt[:, -border_width:, :] = 0.0  # Bottom
        sino_gt[:, :, :border_width] = 0.0  # Left
        sino_gt[:, :, -border_width:] = 0.0  # Right
        self.sino_gt = jnp.array(sino_gt)

        # Normalize the sinogram
        self.sino_gt = self.sino_gt / jnp.percentile(self.sino_gt, 98)
        self.ideal_obj_scan = self.maximum_intensity * jnp.exp(-self.sino_gt)
        # Set the mean and standard deviation for the dark scan
        # These values are estimated empirically from NSI data.
        dark_mean = 0.02
        dark_stddev = 0.001

        # Generate a single dark scan
        self.dark_scan = self.generate_dark_scan((1,) + self.sinogram_shape[1:],
                                                 mean=dark_mean, stddev=dark_stddev, seed=44)

        # Create blank scan using the maximum intensity plus a realization of the dark scan.
        # Then repeat for the object scan, using the noise-free scan plus a new dark scan.
        self.blank_scan = self.maximum_intensity + self.generate_dark_scan((1,) + self.sinogram_shape[1:],
                                                                           mean=dark_mean, stddev=dark_stddev, seed=42)
        self.obj_scan = self.ideal_obj_scan + self.generate_dark_scan(self.sinogram_shape, mean=dark_mean,
                                                                      stddev=dark_stddev, seed=43)

        # Randomly generate defective pixel coordinates.
        np.random.seed(25)
        num_defective_pixels = 15
        defective_pixels = [
            (np.random.randint(0, self.num_det_rows - 1), np.random.randint(0, self.num_det_channels - 1)) for j in
            range(num_defective_pixels)]
        # Include (0, 0) to test the ability to crop when there are defective pixels.
        defective_pixels = [(0, 0), (self.num_det_rows-1, self.num_det_channels-1)] + defective_pixels
        self.defective_pixel_array = np.array(defective_pixels)

        # Randomly set other pixels to nan to test the function's ability to recover
        nan_pixels = [(np.random.randint(0, self.num_views - 1), np.random.randint(0, self.num_det_rows - 1),
                       np.random.randint(0, self.num_det_channels - 1)) for j in range(num_defective_pixels)]

        obj_scan = np.array(self.obj_scan)
        for index in nan_pixels:
            obj_scan[index[0], index[1], index[2]] = np.nan
        self.obj_scan = jnp.array(obj_scan)

        # Set the tolerances for the test
        self.preprocessing_tolerance = {'atol': 0.14, 'nrmse_tol': 0.0015, 'pct99_tol': 0.0018}

    def test_preprocessing(self):
        """Test if background offset correction is consistent between JAX and GDT implementations."""
        obj_scan, blank_scan, dark_scan, defective_pixel_array = preprocess.crop_scans(self.obj_scan, self.blank_scan, self.dark_scan,
                                                                                       defective_pixel_array=self.defective_pixel_array,
                                                                                       crop_pixels_sides=self.crop_pixels_sides,
                                                                                       crop_pixels_top=self.crop_pixels_top,
                                                                                       crop_pixels_bottom=self.crop_pixels_bottom)
        sino_computed = preprocess.compute_sino_transmission(obj_scan, blank_scan, dark_scan,
                                                             defective_pixel_array=defective_pixel_array)

        # Compute background offsets
        background_offset = preprocess.estimate_background_offset(sino_computed, edge_width=self.edge_width)
        print("background_offset = ", background_offset)
        sino_computed = sino_computed - background_offset

        sino_gt_cropped, _, _, _ = preprocess.crop_scans(self.sino_gt, self.blank_scan, self.dark_scan,
                                                         crop_pixels_sides=self.crop_pixels_sides,
                                                         crop_pixels_top=self.crop_pixels_top,
                                                         crop_pixels_bottom=self.crop_pixels_bottom)
        abs_sino_diff = np.abs(sino_computed - sino_gt_cropped)
        max_diff = np.max(np.abs(abs_sino_diff))
        nrmse = np.linalg.norm(abs_sino_diff) / np.linalg.norm(sino_gt_cropped)
        pct99 = np.percentile(abs_sino_diff, 99)

        print('Difference between gt sino and estimated sino: max abs = {:.4f}, nrmse = {:.4f}'.format(max_diff, nrmse))
        print('99% of absolute sinogram differences are less than {:.4f}'.format(pct99))

        tolerance = self.preprocessing_tolerance['atol']
        tolerance_mean = self.preprocessing_tolerance['nrmse_tol']
        tolerance_pct99 = self.preprocessing_tolerance['pct99_tol']

        # Check if differences are within tolerance
        self.assertTrue(
            max_diff < tolerance and nrmse < tolerance_mean and pct99 < tolerance_pct99,
            f"Sinograms differ more than the tolerance. "
            f"Max diff={max_diff:.4f} (tolerance: {tolerance}), "
            f"NRMSE={nrmse:.4f} (tolerance: {tolerance_mean}), 99th percentile={pct99:.4f} (tolerance: {tolerance_pct99})"
        )
        self.assertFalse(np.isnan(sino_computed).any(), "Error: sino_computed contains NaN values!")


if __name__ == '__main__':
    unittest.main()

