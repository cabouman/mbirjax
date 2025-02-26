import unittest
import numpy as np
import jax.numpy as jnp
from mbirjax.preprocess.nsi import load_scans_and_params
from mbirjax.preprocess import compute_sino_transmission_jax, compute_sino_transmission
from mbirjax.preprocess import estimate_background_offset_jax, estimate_background_offset
from mbirjax.preprocess import correct_det_rotation_batch_pix, correct_det_rotation

class TestNSIPreprocessing(unittest.TestCase):
    """
    Unit tests for NSI dataset preprocessing functions.
    Tests:
    1. Sinogram computation using JAX vs. GDT.
    2. Geometry parameter consistency.
    3. Background offset correction.
    """

    def setUp(self):
        """Set up parameters and load scans before each test."""
        # Define test parameters
        self.dataset_dir = "/home/li5273/PycharmProjects/mbirjax_applications/nsi/demo_data/vert_no_metal"
        self.downsample_factor = [8, 8]
        self.subsample_view_factor = 1
        self.crop_region = ((0, 1), (0, 1))  # Default: use entire scan
        self.atol = 1e-5  # Tolerance for numerical checks

        # Load scans and parameters
        self.obj_scan, self.blank_scan, self.dark_scan, self.cone_beam_params, self.optional_params, self.defective_pixel_list = \
            load_scans_and_params(self.dataset_dir,
                                  downsample_factor=self.downsample_factor,
                                  crop_region=self.crop_region,
                                  subsample_view_factor=self.subsample_view_factor)
        self.sino_gdt, _ = compute_sino_transmission(self.obj_scan, self.blank_scan, self.dark_scan, self.defective_pixel_list)
        self.det_rotation = self.optional_params["det_rotation"]

    def test_sinogram_computation(self):
        """Test if sinograms computed by JAX and GDT are numerically close."""
        sino_jax, _ = compute_sino_transmission_jax(self.obj_scan, self.blank_scan, self.dark_scan, self.defective_pixel_list)


        # Compare sinograms
        self.assertTrue(np.allclose(sino_jax, self.sino_gdt, atol=self.atol),
                        f"Sinograms differ more than {self.atol}. Max diff: {np.max(np.abs(sino_jax - self.sino_gdt))}")

    def test_background_offset_correction(self):
        """Test if background offset correction is consistent between JAX and GDT implementations."""

        # Compute background offsets
        bg_offset_jax = estimate_background_offset_jax(self.sino_gdt)
        bg_offset_gdt = estimate_background_offset(self.sino_gdt)

        # Compare offsets
        self.assertAlmostEqual(bg_offset_jax, bg_offset_gdt, places=5,
                               msg=f"Background offsets do not match! JAX: {bg_offset_jax}, GDT: {bg_offset_gdt}")

    def test_detector_rotation_correction(self):
            """Test if sinograms corrected using batch and regular rotation methods are numerically close."""
            sino_rotated_batch = correct_det_rotation_batch_pix(self.sino_gdt, det_rotation=self.det_rotation)
            sino_rotated_orig = correct_det_rotation(self.sino_gdt, weights=None, det_rotation=self.det_rotation)

            # Compare results
            self.assertTrue(np.allclose(sino_rotated_batch, sino_rotated_orig, atol=self.atol),
                            f"Rotated sinograms differ more than {self.atol}. Max diff: {np.max(np.abs(sino_rotated_batch - sino_rotated_orig))}")


if __name__ == '__main__':
    unittest.main()
