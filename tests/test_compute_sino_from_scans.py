import unittest
import numpy as np
import jax
import jax.numpy as jnp
import sys
import os
import importlib.util
import matplotlib.pyplot as plt

# Define the local package path
source_path = "/Users/a124601/Desktop/Research_Purdue/MAR/Slides/0220/mbirjax_preprocessing_gpu/mbirjax"

# Explicitly remove any previously loaded mbirjax from sys.modules
if "mbirjax" in sys.modules:
    del sys.modules["mbirjax"]

# Manually load the local mbirjax package
package_name = "mbirjax"
spec = importlib.util.spec_from_file_location(package_name, os.path.join(source_path, "__init__.py"))
mbirjax = importlib.util.module_from_spec(spec)
sys.modules[package_name] = mbirjax
spec.loader.exec_module(mbirjax)

# Verify that the local version is loaded
print("mbirjax loaded from:", mbirjax.__file__)

from mbirjax.preprocess.nsi import load_scans_and_params
from mbirjax.preprocess.utilities import compute_sino_transmission_jax, compute_sino_transmission
from mbirjax.preprocess.utilities import estimate_background_offset_jax, estimate_background_offset
from mbirjax.preprocess.utilities import correct_det_rotation_batch_pix, correct_det_rotation

class TestNSIPreprocessing(unittest.TestCase):
    """
    Unit tests for NSI dataset preprocessing functions.
    Tests:
    1. Sinogram computation using JAX vs. GDT.
    2. Geometry parameter consistency.
    3. Background offset correction.
    """
    @staticmethod
    def generate_dark_scan(shape, mean=0, stddev=1, clip_negative=True, seed=None):
        """
        Generate a random dark scan with Gaussian noise.

        Parameters:
        - shape (tuple): Shape of the dark scan (e.g., (height, width)).
        - mean (float): Mean of the Gaussian noise (default: 0.0).
        - stddev (float): Standard deviation of the Gaussian noise (default: 0.01).
        - clip_negative (bool): Whether to clip negative values to zero (default: True).

        Returns:
        - dark_scan (numpy.ndarray): Simulated dark scan.
        """
        if seed is not None:
            np.random.seed(seed)
        # Generate Gaussian noise
        dark_scan = np.random.normal(mean, stddev, size=shape)

        # Clip negative values if needed
        if clip_negative:
            dark_scan = np.clip(dark_scan, 0, None)

        return dark_scan

    def setUp(self):
        """Set up parameters and initialize models before each test."""
        # Sinogram parameters
        self.num_views = 128
        self.num_det_rows = 128
        self.num_det_channels = 128
        self.sinogram_shape = (self.num_views, self.num_det_rows, self.num_det_channels)

        # Geometry parameters - FDK
        self.source_detector_dist = 4 * self.num_det_channels
        self.source_iso_dist = self.source_detector_dist / 2
        start_angle = -jnp.pi  # For testing purposes, we use a full 360 degrees
        end_angle = jnp.pi
        self.fdk_angles = jnp.linspace(start_angle, end_angle, self.num_views, endpoint=False)

        # Initialize CT model
        self.cone_model = mbirjax.ConeBeamModel(self.sinogram_shape,
                                                self.fdk_angles,
                                                source_detector_dist=self.source_detector_dist,
                                                source_iso_dist=self.source_iso_dist)

        # Generate 3D Shepp-Logan phantom and sinogram
        self.phantom = self.cone_model.gen_modified_3d_sl_phantom()
        self.sino_gdt = self.cone_model.forward_project(self.phantom)
        self.sino_gdt = self.sino_gdt / np.max(self.sino_gdt)
        self.idea_obj_scan = jnp.ones_like(self.sino_gdt) * np.exp(-self.sino_gdt)
        mean = np.mean(self.idea_obj_scan) / 100
        stddev = 0.001

        self.blank_scan = jnp.ones_like(self.sino_gdt) + self.generate_dark_scan(self.sinogram_shape, mean=mean, stddev=stddev, seed=42)
        self.obj_scan = self.idea_obj_scan + self.generate_dark_scan(self.sinogram_shape[1:], mean=mean, stddev=stddev, seed=43)
        # Simulate object scan

        self.dark_scan = self.generate_dark_scan(self.sinogram_shape, mean=mean, stddev=stddev, seed=44)

        self.compute_sino_tolerance = {'atol': 1e-2}

    def test_sinogram_computation(self):
        """Test if sinograms computed by JAX and GDT are numerically close."""
        sino_computed, _ = compute_sino_transmission_jax(self.obj_scan, self.blank_scan, self.dark_scan)


        # Compare sinograms
        self.assertTrue(np.allclose(sino_computed, self.sino_gdt, atol=self.compute_sino_tolerance['atol']),
                        f"Sinograms differ more than {self.compute_sino_tolerance['atol']}. Max diff: {np.max(np.abs(sino_computed - self.sino_gdt))}")


if __name__ == '__main__':
    unittest.main()
elif "unittest" in sys.modules:
    unittest.main()
