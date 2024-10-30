import numpy as np
import jax.numpy as jnp
import mbirjax
import unittest


class TestFBPReconstruction(unittest.TestCase):
    """
    Unit test for FBP reconstruction. 
    Specifically tests FBP for parallel beam geometry and cone-beam geometry (FDK method) with a planar detector panel. 
    Tests the accuracy of the reconstruction against the 3 metrics: NRMSE, max_diff, and pct_95.
    """

    def setUp(self):
        """Set up parameters and initialize models before each test."""
        # Sinogram parameters
        self.num_views = 64
        self.num_det_rows = 128
        self.num_det_channels = 128
        self.sinogram_shape = (self.num_views, self.num_det_rows, self.num_det_channels)

        # Geometry parameters
        self.fbp_angles =  jnp.linspace(-jnp.pi * (1/2), jnp.pi * (1/2), self.num_views, endpoint=False)
        self.source_detector_dist = 4 * self.num_det_channels
        self.detector_cone_angle = 2 * np.arctan2(self.num_det_channels / 2, self.source_detector_dist)
        detector_cone_angle = 2 * np.arctan2(self.num_det_channels / 2, self.source_detector_dist)
        start_angle = -(jnp.pi + detector_cone_angle) * (1/2)
        end_angle = (jnp.pi + detector_cone_angle) * (1/2)
        self.cone_angles = jnp.linspace(start_angle, end_angle, self.num_views, endpoint=False)
        self.source_detector_dist = 4 * self.num_det_channels
        self.source_iso_dist = self.source_detector_dist / 2
        
        # Initialize both models
        self.parallel_model = mbirjax.ParallelBeamModel(self.sinogram_shape, self.fbp_angles)
        self.cone_model = mbirjax.ConeBeamModel(self.sinogram_shape, 
                                                self.cone_angles,                             
                                                source_detector_dist=self.source_detector_dist, 
                                                source_iso_dist=self.source_iso_dist)

        # Generate 3D Shepp-Logan phantom and sinogram
        self.fbp_phantom = self.parallel_model.gen_modified_3d_sl_phantom()
        self.fbp_sino = self.parallel_model.forward_project(self.fbp_phantom)
        self.fdk_phantom = self.cone_model.gen_modified_3d_sl_phantom()
        self.fdk_sino = self.cone_model.forward_project(self.fdk_phantom)

        # Set tolerances for the metrics - FBP is deterministic, so results should never go above these.
        self.fbp_tolerances = {'nrmse': 0.24050 + 1e-6, 'max_diff': 0.43444 + 1e-6, 'pct_95': 0.09220 + 1e-6}
        self.fdk_tolerances = {'nrmse': 0.20, 'max_diff': 0.40, 'pct_95': 0.05}

    def test_fbp_reconstruction(self):
        """Test the FBP reconstruction against the defined tolerances."""
        # Perform FBP reconstruction
        filter_name = "ramp"
        recon = self.parallel_model.fbp_recon(self.fbp_sino, filter_name=filter_name)
        recon.block_until_ready()

        # Compute the statistics
        max_diff = np.amax(np.abs(self.fbp_phantom - recon))
        nrmse = np.linalg.norm(recon - self.fbp_phantom) / np.linalg.norm(self.fbp_phantom)
        pct_95 = np.percentile(np.abs(recon - self.fbp_phantom), 95)

        # Verify that the computed stats are within tolerances
        self.assertTrue(max_diff < self.fbp_tolerances['max_diff'], f"Max difference too high: {max_diff}")
        self.assertTrue(nrmse < self.fbp_tolerances['nrmse'], f"NRMSE too high: {nrmse}")
        self.assertTrue(pct_95 < self.fbp_tolerances['pct_95'], f"95th percentile difference too high: {pct_95}")

    def test_fdk_reconstruction(self):
        """Test the FDK reconstruction against the defined tolerances."""
        # Perform FBP reconstruction
        filter_name = "ramp"
        recon = self.cone_model.fdk_recon(self.fdk_sino, filter_name=filter_name)
        recon.block_until_ready()

        # Compute the statistics
        max_diff = np.amax(np.abs(self.fdk_phantom - recon))
        nrmse = np.linalg.norm(recon - self.fdk_phantom) / np.linalg.norm(self.fdk_phantom)
        pct_95 = np.percentile(np.abs(recon - self.fdk_phantom), 95)

        # Verify that the computed stats are within tolerances
        self.assertTrue(max_diff < self.fdk_tolerances['max_diff'], f"Max difference too high: {max_diff}")
        self.assertTrue(nrmse < self.fdk_tolerances['nrmse'], f"NRMSE too high: {nrmse}")
        self.assertTrue(pct_95 < self.fdk_tolerances['pct_95'], f"95th percentile difference too high: {pct_95}")

if __name__ == '__main__':
    unittest.main()