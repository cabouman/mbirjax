import unittest
import numpy as np
import jax
import jax.numpy as jnp
import scipy
import mbirjax as mj


class TestFBPReconstruction(unittest.TestCase):
    """
    Unit tests for verifying the reconstruction accuracy of the FBP and FDK algorithms in MBIRJAX.
    """
    
    def setUp(self):
        """Set up before each test method."""
        
        # Choose the geometry types
        self.geometry_types = mj._utils._geometry_types_for_tests.copy()
        self.geometry_types.remove("translation")
        self.geometry_types.remove("anisotropic_translation")
        
        # Set tolerances for the metrics - FBP is deterministic, so results should never go above these.
        self.parallel_tolerances = {'nrmse': 0.25, 'max_diff': 0.43, 'pct_95': 0.091}
        self.anisotropic_parallel_tolerances = {'nrmse': 0.23, 'max_diff': 0.49, 'pct_95': 0.074}
        self.cone_tolerances = {'nrmse': 0.35, 'max_diff': 0.59, 'pct_95': 0.144}
        self.anisotropic_cone_tolerances = {'nrmse': 0.49, 'max_diff': 0.89, 'pct_95': 0.213}
        self.helical_cone_tolerances = {'nrmse': 0.41, 'max_diff': 0.65, 'pct_95': 0.074}
        self.all_tolerances = [self.parallel_tolerances, self.anisotropic_parallel_tolerances, self.cone_tolerances,
                               self.anisotropic_cone_tolerances, self.helical_cone_tolerances]
        if len(self.geometry_types) != len(self.all_tolerances):
            raise IndexError('The list of geometry types does not match the list of test tolerances for the geometry types.')
        
        # Set parameters
        self.num_views = 64
        self.num_det_rows = 40
        self.num_det_channels = 128
        self.sharpness = 0.0
        
        # These can be adjusted to scale voxel aspect ratios for the anisotropic cases
        self.voxel_row_aspect = 1.9
        self.voxel_slice_aspect = 2.9 # cone beam only

        # These can be adjusted to describe the geometry in the cone beam case.
        # np.Inf is an allowable value, in which case this is essentially parallel beam
        self.source_detector_dist = 4 * self.num_det_channels
        self.source_iso_dist = self.source_detector_dist / 2
        
        # These can be adjusted to describe the geometry in the helical cone beam case.
        self.helical_pitch = 0.5
        self.helical_z_range = 80.0
        self.helical_z_center = 40.0

        # Initialize sinogram
        self.sinogram_shape = (self.num_views, self.num_det_rows, self.num_det_channels)
        self.angles = None
        self.helical_z_shifts = None
    
    
    def tearDown(self):
        """Clean up after each test method."""
        pass
    
    
    def set_view_params(self, geometry_type):
        if (geometry_type == 'cone') | (geometry_type == 'anisotropic_cone') | (geometry_type == 'helical_cone'):
            detector_cone_angle = 2 * np.arctan2(self.num_det_channels / 2, self.source_detector_dist)
            start_angle = -(np.pi + detector_cone_angle/2)
            end_angle = (np.pi + detector_cone_angle/2)
            self.angles = jnp.linspace(start_angle, end_angle, self.num_views, endpoint=False)
        else:
            self.angles = jnp.linspace(-jnp.pi * (1/2), jnp.pi * (1/2), self.num_views, endpoint=False)
        
        if geometry_type == 'helical_cone':
            magnification = self.source_detector_dist / self.source_iso_dist
            det_height_iso = self.num_det_rows / magnification
            z_per_rot = self.helical_pitch * det_height_iso
            dz_per_view = z_per_rot / self.num_views
            view_offsets = jnp.arange(self.num_views) - (self.num_views - 1) / 2
            self.helical_z_shifts = self.helical_z_center + dz_per_view * view_offsets
    
    
    def get_model(self, geometry_type):
        if (geometry_type == 'cone') | (geometry_type == 'anisotropic_cone'):
            ct_model = mj.ConeBeamModel(self.sinogram_shape, self.angles,
                                             source_detector_dist=self.source_detector_dist,
                                             source_iso_dist=self.source_iso_dist)
        elif geometry_type == 'helical_cone':
            ct_model = mj.ConeBeamModel(self.sinogram_shape, self.angles, helical_z_shifts=self.helical_z_shifts,
                                             source_detector_dist=self.source_detector_dist,
                                             source_iso_dist=self.source_iso_dist)
        elif (geometry_type == 'parallel') | (geometry_type == 'anisotropic_parallel'):
            ct_model = mj.ParallelBeamModel(self.sinogram_shape, self.angles)
        
        else:
            raise ValueError('Invalid geometry type.  Expected cone or parallel, got {}'.format(geometry_type))

        return ct_model
    
    
    def test_all_FBP(self):
        for geometry_type, tolerances in zip(self.geometry_types, self.all_tolerances):
            with self.subTest(geometry_type=geometry_type):
                if (geometry_type == 'parallel') | (geometry_type == 'anisotropic_parallel'):
                    print("Testing FBP with", geometry_type)
                if (geometry_type == 'cone') | (geometry_type == 'anisotropic_cone') | (geometry_type == 'helical_cone'):
                    print("Testing FDK with", geometry_type)
                self.verify_FBP(geometry_type, tolerances)
    
    
    def verify_FBP(self, geometry_type, tolerances):
        """Test the FBP reconstructions against the defined tolerances."""
        self.set_view_params(geometry_type)
        ct_model = self.get_model(geometry_type)
        
        # Generate 3D Shepp Logan phantom
        print('  Creating phantom')
        recon_shape = ct_model.get_params('recon_shape')
        phantom_shape = recon_shape
        embed_slice_start = 0
        embed_slice_stop = recon_shape[2]
        if geometry_type == 'helical_cone':
            embed_slice_start, embed_slice_stop = mj.get_helical_half_rotation_slice_range(
                ct_model,
                self.helical_pitch,
                self.helical_z_shifts
            )
            phantom_shape = (
                recon_shape[0],
                recon_shape[1],
                embed_slice_stop - embed_slice_start,
            )
        phantom_core = mj.generate_3d_shepp_logan_low_dynamic_range(phantom_shape)
        if geometry_type == 'helical_cone':
            phantom = jnp.zeros(recon_shape)
            phantom = phantom.at[:, :, embed_slice_start:embed_slice_stop].set(phantom_core)
        else:
            phantom = phantom_core
        
        # Generate synthetic sinogram data
        print('  Creating sinogram')
        sinogram = ct_model.forward_project(phantom)
        
        # Set the recon voxel aspect ratio after generating the sinogram
        if geometry_type == 'anisotropic_cone':
            ct_model.set_params(voxel_row_aspect=self.voxel_row_aspect)
            ct_model.set_params(voxel_slice_aspect=self.voxel_slice_aspect)
            ct_model.auto_set_recon_geometry()
        if geometry_type == 'anisotropic_parallel':
            ct_model.set_params(voxel_row_aspect=self.voxel_row_aspect)
            ct_model.auto_set_recon_geometry()
            
        # Perform FBP reconstruction
        print('  Starting recon')
        filter_name = "ramp"
        recon = ct_model.direct_recon(sinogram, filter_name=filter_name)
        
        # if anisotropic, rescale the recon to the phantom shape
        if (geometry_type == 'anisotropic_cone') | (geometry_type == 'anisotropic_parallel'):
            phantom_temp = scipy.ndimage.zoom(phantom, zoom=(np.shape(recon)[0] / np.shape(phantom)[0],
                                                             np.shape(recon)[1] / np.shape(phantom)[1],
                                                             np.shape(recon)[2] / np.shape(phantom)[2]))
            phantom = scipy.ndimage.zoom(phantom_temp, zoom=(np.shape(phantom)[0] / np.shape(phantom_temp)[0],
                                                             np.shape(phantom)[1] / np.shape(phantom_temp)[1],
                                                             np.shape(phantom)[2] / np.shape(phantom_temp)[2]))
            recon = scipy.ndimage.zoom(recon, zoom=(np.shape(phantom)[0] / np.shape(recon)[0],
                                                    np.shape(phantom)[1] / np.shape(recon)[1],
                                                    np.shape(phantom)[2] / np.shape(recon)[2]))
            
        # Compute the statistics
        max_diff = np.amax(np.abs(phantom - recon))
        nrmse = np.linalg.norm(recon - phantom) / np.linalg.norm(phantom)
        pct_95 = np.percentile(np.abs(recon - phantom), 95)
        print('  nrmse = {:.3f}'.format(nrmse))
        print('  max_diff = {:.3f}'.format(max_diff))
        print('  pct_95 = {:.3f}'.format(pct_95))
        
        # Verify that the computed stats are within tolerances
        self.assertTrue(max_diff < tolerances['max_diff'], f"Max difference too high: {max_diff}")
        self.assertTrue(nrmse < tolerances['nrmse'], f"NRMSE too high: {nrmse}")
        self.assertTrue(pct_95 < tolerances['pct_95'], f"95th percentile difference too high: {pct_95}")

if __name__ == '__main__':
    unittest.main()