import numpy as np
import jax
import jax.numpy as jnp
import mbirjax
import unittest


class TestProjectors(unittest.TestCase):
    """
    Test the adjoint property of the forward and back projectors, both the full versions and the sparse voxel version.
    This means if x is an image, and y is a sinogram, then <y, Ax> = <Aty, x>.
    The code below verifies this for the full forward and back projectors and the versions that specify a
    subset of voxels in x.
    This code also verifies that first applying the full back projector and selecting the voxels to get (Aty)[ss]
    is the same as using the subset back projector with the specified set of voxels.
    """

    def setUp(self):
        """Set up before each test method."""
        np.random.seed(0)  # Set a seed to avoid variations due to partition creation.
        # Choose the geometry type
        self.geometry_types = ['parallel', 'cone']
        parallel_tolerances = {'nrmse': 0.15, 'max_diff': 0.38, 'pct_95': 0.04}
        cone_tolerances = {'nrmse': 0.19, 'max_diff': 0.56, 'pct_95': 0.05}
        self.all_tolerances = [parallel_tolerances, cone_tolerances]

        # Set parameters
        self.num_views = 64
        self.num_det_rows = 40
        self.num_det_channels = 128
        self.sharpness = 0.0

        # These can be adjusted to describe the geometry in the cone beam case.
        # np.Inf is an allowable value, in which case this is essentially parallel beam
        self.source_detector_dist = 4 * self.num_det_channels
        self.source_iso_dist = self.source_detector_dist

        # Initialize sinogram
        self.sinogram_shape = (self.num_views, self.num_det_rows, self.num_det_channels)
        self.angles = None

    def tearDown(self):
        """Clean up after each test method."""
        pass

    def set_angles(self, geometry_type):
        if geometry_type == 'cone':
            detector_cone_angle = 2 * np.arctan2(self.num_det_channels / 2, self.source_detector_dist)
        else:
            detector_cone_angle = 0
        start_angle = -(np.pi + detector_cone_angle) * (1 / 2)
        end_angle = (np.pi + detector_cone_angle) * (1 / 2)
        self.angles = jnp.linspace(start_angle, end_angle, self.num_views, endpoint=False)

    def get_model(self, geometry_type):
        if geometry_type == 'cone':
            ct_model = mbirjax.ConeBeamModel(self.sinogram_shape, self.angles,
                                             source_detector_dist=self.source_detector_dist,
                                             source_iso_dist=self.source_iso_dist)
        elif geometry_type == 'parallel':
            ct_model = mbirjax.ParallelBeamModel(self.sinogram_shape, self.angles)
        else:
            raise ValueError('Invalid geometry type.  Expected cone or parallel, got {}'.format(geometry_type))

        return ct_model

    def test_all_vcd(self):
        for geometry_type, tolerances in zip(self.geometry_types, self.all_tolerances):
            with self.subTest(geometry_type=geometry_type):
                print("Testing vcd with", geometry_type)
                self.verify_vcd(geometry_type, tolerances)

    def verify_vcd(self, geometry_type, tolerances):
        """
        Verify that the vcd reconstructions for a simple phantom are within tolerance
        """
        self.set_angles(geometry_type)
        ct_model = self.get_model(geometry_type)

        # Generate 3D Shepp Logan phantom
        print('  Creating phantom')
        phantom = ct_model.gen_modified_3d_sl_phantom()

        # Generate synthetic sinogram data
        print('  Creating sinogram')
        sinogram = ct_model.forward_project(phantom)

        # Set reconstruction parameter values
        ct_model.set_params(verbose=0)

        # ##########################
        # Perform VCD reconstruction
        print('  Starting recon')
        recon, recon_params = ct_model.recon(sinogram)
        recon.block_until_ready()

        max_diff = np.amax(np.abs(phantom - recon))
        nrmse = np.linalg.norm(recon - phantom) / np.linalg.norm(phantom)
        pct_95 = np.percentile(np.abs(recon - phantom), 95)
        print('  nrmse = {:.3f}'.format(nrmse))

        self.assertTrue(max_diff < tolerances['max_diff'] and
                        nrmse < tolerances['nrmse'] and
                        pct_95 < tolerances['pct_95'])


if __name__ == '__main__':
    unittest.main()
