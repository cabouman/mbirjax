import tempfile
import unittest
import warnings
import numpy as np
import jax
import jax.numpy as jnp
import mbirjax as mj


class TestVCD(unittest.TestCase):
    """
    Unit tests for verifying the reconstruction accuracy of the VCD algorithm in MBIRJAX.
    """

    def setUp(self):
        """Set up before each test method."""
        np.random.seed(0)  # Set a seed to avoid variations due to partition creation.
        # Choose the geometry type
        self.geometry_types = mj._utils._geometry_types_for_tests
        parallel_tolerances = {'nrmse': 0.15, 'max_diff': 0.38, 'pct_95': 0.04}
        cone_tolerances = {'nrmse': 0.19, 'max_diff': 0.56, 'pct_95': 0.05}
        translation_tolerances = {'nrmse': 0.8, 'max_diff': 1.05, 'pct_95': 0.03}
        self.all_tolerances = [parallel_tolerances, cone_tolerances, translation_tolerances]
        if len(self.geometry_types) != len(self.all_tolerances):
            raise IndexError('The list of geometry types does not match the list of test tolerances for the geometry types.')

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
        self.translation_vectors = None

    def tearDown(self):
        """Clean up after each test method."""
        pass

    def set_view_params(self, geometry_type):
        if geometry_type == 'cone':
            detector_cone_angle = 2 * np.arctan2(self.num_det_channels / 2, self.source_detector_dist)
        else:
            detector_cone_angle = 0
        start_angle = -(np.pi + detector_cone_angle) * (1 / 2)
        end_angle = (np.pi + detector_cone_angle) * (1 / 2)
        self.angles = jnp.linspace(start_angle, end_angle, self.num_views, endpoint=False)

        num_x_translations = 7
        num_z_translations = 7
        x_spacing = 22
        z_spacing = 22

        # Generate translation vectors
        translation_vectors = mj.gen_translation_vectors(num_x_translations, num_z_translations, x_spacing, z_spacing)
        self.translation_vectors = translation_vectors

    def get_model(self, geometry_type):
        if geometry_type == 'parallel':
            ct_model = mj.ParallelBeamModel(self.sinogram_shape, self.angles)
        elif geometry_type == 'cone':
            ct_model = mj.ConeBeamModel(self.sinogram_shape, self.angles,
                                             source_detector_dist=self.source_detector_dist,
                                             source_iso_dist=self.source_iso_dist)
        elif geometry_type == 'translation':
            ct_model = mj.TranslationModel(self.sinogram_shape, self.translation_vectors,
                                                source_detector_dist=self.source_detector_dist,
                                                source_iso_dist=self.source_iso_dist)
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
        self.set_view_params(geometry_type)
        ct_model = self.get_model(geometry_type)

        # Generate 3D Shepp Logan phantom
        print('  Creating phantom')

        if geometry_type == 'translation':
            recon_shape = ct_model.get_params('recon_shape')
            words = ["Purdue", "Presents", "Translation", "Tomography"]
            phantom = mj.gen_translation_phantom(recon_shape=recon_shape, option='text', words=words)
        else:
            phantom = ct_model.gen_modified_3d_sl_phantom()

        # Generate synthetic sinogram data
        print('  Creating sinogram')
        sinogram = ct_model.forward_project(phantom)

        # Set reconstruction parameter values
        ct_model.set_params(verbose=0)

        # ##########################
        # Perform VCD reconstruction
        sinogram = jax.device_put(sinogram, ct_model.main_device)
        print('  Starting recon')
        recon, recon_dict = ct_model.recon(sinogram)
        recon.block_until_ready()
        max_diff = np.amax(np.abs(phantom - recon))
        nrmse = np.linalg.norm(recon - phantom) / np.linalg.norm(phantom)
        pct_95 = np.percentile(np.abs(recon - phantom), 95)
        print('  nrmse = {:.3f}'.format(nrmse))
        print('  max_diff = {:.3f}'.format(max_diff))
        print('  pct_95 = {:.3f}'.format(pct_95))

        self.assertTrue(max_diff < tolerances['max_diff'] and
                        nrmse < tolerances['nrmse'] and
                        pct_95 < tolerances['pct_95'])

        print('  Testing hdf5 save and load')
        notes = "Testing save/load"
        with tempfile.NamedTemporaryFile('w') as file:
            filepath = file.name
            ct_model.save_recon_hdf5(filepath, recon, recon_dict)
            loaded_recon, loaded_recon_dict, new_model = mj.TomographyModel.load_recon_hdf5(str(filepath),
                                                                                                 recreate_model=True)
            loaded_notes = loaded_recon_dict['notes']

            assert np.allclose(recon, loaded_recon)
            assert np.allclose(ct_model.get_params('sigma_x'), new_model.get_params('sigma_x')) # just one representative parameter
            assert recon_dict['notes'] == loaded_notes


if __name__ == '__main__':
    unittest.main()
