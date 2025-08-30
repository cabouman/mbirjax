import unittest
import numpy as np
import jax
import jax.numpy as jnp
import mbirjax as mj
import pickle

# use_gpu: 'automatic', 'full', 'sinograms', 'projections', 'none'
# geometry: 'cone', 'parallel'
# direction: forward, back

class TestProjectionCone(unittest.TestCase):
    """
    Unit tests for verifying the projection accuracy of the cone beam model in MBIRJAX.
    """

    control_phantom_filepath = None
    control_sinogram_filepath = None
    control_recon_filepath = None
    control_params_filepath = None

    @classmethod
    def setUpClass(cls):
        """Set up once before all tests."""
        # TODO: create a ./data dir and download data into it from depot

        output_directory = f"/depot/bouman/users/ncardel"

        cls.control_phantom_filepath = output_directory + '/cone_phantom_32.npy'
        cls.control_sinogram_filepath = output_directory + '/cone_sinogram_32.npy'
        cls.control_recon_filepath = output_directory + '/cone_recon_32.npy'
        cls.control_params_filepath = output_directory + '/cone_params_32.pkl'

    @classmethod
    def tearDownClass(cls):
        """Clean up once before all tests."""
        # TODO: delete files that were downloaded
        pass

    def setUp(self):
        """Set up before each test method."""
        # load phantom, sinogram, and recon
        self.control_phantom = np.load(self.control_phantom_filepath)
        self.control_sinogram = np.load(self.control_sinogram_filepath)
        self.control_recon = np.load(self.control_recon_filepath)

        # params
        with open(self.control_params_filepath, "rb") as f:
            self.control_params = pickle.load(f)
        angles = self.control_params['angles']
        source_detector_dist = self.control_params['source_detector_dist']
        source_iso_dist = self.control_params['source_iso_dist']

        # create projection model
        self.projection_model = mj.ConeBeamModel(self.control_sinogram.shape, angles,
                                                      source_detector_dist=source_detector_dist,
                                                      source_iso_dist=source_iso_dist)

    def tearDown(self):
        """Clean up after each test method."""
        pass

    def test_sparse_forward_project(self):

        # perform forward projection on phantom and make into sinogram
        sinogram = self.projection_model.forward_project(self.control_phantom)
        sinogram = jax.device_put(sinogram)  # move to host

        # compare with control sinogram
        assert jnp.allclose(sinogram, self.control_sinogram, atol=1e-03)

    def test_sparse_back_project(self):

        # perform back projection on sinogram and make into recon
        recon = self.projection_model.back_project(self.control_sinogram)
        recon = jax.device_put(recon)  # move to host

        # compare with control recon
        assert jnp.allclose(recon, self.control_recon, atol=1e-03)


if __name__ == '__main__':
    unittest.main()
