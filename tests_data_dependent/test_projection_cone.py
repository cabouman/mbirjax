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
    Unit tests for verifying the projection accuracy of all beam geometries in MBIRJAX.
    """

    phantom_filepath = None
    sinogram_filepath = None
    recon_filepath = None
    params_filepath = None

    @classmethod
    def setUpClass(cls):
        # Runs once before all tests
        output_directory = f"/depot/bouman/users/ncardel"

        cls.phantom_filepath = output_directory + '/cone_phantom_32.npy'
        cls.sinogram_filepath = output_directory + '/cone_sinogram_32.npy'
        cls.recon_filepath = output_directory + '/cone_recon_32.npy'
        cls.params_filepath = output_directory + '/cone_params_32.pkl'

    @classmethod
    def tearDownClass(cls):
        # Runs once after all tests
        print("Cleaning up resources...")

    def setUp(self):
        """Set up before each test method."""
        self.phantom = np.load(self.phantom_filepath)
        self.sinogram = np.load(self.sinogram_filepath)
        self.recon = np.load(self.recon_filepath)
        with open(self.params_filepath, "rb") as f:
            self.params = pickle.load(f)

    def tearDown(self):
        """Clean up after each test method."""
        pass

    def test_sparse_forward_project(self):
        # params
        angles = self.params['angles']
        source_detector_dist = self.params['source_detector_dist']
        source_iso_dist = self.params['source_iso_dist']

        # create back projection model
        forward_projection_model = mj.ConeBeamModel(self.sinogram.shape, angles,
                                                 source_detector_dist=source_detector_dist,
                                                 source_iso_dist=source_iso_dist)

        # perform forward projection and make into sinogram
        forward_projection = forward_projection_model.forward_project(self.phantom)
        forward_projection.block_until_ready()
        sinogram = jax.device_put(forward_projection)  # move to host

        # compare with control recon
        assert jnp.allclose(sinogram, self.sinogram, atol=1e-03)

    def test_sparse_back_project(self):
        # params
        angles = self.params['angles']
        source_detector_dist = self.params['source_detector_dist']
        source_iso_dist = self.params['source_iso_dist']

        # create back projection model
        back_projection_model = mj.ConeBeamModel(self.sinogram.shape, angles,
                                                 source_detector_dist=source_detector_dist,
                                                 source_iso_dist=source_iso_dist)

        # get recon shape and partition pixel indices
        recon_shape, granularity = back_projection_model.get_params(['recon_shape', 'granularity'])
        recon_rows, recon_cols, recon_slices = recon_shape
        partitions = mj.gen_set_of_pixel_partitions(recon_shape, granularity)
        pixel_indices = partitions[0][0]

        # perform back projection and reshape into recon
        back_projection = back_projection_model.sparse_back_project(self.sinogram, pixel_indices)


        back_projection.block_until_ready()
        recon = jnp.zeros((recon_rows * recon_cols, recon_slices)).at[pixel_indices].add(back_projection)
        recon = back_projection_model.reshape_recon(recon)
        recon = jax.device_put(recon)  # move to host

        # compare with control recon
        assert jnp.allclose(recon, self.recon, atol=1e-03)


if __name__ == '__main__':
    unittest.main()
