import os
import unittest
import h5py
import jax
import jax.numpy as jnp
import mbirjax as mj

import shutil

# use_gpu: 'automatic', 'full', 'sinograms', 'projections', 'none'
# geometry: 'cone', 'parallel'
# direction: forward, back

class TestProjectionCone(unittest.TestCase):
    """
    Unit tests for verifying the projection accuracy of the cone beam model in MBIRJAX.
    """

    source_filepath = f"/depot/bouman/users/ncardel/cone_32_data.h5"
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    dest_filepath = None

    @classmethod
    def setUpClass(cls):
        """Set up once before all tests."""

        if not os.path.exists(cls.data_dir):
            os.makedirs(cls.data_dir)

        # TODO: download and untar
        filename = os.path.basename(cls.source_filepath)
        cls.dest_filepath = f"{cls.data_dir}/{filename}"
        shutil.copy(cls.source_filepath, cls.dest_filepath)

    @classmethod
    def tearDownClass(cls):
        """Clean up once before all tests."""
        if os.path.exists(cls.dest_filepath):
            os.remove(cls.dest_filepath)

    def setUp(self):
        """Set up before each test method."""

        with h5py.File(self.dest_filepath, "r") as f:
            self.control_phantom = f["phantom"][:]
            self.control_sinogram = f["sinogram"][:]
            self.control_recon = f["recon"][:]
            self.control_params = f.attrs["params"]

        # create projection model
        self.projection_model = mj.ConeBeamModel.from_file(self.control_params)

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
