import os
import shutil
import unittest
import h5py
import jax
import jax.numpy as jnp
import mbirjax as mj


# use_gpu: 'automatic', 'full', 'sinograms', 'projections', 'none'

class TestProjectionCone(unittest.TestCase):
    """
    Unit tests for verifying the projection accuracy of the cone beam model in MBIRJAX.
    """

    source_filepath = f"/depot/bouman/users/ncardel/cone_32_data.tar.gz"
    tmp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")
    dest_filepath = None

    @classmethod
    def setUpClass(cls):
        """Set up once before all tests."""
        if os.path.exists(cls.tmp_dir):
            shutil.rmtree(cls.tmp_dir)
        cls.dest_filepath = mj.download_and_extract(cls.source_filepath, cls.tmp_dir)

    @classmethod
    def tearDownClass(cls):
        """Clean up once before all tests."""
        if os.path.exists(cls.tmp_dir):
            shutil.rmtree(cls.tmp_dir)

    def setUp(self):
        """Set up before each test method."""

        # reload the arrays and params
        with h5py.File(self.dest_filepath, "r") as f:
            self.control_phantom = f["phantom"][:]
            self.control_sinogram = f["sinogram"][:]
            self.control_recon = f["recon"][:]
            self.control_params = f.attrs["params"]

        # recreate projection model
        self.projection_model = mj.ConeBeamModel.from_file(self.control_params)

    def tearDown(self):
        """Clean up after each test method."""
        pass

    def test_sparse_forward_project_automatic(self):

        # perform forward projection on phantom and make into sinogram
        self.projection_model.set_params(use_gpu='automatic')
        sinogram = self.projection_model.forward_project(self.control_phantom)
        sinogram = jax.device_put(sinogram)  # move to host

        # compare with control sinogram
        assert jnp.allclose(sinogram, self.control_sinogram, atol=1e-03)

    def test_sparse_forward_project_full(self):

        # perform forward projection on phantom and make into sinogram
        self.projection_model.set_params(use_gpu='full')
        sinogram = self.projection_model.forward_project(self.control_phantom)
        sinogram = jax.device_put(sinogram)  # move to host

        # compare with control sinogram
        assert jnp.allclose(sinogram, self.control_sinogram, atol=1e-03)

    def test_sparse_forward_project_sinograms(self):

        # perform forward projection on phantom and make into sinogram
        self.projection_model.set_params(use_gpu='sinograms')
        sinogram = self.projection_model.forward_project(self.control_phantom)
        sinogram = jax.device_put(sinogram)  # move to host

        # compare with control sinogram
        assert jnp.allclose(sinogram, self.control_sinogram, atol=1e-03)

    def test_sparse_forward_project_projections(self):
        # perform forward projection on phantom and make into sinogram
        self.projection_model.set_params(use_gpu='projections')
        sinogram = self.projection_model.forward_project(self.control_phantom)
        sinogram = jax.device_put(sinogram)  # move to host

        # compare with control sinogram
        assert jnp.allclose(sinogram, self.control_sinogram, atol=1e-03)

    def test_sparse_forward_project_none(self):
        # perform forward projection on phantom and make into sinogram
        self.projection_model.set_params(use_gpu='none')
        sinogram = self.projection_model.forward_project(self.control_phantom)
        sinogram = jax.device_put(sinogram)  # move to host

        # compare with control sinogram
        assert jnp.allclose(sinogram, self.control_sinogram, atol=1e-03)

    def test_sparse_back_project_automatic(self):

        # perform back projection on sinogram and make into recon
        self.projection_model.set_params(use_gpu='automatic')
        recon = self.projection_model.back_project(self.control_sinogram)
        recon = jax.device_put(recon)  # move to host

        # compare with control recon
        assert jnp.allclose(recon, self.control_recon, atol=1e-03)

    def test_sparse_back_project_full(self):

        # perform back projection on sinogram and make into recon
        self.projection_model.set_params(use_gpu='full')
        recon = self.projection_model.back_project(self.control_sinogram)
        recon = jax.device_put(recon)  # move to host

        # compare with control recon
        assert jnp.allclose(recon, self.control_recon, atol=1e-03)

    def test_sparse_back_project_sinograms(self):

        # perform back projection on sinogram and make into recon
        self.projection_model.set_params(use_gpu='sinograms')
        recon = self.projection_model.back_project(self.control_sinogram)
        recon = jax.device_put(recon)  # move to host

        # compare with control recon
        assert jnp.allclose(recon, self.control_recon, atol=1e-03)

    def test_sparse_back_project_projections(self):

        # perform back projection on sinogram and make into recon
        self.projection_model.set_params(use_gpu='projections')
        recon = self.projection_model.back_project(self.control_sinogram)
        recon = jax.device_put(recon)  # move to host

        # compare with control recon
        assert jnp.allclose(recon, self.control_recon, atol=1e-03)

    def test_sparse_back_project_none(self):

        # perform back projection on sinogram and make into recon
        self.projection_model.set_params(use_gpu='none')
        recon = self.projection_model.back_project(self.control_sinogram)
        recon = jax.device_put(recon)  # move to host

        # compare with control recon
        assert jnp.allclose(recon, self.control_recon, atol=1e-03)


# TODO: create some tests that should fail

if __name__ == '__main__':
    unittest.main()
