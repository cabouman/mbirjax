import os
import shutil
import unittest
import h5py
import jax
import jax.numpy as jnp
import mbirjax as mj


class TestProjectionCone(unittest.TestCase):
    """
    Unit tests for verifying the projection accuracy of the cone beam model in MBIRJAX.
    """
    USE_GPU_OPTS = ["automatic", "full", "sinograms", "projections", "none"]

    SOURCE_FILEPATH = f"/depot/bouman/users/ncardel/cone_32_data.tar.gz"
    TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")
    DATA_FILEPATH = None

    @classmethod
    def setUpClass(cls):
        """Set up once before all tests."""
        if os.path.exists(cls.TMP_DIR):
            shutil.rmtree(cls.TMP_DIR)
        cls.DATA_FILEPATH = mj.download_and_extract(cls.SOURCE_FILEPATH, cls.TMP_DIR)

    @classmethod
    def tearDownClass(cls):
        """Clean up once before all tests."""
        if os.path.exists(cls.TMP_DIR):
            shutil.rmtree(cls.TMP_DIR)

    def setUp(self):
        """Set up before each test method."""

        # reload the arrays and params
        with h5py.File(self.DATA_FILEPATH, "r") as f:
            self.control_phantom = f["phantom"][:]
            self.control_sinogram = f["sinogram"][:]
            self.control_recon = f["recon"][:]
            self.control_params = f.attrs["params"]

        # recreate projection model
        self.projection_model = mj.ConeBeamModel.from_file(self.control_params)

    def tearDown(self):
        """Clean up after each test method."""
        pass

    def test_sparse_forward_project(self):
        for opt in self.USE_GPU_OPTS:
            with self.subTest(use_gpu=opt):
                self.projection_model.set_params(use_gpu=opt)
                sinogram = self.projection_model.forward_project(self.control_phantom)
                sinogram = jax.device_put(sinogram)  # move to host

                # compare with control sinogram
                assert jnp.allclose(sinogram, self.control_sinogram, atol=1e-03)

    def test_sparse_forward_project_rejects_biased_input_at_tol(self):
        self.projection_model.set_params(use_gpu='automatic')
        sinogram = self.projection_model.forward_project(self.control_phantom + 1e-03)
        sinogram = jax.device_put(sinogram)  # move to host

        # compare with control sinogram
        assert not jnp.allclose(sinogram, self.control_sinogram, atol=1e-03)

    def test_forward_project_zero_tolerance_expected_failure(self):
        """Even tiny FP roundoff should break equality at zero tolerance."""
        self.projection_model.set_params(use_gpu='automatic')
        s = self.projection_model.forward_project(self.control_phantom)
        s = jax.device_put(s)
        self.assertFalse(jnp.allclose(s, self.control_sinogram, rtol=0.0, atol=0.0))

    def test_sparse_back_project(self):
        for opt in self.USE_GPU_OPTS:
            with self.subTest(use_gpu=opt):
                self.projection_model.set_params(use_gpu=opt)
                recon = self.projection_model.back_project(self.control_sinogram)
                recon = jax.device_put(recon)  # move to host

                # compare with control recon
                assert jnp.allclose(recon, self.control_recon, atol=1e-03)

    def test_sparse_back_project_rejects_biased_input_at_tol(self):
        self.projection_model.set_params(use_gpu='automatic')
        recon = self.projection_model.back_project(self.control_sinogram + 1e-03)
        recon = jax.device_put(recon)  # move to host

        # compare with control recon
        assert not jnp.allclose(recon, self.control_recon, atol=1e-03)

    def test_back_project_zero_tolerance_expected_failure(self):
        """Back-projection equality at zero tolerance (intentionally too strict)."""
        self.projection_model.set_params(use_gpu='automatic')
        r = self.projection_model.back_project(self.control_sinogram)
        r = jax.device_put(r)
        self.assertFalse(jnp.allclose(r, self.control_recon, rtol=0.0, atol=0.0))


if __name__ == '__main__':
    unittest.main()
