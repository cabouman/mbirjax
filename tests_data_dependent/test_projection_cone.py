import os
import shutil
import unittest
import h5py
import jax
import jax.numpy as jnp
import mbirjax as mj


class TestProjectionCone(unittest.TestCase):
    """
    Unit tests for verifying the projection accuracy
    of the cone-beam model in MBIRJAX.
    """

    # Test parameters
    USE_GPU_OPTS = ["automatic", "full", "sinograms", "projections", "none"]
    ATOL = 1e-3

    # Test data locations
    SOURCE_FILEPATH = "/depot/bouman/users/ncardel/cone_32_data.tar.gz"
    TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")
    DATA_FILEPATH = None

    @classmethod
    def setUpClass(cls):
        """Run once before all tests: fetch and unpack test data into a temp dir."""
        if os.path.exists(cls.TMP_DIR):
            shutil.rmtree(cls.TMP_DIR)
        cls.DATA_FILEPATH = mj.download_and_extract(cls.SOURCE_FILEPATH, cls.TMP_DIR)

    @classmethod
    def tearDownClass(cls):
        """Run once after all tests: remove the temp dir."""
        if os.path.exists(cls.TMP_DIR):
            shutil.rmtree(cls.TMP_DIR)

    def setUp(self):
        """Run before each test: load control arrays/params and (re)create the model."""
        with h5py.File(self.DATA_FILEPATH, "r") as f:
            self.control_phantom = f["phantom"][:]
            self.control_sinogram = f["sinogram"][:]
            self.control_recon = f["recon"][:]
            self.control_params = f.attrs["params"]

        # Recreate a fresh projection model for each test
        self.projection_model = mj.ConeBeamModel.from_file(self.control_params)

    def tearDown(self):
        """Run after each test (no-op placeholder for now)."""
        pass

    # ---------- Forward projection tests ----------

    def test_sparse_forward_project(self):
        """Forward-project the control phantom and compare against control sinogram."""
        for opt in self.USE_GPU_OPTS:
            with self.subTest(use_gpu=opt):
                self.projection_model.set_params(use_gpu=opt)
                sinogram = self.projection_model.forward_project(self.control_phantom)
                sinogram = jax.device_put(sinogram)  # ensure array is materialized on device
                self.assertTrue(
                    jnp.allclose(sinogram, self.control_sinogram, atol=self.ATOL),
                    msg=f"forward mismatch with use_gpu={opt}",
                )

    def test_sparse_forward_project_rejects_biased_input_at_tol(self):
        """
        Adding a small bias to the phantom should change the sinogram enough
        to break equality at the chosen tolerance.
        """
        self.projection_model.set_params(use_gpu="automatic")
        sinogram = self.projection_model.forward_project(self.control_phantom + 1e-3)
        sinogram = jax.device_put(sinogram)  # ensure array is materialized on device
        self.assertFalse(
            jnp.allclose(sinogram, self.control_sinogram, atol=self.ATOL),
            msg="forward unexpectedly allclose with biased input",
        )

    def test_forward_project_zero_tolerance_not_equal(self):
        """
        With zero tolerances, exact equality is (intentionally) too strict.
        We expect inequality due to floating-point roundoff.
        """
        self.projection_model.set_params(use_gpu="automatic")
        s = self.projection_model.forward_project(self.control_phantom)
        s = jax.device_put(s)
        self.assertFalse(
            jnp.allclose(s, self.control_sinogram, rtol=0.0, atol=0.0),
            msg="forward unexpectedly equal at zero tolerance",
        )

    # ---------- Back-projection tests ----------

    def test_sparse_back_project(self):
        """Back-project the control sinogram and compare against control reconstruction."""
        for opt in self.USE_GPU_OPTS:
            with self.subTest(use_gpu=opt):
                self.projection_model.set_params(use_gpu=opt)
                recon = self.projection_model.back_project(self.control_sinogram)
                recon = jax.device_put(recon)  # ensure array is materialized on device
                self.assertTrue(
                    jnp.allclose(recon, self.control_recon, atol=self.ATOL),
                    msg=f"back-projection mismatch with use_gpu={opt}",
                )

    def test_sparse_back_project_rejects_biased_input_at_tol(self):
        """
        Adding a small bias to the sinogram should change the reconstruction
        enough to break equality at the chosen tolerance.
        """
        self.projection_model.set_params(use_gpu="automatic")
        recon = self.projection_model.back_project(self.control_sinogram + 1e-3)
        recon = jax.device_put(recon)  # ensure array is materialized on device
        self.assertFalse(
            jnp.allclose(recon, self.control_recon, atol=self.ATOL),
            msg="back-projection unexpectedly allclose with biased input",
        )

    def test_back_project_zero_tolerance_not_equal(self):
        """
        With zero tolerances, exact equality is (intentionally) too strict.
        We expect inequality due to floating-point roundoff.
        """
        self.projection_model.set_params(use_gpu="automatic")
        r = self.projection_model.back_project(self.control_sinogram)
        r = jax.device_put(r)
        self.assertFalse(
            jnp.allclose(r, self.control_recon, rtol=0.0, atol=0.0),
            msg="back-projection unexpectedly equal at zero tolerance",
        )


if __name__ == "__main__":
    unittest.main()
