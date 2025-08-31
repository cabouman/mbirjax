# test_projection.py
import os, shutil, unittest, h5py, jax, jax.numpy as jnp
import mbirjax as mj

class ProjectionTestBase(unittest.TestCase):
    """
    Reusable test suite for a projection model. Subclasses must set:
      - SOURCE_FILEPATH (str)
      - MODEL (class with .from_file)
    """
    USE_GPU_OPTS = ["automatic", "full", "sinograms", "projections", "none"]
    ATOL = 1e-3

    # To be overridden in subclasses:
    SOURCE_FILEPATH = None
    MODEL = None

    TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")
    DATA_FILEPATH = None

    @classmethod
    def setUpClass(cls):

        # Don't run the abstract/base suite directly
        if cls is ProjectionTestBase:
            raise unittest.SkipTest("ProjectionTestBase is a base class; skipping.")

        assert cls.SOURCE_FILEPATH and cls.MODEL, \
            "Subclasses must define MODEL, SOURCE_FILEPATH"

        if os.path.exists(cls.TMP_DIR):
            shutil.rmtree(cls.TMP_DIR)
        cls.DATA_FILEPATH = mj.download_and_extract(cls.SOURCE_FILEPATH, cls.TMP_DIR)

    @classmethod
    def tearDownClass(cls):
        if cls.TMP_DIR and os.path.exists(cls.TMP_DIR):
            shutil.rmtree(cls.TMP_DIR)

    def setUp(self):
        with h5py.File(self.DATA_FILEPATH, "r") as f:
            self.control_phantom = f["phantom"][:]
            self.control_sinogram = f["sinogram"][:]
            self.control_recon = f["recon"][:]
            self.control_params = f.attrs["params"]
        self.projection_model = self.MODEL.from_file(self.control_params)

    # ---------- Forward projection tests ----------

    def test_sparse_forward_project(self):
        """Forward-project the control phantom and compare against control sinogram."""
        for opt in self.USE_GPU_OPTS:
            with self.subTest(geometry=self.MODEL.__name__, use_gpu=opt):
                self.projection_model.set_params(use_gpu=opt)
                sinogram = self.projection_model.forward_project(self.control_phantom)
                sinogram = jax.device_put(sinogram)
                self.assertTrue(
                    jnp.allclose(sinogram, self.control_sinogram, atol=self.ATOL),
                    msg=f"[{self.MODEL.__name__}] forward mismatch (use_gpu={opt})",
                )

    def test_sparse_forward_project_rejects_biased_input_at_tol(self):
        """
        Adding a small bias to the phantom should change the sinogram enough
        to break equality at the chosen tolerance.
        """
        self.projection_model.set_params(use_gpu="automatic")
        sinogram = self.projection_model.forward_project(self.control_phantom + 1e-3)
        sinogram = jax.device_put(sinogram)
        self.assertFalse(
            jnp.allclose(sinogram, self.control_sinogram, atol=self.ATOL),
            msg=f"[{self.MODEL.__name__}] forward unexpectedly allclose with biased input",
        )

    def test_forward_project_zero_tolerance_not_equal(self):
        """
        With zero tolerances, exact equality is (intentionally) too strict.
        We expect inequality due to floating-point roundoff.
        """
        self.projection_model.set_params(use_gpu="automatic")
        sinogram = self.projection_model.forward_project(self.control_phantom)
        sinogram = jax.device_put(sinogram)
        self.assertFalse(
            jnp.allclose(sinogram, self.control_sinogram, rtol=0.0, atol=0.0),
            msg=f"[{self.MODEL.__name__}] forward unexpectedly equal at zero tol",
        )

    # ---------- Back-projection tests ----------

    def test_sparse_back_project(self):
        """Back-project the control sinogram and compare against control reconstruction."""
        for opt in self.USE_GPU_OPTS:
            with self.subTest(geometry=self.MODEL.__name__, use_gpu=opt):
                self.projection_model.set_params(use_gpu=opt)
                recon = self.projection_model.back_project(self.control_sinogram)
                recon = jax.device_put(recon)
                self.assertTrue(
                    jnp.allclose(recon, self.control_recon, atol=self.ATOL),
                    msg=f"[{self.MODEL.__name__}] back-projection mismatch (use_gpu={opt})",
                )

    def test_sparse_back_project_rejects_biased_input_at_tol(self):
        """
        Adding a small bias to the sinogram should change the reconstruction
        enough to break equality at the chosen tolerance.
        """
        self.projection_model.set_params(use_gpu="automatic")
        recon = self.projection_model.back_project(self.control_sinogram + 1e-3)
        recon = jax.device_put(recon)
        self.assertFalse(
            jnp.allclose(recon, self.control_recon, atol=self.ATOL),
            msg=f"[{self.MODEL.__name__}] back-projection unexpectedly allclose with biased input",
        )

    def test_back_project_zero_tolerance_not_equal(self):
        """
        With zero tolerances, exact equality is (intentionally) too strict.
        We expect inequality due to floating-point roundoff.
        """
        self.projection_model.set_params(use_gpu="automatic")
        recon = self.projection_model.back_project(self.control_sinogram)
        recon = jax.device_put(recon)
        self.assertFalse(
            jnp.allclose(recon, self.control_recon, rtol=0.0, atol=0.0),
            msg=f"[{self.MODEL.__name__}] back-projection unexpectedly equal at zero tol",
        )

# ---- Concrete geometry variants ----

class TestProjectionCone(ProjectionTestBase):
    MODEL = mj.ConeBeamModel
    SOURCE_FILEPATH = "/depot/bouman/users/ncardel/cone_32_data.tar.gz"

class TestProjectionParallel(ProjectionTestBase):
    MODEL = mj.ParallelBeamModel
    SOURCE_FILEPATH = "/depot/bouman/users/ncardel/parallel_32_data.tar.gz"

if __name__ == "__main__":
    unittest.main()
