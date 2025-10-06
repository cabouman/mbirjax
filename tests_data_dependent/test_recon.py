# test_recon.py
import os, shutil, pytest, pathlib, unittest, h5py, jax, jax.numpy as jnp
import mbirjax as mj

@pytest.mark.data_dependent
class ReconTestBase(unittest.TestCase):
    """
    Reusable test suite for a projection model. Subclasses must set:
      - MODEL (class with .from_file)
      - SOURCE_FILEPATH (str)
    """
    USE_GPU_OPTS = ["automatic", "full", "sinograms", "projections", "none"]
    ATOL = 1e-3

    # To be overridden in subclasses:
    MODEL = None
    SOURCE_FILEPATH = None
    TOLERANCES = None

    TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")
    DATA_FILEPATH = None

    @classmethod
    def setUpClass(cls):

        # Don't run the abstract/base suite directly
        if cls is ReconTestBase:
            raise unittest.SkipTest("ReconTestBase is a base class; skipping.")

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

    # ---------- Recon tests ----------

    def test_recon(self):
        """Back-project the control sinogram and compare against control reconstruction."""
        for opt in self.USE_GPU_OPTS:
            with self.subTest(geometry=self.MODEL.__name__, use_gpu=opt):
                self.projection_model.set_params(use_gpu=opt)
                filename = pathlib.Path(self.SOURCE_FILEPATH).stem
                recon, _ = self.projection_model.recon(self.control_sinogram,
                                                       max_iterations=15,
                                                       stop_threshold_change_pct=0,
                                                       logfile_path=f'./logs/recon_{filename}_{opt}.log')
                recon.block_until_ready()
                recon = jax.device_put(recon)

                max_diff = jnp.amax(jnp.abs(self.control_recon - recon))
                nrmse = jnp.linalg.norm(recon - self.control_recon) / jnp.linalg.norm(self.control_recon)
                pct_95 = jnp.percentile(jnp.abs(recon - self.control_recon), 95)

                self.assertTrue(max_diff < self.TOLERANCES['max_diff'] and
                                nrmse < self.TOLERANCES['nrmse'] and
                                pct_95 < self.TOLERANCES['pct_95'],
                                msg=f"[{self.MODEL.__name__}] recon mismatch (use_gpu={opt})",
                )

    def test_recon_biased_input_at_tol(self):
        """
        Adding a small bias to the sinogram should change the reconstruction
        enough to break equality at the chosen tolerance.
        """
        self.projection_model.set_params(use_gpu="automatic")
        recon, _ = self.projection_model.recon(self.control_sinogram + 1,
                                               max_iterations=15,
                                               stop_threshold_change_pct=0)
        recon.block_until_ready()
        recon = jax.device_put(recon)

        max_diff = jnp.amax(jnp.abs(self.control_recon - recon))
        nrmse = jnp.linalg.norm(recon - self.control_recon) / jnp.linalg.norm(self.control_recon)
        pct_95 = jnp.percentile(jnp.abs(recon - self.control_recon), 95)

        self.assertFalse(max_diff < self.TOLERANCES['max_diff'] and
                        nrmse < self.TOLERANCES['nrmse'] and
                        pct_95 < self.TOLERANCES['pct_95'],
                        msg=f"[{self.MODEL.__name__}] recon unexpectedly allclose with biased input",
        )

    def test_recon_zero_tolerance_not_equal(self):
        """
        With zero tolerances, exact equality is (intentionally) too strict.
        We expect inequality due to floating-point roundoff.
        """
        self.projection_model.set_params(use_gpu="automatic")
        recon, _ = self.projection_model.recon(self.control_sinogram,
                                               max_iterations=15,
                                               stop_threshold_change_pct=0)
        recon = jax.device_put(recon)
        self.assertFalse(
            jnp.allclose(recon, self.control_recon, rtol=0.0, atol=0.0),
            msg=f"[{self.MODEL.__name__}] recon unexpectedly equal at zero tol",
        )

# ---- Concrete geometry variants ----

@pytest.mark.data_dependent
class TestReconCone(ReconTestBase):
    MODEL = mj.ConeBeamModel
    SOURCE_FILEPATH = "/depot/bouman/data/unit_test_data/cone_32_recon_data.tgz"
    TOLERANCES = {'nrmse': 0.05, 'max_diff': 0.12, 'pct_95': 0.02}

@pytest.mark.data_dependent
class TestReconParallel(ReconTestBase):
    MODEL = mj.ParallelBeamModel
    SOURCE_FILEPATH = "/depot/bouman/data/unit_test_data/parallel_32_recon_data.tgz"
    TOLERANCES = {'nrmse': 0.08, 'max_diff': 0.12, 'pct_95': 0.032}


if __name__ == "__main__":
    unittest.main()