import os, shutil, pytest, pathlib, unittest, h5py, warnings, jax, jax.numpy as jnp
import mbirjax as mj
from _test_data_dependent_utils import sha256_file

class ReconBase:
    """
    Reusable test suite for a projection model. Subclasses must set:
      - MODEL (class with .from_file)
      - SOURCE_FILEPATH (str)
    """
    HAS_GPU = any(d.platform == "gpu" for d in jax.devices())
    USE_GPU_OPTS = ["automatic", "full", "sinograms", "projections", "none"] if HAS_GPU else ["none"]
    ATOL = 1e-3
    TEST_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(TEST_DIR, "data")
    LOG_DIR = os.path.join(TEST_DIR, "logs")

    # these fields will be set by the setUpClass method
    control_phantom = None
    control_sinogram = None
    control_recon = None
    control_params = None
    projection_model =  None
    data_filepath = None

    # To be overridden in subclasses:
    MODEL = None
    SOURCE_FILEPATH = None
    DATA_FILE_SHA256 = None
    TOLERANCES = None

    def subTest(self, *a, **kw):     return unittest.TestCase.subTest(self, *a, **kw)

    @classmethod
    def setUpClass(cls):

        if not cls.HAS_GPU:
            warnings.warn("No GPUs found. Only use_gpu='none' unit test will be performed.")

        assert cls.SOURCE_FILEPATH and cls.MODEL, \
            "Subclasses must define MODEL, SOURCE_FILEPATH"

        # delete the data directory and all its contents
        if os.path.exists(cls.DATA_DIR):
            shutil.rmtree(cls.DATA_DIR)
        cls.data_filepath = mj.download_and_extract(cls.SOURCE_FILEPATH, cls.DATA_DIR)

        # verify the file contents with sha256
        try:
            p = pathlib.Path(cls.data_filepath)
            actual = sha256_file(p)
            if actual.lower() != cls.DATA_FILE_SHA256.lower():
                warnings.warn(f"Checksum mismatch for {p.name}: expected {cls.DATA_FILE_SHA256}, got {actual}. "
                              "Failures may be due to unexpected input data.")
        except Exception as e:
            warnings.warn(f"Checksum skipped for {cls.data_filepath}: {e}")

    @classmethod
    def tearDownClass(cls):
        # delete the data directory and all its contents
        if cls.DATA_DIR and os.path.exists(cls.DATA_DIR):
            shutil.rmtree(cls.DATA_DIR)

    def setUp(self):
        with h5py.File(self.data_filepath, "r") as f:
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
                                                       logfile_path=os.path.join(self.LOG_DIR,
                                                                                 f'recon_{filename}_{opt}.log'))
                recon.block_until_ready()
                recon = jax.device_put(recon)

                max_diff = jnp.amax(jnp.abs(self.control_recon - recon))
                nrmse = jnp.linalg.norm(recon - self.control_recon) / jnp.linalg.norm(self.control_recon)
                pct_95 = jnp.percentile(jnp.abs(recon - self.control_recon), 95)

                all_within = (
                        float(max_diff) < self.TOLERANCES['max_diff'] and
                        float(nrmse) < self.TOLERANCES['nrmse'] and
                        float(pct_95) < self.TOLERANCES['pct_95']
                )
                assert all_within, f"[{self.MODEL.__name__}] recon mismatch (use_gpu={opt})"

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

        all_within = (
                float(max_diff) < self.TOLERANCES['max_diff'] and
                float(nrmse) < self.TOLERANCES['nrmse'] and
                float(pct_95) < self.TOLERANCES['pct_95']
        )
        assert not all_within, f"[{self.MODEL.__name__}] recon unexpectedly allclose with biased input"

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
        assert not bool(jnp.allclose(recon, self.control_recon, rtol=0.0, atol=0.0)), \
            f"[{self.MODEL.__name__}] recon unexpectedly equal at zero tol"

# ---- Concrete geometry variants ----

@pytest.mark.data_dependent
class TestReconCone(ReconBase, unittest.TestCase):
    MODEL = mj.ConeBeamModel
    SOURCE_FILEPATH = "https://www.datadepot.rcac.purdue.edu/bouman/data/unit_test_data/cone_32_recon_data.tgz"
    DATA_FILE_SHA256 = '7053ccf75298f587607644f3e96fbb3257c9f850704bcd16b484c5de9dcc9441'
    TOLERANCES = {'nrmse': 0.05, 'max_diff': 0.12, 'pct_95': 0.02}

@pytest.mark.data_dependent
class TestReconParallel(ReconBase, unittest.TestCase):
    MODEL = mj.ParallelBeamModel
    SOURCE_FILEPATH = "https://www.datadepot.rcac.purdue.edu/bouman/data/unit_test_data/parallel_32_recon_data.tgz"
    DATA_FILE_SHA256 = 'b0210a75c8a82659530d299d7cef2e5d5d296e3dbf2e51841f7d6f8f208fbf8a'
    TOLERANCES = {'nrmse': 0.08, 'max_diff': 0.12, 'pct_95': 0.032}

if __name__ == "__main__":
    unittest.main()