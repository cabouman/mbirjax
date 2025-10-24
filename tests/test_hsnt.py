import os
import tempfile
import unittest
import numpy as np
import mbirjax.hsnt as hsnt


class TestHSNT(unittest.TestCase):
    """
    Core functionality tests for mbirjax.hsnt:
      - dehydrate / rehydrate
      - hyper_denoise
      - _estimate_subspace_dimension
      - create_hsnt_metadata / export_hsnt_data_hdf5 / import_hsnt_data_hdf5
      - generate_hyper_data
    """

    def setUp(self):
        np.random.seed(0)
        # Small hyperspectral dataset
        self.detector_rows = 16
        self.detector_columns = 16
        self.wavelengths = 10  # N_k
        self.subspace_dim = 2  # N_s

        # Subspace data for low-rank simulation
        self.subspace_data = np.abs(np.random.randn(self.detector_rows, self.detector_columns, self.subspace_dim)).astype(np.float32)
        self.subspace_basis = np.abs(np.random.randn(self.subspace_dim, self.wavelengths)).astype(np.float32)

        # Full hyperspectral clean data (matmul along last axis of subspace_data with subspace_basis)
        self.clean = self.subspace_data @ self.subspace_basis  # shape (detector_rows, detector_columns, wavelengths)
        # Add noise
        self.noisy = (self.clean + 0.2 * np.random.randn(*self.clean.shape)).astype(np.float32)

        # Temp HDF5 file
        self.tmpdir = tempfile.mkdtemp()
        self.h5file = os.path.join(self.tmpdir, "hsnt_test.h5")

    def tearDown(self):
        try:
            if os.path.exists(self.h5file):
                os.remove(self.h5file)
            os.rmdir(self.tmpdir)
        except Exception:
            pass

    def test_dehydrate_rehydrate_with_identity_basis(self):
        """Dehydrate + rehydrate with identity basis should recover data."""
        identity_basis = np.eye(self.wavelengths, dtype=np.float64)
        subspace_data, subspace_basis, dataset_type = hsnt.dehydrate(self.clean, subspace_basis=identity_basis, verbose=0)
        self.assertIsInstance(subspace_data, np.ndarray)
        self.assertIsInstance(subspace_basis, np.ndarray)
        self.assertEqual(dataset_type, 'attenuation')

        rehydrated = hsnt.rehydrate([subspace_data, subspace_basis, dataset_type])
        self.assertEqual(rehydrated.shape, self.clean.shape)
        rel_err = np.linalg.norm(rehydrated - self.clean) / (np.linalg.norm(self.clean) + 1e-12)
        self.assertLess(rel_err, 1e-6, f"Rehydration with identity basis failed (rel_err={rel_err})")

    def test_dehydrate_with_fixed_subspace_dimension(self):
        """Dehydrate with fixed subspace dimension should produce outputs with expected shapes."""
        dehydrated = hsnt.dehydrate(self.clean, subspace_dimension=self.subspace_dim, verbose=0)
        self.assertIsInstance(dehydrated, list)
        subspace_data, subspace_basis, dataset_type = dehydrated
        self.assertEqual(subspace_data.shape[-1], self.subspace_dim)
        self.assertEqual(subspace_basis.shape, (self.subspace_dim, self.wavelengths))
        self.assertEqual(dataset_type, 'attenuation')

        rehydrated = hsnt.rehydrate(dehydrated)
        self.assertEqual(rehydrated.shape, self.clean.shape)

    def test_hyper_denoise_reduces_noise(self):
        """hyper_denoise should reduce noise relative to noisy input."""
        before_std = np.std(self.noisy - self.clean)
        denoised = hsnt.hyper_denoise(self.noisy, subspace_dimension=self.subspace_dim, verbose=0)
        self.assertEqual(denoised.shape, self.noisy.shape)
        after_std = np.std(denoised - self.clean)
        self.assertLess(after_std, before_std, f"Denoising did not reduce noise (before={before_std}, after={after_std})")

    def test_estimate_subspace_dimension_sanity(self):
        """Estimate subspace dimension on the setup data."""
        # reshape to 2D (pixels x wavelengths)
        X = self.clean.reshape(-1, self.wavelengths)
        est = hsnt._estimate_subspace_dimension(X, safety_factor=1, verbose=0)
        self.assertIsInstance(est, int)
        self.assertGreaterEqual(est, 1)
        self.assertLessEqual(est, self.wavelengths)  # can't exceed wavelengths

    def test_hdf5_export_import_round_trip(self):
        """Test create_hsnt_metadata, export_hsnt_data_hdf5 and import_hsnt_data_hdf5 round-trip."""
        metadata_dict = hsnt.create_hsnt_metadata(dataset_name="synthetic_test", dataset_type="attenuation",
                                                  wavelengths=np.linspace(1.0, 2.0, self.wavelengths))

        hsnt.export_hsnt_data_hdf5(self.h5file, self.clean, metadata_dict)

        dataset_names = hsnt.import_hsnt_list_hdf5(self.h5file)
        self.assertIn("synthetic_test", dataset_names)

        imported_data, imported_metadata_dict = hsnt.import_hsnt_data_hdf5(self.h5file, "synthetic_test")
        self.assertEqual(imported_data.shape, self.clean.shape)
        diff = np.linalg.norm(imported_data - self.clean)
        self.assertLess(diff, 1e-6, f"HDF5 round trip data mismatch (diff={diff})")
        self.assertEqual(imported_metadata_dict["dataset_name"], "synthetic_test")
        self.assertTrue(np.allclose(imported_metadata_dict["wavelengths"], metadata_dict["wavelengths"]))

    def test_generate_hyper_data_basic(self):
        """generate_hyper_data returns correctly shaped and non-empty arrays."""
        material_basis = np.abs(np.random.randn(3, self.wavelengths)).astype(np.float32)
        noisy_proj, gt_proj = hsnt.generate_hyper_data(material_basis,
                                                       detector_rows=self.detector_rows,
                                                       detector_columns=self.detector_columns,
                                                       verbose=0)
        self.assertEqual(noisy_proj.shape, (self.detector_rows, self.detector_columns, self.wavelengths))
        self.assertEqual(gt_proj.shape, (self.detector_rows, self.detector_columns, self.wavelengths))
        self.assertTrue(np.isfinite(noisy_proj).all())
        self.assertTrue(np.isfinite(gt_proj).all())

    def test_generate_hyper_data_invalid_basis_raises(self):
        """generate_hyper_data should raise if material_basis has wrong number of rows."""
        bad_basis = np.abs(np.random.randn(2, self.wavelengths)).astype(np.float32)
        with self.assertRaises(ValueError):
            hsnt.generate_hyper_data(bad_basis,
                                     detector_rows=self.detector_rows,
                                     detector_columns=self.detector_columns,
                                     verbose=0)


if __name__ == "__main__":
    unittest.main()

