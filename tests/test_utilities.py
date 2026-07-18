import warnings
import numpy as np
import jax.numpy as jnp
import mbirjax as mj
import unittest


class TestUtilities(unittest.TestCase):
    """
    Test various utilities
    """

    def setUp(self):
        """Set up before each test method."""
        pass

    def tearDown(self):
        """Clean up after each test method."""
        pass

    def test_parameter_names(self):

        print('Testing consistency of parameter names in ParameterHandler')
        consistent = mj._utils.update_param_literal(verify_match_and_exit=True)
        if not consistent:
            warnings.warn('Run mbirjax._utils.update_param_literal() to update ParamNames in ParameterHandler')
        assert consistent

    def test_merge_log_files(self):

        print('Testing merge_log_files')
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            merged_path = os.path.join(tmp_dir, 'merged.log')
            path_a = os.path.join(tmp_dir, 'a.log')
            path_b = os.path.join(tmp_dir, 'b.log')
            with open(path_a, 'w') as f:
                f.write('content a\n')
            with open(path_b, 'w') as f:
                f.write('content b\n')

            # Missing temps are skipped; existing ones appear in order under their headers, then are removed.
            mj.merge_log_files(merged_path, [('first', path_a), ('missing', path_a + '.nope'), ('second', path_b)])
            with open(merged_path, 'r') as f:
                text = f.read()
            self.assertEqual(text, '======== first ========\ncontent a\n'
                                   '======== second ========\ncontent b\n')
            self.assertNotIn('missing', text)
            self.assertFalse(os.path.exists(path_a))
            self.assertFalse(os.path.exists(path_b))

            # No temps at all (including None paths): no output file is written.
            merged_path_2 = os.path.join(tmp_dir, 'merged2.log')
            mj.merge_log_files(merged_path_2, [('first', path_a), ('none', None)])
            self.assertFalse(os.path.exists(merged_path_2))

    def test_concatenate_function_in_batches(self):

        print('Testing concatenate_function_in_batches')

        # Test multiple inputs, multiple outputs, batch size that divides the total ("even") and that doesn't ("odd")
        # Outputs that are the same size and outputs that are different
        def identity(j):
            return j

        def two_output(j):
            return j, j * 2

        fixed_output_size = 5

        def different_outputs(j, k):
            return j + 2 * k, np.ones(fixed_output_size)

        data_to_batch = np.arange(8)
        batch_size_even = 4
        batch_size_odd = 3

        # Single input, single output
        target = data_to_batch
        for batch_size in [batch_size_even, batch_size_odd]:
            output = mj.concatenate_function_in_batches(identity, data_to_batch, batch_size)
            assert (jnp.allclose(output, target))

        # Single input, multiple outputs
        target = data_to_batch, 2 * data_to_batch
        for batch_size in [batch_size_even, batch_size_odd]:
            output = mj.concatenate_function_in_batches(two_output, data_to_batch, batch_size)
            assert (jnp.allclose(output[0], target[0]))
            assert (jnp.allclose(output[1], target[1]))

        # Multiple inputs, multiple and different sized outputs
        input_data = (data_to_batch, data_to_batch)
        for batch_size in [batch_size_even, batch_size_odd]:
            output = mj.concatenate_function_in_batches(different_outputs, input_data, batch_size)
            num_batches = jnp.ceil(input_data[0].shape[0] / batch_size).astype(int)
            target = 3 * data_to_batch, jnp.ones(fixed_output_size * num_batches)
            assert (jnp.allclose(output[0], target[0]))
            assert (jnp.allclose(output[1], target[1]))

    def test_sum_function_in_batches(self):

        print('Testing sum_function_in_batches')

        # Test multiple inputs, multiple outputs, batch size that divides the total ("even") and that doesn't ("odd")
        # Outputs that are the same size and outputs that are different
        def simple_sum(j):
            return jnp.sum(j)

        def two_output_sum(j):
            return jnp.sum(j), jnp.sum(j * 2)

        fixed_output_size = 5

        def different_outputs(j, k, factor):
            return jnp.sum(j + factor * k), np.ones(fixed_output_size)

        data_to_batch = np.arange(8)
        batch_size_even = 4
        batch_size_odd = 3

        # Single input, single output
        target = jnp.sum(data_to_batch)
        for batch_size in [batch_size_even, batch_size_odd]:
            output = mj.sum_function_in_batches(simple_sum, data_to_batch, batch_size)
            assert(jnp.allclose(output, target))

        # Single input, multiple outputs
        target = jnp.sum(data_to_batch), jnp.sum(2 * data_to_batch)
        for batch_size in [batch_size_even, batch_size_odd]:
            output = mj.sum_function_in_batches(two_output_sum, data_to_batch, batch_size)
            assert (jnp.allclose(output[0], target[0]))
            assert (jnp.allclose(output[1], target[1]))

        # Multiple inputs, multiple and different sized outputs
        input_data = (data_to_batch, data_to_batch)
        mult_factor = 3
        extra_args = (mult_factor, )
        for batch_size in [batch_size_even, batch_size_odd]:
            output = mj.sum_function_in_batches(different_outputs, input_data, batch_size, extra_args)
            num_batches = jnp.ceil(input_data[0].shape[0] / batch_size).astype(int)
            target = jnp.sum((mult_factor + 1) * data_to_batch), jnp.ones(fixed_output_size) * num_batches
            assert (jnp.allclose(output[0], target[0]))
            assert (jnp.allclose(output[1], target[1]))


class TestExportReconHostResidence(unittest.TestCase):
    """The HDF5 export path must stay on the HOST for a host recon, so a large volume is never copied
    back onto a single device (the OOM Charlie hit at downsampling 1, f32[1370,1880,1880] ~ 18 GiB).

    This is a host-RESIDENCE guard (the real failure is placement, deterministic at any size), not an
    actual large run: feed a small host recon and assert nothing is promoted to a jax device, plus a
    small export -> import round-trip.
    """

    def test_apply_cylindrical_mask_host_in_host_out(self):
        import mbirjax.preprocess as mjp
        recon = np.random.RandomState(0).rand(16, 16, 8).astype(np.float32)
        out = mjp.apply_cylindrical_mask(recon, radial_margin=2, top_margin=1, bottom_margin=1)
        self.assertIsInstance(out, np.ndarray)             # host in -> host out (no device promotion)
        self.assertEqual(out.dtype, np.float32)
        np.testing.assert_array_equal(out[:, :, 0], 0)     # top margin zeroed
        np.testing.assert_array_equal(out[:, :, -1], 0)    # bottom margin zeroed
        self.assertEqual(float(out[0, 0, 4]), 0.0)         # a corner outside the cylinder is zeroed

    def test_apply_cylindrical_mask_jax_in_jax_out(self):
        import jax
        import mbirjax.preprocess as mjp
        recon = jnp.asarray(np.random.RandomState(1).rand(16, 16, 8).astype(np.float32))
        out = mjp.apply_cylindrical_mask(recon)
        self.assertIsInstance(out, jax.Array)              # jax in -> jax out (on-device preserved)

    def test_export_import_roundtrip_host(self):
        import os, tempfile
        recon = np.random.RandomState(2).rand(12, 10, 6).astype(np.float32)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'recon.h5')
            mj.export_recon_hdf5(path, recon, recon_dict=None, remove_flash=True)
            self.assertTrue(os.path.exists(path))
            loaded, _ = mj.import_recon_hdf5(path)
            self.assertIsInstance(loaded, np.ndarray)      # host array back
            self.assertEqual(loaded.shape, recon.shape)    # round-trips to (row, col, slice)

    def test_export_import_roundtrip_values(self):
        # No flash mask, so the round-trip must be EXACT: catches any re-introduced
        # axis flip or transpose mismatch between export and import (the import-side
        # slice-axis reversal removed in e80f4d0 would fail this test).
        import os, tempfile
        recon = np.random.RandomState(3).rand(12, 10, 6).astype(np.float32)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'recon.h5')
            mj.export_recon_hdf5(path, recon, recon_dict=None)
            loaded, _ = mj.import_recon_hdf5(path)
            np.testing.assert_array_equal(loaded, recon)


class TestIsOomClassifier(unittest.TestCase):
    """is_oom scans the FULL traceback, whose file paths can embed a marker as a substring: a
    checkout named "mbirjax_headroom" put "oom" in every traceback and classified every error as
    an OOM.  These tests are path-independent (the sharding-hooks non-OOM test only catches the
    bug when the checkout path itself contains "oom")."""

    def test_real_oom_signatures_classify_true(self):
        for message in ('XLA: RESOURCE_EXHAUSTED: Out of memory while trying to allocate 1073741824 bytes',
                        'OOM when allocating tensor with shape[1024,1024,1024]',
                        'Failed to create cuFFT batched plan with scratch allocator',
                        'Failed to allocate work area',
                        'MemoryError: std::bad_alloc'):
            self.assertTrue(mj._utils.is_oom(message), message)

    def test_marker_inside_path_word_does_not_classify(self):
        fake_traceback = ('Traceback (most recent call last):\n'
                          '  File "/home/user/mbirjax_headroom/mbirjax/tomography_model.py", line 1\n'
                          '  File "/home/user/tests/test_non_oom_error.py", line 2\n'
                          'jax.errors.JaxRuntimeError: XLA computation has a shape mismatch')
        self.assertFalse(mj._utils.is_oom(fake_traceback))


class TestGetAllParamsBuildModel(unittest.TestCase):
    """get_all_params is the single source of truth for reading a model's params back out; build_model
    reconstructs a model from (required, optional, regularization).  The round-trip must reproduce the
    model, and (required, optional) alone must be a self-contained model description (build_model
    resolves the class from required['geometry_type'])."""

    @staticmethod
    def _angles(n=20):
        return np.linspace(0, np.pi, n, endpoint=False)

    def _assert_roundtrip(self, model):
        req, opt, reg = model.get_all_params()
        self.assertEqual(req['geometry_type'], str(type(model)))
        self.assertNotIn('geometry_type', opt)
        rebuilt = mj.build_model(req, opt, reg)
        self.assertIs(type(rebuilt), type(model))
        self.assertEqual(tuple(rebuilt.get_params('recon_shape')), tuple(model.get_params('recon_shape')))
        self.assertAlmostEqual(float(rebuilt.get_params('delta_voxel')), float(model.get_params('delta_voxel')))

    def test_cone_roundtrip_unpacks_view_components(self):
        m = mj.ConeBeamModel((20, 32, 32), self._angles(), source_detector_dist=128, source_iso_dist=64)
        req, opt, reg = m.get_all_params()
        self.assertIn('angles', req)              # view components unpacked into constructor args
        self.assertIn('helical_z_shifts', req)
        self.assertNotIn('view_params_array', opt)   # the packed array is rebuilt from the components
        self._assert_roundtrip(m)

    def test_parallel_and_translation_roundtrip(self):
        self._assert_roundtrip(mj.ParallelBeamModel((20, 32, 32), self._angles()))
        tv = np.random.RandomState(0).randn(20, 3) * 5
        self._assert_roundtrip(mj.TranslationModel((20, 24, 24), tv, source_detector_dist=500, source_iso_dist=250))

    def test_regularization_bucket_membership(self):
        _, opt, reg = mj.ConeBeamModel((20, 32, 32), self._angles(),
                                       source_detector_dist=128, source_iso_dist=64).get_all_params()
        self.assertEqual(set(reg),
                         {'sigma_y', 'sigma_x', 'sigma_prox', 'snr_db', 'sharpness', 'auto_regularize_flag'})
        for k in ('p', 'q', 'T', 'qggmrf_nbr_wts'):   # prior SHAPE params are structural -> optional
            self.assertIn(k, opt)
            self.assertNotIn(k, reg)

    def test_build_model_from_required_only(self):
        req, _, _ = mj.ConeBeamModel((20, 32, 32), self._angles(),
                                     source_detector_dist=128, source_iso_dist=64).get_all_params()
        self.assertIs(type(mj.build_model(req)), mj.ConeBeamModel)   # optional/regularization default None

    def test_pinned_recon_shape_survives_auto(self):
        req, opt, _ = mj.ConeBeamModel((20, 32, 32), self._angles(),
                                       source_detector_dist=128, source_iso_dist=64).get_all_params()
        opt['recon_shape'] = (40, 40, 40)
        self.assertEqual(tuple(mj.build_model(req, opt).get_params('recon_shape')), (40, 40, 40))

    def test_build_model_unknown_geometry_type_raises(self):
        with self.assertRaises(ValueError):
            mj.build_model({'geometry_type': 'not-a-class', 'sinogram_shape': (1, 1, 1), 'angles': [0.0]})


if __name__ == '__main__':
    unittest.main()
