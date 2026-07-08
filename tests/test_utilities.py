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


if __name__ == '__main__':
    unittest.main()
