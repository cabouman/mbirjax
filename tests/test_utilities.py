import numpy as np
import jax
import jax.numpy as jnp
import mbirjax
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
            output = mbirjax.concatenate_function_in_batches(identity, data_to_batch, batch_size)
            assert(jnp.allclose(output, target))

        # Single input, multiple outputs
        target = data_to_batch, 2 * data_to_batch
        for batch_size in [batch_size_even, batch_size_odd]:
            output = mbirjax.concatenate_function_in_batches(two_output, data_to_batch, batch_size)
            assert (jnp.allclose(output[0], target[0]))
            assert (jnp.allclose(output[1], target[1]))

        # Multiple inputs, multiple and different sized outputs
        input_data = (data_to_batch, data_to_batch)
        for batch_size in [batch_size_even, batch_size_odd]:
            output = mbirjax.concatenate_function_in_batches(different_outputs, input_data, batch_size)
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
            output = mbirjax.sum_function_in_batches(simple_sum, data_to_batch, batch_size)
            assert(jnp.allclose(output, target))

        # Single input, multiple outputs
        target = jnp.sum(data_to_batch), jnp.sum(2 * data_to_batch)
        for batch_size in [batch_size_even, batch_size_odd]:
            output = mbirjax.sum_function_in_batches(two_output_sum, data_to_batch, batch_size)
            assert (jnp.allclose(output[0], target[0]))
            assert (jnp.allclose(output[1], target[1]))

        # Multiple inputs, multiple and different sized outputs
        input_data = (data_to_batch, data_to_batch)
        mult_factor = 3
        extra_args = (mult_factor, )
        for batch_size in [batch_size_even, batch_size_odd]:
            output = mbirjax.sum_function_in_batches(different_outputs, input_data, batch_size, extra_args)
            num_batches = jnp.ceil(input_data[0].shape[0] / batch_size).astype(int)
            target = jnp.sum((mult_factor + 1) * data_to_batch), jnp.ones(fixed_output_size) * num_batches
            assert (jnp.allclose(output[0], target[0]))
            assert (jnp.allclose(output[1], target[1]))


if __name__ == '__main__':
    unittest.main()
