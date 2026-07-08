"""Ground-truth tests for the generic batching helpers in mbirjax/projectors.py.

sum_function_in_batches reads its full batches as dynamic_slice windows (the input-reshape
form measurably copied the batched input on GPU; see
plans/projector_batching/projector_batching_characterization.md).  These tests pin the
helpers' CONTRACT against the un-batched ground truth -- the plain sum / the plain map --
across every batching branch: n < B (single call), n == B, B | n (no remainder), ragged,
and batch_size=None.  Tolerances are relative: batching regroups the summation, so exact
equality is not the contract (project rule).
"""
import unittest

import numpy as np
import jax.numpy as jnp

from mbirjax.projectors import sum_function_in_batches, concatenate_function_in_batches

# (num_input_points, batch_size) covering each branch of the batching logic.
BATCH_CASES = [(5, 8), (8, 8), (12, 3), (10, 3), (7, None)]


class TestSumFunctionInBatches(unittest.TestCase):

    def test_single_array_matches_plain_sum(self):
        rng = np.random.default_rng(0)
        for n, b in BATCH_CASES:
            data = jnp.asarray(rng.standard_normal((n, 4), dtype=np.float32))
            out = sum_function_in_batches(lambda x: jnp.sum(x, axis=0), data, b)
            np.testing.assert_allclose(out, np.sum(np.asarray(data), axis=0),
                                       rtol=1e-5, err_msg=f'n={n}, batch={b}')

    def test_multiple_arrays_and_extra_args(self):
        rng = np.random.default_rng(1)
        scale = jnp.asarray(2.5, dtype=jnp.float32)     # non-batched extra argument

        def weighted(x, w, c):
            return c * jnp.sum(x * w, axis=0)

        for n, b in BATCH_CASES:
            x = jnp.asarray(rng.standard_normal((n, 3), dtype=np.float32))
            w = jnp.asarray(rng.standard_normal((n, 3), dtype=np.float32))
            out = sum_function_in_batches(weighted, (x, w), b, extra_args=(scale,))
            truth = 2.5 * np.sum(np.asarray(x) * np.asarray(w), axis=0)
            np.testing.assert_allclose(out, truth, rtol=1e-5, err_msg=f'n={n}, batch={b}')


class TestConcatenateFunctionInBatches(unittest.TestCase):

    def test_matches_plain_map(self):
        rng = np.random.default_rng(2)
        for n, b in BATCH_CASES:
            data = jnp.asarray(rng.standard_normal((n, 4), dtype=np.float32))
            out = concatenate_function_in_batches(lambda x: 3.0 * x, data, b)
            np.testing.assert_allclose(out, 3.0 * np.asarray(data),
                                       rtol=1e-6, err_msg=f'n={n}, batch={b}')


if __name__ == '__main__':
    unittest.main()
