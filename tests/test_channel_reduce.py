"""Ground-truth tests for channel_scatter_reduce (mbirjax/projectors.py).

The forward kernels' channel scatter is platform-split: scatter-add (CPU) vs sorted
segment-sum (GPU) -- see plans/projector_kernels/fwd_back_findings.md.  These tests pin
BOTH implementations against a plain numpy reference, on both sides of the contract:

  * duplicate indices (many pixels binning into one channel -- the production regime),
  * the boundary contract (indices CLIPPED into range with their weights zeroed -- out-of-range
    taps must contribute exactly zero on both paths),
  * cross-implementation agreement.  On integer-valued inputs the float32 sums are exact, so
    the two implementations must agree EXACTLY there; on random inputs they may differ only in
    summation order, so the comparison is relative (project rule: reordering, never exactness).
"""
import unittest

import numpy as np
import jax.numpy as jnp

from mbirjax.projectors import (channel_scatter_reduce, _channel_reduce_scatter_add,
                                _channel_reduce_sort_segsum)

IMPLEMENTATIONS = [('scatter_add', _channel_reduce_scatter_add),
                   ('sort_segsum', _channel_reduce_sort_segsum)]


def numpy_reference(n, A, values, num_out):
    """out[c, :] = sum over (tap, pixel) with n[tap, pixel] == c of A[tap, pixel] * values[pixel, :]."""
    n, A, values = np.asarray(n), np.asarray(A, dtype=np.float64), np.asarray(values, dtype=np.float64)
    out = np.zeros((num_out, values.shape[1]))
    for k in range(n.shape[0]):
        for p in range(n.shape[1]):
            out[n[k, p]] += A[k, p] * values[p]
    return out


def make_case(rng, num_taps=3, num_pixels=500, num_cols=7, num_out=32, integer_valued=False):
    """A contract-shaped case: clipped indices, weights zeroed where the unclipped tap was
    out of range, heavy duplication (num_pixels >> num_out)."""
    centers = rng.integers(-1, num_out + 1, size=num_pixels)      # includes out-of-range taps
    n_raw = centers[None, :] + np.arange(num_taps)[:, None] - num_taps // 2
    in_range = (n_raw >= 0) & (n_raw < num_out)
    if integer_valued:
        A = rng.integers(0, 4, size=(num_taps, num_pixels)).astype(np.float32)
        values = rng.integers(0, 4, size=(num_pixels, num_cols)).astype(np.float32)
    else:
        A = rng.random((num_taps, num_pixels), dtype=np.float32)
        values = rng.standard_normal((num_pixels, num_cols)).astype(np.float32)
    A = A * in_range
    n = np.clip(n_raw, 0, num_out - 1)
    return jnp.asarray(n), jnp.asarray(A), jnp.asarray(values), num_out


class TestChannelScatterReduce(unittest.TestCase):

    def test_both_implementations_match_reference(self):
        rng = np.random.default_rng(0)
        n, A, values, num_out = make_case(rng)
        ref = numpy_reference(n, A, values, num_out)
        scale = np.max(np.abs(ref))
        for name, impl in IMPLEMENTATIONS:
            out = np.asarray(impl(n, A, values, num_out))
            np.testing.assert_allclose(out, ref, rtol=0, atol=1e-5 * scale,
                                       err_msg=f'implementation {name}')

    def test_implementations_agree_exactly_on_integer_inputs(self):
        # Small-integer inputs sum exactly in float32, so any disagreement is a real indexing
        # or masking bug, not summation order.
        rng = np.random.default_rng(1)
        n, A, values, num_out = make_case(rng, integer_valued=True)
        out_scatter = np.asarray(_channel_reduce_scatter_add(n, A, values, num_out))
        out_sorted = np.asarray(_channel_reduce_sort_segsum(n, A, values, num_out))
        np.testing.assert_array_equal(out_scatter, out_sorted)
        np.testing.assert_array_equal(out_scatter, numpy_reference(n, A, values, num_out))

    def test_zero_weight_boundary_taps_contribute_nothing(self):
        # All taps out of range (weights zeroed, indices clipped to the edges): both paths
        # must return exactly zero -- no reliance on scatter drop semantics.
        num_out, num_cols = 8, 3
        n = jnp.asarray([[0, 0, num_out - 1, num_out - 1]])       # clipped edge indices
        A = jnp.zeros((1, 4))
        values = jnp.ones((4, num_cols))
        for name, impl in IMPLEMENTATIONS:
            out = np.asarray(impl(n, A, values, num_out))
            np.testing.assert_array_equal(out, np.zeros((num_out, num_cols)),
                                          err_msg=f'implementation {name}')

    def test_dispatch_by_flag(self):
        # channel_scatter_reduce(use_sorted=...) must return the same values as the
        # implementation it dispatches to.  num_cols here (7) is inside the sorted window
        # [MIN_COLS, MAX_COLS], so use_sorted=1 really routes to the sorted segment-sum
        # (the wide-column fallback is exercised in test_wide_columns_fall_back_to_scatter).
        rng = np.random.default_rng(2)
        n, A, values, num_out = make_case(rng, integer_valued=True)
        np.testing.assert_array_equal(
            np.asarray(channel_scatter_reduce(n, A, values, num_out, use_sorted=0)),
            np.asarray(_channel_reduce_scatter_add(n, A, values, num_out)))
        np.testing.assert_array_equal(
            np.asarray(channel_scatter_reduce(n, A, values, num_out, use_sorted=1)),
            np.asarray(_channel_reduce_sort_segsum(n, A, values, num_out)))

    def test_wide_columns_fall_back_to_scatter(self):
        # Above SORTED_CHANNEL_REDUCE_MAX_COLS the sorted segment-sum lowering collapses
        # (a measured cliff, ~18x per column at 1536; fwd_guard_cliff.py), so
        # channel_scatter_reduce must ROUTE wide-column calls to the scatter-add even when
        # use_sorted is set -- value-equal, just avoiding the cliff.  The cap is inclusive:
        # at exactly MAX_COLS the sorted path is still used.  Values alone can't prove the
        # route (the two reductions agree), so spy on which implementation runs.
        import mbirjax.projectors as P
        rng = np.random.default_rng(4)
        called = []

        def spy(tag, fn):
            def wrapped(*a, **k):
                called.append(tag)
                return fn(*a, **k)
            return wrapped

        orig_sorted, orig_scatter = P._channel_reduce_sort_segsum, P._channel_reduce_scatter_add
        P._channel_reduce_sort_segsum = spy('sorted', orig_sorted)
        P._channel_reduce_scatter_add = spy('scatter', orig_scatter)
        try:
            for num_cols, expect in [(P.SORTED_CHANNEL_REDUCE_MAX_COLS, 'sorted'),
                                     (P.SORTED_CHANNEL_REDUCE_MAX_COLS + 1, 'scatter')]:
                called.clear()
                n, A, values, num_out = make_case(rng, num_cols=num_cols)
                out = np.asarray(
                    P.channel_scatter_reduce(n, A, values, num_out, use_sorted=1))
                self.assertEqual(called, [expect],
                                 msg=f'num_cols={num_cols} routed via {called}, expected [{expect}]')
                ref = numpy_reference(n, A, values, num_out)          # value-correct either way
                np.testing.assert_allclose(out, ref, rtol=0,
                                           atol=1e-5 * np.max(np.abs(ref)))
        finally:
            P._channel_reduce_sort_segsum = orig_sorted
            P._channel_reduce_scatter_add = orig_scatter

    def test_single_tap_and_single_pixel_shapes(self):
        # Degenerate shapes the kernels can produce (psf_radius=0; tiny pixel batches).
        rng = np.random.default_rng(3)
        for num_taps, num_pixels in [(1, 6), (3, 1), (1, 1)]:
            n, A, values, num_out = make_case(rng, num_taps=num_taps, num_pixels=num_pixels,
                                              num_cols=2, num_out=4)
            ref = numpy_reference(n, A, values, num_out)
            for name, impl in IMPLEMENTATIONS:
                out = np.asarray(impl(n, A, values, num_out))
                np.testing.assert_allclose(out, ref, rtol=0, atol=1e-6 + 1e-5 * np.max(np.abs(ref)),
                                           err_msg=f'{name} taps={num_taps} pixels={num_pixels}')


if __name__ == '__main__':
    unittest.main()
