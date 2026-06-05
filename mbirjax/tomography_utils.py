import jax
import jax.numpy as jnp


# Row batch (B) for apply_row_filter: detector rows convolved per scan step.  The
# cuFFT work area is ~ B * fft_len(c).  GPU sweeps (H100) show large B is ~10x
# faster than small B, with the benefit past 1024 shrinking as c grows while the
# work area grows — so 1024 is near-optimal across c = 512..1624 and keeps the
# peak at the ~2x (input + output) memory floor.  Could later become c-aware
# (B ~ memory_budget / fft_len(c)), going larger only for small c.
ROW_FILTER_BATCH = 1024


@jax.jit
def apply_row_filter(block, filter_arr):
    """Apply a per-row 1-D filter to every detector row of a sinogram block.

    Geometry-agnostic: ``block`` is (views, rows, channels) and ``filter_arr`` is
    the 1-D recon filter (length 2*channels - 1 from generate_direct_recon_filter,
    already scaled by the caller).  Each detector row is convolved with the filter
    ('valid' mode, so the output length stays == channels).  This is the shared
    FBP/FDK filter kernel for every geometry (parallel beam now; cone beam and the
    rest as they migrate).

    Implementation — a ``lax.scan`` over overlapping B-row windows that writes each
    window straight into a preallocated output:
      - Flatten (views, rows) into one row axis and convolve B rows per step
        (vmap over the B), so the cuFFT work area is ~ B * fft_len(c), bounded by
        B alone (ROW_FILTER_BATCH), independent of geometry and device count.
      - vmap supplies the B-way parallelism, so lax.map's batch_size code path
        (jax-ml/jax#27591 — wrong results for large batch_size) is avoided and B
        is a free knob.
      - The last window is clamped in-bounds; when B doesn't divide views*rows it
        overlaps the previous window by a few rows, which are recomputed and
        rewritten identically (the filter is per-row → idempotent).  No padding
        copy and no concat, and the output is updated in place, so the peak stays
        at the input + output floor (~2x the shard) regardless of divisibility.

    @jax.jit'd — the jit is load-bearing for multi-device scaling (eager,
    op-by-op dispatch barely scales on the CPU/GPU backends).

    Args:
        block (jax array):      Shape (views, rows, channels), contiguous.
        filter_arr (jax array): Shape (2*channels - 1,), the (pre-scaled) filter.

    Returns:
        jax array of shape (views, rows, channels).
    """
    n_views, n_rows, n_channels = block.shape
    total_rows = n_views * n_rows
    batch = min(ROW_FILTER_BATCH, total_rows)        # don't batch past a tiny shard
    n_steps = (total_rows + batch - 1) // batch       # ceil → number of windows
    rows = block.reshape(total_rows, n_channels)
    c_out = n_channels                                # mode='valid': out length == c

    # Start row of each B-row window.  The last start is clamped to
    # total_rows - batch so the final window stays in-bounds; when batch does not
    # divide total_rows it overlaps the previous window by a few rows, which are
    # recomputed and rewritten with identical values (idempotent for a per-row
    # filter).  When batch divides total_rows the clamp is a no-op (no overlap).
    starts = jnp.array(
        [min(i * batch, total_rows - batch) for i in range(n_steps)],
        dtype=jnp.int32)

    def _convolve_row(row):
        return jax.scipy.signal.fftconvolve(row, filter_arr, mode='valid')

    def _filter_window(out, start):
        # Slice a (batch, c) window, convolve its rows, and write the result
        # straight into the preallocated output buffer (the scan carry) — no
        # padding copy and no concat, so the only full-size arrays are `rows` (a
        # view of `block`) and `out`.
        window = jax.lax.dynamic_slice(rows, (start, 0), (batch, n_channels))
        filtered = jax.vmap(_convolve_row)(window)               # (batch, c_out)
        return jax.lax.dynamic_update_slice(out, filtered, (start, 0)), None

    out = jnp.zeros((total_rows, c_out), dtype=rows.dtype)
    out, _ = jax.lax.scan(_filter_window, out, starts)
    return out.reshape(n_views, n_rows, c_out)


def generate_direct_recon_filter(num_channels, filter_name="ramp"):
    """
    Creates the specified space domain filter of size (2*num_channels - 1).

    Currently supported filters include: \"ramp\", which corresponds to a ramp in frequency domain.

    Args:
        num_channels (int): Number of detector channels in the sinogram.
        filter_name (string, optional): Name of the filter to be generated. Defaults to "ramp."

    Returns:
        filter (jnp): The computed filter (filter.size = 2*num_channels + 1).
    """

    # If you want to add a new filter, place its name into supported_filters, and ...
    # ... create a new if statement with the filter math.
    # TODO:  Anyone who adds a second filter will need to address how to document the set of available filters
    # in a way that is easy to maintain and appears correctly on readthedocs.  Also, any new filters will need
    # to have the proper scaling.
    supported_filters = ["ramp"]

    # Raise error if filter is not supported.
    if filter_name not in supported_filters:
        raise ValueError(f"Unsupported filter. Supported filters are: {', '.join(supported_filters)}.")

    n = jnp.arange(-num_channels + 1, num_channels)  # ex: num_channels = 3, -> n = [-2, -1, 0, 1, 2]

    recon_filter = 0
    if filter_name == "ramp":
        recon_filter = (1 / 2) * jnp.sinc(n) - (1 / 4) * (jnp.sinc(n / 2)) ** 2

    return recon_filter


