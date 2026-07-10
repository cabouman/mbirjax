"""Shared driver for the scan -> sinogram preprocessing pipeline.

The per-stage transforms in :mod:`mbirjax.preprocess.utilities` are pure device-array *kernels* (the
math only).  This module owns the **single** copy of the batching + host<->device transfer +
in-place-fill scaffolding that was previously duplicated inside each of ``compute_sino_transmission`` /
``downsample_view_data`` / ``correct_det_rotation``.

The driver supports two modes: a single-device sequential view-batch loop (the default, used by the
legacy per-stage public functions), and a multi-device view-sharded mode (used by the fused
``scan_to_sino``) where contiguous view shards run concurrently, one per device.  In both modes the
host output is pre-allocated once and each batch's result is written directly into its view-slice, so
the host footprint is the input + the single output (~2x) rather than input + per-shard gather +
concatenate destination (~3x).  See ``experiments/sharding/plans/preprocessing_pipeline_refactor_plan.md``.
"""
import numpy as np
import jax
import jax.numpy as jnp


def _fill_view_batches(array, kernel, output, batch_size, device, lo, hi, desc=None):
    """Run ``kernel`` over views ``[lo, hi)`` of ``array`` in ``batch_size`` chunks on ``device``,
    writing each batch's host result directly into ``output[j:...]`` (a pre-allocated host array).

    Runs under ``jax.default_device(device)`` so the kernel's ops -- and any HOST constants it closes
    over (which auto-promote on first use) -- land on ``device``.  Host->device per batch, device->host
    per batch.  Writing in place (rather than collecting per-batch results and concatenating) keeps the
    host footprint at the input + the single output array, with only one batch's result transiently live.
    """
    import tqdm
    steps = range(lo, hi, batch_size)
    if desc is not None:
        steps = tqdm.tqdm(steps, desc=desc)
    with jax.default_device(device):
        for j in steps:
            end = min(j + batch_size, hi)
            batch = jax.device_put(array[j:end], device)
            output[j:end] = np.array(kernel(batch))


def map_view_batches(array, kernel, batch_size, desc=None, devices=None):
    """Apply a per-batch device kernel across the leading (view) axis.

    Single device (``devices`` is None or length 1): a sequential view-batch loop -- each contiguous
    batch of ``batch_size`` views is moved to the device, passed through ``kernel`` (a pure
    device-array -> device-array transform), and written back into the host output.  This bounds device
    memory to ``batch_size`` views.

    Multiple devices: the views are split into contiguous, in-order shards (one per device) and each
    shard is processed in its own thread on its own device (via :func:`run_per_device`, which sets
    ``jax.default_device``).  ``kernel`` must be **device-agnostic** -- it should close over HOST
    constants (NumPy), which auto-promote to each batch's device, NOT arrays already committed to one
    device.  Each worker writes its disjoint view-slice of the shared host output in view order.
    Per-view kernels (no cross-view reduction) make this embarrassingly parallel with no cross-device
    communication.

    The host output is pre-allocated once (its shape/dtype probed from the first batch, since a kernel
    may change the trailing detector dims, e.g. downsampling) and filled in place, so the host footprint
    is input + output (~2x) regardless of device count.

    Args:
        array (numpy or jax array): data batched along axis 0 (views).
        kernel (callable): ``device_batch -> device_batch``; per-view, no host transfer inside.
        batch_size (int): number of views per on-device batch.
        desc (str or None, optional): tqdm label (single-device path only).
        devices (sequence or None): devices to spread the views over.  ``None`` means a single device
            (``jax.devices()[0]``) -- sharding is opt-in by passing several devices.

    Returns:
        numpy.ndarray: the per-batch kernel outputs assembled along axis 0 (view order).
    """
    devices = [jax.devices()[0]] if devices is None else list(devices)
    num_views = array.shape[0]

    # Probe the kernel's output shape/dtype on the first batch so a SINGLE host output array can be
    # pre-allocated and every worker can write its view-slice in place.  This avoids the per-shard result
    # lists + the final concatenate, which together held a second and third full-size copy of the result
    # on the host (input + per-shard gather + concatenate destination ~= 3x the result).  Writing in
    # place bounds the host footprint to input + output (~2x), independent of device count.  The probe
    # runs on devices[0] -- which also owns view 0's shard -- so the computed values are unchanged.
    probe_hi = min(batch_size, num_views)
    with jax.default_device(devices[0]):
        probe = np.array(kernel(jax.device_put(array[0:probe_hi], devices[0])))
    output = np.empty((num_views,) + probe.shape[1:], dtype=probe.dtype)
    output[0:probe_hi] = probe
    del probe

    if len(devices) <= 1:
        _fill_view_batches(array, kernel, output, batch_size, devices[0], probe_hi, num_views, desc=desc)
        return output

    # Multi-device: contiguous, in-order view shards, one per device.  Each worker BLOCKS per batch on
    # the device->host transfer (np.array), so we use run_per_device's THREAD POOL (one thread per
    # device) to get true cross-device concurrency: while one thread waits on its transfer/compute, the
    # others run on their own devices (JAX releases the GIL during XLA execution and transfers).  This is
    # the opposite of a single-threaded async-dispatch loop, where any per-step block serializes the
    # devices.  Note: a CPU with forced virtual devices shares the physical cores and shows NO speedup --
    # the parallelism is real only on separate hardware (e.g. multiple GPUs).  Workers write disjoint
    # view-slices of the shared host ``output`` (safe: the slices never overlap).
    from mbirjax import _sharding as mjs
    view_ranges = np.array_split(np.arange(num_views), len(devices))

    def worker(i, device):
        rng = view_ranges[i]
        if len(rng) == 0:
            return
        lo = max(int(rng[0]), probe_hi)  # views [0:probe_hi] are already filled by the probe
        hi = int(rng[-1]) + 1
        if lo >= hi:
            return
        _fill_view_batches(array, kernel, output, batch_size, device, lo, hi)

    mjs.run_per_device(devices, worker)
    return output
