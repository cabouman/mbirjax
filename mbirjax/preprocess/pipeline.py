"""Shared driver for the scan -> sinogram preprocessing pipeline.

The per-stage transforms in :mod:`mbirjax.preprocess.utilities` are pure device-array *kernels* (the
math only).  This module owns the **single** copy of the batching + host<->device transfer +
concatenate scaffolding that was previously duplicated inside each of ``compute_sino_transmission`` /
``downsample_view_data`` / ``correct_det_rotation``.

The driver supports two modes: a single-device sequential view-batch loop (the default, used by the
legacy per-stage public functions), and a multi-device view-sharded mode (used by the fused
``scan_to_sino``) where contiguous view shards run concurrently, one per device.  See
``experiments/sharding/plans/preprocessing_pipeline_refactor_plan.md``.
"""
import numpy as np
import jax
import jax.numpy as jnp


def _run_view_batches(array, kernel, batch_size, device, lo, hi, desc=None):
    """Run ``kernel`` over views ``[lo, hi)`` of ``array`` in ``batch_size`` chunks, all on ``device``.

    Runs under ``jax.default_device(device)`` so the kernel's ops -- and any HOST constants it closes
    over (which auto-promote on first use) -- land on ``device``.  Host->device per batch, device->host
    per batch; returns the concatenated host result (or ``None`` if the range is empty).
    """
    import tqdm
    steps = range(lo, hi, batch_size)
    if desc is not None:
        steps = tqdm.tqdm(steps, desc=desc)
    out = []
    with jax.default_device(device):
        for j in steps:
            batch = jax.device_put(array[j:min(j + batch_size, hi)], device)
            out.append(np.array(kernel(batch)))
    return np.concatenate(out, axis=0) if out else None


def map_view_batches(array, kernel, batch_size, desc=None, devices=None):
    """Apply a per-batch device kernel across the leading (view) axis.

    Single device (``devices`` is None or length 1): a sequential view-batch loop -- each contiguous
    batch of ``batch_size`` views is moved to the device, passed through ``kernel`` (a pure
    device-array -> device-array transform), and brought back to the host; results are concatenated.
    This bounds device memory to ``batch_size`` views.

    Multiple devices: the views are split into contiguous, in-order shards (one per device) and each
    shard is processed in its own thread on its own device (via :func:`run_per_device`, which sets
    ``jax.default_device``).  ``kernel`` must be **device-agnostic** -- it should close over HOST
    constants (NumPy), which auto-promote to each batch's device, NOT arrays already committed to one
    device.  Per-device host results are concatenated in view order.  Per-view kernels (no cross-view
    reduction) make this embarrassingly parallel with no cross-device communication.

    Args:
        array (numpy or jax array): data batched along axis 0 (views).
        kernel (callable): ``device_batch -> device_batch``; per-view, no host transfer inside.
        batch_size (int): number of views per on-device batch.
        desc (str or None, optional): tqdm label (single-device path only).
        devices (sequence or None): devices to spread the views over.  ``None`` means a single device
            (``jax.devices()[0]``) -- sharding is opt-in by passing several devices.

    Returns:
        numpy.ndarray: concatenation of the per-batch kernel outputs along axis 0 (view order).
    """
    devices = [jax.devices()[0]] if devices is None else list(devices)
    num_views = array.shape[0]

    if len(devices) <= 1:
        return _run_view_batches(array, kernel, batch_size, devices[0], 0, num_views, desc=desc)

    # Multi-device: contiguous, in-order view shards, one per device, run concurrently.
    from mbirjax import _sharding as mjs
    view_ranges = np.array_split(np.arange(num_views), len(devices))

    def worker(i, device):
        rng = view_ranges[i]
        if len(rng) == 0:
            return None
        return _run_view_batches(array, kernel, batch_size, device, int(rng[0]), int(rng[-1]) + 1)

    results = mjs.run_per_device(devices, worker)
    return np.concatenate([r for r in results if r is not None], axis=0)
