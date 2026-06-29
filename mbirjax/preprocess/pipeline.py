"""Shared driver for the scan -> sinogram preprocessing pipeline.

The per-stage transforms in :mod:`mbirjax.preprocess.utilities` are pure device-array *kernels* (the
math only).  This module owns the **single** copy of the batching + host<->device transfer +
concatenate scaffolding that was previously duplicated inside each of ``compute_sino_transmission`` /
``downsample_view_data`` / ``correct_det_rotation``.

Keeping that scaffolding in one place is the foundation for the next steps in the preprocessing
refactor (see ``experiments/sharding/plans/preprocessing_pipeline_refactor_plan.md``): fusing the
stages so the data stays on-device across them (one upload / one gather), and view-sharding the loop
across devices.
"""
import numpy as np
import jax.numpy as jnp


def map_view_batches(array, kernel, batch_size, desc=None):
    """Apply a per-batch device kernel across the leading (view) axis.

    Each contiguous batch of ``batch_size`` views is moved to the device (``jnp.array``), passed
    through ``kernel`` (a pure device-array -> device-array transform), and brought back to the host
    (``np.array``); the per-batch results are concatenated into one host NumPy array.  This bounds
    device memory to ``batch_size`` views while leaving the math entirely to ``kernel``.

    Args:
        array (numpy or jax array): data batched along axis 0 (views).
        kernel (callable): ``device_batch -> device_batch``; must not transfer to/from host itself.
        batch_size (int): number of views per batch.
        desc (str or None, optional): tqdm progress-bar label (the bar is always shown, matching the
            prior per-function loops; ``None`` shows an unlabeled bar).

    Returns:
        numpy.ndarray: concatenation of the per-batch kernel outputs along axis 0.
    """
    import tqdm
    num_views = array.shape[0]
    out = []
    for i in tqdm.tqdm(range(0, num_views, batch_size), desc=desc):
        batch = jnp.array(array[i:min(i + batch_size, num_views)])
        out.append(np.array(kernel(batch)))
    return np.concatenate(out, axis=0)
