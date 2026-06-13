"""
mbirjax._sharding.thread_execution
──────────────────────────────────
Per-device threaded execution and sharded-array assembly.

The sharding scheme (Path G) runs one Python thread per device.  Each thread
sets its JAX default device and computes that device's local result; the results
(each already resident on its own device) are then assembled into a single
logically-global jax.Array via make_array_from_single_device_arrays — no extra
data movement.

These two steps are kept as separate functions so they compose: run_per_device
owns the fan-out; assemble_sharded owns the assembly.  A caller that wants the
per-device results without assembling them (or wants to assemble results it
produced some other way) can use either alone.
"""

import contextlib
from concurrent.futures import ThreadPoolExecutor

import jax


@contextlib.contextmanager
def device_pool(n):
    """A reusable thread pool for repeated run_per_device calls.

    Yields a ``ThreadPoolExecutor`` with ``n`` workers and closes it on exit.
    Pass the yielded pool as ``run_per_device(..., executor=pool)`` so a loop of
    many per-device fan-outs (e.g. the slice-band streaming in sharded back
    projection, which calls run_per_device once per band) reuses one pool instead
    of creating and tearing down a fresh ``ThreadPoolExecutor`` per call.

    Thread reuse is safe: each task sets its own ``jax.default_device`` (see
    run_per_device), so there is no thread-to-device affinity to preserve.

    Args:
        n (int): number of worker threads (typically the device count).

    Yields:
        ThreadPoolExecutor: the pool to hand to run_per_device.
    """
    executor = ThreadPoolExecutor(max_workers=n)
    try:
        yield executor
    finally:
        executor.shutdown(wait=True)


def run_per_device(devices, worker_fn, executor=None):
    """Run worker_fn once per device, each in its own thread on that device.

    Each thread runs under `jax.default_device(device)` (which is thread-local
    in JAX, so the threads do not interfere), calls worker_fn(i, device), and
    returns its result.  Results are returned in device order (result[i]
    corresponds to devices[i]), not completion order.

    IMPORTANT: this function does NOT call block_until_ready on the results.
    JAX dispatch is asynchronous; leaving the results unblocked lets a caller
    overlap the next batch's data transfer with the current batch's compute.
    Callers that need the values materialized (e.g. before assembling or reading)
    should block explicitly.

    Args:
        devices (sequence): devices to run on; one thread per device.
        worker_fn (callable): worker_fn(i, device) -> result, where i is the
            index into devices and device is devices[i].
        executor (ThreadPoolExecutor, optional): a pool to submit to (e.g. from
            device_pool()).  When given, it is reused and NOT closed -- this
            avoids creating a fresh pool on every call in a tight loop.  When None
            (default), a private pool is created and closed for this one call, so
            existing single-shot callers are unchanged.

    Returns:
        list: results in device order.
    """
    devices = list(devices)
    n = len(devices)

    def _run_on_device(i):
        with jax.default_device(devices[i]):
            return worker_fn(i, devices[i])

    if executor is not None:
        return list(executor.map(_run_on_device, range(n)))
    # max_workers == n so every device gets a thread and they run concurrently.
    with ThreadPoolExecutor(max_workers=n) as pool:
        return list(pool.map(_run_on_device, range(n)))


def assemble_sharded(per_device_arrays, global_shape, sharding):
    """Assemble per-device arrays into one logically-global sharded jax.Array.

    Thin wrapper over jax.make_array_from_single_device_arrays.  Each entry of
    per_device_arrays must already be resident on the device that owns its shard
    under `sharding`; no data is moved.

    Args:
        per_device_arrays (sequence): one on-device array per device shard.
        global_shape (tuple): shape of the assembled global array.
        sharding (jax.sharding.Sharding): the target sharding (e.g. a
            NamedSharding over the mesh).

    Returns:
        jax.Array: the assembled global array with the given sharding.
    """
    return jax.make_array_from_single_device_arrays(
        global_shape, sharding, list(per_device_arrays)
    )
