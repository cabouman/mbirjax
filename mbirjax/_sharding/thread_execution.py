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

from concurrent.futures import ThreadPoolExecutor

import jax


def run_per_device(devices, worker_fn):
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

    Returns:
        list: results in device order.
    """
    devices = list(devices)
    n = len(devices)

    def _run_on_device(i):
        with jax.default_device(devices[i]):
            return worker_fn(i, devices[i])

    # max_workers == n so every device gets a thread and they run concurrently.
    with ThreadPoolExecutor(max_workers=n) as executor:
        results = list(executor.map(_run_on_device, range(n)))
    return results


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
