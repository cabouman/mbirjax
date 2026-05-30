"""
mbirjax._sharding.transfer
──────────────────────────
Safe cross-device data movement.

Why this exists
───────────────
On some GPUs (verified: NVIDIA L40S, JAX 0.10.1) `jax.device_put` of a
*device-resident* array to another device silently produces zeros on the
destination — no error is raised.  On others (verified: NVIDIA H100) the same
operation is correct.  Reading a shard to host with np.asarray is always safe,
and device_put of a *host* (numpy) array is always safe.

So all cross-device movement in the sharding code goes through move_shard(),
which picks its path from a single boolean (is the direct device-to-device copy
safe on this hardware?).  That boolean is determined once, empirically, by
is_dev2dev_safe() — we probe the actual hardware rather than maintaining a
hard-coded list of good/bad GPU models, so the code auto-adapts to new hardware
and to the eventual upstream fix.

See experiments/sharding/parallel_performance/device_put_check.py for the
standalone probe that established this behavior on H100 vs L40S.
"""

import warnings

import numpy as np
import jax
import jax.numpy as jnp

# Set to True after the first time we fall back to a host bounce, so the
# (informational) warning is emitted only once per process.
_warned_host_bounce = False


def is_dev2dev_safe(devices) -> bool:
    """Empirically test whether direct device-to-device device_put is correct.

    Moves a small known array from devices[0] to devices[1] and checks that the
    value survives.  On hardware where the device_put corruption bug is present
    (e.g. L40S), the destination reads back as zeros and this returns False.

    Args:
        devices (sequence): the devices that will participate in sharding.

    Returns:
        bool: True if a single device is given (nothing to move), or if the
            dev0 -> dev1 copy round-trips correctly; False if it corrupts.
    """
    devices = list(devices)
    if len(devices) < 2:
        return True  # single device: no cross-device copy ever happens

    # Distinctive nonzero pattern so corruption-to-zero is unmistakable.
    probe = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    src = jax.device_put(jnp.asarray(probe), devices[0])
    dst = jax.device_put(src, devices[1])          # the operation under test
    return bool(np.array_equal(np.asarray(dst), probe))


def move_shard(x, target_device, dev2dev_safe=True):
    """Place array x on target_device, choosing a hardware-safe path.

    Args:
        x: a JAX array (possibly resident on another device) or a numpy array.
        target_device: the device to place x on.
        dev2dev_safe (bool): the cached result of is_dev2dev_safe() for this
            mesh.  When True we copy directly; when False we route through host
            memory (read to numpy, then device_put), which is always correct but
            costs a host round-trip.

    Returns:
        A JAX array resident on target_device.
    """
    if dev2dev_safe:
        return jax.device_put(x, target_device)

    # Host-bounce fallback.  Reading a shard to host is always safe; device_put
    # of a host array is always safe.  Warn once so the degraded path is visible.
    global _warned_host_bounce
    if not _warned_host_bounce:
        _warned_host_bounce = True
        warnings.warn(
            "Direct device-to-device transfer is unsafe on this hardware "
            "(device_put corruption detected); routing cross-device transfers "
            "through host memory.  This is correct but slower.  See "
            "mbirjax._sharding.transfer for details.",
            stacklevel=2,
        )
    return jax.device_put(np.asarray(x), target_device)
