"""
pytest configuration for the mbirjax test suite.

Sets XLA_FLAGS before JAX is first imported so that multi-device sharding tests
can run on CPU when no real GPUs are present.  The virtual device count is set
to the number of CPU cores actually available to this process (using
sched_getaffinity on Linux; cpu_count on macOS) rather than a hardcoded value.

This mirrors the logic in mbirjax._device_setup but is reproduced here so that
conftest.py works correctly regardless of which mbirjax installation (editable
source vs site-packages copy) Python resolves first.

Device-preference policy for sharding tests
───────────────────────────────────────────
Use preferred_devices(n) in test setUp/setUpClass to pick devices:

  1. Real GPUs (jax.devices('gpu')) — used when ≥n GPUs are available.
  2. Virtual CPU devices (jax.devices('cpu')) — fallback set by XLA_FLAGS below.

Sharding tests automatically exercise real hardware on a GPU cluster and fall
back to virtual CPUs on a laptop or CI machine.  The tests are identical either
way.
"""
import os


def _count_available_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))   # Linux — process CPU set
    except AttributeError:
        return os.cpu_count() or 1             # macOS / Windows fallback


# Must be set before any JAX import.  setdefault leaves cluster-set values
# untouched.  The flag only affects the CPU backend; GPU machines ignore it.
os.environ.setdefault(
    "XLA_FLAGS",
    f"--xla_force_host_platform_device_count={_count_available_cpus()}"
)

import jax


def preferred_devices(n: int):
    """Return a list of n devices for sharding tests.

    Prefers real GPUs over virtual CPU devices so that sharding tests exercise
    real hardware on a GPU cluster and fall back to virtual CPUs on a laptop.

    Returns None if fewer than n devices of any kind are available (the caller
    should raise unittest.SkipTest in that case).
    """
    try:
        gpus = jax.devices('gpu')
        if len(gpus) >= n:
            return gpus[:n]
    except RuntimeError:
        pass
    cpus = jax.devices('cpu')
    if len(cpus) >= n:
        return cpus[:n]
    return None
