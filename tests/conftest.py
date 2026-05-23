"""
pytest configuration for the mbirjax test suite.

Sets XLA_FLAGS before JAX is first imported so that multi-device sharding tests
can run on CPU when no real GPUs are present.  setdefault is used so that an
environment that already sets XLA_FLAGS (e.g. a GPU cluster) is not overridden.

Device-preference policy for sharding tests
───────────────────────────────────────────
Use preferred_devices(n) in test setUp/setUpClass to pick devices:

  1. Real GPUs (jax.devices('gpu')) — used when ≥n GPUs are available.
  2. Virtual CPU devices (jax.devices('cpu')) — fallback guaranteed by XLA_FLAGS.

This means sharding tests automatically exercise real hardware on a GPU cluster
and fall back to CPU on a laptop or CI machine.  The tests themselves are
identical in either case.

Note: --xla_force_host_platform_device_count=4 only affects the CPU backend;
it does not alter GPU device count or any GPU-using test.
"""
import os

# Must be set before any test module imports jax.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

# Import jax here (after env var) so preferred_devices is ready immediately.
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
