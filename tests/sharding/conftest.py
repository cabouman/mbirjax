"""
pytest configuration for the sharding test subpackage (tests/sharding/).

The parent ``tests/conftest.py`` sets the XLA virtual-device flag *before any JAX
import* (load-bearing for the whole suite) and is loaded first, hierarchically,
so these tests inherit it.  This file adds the one helper the sharding tests
share — ``preferred_devices(n)`` — importable via ``from conftest import
preferred_devices`` (pytest puts this directory on sys.path, so ``conftest``
resolves here).
"""
import jax


def preferred_devices(n: int):
    """Return a list of n devices for sharding tests, real GPUs preferred.

    Prefers real GPUs over virtual CPU devices, so sharding tests exercise real
    hardware on a GPU cluster and fall back to virtual CPUs on a laptop; the tests
    are identical either way.

    Returns None if fewer than n GPUs are available when there is at least one
    GPU, and None if fewer than n CPUs are available and no GPUs are present.
    """
    try:
        gpus = jax.devices('gpu')
        if len(gpus) >= n:
            return gpus[:n]
        return None
    except RuntimeError:
        pass
    cpus = jax.devices('cpu')
    if len(cpus) >= n:
        return cpus[:n]
    return None
