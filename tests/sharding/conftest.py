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
import jax.numpy as jnp
import numpy as np


def rel_max_err(out, ref):
    """Worst-element error as a fraction of the array's scale: max|out - ref| / max|ref|.

    The SCALE-INVARIANT gate for "sharded == single-device to float noise".  Prefer this over a
    fixed ``atol`` or a per-element ``rtol`` for any sharded-vs-reference comparison, because:

      - The sharded and single-device results differ only by float SUMMATION ORDER (the
        reduce-scatter / all-gather sum partials per device, then across devices, in a different
        order than the monolithic sum).  Measured on CPU this difference is PROCESS-
        NONDETERMINISTIC: usually exactly 0, occasionally ~1e-7 of the PEAK magnitude (identical
        across device counts within a process -> a per-process XLA reduction-order / autotune
        choice, not per-call reorder).  So the noise floor scales with the PEAK, not with a fixed
        absolute constant.
      - A fixed ``atol`` is therefore calibrated to one value scale: it silently passes a small-
        magnitude operator and FALSE-FAILS (or flakes on) a large-magnitude one for the SAME
        relative noise -- e.g. a Hessian diagonal (coeff_power=2 squares the coefficients, peak
        ~1e3-1e4) vs a back projection (peak ~1e2) vs a tiny test operator (peak ~1).
      - A per-element ``rtol`` (atol=0) is worse still: it divides by each element's OWN value, so
        a near-zero entry gets a near-zero threshold while its noise is set by the SIGNAL scale
        (the residue of large terms that cancelled) -> the relative diff explodes on near-zero
        entries.

    ``max|out-ref| / max|ref|`` clears the worst-case noise (~1e-7 of peak) at TOL=1e-5 with ~100x
    margin, while still catching a real sharding bug (which perturbs many elements by O(signal), so
    rel_max ~ O(1)).  Returns max|out| when ref is all-zero (any nonzero output is then an error).
    """
    out = np.asarray(out)
    ref = np.asarray(ref)
    denom = float(np.max(np.abs(ref)))
    if denom == 0.0:
        return float(np.max(np.abs(out)))
    return float(np.max(np.abs(out - ref)) / denom)


def assert_sharded_allclose(out, ref, msg="", tol=1e-5):
    """Assert a sharded result matches its single-device reference to float noise, via the
    SCALE-INVARIANT gate ``rel_max_err(out, ref) <= tol`` (see :func:`rel_max_err` for why a fixed
    ``atol`` / per-element ``rtol`` is the wrong ruler for reduce-scatter reorder noise).  Use
    tol=1e-5 for single-shot projectors, 1e-4 for an iterated VCD recon.  Raises AssertionError
    carrying the measured error.  Reserve plain ``assert_allclose``/exact-equality for the OTHER
    invariants -- constructed-zero (padded entries == 0) and data-movement (gather/scatter/halo
    round trips) -- where a relative gate would mask the very corruption the test exists to catch.
    """
    err = rel_max_err(out, ref)
    if err > tol:
        prefix = f"{msg} -- " if msg else ""
        raise AssertionError(
            f"{prefix}sharded vs single-device rel_max_err={err:.2e} > tol={tol:.1e}")


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


# ---------------------------------------------------------------------------
# Shared sharded-projector test helpers.
#
# These three were byte-identical across the three per-geometry sharded test
# modules (test_cone_sharded / test_multiaxis_sharded / test_translation_sharded)
# and are lifted here so the merged tests/sharding/test_geometry_sharded.py can
# import them once.  They are geometry-NEUTRAL: _usable_device_counts drives the
# device sweep off the model's own sinogram_shard_axis()/recon_shard_axis(), so it
# works for cone, multiaxis, and translation without change.
# ---------------------------------------------------------------------------
def _random_sino(model, seed=0):
    shape = model.get_params('sinogram_shape')
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.standard_normal(shape, dtype=np.float32))


def _random_recon(model, seed=1):
    shape = tuple(int(x) for x in model.get_params('recon_shape'))
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.standard_normal(shape, dtype=np.float32))


def _usable_device_counts(model):
    """Available device counts > 1 (from conftest) that divide both sharded axes."""
    sino_shape = model.get_params('sinogram_shape')
    recon_shape = model.get_params('recon_shape')
    sino_axis = model.sinogram_shard_axis() % len(sino_shape)
    recon_axis = model.recon_shard_axis() % len(recon_shape)
    counts = []
    for n in (2, 4):
        devs = preferred_devices(n)
        if devs is None:
            continue
        if sino_shape[sino_axis] % n == 0 and recon_shape[recon_axis] % n == 0:
            counts.append((n, devs))
    return counts
