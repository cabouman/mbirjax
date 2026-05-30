"""
pytest configuration for the mbirjax test suite.

What conftest.py is
───────────────────
`conftest.py` is a pytest *special* file, not a test module (it intentionally
contains no tests).  pytest auto-discovers it and applies it to every test file
in this directory and below — no import is needed for its setup to take effect.
It serves two roles here:

  1. Process setup that must happen *before any test imports JAX.*  conftest is
     the only reliable "runs first, before everything" hook pytest provides, so
     the XLA virtual-device flag is set here (see below).  A test that did
     `import jax` before the flag was set would lock JAX into 1 CPU device.
  2. Shared helpers for test files — `preferred_devices(n)` is importable from
     any test via `from conftest import preferred_devices`.

Every sharding test file (Phases A–F of the implementation plan) relies on this
file; none of them needs to repeat the device-flag setup or the device-pick
logic.

XLA device flag
───────────────
Sets XLA_FLAGS before JAX is first imported so that multi-device sharding tests
can run on CPU when no real GPUs are present.  The virtual device count uses the
same policy as mbirjax._device_setup (env override → macOS P-cores → Linux
affinity → fallback), capped, so behavior matches a normal `import mbirjax`.

This mirrors the logic in mbirjax._device_setup but is reproduced here so that
conftest.py works correctly regardless of which mbirjax installation (editable
source vs site-packages copy) Python resolves first, and so the flag is set even
before mbirjax is imported by a test.  The slight redundancy is intentional:
conftest guards the *test* process regardless of mbirjax import order, while
mbirjax._device_setup guards normal *library* use.

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
import sys

# Keep in sync with mbirjax/_device_setup.py DEFAULT_MAX_CPU_DEVICES.
DEFAULT_MAX_CPU_DEVICES = 8


def _performance_core_count():
    """macOS performance (P) core count via sysctl, or None if unavailable."""
    if sys.platform != "darwin":
        return None
    try:
        import subprocess
        out = subprocess.run(
            ["sysctl", "-n", "hw.perflevel0.physicalcpu"],
            capture_output=True, text=True, timeout=2.0,
        )
        n = int(out.stdout.strip())
        return n if n > 0 else None
    except Exception:
        return None


def _count_available_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))   # Linux — process CPU set
    except AttributeError:
        return os.cpu_count() or 1             # macOS / Windows fallback


def _resolve_num_cpu_devices() -> int:
    override = os.environ.get("MBIRJAX_NUM_CPU_DEVICES")
    if override:
        try:
            n = int(override)
            if n >= 1:
                return n
        except ValueError:
            pass
    cap = DEFAULT_MAX_CPU_DEVICES
    p_cores = _performance_core_count()
    if p_cores is not None:
        return min(p_cores, cap)
    return min(_count_available_cpus(), cap)


# Must be set before any JAX import.  setdefault leaves cluster-set values
# untouched.  The flag only affects the CPU backend; GPU machines ignore it.
os.environ.setdefault(
    "XLA_FLAGS",
    f"--xla_force_host_platform_device_count={_resolve_num_cpu_devices()}"
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
