"""
pytest configuration for the mbirjax test suite.

What conftest.py is
───────────────────
`conftest.py` is a pytest *special* file, not a test module (it intentionally
contains no tests).  pytest auto-discovers it and applies it to every test file
in this directory and below — no import is needed for its setup to take effect.
Its role here is process setup that must happen *before any test imports JAX*:
conftest is the only reliable "runs first, before everything" hook pytest
provides, so the XLA virtual-device flag is set here (see below).  A test that
did `import jax` before the flag was set would lock JAX into 1 CPU device.

This applies hierarchically to every test in tests/ and below, including the
sharding subpackage (tests/sharding/) — whose own conftest adds the
`preferred_devices(n)` device-picker that the sharding tests share.

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

# Matches mbirjax/_device_setup.py's DEFAULT_MAX_CPU_DEVICES (2): with auto-sharding
# on by default, every bare-model test already exercises the multi-device sharded path
# at 2 devices, and capping at 2 keeps the suite time acceptable (8 virtual devices ran
# the legacy suite ~1.9x slower on tiny, overhead-bound test problems).  The 4/8-device
# sweep legs in tests/sharding/ skip at this cap -- raise the env override for a fuller
# sweep (e.g. MBIRJAX_NUM_CPU_DEVICES=4 pytest tests/sharding/).  The resolution policy
# below otherwise mirrors mbirjax._device_setup.
DEFAULT_MAX_CPU_DEVICES = 2


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

# Quiet the benign jaxlib C++ chatter in the TEST process.  This conftest imports jax
# below -- BEFORE any test imports mbirjax -- so mbirjax._device_setup's own
# TF_CPP_MIN_LOG_LEVEL setdefault comes too late here, and pytest runs on GPU show the
# harmless VMM "CUDA_ERROR_NOT_PERMITTED ... will retry with simpler handle types"
# W-lines that normal library use hides.  Level 2 drops INFO+WARNING, keeps ERROR.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Don't preallocate a GPU memory pool for the test process.  The default behavior grabs
# 75% of EVERY visible GPU at backend init and, when a device is partially occupied
# (another job / leftover allocation), logs an ERROR-level back-off sequence
# ("Failed to allocate 59.38GiB ... 53.45GiB ...", each retry 0.9x the last) before
# succeeding with whatever fits -- alarming noise in a green run, and impolite on shared
# nodes.  Tests are small and time nothing, so on-demand allocation costs them nothing.
# setdefault keeps this overridable for deliberate memory experiments.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

# NOTE on jit-compilation caching (measured 2026-06-11, so nobody re-attempts it):
# mbirjax already enables JAX's persistent compilation cache at import
# (tomography_model.py sets jax_compilation_cache_dir = /tmp/jax_cache), and it is
# nearly irrelevant to suite time anyway -- a true cold run (cache cleared) is only
# ~3 s slower than warm (161 vs 158 s; just 12 programs exceed even a 0.25 s XLA
# compile).  The per-model first-call cost that LOOKS like compilation is mostly
# Python-side TRACING/lowering of each fresh model's jitted closures, which no
# persistent cache can skip.  Wall-clock levers that do work: fewer/cheaper recon
# iterations (done) and pytest-xdist parallelism.


def preferred_devices(n: int):
    """Return a list of n devices for sharding tests.

    Prefers real GPUs over virtual CPU devices so that sharding tests exercise
    real hardware on a GPU cluster and fall back to virtual CPUs on a laptop.

    Returns None if fewer than n GPUs are available when there is at least one
    GPU and return None if fewer than n CPUs are available and no GPUs are available.
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
