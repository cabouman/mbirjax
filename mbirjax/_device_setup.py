"""
mbirjax._device_setup
─────────────────────
Runs automatically when mbirjax is imported.  Must be imported as the very
first line of mbirjax/__init__.py, before any JAX import reaches this process.

What this module does
─────────────────────
On CPU-only machines it sets XLA_FLAGS so JAX sees a sensible number of virtual
CPU devices, enabling multi-device sharding (and sharding tests) without real
GPUs.  On GPU machines the flag is set but ignored by the GPU backend, so it is
always safe to apply.

If JAX was already imported before mbirjax (only a problem on CPU-only
machines), a warning is issued with the exact environment variable to set.

Choosing the virtual device count
─────────────────────────────────
Empirically (Apple M3 Max, 10 P + 4 E cores) throughput on CPU plateaus and
then regresses past ~6-8 devices: work is split evenly, so the slow efficiency
cores and memory-bandwidth saturation make extra devices a net loss.  So the
default is capped (DEFAULT_MAX_CPU_DEVICES) rather than "one per core".

Resolution order (first that applies wins):
  1. Env var MBIRJAX_NUM_CPU_DEVICES - explicit user/cluster override.
  2. macOS: min(performance_cores, cap)   [P-cores via sysctl]
  3. Linux: min(len(sched_getaffinity(0)), cap)   [respects job allocation]
  4. Fallback: min(os.cpu_count(), cap)
"""

import os
import sys
import warnings
from typing import Optional

# Default upper bound on virtual CPU devices.  See module docstring: throughput
# plateaus/regresses past this on tested hardware.  Override with the env var
# MBIRJAX_NUM_CPU_DEVICES for deliberate performance tuning.
DEFAULT_MAX_CPU_DEVICES = 2


def _performance_core_count() -> Optional[int]:
    """Return the number of macOS performance (P) cores, or None if unknown.

    On Apple Silicon, sysctl hw.perflevel0.physicalcpu reports the P-core count
    (perflevel1 is the slower E-cores).  Splitting work across the E-cores hurts
    because every device gets an equal share and the E-cores straggle, so we
    prefer to size by P-cores on macOS.  Returns None on non-macOS or if the
    sysctl is unavailable, so the caller can fall back.
    """
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
    """Return the number of CPU cores available to this process.

    On Linux, os.sched_getaffinity(0) returns the set of logical CPUs the
    process is actually allowed to use, respecting taskset, cgroup CPU quotas,
    and SLURM --cpus-per-task.  This is the right number inside containers and
    cluster jobs.

    On macOS (and as a fallback), os.cpu_count() returns the total logical core
    count.  On Apple Silicon there is no hyperthreading, so logical == physical.
    """
    try:
        return len(os.sched_getaffinity(0))   # Linux - process CPU set
    except AttributeError:
        return os.cpu_count() or 1             # macOS / Windows fallback


def _resolve_num_cpu_devices() -> int:
    """Decide how many virtual CPU devices to request (see module docstring)."""
    # 1. Explicit override.
    override = os.environ.get("MBIRJAX_NUM_CPU_DEVICES")
    if override:
        try:
            n = int(override)
            if n >= 1:
                return n
        except ValueError:
            warnings.warn(
                f"MBIRJAX_NUM_CPU_DEVICES={override!r} is not a positive "
                f"integer; ignoring.", stacklevel=2,
            )

    cap = DEFAULT_MAX_CPU_DEVICES

    # 2. macOS: size by performance cores.
    p_cores = _performance_core_count()
    if p_cores is not None:
        return min(p_cores, cap)

    # 3/4. Linux affinity set, or generic fallback.
    return min(_count_available_cpus(), cap)


def _setup_devices() -> None:
    """Configure JAX virtual CPU devices, or warn if the window has passed.

    Key insight: --xla_force_host_platform_device_count only affects the CPU
    backend.  JAX's GPU backend is a completely separate runtime and is
    unaffected by this flag, so setting it is always safe regardless of whether
    GPUs are present.

    os.environ.setdefault() is used so that any value already set by the user or
    by a cluster job scheduler is left untouched.
    """
    n_devices = _resolve_num_cpu_devices()
    flag = f"--xla_force_host_platform_device_count={n_devices}"

    if "jax" in sys.modules:
        # JAX has already initialised its backends.  Setting XLA_FLAGS now has
        # no effect.  On GPU machines this is harmless; on CPU-only machines the
        # user is stuck with however many virtual devices JAX created (usually 1).
        import jax
        try:
            if jax.devices("gpu"):
                return   # GPU backend found - nothing to do
        except RuntimeError:
            pass

        # CPU-only machine and JAX already imported.  Warn only if the user did
        # not already configure the virtual device count themselves.
        if "xla_force_host_platform_device_count" not in os.environ.get("XLA_FLAGS", ""):
            n_current = len(jax.devices("cpu"))
            warnings.warn(
                f"mbirjax was imported after JAX on a CPU-only system. "
                f"JAX has already initialised with {n_current} CPU device(s); "
                f"{n_devices} could be available. "
                f"For optimal multi-device performance, import mbirjax before "
                f"JAX, or set this environment variable before any Python "
                f"import:\n\n"
                f"    XLA_FLAGS='{flag}'\n",
                stacklevel=2,
            )
        return

    # JAX not yet imported - safe to set the flag now.
    os.environ.setdefault("XLA_FLAGS", flag)


def _quiet_benign_xla_logs() -> None:
    """Hide jaxlib's benign multi-GPU allocator chatter by raising its C++ log level.

    On the first multi-GPU allocation XLA's VMM allocator probes advanced
    memory-handle types (FABRIC / POSIX_FD) for fast inter-GPU sharing.  In
    environments that forbid them (most single-node jobs / containers) it logs a
    scary-looking but HARMLESS warning -- ``cuMemCreate with FABRIC+POSIX_FD handle
    types failed: CUDA_ERROR_NOT_PERMITTED; will retry with simpler handle types``
    -- and then succeeds with a fallback handle.  These are C++ WARNING-level logs;
    ``TF_CPP_MIN_LOG_LEVEL=2`` drops INFO+WARNING while keeping ERROR/FATAL (and all
    Python tracebacks / ``warnings.warn`` messages, which use a different path), so
    real failures still surface.

    Set via ``setdefault`` so it is overridable: export ``TF_CPP_MIN_LOG_LEVEL=0``
    (or ``1``) before importing mbirjax to get the full jaxlib logs back.  Like the
    device flag above, this only takes effect if applied BEFORE jax is imported
    (the value is read once at jaxlib init); if jax is already imported it is a
    harmless no-op.
    """
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


# ── Device accessors (single source of truth for jax.devices) ─────────────────
# The rest of mbirjax queries the available devices through these helpers instead
# of calling jax.devices() ad hoc, so device discovery lives in ONE place.  JAX
# already caches its backend's device list per process, so these add naming and a
# single chokepoint, not a second cache.  jax is imported lazily (inside the
# functions) so importing this module never initialises a JAX backend -- the XLA
# device-count flag set above must take effect BEFORE the first jax import.

def gpu_devices():
    """All GPU devices as a tuple, or () when there is no GPU backend."""
    import jax
    try:
        return tuple(jax.devices("gpu"))
    except RuntimeError:
        return ()


def cpu_devices():
    """All (possibly virtual) CPU devices as a tuple."""
    import jax
    return tuple(jax.devices("cpu"))


def default_devices():
    """The default-platform devices: GPUs if any are present, otherwise CPUs.

    Mirrors ``jax.devices()`` (no argument), which returns the highest-priority
    available backend's devices -- GPU when a GPU backend exists, else CPU.
    Returned as a list so callers can index/slice it directly.
    """
    return list(gpu_devices()) or list(cpu_devices())


def get_device_platform(device):
    """Uppercase platform name for a JAX device: ``'CPU'``, ``'GPU'``, or ``'TPU'``.

    Reads ``device.platform`` (JAX reports ``'gpu'`` for both CUDA and ROCm), so this is the
    single place mbirjax turns a device into a human platform name -- prefer it over ad-hoc
    ``device.platform == 'gpu'`` compares or ``device.device_kind`` sniffing.
    """
    return {'cpu': 'CPU', 'tpu': 'TPU'}.get(device.platform, 'GPU')


def get_platform():
    """Platform the default run will use: ``get_device_platform(default_devices()[0])``.

    ``'GPU'`` when a GPU backend is present, else ``'CPU'``.  The zero-argument counterpart
    to :func:`get_device_platform` for "what platform am I on right now."
    """
    return get_device_platform(default_devices()[0])


def _disable_tf32_matmul_default() -> None:
    """Default float32 matmuls to FULL float32 precision (opt out of TF32).

    On Ampere/Hopper GPUs, XLA lowers float32 matmul/tensordot (dot_general) to TF32 tensor
    cores BY DEFAULT: the multiply INPUTS are rounded to a 10-bit mantissa (~1e-3 relative,
    vs float32's ~1e-7; accumulation stays float32).  mbirjax is a quantitative library whose
    correctness gates sit at ~1e-5 with a measured ~5e-6 cross-platform floor, so a silent
    TF32 dot is a DIFFERENT OPERATOR, not a faster implementation -- and it would make GPU
    and CPU results diverge at the 1e-3 level.  The projector kernels contain no dots, but
    denoising.py (step-size tensordots), hsnt.py (subspace rehydration matmul), and vcls.py
    (Gram matrices) do, and any future matmul would silently inherit TF32.

    setdefault, per this module's convention: an environment value set by the user or job
    script wins, and code that deliberately wants TF32 can still use a local
    ``jax.default_matmul_precision(...)`` context, which overrides the global default.
    JAX reads this env var at import, so it must be set before JAX initialises (this module
    is imported first, and _setup_devices already warns when JAX beat us here).
    """
    os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "float32")


_quiet_benign_xla_logs()
_disable_tf32_matmul_default()
_setup_devices()
