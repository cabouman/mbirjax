"""
GPU multi-device sanity test.

Checks three things independently, without any FBP or CT code:

  Test 1 — device_put
    Explicitly place a persistent array on each GPU and read memory stats while
    the arrays are still live.  Confirms JAX can address GPU 1 at all.

  Test 2 — default_device in threads
    Each thread calls jax.default_device(gpu_i), allocates and runs a simple JIT'd
    function, then holds the result on-device while memory stats are sampled.
    Confirms threading + default_device dispatches to the right GPU.

  Test 3 — parallel speedup
    Run a compute-heavy JIT'd function sequentially on GPU 0 twice, then in parallel
    on GPU 0 and GPU 1 via threads.  Expected: parallel ≈ sequential / 2.
    Confirms the two GPUs can run concurrently.
"""

import os

# ── Imports ───────────────────────────────────────────────────────────────────

import jax
import jax.numpy as jnp
import numpy as np
import concurrent.futures
import time

# ── Helpers ───────────────────────────────────────────────────────────────────

def mem_gb(device):
    """Current JAX pool bytes_in_use for a device, in GB.  Returns -1 on error."""
    try:
        return device.memory_stats()['bytes_in_use'] / 1024**3
    except Exception:
        return -1.0


def print_mem(label, devices):
    parts = "  ".join(f"{d}: {mem_gb(d):.4f} GB" for d in devices)
    print(f"  [{label}]  {parts}")


# ── Devices ───────────────────────────────────────────────────────────────────

all_devices = jax.devices()
print(f"Available devices ({len(all_devices)}): {all_devices}")

gpus = [d for d in all_devices if 'cpu' not in d.device_kind.lower()]
if len(gpus) < 2:
    print(f"\nOnly {len(gpus)} GPU(s) found — tests requiring 2 GPUs will be skipped.")
    n_gpus = len(gpus)
else:
    n_gpus = 2
    gpus = gpus[:2]

# ═══════════════════════════════════════════════════════════════════════════════
# Test 1 — device_put
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("Test 1: explicit device_put to each GPU")
print("=" * 60)

ARRAY_MB = 256   # size of each test array in MB
n_floats = ARRAY_MB * 1024**2 // 4
arrays = []

print_mem("before", gpus)

for i, gpu in enumerate(gpus):
    x = jax.device_put(jnp.ones(n_floats, dtype=jnp.float32), gpu)
    jax.block_until_ready(x)
    arrays.append(x)
    print_mem(f"after gpu{i} alloc", gpus)

# All arrays live — memory should reflect each device.
print_mem("all live", gpus)

# Release and confirm drop.
del arrays
jax.effects_barrier()   # flush any pending deallocations
print_mem("after del", gpus)

# ═══════════════════════════════════════════════════════════════════════════════
# Test 2 — default_device in threads
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("Test 2: jax.default_device inside threads")
print("=" * 60)

@jax.jit
def simple_compute(x):
    """A trivial JIT'd function whose result stays on-device until we say so."""
    return jnp.sum(x ** 2)

# Warmup on each device sequentially so compilation doesn't skew the parallel test.
for gpu in gpus:
    with jax.default_device(gpu):
        _ = simple_compute(jnp.ones(n_floats))

print_mem("before threads", gpus)

held_results = [None] * n_gpus
hold_events = [concurrent.futures.Future() for _ in range(n_gpus)]
release_events = [concurrent.futures.Future() for _ in range(n_gpus)]

def hold_on_device(i):
    """Allocate, compute, hold result on device, then wait for a release signal."""
    with jax.default_device(gpus[i]):
        x = jnp.ones(n_floats, dtype=jnp.float32)
        result = simple_compute(x)
        jax.block_until_ready(result)
        held_results[i] = result        # keep reference → memory stays allocated
    hold_events[i].set_result(True)     # signal: memory is live, sample now
    release_events[i].result()          # wait for main thread to say "ok, release"
    held_results[i] = None              # release

with concurrent.futures.ThreadPoolExecutor(max_workers=n_gpus) as executor:
    futures = [executor.submit(hold_on_device, i) for i in range(n_gpus)]

    # Wait until both threads have data on their devices.
    for ev in hold_events:
        ev.result()

    print_mem("while results held on device", gpus)

    # Let threads finish.
    for ev in release_events:
        ev.set_result(True)

    # Collect results.
    for f in futures:
        f.result()

print_mem("after threads finish", gpus)

# ═══════════════════════════════════════════════════════════════════════════════
# Test 3 — parallel speedup
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("Test 3: sequential vs parallel compute")
print("=" * 60)

# A moderately compute-intensive function.
MAT = 4096
rng = np.random.default_rng(0)
mat_np = rng.standard_normal((MAT, MAT)).astype(np.float32)

@jax.jit
def compute_heavy(x):
    # Several matrix multiplications to give the GPU something to do.
    for _ in range(8):
        x = jnp.dot(x, x.T) / MAT
    return x

# Warmup on both devices.
for gpu in gpus:
    with jax.default_device(gpu):
        _ = compute_heavy(jnp.array(mat_np))
for gpu in gpus:
    with jax.default_device(gpu):
        jax.block_until_ready(compute_heavy(jnp.array(mat_np)))

# Sequential: run on GPU 0 twice in a row.
N_RUNS = 3
times_seq = []
for _ in range(N_RUNS):
    t0 = time.time()
    for gpu in gpus:
        with jax.default_device(gpu):
            r = compute_heavy(jnp.array(mat_np))
            jax.block_until_ready(r)
    times_seq.append(time.time() - t0)

t_seq = min(times_seq)
print(f"  Sequential (gpu0 then gpu1): {t_seq:.3f} s")

# Parallel: both devices simultaneously via threads.
def run_gpu(i):
    with jax.default_device(gpus[i]):
        r = compute_heavy(jnp.array(mat_np))
        jax.block_until_ready(r)

times_par = []
for _ in range(N_RUNS):
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_gpus) as executor:
        list(executor.map(run_gpu, range(n_gpus)))
    times_par.append(time.time() - t0)

t_par = min(times_par)
speedup = t_seq / t_par
print(f"  Parallel   (gpu0 || gpu1):   {t_par:.3f} s")
print(f"  Speedup: {speedup:.2f}x  (ideal: {n_gpus:.1f}x)")

if n_gpus >= 2:
    if speedup > 1.5:
        print("  ✓ Both GPUs are running concurrently.")
    elif speedup > 1.1:
        print("  ~ Partial parallelism — some overlap but not full concurrency.")
    else:
        print("  ✗ No parallelism detected — GPUs appear to be serialised.")

print()
