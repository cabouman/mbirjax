"""
Test sharding infrastructure for fbp_filter using JAX virtual CPU devices.

JAX can simulate multiple independent devices on a single CPU via the
XLA_FLAGS environment variable.  This must be set before any JAX import,
so it lives at the very top of this file.

Run with:
    python fbp_sharding_sandbox.py

Expected output (all diffs should be 0.0 or float-epsilon):
    Test 0 (unsharded self-consistency): 0.0
    Test 1 (sharded fbp_filter vs baseline): <float epsilon>
    Test 2 (manual chunk vs baseline): 0.0
"""

import os
# Must be set before any JAX import to create virtual CPU devices.
# This gives us 4 independent TFRT_CPU devices that the full sharding
# machinery (NamedSharding, shard_map, etc.) treats as real devices.
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

import jax
import jax.numpy as jnp
import numpy as np
import mbirjax

N_SHARDS = 4
devices = jax.devices()
print(f"Devices ({len(devices)}): {devices}")
assert len(devices) == N_SHARDS, f"Expected {N_SHARDS} devices, got {len(devices)}"

# Use 64 detector rows so each shard gets 16 rows.
sino_shape = (180, 64, 64)
model = mbirjax.ParallelBeamModel(sino_shape, angles=np.linspace(0, np.pi, 180))

rng = jax.random.PRNGKey(42)
sino = jax.random.normal(rng, sino_shape)

# ------------------------------------------------------------------
# Test 0: unsharded baseline self-consistency
# ------------------------------------------------------------------
filtered_full = np.array(model.fbp_filter(sino))

save_output = False
if save_output:
    np.savez('fbp_sharding_baseline.npz', filtered=filtered_full)
    print("Baseline saved.")
else:
    data = np.load('fbp_sharding_baseline.npz')
    print('Test 0 (unsharded self-consistency):',
          np.max(np.abs(filtered_full - data['filtered'])))  # expect 0.0

# ------------------------------------------------------------------
# Test 1: sharded fbp_filter via configure_sharding / _maybe_shard
# ------------------------------------------------------------------
# Enable sharding: creates a 1-axis mesh over the 4 virtual devices
# and sets main_device = sinogram_device = None so _device_put is a no-op.
model.configure_sharding(devices)

sino_sharded = model._maybe_shard(sino, axis=1)
print(f"\nSinogram sharding: {sino_sharded.sharding}")
# Each of the 4 devices holds sino[:, 0:16, :], sino[:, 16:32, :], etc.

filtered_sharded = model.fbp_filter(sino_sharded)
print(f"Filtered sharding: {filtered_sharded.sharding}")
# fbp_filter uses shard_map internally when self.mesh is set, so each device
# receives a contiguous local shard before the FFT runs.  NamedSharding + SPMD
# alone fails here because XLA's CPU FFT thunk requires row-major layout and
# a 3D array sharded along its middle axis presents strided buffers.

filtered_gathered = np.array(model._maybe_gather(filtered_sharded, axis=1))
print('Test 1 (sharded fbp_filter vs baseline):',
      np.max(np.abs(filtered_gathered - data['filtered'])))  # expect 0.0 or float epsilon

# ------------------------------------------------------------------
# Test 2: manual chunk simulation (no JAX sharding machinery)
# Confirms fbp_filter is truly slice-parallel, independent of sharding infra.
# ------------------------------------------------------------------
model.configure_sharding(None)  # restore single-device mode

shard_size = sino_shape[1] // N_SHARDS
chunk_results = []
for i in range(N_SHARDS):
    chunk = sino[:, i * shard_size:(i + 1) * shard_size, :]
    chunk_results.append(np.array(model.fbp_filter(chunk)))
filtered_chunked = np.concatenate(chunk_results, axis=1)

print('Test 2 (manual chunk vs baseline):',
      np.max(np.abs(filtered_chunked - data['filtered'])))  # expect 0.0
