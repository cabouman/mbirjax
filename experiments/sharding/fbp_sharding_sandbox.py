"""
Test sharding infrastructure for fbp_filter using JAX virtual CPU devices
(and real GPUs when available).

JAX can simulate multiple independent devices on a single CPU via the
XLA_FLAGS environment variable.  This must be set before any JAX import,
so it lives at the very top of this file.

Run with (CPU virtual devices):
    python fbp_sharding_sandbox.py

On a real multi-GPU machine, remove or comment out the XLA_FLAGS line and
set N_SHARDS to match the number of GPUs.

Expected output (all diffs should be 0.0 or float-epsilon):
    Test 0 (unsharded self-consistency): 0.0
    Test 1a (data routing via shard_map): 0.0
    Test 1b (sharded fbp_filter vs baseline): ~0.0
    Test 2  (manual chunk vs baseline): 0.0
"""

import os
# Comment out the line below to use real GPUs instead of virtual CPU devices.
# Must be set before any JAX import.
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

import jax
import jax.numpy as jnp
import numpy as np
import mbirjax

N_SHARDS = len(jax.devices())  # auto-detect: 4 virtual CPUs or however many real GPUs
devices = jax.devices()
print(f"Devices ({N_SHARDS}): {devices}")

# Use a number of rows divisible by N_SHARDS (64 works for 1, 2, 4, 8 GPUs).
sino_shape = (180, 64, 64)
assert sino_shape[1] % N_SHARDS == 0, f"sino rows ({sino_shape[1]}) must be divisible by N_SHARDS ({N_SHARDS})"
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
    print("Baseline saved.  Set save_output=False and re-run to run all tests.")
    import sys; sys.exit(0)  # nothing else to do until baseline exists

data = np.load('fbp_sharding_baseline.npz')
print('Test 0 (unsharded self-consistency):',
      np.max(np.abs(filtered_full - data['filtered'])))  # expect 0.0

# ------------------------------------------------------------------
# Test 1a: data-routing diagnostics — isolate exactly where the failure is.
# ------------------------------------------------------------------
model.configure_sharding(devices)

sino_sharded = model._maybe_shard(sino, axis=1)
print(f"\nSinogram sharding:    {sino_sharded.sharding}")
print(f"sino_sharded.shape:   {sino_sharded.shape}  (global; should be {sino_shape})")

P = jax.sharding.PartitionSpec
sino_np = np.array(sino)

# --- 1a-0: gather the sharded sino WITHOUT any shard_map ---------
# If this fails, the bug is in _maybe_shard or _maybe_gather, not shard_map.
direct_gathered = np.array(model._maybe_gather(sino_sharded, axis=1))
print(f"\nTest 1a-0 (gather after _maybe_shard, no shard_map):")
print(f"  shape: {direct_gathered.shape}  (should be {sino_shape})")
print(f"  max diff vs sino: {np.max(np.abs(direct_gathered - sino_np))}")  # expect 0.0

# --- 1a-1: shard_map identity, gather, check per-shard correctness --
# Shows whether each shard's gathered output matches the correct rows.
identity_out = jax.shard_map(
    lambda s: s * 1.0,
    mesh=model.mesh,
    in_specs=P(None, 'slices', None),
    out_specs=P(None, 'slices', None),
)(sino_sharded)
identity_gathered = np.array(model._maybe_gather(identity_out, axis=1))
shard_rows = sino_shape[1] // N_SHARDS
print(f"\nTest 1a-1 (shard_map identity, gathered):")
print(f"  shape: {identity_gathered.shape}  (should be {sino_shape})")
print(f"  overall max diff vs sino: {np.max(np.abs(identity_gathered - sino_np))}")
for i in range(N_SHARDS):
    lo, hi = i * shard_rows, (i + 1) * shard_rows
    diff = np.max(np.abs(identity_gathered[:, lo:hi, :] - sino_np[:, lo:hi, :]))
    print(f"  shard {i} rows {lo:3d}:{hi:3d} vs sino rows {lo:3d}:{hi:3d}: {diff}")

# ------------------------------------------------------------------
# Test 1b: sharded fbp_filter via configure_sharding / _maybe_shard
# fbp_filter uses jax.shard_map internally (not the deprecated
# jax.experimental.shard_map) with vmap over views rather than
# lax.map(batch_size), which avoids wrong cuFFT results on GPU.
# ------------------------------------------------------------------
filtered_sharded = model.fbp_filter(sino_sharded)
print(f"Filtered sharding: {filtered_sharded.sharding}")

filtered_gathered = np.array(model._maybe_gather(filtered_sharded, axis=1))
print('Test 1b (sharded fbp_filter vs baseline):',
      np.max(np.abs(filtered_gathered - data['filtered'])))  # expect ~0.0

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
