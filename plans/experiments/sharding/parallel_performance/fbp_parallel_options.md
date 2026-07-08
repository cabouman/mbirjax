# Multi-GPU Parallelism Options for `fbp_filter`

## Context

`fbp_filter` applies a 1D convolution along the channels axis independently for every
`(view, row)` pair — embarrassingly parallel along both axes. The first multi-device
implementation used `shard_map` + `lax.map` and proved 3–5× **slower** than single-device,
despite correct results. This document records the analysis and alternatives.

---

## Why `shard_map` + `lax.map` is slow

`shard_map` invokes XLA's **SPMD partitioner**, which treats every operation inside the
map as part of a globally-partitioned computation. This changes the XLA lowering of
`lax.map` in two harmful ways:

1. **More intermediate materializations.** Peak memory inside shard_map was ~32 GB vs
   ~24 GB for the identical single-device computation.
2. **Different (worse) XLA scan code.** The SPMD scan does not fuse the cuFFT calls
   as efficiently as the standard scan.

The smoking-gun evidence: timing with `n_devices=1` and `n_devices=2` inside shard_map
was essentially identical — the shard boundary provides no parallelism, all the cost is
the SPMD compilation overhead.

Observed times (size=512, 3-run min):

| config          | time   | vs unsharded |
|-----------------|--------|--------------|
| unsharded       | 0.195s | baseline     |
| shard_map 1-dev | 1.081s | 0.18×        |
| shard_map 2-dev | 1.231s | 0.16×        |

Root cause: `lax.map` compiles to fast sequential XLA outside SPMD context and slow SPMD
scan inside `shard_map`. This is a JAX/XLA issue, not an algorithmic one.

Note: a plain `NamedSharding` + `jit` approach was ruled out earlier because XLA's cuFFT
thunk requires row-major contiguous memory (`IsMonotonicWithDim0Major`). Sharding a 3D
sinogram along its **middle** axis (rows) produces strided per-device views that fail
this check. `shard_map` was the fix — it hands each device a fresh contiguous local
array — but brings the SPMD overhead described above.

---

## Option A: `pmap` ← **current implementation**

### Concept

`pmap` uses **device-parallel compilation** (not SPMD). Each device compiles and runs
an independent copy of the function on its local data. No SPMD partitioner is involved,
so `lax.map` inside `pmap` compiles identically to `lax.map` outside any multi-device
context. Expected result: per-device timing ≈ single-device timing, so two devices → ~2×
speedup.

### Mechanics

`pmap` requires the parallelized axis to be axis 0 of the input, so we transpose and
reshape before calling and undo the operation on the output:

```
sinogram:  (views, rows, channels)
           np.asarray → transpose(1,0,2) → reshape(n_dev, local_rows, views, channels)
pmap fn:   receives (local_rows, views, channels) per device
           transpose(1,0,2) → (views, local_rows, channels)
           lax.map + vmap as before
           transpose back → (local_rows, views, channels_filtered)
gather:    result (n_dev, local_rows, views, channels_filtered)
           np.asarray → reshape(rows, views, channels_filtered)
           transpose(1,0,2) → (views, rows, channels_filtered)
           jnp.array → uncommitted JAX array
```

The numpy roundtrip for the input (`np.asarray` before the reshape) sidesteps the JAX
0.10.0 GPU scatter bug (see `status.md`) — exactly the same workaround as `_maybe_shard`.
Reading from GPUs (the gather direction) is unaffected by the bug, so `np.asarray(result)`
is safe.

### Trade-offs

| | |
|---|---|
| **Pro** | Device-parallel XLA — no SPMD overhead |
| **Pro** | numpy source for the split → bypasses JAX scatter bug naturally |
| **Pro** | No Mesh / NamedSharding / PartitionSpec needed |
| **Con** | `pmap` is deprecated in newer JAX (officially replaced by `shard_map`) |
| **Con** | Two numpy roundtrips per `fbp_filter` call (split → pmap, gather → JNP) |
| **Con** | Requires transpose + reshape logic; output must be un-transposed |
| **Con** | Compiled once per `(n_devices, local_rows, num_views, num_channels)` tuple |

---

## Option B: Python threads with per-device JIT

### Concept

Split the sinogram into n_devices numpy chunks along axis 1, launch one thread per
device, each calling the standard single-device JIT'd filter with `jax.default_device`
set. JAX's async GPU dispatch allows the kernel queues to overlap even under the GIL.

```python
shards = np.split(np.asarray(sinogram).transpose(1,0,2), n_devices, axis=0)
# shards[i]: (local_rows, views, channels)

def process(device, shard_np):
    with jax.default_device(device):
        shard = jnp.array(shard_np.transpose(1,0,2))  # → (views, local_rows, channels)
        return filter_local_jit(shard)  # existing single-device function

with ThreadPoolExecutor(max_workers=n_devices) as pool:
    results = list(pool.map(lambda args: process(*args), zip(devices, shards)))

filtered = jnp.concatenate(results, axis=1)  # (views, rows, channels_filtered)
```

### Trade-offs

| | |
|---|---|
| **Pro** | Reuses the existing single-device JIT path unchanged |
| **Pro** | Not deprecated — works with any JAX version |
| **Pro** | Conceptually the simplest multi-device design |
| **Con** | `jax.default_device` thread-safety is documented but less battle-tested |
| **Con** | Python GIL may limit true concurrency for CPU/virtual-device testing |
| **Con** | ThreadPoolExecutor overhead for short-running tasks |

---

## Option C: `jit` + NamedSharding (GSPMD) — ruled out

Annotating the sinogram with `NamedSharding(mesh, P(None,'slices',None))` and calling
`jit(filter_fn)` lets XLA's GSPMD partitioner distribute the work. **Rejected** because
sharding along the middle axis of a 3D array produces strided per-device views, which
XLA's cuFFT thunk rejects with `IsMonotonicWithDim0Major` errors.

---

## Bugs found in first pmap implementation (now fixed)

### Bug 1: pmap recompiles on every call

`jax.pmap` caches compiled XLA code keyed on the **Python function object identity**,
not on the computation graph.  Defining `apply_filter_pmap` inline inside `fbp_filter`
creates a new closure on every call, causing a full XLA recompilation (~300 ms at
size=512) even though the timed runs come after a warmup call.

Fix: cache the pmap-wrapped function on `self._pmap_filter_cache`, keyed by every
parameter that affects the compiled computation
`(device_ids, num_channels, filter_name, scaling_factor, body_views, tail_views, view_batch_size)`.
The cached function is compiled once (during warmup) and reused for all timed calls.

### Bug 2: non-contiguous memory layout inside pmap

First layout: `sino_np.transpose(1,0,2).reshape(n_devices, local_rows, views, channels)`.
Each device received `(local_rows, views, channels)`; inside pmap, `shard.transpose(1,0,2)`
was called to produce `(views, local_rows, channels)`.  That transpose makes the
`local_rows` dimension non-contiguous in memory (its stride equals `views * channels`).
Every cuFFT call then operated on strided data, which forces an internal copy.

Fix: use `np.stack(np.split(sino_np, n_devices, axis=1))` to produce
`(n_devices, views, local_rows, channels)`.  Each device shard is already
`(views, local_rows, channels)` in C order — fully contiguous, no transpose needed.
Gather becomes `np.concatenate(list(result_np), axis=1)` with no transpose either.

### Why pmap ultimately failed on real GPU

After the layout and caching fixes, pmap worked well on virtual CPU devices
(4-device run at size=512: 3.28× speedup, 1-device ≈ baseline).  But on real GPU:

- n_devices=1 pmap: still ~6× slower than unsharded
- n_devices=2 pmap: slightly faster than 1-device, but still ~4× slower
- GPU 1 memory stayed at 0.00 GB even for n_devices=2

Root cause: on GPU XLA backends, `jax.pmap` internally uses the same XLA SPMD
partitioner as `shard_map`.  The CPU backend uses a different (device-parallel)
path that avoids SPMD.  This is confirmed by the CPU vs GPU asymmetry:
CPU 1-device pmap ≈ baseline; GPU 1-device pmap ≈ 6× slower.

The GPU XLA SPMD path may be putting all work on GPU 0 (GPU 1 memory = 0),
or it may be using both GPUs with the same SPMD overhead as shard_map.
Either way, SPMD is the bottleneck, not device count.

## Current status: Option B (threading) ← current implementation

The final implementation uses Python threads with per-device `jax.jit`.  Each thread:
1. Sets `jax.default_device(device_i)` via the context manager (thread-local in JAX)
2. Transfers its numpy shard to device i via `jnp.array(shard_np)`
3. Calls the module-level `_apply_fbp_filter_to_shard` (standard JIT, not SPMD)
4. Blocks until ready, then gathers to numpy

The JIT cache key includes the module-level function identity (stable across all calls)
plus abstract input types, so compilation happens once per (size, view_batch_size) pair
during the warmup call and is reused by all threads on all timed calls.

JAX releases the GIL during device dispatch, so both GPU kernels run concurrently
despite the Python GIL.

If `pmap` gives the expected near-linear speedup, it will remain as the production path
with a comment noting the deprecation risk. If it also underperforms, Option B (threads)
is the next candidate.

---

## Deprecation note for `pmap`

JAX officially deprecates `pmap` in favor of `shard_map` + SPMD sharding. The JAX
team's position is that SPMD is better for composability (gradients, nested transforms,
etc.). For `fbp_filter` specifically — a pure forward pass with no gradients and no
cross-device collectives — the SPMD overhead is pure cost and `pmap` is the better fit.

If a future JAX version removes `pmap` or if we need gradient support, the correct
migration path is to wait for XLA to fix the `lax.map`-inside-`shard_map` performance
regression, then revert to `shard_map`.
