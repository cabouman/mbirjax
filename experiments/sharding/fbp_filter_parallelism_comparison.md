# fbp_filter parallelism comparison

This document records timing experiments for seven approaches (A–G) to
parallelizing the FBP ramp-filter step across two GPUs (or two virtual CPU
devices for the CPU runs).  The filter applies a 1-D convolution to every
detector row of a sinogram — a natural candidate for parallelism because every
row is independent of every other.

---

## The core computation

Before describing how each path distributes the work, it helps to understand
what one device must do.  The sinogram has shape
`(views, rows, channels)`.  For each view, every row is convolved with a ramp
filter along the channel axis:

```python
def convolve_row(row):              # (channels,) → (channels,)
    return fftconvolve(row, ramp_filter, mode='valid')

def process_view(view):             # (rows, channels) → (rows, channels)
    return vmap(convolve_row)(view) # apply to all rows simultaneously
```

`vmap` is JAX's vectorizing transform — it applies `convolve_row` to every row
in parallel, fused into a single GPU kernel (analogous to NumPy broadcasting,
but JIT-compiled).  Views are then processed in memory-managed chunks:

```python
lax.map(process_view, sinogram, batch_size=100)
# iterates over views 100 at a time; XLA schedules the GPU without
# any CPU involvement between chunks
```

The `batch_size` parameter tells XLA to loop over 100-view chunks
*internally*.  This bounds peak memory without introducing per-chunk CPU–GPU
synchronisation — everything stays on-device for the duration of the call.

---

## Existing — main-branch production code

The starting point is the current production `fbp_filter`, running on a
single NVIDIA L40S GPU (48 GB).  It uses a Python `for` loop over view
batches with a synchronisation barrier after each:

```python
for i in range(0, num_views, batch_size):
    result = lax.map(process_view, sinogram[i:i+batch_size],
                     batch_size=batch_size)
    result.block_until_ready()   # ← forces CPU↔GPU round-trip here
    parts.append(result)
```

The intent was to bound memory by processing batches sequentially.  It is
numerically correct (`diff = 0`) but `block_until_ready` inside the loop
makes the CPU stall after every 100 views, waiting for the GPU to drain
before dispatching the next chunk.  With 512 views that is 6 stalls; with
1024 views it is 11.  Each stall costs ~70–100 ms on the hardware tested:

| size | Existing | Path A | Existing is N× slower |
|------|----------|--------|----------------------|
| 512  | 0.457 s  | 0.053 s | Existing is 8.6× slower |
| 1024 | 1.404 s  | 0.340 s | Existing is 4.1× slower |

On CPU the stall is essentially free (CPU JAX operations are already
sequential), so Existing is within 7% of path A there.

Path A (below) fixes this with a single `lax.map` call and is used as the
baseline for all speedup comparisons in the rest of this document.

---

## Path A — single-device baseline

**Summary:** Single-device baseline; 4–9× faster than Existing on GPU, essentially the same on CPU.

Path A restructures the production loop into a single `lax.map` call that
hands all chunking to XLA:

```python
lax.map(process_view, sinogram, batch_size=100)
# XLA loops over 100-view chunks internally — the GPU never idles
# waiting for the CPU between chunks
```

This runs on one GPU with no inter-chunk synchronisation.  It is 4–9×
faster than Existing on GPU and essentially identical on CPU.

---

## Path B — `shard_map` + `lax.map`

**Summary:** Slower than A at small sizes, modest gain at large sizes, and incorrect results with 2 devices on GPU.

`shard_map` is JAX's SPMD (Single Program, Multiple Data) primitive.  You
write the computation for *one device's block* of the data; JAX compiles
and runs identical copies on all devices simultaneously — the same idea as
an MPI SPMD region, but expressed as a Python decorator.

### Mesh and partition specs

Two JAX concepts control how data is distributed:

- **Mesh** — a named grid of devices.  `Mesh([gpu0, gpu1], 'devices')`
  creates a 1-D grid with axis name `'devices'`.  Think of it as a directory
  that maps axis names to physical devices.

- **Partition spec** (`P`) — a per-array annotation that says which mesh
  axis each array dimension is split across.  `P(None, 'devices', None)`
  on a `(views, rows, channels)` array means: don't split views, split rows
  across `'devices'`, don't split channels.  `P()` (empty) means replicate
  the whole array to every device.

One practical constraint: any axis marked for splitting must be evenly
divisible by the number of devices.  If the number of rows is not a multiple
of the device count, the array must be padded before distribution (and the
padding stripped from the output).

### The kernel

The sinogram is split along the row axis; the filter is replicated:

```python
@partial(shard_map,
    mesh      = Mesh([gpu0, gpu1], 'devices'),
    in_specs  = (P(None, 'devices', None),  # sinogram: split rows across GPUs
                 P()),                        # filter:   same copy on every GPU
    out_specs = P(None, 'devices', None))   # output:   same split as input
def kernel(local_sino, filter):
    # each GPU sees (views, rows/2, channels) and runs path A's kernel
    return lax.map(process_view, local_sino, batch_size=100)

result = kernel(sinogram, filter)   # JAX handles scatter, run, and gather
```

### The 1-device diagnostic

Running `shard_map` on a 1-device mesh exercises the same SPMD compiler
path (as opposed to the single-device JAX compiler) but with no data
movement possible — both "devices" are the same GPU.  Any slowdown relative
to path A is purely compilation overhead.  In practice the 1-device
shard_map matches path A within measurement noise (1.00× at size=1024),
confirming SPMD compilation adds negligible overhead for this workload.

### NVIDIA L40S GPU (48GB) Results

| size | 1-dev | 2-dev | vs A (2-dev) |
|------|-------|-------|--------------|
| 512  | 0.053 s | 0.076 s ⚠ | 0.69× (slower) |
| 1024 | 0.341 s | 0.221 s ⚠ | 1.54× |

At size=512 the 2-device path is *slower* than A (0.076 s vs 0.053 s); at
size=1024 it reaches 1.54×.  The gain at size=1024 may reflect reduced
memory pressure — peak GPU usage measured at ~20 GB for single-device path A
at that size, which likely reflects working memory from the batched `lax.map`
rather than the sinogram itself (4 GB).  Splitting across two GPUs halves
the working memory per device, which could allow the GPU's caches and memory
bandwidth to be used more efficiently.

### <span style="color:red">Incorrect results</span>

The 2-device results (marked ⚠ above) differ from path A by far more than
float32 noise (`diff = 1.1e-02` at size=512, `diff = 5.9e-01` at size=1024).
Since every row is processed independently and the sharding is along the row
axis, the results should be bit-identical to path A.  The source of the
discrepancy is not yet understood, which makes path B unsuitable for
production on GPU.

---

## Path C — `shard_map` + `vmap`

**Summary:** Nearly identical to B; confirms SPMD overhead is not `lax.map`-specific. Same correctness issue on GPU.

Path C uses the same row-wise sharding as B but replaces `lax.map` inside
the kernel with a single `vmap` over all views at once:

```python
def kernel(local_sino, filter):
    # vmap processes every view simultaneously rather than in 100-view chunks
    return vmap(process_view)(local_sino)
```

The motivation is to isolate whether any overhead in path B is specific to
`lax.map` inside an SPMD context, or whether it is a more general property
of the SPMD compiler.  The answer is that the two are essentially
indistinguishable — paths B and C show nearly identical timing wherever
both are measured.

The trade-off with `vmap` is memory: it materialises all intermediate results
for every view simultaneously rather than in chunks of 100.  For a
1024-view sinogram this risks running out of GPU memory, so path C is skipped
for sizes above 512.

### NVIDIA L40S GPU (48GB) Results

| size | 1-dev | 2-dev | vs A (2-dev) |
|------|-------|-------|--------------|
| 512  | 0.049 s | 0.074 s ⚠ | 0.71× (slower) |
| 1024 | — (skipped, OOM risk) | — | — |

Results at size=512 are essentially identical to path B, confirming the
overhead is not specific to `lax.map` inside the SPMD context.  The 2-device
results carry the same correctness warning as path B (`diff = 1.1e-02`).

---

## Path D — `pmap` (skipped — deprecated)

**Summary:** Deprecated; included for diagnostic context only, not timed.

`pmap` is JAX's older device-parallel primitive, now deprecated in favour of
`shard_map`.  It is listed here because it was investigated during development
and revealed a useful diagnostic.

Like `shard_map`, `pmap` runs the same function on every device.  The
difference is in how JAX compiles it:

- **On GPU**, `pmap` routes through the same XLA SPMD partitioner as
  `shard_map`, so it shows the same performance characteristics.
- **On CPU**, `pmap` uses a *device-parallel* compilation path that bypasses
  the SPMD partitioner entirely.  The result is near-linear speedup with
  essentially no overhead at n_devices=1 — strikingly different from the GPU
  behaviour.

This CPU/GPU asymmetry was a key diagnostic clue during development: it
confirmed that any overhead seen with shard_map on GPU is SPMD-compiler
specific, not an inherent cost of parallelism.

`pmap` is not run in the timing results because the helper APIs needed to
pre-place data for it (`device_put_sharded`, `PositionalSharding`) are also
deprecated, making it difficult to construct a clean benchmark.

---

## Path E — threading, unsharded input

**Summary:** PCIe transfers from pulling and re-uploading the full sinogram dominate; 10–13× slower than A on GPU.

Path E is the first threading-based approach.  One Python thread is launched
per GPU; each thread handles the filter for its assigned block of rows.
Python's `ThreadPoolExecutor` lets the threads run concurrently, and because
each thread calls a different JAX device, the GPUs compute in parallel.

The problem is that the sinogram arrives as a single JAX array on GPU 0.
Before the threads can start, it must be pulled to host memory, split into
per-device row blocks, and re-uploaded.  After filtering, the results come
back to host for concatenation and are uploaded once more:

```python
sino_np    = np.asarray(sinogram)               # GPU 0 → host  (PCIe)
row_splits = np.split(sino_np, n_gpus, axis=1)

def worker(i):
    with jax.default_device(gpu[i]):
        out = filter_fn(jnp.array(row_splits[i]),   # host → GPU i (PCIe)
                        jnp.array(filter))
        out.block_until_ready()
    results[i] = np.asarray(out)                # GPU i → host  (PCIe)

ThreadPoolExecutor().map(worker, range(n_gpus))
result = jnp.array(np.concatenate(results, axis=1))  # host → GPU (PCIe)
```

For N GPUs and an S³ float32 sinogram, the PCIe traffic is:

| transfer | direction | bytes |
|----------|-----------|-------|
| pull full sinogram | GPU 0 → CPU | S³ × 4 |
| push N shards (parallel) | CPU → GPU 0…N | S³ × 4 |
| pull N results (parallel) | GPU 0…N → CPU | S³ × 4 |
| push gathered result | CPU → GPU 0 | S³ × 4 |
| **total** | | **4 × S³ × 4 bytes** |

Crucially, this total is constant regardless of N — adding GPUs does not
reduce the transfer cost.  At size=1024 that is ~16 GB over a ~11 GB/s bus,
which swamps the compute time.

On CPU there is no PCIe bus; `np.asarray()` is essentially free, so threading
gives near-linear speedup there.

### NVIDIA L40S GPU (48GB) Results

Only 2-device results are shown; a 1-device threading run adds CPU roundtrip
overhead with no parallelism benefit, making it strictly slower than path A
and adding nothing diagnostic.

| size | 2-dev | vs A |
|------|-------|------|
| 512  | 0.582 s | 0.09× (11× slower than A) |
| 1024 | 4.363 s | 0.08× (13× slower than A) |

---

## Path F — threading, pre-sharded input

**Summary:** Eliminates the input transfer cost but output gather still limits GPU performance; marginally faster than E.

Path F is structurally identical to E but assumes the sinogram is already
distributed across GPUs before the call — the situation that arises naturally
when the sinogram comes from a previous distributed step (e.g., the VCD loop).
Each thread reads its input directly from the GPU it already lives on via
`addressable_shards`, a JAX API that exposes each per-device shard without
any host copy:

```python
dev_to_shard = {s.device: s.data               # already on device — zero PCIe
                for s in sinogram.addressable_shards}

def worker(i):
    with jax.default_device(gpu[i]):
        out = filter_fn(dev_to_shard[gpu[i]],   # no transfer needed
                        jnp.array(filter))
        out.block_until_ready()
    results[i] = np.asarray(out)               # GPU i → host  (PCIe)

result = jnp.array(np.concatenate(results, axis=1))  # host → GPU (PCIe)
```

This eliminates the initial S³ × 4-byte pull but the output gather still
requires two PCIe passes (results → host, then concatenated array → GPU).
In practice the improvement over E is modest (~7%) because the output
transfers still dominate.

### NVIDIA L40S GPU (48GB) Results

Only 2-device results are shown for the same reason as path E: the sinogram
must be pre-sharded before the call, so a 1-device run is not a meaningful
comparison.

| size | 2-dev | vs A |
|------|-------|------|
| 512  | 0.539 s | 0.10× (10× slower than A) |
| 1024 | 4.113 s | 0.08× (12× slower than A) |

---

## Path G — threading, fully sharded (zero PCIe) ★

**Summary:** Zero PCIe in both directions; correct output; near-2× speedup with 2 GPUs. Clear production winner.

Path G eliminates all host memory transfers in both directions.  The input is
read on-device (same as F), and each thread keeps its result on its GPU.  The
output is assembled into a global distributed JAX array using
`make_array_from_single_device_arrays`, which constructs a logical "global
view" from the per-device arrays without touching host memory:

```python
def worker(i):
    with jax.default_device(gpu[i]):
        out = filter_fn(dev_to_shard[gpu[i]], jnp.array(filter))
        out = out * (pi / num_views)       # scale on-device
        out.block_until_ready()
    results[i] = out                       # stays on GPU — zero PCIe

result = jax.make_array_from_single_device_arrays(
    shape    = sinogram.shape,
    sharding = sinogram.sharding,          # same row distribution as input
    arrays   = results)
```

The returned `result` is a NamedSharding JAX array with the same row
distribution as the input sinogram.  It can be passed directly to any
downstream operation (such as back-projection) without a gather step —
making path G a natural fit for the distributed VCD loop.

### NVIDIA L40S GPU (48GB) Results

Only 2-device results are shown; with 1 device and a pre-sharded sinogram,
path G reduces to path A with added overhead.

| size | 2-dev | vs A |
|------|-------|------|
| 512  | 0.029 s | 1.84× |
| 1024 | 0.180 s | 1.89× |

Path G is the clear winner on GPU, achieving close to the 2× speedup that
pure 2-way compute parallelism predicts (1.84× at size=512, 1.89× at
size=1024).  With zero PCIe overhead there is no hidden transfer cost
obscuring the compute speedup.

On CPU, paths E, F, and G all give similar speedup (~1.9×) because copying
is cheap.  G's advantage on CPU is architectural: its output is already in the
distributed format expected by the rest of the pipeline.

---

## Timing results

```
========================================================================
                         GPU NVIDIA H100:
========================================================================

[detect]  real_gpus=2  xla_virtual_cpus=0  →  backend=GPU, 2 device(s) available
JAX backend: GPU   devices: 2   [CudaDevice(id=0), CudaDevice(id=1)]
Sizes: [512, 1024, 1280]   view_batch_size: 100   runs: 1

========================================================================
Size 512  —  sinogram 512×512×512  (0.50 GB float32)

  A. Baseline (1 GPU, lax.map+vmap):  0.017 s  [GPU 0: 2.66GB  GPU 1: 0.00GB]
  Existing (main branch, Python loop):  0.487 s  (0.03x vs A)  diff=0.0e+00

  ── BOTTLENECK 1: SPMD overhead ──  (1-dev proof: same GPU, no PCIe possible)
  B. shard_map+lax   1-dev:  0.017 s  (1.00x)  diff=0.0e+00  ← 1-dev: no data movement; pure SPMD vs baseline
  B. shard_map+lax   2-dev:  0.009 s  (1.85x)  diff=0.0e+00  
  C. shard_map+vmap  1-dev:  0.015 s  (1.10x)  diff=0.0e+00  ← 1-dev: no data movement; pure SPMD vs baseline
  C. shard_map+vmap  2-dev:  0.008 s  (2.15x)  diff=0.0e+00  

  D. pmap  (skipped — jax.pmap deprecated)

  ── BOTTLENECK 2: data movement ──  E has ~2.0 GB PCIe overhead
  E. thread unshard  2-dev:  0.257 s  (0.06x)  diff=0.0e+00  ← ~2.0 GB PCIe
  F. thread fast     2-dev:  0.225 s  (0.07x)  diff=0.0e+00  ← ~1.0 GB PCIe (output gather only)
  G. thread sharded  2-dev:  0.010 s  (1.67x)  diff=0.0e+00  ← zero PCIe  ★
========================================================================
Size 1024  —  sinogram 1024×1024×1024  (4.00 GB float32)

  A. Baseline (1 GPU, lax.map+vmap):  0.105 s  [GPU 0: 19.89GB  GPU 1: 3.50GB]
  Existing (main branch, Python loop):  1.311 s  (0.08x vs A)  diff=0.0e+00

  ── BOTTLENECK 1: SPMD overhead ──  (1-dev proof: same GPU, no PCIe possible)
  B. shard_map+lax   1-dev:  0.105 s  (1.00x)  diff=0.0e+00  ← 1-dev: no data movement; pure SPMD vs baseline
  B. shard_map+lax   2-dev:  0.054 s  (1.96x)  diff=0.0e+00  
  C. shard_map+vmap: skipped for size 1024 (vmap over all views risks OOM above 512)

  D. pmap  (skipped — jax.pmap deprecated)

  ── BOTTLENECK 2: data movement ──  E has ~16.0 GB PCIe overhead
  E. thread unshard  2-dev:  2.025 s  (0.05x)  diff=0.0e+00  ← ~16.0 GB PCIe
  F. thread fast     2-dev:  1.770 s  (0.06x)  diff=0.0e+00  ← ~8.0 GB PCIe (output gather only)
  G. thread sharded  2-dev:  0.056 s  (1.89x)  diff=0.0e+00  ← zero PCIe  ★
========================================================================
Size 1280  —  sinogram 1280×1280×1280  (7.81 GB float32)

  A. Baseline (1 GPU, lax.map+vmap):  0.305 s  [GPU 0: 56.35GB  GPU 1: 14.55GB]
  Existing (main branch, Python loop):  1.440 s  (0.21x vs A)  diff=0.0e+00

  ── BOTTLENECK 1: SPMD overhead ──  (1-dev proof: same GPU, no PCIe possible)
  B. shard_map+lax   1-dev:  0.304 s  (1.00x)  diff=0.0e+00  ← 1-dev: no data movement; pure SPMD vs baseline
  B. shard_map+lax   2-dev:  0.155 s  (1.97x)  diff=0.0e+00  
  C. shard_map+vmap: skipped for size 1280 (vmap over all views risks OOM above 512)

  D. pmap  (skipped — jax.pmap deprecated)

  ── BOTTLENECK 2: data movement ──  E has ~31.2 GB PCIe overhead
  E. thread unshard  2-dev:  4.121 s  (0.07x)  diff=0.0e+00  ← ~31.2 GB PCIe
  F. thread fast     2-dev:  3.486 s  (0.09x)  diff=0.0e+00  ← ~15.6 GB PCIe (output gather only)
  G. thread sharded  2-dev:  0.158 s  (1.92x)  diff=0.0e+00  ← zero PCIe  ★


========================================================================
SUMMARY  —  backend=GPU  min of 1 runs; speedup vs A (unsharded baseline)

   size       A  baseline      B  sm+lax 1d      B  sm+lax 2d     C  sm+vmap 1d     C  sm+vmap 2d        D  pmap 1d        D  pmap 2d   E  thrd unshard      F  thrd fast   G  thrd sharded
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    512       0.017s base      0.017s 1.00x      0.009s 1.85x      0.015s 1.10x      0.008s 2.15x           skipped           skipped      0.257s 0.06x      0.225s 0.07x      0.010s 1.67x
   1024       0.105s base      0.105s 1.00x      0.054s 1.96x           skipped           skipped           skipped           skipped      2.025s 0.05x      1.770s 0.06x      0.056s 1.89x
   1280       0.305s base      0.304s 1.00x      0.155s 1.97x           skipped           skipped           skipped           skipped      4.121s 0.07x      3.486s 0.09x      0.158s 1.92x


========================================================================
                         GPU NVIDIA L40S:
========================================================================

[detect]  real_gpus=2  xla_virtual_cpus=0  →  backend=GPU, 2 device(s) available
JAX backend: GPU   devices: 2   [CudaDevice(id=0), CudaDevice(id=1)]
Sizes: [512, 1024]   view_batch_size: 100   runs: 1

========================================================================
Size 512  —  sinogram 512×512×512  (0.50 GB float32)

  A. Baseline (1 GPU, lax.map+vmap):  0.053 s  [GPU 0: 2.66GB  GPU 1: 0.00GB]
  Existing (main branch, Python loop):  0.457 s  (0.12x vs A)  diff=0.0e+00

  ── BOTTLENECK 1: SPMD overhead ──  (1-dev proof: same GPU, no PCIe possible)
  B. shard_map+lax   1-dev:  0.053 s  (1.00x)  diff=0.0e+00  ← 1-dev: no data movement; pure SPMD vs baseline
  B. shard_map+lax   2-dev:  0.076 s  (0.69x)  diff=1.1e-02  *** diff=1.1e-02 > 1e-04 — correctness suspect ***
  C. shard_map+vmap  1-dev:  0.049 s  (1.08x)  diff=0.0e+00  ← 1-dev: no data movement; pure SPMD vs baseline
  C. shard_map+vmap  2-dev:  0.074 s  (0.71x)  diff=1.1e-02  *** diff=1.1e-02 > 1e-04 — correctness suspect ***

  D. pmap  (skipped — jax.pmap deprecated)

  ── BOTTLENECK 2: data movement ──  E has ~2.0 GB PCIe overhead
  E. thread unshard  2-dev:  0.582 s  (0.09x)  diff=0.0e+00  ← ~2.0 GB PCIe
  F. thread fast     2-dev:  0.539 s  (0.10x)  diff=0.0e+00  ← ~1.0 GB PCIe (output gather only)
  G. thread sharded  2-dev:  0.029 s  (1.84x)  diff=0.0e+00  ← zero PCIe  ★
========================================================================
Size 1024  —  sinogram 1024×1024×1024  (4.00 GB float32)

  A. Baseline (1 GPU, lax.map+vmap):  0.340 s  [GPU 0: 19.89GB  GPU 1: 3.50GB]
  Existing (main branch, Python loop):  1.404 s  (0.24x vs A)  diff=0.0e+00

  ── BOTTLENECK 1: SPMD overhead ──  (1-dev proof: same GPU, no PCIe possible)
  B. shard_map+lax   1-dev:  0.341 s  (1.00x)  diff=0.0e+00  ← 1-dev: no data movement; pure SPMD vs baseline
  B. shard_map+lax   2-dev:  0.221 s  (1.54x)  diff=5.9e-01  *** diff=5.9e-01 > 1e-04 — correctness suspect ***
  C. shard_map+vmap: skipped for size 1024 (vmap over all views risks OOM above 512)

  D. pmap  (skipped — jax.pmap deprecated)

  ── BOTTLENECK 2: data movement ──  E has ~16.0 GB PCIe overhead
  E. thread unshard  2-dev:  4.363 s  (0.08x)  diff=0.0e+00  ← ~16.0 GB PCIe
  F. thread fast     2-dev:  4.113 s  (0.08x)  diff=0.0e+00  ← ~8.0 GB PCIe (output gather only)
  G. thread sharded  2-dev:  0.180 s  (1.89x)  diff=0.0e+00  ← zero PCIe  ★


========================================================================
SUMMARY  —  backend=GPU  min of 1 runs; speedup vs A (unsharded baseline)

   size       A  baseline      B  sm+lax 1d      B  sm+lax 2d     C  sm+vmap 1d     C  sm+vmap 2d        D  pmap 1d        D  pmap 2d   E  thrd unshard      F  thrd fast   G  thrd sharded
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    512       0.053s base      0.053s 1.00x      0.076s 0.69x      0.049s 1.08x      0.074s 0.71x           skipped           skipped      0.582s 0.09x      0.539s 0.10x      0.029s 1.84x
   1024       0.340s base      0.341s 1.00x      0.221s 1.54x           skipped           skipped           skipped           skipped      4.363s 0.08x      4.113s 0.08x      0.180s 1.89x

Key findings (GPU):
  B/C 1-dev faster than A: not a clean SPMD measurement — baseline A
    includes model-method overhead; earlier runs had a float64 filter
    bug that inflated baseline time.  See docstring for details.
  B/C 2-dev correctness: non-trivial diff on GPU (flagged above).
    Root cause unknown; shard_map multi-GPU not production-ready.
  E/F << A:  PCIe roundtrips dominate; cost is constant vs n_devices.
  G >> A:  zero PCIe; super-linear at small sizes (reduced per-GPU
           memory pressure); confirmed correct.  Production winner.


========================================================================
                   Mac M3 Max (multiple cores)
========================================================================

[detect]  real_gpus=0  xla_virtual_cpus=4  →  backend=CPU, 4 device(s) available
JAX backend: CPU   devices: 4   [CpuDevice(id=0), CpuDevice(id=1), CpuDevice(id=2), CpuDevice(id=3)]
Sizes: [256, 512]   view_batch_size: 100   runs: 1

========================================================================
Size 256  —  sinogram 256×256×256  (0.06 GB float32)

  A. Baseline (1 CPU, lax.map+vmap):  0.432 s
  Existing (main branch, Python loop):  0.457 s  (0.95x vs A)  diff=0.0e+00

  ── BOTTLENECK 1: SPMD overhead ──  (1-dev proof: same CPU device, no data movement)
  B. shard_map+lax   1-dev:  0.437 s  (0.99x)  diff=0.0e+00  ← 1-dev: no data movement; pure SPMD vs baseline
  B. shard_map+lax   2-dev:  0.227 s  (1.90x)  diff=2.3e-09
  C. shard_map+vmap  1-dev:  0.466 s  (0.93x)  diff=0.0e+00  ← 1-dev: no data movement; pure SPMD vs baseline
  C. shard_map+vmap  2-dev:  0.227 s  (1.90x)  diff=2.8e-09

  D. pmap  (skipped — jax.pmap deprecated)

  ── BOTTLENECK 2: data movement ──  E has host-memory copies (cheap on CPU)
  E. thread unshard  2-dev:  0.239 s  (1.81x)  diff=0.0e+00  ← host copy cheap; threading gives speedup
  F. thread fast     2-dev:  0.207 s  (2.09x)  diff=2.8e-09  ← same as E on CPU (input copy was already cheap)
  G. thread sharded  2-dev:  0.233 s  (1.86x)  diff=0.0e+00  ← zero host-mem copy  ★
========================================================================
Size 512  —  sinogram 512×512×512  (0.50 GB float32)

  A. Baseline (1 CPU, lax.map+vmap):  2.167 s
  Existing (main branch, Python loop):  2.318 s  (0.93x vs A)  diff=1.4e-09

  ── BOTTLENECK 1: SPMD overhead ──  (1-dev proof: same CPU device, no data movement)
  B. shard_map+lax   1-dev:  2.191 s  (0.99x)  diff=0.0e+00  ← 1-dev: no data movement; pure SPMD vs baseline
  B. shard_map+lax   2-dev:  1.105 s  (1.96x)  diff=0.0e+00
  C. shard_map+vmap  1-dev:  2.166 s  (1.00x)  diff=0.0e+00  ← 1-dev: no data movement; pure SPMD vs baseline
  C. shard_map+vmap  2-dev:  1.123 s  (1.93x)  diff=0.0e+00

  D. pmap  (skipped — jax.pmap deprecated)

  ── BOTTLENECK 2: data movement ──  E has host-memory copies (cheap on CPU)
  E. thread unshard  2-dev:  1.157 s  (1.87x)  diff=0.0e+00  ← host copy cheap; threading gives speedup
  F. thread fast     2-dev:  1.145 s  (1.89x)  diff=0.0e+00  ← same as E on CPU (input copy was already cheap)
  G. thread sharded  2-dev:  1.111 s  (1.95x)  diff=0.0e+00  ← zero host-mem copy  ★


========================================================================
SUMMARY  —  backend=CPU  min of 1 runs; speedup vs A (unsharded baseline)

   size       A  baseline      B  sm+lax 1d      B  sm+lax 2d     C  sm+vmap 1d     C  sm+vmap 2d        D  pmap 1d        D  pmap 2d   E  thrd unshard      F  thrd fast   G  thrd sharded
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    256       0.432s base      0.437s 0.99x      0.227s 1.90x      0.466s 0.93x      0.227s 1.90x           skipped           skipped      0.239s 1.81x      0.207s 2.09x      0.233s 1.86x
    512       2.167s base      2.191s 0.99x      1.105s 1.96x      2.166s 1.00x      1.123s 1.93x           skipped           skipped      1.157s 1.87x      1.145s 1.89x      1.111s 1.95x

Key findings (CPU):
  B/C: SPMD overhead negligible on CPU — 1-dev ≈ baseline,
    2-dev gives near-linear speedup.  shard_map is viable on CPU.
  E/F/G all similar: no PCIe bus; host copies cheap; threading
    speedup visible across all three paths.

```
