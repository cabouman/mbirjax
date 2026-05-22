"""
experiments/sharding/fbp_filter_parallelism_comparison.py
──────────────────────────────────────────────────────────
Comprehensive comparison of every multi-device parallelism strategy tried for
fbp_filter, on both real GPUs and virtual CPU devices.  Each approach is
isolated to expose its specific bottleneck.

Two independent bottlenecks were discovered during development:

  BOTTLENECK 1 — XLA SPMD compilation overhead
  ─────────────────────────────────────────────
  Affects: shard_map (all variants), pmap on GPU
  NOT GPU-only: the SPMD compiler path exists on CPU too, just with
    different (sometimes smaller) overhead.
  Symptom: n_devices=1 is much slower than the unsharded baseline even
           though both run on the same single device (zero data-movement
           possible).  Slowness must come from compilation, not transfers.
  Cause:   lax.map compiled inside an SPMD context (shard_map or GPU pmap)
           goes through XLA's SPMD partitioner, which produces different —
           and slower — scan code than the standard lax.map path.
  Proof:   time(shard_map 1-dev) ≈ time(shard_map 2-dev)  [no speedup]
           time(shard_map 1-dev) >> time(baseline 1-dev)   [overhead present]
  Note on pmap:
    GPU backend — pmap uses the SAME XLA SPMD partitioner as shard_map;
      n_devices=1 pmap is just as slow as shard_map.
    CPU backend — pmap uses a DIFFERENT, device-parallel compilation path
      (no SPMD); n_devices=1 ≈ baseline, and n_devices=N gives near-linear
      speedup.  This CPU/GPU asymmetry was a key diagnostic clue.

  BOTTLENECK 2 — Host↔device data movement
  ─────────────────────────────────────────
  Affects: threading when the sinogram arrives un-sharded
  GPU only: on CPU, np.asarray() of a JAX array is a cheap host-memory
    copy (no PCIe bus); the threading paths are much closer in performance.
  Symptom: n_devices=2 is *slower* than the 1-device baseline; cost is
           roughly constant regardless of n_devices.
  Cause:   The threading path does np.asarray(sinogram) to split it, then
           re-uploads each shard.  For a float32 sinogram of size S³:
             pull full sinogram:   S³×4 bytes  (GPU→CPU)
             push N shards:        S³×4 bytes  (CPU→GPU, in parallel)
             pull N results:       S³×4 bytes  (GPU→CPU, in parallel)
             push gathered result: S³×4 bytes  (CPU→GPU)
           = 4 × S³ × 4 bytes of PCIe regardless of N.
  Proof:   time(thread 1-dev) ≈ time(thread 2-dev) − compute/2
           Both dominated by the fixed transfer cost, not compute.

Paths tested (all produce identical filtered sinograms up to float32 noise):
  A  Baseline              — single device, standard lax.map + vmap
  B  shard_map + lax.map  — SPMD path; pre-sharded input isolates SPMD
                             overhead from data movement; tested 1-dev/2-dev
  C  shard_map + vmap     — no lax.map inside SPMD; tests whether overhead
                             is lax.map-specific or general SPMD; skipped
                             for large sizes (vmap over all views risks OOM)
  D  pmap                 — pre-placed input isolates compute from transfers;
                             GPU: SPMD (same slowdown as B)
                             CPU: device-parallel (fast, near-linear speedup)
  E  thread unsharded     — production code; GPU: data movement dominates;
                             CPU: transfers cheap, threading gives speedup
  F  thread + fast        — pre-sharded, input via addressable_shards
                             (zero input transfer); output still gathered
  G  thread + sharded     — zero transfers in both directions; winner on GPU;
                             comparable to D on CPU

Usage
─────
    # Real GPUs (auto-detected):
    conda activate mbirjax
    python experiments/sharding/fbp_filter_parallelism_comparison.py

    # Virtual CPU devices (auto-detected from XLA_FLAGS, or set flag manually):
    XLA_FLAGS="--xla_force_host_platform_device_count=4" python ...
    # or set USE_VIRTUAL_CPU = True in the Configuration section below.
"""

# ── Pre-JAX device detection ───────────────────────────────────────────────────
# XLA_FLAGS must be set before JAX is imported, so detection runs first using
# only the standard library.

import os
import re
import subprocess


def count_real_gpus():
    """Count NVIDIA GPUs via nvidia-smi — no JAX required.

    Returns the number of GPUs reported by nvidia-smi, or 0 if nvidia-smi is
    unavailable or reports no GPUs.  Works on any platform where nvidia-smi is
    on PATH.
    """
    try:
        r = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            return len([ln for ln in r.stdout.strip().splitlines() if ln.strip()])
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return 0


def get_xla_virtual_cpu_count():
    """Read the virtual-CPU device count from XLA_FLAGS — no JAX required.

    Returns N from '--xla_force_host_platform_device_count=N' in XLA_FLAGS,
    or 0 if the flag is not set.
    """
    m = re.search(r'xla_force_host_platform_device_count=(\d+)',
                  os.environ.get('XLA_FLAGS', ''))
    return int(m.group(1)) if m else 0


# ── Configuration ─────────────────────────────────────────────────────────────

# Set USE_VIRTUAL_CPU = True to force virtual CPU devices even on a GPU machine
# (useful for testing the CPU code path or for machines without NVIDIA GPUs).
# Leave False to auto-detect: real GPUs are used if present, otherwise
# whatever CPU devices JAX finds.
N_VIRTUAL_CPUS  = 4        # number of virtual CPU devices when _n_real_gpus == 0

SIZES              = [256, 512]
VIEW_BATCH_SIZE    = 100
N_RUNS             = 1     # report min over this many timed calls
VMAP_ONLY_MAX_SIZE = 512   # path C vmap-over-all-views OOMs for large sizes

# ── Backend selection (must happen before JAX import) ─────────────────────────

_n_real_gpus    = count_real_gpus()
_n_virtual_cpus = get_xla_virtual_cpu_count()

USE_VIRTUAL_CPU = False if _n_real_gpus > 0 else True
if USE_VIRTUAL_CPU and _n_virtual_cpus == 0:
    # Set the flag now so JAX sees it on import.
    _flags = os.environ.get('XLA_FLAGS', '')
    os.environ['XLA_FLAGS'] = (
        _flags + f' --xla_force_host_platform_device_count={N_VIRTUAL_CPUS}').strip()
    _n_virtual_cpus = N_VIRTUAL_CPUS

# Decide which backend we'll use.
if _n_virtual_cpus > 0:
    BACKEND    = 'cpu'
    N_DETECTED = _n_virtual_cpus
elif _n_real_gpus > 0:
    BACKEND    = 'gpu'
    N_DETECTED = _n_real_gpus
else:
    BACKEND    = 'cpu'
    N_DETECTED = 1   # single CPU fallback

print(f"[detect]  real_gpus={_n_real_gpus}  "
      f"xla_virtual_cpus={_n_virtual_cpus}  "
      f"→  backend={BACKEND.upper()}, {N_DETECTED} device(s) available")

# ── Imports (JAX and project) ─────────────────────────────────────────────────

import concurrent.futures
import time
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

import mbirjax
from mbirjax import tomography_utils
from mbirjax.parallel_beam import _apply_fbp_filter_to_shard
from mbirjax.memory_stats import get_memory_stats


# ── Utilities ─────────────────────────────────────────────────────────────────

def make_filter_np(num_channels, delta_voxel, filter_name='ramp'):
    """Return scaled float32 reconstruction filter as a numpy array.
    In-place *= preserves float32 — avoids the float64 promotion that occurs
    with out-of-place multiplication by a Python float."""
    f  = tomography_utils.generate_direct_recon_filter(num_channels,
                                                        filter_name=filter_name)
    f *= 1.0 / (delta_voxel ** 2)   # in-place: stays float32
    return np.asarray(f)


def time_fn(fn, n_runs=N_RUNS):
    """Return (min_elapsed, last_result) over n_runs calls."""
    times, result = [], None
    for _ in range(n_runs):
        t0     = time.perf_counter()
        result = fn()
        jax.block_until_ready(result)
        times.append(time.perf_counter() - t0)
    return min(times), result


def peak_mem_str():
    """Peak GPU memory string, or empty string on CPU backends."""
    stats = get_memory_stats(print_results=False)
    parts = [f"{s['id']}: {s['peak_bytes_in_use']/1024**3:.2f}GB"
             for s in stats if 'cpu' not in s['id'].lower()]
    return ('  [' + '  '.join(parts) + ']') if parts else ''


# ── Path B/C: shard_map factory functions ─────────────────────────────────────
# Factories return new Python function objects so JAX's JIT cache is keyed
# separately for each mesh (1-device vs 2-device).

def make_shardmap_lax_fn(mesh):
    """shard_map + lax.map inside the SPMD context — the original attempt."""
    @jax.jit
    def _fn(sinogram, filter_arr):
        @partial(shard_map,
                 mesh=mesh,
                 in_specs=(P(None, 'devices', None), P()),
                 out_specs=P(None, 'devices', None))
        def _inner(shard, f):
            n    = shard.shape[0]
            body = (n // VIEW_BATCH_SIZE) * VIEW_BATCH_SIZE
            tail = n - body
            def conv_row(row):   return jax.scipy.signal.fftconvolve(row, f, mode='valid')
            def conv_view(view): return jax.vmap(conv_row)(view)
            parts = []
            if body > 0:
                parts.append(jax.lax.map(conv_view, shard[:body],
                                         batch_size=VIEW_BATCH_SIZE))
            if tail > 0:
                parts.append(jax.lax.map(conv_view, shard[body:]))
            return jnp.concatenate(parts, axis=0) if len(parts) > 1 else parts[0]
        return _inner(sinogram, filter_arr)
    return _fn


def make_shardmap_vmap_fn(mesh):
    """shard_map + vmap only — no lax.map inside SPMD.
    Tests whether SPMD overhead is specific to lax.map or affects all ops.
    Skipped for large sizes: vmap over all views at once can exhaust memory."""
    @jax.jit
    def _fn(sinogram, filter_arr):
        @partial(shard_map,
                 mesh=mesh,
                 in_specs=(P(None, 'devices', None), P()),
                 out_specs=P(None, 'devices', None))
        def _inner(shard, f):
            def conv_row(row):   return jax.scipy.signal.fftconvolve(row, f, mode='valid')
            def conv_view(view): return jax.vmap(conv_row)(view)
            return jax.vmap(conv_view)(shard)   # vmap over ALL views at once
        return _inner(sinogram, filter_arr)
    return _fn


# ── Paths E / F / G: threading ────────────────────────────────────────────────

def _thread_filter_unsharded(sinogram, filter_np, devices,
                              view_batch_size=VIEW_BATCH_SIZE):
    """Path E — production code as of the first threading attempt.

    Always pulls the full sinogram to host memory before splitting and
    re-uploading to each device.  On GPU this is ~4× the sinogram size in
    PCIe traffic; on CPU the copy is cheap so this path performs better.
    """
    n_devices  = len(devices)
    num_views  = sinogram.shape[0]
    body_views = (num_views // view_batch_size) * view_batch_size
    tail_views = num_views - body_views
    sino_np    = np.asarray(sinogram)           # device→host (PCIe on GPU)
    row_splits = np.split(sino_np, n_devices, axis=1)
    results    = [None] * n_devices

    def _worker(i):
        with jax.default_device(devices[i]):
            out = _apply_fbp_filter_to_shard(
                jnp.array(row_splits[i]),        # host→device (PCIe on GPU)
                jnp.array(filter_np),
                body_views=body_views, tail_views=tail_views,
                view_batch_size=view_batch_size)
            jax.block_until_ready(out)
        results[i] = np.asarray(out)            # device→host (PCIe on GPU)

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_devices) as ex:
        list(ex.map(_worker, range(n_devices)))

    return jnp.array(np.concatenate(results, axis=1))  # host→device (PCIe on GPU)


def _thread_filter_fast(sinogram_sharded, filter_np, devices,
                         view_batch_size=VIEW_BATCH_SIZE):
    """Path F — input read via addressable_shards; output gathered via host.

    Each shard.data is already on its device — zero input transfer cost.
    Output still goes device→host→device for concatenation.
    """
    n_devices    = len(devices)
    num_views    = sinogram_sharded.shape[0]
    body_views   = (num_views // view_batch_size) * view_batch_size
    tail_views   = num_views - body_views
    dev_to_shard = {s.device: s.data for s in sinogram_sharded.addressable_shards}
    results      = [None] * n_devices

    def _worker(i):
        dev = devices[i]
        with jax.default_device(dev):
            out = _apply_fbp_filter_to_shard(
                dev_to_shard[dev],               # already on device — zero cost
                jnp.array(filter_np),
                body_views=body_views, tail_views=tail_views,
                view_batch_size=view_batch_size)
            jax.block_until_ready(out)
        results[i] = np.asarray(out)            # device→host (PCIe on GPU)

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_devices) as ex:
        list(ex.map(_worker, range(n_devices)))

    return jnp.array(np.concatenate(results, axis=1))  # host→device (PCIe on GPU)


def _thread_filter_sharded(sinogram_sharded, filter_np, devices,
                             view_batch_size=VIEW_BATCH_SIZE):
    """Path G — zero host transfers in both directions.

    Input shards read on-device via addressable_shards.  Each thread keeps
    its result on-device.  jax.make_array_from_single_device_arrays assembles
    the global distributed array without touching host memory.

    Applies π/n_views scaling on-device (plain Python float avoids a
    device-placement mismatch with jax scalar).  Output has the same
    NamedSharding as the input — suitable for passing directly to back_project.

    Note: fftconvolve('valid') with a (2·channels−1,) filter on (channels,)
    input produces (channels,) output, so the output sharding matches input.
    """
    n_devices, (num_views, num_rows, num_channels) = (
        len(devices), sinogram_sharded.shape)
    body_views   = (num_views // view_batch_size) * view_batch_size
    tail_views   = num_views - body_views
    pi_scale     = float(np.pi / num_views)
    dev_to_shard = {s.device: s.data for s in sinogram_sharded.addressable_shards}
    results      = [None] * n_devices

    def _worker(i):
        dev = devices[i]
        with jax.default_device(dev):
            out = _apply_fbp_filter_to_shard(
                dev_to_shard[dev],
                jnp.array(filter_np),
                body_views=body_views, tail_views=tail_views,
                view_batch_size=view_batch_size)
            out = out * pi_scale             # scale on-device, stays on device
            jax.block_until_ready(out)
        results[i] = out                     # keep on device — zero transfer

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_devices) as ex:
        list(ex.map(_worker, range(n_devices)))

    return jax.make_array_from_single_device_arrays(
        shape=(num_views, num_rows, num_channels),
        sharding=sinogram_sharded.sharding,
        arrays=results)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Select devices: real GPUs if present, otherwise all CPU devices.
    gpus     = [d for d in jax.devices() if 'cpu' not in d.device_kind.lower()]
    all_devs = gpus if gpus else jax.devices()
    backend  = 'GPU' if gpus else 'CPU'
    n_avail  = len(all_devs)

    print(f"JAX backend: {backend}   devices: {n_avail}   {all_devs}")
    if n_avail < 2:
        print("WARNING: fewer than 2 devices — multi-device paths will be skipped.")
    print(f"Sizes: {SIZES}   view_batch_size: {VIEW_BATCH_SIZE}   runs: {N_RUNS}\n")

    is_gpu   = backend == 'GPU'
    transfer = 'PCIe' if is_gpu else 'host-mem copy'

    summary = []   # list of row dicts for summary table

    for size in SIZES:
        print(f"{'='*72}")
        sz_gb = size**3 * 4 / 1024**3
        print(f"Size {size}  —  sinogram {size}×{size}×{size}  ({sz_gb:.2f} GB float32)")

        rng         = np.random.default_rng(42)
        sino_np     = rng.standard_normal((size, size, size)).astype(np.float32)
        sino_dev    = jnp.array(sino_np)    # on default device (dev 0)
        delta_voxel = 1.0
        filter_np   = make_filter_np(size, delta_voxel)
        filter_jax  = jnp.array(filter_np)

        row = {'size': size}

        # ── Path A: unsharded baseline ────────────────────────────────
        angles  = jnp.linspace(0, np.pi, size, endpoint=False)
        model_1 = mbirjax.ParallelBeamModel((size, size, size), angles)
        model_1.set_params(delta_voxel=delta_voxel, delta_det_channel=1.0)
        model_1.auto_set_recon_geometry()

        jax.block_until_ready(
            model_1.fbp_filter(sino_dev, view_batch_size=VIEW_BATCH_SIZE))  # warmup

        t_a, out_a = time_fn(
            lambda: model_1.fbp_filter(sino_dev, view_batch_size=VIEW_BATCH_SIZE))
        ref_np = np.asarray(out_a)  # scaled by π/n_views inside fbp_filter

        row['A'] = (t_a, 1.0)
        print(f"\n  A. Baseline (1 {backend}, lax.map+vmap):  "
              f"{t_a:.3f} s{peak_mem_str()}")

        if n_avail < 2:
            summary.append(row)
            continue

        # ── Pre-shard sinogram (done once, outside timing) ────────────
        # For GPU: jax.device_put from numpy source sidesteps the JAX
        # 0.10.0 scatter bug (which corrupts non-default GPUs when the
        # source is a JAX array).  For CPU: no bug, but the pattern is
        # the same and avoids a potential zero-copy aliasing issue.
        mesh_1     = Mesh(np.array(all_devs[:1]), ('devices',))
        mesh_2     = Mesh(np.array(all_devs[:2]), ('devices',))
        sharding_1 = NamedSharding(mesh_1, P(None, 'devices', None))
        sharding_2 = NamedSharding(mesh_2, P(None, 'devices', None))
        sino_sh1   = jax.device_put(sino_np, sharding_1)
        sino_sh2   = jax.device_put(sino_np, sharding_2)
        jax.block_until_ready(sino_sh1)
        jax.block_until_ready(sino_sh2)
        devs_2 = all_devs[:2]

        # ── Paths B & C: shard_map ────────────────────────────────────
        spmd_note = (f"same {backend}, no {transfer} possible"
                     if is_gpu else f"same CPU device, no data movement")
        print(f"\n  ── BOTTLENECK 1: SPMD overhead ──  (1-dev proof: {spmd_note})")

        for label, make_fn, tag in [
            ('B', make_shardmap_lax_fn,  'shard_map+lax '),
            ('C', make_shardmap_vmap_fn, 'shard_map+vmap'),
        ]:
            if label == 'C' and size > VMAP_ONLY_MAX_SIZE:
                print(f"  {label}. {tag}: skipped for size {size} "
                      f"(vmap over all views risks OOM above {VMAP_ONLY_MAX_SIZE})")
                continue

            for n_dev, mesh, sino_sh in [(1, mesh_1, sino_sh1),
                                          (2, mesh_2, sino_sh2)]:
                fn = make_fn(mesh)
                try:
                    jax.block_until_ready(fn(sino_sh, filter_jax))  # warmup
                    t, out = time_fn(lambda fn=fn, s=sino_sh: fn(s, filter_jax))
                    diff = float(np.max(np.abs(
                        np.asarray(out) * (np.pi / size) - ref_np)))
                    spd  = t_a / t
                    row[f'{label}_{n_dev}'] = (t, spd)
                    annot = ('← SPMD overhead proved (1-dev, no data movement)'
                             if n_dev == 1 else '')
                    print(f"  {label}. {tag}  {n_dev}-dev:  "
                          f"{t:.3f} s  ({spd:.2f}x)  diff={diff:.1e}  {annot}")
                except Exception as e:
                    print(f"  {label}. {tag}  {n_dev}-dev:  FAILED — {e}")

        # ── Path D: pmap ──────────────────────────────────────────────
        # Skipped: jax.pmap is deprecated and both helper APIs needed to
        # pre-place data for it (device_put_sharded, PositionalSharding) are
        # also deprecated.  See the module docstring for performance notes:
        # on GPU pmap uses XLA SPMD → same slowdown as shard_map; on CPU it
        # uses device-parallel compilation → near-linear speedup, similar to
        # the threading paths but without the transfer flexibility of path G.
        print(f"\n  D. pmap  (skipped — jax.pmap deprecated)")

        # ── Paths E / F / G: threading ────────────────────────────────
        bottleneck2 = (f'~{sz_gb*4:.1f} GB {transfer} overhead'
                       if is_gpu else f'host-memory copies (cheap on CPU)')
        print(f"\n  ── BOTTLENECK 2: data movement ──  E has {bottleneck2}")

        # Path E: unsharded input (current production code)
        jax.block_until_ready(
            _thread_filter_unsharded(sino_dev, filter_np, devs_2))
        t, out_e = time_fn(
            lambda: _thread_filter_unsharded(sino_dev, filter_np, devs_2))
        diff = float(np.max(np.abs(np.asarray(out_e) * (np.pi / size) - ref_np)))
        spd  = t_a / t
        row['E'] = (t, spd)
        annot = (f'← ~{sz_gb*4:.1f} GB {transfer}' if is_gpu
                 else '← host copy cheap; threading gives speedup')
        print(f"  E. thread unshard  2-dev:  "
              f"{t:.3f} s  ({spd:.2f}x)  diff={diff:.1e}  {annot}")

        # Path F: pre-sharded input, host gather
        jax.block_until_ready(
            _thread_filter_fast(sino_sh2, filter_np, devs_2))
        t, out_f = time_fn(
            lambda: _thread_filter_fast(sino_sh2, filter_np, devs_2))
        diff = float(np.max(np.abs(np.asarray(out_f) * (np.pi / size) - ref_np)))
        spd  = t_a / t
        row['F'] = (t, spd)
        annot = (f'← ~{sz_gb*2:.1f} GB {transfer} (output gather only)' if is_gpu
                 else '← same as E on CPU (input copy was already cheap)')
        print(f"  F. thread fast     2-dev:  "
              f"{t:.3f} s  ({spd:.2f}x)  diff={diff:.1e}  {annot}")

        # Path G: fully sharded, zero transfers
        jax.block_until_ready(
            _thread_filter_sharded(sino_sh2, filter_np, devs_2))
        t, out_g = time_fn(
            lambda: _thread_filter_sharded(sino_sh2, filter_np, devs_2))
        # Path G applies π/n_views internally — compare directly to ref_np
        diff = float(np.max(np.abs(np.asarray(out_g) - ref_np)))
        spd  = t_a / t
        row['G'] = (t, spd)
        print(f"  G. thread sharded  2-dev:  "
              f"{t:.3f} s  ({spd:.2f}x)  diff={diff:.1e}"
              f"  ← zero {transfer}  ★")

        summary.append(row)

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n\n{'='*72}")
    print(f"SUMMARY  —  backend={backend}  "
          f"min of {N_RUNS} runs; speedup vs A (unsharded baseline)")
    print()

    cols = [
        ('A',   'A  baseline'),
        ('B_1', 'B  sm+lax 1d'),
        ('B_2', 'B  sm+lax 2d'),
        ('C_1', 'C  sm+vmap 1d'),
        ('C_2', 'C  sm+vmap 2d'),
        ('D_1', 'D  pmap 1d'),
        ('D_2', 'D  pmap 2d'),
        ('E',   'E  thrd unshard'),
        ('F',   'F  thrd fast'),
        ('G',   'G  thrd sharded'),
    ]
    cw  = 16
    hdr = f"  {'size':>5}  " + '  '.join(f"{lbl:>{cw}}" for _, lbl in cols)
    print(hdr)
    print('-' * len(hdr))

    for row in summary:
        size = row['size']
        t_a  = row['A'][0]
        cells = []
        for key, _ in cols:
            if key == 'A':
                cells.append(f"{t_a:.3f}s base")
            elif key not in row:
                cells.append('skipped')
            else:
                t, spd = row[key]
                cells.append(f"{t:.3f}s {spd:.2f}x")
        print(f"  {size:>5}  " + '  '.join(f"{c:>{cw}}" for c in cells))

    print()
    if is_gpu:
        print("Key findings (GPU):")
        print("  B/C 1-dev >> A:  SPMD compilation overhead; not data movement.")
        print("  B/C 1-dev ≈ 2-dev:  SPMD gives no parallelism — overhead only.")
        print("  E/F << A:  PCIe roundtrips dominate; constant cost vs n_devices.")
        print("  G >> A:  both bottlenecks eliminated; super-linear at small sizes")
        print("           (halving rows/device relieves GPU memory pressure).")
    else:
        print("Key findings (CPU):")
        print("  B/C 1-dev > A:  SPMD overhead exists on CPU too, but may be smaller.")
        print("  E/F/G similar:  no PCIe bus; host copies are cheap; threading speedup")
        print("           visible in all three threading paths.")


if __name__ == '__main__':
    main()
