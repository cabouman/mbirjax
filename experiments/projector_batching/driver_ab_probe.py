"""Driver-level v1-vs-v2 A/B at realistic shapes -- the timing gate for the default flip.

Times the three projector drivers in isolation (everything above projectors.py is
version-independent), at shapes matching what the 1024^3/2-GPU recon actually feeds them.
This covers the two mechanisms the earlier gpu_knee_probe runs never timed on GPU:

  1. map_in_balanced_batches with MANY batches: the earlier probes used pixels=2048 -- a
     single pixel batch, so the map helper was inert.  Real calls run tens-to-hundreds of
     pixel batches, where v2's balanced width can be 2047 instead of 2048.
  2. the BAND back driver (_sparse_back_project_band) -- the multi-GPU back path; earlier
     probes ran only the monolithic back.

The view axis runs both DIVISIBLE (512 = 4 x 128; v2 zero-init scan) and RAGGED (471; v1
odd batch 87 vs v2 balanced 118x3+117) counts.  Each cell also cross-checks v2 vs v1 with
the scale-invariant rel-max (expect <= ~1e-5; sum-order noise only).

Run on ONE GPU:

    CUDA_VISIBLE_DEVICES=0 python driver_ab_probe.py

Report back the full stdout.  Cell work is ~26G (view,pixel) pairs for back/forward
(~3-10 s warm per call on H100), 12 timed cells + compiles: expect ~5-15 minutes total.
"""

import os
os.environ.setdefault('MBIRJAX_NUM_CPU_DEVICES', '1')   # harmless on GPU hosts

import time
import numpy as np
import mbirjax                                           # must precede jax (env binding)

import jax
import jax.numpy as jnp
from mbirjax.projectors import (
    ProjectorParams,
    _jit_sparse_forward_project, _jit_sparse_forward_project_v2,
    _jit_sparse_back_project, _jit_sparse_back_project_v2,
    _jit_sparse_back_project_band, _jit_sparse_back_project_band_v2)

# ----------------------------------------------------------------------------------
# Config -- edit here.  Defaults mirror the 1024^3 / 2-GPU recon's driver calls.
# ----------------------------------------------------------------------------------
N = 1024                     # detector N x N, recon slices N (cone)
VIEW_COUNTS = [512, 471]     # views owned per device: divisible by 128 / ragged
NUM_PIXELS = 51400           # ~ a granularity-16 subset of the 1024^2 RoR (26 pixel batches)
BAND_SLICES = 256            # _slice_band_length at these shapes (reduce bound, n_dev=2)
BAND_G0 = 128                # a non-edge band start (traced, like the real band loop)
VIEW_BATCH = 128
PIXEL_BATCH = 2048
TIMED_REPS = 3

BYTES_F32 = 4


_START = time.perf_counter()


def log(msg):
    """Timestamped, flushed progress line -- nothing should be silent for long."""
    print(f'[{time.perf_counter() - _START:7.1f}s] {msg}', flush=True)


def rel_max_err(out, ref):
    out, ref = np.asarray(out), np.asarray(ref)
    denom = float(np.max(np.abs(ref)))
    return float(np.max(np.abs(out - ref))) / denom if denom else float(np.max(np.abs(out)))


def time_call(fn, reps, label):
    log(f'  {label}: compile + warm-up call ...')
    t0 = time.perf_counter()
    out = fn()
    jax.block_until_ready(out)          # warm-up (first execution after compile)
    del out
    log(f'  {label}: warm-up done ({time.perf_counter() - t0:.1f} s incl. compile)')
    best = float('inf')
    for rep in range(reps):
        t0 = time.perf_counter()
        out = fn()
        jax.block_until_ready(out)
        elapsed = time.perf_counter() - t0
        best = min(best, elapsed)
        del out                          # one live result at a time
        log(f'  {label}: rep {rep + 1}/{reps} {elapsed * 1e3:.1f} ms')
    return best


def main():
    dev = jax.devices()[0]
    print(f'device: {dev.device_kind} ({dev.platform}), jax {jax.__version__}')
    print(f'N={N}, pixels={NUM_PIXELS}, view_batch={VIEW_BATCH}, pixel_batch={PIXEL_BATCH}, '
          f'band L={BAND_SLICES} g0={BAND_G0}, reps={TIMED_REPS}\n')

    max_views = max(VIEW_COUNTS)
    angles = jnp.linspace(0, np.pi, max_views, endpoint=False)
    model = mbirjax.ConeBeamModel((max_views, N, N), angles,
                                  source_detector_dist=4.0 * N, source_iso_dist=2.0 * N)
    sinogram_shape, recon_shape = model.get_params(['sinogram_shape', 'recon_shape'])
    pp = ProjectorParams(tuple(sinogram_shape), tuple(recon_shape),
                         model.get_geometry_parameters())
    view_params = model.projector_functions.view_params_array

    rng = np.random.default_rng(0)
    pixel_indices = jnp.array(rng.choice(recon_shape[0] * recon_shape[1], size=NUM_PIXELS,
                                         replace=False), dtype=jnp.int32)
    voxel_values = jnp.array(rng.standard_normal((NUM_PIXELS, recon_shape[2])),
                             dtype=jnp.float32)
    full_sino = jnp.array(rng.standard_normal(tuple(sinogram_shape)), dtype=jnp.float32)

    for num_views in VIEW_COUNTS:
        owned = jnp.arange(num_views)
        local_sino = full_sino[owned]
        tag = 'divisible' if num_views % VIEW_BATCH == 0 else 'ragged'
        print(f'=== views={num_views} ({tag}) ===')

        def fwd(driver):
            return lambda: driver(
                view_params, voxel_values, pixel_indices,
                fwd_kernel=model.forward_project_pixel_batch_to_one_view,
                projector_params=pp, pixel_batch_size=PIXEL_BATCH,
                view_batch_size=VIEW_BATCH, owned_view_indices=owned)

        def back(driver):
            return lambda: driver(
                view_params, local_sino, pixel_indices,
                back_kernel=model.back_project_one_view_to_pixel_batch,
                projector_params=pp, pixel_batch_size=PIXEL_BATCH,
                view_batch_size=VIEW_BATCH, coeff_power=1, owned_view_indices=owned)

        def band(driver):
            return lambda: driver(
                view_params, local_sino, pixel_indices, BAND_G0, BAND_SLICES,
                back_band_kernel=model.back_project_one_view_to_band,
                projector_params=pp, pixel_batch_size=PIXEL_BATCH,
                view_batch_size=VIEW_BATCH, coeff_power=1, owned_view_indices=owned)

        for name, make in [('forward', fwd), ('back', back), ('band', band)]:
            cell = VIEW_COUNTS.index(num_views) * 3 + ['forward', 'back', 'band'].index(name) + 1
            total = 3 * len(VIEW_COUNTS)
            log(f'cell {cell}/{total}: {name}, views={num_views}')
            v1_call = make({'forward': _jit_sparse_forward_project,
                            'back': _jit_sparse_back_project,
                            'band': _jit_sparse_back_project_band}[name])
            v2_call = make({'forward': _jit_sparse_forward_project_v2,
                            'back': _jit_sparse_back_project_v2,
                            'band': _jit_sparse_back_project_band_v2}[name])
            t1 = time_call(v1_call, TIMED_REPS, f'{name} v1')
            t2 = time_call(v2_call, TIMED_REPS, f'{name} v2')
            # Correctness cross-check on the same inputs (sum-order noise only).
            log(f'  {name}: correctness cross-check (one more call per version) ...')
            err = rel_max_err(v2_call(), v1_call())
            print(f'RESULT {name:8s} views={num_views:4d}  v1 {t1 * 1e3:8.1f} ms   '
                  f'v2 {t2 * 1e3:8.1f} ms   v2/v1 {t2 / t1:5.3f}   rel_max {err:.1e}',
                  flush=True)
        print(flush=True)


if __name__ == '__main__':
    main()
