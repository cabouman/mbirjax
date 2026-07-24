"""Driver-level old-vs-new A/B of the windowed-read batching patch, at realistic shapes.

OLD = the frozen pre-change sum_function_in_batches (reshape + scan-over-xs; see
reference_batching.py in this directory -- the package itself carries only ONE batching
implementation).  NEW = the live mbirjax.projectors.  The single changed mechanism is how
the sum-axis batches are READ (dynamic_slice windows vs a pre-reshaped input copy); the
batch sizes, order, and partial sums are IDENTICAL, so the rel-max cross-check per cell
should sit at different-executable noise (~1e-7 class), and any timing/memory difference is
attributable to the input copy alone.

Cells span the VCD granularity levels (823k..6.4k pixels), band lengths (103/256), and
divisible + ragged view counts, for all three drivers.  Forward runs only at the small
pixel cells (its sum axis IS the pixel axis, so multi-pixel-batch cells do exercise the
change; the 823k forward would cost minutes per call for the same information).

Run on ONE GPU, from this directory (reference_batching.py must be importable):

    CUDA_VISIBLE_DEVICES=0 python driver_ab_probe.py

Report back the full stdout.  Expect ~10-20 minutes, dominated by the 823k cells.
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
    _jit_sparse_forward_project, _jit_sparse_back_project, _jit_sparse_back_project_band)
# The 'old' side of the A/B: the frozen pre-change batching, kept OUTSIDE the package
# (reference_batching.py, a sibling module in this directory).
from reference_batching import (
    _jit_sparse_forward_project_reference,
    _jit_sparse_back_project_reference,
    _jit_sparse_back_project_band_reference)

# ----------------------------------------------------------------------------------
# Config -- edit here.  Defaults mirror the 1024^3 / 2-GPU recon's driver calls.
# ----------------------------------------------------------------------------------
N = 1024                     # detector N x N, recon slices N (cone)
VIEW_COUNTS = [512, 471]     # views owned per device: divisible by 128 / ragged
# (num_pixels, band_L) cells.  num_pixels spans the VCD granularity levels of a 1024^2 RoR
# (823k=coarse ... 6.4k=finest); band_L follows _slice_band_length at n_dev=2 (reduce bound
# 256, compute bound 100e6/num_pixels balanced over 512 slices -> ~103 at the coarse level).
PIXEL_CELLS = [
    (823000, 103),           # granularity 1 (+ the Hessian's pixel count)
    (51400, 256),            # granularity 16 (the original probe cell)
    (6430, 256),             # granularity 128
]
BAND_G0 = 128                # a non-edge band start (traced, like the real band loop)
VIEW_BATCH = 128
PIXEL_BATCH = 2048
COEFF_POWER = 1              # 2 = the Hessian-diagonal path (back/band only)
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
    print(f'N={N}, pixel cells={PIXEL_CELLS}, view_batch={VIEW_BATCH}, '
          f'pixel_batch={PIXEL_BATCH}, g0={BAND_G0}, coeff_power={COEFF_POWER}, '
          f'reps={TIMED_REPS}\n')

    max_views = max(VIEW_COUNTS)
    angles = jnp.linspace(0, np.pi, max_views, endpoint=False)
    model = mbirjax.ConeBeamModel((max_views, N, N), angles,
                                  source_detector_dist=4.0 * N, source_iso_dist=2.0 * N)
    sinogram_shape, recon_shape = model.get_params(['sinogram_shape', 'recon_shape'])
    pp = ProjectorParams(tuple(sinogram_shape), tuple(recon_shape),
                         model.get_geometry_parameters())
    view_params = model.projector_functions.view_params_array

    rng = np.random.default_rng(0)
    full_sino = jnp.array(rng.standard_normal(tuple(sinogram_shape)), dtype=jnp.float32)

    # Forward's sum axis is the PIXEL axis, so the small multi-pixel-batch cells already
    # exercise its changed read path; the 823k forward would cost ~minutes/call for the
    # same information, so it is skipped.
    FORWARD_PIXEL_LIMIT = 60000

    cell_num, total = 0, sum((2 if npix > FORWARD_PIXEL_LIMIT else 3) * len(VIEW_COUNTS)
                             for npix, _ in PIXEL_CELLS)
    for num_pixels, band_slices in PIXEL_CELLS:
        pixel_indices = jnp.array(rng.choice(recon_shape[0] * recon_shape[1],
                                             size=num_pixels, replace=False), dtype=jnp.int32)
        voxel_values = jnp.array(rng.standard_normal((num_pixels, recon_shape[2])),
                                 dtype=jnp.float32)

        for num_views in VIEW_COUNTS:
            owned = jnp.arange(num_views)
            local_sino = full_sino[owned]
            tag = 'divisible' if num_views % VIEW_BATCH == 0 else 'ragged'
            print(f'=== pixels={num_pixels}, band L={band_slices}, views={num_views} ({tag}), '
                  f'coeff_power={COEFF_POWER} ===', flush=True)

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
                    view_batch_size=VIEW_BATCH, coeff_power=COEFF_POWER,
                    owned_view_indices=owned)

            def band(driver):
                return lambda: driver(
                    view_params, local_sino, pixel_indices, BAND_G0, band_slices,
                    back_band_kernel=model.back_project_one_view_to_band,
                    projector_params=pp, pixel_batch_size=PIXEL_BATCH,
                    view_batch_size=VIEW_BATCH, coeff_power=COEFF_POWER,
                    owned_view_indices=owned)

            cases = [('back', back), ('band', band)]
            if num_pixels <= FORWARD_PIXEL_LIMIT:
                cases.insert(0, ('forward', fwd))
            for name, make in cases:
                cell_num += 1
                log(f'cell {cell_num}/{total}: {name}, pixels={num_pixels}, views={num_views}')
                old_call = make({'forward': _jit_sparse_forward_project_reference,
                                 'back': _jit_sparse_back_project_reference,
                                 'band': _jit_sparse_back_project_band_reference}[name])
                new_call = make({'forward': _jit_sparse_forward_project,
                                 'back': _jit_sparse_back_project,
                                 'band': _jit_sparse_back_project_band}[name])
                t1 = time_call(old_call, TIMED_REPS, f'{name} old(reference)')
                t2 = time_call(new_call, TIMED_REPS, f'{name} new(live)')
                # Correctness cross-check on the same inputs (sum-order noise only).
                log(f'  {name}: correctness cross-check (one more call per version) ...')
                err = rel_max_err(new_call(), old_call())
                print(f'RESULT {name:8s} pixels={num_pixels:7d} L={band_slices:4d} '
                      f'views={num_views:4d}  old {t1 * 1e3:8.1f} ms   new {t2 * 1e3:8.1f} ms   '
                      f'new/old {t2 / t1:5.3f}   rel_max {err:.1e}', flush=True)
            print(flush=True)

        del pixel_indices, voxel_values     # free before the next (possibly larger) cell


if __name__ == '__main__':
    main()
