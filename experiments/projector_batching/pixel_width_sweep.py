"""Pixel-batch-width sweep on the projector drivers -- testing the power-of-2 hypothesis.

Motivation (see the decision record in batching_refactor_design.md): at identical shapes
(51.4k pixels, L=256, 512 views, H100) the retired v2 code ran the BAND back driver ~10%
faster than v1 while the monolithic back driver was at parity, and elimination left one
candidate: v2's pixel-batch width was 1977 where v1's is 2048.  Hypothesis: a POWER-OF-2
pixel stride causes cache-set conflicts / partition camping in the band kernel's
[views x pixels x L] intermediates -- the kernel ncu profiling flagged as memory-ACCESS-
PATTERN-bound -- and an off-power width breaks the conflict pattern.  The monolithic back
kernel's different access pattern would explain its indifference.

This sweep measures warm time vs pixel_batch_size (a static driver argument -- no library
changes needed) on the LIVE code, for the band driver (primary, at two pixel counts /
band lengths) and the monolithic back + forward drivers (controls, expected flat).  The
sweep discriminates a power-of-2 CLIFF (2048 slow; 2040/2047/2049 fast) from a broad curve
(gradual in width) from a null result (v2's win came from something still unidentified).

fp note: the pixel axis is the CONCATENATE axis in back/band (per-pixel outputs
independent), so pixel width regroups NO sums there -- the rel_max column vs the width-2048
output should be ~0.  Forward's pixel axis IS a sum axis (rel ~1e-7 expected).

Run on ONE GPU:

    CUDA_VISIBLE_DEVICES=0 python pixel_width_sweep.py

Report back the full stdout (RESULT lines carry the verdict).  Expect ~10 minutes.
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

# ----------------------------------------------------------------------------------
# Config -- edit here
# ----------------------------------------------------------------------------------
N = 1024                     # detector N x N, recon slices N (cone)
NUM_VIEWS = 512              # divisible by the view batch -> the view axis is inert
VIEW_BATCH = 128
# Pixel-batch widths: dense around 2048 to discriminate a power-of-2 cliff from a broad
# curve; 1977 is the exact width the v2 code used at the 51.4k cell.
PIXEL_WIDTHS = [1536, 1792, 1920, 1977, 2016, 2040, 2047, 2048, 2049, 2056, 2176, 2304]
REFERENCE_WIDTH = 2048       # ratios and the fp cross-check are against this width
# (driver, num_pixels, band_L) cells.  band at both granularity-16 and coarse shapes;
# back/forward as controls at the cell where the v2 effect was observed.
CELLS = [
    ('band', 51400, 256),
    ('band', 823000, 103),
    ('back', 51400, None),
    ('forward', 51400, None),
]
BAND_G0 = 128                # a non-edge band start (traced, like the real band loop)
COEFF_POWER = 1
TIMED_REPS = 3

_START = time.perf_counter()


def log(msg):
    print(f'[{time.perf_counter() - _START:7.1f}s] {msg}', flush=True)


def rel_max_err(out, ref):
    out, ref = np.asarray(out), np.asarray(ref)
    denom = float(np.max(np.abs(ref)))
    return float(np.max(np.abs(out - ref))) / denom if denom else float(np.max(np.abs(out)))


def time_call(fn, reps, label):
    log(f'  {label}: compile + warm-up call ...')
    t0 = time.perf_counter()
    out = fn()
    jax.block_until_ready(out)
    del out
    log(f'  {label}: warm-up done ({time.perf_counter() - t0:.1f} s incl. compile)')
    best = float('inf')
    for rep in range(reps):
        t0 = time.perf_counter()
        out = fn()
        jax.block_until_ready(out)
        elapsed = time.perf_counter() - t0
        best = min(best, elapsed)
        del out
        log(f'  {label}: rep {rep + 1}/{reps} {elapsed * 1e3:.1f} ms')
    return best


def main():
    dev = jax.devices()[0]
    print(f'device: {dev.device_kind} ({dev.platform}), jax {jax.__version__}')
    print(f'N={N}, views={NUM_VIEWS}, view_batch={VIEW_BATCH}, widths={PIXEL_WIDTHS}, '
          f'cells={CELLS}, reps={TIMED_REPS}\n')

    angles = jnp.linspace(0, np.pi, NUM_VIEWS, endpoint=False)
    model = mbirjax.ConeBeamModel((NUM_VIEWS, N, N), angles,
                                  source_detector_dist=4.0 * N, source_iso_dist=2.0 * N)
    sinogram_shape, recon_shape = model.get_params(['sinogram_shape', 'recon_shape'])
    pp = ProjectorParams(tuple(sinogram_shape), tuple(recon_shape),
                         model.get_geometry_parameters())
    view_params = model.projector_functions.view_params_array
    owned = jnp.arange(NUM_VIEWS)

    rng = np.random.default_rng(0)
    sinogram = jnp.array(rng.standard_normal(tuple(sinogram_shape)), dtype=jnp.float32)

    for cell_idx, (driver_name, num_pixels, band_l) in enumerate(CELLS):
        pixel_indices = jnp.array(rng.choice(recon_shape[0] * recon_shape[1],
                                             size=num_pixels, replace=False), dtype=jnp.int32)
        voxel_values = jnp.array(rng.standard_normal((num_pixels, recon_shape[2])),
                                 dtype=jnp.float32)

        def make_call(width):
            if driver_name == 'forward':
                return lambda: _jit_sparse_forward_project(
                    view_params, voxel_values, pixel_indices,
                    fwd_kernel=model.forward_project_pixel_batch_to_one_view,
                    projector_params=pp, pixel_batch_size=width,
                    view_batch_size=VIEW_BATCH, owned_view_indices=owned)
            if driver_name == 'back':
                return lambda: _jit_sparse_back_project(
                    view_params, sinogram, pixel_indices,
                    back_kernel=model.back_project_one_view_to_pixel_batch,
                    projector_params=pp, pixel_batch_size=width,
                    view_batch_size=VIEW_BATCH, coeff_power=COEFF_POWER,
                    owned_view_indices=owned)
            return lambda: _jit_sparse_back_project_band(
                view_params, sinogram, pixel_indices, BAND_G0, band_l,
                back_band_kernel=model.back_project_one_view_to_band,
                projector_params=pp, pixel_batch_size=width,
                view_batch_size=VIEW_BATCH, coeff_power=COEFF_POWER,
                owned_view_indices=owned)

        print(f'=== cell {cell_idx + 1}/{len(CELLS)}: {driver_name}, pixels={num_pixels}'
              + (f', L={band_l}' if band_l else '') + ' ===', flush=True)

        # Reference output + time at the current default width.
        ref_call = make_call(REFERENCE_WIDTH)
        t_ref = time_call(ref_call, TIMED_REPS, f'{driver_name} width={REFERENCE_WIDTH} (ref)')
        ref_out = np.asarray(ref_call())

        results = [(REFERENCE_WIDTH, t_ref, 0.0)]
        for width in PIXEL_WIDTHS:
            if width == REFERENCE_WIDTH:
                continue
            call = make_call(width)
            t = time_call(call, TIMED_REPS, f'{driver_name} width={width}')
            err = rel_max_err(call(), ref_out)
            results.append((width, t, err))

        del ref_out
        for width, t, err in sorted(results):
            print(f'RESULT {driver_name:8s} pixels={num_pixels:7d} width={width:5d}  '
                  f'{t * 1e3:8.1f} ms   vs2048 {t / t_ref:5.3f}   rel_max {err:.1e}',
                  flush=True)
        print(flush=True)
        del pixel_indices, voxel_values


if __name__ == '__main__':
    main()
