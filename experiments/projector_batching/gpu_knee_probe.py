"""GPU probe: vmap-width knee + transient-memory coefficient for the projector drivers.

Answers the three GPU-only questions left open by the CPU characterization
(projector_batching_characterization.md section 6):

  1. KNEE: is warm per-item time flat in view_batch_size down to ~B/2?  (Balanced
     batching can shrink the batch to B* = ceil(n/ceil(n/B_max)), worst case ~0.5*B_max;
     if time is flat down there, balancing is free in time as well as memory.)
  2. K COEFFICIENT: compiled temp bytes vs view_batch_size -> the effective number of
     coexisting [vb x pixel_batch x det_rows] f32 buffers (CPU measured 7.2 forward /
     1.5 back; a k ~4.4 for GPU can be DERIVED from the 1024^3 OOM comment at
     tomography_model.py ~L131, but that was the full recon path, not the isolated
     driver -- this probe measures the driver alone, one pixel batch).
  3. HIDDEN COPIES: the constant term of that fit, in units of the owned sinogram /
     cylinder block (CPU showed no hidden full-array copies; verify the GPU lowering).

Run on ONE GPU (the drivers are per-device programs; sharding never enters):

    CUDA_VISIBLE_DEVICES=0 python gpu_knee_probe.py

Expected runtime: dominated by ~2 x len(VIEW_BATCH_SIZES) x len(SIZES) compiles
(a few minutes each worst case); trim VIEW_BATCH_SIZES or SIZES below if needed.
Report back the full stdout -- the RESULTS blocks are self-describing.

Measurement hygiene (lessons.md section 5): warm-up call labeled and excluded; inputs
created on device before timing; one result live at a time; per-call timing in one
process; first-call (compile) time reported separately.
"""

import os
os.environ.setdefault('MBIRJAX_NUM_CPU_DEVICES', '1')   # harmless on GPU hosts

import time
import numpy as np
import mbirjax                                           # must precede jax (env binding)

import jax
import jax.numpy as jnp
from mbirjax.projectors import (ProjectorParams, _jit_sparse_forward_project,
                                _jit_sparse_back_project)

# ----------------------------------------------------------------------------------
# Config -- edit here
# ----------------------------------------------------------------------------------
SIZES = [256, 512]            # detector N x N, recon slices N (cone)
NUM_VIEWS = 3840              # divisible by every entry below -> no ragged tail,
                              # so the sweep isolates pure vmap width
VIEW_BATCH_SIZES = [16, 32, 48, 64, 80, 96, 128, 160, 192, 256]
PIXEL_BATCH = 2048            # current hardwired pixel batch
NUM_PIXELS = 2048             # ONE pixel batch -> pixel axis inert, view axis isolated
TIMED_REPS = 3

BYTES_F32 = 4


def build(N):
    angles = jnp.linspace(0, np.pi, NUM_VIEWS, endpoint=False)
    model = mbirjax.ConeBeamModel((NUM_VIEWS, N, N), angles,
                                  source_detector_dist=4.0 * N, source_iso_dist=2.0 * N)
    sinogram_shape, recon_shape = model.get_params(['sinogram_shape', 'recon_shape'])
    pp = ProjectorParams(tuple(sinogram_shape), tuple(recon_shape), model.get_geometry_parameters())
    view_params = model.projector_functions.view_params_array
    voxel_values = jnp.zeros((NUM_PIXELS, recon_shape[2]), dtype=jnp.float32)
    pixel_indices = jnp.arange(NUM_PIXELS, dtype=jnp.int32)
    sinogram = jnp.zeros(tuple(sinogram_shape), dtype=jnp.float32)
    return model, pp, view_params, voxel_values, pixel_indices, sinogram


def time_call(fn, reps):
    out = fn()
    jax.block_until_ready(out)          # warm-up / first call after compile
    del out
    best = float('inf')
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        jax.block_until_ready(out)
        best = min(best, time.perf_counter() - t0)
        del out                          # one live result at a time
    return best


def main():
    dev = jax.devices()[0]
    print(f'device: {dev.device_kind} ({dev.platform}), jax {jax.__version__}')
    print(f'views={NUM_VIEWS}, pixels={NUM_PIXELS}, pixel_batch={PIXEL_BATCH}, reps={TIMED_REPS}\n')

    for N in SIZES:
        model, pp, view_params, voxel_values, pixel_indices, sinogram = build(N)
        det_rows = N
        sino_bytes = NUM_VIEWS * det_rows * N * BYTES_F32
        cyl_bytes = NUM_PIXELS * N * BYTES_F32

        def fwd_pair(driver):
            def call(vb):
                return driver(
                    view_params, voxel_values, pixel_indices,
                    fwd_kernel=model.forward_project_pixel_batch_to_one_view,
                    projector_params=pp, pixel_batch_size=PIXEL_BATCH, view_batch_size=vb)
            return (lambda vb: driver.lower(
                        view_params, voxel_values, pixel_indices,
                        fwd_kernel=model.forward_project_pixel_batch_to_one_view,
                        projector_params=pp, pixel_batch_size=PIXEL_BATCH, view_batch_size=vb),
                    call)

        def back_pair(driver):
            def call(vb):
                return driver(
                    view_params, sinogram, pixel_indices,
                    back_kernel=model.back_project_one_view_to_pixel_batch,
                    projector_params=pp, pixel_batch_size=PIXEL_BATCH, view_batch_size=vb,
                    coeff_power=1)
            return (lambda vb: driver.lower(
                        view_params, sinogram, pixel_indices,
                        back_kernel=model.back_project_one_view_to_pixel_batch,
                        projector_params=pp, pixel_batch_size=PIXEL_BATCH, view_batch_size=vb,
                        coeff_power=1),
                    call)

        drivers = [
            ('forward',) + fwd_pair(_jit_sparse_forward_project),
            ('back',) + back_pair(_jit_sparse_back_project),
        ]

        for name, lower, call in drivers:
            print(f'--- {name}, N={N} (sino {sino_bytes / 1e9:.2f} GB) ---')
            print(f'{"vb":>4s} {"compile_s":>10s} {"warm_ms":>9s} {"per_Mitem_us":>13s} {"temp_MB":>9s}')
            per_vb = []
            for vb in VIEW_BATCH_SIZES:
                t0 = time.perf_counter()
                compiled = lower(vb).compile()
                compile_s = time.perf_counter() - t0
                ma = compiled.memory_analysis()
                temp = ma.temp_size_in_bytes if ma is not None else float('nan')

                best = time_call(lambda: call(vb), TIMED_REPS)
                per_item_us = best / (NUM_VIEWS * NUM_PIXELS) * 1e6 * 1e6  # us per M items
                per_vb.append((vb, temp, best))
                print(f'{vb:4d} {compile_s:10.1f} {best * 1e3:9.1f} {per_item_us:13.2f} '
                      f'{temp / 1e6:9.1f}')

            # Fits: temp slope in buffer units; time flatness vs the vb=128 reference.
            # Only meaningful with >= 2 distinct vb values (a trimmed config may have 1).
            (v0, t0_, _), (v1, t1_, _) = per_vb[0], per_vb[-1]
            if v1 > v0:
                slope = (t1_ - t0_) / (v1 - v0)
                const = t0_ - slope * v0
                unit = PIXEL_BATCH * det_rows * BYTES_F32
                print(f'RESULTS {name} N={N}: k = {slope / unit:.2f} buffers of '
                      f'[vb x {PIXEL_BATCH} x {det_rows}] f32; '
                      f'const = {const / sino_bytes:.2f} x sino '
                      f'/ {const / cyl_bytes:.2f} x cylinders')
            ref = next((t for vb, _, t in per_vb if vb == 128), per_vb[-1][2])
            print('RESULTS time vs vb=128: ' + '  '.join(
                f'{vb}:{t / ref:.2f}x' for vb, _, t in per_vb) + '\n')

        # Free this size's arrays before the next size
        del model, pp, view_params, voxel_values, pixel_indices, sinogram


if __name__ == '__main__':
    main()
