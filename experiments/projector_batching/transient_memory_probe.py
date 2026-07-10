"""Probe the projector drivers' compiled transient memory vs view_batch_size.

Answers three characterization questions with authoritative introspection
(compiled.memory_analysis()), not the source comments:

  1. How many coexisting [view_batch x pixel_batch x det_rows]-scale buffers does the
     forward driver's transient actually hold (the 'k' in the memory model
     view_batch ~ budget / (k * pixel_batch * det_rows * 4B))?
  2. Does the back driver's input reshape of the sinogram (projectors.py
     sum_function_in_batches, data_batched) materialize a full view-shard copy?
  3. Does the forward driver's output concatenate (concatenate_function_in_batches)
     materialize an extra owned-sinogram copy?

CPU-only, single device (the drivers are per-device programs; sharding never enters).
CAVEAT: fusion/buffer-assignment decisions are backend-specific -- CPU numbers give the
model's SHAPE; the GPU constants need a cluster run (flagged in the notes doc).

Run:  python transient_memory_probe.py   (no arguments; edit the config below)
"""

import os
os.environ.setdefault('MBIRJAX_NUM_CPU_DEVICES', '1')   # single device; must precede jax init

import numpy as np
import mbirjax

import jax.numpy as jnp
from mbirjax.projectors import (ProjectorParams, _jit_sparse_forward_project,
                                _jit_sparse_back_project)
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reference_batching import (_jit_sparse_forward_project_reference,
                                _jit_sparse_back_project_reference)

# ----------------------------------------------------------------------------------
# Config -- edit here
# ----------------------------------------------------------------------------------
N = 128                       # recon N^3, detector N x N
NUM_VIEWS = 384               # 3 full view batches at vb=128 (plus tails at other vb)
PIXEL_BATCH = 2048            # the current hardwired pixel batch
NUM_PIXELS = 3 * PIXEL_BATCH + 500   # 3 full pixel batches + a ragged tail
VIEW_BATCH_SIZES = [16, 32, 64, 128]

BYTES_F32 = 4


def build_model():
    angles = jnp.linspace(0, np.pi, NUM_VIEWS, endpoint=False)
    model = mbirjax.ConeBeamModel((NUM_VIEWS, N, N), angles,
                                  source_detector_dist=4.0 * N, source_iso_dist=2.0 * N)
    return model


def analyze(compiled, label):
    ma = compiled.memory_analysis()
    if ma is None:
        print(f'{label}: memory_analysis() unavailable on this backend')
        return None
    return ma


def main():
    model = build_model()
    sinogram_shape, recon_shape = model.get_params(['sinogram_shape', 'recon_shape'])
    geometry_params = model.get_geometry_parameters()
    pp = ProjectorParams(tuple(sinogram_shape), tuple(recon_shape), geometry_params)
    view_params = model.projector_functions.view_params_array

    num_slices = recon_shape[2]
    det_rows = sinogram_shape[1]
    voxel_values = jnp.zeros((NUM_PIXELS, num_slices), dtype=jnp.float32)
    pixel_indices = jnp.arange(NUM_PIXELS, dtype=jnp.int32)
    sinogram = jnp.zeros(tuple(sinogram_shape), dtype=jnp.float32)

    sino_bytes = int(np.prod(sinogram_shape)) * BYTES_F32
    cyl_bytes = NUM_PIXELS * num_slices * BYTES_F32
    print(f'cone N={N}, views={NUM_VIEWS}, pixels={NUM_PIXELS} '
          f'(sino {sino_bytes / 1e6:.1f} MB, cylinders {cyl_bytes / 1e6:.1f} MB)\n')

    def fwd_lower(driver):
        return lambda vb: driver.lower(
            view_params, voxel_values, pixel_indices,
            fwd_kernel=model.forward_project_pixel_batch_to_one_view,
            projector_params=pp, pixel_batch_size=PIXEL_BATCH, view_batch_size=vb)

    def back_lower(driver):
        return lambda vb: driver.lower(
            view_params, sinogram, pixel_indices,
            back_kernel=model.back_project_one_view_to_pixel_batch,
            projector_params=pp, pixel_batch_size=PIXEL_BATCH, view_batch_size=vb,
            coeff_power=1)

    results = {}
    for name, lower in [
        ('forward old(ref)', fwd_lower(_jit_sparse_forward_project_reference)),
        ('forward new(live)', fwd_lower(_jit_sparse_forward_project)),
        ('back old(ref)', back_lower(_jit_sparse_back_project_reference)),
        ('back new(live)', back_lower(_jit_sparse_back_project)),
    ]:
        print(f'--- {name} driver ---')
        per_vb = []
        for vb in VIEW_BATCH_SIZES:
            ma = analyze(lower(vb).compile(), f'{name} vb={vb}')
            if ma is None:
                break
            per_vb.append((vb, ma.temp_size_in_bytes))
            print(f'  vb={vb:4d}: temp {ma.temp_size_in_bytes / 1e6:9.1f} MB   '
                  f'args {ma.argument_size_in_bytes / 1e6:7.1f} MB   '
                  f'out {ma.output_size_in_bytes / 1e6:7.1f} MB')
        results[name] = per_vb
        if len(per_vb) >= 2:
            # Fit temp ~= slope * vb + const from the extreme points; express the slope
            # as an effective count of [vb x pixel_batch x det_rows] f32 buffers.
            (v0, t0), (v1, t1) = per_vb[0], per_vb[-1]
            slope = (t1 - t0) / (v1 - v0)
            unit = PIXEL_BATCH * det_rows * BYTES_F32
            const = t0 - slope * v0
            print(f'  fit: temp = {slope / 1e6:.3f} MB/view * vb + {const / 1e6:.1f} MB')
            print(f'  slope = {slope / unit:.2f} buffers of [vb x {PIXEL_BATCH} x {det_rows}] f32')
            print(f'  const term = {const / sino_bytes:.2f} x sinogram '
                  f'/ {const / cyl_bytes:.2f} x cylinder block\n')


if __name__ == '__main__':
    main()
