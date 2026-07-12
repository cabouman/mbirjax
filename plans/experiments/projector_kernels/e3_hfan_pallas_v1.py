"""E3 first real kernel, v1: the HFAN-FORWARD CSR segment-walk in Pallas-Triton.

Replaces the sorted-reduce channel scatter (the fusion that is 86-88% of parallel
forward device time) with a GATHER kernel over an eagerly-precomputed sorted structure
(enabled by concrete centers + fixed partitions — the Phase-D idiom extended):

  precompute (eager XLA/numpy, per view): sort the T*P (tap, pixel) pairs by channel ->
      sorted (weight, pixel-id) streams + per-channel segment starts (searchsorted).
  kernel (grid = (V, C)): one program per (view, channel) walks its segment
      [start[v,c], start[v,c+1]) with a dynamic-trip-count fori_loop, ref-gathers pixel
      rows from the L2-resident values tile, accumulates the (B,)-vector in registers,
      stores the output row ONCE.  No atomics, no runtime sort, static control flow.
      Load imbalance (channel skew on raster batches) is v1's accepted risk — v2 would
      even-partition the tap stream with boundary fixup (ModernGPU segreduce) if the
      skewed case suffers.

Cases: (a) raster full-grid pixel batch (REAL channel skew — the campaign's collision
lesson) and (b) a random VCD-subset batch (uniform, ~24 taps/channel), both at the
1024^3-class parallel cell (P=8192, B=252->256 padded, C=992, V=128, T=3), geometry from
the real model so n_p/W/centers are production-realistic.

Baseline: the library kernel path (horizontal_fan_project with use_sorted=True) vmapped
over views, timed as the campaign benches time kernels.  Value gate: rel-max <= 1e-5
(summation order within a channel run matches the sorted stream, so agreement may be
bitwise; not required).  Success bar (plan E3): >=1.5-2x at BOTH cases.

Run:  python plans/experiments/projector_kernels/e3_hfan_pallas_v1.py   (constants below)
"""
import os

# ── Config ────────────────────────────────────────────────────────────────────
SINO_SHAPE = (1024, 1008, 992)      # 1024^3-class parallel cell
P_BATCH = 8192
BAND = 252                          # fwd_slice_band at this cell (balanced 1008/4)
B_PAD = 256                         # Triton power-of-2 requirement
VIEW_BATCH = 128
SUBSET_PIXELS = 8192                # VCD-style random batch (same P for apples-to-apples)
NUM_WARPS_SWEEP = [1, 2, 4]
WARMUP, TRIALS = 2, 10

os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.85')
import time                        # noqa: E402
from functools import partial      # noqa: E402
import numpy as np                 # noqa: E402
import mbirjax                     # noqa: E402
import jax                         # noqa: E402
import jax.numpy as jnp            # noqa: E402
from jax.experimental import pallas as pl          # noqa: E402
from mbirjax.parallel_beam import ParallelBeamModel            # noqa: E402
from mbirjax.projectors import (ProjectorParams, horizontal_fan_project,   # noqa: E402
                                _jit_compute_scatter_centers)

T = 3          # psf_width at radius 1 (asserted against the model below)


def build_case(kind, sino_shape, p_batch, band, view_batch):
    """Real-geometry inputs for one pixel batch: values, streams, and the XLA baseline args."""
    num_views, num_det_rows, num_det_channels = sino_shape
    angles = np.linspace(0, np.pi, num_views, endpoint=False)
    model = mbirjax.ParallelBeamModel(sino_shape, angles)
    model.configure_devices(1)
    args = (tuple(model.get_params('sinogram_shape')),
            tuple(model.get_params('recon_shape')), model.get_geometry_parameters())
    args += tuple(int(bool(getattr(model.tiles, f, 0))) for f in ProjectorParams._fields[3:])
    pp = ProjectorParams(*args)
    recon_shape = model.get_params('recon_shape')
    rng = np.random.default_rng(0)
    full_idx = np.asarray(mbirjax.gen_full_indices(
        recon_shape, use_ror_mask=model.get_params('use_ror_mask')))
    if kind == 'raster':
        idx = jnp.asarray(full_idx[:p_batch].astype(np.int32))     # a driver scan step
    else:
        idx = jnp.asarray(np.sort(rng.choice(full_idx, size=min(p_batch, len(full_idx)),
                                             replace=False)).astype(np.int32))
    values = jnp.asarray(rng.random((len(idx), band), dtype=np.float32))
    view_params = jnp.asarray(model.projector_functions.view_params_array)[:view_batch]
    n_pc = _jit_compute_scatter_centers(
        jnp.asarray(model.projector_functions.view_params_array), idx,
        ParallelBeamModel.compute_channel_coordinate, pp, pixels_major=False)[:view_batch]
    assert pp.geometry_params.psf_radius * 2 + 1 == T
    return model, pp, idx, values, view_params, n_pc


def xla_baseline(pp, idx, values, view_params, n_pc):
    """The library hfan (sorted reduce) vmapped over views -> (V, C, band)."""
    gp = pp.geometry_params
    num_det_channels = pp.sinogram_shape[2]

    def one_view(single_view_params, centers):
        n_p, W_p_c, footprint = ParallelBeamModel.compute_proj_data(
            idx, single_view_params, pp)
        delta_voxel_row = gp.voxel_row_aspect * gp.delta_voxel
        scale = (delta_voxel_row * gp.delta_voxel) / footprint
        return horizontal_fan_project(n_p, centers, W_p_c, scale, values,
                                      num_det_channels, gp.psf_radius, use_sorted=True)
    return jax.vmap(one_view)(view_params, n_pc)


def precompute_streams(pp, idx, view_params, n_pc):
    """Eager per-view sorted (weight, pixel) streams + channel segment starts."""
    gp = pp.geometry_params
    C = pp.sinogram_shape[2]

    def one_view(single_view_params, centers):
        n_p, W_p_c, footprint = ParallelBeamModel.compute_proj_data(
            idx, single_view_params, pp)
        delta_voxel_row = gp.voxel_row_aspect * gp.delta_voxel
        scale = (delta_voxel_row * gp.delta_voxel) / footprint
        offs = jnp.arange(-gp.psf_radius, gp.psf_radius + 1)
        n = centers[None, :] + offs[:, None]                       # (T, P)
        L_max = jnp.minimum(1.0, W_p_c)
        A = scale * jnp.clip((W_p_c + 1.0) / 2.0 - jnp.abs(n_p - n), 0.0, L_max)
        A = A * ((n >= 0) & (n < C))
        n = jnp.clip(n, 0, C - 1)
        flat_n = n.reshape(-1)
        sorted_n, order = jax.lax.sort_key_val(flat_n, jnp.arange(flat_n.shape[0]))
        pix = (order % num_pixels).astype(jnp.int32)
        wts = A.reshape(-1)[order]
        starts = jnp.searchsorted(sorted_n, jnp.arange(C + 1)).astype(jnp.int32)
        return wts, pix, starts

    num_pixels = len(idx)
    wts, pix, starts = jax.jit(jax.vmap(one_view))(view_params, n_pc)
    return (jax.block_until_ready(wts), jax.block_until_ready(pix),
            jax.block_until_ready(starts))


def hfan_kernel(starts_ref, ends_ref, wt_ref, pix_ref, vals_ref, out_ref):
    """One program per (view, channel): walk the segment, gather rows, store once."""
    start = starts_ref[0, 0]
    end = ends_ref[0, 0]

    def body(i, acc):
        p = pix_ref[0, i]
        wgt = wt_ref[0, i]
        return acc + wgt * vals_ref[p, :]           # ref-level dynamic row gather
    acc = jax.lax.fori_loop(start, end, body, jnp.zeros((out_ref.shape[-1],), jnp.float32))
    out_ref[0, 0, :] = acc


def make_hfan_call(V, C, TP, P, b_pad, num_warps, interpret=False):
    from jax.experimental.pallas import triton as pltriton
    kw = ({} if interpret else
          {'compiler_params': pltriton.CompilerParams(num_warps=num_warps)})
    return pl.pallas_call(
        hfan_kernel,
        out_shape=jax.ShapeDtypeStruct((V, C, b_pad), jnp.float32),
        grid=(V, C),
        in_specs=[pl.BlockSpec((1, 1), lambda v, c: (v, c)),        # starts[v, c]
                  pl.BlockSpec((1, 1), lambda v, c: (v, c)),        # ends = starts[v, c+1]
                  pl.BlockSpec((1, TP), lambda v, c: (v, 0)),       # the view's wt stream
                  pl.BlockSpec((1, TP), lambda v, c: (v, 0)),       # the view's pix stream
                  pl.BlockSpec((P, b_pad), lambda v, c: (0, 0))],   # shared values tile
        out_specs=pl.BlockSpec((1, 1, b_pad), lambda v, c: (v, c, 0)),
        interpret=interpret, **kw)


def bench(fn, args, name):
    for _ in range(WARMUP):
        jax.block_until_ready(fn(*args))
    ts = []
    for _ in range(TRIALS):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append(time.perf_counter() - t0)
    med = float(np.median(ts))
    print(f'  {name:26s} {med * 1e3:9.3f} ms/call', flush=True)
    return med


def main():
    on_gpu = jax.devices()[0].platform == 'gpu'
    print(f'jax {jax.__version__}  devices={jax.devices()}', flush=True)
    # Tiny geometry for the CPU interpret gate (grid programs execute sequentially in
    # interpret mode); production shapes on GPU.
    sino_shape, p_batch, band, b_pad, view_batch = (
        (SINO_SHAPE, P_BATCH, BAND, B_PAD, VIEW_BATCH) if on_gpu
        else ((64, 24, 32), 256, 16, 16, 2))
    for kind in ('raster', 'subset'):
        print(f'\n===== case: {kind} =====', flush=True)
        model, pp, idx, values, view_params, n_pc = build_case(
            kind, sino_shape, p_batch, band, view_batch)
        V, C = view_batch, pp.sinogram_shape[2]
        P = len(idx)
        TP = T * P
        wts, pix, starts = precompute_streams(pp, idx, view_params, n_pc)
        seg_starts = starts[:, :-1]                    # (V, C)
        seg_ends = starts[:, 1:]                       # (V, C)
        vals_pad = jnp.pad(values, ((0, 0), (0, b_pad - band)))
        counts = np.asarray(seg_ends - seg_starts)
        print(f'P={P} C={C} taps/channel: mean {counts.mean():.1f} '
              f'max {counts.max()} (skew {counts.max() / max(counts.mean(), 1e-9):.1f}x)',
              flush=True)

        base_fn = jax.jit(lambda vp, npc: xla_baseline(pp, idx, values, vp, npc))
        ref = jax.block_until_ready(base_fn(view_params, n_pc))    # (V, C, band)
        scale = float(jnp.max(jnp.abs(ref)))

        if not on_gpu:
            out = make_hfan_call(V, C, TP, P, b_pad, 1, interpret=True)(
                seg_starts, seg_ends, wts, pix, vals_pad)[..., :band]
            rel = float(jnp.max(jnp.abs(out - ref)) / scale)
            print(f'[interpret {kind}] rel {rel:.3g} {"PASS" if rel < 1e-5 else "FAIL"}',
                  flush=True)
            continue

        t_ref = bench(base_fn, (view_params, n_pc), 'xla_sorted_reduce')
        for nw in NUM_WARPS_SWEEP:
            try:
                f = jax.jit(make_hfan_call(V, C, TP, P, b_pad, nw))
                out = jax.block_until_ready(
                    f(seg_starts, seg_ends, wts, pix, vals_pad))[..., :band]
                rel = float(jnp.max(jnp.abs(out - ref)) / scale)
                t = bench(f, (seg_starts, seg_ends, wts, pix, vals_pad),
                          f'pallas_hfan_w{nw}')
                print(f'[{kind} nw={nw}] rel err {rel:.3g} '
                      f'{"PASS" if rel < 1e-5 else "FAIL"}; '
                      f'speedup vs XLA: {t_ref / t:.2f}x', flush=True)
            except Exception as e:
                print(f'[{kind} nw={nw}] FAILED: {type(e).__name__}: {str(e)[:300]}',
                      flush=True)


if __name__ == '__main__':
    main()
