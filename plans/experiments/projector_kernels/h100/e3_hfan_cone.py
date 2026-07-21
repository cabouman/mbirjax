"""E3 cone hfan kernel: the CSR segment-walk applied to CONE's horizontal fan (the
72%-share piece of cone forward, `gpu_headroom_findings.md` cone-split section).

Structural deltas from the parallel bench (e3_hfan_pallas_v{1,2,3}.py):
  * weight scale is PER-PIXEL (per-pixel magnification) — enters the precomputed
    weights; the kernel is unchanged.
  * the hfan input is the vertical fan's PER-VIEW output: values (V, P, num_det_rows)
    rather than one shared (P, band) tile — the gather source is the view's own slice
    (~16 MB at this cell, still L2-resident per view).
  * accumulation width = full detector rows, padded to the next power of 2 (Triton
    shape rule); the row count is arbitrary per dataset — nothing special about this
    cell's 1008.

Variants carried from the parallel verdict: v3 two-phase (uniform-batch winner) and
v2b hybrid single-launch (skew winner), caps {64, 128}; cases raster + subset.
Baseline: the library cone hfan (sorted reduce) vmapped over views, fed the same
per-view values (the cone_fwd_split_ab hfan_only isolation).

Run:  python plans/experiments/projector_kernels/e3_hfan_cone.py   (constants below)
"""
import importlib.util
import os

# ── Config ────────────────────────────────────────────────────────────────────
SINO_SHAPE = (1024, 1008, 992)      # the 1024^3-class cone cell (rows arbitrary)
P_BATCH = 4096                      # cone >=768-slice GPU policy
VIEW_BATCH = 128
CAP_SWEEP = [64, 128]
NUM_WARPS = 1
WARMUP, TRIALS = 2, 10

os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.85')
import time                        # noqa: E402
from functools import partial      # noqa: E402
import numpy as np                 # noqa: E402
import mbirjax                     # noqa: E402
import jax                         # noqa: E402
import jax.numpy as jnp            # noqa: E402
from jax.experimental import pallas as pl          # noqa: E402
from jax.experimental.pallas import triton as pltriton   # noqa: E402
from mbirjax.cone_beam import ConeBeamModel        # noqa: E402
from mbirjax.projectors import (ProjectorParams,   # noqa: E402
                                _jit_compute_scatter_centers)

_here = os.path.dirname(os.path.abspath(__file__))


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_here, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


v2 = _load('hfan_v2', 'e3_hfan_pallas_v2.py')      # split_segments (single-launch)
v3 = _load('hfan_v3', 'e3_hfan_pallas_v3.py')      # split_segments_two_phase

T = 3


def next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def build_case(kind, sino_shape, p_batch, view_batch):
    num_views, num_det_rows, num_det_channels = sino_shape
    angles = np.linspace(0, np.pi, num_views, endpoint=False)
    model = mbirjax.ConeBeamModel(sino_shape, angles,
                                  source_detector_dist=4.0 * num_det_channels,
                                  source_iso_dist=2.0 * num_det_channels)
    model.configure_devices(1)
    args = (tuple(model.get_params('sinogram_shape')),
            tuple(model.get_params('recon_shape')), model.get_geometry_parameters())
    args += tuple(int(bool(getattr(model.tiles, f, 0))) for f in ProjectorParams._fields[3:])
    pp = ProjectorParams(*args)
    assert pp.geometry_params.psf_radius * 2 + 1 == T
    recon_shape = model.get_params('recon_shape')
    rng = np.random.default_rng(0)
    full_idx = np.asarray(mbirjax.gen_full_indices(
        recon_shape, use_ror_mask=model.get_params('use_ror_mask')))
    if kind == 'raster':
        idx = jnp.asarray(full_idx[:p_batch].astype(np.int32))
    else:
        idx = jnp.asarray(np.sort(rng.choice(full_idx, size=min(p_batch, len(full_idx)),
                                             replace=False)).astype(np.int32))
    # Per-VIEW hfan inputs (the vertical fan's output shape); random values are fine —
    # the kernels are agnostic to provenance, and baseline/kernel consume the same array.
    values = jnp.asarray(rng.random((view_batch, len(idx), num_det_rows),
                                    dtype=np.float32))
    view_params = jnp.asarray(model.projector_functions.view_params_array)[:view_batch]
    n_pc = _jit_compute_scatter_centers(
        jnp.asarray(model.projector_functions.view_params_array), idx,
        ConeBeamModel.compute_channel_coordinate, pp, pixels_major=False)[:view_batch]
    return model, pp, idx, values, view_params, n_pc


def xla_baseline(pp, idx, values, view_params, n_pc):
    """The library cone hfan (sorted reduce) vmapped over views -> (V, rows, C),
    transposed to (V, C, rows) to match the kernels' channel-major output."""
    def one_view(vals_v, single_view_params, centers):
        return ConeBeamModel.forward_horizontal_fan_pixel_batch_to_one_view(
            vals_v, idx, single_view_params, centers, pp)
    out = jax.vmap(one_view)(values, view_params, n_pc)     # (V, rows, C)
    return jnp.swapaxes(out, 1, 2)                          # (V, C, rows)


def precompute_streams(pp, idx, view_params, n_pc):
    """Sorted (weight, pixel) streams + segment starts, cone weights (per-pixel scale)."""
    gp = pp.geometry_params
    C = pp.sinogram_shape[2]
    num_pixels = len(idx)

    def one_view(single_view_params, centers):
        n_p, W_p_c, footprint_xy = ConeBeamModel.compute_horizontal_data(
            idx, single_view_params, pp)
        delta_voxel_row = gp.voxel_row_aspect * gp.delta_voxel
        scale = (delta_voxel_row * gp.delta_voxel) / footprint_xy   # PER-PIXEL (cone)
        offs = jnp.arange(-gp.psf_radius, gp.psf_radius + 1)
        n = centers[None, :] + offs[:, None]
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

    wts, pix, starts = jax.jit(jax.vmap(one_view))(view_params, n_pc)
    return (jax.block_until_ready(wts), jax.block_until_ready(pix),
            jax.block_until_ready(starts))


# ── Kernels: identical walks, per-VIEW values block ───────────────────────────
def cone_kernel(seg_ref, wt_ref, pix_ref, vals_ref, *rest, atomic):
    out_ref = rest[-1]
    v = pl.program_id(0)
    start = seg_ref[0, 0, 0]
    end = seg_ref[0, 0, 1]
    c = seg_ref[0, 0, 2]

    def body(i, acc):
        p = pix_ref[0, i]
        wgt = wt_ref[0, i]
        return acc + wgt * vals_ref[0, p, :]        # this view's (P, rows) slice
    acc = jax.lax.fori_loop(start, end, body,
                            jnp.zeros((out_ref.shape[-1],), jnp.float32))
    if atomic:
        pltriton.atomic_add(out_ref, (v, c, slice(None)), acc)
    else:
        out_ref[v, c, :] = acc


def cone_kernel_hybrid(seg_ref, wt_ref, pix_ref, vals_ref, zeros_ref, out_ref):
    v = pl.program_id(0)
    start = seg_ref[0, 0, 0]
    end = seg_ref[0, 0, 1]
    c = seg_ref[0, 0, 2]
    split = seg_ref[0, 0, 3]

    def body(i, acc):
        p = pix_ref[0, i]
        wgt = wt_ref[0, i]
        return acc + wgt * vals_ref[0, p, :]
    acc = jax.lax.fori_loop(start, end, body,
                            jnp.zeros((out_ref.shape[-1],), jnp.float32))

    @pl.when(split == 0)
    def _store():
        out_ref[v, c, :] = acc

    @pl.when(split == 1)
    def _atomic():
        pltriton.atomic_add(out_ref, (v, c, slice(None)), acc)


def _common_specs(V, n_virt, TP, P, r_pad, seg_width):
    return [pl.BlockSpec((1, 1, seg_width), lambda v, s: (v, s, 0)),
            pl.BlockSpec((1, TP), lambda v, s: (v, 0)),
            pl.BlockSpec((1, TP), lambda v, s: (v, 0)),
            pl.BlockSpec((1, P, r_pad), lambda v, s: (v, 0, 0))]   # per-view values


def make_two_phase(V, C, n1, n2, TP, P, r_pad, interpret=False):
    kw = ({} if interpret else
          {'compiler_params': pltriton.CompilerParams(num_warps=NUM_WARPS)})
    p1 = pl.pallas_call(
        partial(cone_kernel, atomic=False),
        out_shape=jax.ShapeDtypeStruct((V, C + 1, r_pad), jnp.float32),
        grid=(V, n1),
        in_specs=_common_specs(V, n1, TP, P, r_pad, 4),
        out_specs=pl.BlockSpec((V, C + 1, r_pad), lambda v, s: (0, 0, 0)),
        interpret=interpret, **kw)
    p2 = pl.pallas_call(
        partial(cone_kernel, atomic=True),
        out_shape=jax.ShapeDtypeStruct((V, C + 1, r_pad), jnp.float32),
        grid=(V, n2),
        in_specs=_common_specs(V, n2, TP, P, r_pad, 4)
                 + [pl.BlockSpec((V, C + 1, r_pad), lambda v, s: (0, 0, 0))],
        out_specs=pl.BlockSpec((V, C + 1, r_pad), lambda v, s: (0, 0, 0)),
        input_output_aliases={4: 0},
        interpret=interpret, **kw)

    def call(seg1, seg2, wts, pix, vals):
        return p2(seg2, wts, pix, vals, p1(seg1, wts, pix, vals))
    return call


def make_hybrid(V, C, n_virt, TP, P, r_pad, interpret=False):
    kw = ({} if interpret else
          {'compiler_params': pltriton.CompilerParams(num_warps=NUM_WARPS)})
    inner = pl.pallas_call(
        cone_kernel_hybrid,
        out_shape=jax.ShapeDtypeStruct((V, C + 1, r_pad), jnp.float32),
        grid=(V, n_virt),
        in_specs=_common_specs(V, n_virt, TP, P, r_pad, 4)
                 + [pl.BlockSpec((V, C + 1, r_pad), lambda v, s: (0, 0, 0))],
        out_specs=pl.BlockSpec((V, C + 1, r_pad), lambda v, s: (0, 0, 0)),
        input_output_aliases={4: 0},
        interpret=interpret, **kw)

    def call(seg, wts, pix, vals):
        return inner(seg, wts, pix, vals,
                     jnp.zeros((V, C + 1, r_pad), jnp.float32))
    return call


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
    sino_shape, p_batch, view_batch = ((SINO_SHAPE, P_BATCH, VIEW_BATCH) if on_gpu
                                       else ((64, 24, 32), 256, 2))
    for kind in ('raster', 'subset'):
        print(f'\n===== case: {kind} =====', flush=True)
        model, pp, idx, values, view_params, n_pc = build_case(
            kind, sino_shape, p_batch, view_batch)
        V, C = view_batch, pp.sinogram_shape[2]
        rows = pp.sinogram_shape[1]
        r_pad = next_pow2(rows)
        P = len(idx)
        TP = T * P
        wts, pix, starts = precompute_streams(pp, idx, view_params, n_pc)
        starts_np = np.asarray(starts)
        counts = starts_np[:, 1:] - starts_np[:, :-1]
        print(f'P={P} C={C} rows={rows}->{r_pad} taps/channel mean '
              f'{counts.mean():.1f} max {counts.max()}', flush=True)
        vals_pad = jnp.pad(values, ((0, 0), (0, 0), (0, r_pad - rows)))

        base_fn = jax.jit(lambda vals, vp, npc: xla_baseline(pp, idx, values, vp, npc))
        ref = jax.block_until_ready(base_fn(values, view_params, n_pc))   # (V, C, rows)
        scale = float(jnp.max(jnp.abs(ref)))

        if not on_gpu:
            seg1, n1, seg2, n2 = v3.split_segments_two_phase(starts_np, 8, C)
            out = make_two_phase(V, C, n1, n2, TP, P, r_pad, interpret=True)(
                seg1, seg2, wts, pix, vals_pad)[:, :C, :rows]
            rel = float(jnp.max(jnp.abs(out - ref)) / scale)
            print(f'[interpret {kind} two_phase] rel {rel:.3g} '
                  f'{"PASS" if rel < 1e-5 else "FAIL"}', flush=True)
            seg, n_virt = v2.split_segments(starts_np, 8, C)
            out = make_hybrid(V, C, n_virt, TP, P, r_pad, interpret=True)(
                seg, wts, pix, vals_pad)[:, :C, :rows]
            rel = float(jnp.max(jnp.abs(out - ref)) / scale)
            print(f'[interpret {kind} hybrid] rel {rel:.3g} '
                  f'{"PASS" if rel < 1e-5 else "FAIL"}', flush=True)
            continue

        t_ref = bench(base_fn, (values, view_params, n_pc), 'xla_sorted_reduce')
        for cap in CAP_SWEEP:
            seg1, n1, seg2, n2 = v3.split_segments_two_phase(starts_np, cap, C)
            try:
                call = make_two_phase(V, C, n1, n2, TP, P, r_pad)
                f = jax.jit(lambda a, b, w, px, vl: call(a, b, w, px, vl))
                out = jax.block_until_ready(
                    f(seg1, seg2, wts, pix, vals_pad))[:, :C, :rows]
                rel = float(jnp.max(jnp.abs(out - ref)) / scale)
                t = bench(f, (seg1, seg2, wts, pix, vals_pad), f'two_phase_cap{cap}')
                print(f'[{kind} two_phase_cap{cap}] n1={n1} n2={n2} rel {rel:.3g} '
                      f'{"PASS" if rel < 1e-5 else "FAIL"}; '
                      f'speedup vs XLA: {t_ref / t:.2f}x', flush=True)
            except Exception as e:
                print(f'[{kind} two_phase_cap{cap}] FAILED: {type(e).__name__}: '
                      f'{str(e)[:250]}', flush=True)

            seg, n_virt = v2.split_segments(starts_np, cap, C)
            try:
                call = make_hybrid(V, C, n_virt, TP, P, r_pad)
                f = jax.jit(lambda s, w, px, vl: call(s, w, px, vl))
                out = jax.block_until_ready(f(seg, wts, pix, vals_pad))[:, :C, :rows]
                rel = float(jnp.max(jnp.abs(out - ref)) / scale)
                t = bench(f, (seg, wts, pix, vals_pad), f'hybrid_cap{cap}')
                print(f'[{kind} hybrid_cap{cap}] n_virt={n_virt} rel {rel:.3g} '
                      f'{"PASS" if rel < 1e-5 else "FAIL"}; '
                      f'speedup vs XLA: {t_ref / t:.2f}x', flush=True)
            except Exception as e:
                print(f'[{kind} hybrid_cap{cap}] FAILED: {type(e).__name__}: '
                      f'{str(e)[:250]}', flush=True)


if __name__ == '__main__':
    main()
