"""E3b back-projection kernel v1: register-tile across views + row-chunk L2 residency
(design: gpu_headroom_plan.md section E3b, Greg-approved 2026-07-12).

The kernel: grid = (row_chunk, pixel), row-chunk SLOWEST so all concurrent programs
gather from the same (V, C, RC) sinogram slice (RC=128 -> ~65 MB ~ L2 — the direct
attack on the E2a transaction-bound gathers).  Each program holds out[p, r*RC:(r+1)*RC]
in registers, loops all V views x T taps (T python-unrolled), ref-gathers RC-float row
chunks from the channel-major sinogram, one store.  No sort, no segments, no atomics —
back is uniform work by construction.  coeff_power in {1, 2} via the weight precompute
(the Hessian path).

Value gates: rel <= 1e-5 vs the library stacked-gather baseline (both coeff_powers),
PLUS the adjoint check <A x, y> = <x, B y> against the XLA forward at matched shapes
(the pallas pair ships together; the pair-vs-pair adjoint test lands in E4's suite).

Sweeps: RC in {64, 128, 256} x num_warps in {1, 2, 4}; raster + subset batches at the
1024^3-class parallel cell.  Success bar: >=1.5-2x at both.

Run:  python plans/experiments/projector_kernels/e3_back_pallas_v1.py  (constants below)
"""
import importlib.util
import os

# ── Config ────────────────────────────────────────────────────────────────────
RC_SWEEP = [64, 128, 256]
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
from jax.experimental.pallas import triton as pltriton   # noqa: E402
from mbirjax.parallel_beam import ParallelBeamModel      # noqa: E402

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('v1f', os.path.join(_here, 'e3_hfan_pallas_v1.py'))
v1f = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v1f)      # build_case, T, shapes, xla fwd baseline (adjoint cell)

T = v1f.T


def next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


def xla_back_baseline(pp, idx, sino_vrc, view_params, n_pc, coeff_power):
    """The library back kernel (stacked gather on GPU policy) vmapped + view-summed."""
    def one_view(view, single_view_params, centers):
        return ParallelBeamModel.back_project_one_view_to_pixel_batch(
            view, idx, single_view_params, centers, pp, coeff_power)
    per_view = jax.vmap(one_view)(sino_vrc, view_params, n_pc)
    return jnp.sum(per_view, axis=0)                     # (P, R)


def precompute_weights(pp, idx, view_params, n_pc, coeff_power):
    """A (V, T, P) f32 (zeroed out of range, squared for the Hessian path) — no sort."""
    gp = pp.geometry_params
    C = pp.sinogram_shape[2]

    def one_view(single_view_params, centers):
        n_p, W_p_c, footprint = ParallelBeamModel.compute_proj_data(
            idx, single_view_params, pp)
        delta_voxel_row = gp.voxel_row_aspect * gp.delta_voxel
        scale = (delta_voxel_row * gp.delta_voxel) / footprint
        offs = jnp.arange(-gp.psf_radius, gp.psf_radius + 1)
        n = centers[None, :] + offs[:, None]
        L_max = jnp.minimum(1.0, W_p_c)
        A = scale * jnp.clip((W_p_c + 1.0) / 2.0 - jnp.abs(n_p - n), 0.0, L_max)
        A = A * ((n >= 0) & (n < C))
        return A ** coeff_power                          # (T, P)

    return jax.block_until_ready(jax.jit(jax.vmap(one_view))(view_params, n_pc))


def back_kernel(centers_ref, w_ref, sino_ref, out_ref, *, rc, V, C, psf_radius):
    """One program per (row-chunk, pixel): registers accumulate across ALL views."""
    def vbody(v, acc):
        c0 = centers_ref[v, 0]
        for t in range(T):                               # static unroll
            cc = jnp.clip(c0 + (t - psf_radius), 0, C - 1)   # weights already 0 OOR
            acc = acc + w_ref[v, t, 0] * sino_ref[v, cc, :]  # (rc,) row-chunk gather
        return acc
    acc = jax.lax.fori_loop(0, V, vbody, jnp.zeros((rc,), jnp.float32))
    out_ref[0, :] = acc


def make_back_call(V, C, P, r_pad, rc, num_warps, psf_radius, interpret=False):
    kw = ({} if interpret else
          {'compiler_params': pltriton.CompilerParams(num_warps=num_warps)})
    return pl.pallas_call(
        partial(back_kernel, rc=rc, V=V, C=C, psf_radius=psf_radius),
        out_shape=jax.ShapeDtypeStruct((P, r_pad), jnp.float32),
        grid=(r_pad // rc, P),                 # row-chunk SLOWEST: the L2-phase design
        in_specs=[pl.BlockSpec((V, 1), lambda r, p: (0, p)),        # centers column
                  pl.BlockSpec((V, T, 1), lambda r, p: (0, 0, p)),  # weights column
                  pl.BlockSpec((V, C, rc), lambda r, p: (0, 0, r))],  # sino row-chunk
        out_specs=pl.BlockSpec((1, rc), lambda r, p: (p, r)),
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
    sino_shape, p_batch, view_batch = ((v1f.SINO_SHAPE, v1f.P_BATCH, v1f.VIEW_BATCH)
                                       if on_gpu else ((64, 24, 32), 256, 2))
    rows = sino_shape[1]
    for kind in ('raster', 'subset'):
        print(f'\n===== case: {kind} =====', flush=True)
        # band arg irrelevant for back; reuse the fwd build for idx/views/centers.
        model, pp, idx, _values, view_params, n_pc = v1f.build_case(
            kind, sino_shape, p_batch, min(rows, 16), view_batch)
        V, C = view_batch, pp.sinogram_shape[2]
        P = len(idx)
        r_pad = next_pow2(rows)
        psf_radius = pp.geometry_params.psf_radius
        rng = np.random.default_rng(1)
        sino_vrc = jnp.asarray(rng.random((V, rows, C), dtype=np.float32))
        sino_cm = jnp.pad(jnp.swapaxes(sino_vrc, 1, 2), ((0, 0), (0, 0), (0, r_pad - rows)))
        jax.block_until_ready((sino_vrc, sino_cm))

        for coeff_power in (1, 2):
            wts = precompute_weights(pp, idx, view_params, n_pc, coeff_power)
            base_fn = jax.jit(lambda s, vp, npc, _cp=coeff_power: xla_back_baseline(
                pp, idx, s, vp, npc, _cp))
            ref = jax.block_until_ready(base_fn(sino_vrc, view_params, n_pc))   # (P, R)
            scale = float(jnp.max(jnp.abs(ref)))

            if not on_gpu:
                rc = r_pad // 2
                out = make_back_call(V, C, P, r_pad, rc, 1, psf_radius, interpret=True)(
                    jnp.asarray(n_pc), wts, sino_cm)[:, :rows]
                rel = float(jnp.max(jnp.abs(out - ref)) / scale)
                print(f'[interpret {kind} cp{coeff_power}] rel {rel:.3g} '
                      f'{"PASS" if rel < 1e-5 else "FAIL"}', flush=True)
                continue

            t_ref = bench(base_fn, (sino_vrc, view_params, n_pc),
                          f'xla_back_cp{coeff_power}')
            rc_list = RC_SWEEP if coeff_power == 1 else RC_SWEEP[1:2]
            nw_list = NUM_WARPS_SWEEP if coeff_power == 1 else NUM_WARPS_SWEEP[:1]
            for rc in rc_list:
                for nw in nw_list:
                    tag = f'cp{coeff_power}_rc{rc}_w{nw}'
                    try:
                        f = jax.jit(make_back_call(V, C, P, r_pad, rc, nw, psf_radius))
                        out = jax.block_until_ready(
                            f(jnp.asarray(n_pc), wts, sino_cm))[:, :rows]
                        rel = float(jnp.max(jnp.abs(out - ref)) / scale)
                        t = bench(f, (jnp.asarray(n_pc), wts, sino_cm), tag)
                        print(f'[{kind} {tag}] rel {rel:.3g} '
                              f'{"PASS" if rel < 1e-5 else "FAIL"}; '
                              f'speedup vs XLA: {t_ref / t:.2f}x', flush=True)
                    except Exception as e:
                        print(f'[{kind} {tag}] FAILED: {type(e).__name__}: '
                              f'{str(e)[:250]}', flush=True)

    # Adjoint cell (parallel, unbanded: slices == det rows): <fwd(x), y> == <x, back(y)>.
    print('\n===== adjoint check =====', flush=True)
    model, pp, idx, _v, view_params, n_pc = v1f.build_case(
        'subset', sino_shape, min(p_batch, 2048), rows, view_batch)
    V, C, P = view_batch, pp.sinogram_shape[2], len(idx)
    r_pad = next_pow2(rows)
    psf_radius = pp.geometry_params.psf_radius
    rng = np.random.default_rng(2)
    x = jnp.asarray(rng.random((P, rows), dtype=np.float32))
    y = jnp.asarray(rng.random((V, C, rows), dtype=np.float32))     # channel-major
    fwd = jax.jit(lambda vp, npc: v1f.xla_baseline(pp, idx, x, vp, npc))  # (V, C, rows)
    ax_dot_y = float(jnp.vdot(fwd(view_params, n_pc), y))
    wts = precompute_weights(pp, idx, view_params, n_pc, 1)
    y_pad = jnp.pad(y, ((0, 0), (0, 0), (0, r_pad - rows)))
    rc = 128 if r_pad % 128 == 0 else r_pad
    back = (make_back_call(V, C, P, r_pad, rc, 1, psf_radius,
                           interpret=not on_gpu))
    by = back(jnp.asarray(n_pc), wts, y_pad)[:, :rows]
    x_dot_by = float(jnp.vdot(x, by))
    rel = abs(ax_dot_y - x_dot_by) / max(abs(ax_dot_y), 1e-30)
    print(f'<Ax,y> = {ax_dot_y:.6e}  <x,By> = {x_dot_by:.6e}  rel diff {rel:.3g} '
          f'{"PASS" if rel < 1e-5 else "FAIL"}', flush=True)


if __name__ == '__main__':
    main()
