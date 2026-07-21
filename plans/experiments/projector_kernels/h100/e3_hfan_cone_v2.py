"""E3 cone hfan v2: row-chunked grid (cone v1 verdict: values pass, speed 1.14x/0.84x).

Cone v1's three deficits and the v2 responses:
  * 4 KB row vectors moved by one warp  -> chunk the ROW axis across the grid: each
    program handles (channel segment) x (row chunk of RC), restoring ~1 KB-class vector
    work and multiplying program parallelism by rows/RC.  Stream reads repeat per chunk
    (scalar loads — cheap).
  * parallel-tuned num_warps=1          -> warps swept {1, 2, 4}.
  * per-view values (no shared L2 tile) -> row chunking also shrinks each program's
    working slice; grid stays view-major (row-chunk dim fastest).
Two-phase variant only: the hybrid's zeros-init tax scales with the output (520 MB for
cone vs 130 MB for parallel) — strictly worse here.

Run:  python plans/experiments/projector_kernels/e3_hfan_cone_v2.py  (constants below)
"""
import importlib.util
import os

# ── Config ────────────────────────────────────────────────────────────────────
CAP = 64
ROW_CHUNKS = [256, 512]
NUM_WARPS_SWEEP = [1, 2, 4]
WARMUP, TRIALS = 2, 10

os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.85')
import time                        # noqa: E402
from functools import partial      # noqa: E402
import numpy as np                 # noqa: E402
import jax                         # noqa: E402
import jax.numpy as jnp            # noqa: E402
from jax.experimental import pallas as pl          # noqa: E402
from jax.experimental.pallas import triton as pltriton   # noqa: E402

_here = os.path.dirname(os.path.abspath(__file__))


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_here, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cone = _load('hfan_cone', 'e3_hfan_cone.py')       # build_case, precompute, baseline
v3 = _load('hfan_v3', 'e3_hfan_pallas_v3.py')      # split_segments_two_phase


def kernel_rc(seg_ref, wt_ref, pix_ref, vals_ref, *rest, atomic, rc):
    out_ref = rest[-1]
    v = pl.program_id(0)
    r = pl.program_id(2)
    start = seg_ref[0, 0, 0]
    end = seg_ref[0, 0, 1]
    c = seg_ref[0, 0, 2]

    def body(i, acc):
        p = pix_ref[0, i]
        wgt = wt_ref[0, i]
        return acc + wgt * vals_ref[0, p, :]        # (rc,) slice of this view's rows
    acc = jax.lax.fori_loop(start, end, body, jnp.zeros((rc,), jnp.float32))
    if atomic:
        pltriton.atomic_add(out_ref, (v, c, pl.dslice(r * rc, rc)), acc)
    else:
        out_ref[v, c, pl.dslice(r * rc, rc)] = acc


def make_two_phase_rc(V, C, n1, n2, TP, P, r_pad, rc, num_warps, interpret=False):
    kw = ({} if interpret else
          {'compiler_params': pltriton.CompilerParams(num_warps=num_warps)})
    n_rc = r_pad // rc

    def specs(n_virt):
        return [pl.BlockSpec((1, 1, 4), lambda v, s, r: (v, s, 0)),
                pl.BlockSpec((1, TP), lambda v, s, r: (v, 0)),
                pl.BlockSpec((1, TP), lambda v, s, r: (v, 0)),
                pl.BlockSpec((1, P, rc), lambda v, s, r: (v, 0, r))]
    p1 = pl.pallas_call(
        partial(kernel_rc, atomic=False, rc=rc),
        out_shape=jax.ShapeDtypeStruct((V, C + 1, r_pad), jnp.float32),
        grid=(V, n1, n_rc),
        in_specs=specs(n1),
        out_specs=pl.BlockSpec((V, C + 1, r_pad), lambda v, s, r: (0, 0, 0)),
        interpret=interpret, **kw)
    p2 = pl.pallas_call(
        partial(kernel_rc, atomic=True, rc=rc),
        out_shape=jax.ShapeDtypeStruct((V, C + 1, r_pad), jnp.float32),
        grid=(V, n2, n_rc),
        in_specs=specs(n2) + [pl.BlockSpec((V, C + 1, r_pad), lambda v, s, r: (0, 0, 0))],
        out_specs=pl.BlockSpec((V, C + 1, r_pad), lambda v, s, r: (0, 0, 0)),
        input_output_aliases={4: 0},
        interpret=interpret, **kw)

    def call(seg1, seg2, wts, pix, vals):
        return p2(seg2, wts, pix, vals, p1(seg1, wts, pix, vals))
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
    sino_shape, p_batch, view_batch = ((cone.SINO_SHAPE, cone.P_BATCH, cone.VIEW_BATCH)
                                       if on_gpu else ((64, 24, 32), 256, 2))
    for kind in ('raster', 'subset'):
        print(f'\n===== case: {kind} =====', flush=True)
        model, pp, idx, values, view_params, n_pc = cone.build_case(
            kind, sino_shape, p_batch, view_batch)
        V, C = view_batch, pp.sinogram_shape[2]
        rows = pp.sinogram_shape[1]
        r_pad = cone.next_pow2(rows)
        P = len(idx)
        TP = cone.T * P
        wts, pix, starts = cone.precompute_streams(pp, idx, view_params, n_pc)
        starts_np = np.asarray(starts)
        vals_pad = jnp.pad(values, ((0, 0), (0, 0), (0, r_pad - rows)))
        base_fn = jax.jit(lambda vals, vp, npc: cone.xla_baseline(pp, idx, values, vp, npc))
        ref = jax.block_until_ready(base_fn(values, view_params, n_pc))
        scale = float(jnp.max(jnp.abs(ref)))

        if not on_gpu:
            seg1, n1, seg2, n2 = v3.split_segments_two_phase(starts_np, 8, C)
            rc = r_pad // 2
            out = make_two_phase_rc(V, C, n1, n2, TP, P, r_pad, rc, 1, interpret=True)(
                seg1, seg2, wts, pix, vals_pad)[:, :C, :rows]
            rel = float(jnp.max(jnp.abs(out - ref)) / scale)
            print(f'[interpret {kind} rc={rc}] rel {rel:.3g} '
                  f'{"PASS" if rel < 1e-5 else "FAIL"}', flush=True)
            continue

        t_ref = bench(base_fn, (values, view_params, n_pc), 'xla_sorted_reduce')
        seg1, n1, seg2, n2 = v3.split_segments_two_phase(starts_np, CAP, C)
        for rc in ROW_CHUNKS:
            for nw in NUM_WARPS_SWEEP:
                tag = f'rc{rc}_w{nw}'
                try:
                    call = make_two_phase_rc(V, C, n1, n2, TP, P, r_pad, rc, nw)
                    f = jax.jit(lambda a, b, w, px, vl: call(a, b, w, px, vl))
                    out = jax.block_until_ready(
                        f(seg1, seg2, wts, pix, vals_pad))[:, :C, :rows]
                    rel = float(jnp.max(jnp.abs(out - ref)) / scale)
                    t = bench(f, (seg1, seg2, wts, pix, vals_pad), tag)
                    print(f'[{kind} {tag}] rel {rel:.3g} '
                          f'{"PASS" if rel < 1e-5 else "FAIL"}; '
                          f'speedup vs XLA: {t_ref / t:.2f}x', flush=True)
                except Exception as e:
                    print(f'[{kind} {tag}] FAILED: {type(e).__name__}: {str(e)[:250]}',
                          flush=True)


if __name__ == '__main__':
    main()
