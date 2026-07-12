"""E3 hfan-forward kernel v2: bound the channel-skew stragglers (v1 raster = 1.42x,
subset = 1.96x; the miss is load imbalance in the one-program-per-channel grid).

v2 = CAP-AND-SPLIT in the precompute: any channel segment longer than CAP taps is split
into VIRTUAL segments of <= CAP taps mapping to the same output channel — every program
walks a bounded segment (balanced grid), at the cost of multiple programs adding into
split channels.  Two store strategies, raced:

  v2a all-atomic : every program pl.atomic_add's its partial row into a zero-initialized
                   output (input_output_aliases zeros); no flags, simplest.
  v2b hybrid     : unsplit channels (the vast majority) DIRECT-store as in v1; only
                   split channels' programs use atomics (a precomputed per-segment flag,
                   pl.when).

Both keep v1's structure otherwise: dynamic-trip fori_loop, ref-level row gathers,
register accumulation.  Empty pad segments target a scratch channel row C (output is
(V, C+1, B), sliced to C).  CAP swept.  The stream PRECOMPUTE is timed this round (the
open accounting item: amortized across VCD iterations, charged for one-shot calls).

Value gates: rel-max <= 1e-5 vs the XLA baseline (atomic order reassociates sums — the
float-gate rules apply, not bitwise).  Success bar: raster >= 1.5x (subset should hold
>= ~1.9x).

Run:  python plans/experiments/projector_kernels/e3_hfan_pallas_v2.py  (constants below)
"""
import importlib.util
import os

# ── Config ────────────────────────────────────────────────────────────────────
CAP_SWEEP = [16, 32, 64]
NUM_WARPS = 1                       # flat in v1; fixed here
WARMUP, TRIALS = 2, 10

os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.85')
import time                        # noqa: E402
import numpy as np                 # noqa: E402
import jax                         # noqa: E402
import jax.numpy as jnp            # noqa: E402
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as pltriton          # noqa: E402

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('v1', os.path.join(_here, 'e3_hfan_pallas_v1.py'))
v1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v1)


def split_segments(starts_np, cap, C):
    """Host-side cap-and-split: per view, (seg_start, seg_end, seg_channel, seg_split)
    arrays padded to a common length; split=1 marks segments of channels that were
    split (their stores must be atomic).  Pad segments target channel C (scratch row)."""
    V = starts_np.shape[0]
    per_view = []
    for v in range(V):
        s = starts_np[v]
        st, en, ch, sp = [], [], [], []
        for c in range(C):
            a, b = int(s[c]), int(s[c + 1])
            n = b - a
            if n == 0:
                continue
            nseg = -(-n // cap)
            for k in range(nseg):
                st.append(a + k * cap)
                en.append(min(a + (k + 1) * cap, b))
                ch.append(c)
                sp.append(1 if nseg > 1 else 0)
        per_view.append((st, en, ch, sp))
    n_virt = max(len(p[0]) for p in per_view)
    seg = np.zeros((V, n_virt, 4), dtype=np.int32)
    seg[:, :, 2] = C                                  # pad segments -> scratch channel
    for v, (st, en, ch, sp) in enumerate(per_view):
        k = len(st)
        seg[v, :k, 0], seg[v, :k, 1] = st, en
        seg[v, :k, 2], seg[v, :k, 3] = ch, sp
        seg[v, k:, 3] = 1                             # pads store via atomic (add zero)
    return jnp.asarray(seg), n_virt


def hfan_kernel_v2(seg_ref, wt_ref, pix_ref, vals_ref, zeros_ref, out_ref, *, hybrid):
    # The out block is the WHOLE (V, C+1, B) array (atomic targets are data-dependent),
    # so the view index comes from the grid — NOT a literal (the interpret gate caught a
    # v=0 collision bug here).
    v = pl.program_id(0)
    start = seg_ref[0, 0, 0]
    end = seg_ref[0, 0, 1]
    c = seg_ref[0, 0, 2]
    split = seg_ref[0, 0, 3]

    def body(i, acc):
        p = pix_ref[0, i]
        wgt = wt_ref[0, i]
        return acc + wgt * vals_ref[p, :]
    acc = jax.lax.fori_loop(start, end, body,
                            jnp.zeros((out_ref.shape[-1],), jnp.float32))
    if hybrid:
        @pl.when(split == 0)
        def _store():
            out_ref[v, c, :] = acc

        @pl.when(split == 1)
        def _atomic():
            pltriton.atomic_add(out_ref, (v, c, slice(None)), acc)
    else:
        pltriton.atomic_add(out_ref, (v, c, slice(None)), acc)


def make_v2_call(V, C, n_virt, TP, P, b_pad, hybrid, interpret=False):
    from functools import partial
    from jax.experimental.pallas import triton as pltriton
    kw = ({} if interpret else
          {'compiler_params': pltriton.CompilerParams(num_warps=NUM_WARPS)})
    return pl.pallas_call(
        partial(hfan_kernel_v2, hybrid=hybrid),
        out_shape=jax.ShapeDtypeStruct((V, C + 1, b_pad), jnp.float32),
        grid=(V, n_virt),
        in_specs=[pl.BlockSpec((1, 1, 4), lambda v, s: (v, s, 0)),   # seg table row
                  pl.BlockSpec((1, TP), lambda v, s: (v, 0)),
                  pl.BlockSpec((1, TP), lambda v, s: (v, 0)),
                  pl.BlockSpec((P, b_pad), lambda v, s: (0, 0)),
                  pl.BlockSpec((V, C + 1, b_pad), lambda v, s: (0, 0, 0))],
        out_specs=pl.BlockSpec((V, C + 1, b_pad), lambda v, s: (0, 0, 0)),
        input_output_aliases={4: 0},                  # zeros -> output (donated init)
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
    sino_shape, p_batch, band, b_pad, view_batch = (
        (v1.SINO_SHAPE, v1.P_BATCH, v1.BAND, v1.B_PAD, v1.VIEW_BATCH) if on_gpu
        else ((64, 24, 32), 256, 16, 16, 2))
    for kind in ('raster', 'subset'):
        print(f'\n===== case: {kind} =====', flush=True)
        model, pp, idx, values, view_params, n_pc = v1.build_case(
            kind, sino_shape, p_batch, band, view_batch)
        V, C = view_batch, pp.sinogram_shape[2]
        P = len(idx)
        TP = v1.T * P

        t0 = time.perf_counter()
        wts, pix, starts = v1.precompute_streams(pp, idx, view_params, n_pc)
        t_pre_cold = time.perf_counter() - t0
        t0 = time.perf_counter()
        wts, pix, starts = v1.precompute_streams(pp, idx, view_params, n_pc)
        t_pre_warm = time.perf_counter() - t0
        print(f'stream precompute: warm {t_pre_warm * 1e3:.2f} ms '
              f'(cold {t_pre_cold * 1e3:.0f} ms)', flush=True)

        vals_pad = jnp.pad(values, ((0, 0), (0, b_pad - band)))
        starts_np = np.asarray(starts)
        base_fn = jax.jit(lambda vp, npc: v1.xla_baseline(pp, idx, values, vp, npc))
        ref = jax.block_until_ready(base_fn(view_params, n_pc))
        scale = float(jnp.max(jnp.abs(ref)))

        if not on_gpu:
            seg, n_virt = split_segments(starts_np, 8, C)
            for hybrid in (False, True):
                call = make_v2_call(V, C, n_virt, TP, P, b_pad, hybrid, interpret=True)
                out = call(seg, wts, pix, vals_pad,
                           jnp.zeros((V, C + 1, b_pad), jnp.float32))[:, :C, :band]
                rel = float(jnp.max(jnp.abs(out - ref)) / scale)
                print(f'[interpret {kind} hybrid={hybrid}] rel {rel:.3g} '
                      f'{"PASS" if rel < 1e-5 else "FAIL"}', flush=True)
            continue

        t_ref = bench(base_fn, (view_params, n_pc), 'xla_sorted_reduce')
        for cap in CAP_SWEEP:
            seg, n_virt = split_segments(starts_np, cap, C)
            n_split = int((np.asarray(seg[:, :, 3]) == 1).sum() / V)
            for hybrid in (False, True):
                tag = f'v2{"b_hyb" if hybrid else "a_atom"}_cap{cap}'
                try:
                    # Zeros are allocated INSIDE the compiled call: the pallas output
                    # aliases (donates) them, so a caller-held buffer cannot be reused
                    # across bench iterations.
                    f = jax.jit(lambda s, w, px, vl, _h=hybrid, _n=n_virt:
                                make_v2_call(V, C, _n, TP, P, b_pad, _h)(
                                    s, w, px, vl,
                                    jnp.zeros((V, C + 1, b_pad), jnp.float32)))
                    out = jax.block_until_ready(
                        f(seg, wts, pix, vals_pad))[:, :C, :band]
                    rel = float(jnp.max(jnp.abs(out - ref)) / scale)
                    t = bench(f, (seg, wts, pix, vals_pad), tag)
                    print(f'[{kind} {tag}] n_virt={n_virt} (~{n_split}/view '
                          f'atomic) rel {rel:.3g} {"PASS" if rel < 1e-5 else "FAIL"}; '
                          f'speedup vs XLA: {t_ref / t:.2f}x', flush=True)
                except Exception as e:
                    print(f'[{kind} {tag}] FAILED: {type(e).__name__}: {str(e)[:250]}',
                          flush=True)


if __name__ == '__main__':
    main()
