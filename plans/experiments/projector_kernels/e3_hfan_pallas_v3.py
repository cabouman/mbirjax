"""E3 hfan-forward kernel v3: two-phase store+atomic (kill v2's zeros-init cost).

v2 verdict: raster 1.59x (bar crossed) but subset regressed 1.96->1.78x, mostly the
130 MB zeros-init write that the atomic path requires.  v3 splits the launch:
  phase 1 (store): every channel's FIRST segment (or an empty-channel zero segment)
      DIRECT-stores its partial — every output row written exactly once, no zeros pass,
      no cross-program race.
  phase 2 (atomic): only the REMAINING segments of split channels atomic-add
      (~450/view raster at cap 64, ~1/view subset — nearly empty).
Caps swept wider (interior optimum between balance and split count).

(v2 docstring follows for the record.)
E3 hfan-forward kernel v2: bound the channel-skew stragglers (v1 raster = 1.42x,
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
CAP_SWEEP = [32, 64, 96, 128]
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


def split_segments_two_phase(starts_np, cap, C):
    """Host-side cap-and-split, two-phase: phase-1 = the FIRST segment of every channel
    (including empty channels: start==end -> stores zero), so every output row is
    written exactly once with no init pass; phase-2 = the remaining segments of split
    channels (atomic adds).  Pad rows target the scratch channel C."""
    V = starts_np.shape[0]
    p1_view, p2_view = [], []
    for v in range(V):
        s = starts_np[v]
        p1, p2 = [], []
        for c in range(C):
            a, b = int(s[c]), int(s[c + 1])
            nseg = max(1, -(-(b - a) // cap))
            p1.append((a, min(a + cap, b), c))
            for k in range(1, nseg):
                p2.append((a + k * cap, min(a + (k + 1) * cap, b), c))
        p1_view.append(p1)
        p2_view.append(p2)

    def pack(per_view, pad_channel):
        n = max(1, max(len(p) for p in per_view))
        seg = np.zeros((V, n, 4), dtype=np.int32)
        seg[:, :, 2] = pad_channel
        for v, rows in enumerate(per_view):
            for k, (a, b, c) in enumerate(rows):
                seg[v, k, 0], seg[v, k, 1], seg[v, k, 2] = a, b, c
        return jnp.asarray(seg), n
    seg1, n1 = pack(p1_view, C)      # pads store zero to the scratch row (harmless)
    seg2, n2 = pack(p2_view, C)      # pads atomic-add zero to the scratch row
    return seg1, n1, seg2, n2


def hfan_kernel_v3(seg_ref, wt_ref, pix_ref, vals_ref, *rest, atomic):
    # atomic phase receives (aliased phase-1 ref, out ref) — same buffer, both passed.
    out_ref = rest[-1]
    # The out block is the WHOLE (V, C+1, B) array (dynamic channel targets), so the
    # view index comes from the grid — not a literal (interpret gate caught that in v2).
    v = pl.program_id(0)
    start = seg_ref[0, 0, 0]
    end = seg_ref[0, 0, 1]
    c = seg_ref[0, 0, 2]

    def body(i, acc):
        p = pix_ref[0, i]
        wgt = wt_ref[0, i]
        return acc + wgt * vals_ref[p, :]
    acc = jax.lax.fori_loop(start, end, body,
                            jnp.zeros((out_ref.shape[-1],), jnp.float32))
    if atomic:
        pltriton.atomic_add(out_ref, (v, c, slice(None)), acc)
    else:
        out_ref[v, c, :] = acc


def make_v3_phase(V, C, n_virt, TP, P, b_pad, atomic, interpret=False):
    """One phase: store (init) or atomic (accumulate into the phase-1 result via
    input_output_aliases on the phase-1 output)."""
    from functools import partial
    from jax.experimental.pallas import triton as pltriton
    kw = ({} if interpret else
          {'compiler_params': pltriton.CompilerParams(num_warps=NUM_WARPS)})
    specs = [pl.BlockSpec((1, 1, 4), lambda v, s: (v, s, 0)),   # seg table row
             pl.BlockSpec((1, TP), lambda v, s: (v, 0)),
             pl.BlockSpec((1, TP), lambda v, s: (v, 0)),
             pl.BlockSpec((P, b_pad), lambda v, s: (0, 0))]
    alias = {}
    if atomic:
        specs.append(pl.BlockSpec((V, C + 1, b_pad), lambda v, s: (0, 0, 0)))
        alias = {4: 0}                       # phase-1 result -> in-place accumulation
    return pl.pallas_call(
        partial(hfan_kernel_v3, atomic=atomic),
        out_shape=jax.ShapeDtypeStruct((V, C + 1, b_pad), jnp.float32),
        grid=(V, n_virt),
        in_specs=specs,
        out_specs=pl.BlockSpec((V, C + 1, b_pad), lambda v, s: (0, 0, 0)),
        input_output_aliases=alias,
        interpret=interpret, **kw)


def make_v3_call(V, C, n1, n2, TP, P, b_pad, interpret=False):
    phase1 = make_v3_phase(V, C, n1, TP, P, b_pad, atomic=False, interpret=interpret)
    phase2 = make_v3_phase(V, C, n2, TP, P, b_pad, atomic=True, interpret=interpret)

    def call(seg1, seg2, wts, pix, vals):
        out = phase1(seg1, wts, pix, vals)
        return phase2(seg2, wts, pix, vals, out)
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
            seg1, n1, seg2, n2 = split_segments_two_phase(starts_np, 8, C)
            call = make_v3_call(V, C, n1, n2, TP, P, b_pad, interpret=True)
            out = call(seg1, seg2, wts, pix, vals_pad)[:, :C, :band]
            rel = float(jnp.max(jnp.abs(out - ref)) / scale)
            print(f'[interpret {kind}] rel {rel:.3g} '
                  f'{"PASS" if rel < 1e-5 else "FAIL"}', flush=True)
            continue

        t_ref = bench(base_fn, (view_params, n_pc), 'xla_sorted_reduce')
        for cap in CAP_SWEEP:
            seg1, n1, seg2, n2 = split_segments_two_phase(starts_np, cap, C)
            tag = f'v3_cap{cap}'
            try:
                call = make_v3_call(V, C, n1, n2, TP, P, b_pad)
                f = jax.jit(lambda s1, s2, w, px, vl: call(s1, s2, w, px, vl))
                out = jax.block_until_ready(
                    f(seg1, seg2, wts, pix, vals_pad))[:, :C, :band]
                rel = float(jnp.max(jnp.abs(out - ref)) / scale)
                t = bench(f, (seg1, seg2, wts, pix, vals_pad), tag)
                print(f'[{kind} {tag}] n1={n1} n2={n2} rel {rel:.3g} '
                      f'{"PASS" if rel < 1e-5 else "FAIL"}; '
                      f'speedup vs XLA: {t_ref / t:.2f}x', flush=True)
            except Exception as e:
                print(f'[{kind} {tag}] FAILED: {type(e).__name__}: {str(e)[:250]}',
                      flush=True)


if __name__ == '__main__':
    main()
