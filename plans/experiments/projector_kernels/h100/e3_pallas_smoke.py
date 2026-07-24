"""E3 step zero: Pallas mechanics smoke on the pinned stack (jax 0.10.1, H100).

Before designing the real band kernel, answer three vehicle questions cheaply
(gpu_headroom_plan.md E3; appendix_pallas_assessment.md risks):
  1. Does pallas_call COMPILE AND RUN CORRECTLY on gautschi's H100 + jax 0.10.1 for a
     computed-index row-gather kernel (the vertical-fan access pattern — the documented
     DSL weak spot)?
  2. Which backend works: the default (Mosaic GPU on Hopper) and/or Triton
     (compiler_params fallback)?
  3. Ballpark: how far is a naive Pallas version from the XLA equivalent at a
     production-ish shape?  (A mechanics check, NOT the E3 success bar.)

The smoke kernel is the vertical-fan band gather essence WITH the slice-index-SET
signature (the parity/band-unifying interface): given per-pixel detector columns
col (P, R), per-(pixel, band-slice) center rows m (P, L) for an ARBITRARY slice-index
set of length L, and tap weights w (P, L, T):
    out[p, l] = sum_t w[p, l, t] * col[p, clip(m[p, l] + t - T//2, 0, R-1)]

Value gate: bitwise/allclose vs the XLA reference in interpret mode (CPU) and on GPU.

Run:  python plans/experiments/projector_kernels/e3_pallas_smoke.py   (constants below)
"""
import os

# ── Config ────────────────────────────────────────────────────────────────────
P, R, L, T = 8192, 1008, 126, 3     # pixels, det rows, slice-set size (stride-8), taps
PIXEL_TILE = 8
WARMUP, TRIALS = 2, 10

os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.7')
import time                     # noqa: E402
import numpy as np              # noqa: E402
import jax                      # noqa: E402
import jax.numpy as jnp         # noqa: E402
from jax.experimental import pallas as pl    # noqa: E402


def xla_reference(col, m, w):
    """The XLA formulation: per-tap take_along_axis + weighted sum."""
    out = jnp.zeros(m.shape, dtype=col.dtype)
    for t in range(T):
        rows = jnp.clip(m + (t - T // 2), 0, R - 1)
        out = out + w[..., t] * jnp.take_along_axis(col, rows, axis=1)
    return out


def gather_kernel(col_ref, m_ref, w_ref, out_ref):
    cols = col_ref[...]                     # (TILE, R)
    m = m_ref[...]                          # (TILE, L)
    w = w_ref[...]                          # (TILE, L, T)
    acc = jnp.zeros(m.shape, dtype=cols.dtype)
    for t in range(T):
        rows = jnp.clip(m + (t - T // 2), 0, R - 1)
        acc = acc + w[..., t] * jnp.take_along_axis(cols, rows, axis=1)
    out_ref[...] = acc


def make_pallas_call(interpret=False, compiler_params=None):
    return pl.pallas_call(
        gather_kernel,
        out_shape=jax.ShapeDtypeStruct((P, L), jnp.float32),
        grid=(P // PIXEL_TILE,),
        in_specs=[pl.BlockSpec((PIXEL_TILE, R), lambda i: (i, 0)),
                  pl.BlockSpec((PIXEL_TILE, L), lambda i: (i, 0)),
                  pl.BlockSpec((PIXEL_TILE, L, T), lambda i: (i, 0, 0))],
        out_specs=pl.BlockSpec((PIXEL_TILE, L), lambda i: (i, 0)),
        interpret=interpret,
        **({'compiler_params': compiler_params} if compiler_params else {}))


def bench(fn, args, name):
    for _ in range(WARMUP):
        jax.block_until_ready(fn(*args))
    ts = []
    for _ in range(TRIALS):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append(time.perf_counter() - t0)
    med = float(np.median(ts))
    print(f'  {name:22s} {med * 1e6:9.1f} us/call', flush=True)
    return med


def main():
    on_gpu = jax.devices()[0].platform == 'gpu'
    print(f'jax {jax.__version__}  devices={jax.devices()}  '
          f'P={P} R={R} L={L} (slice-set stride 8) T={T}', flush=True)
    rng = np.random.default_rng(0)
    col = jnp.asarray(rng.random((P, R), dtype=np.float32))
    slice_set = np.arange(0, 8 * L, 8)                      # the strided slice-index SET
    m = jnp.asarray(rng.integers(0, R, size=(P, L)).astype(np.int32))
    w = jnp.asarray(rng.random((P, L, T), dtype=np.float32))
    jax.block_until_ready((col, m, w))
    print(f'slice set: {len(slice_set)} indices, stride 8 (parity-style)', flush=True)

    ref_fn = jax.jit(xla_reference)
    ref = jax.block_until_ready(ref_fn(col, m, w))

    # 1. Interpret mode (any platform): semantics gate.  Scale-invariant rel-max per the
    # float-gate rules (FMA/summation context differs between interpret and jit — a few
    # ULP is expected; a semantics bug would be O(1)).
    out_i = make_pallas_call(interpret=True)(col, m, w)
    rel_i = float(jnp.max(jnp.abs(out_i - ref)) / jnp.max(jnp.abs(ref)))
    print(f'[interpret] rel max err vs XLA ref: {rel_i:.3g} '
          f'{"PASS" if rel_i < 1e-6 else "FAIL — investigate"}', flush=True)

    if not on_gpu:
        print('[gpu] no GPU present — compile/perf checks deferred to the cluster run')
        return

    t_ref = bench(ref_fn, (col, m, w), 'xla_reference')

    # 2. Default backend (Mosaic GPU on Hopper at this pin).
    try:
        f = jax.jit(make_pallas_call())
        out = jax.block_until_ready(f(col, m, w))
        err = float(jnp.max(jnp.abs(out - ref)))
        t = bench(f, (col, m, w), 'pallas_default')
        print(f'[pallas default backend] max abs err {err:.3g}; '
              f'{t / t_ref:.2f}x the XLA time', flush=True)
    except Exception as e:
        print(f'[pallas default backend] FAILED: {type(e).__name__}: {str(e)[:400]}',
              flush=True)

    # 3. Triton backend fallback.
    try:
        from jax.experimental.pallas import triton as pltriton
        f = jax.jit(make_pallas_call(compiler_params=pltriton.CompilerParams()))
        out = jax.block_until_ready(f(col, m, w))
        err = float(jnp.max(jnp.abs(out - ref)))
        t = bench(f, (col, m, w), 'pallas_triton')
        print(f'[pallas triton backend] max abs err {err:.3g}; '
              f'{t / t_ref:.2f}x the XLA time', flush=True)
    except Exception as e:
        print(f'[pallas triton backend] FAILED: {type(e).__name__}: {str(e)[:400]}',
              flush=True)


if __name__ == '__main__':
    main()
