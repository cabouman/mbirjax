"""E3 step zero, round 3: batch the remaining backend-constraint fixes (rounds 1-2:
e3_pallas_smoke{,2}.py; findings doc has the verdicts).

Round-2 rejections: Triton = "Unimplemented primitive: slice" (the w[..., t] tap index
of a loaded array); MGPU = 128-byte warpgroup alignment on block copies (126- and
252-float minor dims).  Round-3 variants, ALL with per-tap weight inputs (no slicing of
loaded arrays anywhere) and fully padded/aligned shapes:

  triton_pertap : R->1024, L->128; gather via jnp.take_along_axis
  triton_ixgather: same, gather via basic integer-array indexing (different lowering path)
  mgpu_aligned  : R->1024 in 4 chunks of 256 (<=256/dim cap, 1024 B aligned), L->128,
                  per-chunk masked tap accumulation

Gates: interpret rel < 1e-6 vs the unpadded XLA reference (valid region); GPU value +
timing vs XLA.  One of these compiling and matching is the round's success; timing is
calibration only.

Run:  python plans/experiments/projector_kernels/e3_pallas_smoke3.py
"""
import os

# ── Config ────────────────────────────────────────────────────────────────────
P, R, L, T = 8192, 1008, 126, 3
R_PAD, L_PAD = 1024, 128
R_CHUNK = 256
PIXEL_TILE = 8
WARMUP, TRIALS = 2, 10

os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.7')
import time                     # noqa: E402
import numpy as np              # noqa: E402
import jax                      # noqa: E402
import jax.numpy as jnp         # noqa: E402
from jax.experimental import pallas as pl    # noqa: E402


def xla_reference(col, m, w):
    out = jnp.zeros(m.shape, dtype=col.dtype)
    for t in range(T):
        rows = jnp.clip(m + (t - T // 2), 0, R - 1)
        out = out + w[..., t] * jnp.take_along_axis(col, rows, axis=1)
    return out


def make_kernel(gather_mode):
    """Per-tap weights as separate refs; no slicing of loaded arrays."""
    def kernel(col_ref, m_ref, w0_ref, w1_ref, w2_ref, out_ref):
        cols = col_ref[...]                          # (TILE, R_PAD)
        m = m_ref[...]                               # (TILE, L_PAD)
        acc = jnp.zeros(m.shape, dtype=cols.dtype)
        for t, w_ref in enumerate((w0_ref, w1_ref, w2_ref)):
            w_t = w_ref[...]
            rows = jnp.clip(m + (t - T // 2), 0, R - 1)
            if gather_mode == 'take':
                g = jnp.take_along_axis(cols, rows, axis=1)
            else:
                g = cols[jnp.arange(cols.shape[0])[:, None], rows]
            acc = acc + w_t * g
        out_ref[...] = acc
    return kernel


def make_call(gather_mode, interpret=False, compiler_params=None):
    return pl.pallas_call(
        make_kernel(gather_mode),
        out_shape=jax.ShapeDtypeStruct((P, L_PAD), jnp.float32),
        grid=(P // PIXEL_TILE,),
        in_specs=[pl.BlockSpec((PIXEL_TILE, R_PAD), lambda i: (i, 0)),
                  pl.BlockSpec((PIXEL_TILE, L_PAD), lambda i: (i, 0))]
                 + [pl.BlockSpec((PIXEL_TILE, L_PAD), lambda i: (i, 0))] * T,
        out_specs=pl.BlockSpec((PIXEL_TILE, L_PAD), lambda i: (i, 0)),
        interpret=interpret,
        **({'compiler_params': compiler_params} if compiler_params else {}))


def chunked_kernel(col_ref, m_ref, w0_ref, w1_ref, w2_ref, out_ref):
    c = pl.program_id(1)
    base = c * R_CHUNK
    cols = col_ref[...]                              # (TILE, R_CHUNK)
    m = m_ref[...]                                   # (TILE, L_PAD)
    acc = jnp.zeros(m.shape, dtype=cols.dtype)
    for t, w_ref in enumerate((w0_ref, w1_ref, w2_ref)):
        w_t = w_ref[...]
        rows = jnp.clip(m + (t - T // 2), 0, R - 1)
        local = rows - base
        ok = (local >= 0) & (local < R_CHUNK)
        g = jnp.take_along_axis(cols, jnp.clip(local, 0, R_CHUNK - 1), axis=1)
        acc = acc + jnp.where(ok, w_t * g, 0.0)
    @pl.when(c == 0)
    def _init():
        out_ref[...] = jnp.zeros_like(out_ref)
    out_ref[...] += acc


def make_chunked(interpret=False):
    return pl.pallas_call(
        chunked_kernel,
        out_shape=jax.ShapeDtypeStruct((P, L_PAD), jnp.float32),
        grid=(P // PIXEL_TILE, R_PAD // R_CHUNK),
        in_specs=[pl.BlockSpec((PIXEL_TILE, R_CHUNK), lambda i, c: (i, c)),
                  pl.BlockSpec((PIXEL_TILE, L_PAD), lambda i, c: (i, 0))]
                 + [pl.BlockSpec((PIXEL_TILE, L_PAD), lambda i, c: (i, 0))] * T,
        out_specs=pl.BlockSpec((PIXEL_TILE, L_PAD), lambda i, c: (i, 0)),
        interpret=interpret)


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
    print(f'jax {jax.__version__}  devices={jax.devices()}', flush=True)
    rng = np.random.default_rng(0)
    col = jnp.asarray(rng.random((P, R), dtype=np.float32))
    m = jnp.asarray(rng.integers(0, R, size=(P, L)).astype(np.int32))
    w = jnp.asarray(rng.random((P, L, T), dtype=np.float32))
    col_pad = jnp.pad(col, ((0, 0), (0, R_PAD - R)))
    m_pad = jnp.pad(m, ((0, 0), (0, L_PAD - L)))
    w_taps = [jnp.pad(w[..., t], ((0, 0), (0, L_PAD - L))) for t in range(T)]
    args = (col_pad, m_pad, *w_taps)
    jax.block_until_ready(args)

    ref_fn = jax.jit(xla_reference)
    ref = jax.block_until_ready(ref_fn(col, m, w))
    scale = float(jnp.max(jnp.abs(ref)))

    for name, mk in (('take', lambda: make_call('take', interpret=True)),
                     ('ixgather', lambda: make_call('ix', interpret=True)),
                     ('chunked', lambda: make_chunked(interpret=True))):
        out = mk()(*args)[:, :L]
        rel = float(jnp.max(jnp.abs(out - ref)) / scale)
        print(f'[interpret {name}] rel {rel:.3g} {"PASS" if rel < 1e-6 else "FAIL"}',
              flush=True)

    if not on_gpu:
        print('[gpu] no GPU — deferred to the cluster run')
        return

    t_ref = bench(ref_fn, (col, m, w), 'xla_reference')
    from jax.experimental.pallas import triton as pltriton
    candidates = [
        ('triton_pertap', lambda: jax.jit(make_call(
            'take', compiler_params=pltriton.CompilerParams()))),
        ('triton_ixgather', lambda: jax.jit(make_call(
            'ix', compiler_params=pltriton.CompilerParams()))),
        ('mgpu_aligned', lambda: jax.jit(make_chunked())),
    ]
    for name, mk in candidates:
        try:
            f = mk()
            out = jax.block_until_ready(f(*args))[:, :L]
            rel = float(jnp.max(jnp.abs(out - ref)) / scale)
            t = bench(f, args, name)
            print(f'[{name}] rel err {rel:.3g}; {t / t_ref:.2f}x XLA', flush=True)
        except Exception as e:
            print(f'[{name}] FAILED: {type(e).__name__}: {str(e)[:300]}', flush=True)


if __name__ == '__main__':
    main()
