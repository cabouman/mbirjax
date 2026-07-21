"""E3 step zero, round 2: fix both backends' round-1 rejections (e3_pallas_smoke.py).

Round-1 verdicts (job 13473088): interpret PASS; Mosaic GPU rejected the (8, 1008) col
block copy (async copies capped at 256 elements/dim); Triton rejected non-power-of-2
shapes.  Round 2:

  pallas_triton_padded : pad R 1008->1024, L 126->128, T 3->4 (zero weights) — all
                         power-of-2; Triton's pointer-style gathers should lower.
  pallas_mgpu_chunked  : R-chunked col staging (4 chunks of 252 <= 256/dim), taps
                         accumulated per chunk with an in-range mask — pure BlockSpec
                         machinery, no exotic MGPU API.

Value gates: rel max err vs the unpadded XLA reference on the valid region (< 1e-6;
float-gate rules).  Timing vs the XLA reference calibrates viability, not the E3 bar.

Run:  python plans/experiments/projector_kernels/e3_pallas_smoke2.py
"""
import os

# ── Config ────────────────────────────────────────────────────────────────────
P, R, L, T = 8192, 1008, 126, 3
R_PAD, L_PAD, T_PAD = 1024, 128, 4
PIXEL_TILE = 8
R_CHUNK = 252                    # 1008 / 4, <= 256 per-dim async-copy cap
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


# ── Triton variant: power-of-2 padded shapes ─────────────────────────────────
def kernel_padded(col_ref, m_ref, w_ref, out_ref):
    cols = col_ref[...]                      # (TILE, R_PAD)
    m = m_ref[...]                           # (TILE, L_PAD)
    w = w_ref[...]                           # (TILE, L_PAD, T_PAD)
    acc = jnp.zeros(m.shape, dtype=cols.dtype)
    for t in range(T_PAD):
        rows = jnp.clip(m + (t - T // 2), 0, R - 1)      # pad taps carry zero weight
        acc = acc + w[..., t] * jnp.take_along_axis(cols, rows, axis=1)
    out_ref[...] = acc


def make_triton(compiler_params):
    return pl.pallas_call(
        kernel_padded,
        out_shape=jax.ShapeDtypeStruct((P, L_PAD), jnp.float32),
        grid=(P // PIXEL_TILE,),
        in_specs=[pl.BlockSpec((PIXEL_TILE, R_PAD), lambda i: (i, 0)),
                  pl.BlockSpec((PIXEL_TILE, L_PAD), lambda i: (i, 0)),
                  pl.BlockSpec((PIXEL_TILE, L_PAD, T_PAD), lambda i: (i, 0, 0))],
        out_specs=pl.BlockSpec((PIXEL_TILE, L_PAD), lambda i: (i, 0)),
        compiler_params=compiler_params)


# ── MGPU variant: R-chunked staging with per-chunk masked accumulation ────────
def kernel_chunked(col_ref, m_ref, w_ref, out_ref):
    # col_ref block is (TILE, R_CHUNK) for chunk c = program_id(1); accumulate the taps
    # whose row lands in this chunk.  Output block revisited across the chunk grid dim.
    c = pl.program_id(1)
    base = c * R_CHUNK
    cols = col_ref[...]                      # (TILE, R_CHUNK)
    m = m_ref[...]                           # (TILE, L)
    w = w_ref[...]                           # (TILE, L, T)
    acc = jnp.zeros(m.shape, dtype=cols.dtype)
    for t in range(T):
        rows = jnp.clip(m + (t - T // 2), 0, R - 1)
        local = rows - base
        in_chunk = (local >= 0) & (local < R_CHUNK)
        gathered = jnp.take_along_axis(cols, jnp.clip(local, 0, R_CHUNK - 1), axis=1)
        acc = acc + jnp.where(in_chunk, w[..., t] * gathered, 0.0)
    @pl.when(c == 0)
    def _init():
        out_ref[...] = jnp.zeros_like(out_ref)
    out_ref[...] += acc


def make_mgpu_chunked():
    return pl.pallas_call(
        kernel_chunked,
        out_shape=jax.ShapeDtypeStruct((P, L), jnp.float32),
        grid=(P // PIXEL_TILE, R // R_CHUNK),
        in_specs=[pl.BlockSpec((PIXEL_TILE, R_CHUNK), lambda i, c: (i, c)),
                  pl.BlockSpec((PIXEL_TILE, L), lambda i, c: (i, 0)),
                  pl.BlockSpec((PIXEL_TILE, L, T), lambda i, c: (i, 0, 0))],
        out_specs=pl.BlockSpec((PIXEL_TILE, L), lambda i, c: (i, 0)))


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
    w_pad = jnp.pad(w, ((0, 0), (0, L_PAD - L), (0, T_PAD - T)))
    jax.block_until_ready((col, m, w, col_pad, m_pad, w_pad))

    ref_fn = jax.jit(xla_reference)
    ref = jax.block_until_ready(ref_fn(col, m, w))
    ref_scale = float(jnp.max(jnp.abs(ref)))

    # Interpret-mode gates for both formulations (any platform).
    out_c = make_mgpu_chunked()(col, m, w) if False else None  # interpret separately below
    for name, mk, args, valid in (
            ('padded', lambda **kw: pl.pallas_call(
                kernel_padded, out_shape=jax.ShapeDtypeStruct((P, L_PAD), jnp.float32),
                grid=(P // PIXEL_TILE,),
                in_specs=[pl.BlockSpec((PIXEL_TILE, R_PAD), lambda i: (i, 0)),
                          pl.BlockSpec((PIXEL_TILE, L_PAD), lambda i: (i, 0)),
                          pl.BlockSpec((PIXEL_TILE, L_PAD, T_PAD), lambda i: (i, 0, 0))],
                out_specs=pl.BlockSpec((PIXEL_TILE, L_PAD), lambda i: (i, 0)),
                interpret=True), (col_pad, m_pad, w_pad), lambda o: o[:, :L]),
            ('chunked', lambda **kw: pl.pallas_call(
                kernel_chunked, out_shape=jax.ShapeDtypeStruct((P, L), jnp.float32),
                grid=(P // PIXEL_TILE, R // R_CHUNK),
                in_specs=[pl.BlockSpec((PIXEL_TILE, R_CHUNK), lambda i, c: (i, c)),
                          pl.BlockSpec((PIXEL_TILE, L), lambda i, c: (i, 0)),
                          pl.BlockSpec((PIXEL_TILE, L, T), lambda i, c: (i, 0, 0))],
                out_specs=pl.BlockSpec((PIXEL_TILE, L), lambda i, c: (i, 0)),
                interpret=True), (col, m, w), lambda o: o)):
        out = valid(mk()(*args))
        rel = float(jnp.max(jnp.abs(out - ref)) / ref_scale)
        print(f'[interpret {name}] rel max err {rel:.3g} '
              f'{"PASS" if rel < 1e-6 else "FAIL"}', flush=True)

    if not on_gpu:
        print('[gpu] no GPU — compile/perf deferred to the cluster run')
        return

    t_ref = bench(ref_fn, (col, m, w), 'xla_reference')

    try:
        from jax.experimental.pallas import triton as pltriton
        f = jax.jit(make_triton(pltriton.CompilerParams()))
        out = jax.block_until_ready(f(col_pad, m_pad, w_pad))[:, :L]
        rel = float(jnp.max(jnp.abs(out - ref)) / ref_scale)
        t = bench(f, (col_pad, m_pad, w_pad), 'pallas_triton_padded')
        print(f'[triton padded] rel err {rel:.3g}; {t / t_ref:.2f}x XLA', flush=True)
    except Exception as e:
        print(f'[triton padded] FAILED: {type(e).__name__}: {str(e)[:400]}', flush=True)

    try:
        f = jax.jit(make_mgpu_chunked())
        out = jax.block_until_ready(f(col, m, w))
        rel = float(jnp.max(jnp.abs(out - ref)) / ref_scale)
        t = bench(f, (col, m, w), 'pallas_mgpu_chunked')
        print(f'[mgpu chunked] rel err {rel:.3g}; {t / t_ref:.2f}x XLA', flush=True)
    except Exception as e:
        print(f'[mgpu chunked] FAILED: {type(e).__name__}: {str(e)[:400]}', flush=True)


if __name__ == '__main__':
    main()
