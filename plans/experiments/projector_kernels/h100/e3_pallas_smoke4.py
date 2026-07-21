"""E3 step zero, round 4: REF-level gathers (the paged_attention idiom).

Round-3 finding: neither backend implements the HLO `gather` primitive in-kernel — but
in-tree paged_attention gathers fine at this pin by indexing the REF with an integer
array (`k_pages_ref[block_tables]`), which lowers to Triton pointer loads.  Round 3
loaded the block into an array and gathered THE ARRAY (an in-kernel HLO gather).  Round
4 indexes refs directly, three sub-variants (Triton backend; MGPU's lane-semantics
gather gap is deeper — parked):

  refix_tile1 : grid over pixels (TILE=1); 1-D index-array ref-indexing per pixel
  refix_tile8 : TILE=8; 2-D advanced ref-indexing (row ids x gathered rows)
  refix_flat  : col flattened to (P*R,); whole-array ref window; 2-D flat-index gather

Gates as before: interpret rel < 1e-6; GPU value + timing vs XLA.

Run:  python plans/experiments/projector_kernels/e3_pallas_smoke4.py
"""
import os

# ── Config ────────────────────────────────────────────────────────────────────
P, R, L, T = 8192, 1008, 126, 3
R_PAD, L_PAD = 1024, 128
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


def kernel_tile1(col_ref, m_ref, w0_ref, w1_ref, w2_ref, out_ref):
    m = m_ref[0, :]                                   # (L_PAD,)
    acc = jnp.zeros((L_PAD,), dtype=jnp.float32)
    for t, w_ref in enumerate((w0_ref, w1_ref, w2_ref)):
        rows = jnp.clip(m + (t - T // 2), 0, R - 1)
        acc = acc + w_ref[0, :] * col_ref[0, rows]    # REF gather, 1-D index array
    out_ref[0, :] = acc


def kernel_tile8(col_ref, m_ref, w0_ref, w1_ref, w2_ref, out_ref):
    tile = m_ref.shape[0]
    m = m_ref[...]
    ii = jnp.arange(tile)[:, None]
    acc = jnp.zeros(m.shape, dtype=jnp.float32)
    for t, w_ref in enumerate((w0_ref, w1_ref, w2_ref)):
        rows = jnp.clip(m + (t - T // 2), 0, R - 1)
        acc = acc + w_ref[...] * col_ref[ii, rows]    # REF gather, 2-D advanced index
    out_ref[...] = acc


def kernel_flat(col_ref, m_ref, w0_ref, w1_ref, w2_ref, out_ref, *, tile):
    i = pl.program_id(0)
    m = m_ref[...]                                    # (tile, L_PAD)
    pix = i * tile + jnp.arange(tile)[:, None]        # global pixel ids
    acc = jnp.zeros(m.shape, dtype=jnp.float32)
    for t, w_ref in enumerate((w0_ref, w1_ref, w2_ref)):
        rows = jnp.clip(m + (t - T // 2), 0, R - 1)
        acc = acc + w_ref[...] * col_ref[pix * R_PAD + rows]   # flat REF gather
    out_ref[...] = acc


def make(variant, interpret=False):
    from functools import partial
    from jax.experimental.pallas import triton as pltriton
    kw = {} if interpret else {'compiler_params': pltriton.CompilerParams()}
    if variant == 'tile1':
        tile = 1
        kernel, col_spec = kernel_tile1, pl.BlockSpec((1, R_PAD), lambda i: (i, 0))
    elif variant == 'tile8':
        tile = 8
        kernel, col_spec = kernel_tile8, pl.BlockSpec((tile, R_PAD), lambda i: (i, 0))
    else:
        tile = 8
        kernel = partial(kernel_flat, tile=tile)
        col_spec = pl.BlockSpec((P * R_PAD,), lambda i: (0,))   # whole flat array window
    return pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((P, L_PAD), jnp.float32),
        grid=(P // tile,),
        in_specs=[col_spec] + [pl.BlockSpec((tile, L_PAD), lambda i: (i, 0))] * 4,
        out_specs=pl.BlockSpec((tile, L_PAD), lambda i: (i, 0)),
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
    print(f'  {name:16s} {med * 1e6:9.1f} us/call', flush=True)
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
    args2d = (col_pad, m_pad, *w_taps)
    args_flat = (col_pad.reshape(-1), m_pad, *w_taps)
    jax.block_until_ready((args2d, args_flat))

    ref_fn = jax.jit(xla_reference)
    ref = jax.block_until_ready(ref_fn(col, m, w))
    scale = float(jnp.max(jnp.abs(ref)))

    grids = {'tile1': args2d, 'tile8': args2d, 'flat': args_flat}
    for variant, args in grids.items():
        try:
            out = make(variant, interpret=True)(*args)[:, :L]
            rel = float(jnp.max(jnp.abs(out - ref)) / scale)
            print(f'[interpret {variant}] rel {rel:.3g} '
                  f'{"PASS" if rel < 1e-6 else "FAIL"}', flush=True)
        except Exception as e:
            print(f'[interpret {variant}] FAILED: {type(e).__name__}: {str(e)[:200]}',
                  flush=True)

    if not on_gpu:
        print('[gpu] no GPU — deferred to the cluster run')
        return

    t_ref = bench(ref_fn, (col, m, w), 'xla_reference')
    for variant, args in grids.items():
        try:
            f = jax.jit(make(variant))
            out = jax.block_until_ready(f(*args))[:, :L]
            rel = float(jnp.max(jnp.abs(out - ref)) / scale)
            t = bench(f, args, f'triton_{variant}')
            print(f'[triton_{variant}] rel err {rel:.3g}; {t / t_ref:.2f}x XLA',
                  flush=True)
        except Exception as e:
            print(f'[triton_{variant}] FAILED: {type(e).__name__}: {str(e)[:250]}',
                  flush=True)


if __name__ == '__main__':
    main()
