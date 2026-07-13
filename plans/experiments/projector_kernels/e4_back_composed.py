"""E4 preview: the COMPOSED back projection — the Pallas kernel driving a full
production-shaped call (all 771k ROR pixels x 1024 views at the 1024^3 parallel cell)
vs the warm library `sparse_back_project`.

This answers the composition question ahead of the E4 design commitment: at 16-26x
kernel speed the old driver (94-step pixel scan, transfer chunks) would dominate — but
the Pallas kernel has NO scan carry, so the composed driver is just:

    for each view-chunk of 128:                 (keeps the L2-phase slice at ~130 MB)
        centers  <- _jit_compute_scatter_centers(chunk)          [timed]
        weights  <- the weight formula (V,T,P) — no sort         [timed]
        sino_cm  <- transpose+pad the chunk channel-major        [timed]
        out     += pallas kernel, ONE grid over (row-chunks, ALL pixels)   [timed]

Every cost is charged: precompute, layout, kernel, accumulation.  Value gate: rel <=
1e-5 vs the library result.  Also the Hessian path (coeff_power=2, weights**2).

Run (gautschi 1 GPU):  python plans/experiments/projector_kernels/e4_back_composed.py
"""
import importlib.util
import os

# ── Config ────────────────────────────────────────────────────────────────────
SINO_SHAPE = (1024, 1008, 992)
VIEW_CHUNK = 128
RC, NUM_WARPS = 256, 2              # the bench subset-best config
WARMUP, TRIALS = 1, 3

os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.9')
import time                        # noqa: E402
import numpy as np                 # noqa: E402
import mbirjax                     # noqa: E402
import jax                         # noqa: E402
import jax.numpy as jnp            # noqa: E402
from mbirjax.parallel_beam import ParallelBeamModel      # noqa: E402
from mbirjax.projectors import ProjectorParams, _jit_compute_scatter_centers  # noqa: E402

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('bk', os.path.join(_here, 'e3_back_pallas_v1.py'))
bk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bk)


def main():
    num_views, rows, C = SINO_SHAPE
    r_pad = bk.next_pow2(rows)
    angles = np.linspace(0, np.pi, num_views, endpoint=False)
    model = mbirjax.ParallelBeamModel(SINO_SHAPE, angles)
    model.configure_devices(1)
    args = (tuple(model.get_params('sinogram_shape')),
            tuple(model.get_params('recon_shape')), model.get_geometry_parameters())
    args += tuple(int(bool(getattr(model.tiles, f, 0))) for f in ProjectorParams._fields[3:])
    pp = ProjectorParams(*args)
    psf_radius = pp.geometry_params.psf_radius
    recon_shape = model.get_params('recon_shape')
    idx = mbirjax.gen_full_indices(recon_shape, use_ror_mask=model.get_params('use_ror_mask'))
    P = len(idx)
    rng = np.random.default_rng(0)
    sino = model._shard_sinogram(rng.random(SINO_SHAPE, dtype=np.float32))
    view_params_all = jnp.asarray(model.projector_functions.view_params_array)
    jax.block_until_ready(sino)
    print(f'P={P} views={num_views} chunks of {VIEW_CHUNK}; rc={RC} w={NUM_WARPS}',
          flush=True)

    # ── Library baseline (warm) ───────────────────────────────────────────────
    for _ in range(WARMUP):
        jax.block_until_ready(model.sparse_back_project(sino, idx))
    ts = []
    for _ in range(TRIALS):
        t0 = time.perf_counter()
        ref = jax.block_until_ready(model.sparse_back_project(sino, idx))
        ts.append(time.perf_counter() - t0)
    t_lib = float(np.median(ts))
    print(f'library sparse_back_project: {t_lib:.3f} s', flush=True)
    ref = jnp.asarray(ref)
    scale = float(jnp.max(jnp.abs(ref)))

    # ── Composed pallas path ──────────────────────────────────────────────────
    # Weights builder jitted ONCE (module-level-jit structure): the retrace-per-call
    # hypothesis says the 1.83 s weights cost was host tracing, not device work.
    gp = pp.geometry_params

    def _weights(view_params, centers, coeff_power):
        def one_view(single_view_params, cts):
            n_p, W_p_c, footprint = ParallelBeamModel.compute_proj_data(
                idx, single_view_params, pp)
            delta_voxel_row = gp.voxel_row_aspect * gp.delta_voxel
            scale = (delta_voxel_row * gp.delta_voxel) / footprint
            offs = jnp.arange(-gp.psf_radius, gp.psf_radius + 1)
            n = cts[None, :] + offs[:, None]
            L_max = jnp.minimum(1.0, W_p_c)
            A = scale * jnp.clip((W_p_c + 1.0) / 2.0 - jnp.abs(n_p - n), 0.0, L_max)
            A = A * ((n >= 0) & (n < C))
            return A ** coeff_power
        return jax.vmap(one_view)(view_params, centers)

    weights_jit = jax.jit(_weights, static_argnums=2)

    n_chunks = num_views // VIEW_CHUNK
    kern = jax.jit(bk.make_back_call(VIEW_CHUNK, C, P, r_pad, RC, NUM_WARPS, psf_radius))
    to_cm = jax.jit(lambda s: jnp.pad(jnp.swapaxes(s, 1, 2),
                                      ((0, 0), (0, 0), (0, r_pad - rows))))
    add = jax.jit(lambda a, b: a + b, donate_argnums=0)

    def composed(coeff_power):
        parts = dict(centers=0.0, weights=0.0, layout=0.0, kernel=0.0, accum=0.0)
        out = None
        for c in range(n_chunks):
            vsl = slice(c * VIEW_CHUNK, (c + 1) * VIEW_CHUNK)
            t0 = time.perf_counter()
            n_pc = _jit_compute_scatter_centers(
                view_params_all[vsl], idx, ParallelBeamModel.compute_channel_coordinate,
                pp, pixels_major=False)
            jax.block_until_ready(n_pc)
            parts['centers'] += time.perf_counter() - t0
            t0 = time.perf_counter()
            wts = jax.block_until_ready(
                weights_jit(view_params_all[vsl], jnp.asarray(n_pc), coeff_power))
            parts['weights'] += time.perf_counter() - t0
            t0 = time.perf_counter()
            sino_cm = jax.block_until_ready(to_cm(sino[vsl]))
            parts['layout'] += time.perf_counter() - t0
            t0 = time.perf_counter()
            chunk_out = jax.block_until_ready(kern(jnp.asarray(n_pc), wts, sino_cm))
            parts['kernel'] += time.perf_counter() - t0
            t0 = time.perf_counter()
            out = chunk_out if out is None else jax.block_until_ready(add(out, chunk_out))
            parts['accum'] += time.perf_counter() - t0
            del n_pc, wts, sino_cm, chunk_out
        return out[:, :rows], parts

    for _ in range(WARMUP):
        out, _ = composed(1)
    ts, all_parts = [], None
    for _ in range(TRIALS):
        t0 = time.perf_counter()
        out, parts = composed(1)
        ts.append(time.perf_counter() - t0)
        all_parts = parts
    t_comp = float(np.median(ts))
    rel = float(jnp.max(jnp.abs(out - ref)) / scale)
    print(f'composed pallas back: {t_comp:.3f} s  rel {rel:.3g} '
          f'{"PASS" if rel < 1e-5 else "FAIL"}', flush=True)
    print('  breakdown: ' + '  '.join(f'{k} {v:.3f}s' for k, v in all_parts.items()),
          flush=True)
    print(f'COMPOSED SPEEDUP: {t_lib / t_comp:.2f}x  (library {t_lib:.2f} s -> '
          f'{t_comp:.2f} s)', flush=True)

    # Hessian path once (vs library coeff_power=2).
    for _ in range(WARMUP):
        jax.block_until_ready(model.sparse_back_project(sino, idx, coeff_power=2))
    t0 = time.perf_counter()
    ref2 = jax.block_until_ready(model.sparse_back_project(sino, idx, coeff_power=2))
    t_lib2 = time.perf_counter() - t0
    out2, _ = composed(2)
    t0 = time.perf_counter()
    out2, _ = composed(2)
    t_comp2 = time.perf_counter() - t0
    rel2 = float(jnp.max(jnp.abs(out2 - jnp.asarray(ref2))) / scale)
    print(f'hessian: library {t_lib2:.2f} s -> composed {t_comp2:.2f} s '
          f'({t_lib2 / t_comp2:.2f}x)  rel {rel2:.3g} '
          f'{"PASS" if rel2 < 1e-5 else "FAIL"}', flush=True)


if __name__ == '__main__':
    main()
