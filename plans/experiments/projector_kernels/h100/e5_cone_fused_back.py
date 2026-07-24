"""E5: the cone fused-vfan back-projection kernel spike (2026-07-13).

Design + panel amendments: gpu_headroom_findings.md, "Cone fused-vfan back kernel"
(the affine row-center fact m(v,p,l) = m0 + W_p_r*l enables an in-kernel vertical
fan from per-(view,pixel) scalars only).  Gates (Greg 2026-07-13): gradient rel-max
<= 1e-5, Hessian <= 1e-4 (preconditioner argument; the f32 affine-m rounding
sequence differs from the XLA chain's by ~1-2 ULP and squared weights do not
cancel), plus the adjoint identity vs the XLA cone forward.  Bar: >= 1.5x the
best-XLA per-owner band sweep (21.2 s at vb16; 27.8 s at the production vb512);
panel-honest expectation 2.5-4.5 s (5-8x).

Structure (single process, e3-style):
  0. DAY-0 PROBE: the vector ref-gather sino_ref[v, cc, m_vec] and floor-rounding
     must lower on the Triton backend -- no shipped kernel exercises a vector
     indexer.  Abort with a clear message if not.
  1. Precompute builders (module-level jits): hfan {c0 concrete centers, Wchan(T)}
     via the geometry-generic machinery + an experiment-local cone hfan_data wrapper
     (weight_scale = delta_voxel_row*delta_voxel/footprint_xy, matching
     back_horizontal_fan_one_view_to_pixel_batch); vfan {m0, W_p_r} from
     compute_vertical_data_single_pixel at slice 0 (slope == W_p_r EXACTLY --
     asserted at build).
  2. The fused kernel: grid (slice-chunk, pixel); per program: acc[LC] registers,
     view loop; m affine, floor(m+0.5) centers (safety = the W_p_r <= 2r invariant),
     division-free Inf-safe divisor 1/sqrt(1+(v_p/sdd)^2), 3 row weights x 3
     channel-scalar factored FMAs.  coeff_power static (2 squares both factors
     AFTER the divisor, matching vertical_fan_band_gather's order).
  3. Bench: XLA baseline = sparse_back_project_band sweep at BASELINE_VB; fused
     sweep over LC in {16,32,64,128} x num_warps in {1,2}; cp in {1,2} gates;
     adjoint <A x, y> = <x, B y> vs the XLA cone forward.  Edge-vs-central band
     walls reported separately (the drift-band caveat).

Run on one H100 (e5_cone_fused_back.slurm); smoke on CPU with E5_SMOKE=1
(interpret mode; probe skipped -- interpret always "lowers").
"""
import functools
import os
import time
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as pltriton

import mbirjax
from mbirjax.cone_beam import ConeBeamModel

# ── Config ────────────────────────────────────────────────────────────────────
SHARD_VIEWS = 512
SINO_ROWS, SINO_CHANNELS = 1024, 1024
BAND_L = 115
BASELINE_VB = 512                       # the production operating point (27.8 s)
LC_SWEEP = (16, 32, 64, 128)
WARP_SWEEP = (1, 2)
VIEW_CHUNK = 128                        # BACK_VIEW_CHUNK_CAP (weights ~2.5 GB bound)
TRIALS = 3
GRAD_TOL, HESS_TOL, ADJ_TOL = 1e-5, 1e-4, 1e-5
SMOKE = os.environ.get('E5_SMOKE') == '1'
if SMOKE:
    SHARD_VIEWS, SINO_ROWS, SINO_CHANNELS, BAND_L = 16, 24, 32, 7
    LC_SWEEP, WARP_SWEEP, VIEW_CHUNK, BASELINE_VB = (4, 8), (1,), 8, 16

T0 = time.perf_counter()


def note(msg):
    print(f'[{time.perf_counter() - T0:8.2f}s] {msg}', flush=True)


def next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


# ── 0. Day-0 lowering probe ───────────────────────────────────────────────────
def probe_vector_gather(interpret):
    """A minimal kernel using BOTH constructs the fused kernel depends on: a vector
    integer ref-gather indexed at (scalar, scalar, vector), and floor-rounding."""
    def kern(x_ref, idx_ref, o_ref):
        m = idx_ref[0, :]                              # (LC,) f32
        mc = jnp.floor(m + 0.5).astype(jnp.int32)      # round emulation
        o_ref[0, :] = x_ref[0, 1, mc]                  # vector gather at fixed (v, c)
    x = jnp.arange(2 * 3 * 8, dtype=jnp.float32).reshape(2, 3, 8)
    idx = jnp.asarray([[0.6, 2.4, 4.5, 7.0]], dtype=jnp.float32)
    # Triton params are LOAD-BEARING: a bare pallas_call defaults to the Mosaic GPU
    # backend on Hopper (warpgroup-divisible copy constraints, different lowering) --
    # the probe must exercise the SAME backend the fused kernel uses.
    kw = ({} if interpret else
          {'compiler_params': pltriton.CompilerParams(num_warps=1)})
    call = pl.pallas_call(kern,
                          out_shape=jax.ShapeDtypeStruct((1, 4), jnp.float32),
                          interpret=interpret, **kw)
    out = np.asarray(jax.jit(call)(x, idx))
    expect = np.asarray([x[0, 1, i] for i in (1, 2, 5, 7)])
    assert np.array_equal(out[0], expect), (out, expect)


# ── 1. Precompute builders ────────────────────────────────────────────────────
def cone_hfan_data(pixel_indices, single_view_params, projector_params):
    """(n_p, W_p_c, weight_scale) for cone -- the compute_hfan_data equivalent,
    matching back_horizontal_fan_one_view_to_pixel_batch's weight scale."""
    gp = projector_params.geometry_params
    n_p, W_p_c, footprint_xy = ConeBeamModel.compute_horizontal_data(
        pixel_indices, single_view_params, projector_params)
    weight_scale = (gp.voxel_row_aspect * gp.delta_voxel * gp.delta_voxel) / footprint_xy
    return n_p, W_p_c, weight_scale


@partial(jax.jit, static_argnames=['projector_params', 'coeff_power'])
def _jit_hfan_weights(view_params_array, pixel_indices, projector_params,
                      coeff_power=1, owned_view_indices=()):
    """Cone channel-tap weights (V, T, P), the trapezoid formula ** coeff_power --
    mirrors _pallas_kernels._jit_compute_back_weights with the local hfan data."""
    if len(owned_view_indices) > 0:
        view_params_array = view_params_array[jnp.asarray(owned_view_indices)]
    gp = projector_params.geometry_params
    num_channels = projector_params.sinogram_shape[2]

    def one_view(svp):
        n_p, W_p_c, scale = cone_hfan_data(pixel_indices, svp, projector_params)
        centers = jnp.round(n_p).astype(jnp.int32)
        offs = jnp.arange(-gp.psf_radius, gp.psf_radius + 1)
        n = centers[None, :] + offs[:, None]
        L_max = jnp.minimum(1.0, W_p_c)
        A = scale * jnp.clip((W_p_c + 1.0) / 2.0 - jnp.abs(n_p - n), 0.0, L_max)
        A = A * ((n >= 0) & (n < num_channels))
        return A ** coeff_power
    return jax.vmap(one_view)(view_params_array)


@partial(jax.jit, static_argnames=['projector_params'])
def _jit_vfan_scalars(view_params_array, pixel_indices, projector_params,
                      owned_view_indices=()):
    """(m0, W_p_r) per (view, pixel): the affine row-center anchor at GLOBAL slice 0
    and the slope==width scalar (verified equal at build time by the caller)."""
    if len(owned_view_indices) > 0:
        view_params_array = view_params_array[jnp.asarray(owned_view_indices)]

    def one_view(svp):
        def one_pixel(pidx):
            m_p, _, W_p_r, _ = ConeBeamModel.compute_vertical_data_single_pixel(
                pidx, jnp.arange(2), svp, projector_params)   # slices {0, 1}
            return m_p[0], m_p[1] - m_p[0], W_p_r if jnp.ndim(W_p_r) == 0 else W_p_r[0]
        return jax.vmap(one_pixel)(pixel_indices)
    m0, slope, wpr = jax.vmap(one_view)(view_params_array)
    return m0, slope, wpr


# ── 2. The fused kernel ───────────────────────────────────────────────────────
def _fused_kernel(c0_ref, wc_ref, m0_ref, wpr_ref, g0_ref, sino_ref, out_ref, *,
                  lc, num_views, num_channels, num_rows, psf_radius, psf_width,
                  coeff_power, delta_det_row, det_row_offset, det_center_row, inv_sdd):
    """One program per (slice-chunk, pixel): registers hold out[p, l0:l0+lc]; the
    view loop forms the vertical fan in-kernel from the affine (m0, W_p_r)."""
    l_vec = g0_ref[0] + pl.program_id(0) * lc + jnp.arange(lc).astype(jnp.float32)

    def vbody(v, acc):
        c0 = c0_ref[v, 0]
        m0 = m0_ref[v, 0]
        wpr = wpr_ref[v, 0]
        m = m0 + wpr * l_vec                                  # affine row centers
        mc = jnp.floor(m + 0.5).astype(jnp.int32)             # W<=2r makes flips inert
        v_p = (m - det_center_row) * delta_det_row - det_row_offset
        # The XLA vfan DIVIDES by cos(phi); division-free that is a MULTIPLY by
        # 1/cos(phi) = sqrt(1 + (v_p/sdd)^2) (Inf-safe: inv_sdd = 0 -> exactly 1).
        inv_cos = jnp.sqrt(1.0 + (v_p * inv_sdd) ** 2)
        L_max = jnp.minimum(1.0, wpr)
        for tr in range(psf_width):                           # static unroll
            mt = mc + (tr - psf_radius)
            w_row = jnp.clip((wpr + 1.0) / 2.0 - jnp.abs(m - mt.astype(jnp.float32)),
                             0.0, L_max) * inv_cos
            w_row = w_row * ((mt >= 0) & (mt < num_rows))
            if coeff_power == 2:
                w_row = w_row * w_row
            mr = jnp.clip(mt, 0, num_rows - 1)
            row_vals = jnp.zeros((lc,), jnp.float32)
            for tc in range(psf_width):                       # static unroll
                cc = jnp.clip(c0 + (tc - psf_radius), 0, num_channels - 1)
                row_vals = row_vals + wc_ref[v, tc, 0] * sino_ref[v, cc, mr]
            acc = acc + w_row * row_vals
        return acc
    acc = jax.lax.fori_loop(0, num_views, vbody, jnp.zeros((lc,), jnp.float32))
    out_ref[0, :] = acc


@functools.cache
def _make_fused_call(num_views, num_channels, num_rows, num_pixels, l_padded, lc,
                     psf_radius, coeff_power, num_warps, geom_consts, interpret):
    kw = ({} if interpret else
          {'compiler_params': pltriton.CompilerParams(num_warps=num_warps)})
    delta_det_row, det_row_offset, det_center_row, inv_sdd = geom_consts
    return pl.pallas_call(
        partial(_fused_kernel, lc=lc, num_views=num_views, num_channels=num_channels,
                num_rows=num_rows, psf_radius=psf_radius, psf_width=2 * psf_radius + 1,
                coeff_power=coeff_power, delta_det_row=delta_det_row,
                det_row_offset=det_row_offset, det_center_row=det_center_row,
                inv_sdd=inv_sdd),
        out_shape=jax.ShapeDtypeStruct((num_pixels, l_padded), jnp.float32),
        grid=(l_padded // lc, num_pixels),                    # slice-chunk SLOWEST
        in_specs=[
            pl.BlockSpec((num_views, 1), lambda s, p: (0, p)),            # c0
            pl.BlockSpec((num_views, 2 * psf_radius + 1, 1),
                         lambda s, p: (0, 0, p)),                         # Wchan
            pl.BlockSpec((num_views, 1), lambda s, p: (0, p)),            # m0
            pl.BlockSpec((num_views, 1), lambda s, p: (0, p)),            # W_p_r
            pl.BlockSpec((1,), lambda s, p: (0,)),                        # g0 scalar
            pl.BlockSpec((num_views, num_channels, num_rows),
                         lambda s, p: (0, 0, 0)),                         # sino (ref)
        ],
        out_specs=pl.BlockSpec((1, lc), lambda s, p: (p, s)),
        interpret=interpret, **kw)


_accum = jax.jit(lambda a, b: a + b, donate_argnums=0)


def fused_band_sweep(model, sino_cm, idx, bands, lc, num_warps, coeff_power,
                     interpret=False):
    """All bands, view-chunked; returns (P, num_slices).  sino_cm: (V, C, rows)."""
    from mbirjax.projectors import _jit_compute_scatter_centers
    pf = model.projector_functions
    pp = getattr(pf, 'projector_params', None)
    if pp is None:
        # Local smoke resolves the MAIN-tree mbirjax, which predates the increment-1
        # projector_params exposure -- rebuild it the way create_projectors does.
        from mbirjax.projectors import ProjectorParams
        tiles = model.tiles
        pp = ProjectorParams(model.get_params('sinogram_shape'),
                             model.get_params('recon_shape'),
                             model.get_geometry_parameters(),
                             int(bool(tiles is not None and tiles.sort_by_channel)),
                             int(bool(tiles is not None and tiles.back_stacked_gather)))
    num_views, num_channels, num_rows = sino_cm.shape
    num_pixels = int(idx.shape[0])
    psf_radius = pp.geometry_params.psf_radius
    gp = pp.geometry_params
    sdd = gp.source_detector_dist
    geom_consts = (float(gp.delta_det_row), float(gp.det_row_offset),
                   (num_rows - 1) / 2.0, 0.0 if np.isinf(sdd) else float(1.0 / sdd))
    l_padded = next_pow2(bands[0][1])
    out = None
    for v0 in range(0, num_views, VIEW_CHUNK):
        v1 = min(v0 + VIEW_CHUNK, num_views)
        owned = np.arange(v0, v1)
        c0 = _jit_compute_scatter_centers(pf.view_params_array, idx,
                                          model.compute_channel_coordinate, pp,
                                          pixels_major=False, owned_view_indices=owned)
        wc = _jit_hfan_weights(pf.view_params_array, idx, pp,
                               coeff_power=coeff_power, owned_view_indices=owned)
        m0, slope, wpr = _jit_vfan_scalars(pf.view_params_array, idx, pp,
                                           owned_view_indices=owned)
        if v0 == 0:
            dev = float(jnp.max(jnp.abs(slope - wpr)))
            note(f'  slope==W_p_r check: max|diff|={dev:.2e}')
        kern = _make_fused_call(v1 - v0, num_channels, num_rows, num_pixels, l_padded,
                                min(lc, l_padded), psf_radius, coeff_power, num_warps,
                                geom_consts, interpret)
        chunk_bands = []
        for g0, length in bands:
            r = kern(c0, wc, m0, wpr,
                     jnp.asarray([float(g0)], jnp.float32), sino_cm[v0:v1])
            chunk_bands.append(r[:, :length])
        chunk = jnp.concatenate(chunk_bands, axis=1)
        out = chunk if out is None else _accum(out, chunk)
    return out


# ── 3. Bench ──────────────────────────────────────────────────────────────────
def main():
    interpret = SMOKE
    if not SMOKE:
        note('day-0 probe: vector ref-gather + floor rounding on Triton')
        probe_vector_gather(interpret=False)
        note('probe PASSED')
    else:
        probe_vector_gather(interpret=True)
        note('probe (interpret) PASSED')

    views, rows, channels = SHARD_VIEWS, SINO_ROWS, SINO_CHANNELS
    angles = np.linspace(-np.pi / 2, np.pi / 2, views, endpoint=False)
    model = mbirjax.ConeBeamModel((views, rows, channels), angles,
                                  source_detector_dist=4 * channels,
                                  source_iso_dist=4 * channels)
    model.configure_devices(1)
    recon_shape = model.get_params('recon_shape')
    num_slices = recon_shape[2]
    idx = mbirjax.gen_full_indices(recon_shape, use_ror_mask=True)
    rng = np.random.default_rng(0)
    sino = jnp.asarray(rng.random((views, rows, channels), dtype=np.float32))
    sino_cm = jnp.swapaxes(sino, 1, 2)                       # (V, C, rows), no pad
    jax.block_until_ready(sino_cm)
    bands = [(g0, min(BAND_L, num_slices - g0)) for g0 in range(0, num_slices, BAND_L)]
    note(f'shapes: P={len(idx)} slices={num_slices} bands={len(bands)} '
         f'L_pad={next_pow2(BAND_L)}')

    # XLA baselines + references (both coeff powers).
    refs, base_walls = {}, {}
    model.tiles = model.tiles._replace(back_view_batch=BASELINE_VB)
    for cp in (1, 2):
        def xla_sweep():
            outs = [model.projector_functions.sparse_back_project_band(
                sino, idx, g0, min(BAND_L, num_slices), coeff_power=cp)[:, :length]
                for g0, length in bands]
            return jnp.concatenate(outs, axis=1)
        r = jax.block_until_ready(xla_sweep())
        t = time.perf_counter()
        jax.block_until_ready(xla_sweep())
        base_walls[cp] = time.perf_counter() - t
        refs[cp] = np.asarray(r)
        note(f'XLA baseline cp={cp}: {base_walls[cp]:.3f}s')

    # Fused sweep.
    results = []
    for lc in LC_SWEEP:
        for nw in WARP_SWEEP:
            row = dict(lc=lc, warps=nw)
            for cp in (1, 2):
                out = jax.block_until_ready(fused_band_sweep(
                    model, sino_cm, idx, bands, lc, nw, cp, interpret=interpret))
                rel = float(np.max(np.abs(np.asarray(out) - refs[cp]))
                            / np.max(np.abs(refs[cp])))
                tol = GRAD_TOL if cp == 1 else HESS_TOL
                ts = []
                for _ in range(TRIALS):
                    t = time.perf_counter()
                    jax.block_until_ready(fused_band_sweep(
                        model, sino_cm, idx, bands, lc, nw, cp, interpret=interpret))
                    ts.append(time.perf_counter() - t)
                wall = sorted(ts)[len(ts) // 2]
                row[f'cp{cp}'] = (wall, rel, rel < tol)
                note(f'  lc={lc} warps={nw} cp={cp}: wall={wall:.3f}s '
                     f'({base_walls[cp] / wall:.2f}x) rel={rel:.2e} '
                     f'{"PASS" if rel < tol else "FAIL"}')
            results.append(row)

    # Adjoint vs the XLA cone forward, using the best passing config.
    ok = [r for r in results if r['cp1'][2] and r['cp2'][2]]
    if ok:
        best = min(ok, key=lambda r: r['cp1'][0])
        x = jnp.asarray(rng.random((len(idx), num_slices), dtype=np.float32))
        ax = jnp.asarray(model.sparse_forward_project(x, idx))
        by = fused_band_sweep(model, sino_cm, idx, bands, best['lc'], best['warps'],
                              1, interpret=interpret)
        lhs = float(jnp.vdot(ax, sino))
        rhs = float(jnp.vdot(x, by))
        adj = abs(lhs - rhs) / max(abs(lhs), 1e-30)
        note(f'adjoint (lc={best["lc"]}, warps={best["warps"]}): rel={adj:.2e} '
             f'{"PASS" if adj < ADJ_TOL else "FAIL"}')
    print('=== e5 done ===', flush=True)


if __name__ == '__main__':
    main()
