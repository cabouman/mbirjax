"""E6: the cone fused forward kernel spike (architecture (C), panel-amended design --
gpu_headroom_findings.md "Cone forward fused kernel").

The shipped forward segment walk with ONE substitution: the per-pixel contribution
wchan * vals[pix, :] becomes wchan * resample(vals[pix, :]) -- the inverse-affine
vertical fan, tap window +-gp.bp_psf_radius (the XLA vfan's OWN window: gate-safe by
construction), per-tap cos-phi divisor at the tap slice, zero-weight slice mask with
clamped gather indices (load-bearing).

Gates (floor-calibrated per the panel: at 1024 rows two correct f32 impls differ by
~1.3e-4 max-rel; forward has no sqrt-V averaging): PASS = nrmse <= 2e-5 AND
max-rel <= 3e-4 vs the XLA cone forward; adjoint vs the XLA cone back at 1e-5
(inner products average).  Bar >= 1.5x vs XLA full-grid (19.4 s class); expect
2.2-3.9x.  Sweeps: num_warps {1, 2} (the panel's register-pressure escape hatch).

Self-contained (builders inlined) so CPU interpret smoke runs against any tree:
E6_SMOKE=1 shrinks shapes.
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

SINO_SHAPE = (1024, 1024, 1024)
SUBSET_P = 6026
FWD_SEGMENT_CAP = 64
VIEW_CHUNK = 128
WARP_SWEEP = (1, 2)
TRIALS = 3
NRMSE_TOL, MAXREL_TOL, ADJ_TOL = 2e-5, 3e-4, 1e-5
SMOKE = os.environ.get('E6_SMOKE') == '1'
if SMOKE:
    SINO_SHAPE, SUBSET_P, VIEW_CHUNK, WARP_SWEEP = (16, 24, 32), 40, 8, (1,)

T0 = time.perf_counter()


def note(msg):
    print(f'[{time.perf_counter() - T0:8.2f}s] {msg}', flush=True)


def next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


# ── Builders (inlined; mirror _pallas_kernels) ────────────────────────────────
def cone_hfan_data(pixel_indices, svp, pp):
    gp = pp.geometry_params
    n_p, W_p_c, footprint = ConeBeamModel.compute_horizontal_data(pixel_indices, svp, pp)
    return n_p, W_p_c, (gp.voxel_row_aspect * gp.delta_voxel * gp.delta_voxel) / footprint


@partial(jax.jit, static_argnames=['pp'])
def _jit_fwd_streams(view_params_array, pixel_indices, pp, owned=()):
    if len(owned) > 0:
        view_params_array = view_params_array[jnp.asarray(owned)]
    gp = pp.geometry_params
    C = pp.sinogram_shape[2]
    P = pixel_indices.shape[0]

    def one_view(svp):
        n_p, W_p_c, scale = cone_hfan_data(pixel_indices, svp, pp)
        centers = jnp.round(n_p).astype(jnp.int32)
        offs = jnp.arange(-gp.psf_radius, gp.psf_radius + 1)
        n = centers[None, :] + offs[:, None]
        L_max = jnp.minimum(1.0, W_p_c)
        A = scale * jnp.clip((W_p_c + 1.0) / 2.0 - jnp.abs(n_p - n), 0.0, L_max)
        A = A * ((n >= 0) & (n < C))
        n = jnp.clip(n, 0, C - 1)
        flat = n.reshape(-1)
        sn, order = jax.lax.sort_key_val(flat, jnp.arange(flat.shape[0]))
        return A.reshape(-1)[order], (order % P).astype(jnp.int32), \
            jnp.searchsorted(sn, jnp.arange(C + 1)).astype(jnp.int32)
    return jax.vmap(one_view)(view_params_array)


def _split_two_phase(starts, cap, n2):
    s0, s1 = starts[:-1], starts[1:]
    C = s0.shape[0]
    chan = jnp.arange(C, dtype=jnp.int32)
    seg1 = jnp.stack([s0, jnp.minimum(s0 + cap, s1), chan, jnp.zeros_like(chan)], 1)
    nseg2 = jnp.maximum(0, -(-(s1 - s0) // cap) - 1)
    cum = jnp.cumsum(nseg2)
    j = jnp.arange(n2, dtype=jnp.int32)
    ch = jnp.minimum(jnp.searchsorted(cum, j, side='right'), C - 1).astype(jnp.int32)
    k = j - (cum[ch] - nseg2[ch]) + 1
    a = s0[ch] + k * cap
    b = jnp.minimum(a + cap, s1[ch])
    valid = j < cum[-1]
    return seg1, jnp.stack([jnp.where(valid, a, 0), jnp.where(valid, b, 0),
                            jnp.where(valid, ch, C), jnp.zeros_like(j)], 1)


@partial(jax.jit, static_argnames=['pp'])
def _jit_vfan_scalars(view_params_array, pixel_indices, pp, owned=()):
    if len(owned) > 0:
        view_params_array = view_params_array[jnp.asarray(owned)]

    def one_view(svp):
        def one_pixel(pidx):
            m_p, _, W, _ = ConeBeamModel.compute_vertical_data_single_pixel(
                pidx, jnp.arange(1), svp, pp)
            return m_p[0], W if jnp.ndim(W) == 0 else W[0]
        return jax.vmap(one_pixel)(pixel_indices)
    return jax.vmap(one_view)(view_params_array)


# ── The E6 kernel ─────────────────────────────────────────────────────────────
def _e6_kernel(seg_ref, wt_ref, pix_ref, m0_ref, wpr_ref, vals_ref, *rest, atomic,
               bp, num_slices, rows_padded, delta_det_row, det_row_offset,
               det_center_row, inv_sdd):
    out_ref = rest[-1]
    v = pl.program_id(0)
    start, end, c = seg_ref[0, 0, 0], seg_ref[0, 0, 1], seg_ref[0, 0, 2]
    m_vec = jnp.arange(rows_padded).astype(jnp.float32)

    def body(i, acc):
        p = pix_ref[0, i]
        wchan = wt_ref[0, i]
        m0 = m0_ref[0, p]                                  # ref-gather at pixel id
        wpr = wpr_ref[0, p]
        l_c = jnp.floor((m_vec - m0) / wpr + 0.5).astype(jnp.int32)
        L_max = jnp.minimum(1.0, wpr)
        contrib = jnp.zeros((rows_padded,), jnp.float32)
        for tl in range(-bp, bp + 1):                      # static unroll (XLA's window)
            l = l_c + tl
            m_p = m0 + wpr * l.astype(jnp.float32)
            v_p = (m_p - det_center_row) * delta_det_row - det_row_offset
            inv_cos = jnp.sqrt(1.0 + (v_p * inv_sdd) ** 2)  # per-tap divisor, Inf-safe
            w = jnp.clip((wpr + 1.0) / 2.0 - jnp.abs(m_p - m_vec), 0.0, L_max) * inv_cos
            w = w * ((l >= 0) & (l < num_slices))          # zero-weight mask: load-bearing
            lc = jnp.clip(l, 0, num_slices - 1)
            contrib = contrib + w * vals_ref[p, lc]        # vector ref-gather
        return acc + wchan * contrib
    acc = jax.lax.fori_loop(start, end, body, jnp.zeros((rows_padded,), jnp.float32))
    if atomic:
        @pl.when(end > start)
        def _add():
            pltriton.atomic_add(out_ref, (v, c, slice(None)), acc)
    else:
        out_ref[v, c, :] = acc


@functools.cache
def _make_e6_phase(vc, C, n_seg, taps, P, num_slices, rows_padded, atomic, bp,
                   geom_consts, num_warps, interpret=False):
    kw = ({} if interpret else
          {'compiler_params': pltriton.CompilerParams(num_warps=num_warps)})
    ddr, droff, dcen, isdd = geom_consts
    specs = [pl.BlockSpec((1, 1, 4), lambda v, s: (v, s, 0)),
             pl.BlockSpec((1, taps), lambda v, s: (v, 0)),
             pl.BlockSpec((1, taps), lambda v, s: (v, 0)),
             pl.BlockSpec((1, P), lambda v, s: (v, 0)),            # m0 (per view!)
             pl.BlockSpec((1, P), lambda v, s: (v, 0)),            # W_p_r
             pl.BlockSpec((P, num_slices), lambda v, s: (0, 0))]   # shared x tile
    alias = {}
    if atomic:
        specs.append(pl.BlockSpec((vc, C + 1, rows_padded), lambda v, s: (0, 0, 0)))
        alias = {6: 0}
    return pl.pallas_call(
        partial(_e6_kernel, atomic=atomic, bp=bp, num_slices=num_slices,
                rows_padded=rows_padded, delta_det_row=ddr, det_row_offset=droff,
                det_center_row=dcen, inv_sdd=isdd),
        out_shape=jax.ShapeDtypeStruct((vc, C + 1, rows_padded), jnp.float32),
        grid=(vc, n_seg), in_specs=specs,
        out_specs=pl.BlockSpec((vc, C + 1, rows_padded), lambda v, s: (0, 0, 0)),
        input_output_aliases=alias, interpret=interpret, **kw)


@functools.cache
def _make_e6_chunk_fn(pp, vc, C, n2, P, num_slices, rows, rows_padded, bp,
                      geom_consts, num_warps, interpret):
    taps = (2 * pp.geometry_params.psf_radius + 1) * P
    p1 = _make_e6_phase(vc, C, C, taps, P, num_slices, rows_padded, False, bp,
                        geom_consts, num_warps, interpret)
    p2 = _make_e6_phase(vc, C, n2, taps, P, num_slices, rows_padded, True, bp,
                        geom_consts, num_warps, interpret)

    def chunk_fn(view_params_array, pixel_indices, vals, owned):
        wts, pix, starts = _jit_fwd_streams(view_params_array, pixel_indices, pp,
                                            owned=owned)
        seg1, seg2 = jax.vmap(lambda s: _split_two_phase(s, FWD_SEGMENT_CAP, n2))(starts)
        m0, wpr = _jit_vfan_scalars(view_params_array, pixel_indices, pp, owned=owned)
        out = p2(seg2, wts, pix, m0, wpr, vals, p1(seg1, wts, pix, m0, wpr, vals))
        # Padded rows carry REAL garbage (the inverse affine doesn't know padding);
        # the trim here is load-bearing -- same contract note as the cone back driver.
        return jnp.swapaxes(out[:, :C, :rows], 1, 2)
    return jax.jit(chunk_fn)


def get_pp(model):
    pp = getattr(model.projector_functions, 'projector_params', None)
    if pp is None:                    # local smoke resolves the MAIN-tree mbirjax
        from mbirjax.projectors import ProjectorParams
        tiles = model.tiles
        pp = ProjectorParams(model.get_params('sinogram_shape'),
                             model.get_params('recon_shape'),
                             model.get_geometry_parameters(),
                             int(bool(tiles is not None and tiles.sort_by_channel)),
                             int(bool(tiles is not None and tiles.back_stacked_gather)))
    return pp


def e6_forward(model, values, idx, num_warps, interpret=False):
    pf = model.projector_functions
    pp = get_pp(model)
    C = pp.sinogram_shape[2]
    P, num_slices = values.shape
    gp = pp.geometry_params
    rows = pp.sinogram_shape[1]
    rows_padded = next_pow2(rows)
    sdd = gp.source_detector_dist
    geom_consts = (float(gp.delta_det_row), float(gp.det_row_offset),
                   (rows - 1) / 2.0, 0.0 if np.isinf(sdd) else float(1.0 / sdd))
    bp = int(gp.bp_psf_radius)
    num_views = pf.view_params_array.shape[0]
    taps = (2 * gp.psf_radius + 1) * P
    n2 = max(1, taps // FWD_SEGMENT_CAP)
    chunks = []
    for v0 in range(0, num_views, VIEW_CHUNK):
        owned = np.arange(v0, min(v0 + VIEW_CHUNK, num_views))
        fn = _make_e6_chunk_fn(pp, len(owned), C, n2, P, num_slices, rows,
                               rows_padded, bp, geom_consts, num_warps, interpret)
        chunks.append(fn(pf.view_params_array, idx, values, owned))
    return chunks[0] if len(chunks) == 1 else jnp.concatenate(chunks, axis=0)


def main():
    interpret = SMOKE
    views, rows, channels = SINO_SHAPE
    angles = np.linspace(-np.pi / 2, np.pi / 2, views, endpoint=False)
    model = mbirjax.ConeBeamModel(SINO_SHAPE, angles,
                                  source_detector_dist=4 * channels,
                                  source_iso_dist=4 * channels)
    model.configure_devices(1)
    recon_shape = model.get_params('recon_shape')
    idx_full = mbirjax.gen_full_indices(recon_shape, use_ror_mask=True)
    rng = np.random.default_rng(0)
    gp = get_pp(model).geometry_params
    note(f'P_full={len(idx_full)} slices={recon_shape[2]} bp={int(gp.bp_psf_radius)}')

    for tag, idx in (('subset', jnp.asarray(np.sort(rng.choice(
            np.asarray(idx_full), size=min(SUBSET_P, len(idx_full) // 2),
            replace=False)))), ('full', idx_full)):
        vals = jnp.asarray(rng.random((len(idx), recon_shape[2]), dtype=np.float32))
        ref = jax.block_until_ready(
            model.projector_functions.sparse_forward_project(vals, idx))
        t = time.perf_counter()
        jax.block_until_ready(model.projector_functions.sparse_forward_project(vals, idx))
        base = time.perf_counter() - t
        note(f'XLA {tag}: {base:.3f}s')
        refn = np.asarray(ref)
        for nw in WARP_SWEEP:
            out = jax.block_until_ready(e6_forward(model, vals, idx, nw,
                                                   interpret=interpret))
            o = np.asarray(out)
            maxrel = float(np.max(np.abs(o - refn)) / np.max(np.abs(refn)))
            nrmse = float(np.linalg.norm(o - refn) / np.linalg.norm(refn))
            ts = []
            for _ in range(TRIALS):
                t = time.perf_counter()
                jax.block_until_ready(e6_forward(model, vals, idx, nw,
                                                 interpret=interpret))
                ts.append(time.perf_counter() - t)
            wall = sorted(ts)[len(ts) // 2]
            ok = nrmse <= NRMSE_TOL and maxrel <= MAXREL_TOL
            note(f'  E6 {tag} warps={nw}: wall={wall:.3f}s ({base / wall:.2f}x) '
                 f'nrmse={nrmse:.2e} maxrel={maxrel:.2e} {"PASS" if ok else "FAIL"}')

    # Adjoint: <A x, y> vs <x, B y> with B = the XLA cone back.
    x = jnp.asarray(rng.random((len(idx_full), recon_shape[2]), dtype=np.float32))
    y = jnp.asarray(rng.random(SINO_SHAPE, dtype=np.float32))
    ax = e6_forward(model, x, idx_full, WARP_SWEEP[0], interpret=interpret)
    if hasattr(model.tiles, 'back_pallas'):        # absent in the main-tree smoke
        model.tiles = model.tiles._replace(back_pallas=False)
    by = model.sparse_back_project(y, idx_full)
    lhs, rhs = float(jnp.vdot(ax, y)), float(jnp.vdot(x, jnp.asarray(by)))
    adj = abs(lhs - rhs) / max(abs(lhs), 1e-30)
    note(f'adjoint: rel={adj:.2e} {"PASS" if adj < ADJ_TOL else "FAIL"}')
    print('=== e6 done ===', flush=True)


if __name__ == '__main__':
    main()
