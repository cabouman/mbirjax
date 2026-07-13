"""Pallas (GPU) projector kernels — the custom-kernel path from the 2026-07 GPU-headroom
campaign (design: plans/projector_kernels/e4_integration_design.md; measured record:
plans/projector_kernels/gpu_headroom_findings.md; reproducible benches:
plans/experiments/projector_kernels/e3_*.py, e4_back_composed.py).

Increment 1: the PARALLEL-BEAM single-device BACK projection (16-26x kernel-level,
9.17x composed/gated at the 1024^3 cell; Hessian path the same).
Increment 2: the PARALLEL-BEAM forward horizontal fan for SUBSET-SIZED calls (the VCD
fine tail; 2.13x measured) -- the CSR segment-walk with the two-phase store+atomic
launch.  Full-grid forward stays on the XLA sorted reduce by policy: its measured win
is smaller (1.59x) and the python-loop driver overhead at ~3000 pixel batches would
erode it; a batched-grid variant can revisit this later.

How the kernel works (the "register-tile + L2-phase" design)
-------------------------------------------------------------
Back projection is, per pixel p and detector row r:

    out[p, r] = sum over views v and psf taps t of  A[v,t,p] * sino[v, center[v,p]+t, r]

The XLA path evaluates this per-view and sums; its cost is transaction-bound row
gathers.  This kernel instead runs ONE small GPU program per (row-chunk, pixel):

  * the program holds out[p, r0:r0+RC] in REGISTERS and loops over ALL views x taps,
    so the view sum never touches memory (the "register tile");
  * the row-chunk grid dimension is SLOWEST, so every concurrently-running program
    gathers from the same (views, channels, RC) slice of the channel-major sinogram --
    ~130 MB at the production shape, mostly L2-resident (the "L2 phase");
  * work is perfectly uniform (every pixel has exactly psf_width taps): no sort, no
    atomics, each output cell written exactly once.

Weights are the SAME trapezoid formula as the XLA kernels (adjointness is preserved by
construction; only the summation ORDER differs, so results agree to float reordering
noise and are gated at the standard relative tolerance).  The integer channel centers
are the existing concrete-centers arrays (the rounding-bug contract carries over).

Updating / retiring this path
-----------------------------
Constants (ROW_CHUNK, NUM_WARPS, the arch allowlist) come from the bench scripts named
above -- rerun those to revalidate on a new architecture or jax version, then extend
_ARCH_ALLOWLIST.  `is_available()` probes an actual tiny-kernel compile once per
process, so an incompatible toolchain falls back to the XLA path silently.  The env
variable MBIRJAX_DISABLE_PALLAS=1 forces the XLA path everywhere (the escape hatch).
"""
import functools
import os
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import triton as pltriton

# Measured on H100 (e3_back_pallas_v1.py sweep): rc=256 dominates (fewer per-chunk
# weight/center re-reads); num_warps flat between 1 and 2, 2 marginally better at the
# subset shape.  Revalidate per arch before extending the allowlist.
ROW_CHUNK = 256
NUM_WARPS = 2
# Forward (e3_hfan_pallas_v3.py sweep): segment cap 64 balances the tap walk against
# split-segment atomics at subset uniformity; num_warps=1 (per-program work is one
# warp's worth).  The subset-size gate: pallas forward only when the call fits ONE
# pixel batch (the measured 2.13x shape); larger calls keep the XLA sorted reduce.
FWD_SEGMENT_CAP = 64
FWD_NUM_WARPS = 1
# Device kinds where the kernels have been measured (substring match on device_kind).
_ARCH_ALLOWLIST = ('H100',)


def next_pow2(n):
    """Smallest power of two >= n (the Triton backend requires power-of-2 in-kernel
    vector shapes; only the ROW axis is padded -- see the design doc's padding note)."""
    p = 1
    while p < n:
        p *= 2
    return p


@functools.cache
def is_available():
    """True when the pallas back path may be used: enabled, GPU, measured arch, and a
    probe kernel actually compiles (a toolchain-drift guard).  Cached per process."""
    if os.environ.get('MBIRJAX_DISABLE_PALLAS', '0') == '1':
        return False
    try:
        dev = jax.devices()[0]
        if dev.platform != 'gpu':
            return False
        if not any(a in getattr(dev, 'device_kind', '') for a in _ARCH_ALLOWLIST):
            return False

        def probe_kernel(x_ref, o_ref):
            o_ref[...] = x_ref[...] * 2.0

        probe = pl.pallas_call(
            probe_kernel,
            out_shape=jax.ShapeDtypeStruct((32,), jnp.float32),
            compiler_params=pltriton.CompilerParams(num_warps=1))
        out = jax.jit(probe)(jnp.ones((32,), jnp.float32))
        return bool(jax.block_until_ready(out)[0] == 2.0)
    except Exception:
        return False


@partial(jax.jit, static_argnames=['hfan_data_fn', 'projector_params', 'coeff_power'])
def _jit_compute_back_weights(view_params_array, pixel_indices, hfan_data_fn,
                              projector_params, coeff_power=1, owned_view_indices=()):
    """Per-(view, tap, pixel) trapezoid weights A (V, T, P) f32, zeroed out of range,
    raised to coeff_power (2 = the Hessian diagonal).  MODULE-LEVEL jit -- traced once
    (a per-call jit would re-trace every call and cost ~200 ms of host time; see
    lessons.md section 5).  hfan_data_fn is the geometry's float chain (e.g.
    ParallelBeamModel.compute_hfan_data) so the weights match the XLA kernels' exactly.
    """
    if len(owned_view_indices) > 0:
        view_params_array = view_params_array[jnp.asarray(owned_view_indices)]
    gp = projector_params.geometry_params
    num_channels = projector_params.sinogram_shape[2]

    def one_view(single_view_params):
        n_p, W_p_c, weight_scale = hfan_data_fn(pixel_indices, single_view_params,
                                                projector_params)
        centers = jnp.round(n_p).astype(jnp.int32)     # == the concrete-centers values
        offs = jnp.arange(-gp.psf_radius, gp.psf_radius + 1)
        n = centers[None, :] + offs[:, None]           # (T, P)
        L_max = jnp.minimum(1.0, W_p_c)
        A = weight_scale * jnp.clip((W_p_c + 1.0) / 2.0 - jnp.abs(n_p - n), 0.0, L_max)
        A = A * ((n >= 0) & (n < num_channels))
        return A ** coeff_power
    return jax.vmap(one_view)(view_params_array)


def _back_kernel(centers_ref, w_ref, sino_ref, out_ref, *, rc, num_views, num_channels,
                 psf_radius, psf_width):
    """One program per (row-chunk, pixel); see the module docstring."""
    def vbody(v, acc):
        c0 = centers_ref[v, 0]
        for t in range(psf_width):                     # static unroll (psf_width taps)
            cc = jnp.clip(c0 + (t - psf_radius), 0, num_channels - 1)  # weights 0 OOR
            acc = acc + w_ref[v, t, 0] * sino_ref[v, cc, :]
        return acc
    acc = jax.lax.fori_loop(0, num_views, vbody, jnp.zeros((rc,), jnp.float32))
    out_ref[0, :] = acc


@functools.cache
def _make_back_call(num_views, num_channels, num_pixels, rows_padded, psf_radius,
                    interpret=False):
    # Cached on the static shape tuple: the returned pallas_call object must be REUSED
    # across driver calls, or every call would rebuild (and re-lower) the kernel -- the
    # same per-call-construction trap as the jit-retrace lesson (lessons.md section 5).
    psf_width = 2 * psf_radius + 1
    # Small problems: the measured ROW_CHUNK can exceed the (padded) row count -- clamp,
    # or the grid's row dimension would be zero and the kernel would never run.
    rc = min(ROW_CHUNK, rows_padded)
    kw = ({} if interpret else
          {'compiler_params': pltriton.CompilerParams(num_warps=NUM_WARPS)})
    return pl.pallas_call(
        partial(_back_kernel, rc=rc, num_views=num_views,
                num_channels=num_channels, psf_radius=psf_radius, psf_width=psf_width),
        out_shape=jax.ShapeDtypeStruct((num_pixels, rows_padded), jnp.float32),
        grid=(rows_padded // rc, num_pixels),          # row-chunk SLOWEST: the L2 phase
        in_specs=[
            pl.BlockSpec((num_views, 1), lambda r, p: (0, p)),
            pl.BlockSpec((num_views, psf_width, 1), lambda r, p: (0, 0, p)),
            pl.BlockSpec((num_views, num_channels, rc), lambda r, p: (0, 0, r)),
        ],
        out_specs=pl.BlockSpec((1, rc), lambda r, p: (p, r)),
        interpret=interpret, **kw)


# Channel-major + row-padded view chunk; jitted once (shapes static per chunk size).
@partial(jax.jit, static_argnames=['rows_padded'])
def _to_channel_major(sino_chunk, rows_padded):
    rows = sino_chunk.shape[1]
    return jnp.pad(jnp.swapaxes(sino_chunk, 1, 2), ((0, 0), (0, 0),
                                                    (0, rows_padded - rows)))


_accumulate = jax.jit(lambda a, b: a + b, donate_argnums=0)


def back_project_single_device(model, sinogram, pixel_indices, coeff_power=1,
                               output_device=None, interpret=False):
    """The pallas single-device back-projection driver (the composed-prototype
    structure, productized): view-chunk loop x [centers, weights, channel-major layout,
    ONE kernel grid over ALL pixels], accumulating across chunks.  No pixel scan, no
    transfer chunking -- the kernel has no scan carry, so the whole pixel set goes in
    one grid.  ``interpret=True`` runs the kernel in pallas interpret mode (CPU-capable;
    used by the correctness tests -- the selection policy never routes here on CPU).
    """
    from mbirjax.projectors import _jit_compute_scatter_centers   # lazy: avoids an
    # import cycle at package init (tomography_model -> this module -> projectors).
    pf = model.projector_functions
    pp = pf.projector_params
    sinogram = model._shard_sinogram(sinogram)
    num_views, rows, num_channels = sinogram.shape
    rows_padded = next_pow2(rows)
    psf_radius = pp.geometry_params.psf_radius
    num_pixels = int(pixel_indices.shape[0])
    pixel_indices = jax.device_put(pixel_indices, model.sino_placement.devices[0])

    view_chunk = min(model.tiles.back_view_batch, num_views)
    kern = _make_back_call(view_chunk, num_channels, num_pixels, rows_padded,
                           psf_radius, interpret=interpret)

    out = None
    for v0 in range(0, num_views, view_chunk):
        v1 = min(v0 + view_chunk, num_views)
        if v1 - v0 < view_chunk:
            # Ragged tail chunk: fall through with a chunk-sized kernel of its own.
            kern_tail = _make_back_call(v1 - v0, num_channels, num_pixels, rows_padded,
                                        psf_radius, interpret=interpret)
        owned = jnp.arange(v0, v1)
        centers = _jit_compute_scatter_centers(
            pf.view_params_array, pixel_indices, model.compute_channel_coordinate, pp,
            pixels_major=False, owned_view_indices=owned)
        weights = _jit_compute_back_weights(
            pf.view_params_array, pixel_indices, model.compute_hfan_data, pp,
            coeff_power=coeff_power, owned_view_indices=owned)
        sino_cm = _to_channel_major(sinogram[v0:v1], rows_padded=rows_padded)
        k = kern if (v1 - v0) == view_chunk else kern_tail
        chunk_out = k(centers, weights, sino_cm)
        out = chunk_out if out is None else _accumulate(out, chunk_out)

    out = out[:, :rows]
    if output_device is not None:
        out = jax.device_put(out, output_device)
    return out



# ══════════════════════════════════════════════════════════════════════════════
# Increment 2: the forward horizontal fan (subset-sized calls, parallel beam)
#
# out[c, :] = sum over taps (t, p) with center[p]+t == c of A[t,p] * values[p, :]
#
# The scatter becomes a GATHER over a channel-sorted contributor stream built per
# (view-chunk) by a module-level jit (sort + searchsorted -- the same ~2-3% sort cost
# the XLA path pays, paid once per call here), plus a host-side cap-and-split that
# bounds every program's segment (skew guard).  Two launches (the "two-phase" variant,
# the measured subset winner): phase 1 STORES every channel's first segment -- each
# output row written exactly once, no zero-fill, no cross-program race; phase 2
# atomically adds the few remainder segments of over-cap channels (~empty at subset
# uniformity).  The values tile (P x band) is shared by ALL views -- L2-hot, which is
# where the win comes from.
# ══════════════════════════════════════════════════════════════════════════════

@partial(jax.jit, static_argnames=['hfan_data_fn', 'projector_params'])
def _jit_compute_fwd_streams(view_params_array, pixel_indices, hfan_data_fn,
                             projector_params, owned_view_indices=()):
    """Channel-sorted (weights, pixel-ids) streams + per-channel segment starts, per
    view: wts (V, T*P) f32, pix (V, T*P) i32, starts (V, C+1) i32.  Module-level jit
    (traced once -- the retrace lesson, lessons.md section 5)."""
    if len(owned_view_indices) > 0:
        view_params_array = view_params_array[jnp.asarray(owned_view_indices)]
    gp = projector_params.geometry_params
    num_channels = projector_params.sinogram_shape[2]
    num_pixels = pixel_indices.shape[0]

    def one_view(single_view_params):
        n_p, W_p_c, weight_scale = hfan_data_fn(pixel_indices, single_view_params,
                                                projector_params)
        centers = jnp.round(n_p).astype(jnp.int32)
        offs = jnp.arange(-gp.psf_radius, gp.psf_radius + 1)
        n = centers[None, :] + offs[:, None]
        L_max = jnp.minimum(1.0, W_p_c)
        A = weight_scale * jnp.clip((W_p_c + 1.0) / 2.0 - jnp.abs(n_p - n), 0.0, L_max)
        A = A * ((n >= 0) & (n < num_channels))
        n = jnp.clip(n, 0, num_channels - 1)
        flat_n = n.reshape(-1)
        # sort_key_val returns keys and permutation TOGETHER (the rounding-hazard-safe
        # form, as in projectors._channel_reduce_sort_segsum).
        sorted_n, order = jax.lax.sort_key_val(flat_n, jnp.arange(flat_n.shape[0]))
        pix = (order % num_pixels).astype(jnp.int32)
        wts = A.reshape(-1)[order]
        starts = jnp.searchsorted(sorted_n, jnp.arange(num_channels + 1)).astype(jnp.int32)
        return wts, pix, starts
    return jax.vmap(one_view)(view_params_array)


def _split_two_phase_np(starts_np, cap, num_channels):
    """Vectorized host-side cap-and-split (numpy; ~ms at subset sizes): phase 1 = the
    FIRST segment of every channel (empty channels store zero -- the no-zero-fill
    trick); phase 2 = remainder segments of over-cap channels, padded per view to a
    common count (pads target the scratch channel row C: atomic add of zero)."""
    V = starts_np.shape[0]
    s0, s1 = starts_np[:, :-1], starts_np[:, 1:]                       # (V, C)
    seg1 = np.zeros((V, num_channels, 4), dtype=np.int32)
    seg1[:, :, 0] = s0
    seg1[:, :, 1] = np.minimum(s0 + cap, s1)
    seg1[:, :, 2] = np.arange(num_channels)[None, :]

    counts = s1 - s0
    nseg2 = np.maximum(0, -(-counts // cap) - 1)                       # (V, C)
    n2 = max(1, int(nseg2.sum(axis=1).max()))
    seg2 = np.zeros((V, n2, 4), dtype=np.int32)
    seg2[:, :, 2] = num_channels                                       # pad -> scratch row
    for v in range(V):                                                 # V-loop of vector ops
        ch = np.repeat(np.arange(num_channels), nseg2[v])
        if len(ch) == 0:
            continue
        # k-th remainder segment of its channel: 1-based within-channel index
        k = np.arange(len(ch)) - np.repeat(np.cumsum(nseg2[v]) - nseg2[v], nseg2[v]) + 1
        a = s0[v, ch] + k * cap
        seg2[v, :len(ch), 0] = a
        seg2[v, :len(ch), 1] = np.minimum(a + cap, s1[v, ch])
        seg2[v, :len(ch), 2] = ch
    return jnp.asarray(seg1), jnp.asarray(seg2), n2


def _fwd_kernel(seg_ref, wt_ref, pix_ref, vals_ref, *rest, atomic):
    """One program per (view, segment): walk the sorted contributor segment,
    ref-gather pixel rows from the shared values tile, one store (phase 1) or one
    atomic add (phase 2)."""
    out_ref = rest[-1]
    v = pl.program_id(0)
    start = seg_ref[0, 0, 0]
    end = seg_ref[0, 0, 1]
    c = seg_ref[0, 0, 2]

    def body(i, acc):
        return acc + wt_ref[0, i] * vals_ref[pix_ref[0, i], :]
    acc = jax.lax.fori_loop(start, end, body,
                            jnp.zeros((out_ref.shape[-1],), jnp.float32))
    if atomic:
        pltriton.atomic_add(out_ref, (v, c, slice(None)), acc)
    else:
        out_ref[v, c, :] = acc


@functools.cache
def _make_fwd_phase(num_views, num_channels, n_seg, taps, num_pixels, band_padded,
                    atomic, interpret=False):
    # Cached on the static shapes -- see _make_back_call's caching note.
    kw = ({} if interpret else
          {'compiler_params': pltriton.CompilerParams(num_warps=FWD_NUM_WARPS)})
    specs = [pl.BlockSpec((1, 1, 4), lambda v, s: (v, s, 0)),          # segment row
             pl.BlockSpec((1, taps), lambda v, s: (v, 0)),             # weights stream
             pl.BlockSpec((1, taps), lambda v, s: (v, 0)),             # pixel-id stream
             pl.BlockSpec((num_pixels, band_padded), lambda v, s: (0, 0))]  # shared tile
    alias = {}
    if atomic:
        specs.append(pl.BlockSpec((num_views, num_channels + 1, band_padded),
                                  lambda v, s: (0, 0, 0)))
        alias = {4: 0}                        # phase-1 result accumulated in place
    return pl.pallas_call(
        partial(_fwd_kernel, atomic=atomic),
        out_shape=jax.ShapeDtypeStruct((num_views, num_channels + 1, band_padded),
                                       jnp.float32),
        grid=(num_views, n_seg),
        in_specs=specs,
        out_specs=pl.BlockSpec((num_views, num_channels + 1, band_padded),
                               lambda v, s: (0, 0, 0)),
        input_output_aliases=alias,
        interpret=interpret, **kw)


def forward_project_subset(model, voxel_values, pixel_indices, owned_view_indices=(),
                           interpret=False):
    """The pallas forward driver for SUBSET-SIZED calls (one pixel batch): view-chunk
    loop x [streams, host cap-split, two-phase kernel], outputs concatenated to the
    library orientation (views, band_rows, channels).  values (P, band) is the shared
    gather tile; band is whatever the caller passes (a slice band or full cylinders).
    """
    pf = model.projector_functions
    pp = pf.projector_params
    num_channels = pp.sinogram_shape[2]
    num_pixels, band = voxel_values.shape
    band_padded = next_pow2(band)
    vals_pad = jnp.pad(voxel_values, ((0, 0), (0, band_padded - band)))

    if len(owned_view_indices) > 0:
        owned_all = jnp.asarray(owned_view_indices)
    else:
        owned_all = jnp.arange(pf.view_params_array.shape[0])
    num_views = int(owned_all.shape[0])
    view_chunk = min(model.tiles.fwd_view_batch, num_views)

    chunks = []
    for v0 in range(0, num_views, view_chunk):
        owned = owned_all[v0:min(v0 + view_chunk, num_views)]
        vc = int(owned.shape[0])
        wts, pix, starts = _jit_compute_fwd_streams(
            pf.view_params_array, pixel_indices, model.compute_hfan_data, pp,
            owned_view_indices=owned)
        seg1, seg2, n2 = _split_two_phase_np(np.asarray(starts), FWD_SEGMENT_CAP,
                                             num_channels)
        taps = int(wts.shape[1])
        p1 = _make_fwd_phase(vc, num_channels, num_channels, taps, num_pixels,
                             band_padded, atomic=False, interpret=interpret)
        p2 = _make_fwd_phase(vc, num_channels, n2, taps, num_pixels,
                             band_padded, atomic=True, interpret=interpret)
        out = p2(seg2, wts, pix, vals_pad, p1(seg1, wts, pix, vals_pad))
        chunks.append(out[:, :num_channels, :band])       # drop scratch row + padding
    full = chunks[0] if len(chunks) == 1 else jnp.concatenate(chunks, axis=0)
    return jnp.swapaxes(full, 1, 2)                       # -> (views, band_rows, channels)
