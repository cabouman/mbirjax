"""Pallas (GPU) projector kernels — the custom-kernel path from the 2026-07 GPU-headroom
campaign (design: plans/projector_kernels/e4_integration_design.md; measured record:
plans/projector_kernels/gpu_headroom_findings.md; reproducible benches:
plans/experiments/projector_kernels/e3_*.py, e4_back_composed.py).

What lives here, and where it is dispatched from
------------------------------------------------
* BACK projection (gradient and Hessian), single device: ~9x the XLA path at the
  1024^3 cell, all pixel-batch sizes.  Dispatched from
  ``TomographyModel._sparse_back_project_single_device`` (reached only through the
  GPU n=1 short-circuit -- multi-device recons use the banded XLA path) when
  ``model.tiles.back_pallas`` is set.
* FORWARD projection, single device, ALL pixel counts: ~2.6x at the 6,026-pixel
  fine-tail cell, 3.2-3.8x at full grid.  Dispatched from
  ``projectors.sparse_forward_project_public`` when ``model.tiles.fwd_pallas`` is
  set.  There is deliberately NO pixel-count guard: the 2026-07-13 P x band sweep
  (plans/projector_kernels/fwd_guard_sweep.md, 70 value-gated cells) measured pallas
  faster at EVERY point with no crossover -- past L2 the kernel streams at near the
  HBM traffic bound while XLA pays the same traffic plus its sort/scatter constant
  factor.  Do not reintroduce a size guard without new measurements.
Both flags are set in ``ParallelBeamModel._select_tile_policy`` (GPU only), gated by
``availability()`` below.  Inspect a live model with ``model.get_compute_config()``.

How the back kernel works (the "register-tile + L2-phase" design)
-----------------------------------------------------------------
Back projection is, per pixel p and detector row r:

    out[p, r] = sum over views v and psf taps t of  A[v,t,p] * sino[v, center[v,p]+t, r]

The XLA path evaluates this per-view and sums; its cost is transaction-bound row
gathers.  This kernel instead runs ONE small GPU program per (row-chunk, pixel):

  * the program holds out[p, r0:r0+RC] in REGISTERS and loops over ALL views x taps,
    so the view sum never touches memory (the "register tile") -- the main thing the
    whole-array XLA model cannot express;
  * the row-chunk grid dimension is SLOWEST, so every concurrently-running program
    gathers from the same (views, channels, RC) slice of the channel-major sinogram --
    mostly L2-resident (the "L2 phase");
  * work is perfectly uniform (every pixel has exactly psf_width taps): no sort, no
    atomics, each output cell written exactly once.

The forward kernel is described at its own section divider below.

Weights are the SAME trapezoid formula as the XLA kernels (adjointness is preserved by
construction; only the summation ORDER differs, so results agree to float reordering
noise and are gated at the standard relative tolerance).  The integer channel centers
are the existing concrete-centers arrays (the rounding-bug contract carries over).

Constraints that must hold (violations measured as large regressions)
---------------------------------------------------------------------
* Every launch/grid/block shape derives from ARRAY SHAPES only, never from data.  A
  data-dependent shape changes a cache key per VCD subset and triggers a Triton
  recompile inside the recon loop.
* No host<->device synchronization in any per-call path (no ``np.asarray`` /
  ``device_get`` of a device array): one sync per view chunk stalls the pipeline and
  flips the forward kernel's win into a loss.
* ``pl.pallas_call`` objects and jitted wrappers are constructed ONCE per static
  shape (functools.cache) -- per-call construction re-lowers/re-traces every call.
* In-kernel gather of a block-LOADED array does not lower on either pallas backend at
  the pinned jax version; gathers must be REF-LEVEL indexing (``ref[idx, :]``), which
  is what every kernel here does.  Whole-array BlockSpecs are pointer refs on the
  Triton backend, not materialized copies (interpret mode does materialize them --
  acceptable for tests only).

Updating / retiring this path
-----------------------------
Constants (ROW_CHUNK, NUM_WARPS, FWD_SEGMENT_CAP, the arch allowlist) come from the
bench scripts named above -- rerun those to revalidate on a new architecture or jax
version, then extend _ARCH_ALLOWLIST.  ``availability()`` probes an actual tiny-kernel
compile once per process, so an incompatible toolchain falls back to the XLA path
silently.  The env variable MBIRJAX_DISABLE_PALLAS=1 forces the XLA path everywhere
(the escape hatch).  Retiring a kernel = removing its TilePolicy flag line; every call
site keeps the XLA fallback compiled-in.
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
# Cap on the back driver's view chunk, independent of the (XLA-tuned) TilePolicy
# back_view_batch: the per-chunk weights transient is chunk*T*P*4 B, which at the
# SHARDED policy's view batch of 512 reaches ~2.4x the sino shard per owner
# (T*pi/4, size- and n-independent -- the w2_inc3 gate measured +5.4 GB at 1024^3
# n=2).  Chunking itself costs ~nothing (increment 1 gated 9.17x running 8 chunks),
# so 128 holds the transient at <= 0.6x shard at production shapes.
BACK_VIEW_CHUNK_CAP = 128
# Forward (e3_hfan_pallas_v3.py sweep): segment cap 64 balances the tap walk against
# split-segment atomics at subset uniformity; num_warps=1 (per-program work is one
# warp's worth).  All pixel counts route here (no size guard -- see the module
# docstring and fwd_guard_sweep.md).
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
def availability():
    """(usable, reason): whether the pallas paths may be used, and why not when they
    can't -- enabled, GPU, measured arch, and a probe kernel actually compiles (a
    toolchain-drift guard).  Cached per process.  The reason string is user-facing
    (surfaced by ``TomographyModel.get_compute_config``): the fallback to XLA is
    silent at run time by design, so this is where a benchmark learns WHY a node is
    not using the custom kernels."""
    if os.environ.get('MBIRJAX_DISABLE_PALLAS', '0') == '1':
        return False, 'disabled by MBIRJAX_DISABLE_PALLAS=1'
    try:
        dev = jax.devices()[0]
        if dev.platform != 'gpu':
            return False, 'not a GPU platform ({})'.format(dev.platform)
        kind = getattr(dev, 'device_kind', '')
        if not any(a in kind for a in _ARCH_ALLOWLIST):
            return False, 'device kind {!r} not in the measured allowlist {}'.format(
                kind, _ARCH_ALLOWLIST)

        def probe_kernel(x_ref, o_ref):
            o_ref[...] = x_ref[...] * 2.0

        probe = pl.pallas_call(
            probe_kernel,
            out_shape=jax.ShapeDtypeStruct((32,), jnp.float32),
            compiler_params=pltriton.CompilerParams(num_warps=1))
        out = jax.jit(probe)(jnp.ones((32,), jnp.float32))
        if bool(jax.block_until_ready(out)[0] == 2.0):
            return True, 'available ({})'.format(kind)
        return False, 'probe kernel returned a wrong value'
    except Exception as e:
        return False, 'probe kernel failed to compile/run: {}'.format(e)


def is_available():
    """True when the pallas kernel paths may be used; see ``availability``."""
    return availability()[0]


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
    """One program per (row-chunk, pixel); see the module docstring.

    The refs are the per-program BLOCKS declared in _make_back_call: this program's
    pixel column of centers/weights, the full (V, C, rc) sinogram row-chunk slice, and
    its own (1, rc) output row.  ``acc`` lives in registers for the whole view loop --
    the kernel's entire point -- so rc (x psf_width live sino reads) must stay small
    enough to avoid register spills; revalidate ROW_CHUNK if psf_width grows.
    """
    def vbody(v, acc):
        c0 = centers_ref[v, 0]
        for t in range(psf_width):                     # static unroll (psf_width taps)
            # Out-of-range taps have weight EXACTLY zero (the weight builder masks
            # them), so clamping the channel index only redirects reads that
            # contribute nothing -- it exists to keep the ref access in bounds.
            cc = jnp.clip(c0 + (t - psf_radius), 0, num_channels - 1)
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
    # BlockSpec semantics (the recurring trap): the index_map returns BLOCK indices,
    # in units of the block shape -- (0, p) with block (num_views, 1) means "all
    # views, pixel column p", NOT an element offset.  On the Triton backend a block is
    # a POINTER into the operand (loads happen only at ref-indexing sites in the
    # kernel), so the (num_views, num_channels, rc) sinogram block below is not a
    # copy; concurrently-running programs share it through L2.  The grid is iterated
    # LAST-dimension-fastest, so putting the row-chunk FIRST makes it slowest -- all
    # pixels of one row-chunk run before the next chunk begins (the L2 phase).
    return pl.pallas_call(
        partial(_back_kernel, rc=rc, num_views=num_views,
                num_channels=num_channels, psf_radius=psf_radius, psf_width=psf_width),
        out_shape=jax.ShapeDtypeStruct((num_pixels, rows_padded), jnp.float32),
        grid=(rows_padded // rc, num_pixels),          # row-chunk SLOWEST: the L2 phase
        in_specs=[
            pl.BlockSpec((num_views, 1), lambda r, p: (0, p)),             # centers col
            pl.BlockSpec((num_views, psf_width, 1), lambda r, p: (0, 0, p)),  # weights col
            pl.BlockSpec((num_views, num_channels, rc), lambda r, p: (0, 0, r)),  # sino chunk
        ],
        out_specs=pl.BlockSpec((1, rc), lambda r, p: (p, r)),
        interpret=interpret, **kw)


# Channel-major + row-padded view chunk; jitted once (shapes static per chunk size).
# The kernel reads sino[v, channel, row_chunk]: channel-major puts each (view,
# channel)'s row run CONTIGUOUS, so one tap's rc-float read is one coalesced load.
# This copy is the back path's extra memory (~one sino chunk); a transpose-free
# gather variant could reclaim it at the cost of strided kernel reads.
@partial(jax.jit, static_argnames=['rows_padded'])
def _to_channel_major(sino_chunk, rows_padded):
    rows = sino_chunk.shape[1]
    return jnp.pad(jnp.swapaxes(sino_chunk, 1, 2), ((0, 0), (0, 0),
                                                    (0, rows_padded - rows)))


# Donated add: the accumulator's buffer is reused, so cross-chunk accumulation holds
# two (num_pixels, rows_padded) arrays live, not three.
_accumulate = jax.jit(lambda a, b: a + b, donate_argnums=0)


def back_project_single_device(model, sinogram, pixel_indices, coeff_power=1,
                               output_device=None, interpret=False,
                               owned_view_indices=()):
    """The pallas single-device back-projection driver (the composed-prototype
    structure, productized): view-chunk loop x [centers, weights, channel-major layout,
    ONE kernel grid over ALL pixels], accumulating across chunks.  No pixel scan, no
    transfer chunking -- the kernel has no scan carry, so the whole pixel set goes in
    one grid.  ``interpret=True`` runs the kernel in pallas interpret mode (CPU-capable;
    used by the correctness tests -- the selection policy never routes here on CPU).

    Two calling modes:
    * DEFAULT (``owned_view_indices`` empty): the whole-model n=1 path -- the sinogram
      covers all views and the driver places it via the model.
    * PER-OWNER (``owned_view_indices`` = the caller's GLOBAL view indices): the
      multi-device band path's per-view-owner call.  ``sinogram`` is that owner's
      LOCAL (views, band_rows, channels) block, already resident on its device -- the
      driver must NOT reshard or re-place anything (view-owners run concurrently, one
      thread each; every output must stay on its caller's device).

    View chunking exists for MEMORY, not speed: the weights are (V, T, P) f32, so a
    full-grid call built for all views at once would hold multi-GB weight arrays;
    back_view_batch bounds that (and the sino_cm copy) per chunk.
    """
    from mbirjax.projectors import _jit_compute_scatter_centers   # lazy: avoids an
    # import cycle at package init (tomography_model -> this module -> projectors).
    pf = model.projector_functions
    pp = pf.projector_params
    if len(owned_view_indices) > 0:
        owned_all = np.asarray(owned_view_indices)
    else:
        owned_all = None
        sinogram = model._shard_sinogram(sinogram)
        pixel_indices = jax.device_put(pixel_indices, model.sino_placement.devices[0])
    num_views, rows, num_channels = sinogram.shape
    rows_padded = next_pow2(rows)
    psf_radius = pp.geometry_params.psf_radius
    num_pixels = int(pixel_indices.shape[0])

    view_chunk = min(model.tiles.back_view_batch, BACK_VIEW_CHUNK_CAP, num_views)
    kern = _make_back_call(view_chunk, num_channels, num_pixels, rows_padded,
                           psf_radius, interpret=interpret)

    out = None
    for v0 in range(0, num_views, view_chunk):
        v1 = min(v0 + view_chunk, num_views)
        if v1 - v0 < view_chunk:
            # Ragged tail chunk: fall through with a chunk-sized kernel of its own.
            kern_tail = _make_back_call(v1 - v0, num_channels, num_pixels, rows_padded,
                                        psf_radius, interpret=interpret)
        # Chunk view ids: LOCAL positions by default, the caller's GLOBAL ids in
        # per-owner mode (numpy -- the jitted builders gather view params in-jit).
        owned = np.arange(v0, v1) if owned_all is None else owned_all[v0:v1]
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
# The forward horizontal fan (subset-sized calls, parallel beam)
#
# out[c, :] = sum over taps (t, p) with center[p]+t == c of A[t,p] * values[p, :]
#
# The scatter becomes a GATHER over a channel-sorted contributor stream (sort +
# searchsorted -- the same ~2-3% sort cost the XLA path pays, paid once per call
# here), plus a cap-and-split that bounds every program's segment (skew guard: one
# hot channel must not stall a whole launch).  Two launches (the "two-phase"
# variant, the measured subset winner): phase 1 STORES every channel's first
# segment -- each output row written exactly once, no zero-fill, no cross-program
# race; phase 2 atomically adds the few remainder segments of over-cap channels
# (~empty at subset uniformity).  The values tile (P x band) is shared by ALL views
# -- L2-hot, which is where the win comes from (and why the dispatch guard is a
# PIXEL-COUNT limit: the tile is P*band*4 bytes against ~50 MB of L2 on H100).
#
# The split runs ON DEVICE with a static segment bound n2 = (T*P) // cap, and the
# whole per-chunk computation -- streams, split, both kernel phases -- is ONE cached
# jit.  Do NOT "simplify" back toward the obvious alternatives; both were measured
# to flip the win into a 0.68x loss:
#   * sizing phase 2 by the actual (data-dependent) segment count -- every VCD subset
#     then has its own kernel shape, a Triton recompile per subset inside the loop;
#   * splitting on host (pulling `starts` off-device) -- one pipeline stall per view
#     chunk.
# Unused bound slots are (start == end) rows aimed at the scratch channel: a
# zero-trip loop, and the atomic is pl.when-guarded so a pad program touches nothing
# but its own 4-int segment row (launch-only cost).
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
        # Out-of-range taps are NOT dropped: their weights are zeroed but their
        # stream slots remain (channel clipped to a boundary).  _split_two_phase's
        # static bound relies on this -- counts sum to EXACTLY T*P every view.
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


def _split_two_phase(starts, cap, n2):
    """Device-side cap-and-split for ONE view (the caller vmaps over views): phase 1 =
    the FIRST segment of every channel (empty channels store zero -- the no-zero-fill
    trick); phase 2 = remainder segments of over-cap channels, in n2 slots (a STATIC
    bound: n2 >= sum of ceil(count/cap) - 1 because each term is <= count/cap; unused
    slots become (0, 0) rows aimed at the scratch channel row C: atomic add of zero).

    ``starts`` is the (C + 1,) searchsorted boundary array; channel c's contributors
    occupy the sorted-stream range [starts[c], starts[c + 1]).  Each returned segment
    row is (start, end, channel, 0) -- the layout _fwd_kernel reads.
    """
    s0, s1 = starts[:-1], starts[1:]
    num_channels = s0.shape[0]
    chan = jnp.arange(num_channels, dtype=jnp.int32)
    seg1 = jnp.stack([s0, jnp.minimum(s0 + cap, s1), chan, jnp.zeros_like(chan)],
                     axis=1)

    nseg2 = jnp.maximum(0, -(-(s1 - s0) // cap) - 1)             # remainder segs per channel
    cum = jnp.cumsum(nseg2)
    j = jnp.arange(n2, dtype=jnp.int32)
    # Slot j belongs to the channel whose cumulative range contains it; k is the
    # 1-based remainder-segment index within that channel (segment k covers
    # [s0 + k*cap, s0 + (k+1)*cap) clipped to the channel's range).
    ch = jnp.minimum(jnp.searchsorted(cum, j, side='right'), num_channels - 1)
    ch = ch.astype(jnp.int32)
    k = j - (cum[ch] - nseg2[ch]) + 1
    a = s0[ch] + k * cap
    b = jnp.minimum(a + cap, s1[ch])
    valid = j < cum[-1]
    seg2 = jnp.stack([jnp.where(valid, a, 0), jnp.where(valid, b, 0),
                      jnp.where(valid, ch, num_channels), jnp.zeros_like(j)], axis=1)
    return seg1, seg2


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
        # Guarded: at subset uniformity most phase-2 slots are (start == end) pads all
        # aimed at ONE scratch row per view, and unguarded adds-of-zero would serialize
        # on those addresses.  Phase 1's store stays unconditional -- storing zeros for
        # empty channels IS the no-zero-fill trick.
        @pl.when(end > start)
        def _add():
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
        # Phase 2 accumulates IN PLACE on phase 1's output (operand 4 aliases the
        # result).  The alias also gives XLA the data dependency that orders the two
        # launches -- nothing else forces p1 to finish before p2's atomics begin.
        specs.append(pl.BlockSpec((num_views, num_channels + 1, band_padded),
                                  lambda v, s: (0, 0, 0)))
        alias = {4: 0}
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


@functools.cache
def _make_fwd_chunk_fn(hfan_data_fn, projector_params, vc, num_channels, n2,
                       num_pixels, band, band_padded, interpret):
    """One jitted function per static shape key covering a whole view chunk: streams,
    device split, both kernel phases, scratch/pad trim.  Cached so repeated calls (every
    VCD subset iteration at a given granularity) reuse ONE traced program -- no host
    sync and no retrace anywhere in the loop (lessons.md sections 3 and 5).

    Cache-key note: ``hfan_data_fn`` must be a plain function or @staticmethod (stable
    object identity across accesses).  Passing a BOUND method would fragment this
    cache per model instance and pin every model in memory for the process lifetime.
    ``projector_params`` is a hashable NamedTuple (the established static-arg pattern).
    """
    taps = (2 * projector_params.geometry_params.psf_radius + 1) * num_pixels
    p1 = _make_fwd_phase(vc, num_channels, num_channels, taps, num_pixels,
                         band_padded, atomic=False, interpret=interpret)
    p2 = _make_fwd_phase(vc, num_channels, n2, taps, num_pixels,
                         band_padded, atomic=True, interpret=interpret)

    def chunk_fn(view_params_array, pixel_indices, vals_pad, owned):
        wts, pix, starts = _jit_compute_fwd_streams(
            view_params_array, pixel_indices, hfan_data_fn, projector_params,
            owned_view_indices=owned)
        seg1, seg2 = jax.vmap(
            lambda s: _split_two_phase(s, FWD_SEGMENT_CAP, n2))(starts)
        out = p2(seg2, wts, pix, vals_pad, p1(seg1, wts, pix, vals_pad))
        # Trim scratch row + padding and reorient to (views, band_rows, channels) in
        # ONE fused copy (XLA merges the slice and the transpose).
        return jnp.swapaxes(out[:, :num_channels, :band], 1, 2)
    return jax.jit(chunk_fn)


def forward_project_subset(model, voxel_values, pixel_indices, owned_view_indices=(),
                           interpret=False):
    """The pallas forward driver (all pixel counts; the name predates the guard drop):
    a view-chunk loop of fused (streams + split + two-phase kernel) jit calls, outputs
    concatenated to the library orientation (views, band_rows, channels).  values
    (P, band) is the shared gather tile; band is whatever the caller passes (a slice
    band or full cylinders)."""
    pf = model.projector_functions
    pp = pf.projector_params
    num_channels = pp.sinogram_shape[2]
    num_pixels, band = voxel_values.shape
    band_padded = next_pow2(band)
    # Zero-eager-ops discipline (lessons.md section 3): the pad is skipped when the
    # band is already a power of two, view bookkeeping stays in numpy (the jit call
    # machinery transfers those tiny index arrays -- no separate eager dispatch), and
    # the reorientation happens inside the fused chunk jit.
    vals_pad = (voxel_values if band_padded == band else
                jnp.pad(voxel_values, ((0, 0), (0, band_padded - band))))

    if len(owned_view_indices) > 0:
        owned_all = np.asarray(owned_view_indices)
    else:
        owned_all = np.arange(pf.view_params_array.shape[0])
    num_views = int(owned_all.shape[0])
    view_chunk = min(model.tiles.fwd_view_batch, num_views)
    taps = (2 * pp.geometry_params.psf_radius + 1) * num_pixels
    n2 = max(1, taps // FWD_SEGMENT_CAP)          # static remainder-segment bound

    chunks = []
    for v0 in range(0, num_views, view_chunk):
        owned = owned_all[v0:min(v0 + view_chunk, num_views)]
        fn = _make_fwd_chunk_fn(model.compute_hfan_data, pp, int(owned.shape[0]),
                                num_channels, n2, num_pixels, band, band_padded,
                                interpret)
        chunks.append(fn(pf.view_params_array, pixel_indices, vals_pad, owned))
    # Chunk outputs are already (views, band_rows, channels).
    return chunks[0] if len(chunks) == 1 else jnp.concatenate(chunks, axis=0)


# ══════════════════════════════════════════════════════════════════════════════
# The cone fused-vfan back projection (single device AND the n>=2 band path)
#
# out[p, l] = sum over views v, row taps tr, channel taps tc of
#             Wrow[v,p,l,tr] * Wchan[v,p,tc] * sino[v, m(v,p,l)+tr-r, c(v,p)+tc-r]
#
# The enabling geometry fact (verified analytically + numerically, design section in
# gpu_headroom_findings.md): the projected detector-row center is EXACTLY affine in
# the slice index, m(v,p,l) = m0(v,p) + W_p_r(v,p) * l (the slope IS the projected
# row width), for flat and curved detectors alike -- so the vertical fan needs no
# per-slice precompute: the kernel forms row centers, trapezoid row weights, and the
# 1/cos(phi) divisor in-kernel from two scalars per (view, pixel).  The register
# tile holds out[p, l0:l0+LC] across the whole view loop (increment 1's design, 3x3
# taps).  Value contract (Greg 2026-07-13): gradient rel <= 1e-5; Hessian <= 1e-4 --
# the in-kernel f32 affine is a different rounding sequence than the XLA chain (~1-2
# ULP of m), which squared weights do not cancel (measured 2.0e-5 at the 1024 cell).
# Row centers use floor(m + 0.5) (jnp.round has no Triton lowering); safe because
# W_p_r <= 2*psf_radius makes a center flip move only a zero-weight tap.
# ══════════════════════════════════════════════════════════════════════════════

# Measured (e5_cone_fused_back.py sweep, H100): slice-chunk 128 dominates (the
# per-(view,pixel) scalar re-stream amortizes across the chunk -- the rc=256 lesson);
# num_warps=1 wins at every chunk size.
CONE_LC = 128
CONE_NUM_WARPS = 1


@partial(jax.jit, static_argnames=['projector_params'])
def _jit_compute_vfan_scalars(view_params_array, pixel_indices, projector_params,
                              owned_view_indices=()):
    """(m0, W_p_r) per (view, pixel): the affine row-center anchor at GLOBAL slice 0
    and the slope==width scalar.  MODULE-LEVEL jit (the retrace lesson)."""
    from mbirjax.cone_beam import ConeBeamModel     # lazy: avoids an import cycle
    if len(owned_view_indices) > 0:
        view_params_array = view_params_array[jnp.asarray(owned_view_indices)]

    def one_view(svp):
        def one_pixel(pidx):
            m_p, _, W_p_r, _ = ConeBeamModel.compute_vertical_data_single_pixel(
                pidx, jnp.arange(1), svp, projector_params)
            return m_p[0], W_p_r if jnp.ndim(W_p_r) == 0 else W_p_r[0]
        return jax.vmap(one_pixel)(pixel_indices)
    return jax.vmap(one_view)(view_params_array)


def _cone_back_kernel(c0_ref, wc_ref, m0_ref, wpr_ref, g0_ref, sino_ref, out_ref, *,
                      lc, num_views, num_channels, num_rows, psf_radius, psf_width,
                      coeff_power, delta_det_row, det_row_offset, det_center_row,
                      inv_sdd):
    """One program per (slice-chunk, pixel); see the section comment above."""
    l_vec = g0_ref[0] + pl.program_id(0) * lc + jnp.arange(lc).astype(jnp.float32)

    def vbody(v, acc):
        c0 = c0_ref[v, 0]
        m = m0_ref[v, 0] + wpr_ref[v, 0] * l_vec              # exactly affine
        wpr = wpr_ref[v, 0]
        mc = jnp.floor(m + 0.5).astype(jnp.int32)             # W<=2r makes flips inert
        v_p = (m - det_center_row) * delta_det_row - det_row_offset
        # The XLA vfan DIVIDES by cos(phi): multiply by 1/cos(phi) =
        # sqrt(1 + (v_p/sdd)^2); inv_sdd = 0 at sdd = Inf gives exactly 1 (Inf-safe).
        inv_cos = jnp.sqrt(1.0 + (v_p * inv_sdd) ** 2)
        L_max = jnp.minimum(1.0, wpr)
        for tr in range(psf_width):                           # static unroll
            mt = mc + (tr - psf_radius)
            w_row = jnp.clip((wpr + 1.0) / 2.0 - jnp.abs(m - mt.astype(jnp.float32)),
                             0.0, L_max) * inv_cos
            w_row = w_row * ((mt >= 0) & (mt < num_rows))     # real rows, not padded
            if coeff_power == 2:
                w_row = w_row * w_row                         # square AFTER the divisor
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
def _make_cone_back_call(num_views, num_channels, rows_padded, num_rows, num_pixels,
                         l_padded, lc, psf_radius, coeff_power, geom_consts,
                         interpret=False):
    # Cached on static shapes + the per-geometry scalar constants (hashable floats);
    # see _make_back_call's caching note.  rows_padded pads the CHANNEL-MAJOR block's
    # last axis to a power of two (production cone rows are often non-pow2); the
    # kernel masks and clips against the REAL num_rows, so padding is never read.
    kw = ({} if interpret else
          {'compiler_params': pltriton.CompilerParams(num_warps=CONE_NUM_WARPS)})
    delta_det_row, det_row_offset, det_center_row, inv_sdd = geom_consts
    return pl.pallas_call(
        partial(_cone_back_kernel, lc=lc, num_views=num_views,
                num_channels=num_channels, num_rows=num_rows, psf_radius=psf_radius,
                psf_width=2 * psf_radius + 1, coeff_power=coeff_power,
                delta_det_row=delta_det_row, det_row_offset=det_row_offset,
                det_center_row=det_center_row, inv_sdd=inv_sdd),
        out_shape=jax.ShapeDtypeStruct((num_pixels, l_padded), jnp.float32),
        grid=(l_padded // lc, num_pixels),                    # slice-chunk SLOWEST
        in_specs=[
            pl.BlockSpec((num_views, 1), lambda s, p: (0, p)),             # c0
            pl.BlockSpec((num_views, 2 * psf_radius + 1, 1),
                         lambda s, p: (0, 0, p)),                          # Wchan
            pl.BlockSpec((num_views, 1), lambda s, p: (0, p)),             # m0
            pl.BlockSpec((num_views, 1), lambda s, p: (0, p)),             # W_p_r
            pl.BlockSpec((1,), lambda s, p: (0,)),                         # g0 scalar
            pl.BlockSpec((num_views, num_channels, rows_padded),
                         lambda s, p: (0, 0, 0)),                          # sino ref
        ],
        out_specs=pl.BlockSpec((1, lc), lambda s, p: (p, s)),
        interpret=interpret, **kw)


def cone_back_project_band(model, sinogram, pixel_indices, g0, num_band_slices,
                           owned_view_indices=(), coeff_power=1, interpret=False):
    """Cone banded back projection of one view-owner's FULL-ROW views onto the global
    slice band [g0, g0 + num_band_slices) -- the pallas replacement for the XLA banded
    path (a cone slice draws from a RANGE of detector rows, so the rows are never
    cropped).  ``sinogram``: the owner's local (views, rows, channels) block; per-owner
    placement rules as in ``back_project_single_device``.  Returns (num_pixels, L).

    PADDED global slices (l >= the real slice count) are NOT zeroed here, unlike the
    XLA fan's in-kernel mask: inertness is delegated to the sharded assembly's
    ``_mask_padded_slices`` -- the one downstream consumer.  A new consumer of these
    partials must mask padded slices itself."""
    from mbirjax.projectors import _jit_compute_scatter_centers
    pf = model.projector_functions
    pp = pf.projector_params
    num_views, rows, num_channels = sinogram.shape
    rows_padded = next_pow2(rows)
    psf_radius = pp.geometry_params.psf_radius
    num_pixels = int(pixel_indices.shape[0])
    gp = pp.geometry_params
    sdd = gp.source_detector_dist
    geom_consts = (float(gp.delta_det_row), float(gp.det_row_offset),
                   (rows - 1) / 2.0,
                   0.0 if np.isinf(sdd) else float(1.0 / sdd))
    if len(owned_view_indices) > 0:
        owned_all = np.asarray(owned_view_indices)
    else:
        owned_all = np.arange(num_views)
    lc = min(CONE_LC, next_pow2(num_band_slices))
    l_padded = -(-num_band_slices // lc) * lc
    g0_arr = jnp.asarray([float(g0)], jnp.float32)

    out = None
    view_chunk = min(model.tiles.back_view_batch, BACK_VIEW_CHUNK_CAP, num_views)
    for v0 in range(0, num_views, view_chunk):
        v1 = min(v0 + view_chunk, num_views)
        owned = owned_all[v0:v1]
        c0 = _jit_compute_scatter_centers(pf.view_params_array, pixel_indices,
                                          model.compute_channel_coordinate, pp,
                                          pixels_major=False, owned_view_indices=owned)
        wc = _jit_compute_back_weights(pf.view_params_array, pixel_indices,
                                       model.compute_hfan_data, pp,
                                       coeff_power=coeff_power,
                                       owned_view_indices=owned)
        m0, wpr = _jit_compute_vfan_scalars(pf.view_params_array, pixel_indices, pp,
                                            owned_view_indices=owned)
        sino_cm = _to_channel_major(sinogram[v0:v1], rows_padded=rows_padded)
        kern = _make_cone_back_call(v1 - v0, num_channels, rows_padded, rows,
                                    num_pixels, l_padded, lc, psf_radius,
                                    coeff_power, geom_consts, interpret=interpret)
        chunk = kern(c0, wc, m0, wpr, g0_arr, sino_cm)
        out = chunk if out is None else _accumulate(out, chunk)
    return out[:, :num_band_slices]


def cone_back_project_single_device(model, sinogram, pixel_indices, coeff_power=1,
                                    output_device=None, interpret=False):
    """Cone n=1 back projection through the fused kernel: the full slice range as ONE
    launch per view chunk (l padded to a multiple of CONE_LC -- grid dims need not be
    powers of two, so the pad waste is < one slice-chunk)."""
    sinogram = model._shard_sinogram(sinogram)
    pixel_indices = jax.device_put(pixel_indices, model.sino_placement.devices[0])
    num_slices = model.get_params('recon_shape')[2]
    out = cone_back_project_band(model, sinogram, pixel_indices, 0, num_slices,
                                 coeff_power=coeff_power, interpret=interpret)
    if output_device is not None:
        out = jax.device_put(out, output_device)
    return out
