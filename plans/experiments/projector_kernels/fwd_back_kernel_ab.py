"""A/B microbench: WHY is the parallel-beam FORWARD kernel slower than BACK?

Times mbirjax.parallel_beam.forward_project_pixel_batch_to_one_view against
back_project_one_view_to_pixel_batch in the production shape (vmap over a view batch,
full-grid pixel batch), then re-times the forward kernel with its channel SCATTER-ADD
replaced by candidate implementations, to attribute the fwd/back gap and rank fixes.

Hypothesis under test: forward's inner loop is a scatter-add with heavily duplicated
indices (num_pixels >> num_channels, so ~num_pixels/num_channels collisions per channel),
while back's is a conflict-free gather + dense accumulate -- and the scatter is the gap.

Variants (all value-equal to the forward kernel; checked against it):
  * fwd_asis       : the library kernel (3 separate .at[n,:].add scatters, one per psf offset)
  * fwd_1scatter   : ONE stacked scatter of all 3 offsets ((3P,) indices, (3P,S) updates)
  * fwd_segsum     : jax.ops.segment_sum over the stacked (3P,) indices (unsorted)
  * fwd_sortsegsum : argsort the stacked indices per view, segment_sum(indices_are_sorted=True)
  * fwd_matmul     : dense (C,P) weight matrix (sum of the 3 one-hot*A) @ (P,S) voxel values
  * fwd_gathertable: gather-based (adjoint-shaped) forward -- per-view channel->pixel index +
                     weight tables (host-precomputed; legitimate because angles and the pixel
                     grid are static per model, so a real implementation caches them), then the
                     kernel is a pure GATHER + contiguous reduce, exactly back-kernel-shaped
  * fwd_noscatter  : LOWER BOUND, not value-equal -- all per-offset compute, psf-axis reduce,
                     but NO channel scatter (bounds "everything except the scatter")

Value check: robust, not max-error -- different XLA programs contract n_p differently (FMA),
and at exact .5 rounding ties a tap legally moves one channel, so isolated elements differ at
~1e-3 between value-equal programs.  We report the FRACTION of elements with rel err > 1e-4
and flag only if > 0.5%.

Run:  python plans/experiments/projector_kernels/fwd_back_kernel_ab.py
(no CLI args; edit the constants below).  CPU by default; on a GPU node it uses the GPU.
"""
import os

# ── Run parameters (edit here) ────────────────────────────────────────────────
SINO_SHAPES = [(128, 208, 160), (128, 448, 384)]  # (num_views, det_rows, det_channels);
                                                  # recon is auto-derived (rows=cols=channels)
# PRODUCTION kernel shape: the forward driver scans pixels at pixel_batch_size_for_vmap
# (2048) and vmaps the kernel over fwd_view_batch_size_for_vmap (128) angles, so the kernel
# sees (2048, S) voxels -> ~2048*3/C collisions per channel.  None = the full pixel grid in
# one call (the v1 shape; much higher collision counts, OOMs at 1024-class sizes).
PIXEL_BATCH = 2048
VIEW_BATCH = 128      # production fwd vmap width
WARMUP = 2
TRIALS = 5
CHECK_VALUES = True   # verify every variant matches fwd_asis (robust fraction metric)
RUN_DRIVER_LEVEL = True   # also time the REAL sparse_forward/back_project drivers (full grid)
# Band-width sweep: the banded forward hands the kernel B-slice bands (band length B, as in
# tomography_model._slice_band_length); the sorted reduce's per-view sort cost amortizes over
# B, so it loses at small B (end-to-end: lost at B=24, won at B=63).  Sweep scatter vs sorted
# at these band widths to place the dispatch threshold.
SLICE_SWEEP = []          # e.g. [16, 24, 32, 40, 48, 64, 96]; [] = skip
RUN_GATHERTABLE = False   # padded (C,K) tables: dead end (per-channel counts skew ~17x ->
                          # padding waste); the unpadded equivalent IS fwd_sortsegsum

os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.94')  # before jax init (project rule)
import time                       # noqa: E402
from functools import partial     # noqa: E402
import numpy as np                # noqa: E402
import mbirjax                    # noqa: E402  (device setup side effect precedes jax use)
import jax                        # noqa: E402
import jax.numpy as jnp           # noqa: E402


def build(sino_shape):
    """Model + production-shaped kernel inputs for one problem size."""
    num_views, num_det_rows, num_det_channels = sino_shape
    angles = np.linspace(0, np.pi, num_views, endpoint=False)
    model = mbirjax.ParallelBeamModel(sino_shape, angles)
    model.configure_devices(1)   # single device: kernel-level comparison, no sharding
    from mbirjax.projectors import ProjectorParams
    # The kernel-algorithm flag mirrors create_projectors so the library kernel (fwd_asis)
    # dispatches as production would.  Compat chain across library generations: current =
    # sort_by_channel (from the tile policy); earlier = backend_gpu (platform); oldest = the
    # 3-field ProjectorParams (no flag).
    args = (tuple(model.get_params('sinogram_shape')), tuple(model.get_params('recon_shape')),
            model.get_geometry_parameters())
    if 'sort_by_channel' in ProjectorParams._fields:
        args += (int(bool(model.tiles.sort_by_channel)),)
    elif 'backend_gpu' in ProjectorParams._fields:
        args += (int(model.sino_placement.devices[0].platform == 'gpu'),)
    pp = ProjectorParams(*args)
    recon_shape = model.get_params('recon_shape')
    num_pixels = recon_shape[0] * recon_shape[1]
    rng = np.random.default_rng(0)
    voxel_values = jnp.asarray(rng.random((num_pixels, recon_shape[2]), dtype=np.float32))
    pixel_indices = jnp.arange(num_pixels)
    sino_batch = jnp.asarray(
        rng.random((VIEW_BATCH, num_det_rows, num_det_channels), dtype=np.float32))
    view_angles = jnp.asarray(angles[:VIEW_BATCH])
    return model, pp, voxel_values, pixel_indices, sino_batch, view_angles


# ── Forward-kernel variants ───────────────────────────────────────────────────
# Each takes (voxel_values, pixel_indices, angle, projector_params) and returns the
# (num_input_slices, num_det_channels) view, exactly like the library kernel.
def _proj_pieces(voxel_values, pixel_indices, angle, projector_params):
    """The shared per-offset compute: channel indices n (3,P) and weights A (3,P)."""
    gp = projector_params.geometry_params
    num_views, num_det_rows, num_det_channels = projector_params.sinogram_shape
    n_p, n_p_center, W_p_c, footprint_xy = mbirjax.ParallelBeamModel.compute_proj_data(
        pixel_indices, angle, projector_params)
    L_max = jnp.minimum(1.0, W_p_c)
    delta_voxel_row = gp.voxel_row_aspect * gp.delta_voxel
    offsets = jnp.arange(-gp.psf_radius, gp.psf_radius + 1)
    n = n_p_center[None, :] + offsets[:, None]                     # (3, P)
    abs_delta = jnp.abs(n_p[None, :] - n)
    L = jnp.clip((W_p_c + 1.0) / 2.0 - abs_delta, 0.0, L_max)
    A = ((delta_voxel_row * gp.delta_voxel) / footprint_xy) * L    # (3, P)
    A = A * ((n >= 0) & (n < num_det_channels))
    n = jnp.clip(n, 0, num_det_channels - 1)                       # weights already 0 outside
    return n, A, num_det_channels


def fwd_1scatter(voxel_values, pixel_indices, angle, projector_params):
    n, A, C = _proj_pieces(voxel_values, pixel_indices, angle, projector_params)
    updates = A.reshape(-1)[:, None] * jnp.tile(voxel_values, (n.shape[0], 1))   # (3P, S)
    out = jnp.zeros((C, voxel_values.shape[1]))
    return out.at[n.reshape(-1), :].add(updates).T


def fwd_segsum(voxel_values, pixel_indices, angle, projector_params):
    n, A, C = _proj_pieces(voxel_values, pixel_indices, angle, projector_params)
    updates = A.reshape(-1)[:, None] * jnp.tile(voxel_values, (n.shape[0], 1))
    return jax.ops.segment_sum(updates, n.reshape(-1), num_segments=C).T


def fwd_sortsegsum(voxel_values, pixel_indices, angle, projector_params):
    n, A, C = _proj_pieces(voxel_values, pixel_indices, angle, projector_params)
    flat_n = n.reshape(-1)
    order = jnp.argsort(flat_n)
    updates = A.reshape(-1)[:, None] * jnp.tile(voxel_values, (n.shape[0], 1))
    return jax.ops.segment_sum(updates[order], flat_n[order], num_segments=C,
                               indices_are_sorted=True).T


def fwd_matmul(voxel_values, pixel_indices, angle, projector_params):
    n, A, C = _proj_pieces(voxel_values, pixel_indices, angle, projector_params)
    # Dense (C, P) weights: sum over the 3 offsets of one_hot(n)*A.  P*C floats per view.
    W = jnp.zeros((C, n.shape[1]))
    for k in range(n.shape[0]):
        W = W.at[n[k], jnp.arange(n.shape[1])].add(A[k])
    return (W @ voxel_values).T


def fwd_noscatter(voxel_values, pixel_indices, angle, projector_params):
    """LOWER BOUND (wrong values): the per-offset compute + psf reduce, no channel scatter."""
    n, A, C = _proj_pieces(voxel_values, pixel_indices, angle, projector_params)
    acc = jnp.einsum('kp,ps->ps', A, voxel_values)                 # (P, S) psf-axis reduce
    return acc[:C, :].T                                            # shape-compatible crop


def build_gather_tables(voxel_shape, pixel_indices, view_angles, projector_params):
    """Host-side (numpy) per-view channel->pixel tables for the gather-based forward.

    For each view: the (3P,) stacked (channel, pixel, weight) triples, sorted by channel and
    padded per channel to a common K (pad = pixel 0 with weight 0).  Returns
    (Idx (V,C,K) int32, Wt (V,C,K) float32, K).  A real implementation would cache these per
    (angle set, pixel batch) -- both static per model -- so the build cost amortizes; it is
    reported separately, outside the timed kernel.
    """
    V = len(view_angles)
    C = projector_params.sinogram_shape[2]
    per_view = []
    kmax = 0
    for a in np.asarray(view_angles):
        n, A, _ = _proj_pieces(jnp.zeros(voxel_shape), pixel_indices, jnp.float32(a),
                               projector_params)
        n = np.asarray(n).reshape(-1)
        A = np.asarray(A).reshape(-1)
        pix = np.tile(np.arange(voxel_shape[0], dtype=np.int32), n.shape[0] // voxel_shape[0])
        order = np.argsort(n, kind='stable')
        n, A, pix = n[order], A[order], pix[order]
        counts = np.bincount(n, minlength=C)
        kmax = max(kmax, int(counts.max()))
        per_view.append((n, A, pix, counts))
    Idx = np.zeros((V, C, kmax), dtype=np.int32)
    Wt = np.zeros((V, C, kmax), dtype=np.float32)
    for v, (n, A, pix, counts) in enumerate(per_view):
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        for c in range(C):
            k = counts[c]
            Idx[v, c, :k] = pix[starts[c]:starts[c] + k]
            Wt[v, c, :k] = A[starts[c]:starts[c] + k]
    return jnp.asarray(Idx), jnp.asarray(Wt), kmax


GATHER_K_CHUNK = 64   # bounds the gather transient to (V, C, k_chunk, S) under the view vmap


def fwd_gathertable(voxel_values, idx_v, wt_v):
    """Gather-based forward for ONE view given its (C,K) tables.

    Scans K in chunks so the gathered transient is (C, k_chunk, S), not (C, K, S) --
    the unchunked form OOMs (hundreds of GB at production sizes).
    """
    C, K = idx_v.shape
    S = voxel_values.shape[1]
    n_chunks = -(-K // GATHER_K_CHUNK)
    pad = n_chunks * GATHER_K_CHUNK - K
    idx_c = jnp.pad(idx_v, ((0, 0), (0, pad))).reshape(C, n_chunks, -1).transpose(1, 0, 2)
    wt_c = jnp.pad(wt_v, ((0, 0), (0, pad))).reshape(C, n_chunks, -1).transpose(1, 0, 2)

    def body(acc, args):
        ii, ww = args                                              # (C, k_chunk) each
        return acc + jnp.einsum('ck,cks->cs', ww, voxel_values[ii]), None

    acc, _ = jax.lax.scan(body, jnp.zeros((C, S)), (idx_c, wt_c))
    return acc.T


# ── Timing helpers ────────────────────────────────────────────────────────────
def time_fn(fn, *args):
    for _ in range(WARMUP):
        jax.block_until_ready(fn(*args))
    ts = []
    for _ in range(TRIALS):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append(time.perf_counter() - t0)
    return min(ts)


def main():
    print(f"devices: {jax.devices()}   pixel_batch={PIXEL_BATCH}  view_batch={VIEW_BATCH}  "
          f"warmup={WARMUP} trials={TRIALS}")
    fwd_lib = mbirjax.ParallelBeamModel.forward_project_pixel_batch_to_one_view
    back_lib = mbirjax.ParallelBeamModel.back_project_one_view_to_pixel_batch

    for sino_shape in SINO_SHAPES:
        model, pp, vox_full, idx_full, sino_batch, view_angles = build(sino_shape)
        recon_shape = model.get_params('recon_shape')
        P_full, S = vox_full.shape
        C = sino_shape[2]
        # Kernel-level inputs: one PRODUCTION pixel batch (the driver scans these).
        pb = P_full if PIXEL_BATCH is None else min(PIXEL_BATCH, P_full)
        vox, idx = vox_full[:pb], idx_full[:pb]
        print(f"\n== sino {sino_shape}  recon {tuple(recon_shape)}  "
              f"kernel sees P={pb} pixels (of {P_full}), S={S} slices, C={C} channels, "
              f"~{max(1, 3 * pb // C)} collisions/channel ==", flush=True)

        # Kernel-level: forward vmaps over angles (same voxel batch); back vmaps over
        # (views, angles) with the SAME pixel batch, then sums over views (as its driver does).
        def make_fwd_batch(kernel, f32_matmul=False):
            @jax.jit
            def f(vox_, idx_, angles_):
                if f32_matmul:   # baked at trace time: forces full-f32 matmul (no TF32)
                    with jax.default_matmul_precision('float32'):
                        return jax.vmap(lambda a: kernel(vox_, idx_, a, pp))(angles_)
                return jax.vmap(lambda a: kernel(vox_, idx_, a, pp))(angles_)
            return f

        @jax.jit
        def back_batch(sino_, idx_, angles_):
            per_view = jax.vmap(back_lib, in_axes=(0, None, 0, None, None))(
                sino_, idx_, angles_, pp, 1)
            return jnp.sum(per_view, axis=0)

        t_back = time_fn(back_batch, sino_batch, idx, view_angles)
        print(f"  back (vmap+sum, reference) : {1e3 * t_back:9.2f} ms", flush=True)
        ref = None

        def check(out):
            # Robust: fraction of elements deviating (see header) -- FMA/tie effects make
            # isolated elements differ ~1e-3 between value-equal programs.
            scale = np.max(np.abs(ref)) or 1.0
            frac = np.mean(np.abs(out - ref) / scale > 1e-4)
            return f"  frac>1e-4={frac:.2e}" + ("  MISMATCH!" if frac > 5e-3 else "")

        def report(name, t, ok):
            print(f"  {name:26s} : {1e3 * t:9.2f} ms   {t / t_back:5.2f}x back{ok}", flush=True)

        variants = [("fwd_asis", fwd_lib, {}), ("fwd_1scatter", fwd_1scatter, {}),
                    ("fwd_segsum", fwd_segsum, {}), ("fwd_sortsegsum", fwd_sortsegsum, {}),
                    ("fwd_matmul", fwd_matmul, {}),
                    ("fwd_matmul_f32", fwd_matmul, {"f32_matmul": True}),
                    ("fwd_noscatter", fwd_noscatter, {})]
        for name, kern, kw in variants:
            fn = make_fwd_batch(kern, **kw)
            t = time_fn(fn, vox, idx, view_angles)
            ok = ""
            if CHECK_VALUES and name != "fwd_noscatter":
                out = np.asarray(fn(vox, idx, view_angles))
                if ref is None:
                    ref = out
                else:
                    ok = check(out)
            report(name, t, ok)

        if RUN_GATHERTABLE:
            # Padded (C,K) tables: kept for reference; K skews ~17x above the mean count, so
            # the padding waste kills it -- the unpadded equivalent is fwd_sortsegsum.
            t0 = time.perf_counter()
            Idx, Wt, K = build_gather_tables(vox.shape, idx, view_angles, pp)
            t_build = time.perf_counter() - t0

            @jax.jit
            def gather_batch(vox_, Idx_, Wt_):
                return jax.vmap(fwd_gathertable, in_axes=(None, 0, 0))(vox_, Idx_, Wt_)

            t = time_fn(gather_batch, vox, Idx, Wt)
            ok = check(np.asarray(gather_batch(vox, Idx, Wt))) if CHECK_VALUES else ""
            report(f"fwd_gathertable (K={K})", t,
                   ok + f"  [table build {t_build:.2f}s host, cacheable]")

        if SLICE_SWEEP:
            # Band-width sweep: the kernel projects a B-slice band (voxel_values[:, :B]); time
            # the scatter vs sorted reductions at each B to place the dispatch threshold.
            print("  band-width sweep (band length B): scatter vs sorted, ms per 128-view call",
                  flush=True)
            for b_band in SLICE_SWEEP:
                vox_band = vox[:, :b_band]
                t_sc = time_fn(make_fwd_batch(fwd_lib), vox_band, idx, view_angles)
                t_ss = time_fn(make_fwd_batch(fwd_sortsegsum), vox_band, idx, view_angles)
                verdict = "sorted wins" if t_ss < t_sc else "scatter wins"
                print(f"    B={b_band:4d}: scatter {1e3 * t_sc:7.2f}  sorted {1e3 * t_ss:7.2f}"
                      f"   ({verdict}, ratio {t_ss / t_sc:4.2f})", flush=True)

        if RUN_DRIVER_LEVEL:
            # Ground truth: the REAL drivers on the full pixel grid (includes the pixel scan /
            # view batching exactly as production runs them at this device layout).
            pf = model.projector_functions
            t_fwd_drv = time_fn(lambda: pf.sparse_forward_project(vox_full, idx_full))
            t_back_drv = time_fn(lambda: pf.sparse_back_project(
                jnp.asarray(np.random.default_rng(1).random(sino_shape, dtype=np.float32)),
                idx_full))
            print(f"  driver sparse_forward      : {1e3 * t_fwd_drv:9.2f} ms   "
                  f"{t_fwd_drv / t_back_drv:5.2f}x driver back ({1e3 * t_back_drv:.2f} ms)",
                  flush=True)


if __name__ == "__main__":
    main()
