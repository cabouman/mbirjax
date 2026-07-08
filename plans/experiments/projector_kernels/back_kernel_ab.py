"""A/B microbench: where does the parallel-beam BACK projection kernel time go?

Back projection is the long pole now that forward is fixed (H100 model level: back 18.3 s vs
forward 8.2 s at 1024^3 n=1).  Back's kernel is gather-based (no scatter), so the forward
playbook does not transfer directly; the suspects, from the June profiling and code reading:

  * THE VIEW REDUCE: the driver vmaps the per-view kernel then reduces with
    ``jnp.sum(axis=0)`` over a materialized (V, P, B) stack (~470 MB at production shape) --
    ncu previously fingerprinted "the accumulate kernel" as memory-access-pattern-bound.
  * THE TRANSPOSE: each per-view call transposes its view to channel-major inside the vmap
    (the June multi-GPU profiling flagged the band transpose as a limiter).
  * THE PER-TAP GATHERS: 3 separate gather+FMA passes per view.

Variants (all value-equal to production up to summation order; checked):
  back_asis        : production composition -- vmap(kernel) over views, then sum(axis=0)
  back_scan        : lax.scan over views, (P, B) carry += kernel(view)  [no (V,P,B) stack]
  back_chunk8/32   : hybrid -- scan over view chunks, vmap+sum inside each chunk
  back_taps1gather : kernel variant: ONE stacked gather (T*P, B) + reshape-sum over taps
  back_pretranspose: transpose the WHOLE (V, rows, C) batch to channel-major once, outside
                     the vmap (kernel reads it directly)
  back_nogather    : LOWER BOUND, wrong values -- fixed row instead of the data-dependent
                     gather (bounds "everything except the gather")

Run:  python plans/experiments/projector_kernels/back_kernel_ab.py   (edit constants below).
"""
import os

# ── Run parameters (edit here) ────────────────────────────────────────────────
SINO_SHAPES = [(128, 208, 160), (128, 448, 384)]  # (num_views, det_rows, det_channels)
PIXEL_BATCH = 2048    # production kernel shape: back driver concatenates 2048-pixel batches
VIEW_BATCH = 128      # production back view batch at n=1 (the GPU short-circuit path)
WARMUP = 2
TRIALS = 5
CHECK_VALUES = True
RUN_DRIVER_LEVEL = True   # also time the REAL sparse_back_project driver (full pixel grid)

os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.94')  # before jax init (project rule)
import time                       # noqa: E402
import numpy as np                # noqa: E402
import mbirjax                    # noqa: E402  (device setup precedes jax)
import jax                        # noqa: E402
import jax.numpy as jnp           # noqa: E402


def build(sino_shape):
    num_views, num_det_rows, num_det_channels = sino_shape
    angles = np.linspace(0, np.pi, num_views, endpoint=False)
    model = mbirjax.ParallelBeamModel(sino_shape, angles)
    model.configure_devices(1)
    from mbirjax.projectors import ProjectorParams
    args = (tuple(model.get_params('sinogram_shape')), tuple(model.get_params('recon_shape')),
            model.get_geometry_parameters())
    # Mirror create_projectors for EVERY kernel-algorithm flag field, so back_asis measures
    # the kernel exactly as production dispatches it on this library generation.
    args += tuple(int(bool(getattr(model.tiles, f, 0))) for f in ProjectorParams._fields[3:])
    pp = ProjectorParams(*args)
    recon_shape = model.get_params('recon_shape')
    num_pixels = recon_shape[0] * recon_shape[1]
    rng = np.random.default_rng(0)
    pb = min(PIXEL_BATCH, num_pixels)
    pixel_indices = jnp.arange(pb)
    sino_batch = jnp.asarray(
        rng.random((VIEW_BATCH, num_det_rows, num_det_channels), dtype=np.float32))
    view_angles = jnp.asarray(angles[:VIEW_BATCH])
    return model, pp, pixel_indices, sino_batch, view_angles


# ── Kernel/composition variants ───────────────────────────────────────────────
def _back_pieces(pixel_indices, angle, projector_params):
    """Shared per-view weights: channel indices n (T,P) and weights A (T,P), clipped/masked."""
    gp = projector_params.geometry_params
    num_det_channels = projector_params.sinogram_shape[2]
    n_p, n_p_center, W_p_c, footprint_xy = mbirjax.ParallelBeamModel.compute_proj_data(
        pixel_indices, angle, projector_params)
    L_max = jnp.minimum(1.0, W_p_c)
    delta_voxel_row = gp.voxel_row_aspect * gp.delta_voxel
    offsets = jnp.arange(-gp.psf_radius, gp.psf_radius + 1)
    n = n_p_center[None, :] + offsets[:, None]
    L = jnp.clip((W_p_c + 1.0) / 2.0 - jnp.abs(n_p[None, :] - n), 0.0, L_max)
    A = ((delta_voxel_row * gp.delta_voxel) / footprint_xy) * L
    A = A * ((n >= 0) & (n < num_det_channels))
    n = jnp.clip(n, 0, num_det_channels - 1)
    return n, A


def back_taps1gather(sinogram_view, pixel_indices, angle, projector_params, coeff_power=1):
    """ONE stacked (T*P, B) gather + reshape-sum over taps (vs 3 gather+FMA passes)."""
    n, A = _back_pieces(pixel_indices, angle, projector_params)
    sino_T = sinogram_view.T
    gathered = sino_T[n.reshape(-1), :]                          # (T*P, B)
    weighted = (A.reshape(-1) ** coeff_power)[:, None] * gathered
    return weighted.reshape(n.shape[0], n.shape[1], -1).sum(axis=0)


def back_kernel_pretransposed(sino_T, pixel_indices, angle, projector_params, coeff_power=1):
    """The library kernel body, but taking an ALREADY channel-major view (C, rows)."""
    n, A = _back_pieces(pixel_indices, angle, projector_params)
    out = jnp.zeros((pixel_indices.shape[0], sino_T.shape[1]))
    for k in range(n.shape[0]):
        out = out + (A[k] ** coeff_power)[:, None] * sino_T[n[k], :]
    return out


def back_nogather(sinogram_view, pixel_indices, angle, projector_params, coeff_power=1):
    """LOWER BOUND (wrong values): fixed row instead of the data-dependent gather."""
    n, A = _back_pieces(pixel_indices, angle, projector_params)
    sino_T = sinogram_view.T
    out = jnp.zeros((pixel_indices.shape[0], sino_T.shape[1]))
    for k in range(n.shape[0]):
        out = out + (A[k] ** coeff_power)[:, None] * sino_T[0:1, :]   # no gather
    return out


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
    back_lib = mbirjax.ParallelBeamModel.back_project_one_view_to_pixel_batch

    for sino_shape in SINO_SHAPES:
        model, pp, idx, sino_batch, view_angles = build(sino_shape)
        recon_shape = model.get_params('recon_shape')
        P = idx.shape[0]
        print(f"\n== sino {sino_shape}  recon {tuple(recon_shape)}  kernel sees P={P} pixels, "
              f"B={sino_shape[1]} rows, C={sino_shape[2]} channels ==", flush=True)

        def make(comp):
            @jax.jit
            def f(sino_, idx_, angles_):
                return comp(sino_, idx_, angles_)
            return f

        # Production: vmap the kernel over views, sum over the view axis.
        def comp_asis(sino_, idx_, angles_):
            per_view = jax.vmap(back_lib, in_axes=(0, None, 0, None, None))(
                sino_, idx_, angles_, pp, 1)
            return jnp.sum(per_view, axis=0)

        # Scan over views: (P, B) carry, no (V, P, B) stack.
        def comp_scan(sino_, idx_, angles_):
            def body(acc, va):
                view, angle = va
                return acc + back_lib(view, idx_, angle, pp, 1), None
            acc0 = jnp.zeros((idx_.shape[0], sino_.shape[1]))
            acc, _ = jax.lax.scan(body, acc0, (sino_, angles_))
            return acc

        # Hybrid: scan over chunks of views, vmap+sum inside the chunk.
        def make_comp_chunk(chunk):
            def comp(sino_, idx_, angles_):
                V = sino_.shape[0]
                nch = V // chunk
                sino_c = sino_[:nch * chunk].reshape(nch, chunk, *sino_.shape[1:])
                ang_c = angles_[:nch * chunk].reshape(nch, chunk)
                def body(acc, sa):
                    s, a = sa
                    pv = jax.vmap(back_lib, in_axes=(0, None, 0, None, None))(s, idx_, a, pp, 1)
                    return acc + jnp.sum(pv, axis=0), None
                acc0 = jnp.zeros((idx_.shape[0], sino_.shape[1]))
                acc, _ = jax.lax.scan(body, acc0, (sino_c, ang_c))
                for v in range(nch * chunk, V):    # ragged tail (none when chunk | V)
                    acc = acc + back_lib(sino_[v], idx_, angles_[v], pp, 1)
                return acc
            return comp

        # Stacked-tap kernel, production composition.
        def comp_taps1gather(sino_, idx_, angles_):
            per_view = jax.vmap(back_taps1gather, in_axes=(0, None, 0, None, None))(
                sino_, idx_, angles_, pp, 1)
            return jnp.sum(per_view, axis=0)

        # Transpose hoisted out of the vmap: one (V, C, rows) transpose, then the kernel.
        def comp_pretranspose(sino_, idx_, angles_):
            sino_T = jnp.transpose(sino_, (0, 2, 1))
            per_view = jax.vmap(back_kernel_pretransposed, in_axes=(0, None, 0, None, None))(
                sino_T, idx_, angles_, pp, 1)
            return jnp.sum(per_view, axis=0)

        # Lower bound: no data-dependent gather.
        def comp_nogather(sino_, idx_, angles_):
            per_view = jax.vmap(back_nogather, in_axes=(0, None, 0, None, None))(
                sino_, idx_, angles_, pp, 1)
            return jnp.sum(per_view, axis=0)

        variants = [("back_asis", comp_asis), ("back_scan", comp_scan),
                    ("back_chunk8", make_comp_chunk(8)), ("back_chunk32", make_comp_chunk(32)),
                    ("back_taps1gather", comp_taps1gather),
                    ("back_pretranspose", comp_pretranspose),
                    ("back_nogather", comp_nogather)]
        ref = None
        t_ref = None
        for name, comp in variants:
            fn = make(comp)
            t = time_fn(fn, sino_batch, idx, view_angles)
            ok = ""
            if CHECK_VALUES and name != "back_nogather":
                out = np.asarray(fn(sino_batch, idx, view_angles))
                if ref is None:
                    ref, t_ref = out, t
                else:
                    scale = np.max(np.abs(ref)) or 1.0
                    frac = np.mean(np.abs(out - ref) / scale > 1e-4)
                    ok = f"  frac>1e-4={frac:.2e}" + ("  MISMATCH!" if frac > 5e-3 else "")
            print(f"  {name:20s} : {1e3 * t:9.2f} ms   {t / t_ref:5.2f}x asis{ok}", flush=True)

        if RUN_DRIVER_LEVEL:
            pf = model.projector_functions
            num_pixels = recon_shape[0] * recon_shape[1]
            idx_full = jnp.arange(num_pixels)
            rng = np.random.default_rng(1)
            sino_full = jnp.asarray(rng.random(sino_shape, dtype=np.float32))
            t_drv = time_fn(lambda: pf.sparse_back_project(sino_full, idx_full))
            print(f"  driver sparse_back        : {1e3 * t_drv:9.2f} ms  (full grid, "
                  f"{sino_shape[0]} views)", flush=True)


if __name__ == "__main__":
    main()
