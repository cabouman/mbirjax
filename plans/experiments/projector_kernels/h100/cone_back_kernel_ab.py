"""A/B microbench: attribute CONE back-projection kernel time (horizontal vs vertical fan).

Cone back = a horizontal fan (per-tap channel GATHER, line-for-line the parallel back kernel
that just got the stacked-gather treatment) followed by a banded vertical fan (rows -> recon
slices, a rolled lax.map over slice bands) and an assembly transpose.  This bench splits the
monolithic (n=1) kernel's time between the pieces and measures the stacked-gather variant of
the horizontal fan in isolation AND substituted into the full kernel.

Variants (production shape: 2048-pixel batch vmapped over 128 views):
  cone_back_asis     : the library kernel, vmap over views + sum (as the driver composes it)
  hfan_only          : just the horizontal fan (share isolator; different output, unchecked)
  vfan_only          : just the band map + assembly fed a fixed cylinder (share isolator)
  hfan_stacked_only  : the stacked-gather horizontal fan alone (vs hfan_only)
  full_hfan_stacked  : the FULL kernel with the stacked horizontal fan substituted
                       (value-checked against cone_back_asis)

Run:  python plans/experiments/projector_kernels/cone_back_kernel_ab.py   (edit constants below).
"""
import os

# ── Run parameters (edit here) ────────────────────────────────────────────────
SINO_SHAPES = [(128, 448, 384), (128, 1008, 992)]   # (num_views, det_rows, det_channels)
PIXEL_BATCH = 2048
VIEW_BATCH = 128
WARMUP = 2
TRIALS = 5

os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.94')
import time                       # noqa: E402
import numpy as np                # noqa: E402
import mbirjax                    # noqa: E402
import jax                        # noqa: E402
import jax.numpy as jnp           # noqa: E402
from mbirjax.cone_beam import ConeBeamModel, CONE_SLICE_BAND_SIZE   # noqa: E402


def build(sino_shape):
    num_views, num_det_rows, num_det_channels = sino_shape
    angles = np.linspace(0, np.pi, num_views, endpoint=False)
    model = mbirjax.ConeBeamModel(sino_shape, angles,
                                  source_detector_dist=4.0 * num_det_channels,
                                  source_iso_dist=2.0 * num_det_channels)
    model.configure_devices(1)
    from mbirjax.projectors import ProjectorParams
    args = (tuple(model.get_params('sinogram_shape')), tuple(model.get_params('recon_shape')),
            model.get_geometry_parameters())
    args += tuple(int(bool(getattr(model.tiles, f, 0))) for f in ProjectorParams._fields[3:])
    pp = ProjectorParams(*args)
    rng = np.random.default_rng(0)
    idx = jnp.arange(min(PIXEL_BATCH, model.get_params('recon_shape')[0] ** 2))
    sino_batch = jnp.asarray(
        rng.random((VIEW_BATCH, num_det_rows, num_det_channels), dtype=np.float32))
    view_params = jnp.asarray(model.projector_functions.view_params_array)[:VIEW_BATCH]
    return model, pp, idx, sino_batch, view_params


def hfan_stacked(sinogram_view, pixel_indices, single_view_params, projector_params,
                 coeff_power=1):
    """The cone horizontal fan back with ONE stacked (psf_width * num_pixels, rows) gather."""
    gp = projector_params.geometry_params
    num_det_channels = projector_params.sinogram_shape[2]
    num_pixels = pixel_indices.shape[0]
    n_p, n_p_center, W_p_c, footprint_xy = ConeBeamModel.compute_horizontal_data(
        pixel_indices, single_view_params, projector_params)
    L_max = jnp.minimum(1, W_p_c)
    delta_voxel_row = gp.voxel_row_aspect * gp.delta_voxel
    sino_T = sinogram_view.T
    offsets = jnp.arange(-gp.psf_radius, gp.psf_radius + 1)
    n = n_p_center[None, :] + offsets[:, None]                # (psf_width, num_pixels)
    L = jnp.clip((W_p_c + 1) / 2 - jnp.abs(n_p - n), 0, L_max)
    A = ((delta_voxel_row * gp.delta_voxel) / footprint_xy) * L
    A = A * ((n >= 0) & (n < num_det_channels))
    A = A ** coeff_power
    n = jnp.clip(n, 0, num_det_channels - 1)
    gathered = sino_T[n.reshape(-1), :]
    weighted = A.reshape(-1)[:, None] * gathered
    return weighted.reshape(n.shape[0], num_pixels, -1).sum(axis=0)


def make_full_kernel(hfan):
    """The full monolithic kernel with a swappable horizontal fan (mirrors the library body)."""
    def kernel(sinogram_view, pixel_indices, single_view_params, projector_params,
               coeff_power=1):
        num_recon_slices = projector_params.recon_shape[2]
        num_pixels = pixel_indices.shape[0]
        det_voxel_cylinder = hfan(sinogram_view, pixel_indices, single_view_params,
                                  projector_params, coeff_power)
        band_size = min(CONE_SLICE_BAND_SIZE, num_recon_slices)
        num_bands = (num_recon_slices + band_size - 1) // band_size
        band_starts = band_size * jnp.arange(num_bands)

        def back_one_band(g0):
            return ConeBeamModel.back_vertical_fan_band_pixel_batch(
                det_voxel_cylinder, pixel_indices, single_view_params, projector_params,
                g0, band_size, coeff_power=coeff_power)

        bands = jax.lax.map(back_one_band, band_starts)
        out = jnp.transpose(bands, (1, 0, 2)).reshape(num_pixels, num_bands * band_size)
        return jax.lax.slice_in_dim(out, 0, num_recon_slices, axis=1)
    return kernel


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
    print(f"devices: {jax.devices()}   pixel_batch={PIXEL_BATCH}  view_batch={VIEW_BATCH}")
    back_lib = ConeBeamModel.back_project_one_view_to_pixel_batch
    hfan_lib = ConeBeamModel.back_horizontal_fan_one_view_to_pixel_batch

    for sino_shape in SINO_SHAPES:
        model, pp, idx, sino_batch, view_params = build(sino_shape)
        print(f"\n== sino {sino_shape}  recon {tuple(model.get_params('recon_shape'))}  "
              f"P={idx.shape[0]} pixels ==", flush=True)

        def vmap_sum(kernel):
            @jax.jit
            def f(sino_, idx_, vp_):
                pv = jax.vmap(kernel, in_axes=(0, None, 0, None, None))(sino_, idx_, vp_, pp, 1)
                return jnp.sum(pv, axis=0)
            return f

        # Vertical-only isolator: fixed cylinder through the band map + assembly.
        @jax.jit
        def vfan_only(sino_, idx_, vp_):
            dvc = jnp.ones((idx_.shape[0], sino_shape[1]))
            def one_view(single_vp):
                return make_full_kernel(lambda *a: dvc)(sino_[0], idx_, single_vp, pp, 1)
            return jnp.sum(jax.vmap(one_view)(vp_), axis=0)

        variants = [("cone_back_asis", vmap_sum(back_lib), True),
                    ("hfan_only", vmap_sum(hfan_lib), False),
                    ("vfan_only", vfan_only, False),
                    ("hfan_stacked_only", vmap_sum(hfan_stacked), False),
                    ("full_hfan_stacked", vmap_sum(make_full_kernel(hfan_stacked)), True)]
        ref = None
        t_ref = None
        for name, fn, check in variants:
            t = time_fn(fn, sino_batch, idx, view_params)
            ok = ""
            if check:
                out = np.asarray(fn(sino_batch, idx, view_params))
                if ref is None:
                    ref, t_ref = out, t
                else:
                    scale = np.max(np.abs(ref)) or 1.0
                    frac = np.mean(np.abs(out - ref) / scale > 1e-4)
                    ok = f"  frac>1e-4={frac:.2e}" + ("  MISMATCH!" if frac > 5e-3 else "")
            print(f"  {name:20s} : {1e3 * t:9.2f} ms   {t / t_ref:5.2f}x asis{ok}", flush=True)


if __name__ == "__main__":
    main()
