"""A/B microbench: does the stacked-gather horizontal fan help the MULTIAXIS or TRANSLATION
back kernels IN COMPOSITION?

Cone's lesson (cone_back_kernel_ab.py): the stacked hfan wins in isolation (0.57-0.59x) but
is a 1.00x no-op substituted into the full kernel -- XLA overlaps the hfan gather with the
vertical-fan band work.  Multiaxis and translation back have the same hfan -> banded-vfan
composition, so the expectation is the same no-op; this bench decides with one GPU job
whether `back_stacked_gather` should be wired for either geometry (per the campaign rule:
measure the COMPOSITION, not the pieces).

Variants per geometry (production shape: 2048-pixel batch vmapped over the view batch):
  back_asis          : the library kernel, vmap over views + sum (as the driver composes it)
  hfan_only          : just the horizontal fan (share isolator)
  vfan_only          : just the band map + assembly fed a fixed cylinder (share isolator)
  hfan_stacked_only  : the stacked-gather horizontal fan alone (vs hfan_only)
  full_hfan_stacked  : the FULL kernel with the stacked hfan substituted (value-checked)

Translation runs at BOTH the harness TCT geometry (psf_radius 1) and a wide-psf variant
(delta_voxel scaled so psf_radius = 3; the large-cone-angle regime), since the hfan share
grows with the tap count.

Run:  python plans/experiments/projector_kernels/mt_back_kernel_ab.py   (edit constants below).
"""
import os

# ── Run parameters (edit here) ────────────────────────────────────────────────
MULTIAXIS_SINO_SHAPES = [(128, 224, 192), (128, 448, 384)]   # (view batch, det_rows, det_channels)
TRANSLATION_DETECTORS = [(256, 256), (512, 512)]             # (det_rows, det_channels); 15 views fixed
TRANSLATION_WIDE_PSF_SCALE = 5.0    # delta_voxel multiplier -> psf_radius 3 (psf_width 7)
PIXEL_BATCH = 2048
WARMUP = 2
TRIALS = 5

os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.94')
import time                       # noqa: E402
import numpy as np                # noqa: E402
import mbirjax                    # noqa: E402
import jax                        # noqa: E402
import jax.numpy as jnp           # noqa: E402
from mbirjax.multiaxis_parallel import MultiAxisParallelModel, MULTIAXIS_SLICE_BAND_SIZE  # noqa: E402
from mbirjax.translation_model import TranslationModel, TRANSLATION_SLICE_BAND_SIZE       # noqa: E402
from mbirjax.projectors import ProjectorParams   # noqa: E402


# ── Model builders (mirror the harness geometries in performance_tracking.make_model) ────
def finish(model):
    model.configure_devices(1)
    args = (tuple(model.get_params('sinogram_shape')), tuple(model.get_params('recon_shape')),
            model.get_geometry_parameters())
    args += tuple(int(bool(getattr(model.tiles, f, 0))) for f in ProjectorParams._fields[3:])
    pp = ProjectorParams(*args)
    rng = np.random.default_rng(0)
    num_views, num_det_rows, num_det_channels = model.get_params('sinogram_shape')
    n_pix = model.get_params('recon_shape')[0] * model.get_params('recon_shape')[1]
    idx = jnp.arange(min(PIXEL_BATCH, n_pix))
    sino_batch = jnp.asarray(
        rng.random((num_views, num_det_rows, num_det_channels), dtype=np.float32))
    view_params = jnp.asarray(model.projector_functions.view_params_array)
    return model, pp, idx, sino_batch, view_params


def build_multiaxis(sino_shape):
    num_views, n_rows, n_channels = sino_shape
    azimuths = np.linspace(0, np.pi, num_views, endpoint=False)
    elevation = np.full(num_views, np.deg2rad(25.0))
    model = mbirjax.MultiAxisParallelModel(sino_shape, np.column_stack([azimuths, elevation]))
    pb = mbirjax.ParallelBeamModel(sino_shape, azimuths)   # harness recon convention
    model.set_params(recon_shape=pb.get_params('recon_shape'))
    return finish(model)


def build_translation(detector, psf_scale=1.0):
    n_rows, n_channels = detector
    sdd, sid, delta_det = 190000.0, 70000.0, 75.0          # harness TCT geometry
    num_x, num_z = 5, 3
    delta_voxel = delta_det / (sdd / sid) * psf_scale
    cols, slices = n_channels, n_rows
    x_spacing = cols * delta_voxel / (num_x - 1)
    z_spacing = slices * delta_voxel / (num_z - 1)
    tv = mbirjax.gen_translation_vectors(num_x, num_z, x_spacing, z_spacing)
    model = mbirjax.TranslationModel((int(tv.shape[0]), n_rows, n_channels), tv,
                                     source_detector_dist=sdd, source_iso_dist=sid)
    model.set_params(delta_det_channel=delta_det, delta_det_row=delta_det)
    model.set_params(delta_voxel=delta_voxel, voxel_row_aspect=1.0, voxel_slice_aspect=1.0)
    model.set_params(recon_shape=(40, cols, slices))
    return finish(model)


# ── Stacked-gather horizontal fans (mirror each library hfan; ONE (T*P, rows) gather) ────
def ma_hfan_stacked(sinogram_view, pixel_indices, single_view_params, projector_params,
                    coeff_power=1):
    gp = projector_params.geometry_params
    num_det_rows, num_det_channels = projector_params.sinogram_shape[1:]
    azimuth = single_view_params[0]
    num_pixels = pixel_indices.shape[0]
    delta_voxel_row = gp.voxel_row_aspect * gp.delta_voxel
    row_idx, col_idx = jnp.unravel_index(pixel_indices, projector_params.recon_shape[:2])
    x, _, _ = MultiAxisParallelModel.recon_ijk_to_xyz(
        row_idx, col_idx, 0, gp.delta_voxel, gp.voxel_row_aspect, gp.voxel_slice_aspect,
        projector_params.recon_shape, gp.recon_slice_offset, azimuth)
    _, n_p = MultiAxisParallelModel.detector_uv_to_mn(
        x, 0.0, gp.delta_det_channel, gp.delta_det_row, gp.det_channel_offset,
        gp.det_row_offset, num_det_rows, num_det_channels)
    n_p_center = jnp.round(n_p).astype(int)
    footprint_xy = jnp.maximum(jnp.abs(jnp.cos(azimuth)) * gp.delta_voxel,
                               jnp.abs(jnp.sin(azimuth)) * delta_voxel_row)
    W_p_c = footprint_xy / gp.delta_det_channel
    L_max = jnp.minimum(1.0, W_p_c)
    scale = (gp.delta_voxel * delta_voxel_row) / footprint_xy
    sino_T = sinogram_view.T
    offsets = jnp.arange(-gp.psf_radius, gp.psf_radius + 1)
    n = n_p_center[None, :] + offsets[:, None]                # (psf_width, num_pixels)
    L = jnp.clip((W_p_c + 1.0) / 2.0 - jnp.abs(n_p - n), 0.0, L_max)
    A = (L * scale) * ((n >= 0) & (n < num_det_channels))
    A = A ** coeff_power
    n = jnp.clip(n, 0, num_det_channels - 1)
    gathered = sino_T[n.reshape(-1), :]
    weighted = A.reshape(-1)[:, None] * gathered
    return weighted.reshape(offsets.shape[0], num_pixels, -1).sum(axis=0)


def tr_hfan_stacked(sinogram_view, pixel_indices, single_view_params, projector_params,
                    coeff_power=1):
    gp = projector_params.geometry_params
    num_det_channels = projector_params.sinogram_shape[2]
    num_pixels = pixel_indices.shape[0]
    n_p, n_p_center, W_p_c, cos_theta_p = TranslationModel.compute_horizontal_data(
        pixel_indices, single_view_params, projector_params)
    L_max = jnp.minimum(1, W_p_c)
    delta_voxel_row = gp.voxel_row_aspect * gp.delta_voxel
    sino_T = sinogram_view.T
    offsets = jnp.arange(-gp.psf_radius, gp.psf_radius + 1)
    n = n_p_center[None, :] + offsets[:, None]                # (psf_width, num_pixels)
    L = jnp.clip((W_p_c + 1) / 2 - jnp.abs(n_p - n), 0, L_max)
    A = (delta_voxel_row * L / cos_theta_p) * ((n >= 0) & (n < num_det_channels))
    A = A ** coeff_power
    n = jnp.clip(n, 0, num_det_channels - 1)
    gathered = sino_T[n.reshape(-1), :]
    weighted = A.reshape(-1)[:, None] * gathered
    return weighted.reshape(offsets.shape[0], num_pixels, -1).sum(axis=0)


def make_full_kernel(hfan, geom_cls, band_size_const):
    """The full back kernel with a swappable horizontal fan (mirrors each library body)."""
    def kernel(sinogram_view, pixel_indices, single_view_params, projector_params,
               coeff_power=1):
        num_recon_slices = projector_params.recon_shape[2]
        num_pixels = pixel_indices.shape[0]
        rows_data = hfan(sinogram_view, pixel_indices, single_view_params,
                         projector_params, coeff_power)
        band_size = min(band_size_const, num_recon_slices)
        num_bands = (num_recon_slices + band_size - 1) // band_size
        band_starts = band_size * jnp.arange(num_bands)

        def back_one_band(g0):
            return geom_cls.back_vertical_fan_band_pixel_batch(
                rows_data, pixel_indices, single_view_params, projector_params,
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


def run_case(tag, built, geom_cls, hfan_stacked, band_size_const):
    model, pp, idx, sino_batch, view_params = built
    psf_radius = pp.geometry_params.psf_radius
    print(f"\n== {tag}  sino {tuple(model.get_params('sinogram_shape'))}  "
          f"recon {tuple(model.get_params('recon_shape'))}  P={idx.shape[0]} pixels  "
          f"psf_radius={psf_radius} ==", flush=True)
    back_lib = geom_cls.back_project_one_view_to_pixel_batch
    hfan_lib = geom_cls.back_horizontal_fan_one_view_to_pixel_batch
    num_det_rows = model.get_params('sinogram_shape')[1]

    def vmap_sum(kernel):
        @jax.jit
        def f(sino_, idx_, vp_):
            pv = jax.vmap(kernel, in_axes=(0, None, 0, None, None))(sino_, idx_, vp_, pp, 1)
            return jnp.sum(pv, axis=0)
        return f

    @jax.jit
    def vfan_only(sino_, idx_, vp_):
        dvc = jnp.ones((idx_.shape[0], num_det_rows))
        def one_view(single_vp):
            return make_full_kernel(lambda *a: dvc, geom_cls, band_size_const)(
                sino_[0], idx_, single_vp, pp, 1)
        return jnp.sum(jax.vmap(one_view)(vp_), axis=0)

    variants = [("back_asis", vmap_sum(back_lib), True),
                ("hfan_only", vmap_sum(hfan_lib), False),
                ("vfan_only", vfan_only, False),
                ("hfan_stacked_only", vmap_sum(hfan_stacked), False),
                ("full_hfan_stacked",
                 vmap_sum(make_full_kernel(hfan_stacked, geom_cls, band_size_const)), True)]
    ref, t_ref = None, None
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


def main():
    print(f"devices: {jax.devices()}   pixel_batch={PIXEL_BATCH}")
    for sino_shape in MULTIAXIS_SINO_SHAPES:
        run_case("multiaxis", build_multiaxis(sino_shape),
                 MultiAxisParallelModel, ma_hfan_stacked, MULTIAXIS_SLICE_BAND_SIZE)
    for detector in TRANSLATION_DETECTORS:
        run_case("translation", build_translation(detector),
                 TranslationModel, tr_hfan_stacked, TRANSLATION_SLICE_BAND_SIZE)
    run_case(f"translation psf x{TRANSLATION_WIDE_PSF_SCALE:g}",
             build_translation(TRANSLATION_DETECTORS[0], TRANSLATION_WIDE_PSF_SCALE),
             TranslationModel, tr_hfan_stacked, TRANSLATION_SLICE_BAND_SIZE)


if __name__ == "__main__":
    main()
