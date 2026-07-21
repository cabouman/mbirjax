"""A/B microbench: does the SORTED channel reduction still win for TRANSLATION forward at
wide psf (psf_radius 2-3, i.e. 5-7 taps)?

The crossover constant (SORTED_CHANNEL_REDUCE_MIN_COLS) and the fused stacked transient were
measured on the parallel-beam kernel at psf_width 3.  Translation often runs at large cone
angle (high magnification), where psf_radius is 2-3 -- both the sort's element count
(psf_width * num_pixels) and the scatter's tap count scale with the width, so the crossover
should be roughly stable, but this verifies it and watches the stacked transient's memory.

Per psf scale (delta_voxel multiplier -> psf_radius 1 / 2 / 3):
  hfan_scatter / hfan_sorted : the horizontal fan ALONE, both branches (the clean crossover
                               measurement; input = random (P, det_rows) rows_data)
  fwd_scatter / fwd_sorted   : the FULL forward kernel (vertical fan + hfan), both branches
                               (the composition the harness cells see; value-checked)
Memory: compiled memory_analysis() temp bytes per variant (authoritative, no subprocess
isolation needed -- this watches the (psf_width * num_pixels, det_rows) stacked transient).

Run:  python plans/experiments/projector_kernels/translation_fwd_psf_ab.py  (edit constants below).
"""
import os

# ── Run parameters (edit here) ────────────────────────────────────────────────
DETECTOR = (256, 256)               # (det_rows, det_channels); 15 views fixed (harness TCT)
PSF_SCALES = [1.0, 3.0, 5.0]        # delta_voxel multipliers -> psf_radius 1 / 2 / 3
PIXEL_BATCH = 2048
WARMUP = 2
TRIALS = 5

os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.94')
import time                       # noqa: E402
import numpy as np                # noqa: E402
import mbirjax                    # noqa: E402
import jax                        # noqa: E402
import jax.numpy as jnp           # noqa: E402
from mbirjax.translation_model import TranslationModel   # noqa: E402
from mbirjax.projectors import ProjectorParams           # noqa: E402


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
    model.configure_devices(1)
    base = (tuple(model.get_params('sinogram_shape')), tuple(model.get_params('recon_shape')),
            model.get_geometry_parameters())
    pp0 = ProjectorParams(*base, 0, 0)     # scatter branch
    pp1 = ProjectorParams(*base, 1, 0)     # sorted branch
    return model, pp0, pp1


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
    print(f"devices: {jax.devices()}   detector={DETECTOR}  pixel_batch={PIXEL_BATCH}")
    for psf_scale in PSF_SCALES:
        model, pp0, pp1 = build_translation(DETECTOR, psf_scale)
        psf_radius = pp0.geometry_params.psf_radius
        recon_shape = model.get_params('recon_shape')
        num_det_rows = model.get_params('sinogram_shape')[1]
        rng = np.random.default_rng(0)
        n_pix = min(PIXEL_BATCH, recon_shape[0] * recon_shape[1])
        idx = jnp.arange(n_pix)
        vox = jnp.asarray(rng.random((n_pix, recon_shape[2]), dtype=np.float32))
        rows_data = jnp.asarray(rng.random((n_pix, num_det_rows), dtype=np.float32))
        view_params = jnp.asarray(model.projector_functions.view_params_array)
        print(f"\n== psf_scale {psf_scale:g}  psf_radius {psf_radius} "
              f"(psf_width {2 * psf_radius + 1})  recon {tuple(recon_shape)} ==", flush=True)

        def vmapped(kernel, pp, data):
            @jax.jit
            def f(data_, idx_, vp_):
                return jax.vmap(kernel, in_axes=(None, None, 0, None))(data_, idx_, vp_, pp)
            return f, (data, idx, view_params)

        variants = [
            ("hfan_scatter", *vmapped(TranslationModel.forward_horizontal_fan_pixel_batch_to_one_view, pp0, rows_data)),
            ("hfan_sorted",  *vmapped(TranslationModel.forward_horizontal_fan_pixel_batch_to_one_view, pp1, rows_data)),
            ("fwd_scatter",  *vmapped(TranslationModel.forward_project_pixel_batch_to_one_view, pp0, vox)),
            ("fwd_sorted",   *vmapped(TranslationModel.forward_project_pixel_batch_to_one_view, pp1, vox)),
        ]
        results = {}
        for name, fn, args in variants:
            t = time_fn(fn, *args)
            # Authoritative per-program memory: compiled temp allocation (watches the
            # (psf_width * num_pixels, det_rows) stacked transient of the sorted branch).
            mem = fn.lower(*args).compile().memory_analysis()
            temp_mb = getattr(mem, 'temp_size_in_bytes', 0) / 2**20
            results[name] = (t, np.asarray(fn(*args)))
            print(f"  {name:14s} : {1e3 * t:9.2f} ms   temps {temp_mb:8.1f} MB", flush=True)
        for pair in (("hfan_scatter", "hfan_sorted"), ("fwd_scatter", "fwd_sorted")):
            t0, out0 = results[pair[0]]
            t1, out1 = results[pair[1]]
            scale = np.max(np.abs(out0)) or 1.0
            frac = np.mean(np.abs(out1 - out0) / scale > 1e-4)
            flag = "  MISMATCH!" if frac > 5e-3 else ""
            print(f"  {pair[1]:14s} vs {pair[0]:14s}: {t0 / t1:5.2f}x speedup   "
                  f"frac>1e-4={frac:.2e}{flag}", flush=True)


if __name__ == "__main__":
    main()
