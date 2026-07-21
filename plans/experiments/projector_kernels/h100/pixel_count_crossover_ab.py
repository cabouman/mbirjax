"""A/B microbench: the sorted channel reduce vs scatter as a function of PIXEL COUNT.

Motivation: the 15x257x257 translation forward cell regressed 0.86x with sort_by_channel on,
while 15x256x256 won 1.20x.  Hypothesis: 257 gives 40*257 = 10280 pixels = 5*2048 + a RAGGED
40-pixel batch, and the sorted reduce's fixed sort cost cannot amortize over 40 pixels (the
same mechanism as the cone VCD watch item: forward on small pixel subsets).  If confirmed,
the clean fix is a static num_pixels guard INSIDE channel_scatter_reduce (trace-time branch
on a static shape; the tiny batch falls back to scatter within the same compiled program).

Part 1 (crossover): the translation forward kernel, sorted vs scatter, across pixel counts
at the 257-detector shape, vmapped over the 15 views (the production composition).  This
locates the num_pixels crossover for the guard threshold.
Part 2 (mechanism confirm, model level): sparse_forward_project at 15x257x257 with the
default fwd_pixel_batch (2048 -> ragged 40) vs 2056 (10280 = 5*2056, no ragged batch), both
flag states.  If the hypothesis holds, 2056 recovers the sorted win.

Run:  python plans/experiments/projector_kernels/pixel_count_crossover_ab.py
"""
import os

# ── Run parameters (edit here) ────────────────────────────────────────────────
# RESULTS (H100, 2026-07-08, three runs; the numbers live in fwd_back_findings.md):
#  1. detector 257: the ragged-batch hypothesis is REFUTED -- Part 2 showed batch 2056
#     (ragged=0) does NOT recover the sorted win, and Part 1's pixel-count sweep is
#     non-monotonic (0.93-1.20x).
#  2. square detectors 256..1025: sorted wins 0.65-0.90x at powers of 2, ties/loses
#     slightly at odd sizes (sub-ms kernels; second-order).
#  3. the REAL TCT shapes (1936x3064, 1883x3064/3065): sorted is 4.5-6.5x SLOWER -- the
#     channel-collision cliff (psf_width * num_pixels / num_det_channels ~ 2: the scatter
#     has almost no duplicate-channel collisions to eliminate, and XLA's near-empty
#     segment-sum lowering is pathological).  CONCLUSION: translation's policy keeps the
#     scatter path; SORTED_CHANNEL_REDUCE_MIN_COLLISION_RATIO = 4 guards multiaxis.
DETECTOR = (257, 257)                # (det_rows, det_channels); 15 views fixed (harness TCT)
DETECTORS = [(1936, 3064), (1883, 3064), (1883, 3065)]   # run-3 set; run 2 used 256..1025 squares
PIXEL_COUNTS = [1024, 2048]
MODEL_PIXEL_BATCHES = [2048, 2056]   # 2056 divides 40*257 = 10280 exactly
RUN_MODEL_LEVEL = False              # Part 2 answered on the first run
WARMUP = 2
TRIALS = 10

os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.94')
import time                       # noqa: E402
import numpy as np                # noqa: E402
import mbirjax                    # noqa: E402
import jax                        # noqa: E402
import jax.numpy as jnp           # noqa: E402
from mbirjax.translation_model import TranslationModel   # noqa: E402
from mbirjax.projectors import ProjectorParams           # noqa: E402


def build_translation(detector):
    n_rows, n_channels = detector
    sdd, sid, delta_det = 190000.0, 70000.0, 75.0          # harness TCT geometry
    num_x, num_z = 5, 3
    delta_voxel = delta_det / (sdd / sid)
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
    return model, ProjectorParams(*base, 0, 0), ProjectorParams(*base, 1, 0)


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
    print(f"devices: {jax.devices()}")
    kern = TranslationModel.forward_project_pixel_batch_to_one_view
    rng = np.random.default_rng(0)

    print("\n-- Part 1: kernel-level sorted vs scatter, detector-size x pixel-count sweep "
          "(vmap over the 15 views) --")
    for detector in DETECTORS:
        model, pp0, pp1 = build_translation(detector)
        recon_shape = model.get_params('recon_shape')
        view_params = jnp.asarray(model.projector_functions.view_params_array)
        for n_pix in PIXEL_COUNTS:
            idx = jnp.arange(n_pix)
            vox = jnp.asarray(rng.random((n_pix, recon_shape[2]), dtype=np.float32))

            def vmapped(pp):
                @jax.jit
                def f(vox_, idx_, vp_):
                    return jax.vmap(kern, in_axes=(None, None, 0, None))(vox_, idx_, vp_, pp)
                return f

            t0 = time_fn(vmapped(pp0), vox, idx, view_params)
            t1 = time_fn(vmapped(pp1), vox, idx, view_params)
            print(f"  det={detector[0]:5d}  P={n_pix:5d} : scatter {1e3 * t0:8.3f} ms   "
                  f"sorted {1e3 * t1:8.3f} ms   sorted/scatter {t1 / t0:5.2f}x", flush=True)

    if not RUN_MODEL_LEVEL:
        return
    model, pp0, pp1 = build_translation(DETECTOR)
    recon_shape = model.get_params('recon_shape')
    print("\n-- Part 2: model-level sparse_forward_project, pixel-batch ablation --")
    idx_full = mbirjax.gen_full_indices(recon_shape, use_ror_mask=False)
    vox_full = jnp.asarray(rng.random((len(idx_full), recon_shape[2]), dtype=np.float32))
    print(f"  full grid: {len(idx_full)} pixels")
    for batch in MODEL_PIXEL_BATCHES:
        for flag, label in ((False, 'scatter'), (True, 'sorted ')):
            model.tiles = model.tiles._replace(fwd_pixel_batch=batch, sort_by_channel=flag)
            # sort_by_channel is baked into static ProjectorParams at projector creation,
            # so force a rebuild to pick up the flag change.
            model.projector_functions.create_projectors(model)
            fn = model.sparse_forward_project
            t = time_fn(fn, vox_full, idx_full)
            ragged = len(idx_full) % batch
            print(f"  batch={batch}  ragged={ragged:5d}  {label} : {1e3 * t:8.2f} ms",
                  flush=True)


if __name__ == "__main__":
    main()
