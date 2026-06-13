"""
experiments/sharding/scaling_tests/cone_channel_major_ablation.py
─────────────────────────────────────────────────────────────────
Apples-to-apples ablation for the cone channel-major horizontal fans (P6
increment A).  The before/after-across-RUNS comparison is confounded by GPU
run-to-run variance (the pre/post fan-split runs showed even UNTOUCHED code moving
~1.9× between runs).  This bench instead times the OLD (row-major) and NEW
(channel-major) horizontal kernels in ONE process, interleaved, on identical inputs
— so GPU clock/occupancy is shared and run-level variance cancels.  The reported
old/new ratio is the variance-free speedup attributable to the layout change.

Both kernels call the SAME `compute_horizontal_data` (unchanged); they differ ONLY
in the accumulation layout (row-major scatter/gather vs channel-major), so this is
a true single-variable ablation.  It also asserts old≈new (tight allclose) — a
correctness cross-check that the transpose preserved the math.

Forward horizontal: (pixels, det_rows) -> (det_rows, channels) channel scatter.
Back horizontal:    (det_rows, channels) -> (pixels, det_rows) channel gather.

Run from the BETA worktree root (no CLI args; edit the config block):

    python experiments/sharding/scaling_tests/cone_channel_major_ablation.py
"""
import os
import gc
from collections import namedtuple

import numpy as np
import scaling_common as sc

import mbirjax                       # device-setup side effect before jax init
from mbirjax import ConeBeamModel
import jax
import jax.numpy as jnp
from functools import partial


# ── Run configuration (edit here; no CLI args) ────────────────────────────────
SIZES_CPU = [(64, 64, 64), (128, 128, 128), (256, 256, 256)]
SIZES_GPU = [(256, 256, 256), (512, 512, 512), (1024, 1024, 1024)]
VIEWS = 8                       # capped per size by the transient budget below
TRANSIENT_BUDGET_GB = 4.0
SDD_OVER_CHANNELS = 4.0
WARMUP, TRIALS = 2, 5           # more trials than usual: we want a tight min ratio
SEED = 0
ALLCLOSE_RTOL, ALLCLOSE_ATOL = 1e-4, 1e-4   # old vs new (computed floats, not exact)


def build_model(size):
    n_views, n_rows, n_channels = size
    angles = np.linspace(0, np.pi, n_views, endpoint=False)
    m = ConeBeamModel((n_views, n_rows, n_channels), angles,
                      source_detector_dist=SDD_OVER_CHANNELS * n_channels,
                      source_iso_dist=SDD_OVER_CHANNELS * n_channels / 2.0)
    m.set_params(verbose=0)
    return m


def make_projector_params(model):
    gp = model.get_geometry_parameters()
    sinogram_shape, recon_shape = model.get_params(['sinogram_shape', 'recon_shape'])
    PP = namedtuple('ProjectorParams', ['sinogram_shape', 'recon_shape', 'geometry_params'])
    return PP(sinogram_shape, recon_shape, gp)


# ── OLD (row-major) horizontal kernels — verbatim pre-channel-major bodies ─────
# These differ from the current ConeBeamModel kernels ONLY in the accumulation
# layout: the OLD forward scatters into sinogram_view[:, n] (stride num_det_channels)
# and the OLD back gathers sinogram_view[:, n] (same stride).  Everything else
# (compute_horizontal_data, the psf loop, the weights) is identical and shared.
def old_forward_horiz(voxel_values, pixel_indices, single_view_params, projector_params):
    gp = projector_params.geometry_params
    num_views, num_det_rows, num_det_channels = projector_params.sinogram_shape
    n_p, n_p_center, W_p_c, footprint_xy = ConeBeamModel.compute_horizontal_data(
        pixel_indices, single_view_params, projector_params)
    L_max = jnp.minimum(1, W_p_c)
    delta_voxel_row = gp.voxel_row_aspect * gp.delta_voxel
    sinogram_view = jnp.zeros((num_det_rows, num_det_channels))
    for n_offset in jnp.arange(start=-gp.psf_radius, stop=gp.psf_radius + 1):
        n = n_p_center + n_offset
        abs_delta_p_c_n = jnp.abs(n_p - n)
        L_p_c_n = jnp.clip((W_p_c + 1) / 2 - abs_delta_p_c_n, 0, L_max)
        A_chan_n = ((delta_voxel_row * gp.delta_voxel) / footprint_xy) * L_p_c_n
        A_chan_n *= (n >= 0) * (n < num_det_channels)
        sinogram_view = sinogram_view.at[:, n].add(A_chan_n.reshape((1, -1)) * voxel_values.T)
    return sinogram_view


def old_back_horiz(sinogram_view, pixel_indices, single_view_params, projector_params, coeff_power=1):
    gp = projector_params.geometry_params
    num_views, num_det_rows, num_det_channels = projector_params.sinogram_shape
    num_pixels = pixel_indices.shape[0]
    n_p, n_p_center, W_p_c, footprint_xy = ConeBeamModel.compute_horizontal_data(
        pixel_indices, single_view_params, projector_params)
    L_max = jnp.minimum(1, W_p_c)
    delta_voxel_row = gp.voxel_row_aspect * gp.delta_voxel
    det_voxel_cylinder = jnp.zeros((num_pixels, num_det_rows))
    for n_offset in jnp.arange(start=-gp.psf_radius, stop=gp.psf_radius + 1):
        n = n_p_center + n_offset
        abs_delta_p_c_n = jnp.abs(n_p - n)
        L_p_c_n = jnp.clip((W_p_c + 1) / 2 - abs_delta_p_c_n, 0, L_max)
        A_chan_n = ((delta_voxel_row * gp.delta_voxel) / footprint_xy) * L_p_c_n
        A_chan_n *= (n >= 0) * (n < num_det_channels)
        A_chan_n = A_chan_n ** coeff_power
        det_voxel_cylinder = jnp.add(det_voxel_cylinder, A_chan_n.reshape((-1, 1)) * sinogram_view[:, n].T)
    return det_voxel_cylinder


# NEW (channel-major) horizontal kernels — defined INLINE here (NOT referencing the
# installed ConeBeamModel methods) so this ablation measures the LAYOUT difference
# independent of the installed build.  If "new" pointed at the installed method and
# the build were stale (still row-major), new would equal old and the ablation would
# report a FALSE 1.00x null.  Inlining both layouts removes that failure mode; a
# separate source check (see report_installed_kernel_layout) flags a stale install.
def new_forward_horiz(voxel_values, pixel_indices, single_view_params, projector_params):
    gp = projector_params.geometry_params
    num_views, num_det_rows, num_det_channels = projector_params.sinogram_shape
    n_p, n_p_center, W_p_c, footprint_xy = ConeBeamModel.compute_horizontal_data(
        pixel_indices, single_view_params, projector_params)
    L_max = jnp.minimum(1, W_p_c)
    delta_voxel_row = gp.voxel_row_aspect * gp.delta_voxel
    sinogram_view_T = jnp.zeros((num_det_channels, num_det_rows))      # channel-major
    for n_offset in jnp.arange(start=-gp.psf_radius, stop=gp.psf_radius + 1):
        n = n_p_center + n_offset
        abs_delta_p_c_n = jnp.abs(n_p - n)
        L_p_c_n = jnp.clip((W_p_c + 1) / 2 - abs_delta_p_c_n, 0, L_max)
        A_chan_n = ((delta_voxel_row * gp.delta_voxel) / footprint_xy) * L_p_c_n
        A_chan_n *= (n >= 0) * (n < num_det_channels)
        sinogram_view_T = sinogram_view_T.at[n, :].add(A_chan_n.reshape((-1, 1)) * voxel_values)
    return sinogram_view_T.T


def new_back_horiz(sinogram_view, pixel_indices, single_view_params, projector_params, coeff_power=1):
    gp = projector_params.geometry_params
    num_views, num_det_rows, num_det_channels = projector_params.sinogram_shape
    num_pixels = pixel_indices.shape[0]
    n_p, n_p_center, W_p_c, footprint_xy = ConeBeamModel.compute_horizontal_data(
        pixel_indices, single_view_params, projector_params)
    L_max = jnp.minimum(1, W_p_c)
    delta_voxel_row = gp.voxel_row_aspect * gp.delta_voxel
    sinogram_view_T = sinogram_view.T            # channel-major (num_det_channels, num_det_rows)
    det_voxel_cylinder = jnp.zeros((num_pixels, num_det_rows))
    for n_offset in jnp.arange(start=-gp.psf_radius, stop=gp.psf_radius + 1):
        n = n_p_center + n_offset
        abs_delta_p_c_n = jnp.abs(n_p - n)
        L_p_c_n = jnp.clip((W_p_c + 1) / 2 - abs_delta_p_c_n, 0, L_max)
        A_chan_n = ((delta_voxel_row * gp.delta_voxel) / footprint_xy) * L_p_c_n
        A_chan_n *= (n >= 0) * (n < num_det_channels)
        A_chan_n = A_chan_n ** coeff_power
        det_voxel_cylinder = jnp.add(det_voxel_cylinder, A_chan_n.reshape((-1, 1)) * sinogram_view_T[n, :])
    return det_voxel_cylinder


def report_installed_kernel_layout():
    """Print whether the INSTALLED ConeBeamModel horizontal kernels are channel-major
    (the current edit) or row-major (a STALE build).  This is the build-currency check
    that distinguishes a real GPU 'no difference' from a stale-build false null."""
    import inspect
    for name, fn in [("forward_horizontal_fan_pixel_batch_to_one_view",
                      ConeBeamModel.forward_horizontal_fan_pixel_batch_to_one_view),
                     ("back_horizontal_fan_one_view_to_pixel_batch",
                      ConeBeamModel.back_horizontal_fan_one_view_to_pixel_batch)]:
        src = inspect.getsource(fn)
        layout = "channel-major (CURRENT)" if "sinogram_view_T" in src else "row-major (STALE?)"
        print(f"  installed {name}: {layout}")
    print(f"  installed mbirjax: {mbirjax.__file__}")


def vmapped(fn, idx, static_pp_argnum):
    """jit a view-vmapped version of a horizontal kernel, idx closed over."""
    @partial(jax.jit, static_argnums=(static_pp_argnum,))
    def g(arr_per_view, view_params, projector_params):
        f = lambda a, svp: fn(a, idx, svp, projector_params)
        return jax.vmap(f)(arr_per_view, view_params)
    return g


def run_one_size(size):
    model = build_model(size)
    pp = make_projector_params(model)
    recon_shape = tuple(int(x) for x in model.get_params('recon_shape'))
    n_views, n_rows, n_channels = size
    idx = jnp.asarray(mbirjax.gen_full_indices(
        recon_shape, use_ror_mask=model.get_params('use_ror_mask')))
    num_pixels = int(idx.shape[0])
    budget_views = int(TRANSIENT_BUDGET_GB * (1024 ** 3) // max(1, num_pixels * n_rows * 4))
    V = max(2, min(VIEWS, n_views, budget_views))
    vparams = jnp.asarray(model.get_params('view_params_array'))[:V]

    rng = np.random.default_rng(SEED)
    # Forward-horizontal input: the vertical-fan output shape (pixels, det_rows).
    fwd_in = jnp.asarray(rng.standard_normal((V, num_pixels, n_rows), dtype=np.float32))
    # Back-horizontal input: a sinogram view batch (det_rows, channels).
    bwd_in = jnp.asarray(rng.standard_normal((V, n_rows, n_channels), dtype=np.float32))

    of = vmapped(old_forward_horiz, idx, 2)
    nf = vmapped(new_forward_horiz, idx, 2)
    ob = vmapped(old_back_horiz, idx, 2)
    nb = vmapped(new_back_horiz, idx, 2)

    # Correctness cross-check (old ≈ new): forward outputs are (V, rows, channels);
    # back outputs (V, pixels, rows).  Computed floats -> tight allclose, not exact.
    of_out, nf_out = jax.block_until_ready(of(fwd_in, vparams, pp)), jax.block_until_ready(nf(fwd_in, vparams, pp))
    ob_out, nb_out = jax.block_until_ready(ob(bwd_in, vparams, pp)), jax.block_until_ready(nb(bwd_in, vparams, pp))
    fwd_ok = bool(np.allclose(np.asarray(of_out), np.asarray(nf_out), rtol=ALLCLOSE_RTOL, atol=ALLCLOSE_ATOL))
    bwd_ok = bool(np.allclose(np.asarray(ob_out), np.asarray(nb_out), rtol=ALLCLOSE_RTOL, atol=ALLCLOSE_ATOL))

    # Interleave old/new so any slow drift in GPU clocks affects both equally.
    t_of = t_nf = t_ob = t_nb = None
    for _ in range(1):
        t_of, _ = sc.time_op(lambda: of(fwd_in, vparams, pp), WARMUP, TRIALS)
        t_nf, _ = sc.time_op(lambda: nf(fwd_in, vparams, pp), WARMUP, TRIALS)
        t_ob, _ = sc.time_op(lambda: ob(bwd_in, vparams, pp), WARMUP, TRIALS)
        t_nb, _ = sc.time_op(lambda: nb(bwd_in, vparams, pp), WARMUP, TRIALS)

    return {
        "size": sc.size_label(size), "recon": list(recon_shape), "views": V,
        "fwd_old_ms": t_of["min_ms"], "fwd_new_ms": t_nf["min_ms"],
        "bwd_old_ms": t_ob["min_ms"], "bwd_new_ms": t_nb["min_ms"],
        "fwd_allclose": fwd_ok, "bwd_allclose": bwd_ok,
    }


def main():
    plat, _ = sc.detect_platform()
    sizes = SIZES_GPU if plat == "gpu" else SIZES_CPU
    print("=" * 78)
    print(f"  cone channel-major ablation ({plat}, same-process old-vs-new, "
          f"{WARMUP}+{TRIALS} timing)")
    print("=" * 78)
    # Build-currency check: both kernels under test are INLINE (build-independent),
    # but report whether the INSTALLED kernels are current or stale so a stale build
    # is never mistaken for a real 'no difference'.
    report_installed_kernel_layout()
    rows = []
    for size in sizes:
        r = run_one_size(size)
        rows.append(r)
        fr = r["fwd_old_ms"] / r["fwd_new_ms"]
        br = r["bwd_old_ms"] / r["bwd_new_ms"]
        print(f"\n  {r['size']}  (recon {tuple(r['recon'])}, V={r['views']})")
        print(f"    FWD horiz  old={r['fwd_old_ms']:9.3f}  new={r['fwd_new_ms']:9.3f} ms"
              f"   speedup old/new = {fr:5.2f}x   old~=new: {r['fwd_allclose']}")
        print(f"    BACK horiz old={r['bwd_old_ms']:9.3f}  new={r['bwd_new_ms']:9.3f} ms"
              f"   speedup old/new = {br:5.2f}x   old~=new: {r['bwd_allclose']}")
        gc.collect()

    sc.save_yaml(os.path.join(sc.RESULTS_DIR, f"cone_channel_major_ablation_{plat}.yaml"),
                 {"kind": "cone_channel_major_ablation", "platform": plat,
                  "views": VIEWS, "warmup": WARMUP, "trials": TRIALS,
                  "sdd_over_channels": SDD_OVER_CHANNELS, "rows": rows})
    print("\n  speedup = old(row-major)/new(channel-major), SAME process (variance-free).")
    print("\nDone.")


if __name__ == "__main__":
    main()
