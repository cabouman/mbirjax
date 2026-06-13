"""
experiments/sharding/scaling_tests/cone_fan_split_microbench.py
───────────────────────────────────────────────────────────────
P6 Step 0 companion — localize the cone back-projection cost between its two
stages and inform the A-vs-B-vs-fused structure choice (plan:
p6_projector_rework_proposal.md §8a-split, §2/§6).

Cone back projection = HORIZONTAL fan (sino rows×channels -> pixels×rows; gather
over channels, band-INDEPENDENT, materializes the pixels×rows transient) then
VERTICAL fan (pixels×rows -> pixels×slices; the natural slice axis).  The CPU
baseline showed a ~×110 back-projection cliff at 256³; this bench answers:

  1. Which stage dominates, and how does each scale with N (where is the cliff)?
  2. Does fusing the two fans into one per-view kernel (vs two separate vmapped
     calls with a materialized pixels×rows intermediate) change time — a proxy for
     whether XLA can elide the intermediate (the memory question behind the
     fused / sino-accumulator structure Greg proposed)?

Timing only, single device, CPU.  Peak-memory elision is a GPU item (CPU RSS is a
process high-water mark, a poor per-op ruler — flagged, not measured here).  This
is a focused experiment script (single role), so it imports mbirjax at top and
reuses scaling_common.time_op for identical timing discipline.

Run from the BETA worktree root (no CLI args; edit the config block):

    python experiments/sharding/scaling_tests/cone_fan_split_microbench.py
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
# CPU and GPU size ladders (GPU adds 512³/1024³).  Auto-selected by platform.
SIZES_CPU = [(64, 64, 64), (128, 128, 128), (256, 256, 256)]
SIZES_GPU = [(256, 256, 256), (512, 512, 512), (1024, 1024, 1024)]
# Max view batch for the split.  The ACTUAL view count per size is capped so the
# pixels×rows det-cylinder transient stays under TRANSIENT_BUDGET_GB (else 1024³
# with 8 views would allocate ~25 GB).  Times are normalized PER VIEW so cross-size
# scaling stays apples-to-apples even when the cap lowers V.
VIEWS_FOR_SPLIT = 8
TRANSIENT_BUDGET_GB = 4.0
SDD_OVER_CHANNELS = 4.0           # magnification 2 (matches the baseline + test suite)
WARMUP, TRIALS = 1, 3
SEED = 0


def build_model(size):
    n_views, n_rows, n_channels = size
    angles = np.linspace(0, np.pi, n_views, endpoint=False)
    m = ConeBeamModel((n_views, n_rows, n_channels), angles,
                      source_detector_dist=SDD_OVER_CHANNELS * n_channels,
                      source_iso_dist=SDD_OVER_CHANNELS * n_channels / 2.0)
    m.set_params(verbose=0)
    return m


def make_projector_params(model):
    """Rebuild the ProjectorParams namedtuple exactly as projectors.py does, so the
    cone fan static methods can be called directly (it is the hashable static arg)."""
    gp = model.get_geometry_parameters()
    sinogram_shape, recon_shape = model.get_params(['sinogram_shape', 'recon_shape'])
    PP = namedtuple('ProjectorParams', ['sinogram_shape', 'recon_shape', 'geometry_params'])
    return PP(sinogram_shape, recon_shape, gp)


def run_one_size(size):
    model = build_model(size)
    pp = make_projector_params(model)
    recon_shape = tuple(int(x) for x in model.get_params('recon_shape'))
    n_views, n_rows, n_channels = size
    idx = mbirjax.gen_full_indices(recon_shape, use_ror_mask=model.get_params('use_ror_mask'))
    idx = jnp.asarray(idx)
    num_pixels = int(idx.shape[0])
    num_slices = recon_shape[2]
    # Cap the view batch so the pixels×rows det-cylinder transient stays under
    # budget (1024³: pix·rows·4B ≈ 3.2 GB/view).  Times are reported per-view.
    budget_views = int(TRANSIENT_BUDGET_GB * (1024 ** 3) // max(1, num_pixels * n_rows * 4))
    V = max(2, min(VIEWS_FOR_SPLIT, n_views, budget_views))

    vparams = jnp.asarray(model.get_params('view_params_array'))[:V]   # (V, 2): angle, z_shift
    rng = np.random.default_rng(SEED)
    sino_vb = jnp.asarray(rng.standard_normal((V, n_rows, n_channels), dtype=np.float32))

    H = ConeBeamModel.back_horizontal_fan_one_view_to_pixel_batch
    Vfan = ConeBeamModel.back_vertical_fan_one_view_to_pixel_batch
    Full = ConeBeamModel.back_project_one_view_to_pixel_batch

    # HORIZONTAL only: vmap over views -> (V, num_pixels, num_rows) det cylinders.
    @partial(jax.jit, static_argnums=(2,))
    def horiz_only(sino_views, view_params, projector_params):
        f = lambda sv, svp: H(sv, idx, svp, projector_params, 1)
        return jax.vmap(f)(sino_views, view_params)

    det_cyls = horiz_only(sino_vb, vparams, pp)
    det_cyls = jax.block_until_ready(det_cyls)

    # VERTICAL only: vmap over views of the vertical fan on the det cylinders, summed
    # over views -> (num_pixels, num_slices).  Matches how the projector sums views.
    @partial(jax.jit, static_argnums=(2,))
    def vert_only(det_cylinders, view_params, projector_params):
        f = lambda dc, svp: Vfan(dc, idx, svp, projector_params, 1)
        return jnp.sum(jax.vmap(f)(det_cylinders, view_params), axis=0)

    # FUSED: one per-view kernel does horizontal THEN vertical internally, summed
    # over views.  Two separate calls (horiz_only then vert_only) materialize the
    # (V, num_pixels, num_rows) intermediate across a function boundary; fusing gives
    # XLA the chance to elide it.  Equal time => fusion buys little; faster => the
    # intermediate's materialization/traffic was costing us.
    @partial(jax.jit, static_argnums=(2,))
    def fused(sino_views, view_params, projector_params):
        f = lambda sv, svp: Full(sv, idx, svp, projector_params, 1)
        return jnp.sum(jax.vmap(f)(sino_views, view_params), axis=0)

    t_h, _ = sc.time_op(lambda: horiz_only(sino_vb, vparams, pp), WARMUP, TRIALS)
    t_v, _ = sc.time_op(lambda: vert_only(det_cyls, vparams, pp), WARMUP, TRIALS)
    t_f, _ = sc.time_op(lambda: fused(sino_vb, vparams, pp), WARMUP, TRIALS)

    # ── FORWARD split (vertical then horizontal) ──────────────────────────────
    # Forward vertical: (pixels, slices) -> (pixels, det_rows), same recon for all
    # views (vmap only over view params).  Forward horizontal: (pixels, det_rows)
    # -> (det_rows, channels), the channel SCATTER (the channel-major question).
    FwdV = ConeBeamModel.forward_vertical_fan_pixel_batch_to_one_view
    FwdH = ConeBeamModel.forward_horizontal_fan_pixel_batch_to_one_view
    cylinders = jnp.asarray(rng.standard_normal((num_pixels, num_slices), dtype=np.float32))

    @partial(jax.jit, static_argnums=(2,))
    def fwd_vert_only(voxels, view_params, projector_params):
        f = lambda svp: FwdV(voxels, idx, svp, projector_params)
        return jax.vmap(f)(view_params)                       # (V, pixels, det_rows)

    fwd_det_cyls = jax.block_until_ready(fwd_vert_only(cylinders, vparams, pp))

    @partial(jax.jit, static_argnums=(2,))
    def fwd_horiz_only(det_cylinders, view_params, projector_params):
        f = lambda dc, svp: FwdH(dc, idx, svp, projector_params)
        return jax.vmap(f)(det_cylinders, view_params)        # (V, det_rows, channels)

    t_fv, _ = sc.time_op(lambda: fwd_vert_only(cylinders, vparams, pp), WARMUP, TRIALS)
    t_fh, _ = sc.time_op(lambda: fwd_horiz_only(fwd_det_cyls, vparams, pp), WARMUP, TRIALS)

    transient_mb = V * num_pixels * n_rows * 4 / (1024 ** 2)   # pixels×rows intermediate
    return {
        "size": sc.size_label(size), "recon": list(recon_shape),
        "views_timed": V, "num_pixels": num_pixels, "num_slices": num_slices,
        "horiz_ms": t_h["min_ms"], "vert_ms": t_v["min_ms"], "fused_ms": t_f["min_ms"],
        "separate_ms": t_h["min_ms"] + t_v["min_ms"],
        "fwd_vert_ms": t_fv["min_ms"], "fwd_horiz_ms": t_fh["min_ms"],
        "pixels_rows_transient_mb": transient_mb,
    }


def main():
    plat, _ = sc.detect_platform()
    sizes = SIZES_GPU if plat == "gpu" else SIZES_CPU
    print("=" * 78)
    print(f"  cone fan split micro-bench ({plat}, <= {VIEWS_FOR_SPLIT}-view batch, "
          f"{WARMUP}+{TRIALS} timing; times shown PER VIEW)")
    print("=" * 78)
    rows = []
    for size in sizes:
        r = run_one_size(size)
        # Per-view normalization (V is capped by the transient budget and may vary
        # with size, so per-view keeps cross-size scaling apples-to-apples).
        V = r["views_timed"]
        for k in ("horiz_ms", "vert_ms", "fused_ms", "separate_ms",
                  "fwd_vert_ms", "fwd_horiz_ms"):
            r[k + "_per_view"] = r[k] / V
        rows.append(r)
        print(f"\n  {r['size']}  (recon {tuple(r['recon'])}, pix={r['num_pixels']}, "
              f"V={V})    [ms per view]")
        print(f"    BACK  horizontal : {r['horiz_ms_per_view']:9.3f}")
        print(f"    BACK  vertical   : {r['vert_ms_per_view']:9.3f}   (H/V = {r['horiz_ms']/r['vert_ms']:.2f})")
        print(f"    BACK  separate   : {r['separate_ms_per_view']:9.3f}"
              f"   fused = {r['fused_ms_per_view']:9.3f}  (fused/sep = {r['fused_ms']/r['separate_ms']:.2f})")
        print(f"    FWD   vertical   : {r['fwd_vert_ms_per_view']:9.3f}")
        print(f"    FWD   horizontal : {r['fwd_horiz_ms_per_view']:9.3f}   "
              f"(H/V = {r['fwd_horiz_ms']/r['fwd_vert_ms']:.2f})")
        print(f"    pixels×rows transient    : {r['pixels_rows_transient_mb']:8.1f} MB "
              f"(for {V} views)")
        gc.collect()   # release this size's arrays before the next (GPU residue hygiene)

    # Scaling of each stage (where is the cliff?), on PER-VIEW times.
    print("\n  " + "=" * 66)
    print("  per-stage scaling (x factor vs previous size, PER VIEW; ideal N^4 = x16)")
    for i in range(1, len(rows)):
        a, b = rows[i - 1], rows[i]
        f = lambda k: b[k + "_per_view"] / a[k + "_per_view"]
        print(f"    {a['size']} -> {b['size']}: "
              f"BACK horiz x{f('horiz_ms'):.1f} vert x{f('vert_ms'):.1f}  |  "
              f"FWD vert x{f('fwd_vert_ms'):.1f} horiz x{f('fwd_horiz_ms'):.1f}")

    sc.save_yaml(os.path.join(sc.RESULTS_DIR, f"cone_fan_split_{plat}.yaml"),
                 {"kind": "cone_fan_split", "platform": plat,
                  "views_for_split": VIEWS_FOR_SPLIT,
                  "transient_budget_gb": TRANSIENT_BUDGET_GB,
                  "warmup": WARMUP, "trials": TRIALS,
                  "sdd_over_channels": SDD_OVER_CHANNELS, "rows": rows})
    print("\n  NOTE: fused-vs-separate PEAK MEMORY (does XLA elide the pixels×rows "
          "intermediate?) is deprioritized — fusion showed no time win on CPU; the "
          "baseline driver's per-device peak_bytes_in_use is the memory ruler.")
    print("\nDone.")


if __name__ == "__main__":
    main()
