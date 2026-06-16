"""
experiments/sharding/scaling_tests/cone_forward_structure_compare.py
────────────────────────────────────────────────────────────────────
Direct, empirical comparison of three CONE forward-projection structures, to choose
the sharded-forward design by measurement (not formulas).  One view-owner produces a
batch of views from a recon whose slice axis is split across ``n`` devices -- the
setting where the structures differ in per-device peak memory.

  * MONO -- monolithic forward, full cylinder already local on the owner (the
            single-device reference / "current").  vertical(all slices) -> horizontal.
  * C    -- all-gather: the owner GATHERS all slice shards into the full cylinder
            (P x S), then runs the monolithic forward.  Simple; holds the full
            cylinder on the owner.
  * B    -- band-stream: the owner STREAMS one slice band at a time from its
            slice-owner, runs a WINDOWED vertical fan (only the detector rows the
            band reaches) into a det-cylinder accumulator (V x P_b x R), then ONE
            horizontal fan.  Holds a band (P_b x L) + the accumulator; never the
            full cylinder.

The recon is created as per-device slice shards (the full cylinder is NOT resident
on the owner), so C's gather and B's stream show their real per-device memory.

Two worker modes, each a fresh subprocess (clean peak_bytes_in_use):
  * correct -- compute MONO, C, B and assert C,B agree with MONO (allclose).
  * measure --variant -- run ONLY that variant; report wall time + per-device peak.

Circular scan (zero z-shift) so the band row-window is shared across views (helical
correctness is gated in tests/geometries/test_cone_banded.py).

Run from the BETA worktree root (no CLI args; edit the config block):

    python experiments/sharding/scaling_tests/cone_forward_structure_compare.py
"""
import os
import sys
import gc
import argparse
from collections import namedtuple

import scaling_common as sc
import numpy as np

# mbirjax/jax imported only inside workers (orchestrator stays JAX-free).


# ── Run configuration (edit here; no CLI args) ────────────────────────────────
DEVICE_COUNTS = [2, 4]                  # n=1 is degenerate (nothing to gather/stream)
SIZES = {  # sinogram (n_views, n_det_rows, n_det_channels); recon auto-derived
    "cpu": [(8, 64, 64), (8, 128, 128), (8, 256, 256)],
    "gpu": [(8, 256, 256), (8, 512, 512), (8, 1024, 1024)],
}
VIEWS_PER_OWNER = 4          # views the measured owner produces
PIXEL_BATCH = 8192           # pixels per batch (bounds the V x P_b x R transient)
BANDS_PER_SHARD = 2          # sub-bands within each device's slice shard (B)
SDD_OVER_CHANNELS = 4.0      # magnification 2
WINDOW_MARGIN = 4            # extra detector rows beyond the worst-case band window
WARMUP, TRIALS = 1, 3
SEED = 0
VARIANTS = ("mono", "C", "B")
CORRECTNESS_TOL = 1e-3       # max_abs(B or C vs MONO); computed floats


# ── Geometry / model ──────────────────────────────────────────────────────────
def make_model(size):
    import mbirjax
    n_views, n_rows, n_channels = size
    angles = np.linspace(0, np.pi, n_views, endpoint=False)   # circular (z-shift 0)
    sdd = SDD_OVER_CHANNELS * n_channels
    model = mbirjax.ConeBeamModel((n_views, n_rows, n_channels), angles,
                                  source_detector_dist=sdd, source_iso_dist=sdd / 2.0)
    model.set_params(verbose=0)
    return model


def make_projector_params(model):
    gp = model.get_geometry_parameters()
    sinogram_shape, recon_shape = model.get_params(['sinogram_shape', 'recon_shape'])
    PP = namedtuple('ProjectorParams', ['sinogram_shape', 'recon_shape', 'geometry_params'])
    return PP(sinogram_shape, recon_shape, gp)


def balanced_bounds(extent, num_bands):
    base, rem = divmod(extent, num_bands)
    bounds, start = [], 0
    for k in range(num_bands):
        length = base + (1 if k < rem else 0)
        bounds.append((start, start + length))
        start += length
    return bounds


def shard_bounds(S, n_dev):
    """Slice-shard bounds (each device owns a contiguous slice range)."""
    return balanced_bounds(S, n_dev)


def all_band_bounds(S, n_dev):
    """Global band bounds: BANDS_PER_SHARD sub-bands inside each device's shard, so
    every band is contained in one shard (it can be streamed from one device)."""
    bounds = []
    for (s0, s1) in shard_bounds(S, n_dev):
        for (b0, b1) in balanced_bounds(s1 - s0, min(BANDS_PER_SHARD, s1 - s0)):
            bounds.append((s0 + b0, s0 + b1))
    return bounds


def magnification_bounds(model):
    M = float(model.get_magnification())
    sdd = float(model.get_params('source_detector_dist'))
    recon_shape = model.get_params('recon_shape')
    dv = float(model.get_params('delta_voxel'))
    vra = float(model.get_params('voxel_row_aspect'))
    half = 0.5 * max(recon_shape[0] * vra * dv, recon_shape[1] * dv)
    if np.isinf(sdd):
        return M, M
    return 1.0 / (1.0 / M + half / sdd), 1.0 / (1.0 / M - half / sdd)


def band_windows(model, pp, bounds):
    """Per-band detector-row window [r0, r0+W) (circular scan), over all pixels
    (worst-case magnification).  r0 per band (clipped in-detector); one static W."""
    gp = pp.geometry_params
    R = pp.sinogram_shape[1]
    S = pp.recon_shape[2]
    dvs = float(gp.voxel_slice_aspect) * float(gp.delta_voxel)
    delta_row = float(gp.delta_det_row)
    row_off = float(gp.det_row_offset)
    det_center = (R - 1) / 2.0
    offset = float(gp.recon_slice_offset)
    mag_min, mag_max = magnification_bounds(model)
    m_of = lambda mag, z: (mag * z + row_off) / delta_row + det_center
    spans, m_los = [], []
    for (g0, g1) in bounds:
        z_lo = dvs * (g0 - (S - 1) / 2.0) + offset
        z_hi = dvs * (g1 - 1 - (S - 1) / 2.0) + offset
        corners = [m_of(mag, z) for mag in (mag_min, mag_max) for z in (z_lo, z_hi)]
        m_los.append(min(corners))
        spans.append(max(corners) - min(corners))
    bp = int(gp.bp_psf_radius)
    W = min(int(np.ceil(max(spans))) + 2 * bp + 2 * WINDOW_MARGIN, R)
    r0_list = [int(np.clip(int(np.floor(m_lo)) - bp - WINDOW_MARGIN, 0, R - W)) for m_lo in m_los]
    return r0_list, W


# ── Kernels ─────────────────────────────────────────────────────────────────
def forward_vertical_windowed_one_pixel(band_values, pixel_mag, svp, pp, g0, r0, W):
    """Windowed forward vertical fan: a band of L slices -> (W,) contribution to
    detector rows [r0, r0+W).  Anchored on the problem slice count S and global index
    k = g0 + arange(L); same physics as the monolithic vertical, restricted to the
    window rows.

    Takes ``pixel_mag`` PRECOMPUTED (it depends on the pixel + view, NOT the band), so
    the band loop does not recompute the per-pixel geometry every band -- the
    optimization that makes B competitive (without it B was ~10x slower on CPU)."""
    import jax.numpy as jnp
    helical_z_shift = svp[1]
    gp = pp.geometry_params
    R = pp.sinogram_shape[1]
    S = pp.recon_shape[2]
    L = band_values.shape[0]
    dvs = gp.voxel_slice_aspect * gp.delta_voxel

    k_global = g0 + jnp.arange(L)
    z = dvs * (k_global - (S - 1) / 2.0) + (gp.recon_slice_offset - helical_z_shift)
    cos_phi = jnp.cos(jnp.arctan2(pixel_mag * z, gp.source_detector_dist))
    band_valid = (k_global >= 0) & (k_global < S)
    scaled = jnp.where(band_valid, band_values, 0.0) / cos_phi

    W_p_r = (pixel_mag * dvs) / gp.delta_det_row
    slope = W_p_r
    L_max = jnp.minimum(1, W_p_r)
    det_center = (R - 1) / 2.0
    m_center = r0 + jnp.arange(W)
    v_m = (m_center - det_center) * gp.delta_det_row - gp.det_row_offset
    z_m = v_m / pixel_mag
    k_m_local = (z_m - (gp.recon_slice_offset - helical_z_shift)) / dvs + (S - 1) / 2.0 - g0
    k_m_center = jnp.round(k_m_local).astype(int)
    m_p = slope * (k_m_center - k_m_local[0]) + m_center[0]

    out = jnp.zeros(W)
    for k_off in jnp.arange(start=-gp.bp_psf_radius, stop=gp.bp_psf_radius + 1):
        k_local = k_m_center + k_off
        k_glob = k_local + g0
        absd = jnp.abs(m_p + slope * k_off - m_center)
        A = jnp.clip((W_p_r + 1) / 2 - absd, 0, L_max)
        A *= (k_glob >= 0) * (k_glob < S)
        A *= (k_local >= 0) * (k_local < L)
        A *= (m_center >= 0) * (m_center < R)
        out = out + A * scaled[k_local]
    return out


# The three structures are defined as JITTED computations inside worker() (so the
# orchestrator can import this module without initializing a JAX backend).


# ── Worker ──────────────────────────────────────────────────────────────────
def _setup(size, n_dev):
    """Build model/params/indices/views/sharded-recon for a size and device count."""
    import mbirjax, jax
    import jax.numpy as jnp
    model = make_model(size)
    pp = make_projector_params(model)
    recon_shape = tuple(int(x) for x in model.get_params('recon_shape'))
    S = recon_shape[2]
    idx = jnp.asarray(mbirjax.gen_full_indices(recon_shape, use_ror_mask=model.get_params('use_ror_mask')))
    P = int(idx.shape[0])
    Vp = min(VIEWS_PER_OWNER, size[0])
    view_params = jnp.asarray(model.get_params('view_params_array'))[:Vp]
    rng = np.random.default_rng(SEED)
    voxels_host = rng.standard_normal((P, S), np.float32)
    devices = sc.pick_devices(n_dev)
    # Slice-shard the recon: shard s lives ONLY on device s (full cylinder never on owner).
    shards = [(jax.device_put(voxels_host[:, s0:s1], devices[s]), (s0, s1))
              for s, (s0, s1) in enumerate(shard_bounds(S, n_dev))]
    return model, pp, idx, view_params, voxels_host, devices, S


def worker(mode, size_label, n_dev, variant, warmup, trials, out_file):
    import mbirjax  # device-setup side effect; must precede jax init
    import jax
    import jax.numpy as jnp
    import mbirjax._sharding as mjs
    from functools import partial
    size = tuple(int(x) for x in size_label.split("x"))
    model, pp, idx, view_params, voxels_host, devices, S = _setup(size, n_dev)
    dev2dev_safe = mjs.is_dev2dev_safe(devices)
    owner = devices[0]
    R, V, P = pp.sinogram_shape[1], int(view_params.shape[0]), int(idx.shape[0])
    bounds = all_band_bounds(S, n_dev)
    r0_list, W = band_windows(model, pp, bounds)
    FWD1 = mbirjax.ConeBeamModel.forward_project_pixel_batch_to_one_view
    HORIZ1 = mbirjax.ConeBeamModel.forward_horizontal_fan_pixel_batch_to_one_view
    idx_owner = jax.device_put(idx, owner)
    vp_owner = jax.device_put(view_params, owner)

    # ── the three structures, JITTED so the comparison is apples-to-apples ──────
    MAG = mbirjax.ConeBeamModel.compute_y_mag_for_pixel

    @jax.jit                                    # monolithic per-pixel-batch forward
    def mono_pb(vb, ib):
        return jax.vmap(lambda svp: FWD1(vb, ib, svp, pp))(vp_owner)        # (V, R, C)

    @jax.jit                                    # per-(view, pixel) magnification, ONCE
    def precompute_mag(ib):
        def per_view(svp):
            return jax.vmap(lambda pix: MAG(pix, svp[0], pp.recon_shape, pp)[1])(ib)  # (n_p,)
        return jax.vmap(per_view)(vp_owner)                                 # (V, n_p)

    @partial(jax.jit, donate_argnums=0)         # windowed vertical of one band -> acc
    def b_add(acc, band, mags, g0, r0):
        # ``mags`` (V, n_p) is the precomputed per-(view, pixel) magnification, so the
        # per-pixel geometry is NOT recomputed for every band (the hoist).
        def per_view(svp, mag_v):
            return jax.vmap(lambda bv, m: forward_vertical_windowed_one_pixel(
                bv, m, svp, pp, g0, r0, W), in_axes=(0, 0))(band, mag_v)
        win = jax.vmap(per_view)(vp_owner, mags)                            # (V, n_p, W)
        cur = jax.lax.dynamic_slice(acc, (0, 0, r0), win.shape)
        return jax.lax.dynamic_update_slice(acc, cur + win, (0, 0, r0))

    @jax.jit                                    # horizontal fan once per pixel batch
    def b_horiz(acc, ib):
        return jax.vmap(lambda svp, dc: HORIZ1(dc, ib, svp, pp))(vp_owner, acc)

    def make_shards():
        return [(jax.device_put(voxels_host[:, s0:s1], devices[s]), (s0, s1))
                for s, (s0, s1) in enumerate(shard_bounds(S, n_dev))]

    def shard_of(g0, shards):
        for i, (_d, (s0, s1)) in enumerate(shards):
            if s0 <= g0 < s1:
                return i, s0
        raise ValueError(f"band start {g0} not in a shard")

    def from_full(full_cyl):                    # MONO / C compute from a local cylinder
        out = None
        for p0 in range(0, P, PIXEL_BATCH):
            part = mono_pb(full_cyl[p0:p0 + PIXEL_BATCH], idx_owner[p0:p0 + PIXEL_BATCH])
            out = part if out is None else out + part
        return out

    def do_mono():
        return from_full(jax.device_put(voxels_host, owner))

    def do_C():                                 # gather (per pixel-batch), monolithic
        shards = make_shards()
        out = None
        for p0 in range(0, P, PIXEL_BATCH):
            # Gather only THIS pixel batch's slices from each shard (bounds the
            # gathered cylinder to (P_b, S), the fair / best-C memory).
            full_pb = jnp.concatenate(
                [mjs.move_shard(d[p0:p0 + PIXEL_BATCH], owner, dev2dev_safe=dev2dev_safe)
                 for d, _ in shards], axis=1)
            part = mono_pb(full_pb, idx_owner[p0:p0 + PIXEL_BATCH])
            out = part if out is None else out + part
        return out

    def do_B():                                 # stream bands, windowed accumulate
        shards = make_shards()
        out = None
        for p0 in range(0, P, PIXEL_BATCH):
            ib = idx_owner[p0:p0 + PIXEL_BATCH]
            n_p = int(ib.shape[0])
            mags = precompute_mag(ib)                      # (V, n_p), ONCE for all bands
            acc = jnp.zeros((V, n_p, R), device=owner)
            for (g0, g1), r0 in zip(bounds, r0_list):
                si, s0 = shard_of(g0, shards)
                data, _ = shards[si]
                band = mjs.move_shard(data[p0:p0 + PIXEL_BATCH, (g0 - s0):(g1 - s0)],
                                      owner, dev2dev_safe=dev2dev_safe)
                acc = b_add(acc, band, mags, g0, r0)
            part = b_horiz(acc, ib)
            out = part if out is None else out + part
        return out

    runners = {"mono": do_mono, "C": do_C, "B": do_B}

    if mode == "correct":
        ref = np.asarray(jax.block_until_ready(do_mono()))
        outC = np.asarray(jax.block_until_ready(do_C()))
        outB = np.asarray(jax.block_until_ready(do_B()))
        sc.write_worker_result(out_file, {
            "size": size_label, "n_dev": n_dev, "num_bands": len(bounds), "W": int(W),
            "C_max_abs": float(np.max(np.abs(outC - ref))),
            "B_max_abs": float(np.max(np.abs(outB - ref))),
            "shapes_ok": bool(outC.shape == ref.shape and outB.shape == ref.shape)})
        return

    # measure: run ONLY the chosen variant (clean per-variant peak).
    stats, _ = sc.time_op(runners[variant], warmup, trials)
    mem_mb, mem_kind = sc.peak_memory_mb(devices)
    sc.write_worker_result(out_file, {"size": size_label, "n_dev": n_dev, "variant": variant,
                                      "min_ms": stats["min_ms"], "mem_mb": mem_mb, "mem_kind": mem_kind})


def worker_setup(out_file):
    """Report platform + device count (JAX touched only here, in a subprocess)."""
    import mbirjax  # device-setup side effect; must precede jax init
    plat, max_dev = sc.detect_platform()
    sc.write_worker_result(out_file, {"platform": plat, "max_devices": max_dev,
                                      "device_label": sc.device_label()})


def run_worker(argv):
    p = argparse.ArgumentParser(description="forward-structure compare worker (internal)")
    p.add_argument("--worker", action="store_true")
    p.add_argument("--mode", choices=["setup", "correct", "measure"], required=True)
    p.add_argument("--size", default="0x0x0")
    p.add_argument("--n-dev", type=int, default=1)
    p.add_argument("--variant", default="mono")
    p.add_argument("--warmup", type=int, default=WARMUP)
    p.add_argument("--trials", type=int, default=TRIALS)
    p.add_argument("--out-file", required=True)
    a = p.parse_args(argv)
    if a.mode == "setup":
        worker_setup(a.out_file)
    else:
        worker(a.mode, a.size, a.n_dev, a.variant, a.warmup, a.trials, a.out_file)


# ── Orchestrator ──────────────────────────────────────────────────────────────
def main():
    script = os.path.abspath(__file__)
    worker_env = sc.build_worker_env()
    # Set the CPU virtual-device count BEFORE the (JAX-free) orchestrator asks the
    # setup worker for the platform; ignored on GPU.  Keeps the orchestrator JAX-free
    # (the setup worker is a subprocess) so it holds no device backend during measures.
    worker_env["MBIRJAX_NUM_CPU_DEVICES"] = str(max(DEVICE_COUNTS))
    setup, _ = sc.run_worker(script, ["--worker", "--mode", "setup"], extra_env=worker_env)
    if not setup:
        print("  ERROR: setup worker produced no result; aborting.")
        return
    plat, max_dev = setup["platform"], setup["max_devices"]
    sizes = SIZES[plat]
    size_labels = [sc.size_label(s) for s in sizes]
    device_counts = [n for n in DEVICE_COUNTS if n <= max_dev]
    if not device_counts:
        print(f"  ERROR: no device counts in {DEVICE_COUNTS} fit max_dev={max_dev}; aborting.")
        return
    print("=" * 78)
    print(f"  cone forward-structure compare ({plat})  sizes={size_labels}  n_dev={device_counts}")
    print(f"  (per view-owner: {VIEWS_PER_OWNER} views; pixel batch {PIXEL_BATCH}; "
          f"{BANDS_PER_SHARD} bands/shard)")
    print("=" * 78)

    grid = {}
    for label in size_labels:
        for n in device_counts:
            # Correctness (one subprocess) then a measure subprocess per variant.
            cres, _ = sc.run_worker(script, ["--worker", "--mode", "correct", "--size", label,
                                             "--n-dev", str(n)], extra_env=worker_env)
            cells = {}
            for variant in VARIANTS:
                mres, _ = sc.run_worker(script, ["--worker", "--mode", "measure", "--size", label,
                                                 "--n-dev", str(n), "--variant", variant],
                                        extra_env=worker_env)
                cells[variant] = mres or {}
            grid[(label, n)] = {"correct": cres or {}, "measure": cells}
            c = cres or {}
            corr_ok = c.get("shapes_ok") and max(c.get("C_max_abs", 9), c.get("B_max_abs", 9)) < CORRECTNESS_TOL
            line = f"\n  {label}  n={n}  bands={c.get('num_bands','?')} W={c.get('W','?')}  " \
                   f"correct={'OK' if corr_ok else 'FAIL ' + str((c.get('C_max_abs'), c.get('B_max_abs')))}"
            print(line)
            for variant in VARIANTS:
                m = cells[variant]
                if m:
                    print(f"      {variant:<5s} {m.get('min_ms', float('nan')):9.1f} ms   "
                          f"peak {m.get('mem_mb', float('nan')):9.1f} MB ({m.get('mem_kind','?')})")
                else:
                    print(f"      {variant:<5s} (no result)")
            # B-vs-C summary
            mB, mC = cells.get("B", {}), cells.get("C", {})
            if mB and mC and mC.get("mem_mb"):
                print(f"      -> B/C time {mB['min_ms']/mC['min_ms']:.2f}x   "
                      f"B/C peak {mB['mem_mb']/mC['mem_mb']:.2f}x")

    sc.save_yaml(os.path.join(sc.RESULTS_DIR, f"cone_forward_structure_{plat}.yaml"),
                 {"kind": "cone_forward_structure_compare", "platform": plat,
                  "device_counts": device_counts, "sizes": size_labels,
                  "views_per_owner": VIEWS_PER_OWNER, "pixel_batch": PIXEL_BATCH,
                  "bands_per_shard": BANDS_PER_SHARD,
                  "grid": {f"{k[0]}|n{k[1]}": v for k, v in grid.items()}})
    print("\nDone.")


if __name__ == "__main__":
    if "--worker" in sys.argv:
        run_worker(sys.argv[1:])
    else:
        main()
