"""Before/after harness for the MultiAxisParallelModel convention refactor.

The upcoming refactor (mirror ParallelBeamModel/ConeBeamModel: recon_ijk_to_xyz / geometry_xyz_to_uv
/ detector_uv_to_mn, move the azimuth rotation into recon_ijk_to_xyz) must change NEITHER the
numerics NOR the performance -- it is pure regrouping/renaming of already-traced math. This harness
captures a baseline from the current code, then compares against it after each refactor step.

Four gates:
  1. GOLDEN NUMERICS   -- forward_project and back_project over a config matrix, saved to .npz;
                          after refactor, max abs/rel diff must be < TOL (GPU scatter noise ~1e-6).
  2. REDUCTION         -- multi-axis(el=0) == ParallelBeamModel for zero and nonzero offset.
  3. ADJOINT           -- <Ax,y> == <x, A^T y> for every config.
  4. PERFORMANCE       -- median wall-clock of forward / back / a short recon on a GPU-sized
                          problem; after refactor, no regression beyond a few percent.

Usage (on a GPU node with the editable mbirjax env):
  python refactor_harness.py                 # capture baseline if missing, else compare to it
  python refactor_harness.py --recapture     # overwrite the baseline with current code
The baseline file lives next to this script (refactor_baseline.npz); it is a local reference for
the refactor, not something to commit.
"""
import os, sys, time, argparse
import numpy as np
import jax, jax.numpy as jnp
from mbirjax import ParallelBeamModel, MultiAxisParallelModel

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINE = os.path.join(HERE, "refactor_baseline.npz")
TOL_REL = 1e-5              # above GPU scatter run-to-run noise (~1e-6), below any real change


def nrmse(a, b):
    return float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))


def _build(angles, sino_shape, params, corr):
    m = MultiAxisParallelModel(sino_shape, jnp.asarray(angles))
    m.set_elevation_correction(**corr)
    m.set_params(**params)
    return m


def golden_configs():
    """A matrix exercising elevation (incl. >45), offsets, anisotropy, and each correction mode."""
    N, nv = 32, 10
    sino = (nv, 64, 56)
    az = np.linspace(0, np.pi, nv, endpoint=False)
    def ang(el_amp): return np.stack([az, np.deg2rad(el_amp) * np.cos(2 * az)], 1).astype(np.float32)
    base = dict(recon_shape=(N, N, N), delta_voxel=1.0, delta_det_row=1.0, delta_det_channel=1.0)
    ON = dict(correct_elevation_pathlength=True, broaden_elevation_footprint=False)
    return [
        dict(label="el0",         angles=ang(0),  sino=sino, params=dict(base), corr=ON),
        dict(label="el30",        angles=ang(30), sino=sino, params=dict(base), corr=ON),
        dict(label="el55",        angles=ang(55), sino=sino, params=dict(base), corr=ON),
        dict(label="offsets",     angles=ang(30), sino=sino,
             params={**base, "det_channel_offset": 3.0, "det_row_offset": -2.0}, corr=ON),
        dict(label="anisotropic", angles=ang(30), sino=sino,
             params={**base, "voxel_row_aspect": 1.5, "voxel_slice_aspect": 0.7}, corr=ON),
        dict(label="uncorrected", angles=ang(30), sino=sino, params=dict(base),
             corr=dict(correct_elevation_pathlength=False, broaden_elevation_footprint=False)),
        dict(label="broaden55",   angles=ang(55), sino=sino, params=dict(base),
             corr=dict(correct_elevation_pathlength=True, broaden_elevation_footprint=True)),
    ]


def compute_golden():
    """Return {key: ndarray} of forward and back projections + adjoint inner products per config."""
    out = {}
    for ci, c in enumerate(golden_configs()):
        m = _build(c["angles"], c["sino"], c["params"], c["corr"])
        N = c["params"]["recon_shape"]
        rng = np.random.default_rng(ci)
        x = jnp.asarray(rng.random(N).astype(np.float32))
        y = jnp.asarray(rng.random(c["sino"]).astype(np.float32))
        fwd = m.forward_project(x); bwd = m.back_project(y)
        out[c["label"] + ".fwd"] = np.asarray(fwd)
        out[c["label"] + ".bwd"] = np.asarray(bwd)
        # adjoint inner products (stored so we can report them and check <Ax,y> vs <x,A^Ty>)
        out[c["label"] + ".ip"] = np.asarray([float(jnp.sum(fwd * y)), float(jnp.sum(x * bwd))])
    return out


def reduction_check():
    """multi-axis(el=0) == ParallelBeamModel, for zero and nonzero det_channel_offset."""
    np.random.seed(0)
    N, nv, NR, NC = 24, 8, 24, 41
    az = np.linspace(0, np.pi, nv, endpoint=False).astype(np.float32)
    vol = jnp.asarray(np.random.rand(N, N, N).astype(np.float32))
    res = {}
    for off in [0.0, 3.5, -2.0]:
        pm = ParallelBeamModel((nv, NR, NC), jnp.asarray(az))
        pm.set_params(recon_shape=(N, N, N), delta_voxel=1.0, delta_det_channel=1.0, det_channel_offset=off)
        mm = MultiAxisParallelModel((nv, NR, NC), jnp.asarray(np.stack([az, np.zeros_like(az)], 1)))
        mm.set_params(recon_shape=(N, N, N), delta_voxel=1.0, delta_det_channel=1.0,
                      delta_det_row=1.0, det_channel_offset=off)
        res[off] = nrmse(mm.forward_project(vol), pm.forward_project(vol))
    return res


def _time(fn, n=7):
    r = fn(); jax.block_until_ready(r)                  # warmup / compile
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); r = fn(); jax.block_until_ready(r); ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def perf_timings():
    """Median wall-clock (ms) of forward / back / short recon on a GPU-sized problem."""
    N, nv = 128, 48
    sino = (nv, 176, 200)
    az = np.linspace(0, np.pi, nv, endpoint=False)
    angles = jnp.asarray(np.stack([az, np.deg2rad(30) * np.cos(2 * az)], 1).astype(np.float32))
    m = MultiAxisParallelModel(sino, angles)
    m.set_elevation_correction(correct_elevation_pathlength=True)
    m.set_params(recon_shape=(N, N, N), delta_voxel=1.0, delta_det_row=1.0, delta_det_channel=1.0)
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.random((N, N, N)).astype(np.float32))
    y = jnp.asarray(rng.random(sino).astype(np.float32))
    w = jnp.ones_like(y); init = jnp.zeros((N, N, N), jnp.float32)
    t_fwd = _time(lambda: m.forward_project(x))
    t_bwd = _time(lambda: m.back_project(y))
    t_rec = _time(lambda: m.recon(y, weights=w, init_recon=init, max_iterations=5, print_logs=False)[0], n=3)
    return {"forward_ms": t_fwd * 1e3, "back_ms": t_bwd * 1e3, "recon5_ms": t_rec * 1e3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recapture", action="store_true", help="overwrite the baseline with current code")
    args = ap.parse_args()
    print("jax devices:", jax.devices())

    golden = compute_golden()
    reduction = reduction_check()
    timings = perf_timings()

    print("\nREDUCTION (multi-axis el=0 vs ParallelBeam; should be ~0):")
    for off, v in reduction.items():
        print(f"  det_channel_offset={off:+.1f}: NRMSE={v:.2e}")
    print("\nADJOINT (<Ax,y> vs <x,A^Ty> per config):")
    for c in golden_configs():
        a, b = golden[c["label"] + ".ip"]
        print(f"  {c['label']:>12}: reldiff={abs(a - b) / abs(a):.2e}")
    print("\nPERFORMANCE (median wall-clock):")
    for k, v in timings.items():
        print(f"  {k:>12}: {v:8.2f} ms")

    if args.recapture or not os.path.exists(BASELINE):
        save = dict(golden)
        for k, v in timings.items():
            save["perf." + k] = np.asarray(v)
        np.savez(BASELINE, **save)
        print(f"\nBASELINE CAPTURED -> {BASELINE}\n(Re-run after the refactor to compare.)")
        return

    ref = np.load(BASELINE)
    print(f"\nGOLDEN NUMERICS vs baseline (tol rel={TOL_REL:g}):")
    worst = 0.0; fails = 0
    for c in golden_configs():
        for kind in (".fwd", ".bwd"):
            key = c["label"] + kind
            a = golden[key]; b = ref[key]
            denom = max(float(np.max(np.abs(b))), 1e-12)
            rel = float(np.max(np.abs(a - b))) / denom
            worst = max(worst, rel)
            flag = "" if rel < TOL_REL else "  <<< FAIL"
            if rel >= TOL_REL: fails += 1
            print(f"  {key:>18}: max_rel_diff={rel:.2e}{flag}")
    print(f"  worst={worst:.2e}  ->  {'PASS' if fails == 0 else str(fails) + ' FAIL(S)'}")

    print("\nPERFORMANCE vs baseline (ratio >1 = slower):")
    for k in ("forward_ms", "back_ms", "recon5_ms"):
        now = timings[k]; was = float(ref["perf." + k])
        print(f"  {k:>12}: {was:7.2f} -> {now:7.2f} ms   (x{now / was:.3f})")


if __name__ == "__main__":
    main()
