"""Phase 3 / step B validation: the SHIPPED split_sino_recon (geometry-derived recon
overlap, taper retired, align_split_grid opt-in) on the real Lilly scan, at the two
regimes the P2c investigation measured:

  4x: downsample (4,4), view subsample 2  -- old shipped taper fixed the stripes here
      (6.5e-4); the geometry extension measured 5.7e-4 @15 iters (decaying to 3.2e-4 @60).
  8x: downsample (8,8), view subsample 8  -- old shipped taper FAILED here (6.1e-3 vs
      no-fix 7.9e-3); the extension at the formula value measured 9.0e-4.

Per regime: an unsplit reference, the shipped split (default), and the aligned split
(align_split_grid=True).  The aligned variant's output grid is SHIFTED by its residual
(sub-slice) alignment, so its fair reference is an unsplit recon on the SAME shifted grid
-- run as a fourth job with recon_slice_offset shifted by the reported grid_shift_alu.
Seam metric mirrors the P2c scripts: per-slice RMS(split - ref) over an interior disk,
background = median outside the seam window, verdict = seam max / background.

Data pipeline mirrors the P2c scripts exactly (nsi.compute_sino_and_params, NO auto-crop,
transmission_root weights, regularization from the full sinogram then frozen).  Under the
step-A library the full model's slice axis carries the (offset-blind) visibility
extension from construction, then NSI's recon_slice_offset recenters -- both the unsplit
reference and the split divide the SAME grid, so the comparison is apples-to-apples.

Run on gautschi (one GPU), from the staging directory:  sbatch p3c_lilly_split.slurm
Outputs (volumes as slice-viewer h5, seam tables + summary json) land in OUT_DIR.
"""
import json
import os
import subprocess
import sys
import time

import numpy as np

# ---------------- run parameters (edit here; no CLI args) ----------------
DATASET_DIR = '/scratch/gautschi/buzzard/flash_lilly/D01788'
OUT_DIR = '/depot/bouman/data/mbirjax_metrics/padding'
REGIMES = {
    'lilly_ds4': {'downsample_factor': (4, 4), 'subsample_view_factor': 2,
                  'max_iterations': 15},
    'lilly_ds8': {'downsample_factor': (8, 8), 'subsample_view_factor': 8,
                  'max_iterations': 15},
}
SEAM_VIEW_HALF_WIDTH = 12
INTERIOR_RADIUS_FRAC = 0.85
SEED = 0
EXPECTED_COMMIT = 'fcc0e9e'  # step-B head ("Remove sine taper and use padding instead.")

_ROLE, _JOB = 'P3C_ROLE', 'P3C_JOB'


def _lib_provenance():
    import mbirjax as mj
    repo = os.path.dirname(os.path.dirname(os.path.abspath(mj.__file__)))
    try:
        commit = subprocess.run(['git', '-C', repo, 'rev-parse', '--short', 'HEAD'],
                                capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        commit = 'unknown'
    return mj.__file__, commit


def _build(regime):
    """Model + sinogram + weights for one regime, mirroring the P2c pipeline."""
    import mbirjax as mj
    import mbirjax.preprocess as mjp
    spec = REGIMES[regime]
    sino, cone_params, optional_params = mjp.nsi.compute_sino_and_params(
        DATASET_DIR, downsample_factor=spec['downsample_factor'],
        subsample_view_factor=spec['subsample_view_factor'])
    sino = np.asarray(sino)
    model = mj.ConeBeamModel(**cone_params)
    model.set_params(**optional_params)
    model.set_params(verbose=0)
    weights = np.asarray(mj.gen_weights(sino, 'transmission_root'))
    model.auto_set_regularization_params(sino, weights=weights)
    model.set_params(auto_regularize_flag=False)
    return model, sino, weights


def worker():
    job = json.loads(os.environ[_JOB])
    regime, variant = job['regime'], job['variant']
    lib_file, commit = _lib_provenance()
    print(f'WORKER {regime} {variant}: mbirjax={lib_file} commit={commit}', flush=True)
    if EXPECTED_COMMIT is not None:
        assert commit == EXPECTED_COMMIT, \
            f'live mbirjax commit {commit} != expected {EXPECTED_COMMIT} -- wrong checkout?'

    model, sino, weights = _build(regime)
    max_iterations = REGIMES[regime]['max_iterations']
    print(f'  sino {sino.shape}  recon_shape {tuple(model.get_params("recon_shape"))} '
          f'offset {float(model.get_params("recon_slice_offset")):.4f}', flush=True)

    t0 = time.perf_counter()
    split_params = None
    if variant == 'ref':
        np.random.seed(SEED)
        vol, recon_dict = model.recon(sino, weights=weights, max_iterations=max_iterations,
                                      stop_threshold_change_pct=0.0, print_logs=False)
    elif variant == 'ref_shifted':
        # Reference on the ALIGNED grid: shift recon_slice_offset by the aligned split's
        # reported grid shift (read from its summary, written by the aligned worker earlier).
        with open(os.path.join(OUT_DIR, f'p3c_{regime}_aligned_split_params.json')) as f:
            shift = json.load(f)['grid_shift_alu']
        model.set_params(recon_slice_offset=float(model.get_params('recon_slice_offset')) + shift)
        print(f'  shifted reference grid by {shift:.6f} ALU', flush=True)
        np.random.seed(SEED)
        vol, recon_dict = model.recon(sino, weights=weights, max_iterations=max_iterations,
                                      stop_threshold_change_pct=0.0, print_logs=False)
    else:   # 'default' or 'aligned'
        np.random.seed(SEED)
        vol, recon_dict = model.split_sino_recon(
            sino, weights=weights, max_iterations=max_iterations,
            stop_threshold_change_pct=0.0, print_logs=False,
            align_split_grid=(variant == 'aligned'))
        split_params = recon_dict['split_params']
        print(f'  split_params: {split_params}', flush=True)
        with open(os.path.join(OUT_DIR, f'p3c_{regime}_{variant}_split_params.json'), 'w') as f:
            json.dump(split_params, f, indent=1)
    elapsed = time.perf_counter() - t0

    vol = np.asarray(vol)
    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(os.path.join(OUT_DIR, f'p3c_{regime}_{variant}.npy'), vol)
    print(f'WORKER {regime} {variant}: done ({elapsed:.0f}s, shape {vol.shape})', flush=True)


def seam_report(regime):
    """P2c-style per-slice seam table for the finished regime (host-side, cheap)."""
    import mbirjax as mj  # noqa: F401  (import binds env before jax, mirroring the workers)
    model, _sino, _w = None, None, None
    # The split index and slice pitch come from a fresh model build (no recon needed).
    model, sino, _ = _build(regime)
    shape = tuple(int(x) for x in model.get_params('recon_shape'))
    dslice = model.get_params('voxel_slice_aspect') * model.get_params('delta_voxel')
    slice_off = float(model.get_params('recon_slice_offset'))
    split_index = int(np.round((shape[2] - 1) / 2.0 - slice_off / dslice))
    i = np.arange(shape[0], dtype=np.float32)[:, None] - (shape[0] - 1) / 2.0
    j = np.arange(shape[1], dtype=np.float32)[None, :] - (shape[1] - 1) / 2.0
    disk = np.sqrt(i**2 + j**2) < INTERIOR_RADIUS_FRAC * (min(shape[:2]) / 2.0)
    lo = max(0, split_index - SEAM_VIEW_HALF_WIDTH)
    hi = min(shape[2], split_index + SEAM_VIEW_HALF_WIDTH + 1)

    summary = {'regime': regime, 'split_index': split_index}
    pairs = [('default', 'ref'), ('aligned', 'ref_shifted')]
    for variant, ref_name in pairs:
        ref = np.load(os.path.join(OUT_DIR, f'p3c_{regime}_{ref_name}.npy'))
        split = np.load(os.path.join(OUT_DIR, f'p3c_{regime}_{variant}.npy'))
        rms = np.sqrt(np.mean((split - ref)[disk] ** 2, axis=0))
        bg = float(np.median(np.concatenate([rms[:lo], rms[hi:]])))
        print(f'\n=== {regime} {variant} vs {ref_name}: split at {split_index}; '
              f'background median RMS {bg:.3e}', flush=True)
        for s in range(lo, hi):
            marker = '  <-- split' if s == split_index else ''
            print(f'  slice {s:4d}: RMS {rms[s]:.3e}  ({rms[s]/bg:6.1f}x bg){marker}',
                  flush=True)
        seam_max = float(rms[lo:hi].max())
        print(f'{regime} {variant} seam max RMS: {seam_max:.3e}  ({seam_max/bg:.1f}x bg)',
              flush=True)
        summary[variant] = {'seam_max_rms': seam_max, 'background_rms': bg,
                            'seam_over_bg': seam_max / bg}
    with open(os.path.join(OUT_DIR, f'p3c_{regime}_seam_summary.json'), 'w') as f:
        json.dump(summary, f, indent=1)


def main():
    if os.environ.get(_ROLE) == 'worker':
        worker()
        return
    lib_file, commit = _lib_provenance()
    print(f'DRIVER: mbirjax={lib_file} commit={commit}', flush=True)
    for regime in REGIMES:
        # 'aligned' runs before 'ref_shifted' (which reads the aligned grid shift).
        for variant in ('ref', 'default', 'aligned', 'ref_shifted'):
            out = os.path.join(OUT_DIR, f'p3c_{regime}_{variant}.npy')
            if os.path.exists(out):
                print(f'--- {regime} {variant}: exists, skipped', flush=True)
                continue
            print(f'--- {regime} {variant} ---', flush=True)
            env = dict(os.environ, **{_ROLE: 'worker',
                                      _JOB: json.dumps({'regime': regime, 'variant': variant})})
            proc = subprocess.run([sys.executable, '-u', os.path.abspath(__file__)], env=env)
            if proc.returncode != 0:
                print(f'*** WORKER FAILED: {regime} {variant} (rc={proc.returncode}) ***',
                      flush=True)
                sys.exit(proc.returncode)
        seam_report(regime)
    print('ALL RUNS DONE', flush=True)


if __name__ == '__main__':
    main()
