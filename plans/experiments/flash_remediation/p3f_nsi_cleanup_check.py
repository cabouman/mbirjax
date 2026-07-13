"""Phase 3 / step D validation: the NSI auto-geometry cleanup on the real Lilly scan.

The cleanup: nsi.compute_sino_and_params no longer hand-sets recon_slice_offset, and the
NSI convention becomes construct -> set_params(**optional_params) ->
auto_set_recon_geometry() (matching zeiss), so the automatic shape/extension finally see
the REAL detector pitches and offsets.  Expected effects at Lilly ds4 (4,4)/ss2:

  - shape: (374, 374, ~571) instead of the pitch-inflated (374, 374, 667) -- the
    extension shrinks from R/SID 0.42 (computed with default unit pitches) to the real
    0.21, asymmetric per end because det_row_offset = -3.9 rows enters correctly;
  - offset: the auto per-end value (~ +4.5 slices) instead of the old hand compensation
    (+3.9 slices) -- a sub-slice grid difference, so comparisons below use PHYSICALLY
    z-aligned profiles rather than voxelwise diffs;
  - values: interior unchanged vs the old-flow reference (p3c ref), ends still clean;
  - split_sino_recon at ds8 under the new flow: seam still at the step-B fixed level.

Workers:
  shape      -- build the new-flow ds4 model; report shape/offset (no GPU work).
  recon_ds4  -- 15-iter unsplit recon (new flow); profile-compare vs p3c_lilly_ds4_ref.
  split_ds8  -- unsplit ref + default split at ds8 (new flow); P2c seam metric.

Run on gautschi (one GPU):  sbatch p3f_nsi_cleanup.slurm
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
SEQUENCE = [0, 2, 4, 6, 7]
SEED = 0
MAX_ITERATIONS = 15
SEAM_VIEW_HALF_WIDTH = 12
INTERIOR_RADIUS_FRAC = 0.85
EXPECTED_COMMIT = 'dbc9c3b'  # step-D head ("Match nsi preprocessing to zeiss re recon_slice_offset.")

_ROLE, _JOB = 'P3F_ROLE', 'P3F_JOB'


def _lib_provenance():
    import mbirjax as mj
    repo = os.path.dirname(os.path.dirname(os.path.abspath(mj.__file__)))
    try:
        commit = subprocess.run(['git', '-C', repo, 'rev-parse', '--short', 'HEAD'],
                                capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        commit = 'unknown'
    return mj.__file__, commit


def _build(ds, view_ss):
    """The NEW NSI flow: construct -> optional_params -> auto_set_recon_geometry."""
    import mbirjax as mj
    import mbirjax.preprocess as mjp
    sino, cone_params, optional_params = mjp.nsi.compute_sino_and_params(
        DATASET_DIR, downsample_factor=(ds, ds), subsample_view_factor=view_ss)
    sino = np.asarray(sino)
    assert 'recon_slice_offset' not in optional_params, \
        'cleanup regression: optional_params still carries recon_slice_offset'
    model = mj.ConeBeamModel(**cone_params)
    model.set_params(**optional_params)
    model.auto_set_recon_geometry()
    model.set_params(verbose=0)
    weights = np.asarray(mj.gen_weights(sino, 'transmission_root'))
    model.auto_set_regularization_params(sino, weights=weights)
    model.set_params(auto_regularize_flag=False)
    model.set_params(partition_sequence=SEQUENCE)
    return model, sino, weights


def _recon(model, sino, weights, max_iterations=MAX_ITERATIONS):
    np.random.seed(SEED)
    vol, _ = model.recon(sino, weights=weights, max_iterations=max_iterations,
                         stop_threshold_change_pct=0.0, print_logs=False)
    return np.asarray(vol)


def worker():
    import mbirjax as mj
    job = json.loads(os.environ[_JOB])
    task = job['task']
    lib_file, commit = _lib_provenance()
    print(f'WORKER {task}: mbirjax={lib_file} commit={commit}', flush=True)
    if EXPECTED_COMMIT is not None:
        assert commit == EXPECTED_COMMIT, \
            f'live mbirjax commit {commit} != expected {EXPECTED_COMMIT} -- wrong checkout?'

    if task == 'shape':
        model, sino, _ = _build(4, 2)
        shape = tuple(int(x) for x in model.get_params('recon_shape'))
        offset = float(model.get_params('recon_slice_offset'))
        dslice = model.get_params('voxel_slice_aspect') * model.get_params('delta_voxel')
        print(f'  NEW-FLOW ds4: sino {sino.shape}, recon_shape {shape}, '
              f'recon_slice_offset {offset:.4f} ALU = {offset/dslice:.2f} slices', flush=True)
        with open(os.path.join(OUT_DIR, 'p3f_shape.json'), 'w') as f:
            json.dump({'shape': list(shape), 'offset_alu': offset,
                       'offset_slices': offset / float(dslice), 'commit': commit}, f, indent=1)
        return

    if task == 'recon_ds4':
        model, sino, weights = _build(4, 2)
        vol = _recon(model, sino, weights)
        np.save(os.path.join(OUT_DIR, 'p3f_lilly_ds4_newflow.npy'), vol)

        # Physically z-aligned interior profile comparison vs the OLD-flow reference.
        ref = np.load(os.path.join(OUT_DIR, 'p3c_lilly_ds4_ref.npy'))
        dslice = float(model.get_params('voxel_slice_aspect') * model.get_params('delta_voxel'))
        off_new = float(model.get_params('recon_slice_offset'))
        off_old = 0.4228                       # p3c ref grid (recorded in its job log)
        def z_axis(n, off):
            return dslice * (np.arange(n) - (n - 1) / 2.0) + off
        rr, cc = np.ogrid[:vol.shape[0], :vol.shape[1]]
        disk = ((rr - vol.shape[0]/2)**2 + (cc - vol.shape[1]/2)**2) <= (0.35*vol.shape[0])**2
        prof_new = np.array([float(np.abs(vol[:, :, k][disk]).mean())
                             for k in range(vol.shape[2])])
        prof_old = np.array([float(np.abs(ref[:, :, k][disk]).mean())
                             for k in range(ref.shape[2])])
        z_new, z_old = z_axis(vol.shape[2], off_new), z_axis(ref.shape[2], off_old)
        # Shared z-range, old profile interpolated onto the new grid.
        m = (z_new >= z_old[0]) & (z_new <= z_old[-1])
        old_on_new = np.interp(z_new[m], z_old, prof_old)
        rel = np.abs(prof_new[m] - old_on_new) / max(prof_old.max(), 1e-12)
        interior = slice(60, int(m.sum()) - 60)
        print(f'  profile agreement (old-flow vs new-flow, z-aligned): '
              f'interior max rel {float(rel[interior].max()):.4f}, '
              f'overall max rel {float(rel.max()):.4f} (ends include the trimmed '
              f'over-extension region)', flush=True)
        with open(os.path.join(OUT_DIR, 'p3f_ds4_profile_check.json'), 'w') as f:
            json.dump({'interior_max_rel': float(rel[interior].max()),
                       'overall_max_rel': float(rel.max()),
                       'new_shape': list(vol.shape), 'ref_shape': list(ref.shape)},
                      f, indent=1)
        return

    if task == 'split_ds8':
        model, sino, weights = _build(8, 8)
        shape = tuple(int(x) for x in model.get_params('recon_shape'))
        print(f'  ds8 new-flow recon_shape {shape}', flush=True)
        ref = _recon(model, sino, weights)
        np.random.seed(SEED)
        split, info = model.split_sino_recon(sino, weights=weights,
                                             max_iterations=MAX_ITERATIONS,
                                             stop_threshold_change_pct=0.0, print_logs=False)
        split = np.asarray(split)
        print(f'  split_params: {info["split_params"]}', flush=True)
        dslice = model.get_params('voxel_slice_aspect') * model.get_params('delta_voxel')
        off = float(model.get_params('recon_slice_offset'))
        split_index = int(np.round((shape[2] - 1) / 2.0 - off / float(dslice)))
        rr, cc = np.ogrid[:shape[0], :shape[1]]
        disk = np.sqrt((rr - (shape[0]-1)/2.0)**2 + (cc - (shape[1]-1)/2.0)**2) \
            < INTERIOR_RADIUS_FRAC * (min(shape[:2]) / 2.0)
        rms = np.sqrt(np.mean((split - ref)[disk] ** 2, axis=0))
        lo = max(0, split_index - SEAM_VIEW_HALF_WIDTH)
        hi = min(shape[2], split_index + SEAM_VIEW_HALF_WIDTH + 1)
        bg = float(np.median(np.concatenate([rms[:lo], rms[hi:]])))
        seam_max = float(rms[lo:hi].max())
        print(f'  ds8 split seam max RMS {seam_max:.3e} ({seam_max/bg:.1f}x bg {bg:.3e}), '
              f'split at slice {split_index}', flush=True)
        with open(os.path.join(OUT_DIR, 'p3f_ds8_seam_check.json'), 'w') as f:
            json.dump({'seam_max_rms': seam_max, 'background_rms': bg,
                       'split_params': info['split_params'], 'shape': list(shape)}, f, indent=1)
        return

    raise ValueError(f'unknown task {task}')


def main():
    if os.environ.get(_ROLE) == 'worker':
        worker()
        return
    lib_file, commit = _lib_provenance()
    print(f'DRIVER: mbirjax={lib_file} commit={commit}', flush=True)
    for task in ('shape', 'recon_ds4', 'split_ds8'):
        print(f'--- {task} ---', flush=True)
        env = dict(os.environ, **{_ROLE: 'worker', _JOB: json.dumps({'task': task})})
        proc = subprocess.run([sys.executable, '-u', os.path.abspath(__file__)], env=env)
        if proc.returncode != 0:
            print(f'*** WORKER FAILED: {task} (rc={proc.returncode}) ***', flush=True)
            sys.exit(proc.returncode)
    print('ALL RUNS DONE', flush=True)


if __name__ == '__main__':
    main()
