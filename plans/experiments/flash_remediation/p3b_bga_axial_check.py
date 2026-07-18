"""Phase 3 / step A validation on the Purdue BGA Normal scan: does AXIAL padding alone help?

BGA (`17U1-250TC-Normal_Tomo_No_HART.txrm`) leaves the FoV both LATERALLY and axially --
the severe case.  The lateral truncation is deliberately untreated here (step C is
detect-and-warn; lateral cover-padding is a separate, manual decision), so this experiment
asks a narrower question than the SiC one: with the lateral contamination present in BOTH
variants, does the per-end axial extension by itself improve the reconstruction --
in particular the prominent noise near the CENTER slices and the slow convergence Greg
observes with the old shape?

Settings mirror `mbirjax_applications/zeiss/experiment_zeiss.py` :: 'Purdue BGA Normal
scan' exactly: downsample (2,2), view subsample 2, sharpness 1.5, snr_db 35, no view
alignment, transmission_root weights, partition sequence [0,2,4,6,7]; that script's
max_iterations default is 15 = our first snapshot, and we continue to 50 for the
convergence story (stop threshold 0, fixed iterations -- the flash inflates the change-%%
metric, so comparisons run at matched counts).

Structure mirrors p3a_sic_axial_check.py: a one-time PREPARE step caches the preprocessed
sinogram (the .txrm load + downsample is minutes; the cache also serves the step-C lateral
experiments later), then one subprocess per variant:
  new -- auto_set_recon_geometry at the current library (per-end visibility extension).
  old -- recon shape/offset forced back to the pre-extension automatic values.

Run on gautschi (one GPU), from the staging directory:  sbatch p3b_bga_axial.slurm
Outputs land in OUT_DIR: bga_normal_<variant>_iter<k>.h5 + _log.json (+ the cache).
"""
import json
import os
import subprocess
import sys
import time

import numpy as np

# ---------------- run parameters (edit here; no CLI args) ----------------
SCAN_PATH = '/depot/bouman/data/Zeiss/purdue_BGA/17U1-250TC-Normal_Tomo_No_HART.txrm'
OUT_DIR = '/depot/bouman/data/mbirjax_metrics/padding'
TAG = 'bga_normal_v2x_d2x'
CACHE_H5 = os.path.join(OUT_DIR, f'{TAG}_cache.h5')
CACHE_JSON = os.path.join(OUT_DIR, f'{TAG}_cache.json')
DOWNSAMPLE = (2, 2)          # experiment_zeiss.py 'Purdue BGA Normal scan'
VIEW_SUBSAMPLE = 2
RECON_SETTINGS = {'sharpness': 1.5, 'snr_db': 35.0}
VARIANTS = ('new', 'old')
SEQUENCE = [0, 2, 4, 6, 7]   # the script's 'default' sequence
SNAPSHOTS = [15, 50]         # 15 = experiment_zeiss.py's max_iterations default
SEED = 0
EXPECTED_COMMIT = 'a872695'

_ROLE, _JOB = 'P3B_ROLE', 'P3B_JOB'


def _lib_provenance():
    import mbirjax as mj
    repo = os.path.dirname(os.path.dirname(os.path.abspath(mj.__file__)))
    try:
        commit = subprocess.run(['git', '-C', repo, 'rev-parse', '--short', 'HEAD'],
                                capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        commit = 'unknown'
    return mj.__file__, commit


def prepare():
    """One-time preprocessing -> cache (mirrors experiment_zeiss.py's load + the
    partition-sequence build_cache.py zeiss format so the p3a-style workers can load it)."""
    import mbirjax.preprocess as mjp
    lib_file, commit = _lib_provenance()
    print(f'PREPARE: mbirjax={lib_file} commit={commit}', flush=True)
    sino, geometry_params, optional_params, metadata = mjp.zeiss.compute_sino_and_params(
        SCAN_PATH, downsample_factor=DOWNSAMPLE, subsample_view_factor=VIEW_SUBSAMPLE)
    model_class = 'ParallelBeamModel' if metadata['scanner_type'] == 'ultra' else 'ConeBeamModel'
    os.makedirs(OUT_DIR, exist_ok=True)
    mjp.save_cone_preprocessing(CACHE_H5, sino, geometry_params, optional_params)
    sidecar = {'model_class': model_class, 'auto_set_recon_geometry': True,
               'recon_settings': dict(RECON_SETTINGS),
               'provenance': {'source': SCAN_PATH, 'downsample_factor': list(DOWNSAMPLE),
                              'subsample_view_factor': VIEW_SUBSAMPLE,
                              'per': 'experiment_zeiss.py Purdue BGA Normal scan',
                              'built': time.strftime('%Y-%m-%d'), 'commit': commit}}
    with open(CACHE_JSON, 'w') as f:
        json.dump(sidecar, f, indent=1)
    print(f'PREPARE: cached sino {sino.shape} -> {CACHE_H5}', flush=True)


def _old_auto_shape(model):
    """The PRE-extension automatic recon shape/offset (see p3a_sic_axial_check.py)."""
    num_det_rows = int(model.get_params('sinogram_shape')[1])
    delta_det_row, delta_voxel, voxel_slice_aspect = model.get_params(
        ['delta_det_row', 'delta_voxel', 'voxel_slice_aspect'])
    magnification = model.get_magnification()
    delta_voxel_slice = voxel_slice_aspect * delta_voxel
    z_shifts = np.asarray(model.get_params('view_params_array'))[:, 1]
    h_iso = num_det_rows * (delta_det_row / magnification)
    base_slices = max(1, int(np.ceil((h_iso + (z_shifts.max() - z_shifts.min()))
                                     / delta_voxel_slice)))
    old_offset = float(0.5 * (z_shifts.min() + z_shifts.max()))
    rows, cols, _ = model.get_params('recon_shape')
    return (int(rows), int(cols), base_slices), old_offset


def worker():
    job = json.loads(os.environ[_JOB])
    variant = job['variant']
    lib_file, commit = _lib_provenance()
    print(f'WORKER {TAG} {variant}: mbirjax={lib_file} commit={commit}', flush=True)
    assert commit == EXPECTED_COMMIT, \
        f'live mbirjax commit {commit} != expected {EXPECTED_COMMIT} -- wrong checkout?'

    import mbirjax as mj
    import mbirjax.preprocess as mjp
    sino, geometry_params, optional_params, _ = mjp.load_cone_preprocessing(CACHE_H5)
    with open(CACHE_JSON) as f:
        sidecar = json.load(f)
    model = getattr(mj, sidecar['model_class'])(**geometry_params)
    if optional_params:
        model.set_params(**optional_params)
    model.auto_set_recon_geometry()
    if variant == 'old':
        old_shape, old_offset = _old_auto_shape(model)
        model.set_params(recon_shape=old_shape, recon_slice_offset=old_offset)
    model.set_params(verbose=0, **sidecar['recon_settings'])
    model.set_params(partition_sequence=SEQUENCE)
    weights = mj.gen_weights(sino, weight_type='transmission_root')
    recon_shape = model.get_params('recon_shape')
    offset = float(model.get_params('recon_slice_offset'))
    print(f'  recon_shape={tuple(int(x) for x in recon_shape)} recon_slice_offset={offset:.4f}',
          flush=True)

    np.random.seed(SEED)
    partitions = mj.gen_set_of_pixel_partitions(
        recon_shape, model.get_params('granularity'),
        output_device=model.recon_placement.devices[0],
        use_ror_mask=model.get_params('use_ror_mask'))
    seq_ext = np.asarray(mj.gen_partition_sequence(SEQUENCE, max_iterations=max(SNAPSHOTS)))
    model._log_run_header(0, '~/.mbirjax/logs/recon.log', print_logs=False)
    model.auto_set_regularization_params(sino, weights=weights)

    rows, recon_dev, ckpt, total, prev = [], None, {}, 0.0, 0
    for boundary in sorted(SNAPSHOTS):
        t0 = time.perf_counter()
        recon_dev, loss_vectors, ckpt = model.vcd_recon(
            sino, partitions, seq_ext[prev:boundary], 0.0, weights=weights,
            init_recon=recon_dev, first_iteration=prev,
            init_error_sinogram=ckpt.get('error_sinogram'),
            fm_hessian=ckpt.get('fm_hessian'), return_checkpoint=True)
        total += time.perf_counter() - t0
        changes = [100.0 * float(v) for v in loss_vectors[2]]
        rows += [{'iteration': prev + i + 1, 'change_pct': c, 'time_s': total}
                 for i, c in enumerate(changes)]
        prev = boundary
        recon = np.asarray(recon_dev)[:, :, :recon_shape[2]]
        info = {'dataset': TAG, 'variant': variant, 'iterations': boundary,
                'sequence': SEQUENCE, 'seed': SEED, 'stop_threshold_change_pct': 0.0,
                'weights': 'transmission_root', 'recon_slice_offset': offset,
                'settings': dict(RECON_SETTINGS), 'mbirjax_commit': commit,
                'purpose': 'flash-remediation step-A axial-extension validation (BGA: '
                           'laterally truncated in both variants; axial-only A/B)'}
        recon_dict = model.get_recon_dict(
            recon_params=info,
            notes=f'p3b BGA axial-extension check: variant={variant} iter={boundary}')
        out = os.path.join(OUT_DIR, f'{TAG}_{variant}_iter{boundary}.h5')
        model.save_recon_hdf5(out, recon, recon_dict)
        print(f'  saved {out}  (iter {boundary}, change {changes[-1]:.4f}%, '
              f'cum t={total:.0f}s, shape {recon.shape})', flush=True)

    log = {'dataset': TAG, 'variant': variant, 'sequence': SEQUENCE, 'seed': SEED,
           'recon_shape': [int(x) for x in recon_shape], 'recon_slice_offset': offset,
           'sino_shape': [int(x) for x in sino.shape], 'mbirjax_commit': commit,
           'settings': dict(RECON_SETTINGS), 'total_time_s': total, 'rows': rows}
    with open(os.path.join(OUT_DIR, f'{TAG}_{variant}_log.json'), 'w') as f:
        json.dump(log, f, indent=1)
    print(f'WORKER {TAG} {variant}: done ({total:.0f}s)', flush=True)


def main():
    role = os.environ.get(_ROLE)
    if role == 'worker':
        worker()
        return
    if role == 'prepare':
        prepare()
        return
    # Driver: prepare the cache if missing, then one subprocess per variant.
    if not (os.path.exists(CACHE_H5) and os.path.exists(CACHE_JSON)):
        print('--- prepare cache ---', flush=True)
        env = dict(os.environ, **{_ROLE: 'prepare'})
        proc = subprocess.run([sys.executable, '-u', os.path.abspath(__file__)], env=env)
        if proc.returncode != 0:
            print(f'*** PREPARE FAILED (rc={proc.returncode}) ***', flush=True)
            sys.exit(proc.returncode)
    for variant in VARIANTS:
        print(f'--- {TAG} variant={variant} ---', flush=True)
        env = dict(os.environ, **{_ROLE: 'worker', _JOB: json.dumps({'variant': variant})})
        proc = subprocess.run([sys.executable, '-u', os.path.abspath(__file__)], env=env)
        if proc.returncode != 0:
            print(f'*** WORKER FAILED: {variant} (rc={proc.returncode}) ***', flush=True)
            sys.exit(proc.returncode)
    print('ALL RUNS DONE', flush=True)


if __name__ == '__main__':
    main()
