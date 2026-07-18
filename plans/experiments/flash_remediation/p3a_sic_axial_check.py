"""Phase 3 / step A validation: the per-end axial extension on the REAL SiC scan.

SiC (`SiC-SiC_CompositeFFOV_tomo-A.txrm`) is the axial-truncation case from the
flash-remediation program: in-FoV laterally but extending past the detector in rows, with
end-slice flash + z-ringing at the truncated end (plan: `plans/flash_remediation/`).
This script A/Bs the ONE variable the step-A library change moved -- the recon slab:

  new -- the standard cache load: `auto_set_recon_geometry` at the current library extends
         each end to the cone-beam visibility bound (det_row_offset-aware, per end).
  old -- same load, then recon_shape / recon_slice_offset forced back to the PRE-extension
         automatic values (central-ray slab, centered at mid-travel) via the escape hatch.

Same library, same cached sinogram, same weights/sequence/seed/regularization -- only the
slab differs.  One continuous checkpointed recon per variant (the global-RNG subset stream
is NOT re-seeded between snapshot boundaries), snapshots saved as slice_viewer h5 plus a
per-iteration change%% trajectory json.  Structure mirrors the proven worker in
`mbirjax_metrics/experiments/partition_sequence/recon_and_save.py`.

Run on gautschi (one GPU), from the staging directory:
    sbatch p3a_sic_axial.slurm
Outputs land in OUT_DIR (depot): <tag>_<variant>_iter<k>.h5 + <tag>_<variant>_log.json.
Figures are rendered by a separate script once the volumes exist.
"""
import json
import os
import subprocess
import sys
import time

import numpy as np

# ---------------- run parameters (edit here; no CLI args) ----------------
CACHE_DIR = '/depot/bouman/data/mbirjax_metrics/partition_sequence/cache'
OUT_DIR = '/depot/bouman/data/mbirjax_metrics/padding'
# Round 1 (DONE 2026-07-11, job 13434902): sic_v4x_d4x_nv401_nch512 -- the fast workhorse.
# Round 2: the 1024-class confirmation; recons/sic_v3x_d2x_nv534_nch1024_default_iter50.h5
# (prerelease-era, same sequence/seed machinery) is the cross-check for the old variant.
DATASETS = ['sic_v3x_d2x_nv534_nch1024']
VARIANTS = ('new', 'old')
SEQUENCE = [0, 2, 4, 6, 7]   # the shipped default sequence, pinned explicitly so the runs
                             # stay comparable to the existing recons/ references
SNAPSHOTS = [15, 50]         # save the volume at these iterations (one continuous run)
SEED = 0
EXPECTED_COMMIT = 'a872695'  # step-A head; the worker asserts the live library matches

_ROLE, _JOB = 'P3A_ROLE', 'P3A_JOB'


def _lib_provenance():
    """(mbirjax.__file__, short commit) of the LIVE import -- printed and stored with every
    output, and asserted against EXPECTED_COMMIT (the editable-install meta-path finder can
    silently serve another checkout; see the P2c provenance lesson)."""
    import mbirjax as mj
    repo = os.path.dirname(os.path.dirname(os.path.abspath(mj.__file__)))
    try:
        commit = subprocess.run(['git', '-C', repo, 'rev-parse', '--short', 'HEAD'],
                                capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        commit = 'unknown'
    return mj.__file__, commit


def _old_auto_shape(model):
    """The PRE-extension automatic recon shape/offset: slices = ceil((H_iso + z_travel) /
    delta_voxel_slice) centered at mid-travel -- exactly what auto_set_recon_geometry
    computed before the visibility extension landed (commit a872695^)."""
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


def _setup(tag, variant):
    """Build the model for one variant + the shared recon inputs (mirrors recon_and_save)."""
    import mbirjax as mj
    import mbirjax.preprocess as mjp
    sino, geometry_params, optional_params, _ = mjp.load_cone_preprocessing(
        os.path.join(CACHE_DIR, f'{tag}.h5'))
    with open(os.path.join(CACHE_DIR, f'{tag}.json')) as f:
        sidecar = json.load(f)
    model = getattr(mj, sidecar['model_class'])(**geometry_params)
    if optional_params:
        model.set_params(**optional_params)
    if sidecar['auto_set_recon_geometry']:
        model.auto_set_recon_geometry()
    if variant == 'old':
        old_shape, old_offset = _old_auto_shape(model)
        model.set_params(recon_shape=old_shape, recon_slice_offset=old_offset)
    model.set_params(verbose=0, **sidecar['recon_settings'])
    model.set_params(partition_sequence=SEQUENCE)
    weights = mj.gen_weights(sino, weight_type='transmission_root')
    recon_shape = model.get_params('recon_shape')

    # Production semantics: partitions generated ONCE under the seed; the per-iteration
    # subset permutations then continue the global-RNG stream unbroken across chunks.
    np.random.seed(SEED)
    partitions = mj.gen_set_of_pixel_partitions(
        recon_shape, model.get_params('granularity'),
        output_device=model.recon_placement.devices[0],
        use_ror_mask=model.get_params('use_ror_mask'))
    seq_ext = np.asarray(mj.gen_partition_sequence(SEQUENCE, max_iterations=max(SNAPSHOTS)))
    model._log_run_header(0, '~/.mbirjax/logs/recon.log', print_logs=False)
    model.auto_set_regularization_params(sino, weights=weights)
    return model, sino, weights, partitions, seq_ext, recon_shape


def worker():
    job = json.loads(os.environ[_JOB])
    tag, variant = job['dataset'], job['variant']
    lib_file, commit = _lib_provenance()
    print(f'WORKER {tag} {variant}: mbirjax={lib_file} commit={commit}', flush=True)
    assert commit == EXPECTED_COMMIT, \
        f'live mbirjax commit {commit} != expected {EXPECTED_COMMIT} -- wrong checkout?'

    model, sino, weights, partitions, seq_ext, recon_shape = _setup(tag, variant)
    offset = float(model.get_params('recon_slice_offset'))
    print(f'  recon_shape={tuple(int(x) for x in recon_shape)} recon_slice_offset={offset:.4f}',
          flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    rows, recon_dev, ckpt, total, prev = [], None, {}, 0.0, 0
    for boundary in sorted(SNAPSHOTS):
        t0 = time.perf_counter()
        recon_dev, loss_vectors, ckpt = model.vcd_recon(
            sino, partitions, seq_ext[prev:boundary], 0.0, weights=weights,
            init_recon=recon_dev, first_iteration=prev,
            init_error_sinogram=ckpt.get('error_sinogram'),
            fm_hessian=ckpt.get('fm_hessian'), return_checkpoint=True)
        total += time.perf_counter() - t0
        changes = [100.0 * float(v) for v in loss_vectors[2]]   # nmae update, percent
        rows += [{'iteration': prev + i + 1, 'change_pct': c, 'time_s': total}
                 for i, c in enumerate(changes)]
        prev = boundary
        recon = np.asarray(recon_dev)[:, :, :recon_shape[2]]    # gather + crop pad slices
        info = {'dataset': tag, 'variant': variant, 'iterations': boundary,
                'sequence': SEQUENCE, 'seed': SEED, 'stop_threshold_change_pct': 0.0,
                'weights': 'transmission_root', 'recon_slice_offset': offset,
                'mbirjax_commit': commit,
                'purpose': 'flash-remediation step-A axial-extension validation'}
        recon_dict = model.get_recon_dict(
            recon_params=info,
            notes=f'p3a axial-extension check: {tag} variant={variant} iter={boundary}')
        out = os.path.join(OUT_DIR, f'{tag}_{variant}_iter{boundary}.h5')
        model.save_recon_hdf5(out, recon, recon_dict)
        print(f'  saved {out}  (iter {boundary}, change {changes[-1]:.4f}%, '
              f'cum t={total:.0f}s, shape {recon.shape})', flush=True)

    log = {'dataset': tag, 'variant': variant, 'sequence': SEQUENCE, 'seed': SEED,
           'recon_shape': [int(x) for x in recon_shape], 'recon_slice_offset': offset,
           'sino_shape': [int(x) for x in sino.shape], 'mbirjax_commit': commit,
           'total_time_s': total, 'rows': rows}
    with open(os.path.join(OUT_DIR, f'{tag}_{variant}_log.json'), 'w') as f:
        json.dump(log, f, indent=1)
    print(f'WORKER {tag} {variant}: done ({total:.0f}s)', flush=True)


def main():
    if os.environ.get(_ROLE) == 'worker':
        worker()
        return
    # Driver: one subprocess per (dataset, variant) -- fresh process = clean device memory
    # and honest isolation between variants.
    for tag in DATASETS:
        for variant in VARIANTS:
            print(f'--- {tag} variant={variant} ---', flush=True)
            env = dict(os.environ, **{_ROLE: 'worker',
                                      _JOB: json.dumps({'dataset': tag, 'variant': variant})})
            proc = subprocess.run([sys.executable, '-u', os.path.abspath(__file__)], env=env)
            if proc.returncode != 0:
                print(f'*** WORKER FAILED: {tag} {variant} (rc={proc.returncode}) ***',
                      flush=True)
                sys.exit(proc.returncode)
    print('ALL RUNS DONE', flush=True)


if __name__ == '__main__':
    main()
