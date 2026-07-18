"""Phase 3 / step C follow-up: LATERAL padding added to the (now default) axial extension,
on the two scans the new warning fires for -- so the three-way comparison is complete:
no padding (old default) vs axial (new default) vs axial + lateral (the warning's remedy).

Variants (smallest first, so a memory failure on the big BGA run cannot block the rest):
  lilly_ds4  s=1.25          -- mild one-sided truncation (16% edge fraction); compare
                                against the p3c unsplit reference (same seed/iterations).
  bga        s=1.5 and s=2.0 -- severe truncation (86%); two scales because "cover" is
                                genuinely unknown and the P2b knee is the question: if the
                                two agree we are at/past cover, if s=2 is much better we
                                were still under at 1.5.  Compare against p3b new/old.

Composition detail: scale_recon_shape is deliberately a PURE scaler, but lateral growth
enlarges the support radius R and hence the axial visibility bound -- so after scaling
laterally this script grows the slice axis to the new bound (mirroring the per-end block
in ConeBeamModel.auto_set_recon_geometry, measured against the CURRENT slab: a slab that
already over-covers, like Lilly's NSI-inflated extension, gets no growth).  This is the
"compensating slice growth" the planned cone scale_recon_shape warning will point at.

Run on gautschi (one GPU):  sbatch p3e_lateral_pad.slurm
Outputs land in OUT_DIR: volumes (h5 for BGA, npy for Lilly to pair with p3c), trajectory
logs, and the applied-shape json per variant.
"""
import json
import os
import subprocess
import sys
import time

import numpy as np

# ---------------- run parameters (edit here; no CLI args) ----------------
OUT_DIR = '/depot/bouman/data/mbirjax_metrics/padding'
BGA_CACHE = os.path.join(OUT_DIR, 'bga_normal_v2x_d2x_cache')
LILLY_DIR = '/scratch/gautschi/buzzard/flash_lilly/D01788'
SEQUENCE = [0, 2, 4, 6, 7]
SEED = 0
EXPECTED_COMMIT = '41ecbc2'
JOBS = [
    # (label, kind, lateral_scale, snapshots)
    ('lilly_ds4_lat125', 'lilly', 1.25, [15]),
    ('bga_lat150', 'bga', 1.5, [15, 50]),
    ('bga_lat200', 'bga', 2.0, [15, 50]),
]

_ROLE, _JOB = 'P3E_ROLE', 'P3E_JOB'


def _lib_provenance():
    import mbirjax as mj
    repo = os.path.dirname(os.path.dirname(os.path.abspath(mj.__file__)))
    try:
        commit = subprocess.run(['git', '-C', repo, 'rev-parse', '--short', 'HEAD'],
                                capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        commit = 'unknown'
    return mj.__file__, commit


def extend_axially_to_bound(model):
    """Grow the slice axis (per end) to the visibility bound implied by the CURRENT
    rows/cols, measured against the CURRENT slab.  Mirrors the per-end extension in
    ConeBeamModel.auto_set_recon_geometry; returns (n_bot, n_top) added."""
    import mbirjax as mj
    rows, cols, num_slices = (int(x) for x in model.get_params('recon_shape'))
    delta_voxel, voxel_row_aspect, voxel_slice_aspect = model.get_params(
        ['delta_voxel', 'voxel_row_aspect', 'voxel_slice_aspect'])
    delta_det_row, delta_det_channel = model.get_params(['delta_det_row', 'delta_det_channel'])
    det_row_offset, det_channel_offset = model.get_params(['det_row_offset', 'det_channel_offset'])
    sdd, use_ror_mask = model.get_params(['source_detector_dist', 'use_ror_mask'])
    num_det_rows, num_det_channels = (int(x) for x in model.get_params('sinogram_shape')[1:3])
    offset = float(model.get_params('recon_slice_offset'))
    magnification = model.get_magnification()
    delta_voxel_slice = voxel_slice_aspect * delta_voxel

    support_radius = mj.get_support_radius((rows, cols), voxel_row_aspect * delta_voxel,
                                           delta_voxel, use_ror_mask=use_ror_mask)
    _, v_lo = model.detector_mn_to_uv(-0.5, 0.0, delta_det_channel, delta_det_row,
                                      det_channel_offset, det_row_offset,
                                      num_det_rows, num_det_channels)
    _, v_hi = model.detector_mn_to_uv(num_det_rows - 0.5, 0.0, delta_det_channel,
                                      delta_det_row, det_channel_offset, det_row_offset,
                                      num_det_rows, num_det_channels)
    v_top, v_bot = max(float(v_lo), float(v_hi)), min(float(v_lo), float(v_hi))
    factor = 1.0 / float(magnification) + support_radius / float(sdd)
    half_slab = num_slices * float(delta_voxel_slice) / 2.0
    z_top_have, z_bot_have = offset + half_slab, offset - half_slab
    n_top = max(0, int(np.ceil((v_top * factor - z_top_have) / delta_voxel_slice)))
    n_bot = max(0, int(np.ceil((z_bot_have - v_bot * factor) / delta_voxel_slice)))
    model.set_params(recon_shape=(rows, cols, num_slices + n_top + n_bot),
                     recon_slice_offset=offset + 0.5 * (n_top - n_bot) * float(delta_voxel_slice))
    return n_bot, n_top


def _build(kind):
    import mbirjax as mj
    import mbirjax.preprocess as mjp
    if kind == 'bga':
        sino, geometry_params, optional_params, _ = mjp.load_cone_preprocessing(BGA_CACHE + '.h5')
        with open(BGA_CACHE + '.json') as f:
            sidecar = json.load(f)
        model = getattr(mj, sidecar['model_class'])(**geometry_params)
        model.set_params(**optional_params)
        model.auto_set_recon_geometry()
        recon_settings = sidecar['recon_settings']
    else:   # lilly, mirroring p3c_lilly_split_check
        sino, cone_params, optional_params = mjp.nsi.compute_sino_and_params(
            LILLY_DIR, downsample_factor=(4, 4), subsample_view_factor=2)
        sino = np.asarray(sino)
        model = mj.ConeBeamModel(**cone_params)
        model.set_params(**optional_params)
        recon_settings = {}
    model.set_params(verbose=0, **recon_settings)
    model.set_params(partition_sequence=SEQUENCE)
    weights = np.asarray(mj.gen_weights(sino, 'transmission_root'))
    return model, sino, weights


def worker():
    import mbirjax as mj
    job = json.loads(os.environ[_JOB])
    label, kind, scale, snapshots = job['label'], job['kind'], job['scale'], job['snapshots']
    lib_file, commit = _lib_provenance()
    print(f'WORKER {label}: mbirjax={lib_file} commit={commit}', flush=True)
    assert commit == EXPECTED_COMMIT, \
        f'live mbirjax commit {commit} != expected {EXPECTED_COMMIT} -- wrong checkout?'

    model, sino, weights = _build(kind)
    base_shape = tuple(int(x) for x in model.get_params('recon_shape'))
    added = model.scale_recon_shape(row_scale=scale, col_scale=scale)
    n_bot, n_top = extend_axially_to_bound(model)
    shape = tuple(int(x) for x in model.get_params('recon_shape'))
    offset = float(model.get_params('recon_slice_offset'))
    print(f'  base {base_shape} -> lateral x{scale} (+{added[0]},{added[1]} px) '
          f'-> axial +({n_bot},{n_top}) => {shape}, offset {offset:.4f}', flush=True)
    with open(os.path.join(OUT_DIR, f'p3e_{label}_shape.json'), 'w') as f:
        json.dump({'label': label, 'lateral_scale': scale, 'base_shape': list(base_shape),
                   'final_shape': list(shape), 'axial_added': [n_bot, n_top],
                   'recon_slice_offset': offset, 'commit': commit}, f, indent=1)

    # Regularization from the sinogram (as the p3b/p3c runs did), then frozen.
    model.auto_set_regularization_params(sino, weights=weights)
    model.set_params(auto_regularize_flag=False)

    np.random.seed(SEED)
    partitions = mj.gen_set_of_pixel_partitions(
        shape, model.get_params('granularity'),
        output_device=model.recon_placement.devices[0],
        use_ror_mask=model.get_params('use_ror_mask'))
    seq_ext = np.asarray(mj.gen_partition_sequence(SEQUENCE, max_iterations=max(snapshots)))
    model._log_run_header(0, '~/.mbirjax/logs/recon.log', print_logs=False)

    rows_log, recon_dev, ckpt, total, prev = [], None, {}, 0.0, 0
    for boundary in sorted(snapshots):
        t0 = time.perf_counter()
        recon_dev, loss_vectors, ckpt = model.vcd_recon(
            sino, partitions, seq_ext[prev:boundary], 0.0, weights=weights,
            init_recon=recon_dev, first_iteration=prev,
            init_error_sinogram=ckpt.get('error_sinogram'),
            fm_hessian=ckpt.get('fm_hessian'), return_checkpoint=True)
        total += time.perf_counter() - t0
        changes = [100.0 * float(v) for v in loss_vectors[2]]
        rows_log += [{'iteration': prev + i + 1, 'change_pct': c, 'time_s': total}
                     for i, c in enumerate(changes)]
        prev = boundary
        recon = np.asarray(recon_dev)[:, :, :shape[2]]
        if kind == 'bga':
            info = {'dataset': label, 'lateral_scale': scale, 'iterations': boundary,
                    'sequence': SEQUENCE, 'seed': SEED, 'mbirjax_commit': commit,
                    'purpose': 'flash-remediation step-C lateral+axial padding comparison'}
            recon_dict = model.get_recon_dict(recon_params=info,
                                              notes=f'p3e {label} iter={boundary}')
            out = os.path.join(OUT_DIR, f'p3e_{label}_iter{boundary}.h5')
            model.save_recon_hdf5(out, recon, recon_dict)
        else:
            out = os.path.join(OUT_DIR, f'p3e_{label}_iter{boundary}.npy')
            np.save(out, recon)
        print(f'  saved {out}  (iter {boundary}, change {changes[-1]:.4f}%, '
              f'cum t={total:.0f}s)', flush=True)

    log = {'label': label, 'kind': kind, 'lateral_scale': scale, 'sequence': SEQUENCE,
           'seed': SEED, 'recon_shape': list(shape), 'recon_slice_offset': offset,
           'sino_shape': [int(x) for x in sino.shape], 'mbirjax_commit': commit,
           'total_time_s': total, 'rows': rows_log}
    with open(os.path.join(OUT_DIR, f'p3e_{label}_log.json'), 'w') as f:
        json.dump(log, f, indent=1)
    print(f'WORKER {label}: done ({total:.0f}s)', flush=True)


def main():
    if os.environ.get(_ROLE) == 'worker':
        worker()
        return
    lib_file, commit = _lib_provenance()
    print(f'DRIVER: mbirjax={lib_file} commit={commit}', flush=True)
    failed = []
    for label, kind, scale, snapshots in JOBS:
        marker = os.path.join(OUT_DIR, f'p3e_{label}_log.json')
        if os.path.exists(marker):
            print(f'--- {label}: exists, skipped', flush=True)
            continue
        print(f'--- {label} ---', flush=True)
        env = dict(os.environ, **{_ROLE: 'worker',
                                  _JOB: json.dumps({'label': label, 'kind': kind,
                                                    'scale': scale, 'snapshots': snapshots})})
        proc = subprocess.run([sys.executable, '-u', os.path.abspath(__file__)], env=env)
        if proc.returncode != 0:
            # Keep going: a memory failure on the big BGA scale must not block the others.
            print(f'*** WORKER FAILED: {label} (rc={proc.returncode}) -- continuing ***',
                  flush=True)
            failed.append(label)
    print(f'ALL RUNS DONE (failed: {failed or "none"})', flush=True)


if __name__ == '__main__':
    main()
