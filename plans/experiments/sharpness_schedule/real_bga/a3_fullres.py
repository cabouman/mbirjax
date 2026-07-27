"""A3: FULL-RESOLUTION BGA reproduction -- the regime where the streaking is
"serious" and survives (plan Phase A / A2 full-res arm).

Same structure as a2_bga.py (segmented driver, conservative converged reference,
center + damping-off variants, two-seed at snapshots) at the registry's full-res
setting: no detector downsampling, view factor 4 (2401 -> 601 views, 1532 channels;
recon ~1532 x 1532 x ~970).  The per-iteration hook records BOTH metric v1
(z-constant S + control) and metric v2 (axial-spectrum S_low/S_high/Rz), with
z_step=3 slice subsampling for cost (the low-f_z band this targets is unaffected).

Memory: a single H100 should hold this (~9 GB recon x2 + ~3.6 GB sino x3 + subset
transients); if the job OOMs, resubmit with --gpus-per-node=2 (the driver and
snapshots are sharding-safe via _gather_recon).

Gautschi-only; outputs to scratch.  Run:  python -u a3_fullres.py
"""

import glob
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'driver'))

import mbirjax as mj  # mbirjax must be imported before jax (sets XLA env vars)
import mbirjax.preprocess as mjp

from segmented_driver import run_segmented, run_continuous   # noqa: E402
import metrics                                               # noqa: E402

# ---------------------------------------------------------------- configuration
DATA_PATH = '/depot/bouman/data/Zeiss/purdue_BGA/17U1-250TC-Normal_Tomo_No_HART.txrm'
DOWNSAMPLE_FACTOR = (1, 1)          # full detector resolution
SUBSAMPLE_VIEW_FACTOR = 4           # 2401 -> 601 views (registry full-res setting)
WEIGHT_TYPE = 'transmission_root'

CENTER_S, CENTER_DB = 1.5, 35.0     # registry recon settings (the streaking case)
REFERENCE_S, REFERENCE_DB = 0.0, 30.0
REFERENCE_ITERATIONS = 60
REFERENCE_SEED = 0

MAX_ITERATIONS = 15
SNAPSHOT_ITERATIONS = (0, 1, 4, 9, 14)
Z_STEP = 3                          # slice subsampling for all metrics (cost)

VARIANTS = [
    dict(name='center', sharpness=CENTER_S, snr_db=CENTER_DB,
         seeds=(1, 2, 3), damping='default'),
    dict(name='center_damp_off', sharpness=CENTER_S, snr_db=CENTER_DB,
         seeds=(1, 2), damping='off'),
]

OUTPUT_ROOT = '/scratch/gautschi/buzzard/sharpness_schedule/a3_fullres'
# -------------------------------------------------------------------------------


def load_case():
    sinogram, model = mjp.zeiss.get_sino_and_model(
        DATA_PATH, downsample_factor=DOWNSAMPLE_FACTOR,
        subsample_view_factor=SUBSAMPLE_VIEW_FACTOR)
    model.set_params(verbose=0)
    sinogram = np.asarray(sinogram)
    weights = np.asarray(mj.gen_weights(sinogram, weight_type=WEIGHT_TYPE))
    return model, sinogram, weights


def get_reference(model, sinogram, weights):
    ref_path = os.path.join(OUTPUT_ROOT, 'reference_recon.npy')
    if os.path.exists(ref_path):
        print(f'reference: loading cached {ref_path}', flush=True)
        return np.load(ref_path)
    print(f'reference: computing (sharpness={REFERENCE_S}, snr_db={REFERENCE_DB}, '
          f'{REFERENCE_ITERATIONS} iterations)...', flush=True)
    model.set_params(sharpness=REFERENCE_S, snr_db=REFERENCE_DB)
    t0 = time.time()
    ref = run_continuous(model, sinogram, weights=weights,
                         max_iterations=REFERENCE_ITERATIONS,
                         seed=REFERENCE_SEED)['final_recon']
    print(f'reference: done in {(time.time() - t0) / 60:.1f} min', flush=True)
    np.save(ref_path, ref.astype(np.float32))
    return ref


def make_hook(model, reference, mask, run_dir):
    """v1 + v2 metrics vs the converged reference, per iteration, z_step-subsampled.

    The hook ALSO writes the snapshot volumes to disk itself (it already holds the
    gathered volume for the metrics), so the driver is called with
    snapshot_iterations=() and never accumulates ~9 GB volumes in records -- the
    host-memory OOM that killed job 14202362.  Big temporaries are freed explicitly.
    """
    snap_dir = os.path.join(run_dir, 'snapshots')
    os.makedirs(snap_dir, exist_ok=True)

    def hook(i, recon_device, ckpt, seg_record):
        vol = np.asarray(model._gather_recon(recon_device))
        if i in SNAPSHOT_ITERATIONS:
            np.save(os.path.join(snap_dir, f'it_{i:03d}.npy'),
                    vol.astype(np.float32))
        err = vol - reference
        del vol
        sc = metrics.streak_score(err, mask=mask, z_step=Z_STEP)
        freqs, power = metrics.axial_power_spectrum(err, mask=mask, z_step=Z_STEP)
        v2 = metrics.zcoherence_summary(freqs, power)
        smap = metrics.streak_map(err, z_step=Z_STEP).astype(np.float32)
        del err
        print(f'    it {i:3d}: S={sc["S"]:.4g} ctrl={sc["control"]:.4g} '
              f'S_low={v2["S_low"]:.4g} Rz={v2["Rz"]:.2f} '
              f'alpha={seg_record["alpha"]:.3f} ({seg_record["wall_s"]:.1f}s)',
              flush=True)
        return dict(S=sc['S'], control=sc['control'], streak_map=smap,
                    S_low=v2['S_low'], S_high=v2['S_high'], Rz=v2['Rz'],
                    power=power.astype(np.float32))
    return hook


def save_run(run_dir, records, run_config):
    """a2_bga.save_run + the v2 series (kept in sync by hand -- small)."""
    os.makedirs(os.path.join(run_dir, 'snapshots'), exist_ok=True)
    series = {k: np.asarray(records[k]) for k in
              ('entry', 'num_subsets', 'sigma_x', 'sigma_y', 'fm_rmse_raw',
               'fm_rmse', 'es_rmse', 'nmae', 'alpha', 'wall_s', 'perm_verified')}
    for key in ('S', 'control', 'S_low', 'S_high', 'Rz'):
        series[key] = np.asarray([h[key] for h in records['hook']])
    series['streak_maps'] = np.stack([h['streak_map'] for h in records['hook']])
    series['powers'] = np.stack([h['power'] for h in records['hook']])
    perms = np.empty(len(records['perm']), dtype=object)
    for j, p in enumerate(records['perm']):
        perms[j] = np.asarray(p)
    series['perms'] = perms
    seq = records['seq']
    for e in sorted({int(v) for v in seq[:3]} | {int(seq[-1])}):
        series[f'partition_entry{e}'] = records['partitions_host'][e]
    np.savez_compressed(os.path.join(run_dir, 'records.npz'), **series)
    # Snapshot volumes were written by the hook (see make_hook); only the final
    # recon is written here, and the caller drops it from memory right after.
    np.save(os.path.join(run_dir, 'final_recon.npy'),
            records['final_recon'].astype(np.float32))
    run_config = dict(run_config)
    run_config.update(targets=[float(v) for v in records['targets']],
                      seq=[int(v) for v in records['seq']],
                      z_step=Z_STEP,
                      mbirjax_version=getattr(mj, '__version__', 'unknown'))
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(run_config, f, indent=1)


def two_seed_curves(dir_a, dir_b, mask):
    """v1 + v2 two-seed scores at common snapshot iterations + final.

    DISK-BASED: loads one pair of volumes at a time from the two run directories
    and frees them between iterations (holding every seed's volumes in memory is
    what OOM-killed the first attempt)."""
    def both(path_a, path_b):
        va, vb = np.load(path_a), np.load(path_b)
        sc = metrics.two_seed_score(va, vb, mask=mask, z_step=Z_STEP)
        freqs, power = metrics.two_seed_spectrum(va, vb, mask=mask, z_step=Z_STEP)
        del va, vb
        v2 = metrics.zcoherence_summary(freqs, power)
        return dict(S2=float(sc['S']), control2=float(sc['control']),
                    S2_low=float(v2['S_low']), Rz2=float(v2['Rz']))

    def snaps(d):
        return {int(os.path.basename(p)[3:6]): p for p in
                sorted(glob.glob(os.path.join(d, 'snapshots', 'it_*.npy')))}

    sa, sb = snaps(dir_a), snaps(dir_b)
    out = {'iterations': [], 'points': []}
    for i in sorted(set(sa) & set(sb)):
        out['iterations'].append(int(i))
        out['points'].append(both(sa[i], sb[i]))
    out['final'] = both(os.path.join(dir_a, 'final_recon.npy'),
                        os.path.join(dir_b, 'final_recon.npy'))
    return out


def main():
    t_start = time.time()
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    print(f'A3 full-res BGA -> {OUTPUT_ROOT}', flush=True)

    model, sinogram, weights = load_case()
    print(f'sinogram {sinogram.shape}, recon {model.get_params("recon_shape")}',
          flush=True)
    default_damping = model._dc_damping
    reference = get_reference(model, sinogram, weights)
    mask = metrics.interior_mask(reference.shape)

    summary = dict(config=dict(data=DATA_PATH, downsample=DOWNSAMPLE_FACTOR,
                               view_factor=SUBSAMPLE_VIEW_FACTOR,
                               weight_type=WEIGHT_TYPE, z_step=Z_STEP,
                               reference=[REFERENCE_S, REFERENCE_DB,
                                          REFERENCE_ITERATIONS]),
                   variants={})
    summary_path = os.path.join(OUTPUT_ROOT, 'sweep_summary.json')

    for variant in VARIANTS:
        name = variant['name']
        print(f'\n=== variant {name}: sharpness={variant["sharpness"]} '
              f'snr_db={variant["snr_db"]} damping={variant["damping"]} ===',
              flush=True)
        model.set_params(sharpness=variant['sharpness'], snr_db=variant['snr_db'])
        model._dc_damping = None if variant['damping'] == 'off' else default_damping

        run_dirs = []
        vsum = dict(sharpness=variant['sharpness'], snr_db=variant['snr_db'],
                    damping=variant['damping'], per_seed={})
        for seed in variant['seeds']:
            print(f'  seed {seed}:', flush=True)
            run_dir = os.path.join(OUTPUT_ROOT, name, f'seed{seed}')
            rec = run_segmented(model, sinogram, weights=weights,
                                max_iterations=MAX_ITERATIONS, seed=seed,
                                snapshot_iterations=(),   # the hook snapshots to disk
                                per_iteration_hook=make_hook(model, reference, mask,
                                                             run_dir))
            save_run(run_dir, rec, dict(variant=variant, seed=seed))
            vsum['per_seed'][str(seed)] = dict(
                S=[float(h['S']) for h in rec['hook']],
                control=[float(h['control']) for h in rec['hook']],
                S_low=[float(h['S_low']) for h in rec['hook']],
                Rz=[float(h['Rz']) for h in rec['hook']],
                alpha=[float(v) for v in rec['alpha']],
                targets=[float(v) for v in rec['targets']],
                perm_verified=bool(all(rec['perm_verified'])))
            # Drop the 9 GB final volume (it is on disk now); keep records light.
            rec['final_recon'] = None
            run_dirs.append(run_dir)
        if len(run_dirs) >= 2:
            vsum['two_seed'] = two_seed_curves(run_dirs[0], run_dirs[1], mask)
        summary['variants'][name] = vsum
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=1)
        print(f'  [{name} done, elapsed {(time.time() - t_start) / 60:.1f} min]',
              flush=True)

    print(f'\nA3 full-res complete in {(time.time() - t_start) / 60:.1f} min; '
          f'summary at {summary_path}', flush=True)


if __name__ == '__main__':
    main()
