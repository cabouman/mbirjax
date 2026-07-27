"""A2: real-data reproduction on the downsampled BGA scan (plan Phase A).

Loads `bga_no_hart` (Zeiss .txrm on the depot) at the registry's downsampled
setting, builds transmission_root weights, computes/caches a conservative converged
REFERENCE recon (the ground-truth proxy; heavily regularized, long run -- verify
streak-free visually before trusting reference-based curves), then runs the study
variants through the segmented driver with per-iteration metrics vs the reference.
The two-seed discriminator (computed at the snapshot iterations) is the PRIMARY
metric here; reference-based curves are used relatively, per the plan's proxy-bias
caveat.

Gautschi-only (data lives on the depot); outputs to scratch.
Run:  python -u a2_bga.py    (config constants below; no CLI args)
"""

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
# Registry values for 'bga_no_hart' (experiments/IQ_evaluation/dataset_registry.py),
# inlined so this script is self-contained on the cluster:
DATA_PATH = '/depot/bouman/data/Zeiss/purdue_BGA/17U1-250TC-Normal_Tomo_No_HART.txrm'
DOWNSAMPLE_FACTOR = (3, 3)
SUBSAMPLE_VIEW_FACTOR = 5
WEIGHT_TYPE = 'transmission_root'

CENTER_S, CENTER_DB = 1.5, 35.0       # registry recon settings (the streaking case)

# Conservative converged reference (the ground-truth proxy): heavy regularization,
# long run, fixed seed.  Cached on scratch; delete the file to recompute.
REFERENCE_S, REFERENCE_DB = 0.0, 30.0
REFERENCE_ITERATIONS = 60
REFERENCE_SEED = 0

MAX_ITERATIONS = 15
SNAPSHOT_ITERATIONS = (0, 1, 2, 4, 9, 14)


def _v(name, sharpness, snr_db, seeds=(1, 2), damping='default'):
    return dict(name=name, sharpness=float(sharpness), snr_db=float(snr_db),
                seeds=tuple(seeds), damping=damping)


VARIANTS = [
    _v('center', CENTER_S, CENTER_DB, seeds=(1, 2, 3)),
    _v('center_damp_off', CENTER_S, CENTER_DB, damping='off'),
    _v('sharp3.0', 3.0, CENTER_DB),
]

OUTPUT_ROOT = '/scratch/gautschi/buzzard/sharpness_schedule/a2_bga'
# -------------------------------------------------------------------------------


def load_case():
    """Sinogram + model via the standard Zeiss reader (registry settings), plus
    transmission_root weights.  Matches the IQ_evaluation pipeline's loading."""
    sinogram, model = mjp.zeiss.get_sino_and_model(
        DATA_PATH, downsample_factor=DOWNSAMPLE_FACTOR,
        subsample_view_factor=SUBSAMPLE_VIEW_FACTOR)
    model.set_params(verbose=0)
    sinogram = np.asarray(sinogram)
    weights = np.asarray(mj.gen_weights(sinogram, weight_type=WEIGHT_TYPE))
    return model, sinogram, weights


def get_reference(model, sinogram, weights):
    """Load or compute the conservative converged reference recon (cached)."""
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


def make_hook(model, reference, mask):
    """Per-iteration metrics vs the converged reference (RELATIVE use only) plus the
    per-run streak map for the footprint probe."""
    def hook(i, recon_device, ckpt, seg_record):
        err = np.asarray(model._gather_recon(recon_device)) - reference
        sc = metrics.streak_score(err, mask=mask)
        smap = metrics.streak_map(err).astype(np.float32)
        print(f'    it {i:3d}: S={sc["S"]:.4g} ctrl={sc["control"]:.4g} '
              f'alpha={seg_record["alpha"]:.3f} es_rmse={seg_record["es_rmse"]:.4g} '
              f'({seg_record["wall_s"]:.1f}s)', flush=True)
        return dict(S=sc['S'], control=sc['control'], streak_map=smap)
    return hook


def save_run(run_dir, records, run_config):
    """Same layout as a1_sweep.save_run (kept in sync by hand -- small)."""
    os.makedirs(os.path.join(run_dir, 'snapshots'), exist_ok=True)
    series = {k: np.asarray(records[k]) for k in
              ('entry', 'num_subsets', 'sigma_x', 'sigma_y', 'fm_rmse_raw',
               'fm_rmse', 'es_rmse', 'nmae', 'alpha', 'wall_s', 'perm_verified')}
    series['S'] = np.asarray([h['S'] for h in records['hook']])
    series['control'] = np.asarray([h['control'] for h in records['hook']])
    series['streak_maps'] = np.stack([h['streak_map'] for h in records['hook']])
    perms = np.empty(len(records['perm']), dtype=object)
    for j, p in enumerate(records['perm']):
        perms[j] = np.asarray(p)
    series['perms'] = perms
    seq = records['seq']
    for e in sorted({int(v) for v in seq[:3]} | {int(seq[-1])}):
        series[f'partition_entry{e}'] = records['partitions_host'][e]
    np.savez_compressed(os.path.join(run_dir, 'records.npz'), **series)
    for i, vol in records['snapshots'].items():
        np.save(os.path.join(run_dir, 'snapshots', f'it_{i:03d}.npy'),
                vol.astype(np.float32))
    np.save(os.path.join(run_dir, 'final_recon.npy'),
            records['final_recon'].astype(np.float32))
    run_config = dict(run_config)
    run_config.update(targets=[float(v) for v in records['targets']],
                      seq=[int(v) for v in records['seq']],
                      mbirjax_version=getattr(mj, '__version__', 'unknown'))
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(run_config, f, indent=1)


def two_seed_curves(runs_by_seed, mask):
    seeds = sorted(runs_by_seed)[:2]
    a, b = runs_by_seed[seeds[0]], runs_by_seed[seeds[1]]
    out = {'seeds': [int(s) for s in seeds], 'iterations': [], 'S2': [], 'control2': []}
    for i in sorted(set(a['snapshots']) & set(b['snapshots'])):
        sc = metrics.two_seed_score(a['snapshots'][i], b['snapshots'][i], mask=mask)
        out['iterations'].append(int(i))
        out['S2'].append(float(sc['S']))
        out['control2'].append(float(sc['control']))
    sc = metrics.two_seed_score(a['final_recon'], b['final_recon'], mask=mask)
    out['final_S2'] = float(sc['S'])
    out['final_control2'] = float(sc['control'])
    return out


def main():
    t_start = time.time()
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    print(f'A2 BGA -> {OUTPUT_ROOT}', flush=True)

    model, sinogram, weights = load_case()
    print(f'sinogram {sinogram.shape}, recon {model.get_params("recon_shape")}',
          flush=True)
    default_damping = model._dc_damping
    reference = get_reference(model, sinogram, weights)
    mask = metrics.interior_mask(reference.shape)

    summary = dict(config=dict(data=DATA_PATH, downsample=DOWNSAMPLE_FACTOR,
                               view_factor=SUBSAMPLE_VIEW_FACTOR,
                               weight_type=WEIGHT_TYPE,
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

        runs_by_seed = {}
        vsum = dict(sharpness=variant['sharpness'], snr_db=variant['snr_db'],
                    damping=variant['damping'], per_seed={})
        for seed in variant['seeds']:
            print(f'  seed {seed}:', flush=True)
            rec = run_segmented(model, sinogram, weights=weights,
                                max_iterations=MAX_ITERATIONS, seed=seed,
                                snapshot_iterations=SNAPSHOT_ITERATIONS,
                                per_iteration_hook=make_hook(model, reference, mask))
            save_run(os.path.join(OUTPUT_ROOT, name, f'seed{seed}'), rec,
                     dict(variant=variant, seed=seed))
            runs_by_seed[seed] = rec
            vsum['per_seed'][str(seed)] = dict(
                S=[float(h['S']) for h in rec['hook']],
                control=[float(h['control']) for h in rec['hook']],
                alpha=[float(v) for v in rec['alpha']],
                targets=[float(v) for v in rec['targets']],
                perm_verified=bool(all(rec['perm_verified'])))
        if len(runs_by_seed) >= 2:
            vsum['two_seed'] = two_seed_curves(runs_by_seed, mask)
        summary['variants'][name] = vsum
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=1)
        print(f'  [{name} done, elapsed {(time.time() - t_start) / 60:.1f} min]',
              flush=True)

    print(f'\nA2 BGA complete in {(time.time() - t_start) / 60:.1f} min; '
          f'summary at {summary_path}', flush=True)


if __name__ == '__main__':
    main()
