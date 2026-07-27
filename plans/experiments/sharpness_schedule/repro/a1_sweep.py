"""A1: synthetic cone-beam severity map + formation/decay curves (plan Phase A).

One fixed 256-scale cone case (ball-grid ground truth phantom, transmission noise,
transmission_root weights from the CLEAN sinogram), swept over (sharpness, snr_db)
variants oriented along and across the balance diagonal, plus the center-anchored
validation/mechanism variants (damping off, noise off, long tail).  Every run goes
through the segmented driver, records per-iteration reference metrics + streak maps
in-stream, and saves selected snapshot volumes; each multi-seed variant also gets
two-seed scores (the primary discriminator) at the common snapshot iterations.

Outputs (per run): <out>/<variant>/seed<k>/{config.json, records.npz, snapshots/*.npy}
Sweep-level: <out>/{sweep_summary.json, gt_phantom.npy, sinogram_noisy.npy, ...}
On gautschi the output root is scratch (large files never land in home); locally it
is ./output next to this script.

Run:  python -u a1_sweep.py    (config constants below; no CLI args)
Set A1_SMOKE = True for a tiny local validation config (separate output dir).
"""

import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'driver'))

import mbirjax as mj  # mbirjax must be imported before jax (sets XLA env vars)
import jax.numpy as jnp

from segmented_driver import run_segmented          # noqa: E402
from phantom import ball_grid_phantom               # noqa: E402
from noise import add_transmission_noise            # noqa: E402
import metrics                                      # noqa: E402

# ---------------------------------------------------------------- configuration
A1_SMOKE = False       # True = tiny local validation config (see block below)

SIZE = 256             # detector rows = channels; recon ~ SIZE^3
NUM_VIEWS = 100        # ~0.4 * channels, matching the full-res BGA view sparsity
I0 = 1.0e4             # incident photons/element (transmission noise level)
TARGET_MAX_SINO = 5.0  # phantom scaled so the max line integral is ~5 (physical
                       # transmission regime; gen_weights warns above ~5, and larger
                       # values underflow the Poisson counts at I0=1e4)
NOISE_SEED = 7         # independent of partition seeds; ONE realization for all runs
MAX_ITERATIONS = 15
SNAPSHOT_ITERATIONS = (0, 1, 2, 4, 9, 14)
LONGTAIL_ITERATIONS = 60
LONGTAIL_SNAPSHOTS = (0, 1, 2, 4, 9, 19, 39, 59)

CENTER_S, CENTER_DB = 1.5, 35.0    # registry-matched center (bga_no_hart)
BALANCE_SLOPE = -6.02              # d snr_db per unit sharpness at constant balance


def _v(name, sharpness, snr_db, seeds=(1, 2), noise=True, damping='default',
       max_iterations=None, snapshots=None):
    return dict(name=name, sharpness=float(sharpness), snr_db=float(snr_db),
                seeds=tuple(seeds), noise=bool(noise), damping=damping,
                max_iterations=int(max_iterations or MAX_ITERATIONS),
                snapshots=tuple(snapshots or SNAPSHOT_ITERATIONS))


VARIANTS = [
    _v('center', CENTER_S, CENTER_DB, seeds=(1, 2, 3)),
    # Across the diagonal -- sharpness axis at fixed snr_db (balance varies):
    _v('sharp0.0', 0.0, CENTER_DB),
    _v('sharp1.0', 1.0, CENTER_DB),
    _v('sharp2.0', 2.0, CENTER_DB),
    _v('sharp3.0', 3.0, CENTER_DB),
    # Across the diagonal -- snr_db axis at fixed sharpness:
    _v('snr25', CENTER_S, 25.0),
    _v('snr30', CENTER_S, 30.0),
    _v('snr40', CENTER_S, 40.0),
    _v('snr45', CENTER_S, 45.0),
    # Along the balance diagonal through the center (collapse test, prediction 3):
    _v('diag_t-1', CENTER_S - 1.0, CENTER_DB - BALANCE_SLOPE),
    _v('diag_t+1', CENTER_S + 1.0, CENTER_DB + BALANCE_SLOPE),
    _v('diag_t+2', CENTER_S + 2.0, CENTER_DB + 2 * BALANCE_SLOPE),
    # Metric validation + mechanism discriminators at the center:
    _v('center_damp_off', CENTER_S, CENTER_DB, damping='off'),
    _v('center_noise_off', CENTER_S, CENTER_DB, noise=False),
    # Long tail -- decay rate at the center (prediction 1):
    _v('center_long', CENTER_S, CENTER_DB,
       max_iterations=LONGTAIL_ITERATIONS, snapshots=LONGTAIL_SNAPSHOTS),
]

if A1_SMOKE:
    SIZE, NUM_VIEWS, MAX_ITERATIONS = 64, 48, 4
    SNAPSHOT_ITERATIONS = (0, 1, 3)
    VARIANTS = [
        _v('center', CENTER_S, CENTER_DB, seeds=(1, 2),
           max_iterations=4, snapshots=(0, 1, 3)),
        _v('center_noise_off', CENTER_S, CENTER_DB, seeds=(1,), noise=False,
           max_iterations=4, snapshots=(0, 1, 3)),
    ]

_SCRATCH = '/scratch/gautschi/buzzard/sharpness_schedule'
if os.path.isdir(os.path.dirname(_SCRATCH)):
    OUTPUT_ROOT = os.path.join(_SCRATCH, 'a1_smoke' if A1_SMOKE else 'a1')
else:
    OUTPUT_ROOT = os.path.join(_HERE, 'output', 'a1_smoke' if A1_SMOKE else 'a1')
# -------------------------------------------------------------------------------


def build_case():
    """Model + ground truth phantom + clean/noisy sinograms + weights (built once;
    geometry and data are identical across all variants)."""
    sinogram_shape = (NUM_VIEWS, SIZE, SIZE)
    angles = jnp.linspace(0, 2 * np.pi, NUM_VIEWS, endpoint=False)
    source_detector_dist = 4.0 * SIZE
    model = mj.ConeBeamModel(sinogram_shape, angles,
                             source_detector_dist=source_detector_dist,
                             source_iso_dist=source_detector_dist / 2.0)
    model.set_params(verbose=0)
    gt_phantom = ball_grid_phantom(model.get_params('recon_shape'))
    sino_clean = np.asarray(model.forward_project(gt_phantom))
    # Scale phantom + sinogram (projection is linear) so the max line integral sits
    # in the physical transmission regime -- otherwise counts = I0*exp(-sino)
    # underflow and the noisy sinogram saturates at log(I0).
    scale = TARGET_MAX_SINO / float(sino_clean.max())
    gt_phantom = (gt_phantom * scale).astype(np.float32)
    sino_clean = (sino_clean * scale).astype(np.float32)
    sino_noisy, weights = add_transmission_noise(sino_clean, i0=I0,
                                                 noise_seed=NOISE_SEED)
    return model, gt_phantom, sino_clean, sino_noisy, weights


def make_hook(model, gt_phantom, mask):
    """Per-iteration in-stream metrics vs the ground truth phantom: streak score,
    z-incoherent control, and the (rows, cols) streak map for the footprint probe."""
    def hook(i, recon_device, ckpt, seg_record):
        err = np.asarray(model._gather_recon(recon_device)) - gt_phantom
        sc = metrics.streak_score(err, mask=mask)
        smap = metrics.streak_map(err).astype(np.float32)
        print(f'    it {i:3d}: S={sc["S"]:.4g} ctrl={sc["control"]:.4g} '
              f'alpha={seg_record["alpha"]:.3f} es_rmse={seg_record["es_rmse"]:.4g} '
              f'({seg_record["wall_s"]:.1f}s)', flush=True)
        return dict(S=sc['S'], control=sc['control'], streak_map=smap)
    return hook


def save_run(run_dir, records, run_config):
    """Persist one run: small series in records.npz, snapshot volumes as .npy."""
    os.makedirs(os.path.join(run_dir, 'snapshots'), exist_ok=True)
    series = {k: np.asarray(records[k]) for k in
              ('entry', 'num_subsets', 'sigma_x', 'sigma_y', 'fm_rmse_raw',
               'fm_rmse', 'es_rmse', 'nmae', 'alpha', 'wall_s', 'perm_verified')}
    series['S'] = np.asarray([h['S'] for h in records['hook']])
    series['control'] = np.asarray([h['control'] for h in records['hook']])
    series['streak_maps'] = np.stack([h['streak_map'] for h in records['hook']])
    # Object array built explicitly: np.asarray(..., dtype=object) would collapse
    # equal-length permutations into a 2-D object array instead of a list of arrays.
    perms = np.empty(len(records['perm']), dtype=object)
    for j, p in enumerate(records['perm']):
        perms[j] = np.asarray(p)
    series['perms'] = perms
    # Partitions for the entries the footprint probe needs (first three iterations)
    # plus the tail entry; the full set is regenerable from the seed.
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
    """Two-seed scores at the snapshot iterations common to the first two seeds,
    plus at the final iteration.  Returns a jsonable dict."""
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
    print(f'A1 sweep -> {OUTPUT_ROOT}  (smoke={A1_SMOKE})', flush=True)

    model, gt_phantom, sino_clean, sino_noisy, weights = build_case()
    default_damping = model._dc_damping
    np.save(os.path.join(OUTPUT_ROOT, 'gt_phantom.npy'), gt_phantom)
    np.save(os.path.join(OUTPUT_ROOT, 'sinogram_noisy.npy'), sino_noisy)
    mask = metrics.interior_mask(gt_phantom.shape)

    summary = dict(config=dict(size=SIZE, num_views=NUM_VIEWS, i0=I0,
                               noise_seed=NOISE_SEED, center=[CENTER_S, CENTER_DB],
                               balance_slope=BALANCE_SLOPE), variants={})
    summary_path = os.path.join(OUTPUT_ROOT, 'sweep_summary.json')

    for variant in VARIANTS:
        name = variant['name']
        print(f'\n=== variant {name}: sharpness={variant["sharpness"]} '
              f'snr_db={variant["snr_db"]} noise={variant["noise"]} '
              f'damping={variant["damping"]} ===', flush=True)
        model.set_params(sharpness=variant['sharpness'], snr_db=variant['snr_db'])
        model._dc_damping = None if variant['damping'] == 'off' else default_damping
        sino = sino_noisy if variant['noise'] else sino_clean

        runs_by_seed = {}
        vsum = dict(sharpness=variant['sharpness'], snr_db=variant['snr_db'],
                    noise=variant['noise'], damping=variant['damping'], per_seed={})
        for seed in variant['seeds']:
            print(f'  seed {seed}:', flush=True)
            rec = run_segmented(model, sino, weights=weights,
                                max_iterations=variant['max_iterations'], seed=seed,
                                snapshot_iterations=variant['snapshots'],
                                per_iteration_hook=make_hook(model, gt_phantom, mask))
            run_dir = os.path.join(OUTPUT_ROOT, name, f'seed{seed}')
            save_run(run_dir, rec, dict(variant=variant, seed=seed))
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
        with open(summary_path, 'w') as f:      # rewrite whole summary each variant
            json.dump(summary, f, indent=1)     # (crash-robust incremental record)
        print(f'  [{name} done, elapsed {(time.time() - t_start) / 60:.1f} min]',
              flush=True)

    print(f'\nA1 sweep complete in {(time.time() - t_start) / 60:.1f} min; '
          f'summary at {summary_path}', flush=True)


if __name__ == '__main__':
    main()
