"""b2: Phase B full-resolution confirmation (phase_b_plan.md).

Fresh full-resolution baselines (3 seeds) and the b1-selected winner (3 seeds)
[+ optional runner-up, 2 seeds] on the common snapshot grid, 17 iterations, with the
memory-lean hook (disk snapshots, z_step=3 metrics, no full-volume prior loss).
Two-seed over all pairs; permanent per-run images; snapshot volumes deleted after
each variant's digest (the ~0.6 TB storage note in the plan).

Set WINNER (and optionally RUNNER_UP) below before submitting.
Run on gautschi (2 GPUs):  sbatch run_b2_gautschi.sh
"""

import json
import math
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'driver'))
sys.path.insert(0, os.path.join(_HERE, '..', 'real_bga'))

import mbirjax as mj  # mbirjax must be imported before jax (sets XLA env vars)
import jax.numpy as jnp

from segmented_driver import run_segmented, compute_targets   # noqa: E402
import a3_fullres                                             # noqa: E402
import metrics                                                # noqa: E402
import run_io                                                 # noqa: E402

# ---------------------------------------------------------------- configuration
CENTER_S, CENTER_DB = 1.5, 35.0
ITERATIONS = run_io.PHASE_B_ITERATIONS
SNAPSHOTS = run_io.PHASE_B_SNAPSHOTS
IMAGE_ITERATIONS = (0, 2, 3, 5, 14, 16)
Z_STEP = 3
KEEP_SNAPSHOTS = False        # delete per-run snapshot volumes after the digest

WINNER = None                 # REQUIRED: set from b1 per the decision rules
RUNNER_UP = None              # optional (decision rule 2 only)
SEEDS_MAIN = (1, 2, 3)        # baseline and winner
SEEDS_RUNNER = (1, 2)

OUTPUT_ROOT = '/scratch/gautschi/buzzard/sharpness_schedule/b2'
A3_ROOT = '/scratch/gautschi/buzzard/sharpness_schedule/a3_fullres'
# -------------------------------------------------------------------------------


def main():
    assert WINNER is not None, 'set WINNER from the b1 decision before submitting'
    t0 = time.time()
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    print(f'b2 full-res -> {OUTPUT_ROOT} (winner={WINNER}, runner_up={RUNNER_UP})',
          flush=True)

    model, sinogram, weights = a3_fullres.load_case()
    print(f'sinogram {sinogram.shape}, recon {model.get_params("recon_shape")}',
          flush=True)
    ref_path = os.path.join(A3_ROOT, 'reference_recon.npy')
    assert os.path.exists(ref_path), \
        f'{ref_path} missing (scratch purge?) -- regen via a3_fullres (~23 min)'
    reference = np.load(ref_path)
    mask = metrics.interior_mask(reference.shape)
    # Weights in the model's sinogram device form so the per-iteration weighted
    # reduction runs sharded next to the checkpoint error sinogram.
    weights_device = model._shard_sinogram(jnp.asarray(weights))
    real_sino_size = math.prod(model.get_params('sinogram_shape'))

    fam = WINNER[0]
    variants = [('baseline', None, SEEDS_MAIN),
                (WINNER, run_io.family_offsets(fam, float(WINNER[1:])), SEEDS_MAIN)]
    if RUNNER_UP is not None:
        variants.append((RUNNER_UP,
                         run_io.family_offsets(RUNNER_UP[0], float(RUNNER_UP[1:])),
                         SEEDS_RUNNER))

    summary_path = os.path.join(OUTPUT_ROOT, 'b2_summary.json')
    summary = dict(config=dict(iterations=ITERATIONS, snapshots=list(SNAPSHOTS),
                               winner=WINNER, runner_up=RUNNER_UP, z_step=Z_STEP),
                   variants={})

    for name, offsets, seeds in variants:
        print(f'=== {name} (seeds {list(seeds)}) ===', flush=True)
        run_dirs = []
        for seed in seeds:
            run_dir = os.path.join(OUTPUT_ROOT, name, f'seed{seed}')
            run_dirs.append(run_dir)
            if run_io.run_is_complete(run_dir):
                print(f'  [skip complete] {name}/seed{seed}', flush=True)
                continue
            print(f'  running {name}/seed{seed}:', flush=True)
            model.set_params(sharpness=CENTER_S, snr_db=CENTER_DB)
            os.makedirs(run_dir, exist_ok=True)
            targets = compute_targets(model, sinogram, weights)
            hook = run_io.make_hook(model, reference, mask, run_dir,
                                    targets=targets,
                                    weights_device=weights_device, z_step=Z_STEP,
                                    snapshot_iterations=SNAPSHOTS,
                                    prior_loss=False,
                                    image_iterations=IMAGE_ITERATIONS,
                                    real_sino_size=real_sino_size)
            rec = run_segmented(model, sinogram, weights=weights,
                                max_iterations=ITERATIONS, seed=seed,
                                offsets_by_entry=offsets, snapshot_iterations=(),
                                per_iteration_hook=hook)
            run_io.save_run(run_dir, rec, dict(variant=name, offsets=offsets,
                                               seed=seed, iterations=ITERATIONS,
                                               case='fullres'))
            rec['final_recon'] = None
            run_io.save_run_images(run_dir, reference,
                                   label=f'full-res {name} seed{seed}')

        vsum = dict(per_seed={}, run_dirs=[os.path.basename(d) for d in run_dirs])
        for d in run_dirs:
            rec = np.load(os.path.join(d, 'records.npz'), allow_pickle=True)
            vsum['per_seed'][os.path.basename(d)] = {
                k: [float(v) for v in rec[k]] for k in
                ('S_low', 'Rz', 'S', 'control', 'es_rmse', 'nrmse',
                 'data_term_target', 'precond_prior_share', 'alpha',
                 'sigma_x', 'sigma_y')}
        vsum['two_seed'] = run_io.two_seed_curves(run_dirs, mask, z_step=Z_STEP)
        summary['variants'][name] = vsum
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=1)
        if not KEEP_SNAPSHOTS:
            for d in run_dirs:
                run_io.delete_snapshots(d)
            print(f'  [snapshots deleted for {name}]', flush=True)
        print(f'  [{name} done, {(time.time() - t0) / 60:.1f} min]', flush=True)

    print(f'b2 complete in {(time.time() - t0) / 60:.1f} min; summary at '
          f'{summary_path}', flush=True)


if __name__ == '__main__':
    main()
