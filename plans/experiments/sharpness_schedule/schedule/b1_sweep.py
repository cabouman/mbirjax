"""b1: the Phase B downsampled search (phase_b_plan.md).

Six balance-matched schedule variants (D/S/J at b in {2, 4} dB/level) + baseline on
the downsampled BGA scan, seeds {1, 2}; 17 iterations, the common snapshot grid,
in-stream metrics incl. the target-objective terms, permanent per-run images, and
all-pairs two-seed scores.

STAGE 2 (set WINNER below after the pre-registered decision rules are applied to the
stage-1 results; the script is idempotent, so rerunning executes only the new work):
seed 3 for baseline + winner, the 40-iteration long-tail pair, and the synthetic
no-harm runs (criterion C3).

Run on gautschi:  python -u b1_sweep.py
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
sys.path.insert(0, os.path.join(_HERE, '..', 'repro'))

import mbirjax as mj  # mbirjax must be imported before jax (sets XLA env vars)
import jax.numpy as jnp

from segmented_driver import run_segmented   # noqa: E402
import a2_bga                                # noqa: E402
import a1_sweep                              # noqa: E402
import metrics                               # noqa: E402
import run_io                                # noqa: E402

# ---------------------------------------------------------------- configuration
CENTER_S, CENTER_DB = 1.5, 35.0          # the case's (target) settings
ITERATIONS = run_io.PHASE_B_ITERATIONS   # 17: gate at 14, fallback at 16
SNAPSHOTS = run_io.PHASE_B_SNAPSHOTS     # {0,1,2,3,4,5,9,14,16}
IMAGE_ITERATIONS = (0, 2, 3, 5, 14, 16)  # per-iteration (x,z) error images
SEEDS = (1, 2)
VARIANTS = [('baseline', None)] + [(f'{fam}{b}', run_io.family_offsets(fam, b))
                                   for fam in 'DSJ' for b in (2, 4)]

WINNER = None            # STAGE 2: set to a variant name (e.g. 'D4') per the
                         # pre-registered decision rules, then rerun.
LONGTAIL_ITERATIONS = 40
OUTPUT_ROOT = '/scratch/gautschi/buzzard/sharpness_schedule/b1'
# -------------------------------------------------------------------------------


def run_case(model, sinogram, weights, reference, mask, case_dir, variant_name,
             offsets, seed, iterations, snapshot_its, z_step=1, prior_loss=True):
    """One idempotent run: driver + hook + save + images."""
    run_dir = os.path.join(case_dir, variant_name, f'seed{seed}')
    if run_io.run_is_complete(run_dir):
        print(f'  [skip complete] {variant_name}/seed{seed}', flush=True)
        return run_dir
    print(f'  running {variant_name}/seed{seed}:', flush=True)
    model.set_params(sharpness=CENTER_S, snr_db=CENTER_DB)
    os.makedirs(run_dir, exist_ok=True)
    # Targets are deterministic per case; compute once here for the hook (the
    # driver recomputes identically inside run_segmented).
    from segmented_driver import compute_targets
    targets = compute_targets(model, sinogram, weights)
    w_dev = jnp.asarray(weights)
    hook = run_io.make_hook(model, reference, mask, run_dir, targets=targets,
                            weights_device=w_dev, z_step=z_step,
                            snapshot_iterations=snapshot_its,
                            prior_loss=prior_loss,
                            image_iterations=IMAGE_ITERATIONS,
                            real_sino_size=math.prod(
                                model.get_params('sinogram_shape')))
    rec = run_segmented(model, sinogram, weights=weights,
                        max_iterations=iterations, seed=seed,
                        offsets_by_entry=offsets, snapshot_iterations=(),
                        per_iteration_hook=hook)
    run_io.save_run(run_dir, rec, dict(variant=variant_name, offsets=offsets,
                                       seed=seed, iterations=iterations))
    rec['final_recon'] = None
    run_io.save_run_images(run_dir, reference,
                           label=f'{variant_name} seed{seed}')
    return run_dir


def variant_summary(case_dir, variant_name, mask, z_step=1):
    """All-pairs two-seed + per-seed series digest for one variant."""
    import glob
    dirs = sorted(glob.glob(os.path.join(case_dir, variant_name, 'seed*')))
    vsum = dict(per_seed={}, run_dirs=[os.path.basename(d) for d in dirs])
    for d in dirs:
        rec = np.load(os.path.join(d, 'records.npz'), allow_pickle=True)
        vsum['per_seed'][os.path.basename(d)] = {
            k: [float(v) for v in rec[k]] for k in
            ('S_low', 'Rz', 'S', 'control', 'es_rmse', 'nrmse',
             'data_term_target', 'prior_target', 'precond_prior_share', 'alpha',
             'sigma_x', 'sigma_y')}
    if len(dirs) >= 2:
        vsum['two_seed'] = run_io.two_seed_curves(dirs, mask, z_step=z_step)
    return vsum


def main():
    t0 = time.time()
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    print(f'b1 sweep -> {OUTPUT_ROOT} (winner={WINNER})', flush=True)

    # ---- downsampled BGA case (shared data for every variant) ----
    model, sinogram, weights = a2_bga.load_case()
    reference = np.load(os.path.join(
        '/scratch/gautschi/buzzard/sharpness_schedule/a2_bga',
        'reference_recon.npy'))
    mask = metrics.interior_mask(reference.shape)
    bga_dir = os.path.join(OUTPUT_ROOT, 'bga')

    summary_path = os.path.join(OUTPUT_ROOT, 'b1_summary.json')
    summary = dict(config=dict(iterations=ITERATIONS, snapshots=list(SNAPSHOTS),
                               seeds=list(SEEDS), winner=WINNER), variants={})

    for name, offsets in VARIANTS:
        seeds = list(SEEDS)
        if WINNER is not None and name in ('baseline', WINNER):
            seeds.append(3)
        print(f'=== {name} (seeds {seeds}) ===', flush=True)
        for seed in seeds:
            run_case(model, sinogram, weights, reference, mask, bga_dir, name,
                     offsets, seed, ITERATIONS, SNAPSHOTS)
        summary['variants'][name] = variant_summary(bga_dir, name, mask)
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=1)
        print(f'  [{name} done, {(time.time() - t0) / 60:.1f} min]', flush=True)

    # ---- identity gates (pre-registered): targets equal; fine tail at target ----
    targets_seen = set()
    for name, _ in VARIANTS:
        for d in sorted(os.listdir(os.path.join(bga_dir, name))):
            with open(os.path.join(bga_dir, name, d, 'config.json')) as f:
                cfg = json.load(f)
            targets_seen.add(tuple(cfg['targets']))
            rec = np.load(os.path.join(bga_dir, name, d, 'records.npz'),
                          allow_pickle=True)
            tail = np.asarray(cfg['seq'][:len(rec['sigma_x'])]) >= 7
            assert np.allclose(rec['sigma_x'][tail], cfg['targets'][0], rtol=0), \
                f'fine-tail sigma_x != target in {name}/{d}'
            assert np.allclose(rec['sigma_y'][tail], cfg['targets'][1], rtol=0), \
                f'fine-tail sigma_y != target in {name}/{d}'
    assert len(targets_seen) == 1, f'targets differ across runs: {targets_seen}'
    print('identity gates: targets identical across runs; fine tails at target',
          flush=True)

    # ---- STAGE 2 extras (winner set): long tail + synthetic no-harm ----
    if WINNER is not None:
        winner_offsets = dict(VARIANTS)[WINNER]
        lt_dir = os.path.join(OUTPUT_ROOT, 'longtail')
        for name, offs in (('baseline', None), (WINNER, winner_offsets)):
            for seed in (1, 2):
                run_case(model, sinogram, weights, reference, mask, lt_dir, name,
                         offs, seed, LONGTAIL_ITERATIONS,
                         tuple(sorted(set(SNAPSHOTS) | {19, 29, 39})))
            summary.setdefault('longtail', {})[name] = variant_summary(
                lt_dir, name, mask)

        # Synthetic no-harm (C3): same machinery on the ball-grid case.
        smodel, gt_phantom, sino_clean, sino_noisy, sweights = a1_sweep.build_case()
        smask = metrics.interior_mask(gt_phantom.shape)
        syn_dir = os.path.join(OUTPUT_ROOT, 'synthetic')
        for name, offs in (('baseline', None), (WINNER, winner_offsets)):
            for seed in (1, 2):
                run_dir = os.path.join(syn_dir, name, f'seed{seed}')
                if run_io.run_is_complete(run_dir):
                    continue
                print(f'  synthetic {name}/seed{seed}:', flush=True)
                smodel.set_params(sharpness=CENTER_S, snr_db=CENTER_DB)
                os.makedirs(run_dir, exist_ok=True)
                from segmented_driver import compute_targets
                stargets = compute_targets(smodel, sino_noisy, sweights)
                hook = run_io.make_hook(
                    smodel, gt_phantom, smask, run_dir, targets=stargets,
                    weights_device=jnp.asarray(sweights),
                    snapshot_iterations=SNAPSHOTS, prior_loss=True,
                    image_iterations=IMAGE_ITERATIONS,
                    real_sino_size=math.prod(
                        smodel.get_params('sinogram_shape')))
                rec = run_segmented(smodel, sino_noisy, weights=sweights,
                                    max_iterations=ITERATIONS, seed=seed,
                                    offsets_by_entry=offs,
                                    snapshot_iterations=(),
                                    per_iteration_hook=hook)
                run_io.save_run(run_dir, rec, dict(variant=name, offsets=offs,
                                                   seed=seed, case='synthetic'))
                rec['final_recon'] = None
                run_io.save_run_images(run_dir, gt_phantom,
                                       label=f'synthetic {name} seed{seed}')
            summary.setdefault('synthetic', {})[name] = variant_summary(
                syn_dir, name, smask)

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=1)
    print(f'b1 complete in {(time.time() - t0) / 60:.1f} min; summary at '
          f'{summary_path}', flush=True)


if __name__ == '__main__':
    main()
