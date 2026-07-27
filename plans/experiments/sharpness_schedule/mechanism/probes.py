"""Phase A mechanism probes (plan: "Sweep and mechanism probes"), on the same
synthetic case as a1_sweep:

  coarse_late  -- several fine iterations first, then ONE 4-subset iteration at
                  target sigmas.  Injection there (an S jump + footprint enrichment
                  at that iteration) means coarse granularity itself -- not the early
                  large-residual state -- drives injection, so a schedule must cover
                  coarse iterations whenever they occur.
  q2_control   -- center settings with q = 2.0 (quadratic prior tail, no
                  edge-preserving saturation).  If streaks still persist at q = 2,
                  saturation (P-sat) is not the persistence mechanism.

Both reuse the A1 case builder so the phantom/noise/weights are IDENTICAL to the
sweep.  Outputs mirror the a1_sweep per-run layout, under <scratch>/mechanism/.

Run:  python -u probes.py    (config constants below; no CLI args)
"""

import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'driver'))
sys.path.insert(0, os.path.join(_HERE, '..', 'repro'))

import mbirjax as mj  # mbirjax must be imported before jax (sets XLA env vars)

from segmented_driver import run_segmented                    # noqa: E402
import metrics                                                # noqa: E402
import a1_sweep                                               # noqa: E402

# ---------------------------------------------------------------- configuration
CENTER_S, CENTER_DB = a1_sweep.CENTER_S, a1_sweep.CENTER_DB

# coarse_late: fine (128-subset) iterations to settle the state, then one 4-subset
# iteration -- all at TARGET sigmas.  Entry 7 = 128 subsets, entry 2 = 4 subsets
# (default granularity list).  A DIAGNOSTIC sequence only: fine-to-coarse is never
# an algorithmic recommendation (plan: "Schedule structure").
COARSE_LATE_SEQUENCE = [7, 8, 9, 10, 7, 8, 2, 7]
Q2_ITERATIONS = 15

SEEDS = (1, 2)
_SCRATCH = '/scratch/gautschi/buzzard/sharpness_schedule'
OUTPUT_ROOT = (os.path.join(_SCRATCH, 'mechanism')
               if os.path.isdir(os.path.dirname(_SCRATCH))
               else os.path.join(_HERE, 'output', 'mechanism'))
# -------------------------------------------------------------------------------


def run_probe(model, name, sino, weights, gt_phantom, mask, seeds, max_iterations,
              snapshots, summary, summary_path, t_start):
    """One probe variant through the segmented driver, saved in the A1 layout."""
    runs_by_seed = {}
    vsum = dict(sharpness=float(model.get_params('sharpness')),
                snr_db=float(model.get_params('snr_db')),
                q=float(model.get_params('q')),
                partition_sequence=[int(v) for v in
                                    model.get_params('partition_sequence')],
                per_seed={})
    for seed in seeds:
        print(f'  seed {seed}:', flush=True)
        rec = run_segmented(model, sino, weights=weights,
                            max_iterations=max_iterations, seed=seed,
                            snapshot_iterations=snapshots,
                            per_iteration_hook=a1_sweep.make_hook(
                                model, gt_phantom, mask))
        a1_sweep.save_run(os.path.join(OUTPUT_ROOT, name, f'seed{seed}'), rec,
                          dict(variant=name, seed=seed))
        runs_by_seed[seed] = rec
        vsum['per_seed'][str(seed)] = dict(
            S=[float(h['S']) for h in rec['hook']],
            control=[float(h['control']) for h in rec['hook']],
            alpha=[float(v) for v in rec['alpha']],
            entry=[int(v) for v in rec['entry']],
            targets=[float(v) for v in rec['targets']],
            perm_verified=bool(all(rec['perm_verified'])))
    if len(runs_by_seed) >= 2:
        vsum['two_seed'] = a1_sweep.two_seed_curves(runs_by_seed, mask)
    summary['variants'][name] = vsum
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=1)
    print(f'  [{name} done, elapsed {(time.time() - t_start) / 60:.1f} min]',
          flush=True)


def main():
    t_start = time.time()
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    print(f'mechanism probes -> {OUTPUT_ROOT}', flush=True)

    model, gt_phantom, sino_clean, sino_noisy, weights = a1_sweep.build_case()
    mask = metrics.interior_mask(gt_phantom.shape)
    default_sequence = list(model.get_params('partition_sequence'))
    default_q = float(model.get_params('q'))

    summary = dict(config=dict(coarse_late_sequence=COARSE_LATE_SEQUENCE,
                               q2_iterations=Q2_ITERATIONS), variants={})
    summary_path = os.path.join(OUTPUT_ROOT, 'probes_summary.json')

    # --- coarse_late (center settings, custom sequence, snapshots everywhere) ---
    print(f'\n=== probe coarse_late: sequence {COARSE_LATE_SEQUENCE} ===', flush=True)
    model.set_params(sharpness=CENTER_S, snr_db=CENTER_DB)
    model.set_params(partition_sequence=COARSE_LATE_SEQUENCE)
    run_probe(model, 'coarse_late', sino_noisy, weights, gt_phantom, mask, SEEDS,
              len(COARSE_LATE_SEQUENCE), 'all', summary, summary_path, t_start)
    model.set_params(partition_sequence=default_sequence)

    # --- q2_control (center settings, quadratic prior tail) ---
    print(f'\n=== probe q2_control: q=2.0, {Q2_ITERATIONS} iterations ===', flush=True)
    model.set_params(sharpness=CENTER_S, snr_db=CENTER_DB, q=2.0)
    run_probe(model, 'q2_control', sino_noisy, weights, gt_phantom, mask, SEEDS,
              Q2_ITERATIONS, a1_sweep.SNAPSHOT_ITERATIONS, summary, summary_path,
              t_start)
    model.set_params(q=default_q)

    print(f'\nmechanism probes complete in {(time.time() - t_start) / 60:.1f} min',
          flush=True)


if __name__ == '__main__':
    main()
