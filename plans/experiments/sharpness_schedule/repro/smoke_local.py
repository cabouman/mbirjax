"""Local CPU smoke test of the full Phase A instrument chain at small size.

Exercises: the equivalence gate, a schedule variant, per-iteration snapshots, the
streak metrics (reference-based + two-seed), and the footprint-enrichment machinery.
NOT a science run -- this size is far below the streak regime; it validates code
paths and prints the tables the real runs will produce.

Run:  python smoke_local.py   (config constants below; no CLI args)
"""

import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'driver'))

from segmented_driver import run_segmented                    # noqa: E402
import equivalence_gate                                       # noqa: E402
import metrics                                                # noqa: E402

# ---------------------------------------------------------------- configuration
SIZE = 64
GATE_ITERATIONS = 6
METRIC_ITERATIONS = 5      # short paired runs for the metric/footprint smoke
SEED_A, SEED_B = 1, 2
# -------------------------------------------------------------------------------


def main():
    t0 = time.time()
    print(f'[1/3] equivalence gate (size {SIZE}, {GATE_ITERATIONS} iterations)...')
    passed, report = equivalence_gate.run_gate(size=SIZE, n_iterations=GATE_ITERATIONS)
    print(report)
    print('gate:', 'PASS' if passed else 'FAIL', f'({time.time() - t0:.0f}s)')
    if not passed:
        sys.exit(1)

    print(f'\n[2/3] paired target-parameter runs for metric smoke '
          f'(seeds {SEED_A}/{SEED_B}, {METRIC_ITERATIONS} iterations)...')
    model, gt_phantom, sinogram = equivalence_gate.build_case(SIZE)
    rec_a = run_segmented(model, sinogram, max_iterations=METRIC_ITERATIONS,
                          seed=SEED_A, snapshot_iterations='all')
    rec_b = run_segmented(model, sinogram, max_iterations=METRIC_ITERATIONS,
                          seed=SEED_B)

    print('\n[3/3] metrics smoke...')
    mask = metrics.interior_mask(gt_phantom.shape)
    ref = metrics.streak_score(rec_a['final_recon'] - gt_phantom, mask=mask)
    two = metrics.two_seed_score(rec_a['final_recon'], rec_b['final_recon'], mask=mask)
    print(f'reference-based S = {ref["S"]:.4g}   (z-incoherent control {ref["control"]:.4g})')
    print(f'two-seed        S = {two["S"]:.4g}   (z-incoherent control {two["control"]:.4g})')

    # Footprint enrichment on iteration 0 (the 4-subset iteration): per-run map from
    # the iteration-0 snapshot vs the ground truth phantom, attributed with that
    # run's own partition + update order.
    it0_map = metrics.streak_map(rec_a['snapshots'][0] - gt_phantom)
    entry0 = rec_a['entry'][0]
    enrichment = metrics.footprint_enrichment(
        it0_map, rec_a['partitions_host'][entry0], rec_a['perm'][0])
    order = ', '.join(f'E({r})={v:.3f}' for r, v in enumerate(enrichment))
    print(f'iteration-0 footprint enrichment by update rank: {order}')
    print(f'(update order = subsets {[int(v) for v in rec_a["perm"][0]]} '
          f'of partition entry {entry0})')

    print(f'\nsmoke complete in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
