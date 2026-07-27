"""B0: noise calibration from the EXISTING Phase A runs (no new reconstructions).

Anchors Phase B's pre-registered thresholds before any schedule run
(phase_b_plan.md): all pairwise two-seed S_low values (seed pairs 1-2, 1-3, 2-3) at
every saved snapshot for the baseline at both scales, plus the seed spread and
iteration-13/14 decrement of es_rmse.  Thresholds may be revised ONCE from these
numbers, then frozen.

Run on a compute node:  python -u b0_calibration.py
"""

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'driver'))

import metrics   # noqa: E402
import run_io    # noqa: E402

# ---------------------------------------------------------------- configuration
_SCRATCH = '/scratch/gautschi/buzzard/sharpness_schedule'
CASES = {
    'downsampled': dict(root=os.path.join(_SCRATCH, 'a2_bga'),
                        reference='reference_recon.npy', z_step=1),
    'fullres': dict(root=os.path.join(_SCRATCH, 'a3_fullres'),
                    reference='reference_recon.npy', z_step=3),
}
OUT_PATH = os.path.join(_SCRATCH, 'b1', 'b0_calibration.json')
# -------------------------------------------------------------------------------


def main():
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out = {}
    for tag, cfg in CASES.items():
        root = cfg['root']
        dirs = sorted(d for d in
                      (os.path.join(root, 'center', f'seed{s}') for s in (1, 2, 3))
                      if os.path.isdir(d))
        print(f'[{tag}] {len(dirs)} baseline seeds', flush=True)
        reference = np.load(os.path.join(root, cfg['reference']), mmap_mode='r')
        mask = metrics.interior_mask(reference.shape)
        ts = run_io.two_seed_curves(dirs, mask, z_step=cfg['z_step'])

        finals = [p['final']['S2_low'] for p in ts['pairs']]
        spread = max(finals) / min(finals) if finals else float('nan')
        es14, dec = [], []
        for d in dirs:
            rec = np.load(os.path.join(d, 'records.npz'), allow_pickle=True)
            es = rec['es_rmse']
            es14.append(float(es[14] if len(es) > 14 else es[-1]))
            dec.append(float((es[13] - es[14]) / es[13])
                       if len(es) > 14 else float('nan'))
        out[tag] = dict(two_seed=ts, final_S2_low_per_pair=finals,
                        pair_ratio_max_over_min=spread,
                        es_rmse_it14_per_seed=es14,
                        es_rmse_seed_spread=(max(es14) - min(es14)) / min(es14),
                        es_rmse_late_decrement_per_seed=dec)
        print(f'[{tag}] final S2_low per pair: '
              + ' '.join(f'{v:.4g}' for v in finals)
              + f' | max/min = {spread:.2f}', flush=True)
        print(f'[{tag}] es_rmse@14 per seed: '
              + ' '.join(f'{v:.6g}' for v in es14)
              + f' | spread = {out[tag]["es_rmse_seed_spread"]:.2e}'
              + f' | late decrement/it: '
              + ' '.join(f'{v:.3f}' for v in dec), flush=True)
    with open(OUT_PATH, 'w') as f:
        json.dump(out, f, indent=1)
    print('B0 calibration written to', OUT_PATH, flush=True)


if __name__ == '__main__':
    main()
