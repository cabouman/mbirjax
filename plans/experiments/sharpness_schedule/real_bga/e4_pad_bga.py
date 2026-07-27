"""E4 on the real scan: lateral padding as discriminator and candidate remedy.

The downsampled BGA baseline reconstructed on a laterally enlarged grid
(scale_recon_shape 1.5 — the flash Phase 3 verified value for this scan; the truncated object gets room, so the unexplained
attenuation has a legitimate home outside the original field of view), at the
practical 17-iteration budget, seeds 1-2, registry settings.

Scored on the central crop matching the original grid so every number shares
a ruler with the unpadded baseline: primary = two-seed S_low at iteration 14
vs the b1 baseline's 0.00151 (a >= 2x reduction would beat every schedule
variant); vs-conservative-reference numbers are recorded as a solution-shift
diagnostic (the padded objective differs by design).  Full padded finals are
kept for boundary inspection; per-iteration (x,z) images are permanent.

Run on gautschi:  python -u e4_pad_bga.py
"""

import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'driver'))

import mbirjax as mj  # mbirjax must be imported before jax (sets XLA env vars)

from segmented_driver import run_segmented                    # noqa: E402
import a2_bga                                                 # noqa: E402
import metrics                                                # noqa: E402
import run_io                                                 # noqa: E402

# ---------------------------------------------------------------- configuration
CENTER_S, CENTER_DB = 1.5, 35.0
PAD_SCALE = 1.5
ITERATIONS = run_io.PHASE_B_ITERATIONS      # 17: the practical budget
SNAPSHOT_ITS = run_io.PHASE_B_SNAPSHOTS     # {0,1,2,3,4,5,9,14,16}
SEEDS = (1, 2)
IMAGE_ITS = (0, 2, 3, 5, 14, 16)
OUTPUT_ROOT = '/scratch/gautschi/buzzard/sharpness_schedule/e4_pad'
REFERENCE_PATH = ('/scratch/gautschi/buzzard/sharpness_schedule/a2_bga/'
                  'reference_recon.npy')
# -------------------------------------------------------------------------------


def main():
    t0 = time.time()
    case_dir = os.path.join(OUTPUT_ROOT, f'bga_s{PAD_SCALE:g}')
    os.makedirs(case_dir, exist_ok=True)
    model, sinogram, weights = a2_bga.load_case()
    reference = np.load(REFERENCE_PATH)
    orig_shape = reference.shape
    model.scale_recon_shape(PAD_SCALE, PAD_SCALE)
    model.set_params(sharpness=CENTER_S, snr_db=CENTER_DB)
    pad_shape = model.get_params('recon_shape')
    r0 = (pad_shape[0] - orig_shape[0]) // 2
    c0 = (pad_shape[1] - orig_shape[1]) // 2
    crop_rc = (r0, c0, orig_shape[0], orig_shape[1])
    mask = metrics.interior_mask(orig_shape)
    print(f'e4 pad bga: padded grid {pad_shape} (scale {PAD_SCALE}), '
          f'crop {crop_rc} -> {orig_shape}', flush=True)

    run_dirs = []
    for seed in SEEDS:
        run_dir = os.path.join(case_dir, f'seed{seed}')
        run_dirs.append(run_dir)
        if run_io.run_is_complete(run_dir):
            print(f'[seed{seed}] already complete; skipped', flush=True)
            continue
        print(f'=== padded run seed{seed} ===', flush=True)
        hook = run_io.make_crop_hook(
            model, reference, mask, run_dir, crop_rc=crop_rc, z_step=1,
            snapshot_iterations=SNAPSHOT_ITS, image_iterations=IMAGE_ITS,
            label=f'padded x{PAD_SCALE} seed{seed}')
        rec = run_segmented(model, sinogram, weights=weights,
                            max_iterations=ITERATIONS, seed=seed,
                            per_iteration_hook=hook)
        full_final = rec['final_recon']
        np.save(os.path.join(run_dir, 'final_recon_full.npy'),
                full_final.astype(np.float32))
        rec['final_recon'] = full_final[r0:r0 + orig_shape[0],
                                        c0:c0 + orig_shape[1], :]
        run_io.save_run(run_dir, rec, dict(
            experiment='e4_pad_bga', pad_scale=PAD_SCALE, variant='baseline',
            seed=seed, iterations=ITERATIONS, sharpness=CENTER_S,
            snr_db=CENTER_DB))
        run_io.save_run_images(run_dir, reference,
                               label=f'e4 padded x{PAD_SCALE} seed{seed}')
        print(f'[seed{seed}] done ({(time.time() - t0) / 60:.1f} min)',
              flush=True)

    ts = run_io.two_seed_curves(run_dirs, mask, z_step=1)
    with open(os.path.join(case_dir, 'two_seed.json'), 'w') as f:
        json.dump(ts, f, indent=1)
    pair = ts['pairs'][0]
    at14 = [p['S2_low'] for i, p in zip(pair['iterations'], pair['points'])
            if i == 14]
    print(f'two-seed S2_low@14 (padded) = '
          f'{at14[0] if at14 else float("nan"):.4g}  '
          f'[b1 unpadded baseline: 0.00151]', flush=True)
    print(f'e4 pad bga complete in {(time.time() - t0) / 60:.1f} min',
          flush=True)


if __name__ == '__main__':
    main()
