"""The padded long pair on the real scan: does padding help the persistent
streak field, slowly — or is that field mostly not lateral-truncation-driven?

The laterally padded downsampled BGA baseline run to 60 iterations, seeds 1-2
— the padded analog of the unpadded long pair (e1_longtail_bga.py), sharing
its rulers: per-iteration severity vs the conservative reference, the two-seed
decay, and all-iteration snapshots for per-bin two-seed spectra.

Grid-alignment fix: 1.5 x 510 = 765 gives an ODD padding delta, so the padded
voxel centers sit half a voxel off the original grid and vs-reference numbers
inherit a registration artifact.  Here the scale is chosen so the padded size
has an EVEN delta (>= the verified 1.5x coverage), making the central crop
voxel-exact against the reference and the unpadded runs.

Run on gautschi:  python -u e4_pad_long_bga.py
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
PAD_SCALE_CANDIDATES = (1.502, 1.506, 1.498, 1.51)   # first EVEN-delta wins
ITERATIONS = 60
SEEDS = (1, 2)
IMAGE_ITS = (0, 4, 9, 14, 29, 44, 59)
OUTPUT_ROOT = '/scratch/gautschi/buzzard/sharpness_schedule/e4_pad_long'
REFERENCE_PATH = ('/scratch/gautschi/buzzard/sharpness_schedule/a2_bga/'
                  'reference_recon.npy')
# -------------------------------------------------------------------------------


def main():
    t0 = time.time()
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    reference = np.load(REFERENCE_PATH)
    orig_shape = reference.shape

    model = sinogram = weights = None
    pad_scale = None
    for s in PAD_SCALE_CANDIDATES:
        model, sinogram, weights = a2_bga.load_case()
        model.scale_recon_shape(s, s)
        pad_shape = model.get_params('recon_shape')
        dr = pad_shape[0] - orig_shape[0]
        dc = pad_shape[1] - orig_shape[1]
        if dr % 2 == 0 and dc % 2 == 0 and dr > 0 and dc > 0:
            pad_scale = s
            break
        print(f'scale {s}: padded {pad_shape} -> odd delta ({dr},{dc}); '
              f'retrying', flush=True)
    if pad_scale is None:
        raise SystemExit('no candidate scale gave an even padding delta')
    model.set_params(sharpness=CENTER_S, snr_db=CENTER_DB)
    r0 = (pad_shape[0] - orig_shape[0]) // 2
    c0 = (pad_shape[1] - orig_shape[1]) // 2
    crop_rc = (r0, c0, orig_shape[0], orig_shape[1])
    mask = metrics.interior_mask(orig_shape)
    print(f'e4 pad long: scale {pad_scale} -> padded {pad_shape}, even delta, '
          f'crop {crop_rc} (voxel-exact vs reference)', flush=True)

    case_dir = os.path.join(OUTPUT_ROOT, f'bga_s{pad_scale:g}')
    run_dirs = []
    for seed in SEEDS:
        run_dir = os.path.join(case_dir, f'seed{seed}')
        run_dirs.append(run_dir)
        if run_io.run_is_complete(run_dir):
            print(f'[seed{seed}] already complete; skipped', flush=True)
            continue
        print(f'=== padded long run seed{seed} ===', flush=True)
        hook = run_io.make_crop_hook(
            model, reference, mask, run_dir, crop_rc=crop_rc, z_step=1,
            snapshot_iterations=tuple(range(ITERATIONS)),
            image_iterations=IMAGE_ITS,
            label=f'padded x{pad_scale:g} seed{seed}')
        rec = run_segmented(model, sinogram, weights=weights,
                            max_iterations=ITERATIONS, seed=seed,
                            per_iteration_hook=hook)
        full_final = rec['final_recon']
        np.save(os.path.join(run_dir, 'final_recon_full.npy'),
                full_final.astype(np.float32))
        rec['final_recon'] = full_final[r0:r0 + orig_shape[0],
                                        c0:c0 + orig_shape[1], :]
        run_io.save_run(run_dir, rec, dict(
            experiment='e4_pad_long_bga', pad_scale=pad_scale,
            variant='baseline', seed=seed, iterations=ITERATIONS,
            sharpness=CENTER_S, snr_db=CENTER_DB))
        run_io.save_run_images(run_dir, reference,
                               label=f'padded x{pad_scale:g} seed{seed} (60 it)')
        print(f'[seed{seed}] done ({(time.time() - t0) / 60:.1f} min)',
              flush=True)

    ts = run_io.two_seed_curves(run_dirs, mask, z_step=1)
    with open(os.path.join(case_dir, 'two_seed.json'), 'w') as f:
        json.dump(ts, f, indent=1)
    n = run_io.two_seed_powers(run_dirs, mask,
                               os.path.join(case_dir, 'two_seed_powers.npz'))
    print(f'two-seed curves + {n} per-bin spectra written', flush=True)

    fa = np.load(os.path.join(run_dirs[0], 'final_recon.npy'))
    fb = np.load(os.path.join(run_dirs[1], 'final_recon.npy'))
    diff = fa - fb
    out = dict(pad_scale=pad_scale,
               seed_pair_rel_max=float(np.max(np.abs(diff))
                                       / np.max(np.abs(fa))),
               seed_pair_rms=float(np.sqrt(np.mean(diff ** 2))),
               run_rms=float(np.sqrt(np.mean(fa ** 2))))
    with open(os.path.join(case_dir, 'reconcile.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print(f'reconcile: {out}', flush=True)
    print(f'e4 pad long complete in {(time.time() - t0) / 60:.1f} min',
          flush=True)


if __name__ == '__main__':
    main()
