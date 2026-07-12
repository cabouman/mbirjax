"""R1: real-data schedule protocol on Lilly ds8 (slice_parity_plan.md, R1 section).

Runs the four candidate (S, P) schedules C0-C3 at two sharpness settings on the Lilly
D01788 cone dataset (ds8, view-subsample 8), against 150-iteration references, with the
per-slice/cropped-NRMSE diagnostics and the cone cost accounting from the protocol.

Structure: main() preprocesses ONCE (cached npz in STAGE_DIR), generates references
first (fail-fast), then runs each (schedule, sharpness) arm in its own SUBPROCESS
(fresh jax, honest memory), mask-form parity via the P1 ParityMixin (bitwise-verified
copy of the library updater).  Restart-per-iteration capture, seeded per call.

Run (gautschi, 1 GPU):  sbatch plans/experiments/slice_parity/parity_realdata.slurm
"""
import json
import os
import pickle
import subprocess
import sys
import time

import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_DIR = '/scratch/gautschi/buzzard/flash_lilly/D01788'
STAGE_DIR = os.path.expanduser('~/parity_lilly')
DOWNSAMPLE_FACTOR = (8, 8)
SUBSAMPLE_VIEW_FACTOR = 8
SHARPNESS_LIST = [1.0, 2.5]
NUM_ITERATIONS = 30
REF_ITERATIONS = 150
SEED = 1000
INTERIOR_RADIUS_FRAC = 0.85     # cropped-metric disk (flash-analysis pattern)
AXIAL_CROP_FRAC = 0.10          # exclude this fraction of slices at each end

# (name, partition_sequence, phases_spec) — phases_spec int or per-iteration list.
SCHEDULES = [
    ('C0_default',   [0, 2, 4, 6, 7], 1),
    ('C1_parityall', [0, 2, 4, 6, 7], 2),
    ('C2_composite', [1, 3, 5, 7],    [2, 2, 2, 1]),
    ('C3_flat128',   [7],             1),
]


def load_case():
    """Preprocess once, cache to STAGE_DIR (the zeiss save/reload pattern)."""
    cache = os.path.join(STAGE_DIR, 'lilly_ds8_case.npz')
    pkl = os.path.join(STAGE_DIR, 'lilly_ds8_params.pkl')
    if os.path.exists(cache) and os.path.exists(pkl):
        d = np.load(cache)
        with open(pkl, 'rb') as f:
            cone_params, optional_params = pickle.load(f)
        return d['sino'], d['weights'], cone_params, optional_params
    import mbirjax as mj
    import mbirjax.preprocess as mjp
    sino, cone_params, optional_params = mjp.nsi.compute_sino_and_params(
        DATASET_DIR, downsample_factor=DOWNSAMPLE_FACTOR,
        subsample_view_factor=SUBSAMPLE_VIEW_FACTOR)
    sino = np.asarray(sino)
    weights = np.asarray(mj.gen_weights(sino, 'transmission_root'))
    os.makedirs(STAGE_DIR, exist_ok=True)
    np.savez_compressed(cache, sino=sino, weights=weights)
    with open(pkl, 'wb') as f:
        pickle.dump((cone_params, optional_params), f)
    return sino, weights, cone_params, optional_params


def build_model(cone_params, optional_params, sharpness, sino, parity_cls=True):
    import mbirjax as mj
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from parity_convergence_ab import ParityMixin

    class ParityCone(ParityMixin, mj.ConeBeamModel):
        pass

    cls = ParityCone if parity_cls else mj.ConeBeamModel
    model = cls(**cone_params)
    model.set_params(**optional_params)
    model.set_params(sharpness=sharpness, verbose=0)
    model.auto_set_regularization_params(sino)
    model.set_params(auto_regularize_flag=False)
    return model


def crop_masks(recon_shape):
    nz = recon_shape[2]
    z0, z1 = int(np.ceil(nz * AXIAL_CROP_FRAC)), nz - int(np.ceil(nz * AXIAL_CROP_FRAC))
    i = np.arange(recon_shape[0], dtype=np.float32)[:, None] - (recon_shape[0] - 1) / 2.0
    j = np.arange(recon_shape[1], dtype=np.float32)[None, :] - (recon_shape[1] - 1) / 2.0
    disk = np.sqrt(i ** 2 + j ** 2) < INTERIOR_RADIUS_FRAC * (min(recon_shape[:2]) / 2.0)
    return disk, z0, z1


def worker(cfg):
    os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.94')
    import mbirjax as mj  # noqa: F401  (device init before heavy work)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from parity_convergence_ab import phase_masks

    out = dict(cfg)
    try:
        sino, weights, cone_params, optional_params = load_case()
        model = build_model(cone_params, optional_params, cfg['sharpness'], sino)
        reference = np.load(os.path.join(
            STAGE_DIR, f'ref_sharp{cfg["sharpness"]}.npy'))
        recon_shape = model.get_params('recon_shape')
        disk, z0, z1 = crop_masks(recon_shape)
        ref_crop_norm = np.linalg.norm(reference[disk][:, z0:z1])
        num_slices = recon_shape[2]

        pseq, phases_spec = cfg['pseq'], cfg['phases']
        errs, cropped_lognrmse = [], []
        init_recon = None
        t0 = time.perf_counter()
        for j in range(NUM_ITERATIONS):
            phases_j = (phases_spec if isinstance(phases_spec, int)
                        else phases_spec[min(j, len(phases_spec) - 1)])
            model.parity_masks = phase_masks(num_slices, phases_j)
            model.set_params(partition_sequence=pseq)
            np.random.seed(SEED)
            recon, _ = model.recon(sino, weights=weights, init_recon=init_recon,
                                   first_iteration=j, max_iterations=j + 1,
                                   stop_threshold_change_pct=0, print_logs=False)
            recon = np.asarray(recon)
            diff = recon - reference
            errs.append(np.linalg.norm(diff, axis=(0, 1)))          # per-slice, uncropped
            cropped_lognrmse.append(float(np.log10(
                np.linalg.norm(diff[disk][:, z0:z1]) / ref_crop_norm + 1e-30)))
            init_recon = recon
        out['wall_s'] = round(time.perf_counter() - t0, 1)

        # Idealized cost units per iteration (protocol: back 1, fwd P => (P+1)/2 on cone).
        cost_units = [((phases_spec if isinstance(phases_spec, int)
                        else phases_spec[min(j, len(phases_spec) - 1)]) + 1) / 2.0
                      for j in range(NUM_ITERATIONS)]
        np.savez_compressed(
            os.path.join(STAGE_DIR, f'{cfg["name"]}_sharp{cfg["sharpness"]}.npz'),
            errs=np.stack(errs), cropped_lognrmse=np.array(cropped_lognrmse),
            cost_units=np.array(cost_units), final_recon=recon)
        out['cropped_lognrmse'] = [round(v, 4) for v in cropped_lognrmse]
        out['cum_cost'] = [round(v, 1) for v in np.cumsum(cost_units)]
        out['status'] = 'ok'
    except Exception:
        import traceback
        out['status'] = 'error'
        out['traceback'] = traceback.format_exc()
    print('RESULT ' + json.dumps(out), flush=True)


def main():
    os.makedirs(STAGE_DIR, exist_ok=True)
    print('[preprocess] loading/caching the ds8 case...', flush=True)
    sino, weights, cone_params, optional_params = load_case()
    print(f'[preprocess] sino {sino.shape}', flush=True)

    # References first (fail-fast; library solver, default sequence).
    for sharpness in SHARPNESS_LIST:
        ref_path = os.path.join(STAGE_DIR, f'ref_sharp{sharpness}.npy')
        if os.path.exists(ref_path):
            continue
        print(f'[reference] sharpness {sharpness}: {REF_ITERATIONS} iterations...',
              flush=True)
        model = build_model(cone_params, optional_params, sharpness, sino,
                            parity_cls=False)
        np.random.seed(SEED)
        ref, _ = model.recon(sino, weights=weights, max_iterations=REF_ITERATIONS,
                             stop_threshold_change_pct=0, print_logs=False)
        np.save(ref_path, np.asarray(ref))
        del model, ref

    results = []
    for sharpness in SHARPNESS_LIST:
        for name, pseq, phases in SCHEDULES:
            cfg = dict(name=name, pseq=pseq, phases=phases, sharpness=sharpness)
            proc = subprocess.run([sys.executable, os.path.abspath(__file__),
                                   '--worker', json.dumps(cfg)],
                                  capture_output=True, text=True)
            got = False
            for line in proc.stdout.splitlines():
                if line.startswith('RESULT '):
                    r = json.loads(line[len('RESULT '):])
                    results.append(r)
                    got = True
                    tail = (r['cropped_lognrmse'][-1] if r['status'] == 'ok' else 'ERR')
                    print(f'[{name} s{sharpness}] final cropped log10 NRMSE: {tail} '
                          f'({r.get("wall_s", "?")}s)', flush=True)
            if not got:
                print(f'[no RESULT rc={proc.returncode}] {cfg}\n{proc.stderr[-1500:]}',
                      flush=True)

    with open(os.path.join(STAGE_DIR, 'r1_summary.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n[summary + arrays in {STAGE_DIR}]')


if __name__ == '__main__':
    if len(sys.argv) >= 3 and sys.argv[1] == '--worker':
        worker(json.loads(sys.argv[2]))
    else:
        main()
