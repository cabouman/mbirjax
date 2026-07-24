"""R2 follow-up: capture D0 (default) vs D1 (drop-0) recon VOLUMES at 15/20/30
iterations for visual head-to-head comparison on all real cases (Greg 2026-07-12:
the metrics can mislead — want images).

Reuses parity_realdata's case configs/loaders; plain library solver (no parity
machinery — both schedules are P=1).  Orchestrator stays JAX-free (every GPU arm in a
subprocess).  Volumes + capture_summary.json go to STAGE on scratch; figures are built
separately by r2_recon_figs.py.

Run (gautschi, 1 GPU): sbatch plans/experiments/slice_parity/r2_recon_capture.slurm
"""
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parity_realdata import CASES, SEED, build_model, crop_masks, load_case, stage_dir  # noqa: E402,F401

# ── Config ────────────────────────────────────────────────────────────────────
STAGE = '/scratch/gautschi/buzzard/parity_recons'
SNAP_ITERS = [15, 20, 30]               # save the volume after this many iterations
SHARPNESS_LIST = [1.0, 2.0]
RUN_CASES = ['lilly_ds8', 'z62', 'lilly_ds4']
ARMS = [('D0_default', [0, 2, 4, 6, 7]),
        ('D1_g2start', [2, 4, 6, 7])]


def worker(cfg):
    os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.94')
    import mbirjax as mj  # noqa: F401  (device init before heavy work)
    out = dict(cfg)
    try:
        sino, weights, geom_params, optional_params = load_case(cfg['case'])
        model = build_model(cfg['case'], geom_params, optional_params,
                            cfg['sharpness'], sino, parity_cls=False)
        model.set_params(partition_sequence=cfg['pseq'])
        # Metric tie-in where a 150-iteration reference exists (sanity check that the
        # captured trajectories match the R2 numbers; ds4 s2.0 has no reference).
        ref_path = os.path.join(stage_dir(cfg['case']),
                                f'ref_sharp{cfg["sharpness"]}.npy')
        reference = np.load(ref_path) if os.path.exists(ref_path) else None

        recon, first, metrics = None, 0, {}
        for it in SNAP_ITERS:
            np.random.seed(SEED)        # same per-call partition draw as the runners
            recon, _ = model.recon(sino, weights=weights, init_recon=recon,
                                   first_iteration=first, max_iterations=it,
                                   stop_threshold_change_pct=0, print_logs=False)
            recon = np.asarray(recon)
            np.save(os.path.join(
                STAGE,
                f'{cfg["case"]}_{cfg["name"]}_s{cfg["sharpness"]}_it{it}.npy'), recon)
            if reference is not None:
                disk, z0, z1 = crop_masks(recon.shape)
                num = np.linalg.norm((recon - reference)[disk][:, z0:z1])
                den = np.linalg.norm(reference[disk][:, z0:z1])
                metrics[f'it{it}'] = round(float(np.log10(num / den + 1e-30)), 4)
            first = it
        out['cropped_lognrmse'] = metrics
        out['status'] = 'ok'
    except Exception:
        import traceback
        out['status'] = 'error'
        out['traceback'] = traceback.format_exc()
    print('RESULT ' + json.dumps(out), flush=True)


def main():
    os.makedirs(STAGE, exist_ok=True)
    results = []
    for case in RUN_CASES:
        for s in SHARPNESS_LIST:
            for name, pseq in ARMS:
                cfg = dict(case=case, name=name, pseq=pseq, sharpness=s)
                proc = subprocess.run([sys.executable, os.path.abspath(__file__),
                                       '--worker', json.dumps(cfg)],
                                      capture_output=True, text=True)
                got = False
                for line in proc.stdout.splitlines():
                    if line.startswith('RESULT '):
                        r = json.loads(line[len('RESULT '):])
                        results.append(r)
                        got = True
                        tail = (r['cropped_lognrmse'] if r['status'] == 'ok' else 'ERR')
                        print(f'[{case} {name} s{s}] {tail}', flush=True)
                if not got:
                    print(f'[no RESULT rc={proc.returncode}] {cfg}\n'
                          f'{proc.stderr[-1500:]}', flush=True)
    with open(os.path.join(STAGE, 'capture_summary.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print('=== capture done ===')


if __name__ == '__main__':
    if len(sys.argv) >= 3 and sys.argv[1] == '--worker':
        worker(json.loads(sys.argv[2]))
    else:
        main()
