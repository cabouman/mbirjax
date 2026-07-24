"""inc5 convergence-equivalence gate (Greg-approved design, 2026-07-13).

The trajectory max-norm gate is intrinsically unpassable for reordering-different
implementations (findings: "inc5 VCD divergence").  The replacement, run
ONCE-IN-A-WHILE (never nightly): real data + the parity study's 150-iteration depot
references; the question is "do both paths converge equally well?", measured as
cropped log10 NRMSE vs the reference (the parity conventions, imported from
plans/experiments/slice_parity/parity_realdata.py -- read-only reuse).

Configs per case: off (XLA), on (full pallas incl. the reverted cp=2), ctrl (XLA
with back_view_batch=96 -- the same-class reordering perturbation that CALIBRATES
the noise band).  PASS criterion: |NRMSE_on - NRMSE_off| <= 3 x max(|NRMSE_ctrl -
NRMSE_off|, 1e-4 * NRMSE_off) at each iteration mark; delta-log10 also reported in
the parity study's units for eyeballing.

Cases: lilly_ds8 + z62 at sharpness 2.0 (production-realistic; refs exist on depot).
All cells n=2 (the cone band path), seeded partitions, continuous runs to each mark.
"""
import json
import os
import subprocess
import sys
import time

PARITY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '../..', 'slice_parity')
REFS_DIR = '/depot/bouman/data/mbirjax_metrics/slice_parity/refs'
OUT_DIR = '/scratch/gautschi/buzzard/w2_inc5_convergence'
SHARPNESS = 2.0
ITER_MARKS = (8, 15)
CASES = ('lilly_ds8', 'z62')
CONFIGS = ('off', 'on', 'ctrl')
CELLS = [dict(name=f'{c}_{cfg}_i{k}', case=c, config=cfg, iters=k)
         for c in CASES for cfg in CONFIGS for k in ITER_MARKS]


def worker(cfg):
    import numpy as np
    import jax
    sys.path.insert(0, os.path.abspath(PARITY_DIR))
    from parity_realdata import load_case, build_model, crop_masks

    t0 = time.perf_counter()

    def note(msg):
        print(f'[{time.perf_counter() - t0:8.2f}s] {msg}', flush=True)

    sino, weights, geom_params, optional_params = load_case(cfg['case'])
    model = build_model(cfg['case'], geom_params, optional_params, SHARPNESS,
                        sino, parity_cls=False)
    note(f'model summary={model.device_summary}')
    if cfg['config'] == 'ctrl':
        model.tiles = model.tiles._replace(back_view_batch=96)
    import jax.numpy as jnp
    sino_j = jnp.asarray(sino)
    w_j = jnp.asarray(weights)
    np.random.seed(0)
    out, _ = model.recon(sino_j, weights=w_j, max_iterations=cfg['iters'])
    recon = np.asarray(out)
    note(f'RESULT recon done {cfg["name"]}')

    ref = np.load(os.path.join(REFS_DIR, f"{cfg['case']}_ref_sharp{SHARPNESS}.npy"))
    recon = recon[..., :ref.shape[-1]]                 # device-form crop, padding inert
    disk, z0, z1 = crop_masks(ref.shape)
    d = (recon - ref)[disk, :][:, z0:z1]
    r = ref[disk, :][:, z0:z1]
    nrmse = float(np.linalg.norm(d) / np.linalg.norm(r))
    note(f'RESULT nrmse={nrmse:.6e} log10={np.log10(nrmse):.4f}')
    np.save(os.path.join(OUT_DIR, f"{cfg['name']}_nrmse.npy"), np.asarray([nrmse]))


def orchestrator():
    import numpy as np
    os.makedirs(OUT_DIR, exist_ok=True)
    for cfg in CELLS:
        env = dict(os.environ, W2C_CELL=json.dumps(cfg), CUDA_VISIBLE_DEVICES='0,1')
        if cfg['config'] != 'on':
            env['MBIRJAX_DISABLE_PALLAS'] = '1'
        log_path = os.path.join(OUT_DIR, f"{cfg['name']}.log")
        with open(log_path, 'w') as log:
            rc = subprocess.run([sys.executable, os.path.abspath(__file__)],
                                env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        line = f"[{cfg['name']}] rc={rc}"
        with open(log_path) as log:
            for row in log:
                if 'RESULT ' in row:
                    line += '\n    ' + row.strip()
        print(line, flush=True)
    print('===== convergence-gate verdicts =====', flush=True)
    for case in CASES:
        for k in ITER_MARKS:
            try:
                n = {c: float(np.load(os.path.join(
                    OUT_DIR, f'{case}_{c}_i{k}_nrmse.npy'))[0]) for c in CONFIGS}
            except FileNotFoundError:
                print(f'{case} i{k}: MISSING CELLS', flush=True)
                continue
            band = 3 * max(abs(n['ctrl'] - n['off']), 1e-4 * n['off'])
            delta = abs(n['on'] - n['off'])
            print(f"{case} i{k}: off={n['off']:.6f} on={n['on']:.6f} "
                  f"ctrl={n['ctrl']:.6f} |on-off|={delta:.2e} band={band:.2e} "
                  f"dlog10={np.log10(n['on']) - np.log10(n['off']):+.4f} "
                  f"{'PASS' if delta <= band else 'FAIL'}", flush=True)
    print('=== w2_inc5_convergence done ===', flush=True)


if __name__ == '__main__':
    worker(json.loads(os.environ['W2C_CELL'])) if os.environ.get('W2C_CELL') \
        else orchestrator()
