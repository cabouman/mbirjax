"""inc5 VCD-divergence forensics (2026-07-13).

The cone VCD n=2 trajectory gate fails at rel 8.5e-3 (deterministic; Hessian-to-XLA
ablation left it UNCHANGED at 8.52e-3, refuting the low-Hessian-voxel hypothesis),
while every per-call gate passes at <= 6.5e-6: full-grid and subset back, sorted and
unsorted pixels, both powers, adjoint.  This job localizes the divergence:

  * WHEN: seeded VCD off/on at max_iterations 1, 2, 4 -- rel after each.
  * WHERE: per-slice max|diff| profile (band seams? axial ends?) and radial bins
    (FoV edge?) at each iteration count, printed as compact tables.

Cells run n=2, both configs per iteration count (off saves ref to scratch).
"""
import json
import os
import subprocess
import sys
import time

SINO_SHAPE = (1024, 1024, 1024)
ITER_COUNTS = (1, 2, 4)
OUT_DIR = '/scratch/gautschi/buzzard/w2_inc5_forensics'
CELLS = [dict(name=f'vcd_i{k}_{cfg}', iters=k, on=cfg == 'on')
         for k in ITER_COUNTS for cfg in ('off', 'on')]
# THE CONTROL: two pure-XLA runs differing ONLY in back_view_batch (a summation-
# order perturbation of the same float class as pallas-vs-XLA).  If this pair also
# diverges to ~1e-2 at the edge after 4 iterations, the divergence is INTRINSIC
# ill-conditioning of low-coverage voxels, and the trajectory gate needs an
# interior mask or a convergence metric -- no reordering-different implementation
# could pass a max-norm gate over the extension zone.
CELLS += [dict(name='vcd_ctrl_i4_off', iters=4, on=False, ctrl_ref='ctrl'),
          dict(name='vcd_ctrl_i4_vb96', iters=4, on=False, vb=96, ctrl_ref='ctrl')]
if os.environ.get('W2F_CTRL_ONLY') == '1':
    CELLS = [c for c in CELLS if 'ctrl' in c['name']]


def worker(cfg):
    import numpy as np
    import jax
    import jax.numpy as jnp
    import mbirjax

    t0 = time.perf_counter()

    def note(msg):
        print(f'[{time.perf_counter() - t0:8.2f}s] {msg}', flush=True)

    views, rows, channels = SINO_SHAPE
    angles = np.linspace(-np.pi / 2, np.pi / 2, views, endpoint=False)
    model = mbirjax.ConeBeamModel(SINO_SHAPE, angles,
                                  source_detector_dist=4 * channels,
                                  source_iso_dist=4 * channels)
    note(f'summary={model.device_summary}')
    recon_shape = model.get_params('recon_shape')
    rng = np.random.default_rng(0)
    sino = jnp.asarray(rng.random(SINO_SHAPE, dtype=np.float32))
    jax.block_until_ready(sino)
    model.set_params(verbose=0)
    if cfg.get('vb'):
        model.tiles = model.tiles._replace(back_view_batch=cfg['vb'])
        note(f'control: back_view_batch={cfg["vb"]}')
    np.random.seed(0)                        # identical partitions off/on
    out, _ = model.recon(sino, max_iterations=cfg['iters'])
    result = np.asarray(out)
    note(f'RESULT recon done iters={cfg["iters"]}')

    tag = cfg.get('ctrl_ref', f"vcd_i{cfg['iters']}")
    ref_path = os.path.join(OUT_DIR, f'{tag}_ref.npy')
    if not cfg['on'] and not cfg.get('vb'):
        np.save(ref_path, result)
        note('RESULT ref=saved')
        return
    ref = np.load(ref_path)
    diff = np.abs(result - ref)
    scale = np.max(np.abs(ref))
    note(f"RESULT rel={float(diff.max() / scale):.2e} iters={cfg['iters']}")
    # WHERE, axially: per-slice max|diff| (relative), compact profile.
    per_slice = diff.reshape(-1, diff.shape[-1]).max(axis=0) / scale
    worst = np.argsort(per_slice)[-8:][::-1]
    note('RESULT worst slices (idx: rel): '
         + ', '.join(f'{int(i)}: {per_slice[i]:.1e}' for i in worst))
    n_sl = len(per_slice)
    bins = [per_slice[i * n_sl // 12:(i + 1) * n_sl // 12].max() for i in range(12)]
    note('RESULT slice-profile (12 bins): '
         + ' '.join(f'{b:.0e}' for b in bins))
    # WHERE, radially: max|diff| in 8 radius bins over the pixel grid.
    nr, nc = recon_shape[0], recon_shape[1]
    yy, xx = np.mgrid[0:nr, 0:nc]
    r = np.sqrt((yy - (nr - 1) / 2) ** 2 + (xx - (nc - 1) / 2) ** 2) / (nr / 2)
    dmax_xy = diff.max(axis=-1)
    rbins = [dmax_xy[(r >= lo / 8) & (r < (lo + 1) / 8)].max() / scale
             for lo in range(8)]
    note('RESULT radial-profile (8 bins, center->edge): '
         + ' '.join(f'{b:.0e}' for b in rbins))


def orchestrator():
    os.makedirs(OUT_DIR, exist_ok=True)
    for cfg in CELLS:
        env = dict(os.environ, W2F_CELL=json.dumps(cfg), CUDA_VISIBLE_DEVICES='0,1')
        if not cfg['on']:
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
    print('=== w2_inc5_forensics done ===', flush=True)


if __name__ == '__main__':
    worker(json.loads(os.environ['W2F_CELL'])) if os.environ.get('W2F_CELL') \
        else orchestrator()
