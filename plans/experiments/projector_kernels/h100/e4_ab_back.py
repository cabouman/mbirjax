"""E4 model-level A/B: the pallas back path (increment 1) vs the XLA path, flag-toggled
via MBIRJAX_DISABLE_PALLAS in per-config SUBPROCESSES (fresh jax, honest memory).

Cells (parallel beam, H100 n=1):
  back_full / back_subset / hessian_full  at the 1024^3 cell — wall + peak memory +
      value agreement (rel <= 1e-5 vs the XLA result computed in the same subprocess
      via the env override... the XLA reference comes from the OFF config's saved
      output, compared by the orchestrator);
  vcd_guard  at a 256^3-class cell — full recon wall + final-recon agreement at the
      ITERATED float gate (1e-4), watching the pallas-not-in-CUDA-graphs caveat.

Run (gautschi 1 GPU):  sbatch plans/experiments/projector_kernels/e4_ab_back.slurm
"""
import json
import os
import subprocess
import sys

import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
BIG_SINO = (1024, 1008, 992)
VCD_SINO = (256, 252, 248)
VCD_ITERATIONS = 5
OUT_DIR = '/scratch/gautschi/buzzard/headroom_e4_ab'   # big regenerable npys: SCRATCH, not home (home quota is 25 GB)
WARMUP, TRIALS = 1, 3


def worker(cfg):
    os.environ['MBIRJAX_DISABLE_PALLAS'] = '0' if cfg['pallas'] else '1'
    os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.9')
    import time
    import mbirjax
    import jax
    import jax.numpy as jnp

    out = dict(cfg)
    try:
        cell = cfg['cell']
        sino_shape = BIG_SINO if cell != 'vcd_guard' else VCD_SINO
        angles = np.linspace(0, np.pi, sino_shape[0], endpoint=False)
        model = mbirjax.ParallelBeamModel(sino_shape, angles)
        model.configure_devices(1)
        out['back_pallas_active'] = bool(model.tiles.back_pallas)
        recon_shape = model.get_params('recon_shape')
        rng = np.random.default_rng(0)
        idx = mbirjax.gen_full_indices(recon_shape,
                                       use_ror_mask=model.get_params('use_ror_mask'))
        if cell in ('back_subset', 'fwd_subset'):
            idx = jnp.asarray(np.sort(rng.choice(np.asarray(idx), size=6026,
                                                 replace=False)))

        if cell in ('fwd_subset', 'fwd_full'):
            # fwd_full verifies the POLICY: the pixel-count guard must keep full-grid
            # forward on XLA, so on/off walls should match (and values bitwise-ish).
            values = jnp.asarray(rng.random((len(idx), recon_shape[2]),
                                            dtype=np.float32))
            values = jax.device_put(values, jax.devices()[0])
            jax.block_until_ready(values)
            for _ in range(WARMUP):
                jax.block_until_ready(model.sparse_forward_project(values, idx))
            ts = []
            for _ in range(TRIALS):
                t0 = time.perf_counter()
                r = jax.block_until_ready(model.sparse_forward_project(values, idx))
                ts.append(time.perf_counter() - t0)
            out['wall_s'] = round(float(np.median(ts)), 3)
            result = np.asarray(r)
        elif cell == 'vcd_guard':
            phantom = mbirjax.gen_cube_phantom(recon_shape)
            sino = np.asarray(model.forward_project(phantom))
            np.random.seed(7)
            t0 = time.perf_counter()
            recon, _ = model.recon(sino, max_iterations=VCD_ITERATIONS,
                                   stop_threshold_change_pct=0, print_logs=False)
            out['wall_s'] = round(time.perf_counter() - t0, 2)
            result = np.asarray(recon)
        else:
            coeff_power = 2 if cell == 'hessian_full' else 1
            sino = model._shard_sinogram(rng.random(sino_shape, dtype=np.float32))
            jax.block_until_ready(sino)
            for _ in range(WARMUP):
                jax.block_until_ready(model.sparse_back_project(
                    sino, idx, coeff_power=coeff_power))
            ts = []
            for _ in range(TRIALS):
                t0 = time.perf_counter()
                r = jax.block_until_ready(model.sparse_back_project(
                    sino, idx, coeff_power=coeff_power))
                ts.append(time.perf_counter() - t0)
            out['wall_s'] = round(float(np.median(ts)), 3)
            result = np.asarray(r)

        stats = jax.local_devices()[0].memory_stats() or {}
        out['peak_gb'] = round(stats.get('peak_bytes_in_use', 0) / 2 ** 30, 2)
        os.makedirs(OUT_DIR, exist_ok=True)
        np.save(os.path.join(OUT_DIR, f'{cell}_{"on" if cfg["pallas"] else "off"}.npy'),
                result)
        out['status'] = 'ok'
    except Exception:
        import traceback
        out['status'] = 'error'
        out['traceback'] = traceback.format_exc()
    print('RESULT ' + json.dumps(out), flush=True)


def main():
    cells = ['back_full', 'back_subset', 'hessian_full',
             'fwd_subset', 'fwd_full', 'vcd_guard']
    results = {}
    for cell in cells:
        for pallas in (False, True):
            cfg = dict(cell=cell, pallas=pallas)
            proc = subprocess.run([sys.executable, os.path.abspath(__file__),
                                   '--worker', json.dumps(cfg)],
                                  capture_output=True, text=True)
            for line in proc.stdout.splitlines():
                if line.startswith('RESULT '):
                    r = json.loads(line[len('RESULT '):])
                    results[(cell, pallas)] = r
                    print(f'[{cell} pallas={pallas}] {r.get("status")} '
                          f'wall={r.get("wall_s")}s peak={r.get("peak_gb")}GB '
                          f'active={r.get("back_pallas_active")}', flush=True)
            if proc.returncode != 0:
                print(f'[rc={proc.returncode}] {cfg}\n{proc.stderr[-1500:]}', flush=True)

    print('\n===== A/B summary =====', flush=True)
    for cell in cells:
        off, on = results.get((cell, False)), results.get((cell, True))
        if not (off and on and off['status'] == on['status'] == 'ok'):
            print(f'{cell}: INCOMPLETE', flush=True)
            continue
        a = np.load(os.path.join(OUT_DIR, f'{cell}_off.npy'))
        b = np.load(os.path.join(OUT_DIR, f'{cell}_on.npy'))
        rel = float(np.max(np.abs(a - b)) / max(np.max(np.abs(a)), 1e-30))
        tol = 1e-4 if cell == 'vcd_guard' else 1e-5      # iterated vs single-shot gate
        print(f'{cell}: {off["wall_s"]}s -> {on["wall_s"]}s '
              f'({off["wall_s"] / on["wall_s"]:.2f}x)  '
              f'peak {off["peak_gb"]} -> {on["peak_gb"]} GB  '
              f'rel {rel:.3g} {"PASS" if rel < tol else "FAIL"}', flush=True)


if __name__ == '__main__':
    if len(sys.argv) >= 3 and sys.argv[1] == '--worker':
        worker(json.loads(sys.argv[2]))
    else:
        main()
