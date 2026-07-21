"""E2a: back_view_batch L2-residency sweep at the 1024^3 n=1 cell.

E0 established that back projection's cost is ONE fused kernel randomly gathering 4 KB
rows from its per-step view working set: view_batch (128) x channel-major views (4 MB
each) = 512 MB >> the H100's 50 MB L2, so gathers miss to HBM with poor efficiency.
Shrinking back_view_batch shrinks that working set toward L2 residency (16 views = 64 MB,
8 = 32 MB) at the cost of more view-batch steps (more per-step fixed cost + more output
RMW passes).  Same arithmetic, same values; a pure TilePolicy knob.

This sweeps back_view_batch and reports model-level sparse_back_project wall + device
peak memory.  Full ROR grid (the (b)-workload shape) AND a 6,026-pixel subset (the
production VCD granularity-128 shape) per config.  Subprocess per config (honest memory).

Run:  python plans/experiments/projector_kernels/e2a_back_view_batch.py   (constants below)
"""
import json
import os
import subprocess
import sys

# ── Config ────────────────────────────────────────────────────────────────────
SINO_SHAPE = (1024, 1008, 992)
VIEW_BATCHES = ['default', 64, 32, 16, 8]     # 'default' = the library policy (128)
SUBSET_PIXELS = 6026                          # granularity-128 subset size at this cell
WARMUP = 1
TRIALS_FULL, TRIALS_SUBSET = 2, 5
RESULTS_PATH = os.path.expanduser('~/headroom/results/e2a_back_view_batch.jsonl')


def worker(cfg):
    os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.94')
    import time
    import numpy as np
    import mbirjax
    import jax

    out = dict(cfg)
    try:
        angles = np.linspace(0, np.pi, SINO_SHAPE[0], endpoint=False)
        model = mbirjax.ParallelBeamModel(SINO_SHAPE, angles)
        model.configure_devices(1)
        if cfg['view_batch'] != 'default':
            model.tiles = model.tiles._replace(back_view_batch=int(cfg['view_batch']))
        out['effective_view_batch'] = int(model.tiles.back_view_batch)
        recon_shape = model.get_params('recon_shape')
        full_idx = mbirjax.gen_full_indices(recon_shape,
                                            use_ror_mask=model.get_params('use_ror_mask'))
        rng = np.random.default_rng(0)
        sino = model._shard_sinogram(rng.random(SINO_SHAPE, dtype=np.float32))
        jax.block_until_ready(sino)

        for tag, idx, trials in (('full', full_idx, TRIALS_FULL),
                                 ('subset', np.sort(rng.choice(np.asarray(full_idx),
                                                               size=SUBSET_PIXELS,
                                                               replace=False)),
                                  TRIALS_SUBSET)):
            for _ in range(WARMUP):
                jax.block_until_ready(model.sparse_back_project(sino, idx))
            times = []
            for _ in range(trials):
                t0 = time.perf_counter()
                r = jax.block_until_ready(model.sparse_back_project(sino, idx))
                times.append(time.perf_counter() - t0)
                del r
            out[f'{tag}_back_s'] = float(np.median(times))
            out[f'{tag}_back_s_all'] = [round(t, 5) for t in times]
        stats = jax.local_devices()[0].memory_stats() or {}   # None on CPU
        out['peak_bytes_in_use'] = int(stats.get('peak_bytes_in_use', -1))
        out['status'] = 'ok'
    except Exception:
        import traceback
        out['status'] = 'error'
        out['traceback'] = traceback.format_exc()
    print('RESULT ' + json.dumps(out), flush=True)


def main():
    results = []
    for vb in VIEW_BATCHES:
        cfg = dict(view_batch=vb)
        proc = subprocess.run([sys.executable, os.path.abspath(__file__), '--worker',
                               json.dumps(cfg)], capture_output=True, text=True)
        for line in proc.stdout.splitlines():
            if line.startswith('RESULT '):
                results.append(json.loads(line[len('RESULT '):]))
                print(line, flush=True)
        if proc.returncode != 0:
            print(f'[worker rc={proc.returncode}] {cfg}\n{proc.stderr[-2000:]}', flush=True)

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

    print(f'\n{"=" * 70}\nback_view_batch sweep (1024^3 n=1; working set = vb x 4 MB):')
    print(f'{"view_batch":>10s} {"ws_MB":>6s} {"full_back_s":>12s} {"subset_back_s":>14s} '
          f'{"peak_GB":>8s}')
    for r in results:
        if r.get('status') != 'ok':
            print(f'{str(r["view_batch"]):>10s}  ERROR')
            continue
        vb = r['effective_view_batch']
        print(f'{vb:10d} {vb * 4:6d} {r["full_back_s"]:12.3f} {r["subset_back_s"]:14.4f} '
              f'{r["peak_bytes_in_use"] / 2**30:8.2f}')
    print(f'\n[raw results in {RESULTS_PATH}]')


if __name__ == '__main__':
    if len(sys.argv) >= 3 and sys.argv[1] == '--worker':
        worker(json.loads(sys.argv[2]))
    else:
        main()
