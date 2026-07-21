"""E1 Amdahl gate (part 1): projector wall time vs PIXEL COUNT at the 1024^3 cell.

VCD at production granularity 128 calls the projectors on ~6.3k-pixel subsets -- a single
pixel-batch call, where per-call fixed costs (dispatch, the sorted reduce's sort, ragged
tiles) have nothing to amortize over.  This sweep times sparse_forward_project and
sparse_back_project at subset sizes corresponding to granularities {512, 128, 64, 16, 4, 1}
and prints, per granularity, the PREDICTED per-iteration projector seconds
(= num_subsets x per-call time at that subset size, since one partition pass covers every
pixel).  Together with e1_vcd_trace.py's measured iteration walls, this bounds the
fine-granularity VCD kernel share without running dozens of full recons.

Subsets are seeded random draws from the ROR-masked full grid (mimicking VCD partition
subsets -- random spatial samples, NOT raster order; locality is workload-representative).

Each (geometry, count) runs in its OWN SUBPROCESS (fresh jax, honest memory, an OOM cannot
contaminate later configs) -- the fwd_band_pixel_sweep.py pattern.

Run:  python plans/experiments/projector_kernels/e1_pixel_sweep.py    (edit constants below)
"""
import json
import os
import subprocess
import sys

# ── Config ────────────────────────────────────────────────────────────────────
SINO_SHAPE = (1024, 1008, 992)
GEOMETRIES = ['parallel', 'cone']
GRANULARITIES = [512, 128, 64, 16, 4, 1]     # 1 = the full ROR grid in one call
WARMUP = 1
TRIALS_SMALL, TRIALS_FULL = 5, 2             # more trials where calls are tiny/noisy
RESULTS_PATH = os.path.expanduser('~/headroom/results/e1_pixel_sweep.jsonl')


def worker(cfg):
    os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.94')
    import time
    import numpy as np
    import mbirjax
    import jax

    out = dict(cfg)
    try:
        angles = np.linspace(0, np.pi, SINO_SHAPE[0], endpoint=False)
        if cfg['geometry'] == 'parallel':
            model = mbirjax.ParallelBeamModel(SINO_SHAPE, angles)
        else:
            model = mbirjax.ConeBeamModel(SINO_SHAPE, angles,
                                          source_detector_dist=4 * SINO_SHAPE[2],
                                          source_iso_dist=4 * SINO_SHAPE[2])
        model.configure_devices(1)
        recon_shape = model.get_params('recon_shape')
        full_idx = mbirjax.gen_full_indices(recon_shape,
                                            use_ror_mask=model.get_params('use_ror_mask'))
        g = cfg['granularity']
        count = len(full_idx) if g == 1 else max(1, int(np.ceil(len(full_idx) / g)))
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(np.asarray(full_idx), size=count, replace=False)) \
            if g > 1 else full_idx
        out['num_pixels'] = int(count)

        cylinders = model._shard_recon(rng.random((count, recon_shape[2]), dtype=np.float32))
        sino = model._shard_sinogram(rng.random(SINO_SHAPE, dtype=np.float32))
        jax.block_until_ready((cylinders, sino))

        trials = TRIALS_FULL if g == 1 else TRIALS_SMALL
        for op in ('fwd', 'back'):
            call = ((lambda: model.sparse_forward_project(cylinders, idx)) if op == 'fwd'
                    else (lambda: model.sparse_back_project(sino, idx)))
            for _ in range(WARMUP):
                jax.block_until_ready(call())
            times = []
            for _ in range(trials):
                t0 = time.perf_counter()
                r = jax.block_until_ready(call())
                times.append(time.perf_counter() - t0)
                del r
            out[f'{op}_s'] = float(np.median(times))
            out[f'{op}_s_all'] = [round(t, 5) for t in times]
        out['status'] = 'ok'
    except Exception:
        import traceback
        out['status'] = 'error'
        out['traceback'] = traceback.format_exc()
    print('RESULT ' + json.dumps(out), flush=True)


def main():
    results = []
    for geometry in GEOMETRIES:
        for g in GRANULARITIES:
            cfg = dict(geometry=geometry, granularity=g)
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

    print(f'\n{"=" * 74}\nPer-iteration PROJECTOR-second prediction '
          f'(num_subsets x per-call median, fwd+back):\n{"=" * 74}')
    print(f'{"geometry":10s} {"granularity":>11s} {"pixels/call":>11s} '
          f'{"fwd_s":>9s} {"back_s":>9s} {"iter_proj_s":>12s}')
    for r in results:
        if r.get('status') != 'ok':
            continue
        g = r['granularity']
        iter_s = g * (r['fwd_s'] + r['back_s'])
        print(f'{r["geometry"]:10s} {g:11d} {r["num_pixels"]:11d} '
              f'{r["fwd_s"]:9.4f} {r["back_s"]:9.4f} {iter_s:12.2f}')
    print(f'\n[raw results in {RESULTS_PATH}]')


if __name__ == '__main__':
    if len(sys.argv) >= 3 and sys.argv[1] == '--worker':
        worker(json.loads(sys.argv[2]))
    else:
        main()
