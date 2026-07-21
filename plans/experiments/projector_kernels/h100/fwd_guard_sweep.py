"""Forward-guard sweep: pallas forward_project_subset vs the XLA sorted-reduce path
across pixel count P and band width, to replace the inherited dispatch guard
(pixel_count <= tiles.fwd_pixel_batch = 8192) with a MEASURED, band-aware policy.

Physics of the question: the pallas kernel's win rests on the shared values tile
(P x band x 4 B) staying L2-hot across views (H100 L2 = 50 MB), so the predicted
knees are P ~ 50 MB / (4 * band): ~24.4k / ~12.2k / ~6.1k pixels for bands
512 / 1024 / 2048.  The sweep brackets all three and includes the current guard
value (8192) as an exact point.

Method (modeled on e4_ab_back.py): each (band, P, impl) cell runs in an ISOLATED
subprocess (fresh jax => honest peak memory; the orchestrator stays JAX-free).
band = num_det_rows = num recon slices = the values second dim, via
sino_shape = (VIEWS, band, CHANNELS).  The pallas cell calls the driver
_pallas_kernels.forward_project_subset(model, values, idx) DIRECTLY (bypasses the
wrapper's pixel-count guard -- no library edit); the XLA cell calls the public
model.projector_functions.sparse_forward_project(values, idx) under
MBIRJAX_DISABLE_PALLAS=1 (the production fallback path, including its own pixel
batching at 8192).  Values gate: rel-max <= 1e-5 (single-shot projector gate,
lessons.md section 2), compared by the orchestrator from npys saved to scratch,
deleted after each pair to bound scratch usage.

Run (gautschi 1x H100):  sbatch plans/experiments/projector_kernels/fwd_guard_sweep.slurm
"""
import json
import os
import subprocess
import sys

import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
VIEWS = 1024
CHANNELS = 1024
BANDS = (512, 1024, 2048)            # values second dim (= det rows = recon slices)
PIXELS = (2048, 4096, 6144, 8192,    # 8192 = the current guard, an exact point
          12288, 16384, 24576, 49152)
OUT_DIR = '/scratch/gautschi/buzzard/headroom_fwd_guard'  # big regenerable npys: SCRATCH (home quota 25 GB)
WARMUP, TRIALS = 2, 5                # per-cell warm calls, then median of timed calls
REL_TOL = 1e-5                       # single-shot float gate vs the XLA reference


def cell_paths(band, num_pixels):
    return {impl: os.path.join(OUT_DIR, f'b{band}_p{num_pixels}_{impl}.npy')
            for impl in ('xla', 'pallas')}


def worker(cfg):
    # The env flag must be set BEFORE importing mbirjax/jax: it feeds the cached
    # availability() probe and the tile policy.
    os.environ['MBIRJAX_DISABLE_PALLAS'] = '1' if cfg['impl'] == 'xla' else '0'
    os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.9')
    import time
    import mbirjax
    import jax
    import jax.numpy as jnp

    out = dict(cfg)
    try:
        band, num_pixels = cfg['band'], cfg['num_pixels']
        sino_shape = (VIEWS, band, CHANNELS)
        angles = np.linspace(0, np.pi, VIEWS, endpoint=False)
        model = mbirjax.ParallelBeamModel(sino_shape, angles)
        model.configure_devices(1)
        out['device_kind'] = jax.devices()[0].device_kind
        out['fwd_pallas_policy'] = bool(model.tiles.fwd_pallas)

        # Same rng seed in both impl subprocesses => identical idx and values per
        # (band, P) pair; sorted subset matches the E4 gate cells.
        recon_shape = model.get_params('recon_shape')
        rng = np.random.default_rng(0)
        full = np.asarray(mbirjax.gen_full_indices(
            recon_shape, use_ror_mask=model.get_params('use_ror_mask')))
        idx = jnp.asarray(np.sort(rng.choice(full, size=num_pixels, replace=False)))
        values = jax.device_put(
            jnp.asarray(rng.random((num_pixels, recon_shape[2]), dtype=np.float32)),
            jax.devices()[0])
        jax.block_until_ready(values)

        if cfg['impl'] == 'pallas':
            from mbirjax import _pallas_kernels
            ok, why = _pallas_kernels.availability()
            out['pallas_availability'] = why
            if not ok:
                raise RuntimeError('pallas unavailable on this node: ' + why)
            # Direct driver call: deliberately bypasses the wrapper's
            # pixel_count <= fwd_pixel_batch guard (the guard is what we are measuring).
            call = lambda: _pallas_kernels.forward_project_subset(model, values, idx)
        else:
            assert not model.tiles.fwd_pallas, 'kill switch failed to disable pallas'
            call = lambda: model.projector_functions.sparse_forward_project(values, idx)

        for _ in range(WARMUP):
            jax.block_until_ready(call())
        ts, r = [], None
        for _ in range(TRIALS):
            r = None                       # release the previous result BEFORE the
            t0 = time.perf_counter()       # next allocation (lessons.md section 5)
            r = jax.block_until_ready(call())
            ts.append(time.perf_counter() - t0)
        out['wall_ms'] = round(1000 * float(np.median(ts)), 2)
        out['trials_ms'] = [round(1000 * t, 2) for t in ts]

        stats = jax.local_devices()[0].memory_stats() or {}
        out['peak_gb'] = round(stats.get('peak_bytes_in_use', 0) / 2 ** 30, 2)
        os.makedirs(OUT_DIR, exist_ok=True)
        np.save(cell_paths(band, num_pixels)[cfg['impl']], np.asarray(r))
        out['status'] = 'ok'
    except Exception:
        import traceback
        out['status'] = 'error'
        out['traceback'] = traceback.format_exc()
    print('RESULT ' + json.dumps(out), flush=True)


def run_cell(cfg):
    proc = subprocess.run([sys.executable, os.path.abspath(__file__),
                           '--worker', json.dumps(cfg)],
                          capture_output=True, text=True)
    result = None
    for line in proc.stdout.splitlines():
        if line.startswith('RESULT '):
            result = json.loads(line[len('RESULT '):])
    if proc.returncode != 0 or result is None:
        print(f'[rc={proc.returncode}] {cfg}\n{proc.stderr[-1500:]}', flush=True)
    return result or dict(cfg, status='no-result')


def main():
    rows = []
    for band in BANDS:
        for num_pixels in PIXELS:
            pair = {}
            for impl in ('xla', 'pallas'):
                cfg = dict(band=band, num_pixels=num_pixels, impl=impl)
                r = run_cell(cfg)
                pair[impl] = r
                print(f'[b={band} P={num_pixels} {impl}] {r.get("status")} '
                      f'wall={r.get("wall_ms")}ms peak={r.get("peak_gb")}GB',
                      flush=True)
                if r.get('status') == 'error':
                    print(r.get('traceback', '')[-1500:], flush=True)

            # Compare + delete the pair's npys immediately (bounds scratch to one pair).
            paths = cell_paths(band, num_pixels)
            row = dict(band=band, num_pixels=num_pixels,
                       xla=pair['xla'], pallas=pair['pallas'])
            if all(pair[i].get('status') == 'ok' for i in ('xla', 'pallas')):
                a = np.load(paths['xla'])
                b = np.load(paths['pallas'])
                row['rel_max'] = float(np.max(np.abs(a - b)) /
                                       max(float(np.max(np.abs(a))), 1e-30))
                row['values_pass'] = row['rel_max'] <= REL_TOL
                row['speedup'] = round(pair['xla']['wall_ms'] / pair['pallas']['wall_ms'], 3)
                del a, b
            for p in paths.values():
                if os.path.exists(p):
                    os.remove(p)
            rows.append(row)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, 'fwd_guard_sweep_results.json'), 'w') as f:
        json.dump(rows, f, indent=1)

    print('\n===== fwd guard sweep summary (speedup = XLA wall / pallas wall) =====',
          flush=True)
    print(f'{"band":>5} {"P":>6} {"tile_MB":>8} {"xla_ms":>8} {"pallas_ms":>10} '
          f'{"speedup":>8} {"rel_max":>10} {"gate":>5}', flush=True)
    for row in rows:
        if 'speedup' not in row:
            print(f'{row["band"]:>5} {row["num_pixels"]:>6}  INCOMPLETE', flush=True)
            continue
        tile_mb = row['num_pixels'] * row['band'] * 4 / 2 ** 20
        print(f'{row["band"]:>5} {row["num_pixels"]:>6} {tile_mb:>8.1f} '
              f'{row["xla"]["wall_ms"]:>8.1f} {row["pallas"]["wall_ms"]:>10.1f} '
              f'{row["speedup"]:>8.2f} {row["rel_max"]:>10.3g} '
              f'{"PASS" if row["values_pass"] else "FAIL"}', flush=True)


if __name__ == '__main__':
    if len(sys.argv) >= 3 and sys.argv[1] == '--worker':
        worker(json.loads(sys.argv[2]))
    else:
        main()
