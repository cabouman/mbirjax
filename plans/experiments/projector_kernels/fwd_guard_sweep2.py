"""Forward-guard sweep, part 2: (a) extend the P axis to FULL-GRID (sweep 1 found no
knee up to P=49152 -- pallas 3-6x at bands 512/1024, monotonically rising, so the
guard question becomes "does the win persist to full-grid calls?"), and (b) a views
ablation to attribute sweep 1's anomalous band=2048 XLA column (per-pixel slope 58x
band=1024's): at views=1024, band=2048, channels=1024 the sinogram element count is
EXACTLY 2^31, so the hypothesis is XLA falling onto 64-bit-index slow paths (the
lessons.md section 4 boundary), not a band effect.  Halving views drops the element
count to 2^30 with the band unchanged -- if the per-work cost snaps back in line, the
cliff is the index space, not the band.

Method identical to fwd_guard_sweep.py (isolated subprocesses, JAX-free orchestrator,
warmup + median of timed trials, rel-max <= 1e-5 pallas-vs-XLA gate per pair, npys to
scratch and deleted per pair).  P='full' uses gen_full_indices raster order directly
(the production full-grid call shape).

Run (gautschi 1x H100):  sbatch plans/experiments/projector_kernels/fwd_guard_sweep2.slurm
"""
import json
import os
import subprocess
import sys

import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
CHANNELS = 1024
# (band, num_pixels or 'full', views)
CELLS = [
    # P extension to full grid (bands 512/1024; sweep 1 covered <= 49152)
    (512, 98304, 1024), (512, 196608, 1024), (512, 393216, 1024),
    (512, 786432, 1024), (512, 'full', 1024),
    (1024, 98304, 1024), (1024, 196608, 1024), (1024, 393216, 1024),
    (1024, 786432, 1024), (1024, 'full', 1024),
    # band=2048 single extension point (the column already wins by >100x; this only
    # confirms the trend holds -- its XLA cell costs ~1 min/call, so just one point)
    (2048, 98304, 1024),
    # views ablation: control (normal regime) + discriminator (2^31 hypothesis)
    (1024, 8192, 512),
    (2048, 8192, 512),
]
OUT_DIR = '/scratch/gautschi/buzzard/headroom_fwd_guard'
WARMUP, TRIALS = 2, 3
REL_TOL = 1e-5


def ptag(num_pixels):
    return num_pixels if isinstance(num_pixels, str) else str(num_pixels)


def cell_paths(band, num_pixels, views):
    return {impl: os.path.join(
        OUT_DIR, f'b{band}_p{ptag(num_pixels)}_v{views}_{impl}.npy')
        for impl in ('xla', 'pallas')}


def worker(cfg):
    os.environ['MBIRJAX_DISABLE_PALLAS'] = '1' if cfg['impl'] == 'xla' else '0'
    os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.9')
    import time
    import mbirjax
    import jax
    import jax.numpy as jnp

    out = dict(cfg)
    try:
        band, num_pixels, views = cfg['band'], cfg['num_pixels'], cfg['views']
        sino_shape = (views, band, CHANNELS)
        angles = np.linspace(0, np.pi, views, endpoint=False)
        model = mbirjax.ParallelBeamModel(sino_shape, angles)
        model.configure_devices(1)
        out['device_kind'] = jax.devices()[0].device_kind

        recon_shape = model.get_params('recon_shape')
        rng = np.random.default_rng(0)
        full = np.asarray(mbirjax.gen_full_indices(
            recon_shape, use_ror_mask=model.get_params('use_ror_mask')))
        if num_pixels == 'full':
            idx = jnp.asarray(full)          # raster order: the production full-grid shape
        else:
            idx = jnp.asarray(np.sort(rng.choice(full, size=num_pixels, replace=False)))
        out['resolved_pixels'] = int(idx.shape[0])
        values = jax.device_put(
            jnp.asarray(rng.random((int(idx.shape[0]), recon_shape[2]),
                                   dtype=np.float32)),
            jax.devices()[0])
        jax.block_until_ready(values)

        if cfg['impl'] == 'pallas':
            from mbirjax import _pallas_kernels
            ok, why = _pallas_kernels.availability()
            out['pallas_availability'] = why
            if not ok:
                raise RuntimeError('pallas unavailable on this node: ' + why)
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
        np.save(cell_paths(band, num_pixels, views)[cfg['impl']], np.asarray(r))
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
    for band, num_pixels, views in CELLS:
        pair = {}
        for impl in ('xla', 'pallas'):
            cfg = dict(band=band, num_pixels=num_pixels, views=views, impl=impl)
            r = run_cell(cfg)
            pair[impl] = r
            print(f'[b={band} P={num_pixels} v={views} {impl}] {r.get("status")} '
                  f'wall={r.get("wall_ms")}ms peak={r.get("peak_gb")}GB', flush=True)
            if r.get('status') == 'error':
                print(r.get('traceback', '')[-1500:], flush=True)

        paths = cell_paths(band, num_pixels, views)
        row = dict(band=band, num_pixels=num_pixels, views=views,
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
    with open(os.path.join(OUT_DIR, 'fwd_guard_sweep2_results.json'), 'w') as f:
        json.dump(rows, f, indent=1)

    print('\n===== fwd guard sweep 2 summary (speedup = XLA wall / pallas wall) =====',
          flush=True)
    print(f'{"band":>5} {"P":>7} {"views":>6} {"xla_ms":>9} {"pallas_ms":>10} '
          f'{"speedup":>8} {"rel_max":>10} {"gate":>5}', flush=True)
    for row in rows:
        p_shown = row['xla'].get('resolved_pixels', row['num_pixels'])
        if 'speedup' not in row:
            print(f'{row["band"]:>5} {p_shown:>7} {row["views"]:>6}  INCOMPLETE',
                  flush=True)
            continue
        print(f'{row["band"]:>5} {p_shown:>7} {row["views"]:>6} '
              f'{row["xla"]["wall_ms"]:>9.1f} {row["pallas"]["wall_ms"]:>10.1f} '
              f'{row["speedup"]:>8.2f} {row["rel_max"]:>10.3g} '
              f'{"PASS" if row["values_pass"] else "FAIL"}', flush=True)


if __name__ == '__main__':
    if len(sys.argv) >= 3 and sys.argv[1] == '--worker':
        worker(json.loads(sys.argv[2]))
    else:
        main()
