"""Forward-guard sweep, part 3: close the SMALL-BAND corner before proposing to drop
the pixel-count guard.  Sweeps 1-2 covered bands 512-2048 (pallas >= 3.1x at every
point through full grid), but small problems (256^3-class -- the VCD interactive
regime) run band <= 256 where the pallas per-call fixed cost (~4-5 ms) could bite,
and every swept band was a power of two, so the driver's band-padding path (vals_pad
copy + wasted padded columns) went unmeasured.  Cells:

  - band in {128, 256} x P in {2048, 8192, 24576} at views=1024, channels=1024
    (single-variable: band alone, vs sweep 1's band=512 column);
  - band=768 at P=8192 (pad to 1024: 33% wasted columns, the worst mid-pad case);
  - a small-problem pair at production aspect, sino (256, 256, 256): P=2048 and
    P='full' (~51k) -- the op-level mirror of E4's vcd_guard cell.

Method identical to fwd_guard_sweep.py/2 (isolated subprocesses, warmup + median
trials, rel-max <= 1e-5 gate, scratch npys deleted per pair).

Run (gautschi 1x H100):  sbatch plans/experiments/projector_kernels/fwd_guard_sweep3.slurm
"""
import json
import os
import subprocess
import sys

import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
# (band, num_pixels or 'full', views, channels)
CELLS = [
    (128, 2048, 1024, 1024), (128, 8192, 1024, 1024), (128, 24576, 1024, 1024),
    (256, 2048, 1024, 1024), (256, 8192, 1024, 1024), (256, 24576, 1024, 1024),
    (768, 8192, 1024, 1024),
    (256, 2048, 256, 256), (256, 'full', 256, 256),
]
OUT_DIR = '/scratch/gautschi/buzzard/headroom_fwd_guard'
WARMUP, TRIALS = 2, 5
REL_TOL = 1e-5


def ptag(num_pixels):
    return num_pixels if isinstance(num_pixels, str) else str(num_pixels)


def cell_paths(band, num_pixels, views, channels):
    return {impl: os.path.join(
        OUT_DIR, f'b{band}_p{ptag(num_pixels)}_v{views}_c{channels}_{impl}.npy')
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
        band, num_pixels = cfg['band'], cfg['num_pixels']
        views, channels = cfg['views'], cfg['channels']
        sino_shape = (views, band, channels)
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
        np.save(cell_paths(band, num_pixels, views, channels)[cfg['impl']],
                np.asarray(r))
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
    for band, num_pixels, views, channels in CELLS:
        pair = {}
        for impl in ('xla', 'pallas'):
            cfg = dict(band=band, num_pixels=num_pixels, views=views,
                       channels=channels, impl=impl)
            r = run_cell(cfg)
            pair[impl] = r
            print(f'[b={band} P={num_pixels} v={views} c={channels} {impl}] '
                  f'{r.get("status")} wall={r.get("wall_ms")}ms '
                  f'peak={r.get("peak_gb")}GB', flush=True)
            if r.get('status') == 'error':
                print(r.get('traceback', '')[-1500:], flush=True)

        paths = cell_paths(band, num_pixels, views, channels)
        row = dict(band=band, num_pixels=num_pixels, views=views, channels=channels,
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
    with open(os.path.join(OUT_DIR, 'fwd_guard_sweep3_results.json'), 'w') as f:
        json.dump(rows, f, indent=1)

    print('\n===== fwd guard sweep 3 summary (speedup = XLA wall / pallas wall) =====',
          flush=True)
    print(f'{"band":>5} {"P":>7} {"views":>6} {"chans":>6} {"xla_ms":>9} '
          f'{"pallas_ms":>10} {"speedup":>8} {"rel_max":>10} {"gate":>5}', flush=True)
    for row in rows:
        p_shown = row['xla'].get('resolved_pixels', row['num_pixels'])
        if 'speedup' not in row:
            print(f'{row["band"]:>5} {p_shown:>7} {row["views"]:>6} '
                  f'{row["channels"]:>6}  INCOMPLETE', flush=True)
            continue
        print(f'{row["band"]:>5} {p_shown:>7} {row["views"]:>6} {row["channels"]:>6} '
              f'{row["xla"]["wall_ms"]:>9.1f} {row["pallas"]["wall_ms"]:>10.1f} '
              f'{row["speedup"]:>8.2f} {row["rel_max"]:>10.3g} '
              f'{"PASS" if row["values_pass"] else "FAIL"}', flush=True)


if __name__ == '__main__':
    if len(sys.argv) >= 3 and sys.argv[1] == '--worker':
        worker(json.loads(sys.argv[2]))
    else:
        main()
