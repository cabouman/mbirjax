"""XLA forward band>=2048 cliff: sort_by_channel ablation + fine band sweep.

Follow-up to fwd_guard_sweep 1 (plans/projector_kernels/fwd_guard_sweep.md, Read 2):
the XLA forward path's per-work cost is ~58x worse at band=2048 than band=1024,
categorical at every P, values still correct; the views=512 ablation already refuted
the 2^31 hypothesis, leaving a band-categorical XLA lowering artifact.  This job
localizes it.

Two questions in ONE grid (band x reduce-algorithm), all on the XLA path
(MBIRJAX_DISABLE_PALLAS=1, so sparse_forward_project never routes to pallas):

  1. WHERE does the cliff live?  Toggle the channel reduction between the sorted
     segment-sum (sort_by_channel=1, the GPU default -- projectors.
     _channel_reduce_sort_segsum) and the unrolled scatter-add
     (sort_by_channel=0 -- _channel_reduce_scatter_add).  If the cliff vanishes with
     scatter-add it is the sorted reduce (and the fix is a one-line policy threshold);
     if it persists it is upstream in the base forward kernel's per-view vmap.
     sort_by_channel is baked into the static ProjectorParams at create_projectors, so
     each cell forces model.tiles then rebuilds the projectors.

  2. WHERE is the threshold?  Fine band sweep {1024, 1152, 1280, 1536, 1792, 2048,
     3072, 4096} at fixed P: exactly at 2048 (a power-of-two tiling boundary) vs
     mid-range (a byte-size/register-pressure limit) vs recurring at 4096 (periodic)
     are different mechanisms.

Peak memory per cell is a third discriminator: a jump at the threshold means XLA
MATERIALIZES the (psf_width*P, band) reduce transient there; smooth peak with a time
cliff means a compute-pattern (tiling/vectorization) collapse instead.

Correctness gate (free, apples-to-apples): sorted and scatter-add are value-equal up
to float32 summation order, so per band the on-vs-off outputs must agree at rel-max
<= 1e-5 -- a check that the ablation compares the same computation.

Run (gautschi 1x H100):  sbatch plans/experiments/projector_kernels/fwd_guard_cliff.slurm
"""
import json
import os
import subprocess
import sys

import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
VIEWS = 1024
CHANNELS = 1024
P = 8192                              # fixed pixel count (matches fwd_guard_sweep 1)
BANDS = (1024, 1152, 1280, 1536, 1792, 2048, 3072, 4096)
SORTED = (1, 0)                       # 1 = sorted segment-sum, 0 = unrolled scatter-add
OUT_DIR = '/scratch/gautschi/buzzard/headroom_fwd_guard'
WARMUP, TRIALS = 1, 3
REL_TOL = 1e-5


def cell_paths(band):
    return {s: os.path.join(OUT_DIR, f'cliff_b{band}_sorted{s}.npy') for s in SORTED}


def worker(cfg):
    os.environ['MBIRJAX_DISABLE_PALLAS'] = '1'      # XLA forward path only
    os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.9')
    import time
    import mbirjax
    import jax
    import jax.numpy as jnp

    out = dict(cfg)
    try:
        band, sorted_flag = cfg['band'], cfg['sorted']
        sino_shape = (VIEWS, band, CHANNELS)
        angles = np.linspace(0, np.pi, VIEWS, endpoint=False)
        model = mbirjax.ParallelBeamModel(sino_shape, angles)
        model.configure_devices(1)
        out['device_kind'] = jax.devices()[0].device_kind

        # Force the channel reduction and REBUILD the projectors so the static
        # ProjectorParams.sort_by_channel picks it up (it is baked at create_projectors;
        # setting model.tiles alone would leave the previously baked flag in place).
        model.tiles = model.tiles._replace(sort_by_channel=bool(sorted_flag))
        model.create_projectors()
        out['baked_sort_by_channel'] = int(model.projector_functions.projector_params.sort_by_channel)

        recon_shape = model.get_params('recon_shape')
        rng = np.random.default_rng(0)
        full = np.asarray(mbirjax.gen_full_indices(
            recon_shape, use_ror_mask=model.get_params('use_ror_mask')))
        idx = jnp.asarray(np.sort(rng.choice(full, size=P, replace=False)))
        values = jax.device_put(
            jnp.asarray(rng.random((P, recon_shape[2]), dtype=np.float32)),
            jax.devices()[0])
        jax.block_until_ready(values)

        call = lambda: model.projector_functions.sparse_forward_project(values, idx)
        for _ in range(WARMUP):
            jax.block_until_ready(call())
        ts, r = [], None
        for _ in range(TRIALS):
            r = None                       # release before the next alloc (lessons.md sec 5)
            t0 = time.perf_counter()
            r = jax.block_until_ready(call())
            ts.append(time.perf_counter() - t0)
        out['wall_ms'] = round(1000 * float(np.median(ts)), 2)
        out['trials_ms'] = [round(1000 * t, 2) for t in ts]

        stats = jax.local_devices()[0].memory_stats() or {}
        out['peak_gb'] = round(stats.get('peak_bytes_in_use', 0) / 2 ** 30, 2)
        os.makedirs(OUT_DIR, exist_ok=True)
        np.save(cell_paths(band)[sorted_flag], np.asarray(r))
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
    results = {}
    for band in BANDS:
        for sorted_flag in SORTED:
            cfg = dict(band=band, sorted=sorted_flag)
            r = run_cell(cfg)
            results[(band, sorted_flag)] = r
            print(f'[b={band} sorted={sorted_flag}] {r.get("status")} '
                  f'wall={r.get("wall_ms")}ms peak={r.get("peak_gb")}GB '
                  f'baked={r.get("baked_sort_by_channel")}', flush=True)
            if r.get('status') == 'error':
                print(r.get('traceback', '')[-1500:], flush=True)

        # Per-band correctness gate: sorted vs scatter-add must agree at the float gate.
        paths = cell_paths(band)
        if all(results[(band, s)].get('status') == 'ok' for s in SORTED):
            a = np.load(paths[1])
            b = np.load(paths[0])
            rel = float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(a))), 1e-30))
            results[('rel', band)] = rel
            del a, b
        for p in paths.values():
            if os.path.exists(p):
                os.remove(p)

    os.makedirs(OUT_DIR, exist_ok=True)
    dumpable = {f'{k[0]}_{k[1]}': v for k, v in results.items()}
    with open(os.path.join(OUT_DIR, 'fwd_guard_cliff_results.json'), 'w') as f:
        json.dump(dumpable, f, indent=1)

    # ── Summary: on/off walls + ratio per band, plus the per-band value gate ──
    print('\n===== fwd cliff summary (P={}, views={}, channels={}) ====='.format(
        P, VIEWS, CHANNELS), flush=True)
    print(f'{"band":>5} {"sorted_ms":>10} {"scatter_ms":>11} {"sorted/scatter":>15} '
          f'{"sorted_peak":>12} {"scatter_peak":>13} {"rel(on-off)":>12}', flush=True)
    base_sorted = base_scatter = None
    for band in BANDS:
        so = results.get((band, 1), {})
        sc = results.get((band, 0), {})
        if so.get('status') != 'ok' or sc.get('status') != 'ok':
            print(f'{band:>5}  INCOMPLETE', flush=True)
            continue
        ratio = so['wall_ms'] / sc['wall_ms']
        rel = results.get(('rel', band), float('nan'))
        print(f'{band:>5} {so["wall_ms"]:>10.1f} {sc["wall_ms"]:>11.1f} '
              f'{ratio:>15.2f} {so["peak_gb"]:>12.2f} {sc["peak_gb"]:>13.2f} '
              f'{rel:>12.2g}', flush=True)

    # ── Slope check: per-work cost vs band=1024 baseline, each algorithm ──
    print('\n===== per-work cost vs band=1024 (categorical cliff shows as a jump) =====',
          flush=True)
    for label, flag in (('sorted', 1), ('scatter', 0)):
        base = results.get((1024, flag), {}).get('wall_ms')
        if not base:
            continue
        cells = []
        for band in BANDS:
            w = results.get((band, flag), {}).get('wall_ms')
            if w:
                # normalize by band (work ~ band); a flat curve = linear, a jump = cliff
                cells.append(f'{band}:{(w / band) / (base / 1024):.2f}x')
        print(f'  {label:>8} (per-band-element, 1024=1.0): ' + '  '.join(cells),
              flush=True)


if __name__ == '__main__':
    if len(sys.argv) >= 3 and sys.argv[1] == '--worker':
        worker(json.loads(sys.argv[2]))
    else:
        main()
