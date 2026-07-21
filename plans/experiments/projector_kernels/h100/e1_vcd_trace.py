"""E1 Amdahl gate (part 2): device-kernel vs host share of a PRODUCTION-granularity VCD
iteration at 512^3- and 1024^3-class sizes -- THE gating number for the headroom plan's
(a) track (gpu_headroom_plan.md section 3).

Design -- the window-difference trick: each cell runs, in one process,
    warm-up: recon(max_iterations=1)                   (compiles everything)
    window A: traced recon(max_iterations=1)
    window B: traced recon(max_iterations=2)
with np.random.seed(0) before every recon (identical partitions and subset order) and
stop_threshold_change_pct=0 (exact iteration counts).  B - A = exactly ONE
granularity-128 iteration (128 subsets), with the init block (FBP/FDK + Hessian + error
sinogram) and all one-time costs cancelled EXACTLY.  Device share of an iteration
= (device_self_time_B - device_self_time_A) / (wall_B - wall_A).

Traces are parsed with the June profiling campaign's proven trace_utils (self-time with
nesting handled); raw per-track totals and the top per-event deltas (the iteration's
device-time attribution: projector fusions vs prior vs stats) are saved to RESULTS_PATH
so the device-track classification can be re-derived offline if the label heuristic here
mislabels anything.

Each cell runs in its OWN SUBPROCESS (fresh jax, honest memory).

Run:  python plans/experiments/projector_kernels/e1_vcd_trace.py    (edit constants below)
"""
import glob
import json
import os
import subprocess
import sys

# ── Config ────────────────────────────────────────────────────────────────────
CELLS = [
    dict(geometry='parallel', sino_shape=(512, 504, 496)),
    dict(geometry='cone', sino_shape=(512, 504, 496)),
    dict(geometry='parallel', sino_shape=(1024, 1008, 992)),
    dict(geometry='cone', sino_shape=(1024, 1008, 992)),
]
PARTITION_SEQUENCE = [7]                 # granularity 128 every iteration (production regime)
TRACE_ROOT = os.path.expanduser('~/headroom/results/e1_traces')
RESULTS_PATH = os.path.expanduser('~/headroom/results/e1_vcd_trace.jsonl')
# Where the June campaign's trace parser lives (first existing wins).
TRACE_UTILS_CANDIDATES = [
    os.path.expanduser('~/PycharmProjects/mbirjax_metrics/experiments/profiling'),
    '/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirjax_metrics/experiments/profiling',
]
TOP_EVENT_DELTAS = 25


def _find_perfetto(out_dir):
    """The jax profiler writes <dir>/plugins/profile/<ts>/*.trace.json.gz."""
    hits = glob.glob(os.path.join(out_dir, '**', '*.trace.json.gz'), recursive=True)
    if not hits:
        raise FileNotFoundError(f'no perfetto trace under {out_dir}')
    return max(hits, key=os.path.getmtime)


def worker(cfg):
    os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.94')
    import time
    import numpy as np
    import mbirjax
    import jax

    for cand in TRACE_UTILS_CANDIDATES:
        if os.path.isdir(cand):
            sys.path.insert(0, cand)
            break
    import trace_utils   # noqa: E402  (mbirjax_metrics/experiments/profiling)

    out = dict(cfg)
    try:
        sino_shape = tuple(cfg['sino_shape'])
        angles = np.linspace(0, np.pi, sino_shape[0], endpoint=False)
        if cfg['geometry'] == 'parallel':
            model = mbirjax.ParallelBeamModel(sino_shape, angles)
        else:
            model = mbirjax.ConeBeamModel(sino_shape, angles,
                                          source_detector_dist=4 * sino_shape[2],
                                          source_iso_dist=4 * sino_shape[2])
        model.configure_devices(1)
        model.set_params(partition_sequence=PARTITION_SEQUENCE)
        recon_shape = model.get_params('recon_shape')

        phantom = mbirjax.generate_3d_shepp_logan_low_dynamic_range(recon_shape)
        sinogram = jax.block_until_ready(model.forward_project(phantom))
        del phantom

        def run_recon(n_iter):
            np.random.seed(0)      # identical partitions + subset order in every window
            t0 = time.perf_counter()
            recon, _ = model.recon(sinogram, max_iterations=n_iter,
                                   stop_threshold_change_pct=0, print_logs=False)
            jax.block_until_ready(recon)
            wall = time.perf_counter() - t0
            del recon
            return wall

        out['warmup_wall_s'] = round(run_recon(1), 3)      # compile everything

        # Untraced window pair FIRST: the honest iteration wall, and the check on how much
        # the profiler inflates the traced windows below.
        wall_a0, wall_b0 = run_recon(1), run_recon(2)
        out['iter_wall_untraced_s'] = round(wall_b0 - wall_a0, 3)

        windows = {}
        for tag, n_iter in (('A', 1), ('B', 2)):
            tdir = os.path.join(TRACE_ROOT, f'{cfg["geometry"]}_{sino_shape[0]}_{tag}')
            with jax.profiler.trace(tdir):
                wall = run_recon(n_iter)
            events, tracks, _ = trace_utils.fusion_self_time(_find_perfetto(tdir))
            windows[tag] = dict(wall=wall, tracks=tracks,
                                events={k: v for k, v in events.items()})
            out[f'wall_{tag}_s'] = round(wall, 3)

        # Device share of the differenced iteration.  Heuristic: device timelines are
        # '/device:GPU:...' tracks or stream sub-tracks -- but NOT host-side runtime threads
        # like 'tf_XLAPjRtGpuClient' (which also contain 'gpu').  ALL raw track totals are
        # saved so this classification can be re-derived offline.
        def dev_us(tracks):
            total = 0.0
            for label, us in tracks.items():
                lbl = label.lower()
                if lbl.startswith('tf_') or 'client' in lbl:
                    continue
                if ('/device:' in lbl and 'cpu' not in lbl) or 'stream' in lbl:
                    total += us
            return total

        d_wall = windows['B']['wall'] - windows['A']['wall']
        d_dev = (dev_us(windows['B']['tracks']) - dev_us(windows['A']['tracks'])) / 1e6
        out['iter_wall_s'] = round(d_wall, 3)
        out['iter_device_s'] = round(d_dev, 3)
        out['iter_device_share'] = round(d_dev / d_wall, 4) if d_wall > 0 else None
        out['tracks_A'] = {k: round(v / 1e6, 4) for k, v in windows['A']['tracks'].items()}
        out['tracks_B'] = {k: round(v / 1e6, 4) for k, v in windows['B']['tracks'].items()}

        deltas = []
        ev_a, ev_b = windows['A']['events'], windows['B']['events']
        for name, (us_b, cnt_b) in ev_b.items():
            us_a, cnt_a = ev_a.get(name, (0.0, 0))
            if us_b - us_a > 0:
                deltas.append((us_b - us_a, cnt_b - cnt_a, name))
        deltas.sort(key=lambda t: -t[0])
        out['top_event_deltas'] = [dict(self_s=round(us / 1e6, 4), count=cnt, name=name[:120])
                                   for us, cnt, name in deltas[:TOP_EVENT_DELTAS]]
        out['status'] = 'ok'
    except Exception:
        import traceback
        out['status'] = 'error'
        out['traceback'] = traceback.format_exc()
    print('RESULT ' + json.dumps(out), flush=True)


def main():
    os.makedirs(TRACE_ROOT, exist_ok=True)
    results = []
    for cfg in CELLS:
        proc = subprocess.run([sys.executable, os.path.abspath(__file__), '--worker',
                               json.dumps(cfg)], capture_output=True, text=True)
        got = False
        for line in proc.stdout.splitlines():
            if line.startswith('RESULT '):
                r = json.loads(line[len('RESULT '):])
                results.append(r)
                got = True
                share = r.get('iter_device_share')
                print(f'{r["geometry"]:9s} {r["sino_shape"]}: iter_wall={r.get("iter_wall_s")} s '
                      f'iter_device={r.get("iter_device_s")} s  SHARE={share}'
                      if r['status'] == 'ok' else
                      f'{r["geometry"]:9s} {r["sino_shape"]}: ERROR (see jsonl)', flush=True)
        if not got:
            print(f'[no RESULT, rc={proc.returncode}] {cfg}\n{proc.stderr[-2000:]}', flush=True)

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
    print(f'\n[raw results (tracks + event deltas) in {RESULTS_PATH}]')


if __name__ == '__main__':
    if len(sys.argv) >= 3 and sys.argv[1] == '--worker':
        worker(json.loads(sys.argv[2]))
    else:
        main()
