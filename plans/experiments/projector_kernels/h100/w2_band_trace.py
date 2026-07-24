"""Wave-2 attribution: per-kernel device time of the CONE banded back path at n>=2
(2026-07-13; follows the w2_scaling_baseline result: cone back n=2 ANTI-scales,
18.9 -> 27.5 s, while parallel's band path scales normally).

Question A5 hangs on: is the band kernel's device time still dominated by the
transpose/layout fusion inside the vertical fan (the June ncu attribution,
`input_transpose_fusion` L1-bound), on CURRENT code?  The band path carries
named_scope tags (cone/back/band/{horizontal,vertical}_fan), so a short jax trace
of a few back_project calls attributes by region and by fusion name directly.

Cells (own subprocess each; single node):
  cone_n2, cone_n4   -- the anti-scaling cases
  cone_n1            -- the healthy single-device reference (pixel kernel path)
  par_n2             -- the normally-scaling band path, for contrast
Each: warm-up back_project, then a traced back_project (3 repeats inside one trace);
parse with the June trace_utils (self-time, nesting-safe) and print the top device
events + the named-scope rollup.

Run: python plans/experiments/projector_kernels/w2_band_trace.py  (constants below)
"""
import glob
import json
import os
import subprocess
import sys

# ── Config ────────────────────────────────────────────────────────────────────
SINO_SHAPE = (1024, 1024, 1024)
CELLS = [
    dict(name='cone_n2', geometry='cone', n=2),
    dict(name='cone_n4', geometry='cone', n=4),
    dict(name='cone_n1', geometry='cone', n=1),
    dict(name='par_n2', geometry='parallel', n=2),
]
TRACE_ROOT = '/scratch/gautschi/buzzard/w2_band_traces'
RESULTS_PATH = os.path.expanduser('~/headroom/results/w2_band_trace.jsonl')
TRACE_UTILS_CANDIDATES = [
    os.path.expanduser('~/PycharmProjects/mbirjax_metrics/experiments/profiling'),
    '/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirjax_metrics/experiments/profiling',
]
TOP_EVENTS = 20

if os.environ.get('W2T_SMOKE') == '1':
    SINO_SHAPE = (32, 16, 24)
    CELLS = [dict(name='cone_n1', geometry='cone', n=1)]
    TRACE_ROOT = os.path.join(os.environ.get('TMPDIR', '/tmp'), 'w2_band_traces')
    RESULTS_PATH = os.path.join(TRACE_ROOT, 'results.jsonl')


def _find_perfetto(out_dir):
    hits = glob.glob(os.path.join(out_dir, '**', '*.trace.json.gz'), recursive=True)
    if not hits:
        raise FileNotFoundError(f'no perfetto trace under {out_dir}')
    return max(hits, key=os.path.getmtime)


def worker(cfg):
    os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.94')
    import time
    import numpy as np
    import jax
    import jax.numpy as jnp
    import mbirjax

    views, rows, channels = SINO_SHAPE
    angles = np.linspace(-np.pi / 2, np.pi / 2, views, endpoint=False)
    if cfg['geometry'] == 'cone':
        model = mbirjax.ConeBeamModel(SINO_SHAPE, angles,
                                      source_detector_dist=4 * channels,
                                      source_iso_dist=4 * channels)
    else:
        model = mbirjax.ParallelBeamModel(SINO_SHAPE, angles)
    print(f'devices={jax.devices()} summary={model.device_summary}', flush=True)
    recon_shape = model.get_params('recon_shape')
    idx = mbirjax.gen_full_indices(recon_shape, use_ror_mask=True)
    sino = jnp.asarray(np.random.default_rng(0).random(SINO_SHAPE, dtype=np.float32))
    jax.block_until_ready(sino)

    jax.block_until_ready(model.sparse_back_project(sino, idx))       # compile + warm
    trace_dir = os.path.join(TRACE_ROOT, cfg['name'])
    t0 = time.perf_counter()
    with jax.profiler.trace(trace_dir):
        for _ in range(3):
            jax.block_until_ready(model.sparse_back_project(sino, idx))
    wall = time.perf_counter() - t0
    print(f'traced wall (3 calls) = {wall:.3f}s', flush=True)

    # Parse with the June campaign's trace_utils (self-time, nesting handled).
    for cand in TRACE_UTILS_CANDIDATES:
        if os.path.isdir(cand):
            sys.path.insert(0, cand)
            break
    import trace_utils
    events, tracks, _ = trace_utils.fusion_self_time(_find_perfetto(trace_dir))

    # Device tracks per the E1 heuristic ('/device:' non-cpu or 'stream', excluding
    # tf_*/client host threads); all raw track totals are saved for offline re-check.
    def is_device_track(label):
        lbl = label.lower()
        if lbl.startswith('tf_') or 'client' in lbl:
            return False
        return ('/device:' in lbl and 'cpu' not in lbl) or 'stream' in lbl

    total_device = sum(us for lbl, us in tracks.items() if is_device_track(lbl))
    print(f'total device self-time = {total_device / 1e6:.3f}s over 3 calls '
          f'(wall {wall:.3f}s)', flush=True)
    # Event names carry the HLO op (with named-scope prefixes where present); rank by
    # self-time.  Host runtime rows are filtered by name.
    ranked = [(us, cnt, name) for name, (us, cnt) in events.items()
              if not trace_utils.is_host_runtime(name)]
    ranked.sort(key=lambda t: -t[0])
    for us, cnt, name in ranked[:TOP_EVENTS]:
        share = us / total_device * 100 if total_device else 0.0
        print(f'  {share:6.2f}%  {us / 1e6:8.3f}s  x{cnt:<5d} {name[:130]}', flush=True)
    scoped = {}
    for us, cnt, name in ranked:
        for tag in ('horizontal_fan', 'vertical_fan', 'transpose'):
            if tag in name:
                scoped[tag] = scoped.get(tag, 0) + us
    print(f'scope rollup (s): { {k: round(v / 1e6, 3) for k, v in scoped.items()} }',
          flush=True)
    with open(RESULTS_PATH, 'a') as f:
        f.write(json.dumps(dict(cfg, wall_3calls=wall, total_device_s=total_device / 1e6,
                                tracks={k: round(v / 1e6, 4) for k, v in tracks.items()},
                                top=[dict(self_s=round(us / 1e6, 4), count=cnt,
                                          name=name[:200])
                                     for us, cnt, name in ranked[:TOP_EVENTS]],
                                scoped={k: round(v / 1e6, 4)
                                        for k, v in scoped.items()})) + '\n')


def orchestrator():
    os.makedirs(TRACE_ROOT, exist_ok=True)
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    for cfg in CELLS:
        env = dict(os.environ, W2T_CELL=json.dumps(cfg),
                   CUDA_VISIBLE_DEVICES=','.join(str(i) for i in range(cfg['n'])),
                   MBIRJAX_DISABLE_PALLAS='1')     # XLA path everywhere (the A5 target)
        print(f"[{cfg['name']}] starting", flush=True)
        rc = subprocess.run([sys.executable, os.path.abspath(__file__)],
                            env=env).returncode
        print(f"[{cfg['name']}] rc={rc}", flush=True)
    print('=== w2_band_trace done ===', flush=True)


if __name__ == '__main__':
    if os.environ.get('W2T_CELL'):
        worker(json.loads(os.environ['W2T_CELL']))
    else:
        orchestrator()
