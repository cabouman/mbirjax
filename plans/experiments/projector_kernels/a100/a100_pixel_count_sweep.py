"""A100 campaign follow-up: PIXEL COUNT as the swept axis, not "subset vs full grid".

Why this exists (Greg 2026-07-20, and he is right)
--------------------------------------------------
The step-2 sweep reported two operating points labelled "subset" and "full grid".  Those
labels are not intrinsic: a wider recon's 128-subset piece can hold more pixels than a
narrow recon's entire grid.  The real variable is the ABSOLUTE PIXEL COUNT, and it
matters here for a specific structural reason -- the pallas back path does NO pixel
batching.  ``back_project_single_device``/``cone_back_project_band`` put the whole index
set into ONE kernel grid:

    grid=(rows_padded // rc, num_pixels)      # parallel back  (_make_back_call)
    grid=(l_padded   // lc, num_pixels)       # cone back      (_make_cone_back_call)

and ``num_pixels`` is part of each builder's functools.cache key, so every distinct count
COMPILES ITS OWN KERNEL.  (The fwd_pixel_batch / back_pixel_batch TilePolicy knobs are
XLA-path knobs; the pallas drivers ignore them.)  Pixel count therefore sets the grid's
parallelism, and small counts are occupancy-starved on 108 SMs while large ones saturate.

The counts a REAL 15-iteration recon visits, at recon (688, 688, 810):

    2,895   x12 iterations   128 subsets (partition_sequence [2,4,6,7] repeats its last)
    5,791   x1               64 subsets
   23,162   x1               16 subsets
   92,649   x1                4 subsets
  473,344   x1               the Hessian -- compute_hessian_diagonal uses
                             indices = arange(num_recon_rows * num_recon_cols), i.e. the
                             UNMASKED rectangle (not the ROR set), at coeff_power=2

Step 2 measured 2,896 and 370,596.  The first is exactly the dominant case; the second
corresponds to NO call the recon makes (the ROR-masked full grid) and is 28% smaller than
the Hessian's actual count.  The three middle counts were never measured -- and that gap
matters, because ROW_CHUNK=512 was 5.3% FASTER at 2,896 and 9.4% SLOWER at 370,596, so a
crossover sits somewhere inside the unmeasured range.

This script sweeps the surviving configurations across all five REAL counts, so the
result is a curve with a locatable crossover instead of two labelled endpoints.

Run:  sbatch a100_pixel_count_sweep.slurm
"""
import json
import os
import random
import subprocess
import sys
import time

# ── Config ────────────────────────────────────────────────────────────────────
SINO_SHAPE = (752, 720, 688)
GEOMETRY = os.environ.get('A100_PX_GEOMETRY', 'parallel')   # 'parallel' | 'cone'
CONE_SDD_OVER_CHANNELS = 4.0
# The real operating points: (label, num_subsets or None for the Hessian, coeff_power).
POINTS = [('g128', 128, 1), ('g64', 64, 1), ('g16', 16, 1), ('g4', 4, 1),
          ('hessian', None, 2)]
# Configurations to compare.  VIEW_CHUNK is the candidate; ROW_CHUNK locates the
# crossover that flipped sign between the step-2 endpoints.
CONFIGS = [
    {},                                        # shipped
    {'VIEW_CHUNK': 64},
    {'VIEW_CHUNK': 32},
    {'ROW_CHUNK': 512},
    {'ROW_CHUNK': 512, 'VIEW_CHUNK': 64},
]
PASSES = 2
TRIALS = 10
WARMUP_SECONDS = 0.5
# Overridable so the SAME harness runs on gautschi's H100s (the boundary
# re-run) without editing: the constants under test ship for both arches.
OUT_DIR = os.environ.get('A100_OUT_DIR',
                         '/scratch/gilbreth/buzzard/a100_tuning')
RESULTS = os.path.join(OUT_DIR, 'results', 'pixel_count_sweep_%s.json' % GEOMETRY)
PIXEL_CACHE = os.path.join(OUT_DIR, 'cache',
                           'pxsweep_%s_%dx%dx%d.npz' % ((GEOMETRY,) + SINO_SHAPE))

SHIPPED = ({'ROW_CHUNK': 256, 'NUM_WARPS': 2, 'VIEW_CHUNK': 128}
           if GEOMETRY == 'parallel' else
           {'CONE_LC': 128, 'CONE_NUM_WARPS': 1, 'VIEW_CHUNK': 128})

_ROLE = os.environ.get('A100_PX_ROLE', 'orchestrate')


def _clear_all_kernel_caches():
    from mbirjax import _pallas_kernels as pk
    n = 0
    for name in dir(pk):
        o = getattr(pk, name, None)
        if hasattr(o, 'cache_clear') and hasattr(o, 'cache_info'):
            o.cache_clear()
            n += 1
    return n


def _build_model():
    import numpy as np
    import mbirjax
    n_views, n_rows, n_channels = SINO_SHAPE
    angles = np.linspace(0, np.pi, n_views, endpoint=False)
    if GEOMETRY == 'parallel':
        model = mbirjax.ParallelBeamModel(SINO_SHAPE, angles)
    else:
        sdd = CONE_SDD_OVER_CHANNELS * n_channels
        model = mbirjax.ConeBeamModel(SINO_SHAPE, angles, source_detector_dist=sdd,
                                      source_iso_dist=sdd / 2.0)
    model.configure_devices(1)
    return model


def _pixel_sets(model):
    """All five real operating points, cached to scratch (deterministic given the seed)."""
    import numpy as np
    if os.path.exists(PIXEL_CACHE):
        d = np.load(PIXEL_CACHE)
        return {k: d[k] for k in d.files}
    import mbirjax
    from mbirjax.vcd_utils import gen_set_of_pixel_partitions
    recon_shape = model.get_params('recon_shape')
    sets = {}
    np.random.seed(0)
    for label, ns, _cp in POINTS:
        if ns is None:
            # The Hessian's own index set: the UNMASKED rectangle, matching
            # compute_hessian_diagonal's arange(num_recon_rows * num_recon_cols).
            sets[label] = np.arange(recon_shape[0] * recon_shape[1], dtype=np.int32)
        else:
            part = gen_set_of_pixel_partitions(recon_shape, [ns])[0]
            sets[label] = np.asarray(part[0], dtype=np.int32)
    os.makedirs(os.path.dirname(PIXEL_CACHE), exist_ok=True)
    tmp = PIXEL_CACHE + '.tmp%d.npz' % os.getpid()   # .npz: np.savez appends it otherwise
    np.savez(tmp, **sets)
    os.replace(tmp, PIXEL_CACHE)
    return sets


def run_cell():
    import numpy as np
    import jax
    from mbirjax import _pallas_kernels as pk

    spec = json.loads(os.environ['A100_PX_CELL'])
    consts = dict(SHIPPED, **spec['consts'])
    label = spec['point']
    cp = dict((l, c) for l, _n, c in POINTS)[label]

    for k, v in consts.items():
        if k != 'VIEW_CHUNK':
            setattr(pk, k, v)
    ncaches = _clear_all_kernel_caches()

    model = _build_model()
    model.tiles = model.tiles._replace(back_view_batch=consts['VIEW_CHUNK'])
    sets = _pixel_sets(model)
    dev = model.sino_placement.devices[0]
    idx = jax.device_put(sets[label], dev)
    rng = np.random.default_rng(0)
    sino = jax.device_put(rng.random(SINO_SHAPE, dtype=np.float32), dev)
    jax.block_until_ready(sino)
    num_slices = model.get_params('recon_shape')[2]

    if GEOMETRY == 'parallel':
        builder = pk._make_back_call
        fn = lambda: pk.back_project_single_device(model, sino, idx, coeff_power=cp)
    else:
        builder = pk._make_cone_back_call
        fn = lambda: pk.cone_back_project_band(model, sino, idx, 0, num_slices,
                                               coeff_power=cp)
    misses0 = builder.cache_info().misses
    jax.block_until_ready(fn())
    rebuilt = builder.cache_info().misses > misses0

    t_end = time.perf_counter() + WARMUP_SECONDS
    while time.perf_counter() < t_end:
        jax.block_until_ready(fn())
    best = min(_timed(fn) for _ in range(TRIALS))
    st = jax.devices()[0].memory_stats() or {}
    print('PX_RESULT ' + json.dumps({
        'consts': spec['consts'], 'point': label, 'coeff_power': cp,
        'num_pixels': int(idx.shape[0]), 'time_s': best,
        'peak_GB': int(st.get('peak_bytes_in_use', 0)) / 1e9,
        'rebuilt': bool(rebuilt), 'caches_cleared': ncaches,
        'pass': spec.get('pass'),
    }))


def _timed(fn):
    import jax
    t0 = time.perf_counter()
    jax.block_until_ready(fn())
    return time.perf_counter() - t0


def orchestrate():
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    cells = [{'consts': c, 'point': lbl}
             for lbl, _ns, _cp in POINTS for c in CONFIGS]
    results = []
    for p in range(PASSES):
        order = list(range(len(cells)))
        random.Random(500 + p).shuffle(order)
        for i in order:
            spec = dict(cells[i], **{'pass': p})
            env = dict(os.environ, A100_PX_ROLE='cell',
                       A100_PX_CELL=json.dumps(spec))
            out = subprocess.run([sys.executable, '-u', __file__], env=env,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True)
            rec = None
            for ln in out.stdout.splitlines():
                if ln.startswith('PX_RESULT '):
                    rec = json.loads(ln[len('PX_RESULT '):])
            if rec is None:
                print('   cell FAILED (%s %s) rc=%d' % (spec['consts'], spec['point'],
                                                        out.returncode), flush=True)
                for ln in out.stdout.splitlines()[-8:]:
                    print('     | ' + ln)
                continue
            results.append(rec)
            print('   %-34s %-8s %7d px  %.5f s  %5.2f GB%s'
                  % (json.dumps(rec['consts']) or '{}', rec['point'],
                     rec['num_pixels'], rec['time_s'], rec['peak_GB'],
                     '' if rec['rebuilt'] else '  !! NOT REBUILT'), flush=True)
    with open(RESULTS, 'w') as f:
        json.dump(results, f, indent=2)
    _report(results)
    return 0


def _report(results):
    print('\n' + '=' * 86)
    print('PIXEL-COUNT SWEEP -- %s back projection, sino %s' % (GEOMETRY, SINO_SHAPE))
    print('Each column is a REAL operating point of a 15-iteration recon.')
    print('=' * 86)
    labels = [l for l, _n, _c in POINTS]
    npix = {}
    best = {}
    for r in results:
        key = json.dumps(r['consts'], sort_keys=True)
        npix[r['point']] = r['num_pixels']
        cur = best.get((key, r['point']))
        if cur is None or r['time_s'] < cur[0]:
            best[(key, r['point'])] = (r['time_s'], r['peak_GB'])
    print('%-30s %s' % ('', ''.join('%14s' % ('%s' % l) for l in labels)))
    print('%-30s %s' % ('config \\ pixels',
                        ''.join('%14s' % ('%d' % npix.get(l, 0)) for l in labels)))
    ship_key = json.dumps({}, sort_keys=True)
    for c in CONFIGS:
        key = json.dumps(c, sort_keys=True)
        lbl = json.dumps(c) if c else 'shipped'
        row = ''
        for l in labels:
            v = best.get((key, l))
            row += '%14s' % ('%.5f' % v[0] if v else '-')
        print('%-30s %s' % (lbl, row))
    print()
    print('Relative to shipped (negative = FASTER), per operating point:')
    print('%-30s %s' % ('config', ''.join('%14s' % l for l in labels)))
    for c in CONFIGS[1:]:
        key = json.dumps(c, sort_keys=True)
        row = ''
        for l in labels:
            v, s = best.get((key, l)), best.get((ship_key, l))
            row += '%14s' % ('%+.1f%%' % (100 * (v[0] - s[0]) / s[0]) if v and s else '-')
        print('%-30s %s' % (json.dumps(c), row))
    print()
    print('A config that changes SIGN across the row has a crossover between those two')
    print('pixel counts -- it is not a single best value, and any default that ignores')
    print('the operating point is a compromise.  A config negative EVERYWHERE dominates.')


if __name__ == '__main__':
    if _ROLE == 'cell':
        run_cell()
    else:
        sys.exit(orchestrate())
