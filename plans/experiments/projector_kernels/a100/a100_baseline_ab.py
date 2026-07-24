"""Step 1 of the A100 tuning campaign: a CONTROLLED pallas-on/off A/B of the BGA cone
recon, measuring wall time AND peak device memory.

Why this script exists rather than running `center_slice_zeiss.py` twice:

  * that script np.savez's the sinogram + weights + FDK recon into the CURRENT
    DIRECTORY -- multi-GB into a 25 GB home quota that fails jobs SILENTLY -- and ends
    in an interactive slice_viewer.  Here the preprocessing runs ONCE into a scratch
    cache and every timed cell reloads it, so the A/B varies exactly one thing;
  * a single run of each variant cannot separate a real effect from run-to-run drift.
    Repeats are INTERLEAVED (A B A B ...) so slow-node drift cannot masquerade as a
    variant effect;
  * MEMORY is a first-class result here, not a footnote (Greg 2026-07-20: roughly
    willing to trade 10% more memory for a 30% time reduction, i.e. an accept
    threshold of about 3x more time-saved than memory-spent).  The pallas back path
    allocates a channel-major sinogram copy (_to_channel_major, rows padded to a power
    of two), a (views, taps, pixels) weight array per view chunk, and holds two output
    buffers while accumulating -- none of which the XLA path allocates.  So the
    speedup already measured on this workload (111 s -> 74 s) needs a memory verdict
    before it can be called a good trade.

Measurement rules followed (project lessons.md sections 2-3 and .claude/claude_prompt.md):
  * each cell runs in an ISOLATED SUBPROCESS and the orchestrator stays JAX-FREE, so no
    orchestrator-held device memory pollutes a worker's peak_bytes_in_use (which is a
    process-cumulative high-water mark and cannot be reset);
  * the pallas path is switched with MBIRJAX_DISABLE_PALLAS, which availability()
    reads ONCE per process under functools.cache -- so it must be a per-process env
    var, which is exactly what a subprocess-per-cell design gives;
  * the first cell of each variant is a DISCARDED warmup: jax's persistent compile
    cache makes a cold run much slower, and that cost belongs to neither variant.

Run:  sbatch a100_baseline_ab.slurm      (or: python a100_baseline_ab.py)
All run parameters are in the Config block below -- no command-line arguments.
"""
import json
import os
import pickle
import subprocess
import sys
import time

# ── Config ────────────────────────────────────────────────────────────────────
DATASET = '/depot/bouman/data/Zeiss/purdue_BGA/17U1-250TC-Normal_Tomo_No_HART.txrm'
DOWNSAMPLE_FACTOR = 3          # center_slice_zeiss.py settings, held fixed
SUBSAMPLE_VIEW_FACTOR = 5
SHARPNESS = 1.5
SNR_DB = 35.0
MAX_ITERATIONS = 15
# 0 disables the early stop, forcing EXACTLY MAX_ITERATIONS (Greg 2026-07-20).  The
# default 0.2 lets the two variants stop at different iteration counts -- they differ
# in float rounding -- which would make the wall-time comparison unequal-work and
# quietly credit a convergence difference to the kernels.  Equal work by construction
# beats detecting the inequality afterwards; the check below stays as verification.
STOP_THRESHOLD_CHANGE_PCT = 0.0
REPEATS = 3                    # TIMED repeats per variant (plus 1 discarded warmup)
VARIANTS = ('pallas', 'xla')   # interleaved in this order
OUT_DIR = '/scratch/gilbreth/buzzard/a100_tuning'
CACHE = os.path.join(OUT_DIR, 'cache', 'bga_ds%d_vs%d.npz' % (DOWNSAMPLE_FACTOR,
                                                              SUBSAMPLE_VIEW_FACTOR))
# Model parameters travel in a pickle sidecar, not inside the npz: the required-params
# dict holds numpy view arrays (angles) and nested tuples, and the round trip has to be
# EXACT -- a lossy re-encoding would silently change the geometry under test.
PARAMS = CACHE.replace('.npz', '_params.pkl')
RESULTS = os.path.join(OUT_DIR, 'results', 'baseline_ab.json')
# Time and memory are a PARETO FRONTIER -- both matter, and neither is a constraint the
# other is optimized subject to (Greg 2026-07-20).  So this number is NOT an accept/
# reject gate: it is an INDIFFERENCE SLOPE for orientation, the trade Greg described as
# roughly break-even (10% more memory for 30% less time => 3.0).  A configuration on the
# unfavourable side of it is still reported, and is still on the frontier if nothing
# dominates it -- the choice among non-dominated points is Greg's, not the harness's.
INDIFFERENCE_SLOPE = 3.0

_ROLE = os.environ.get('A100_AB_ROLE', 'orchestrate')


# ── Role: prep (runs once; writes the scratch cache) ──────────────────────────
def prep():
    import numpy as np
    import mbirjax as mj
    import mbirjax.preprocess as mjp

    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    t0 = time.time()
    sinogram, ct_model = mjp.zeiss.get_sino_and_model(
        DATASET, downsample_factor=(DOWNSAMPLE_FACTOR, DOWNSAMPLE_FACTOR),
        subsample_view_factor=SUBSAMPLE_VIEW_FACTOR)
    ct_model.set_params(sharpness=SHARPNESS, snr_db=SNR_DB)
    # Captured AFTER the regularization knobs are set, so `regularization` carries the
    # sharpness/snr_db under test and the rebuild needs no second set_params.
    required, optional, regularization = ct_model.get_all_params()

    direct_recon = ct_model.direct_recon(sinogram)
    weights = mj.gen_weights(sinogram, weight_type='transmission_root')

    sinogram = np.asarray(sinogram, dtype=np.float32)
    np.savez(CACHE, sinogram=sinogram,
             weights=np.asarray(weights, dtype=np.float32),
             direct_recon=np.asarray(direct_recon, dtype=np.float32))
    with open(PARAMS, 'wb') as f:
        pickle.dump((required, optional, regularization), f)

    # Prove the round trip BEFORE any timed cell depends on it: a rebuilt model whose
    # recon_shape differs would silently make every cell a different workload.
    rebuilt = _rebuild()
    same = tuple(rebuilt.get_params('recon_shape')) == tuple(
        ct_model.get_params('recon_shape'))
    print('PREP round-trip recon_shape match: %s (%s vs %s)' % (
        same, tuple(rebuilt.get_params('recon_shape')),
        tuple(ct_model.get_params('recon_shape'))))
    if not same:
        raise SystemExit('FATAL: params round trip changed the recon geometry')

    # The shapes the step-2 kernel sweep must be built at -- recorded here so the sweep
    # never has to guess (or re-run the 7 GB preprocessing) to learn them.
    print('PREP shapes: sinogram=%s recon=%s  (%.1f s)' % (
        sinogram.shape, ct_model.get_params('recon_shape'), time.time() - t0))
    print('PREP cache: %s (%.2f GB)' % (CACHE, os.path.getsize(CACHE) / 1e9))


def _rebuild():
    """Rebuild the model from the cached params via the sanctioned path.

    ``build_model`` (not a bare constructor) because required_params carries a
    ``geometry_type`` entry the constructor does not accept, and because build_model
    applies a pinned ``recon_shape`` AFTER auto_set_recon_geometry -- the ordering a
    hand-rolled rebuild gets wrong, leaving the grid sized with default pitches.
    """
    from mbirjax.utilities import build_model
    with open(PARAMS, 'rb') as f:
        required, optional, regularization = pickle.load(f)
    return build_model(required, optional, regularization)


# ── Role: run (one timed cell; MBIRJAX_DISABLE_PALLAS is already set by the parent) ──
def run_cell():
    import numpy as np
    import mbirjax as mj
    import jax
    from mbirjax import _pallas_kernels as pk

    data = np.load(CACHE, allow_pickle=False)
    sinogram = data['sinogram']
    weights = data['weights']
    direct_recon = data['direct_recon']

    ct_model = _rebuild()
    ct_model.configure_devices(1)

    tiles = ct_model.tiles
    avail, why = pk.availability()
    t0 = time.time()
    recon, recon_dict = ct_model.recon(
        sinogram, init_recon=direct_recon, weights=weights,
        max_iterations=MAX_ITERATIONS,
        stop_threshold_change_pct=STOP_THRESHOLD_CHANGE_PCT)
    # recon() returns a NUMPY array (the device->host transfer already forced
    # completion, so the timing is synchronous either way).  Guarded rather than
    # removed so this stays correct if the return type ever becomes a jax array.
    if hasattr(recon, 'block_until_ready'):
        recon.block_until_ready()
    elapsed = time.time() - t0

    dev = jax.devices()[0]
    stats = dev.memory_stats() or {}
    result = {
        'variant': os.environ.get('A100_AB_VARIANT'),
        'repeat': int(os.environ.get('A100_AB_REPEAT', '-1')),
        'elapsed_s': elapsed,
        'peak_bytes': int(stats.get('peak_bytes_in_use', 0)),
        'pool_bytes': int(stats.get('pool_bytes', 0)),
        'num_allocs': int(stats.get('num_allocs', 0)),
        'largest_alloc_bytes': int(stats.get('largest_alloc_size', 0)),
        'pallas_available': bool(avail),
        'pallas_why': why,
        # The flags actually in force -- the ground truth for "was the kernel used",
        # independent of what the env var was meant to do.
        'fwd_pallas': bool(tiles.fwd_pallas),
        'back_pallas': bool(tiles.back_pallas),
        'sino_shape': list(sinogram.shape),
        'recon_shape': list(ct_model.get_params('recon_shape')),
        'device_kind': dev.device_kind,
    }
    # The 0.2% stop criterion is LIVE by default, and the two variants differ in float
    # rounding -- so they can stop at different iteration counts.  If they do, wall
    # time is not an apples-to-apples comparison and the report must say so rather than
    # quietly credit the difference to the kernels.
    result.update(_convergence_fields(recon_dict))
    print('CELL_RESULT ' + json.dumps(result))


def _convergence_fields(recon_dict):
    """num_iterations + final forward-model RMSE, tolerant of recon_dict's two shapes.

    ``recon()`` may pass recon_dict through convert_subdicts_to_strings (for hdf5
    round-tripping), so recon_params can arrive as a dict OR as its string repr.
    """
    out = {'num_iterations': None, 'final_fm_rmse': None}
    rp = recon_dict.get('recon_params')
    if isinstance(rp, dict):
        out['num_iterations'] = rp.get('num_iterations')
        fm = rp.get('fm_rmse')
        if isinstance(fm, (list, tuple)) and len(fm):
            out['final_fm_rmse'] = float(fm[-1])
    elif isinstance(rp, str):
        import re
        m = re.search(r"'num_iterations':\s*(\d+)", rp)
        if m:
            out['num_iterations'] = int(m.group(1))
    return out


# ── Role: orchestrate (JAX-free: it must hold NO device memory) ───────────────
def orchestrate():
    os.makedirs(os.path.join(OUT_DIR, 'results'), exist_ok=True)
    env_base = dict(os.environ)

    if not (os.path.exists(CACHE) and os.path.exists(PARAMS)):
        print('=== prep (once): preprocessing the BGA scan into the scratch cache ===',
              flush=True)
        env = dict(env_base, A100_AB_ROLE='prep')
        rc = subprocess.call([sys.executable, '-u', __file__], env=env)
        if rc != 0:
            print('prep FAILED (rc=%d)' % rc)
            return 1
    else:
        print('=== prep: reusing cache %s ===' % CACHE, flush=True)

    cells = []
    # One discarded warmup per variant, then REPEATS timed passes, INTERLEAVED.
    order = [(v, -1) for v in VARIANTS]
    for r in range(REPEATS):
        order += [(v, r) for v in VARIANTS]

    results = []
    for variant, repeat in order:
        tag = 'warmup' if repeat < 0 else 'repeat %d' % repeat
        print('=== %s / %s ===' % (variant, tag), flush=True)
        env = dict(env_base, A100_AB_ROLE='run', A100_AB_VARIANT=variant,
                   A100_AB_REPEAT=str(repeat))
        if variant == 'xla':
            env['MBIRJAX_DISABLE_PALLAS'] = '1'
        else:
            env.pop('MBIRJAX_DISABLE_PALLAS', None)
        out = subprocess.run([sys.executable, '-u', __file__], env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True)
        line = None
        for ln in out.stdout.splitlines():
            if ln.startswith('CELL_RESULT '):
                line = ln[len('CELL_RESULT '):]
            else:
                print('   | ' + ln)
        if out.returncode != 0 or line is None:
            print('   CELL FAILED (rc=%d)' % out.returncode)
            continue
        rec = json.loads(line)
        rec['warmup'] = repeat < 0
        results.append(rec)
        print('   -> %.2f s   peak %.2f GB   fwd_pallas=%s back_pallas=%s'
              % (rec['elapsed_s'], rec['peak_bytes'] / 1e9,
                 rec['fwd_pallas'], rec['back_pallas']), flush=True)
        cells.append(rec)

    with open(RESULTS, 'w') as f:
        json.dump(results, f, indent=2)
    _report(results)
    return 0


def _report(results):
    timed = [r for r in results if not r['warmup']]
    print('\n' + '=' * 72)
    print('A100 BASELINE A/B -- BGA cone recon, %d timed repeats per variant'
          % REPEATS)
    print('=' * 72)
    if not timed:
        print('no timed cells succeeded')
        return
    r0 = timed[0]
    print('device      : %s' % r0['device_kind'])
    print('sinogram    : %s   recon: %s' % (tuple(r0['sino_shape']),
                                            tuple(r0['recon_shape'])))
    print('availability: %s' % r0['pallas_why'])
    print()
    print('%-8s %-6s %10s %10s %12s %12s' % ('variant', 'n', 'time_s(min)',
                                             'time_s(med)', 'peak_GB(min)',
                                             'peak_GB(max)'))
    summary = {}
    for v in VARIANTS:
        rs = [r for r in timed if r['variant'] == v]
        if not rs:
            continue
        ts = sorted(r['elapsed_s'] for r in rs)
        ps = sorted(r['peak_bytes'] / 1e9 for r in rs)
        med = ts[len(ts) // 2]
        iters = sorted({r.get('num_iterations') for r in rs})
        summary[v] = {'t_min': ts[0], 't_med': med, 'p_min': ps[0], 'p_max': ps[-1],
                      'spread': (ts[-1] - ts[0]) / ts[0] if ts[0] else 0.0,
                      'flags': (rs[0]['fwd_pallas'], rs[0]['back_pallas']),
                      'iters': iters,
                      'fm_rmse': rs[0].get('final_fm_rmse')}
        print('%-8s %-6d %10.2f %10.2f %12.3f %12.3f'
              % (v, len(rs), ts[0], med, ps[0], ps[-1]))
        print('         (run-to-run time spread %.1f%%; fwd_pallas=%s back_pallas=%s; '
              'iterations %s; final fm_rmse %s)'
              % (100 * summary[v]['spread'], *summary[v]['flags'],
                 summary[v]['iters'], summary[v]['fm_rmse']))

    if 'pallas' in summary and 'xla' in summary:
        p, x = summary['pallas'], summary['xla']
        # Guard against the silent-no-op case: if the flags are identical the env var
        # did not do what it was meant to and the "A/B" compared a variant with itself.
        if p['flags'] == x['flags']:
            print('\n!! WARNING: both variants ran with the SAME pallas flags %s --'
                  % (p['flags'],))
            print('   MBIRJAX_DISABLE_PALLAS did not take effect; this is NOT an A/B.')
            return
        # Control: equal work.  Different iteration counts mean the two variants did
        # different amounts of work, and the time difference is then partly (or wholly)
        # a convergence effect, not a kernel effect.
        # With STOP_THRESHOLD_CHANGE_PCT = 0 the counts should be identical by
        # construction; this is the verification that the setting took effect, not the
        # primary control.
        equal_work = p['iters'] == x['iters']
        if not equal_work:
            print('\n!! CAUTION: iteration counts DIFFER -- pallas %s vs xla %s, '
                  'despite stop_threshold_change_pct=%s.'
                  % (p['iters'], x['iters'], STOP_THRESHOLD_CHANGE_PCT))
            print('   The runs are NOT equal-work, so the wall times below cannot be')
            print('   attributed to the kernels.  Investigate before trusting them.')

        dt = (x['t_med'] - p['t_med']) / x['t_med']      # + = pallas FASTER
        dm = (p['p_max'] - x['p_max']) / x['p_max']      # + = pallas uses MORE memory
        print('\ntime  : pallas is %+.1f%% vs XLA  (%.1f s -> %.1f s)'
              % (-100 * dt, x['t_med'], p['t_med']))
        print('memory: pallas is %+.1f%% vs XLA  (%.2f GB -> %.2f GB)'
              % (100 * dm, x['p_max'], p['p_max']))
        print('        run-to-run time spread: pallas %.1f%%, xla %.1f%% '
              '(an effect below this is noise)'
              % (100 * p['spread'], 100 * x['spread']))
        # Report the PARETO relationship, not a pass/fail.  With two points there are
        # only three cases: one dominates, the other dominates, or both are on the
        # frontier and the choice is a judgement call informed by the slope.
        prefix = 'PARETO' if equal_work else 'PARETO (PROVISIONAL -- UNEQUAL WORK)'
        faster, leaner = dt > 0, dm < 0
        if faster and leaner:
            print('%s: pallas DOMINATES -- faster AND smaller peak. No trade to weigh.'
                  % prefix)
        elif (not faster) and (not leaner):
            print('%s: XLA DOMINATES -- pallas is both slower and larger here.'
                  % prefix)
        else:
            # Non-dominated pair: state the trade in both directions and locate it
            # relative to the indifference slope WITHOUT calling a winner.
            ratio = abs(dt) / max(abs(dm), 1e-9)
            side = ('better than' if ratio > INDIFFERENCE_SLOPE else
                    'worse than' if ratio < INDIFFERENCE_SLOPE else 'at')
            print('%s: BOTH POINTS ARE ON THE FRONTIER -- neither dominates.' % prefix)
            if faster:
                print('   pallas buys %.1f%% less time for %.1f%% more peak memory.'
                      % (100 * dt, 100 * dm))
            else:
                print('   pallas buys %.1f%% less peak memory for %.1f%% more time.'
                      % (-100 * dm, -100 * dt))
            print('   Trade ratio %.1fx, %s the ~%.1fx slope Greg called break-even'
                  % (ratio, side, INDIFFERENCE_SLOPE))
            print('   -- reported for judgement, not auto-accepted or rejected.')


if __name__ == '__main__':
    if _ROLE == 'prep':
        prep()
    elif _ROLE == 'run':
        run_cell()
    else:
        sys.exit(orchestrate())
