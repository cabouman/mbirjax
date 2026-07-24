"""Step 3 of the A100 tuning campaign: does the view-chunk MEMORY saving compose?

The step-2 cone sweep found the only actionable lever on the memory axis, and it is not
a Pallas constant: the effective view chunk (``TilePolicy.back_view_batch``, which the
pallas back driver takes as min(back_view_batch, BACK_VIEW_CHUNK_CAP, num_views)).
Measured on isolated back-projection calls at sinogram (752, 720, 688):

    view chunk 128 (shipped) : 0.0253 s subset / 2.81 GB ; 2.076 s full grid / 7.06 GB
    view chunk  64           : +2.0% time, -19% peak (subset) ; +1.0% time, -9.9% (full)
    view chunk  32           : +6.3% time, -33% peak (subset)

Against Greg's stated indifference slope (time worth roughly 3x memory) those are ~10x
ratios -- attractive.  But a KERNEL-level peak is not a RECON-level peak: the recon's
high-water mark is set by the persistent set (sinogram, weights, error sinogram, recon,
Hessian) TOGETHER with the largest co-live transient, and shrinking one transient only
moves the peak if that transient is what set it.  This script measures the composed
result instead of inferring it -- the project has a documented history of driver-level
wins failing to compose end to end (plans/projector_batching/, twice).

Method: full VCD recons at the step-2 shape, one ISOLATED SUBPROCESS per cell (peak
memory is a process-cumulative high-water mark), interleaved variants, a discarded
warmup, and stop_threshold_change_pct=0 so every variant does EXACTLY the same work.
Reports the (time, peak) Pareto relationship -- both axes, no single-objective verdict.

Run:  sbatch a100_viewchunk_recon_ab.slurm
"""
import json
import os
import subprocess
import sys
import time

# ── Config ────────────────────────────────────────────────────────────────────
SINO_SHAPE = (752, 720, 688)        # the step-2 sweep cell
CONE_SDD_OVER_CHANNELS = 4.0
# 'cone' or 'parallel'.  The parallel sweep found the view chunk DOMINATES at both
# operating points (subset -5.9% time / -19% peak; full grid -19.7% time / -5.3% peak),
# a much stronger effect than cone's, so both geometries get a composition check.
GEOMETRY = os.environ.get('A100_VC_GEOMETRY', 'cone')
VIEW_CHUNKS = (128, 64, 32)         # 128 = shipped (_BACK_VIEW_CAP_SINGLE)
MAX_ITERATIONS = 8                  # enough to reach steady state; the peak is set early
STOP_THRESHOLD_CHANGE_PCT = 0.0     # exactly MAX_ITERATIONS for every variant
REPEATS = 2
SHARPNESS, SNR_DB = 1.0, 30.0
OUT_DIR = '/scratch/gilbreth/buzzard/a100_tuning'
RESULTS = os.path.join(OUT_DIR, 'results',
                       'viewchunk_recon_ab_%s.json'
                       % os.environ.get('A100_VC_GEOMETRY', 'cone'))

_ROLE = os.environ.get('A100_VC_ROLE', 'orchestrate')


def run_cell():
    import numpy as np
    import jax
    import mbirjax

    vc = int(os.environ['A100_VC_CHUNK'])
    n_views, n_rows, n_channels = SINO_SHAPE
    angles = np.linspace(0, np.pi, n_views, endpoint=False)
    sdd = CONE_SDD_OVER_CHANNELS * n_channels
    if GEOMETRY == 'parallel':
        model = mbirjax.ParallelBeamModel(SINO_SHAPE, angles)
    else:
        model = mbirjax.ConeBeamModel(SINO_SHAPE, angles, source_detector_dist=sdd,
                                      source_iso_dist=sdd / 2.0)
    model.configure_devices(1)
    model.set_params(sharpness=SHARPNESS, snr_db=SNR_DB)
    # The knob under test.  Set AFTER configure_devices: _set_device_layout re-runs
    # _select_tile_policy and would overwrite it.
    model.tiles = model.tiles._replace(back_view_batch=vc)

    rng = np.random.default_rng(0)
    sino = rng.random(SINO_SHAPE, dtype=np.float32)      # positive, like a real log-sino
    t0 = time.time()
    recon, recon_dict = model.recon(sino, max_iterations=MAX_ITERATIONS,
                                    stop_threshold_change_pct=STOP_THRESHOLD_CHANGE_PCT)
    if hasattr(recon, 'block_until_ready'):
        recon.block_until_ready()
    elapsed = time.time() - t0

    st = jax.devices()[0].memory_stats() or {}
    rp = recon_dict.get('recon_params')
    iters = rp.get('num_iterations') if isinstance(rp, dict) else None
    print('VC_RESULT ' + json.dumps({
        'view_chunk': vc,
        'effective_view_chunk': int(model.tiles.back_view_batch),
        'elapsed_s': elapsed,
        'peak_GB': int(st.get('peak_bytes_in_use', 0)) / 1e9,
        'num_iterations': iters,
        'recon_shape': list(model.get_params('recon_shape')),
        'geometry': GEOMETRY,
        'back_pallas': bool(model.tiles.back_pallas),
        'fwd_pallas': bool(model.tiles.fwd_pallas),
        'repeat': int(os.environ.get('A100_VC_REPEAT', -1)),
    }))


def orchestrate():
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    order = [(vc, -1) for vc in VIEW_CHUNKS]                    # discarded warmups
    for r in range(REPEATS):
        order += [(vc, r) for vc in VIEW_CHUNKS]                # interleaved
    results = []
    for vc, rep in order:
        env = dict(os.environ, A100_VC_ROLE='cell', A100_VC_CHUNK=str(vc),
                   A100_VC_REPEAT=str(rep))
        print('=== view_chunk %d / %s ===' % (vc, 'warmup' if rep < 0 else 'rep %d' % rep),
              flush=True)
        out = subprocess.run([sys.executable, '-u', __file__], env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        rec = None
        for ln in out.stdout.splitlines():
            if ln.startswith('VC_RESULT '):
                rec = json.loads(ln[len('VC_RESULT '):])
        if rec is None:
            print('   FAILED (rc=%d); tail:' % out.returncode)
            for ln in out.stdout.splitlines()[-15:]:
                print('     | ' + ln)
            continue
        rec['warmup'] = rep < 0
        results.append(rec)
        print('   -> %.1f s   peak %.2f GB   iters=%s  back_pallas=%s'
              % (rec['elapsed_s'], rec['peak_GB'], rec['num_iterations'],
                 rec['back_pallas']), flush=True)

    with open(RESULTS, 'w') as f:
        json.dump(results, f, indent=2)

    timed = [r for r in results if not r['warmup']]
    if not timed:
        print('no timed cells succeeded')
        return 1
    print('\n' + '=' * 74)
    print('END-TO-END view-chunk A/B -- %s VCD, sino %s, recon %s, %d iterations'
          % (GEOMETRY, SINO_SHAPE, tuple(timed[0]['recon_shape']), MAX_ITERATIONS))
    print('=' * 74)
    print('%-12s %-4s %12s %12s %10s' % ('view_chunk', 'n', 'time_s(min)', 'peak_GB(min)',
                                         'iters'))
    summary = {}
    for vc in VIEW_CHUNKS:
        rs = [r for r in timed if r['view_chunk'] == vc]
        if not rs:
            continue
        t = min(r['elapsed_s'] for r in rs)
        m = min(r['peak_GB'] for r in rs)
        summary[vc] = (t, m)
        print('%-12d %-4d %12.1f %12.2f %10s'
              % (vc, len(rs), t, m, {r['num_iterations'] for r in rs}))
    base = summary.get(VIEW_CHUNKS[0])
    if base:
        bt, bm = base
        print('\nrelative to the shipped view chunk %d:' % VIEW_CHUNKS[0])
        for vc in VIEW_CHUNKS[1:]:
            if vc not in summary:
                continue
            t, m = summary[vc]
            dt, dm = (t - bt) / bt, (m - bm) / bm
            note = ('DOMINATES (faster and leaner)' if dt <= 0 and dm <= 0 else
                    'dominated (slower and larger)' if dt >= 0 and dm >= 0 else
                    'trade: %+.1f%% time for %+.1f%% memory (ratio %.1fx)'
                    % (100 * dt, 100 * dm, abs(dm) / max(abs(dt), 1e-9)))
            print('   vc=%-4d %+6.1f%% time  %+6.1f%% peak   %s'
                  % (vc, 100 * dt, 100 * dm, note))
        print('\nThe KERNEL-level saving was -9.9%% (full grid) / -19%% (subset) peak.')
        print('Compare with the end-to-end numbers above: if the recon peak barely moves,')
        print('the projector transient is NOT what sets it and the lever is not useful')
        print('at this size -- which is the thing this script exists to find out.')
    return 0


if __name__ == '__main__':
    if _ROLE == 'cell':
        run_cell()
    else:
        sys.exit(orchestrate())
