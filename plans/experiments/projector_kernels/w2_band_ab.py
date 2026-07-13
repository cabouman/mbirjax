"""Wave-2 band A/B (2026-07-13): the A5 knob sweep + the parallel band-adoption probe.

Context (gpu_headroom_findings.md, jobs 13497819/13497836): cone banded back at n>=2
is dominated by XLA materializing (pixel_batch, L, VIEW_BATCH) per-view-partial
stacks before the view reduction; at current defaults the whole 512-view shard goes
through ONE vmap batch (~480 MB stacks, view-last transposes).  But the banded driver
(_sparse_back_project_band) already accumulates across view batches -- so the A5
restructure may be a pure TilePolicy knob: shrink back_view_batch for the banded call
and the materialization shrinks proportionally.  E2a found the default fastest for
the n=1 PIXEL path; the banded path's economics are different, hence this sweep.

All cells are SINGLE-DEVICE replications of one view-owner's per-band work at the
1024^3 n=2 shard shape (512 views, 1024x1024 detector, L=115 slice bands over the
model's actual slice count -- cone auto-extends to 1152 slices, so 11 band calls
including a cropped tail; a uniform ~1.1x on absolute walls vs production's 10, with
sweep RATIOS unaffected) -- the band-kernel cost is per-device, so no multi-GPU
allocation is needed to gate the kernel.  Both parallel cells pin
back_view_batch=SHARD_VIEWS (the n>=2 sharded policy value; a 1-device layout would
otherwise select the n=1 cap of 128) and the XLA cell times
projector_functions.sparse_back_project directly -- the EXACT call the production
per-owner override makes (parallel_beam._back_project_view_shard_to_band); the
model-level model.sparse_back_project would route through the n=1 transfer-batched
driver, a structurally different program (review finding, wf_2a3f9c18).

Cells (isolated subprocesses):
  cone_vb{512,256,128,64,32,16}: full 9-band sweep through the public
      sparse_back_project_band with back_view_batch replaced; values gated against
      the vb512 reference (identical math, summation order differs).
  par_band_xla / par_band_pallas: the parallel band = a detector-row crop, so a band
      call IS a (512, L, 1024) back projection.  XLA cell runs it with pallas
      disabled (today's n>=2 per-owner path); the pallas cell routes the SAME call
      through the shipped register-tile kernel -- the adoption probe for dispatching
      existing kernels on the multi-device parallel band path.

Outputs: per-cell walls (median of TRIALS after WARMUP), peak GPU memory, value
gates.  Results land in the job log; reference arrays stay on scratch.
"""
import json
import os
import subprocess
import sys
import time

# ── Config ────────────────────────────────────────────────────────────────────
SHARD_VIEWS = 512                       # one n=2 view-owner's shard of 1024 views
SINO_ROWS, SINO_CHANNELS = 1024, 1024
BAND_L = 115                            # matches the traced n=2 banding (9 bands)
WARMUP = 1
TRIALS = 3
OUT_DIR = '/scratch/gautschi/buzzard/w2_band_ab'
REL_TOL = 1e-5
CONE_SWEEP = [512, 256, 128, 64, 32, 16]
CELLS = ([dict(name=f'cone_vb{b}', kind='cone', view_batch=b) for b in CONE_SWEEP]
         + [dict(name='par_band_xla', kind='par', pallas=False),
            dict(name='par_band_pallas', kind='par', pallas=True)])

if os.environ.get('W2AB_SMOKE') == '1':
    SHARD_VIEWS, SINO_ROWS, SINO_CHANNELS, BAND_L = 16, 24, 32, 7
    CONE_SWEEP = [16, 4]
    CELLS = ([dict(name=f'cone_vb{b}', kind='cone', view_batch=b) for b in CONE_SWEEP]
             + [dict(name='par_band_xla', kind='par', pallas=False),
                dict(name='par_band_pallas', kind='par', pallas=True)])
    OUT_DIR = os.path.join(os.environ.get('TMPDIR', '/tmp'), 'w2_band_ab_smoke')


def _bands(num_slices):
    return [(g0, min(BAND_L, num_slices - g0)) for g0 in range(0, num_slices, BAND_L)]


def worker(cfg):
    import numpy as np
    import jax
    import jax.numpy as jnp
    import mbirjax

    t0 = time.perf_counter()

    def note(msg):
        print(f'[{time.perf_counter() - t0:8.2f}s] {msg}', flush=True)

    angles = np.linspace(-np.pi / 2, np.pi / 2, SHARD_VIEWS, endpoint=False)
    rng = np.random.default_rng(0)

    if cfg['kind'] == 'cone':
        sino_shape = (SHARD_VIEWS, SINO_ROWS, SINO_CHANNELS)
        model = mbirjax.ConeBeamModel(sino_shape, angles,
                                      source_detector_dist=4 * SINO_CHANNELS,
                                      source_iso_dist=4 * SINO_CHANNELS)
        model.configure_devices(1)
        recon_shape = model.get_params('recon_shape')
        num_slices = recon_shape[2]
        idx = mbirjax.gen_full_indices(recon_shape, use_ror_mask=True)
        sino = jnp.asarray(rng.random(sino_shape, dtype=np.float32))
        jax.block_until_ready(sino)
        model.tiles = model.tiles._replace(back_view_batch=cfg['view_batch'])
        note(f'cone view_batch={cfg["view_batch"]} bands={_bands(num_slices)}')

        def band_sweep():
            outs = []
            for g0, length in _bands(num_slices):
                # Ragged tail: num_band_slices is static, so pad the last band to
                # BAND_L and crop after (one compiled program for all bands).
                out = model.projector_functions.sparse_back_project_band(
                    sino, idx, g0, min(BAND_L, num_slices))
                outs.append(out[:, :length])
            return jnp.concatenate(outs, axis=1)

        result = jax.block_until_ready(band_sweep())          # compile + warm
        for _ in range(WARMUP - 1):
            jax.block_until_ready(band_sweep())
        ts = []
        for _ in range(TRIALS):
            t = time.perf_counter()
            jax.block_until_ready(band_sweep())
            ts.append(time.perf_counter() - t)
        note(f'RESULT wall={sorted(ts)[len(ts) // 2]:.3f}s trials={["%.3f" % x for x in ts]}')
        ref_path = os.path.join(OUT_DIR, 'cone_ref.npy')
        if cfg['view_batch'] == CONE_SWEEP[0]:
            np.save(ref_path, np.asarray(result))
            note('RESULT ref=saved')
        else:
            ref = np.load(ref_path)
            rel = float(np.max(np.abs(np.asarray(result) - ref)) / np.max(np.abs(ref)))
            note(f'RESULT rel={rel:.2e} {"PASS" if rel < REL_TOL else "FAIL"}')
    else:
        # Parallel band: rows == slices, so one band call is a (V, L, C) problem.
        sino_shape = (SHARD_VIEWS, BAND_L, SINO_CHANNELS)
        model = mbirjax.ParallelBeamModel(sino_shape, angles)
        model.configure_devices(1)
        recon_shape = model.get_params('recon_shape')
        idx = mbirjax.gen_full_indices(recon_shape, use_ror_mask=True)
        n_bands = -(-SINO_ROWS // BAND_L)
        sinos = [jnp.asarray(rng.random(sino_shape, dtype=np.float32))
                 for _ in range(n_bands)]
        jax.block_until_ready(sinos)
        use_pallas = cfg['pallas']
        # The n>=2 sharded policy operating point (a 1-device layout selects the n=1
        # cap of 128); both cells, so the A/B compares at the production value.
        model.tiles = model.tiles._replace(back_view_batch=SHARD_VIEWS)
        note(f'parallel band cell pallas={use_pallas} n_bands={n_bands} '
             f'view_batch={model.tiles.back_view_batch} summary={model.device_summary}')

        try:
            from mbirjax import _pallas_kernels
        except ImportError:
            # Local smoke only: the `mbirjax` env's editable install points at the
            # MAIN worktree (no pallas module).  The cluster env resolves the
            # headroom worktree, where this import succeeds.
            print('RESULT skipped=no-pallas-module-in-this-env', flush=True)
            return

        def band_sweep():
            outs = []
            for s in sinos:                          # 9 bands of fresh data
                if use_pallas:
                    outs.append(_pallas_kernels.back_project_single_device(
                        model, s, idx))
                else:
                    # The production per-owner entry point (see module docstring).
                    outs.append(model.projector_functions.sparse_back_project(s, idx))
            return [jax.block_until_ready(o) for o in outs]

        result = band_sweep()                                  # compile + warm
        for _ in range(WARMUP - 1):
            band_sweep()
        ts = []
        for _ in range(TRIALS):
            t = time.perf_counter()
            band_sweep()
            ts.append(time.perf_counter() - t)
        note(f'RESULT wall={sorted(ts)[len(ts) // 2]:.3f}s trials={["%.3f" % x for x in ts]}')
        ref_path = os.path.join(OUT_DIR, 'par_ref.npy')
        stacked = np.stack([np.asarray(o) for o in result])
        if not use_pallas:
            np.save(ref_path, stacked)
            note('RESULT ref=saved')
        else:
            ref = np.load(ref_path)
            rel = float(np.max(np.abs(stacked - ref)) / np.max(np.abs(ref)))
            note(f'RESULT rel={rel:.2e} {"PASS" if rel < REL_TOL else "FAIL"}')

    stats = mbirjax.get_memory_stats(print_results=False)
    peak = max(s['peak_bytes_in_use'] for s in stats) / 2**30 if stats else 0.0
    note(f'RESULT peak_gb={peak:.2f}')


def orchestrator():
    os.makedirs(OUT_DIR, exist_ok=True)
    for ref in ('cone_ref.npy', 'par_ref.npy'):
        # Never gate against a stale reference from a previous (possibly
        # differently-configured) run (review finding).
        try:
            os.remove(os.path.join(OUT_DIR, ref))
        except FileNotFoundError:
            pass
    summary = []
    for cfg in CELLS:
        env = dict(os.environ, W2AB_CELL=json.dumps(cfg), CUDA_VISIBLE_DEVICES='0')
        if cfg['kind'] == 'cone' or not cfg.get('pallas'):
            env['MBIRJAX_DISABLE_PALLAS'] = '1'    # XLA reference paths
        log_path = os.path.join(OUT_DIR, f"{cfg['name']}.log")
        with open(log_path, 'w') as log:
            rc = subprocess.run([sys.executable, os.path.abspath(__file__)],
                                env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        line = f"[{cfg['name']}] rc={rc}"
        with open(log_path) as log:
            for row in log:
                if 'RESULT ' in row:
                    line += '\n    ' + row.strip()
        print(line, flush=True)
        summary.append(line)
    print('===== w2 band A/B summary =====', flush=True)
    print('\n'.join(summary), flush=True)
    print('=== w2_band_ab done ===', flush=True)


if __name__ == '__main__':
    worker(json.loads(os.environ['W2AB_CELL'])) if os.environ.get('W2AB_CELL') \
        else orchestrator()
