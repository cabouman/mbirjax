"""Cold-compile-time attribution at demo-1 scale (1024^3), triggered by Greg's 2m19s
`_jit_compute_scatter_centers` compile alarm on a first demo run (2026-07-13).

Question: where does a COLD first-run XLA GPU compile spend its time at production
shapes, and which single variable moves it -- autotuning, compile-CPU starvation, or
geometry?  Each cell is an isolated subprocess with a FRESH persistent-cache dir on
scratch (a true cold start; the library's auto-configured cache is what makes second
runs fast).  MBIRJAX_DISABLE_PALLAS=1 everywhere so the measured path is the XLA one
(identical to kernel_investigation, where the alarm was seen).

Cells (single-variable versus cone_default):
  cone_default    demo-1 geometry/shape, stock flags      -- reproduces the observation
  cone_autotune0  + XLA_FLAGS=--xla_gpu_autotune_level=0  -- is it autotuning?
  cone_2cpu       + taskset -c 0,1                        -- is it compile parallelism?
  par_default     ParallelBeamModel, stock flags          -- is it cone-specific?
  cone_warm       cone_default's cache dir, NEW process   -- the cache-hit control

Output: per-step walls (import, model build, forward, back, second forward) plus
timestamped jax compile-log lines and XLA slow-op alarms (stderr), one log per cell in
OUT_DIR.  Run via compile_attribution.slurm; smoke-test locally on CPU with
COMPILE_ATTR_SMOKE=1 (shrinks shapes, uses a temp cache).
"""
import os
import shutil
import subprocess
import sys
import time

# ── Run parameters (edit here; no CLI args) ───────────────────────────────────
SINO_SHAPE = (1024, 1024, 1024)            # (views, det rows, det channels) -- Greg's demo edit
OUT_DIR = '/scratch/gautschi/buzzard/compile_attr'
CELLS = [
    # (name, geometry, xla_flags, cpu_pin, cache_from)
    ('cone_default',   'cone',     '',                              None, None),
    ('cone_autotune0', 'cone',     '--xla_gpu_autotune_level=0',    None, None),
    ('cone_2cpu',      'cone',     '',                              '0,1', None),
    ('par_default',    'parallel', '',                              None, None),
    ('cone_warm',      'cone',     '',                              None, 'cone_default'),
]

if os.environ.get('COMPILE_ATTR_SMOKE') == '1':
    SINO_SHAPE = (16, 12, 16)
    OUT_DIR = os.path.join(os.environ.get('TMPDIR', '/tmp'), 'compile_attr_smoke')


def worker():
    """One cell: build the model, run the demo-1 projector calls, log every compile."""
    t0 = time.perf_counter()

    def note(msg):
        print(f'[{time.perf_counter() - t0:8.2f}s] {msg}', flush=True)

    import logging

    class TimedJaxLog(logging.Handler):
        # jax_log_compiles emits tracing/compilation lines through the 'jax' logger;
        # stamping them onto the shared clock interleaves them with the step walls.
        def emit(self, record):
            msg = record.getMessage()
            if 'ompil' in msg or 'racing' in msg:                  # Compil/compil/Tracing
                note(f'jaxlog: {msg[:300]}')

    logging.getLogger('jax').addHandler(TimedJaxLog())
    logging.getLogger('jax').setLevel(logging.DEBUG)

    import jax
    jax.config.update('jax_log_compiles', True)
    import jax.numpy as jnp
    import numpy as np
    import mbirjax
    affinity = (len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity')
                else os.cpu_count())                      # macOS smoke has no affinity API
    note(f'imports done; jax {jax.__version__}; devices {jax.devices()}; '
         f'ncpu={os.cpu_count()} affinity={affinity}; '
         f'cache_dir={jax.config.jax_compilation_cache_dir}; '
         f'XLA_FLAGS={os.environ.get("XLA_FLAGS", "")!r}')

    geometry = os.environ['COMPILE_ATTR_GEOM']
    views, rows, channels = SINO_SHAPE
    angles = np.linspace(-np.pi / 2, np.pi / 2, views, endpoint=False)
    if geometry == 'cone':
        model = mbirjax.ConeBeamModel(SINO_SHAPE, angles,
                                      source_detector_dist=4 * channels,
                                      source_iso_dist=4 * channels)
    else:
        model = mbirjax.ParallelBeamModel(SINO_SHAPE, angles)
    model.configure_devices(1)
    recon_shape = model.get_params('recon_shape')
    note(f'model built: {geometry} recon_shape={recon_shape}')

    phantom = jnp.ones(recon_shape, dtype=jnp.float32)   # values irrelevant to compile
    jax.block_until_ready(phantom)
    note('phantom ready')

    t = time.perf_counter()
    sino = jax.block_until_ready(model.forward_project(phantom))
    note(f'STEP forward_project (cold): {time.perf_counter() - t:.2f}s')

    t = time.perf_counter()
    jax.block_until_ready(model.forward_project(phantom))
    note(f'STEP forward_project (warm, in-process): {time.perf_counter() - t:.2f}s')

    t = time.perf_counter()
    jax.block_until_ready(model.back_project(sino))
    note(f'STEP back_project (cold): {time.perf_counter() - t:.2f}s')
    note('cell done')


def orchestrator():
    """JAX-free parent: fresh cache dir per cold cell, one subprocess per cell."""
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, geom, xla_flags, cpu_pin, cache_from in CELLS:
        cache_dir = os.path.join(OUT_DIR, f'cache_{cache_from or name}')
        if cache_from is None:
            shutil.rmtree(cache_dir, ignore_errors=True)      # guarantee COLD
            os.makedirs(cache_dir)
        env = dict(os.environ,
                   COMPILE_ATTR_CELL='1', COMPILE_ATTR_GEOM=geom,
                   JAX_COMPILATION_CACHE_DIR=cache_dir,
                   MBIRJAX_DISABLE_PALLAS='1')
        if xla_flags:
            env['XLA_FLAGS'] = (env.get('XLA_FLAGS', '') + ' ' + xla_flags).strip()
        cmd = [sys.executable, os.path.abspath(__file__)]
        if cpu_pin:
            if shutil.which('taskset') is None:           # macOS smoke: no taskset
                print(f'[{name}] SKIPPED (no taskset on this platform)', flush=True)
                continue
            cmd = ['taskset', '-c', cpu_pin] + cmd
        log_path = os.path.join(OUT_DIR, f'{name}.log')
        t = time.perf_counter()
        with open(log_path, 'w') as log:
            rc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        wall = time.perf_counter() - t
        print(f'[{name}] rc={rc} total_wall={wall:.1f}s log={log_path}', flush=True)
        with open(log_path) as log:                          # echo the step lines
            for line in log:
                if 'STEP ' in line or 'slow_operation' in line or 'Very slow' in line:
                    print(f'    {line.rstrip()}', flush=True)
    print('=== compile_attr done ===', flush=True)


if __name__ == '__main__':
    worker() if os.environ.get('COMPILE_ATTR_CELL') else orchestrator()
