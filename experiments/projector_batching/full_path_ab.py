"""Full-recon-path A/B of projector batching v1 vs v2 at scale (the default-flip gate).

Runs the SAME cone-beam VCD recon twice -- once per batching version -- each in a fresh
subprocess (peak_bytes_in_use is a process-cumulative high-water mark, so honest per-config
memory needs one process per config with a JAX-free orchestrator; lessons.md section 5).
Reports per-device peak memory, recon wall time, and the rel-max difference between the two
recons (gate: fp equivalence at 1e-4, the iterated-VCD calibration).

This is also the measurement that reconciles the historical full-path transient (~18.8 GiB
at 1024^3 / view_batch 512; a [vb x pixel_batch x slices x footprint~5] kernel intermediate
per Greg's reconstruction) with the isolated-driver k ~= 1.3 -- see
projector_batching_characterization.md section 4.

Run on the cluster (all GPUs, ~tens of minutes at 1024^3):

    python full_path_ab.py

Report back the FINAL SUMMARY block.  For a laptop smoke test, lower N / NUM_VIEWS /
MAX_ITERATIONS in the config below.
"""

import os
import subprocess
import sys
import time

# ----------------------------------------------------------------------------------
# Config -- edit here
# ----------------------------------------------------------------------------------
N = 1024                    # recon N^3; detector N x N
NUM_VIEWS = 1024
MAX_ITERATIONS = 5
SEED = 0                    # np.random.seed before recon: partitions come from the global
                            # RNG, and an unseeded A/B compares partition noise, not code
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ab_output')
KEEP_RECONS = False         # True to keep the two recon .npy files (N^3 * 4 bytes each)

_ROLE_ENV = 'MBIRJAX_AB_ROLE'
_VERSION_ENV = 'MBIRJAX_PROJECTOR_BATCHING_VERSION'   # read by TomographyModel.__init__


def worker():
    """One measured recon in this process; version comes from the env (set by the parent)."""
    import numpy as np
    import mbirjax                                    # must precede jax (env binding)
    import jax

    version = os.environ[_VERSION_ENV]
    angles = np.linspace(0, np.pi, NUM_VIEWS, endpoint=False)
    model = mbirjax.ConeBeamModel((NUM_VIEWS, N, N), angles,
                                  source_detector_dist=4.0 * N, source_iso_dist=2.0 * N)
    recon_shape = model.get_params('recon_shape')

    phantom = mbirjax.gen_cube_phantom(recon_shape)
    sinogram = np.asarray(model.forward_project(phantom))
    del phantom

    np.random.seed(SEED)                              # reproducible partitions
    t0 = time.perf_counter()
    recon, _ = model.recon(sinogram, max_iterations=MAX_ITERATIONS, print_logs=False)
    recon = np.asarray(recon)
    elapsed = time.perf_counter() - t0

    np.save(os.path.join(OUT_DIR, f'recon_v{version}.npy'), recon)

    # Per-device peak memory (only devices with a backend memory-stats implementation).
    peaks = []
    for dev in jax.local_devices():
        stats = dev.memory_stats()
        if stats and 'peak_bytes_in_use' in stats:
            peaks.append(stats['peak_bytes_in_use'] / 2**30)
    print(f'WORKER v{version}: recon {elapsed:.1f} s; peak GiB/device: '
          + (', '.join(f'{p:.2f}' for p in peaks) if peaks else 'n/a (no memory stats)'))


def orchestrator():
    """JAX-free parent: one subprocess per version, then compare the saved recons."""
    import numpy as np

    os.makedirs(OUT_DIR, exist_ok=True)
    lines = {}
    for version in ('1', '2'):
        env = dict(os.environ, **{_ROLE_ENV: 'worker', _VERSION_ENV: version})
        print(f'--- running batching v{version} ({N}^3, {NUM_VIEWS} views, '
              f'{MAX_ITERATIONS} iters) ---')
        result = subprocess.run([sys.executable, os.path.abspath(__file__)], env=env,
                                capture_output=True, text=True)
        sys.stdout.write(result.stdout)
        if result.returncode != 0:
            # Show the full traceback -- an OOM can surface as an unrelated-looking error.
            sys.stderr.write(result.stderr)
            raise RuntimeError(f'v{version} worker failed (exit {result.returncode})')
        lines[version] = [ln for ln in result.stdout.splitlines() if ln.startswith('WORKER')]

    r1 = np.load(os.path.join(OUT_DIR, 'recon_v1.npy'), mmap_mode='r')
    r2 = np.load(os.path.join(OUT_DIR, 'recon_v2.npy'), mmap_mode='r')
    # rel-max in slabs so the 2 x N^3 arrays never fully materialize in RAM.
    max_diff, max_ref = 0.0, 0.0
    for s in range(0, r1.shape[0], 64):
        a, b = np.asarray(r1[s:s + 64], dtype=np.float64), np.asarray(r2[s:s + 64],
                                                                      dtype=np.float64)
        max_diff = max(max_diff, float(np.max(np.abs(a - b))))
        max_ref = max(max_ref, float(np.max(np.abs(a))))
    rel = max_diff / max_ref if max_ref else max_diff

    print('\n=== FINAL SUMMARY ===')
    for version in ('1', '2'):
        for ln in lines[version]:
            print(ln)
    print(f'rel_max_err(v2 vs v1) = {rel:.2e}   (gate: 1e-4, iterated-VCD calibration)')
    print('PASS' if rel <= 1e-4 else 'FAIL')

    if not KEEP_RECONS:
        for version in ('1', '2'):
            os.remove(os.path.join(OUT_DIR, f'recon_v{version}.npy'))


if __name__ == '__main__':
    if os.environ.get(_ROLE_ENV) == 'worker':
        worker()
    else:
        orchestrator()
