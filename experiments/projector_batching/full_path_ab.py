"""Full-recon-path A/B of the projector batching across CODE STATES, at scale.

The package carries a single batching implementation, so old-vs-new full-path comparison is
TWO RUNS of this script at two git states, compared through a saved baseline recon:

  1. At the baseline commit: set KEEP_RECONS = True, run, note the saved
     ab_output/recon_<label>.npy.
  2. At the new code: set COMPARE_TO to that file's path and run again.

Each case runs in a fresh subprocess (peak_bytes_in_use is a process-cumulative high-water
mark, so honest per-config memory needs one process per config with a JAX-free
orchestrator; lessons.md section 5).  Reports per-device peak memory, recon wall time,
NRMSE vs the known phantom (the quality anchor: two recons that differ from each other but
sit equidistant from truth are both fine), and rel-max between recons (gate: 1e-4, the
iterated-VCD calibration -- see the null-calibration note at CASES).

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
PARTITION_SEQUENCE = None   # None = model default [0,2,4,6,7].  Pin to a single granularity
                            # level to LOCALIZE a timing delta: e.g. [0] = all-coarse
                            # iterations (823k-pixel subsets, short bands), [4] = all
                            # granularity-16, [7] = all-fine (6.4k).
# Cases: (label, view_batch_size, back_pixel_batch_size) -- None = the model default.
# Every case after the first is compared against the first.  A single case is the normal
# cross-code-state mode (compare via COMPARE_TO).  Two cases with different view batches is
# the NULL CALIBRATION of the fp gate (a pure sum-REGROUPING perturbation with the code
# fixed), e.g. [('vb128', None, None), ('vb121', 121, None)].  Two cases with different
# back pixel widths A/Bs the band-kernel width sensitivity (2026-07-04 verdict: driver-level
# width wins did NOT survive the full path on either platform; 2048 stays).
CASES = [
    ('recon', None, None),
]
COMPARE_TO = None           # path to a baseline recon .npy from a run at another code
                            # state; the FIRST case is compared against it
SEED = 0                    # np.random.seed before recon: partitions come from the global
                            # RNG, and an unseeded A/B compares partition noise, not code
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ab_output')
KEEP_RECONS = False         # True to keep the recon .npy files (N^3 * 4 bytes each)

_ROLE_ENV = 'MBIRJAX_AB_ROLE'
_LABEL_ENV = 'MBIRJAX_AB_LABEL'
_VIEW_BATCH_ENV = 'MBIRJAX_AB_VIEW_BATCH'
_PIXEL_BACK_ENV = 'MBIRJAX_AB_PIXEL_BATCH_BACK'


def worker():
    """One measured recon in this process; case config comes from the env (set by the parent)."""
    import numpy as np
    import mbirjax                                    # must precede jax (env binding)
    import jax

    label = os.environ[_LABEL_ENV]
    angles = np.linspace(0, np.pi, NUM_VIEWS, endpoint=False)
    model = mbirjax.ConeBeamModel((NUM_VIEWS, N, N), angles,
                                  source_detector_dist=4.0 * N, source_iso_dist=2.0 * N)
    if PARTITION_SEQUENCE is not None:
        model.set_params(partition_sequence=PARTITION_SEQUENCE)
    view_batch = os.environ.get(_VIEW_BATCH_ENV)
    pixel_back = os.environ.get(_PIXEL_BACK_ENV)
    if view_batch or pixel_back:
        # The projector entry points capture the batch sizes when built, so a post-__init__
        # override must rebuild them.
        if view_batch:
            model.view_batch_size_for_vmap = int(view_batch)
        if pixel_back:
            model.pixel_batch_size_for_vmap_back = int(pixel_back)
        model.create_projectors()
    recon_shape = model.get_params('recon_shape')

    phantom = mbirjax.gen_cube_phantom(recon_shape)
    sinogram = np.asarray(model.forward_project(phantom))
    # Move the phantom to HOST before the recon: keeping the jax array alive parks a full
    # recon-sized buffer on device 0 for the whole run (+4 GiB at 1024^3, observed as an
    # inflated device-0 peak in the 2026-07-04 pixel-width A/B).
    phantom = np.asarray(phantom, dtype=np.float32)

    np.random.seed(SEED)                              # reproducible partitions
    t0 = time.perf_counter()
    recon, _ = model.recon(sinogram, max_iterations=MAX_ITERATIONS, print_logs=False)
    recon = np.asarray(recon)
    elapsed = time.perf_counter() - t0

    np.save(os.path.join(OUT_DIR, f'recon_{label}.npy'), recon)

    # Quality anchor: distance to the KNOWN phantom.  If two cases differ from each other
    # but are equidistant from the truth, neither is degraded.
    phantom = np.asarray(phantom, dtype=np.float32)
    nrmse = float(np.linalg.norm(recon - phantom) / np.linalg.norm(phantom))

    # Per-device peak memory (only devices with a backend memory-stats implementation).
    peaks = []
    for dev in jax.local_devices():
        stats = dev.memory_stats()
        if stats and 'peak_bytes_in_use' in stats:
            peaks.append(stats['peak_bytes_in_use'] / 2**30)
    print(f'WORKER {label}: recon {elapsed:.1f} s; nrmse vs phantom {nrmse:.6f}; '
          f'peak GiB/device: '
          + (', '.join(f'{p:.2f}' for p in peaks) if peaks else 'n/a (no memory stats)'))


def slab_rel_max(path_a, path_b):
    """rel-max between two saved recons, in slabs so 2 x N^3 never materializes in RAM."""
    import numpy as np
    ra = np.load(path_a, mmap_mode='r')
    rb = np.load(path_b, mmap_mode='r')
    max_diff, max_ref = 0.0, 0.0
    for s in range(0, ra.shape[0], 64):
        a = np.asarray(ra[s:s + 64], dtype=np.float64)
        b = np.asarray(rb[s:s + 64], dtype=np.float64)
        max_diff = max(max_diff, float(np.max(np.abs(a - b))))
        max_ref = max(max_ref, float(np.max(np.abs(a))))
    return max_diff / max_ref if max_ref else max_diff


def orchestrator():
    """JAX-free parent: one subprocess per case, then the configured comparisons."""
    os.makedirs(OUT_DIR, exist_ok=True)
    lines = {}
    for label, view_batch, pixel_back in CASES:
        env = dict(os.environ, **{_ROLE_ENV: 'worker', _LABEL_ENV: label})
        if view_batch is not None:
            env[_VIEW_BATCH_ENV] = str(view_batch)
        if pixel_back is not None:
            env[_PIXEL_BACK_ENV] = str(pixel_back)
        seq = 'default' if PARTITION_SEQUENCE is None else PARTITION_SEQUENCE
        print(f'--- running case {label} (view_batch={view_batch or "default"}, '
              f'pixel_batch_back={pixel_back or "default"}, '
              f'{N}^3, {NUM_VIEWS} views, {MAX_ITERATIONS} iters, '
              f'partition_sequence={seq}) ---', flush=True)
        result = subprocess.run([sys.executable, os.path.abspath(__file__)], env=env,
                                capture_output=True, text=True)
        sys.stdout.write(result.stdout)
        if result.returncode != 0:
            # Show the full traceback -- an OOM can surface as an unrelated-looking error.
            sys.stderr.write(result.stderr)
            raise RuntimeError(f'case {label} worker failed (exit {result.returncode})')
        lines[label] = [ln for ln in result.stdout.splitlines() if ln.startswith('WORKER')]

    ref_label = CASES[0][0]
    ref_path = os.path.join(OUT_DIR, f'recon_{ref_label}.npy')
    print('\n=== FINAL SUMMARY ===')
    for label, _, _ in CASES:
        for ln in lines[label]:
            print(ln)
    for label, _, _ in CASES[1:]:
        rel = slab_rel_max(ref_path, os.path.join(OUT_DIR, f'recon_{label}.npy'))
        verdict = 'PASS' if rel <= 1e-4 else 'FAIL'
        print(f'rel_max_err({label} vs {ref_label}) = {rel:.2e}   '
              f'(gate: 1e-4, iterated-VCD calibration)   {verdict}')
    if COMPARE_TO is not None:
        rel = slab_rel_max(COMPARE_TO, ref_path)
        verdict = 'PASS' if rel <= 1e-4 else 'FAIL'
        print(f'rel_max_err({ref_label} vs baseline {os.path.basename(COMPARE_TO)}) = '
              f'{rel:.2e}   (gate: 1e-4, iterated-VCD calibration)   {verdict}')

    if not KEEP_RECONS:
        for label, _, _ in CASES:
            os.remove(os.path.join(OUT_DIR, f'recon_{label}.npy'))


if __name__ == '__main__':
    if os.environ.get(_ROLE_ENV) == 'worker':
        worker()
    else:
        orchestrator()
