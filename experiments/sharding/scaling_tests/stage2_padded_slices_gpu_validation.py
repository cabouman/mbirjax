"""
experiments/sharding/scaling_tests/stage2_padded_slices_gpu_validation.py
─────────────────────────────────────────────────────────────────────────
GPU validation of P5 Step 4 Stage 2 (slice padding) at REALISTIC scale.

The CPU test suite (and the GPU test suite at toy sizes) already gates the
padding machinery's correctness; what it cannot show is the behavior at real
shard sizes: a multi-GPU VCD recon whose SLICE count does not divide the GPU
count, validated against the single-GPU reference, with timing that shows the
padding + masks are in the noise.

What this script does (one process, no CLI args -- edit the config below):

  1. Builds a Shepp-Logan phantom sinogram at (NUM_VIEWS, NUM_SLICES, NUM_CHANNELS)
     with NUM_SLICES chosen to NOT divide the GPU count (1023 = 3*11*31: pads on
     2/4/8 GPUs -> 1024).
  2. MULTI-GPU run first: a bare model (auto-shards across all GPUs by default);
     prints model.device_summary -- expect e.g. '4 x GPU (sharded) (slices padded
     1023->1024)' -- runs recon(), records wall time and per-device
     peak_bytes_in_use.
  3. SINGLE-GPU reference second (configure_devices(1)), same seed/partitions:
     wall time + device-0 peak.  (Order matters: peak_bytes_in_use is process-
     cumulative per device, so the smaller multi-GPU per-device peaks are read
     BEFORE the single-GPU run raises device 0's high-water mark.)
  4. Compares the two recons: NRMSE and max |diff|.  PASS gate: NRMSE <= 1e-3.
     (The default recon path stages qGGMRF halos once per partition pass, which
     differs from per-subset staging only at gen_pixel_partition's few replicated
     pixels -- a documented ~6e-5..3e-4 NRMSE effect, far below the gate.  The
     padding itself contributes NOTHING: padded slices are exactly inert.)
  5. Informational: host RSS before/after the sinogram entry placement, to spot
     any padded host copy (prepare_sino_for_devices streams shard-by-shard; the
     row+slice padding is created on-device, so RSS should grow by ~0, not by a
     sinogram).

Pre-flight: nvidia-smi (check the GPUs are idle and not thermally throttled);
fresh `pip install -e .` if the checkout changed (a stale build once impersonated
a leak).

Expected runtime at the default 1024 x 1023 x 1024, 10 iterations on H100s:
roughly 3-6 min for the multi-GPU run (4 GPUs) plus ~10-15 min single-GPU.

Run from the beta worktree root:

    python experiments/sharding/scaling_tests/stage2_padded_slices_gpu_validation.py
"""
import gc
import time

import numpy as np

# ── Run configuration (edit here; no CLI args) ────────────────────────────────
NUM_VIEWS = 1024
NUM_SLICES = 1023      # = det rows for parallel beam; 1023 does NOT divide 2/4/8 GPUs
NUM_CHANNELS = 1024
MAX_ITERATIONS = 10
SEED = 7               # fixes partitions + subset order so the two runs are comparable
NRMSE_GATE = 1e-3      # halo-once-per-pass band is ~6e-5..3e-4; padding adds exactly 0


def host_rss_mb():
    import resource
    # ru_maxrss is KB on Linux, bytes on macOS.
    import sys
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1024 if sys.platform != 'darwin' else rss / (1024 * 1024)


def device_peaks_mb(devices):
    peaks = {}
    for d in devices:
        try:
            peaks[str(d)] = d.memory_stats()['peak_bytes_in_use'] / 2**20
        except Exception:
            peaks[str(d)] = float('nan')
    return peaks


def run_recon(model, sinogram, label):
    np.random.seed(SEED)
    t0 = time.time()
    recon, recon_dict = model.recon(sinogram, max_iterations=MAX_ITERATIONS,
                                    stop_threshold_change_pct=0.0, print_logs=False)
    recon = np.asarray(recon)   # block + host copy so the timing covers the work
    dt = time.time() - t0
    print(f'[{label}] recon time: {dt:.1f} s   recon shape: {recon.shape}')
    return recon, dt


def main():
    import mbirjax as mj
    import jax

    gpus = jax.devices('gpu')
    print(f'GPUs visible: {len(gpus)}')
    if len(gpus) < 2:
        print('Need >= 2 GPUs for a padded multi-GPU run; exiting.')
        return
    assert NUM_SLICES % len(gpus) != 0, \
        f'NUM_SLICES={NUM_SLICES} divides {len(gpus)} GPUs -- pick a non-dividing count.'

    angles = np.linspace(0, np.pi, NUM_VIEWS, endpoint=False)
    sino_shape = (NUM_VIEWS, NUM_SLICES, NUM_CHANNELS)

    # Phantom + sinogram, generated on the multi-GPU model (any model works; the
    # forward projection output is gathered to a plain real-shape array).
    print('Generating phantom + sinogram ...')
    model_multi = mj.ParallelBeamModel(sino_shape, angles)   # bare: auto-shards all GPUs
    phantom = mj.generate_3d_shepp_logan_low_dynamic_range(
        model_multi.get_params('recon_shape'))
    sinogram = np.asarray(model_multi.forward_project(phantom))
    del phantom
    gc.collect()

    print('\n--- device_summary (expect "slices padded '
          f'{NUM_SLICES}->{NUM_SLICES + (-NUM_SLICES) % len(gpus)}") ---')
    print(model_multi.device_summary)

    # Informational: entry placement should add ~0 host RSS (no padded host copy).
    rss0 = host_rss_mb()
    prepared = model_multi.prepare_sino_for_devices(sinogram)
    rss1 = host_rss_mb()
    sino_mb = sinogram.nbytes / 2**20
    print(f'\nprepare_sino_for_devices: host max-RSS delta {rss1 - rss0:.0f} MB '
          f'(sinogram is {sino_mb:.0f} MB; expect a delta far below that)')

    # 1) Multi-GPU padded run FIRST (per-device peaks must be read before the
    #    single-GPU run raises device 0's cumulative high-water mark).
    recon_multi, t_multi = run_recon(model_multi, prepared, f'{len(gpus)} GPUs (padded)')
    peaks_multi = device_peaks_mb(gpus)
    print(f'[multi] per-device peak_bytes_in_use (MB): '
          + ', '.join(f'{v:.0f}' for v in peaks_multi.values()))
    del prepared
    gc.collect()

    # 2) Single-GPU reference (no padding: one device divides everything).
    model_single = mj.ParallelBeamModel(sino_shape, angles)
    model_single.configure_devices(1)
    print('\n--- single-GPU device_summary ---')
    print(model_single.device_summary)
    recon_single, t_single = run_recon(model_single, sinogram, '1 GPU (reference)')
    peaks_single = device_peaks_mb(gpus[:1])
    print(f'[single] device-0 peak_bytes_in_use (MB): '
          + ', '.join(f'{v:.0f}' for v in peaks_single.values()))

    # 3) Compare.
    diff = recon_multi - recon_single
    nrmse = float(np.linalg.norm(diff) / np.linalg.norm(recon_single))
    max_abs = float(np.max(np.abs(diff)))
    print(f'\nmulti-vs-single: NRMSE = {nrmse:.3e}   max|diff| = {max_abs:.3e}')
    print(f'speedup {len(gpus)} GPUs vs 1: {t_single / t_multi:.2f}x')
    verdict = 'PASS' if nrmse <= NRMSE_GATE else 'FAIL'
    print(f'\n=== {verdict}: NRMSE {nrmse:.3e} vs gate {NRMSE_GATE:.0e} ===')


if __name__ == '__main__':
    main()
