"""E0 attribution (sort share): nsys driver measuring what fraction of the production
FORWARD kernel time is the CUB radix sort.

Intended to run UNDER nsys with the cudaProfilerApi capture range, so the report covers
ONLY the warm measured calls (compile/autotune kernels are excluded by the bracket):

    nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop \
         -o ~/headroom/results/e0_fwd_nsys \
         python -u plans/experiments/projector_kernels/e0_sort_share.py
    nsys stats --report cuda_gpu_kern_sum ~/headroom/results/e0_fwd_nsys.nsys-rep

In the kern_sum output: DeviceRadixSort*/cub* kernels = the sort; the remaining big
fusions are the gather/segment-sum work (join names against the e0_hlo_dump HLO files).
The script also prints wall-clock per call, so kernel-sum seconds can be normalized.

Runs standalone too (no nsys): the cudaProfiler calls are no-ops without a collector.

Run on ONE GPU.  Edit constants below.
"""
import os

# ── Config ────────────────────────────────────────────────────────────────────
SINO_SHAPE = (1024, 1008, 992)
WARMUP = 1
MEASURED_CALLS = 3

os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.94')

import ctypes
import time

import numpy as np


def cuda_profiler(on):
    """cudaProfilerStart/Stop via libcudart -- the nsys capture-range bracket."""
    try:
        rt = ctypes.CDLL('libcudart.so')
    except OSError:
        print('[e0_sort_share] libcudart not loadable; running unbracketed')
        return
    (rt.cudaProfilerStart if on else rt.cudaProfilerStop)()


def main():
    import mbirjax
    import jax
    print(f'jax {jax.__version__}  devices={jax.devices()}  sino={SINO_SHAPE}')

    angles = np.linspace(0, np.pi, SINO_SHAPE[0], endpoint=False)
    model = mbirjax.ParallelBeamModel(SINO_SHAPE, angles)
    model.configure_devices(1)
    recon_shape = model.get_params('recon_shape')
    idx = mbirjax.gen_full_indices(recon_shape, use_ror_mask=model.get_params('use_ror_mask'))
    rng = np.random.default_rng(0)
    cylinders = model._shard_recon(rng.random((len(idx), recon_shape[2]), dtype=np.float32))
    jax.block_until_ready(cylinders)

    for _ in range(WARMUP):
        jax.block_until_ready(model.sparse_forward_project(cylinders, idx))

    cuda_profiler(True)
    times = []
    for _ in range(MEASURED_CALLS):
        t0 = time.perf_counter()
        jax.block_until_ready(model.sparse_forward_project(cylinders, idx))
        times.append(time.perf_counter() - t0)
    cuda_profiler(False)

    print(f'RESULT e0_sort_share: measured_calls={MEASURED_CALLS} '
          f'fwd_wall_s={[round(t, 3) for t in times]} total_wall_s={sum(times):.3f}')
    print('(divide the nsys kern_sum radix-sort seconds by total_wall_s for the sort share)')


if __name__ == '__main__':
    main()
