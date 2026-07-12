"""E0b (nsys replacement): per-kernel time shares of the production forward and back
projections at the 1024^3 n=1 cell, via a jax profiler trace.

nsys is not available inside batch shells on gautschi (`module` is not initialized), so
this uses the same jax-trace + trace_utils machinery as e1_vcd_trace.py: trace a window
of WARM calls, aggregate per-event self-time, and bucket the device kernels:

    sort      -- CUB radix-sort kernels (the one per-step op OUTSIDE the scatter fusion;
                 e0_hlo_dump confirmed cub_sort_pairs in the fwd while body)
    scatter   -- input_scatter_fusion (the fused gather+multiply+sorted-scatter)
    reduce    -- back's input_reduce_fusion (the fused gather+weight+tap/view-sum)
    other     -- everything else on device

The fwd sort share decides approach A1 (hoist the sort); the back breakdown baselines the
view-batch L2-residency experiment (gpu_headroom_plan.md section 4/6, post-E0 update).

Run on ONE GPU:  python plans/experiments/projector_kernels/e0b_kernel_share.py
"""
import glob
import os
import sys

# ── Config ────────────────────────────────────────────────────────────────────
SINO_SHAPE = (1024, 1008, 992)
MEASURED_CALLS = 2
TRACE_ROOT = os.path.expanduser('~/headroom/results/e0b_traces')
TRACE_UTILS_CANDIDATES = [
    os.path.expanduser('~/PycharmProjects/mbirjax_metrics/experiments/profiling'),
    '/Users/gbuzzard/Documents/PyCharm Projects/Research/mbirjax_metrics/experiments/profiling',
]
TOP_N = 25

os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.94')


def _find_perfetto(out_dir):
    hits = glob.glob(os.path.join(out_dir, '**', '*.trace.json.gz'), recursive=True)
    if not hits:
        raise FileNotFoundError(f'no perfetto trace under {out_dir}')
    return max(hits, key=os.path.getmtime)


def bucket(name):
    lname = name.lower()
    if 'radix' in lname or 'cub' in lname or 'sort' in lname:
        return 'sort'
    if 'scatter' in lname:
        return 'scatter'
    if 'reduce' in lname:
        return 'reduce'
    return 'other'


def main():
    import time
    import numpy as np
    import mbirjax
    import jax

    for cand in TRACE_UTILS_CANDIDATES:
        if os.path.isdir(cand):
            sys.path.insert(0, cand)
            break
    import trace_utils   # noqa: E402

    print(f'jax {jax.__version__}  devices={jax.devices()}  sino={SINO_SHAPE}')
    angles = np.linspace(0, np.pi, SINO_SHAPE[0], endpoint=False)
    model = mbirjax.ParallelBeamModel(SINO_SHAPE, angles)
    model.configure_devices(1)
    recon_shape = model.get_params('recon_shape')
    idx = mbirjax.gen_full_indices(recon_shape, use_ror_mask=model.get_params('use_ror_mask'))
    rng = np.random.default_rng(0)
    cylinders = model._shard_recon(rng.random((len(idx), recon_shape[2]), dtype=np.float32))
    jax.block_until_ready(cylinders)
    sino_in = model._shard_sinogram(rng.random(SINO_SHAPE, dtype=np.float32))
    jax.block_until_ready(sino_in)

    for op in ('fwd', 'back'):
        call = ((lambda: model.sparse_forward_project(cylinders, idx)) if op == 'fwd'
                else (lambda: model.sparse_back_project(sino_in, idx)))
        jax.block_until_ready(call())            # warm
        tdir = os.path.join(TRACE_ROOT, op)
        t0 = time.perf_counter()
        with jax.profiler.trace(tdir):
            for _ in range(MEASURED_CALLS):
                jax.block_until_ready(call())
        wall = time.perf_counter() - t0

        events, tracks, _ = trace_utils.fusion_self_time(_find_perfetto(tdir))
        # Device events only: skip host runtime frames (trace_utils' classifier plus the
        # PjitFunction dispatch wrappers it predates); kernels carry XLA fusion / CUB names.
        dev = [(us, cnt, name) for name, (us, cnt) in events.items()
               if not trace_utils.is_host_runtime(name)
               and 'PjitFunction' not in name and 'jit(' not in name]
        dev.sort(key=lambda t: -t[0])
        buckets = {}
        for us, cnt, name in dev:
            buckets[bucket(name)] = buckets.get(bucket(name), 0.0) + us
        total_dev = sum(buckets.values())

        print(f'\n===== {op}: wall {wall:.2f} s for {MEASURED_CALLS} calls '
              f'({wall / MEASURED_CALLS:.2f} s/call) =====')
        print(f'device event self-time total: {total_dev / 1e6:.2f} s '
              f'({total_dev / 1e6 / wall * 100:.0f}% of window wall)')
        for b, us in sorted(buckets.items(), key=lambda t: -t[1]):
            print(f'  bucket {b:8s}: {us / 1e6:7.2f} s  ({us / total_dev * 100:5.1f}% of device)')
        print(f'-- top {TOP_N} device events --')
        for us, cnt, name in dev[:TOP_N]:
            print(f'  {us / 1e6:8.3f} s  x{cnt:6d}  [{bucket(name):7s}] {name[:90]}')


if __name__ == '__main__':
    main()
