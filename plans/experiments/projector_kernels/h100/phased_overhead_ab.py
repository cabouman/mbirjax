"""Attribute the Phase D per-call overhead seen in the VCD cell (+35% at 200^3).

Times, on ONE library (run under old and new snapshots and compare):
  wrapper_small : sparse_forward_project on a VCD-sized pixel SUBSET, called in a loop
                  (the composition VCD sees; per-call time is the decision number)
  wrapper_back  : same for sparse_back_project
  centers_only  : the eager _jit_compute_scatter_centers call alone (new library only)
Loop timing with block_until_ready ONLY at the end of each batch of calls, mirroring how
VCD's dispatch pipeline overlaps -- plus a fully-synchronous per-call variant to separate
dispatch cost from device cost.

Run:  python plans/experiments/projector_kernels/phased_overhead_ab.py
"""
import os

# ── Run parameters (edit here) ────────────────────────────────────────────────
SINO_SHAPE = (200, 208, 160)      # the VCD cell's geometry
PIXEL_COUNTS = [156, 1241, 19856, 25600]   # the EXACT distinct subset sizes the VCD cell uses
CALLS = 50
WARMUP = 5

os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.94')
import time                       # noqa: E402
import numpy as np                # noqa: E402
import mbirjax                    # noqa: E402
import jax                        # noqa: E402
import jax.numpy as jnp           # noqa: E402


def main():
    print(f"devices: {jax.devices()}  mbirjax: {mbirjax.__file__}")
    angles = np.linspace(0, np.pi, SINO_SHAPE[0], endpoint=False)
    model = mbirjax.ParallelBeamModel(SINO_SHAPE, angles)
    model.configure_devices(1)
    recon_shape = model.get_params('recon_shape')
    full = np.asarray(mbirjax.gen_full_indices(recon_shape, use_ror_mask=False))
    rng = np.random.default_rng(0)
    sino = jnp.asarray(rng.random(SINO_SHAPE, dtype=np.float32))
    pf = model.projector_functions

    def time_loop(name, fn):
        for _ in range(WARMUP):
            jax.block_until_ready(fn())
        # pipelined: dispatch CALLS times, sync once (how VCD's queue overlaps)
        t0 = time.perf_counter()
        out = None
        for _ in range(CALLS):
            out = fn()
        jax.block_until_ready(out)
        t_pipe = (time.perf_counter() - t0) / CALLS
        # synchronous: per-call block (upper bound incl. all latency)
        t0 = time.perf_counter()
        for _ in range(CALLS):
            jax.block_until_ready(fn())
        t_sync = (time.perf_counter() - t0) / CALLS
        print(f"  {name:16s}: {1e3 * t_pipe:8.3f} ms/call pipelined   "
              f"{1e3 * t_sync:8.3f} ms/call synchronous", flush=True)

    for n_pix in PIXEL_COUNTS:
        stride = max(1, len(full) // n_pix)
        idx = jnp.asarray(full[::stride][:n_pix])
        vox = jnp.asarray(rng.random((int(idx.shape[0]), recon_shape[2]), dtype=np.float32))
        print(f"-- subset pixels: {idx.shape[0]}  views: {SINO_SHAPE[0]} --")
        time_loop('wrapper_fwd', lambda: pf.sparse_forward_project(vox, idx))
        time_loop('wrapper_back', lambda: pf.sparse_back_project(sino, idx))
        time_loop('wrapper_back_cp2', lambda: pf.sparse_back_project(sino, idx, coeff_power=2))

    # New library only: the centers computation alone.
    try:
        from mbirjax.projectors import _jit_compute_scatter_centers, ProjectorParams
        tiles = model.tiles
        pp = ProjectorParams(tuple(model.get_params('sinogram_shape')),
                             tuple(model.get_params('recon_shape')),
                             model.get_geometry_parameters(),
                             int(bool(tiles.sort_by_channel)),
                             int(bool(tiles.back_stacked_gather)))
        vp = pf.view_params_array
        time_loop('centers_pixmajor', lambda: _jit_compute_scatter_centers(
            vp, idx, channel_coord_fn=model.compute_channel_coordinate,
            projector_params=pp, pixels_major=True))
        time_loop('centers_viewmajor', lambda: _jit_compute_scatter_centers(
            vp, idx, channel_coord_fn=model.compute_channel_coordinate,
            projector_params=pp, pixels_major=False))
    except ImportError:
        print('  (old library: no centers machinery)')


if __name__ == "__main__":
    main()
