"""E1 open item: attribute CONE FORWARD kernel time between the vertical and horizontal
fans at the 1024^3-class cell (gpu_headroom_findings.md pending list; open item #1 in the
June profiling notes, now decision-relevant twice over: it sizes the vfan prize for the
kernel campaign AND for slice-parity efficiency — the vfan is why P>1 schedules cost ~P x
forward today; see slice_parity_plan.md R1 cost accounting).

Variants at the production kernel shape (cone 1024-class policy: 4096-pixel batch vmapped
over 128 views):
  fwd_full  : the library kernel (vfan -> hfan), as the driver composes it
  vfan_only : forward_vertical_fan_pixel_batch_to_one_view alone (share isolator)
  hfan_only : forward_horizontal_fan... fed PRECOMPUTED vfan outputs (share isolator)

Shares need not add to 1.0 exactly (XLA overlaps/fuses differently in composition — the
cone-back lesson); the split still bounds the vfan prize.

Run:  python plans/experiments/projector_kernels/cone_fwd_split_ab.py   (constants below)
"""
import os

# ── Run parameters ────────────────────────────────────────────────────────────
SINO_SHAPE = (1024, 1008, 992)
PIXEL_BATCH = 4096            # the cone >=768-slice GPU policy value
VIEW_BATCH = 128
WARMUP = 2
TRIALS = 5

os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.94')
import time                      # noqa: E402
import numpy as np               # noqa: E402
import mbirjax                   # noqa: E402
import jax                       # noqa: E402
import jax.numpy as jnp          # noqa: E402
from mbirjax.cone_beam import ConeBeamModel        # noqa: E402
from mbirjax.projectors import ProjectorParams, _jit_compute_scatter_centers   # noqa: E402


def main():
    num_views, num_det_rows, num_det_channels = SINO_SHAPE
    angles = np.linspace(0, np.pi, num_views, endpoint=False)
    model = mbirjax.ConeBeamModel(SINO_SHAPE, angles,
                                  source_detector_dist=4.0 * num_det_channels,
                                  source_iso_dist=2.0 * num_det_channels)
    model.configure_devices(1)
    args = (tuple(model.get_params('sinogram_shape')), tuple(model.get_params('recon_shape')),
            model.get_geometry_parameters())
    args += tuple(int(bool(getattr(model.tiles, f, 0))) for f in ProjectorParams._fields[3:])
    pp = ProjectorParams(*args)
    recon_shape = model.get_params('recon_shape')
    num_slices = recon_shape[2]
    print(f'sino {SINO_SHAPE}  recon {recon_shape}  P={PIXEL_BATCH} V={VIEW_BATCH}',
          flush=True)

    rng = np.random.default_rng(0)
    idx = jnp.asarray(np.sort(rng.choice(recon_shape[0] * recon_shape[1],
                                         size=PIXEL_BATCH, replace=False)).astype(np.int32))
    values = jnp.asarray(rng.random((PIXEL_BATCH, num_slices), dtype=np.float32))
    view_params = jnp.asarray(model.projector_functions.view_params_array)[:VIEW_BATCH]
    n_pc = _jit_compute_scatter_centers(
        jnp.asarray(model.projector_functions.view_params_array), idx,
        ConeBeamModel.compute_channel_coordinate, pp,
        pixels_major=False)[:VIEW_BATCH]          # (V, P) int32
    jax.block_until_ready((values, view_params, n_pc))

    fwd_full = jax.jit(jax.vmap(ConeBeamModel.forward_project_pixel_batch_to_one_view,
                                in_axes=(None, None, 0, 0, None)),
                       static_argnums=(4,))
    vfan = jax.jit(jax.vmap(ConeBeamModel.forward_vertical_fan_pixel_batch_to_one_view,
                            in_axes=(None, None, 0, None)),
                   static_argnums=(3,))
    hfan = jax.jit(jax.vmap(ConeBeamModel.forward_horizontal_fan_pixel_batch_to_one_view,
                            in_axes=(0, None, 0, 0, None)),
                   static_argnums=(4,))

    vfan_out = jax.block_until_ready(vfan(values, idx, view_params, pp))   # (V, P, rows)

    variants = {
        'fwd_full': lambda: fwd_full(values, idx, view_params, n_pc, pp),
        'vfan_only': lambda: vfan(values, idx, view_params, pp),
        'hfan_only': lambda: hfan(vfan_out, idx, view_params, n_pc, pp),
    }
    times = {}
    for name, call in variants.items():
        for _ in range(WARMUP):
            jax.block_until_ready(call())
        ts = []
        for _ in range(TRIALS):
            t0 = time.perf_counter()
            jax.block_until_ready(call())
            ts.append(time.perf_counter() - t0)
        times[name] = float(np.median(ts))
        print(f'{name:10s} {times[name] * 1e3:9.2f} ms  (all: '
              f'{[round(t * 1e3, 2) for t in ts]})', flush=True)

    full = times['fwd_full']
    print(f'\nshares of the full kernel: vfan {times["vfan_only"] / full:.2f}, '
          f'hfan {times["hfan_only"] / full:.2f} '
          f'(sum {(times["vfan_only"] + times["hfan_only"]) / full:.2f} — '
          f'overlap/fusion means != 1.00 is expected)', flush=True)


if __name__ == '__main__':
    main()
