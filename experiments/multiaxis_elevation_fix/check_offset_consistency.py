"""Consistency check for the det_channel_offset sign fix.

MultiAxisParallelModel at elevation 0 must equal ParallelBeamModel -- and now for BOTH zero and
NONZERO det_channel_offset. Before the fix, multi-axis used n = (u - off)/Δ + c while parallel/cone
use n = (u + off)/Δ + c, so a nonzero offset shifted the detector the opposite way. This asserts
the two now agree, and that adjointness still holds with an offset present.

Run on a GPU node with the editable mbirjax env:
  python check_offset_consistency.py
"""
import numpy as np, jax.numpy as jnp
from mbirjax import ParallelBeamModel, MultiAxisParallelModel


def nrmse(a, b):
    return float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))


def main():
    np.random.seed(0)
    N, nv, NR, NC = 24, 8, 24, 41
    az = np.linspace(0, np.pi, nv, endpoint=False).astype(np.float32)
    vol = jnp.asarray(np.random.rand(N, N, N).astype(np.float32))

    print("multi-axis(el=0) vs ParallelBeam, forward-projection NRMSE (should be ~0 for every offset):")
    for off in [0.0, 3.5, -2.0]:
        pm = ParallelBeamModel((nv, NR, NC), jnp.asarray(az))
        pm.set_params(recon_shape=(N, N, N), delta_voxel=1.0, delta_det_channel=1.0, det_channel_offset=off)
        mm = MultiAxisParallelModel((nv, NR, NC), jnp.asarray(np.stack([az, np.zeros_like(az)], 1)))
        mm.set_params(recon_shape=(N, N, N), delta_voxel=1.0, delta_det_channel=1.0,
                      delta_det_row=1.0, det_channel_offset=off)
        print(f"  det_channel_offset={off:+.1f}: NRMSE={nrmse(mm.forward_project(vol), pm.forward_project(vol)):.2e}")

    # adjoint must still hold with a nonzero offset and nonzero elevation
    ang = jnp.array([[0.3, np.deg2rad(30)], [1.1, np.deg2rad(15)]])
    m = MultiAxisParallelModel((2, 31, 41), ang)
    m.set_params(recon_shape=(N, N, N), delta_voxel=1.0, delta_det_channel=1.0,
                 delta_det_row=1.0, det_channel_offset=2.5)
    x = jnp.asarray(np.random.rand(N, N, N).astype(np.float32))
    y = jnp.asarray(np.random.rand(2, 31, 41).astype(np.float32))
    lhs = float(jnp.sum(m.forward_project(x) * y)); rhs = float(jnp.sum(x * m.back_project(y)))
    print(f"adjoint (offset=2.5, el=30): reldiff={abs(lhs - rhs) / abs(lhs):.2e}")


if __name__ == "__main__":
    main()
