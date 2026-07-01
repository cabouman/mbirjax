"""Verify whether the maximum(W_p_r, 0.5) floor is needed, per footprint mode.
W_p_r (rows) = _vertical_footprint_phys(az, el, ...)/delta_det_row, minimized over azimuth."""
import numpy as np, jax.numpy as jnp
from mbirjax.multiaxis_parallel import _vertical_footprint_phys

def wpr_range(el_deg, broaden, dv, dvr, dvs, ddr):
    azs = np.linspace(0, np.pi/2, 181)
    vals = [float(_vertical_footprint_phys(a, np.deg2rad(el_deg), dv, dvr, dvs, broaden)) / ddr for a in azs]
    return min(vals), max(vals)

for label, (dv, dvr, dvs, ddr) in [
        ("isotropic dv=dvr=dvs=ddr=1", (1.0, 1.0, 1.0, 1.0)),
        ("thin slices dvs=0.3",        (1.0, 1.0, 0.3, 1.0)),
        ("coarse detector ddr=3",      (1.0, 1.0, 1.0, 3.0))]:
    print(f"\n{label}:  W_p_r (rows), min over azimuth  [floor = 0.5]")
    print(f"{'el':>4} {'narrow min':>12} {'broaden(maxedge) min':>22}")
    for el in [0, 30, 45, 55, 60, 70, 80, 89, 90]:
        nmin, _ = wpr_range(el, False, dv, dvr, dvs, ddr)
        bmin, _ = wpr_range(el, True, dv, dvr, dvs, ddr)
        flagn = " <0.5" if nmin < 0.5 else ""
        flagb = " <0.5" if bmin < 0.5 else ""
        print(f"{el:>4} {nmin:12.4f}{flagn:6} {bmin:16.4f}{flagb}")
