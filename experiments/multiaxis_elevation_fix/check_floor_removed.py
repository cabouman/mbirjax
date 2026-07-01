"""With the floor removed: max-of-edges must stay finite + mass-conserved at ALL elevations;
narrow only degenerates at exactly el=90 (a top-down view past the >45deg warning)."""
import numpy as np, jax.numpy as jnp
from mbirjax import MultiAxisParallelModel
N=31
def total(mode_broaden, eldeg):
    m=MultiAxisParallelModel((1,81,81), jnp.array([[0.3, np.deg2rad(eldeg)]]))
    m.set_elevation_correction(correct_elevation_pathlength=True, broaden_elevation_footprint=mode_broaden)
    m.set_params(recon_shape=(N,N,N), delta_voxel=1.0, delta_det_row=1.0, delta_det_channel=1.0)
    vol=np.zeros((N,N,N),np.float32); vol[N//2,N//2,N//2]=1.0
    return float(jnp.sum(m.forward_project(jnp.asarray(vol))))
print("single-voxel total (floor removed); correct = 1.0, NaN = degenerate")
print(f"{'el':>4} {'narrow':>10} {'max-of-edges':>14}")
for el in [45,60,80,89,90]:
    print(f"{el:>4} {total(False,el):10.4f} {total(True,el):14.4f}")
