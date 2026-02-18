import jax.numpy as jnp
import mbirjax as mj
import matplotlib.pyplot as plt
import time

# 1. Setup Parameters (Rectangle Phantom)
sinogram_shape = (18, 100, 200)
azimuths = jnp.linspace(0, jnp.pi, sinogram_shape[0], endpoint=False)
elevation = jnp.ones_like(azimuths)*jnp.deg2rad(25)
# Input angles as (num_views,2)
angles_zero = jnp.column_stack([azimuths, jnp.zeros_like(azimuths)])
angles_tilt = jnp.column_stack([azimuths, elevation])

# 2. Models
model_pb = mj.ParallelBeamModel(sinogram_shape, azimuths)
model_ma_zero = mj.MultiAxisParallelBeamModel(sinogram_shape, angles_zero)
model_ma_tilt = mj.MultiAxisParallelBeamModel(sinogram_shape, angles_tilt)

recon_shape = (150,150,100)
model_pb.set_params(recon_shape=recon_shape)
model_ma_zero.set_params(recon_shape=recon_shape)
model_ma_tilt.set_params(recon_shape=recon_shape)

# 3. Create a Rectangular Slice Phantom
recon_shape = (150,150,100)
phantom=jnp.zeros((150,150,100))
phantom=phantom.at[25+25:25+95, 25+30:25+60,25:70].set(1.0)

# 4. Forward Project
time_start=time.time()
sino_pb = model_pb.forward_project(phantom)
print('Parallel Beam Forward Projection Time: %.2f seconds' % (time.time()-time_start))
time_start=time.time()
sino_ma_zero = model_ma_zero.forward_project(phantom)
print('Multi-Axis (Elevation 0) Forward Projection Time: %.2f seconds' % (time.time()-time_start))
time_start=time.time()
sino_ma_tilt = model_ma_tilt.forward_project(phantom)
print('Multi-Axis (Elevation 20°) Forward Projection Time: %.2f seconds' % (time.time()-time_start))

# 5. Validation Check
max_diff = jnp.max(jnp.abs(sino_pb - sino_ma_zero))
print(f"Max Difference vs Standard (Elevation 0): {max_diff:.2e}")
# 6. Plotting sino views
vmin=jnp.min(jnp.array([sino_pb, sino_ma_zero, sino_ma_tilt]))
vmax=jnp.max(jnp.array([sino_pb, sino_ma_zero, sino_ma_tilt]))
plt.subplots(1,3,figsize=(15,5))
plt.subplot(1,3,1)
plt.imshow(sino_pb[3, :, :], vmin=vmin, vmax=vmax, cmap='gray', aspect='auto')
plt.title(f'ParallelBeamModel Sinogram View \n Azimuth={jnp.rad2deg(azimuths[3]):.1f}°')
plt.subplot(1,3,2)
plt.imshow(sino_ma_zero[3, :, :], vmin=vmin, vmax=vmax, cmap='gray', aspect='auto')
plt.title(f'MultiAxisParallelBeamModel Sinogram View \n (Azimuth,Elevation)=({jnp.rad2deg(azimuths[3]):.1f}°,0°)')
plt.subplot(1,3,3)
plt.imshow(sino_ma_tilt[3, :, :], vmin=vmin, vmax=vmax, cmap='gray', aspect='auto')
plt.title(f'MultiAxisParallelBeamModel Sinogram View \n (Azimuth,Elevation)=({jnp.rad2deg(azimuths[3]):.1f}°,20°)')
plt.show()

time_start=time.time()
recon_pb=model_pb.direct_recon(sino_pb)
print('Parallel Beam Direct Reconstruction Time: %.2f seconds' % (time.time()-time_start))
time_start=time.time()
recon_ma_zero=model_ma_zero.direct_recon(sino_ma_zero)
print('Multi-Axis (Elevation 0) Direct Reconstruction Time: %.2f seconds' % (time.time()-time_start))
time_start=time.time()
recon_ma_tilt=model_ma_tilt.direct_recon(sino_ma_tilt)
print('Multi-Axis (Elevation 20) Direct Reconstruction Time: %.2f seconds' % (time.time()-time_start))

# Check error
pb_error=jnp.linalg.norm(recon_pb-phantom)/jnp.linalg.norm(phantom)
ma_zero_error=jnp.linalg.norm(recon_ma_zero-phantom)/jnp.linalg.norm(phantom)
ma_tilt_error=jnp.linalg.norm(recon_ma_tilt-phantom)/jnp.linalg.norm(phantom)
print(f"Parallel Beam Direct Recon NRMSE: {pb_error:.2e}")
print(f"Multi-Axis (Elevation 0) Direct Recon NRMSE: {ma_zero_error:.2e}")
print(f"Multi-Axis (Elevation 20°) Direct Recon NRMSE: {ma_tilt_error:.2e}")
# Check Reconstruction Differences
max_diff_recon_zero = jnp.max(jnp.abs(recon_pb - recon_ma_zero))
max_diff_recon_tilt = jnp.max(jnp.abs(recon_pb - recon_ma_tilt))
print(f"Max Direct Reconstruction Difference vs Standard (Elevation 0): {max_diff_recon_zero:.2e}")
print(f"Max Direct Reconstruction Difference vs Standard (Elevation 20°): {max_diff_recon_tilt:.2e}")

# Display Direct Reconstructions

mj.slice_viewer(recon_pb, recon_ma_zero, title='FBP Reconstruction: ParallelBeamModel vs MultiAxisParallelBeamModel with elevation=0', slice_axis=2, slice_label='Slice')
mj.slice_viewer(recon_pb, recon_ma_tilt, title='FBP Reconstruction: ParallelBeamModel vs MultiAxisParallelBeamModel with elevation=20', slice_axis=2, slice_label='Slice')

time_start=time.time()
recon_pb,_=model_pb.recon(sino_pb)
print('Parallel Beam Reconstruction Time: %.2f seconds' % (time.time()-time_start))
time_start=time.time()
recon_ma_zero,_=model_ma_zero.recon(sino_ma_zero)
print('Multi-Axis (Elevation 0) Reconstruction Time: %.2f seconds' % (time.time()-time_start))
time_start=time.time()
recon_ma_tilt,_=model_ma_tilt.recon(sino_ma_tilt)
print('Multi-Axis (Elevation 20°) Reconstruction Time: %.2f seconds' % (time.time()-time_start))

# Check error
pb_error=jnp.linalg.norm(recon_pb-phantom)/jnp.linalg.norm(phantom)
ma_zero_error=jnp.linalg.norm(recon_ma_zero-phantom)/jnp.linalg.norm(phantom)
ma_tilt_error=jnp.linalg.norm(recon_ma_tilt-phantom)/jnp.linalg.norm(phantom)
print(f"Parallel Beam Recon NRMSE: {pb_error:.2e}")
print(f"Multi-Axis (Elevation 0) Recon NRMSE: {ma_zero_error:.2e}")
print(f"Multi-Axis (Azimuth 0) Recon NRMSE: {ma_tilt_error:.2e}")

# Check Reconstruction Differences
max_diff_recon_zero = jnp.max(jnp.abs(recon_pb - recon_ma_zero))
max_diff_recon_tilt = jnp.max(jnp.abs(recon_pb - recon_ma_tilt))
print(f"Max Reconstruction Difference vs Standard (Elevation 0): {max_diff_recon_zero:.2e}")
print(f"Max Reconstruction Difference vs Standard (Elevation 20°): {max_diff_recon_tilt:.2e}")

# Display MAP Reconstructions
mj.slice_viewer(recon_pb, recon_ma_zero, title='MAP Reconstruction: ParallelBeamModel vs MultiAxisParallelBeamModel with elevation=0', slice_axis=2, slice_label='Slice')
mj.slice_viewer(recon_pb, recon_ma_tilt, title='MAP Reconstruction: ParallelBeamModel vs MultiAxisParallelBeamModel with elevation=20', slice_axis=2, slice_label='Slice')