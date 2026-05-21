import jax.numpy as jnp
import jax, numpy as np
import mbirjax

sino_shape = (180, 64, 64)
model = mbirjax.ParallelBeamModel(sino_shape, angles=np.linspace(0, np.pi, 180))
model.set_params(sharpness=0.0)
recon_shape = model.get_params('recon_shape')

rng = jax.random.PRNGKey(1)
num_spatial = recon_shape[0] * recon_shape[1]
flat_recon   = jax.random.normal(rng, (num_spatial, recon_shape[2]))
pixel_indices = jnp.arange(200)

qggmrf_nbr_wts, sigma_x, p, q, T = model.get_params(['qggmrf_nbr_wts', 'sigma_x', 'p', 'q', 'T'])
b = mbirjax.get_b_from_nbr_wts(qggmrf_nbr_wts)
qggmrf_params = (b, sigma_x, p, q, T)

# --- Test 1: halo=None matches saved baseline ---
g, h = mbirjax.qggmrf_gradient_and_hessian_at_indices(flat_recon, recon_shape, pixel_indices, qggmrf_params)

save_output = False
if save_output:
    np.savez('qggmrf_baseline.npz', g=g, h=h)   # run once to save
else:
    data = np.load('qggmrf_baseline.npz')
    print('Test 1 (halo=None vs baseline):',
          np.amax(np.abs(data['g'] - g)), np.amax(np.abs(data['h'] - h)))  # expect 0.0, 0.0

# --- Test 2: explicit mirror halos match baseline ---
left_halo  = flat_recon[:, 0]
right_halo = flat_recon[:, -1]
g2, h2 = mbirjax.qggmrf_gradient_and_hessian_at_indices(flat_recon, recon_shape, pixel_indices, qggmrf_params,
                                                         left_halo=left_halo, right_halo=right_halo)
print('Test 2 (mirror halos vs baseline):',
      np.amax(np.abs(g2 - g)), np.amax(np.abs(h2 - h)))  # expect 0.0, 0.0

# --- Test 3: interior shard self-consistency ---
# Split the recon in half along slices and check that the gradient at the split point
# agrees between the two halves (each using the other half's boundary as a halo).
mid = recon_shape[2] // 2
left_half  = flat_recon[:, :mid]
right_half = flat_recon[:, mid:]

# Left shard: right halo = first slice of right half
gL, _ = mbirjax.qggmrf_gradient_and_hessian_at_indices(
    left_half,  (recon_shape[0], recon_shape[1], mid), pixel_indices, qggmrf_params,
    left_halo=None, right_halo=flat_recon[:, mid])

# Right shard: left halo = last slice of left half
gR, _ = mbirjax.qggmrf_gradient_and_hessian_at_indices(
    right_half, (recon_shape[0], recon_shape[1], mid), pixel_indices, qggmrf_params,
    left_halo=flat_recon[:, mid-1], right_halo=None)

# The gradient at the split point computed from each side should agree
print('Test 3 (shard boundary self-consistency): left[-1] vs right[0]:',
      np.amax(np.abs(gL[:, -1] - g[:, mid-1])),   # left shard's last slice vs full-recon reference
      np.amax(np.abs(gR[:,  0] - g[:, mid])))      # right shard's first slice vs full-recon reference
# Both should be 0.0 -- the sharded halves reproduce the full-recon gradient exactly