Current Status

Forward projection v3
- Added forward_project_pixel_batch_to_one_view_v3 in mbirjax/parallel_beam.py.
- It keeps the original projection math but uses a GEMV + safe scatter approach:
  - Computes A_chan_n weights per offset.
  - Applies a validity mask and clips indices to avoid out-of-bounds scatter.
  - Uses voxel_values.T @ A_chan_n to form a per-channel vector and scatters into sinogram_view.
  - Uses jax.lax.fori_loop to avoid Python-level unrolling.

Comparison plan (v1 vs v2 vs v3)
- Create a micro-benchmark that times each variant on representative sizes:
  - num_pixels ~ 2K, num_det_rows ~ 1-2K, num_det_channels ~ 1-2K, psf_radius ~ 1.
- Run with a fixed set of pixel_indices and angles to make results comparable.
- Record compile time and steady-state runtime for each variant.
- Verify numerical equivalence (max/mean absolute error) against v1 for one or two random seeds.

