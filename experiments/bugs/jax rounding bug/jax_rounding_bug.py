import numpy as np
import jax.numpy as jnp
import mbirjax.parallel_beam
import jax

jax.config.update("jax_default_matmul_precision", "highest")
if __name__ == "__main__":
    """
    This is a script to demonstrate that rounding errors in jax can cause very odd bugs. 

    Jax compiles code to XLA, and different branches through the code can lead to different values for the same
    variable.  In the code below, different slicing of the the input pixels leads the to different values of the
    integer n being used to determine which sinogram entry to access versus how to calculate abs_delta_p_c_n in the
    snippet below from parallel_beam.py, back_project_one_view_to_pixel_batch:

        for n_offset in jnp.arange(start=-gp.psf_radius, stop=gp.psf_radius + 1):
            n = n_p_center + n_offset
            abs_delta_p_c_n = jnp.abs(n_p - n)
            L_p_c_n = jnp.clip((W_p_c + 1.0) / 2.0 - abs_delta_p_c_n, 0.0, L_max)
            A_chan_n = gp.delta_voxel * L_p_c_n / cos_alpha_p_xy
            A_chan_n *= (n >= 0) * (n < num_det_channels)
            A_chan_n = A_chan_n ** coeff_power
            det_voxel_cylinder = jnp.add(det_voxel_cylinder, A_chan_n.reshape((-1, 1)) * sinogram_view[:, n].T)

    One hacky fix to reduce the likelihood of this happening is to add a small value to the index before rounding.
    This would need to be done in parallel_beam.py, compute_proj_data and 
    cone_beam.py, compute_vertical_data_single_pixel and compute_horizontal_data.  
    
    See also tests/test_projectors.py: verify_view_batching
    
    Replace the default version of parallel_beam.py with the version in the folder bugs/jax rounding bug to
    see the effect of inconsistent rounding.  
    """

    # Set parameters
    num_views = 16
    num_det_rows = 5
    num_det_channels = 128
    sharpness = 0.0

    # These can be adjusted to describe the geometry in the cone beam case.
    # np.Inf is an allowable value, in which case this is essentially parallel beam
    source_detector_dist = 4 * num_det_channels
    source_iso_dist = source_detector_dist

    start_angle = -np.pi * (1 / 2)
    end_angle = np.pi * (1 / 2)

    # Initialize sinogram
    sinogram_shape = (num_views, num_det_rows, num_det_channels)
    angles = jnp.linspace(start_angle, end_angle, num_views, endpoint=False)

    # Set up the model
    ct_model = mbirjax.ParallelBeamModel(sinogram_shape, angles)

    # Generate phantom
    recon_shape = ct_model.get_params('recon_shape')
    phantom = mbirjax.gen_cube_phantom(recon_shape)

    # Generate indices of pixels and get the voxel cylinders
    full_indices = mbirjax.gen_pixel_partition(recon_shape, num_subsets=1)[0]
    voxel_values = phantom.reshape((-1,) + recon_shape[2:])[full_indices]

    # Compute forward projection with all the views at once
    sinogram = ct_model.sparse_forward_project(voxel_values, full_indices)

    num_subsets = 5
    view_subsets = jnp.array_split(jnp.arange(num_views), num_subsets)

    sinogram_copy = sinogram.copy()
    sinogram = np.zeros(sinogram.shape)
    view_index = 12
    sinogram[view_index, :, 63] = 2

    view_indices0 = jnp.arange(3605, 3616)
    view_indices1 = jnp.arange(3605, 3617)

    back_projection0 = ct_model.sparse_back_project(sinogram, view_indices0, view_indices=np.array([view_index]))
    back_projection1 = ct_model.sparse_back_project(sinogram, view_indices1, view_indices=np.array([view_index]))

    row_index0, col_index0 = jnp.unravel_index(view_indices0, recon_shape[:2])
    recon0 = jnp.zeros(recon_shape)
    recon0 = recon0.at[row_index0, col_index0].set(back_projection0)

    row_index1, col_index1 = jnp.unravel_index(view_indices1, recon_shape[:2])
    recon1 = jnp.zeros(recon_shape)
    recon1 = recon1.at[row_index1, col_index1].set(back_projection1)

    title = 'Back projections of the same data, but with a set of \npixels that is one more in the right figure than in the left'
    mbirjax.slice_viewer(recon0[20:40, 20:40, :], recon1[20:40, 20:40, :], slice_axis=2, title=title)

