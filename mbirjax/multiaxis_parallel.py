import jax
import jax.numpy as jnp
from functools import partial
from collections import namedtuple
import mbirjax as mj
from mbirjax import TomographyModel
from typing import Literal, Union, overload, Any
import warnings

MultiAxisParallelBeamParamNames = mj.ParamNames | Literal['angles', 'recon_slice_offset']


class MultiAxisParallelModel(TomographyModel):
    """
    Parallel beam geometry allowing for a per-view elevation (tilt) angle.

    This class extends ParallelBeamModel to support a 2-axis rotation geometry:
      - Azimuth (Theta): Rotation around the object's Z-axis (standard tomography rotation, analogous to angles in ParallelBeamModel).
      - Elevation (Phi): Tilt of the ray vector out of the XY plane (extension beyond single-axis ParallelBeamModel).

    When elevation = 0, this model is mathematically equivalent to ParallelBeamModel (mirroring its behavior exactly).

    Novelty/Extension:
        - Parallel beam laminography is a special case of this geometry.
        - Introduces split vertical/horizontal fan projectors (mirrors ConeBeamModel and TranslationModel).
        - Supports arbitrary elevation without assuming slice independence (generalization of ParallelBeamModel).
        - Directional velocity-weighted ramp filter in direct_recon (extension of standard ramp in ParallelBeamModel).

    Args:
        sinogram_shape (tuple): (num_views, num_det_rows, num_det_channels)
        angles (jnp.ndarray): (num_views,2) array.
            - angles[:,0] = Azimuth (radians, analogous to ParallelBeamModel)
            - angles[:,1] = Elevation (radians, unique extension)

    """

    DIRECT_RECON_VIEW_BATCH_SIZE = TomographyModel.DIRECT_RECON_VIEW_BATCH_SIZE

    def __init__(self, sinogram_shape, angles):
        # Validate input shape
        angles = jnp.asarray(angles)
        if angles.ndim != 2 or angles.shape[1] != 2:
            raise ValueError(f"angles must have shape (num_views,2). Got {angles.shape}.")
        if angles.shape[0] != sinogram_shape[0]:
            raise ValueError(
                f"Number of angle pairs ({angles.shape[0]}) must match number of views ({sinogram_shape[0]}).")

        # Check for large elevation angles and warn
        elevations = jnp.abs(angles[:,1])
        if jnp.any(elevations > jnp.pi / 4):  # pi/4 radians = 45 degrees
            warnings.warn("One or more elevation angles exceed 45 degrees. This may degrade approximation quality.")

        view_params_array = angles

        # Initialize base class
        # We define entries_per_cylinder_batch for the split projectors
        self.entries_per_cylinder_batch = 128
        self.bp_psf_radius = 1

        super().__init__(sinogram_shape, angles=view_params_array, view_params_name='angles', recon_slice_offset=0.0)
        self.set_params(geometry_type=str(type(self)))

    def get_psf_radius(self):
        """
        Compute the integer radius of the PSF kernel (mirrors get_psf_radius in ConeBeamModel and TranslationModel).
        """
        delta_det_channel, delta_det_row, delta_voxel, voxel_row_aspect, voxel_slice_aspect = self.get_params(
            ['delta_det_channel', 'delta_det_row', 'delta_voxel', 'voxel_row_aspect', 'voxel_slice_aspect']
        )
        delta_voxel_row = voxel_row_aspect * delta_voxel
        delta_voxel_slice = voxel_slice_aspect * delta_voxel

        # Horizontal radius: the in-plane footprint on the channels is bounded by the larger
        # of the two in-plane voxel pitches (column = delta_voxel, row = delta_voxel_row).
        max_in_plane_pitch = max(delta_voxel, delta_voxel_row)
        psf_radius_u = int(jnp.ceil(jnp.ceil(max_in_plane_pitch / delta_det_channel) / 2))

        # Vertical radius (elevation tilt): the slice pitch projects onto the detector rows.
        psf_radius_v = int(jnp.ceil(jnp.ceil(delta_voxel_slice / delta_det_row) / 2))

        # We use a single radius for the parameter handler, taking the max to be safe (same as ConeBeamModel).
        return max(psf_radius_u, psf_radius_v)

    @overload
    def get_params(self, parameter_names: Union[
        MultiAxisParallelBeamParamNames, list[MultiAxisParallelBeamParamNames]]) -> Any:
        ...

    def get_params(self, parameter_names) -> Any:
        return super().get_params(parameter_names)

    def verify_valid_params(self):
        """Verify parameters match the expected geometry constraints."""
        super().verify_valid_params()
        sinogram_shape = self.get_params('sinogram_shape')
        angles = self.get_params('angles')

        if angles.shape[0] != sinogram_shape[0]:
            raise ValueError(f"View mismatch: {angles.shape[0]} angles for {sinogram_shape[0]} views.")
        if angles.shape[1] != 2:
            raise ValueError("Each view requires exactly 2 angles: [azimuth, elevation].")

    def get_geometry_parameters(self):
        """Package view-independent parameters into a namedtuple for JIT (same pattern as ConeBeamModel)."""
        # 1. Get parameters managed by ParameterHandler (self.params)
        geometry_param_names = [
            'delta_det_channel', 'det_channel_offset', 'delta_det_row',
            'det_row_offset', 'delta_voxel', 'recon_slice_offset',
            'voxel_row_aspect', 'voxel_slice_aspect'
        ]
        geometry_param_values = self.get_params(geometry_param_names)

        # Ensure values are Python scalars (floats) to avoid tracer issues inside projectors.
        geometry_param_values = [float(v) if v is not None else 0.0 for v in geometry_param_values]

        # 2. Append additional parameters not in self.params (same pattern as ConeBeamModel).
        geometry_param_names += ['entries_per_cylinder_batch', 'psf_radius']
        geometry_param_values.append(self.entries_per_cylinder_batch)
        geometry_param_values.append(self.get_psf_radius())

        # The class is shared across instances (make_geometry_params) so the projectors' jit cache
        # is shared rather than re-traced per instance.
        return self.make_geometry_params(geometry_param_names, geometry_param_values)

    def get_magnification(self):
        """For parallel beam geometries, magnification is always 1.0 (same as ParallelBeamModel)."""
        return 1.0

    def auto_set_recon_geometry(self, no_compile=False, no_warning=False):
        """
        Set the reconstruction geometry based on the largest bounding box required
        to project onto the detector at the given angles.
        """
        sinogram_shape = self.get_params('sinogram_shape')
        num_views, num_det_rows, num_det_channels = sinogram_shape
        delta_det_channel, delta_det_row = self.get_params(['delta_det_channel', 'delta_det_row'])
        magnification = self.get_magnification()

        # Physical size of detector
        max_u = (num_det_channels * delta_det_channel) / 2.0
        max_v = (num_det_rows * delta_det_row) / 2.0

        angles = self.get_params('angles')
        elevations = angles[:, 1]

        # 1. XY Radius: Determined by U coverage
        max_R_xy = max_u  # Safe assumption for centering

        # 2. Z Height: Determined by V coverage
        # v = z cos(el) - t sin(el).
        # We need max z to fit in max_v.
        min_cos_el = jnp.min(jnp.abs(jnp.cos(elevations)))
        # Clamp to avoid division by zero (top-down view implies infinite Z capability on detector)
        min_cos_el = jnp.maximum(min_cos_el, 0.1)
        max_R_z = max_v / min_cos_el

        # Honor anisotropic voxels: the column (x) axis uses delta_voxel, the row (y) axis uses
        # delta_voxel_row, and the slice (z) axis uses delta_voxel_slice, so each axis is sized
        # by its own physical extent divided by its own pitch.  With the default aspects (== 1)
        # this reduces to the isotropic sizing (rows == cols, slice pitch == delta_voxel).
        voxel_row_aspect, voxel_slice_aspect = self.get_params(['voxel_row_aspect', 'voxel_slice_aspect'])
        delta_voxel = delta_det_channel
        delta_voxel_row = voxel_row_aspect * delta_voxel
        delta_voxel_slice = voxel_slice_aspect * delta_voxel
        num_recon_cols = int(jnp.floor(2 * max_R_xy / delta_voxel))
        num_recon_rows = int(jnp.floor(2 * max_R_xy / delta_voxel_row))
        num_recon_slices = int(jnp.floor(2 * max_R_z / delta_voxel_slice))

        self.set_params(recon_shape=(num_recon_rows, num_recon_cols, num_recon_slices),
                        delta_voxel=delta_voxel,
                        no_compile=no_compile, no_warning=no_warning)

    # =========================================================================
    # Split Projectors (Vertical then Horizontal)
    # =========================================================================
    # This split mirrors the vertical/horizontal fan approach in ConeBeamModel
    # and TranslationModel, but generalized for elevation tilt.

    @staticmethod
    @partial(jax.jit, static_argnames='projector_params')
    def forward_project_pixel_batch_to_one_view(voxel_values, pixel_indices, single_view_params, projector_params):
        """
        Forward project a set of voxel cylinders to one view.
        Splits the operation into Vertical (Z -> V) and Horizontal (V -> U) steps (mirrors ConeBeamModel).
        """
        # 1. Vertical Projection: Project voxel cylinders (slices) to detector rows
        # Output: (num_pixels, num_det_rows)
        vertical_projector = MultiAxisParallelModel.forward_vertical_fan_pixel_batch_to_one_view
        rows_data = vertical_projector(voxel_values, pixel_indices, single_view_params, projector_params)

        # 2. Horizontal Projection: Scatter pixel-rows to detector channels
        # Output: (num_det_rows, num_det_channels)
        horizontal_projector = MultiAxisParallelModel.forward_horizontal_fan_pixel_batch_to_one_view
        sinogram_view = horizontal_projector(rows_data, pixel_indices, single_view_params, projector_params)

        return sinogram_view

    @staticmethod
    def forward_vertical_fan_pixel_batch_to_one_view(voxel_values, pixel_indices, single_view_params, projector_params):
        """
        Maps (pixels, slices) -> (pixels, rows) using scatter (generalization of TranslationModel).
        """
        # Vmap over the pixel batch
        pixel_map = jax.vmap(MultiAxisParallelModel.forward_vertical_fan_one_pixel_to_one_view,
                             in_axes=(0, 0, None, None))
        new_pixels = pixel_map(voxel_values, pixel_indices, single_view_params, projector_params)
        return new_pixels


    @staticmethod
    def forward_vertical_fan_one_pixel_to_one_view(voxel_cylinder, pixel_index, single_view_params, projector_params):
        """
        Projects a single voxel cylinder (1D array of slices) onto the detector rows using scatter (extension of TranslationModel for elevation).
        SCATTER IMPLEMENTATION: Iterates over slices k, scatters to rows m.
        This allows for slope=0 (top down view).
        """
        gp = projector_params.geometry_params
        num_views, num_det_rows, num_det_channels = projector_params.sinogram_shape
        recon_shape = projector_params.recon_shape
        num_slices = voxel_cylinder.shape[0]

        azimuth, elevation = single_view_params[0], single_view_params[1]

        # Anisotropic voxel pitches: the row (y) axis uses delta_voxel_row and the slice (z)
        # axis uses delta_voxel_slice; the column (x) axis uses delta_voxel.
        delta_voxel_row = gp.voxel_row_aspect * gp.delta_voxel
        delta_voxel_slice = gp.voxel_slice_aspect * gp.delta_voxel

        # 1. Calculate geometry for this pixel column
        row_idx, col_idx = jnp.unravel_index(pixel_index, recon_shape[:2])
        y = ((recon_shape[0] - 1) / 2.0 - row_idx) * delta_voxel_row
        x = (col_idx - (recon_shape[1] - 1) / 2.0) * gp.delta_voxel

        # t is the coordinate along the ray projection in XY plane
        t = -x * jnp.sin(azimuth) + y * jnp.cos(azimuth)

        # Slope: How many detector rows does one voxel step in z cover?
        # v = z * cos(el) - t * sin(el)
        slope_k_to_m = (delta_voxel_slice * jnp.cos(elevation)) / gp.delta_det_row

        # We define m_p (projected row) for the center of the 0-th slice (k=0):
        # z_0 = -(num_slices - 1)/2 * delta_voxel_slice + recon_offset
        z_0 = (0 - (num_slices - 1) / 2.0) * delta_voxel_slice + gp.recon_slice_offset
        v_0 = z_0 * jnp.cos(elevation) - t * jnp.sin(elevation)
        m_p_0 = (v_0 + gp.det_row_offset) / gp.delta_det_row + (num_det_rows - 1) / 2.0

        # W_p_r: projected footprint width of one voxel on rows
        W_p_r = jnp.abs(delta_voxel_slice * jnp.cos(elevation) / gp.delta_det_row)
        W_p_r = jnp.maximum(W_p_r, 0.5)

        scaling = 1.0

        # --- Scatter Logic (Iterate slices, write to rows) ---
        # We iterate over slices in batches to manage loop size, but typically slices < rows.
        # We will use the 'entries_per_cylinder_batch' to chunk the slices.
        slices_per_batch = gp.entries_per_cylinder_batch
        slices_per_batch = min(slices_per_batch, num_slices)
        num_slice_batches = (num_slices + slices_per_batch - 1) // slices_per_batch
        slice_indices = slices_per_batch * jnp.arange(num_slice_batches)

        L_max = jnp.minimum(1.0, W_p_r)

        def project_slice_batch(start_index):
            k_indices = start_index + jnp.arange(slices_per_batch)
            valid_k = (k_indices < num_slices)

            # Projection center for these slices
            m_p = m_p_0 + k_indices * slope_k_to_m
            m_center = jnp.round(m_p).astype(int)

            # Get values (masked)
            vals = jnp.where(valid_k, voxel_cylinder[jnp.clip(k_indices, 0, num_slices - 1)], 0.0)

            # Accumulator for this batch
            batch_det = jnp.zeros(num_det_rows)

            for m_offset in jnp.arange(start=-gp.psf_radius, stop=gp.psf_radius + 1):
                m = m_center + m_offset
                dist = jnp.abs(m_p - m)

                weight = jnp.clip((W_p_r + 1.0) / 2.0 - dist, 0.0, L_max)
                valid_m = (m >= 0) & (m < num_det_rows)

                # Scatter add
                # We want to add (vals * weight) to indices (m)
                # Since m varies per k, we use .at[m].add(...)
                # Only update valid m indices (invalid m will be clipped or masked)

                # We must mask invalid_m to a safe index for the scatter, then add 0
                safe_m = jnp.clip(m, 0, num_det_rows - 1)
                add_val = vals * weight * scaling * valid_m

                batch_det = batch_det.at[safe_m].add(add_val)

            return batch_det, None

        # Map over slice chunks
        det_column_parts, _ = jax.lax.map(project_slice_batch, slice_indices)
        # Sum the contributions from all slice chunks
        det_column = jnp.sum(det_column_parts, axis=0)

        return det_column

    @staticmethod
    def forward_horizontal_fan_pixel_batch_to_one_view(rows_data, pixel_indices, single_view_params, projector_params):
        """
        Maps (pixels, rows) -> (rows, channels) (mirrors horizontal fan in ConeBeamModel and TranslationModel).
        Scatters the vertical strips into the correct horizontal channels.
        """
        gp = projector_params.geometry_params
        num_det_rows, num_det_channels = projector_params.sinogram_shape[1:]
        azimuth = single_view_params[0]

        # Anisotropic in-plane pitches: the row (y) axis uses delta_voxel_row, the column (x)
        # axis uses delta_voxel.
        delta_voxel_row = gp.voxel_row_aspect * gp.delta_voxel

        # Map pixels to u coordinates
        row_idx, col_idx = jnp.unravel_index(pixel_indices, projector_params.recon_shape[:2])
        y = ((projector_params.recon_shape[0] - 1) / 2.0 - row_idx) * delta_voxel_row
        x = (col_idx - (projector_params.recon_shape[1] - 1) / 2.0) * gp.delta_voxel

        u_p = x * jnp.cos(azimuth) + y * jnp.sin(azimuth)
        n_p = (u_p - gp.det_channel_offset) / gp.delta_det_channel + (num_det_channels - 1) / 2.0
        n_p_center = jnp.round(n_p).astype(int)

        # In-plane footprint of the voxel on the channels: the larger of the two in-plane pitches
        # projected onto the u-axis (matches ParallelBeamModel.compute_proj_data).  Reduces to
        # delta_voxel * max(|cos|, |sin|) when voxel_row_aspect == 1.
        footprint_xy = jnp.maximum(jnp.abs(jnp.cos(azimuth)) * gp.delta_voxel,
                                   jnp.abs(jnp.sin(azimuth)) * delta_voxel_row)
        W_p_c = footprint_xy / gp.delta_det_channel
        L_max = jnp.minimum(1.0, W_p_c)

        # Density normalization: in-plane voxel cross-section area / footprint length.
        scale = (gp.delta_voxel * delta_voxel_row) / footprint_xy

        sinogram_view = jnp.zeros((num_det_rows, num_det_channels))

        # Loop over horizontal kernel
        for n_offset in jnp.arange(start=-gp.psf_radius, stop=gp.psf_radius + 1):
            n = n_p_center + n_offset
            dist = jnp.abs(n_p - n)
            weight = jnp.clip((W_p_c + 1.0) / 2.0 - dist, 0.0, L_max)

            valid = (n >= 0) & (n < num_det_channels)

            # This is effectively: sinogram = sinogram.at[:, n].add(rows_data.T)
            # transpose rows_data to (num_rows, num_pixels)
            sinogram_view = sinogram_view.at[:, n].add(rows_data.T * (weight * scale * valid))

        return sinogram_view

    # =========================================================================
    # Split Back Projectors
    # =========================================================================

    @staticmethod
    @partial(jax.jit, static_argnames='projector_params')
    def back_project_one_view_to_pixel_batch(sinogram_view, pixel_indices, single_view_params, projector_params,
                                             coeff_power=1):
        """
        Back project: Horizontal (Channels -> Rows) then Vertical (Rows -> Slices) (mirrors ConeBeamModel).
        """
        # 1. Horizontal Backproj: (rows, channels) -> (pixels, rows)
        horizontal_bp = MultiAxisParallelModel.back_horizontal_fan_one_view_to_pixel_batch
        rows_data = horizontal_bp(sinogram_view, pixel_indices, single_view_params, projector_params, coeff_power)

        # 2. Vertical Backproj: (pixels, rows) -> (pixels, slices)
        vertical_bp = MultiAxisParallelModel.back_vertical_fan_one_view_to_pixel_batch
        voxel_values = vertical_bp(rows_data, pixel_indices, single_view_params, projector_params, coeff_power)

        return voxel_values

    @staticmethod
    def back_horizontal_fan_one_view_to_pixel_batch(sinogram_view, pixel_indices, single_view_params, projector_params,
                                                    coeff_power=1):
        gp = projector_params.geometry_params
        num_det_rows, num_det_channels = projector_params.sinogram_shape[1:]
        azimuth = single_view_params[0]
        num_pixels = pixel_indices.shape[0]

        # Anisotropic in-plane pitches (must match the forward horizontal fan for adjointness).
        delta_voxel_row = gp.voxel_row_aspect * gp.delta_voxel

        # Geometry U
        row_idx, col_idx = jnp.unravel_index(pixel_indices, projector_params.recon_shape[:2])
        y = ((projector_params.recon_shape[0] - 1) / 2.0 - row_idx) * delta_voxel_row
        x = (col_idx - (projector_params.recon_shape[1] - 1) / 2.0) * gp.delta_voxel

        u_p = x * jnp.cos(azimuth) + y * jnp.sin(azimuth)
        n_p = (u_p - gp.det_channel_offset) / gp.delta_det_channel + (num_det_channels - 1) / 2.0
        n_p_center = jnp.round(n_p).astype(int)

        footprint_xy = jnp.maximum(jnp.abs(jnp.cos(azimuth)) * gp.delta_voxel,
                                   jnp.abs(jnp.sin(azimuth)) * delta_voxel_row)
        W_p_c = footprint_xy / gp.delta_det_channel
        L_max = jnp.minimum(1.0, W_p_c)
        scale = (gp.delta_voxel * delta_voxel_row) / footprint_xy

        # Accumulate rows
        det_rows_values = jnp.zeros((num_pixels, num_det_rows))

        for n_offset in jnp.arange(start=-gp.psf_radius, stop=gp.psf_radius + 1):
            n = n_p_center + n_offset
            dist = jnp.abs(n_p - n)
            weight = jnp.clip((W_p_c + 1.0) / 2.0 - dist, 0.0, L_max)

            valid = (n >= 0) & (n < num_det_channels)

            w_total = (weight * scale * valid) ** coeff_power

            # Gather columns: sinogram_view[:, n] is (num_rows, num_pixels) effectively after gather
            cols = sinogram_view[:, jnp.clip(n, 0, num_det_channels - 1)].T  # (num_pixels, num_rows)

            det_rows_values += cols * w_total[:, None]

        return det_rows_values

    @staticmethod
    def back_vertical_fan_one_view_to_pixel_batch(rows_data, pixel_indices, single_view_params, projector_params,
                                                  coeff_power=1):
        # Vmap the per-pixel logic
        pixel_map = jax.vmap(MultiAxisParallelModel.back_vertical_fan_one_view_to_one_pixel,
                             in_axes=(0, 0, None, None, None))
        return pixel_map(rows_data, pixel_indices, single_view_params, projector_params, coeff_power)

    @staticmethod
    def back_vertical_fan_one_view_to_one_pixel(detector_col, pixel_index, single_view_params, projector_params,
                                                coeff_power=1):
        gp = projector_params.geometry_params
        num_views, num_det_rows, num_det_channels = projector_params.sinogram_shape
        recon_shape = projector_params.recon_shape
        num_slices = recon_shape[2]
        azimuth, elevation = single_view_params[0], single_view_params[1]

        # Anisotropic voxel pitches (must match the forward vertical fan for adjointness):
        # row (y) -> delta_voxel_row, slice (z) -> delta_voxel_slice, column (x) -> delta_voxel.
        delta_voxel_row = gp.voxel_row_aspect * gp.delta_voxel
        delta_voxel_slice = gp.voxel_slice_aspect * gp.delta_voxel

        # Geometry
        row_idx, col_idx = jnp.unravel_index(pixel_index, recon_shape[:2])
        y = ((recon_shape[0] - 1) / 2.0 - row_idx) * delta_voxel_row
        x = (col_idx - (recon_shape[1] - 1) / 2.0) * gp.delta_voxel
        t = -x * jnp.sin(azimuth) + y * jnp.cos(azimuth)

        # Map z=0 to m
        z_0 = (0 - (num_slices - 1) / 2.0) * delta_voxel_slice + gp.recon_slice_offset
        v_0 = z_0 * jnp.cos(elevation) - t * jnp.sin(elevation)
        m_p_0 = (v_0 + gp.det_row_offset) / gp.delta_det_row + (num_det_rows - 1) / 2.0

        slope_k_to_m = (delta_voxel_slice * jnp.cos(elevation)) / gp.delta_det_row

        W_p_r = jnp.abs(delta_voxel_slice * jnp.cos(elevation) / gp.delta_det_row)
        W_p_r = jnp.maximum(W_p_r, 0.5)
        L_max = jnp.minimum(1.0, W_p_r)

        # Batching for output slices (Gather logic is fine for Backproj)
        slices_per_batch = gp.entries_per_cylinder_batch
        slices_per_batch = min(slices_per_batch, num_slices)
        num_slice_batches = (num_slices + slices_per_batch - 1) // slices_per_batch
        slice_indices = slices_per_batch * jnp.arange(num_slice_batches)

        def create_voxel_cylinder_slices(start_index):
            k_target = start_index + jnp.arange(slices_per_batch)

            # Forward projection of this k
            m_p_k = m_p_0 + k_target * slope_k_to_m
            m_center = jnp.round(m_p_k).astype(int)

            new_cylinder = jnp.zeros(slices_per_batch)

            for m_offset in jnp.arange(start=-gp.psf_radius, stop=gp.psf_radius + 1):
                m_idx = m_center + m_offset
                dist = jnp.abs(m_p_k - m_idx)

                weight = jnp.clip((W_p_r + 1.0) / 2.0 - dist, 0.0, L_max)
                valid = (m_idx >= 0) & (m_idx < num_det_rows)

                w_total = weight ** coeff_power

                val = detector_col[jnp.clip(m_idx, 0, num_det_rows - 1)]
                new_cylinder += val * w_total * valid

            return new_cylinder, None

        recon_voxel_cylinder, _ = jax.lax.map(create_voxel_cylinder_slices, slice_indices)
        return recon_voxel_cylinder.flatten()[:num_slices]

    # =========================================================================
    # Direct Recon (Directional Filtered Backprojection)
    # =========================================================================
    # This is a novel extension of the standard ramp filter in ParallelBeamModel.

    def direct_recon(self, sinogram, filter_name="ramp", view_batch_size=DIRECT_RECON_VIEW_BATCH_SIZE,
                     output_sharded=False):
        return self.fbp_recon(sinogram, filter_name, view_batch_size, output_sharded=output_sharded)

    def direct_filter(self, sinogram, filter_name="ramp", view_batch_size=None, output_sharded=False):
        """Thin alias for :meth:`fbp_filter` — the FBP filtering step of a direct recon (mirrors
        ParallelBeamModel.direct_filter so ``model.direct_filter`` works uniformly across geometries).
        ``output_sharded`` chooses the output form (plain by default, view-sharded when True)."""
        return self.fbp_filter(sinogram, filter_name=filter_name, output_sharded=output_sharded)

    def fbp_filter(self, sinogram, filter_name="ramp", view_batch_size=None, output_sharded=False):
        """
        Perform FBP filtering on the given sinogram with a standard 1-D ramp filter along
        the detector channels (the same shared path as ``ParallelBeamModel.fbp_filter``).

        This REPLACES the earlier per-view directional 2-D ramp, whose orientation and
        magnitude came from ``jnp.gradient(angles)`` -- the step between LIST-NEIGHBOR views.
        This geometry is a set of SIMULTANEOUS fixed measurements (multiple sources/detectors)
        with no acquisition trajectory, so the ``angles`` list order is arbitrary; an
        order-dependent filter is therefore ill-defined (permuting the arbitrary list changed
        the recon).  The fix treats the recon as stacked 2-D FBP -- a uniform per-view angular
        weight ``pi / num_views`` (folded in by the shared method, the SAME constant as
        parallel beam) and the standard channel ramp.  This is order-INVARIANT and reduces
        EXACTLY to
        ``ParallelBeamModel.fbp_filter`` in the azimuth-only, equally-spaced, zero-elevation
        limit; elevation is approximated in the filter and corrected by the iterative
        ``recon()`` (this direct recon is an initializer -- see the note on ``fbp_recon``).

        The voxel-size scaling is ``1 / (delta_voxel * delta_voxel_row)`` (NOT
        ``1 / delta_voxel**2``, which silently assumed ``voxel_row_aspect == 1``); with
        anisotropic voxels the row pitch ``delta_voxel_row`` enters correctly.

        Args:
            sinogram (jax array): The input sinogram with shape (num_views, num_rows, num_channels).
            filter_name (string, optional): Name of the filter to be used. Defaults to "ramp".
            view_batch_size (int, optional): DEPRECATED and ignored -- the shared row-filter
                kernel sets its own batch (tomography_utils.ROW_FILTER_BATCH).  Kept for back-compat.
            output_sharded (bool, optional): Accepted for API uniformity.  This model runs
                single-device until it is ported to the sharded placement path, so the output
                is the same either way; the argument is plumbed through for that future port.

        Returns:
            filtered_sinogram (jax array): The sinogram after FBP filtering.
        """
        # Voxel-size scaling factor; the pi/num_views angular weight is folded into the
        # (tiny) filter array by the shared method.  Parallel beam has no FDK cosine
        # pre-weight, so row_weight=None.  See ParallelBeamModel.fbp_filter.
        delta_voxel, voxel_row_aspect = self.get_params(['delta_voxel', 'voxel_row_aspect'])
        delta_voxel_row = voxel_row_aspect * delta_voxel
        scaling_factor = 1.0 / (delta_voxel * delta_voxel_row)
        return self._apply_direct_recon_filter(
            sinogram, filter_name, filter_scale=scaling_factor,
            output_sharded=output_sharded, row_weight=None)

    def fbp_recon(self, sinogram, filter_name="ramp", view_batch_size=None, output_sharded=False):
        """
        Perform FBP reconstruction: ramp-filter the sinogram, then back-project (the adjoint).

        Note:
            The ``pi / num_views`` angular weight (in ``fbp_filter``) assumes EQUALLY SPACED
            azimuths over the conventional full angular range and treats the geometry as
            stacked 2-D FBP (elevation approximated in the filter).  Multiaxis parallel beam
            is typically a LIMITED-ANGLE geometry, so this direct reconstruction is only
            approximate; it is intended as an initializer for the iterative ``recon()``, which
            absorbs a global angular mis-weighting (and the elevation approximation) in a few
            iterations.
        """
        filtered_sinogram = self.fbp_filter(sinogram, filter_name, view_batch_size)
        return self.back_project(filtered_sinogram, output_sharded=output_sharded)


# Backward-compatible public API name used throughout docs/examples.
MultiAxisParallelBeamModel = MultiAxisParallelModel
