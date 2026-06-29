import jax
import jax.numpy as jnp
from functools import partial
from collections import namedtuple
import mbirjax as mj
from mbirjax import TomographyModel
from typing import Literal, Union, overload, Any
import warnings

MultiAxisParallelBeamParamNames = mj.ParamNames | Literal['angles', 'recon_slice_offset']


# Default slice-band size for the back projector's rolled vertical-fan loop.  The back
# projector computes the horizontal fan once per view, then walks the recon slice axis in
# bands of this many slices using a jax.lax.map.  The loop is ROLLED (the band body compiles
# once and iterates at runtime), so the compiled program size is independent of the slice
# count -- which can range from tens of slices on one device to thousands across many.  The
# band size bounds the per-band scratch to (pixel_batch x band_size); it is the back
# projector's memory knob, and back_project_one_view_to_band (horizontal fan once + one band)
# is the per-band entry the multi-device reduce-scatter uses once sharding is turned on.
MULTIAXIS_SLICE_BAND_SIZE = 128

# Slice-batch size for the FORWARD vertical fan's rolled jax.lax.map.  The forward vertical
# fan SCATTERS input slices onto detector rows (it does not band the output rows, so it stays
# monolithic over the detector), so this chunks the INPUT slice axis to bound the per-batch
# transient -- a memory/compile knob.  128 matches the old entries_per_cylinder_batch default,
# so the forward output is unchanged.
MULTIAXIS_FORWARD_SLICE_BATCH = 128


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

        # Slice batching for the split projectors is set by the module-level band/batch
        # constants (the back projector bands the slice axis; the forward vertical fan chunks
        # the input slices), not by a per-instance attribute.
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
        geometry_param_names += ['psf_radius']
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
        # The vertical fan anchors the slice<->row map on the REAL slice count (recon_shape[2]);
        # require the input cylinder to have exactly that length so the params anchor and the
        # input agree (the forward is monolithic -- never banded -- so the global slice index
        # equals the local index).  Mirrors TranslationModel.forward_project_pixel_batch_to_one_view.
        num_recon_slices = projector_params.recon_shape[2]
        if voxel_values.shape[0] != pixel_indices.shape[0] or len(voxel_values.shape) < 2 or \
                voxel_values.shape[1] != num_recon_slices:
            raise ValueError('voxel_values must have shape[0:2] = (num_indices, num_slices)')

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
        # Anchor the slice<->z map on the REAL slice count from params (recon_shape[2]), NOT the
        # input cylinder length, so the forward stays consistent with the back projector (which
        # anchors z on recon_shape[2]) and the validity test below is a GLOBAL one (k < S_real).
        # The forward is monolithic -- never banded (decision C) -- so the global slice index
        # equals the local index here; the batch entry asserts the per-pixel cylinder is
        # num_recon_slices long, so this is value-identical today and becomes the inert-padding
        # anchor once the slice axis is zero-padded for sharding.
        num_slices = recon_shape[2]

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
        # Chunk the input slices into batches of MULTIAXIS_FORWARD_SLICE_BATCH to bound the
        # per-batch transient (the forward stays monolithic over the detector rows).
        slices_per_batch = MULTIAXIS_FORWARD_SLICE_BATCH
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

        # Build the view CHANNEL-MAJOR -- (num_det_channels, num_det_rows) rather than
        # (num_det_rows, num_det_channels) -- so the per-pixel channel scatter writes a
        # CONTIGUOUS row (stride 1) instead of a column of stride num_det_channels.  A
        # power-of-2 num_det_channels column stride aliases the CPU cache; the contiguous
        # write avoids it.  Transpose back to (rows, channels) on return (one cheap pass,
        # fused by XLA).  This mirrors ParallelBeamModel and ConeBeamModel.
        sinogram_view_T = jnp.zeros((num_det_channels, num_det_rows))

        # Loop over horizontal kernel
        for n_offset in jnp.arange(start=-gp.psf_radius, stop=gp.psf_radius + 1):
            n = n_p_center + n_offset
            dist = jnp.abs(n_p - n)
            weight = jnp.clip((W_p_c + 1.0) / 2.0 - dist, 0.0, L_max)

            valid = (n >= 0) & (n < num_det_channels)

            # Scatter each pixel's detector column (rows_data[p, :]) into channel n[p].
            sinogram_view_T = sinogram_view_T.at[n, :].add(rows_data * (weight * scale * valid)[:, None])

        return sinogram_view_T.T

    # =========================================================================
    # Split Back Projectors
    # =========================================================================

    @staticmethod
    @partial(jax.jit, static_argnames=['projector_params', 'slice_band_size'])
    def back_project_one_view_to_pixel_batch(sinogram_view, pixel_indices, single_view_params, projector_params,
                                             coeff_power=1, slice_band_size=None):
        """
        Back project one view to multiple voxel cylinders: horizontal fan (channels -> rows)
        ONCE, then walk the recon slice axis in fixed-size bands with a ROLLED jax.lax.map
        (mirrors ConeBeamModel / TranslationModel).

        The horizontal fan does not depend on the recon slice, so it is computed once; the
        vertical fan is then evaluated band-by-band via back_vertical_fan_band_pixel_batch.  The
        band body compiles once and iterates at runtime, so the compiled program does not grow
        with the slice count.  The last band runs past num_recon_slices and is cropped off; the
        banded kernel's global validity clip zeros that padded tail, so the crop is exact.

        Args:
            sinogram_view (2D jax array): one view, shape (num_det_rows, num_det_channels).
            pixel_indices (1D jax array of int): indices into the flattened num_rows x num_cols array.
            single_view_params: [azimuth, elevation] for this view.
            projector_params (namedtuple): (sinogram_shape, recon_shape, geometry_params).
            coeff_power (int): backproject using (A_ij ** coeff_power); 2 for the Hessian diagonal.
            slice_band_size (int or None): number of recon slices per band (the memory knob).
                None uses MULTIAXIS_SLICE_BAND_SIZE; tests pass a small value to exercise the
                multi-band assembly on a small geometry.  Static (sets shapes).

        Returns:
            (len(pixel_indices), num_recon_slices) voxel cylinders.
        """
        num_recon_slices = projector_params.recon_shape[2]
        num_pixels = pixel_indices.shape[0]

        # Horizontal fan once per view -> (num_pixels, num_det_rows).
        rows_data = MultiAxisParallelModel.back_horizontal_fan_one_view_to_pixel_batch(
            sinogram_view, pixel_indices, single_view_params, projector_params, coeff_power)

        # Tile the slice axis into uniform bands; jax.lax.map needs equal-shape iterations, so
        # the last band runs past num_recon_slices and is cropped off below.
        band_size = MULTIAXIS_SLICE_BAND_SIZE if slice_band_size is None else slice_band_size
        band_size = min(band_size, num_recon_slices)
        num_bands = (num_recon_slices + band_size - 1) // band_size
        band_starts = band_size * jnp.arange(num_bands)        # g0 for each band (mapped over)

        def back_one_band(g0):
            # (num_pixels, band_size) back projection onto global slices [g0, g0 + band_size).
            return MultiAxisParallelModel.back_vertical_fan_band_pixel_batch(
                rows_data, pixel_indices, single_view_params, projector_params,
                g0, band_size, coeff_power=coeff_power)

        bands = jax.lax.map(back_one_band, band_starts)        # (num_bands, num_pixels, band_size)
        # Reassemble into (num_pixels, num_bands * band_size): for pixel p and band b, slice
        # b*band_size + l is bands[b, p, l].  Then crop the padded tail back to the real count.
        back_projection = jnp.transpose(bands, (1, 0, 2)).reshape(num_pixels, num_bands * band_size)
        back_projection = jax.lax.slice_in_dim(back_projection, 0, num_recon_slices, axis=1)

        return back_projection

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

        # Read the view CHANNEL-MAJOR -- transpose to (num_det_channels, num_det_rows) up front
        # so the per-pixel gather reads a CONTIGUOUS row (stride 1) instead of a column of
        # stride num_det_channels (the adjoint of the forward kernel's channel-major scatter).
        sinogram_view_T = sinogram_view.T            # (num_det_channels, num_det_rows)
        det_rows_values = jnp.zeros((num_pixels, num_det_rows))

        for n_offset in jnp.arange(start=-gp.psf_radius, stop=gp.psf_radius + 1):
            n = n_p_center + n_offset
            dist = jnp.abs(n_p - n)
            weight = jnp.clip((W_p_c + 1.0) / 2.0 - dist, 0.0, L_max)

            valid = (n >= 0) & (n < num_det_channels)

            w_total = (weight * scale * valid) ** coeff_power

            # Gather each pixel's channel row: (num_pixels, num_det_rows).
            rows = sinogram_view_T[jnp.clip(n, 0, num_det_channels - 1), :]

            det_rows_values += rows * w_total[:, None]

        return det_rows_values

    # ──────────────────────────────────────────────────────────────────────────
    # Banded vertical fan + per-view banded back kernel
    #
    # These back-project a view onto a contiguous band of GLOBAL recon-slice indices
    # [g0, g0+L): the vertical fan is restricted to that slice range, while the horizontal fan
    # is the same for every band and is computed once per view.  The single-device back
    # projector (back_project_one_view_to_pixel_batch) walks the slice axis in bands of
    # MULTIAXIS_SLICE_BAND_SIZE via back_vertical_fan_band_pixel_batch; back_project_one_view_to_band
    # (horizontal fan once + one band) is the per-band entry the multi-device back projector will
    # use (reduce-scatter, when sharding is turned on).
    #
    # Physical z-coordinates come from the problem's recon_shape and the GLOBAL slice index
    # (k = g0 + k_local), never from the length of the band passed in, so a sub-band gives
    # exactly the same coordinates as the full cylinder.  A slice whose global index is at or
    # beyond the real slice count (a padded slice, used when the slice axis is padded to split
    # evenly across devices) receives nothing, so padding is inert.
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def back_vertical_fan_band_one_pixel(detector_col, pixel_index, single_view_params, projector_params,
                                         g0, num_band_slices, coeff_power=1):
        """Back vertical fan for one pixel, producing a BAND of output slices.

        Back-projects this pixel's detector column onto the GLOBAL recon slices [g0, g0+L)
        (output length L = num_band_slices).  The z<->row map is anchored on the problem's real
        slice count (recon_shape[2]), so this just offsets the band indices by g0; padded global
        slices (index >= the real slice count) are zeroed.  Concatenating the per-band outputs
        over a tiling of the slices reconstructs the full vertical fan.

        Args:
            detector_col (1D jax array): shape (num_det_rows,), this pixel's detector column.
            pixel_index (int): index into the flattened num_rows x num_cols array.
            single_view_params: [azimuth, elevation] for this view.
            projector_params (namedtuple): (sinogram_shape, recon_shape, geometry_params).
            g0 (int): global index of the first slice in the band (traced).
            num_band_slices (int): band length L (static).
            coeff_power (int): backproject using (weight ** coeff_power); 2 for the Hessian diagonal.

        Returns:
            1D jax array of shape (num_band_slices,) of voxel values.
        """
        gp = projector_params.geometry_params
        num_views, num_det_rows, num_det_channels = projector_params.sinogram_shape
        recon_shape = projector_params.recon_shape
        num_recon_slices = recon_shape[2]
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

        # Map global slice 0 (z anchored on the REAL slice count) to detector row m.
        z_0 = (0 - (num_recon_slices - 1) / 2.0) * delta_voxel_slice + gp.recon_slice_offset
        v_0 = z_0 * jnp.cos(elevation) - t * jnp.sin(elevation)
        m_p_0 = (v_0 + gp.det_row_offset) / gp.delta_det_row + (num_det_rows - 1) / 2.0

        slope_k_to_m = (delta_voxel_slice * jnp.cos(elevation)) / gp.delta_det_row

        W_p_r = jnp.abs(delta_voxel_slice * jnp.cos(elevation) / gp.delta_det_row)
        W_p_r = jnp.maximum(W_p_r, 0.5)
        L_max = jnp.minimum(1.0, W_p_r)

        slice_indices = g0 + jnp.arange(num_band_slices)        # GLOBAL band indices
        m_p_k = m_p_0 + slice_indices * slope_k_to_m
        m_center = jnp.round(m_p_k).astype(int)

        new_cylinder = jnp.zeros(num_band_slices)
        for m_offset in jnp.arange(start=-gp.psf_radius, stop=gp.psf_radius + 1):
            m_idx = m_center + m_offset
            dist = jnp.abs(m_p_k - m_idx)
            weight = jnp.clip((W_p_r + 1.0) / 2.0 - dist, 0.0, L_max)
            valid = (m_idx >= 0) & (m_idx < num_det_rows)
            w_total = weight ** coeff_power
            val = detector_col[jnp.clip(m_idx, 0, num_det_rows - 1)]
            new_cylinder += val * w_total * valid

        # Padded global slices (index >= the real slice count) are inert.  No-op when
        # g0 + L <= num_recon_slices.
        new_cylinder = new_cylinder * (slice_indices < num_recon_slices)
        return new_cylinder

    @staticmethod
    def back_vertical_fan_band_pixel_batch(rows_data, pixel_indices, single_view_params,
                                           projector_params, g0, num_band_slices, coeff_power=1):
        """Vmap the banded back vertical fan over a pixel batch.

        ``rows_data`` is (num_pixels, num_det_rows); returns (num_pixels, num_band_slices) for
        global slices [g0, g0+L)."""
        pixel_map = jax.vmap(MultiAxisParallelModel.back_vertical_fan_band_one_pixel,
                             in_axes=(0, 0, None, None, None, None, None))
        return pixel_map(rows_data, pixel_indices, single_view_params, projector_params,
                         g0, num_band_slices, coeff_power)

    @staticmethod
    @partial(jax.jit, static_argnames=['projector_params', 'num_band_slices'])
    def back_project_one_view_to_band(sinogram_view, pixel_indices, single_view_params, projector_params,
                                      g0, num_band_slices, coeff_power=1):
        """Banded back projection of one view onto GLOBAL recon slices [g0, g0+L).

        Horizontal fan once (full detector) -> banded vertical fan.  Returns
        (num_pixels, num_band_slices).  ``g0`` is traced; ``num_band_slices`` (= L) is static.
        This is the per-band entry the multi-device reduce-scatter back projector uses on each
        view-owner's shard."""
        rows_data = MultiAxisParallelModel.back_horizontal_fan_one_view_to_pixel_batch(
            sinogram_view, pixel_indices, single_view_params, projector_params, coeff_power=coeff_power)
        return MultiAxisParallelModel.back_vertical_fan_band_pixel_batch(
            rows_data, pixel_indices, single_view_params, projector_params,
            g0, num_band_slices, coeff_power=coeff_power)

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
        ``output_sharded`` chooses the output form (numpy by default, view-sharded when True)."""
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
            sinogram (numpy or jax array): The input sinogram with shape (num_views, num_rows, num_channels).
            filter_name (string, optional): Name of the filter to be used. Defaults to "ramp".
            view_batch_size (int, optional): DEPRECATED and ignored -- the shared row-filter
                kernel sets its own batch (tomography_utils.ROW_FILTER_BATCH).  Kept for back-compat.
            output_sharded (bool, optional): If False (default), return a numpy array; if True,
                return the view-sharded device form (on a single device the same either way).

        Returns:
            filtered_sinogram (numpy or jax array): The sinogram after FBP filtering.
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
        # Keep the pipeline on-device (matches ParallelBeamModel.fbp_recon): fbp_filter returns the
        # view-sharded form (no host round-trip), back_project consumes it directly and decides the
        # output form.  fbp_filter shards a plain input itself, so no explicit entry shard is needed.
        filtered_sinogram = self.fbp_filter(sinogram, filter_name=filter_name, output_sharded=True)
        return self.back_project(filtered_sinogram, output_sharded=output_sharded)


# Backward-compatible public API name used throughout docs/examples.
MultiAxisParallelBeamModel = MultiAxisParallelModel
