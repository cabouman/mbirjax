import jax
import jax.numpy as jnp
from functools import partial
from collections import namedtuple
import mbirjax as mj
from mbirjax import TomographyModel
from mbirjax.projectors import (horizontal_fan_project, horizontal_fan_back,
                                SORTED_CHANNEL_REDUCE_MIN_COLS,
                                SORTED_CHANNEL_REDUCE_MAX_PSF_RADIUS,
                                SORTED_CHANNEL_REDUCE_MIN_COLLISION_RATIO)
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


def _vertical_footprint_phys(azimuth, elevation, delta_voxel, delta_voxel_row, delta_voxel_slice):
    """Physical vertical footprint of one voxel projected onto the detector v-axis.

    The detector v-axis is ``e_v = (sin az sin el, -cos az sin el, cos el)``; a voxel's extent
    along it is the projection of its three edges.  We use the amplitude-method LARGEST-edge
    footprint -- ``max`` over the z-edge and the two in-plane edges projected onto v -- the SAME
    max-of-edges approximation ParallelBeamModel.compute_proj_data uses for ``footprint_xy`` (this
    vertical pass IS 2-D parallel beam in the ``(t,z)->v`` plane at angle ``el``).  For
    ``|el| < ~45 deg`` the z-edge dominates; beyond ``45 deg`` the in-plane ``sin(el)`` edges take
    over.  Because the max never collapses to 0, no footprint floor is needed (like parallel/cone).

    NOTE: this is the ``max`` of edges, NOT their sum -- the sum lays the full trapezoid base down
    as a flat box and over-widens (the trapezoid's effective width is ~the largest edge).
    """
    z_edge = delta_voxel_slice * jnp.abs(jnp.cos(elevation))
    sin_el = jnp.abs(jnp.sin(elevation))
    x_edge = sin_el * jnp.abs(jnp.sin(azimuth)) * delta_voxel
    y_edge = sin_el * jnp.abs(jnp.cos(azimuth)) * delta_voxel_row
    return jnp.maximum(z_edge, jnp.maximum(x_edge, y_edge))


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
        - Order-invariant stacked-2D FBP initializer in direct_recon (standard channel ramp; see fbp_filter).

    Args:
        sinogram_shape (tuple): (num_views, num_det_rows, num_det_channels)
        angles (jnp.ndarray): (num_views,2) array.
            - angles[:,0] = Azimuth (radians, analogous to ParallelBeamModel)
            - angles[:,1] = Elevation (radians, unique extension)

    """

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

        # Vertical radius (elevation tilt): a SINGLE psf_radius serves every view, so size it from
        # the LARGEST vertical footprint over the whole view set.  Per view the max-of-edges
        # footprint (see _vertical_footprint_phys) is
        # max(|cos el|*delta_voxel_slice, |sin el|*(in-plane pitch)), and the two edges peak at
        # OPPOSITE elevations -- the z-edge (|cos el|) is largest at the SMALLEST tilt, the in-plane
        # edge (|sin el|) at the largest -- so we take the max of EACH edge over the ACTUAL
        # elevations.  Using max_v|cos el| / max_v|sin el| (not cos/sin at a single elevation
        # extremum) keeps it tight AND robust to any angle, with no monotonicity assumption.
        # Worst-case azimuth gives an in-plane pitch of max(delta_voxel, delta_voxel_row).
        angles = self.get_params('angles')
        if angles is not None:
            max_abs_cos_el = float(jnp.max(jnp.abs(jnp.cos(angles[:, 1]))))
            max_abs_sin_el = float(jnp.max(jnp.abs(jnp.sin(angles[:, 1]))))
        else:
            max_abs_cos_el, max_abs_sin_el = 1.0, 0.0
        z_edge = max_abs_cos_el * delta_voxel_slice
        in_plane_edge = max_abs_sin_el * max(delta_voxel, delta_voxel_row)
        vertical_footprint = max(z_edge, in_plane_edge)
        psf_radius_v = int(jnp.ceil(jnp.ceil(vertical_footprint / delta_det_row) / 2))

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
    # Coordinate-stage helpers (mirror ParallelBeamModel / ConeBeamModel)
    # =========================================================================
    # These name the same three coordinate stages the other geometries use, so intuition
    # transfers directly.  The ONLY functional difference from parallel/cone is the elevation
    # term in geometry_xyz_to_uv; everything else is the shared convention (object rotates into
    # the detector frame at recon_ijk_to_xyz; x is the detector-channel direction u).

    @staticmethod
    def recon_ijk_to_xyz(i, j, k, delta_voxel, voxel_row_aspect, voxel_slice_aspect,
                         recon_shape, recon_slice_offset, azimuth):
        """Map recon indices (i=row, j=col, k=slice) to continuous xyz, rotating the in-plane
        coordinates by the azimuth -- identical convention to ParallelBeamModel.recon_ij_to_x /
        ConeBeamModel.recon_ijk_to_xyz (the object rotates into the detector frame, so the rotated
        x is the detector channel coordinate u)."""
        num_recon_rows, num_recon_cols, num_recon_slices = recon_shape
        delta_voxel_row = voxel_row_aspect * delta_voxel
        delta_voxel_slice = voxel_slice_aspect * delta_voxel
        y_tilde = delta_voxel_row * (i - (num_recon_rows - 1) / 2.0)
        x_tilde = delta_voxel * (j - (num_recon_cols - 1) / 2.0)
        cosine = jnp.cos(azimuth)
        sine = jnp.sin(azimuth)
        x = cosine * x_tilde - sine * y_tilde
        y = sine * x_tilde + cosine * y_tilde
        z = delta_voxel_slice * (k - (num_recon_slices - 1) / 2.0) + recon_slice_offset
        return x, y, z

    @staticmethod
    def geometry_xyz_to_uv(x, y, z, elevation):
        """Map continuous xyz (in the azimuth-rotated frame) to detector uv.  This is the ONE
        place multi-axis differs from parallel/cone: a nonzero elevation tilts the ray out of the
        plane, so the detector row coordinate v picks up the in-plane depth y --
        ``v = z*cos(el) + y*sin(el)``.  (ParallelBeamModel has v = z; ConeBeamModel has v = mag*z;
        the parallel-beam magnification here is 1.  At el=0 this reduces to v = z.)"""
        u = x
        v = z * jnp.cos(elevation) + y * jnp.sin(elevation)
        return u, v

    @staticmethod
    def detector_uv_to_mn(u, v, delta_det_channel, delta_det_row, det_channel_offset,
                          det_row_offset, num_det_rows, num_det_channels):
        """Map continuous detector uv to fractional detector indices (m=row, n=channel).  The
        offset signs match ParallelBeamModel/ConeBeamModel (detector_uv_to_mn)."""
        n = (u + det_channel_offset) / delta_det_channel + (num_det_channels - 1) / 2.0
        m = (v + det_row_offset) / delta_det_row + (num_det_rows - 1) / 2.0
        return m, n

    @staticmethod
    def compute_channel_coordinate(pixel_indices, single_view_params, projector_params):
        """Continuous projected channel coordinate n_p for one view, via the shared
        coordinate stages (u = azimuth-rotated x).  This is the ONE chain behind the fwd and
        back horizontal fans AND the scatter-centers precompute (the integer center
        round(n_p) is computed OUTSIDE the projector programs; see the base-class docstring
        for the concrete-input contract)."""
        gp = projector_params.geometry_params
        num_det_rows, num_det_channels = projector_params.sinogram_shape[1:]
        azimuth = single_view_params[0]
        row_idx, col_idx = jnp.unravel_index(pixel_indices, projector_params.recon_shape[:2])
        x, _, _ = MultiAxisParallelModel.recon_ijk_to_xyz(
            row_idx, col_idx, 0, gp.delta_voxel, gp.voxel_row_aspect, gp.voxel_slice_aspect,
            projector_params.recon_shape, gp.recon_slice_offset, azimuth)
        u_p = x  # geometry_xyz_to_uv: the channel coordinate u is the azimuth-rotated x
        _, n_p = MultiAxisParallelModel.detector_uv_to_mn(
            u_p, 0.0, gp.delta_det_channel, gp.delta_det_row, gp.det_channel_offset,
            gp.det_row_offset, num_det_rows, num_det_channels)
        return n_p

    def _select_tile_policy(self, on_gpu, num_views, num_slices, n_devices):
        """Multiaxis tiling: on GPU the forward horizontal fan uses the SORTED channel
        reduction (defined at projectors.channel_scatter_reduce; measured 1.2-1.4x forward);
        everything else inherits the base policy.

        The sorted reduce's columns are the FULL detector rows (the horizontal fan consumes
        the vertical fan's (num_pixels, num_det_rows) output and is never row-banded), so
        its per-call sort amortizes at any realistic detector; the column guard below only
        trips on tiny ones.  The psf and channel-collision guards protect the regimes where
        the sorted form is a measured LOSS -- wide psf (large elevation widens the shared
        radius) and wide detectors with few scatter collisions per channel (the reason
        TranslationModel keeps the scatter path entirely).  See the guard constants in
        projectors.py for the measured crossovers.

        Deliberately NOT set: ``back_stacked_gather``.  The stacked horizontal-fan gather
        wins in isolation but changes the FULL multiaxis back kernel not at all -- the
        gather latency hides behind the vertical-fan band work (mt_back_kernel_ab.py).
        """
        tiles = super()._select_tile_policy(on_gpu, num_views, num_slices, n_devices)
        if not on_gpu:
            return tiles
        num_det_rows, num_det_channels = self.get_params('sinogram_shape')[1:]
        psf_width = 2 * self.get_psf_radius() + 1
        collision_ratio = psf_width * tiles.fwd_pixel_batch / num_det_channels
        return tiles._replace(
            sort_by_channel=(num_det_rows >= SORTED_CHANNEL_REDUCE_MIN_COLS
                             and self.get_psf_radius() <= SORTED_CHANNEL_REDUCE_MAX_PSF_RADIUS
                             and collision_ratio >= SORTED_CHANNEL_REDUCE_MIN_COLLISION_RATIO))

    # =========================================================================
    # Split Projectors (Vertical then Horizontal)
    # =========================================================================
    # This split mirrors the vertical/horizontal fan approach in ConeBeamModel
    # and TranslationModel, but generalized for elevation tilt.

    @staticmethod
    @partial(jax.jit, static_argnames='projector_params')
    def forward_project_pixel_batch_to_one_view(voxel_values, pixel_indices, single_view_params, n_p_centers,
                                                projector_params):
        """
        Forward project a set of voxel cylinders to one view.
        Splits the operation into Vertical (Z -> V) and Horizontal (V -> U) steps (mirrors ConeBeamModel).
        n_p_centers: this view's (num_pixels,) concrete integer channel centers (see
        compute_channel_coordinate) -- consumed by the horizontal fan.
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
        sinogram_view = horizontal_projector(rows_data, pixel_indices, single_view_params,
                                             n_p_centers, projector_params)

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

        # 1. Geometry for this pixel column via the shared coordinate stages.  Anchor the slice-0
        #    row m_p_0 (recon_ijk_to_xyz uses recon_shape[2] == num_slices for the z center).
        row_idx, col_idx = jnp.unravel_index(pixel_index, recon_shape[:2])
        x, y, z_0 = MultiAxisParallelModel.recon_ijk_to_xyz(
            row_idx, col_idx, 0, gp.delta_voxel, gp.voxel_row_aspect, gp.voxel_slice_aspect,
            recon_shape, gp.recon_slice_offset, azimuth)
        _, v_0 = MultiAxisParallelModel.geometry_xyz_to_uv(x, y, z_0, elevation)
        m_p_0, _ = MultiAxisParallelModel.detector_uv_to_mn(
            x, v_0, gp.delta_det_channel, gp.delta_det_row, gp.det_channel_offset,
            gp.det_row_offset, num_det_rows, num_det_channels)

        # Slope: detector rows per one-slice step in z.  v = z*cos(el) + y*sin(el) and z steps by
        # delta_voxel_slice, so dm/dk = delta_voxel_slice*cos(el)/delta_det_row.
        slope_k_to_m = (delta_voxel_slice * jnp.cos(elevation)) / gp.delta_det_row

        # W_p_r: projected footprint of one voxel on the detector rows, in row units (max-of-edges;
        # see _vertical_footprint_phys).  No floor is applied: the max-of-edges footprint never
        # collapses to 0 (the in-plane sin(el) edge keeps it positive), so the mass-conserving
        # amplitude below never divides by zero -- parallel/cone likewise do not clamp.
        W_p_r_phys = _vertical_footprint_phys(azimuth, elevation, gp.delta_voxel,
                                              delta_voxel_row, delta_voxel_slice)
        W_p_r = W_p_r_phys / gp.delta_det_row

        # Mass-conserving amplitude: the kernel sums to ~W_p_r rows, so
        # amplitude = (delta_voxel_slice/delta_det_row)/W_p_r makes the total vertical weight
        # = delta_voxel_slice/delta_det_row, INDEPENDENT of elevation (reduces to 1/cos(el) for the
        # z-edge-dominated footprint).  Folded into the voxel VALUES below, mirroring ConeBeamModel.
        scaling = (delta_voxel_slice / gp.delta_det_row) / W_p_r

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

            # Get values (masked), then fold the mass-conserving amplitude into the VALUES -- as
            # ConeBeamModel scales voxel_cylinder by 1/cos(phi) -- so A_row_k stays a pure
            # interpolation weight (no scale in the coefficient, hence no L_p_r intermediary).
            vals = jnp.where(valid_k, voxel_cylinder[jnp.clip(k_indices, 0, num_slices - 1)], 0.0)
            scaled_vals = vals * scaling

            # Accumulator for this batch
            batch_det = jnp.zeros(num_det_rows)

            for m_offset in jnp.arange(start=-gp.psf_radius, stop=gp.psf_radius + 1):
                m = m_center + m_offset
                dist = jnp.abs(m_p - m)

                A_row_k = jnp.clip((W_p_r + 1.0) / 2.0 - dist, 0.0, L_max)  # vertical interpolation weight
                valid_m = (m >= 0) & (m < num_det_rows)
                A_row_k = A_row_k * valid_m  # fold the validity mask in (cf. ConeBeamModel: A_row_k *= valid)

                # Scatter add
                # We want to add (vals * weight) to indices (m)
                # Since m varies per k, we use .at[m].add(...)
                # Only update valid m indices (invalid m will be clipped or masked)

                # We must mask invalid_m to a safe index for the scatter, then add 0
                safe_m = jnp.clip(m, 0, num_det_rows - 1)
                add_val = scaled_vals * A_row_k

                batch_det = batch_det.at[safe_m].add(add_val)

            return batch_det, None

        # Map over slice chunks
        det_column_parts, _ = jax.lax.map(project_slice_batch, slice_indices)
        # Sum the contributions from all slice chunks
        det_column = jnp.sum(det_column_parts, axis=0)

        return det_column

    @staticmethod
    def forward_horizontal_fan_pixel_batch_to_one_view(rows_data, pixel_indices, single_view_params,
                                                       n_p_centers, projector_params):
        """
        Maps (pixels, rows) -> (rows, channels) (mirrors horizontal fan in ConeBeamModel and TranslationModel).
        Scatters the vertical strips into the correct horizontal channels.  n_p_centers is this
        view's concrete integer channel centers (see compute_channel_coordinate).
        """
        gp = projector_params.geometry_params
        num_det_channels = projector_params.sinogram_shape[2]
        azimuth = single_view_params[0]

        # Anisotropic in-plane pitches: the row (y) axis uses delta_voxel_row, the column (x)
        # axis uses delta_voxel.
        delta_voxel_row = gp.voxel_row_aspect * gp.delta_voxel

        # The continuous channel coordinate via the ONE shared chain (also behind the centers
        # precompute, so centers and weights can never disagree).
        n_p = MultiAxisParallelModel.compute_channel_coordinate(pixel_indices, single_view_params,
                                                                projector_params)

        # In-plane footprint of the voxel on the channels: the larger of the two in-plane pitches
        # projected onto the u-axis (matches ParallelBeamModel.compute_proj_data).  Reduces to
        # delta_voxel * max(|cos|, |sin|) when voxel_row_aspect == 1.
        footprint_xy = jnp.maximum(jnp.abs(jnp.cos(azimuth)) * gp.delta_voxel,
                                   jnp.abs(jnp.sin(azimuth)) * delta_voxel_row)
        W_p_c = footprint_xy / gp.delta_det_channel

        # The shared trapezoid-rule fan (see horizontal_fan_project in projectors.py, which
        # owns the channel-major layout rationale and the sort_by_channel branch -- decided
        # by MultiAxisParallelModel._select_tile_policy; trace-time since projector_params is
        # static).  The weight scale -- in-plane voxel cross-section area over the footprint
        # length -- is a per-VIEW scalar here (the footprint depends only on the azimuth).
        weight_scale = (gp.delta_voxel * delta_voxel_row) / footprint_xy
        sinogram_view_T = horizontal_fan_project(n_p, n_p_centers, W_p_c, weight_scale,
                                                 rows_data, num_det_channels, gp.psf_radius,
                                                 use_sorted=projector_params.sort_by_channel)
        return sinogram_view_T.T

    # =========================================================================
    # Split Back Projectors
    # =========================================================================

    @staticmethod
    @partial(jax.jit, static_argnames=['projector_params', 'slice_band_size'])
    def back_project_one_view_to_pixel_batch(sinogram_view, pixel_indices, single_view_params, n_p_centers,
                                             projector_params, coeff_power=1, slice_band_size=None):
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
            sinogram_view, pixel_indices, single_view_params, n_p_centers, projector_params, coeff_power)

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
    def back_horizontal_fan_one_view_to_pixel_batch(sinogram_view, pixel_indices, single_view_params,
                                                    n_p_centers, projector_params, coeff_power=1):
        gp = projector_params.geometry_params
        num_det_channels = projector_params.sinogram_shape[2]
        azimuth = single_view_params[0]

        # Anisotropic in-plane pitches (must match the forward horizontal fan for adjointness).
        delta_voxel_row = gp.voxel_row_aspect * gp.delta_voxel

        # The continuous channel coordinate via the ONE shared chain (matches the forward fan
        # and the centers precompute -- adjoint by construction, even at rounding ties).
        n_p = MultiAxisParallelModel.compute_channel_coordinate(pixel_indices, single_view_params,
                                                                projector_params)

        footprint_xy = jnp.maximum(jnp.abs(jnp.cos(azimuth)) * gp.delta_voxel,
                                   jnp.abs(jnp.sin(azimuth)) * delta_voxel_row)
        W_p_c = footprint_xy / gp.delta_det_channel

        # The shared adjoint trapezoid-rule fan (see horizontal_fan_back in projectors.py,
        # which owns the channel-major layout rationale and the back_stacked_gather branch
        # -- deliberately NOT enabled by multiaxis' policy: measured a 1.00x composition
        # no-op behind the vertical fan).  Weight scale mirrors the forward fan.
        sinogram_view_T = sinogram_view.T            # (num_det_channels, num_det_rows)
        weight_scale = (gp.delta_voxel * delta_voxel_row) / footprint_xy
        return horizontal_fan_back(sinogram_view_T, n_p, n_p_centers, W_p_c, weight_scale,
                                   gp.psf_radius, coeff_power=coeff_power,
                                   use_stacked=projector_params.back_stacked_gather)

    # ──────────────────────────────────────────────────────────────────────────
    # Banded vertical fan + per-view banded back kernel.  The shared contract (global-index
    # z anchoring / tiling-invariance, inert padding, single-device band walk in bands of
    # MULTIAXIS_SLICE_BAND_SIZE + the multi-device per-band entry) is documented ONCE at the
    # canonical note above projectors.vertical_fan_band_gather.
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def back_vertical_fan_band_one_pixel(detector_col, pixel_index, single_view_params, projector_params,
                                         g0, num_band_slices, coeff_power=1):
        """Back vertical fan for one pixel, producing a BAND of output slices.

        Deliberately NOT the shared projectors.vertical_fan_band_gather (which cone and
        translation use verbatim): multiaxis' vertical fan is structurally different --
        pure-L interpolation weights with the mass-conserving amplitude applied POST-loop
        on the cylinder (the adjoint of its forward fan folding the amplitude into the
        voxel VALUES), not an L / cos_alpha per-tap weight.

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

        # Geometry via the shared coordinate stages (must match the forward vertical fan).  z is
        # anchored on the REAL slice count (recon_ijk_to_xyz uses recon_shape[2]); slice-0 -> row.
        row_idx, col_idx = jnp.unravel_index(pixel_index, recon_shape[:2])
        x, y, z_0 = MultiAxisParallelModel.recon_ijk_to_xyz(
            row_idx, col_idx, 0, gp.delta_voxel, gp.voxel_row_aspect, gp.voxel_slice_aspect,
            recon_shape, gp.recon_slice_offset, azimuth)
        _, v_0 = MultiAxisParallelModel.geometry_xyz_to_uv(x, y, z_0, elevation)
        m_p_0, _ = MultiAxisParallelModel.detector_uv_to_mn(
            x, v_0, gp.delta_det_channel, gp.delta_det_row, gp.det_channel_offset,
            gp.det_row_offset, num_det_rows, num_det_channels)

        slope_k_to_m = (delta_voxel_slice * jnp.cos(elevation)) / gp.delta_det_row

        # Footprint and amplitude -- must match the forward vertical fan exactly for adjointness
        # (max-of-edges footprint, no floor, mass-conserving amplitude on the output cylinder).
        W_p_r_phys = _vertical_footprint_phys(azimuth, elevation, gp.delta_voxel,
                                              delta_voxel_row, delta_voxel_slice)
        W_p_r = W_p_r_phys / gp.delta_det_row
        L_max = jnp.minimum(1.0, W_p_r)
        scaling = (delta_voxel_slice / gp.delta_det_row) / W_p_r

        slice_indices = g0 + jnp.arange(num_band_slices)        # GLOBAL band indices
        m_p_k = m_p_0 + slice_indices * slope_k_to_m
        m_center = jnp.round(m_p_k).astype(int)

        new_cylinder = jnp.zeros(num_band_slices)
        for m_offset in jnp.arange(start=-gp.psf_radius, stop=gp.psf_radius + 1):
            m_idx = m_center + m_offset
            dist = jnp.abs(m_p_k - m_idx)
            A_row_k = jnp.clip((W_p_r + 1.0) / 2.0 - dist, 0.0, L_max)  # vertical interpolation weight
            valid = (m_idx >= 0) & (m_idx < num_det_rows)
            A_row_k = A_row_k * valid  # fold the validity mask in (cf. ConeBeamModel: A_row_k *= valid)
            val = detector_col[jnp.clip(m_idx, 0, num_det_rows - 1)]
            new_cylinder += val * (A_row_k ** coeff_power)

        # Apply the mass-conserving amplitude to the back-projected cylinder -- the adjoint of the
        # forward fan folding `scaling` into the voxel values -- so A_row_k stays a pure
        # interpolation weight (mirrors ConeBeamModel).  scaling is per-view (a scalar), so it
        # factors out of the row sum; raised to coeff_power for the Hessian-diagonal path.
        new_cylinder = new_cylinder * (scaling ** coeff_power)

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
    def back_project_one_view_to_band(sinogram_view, pixel_indices, single_view_params, n_p_centers,
                                      projector_params, g0, num_band_slices, coeff_power=1):
        """Banded back projection of one view onto GLOBAL recon slices [g0, g0+L).

        Horizontal fan once (full detector) -> banded vertical fan.  Returns
        (num_pixels, num_band_slices).  ``g0`` is traced; ``num_band_slices`` (= L) is static.
        ``n_p_centers`` is this view's concrete integer channel centers (see compute_channel_coordinate).
        This is the per-band entry the multi-device reduce-scatter back projector uses on each
        view-owner's shard."""
        rows_data = MultiAxisParallelModel.back_horizontal_fan_one_view_to_pixel_batch(
            sinogram_view, pixel_indices, single_view_params, n_p_centers, projector_params,
            coeff_power=coeff_power)
        return MultiAxisParallelModel.back_vertical_fan_band_pixel_batch(
            rows_data, pixel_indices, single_view_params, projector_params,
            g0, num_band_slices, coeff_power=coeff_power)

    # =========================================================================
    # Direct Recon (Filtered Backprojection)
    # =========================================================================
    # Stacked 2-D FBP: the shared parallel-beam channel ramp per detector row with a uniform
    # pi/num_views per-view weight.  Order-invariant; used as an initializer for recon().

    def direct_recon(self, sinogram, filter_name="ramp", output_sharded=False):
        return self.fbp_recon(sinogram, filter_name, output_sharded=output_sharded)

    def direct_filter(self, sinogram, filter_name="ramp", output_sharded=False):
        """Thin alias for :meth:`fbp_filter` — the FBP filtering step of a direct recon (mirrors
        ParallelBeamModel.direct_filter so ``model.direct_filter`` works uniformly across geometries).
        ``output_sharded`` chooses the output form (numpy by default, view-sharded when True)."""
        return self.fbp_filter(sinogram, filter_name=filter_name, output_sharded=output_sharded)

    def fbp_filter(self, sinogram, filter_name="ramp", output_sharded=False):
        """
        Perform FBP filtering on the given sinogram with a standard 1-D ramp filter along
        the detector channels (the same shared path as ``ParallelBeamModel.fbp_filter``).

        The multiaxis geometry is a set of SIMULTANEOUS fixed measurements (multiple
        sources/detectors) with no acquisition trajectory, so the ``angles`` list order is
        arbitrary and the filter must be ORDER-INVARIANT.  The recon is therefore treated as
        stacked 2-D FBP: a uniform per-view angular weight ``pi / num_views`` (folded in by the
        shared method, the SAME constant as parallel beam) times the standard channel ramp.  This
        reduces EXACTLY to ``ParallelBeamModel.fbp_filter`` in the azimuth-only, equally-spaced,
        zero-elevation limit; elevation is approximated in the filter and corrected by the
        iterative ``recon()`` (this direct recon is an initializer -- see the note on ``fbp_recon``).

        The voxel-size scaling is ``1 / (delta_voxel * delta_voxel_row)`` (NOT
        ``1 / delta_voxel**2``, which silently assumed ``voxel_row_aspect == 1``); with
        anisotropic voxels the row pitch ``delta_voxel_row`` enters correctly.

        Args:
            sinogram (numpy or jax array): The input sinogram with shape (num_views, num_rows, num_channels).
            filter_name (string, optional): Name of the filter to be used. Defaults to "ramp".
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

    def fbp_recon(self, sinogram, filter_name="ramp", output_sharded=False):
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
