"""Case-matched "ball grid" ground truth phantom (BGA-like).

A low-density slab (epoxy/package) holding a 2D lattice of small dense spheres
(solder balls) -- the high-contrast features that drive streaking in the real
`bga_no_hart` case.  Shared instrument: used by the repro, mechanism, and schedule
topics.  Build it on the MODEL's recon grid (model.get_params('recon_shape')) so the
phantom and reconstruction share a grid exactly.
"""

import numpy as np


def ball_grid_phantom(recon_shape, *, slab_xy_frac=0.6, slab_z_frac=0.5,
                      slab_value=1.0, ball_value=6.0, ball_radius_frac=0.035,
                      ball_pitch_frac=0.14, ball_layer_z_frac=0.5,
                      dtype=np.float32, return_materials=False,
                      num_metal_types=1):
    """Build the ball-grid phantom.

    Args:
        recon_shape: (rows, cols, slices) -- the model's recon shape.
        slab_xy_frac: slab full width as a fraction of min(rows, cols).  The default
            0.6 keeps the slab corners inside the inscribed ROR ellipse (corner
            radius 0.6/sqrt(2) ~ 0.42 < 0.5).
        slab_z_frac: slab full height as a fraction of slices.
        slab_value: slab (plastic-like) value.
        ball_value: sphere (solder-like) value -- the high-contrast streak driver.
        ball_radius_frac: sphere radius as a fraction of min(rows, cols).
        ball_pitch_frac: ball lattice pitch as a fraction of min(rows, cols).
        ball_layer_z_frac: the ball layer's height as a fraction of the volume
            (0.5 = centered, the default and the Phase A behavior).  An off-center
            layer (e.g. 0.35) keeps the volume center free of object structure, so
            center-slice indicators are not confounded by the balls.  Keep the
            layer within the slab.
        return_materials: when True, also return per-material maps (for the
            polychromatic beam-hardening forward model): the slab-only volume and
            one volume per metal type, with vol == slab_map + sum(metal_maps)
            exactly.
        num_metal_types: number of metal types among the balls (materials mode).
            Balls are assigned by lattice checkerboard, (i + j) % num_metal_types,
            so every metal is interleaved across the grid.  ball_value may be a
            sequence of per-metal reference-energy values (different metals
            differ at the reference energy too, which also makes them separable
            by intensity segmentation); a scalar applies to every metal.

    Returns:
        float32 volume of shape (rows, cols, slices); with return_materials=True,
        the tuple (volume, slab_map, [metal_map_0, ...]).
    """
    rows, cols, slices = (int(v) for v in recon_shape[:3])
    n = min(rows, cols)
    vol = np.zeros((rows, cols, slices), dtype=dtype)

    # Slab: centered cuboid.
    half_r = 0.5 * slab_xy_frac * rows
    half_c = 0.5 * slab_xy_frac * cols
    half_z = 0.5 * slab_z_frac * slices
    rc, cc, zc = (rows - 1) / 2.0, (cols - 1) / 2.0, (slices - 1) / 2.0
    r0, r1 = int(np.ceil(rc - half_r)), int(np.floor(rc + half_r)) + 1
    c0, c1 = int(np.ceil(cc - half_c)), int(np.floor(cc + half_c)) + 1
    z0, z1 = int(np.ceil(zc - half_z)), int(np.floor(zc + half_z)) + 1
    vol[r0:r1, c0:c1, z0:z1] = slab_value

    # Ball lattice: a single 2D layer (like a BGA), kept off the slab edge by a
    # margin so every sphere is fully embedded.  ball_layer_z_frac = 0.5 places it
    # at mid-height (the original behavior, bit-identical).
    zc = (slices - 1) * ball_layer_z_frac
    radius = ball_radius_frac * n
    pitch = ball_pitch_frac * n
    margin = 2.0 * radius + 2.0

    def centers(center, half_extent):
        k = int(np.floor((half_extent - margin) / pitch))
        return center + pitch * np.arange(-k, k + 1) if k >= 0 else np.array([center])

    row_centers = centers(rc, half_r)
    col_centers = centers(cc, half_c)

    # Stamp each sphere in a local window (cheap: window ~ (2r+3)^3 voxels).
    # Materials mode also records which metal type each sphere belongs to
    # (lattice checkerboard) so per-material maps can be separated afterward.
    metal_id = np.full((rows, cols, slices), -1, dtype=np.int8) \
        if return_materials else None
    values = (list(ball_value) if np.ndim(ball_value) else
              [ball_value] * max(1, num_metal_types))
    assert len(values) >= num_metal_types
    rad_i = int(np.ceil(radius)) + 1
    for ir, br in enumerate(row_centers):
        for ic, bc in enumerate(col_centers):
            rr = np.arange(max(0, int(br) - rad_i), min(rows, int(br) + rad_i + 1))
            cw = np.arange(max(0, int(bc) - rad_i), min(cols, int(bc) + rad_i + 1))
            zw = np.arange(max(0, int(zc) - rad_i), min(slices, int(zc) + rad_i + 1))
            dr = (rr - br)[:, None, None]
            dc = (cw - bc)[None, :, None]
            dz = (zw - zc)[None, None, :]
            sphere = dr * dr + dc * dc + dz * dz <= radius * radius
            k = (ir + ic) % num_metal_types
            window = vol[np.ix_(rr, cw, zw)]
            window[sphere] = values[k]
            vol[np.ix_(rr, cw, zw)] = window
            if metal_id is not None:
                idw = metal_id[np.ix_(rr, cw, zw)]
                idw[sphere] = k
                metal_id[np.ix_(rr, cw, zw)] = idw

    if not return_materials:
        return vol
    metal_maps = []
    for k in range(num_metal_types):
        mk = np.zeros_like(vol)
        sel = metal_id == k
        mk[sel] = vol[sel]
        metal_maps.append(mk)
    slab_map = vol - sum(metal_maps)
    return vol, slab_map, metal_maps
