"""
mbirjax._sharding.placement
────────────────────────────
How an array type is distributed across devices (which device owns which shard),
and the one thing that moves between two such distributions: voxel cylinders.

`Placement` is the unit that replaces the scalar ``main_device`` /
``sinogram_device`` fields: it defines how a recon-like (or sino-like) array is
distributed — a list of devices, the axis its array type is sharded on, and a
1-D mesh over those devices.  A single device is the trivial 1-shard case;
``recon_placement`` and ``sino_placement`` may share the same devices but differ
in axis (slice vs view), which is why the axis is part of the placement.

Under view/slice sharding the **only** array that crosses the recon↔sino
boundary is a batch of voxel cylinders (rows of ``flat_recon``); the sinogram is
written locally on its view-shard and never moves.  The two functions here are
that crossing, as an **adjoint pair**:

  - ``move_cylinders_to_sino``  (forward / all-gather): assemble the full
    cylinders (all slices) for a pixel batch onto each sino device.
  - ``sum_cylinders_to_recon``  (back / reduce-scatter): reduce per-sino-device
    partial cylinders over devices and scatter slice-bands to the recon owners.

Both are loops over the placements' shards built on the one transfer primitive
``move_shard``.  The shard counts come from the placements, so the homogeneous
multi-device case (``N×N``) and the single-device / hybrid case (``1×1``) are the
same code with no mode branch; a ``move_shard`` to the same device is a no-op, so
the single-device path carries no overhead.

The adjointness ``<move_cylinders_to_sino(x), y> == <x, sum_cylinders_to_recon(y)>``
is what keeps forward and back projection adjoints (the property the forward/back
adjoint round-trip test relies on).
"""

import numpy as np
import jax
import jax.numpy as jnp

from .transfer import move_shard
from .thread_execution import assemble_sharded


class Placement:
    """Defines how one array type is distributed across devices: the mapping from
    each device to the contiguous block (shard) of the array it owns, determined
    by a sharded axis and a device list.

    Args:
        devices (sequence of jax.Device): the devices this array type lives on.
            A single device is the trivial (1-shard) placement.
        axis (int): the axis of this array type that is partitioned across the
            devices (may be negative; resolved against an array's rank when a
            sharding is built).  Recon-like → the slice axis (-1); sino-like →
            the view axis (0).
        axis_name (str): the mesh axis name referenced by the PartitionSpecs.
    """

    def __init__(self, devices, axis, axis_name="devices"):
        self.devices = list(devices)
        if len(self.devices) < 1:
            raise ValueError("Placement requires at least one device.")
        self.axis = axis
        self.axis_name = axis_name
        # 1-D mesh over these devices; a single device is a trivial 1-device mesh
        # so the single- and multi-device paths share one representation.
        self.mesh = jax.sharding.Mesh(np.array(self.devices), (axis_name,))

    @property
    def n_devices(self):
        return len(self.devices)

    @property
    def is_trivial(self):
        """True when this placement is a single device (1 shard)."""
        return len(self.devices) == 1

    def shard_ranges(self, size):
        """The half-open axis range each device owns when an axis of length
        ``size`` is split into equal contiguous blocks (one per device).

        Args:
            size (int): the length of the sharded axis to split.  Must be
                divisible by the device count (the sharding contract).

        Returns:
            list of (device, (start, end)): the half-open block owned by each
            device, in device order.
        """
        n = len(self.devices)
        if size % n != 0:
            raise ValueError(
                f"Cannot evenly shard axis of size {size} across {n} devices."
            )
        block = size // n
        return [(self.devices[i], (i * block, (i + 1) * block)) for i in range(n)]

    def shard_structure(self, ndim):
        """The NamedSharding that partitions ``axis`` of an ``ndim``-array here.

        Args:
            ndim (int): the rank of the array to be placed.

        Returns:
            jax.sharding.NamedSharding describing how an ``ndim``-array maps onto
            this placement's devices.  Its two parts:

              - ``.mesh`` — the device mesh (this placement's 1-D mesh over
                ``self.devices``, with axis name ``self.axis_name``).
                ``.mesh.devices`` is the device array.
              - ``.spec`` — a ``PartitionSpec``: the mesh axis name
                ``self.axis_name`` at array position ``self.axis`` and ``None``
                everywhere else, i.e. that one array axis is split across the
                devices and every other axis is replicated.  ``.spec[i]`` is the
                entry for array axis ``i`` (the name or ``None``).

            Hand it to ``jax.device_put`` to distribute an array this way, or to
            ``make_array_from_single_device_arrays`` to assemble per-device
            pieces.  On the resulting array, ``.addressable_shards`` gives one
            entry per shard, each with ``.device`` (where it lives), ``.index``
            (the slice of the global array it covers), and ``.data`` (the local
            piece).
        """
        spec = [None] * ndim
        spec[self.axis % ndim] = self.axis_name
        return jax.sharding.NamedSharding(
            self.mesh, jax.sharding.PartitionSpec(*spec)
        )


def move_cylinders_to_sino(cylinders, sino_placement, dev2dev_safe=True):
    """Gather full cylinders (all slices) for a pixel batch onto each sino device.

    Forward projection's all-gather.  ``cylinders`` is a **slice-sharded**
    ``(num_pixels_batch, num_slices)`` array (each device owns a contiguous slice
    band); each sino device needs the *full* cylinder (all slices) to project its
    views, so the slice bands are assembled onto every sino device.

    The adjoint of :func:`sum_cylinders_to_recon`.

    Assumes the sinogram is sharded on an axis **orthogonal** to the recon slice
    axis (currently by view), which is what makes every sino device need all
    slices.  It does not apply to a slice-sharded sinogram: in parallel beam
    slice ``s`` maps only to detector row ``s``, so a slice-sharded sinogram
    leaves each device owning the recon slices and detector rows for its own band
    — projection is then fully local and no cylinder movement (hence no call
    here) is needed.

    Args:
        cylinders (jax.Array): slice-sharded ``(num_pixels_batch, num_slices)``.
        sino_placement (Placement): the devices that will project (only the
            device list is used here — the sino is view-sharded, but the cylinder
            movement only cares about *which devices* receive the cylinders).
        dev2dev_safe (bool): cached hardware probe; forwarded to ``move_shard``.

    Returns:
        dict {sino_device: jax.Array}: the full ``(num_pixels_batch, num_slices)``
        cylinders resident on each sino device.
    """
    # Recon slice-shards in slice order, so concatenation reassembles correctly.
    recon_shards = sorted(cylinders.addressable_shards, key=lambda s: s.index[-1].start)
    pieces = [s.data for s in recon_shards]   # (batch, local_slices) on each recon device
    full = {}
    for dev in sino_placement.devices:
        parts = [move_shard(p, dev, dev2dev_safe=dev2dev_safe) for p in pieces]
        full[dev] = parts[0] if len(parts) == 1 else jnp.concatenate(parts, axis=-1)
    return full


def sum_cylinders_to_recon(partials, recon_placement, dev2dev_safe=True):
    """Reduce per-sino-device partial cylinders and scatter slice-bands to owners.

    Back projection's reduce-scatter.  ``partials`` holds one full
    ``(num_pixels_batch, num_slices)`` cylinder per sino device (that device's
    view contribution); they are summed over devices and each recon owner's slice
    band is landed on its device, yielding a slice-sharded result.

    The adjoint of :func:`move_cylinders_to_sino`.

    Assumes the sinogram is sharded on an axis **orthogonal** to the recon slice
    axis (currently by view), so each device's partial spans all slices and the
    partials must be summed across devices.  Under a slice-sharded sinogram each
    slice has a single contributor (its own device), so there is nothing to sum
    and back projection is local — this reduce-scatter would not apply (and would
    be incorrect if forced).

    Args:
        partials (dict {device: jax.Array}): per-sino-device cylinders, each
            ``(num_pixels_batch, num_slices)``.
        recon_placement (Placement): the recon (slice) placement to land into.
        dev2dev_safe (bool): cached hardware probe; forwarded to ``move_shard``.

    Returns:
        jax.Array: slice-sharded ``(num_pixels_batch, num_slices)`` on
        ``recon_placement`` (the summed cylinders).
    """
    parts = list(partials.values())
    num_pixels_batch, num_slices = parts[0].shape
    owned = []
    for dev, (s0, s1) in recon_placement.shard_ranges(num_slices):
        # Bring each sino device's slice band for this owner to the owner, sum.
        contribs = [move_shard(p[:, s0:s1], dev, dev2dev_safe=dev2dev_safe)
                    for p in parts]
        total = contribs[0]
        for c in contribs[1:]:
            total = total + c
        owned.append(total)   # (batch, s1 - s0) on dev
    return assemble_sharded(owned, (num_pixels_batch, num_slices),
                            recon_placement.shard_structure(2))
