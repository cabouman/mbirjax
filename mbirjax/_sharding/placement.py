"""
mbirjax._sharding.placement
────────────────────────────
How an array type is distributed across devices: which device owns which shard.

`Placement` is the unit that replaces the scalar ``main_device`` /
``sinogram_device`` fields: it defines how a recon-like (or sino-like) array is
distributed — a list of devices, the axis its array type is sharded on, and a
1-D mesh over those devices.  A single device is the trivial 1-shard case;
``recon_placement`` and ``sino_placement`` may share the same devices but differ
in axis (slice vs view), which is why the axis is part of the placement.

Under view/slice sharding the **only** data that crosses the recon↔sino boundary
is voxel-cylinder slice-bands (the sinogram is written locally on its view-shard
and never moves).  That crossing is the banded adjoint pair in
:mod:`mbirjax._sharding.transfer`, both built on the one transfer primitive
``move_shard``:

  - ``broadcast_band_to_views``  (forward / all-gather): copy a slice-band from
    its slice-owner to every view-owner.
  - ``sum_band_to_owner``        (back / reduce-scatter): sum each view-owner's
    band partials onto the band's slice-owner.

(An earlier full-cylinder pair, ``move_cylinders_to_sino`` /
``sum_cylinders_to_recon``, was superseded by this banded pair, which streams the
slice axis and so holds less transient memory.)
"""

import numpy as np
import jax


class Placement:
    """Defines how one array type is distributed across devices: the mapping from
    each device to the contiguous block (shard) of the array it owns, determined
    by a sharded axis and a device list.

    The placement also answers "what is on the devices?" for its sharded axis:
    when ``real_size`` (the problem-owned axis length, from the model params) is
    given and does not divide the device count, the DEVICE form of the axis is
    the next multiple of the device count (``padded_size``), with the tail
    zero-filled and kept exactly inert by the model (entry zero-fill + masking).
    Problem-owned shapes stay in the model params; the padded device shape lives
    only here.

    Args:
        devices (sequence of jax.Device): the devices this array type lives on.
            A single device is the trivial (1-shard) placement.
        axis (int): the axis of this array type that is partitioned across the
            devices (may be negative; resolved against an array's rank when a
            sharding is built).  Recon-like → the slice axis (-1); sino-like →
            the view axis (0).
        axis_name (str): the mesh axis name referenced by the PartitionSpecs.
        real_size (int or None): the problem-owned length of the sharded axis
            (e.g. num_views for a sino placement).  When given, ``padded_size``
            is the device-form length (the smallest multiple of the device count
            >= real_size); when None, padding is unknown/unsupported and only
            the divisible case is valid.
    """

    def __init__(self, devices, axis, axis_name="devices", real_size=None):
        self.devices = list(devices)
        if len(self.devices) < 1:
            raise ValueError("Placement requires at least one device.")
        self.axis = axis
        self.axis_name = axis_name
        self.real_size = int(real_size) if real_size is not None else None
        if self.real_size is None:
            self.padded_size = None
        else:
            n = len(self.devices)
            self.padded_size = ((self.real_size + n - 1) // n) * n
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

    @property
    def is_padded(self):
        """True when the device form of the sharded axis is longer than the
        problem's real axis (real_size does not divide the device count)."""
        return self.padded_size is not None and self.padded_size > self.real_size

    def padded_shard_ranges(self):
        """``shard_ranges`` over the device-form (padded) axis length, plus each
        shard's count of REAL (problem-owned) entries.

        Returns:
            list of (device, (start, end), n_valid): the half-open global block
            each device owns on the padded axis, and how many of its entries are
            real (the rest, ``end - start - n_valid``, are zero-filled padding at
            the end of the axis).  Requires ``real_size`` to have been given.
        """
        if self.padded_size is None:
            raise ValueError("padded_shard_ranges requires real_size to be set.")
        ranges = self.shard_ranges(self.padded_size)
        return [(dev, (start, end), max(0, min(end, self.real_size) - start))
                for dev, (start, end) in ranges]

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
