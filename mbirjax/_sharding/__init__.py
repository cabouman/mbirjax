"""
mbirjax._sharding
─────────────────
Low-level, model-agnostic primitives for multi-device sharding.

These are deliberately small and stateless so they can be unit-tested without a
TomographyModel.  Two concerns are kept separate so they compose:

  - transfer.py          : moving array data between devices safely (handles
                           the hardware-dependent device_put corruption issue).
  - thread_execution.py  : fanning a per-device computation out across threads
                           and assembling the per-device results into one
                           sharded array.
  - placement.py         : `Placement` (where an array type lives across
                           devices).  The banded adjoint pair that moves voxel-
                           cylinder slice-bands between recon and sino placements
                           (sum_band_to_owner / broadcast_band_to_views) lives in
                           transfer.py.

Higher-level code (e.g. TomographyModel.configure_sharding and the projector
phases) imports these via `from mbirjax._sharding import ...`.
"""

from .transfer import (is_dev2dev_safe, move_shard, sum_band_to_owner,
                       broadcast_band_to_views)
from .thread_execution import run_per_device, assemble_sharded, device_pool
from .placement import Placement

__all__ = [
    "is_dev2dev_safe",
    "move_shard",
    "sum_band_to_owner",
    "broadcast_band_to_views",
    "run_per_device",
    "assemble_sharded",
    "device_pool",
    "Placement",
]
