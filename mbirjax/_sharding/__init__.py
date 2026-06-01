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

Higher-level code (e.g. TomographyModel.configure_sharding and the projector
phases) imports these via `from mbirjax._sharding import ...`.
"""

from .transfer import is_dev2dev_safe, move_shard
from .thread_execution import run_per_device, assemble_sharded, device_pool

__all__ = [
    "is_dev2dev_safe",
    "move_shard",
    "run_per_device",
    "assemble_sharded",
    "device_pool",
]
