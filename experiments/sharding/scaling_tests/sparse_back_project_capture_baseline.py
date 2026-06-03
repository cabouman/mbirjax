"""
experiments/sharding/scaling_tests/sparse_back_project_capture_baseline.py
──────────────────────────────────────────────────────────────────────────
Capture the prerelease sparse_back_project reference output for correctness
comparison.

RUN THIS FROM A PRERELEASE CHECKOUT, not the beta worktree.  It builds a small,
deterministic sinogram (fixed seed) at the correctness size, runs the prerelease
``sparse_back_project`` over the full field-of-view indices, and writes ONE
platform-agnostic baseline into beta's baselines/:
  - sparse_back_project.npy   — the reference output array (compact, exact binary)
  - sparse_back_project.yaml  — metadata only (capture platform, prerelease path,
                                size, seed, shape, dtype, capture timing)

There is a SINGLE baseline, not one per platform.  Every beta run — CPU or GPU —
compares against this same reference, so a significant CPU-vs-GPU divergence
shows up as a real difference rather than being hidden by per-platform
self-comparison.  The capture platform is recorded in the metadata so the beta
driver can label a cross-platform check.  We capture on CPU prerelease because it
is deterministic, reproducible, and available everywhere.

Note: beta's single-device (no-mesh) ``sparse_back_project`` is the unchanged
prerelease body (extracted verbatim into ``_sparse_back_project_single_device``),
so this baseline also gives the beta single-device path a near-zero correctness
floor; the value of capturing from prerelease is to catch any future drift.

The correctness size is intentionally small so the stored array is modest.  The
comparison is essentially size-independent: the relevant differences come from
geometry/pixel patterns, not array scale.

The banner prints the resolved mbirjax path — confirm it points at the PRERELEASE
checkout before trusting the captured baseline.

Typical use with a temporary worktree (run once).  PYTHONPATH forces prerelease
onto sys.path so `import mbirjax` resolves to prerelease and NOT the editable
install (which points at the research worktree):

    git worktree add ../mbirjax_prerelease prerelease
    cd ../mbirjax_prerelease
    PYTHONPATH="$PWD" python \
      <beta>/experiments/sharding/scaling_tests/sparse_back_project_capture_baseline.py
    #   → CONFIRM the banner path contains 'mbirjax_prerelease'
    cd -
    git worktree remove ../mbirjax_prerelease

No CLI arguments: the output always goes to beta's baselines/ (resolved from
scaling_common's location), so the script is reproducible with no flags.
"""
import os

# Import mbirjax before jax (device-setup-first ordering).
import mbirjax

import numpy as np

# scaling_common resolves from this script's directory (beta scaling_tests/), so
# the baseline lands in beta's baselines/ even when run from the prerelease tree.
import scaling_common as sc


OP_NAME = "sparse_back_project"
# MUST match sparse_back_project_scaling.CORRECTNESS_SIZE / _SEED.  Non-symmetric
# (distinct views/rows/channels) so shape/axis bugs surface early.
CORRECTNESS_SIZE = (80, 48, 64)
CORRECTNESS_SEED = 1234


def main():
    path = os.path.dirname(mbirjax.__file__)
    print("=" * 72)
    print(f"  mbirjax loaded from: {path}")
    print("  (confirm this is the PRERELEASE checkout — NOT mbirjax/ research")
    print("   or mbirjax_sharding/ beta)")
    print("=" * 72)

    plat, _ = sc.detect_platform()

    n_views, n_rows, n_channels = CORRECTNESS_SIZE
    angles = np.linspace(0, np.pi, n_views, endpoint=False)
    model = mbirjax.ParallelBeamModel((n_views, n_rows, n_channels), angles)

    # Full field-of-view indices (deterministic for this size) and a fixed-seed
    # random sinogram.  These must match what the beta setup worker builds.
    recon_shape, use_ror_mask = model.get_params(['recon_shape', 'use_ror_mask'])
    full_indices = mbirjax.gen_full_indices(recon_shape, use_ror_mask=use_ror_mask)
    rng = np.random.default_rng(CORRECTNESS_SEED)
    sino = rng.random(CORRECTNESS_SIZE, dtype=np.float32)

    # Time the reference run too, so the YAML records prerelease timing context.
    stats, result = sc.time_op(
        lambda: model.sparse_back_project(sino, full_indices), warmup=1, trials=3)
    out = np.asarray(result)

    meta = {
        "op": OP_NAME,
        "captured_on_platform": plat,
        "prerelease_mbirjax_path": path,
        "size": list(CORRECTNESS_SIZE),
        "seed": CORRECTNESS_SEED,
        "capture_timing_ms": stats,
    }
    sc.save_baseline(OP_NAME, out, meta)
    print(f"  reference output shape {out.shape}, {out.size} elements, "
          f"dtype {out.dtype}  (captured on {plat})")


if __name__ == "__main__":
    main()
