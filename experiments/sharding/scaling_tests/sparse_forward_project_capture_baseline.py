"""
experiments/sharding/scaling_tests/sparse_forward_project_capture_baseline.py
─────────────────────────────────────────────────────────────────────────────
Capture the prerelease sparse_forward_project reference output for correctness
comparison (the forward analogue of sparse_back_project_capture_baseline.py).

RUN THIS FROM A PRERELEASE CHECKOUT, not the beta worktree.  It builds a small,
deterministic cylinder array (fixed seed) at the correctness size, runs the
prerelease ``sparse_forward_project`` over the full field-of-view indices, and
writes ONE platform-agnostic baseline into beta's baselines/:
  - sparse_forward_project.npy   — the reference sinogram (compact, exact binary)
  - sparse_forward_project.yaml  — metadata only (capture platform, prerelease
                                   path, size, seed, shape, dtype, capture timing)

The input cylinders are ``default_rng(SEED).standard_normal((num_pixels,
num_slices))`` — the SAME first draw the beta forward setup worker uses as the
adjoint check's ``x_cyl`` — so the beta driver compares its ``forward(x_cyl)``
against this prerelease reference with no extra input plumbing.

There is a SINGLE baseline, not one per platform; every beta run (CPU or GPU)
compares against this same reference, so a CPU-vs-GPU divergence shows up as a
real difference.  We capture on CPU prerelease because it is deterministic.

Note: beta's single-device (no-mesh) ``sparse_forward_project`` is the unchanged
prerelease body (extracted into ``_sparse_forward_project_single_device``); the
parallel-beam kernel was later given a channel-major layout (bit-compatible), so
this baseline ALSO validates that layout change against the independent prerelease
reference (expect a near-zero, float-noise difference).

Typical use with a temporary worktree (run once).  PYTHONPATH forces prerelease
onto sys.path so ``import mbirjax`` resolves to prerelease and NOT the editable
install:

    git worktree add ../mbirjax_prerelease prerelease
    cd ../mbirjax_prerelease
    PYTHONPATH="$PWD" python \
      <beta>/experiments/sharding/scaling_tests/sparse_forward_project_capture_baseline.py
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


OP_NAME = "sparse_forward_project"
# MUST match sparse_forward_project_scaling.CORRECTNESS_SIZE / _SEED.  Non-symmetric
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

    # Full field-of-view indices (deterministic for this size) and fixed-seed
    # random cylinders -- the FIRST draw of default_rng(SEED), which is exactly the
    # beta setup worker's adjoint-check x_cyl, so the beta driver can compare its
    # forward(x_cyl) against this reference directly.
    recon_shape = model.get_params('recon_shape')
    # Prerelease has no ``use_ror_mask`` MODEL param (added on a later branch), so
    # read it from gen_full_indices' default (True) rather than get_params --
    # which is exactly what beta passes (beta's use_ror_mask default is also True),
    # so the field-of-view / num_pixels match.
    full_indices = mbirjax.gen_full_indices(recon_shape, use_ror_mask=True)
    num_slices = recon_shape[2]
    num_pixels = len(full_indices)
    rng = np.random.default_rng(CORRECTNESS_SEED)
    cylinders = rng.standard_normal((num_pixels, num_slices), dtype=np.float32)

    # Time the reference run too, so the YAML records prerelease timing context.
    stats, result = sc.time_op(
        lambda: model.sparse_forward_project(cylinders, full_indices),
        warmup=1, trials=3)
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
    print(f"  reference sinogram shape {out.shape}, {out.size} elements, "
          f"dtype {out.dtype}  (captured on {plat})")


if __name__ == "__main__":
    main()
