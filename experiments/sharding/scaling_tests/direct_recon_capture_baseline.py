"""
experiments/sharding/scaling_tests/direct_recon_capture_baseline.py
───────────────────────────────────────────────────────────────────
Capture the prerelease ``direct_recon`` (FBP) reference output for correctness
comparison by the F2 scaling driver (``direct_recon_scaling.py``).

RUN THIS FROM A PRERELEASE CHECKOUT, not the beta worktree.  It builds a small,
deterministic sinogram (fixed seed) at the correctness size, runs the prerelease
``direct_recon`` (filter + adjoint back projection) over the full reconstruction
volume, and writes ONE platform-agnostic baseline into beta's baselines/:
  - direct_recon.npy   — the reference recon volume (compact, exact binary)
  - direct_recon.yaml  — metadata only (capture platform, prerelease path, size,
                         seed, shape, dtype, capture timing)

There is a SINGLE baseline, not one per platform.  Every beta run — CPU or GPU —
compares against this same reference, so a significant CPU-vs-GPU divergence
shows up as a real difference rather than being hidden by per-platform
self-comparison.  The capture platform is recorded in the metadata so the beta
driver can label a cross-platform check.  We capture on CPU prerelease because it
is deterministic, reproducible, and available everywhere.

This comparison is a FLOAT-NOISE gate, not a bit-exact one: expect a
small nonzero diff dominated by the F1 filter rewrite, propagated through back
projection (measured ~8e-8 max_abs_diff on CPU, 0% above the 1e-4 threshold).
The value of the baseline is to catch real divergence (and cross-platform CPU/GPU
drift) above that float-noise floor.

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
      <beta>/experiments/sharding/scaling_tests/direct_recon_capture_baseline.py
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


OP_NAME = "direct_recon"
# MUST match direct_recon_scaling.CORRECTNESS_SIZE / _SEED.  Non-symmetric
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

    # A fixed-seed random sinogram.  FBP (filter + adjoint back projection) is
    # linear, so a random sinogram is a valid correctness input.  These must
    # match what the beta setup worker builds.
    rng = np.random.default_rng(CORRECTNESS_SEED)
    sino = rng.random(CORRECTNESS_SIZE, dtype=np.float32)

    # Time the reference run too, so the YAML records prerelease timing context.
    stats, result = sc.time_op(
        lambda: model.direct_recon(sino), warmup=1, trials=3)
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
    print(f"  reference recon shape {out.shape}, {out.size} elements, "
          f"dtype {out.dtype}  (captured on {plat})")


if __name__ == "__main__":
    main()
