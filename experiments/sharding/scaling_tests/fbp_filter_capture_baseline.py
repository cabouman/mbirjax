"""
experiments/sharding/scaling_tests/fbp_filter_capture_baseline.py
──────────────────────────────────────────────────────────────────
Capture a prerelease fbp_filter reference output for correctness comparison.

RUN THIS FROM A PRERELEASE CHECKOUT, not the beta worktree.  It builds a small,
deterministic sinogram (fixed seed) at the correctness size, runs the prerelease
fbp_filter, and writes the output (and timing) to
baselines/fbp_filter_<platform>.yaml.  The beta driver (fbp_filter_scaling.py)
regenerates the identical input from the same seed and compares.

The correctness size is intentionally small so the stored array is a few MB.
The comparison is essentially size-independent: the relevant differences come
from geometry/pixel patterns, not array scale.

The banner prints the resolved mbirjax path — confirm it points at the
PRERELEASE checkout before trusting the captured baseline.

Typical use with a temporary worktree (run once):
    git worktree add ../mbirjax_prerelease prerelease
    cd ../mbirjax_prerelease
    python <path-to-this-script> --out <beta>/experiments/sharding/scaling_tests/baselines
    cd -            # back to beta
    git worktree remove ../mbirjax_prerelease
"""
import argparse
import os

# Import mbirjax before jax (device-setup-first ordering).
import mbirjax

import numpy as np
import jax

from ruamel.yaml import YAML


OP_NAME = "fbp_filter"
CORRECTNESS_SIZE = (64, 64, 64)   # MUST match fbp_filter_scaling.CORRECTNESS_SIZE
CORRECTNESS_SEED = 1234           # MUST match fbp_filter_scaling.CORRECTNESS_SEED


def main():
    parser = argparse.ArgumentParser(description="Capture prerelease fbp_filter baseline")
    parser.add_argument("--out", required=True,
                        help="baselines/ directory in the beta worktree to write into")
    args = parser.parse_args()

    path = os.path.dirname(mbirjax.__file__)
    print("=" * 72)
    print(f"  mbirjax loaded from: {path}")
    print(f"  (confirm this is the PRERELEASE checkout)")
    print("=" * 72)

    try:
        plat = "gpu" if jax.devices("gpu") else "cpu"
    except RuntimeError:
        plat = "cpu"

    n_views, n_rows, n_channels = CORRECTNESS_SIZE
    angles = np.linspace(0, np.pi, n_views, endpoint=False)
    model = mbirjax.ParallelBeamModel((n_views, n_rows, n_channels), angles)

    rng = np.random.default_rng(CORRECTNESS_SEED)
    sino = rng.random(CORRECTNESS_SIZE, dtype=np.float32)

    out = np.asarray(model.fbp_filter(sino))

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"{OP_NAME}_{plat}.yaml")
    yaml = YAML()
    yaml.default_flow_style = False
    payload = {
        "op": OP_NAME,
        "platform": plat,
        "mbirjax_path": path,
        "size": list(CORRECTNESS_SIZE),
        "seed": CORRECTNESS_SEED,
        "output_shape": list(out.shape),
        "output": out.reshape(-1).tolist(),
    }
    with open(out_path, "w") as f:
        yaml.dump(payload, f)
    print(f"  wrote prerelease baseline: {out_path}")
    print(f"  output shape {out.shape}, {out.size} elements")


if __name__ == "__main__":
    main()
