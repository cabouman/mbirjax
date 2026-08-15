"""Bridge between the Rust dehydration/hydration UI and the local mbirjax library.

The Rust app writes the hyperspectral matrix (points x bands, float64, .npy),
invokes this script, and reads the denoised matrix back.  The script always
imports the mbirjax package of the repository it lives in (two levels up), so
the UI runs whatever is checked out locally — not an installed copy.

Protocol (stdout, one message per line):
  ##VERSION <mbirjax version> <mbirjax module path>
  ##PROGRESS <fraction 0..1> <stage text>
  ##SUBSPACE_DIM <n>                  (denoise mode)
  ##ESTIMATE <num_materials> <subspace_dimension>   (estimate mode)
Anything else the libraries print is passed through and ignored by the app.
"""

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

import matplotlib

matplotlib.use("Agg")  # mbirjax.hsnt imports pyplot; never open windows here

import numpy as np


def emit(tag, *fields):
    print("##" + tag + " " + " ".join(str(f) for f in fields), flush=True)


def load_hsnt():
    """Import mbirjax/hsnt.py directly from the repository.

    Loading the file (rather than `import mbirjax`) skips the package
    __init__, which pulls in jax — a dependency the denoising does not need.
    """
    import importlib.util

    path = os.path.join(REPO_ROOT, "mbirjax", "hsnt.py")
    spec = importlib.util.spec_from_file_location("mbirjax_hsnt", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def mbirjax_version():
    """The local library's version, from the repository's pyproject.toml."""
    import re

    try:
        with open(os.path.join(REPO_ROOT, "pyproject.toml")) as f:
            m = re.search(r'^version\s*=\s*"([^"]+)"', f.read(), re.MULTILINE)
        return m.group(1) if m else "unknown"
    except OSError:
        return "unknown"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["denoise", "estimate"], default="denoise")
    p.add_argument("--input", required=True, help=".npy matrix, points x bands, float64")
    p.add_argument("--output", help=".npy receiving the denoised matrix (denoise mode)")
    p.add_argument("--dataset-type", choices=["attenuation", "transmission"],
                   default="attenuation")
    p.add_argument("--num-materials", type=int, default=2)
    p.add_argument("--safety-factor", type=float, default=2.0)
    p.add_argument("--beta-loss", choices=["frobenius", "kullback-leibler"],
                   default="frobenius")
    p.add_argument("--max-iter", type=int, default=300)
    p.add_argument("--tolerance", type=float, default=1e-10)
    return p.parse_args()


def main():
    args = parse_args()

    hsnt, hsnt_path = load_hsnt()
    emit("VERSION", mbirjax_version(), hsnt_path)

    emit("PROGRESS", 0.05, "Loading data")
    data = np.load(args.input)

    if args.mode == "estimate":
        # Reproduce dehydrate()'s preprocessing (mbirjax/hsnt.py) so the
        # estimation sees the data exactly as the denoising would.
        epsilon = 1e-3
        data = data.astype(np.float64)
        if args.dataset_type == "transmission":
            data[data < epsilon] = epsilon
            data = -np.log(data)
        data[data < 0] = 0
        emit("PROGRESS", 0.2, "Estimating the subspace dimension")
        dim = hsnt._estimate_subspace_dimension(
            data, safety_factor=args.safety_factor, random_state=0, verbose=0)
        num_materials = max(1, int(round(dim / max(args.safety_factor, 1.0))))
        emit("ESTIMATE", num_materials, dim)
        emit("PROGRESS", 1.0, "Done")
        return

    if not args.output:
        sys.exit("--output is required in denoise mode")

    points, bands = data.shape
    subspace_dimension = int(np.ceil(args.safety_factor * args.num_materials))
    emit("SUBSPACE_DIM", subspace_dimension)
    emit("PROGRESS", 0.15,
         f"Running mbirjax hyper_denoise ({points} spectra x {bands} bands, "
         f"subspace dimension {subspace_dimension})")

    denoised = hsnt.hyper_denoise(
        data,
        dataset_type=args.dataset_type,
        num_materials=args.num_materials,
        safety_factor=args.safety_factor,
        beta_loss=args.beta_loss,
        max_iter=args.max_iter,
        tolerance=args.tolerance,
        verbose=0,
    )

    emit("PROGRESS", 0.9, "Writing result")
    np.save(args.output, np.asarray(denoised, dtype=np.float32))
    emit("PROGRESS", 1.0, "Done")


if __name__ == "__main__":
    main()
