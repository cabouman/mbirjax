"""
experiments/sharding/scaling_tests/vcd_recon_capture_baseline.py
────────────────────────────────────────────────────────────────
Capture the prerelease end-to-end VCD ``recon`` reference volume for the
correctness comparison done by ``vcd_recon_scaling.py``.

RUN THIS FROM A PRERELEASE CHECKOUT, not the beta worktree.  It builds a small,
deterministic random sinogram at the correctness size and runs the prerelease
``recon`` (full multi-granular VCD) for a FIXED number of iterations, then writes
ONE platform-agnostic baseline into beta's baselines/:
  - vcd_recon.npy   — the reference recon volume (compact, exact binary)
  - vcd_recon.yaml  — metadata only (capture platform, prerelease path, size,
                      seed, iterations, shape, dtype, capture timing)

Determinism is the whole game here: VCD draws on the GLOBAL numpy RNG twice — once
to build the pixel partitions (in ``initialize_recon``) and once per iteration to
shuffle subset order (in ``vcd_partition_iterator``).  We therefore seed the
global RNG ONCE with ``np.random.seed(CORRECTNESS_SEED)`` and also draw the random
sinogram from that same global stream, so the entire draw sequence is reproducible.
The beta setup worker (``vcd_recon_scaling.worker_setup``) repeats the identical
sequence, so beta-vs-prerelease is a clean comparison.

NOT a bit-exact gate.  Beta's single-device path is verbatim-prerelease for the
projectors, and the qGGMRF prior was refactored to a halo-aware form whose no-halo
case is bit-exact with prerelease; but iterative accumulation amplifies any
float-reduce-order difference, so expect a small nonzero diff (a FLOAT-NOISE gate
that still catches real divergence and CPU/GPU drift above that floor).

Stop is forced OFF (``stop_threshold_change_pct=0``) so exactly MAX_ITERATIONS run
regardless of platform — otherwise an early-stop difference would masquerade as a
divergence.

Typical use with a temporary worktree (run once).  PYTHONPATH forces prerelease
onto sys.path so ``import mbirjax`` resolves to prerelease and NOT the editable
install:

    git worktree add ../mbirjax_prerelease prerelease
    cd ../mbirjax_prerelease
    PYTHONPATH="$PWD" python \
      <beta>/experiments/sharding/scaling_tests/vcd_recon_capture_baseline.py
    #   → CONFIRM the banner path contains 'mbirjax_prerelease'
    cd -
    git worktree remove ../mbirjax_prerelease

No CLI arguments: the output always goes to beta's baselines/ (resolved from
scaling_common's location), so the script is reproducible with no flags.
"""
import os

import numpy as np

# scaling_common resolves from this script's directory (beta scaling_tests/), so
# the baseline lands in beta's baselines/ even when run from the prerelease tree.
import scaling_common as sc

# NOTE: mbirjax (and therefore jax) is imported LAZILY inside the functions below, NOT
# at module top, so this module is JAX-FREE TO IMPORT.  vcd_recon_scaling.py imports
# constants + run_reference_recon from here at top level; if merely importing this
# module initialized jax, the scaling *orchestrator* would hold a CUDA context and its
# worker subprocesses would fail with CUDA_ERROR_NOT_PERMITTED (the whole point of the
# isolated-subprocess harness is a JAX-free orchestrator).  Device-setup-first still
# holds: each entry point imports mbirjax before any jax use.


OP_NAME = "vcd_recon"
# MUST match vcd_recon_scaling.CORRECTNESS_SIZE / _SEED / _MAX_ITERATIONS.
# Non-symmetric (distinct views/rows/channels) so shape/axis bugs surface early.
CORRECTNESS_SIZE = (80, 48, 64)
CORRECTNESS_SEED = 1234
CORRECTNESS_MAX_ITERATIONS = 5


def run_reference_recon(size, seed, max_iterations):
    """Deterministic single-device VCD recon of a fixed-seed random sinogram.

    Shared verbatim (by copy) with ``vcd_recon_scaling.worker_setup`` so the
    prerelease capture and the beta check do the identical global-RNG draw
    sequence.  Returns (recon_volume_np, timing_stats).
    """
    import mbirjax  # lazy (device-setup-first): keeps this module JAX-free to import
    n_views, n_rows, n_channels = size
    angles = np.linspace(0, np.pi, n_views, endpoint=False)
    model = mbirjax.ParallelBeamModel(size, angles)
    model.set_params(verbose=0)

    def _recon():
        # Seed the GLOBAL RNG inside the timed call so every warmup/trial does the
        # identical work (partitions + subset shuffles), and draw the sinogram from
        # the same stream so a single seed governs the whole run.
        np.random.seed(seed)
        sino = np.random.rand(*size).astype(np.float32)
        recon, _ = model.recon(sino, weights=None, max_iterations=max_iterations,
                               stop_threshold_change_pct=0.0, print_logs=False)
        return recon

    stats, result = sc.time_op(_recon, warmup=1, trials=2)
    return np.asarray(result), stats


def main():
    import mbirjax  # lazy (device-setup-first): keeps this module JAX-free to import
    path = os.path.dirname(mbirjax.__file__)
    print("=" * 72)
    print(f"  mbirjax loaded from: {path}")
    print("  (confirm this is the PRERELEASE checkout — NOT mbirjax/ research")
    print("   or mbirjax_sharding/ beta)")
    print("=" * 72)

    plat, _ = sc.detect_platform()
    out, stats = run_reference_recon(CORRECTNESS_SIZE, CORRECTNESS_SEED,
                                     CORRECTNESS_MAX_ITERATIONS)

    meta = {
        "op": OP_NAME,
        "captured_on_platform": plat,
        "prerelease_mbirjax_path": path,
        "size": list(CORRECTNESS_SIZE),
        "seed": CORRECTNESS_SEED,
        "max_iterations": CORRECTNESS_MAX_ITERATIONS,
        "capture_timing_ms": stats,
    }
    sc.save_baseline(OP_NAME, out, meta)
    print(f"  reference recon shape {out.shape}, {out.size} elements, "
          f"dtype {out.dtype}  (captured on {plat}, {CORRECTNESS_MAX_ITERATIONS} iters)")


if __name__ == "__main__":
    main()
