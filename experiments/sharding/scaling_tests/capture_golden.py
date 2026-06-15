"""
experiments/sharding/scaling_tests/capture_golden.py
────────────────────────────────────────────────────
Manual launcher to capture (or selectively refresh) the GOLDEN reference that the gate compares
against — the "expected state" + de-facto accept-list (a failure recorded in golden is a known
wart and stays quiet; §10a).  Triggered deliberately, never auto-promoted from a nightly.

    python experiments/sharding/scaling_tests/capture_golden.py

It runs the full default sweep against the CURRENT working tree and writes
``<GOLDEN_DIR>/golden_<plat>.yaml``.  For an INTENTIONAL baseline change or a newly-ported
geometry, set ONLY to a subset of geometry/op names: only those cells are recaptured and MERGED
into the existing golden (the rest are left untouched), and a ``refresh_log`` entry records what
moved and when — the deliberate-change audit trail.

Notes:
- The canonical golden is captured at a KNOWN-GOOD commit.  For now this runs the working tree;
  the operational wrapper (deferred) will check out a named commit in a worktree, clean-install,
  then call this — and push the golden to the mbirjax_metrics repo.
- The representative `.npy` deep-diff array (used by the deferred compare_to_baseline) is NOT
  captured here yet; the golden fingerprint YAML is what the gate needs.
"""
import os

import performance_tracking as pt
import scaling_common as sc


# ── CONFIG (edit here) ────────────────────────────────────────────────────────
GOLDEN_DIR = os.path.join(sc.RESULTS_DIR, "golden")   # later: <mbirjax_metrics>/golden/
ONLY = None        # None -> full capture (overwrite); or e.g. ["cone"] / ["direct_filter"]
                   # / ["cone", "vcd_nonconst"] -> recapture just those cells, merge into golden


def main():
    config = pt.Config()          # the full default (nightly) sweep
    print("=" * 72)
    print("  performance_tracking — GOLDEN capture (current working tree)")
    print(f"  golden dir: {GOLDEN_DIR}")
    print(f"  only:       {ONLY or 'ALL (full capture)'}")
    print("=" * 72)
    path = pt.capture_golden(config, GOLDEN_DIR, only=ONLY)
    print("Done." if path else "Capture produced no result.")


if __name__ == "__main__":
    main()
