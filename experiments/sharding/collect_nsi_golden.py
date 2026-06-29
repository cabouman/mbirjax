#!/usr/bin/env python
"""
AD HOC / EPHEMERAL -- golden capture + verify for the preprocessing refactor.

Purpose: capture the CURRENT output of NSI ``compute_sino_and_params`` so the refactored
implementation can be verified against it (see preprocessing_pipeline_refactor_plan.md, Phase 0).
The golden is NOT a permanent fixture: once the new implementation is verified, the new
implementation is the gold standard.  Delete this script and the .h5 when done.

It captures the EXACT output of ``compute_sino_and_params`` -- i.e. BEFORE the downstream
``auto_crop_sino_conebeam`` / ``np.maximum`` steps in Lilly_recon.py, which are out of refactor scope.

Run on the cluster, in the mbirjax env.

  # 1) COLLECT (before the refactor): save the golden
  python collect_nsi_golden.py --data_path /depot/bouman/data/Lilly/Autoinjector_HighRes_Horizontal \
      --downsampling 4 --subsample_view_factor 20 --out /scratch/$USER/nsi_golden_ds4_sv20.h5

  # 2) VERIFY (after the refactor): recompute and compare to the saved golden
  python collect_nsi_golden.py --data_path /depot/bouman/data/Lilly/Autoinjector_HighRes_Horizontal \
      --downsampling 4 --subsample_view_factor 20 --ref /scratch/$USER/nsi_golden_ds4_sv20.h5

Capture two goldens to cover both branches of the pipeline:
  * --downsampling 1  -> exercises the NO-downsample branch (downsample_view_data skipped)
  * --downsampling 4  -> exercises the downsample branch
A large --subsample_view_factor keeps the file small/fast; the per-view kernels do not depend on
the view count, so heavy view subsampling is faithful coverage.
"""
import argparse
import os
import time

import numpy as np
import mbirjax.preprocess as mjp


def _fingerprint(sino):
    """Cheap scalar fingerprint for eyeball comparison across runs (no file load needed)."""
    s = np.asarray(sino)
    return dict(shape=tuple(s.shape), dtype=str(s.dtype),
                sum=float(np.sum(s, dtype=np.float64)),
                min=float(np.min(s)), max=float(np.max(s)),
                mean=float(np.mean(s, dtype=np.float64)))


def _compare_params(name, ref, new):
    """Return list of human-readable diffs between two param dicts."""
    diffs = []
    keys = set(ref) | set(new)
    for k in sorted(keys):
        if k not in ref:
            diffs.append(f"{name}: key '{k}' only in NEW")
        elif k not in new:
            diffs.append(f"{name}: key '{k}' only in REF")
        else:
            a, b = ref[k], new[k]
            if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
                a, b = np.asarray(a), np.asarray(b)
                if a.shape != b.shape or not np.allclose(a, b, rtol=1e-6, atol=1e-6):
                    diffs.append(f"{name}['{k}']: array differs (shapes {a.shape} vs {b.shape})")
            elif a != b:
                diffs.append(f"{name}['{k}']: {a!r} != {b!r}")
    return diffs


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Capture/verify NSI compute_sino_and_params golden.")
    p.add_argument("--data_path", required=True, help="NSI dataset directory.")
    p.add_argument("--downsampling", type=int, default=1, help="Detector row/channel downsample factor.")
    p.add_argument("--subsample_view_factor", type=int, default=1, help="View subsample factor.")
    p.add_argument("--out", type=str, default=None, help="COLLECT: path to save the golden .h5.")
    p.add_argument("--ref", type=str, default=None, help="VERIFY: path to an existing golden .h5 to compare against.")
    p.add_argument("--rtol", type=float, default=1e-5)
    p.add_argument("--atol", type=float, default=1e-5)
    args = p.parse_args()

    downsample_rate = [args.downsampling, args.downsampling]

    t0 = time.time()
    sino, cone_beam_params, optional_params = mjp.nsi.compute_sino_and_params(
        args.data_path,
        downsample_factor=downsample_rate,
        subsample_view_factor=args.subsample_view_factor)
    dt = time.time() - t0

    fp = _fingerprint(sino)
    print(f"\ncompute_sino_and_params wall time: {dt:.1f}s")
    print("sino fingerprint:", fp)
    print("cone_beam_params:", cone_beam_params)
    print("optional_params:", optional_params)

    if args.ref is not None:
        # VERIFY mode: load the saved golden and compare.
        ref_sino, ref_cbp, ref_op, _ = mjp.load_preprocessing(args.ref)
        ok = True
        if np.asarray(ref_sino).shape != np.asarray(sino).shape:
            print(f"\nFAIL: sino shape {np.asarray(sino).shape} != golden {np.asarray(ref_sino).shape}")
            ok = False
        else:
            close = np.allclose(np.asarray(ref_sino), np.asarray(sino), rtol=args.rtol, atol=args.atol)
            maxabs = float(np.max(np.abs(np.asarray(ref_sino) - np.asarray(sino))))
            denom = float(np.max(np.abs(np.asarray(ref_sino)))) + 1e-12
            print(f"\nsino allclose(rtol={args.rtol}, atol={args.atol}): {close}  "
                  f"max abs diff={maxabs:.3e}  max rel diff={maxabs/denom:.3e}")
            ok = ok and close
        param_diffs = (_compare_params("cone_beam_params", ref_cbp, cone_beam_params)
                       + _compare_params("optional_params", ref_op, optional_params))
        if param_diffs:
            ok = False
            print("PARAM DIFFS:")
            for d in param_diffs:
                print("  -", d)
        else:
            print("params: match")
        print("\nRESULT:", "PASS" if ok else "FAIL")
        raise SystemExit(0 if ok else 1)

    # COLLECT mode: save the golden.
    out = args.out or f"nsi_golden_ds{args.downsampling}_sv{args.subsample_view_factor}.h5"
    mjp.save_preprocessing(out, sino, cone_beam_params, optional_params)
    print(f"\nsaved golden -> {os.path.abspath(out)}")
