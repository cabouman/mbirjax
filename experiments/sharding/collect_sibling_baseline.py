#!/usr/bin/env python
"""AD HOC / EPHEMERAL -- before/after baseline for the zeiss / zeiss_tct compute_sino_and_params
output, to verify the Phase 4 sibling migration onto the shared scan_to_sino core.

Like collect_nsi_golden.py but format-aware (the siblings return 4-tuples):
  zeiss     -> (sino, geometry_params, optional_params, zeiss_metadata)
  zeiss_tct -> (sino, translation_params, optional_params, weights)

Capture with the CURRENT (pre-migration) code, then re-run with --ref after migrating; on the same
platform expect a byte-identical match (modulo scan_to_sino's ~1-ULP fusion, which real data showed
exact).  The baseline is NOT kept.

Usage:
  # capture (before migration)
  python collect_sibling_baseline.py --format zeiss     --data_path .../foam512R1N3000_raw_scan.txrm --out .../zeiss_foam_baseline.npz
  python collect_sibling_baseline.py --format zeiss_tct --data_path .../purdue_BGA_xrm              --out .../tct_bga_baseline.npz
  # verify (after migration): swap --out for --ref
"""
import argparse
import time

import numpy as np
import mbirjax.preprocess as mjp


def _fingerprint(a):
    a = np.asarray(a)
    return dict(shape=tuple(a.shape), dtype=str(a.dtype),
                sum=float(np.sum(a, dtype=np.float64)), min=float(np.min(a)), max=float(np.max(a)))


def _run(fmt, data_path, downsample, subsample):
    if fmt == "zeiss":
        sino, geom, opt, extra = mjp.zeiss.compute_sino_and_params(
            data_path, downsample_factor=(downsample, downsample), subsample_view_factor=subsample)
        return np.asarray(sino), dict(geom), dict(opt), ("metadata", dict(extra))
    sino, geom, opt, weights = mjp.zeiss_tct.compute_sino_and_params(data_path)
    return np.asarray(sino), dict(geom), dict(opt), ("weights", _fingerprint(weights))


def _compare_params(name, ref, new):
    diffs = []
    for k in sorted(set(ref) | set(new)):
        if k not in ref:
            diffs.append(f"{name}: '{k}' only in NEW")
        elif k not in new:
            diffs.append(f"{name}: '{k}' only in REF")
        else:
            a, b = ref[k], new[k]
            if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
                a, b = np.asarray(a), np.asarray(b)
                if a.shape != b.shape or not np.allclose(a, b, rtol=1e-6, atol=1e-6):
                    diffs.append(f"{name}['{k}']: array differs")
            elif a != b:
                diffs.append(f"{name}['{k}']: {a!r} != {b!r}")
    return diffs


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--format", required=True, choices=["zeiss", "zeiss_tct"])
    p.add_argument("--data_path", required=True)
    p.add_argument("--downsample", type=int, default=1)
    p.add_argument("--subsample", type=int, default=1)
    p.add_argument("--out")
    p.add_argument("--ref")
    p.add_argument("--rtol", type=float, default=1e-5)
    p.add_argument("--atol", type=float, default=1e-5)
    args = p.parse_args()

    import mbirjax
    print("IMPORTING mbirjax FROM:", mbirjax.__file__)
    t0 = time.time()
    sino, geom, opt, (extra_name, extra) = _run(args.format, args.data_path, args.downsample, args.subsample)
    print(f"compute_sino_and_params wall: {time.time() - t0:.1f}s")
    print("sino fingerprint:", _fingerprint(sino))
    print("geom params:", geom)
    print("optional params:", opt)
    print(f"{extra_name}:", extra)

    if args.ref:
        d = np.load(args.ref, allow_pickle=True)
        ref_sino, ref_geom, ref_opt = d["sino"], d["geom"].item(), d["opt"].item()
        ref_extra = d["extra"].item()
        ok = True
        if ref_sino.shape != sino.shape:
            print(f"FAIL: sino shape {sino.shape} != ref {ref_sino.shape}")
            ok = False
        else:
            close = np.allclose(ref_sino, sino, rtol=args.rtol, atol=args.atol)
            md = float(np.max(np.abs(ref_sino - sino)))
            print(f"sino allclose(rtol={args.rtol},atol={args.atol}): {close}  max abs diff={md:.3e}")
            ok = ok and close
        diffs = _compare_params("geom", ref_geom, geom) + _compare_params("opt", ref_opt, opt)
        if ref_extra != extra:
            diffs.append(f"{extra_name} differ: ref {ref_extra} vs new {extra}")
        if diffs:
            ok = False
            print("DIFFS:")
            for x in diffs:
                print("  -", x)
        else:
            print(f"params + {extra_name} match")
        print("RESULT:", "PASS" if ok else "FAIL")
        raise SystemExit(0 if ok else 1)

    out = args.out or f"{args.format}_baseline.npz"
    np.savez(out, sino=sino, geom=np.array(geom, dtype=object), opt=np.array(opt, dtype=object),
             extra=np.array(extra, dtype=object))
    print(f"saved baseline -> {out}")
