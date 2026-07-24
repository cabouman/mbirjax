"""R1 analysis: apply the slice_parity_plan.md §R1 decision rule to r1_summary.json.

Reads the summary (and per-arm npz files when present) from RESULTS_DIR — copy them
from gautschi first, e.g.:
    scp buzzard@gautschi.rcac.purdue.edu:~/parity_lilly/r1_summary.json \
        plans/experiments/slice_parity/results/r1/
    scp buzzard@gautschi.rcac.purdue.edu:~/parity_lilly/C*_sharp*.npz \
        plans/experiments/slice_parity/results/r1/

Outputs (stdout): per-arm trajectory table (cropped log10 NRMSE vs iteration AND vs
idealized cost), cost-to-quality at the 2.0x / 1.2x marks, the displacement verdict per
candidate, and per-slice error profiles at selected iterations if npz files are local.

Decision rule (plan §R1): a candidate displaces C0 if it reaches the 1.2x mark at
<= 0.8x C0's idealized cost at BOTH sharpness settings and is never worse at the 2.0x
mark.  Ties -> prefer the simpler schedule.
"""
import json
import os

import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
# (results/ subdir, control arm name) — analyzed if the dir holds a *_summary.json.
CASE_SPECS = [
    ('r1', 'C0_default'), ('r1_z62', 'C0_default'), ('r1_lilly_ds4', 'C0_default'),
    ('r2_ds8', 'D0_default'), ('r2_z62', 'D0_default'),
]
QUALITY_FACTORS = [2.0, 1.2]        # multiples of the control's final cropped NRMSE
DISPLACE_COST_FRAC = 0.8            # candidate must hit 1.2x mark at <= this x control
PROFILE_ITERS = [4, 9, 29]          # per-slice profile snapshots (0-based iteration idx)
MID_ITER = 14                       # 0-based index of the production budget (15 iters)


def cost_to_mark(lognrmse, cum_cost, mark_log):
    """First cumulative cost at which the trajectory reaches mark_log (np.inf if never)."""
    for lv, c in zip(lognrmse, cum_cost):
        if lv <= mark_log:
            return c
    return np.inf


def main():
    import glob
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    for case, control in CASE_SPECS:
        results_dir = os.path.join(base, case)
        summaries = glob.glob(os.path.join(results_dir, '*_summary.json'))
        if summaries:
            print(f'\n################ case {case} ################')
            analyze(results_dir, summaries[0], control)


def analyze(RESULTS_DIR, SUMMARY, CONTROL):
    with open(SUMMARY) as f:
        results = [r for r in json.load(f) if r.get('status') == 'ok']
    if not results:
        raise SystemExit(f'no ok results in {SUMMARY}')
    sharpness_vals = sorted({r['sharpness'] for r in results})
    by_key = {(r['name'], r['sharpness']): r for r in results}
    names = list(dict.fromkeys(r['name'] for r in results))

    # ── Trajectories ──────────────────────────────────────────────────────────
    for s in sharpness_vals:
        print(f'\n=== sharpness {s}: cropped log10 NRMSE (iters 1,2,3,5,10,20,30) '
              f'and [final @ total idealized cost, wall s] ===')
        for name in names:
            r = by_key.get((name, s))
            if r is None:
                print(f'  {name:>14}: MISSING')
                continue
            tr, cc = r['cropped_lognrmse'], r['cum_cost']
            picks = [0, 1, 2, 4, 9, 19, 29]
            vals = ' '.join(f'{tr[i]:7.3f}' for i in picks if i < len(tr))
            print(f'  {name:>14}: {vals}   [{tr[-1]:.4f} @ {cc[-1]:.1f}u, '
                  f'{r.get("wall_s", "?")}s]')

    # ── Production budget: state at MID_ITER+1 iterations ────────────────────
    print(f'\n=== state at {MID_ITER + 1} iterations '
          f'(cropped log10 NRMSE; delta vs {CONTROL}: + = lower error) ===')
    for s in sharpness_vals:
        ctrl_mid = by_key[(CONTROL, s)]['cropped_lognrmse'][MID_ITER]
        cells = []
        for name in names:
            r = by_key.get((name, s))
            if r is None:
                continue
            v = r['cropped_lognrmse'][MID_ITER]
            d = ctrl_mid - v
            cells.append(f'{name}={v:.3f} ({d:+.3f}, {(1 - 10 ** (-d)) * 100:+.0f}%)'
                         if name != CONTROL else f'{name}={v:.3f}')
        print(f'  s{s}: ' + '  '.join(cells))

    # ── Decision rule ─────────────────────────────────────────────────────────
    print('\n=== cost-to-quality (idealized units; marks are relative to '
          f'{CONTROL} final per sharpness) ===')
    costs = {}          # (name, s, factor) -> cost
    for s in sharpness_vals:
        c0 = by_key[(CONTROL, s)]
        c0_final = c0['cropped_lognrmse'][-1]
        for factor in QUALITY_FACTORS:
            mark = c0_final + np.log10(factor)
            row = []
            for name in names:
                r = by_key.get((name, s))
                c = (cost_to_mark(r['cropped_lognrmse'], r['cum_cost'], mark)
                     if r else np.inf)
                costs[(name, s, factor)] = c
                row.append(f'{name}={c:.1f}' if np.isfinite(c) else f'{name}=never')
            print(f'  s{s} {factor}x mark ({mark:.3f} log10): ' + '  '.join(row))

    print(f'\n=== verdicts (displace {CONTROL} if cost(1.2x) <= '
          f'{DISPLACE_COST_FRAC}x {CONTROL} at BOTH sharpness AND never worse at 2.0x) ===')
    for name in names:
        if name == CONTROL:
            continue
        ok_12 = all(costs[(name, s, 1.2)] <= DISPLACE_COST_FRAC * costs[(CONTROL, s, 1.2)]
                    for s in sharpness_vals)
        ok_20 = all(costs[(name, s, 2.0)] <= costs[(CONTROL, s, 2.0)]
                    for s in sharpness_vals)
        detail = '  '.join(
            f's{s}: 1.2x {costs[(name, s, 1.2)]:.1f}/{costs[(CONTROL, s, 1.2)]:.1f}u, '
            f'2.0x {costs[(name, s, 2.0)]:.1f}/{costs[(CONTROL, s, 2.0)]:.1f}u'
            for s in sharpness_vals)
        verdict = 'DISPLACES' if (ok_12 and ok_20) else 'does not displace'
        print(f'  {name:>14}: {verdict}   ({detail})')

    # ── Per-slice profiles (optional, needs npz files) ────────────────────────
    for s in sharpness_vals:
        arrs = {}
        for name in names:
            p = os.path.join(RESULTS_DIR, f'{name}_sharp{s}.npz')
            if os.path.exists(p):
                arrs[name] = np.load(p)['errs']    # (iters, num_slices)
        if not arrs:
            continue
        print(f'\n=== sharpness {s}: per-slice error — argmax slice and '
              f'max/median ratio at iterations {[i + 1 for i in PROFILE_ITERS]} ===')
        for name, errs in arrs.items():
            cells = []
            for i in PROFILE_ITERS:
                if i < errs.shape[0]:
                    prof = errs[i]
                    cells.append(f'it{i + 1}: z{int(np.argmax(prof))} '
                                 f'{prof.max() / np.median(prof):.2f}x')
            print(f'  {name:>14}: ' + '   '.join(cells))


if __name__ == '__main__':
    main()
