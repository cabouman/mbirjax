"""Equivalence gate: the segmented driver must reproduce a continuous vcd_recon run.

Same seed => same partitions and same subset orders => the final recons must agree to
float noise, every segment's permutation replay must verify, and the per-iteration
loss series must match.  Run this before trusting any driver-based curve, and rerun
after driver changes or library updates (plan: "Driver and RNG discipline").

Gate threshold: 1e-4 relative max error -- the project's iterated, projector-touching
tolerance (lessons.md "Float correctness gates").  On CPU within one process the
observed error is typically far smaller; the gate reports the actual value.

Also smokes the schedule path: a per-granularity offset variant must run cleanly with
the expected per-segment sigmas (no continuous counterpart exists for that, so the
check is bookkeeping, not values).

Run:  python equivalence_gate.py   (config constants below; no CLI args)
"""

import sys

import numpy as np

import mbirjax as mj  # mbirjax must be imported before jax (sets XLA env vars)
import jax.numpy as jnp

from segmented_driver import (run_segmented, run_continuous, rel_max_err,
                              scheduled_sigmas)
from phantom import ball_grid_phantom

# ---------------------------------------------------------------- configuration
SIZE = 64             # views = det rows = det channels (small local gate size)
N_ITERATIONS = 6      # covers the coarse warmup [2, 4, 6] plus fine iterations
SEED = 0
GATE_TOL = 1e-4       # iterated projector-touching relative-max-error gate
# Schedule used only to smoke the scheduled path (conservative -> target):
SMOKE_OFFSETS = {2: (-2.0, -6.0), 4: (-1.0, -3.0), 6: (-0.5, -1.5)}
# -------------------------------------------------------------------------------


def build_case(size=SIZE, verbose=0):
    """Small cone-beam case: model + ball-grid phantom + noiseless sinogram.

    Magnification 2 (source_iso = source_det / 2) gives a real cone angle -- the
    geometry class where streaking is most prominent.
    """
    num_views = num_det_rows = num_det_channels = size
    sinogram_shape = (num_views, num_det_rows, num_det_channels)
    angles = jnp.linspace(0, 2 * np.pi, num_views, endpoint=False)
    source_detector_dist = 4.0 * num_det_channels
    model = mj.ConeBeamModel(sinogram_shape, angles,
                             source_detector_dist=source_detector_dist,
                             source_iso_dist=source_detector_dist / 2.0)
    model.set_params(verbose=verbose)
    gt_phantom = ball_grid_phantom(model.get_params('recon_shape'))
    sinogram = model.forward_project(gt_phantom)
    return model, gt_phantom, sinogram


def run_gate(size=SIZE, n_iterations=N_ITERATIONS, seed=SEED, verbose=0):
    """Run the gate; return (passed: bool, report: str)."""
    model, _, sinogram = build_case(size, verbose)
    lines = []

    cont = run_continuous(model, sinogram, max_iterations=n_iterations, seed=seed)
    seg = run_segmented(model, sinogram, max_iterations=n_iterations, seed=seed)

    recon_err = rel_max_err(seg['final_recon'], cont['final_recon'])
    # With no offsets the scheduled sigma equals the target, so seg['fm_rmse'] is on
    # the same ruler as the continuous series.
    fm_err = max(abs(a - b) / max(abs(b), 1e-30)
                 for a, b in zip(seg['fm_rmse'], cont['fm_rmse']))
    alpha_err = max(abs(a - b) / max(abs(b), 1e-30)
                    for a, b in zip(seg['alpha'], cont['alpha']))
    perms_ok = all(seg['perm_verified'])

    lines.append(f'final recon rel max err : {recon_err:.3e}  (gate {GATE_TOL:.0e})')
    lines.append(f'fm_rmse max rel diff    : {fm_err:.3e}')
    lines.append(f'alpha   max rel diff    : {alpha_err:.3e}')
    lines.append(f'permutation replay      : {"all verified" if perms_ok else "FAILED"}')

    passed = (recon_err <= GATE_TOL) and (fm_err <= GATE_TOL) and perms_ok

    # Scheduled-path smoke: must run cleanly, sigmas must follow the offsets, and
    # every segment's permutation must still verify.
    sched = run_segmented(model, sinogram, max_iterations=n_iterations, seed=seed,
                          offsets_by_entry=SMOKE_OFFSETS)
    sig_ok = all(np.isclose((sx, sy),
                            scheduled_sigmas(e, sched['targets'], SMOKE_OFFSETS)).all()
                 for e, sx, sy in zip(sched['entry'], sched['sigma_x'], sched['sigma_y']))
    sched_ok = sig_ok and all(sched['perm_verified'])

    # Phase B identity gate: an all-zeros schedule must be a NO-OP -- bitwise equal
    # to the plain segmented run on CPU (same process, same stream, identical
    # sigmas); on GPU, within the float-noise gate (scatter-add atomics).
    import jax
    zero = run_segmented(model, sinogram, max_iterations=n_iterations, seed=seed,
                         offsets_by_entry={2: (0.0, 0.0), 4: (0.0, 0.0),
                                           6: (0.0, 0.0)})
    zero_err = rel_max_err(zero['final_recon'], seg['final_recon'])
    on_cpu = jax.devices()[0].platform == 'cpu'
    zero_ok = (zero_err == 0.0) if on_cpu else (zero_err <= GATE_TOL)
    lines.append(f'zero-offset schedule     : rel err {zero_err:.3e} '
                 f'({"bitwise required, CPU" if on_cpu else "float gate, GPU"}) '
                 f'{"ok" if zero_ok else "FAILED"}')
    passed = passed and zero_ok
    lines.append(f'scheduled-path smoke    : '
                 f'{"ok" if sched_ok else "FAILED"} '
                 f'(sigmas {"match" if sig_ok else "MISMATCH"})')
    lines.append('  iter entry n_sub  sigma_x    sigma_y    fm_rmse(target ruler)')
    for i in range(n_iterations):
        lines.append(f'  {i:4d} {sched["entry"][i]:5d} {sched["num_subsets"][i]:5d} '
                     f'{sched["sigma_x"][i]:9.4g}  {sched["sigma_y"][i]:9.4g}  '
                     f'{sched["fm_rmse"][i]:9.4g}')
    passed = passed and sched_ok
    return passed, '\n'.join(lines)


if __name__ == '__main__':
    passed, report = run_gate()
    print(report)
    print('EQUIVALENCE GATE:', 'PASS' if passed else 'FAIL')
    sys.exit(0 if passed else 1)
