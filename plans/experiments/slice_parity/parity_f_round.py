"""F-round parity experiments (follow-ups to P1; see slice_parity_plan.md P1 RESULTS):

  F1  — the injection regime: default sequence [0,2,4,6,7] (coarse start) with parity-2
        masks applied at (a) no iterations, (b) only the coarse iterations 0-2
        (granularities 1,4,16), (c) all iterations.  Does parity remove the notes'
        slice-19 bad early step where it actually occurs?
  F2  — block aspect-ratio sweep at EQUAL cost, fine end: granularity index {5,6,7} x
        phases {4,2,1} = 128 sub-updates/iter each.  Is P1's compensated winner the
        knee, or does wider-in-plane x thinner-in-z keep winning?
  F2c — the same sweep at the COARSE end (Greg 2026-07-12): {0,1,2} x {4,2,1} = 4
        sub-updates/iter each — isolates the aspect-ratio effect in the
        overshoot-injection regime, complementing F1.
  F3  — tails: baseline [7] vs compensated [6]x2 at 60 iterations (does the compensated
        lead survive its shallower observed slope?).

All cone (P1's null control cleared parallel).  This round uses a DEEP (300-iteration)
converged reference — the P1 100-iteration reference is too shallow for 60-iteration
tails — and re-runs the P1-overlapping arms against it, so all curves here share one
reference.  Reuses the P1 module's verified machinery (ParityMixin passed its bitwise
self-check in P1).

Run:  python plans/experiments/slice_parity/parity_f_round.py    (constants below)
"""
import importlib.util
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location('p1', os.path.join(_here, 'parity_convergence_ab.py'))
p1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p1)

import mbirjax as mj   # noqa: E402  (after p1 sets matplotlib backend)

# ── Config ────────────────────────────────────────────────────────────────────
SEED = p1.SEED
SHARPNESS = p1.SHARPNESS
DEEP_REFERENCE_ITERATIONS = 300
RESULTS_DIR = os.path.join(_here, 'results')

# (name, partition_sequence, phases_spec, num_iterations)
# phases_spec: int = constant; list = per-iteration (extended by its last entry).
VARIANTS = [
    # F1 — injection regime (default coarse-start sequence)
    ('f1_base',    [0, 2, 4, 6, 7], 1,            20),
    ('f1_pcoarse', [0, 2, 4, 6, 7], [2, 2, 2, 1], 20),
    ('f1_pall',    [0, 2, 4, 6, 7], 2,            20),
    # F2 — fine-end aspect-ratio sweep (equal cost: 128 sub-updates/iter)
    ('f2_g5x4', [5], 4, 20),
    ('f2_g6x2', [6], 2, 20),
    ('f2_g7x1', [7], 1, 20),
    ('f2_g7x2', [7], 2, 20),      # P1's parity-2, vs the deep reference (continuity)
    # F2c — coarse-end aspect-ratio sweep (equal cost: 4 sub-updates/iter)
    ('f2c_g0x4', [0], 4, 20),
    ('f2c_g1x2', [1], 2, 20),
    ('f2c_g2x1', [2], 1, 20),
    # F3 — tails
    ('f3_base', [7], 1, 60),
    ('f3_comp', [6], 2, 60),
]
GROUPS = {'F1': ['f1_base', 'f1_pcoarse', 'f1_pall'],
          'F2': ['f2_g5x4', 'f2_g6x2', 'f2_g7x1', 'f2_g7x2'],
          'F2c': ['f2c_g0x4', 'f2c_g1x2', 'f2c_g2x1'],
          'F3': ['f3_base', 'f3_comp']}


def run_variant_scheduled(model, sinogram, pseq, phases_spec, num_iterations):
    """P1's restart driver, extended with a per-iteration phase schedule (the updater is
    created fresh per recon() call, so setting parity_masks per call takes effect)."""
    num_slices = model.get_params('recon_shape')[2]
    model.set_params(partition_sequence=pseq)
    recons, init_recon = [], None
    for j in range(num_iterations):
        phases_j = (phases_spec if isinstance(phases_spec, int)
                    else phases_spec[min(j, len(phases_spec) - 1)])
        model.parity_masks = p1.phase_masks(num_slices, phases_j)
        np.random.seed(SEED)
        recon, _ = model.recon(sinogram, init_recon=init_recon, first_iteration=j,
                               max_iterations=j + 1, stop_threshold_change_pct=0,
                               print_logs=False)
        recons.append(np.asarray(recon))
        init_recon = recon
    return recons


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model = p1.build_model('cone', cls=mj.ConeBeamModel)
    phantom = mj.gen_cube_phantom(model.get_params('recon_shape'))
    sinogram = np.asarray(model.forward_project(phantom))

    ref_path = os.path.join(RESULTS_DIR, f'converged_cone_sharp{SHARPNESS}_deep.npy')
    if os.path.exists(ref_path):
        reference = np.load(ref_path)
    else:
        print(f'[deep reference] {DEEP_REFERENCE_ITERATIONS} iterations...')
        np.random.seed(SEED)
        reference, _ = model.recon(sinogram, max_iterations=DEEP_REFERENCE_ITERATIONS,
                                   stop_threshold_change_pct=0, print_logs=False)
        reference = np.asarray(reference)
        np.save(ref_path, reference)
    old_ref_path = os.path.join(RESULTS_DIR, f'converged_cone_sharp{SHARPNESS}.npy')
    if os.path.exists(old_ref_path):
        old_ref = np.load(old_ref_path)
        drift = np.linalg.norm(reference - old_ref) / np.linalg.norm(reference)
        print(f'[deep vs 100-iter reference] rel diff {drift:.3e} '
              f'(the floor P1 curves sat on)')
    ref_norm = np.linalg.norm(reference)

    summaries = {}
    for name, pseq, phases_spec, iters in VARIANTS:
        t0 = time.perf_counter()
        recons = run_variant_scheduled(p1.build_model('cone'), sinogram, pseq,
                                       phases_spec, iters)
        wall = time.perf_counter() - t0
        errs = np.stack([p1.per_slice_norms(r - reference) for r in recons])
        vol_log_nrmse = np.log10(np.sqrt((errs ** 2).sum(axis=1)) / ref_norm)
        incs = np.stack([p1.per_slice_norms(recons[j + 1] - recons[j])
                         for j in range(len(recons) - 1)])
        np.savez_compressed(os.path.join(RESULTS_DIR, f'{name}.npz'),
                            errs=errs, incs=incs, vol_log_nrmse=vol_log_nrmse)
        summaries[name] = dict(
            partition_sequence=pseq, phases=phases_spec, iterations=iters,
            wall_s=round(wall, 1),
            vol_log_nrmse=[round(float(v), 4) for v in vol_log_nrmse],
            slice19_log_err=[round(float(np.log10(e[19] + 1e-30)), 3) for e in errs],
        )
        print(f'[{name}] wall {wall:.1f}s  final log10 NRMSE {vol_log_nrmse[-1]:.4f}')

    with open(os.path.join(RESULTS_DIR, 'f_round_summary.json'), 'w') as f:
        json.dump(summaries, f, indent=2)

    # Figures: whole-volume log NRMSE per group; slice-19 for F1.
    for group, names in GROUPS.items():
        fig, ax = plt.subplots(figsize=(6, 4))
        for name in names:
            ax.plot(summaries[name]['vol_log_nrmse'], label=name)
        ax.set_xlabel('iteration'), ax.set_ylabel('log10 NRMSE vs deep reference')
        ax.set_title(f'{group} (cone, sharpness {SHARPNESS})'), ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, f'f_round_{group}.png'), dpi=140)
    fig, ax = plt.subplots(figsize=(6, 4))
    for name in GROUPS['F1']:
        ax.plot(summaries[name]['slice19_log_err'], label=name)
    ax.set_xlabel('iteration'), ax.set_ylabel('log10 slice-19 error')
    ax.set_title('F1: the hotspot slice under the coarse-start sequence')
    ax.legend(fontsize=8), fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, 'f_round_F1_slice19.png'), dpi=140)
    print(f'\n[results + figures in {RESULTS_DIR}]')


if __name__ == '__main__':
    main()
