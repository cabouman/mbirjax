"""P1: slice-parity alternation convergence A/B (see plans/slice_parity/slice_parity_plan.md).

Compares standard VCD against slice-parity alternation (even/odd and mod-3 phases layered
on the pixel subsets) on the center-slice-noise repro case (cone 128x40x128, cube phantom,
sharpness 3.0, noiseless — `plans/experiments/bugs_and_artifacts/center slice noise/
center_slice.py`), plus a parallel-beam NULL CONTROL (parallel slices are already
data-decoupled, so parity should change ~nothing there).

MASK-BASED implementation (cost-blind by design — convergence is the question here):
an experiment-local subclass overrides ``create_vcd_subset_updater`` with a verbatim copy
of the library closure (tomography_model.py:3316-3514 at branch commit; see FIDELITY
note) that (a) loops the body over a list of per-slice 0/1 phase masks and (b) multiplies
the update direction by the mask immediately after it is formed — BEFORE the line-search
scalars and the delta forward projection, so alpha is exactly optimal for the masked
direction and the error-sinogram invariant holds automatically.  With masks=None the
copied body takes the identical code path, and a startup SELF-CHECK asserts the copy
reproduces the library solver exactly (protects against copy drift).

Per-iteration capture uses the repro's restart idiom (init_recon chained,
first_iteration=j, max_iterations=j+1) with ``np.random.seed(SEED)`` before every call:
partitions and the subset order are then IDENTICAL across calls and across variants (the
subset order is also identical across iterations — a mild departure from production noted
in the plan; applied uniformly to every variant, so comparisons are fair).

Outputs (RESULTS_DIR): per-variant volumes + per-slice error arrays (npz), diagnostic
figures (png), and a summary json.  Decision numbers get copied into
plans/slice_parity/slice_parity_plan.md per the results-don't-survive-handoff rule.

Run:  python plans/experiments/slice_parity/parity_convergence_ab.py    (constants below)
"""
import json
import os
import time

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import mbirjax as mj
from mbirjax.tomography_model import update_recon, update_error_sinogram

# ── Config ────────────────────────────────────────────────────────────────────
SHARPNESS = 3.0                     # the notes' low-beta / weak-prior hard case
NUM_ITERATIONS = 20                 # long enough to see tail behavior (low-z-freq caveat)
REFERENCE_ITERATIONS = 100          # converged x_inf (cached per geometry)
SEED = 1000
NUM_VIEWS, NUM_DET_ROWS, NUM_DET_CHANNELS = 128, 40, 128   # the repro's shapes
SELF_CHECK = True                   # verify the copied updater == library solver first
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
DIAG_SLICES = [3, 10, 13, 16, 19]   # the repro's per-slice error picks (19 = center)

# (variant name, geometry, partition_sequence, num_phases)  — phases=1 means baseline.
VARIANTS = [
    ('cone_baseline',  'cone',     [7], 1),
    ('cone_parity2',   'cone',     [7], 2),
    ('cone_parity3',   'cone',     [7], 3),
    ('cone_comp2',     'cone',     [6], 2),   # Greg's compensated variant: S/2 x 2 phases
    ('par_baseline',   'parallel', [7], 1),
    ('par_parity2',    'parallel', [7], 2),   # null control
]


# ──────────────────────────────────────────────────────────────────────────────
# The parity updater: a verbatim copy of TomographyModel.create_vcd_subset_updater
# (tomography_model.py:3316-3514) with (1) a phase loop and (2) the one mask line.
# FIDELITY: any library change to the updater must be re-copied here; the SELF_CHECK
# below catches drift by comparing against the unmodified solver.
# Experiment scope: qGGMRF prior only (no prox), positivity OFF (asserted) — the
# positivity clip could resurrect masked entries and would need a re-mask (plan doc).
# ──────────────────────────────────────────────────────────────────────────────
class ParityMixin:
    parity_masks = None     # None => baseline; else list of (num_slices,) float32 masks

    def create_vcd_subset_updater(self, fm_hessian, weights, prox_input=None):
        assert prox_input is None, 'parity experiment supports the qGGMRF prior only'
        positivity_flag = self.get_params('positivity_flag')
        assert not positivity_flag, 'positivity would need a re-mask after its clip'
        fm_constant = 1.0 / (self.get_params('sigma_y') ** 2.0)
        qggmrf_nbr_wts, sigma_x, p, q, T = self.get_params(['qggmrf_nbr_wts', 'sigma_x', 'p', 'q', 'T'])
        b = mj.get_b_from_nbr_wts(qggmrf_nbr_wts)
        qggmrf_params = tuple((b, sigma_x, p, q, T))
        recon_shape = self.get_params('recon_shape')
        max_alpha = self.get_params('max_alpha')
        sparse_back_project = self.sparse_back_project
        sparse_forward_project = self.sparse_forward_project
        try:
            const_weights = False
            sinogram_shape = self.get_params('sinogram_shape')
            if tuple(weights.shape) not in (tuple(sinogram_shape), self._sino_device_shape()):
                raise ValueError('weights must be a constant or have the same shape as sinogram.')
        except AttributeError:
            eps = 1e-5
            if np.abs(weights - 1) > eps:
                raise ValueError('Constant weights must have value 1.')
            const_weights = True

        masks = self.parity_masks   # captured at creation (recon() creates the updater per call)

        def vcd_subset_updater(flat_recon, error_sinogram, pixel_indices, staged_halos=None):
            recon_indices = jax.device_put(
                pixel_indices,
                jax.sharding.NamedSharding(self.recon_placement.mesh,
                                           jax.sharding.PartitionSpec()))
            ell1_total = 0.0
            alphas = []
            phase_masks = masks if masks is not None else [None]
            for mask in phase_masks:
                # --- verbatim library body (one phase sub-update) ---
                prior_grad, prior_hess = self._qggmrf_prior_sharded(
                    flat_recon, pixel_indices, qggmrf_params, staged_halos=staged_halos)

                if not const_weights:
                    weighted_error_sinogram = weights * error_sinogram
                else:
                    weighted_error_sinogram = error_sinogram

                forward_grad = - fm_constant * sparse_back_project(weighted_error_sinogram, pixel_indices)
                forward_hess = fm_constant * fm_hessian[recon_indices]
                delta_recon_at_indices = - ((forward_grad + prior_grad) / (forward_hess + prior_hess))

                # === THE ONE PARITY LINE: restrict the update direction to this phase's
                # slices BEFORE the line-search scalars and the delta projection. ===
                if mask is not None:
                    delta_recon_at_indices = delta_recon_at_indices * mask[None, :]

                prior_linear = jnp.sum(prior_grad * delta_recon_at_indices)
                prior_overrelaxation_factor = 1.0
                prior_quadratic_approx = ((1 / prior_overrelaxation_factor) *
                                          jnp.sum(prior_hess * delta_recon_at_indices ** 2))

                jax.block_until_ready((prior_linear, prior_quadratic_approx))
                del forward_grad, prior_grad, forward_hess, prior_hess

                delta_sinogram = sparse_forward_project(delta_recon_at_indices, pixel_indices)
                forward_linear, forward_quadratic = self.get_forward_lin_quad(
                    weighted_error_sinogram, delta_sinogram, weights, fm_constant, const_weights)

                forward_linear = self._replicate_scalar(forward_linear, self.recon_placement)
                forward_quadratic = self._replicate_scalar(forward_quadratic, self.recon_placement)
                alpha_numerator = forward_linear - prior_linear
                alpha_denominator = forward_quadratic + prior_quadratic_approx + jnp.finfo(jnp.float32).eps
                alpha = alpha_numerator / alpha_denominator
                alpha = jnp.clip(alpha, jnp.finfo(jnp.float32).eps, max_alpha)

                delta_recon_at_indices = alpha * delta_recon_at_indices
                flat_recon = update_recon(flat_recon, recon_indices, delta_recon_at_indices)

                if const_weights:
                    weighted_error_sinogram = None
                error_sinogram = update_error_sinogram(
                    error_sinogram, self._replicate_scalar(alpha, self.sino_placement), delta_sinogram)

                ell1_total = ell1_total + jnp.sum(jnp.abs(delta_recon_at_indices))
                alphas.append(alpha)

                if not const_weights:
                    jax.block_until_ready((flat_recon, error_sinogram))
                    weighted_error_sinogram.delete()
                # --- end verbatim body ---

            alpha_for_subset = alphas[0] if len(alphas) == 1 else jnp.mean(jnp.stack(alphas))
            return flat_recon, error_sinogram, ell1_total, alpha_for_subset

        return vcd_subset_updater


class ParityCone(ParityMixin, mj.ConeBeamModel):
    pass


class ParityParallel(ParityMixin, mj.ParallelBeamModel):
    pass


# ──────────────────────────────────────────────────────────────────────────────
def build_model(geometry, cls=None):
    """The repro's model setup (center_slice.py), for either geometry."""
    sinogram_shape = (NUM_VIEWS, NUM_DET_ROWS, NUM_DET_CHANNELS)
    angles = np.linspace(0, 2 * np.pi, NUM_VIEWS, endpoint=False)
    if geometry == 'cone':
        cls = cls or ParityCone
        model = cls(sinogram_shape, angles,
                    source_detector_dist=12 * NUM_DET_CHANNELS,
                    source_iso_dist=4 * NUM_DET_CHANNELS)
        model.set_params(delta_voxel=0.28, det_row_offset=0.4)
    else:
        cls = cls or ParityParallel
        model = cls(sinogram_shape, angles)
    model.set_params(sharpness=SHARPNESS, verbose=0)
    return model


def phase_masks(num_slices, num_phases):
    if num_phases == 1:
        return None
    return [jnp.asarray((np.arange(num_slices) % num_phases == i).astype(np.float32))
            for i in range(num_phases)]


def run_variant(model, sinogram, partition_sequence, num_phases, num_iterations):
    """Restart-per-iteration capture; returns the per-iteration volume list (incl. init)."""
    num_slices = model.get_params('recon_shape')[2]
    model.parity_masks = phase_masks(num_slices, num_phases)
    model.set_params(partition_sequence=partition_sequence)
    recons, init_recon = [], None
    for j in range(num_iterations):
        np.random.seed(SEED)     # identical partitions + subset order across calls/variants
        recon, _ = model.recon(sinogram, init_recon=init_recon, first_iteration=j,
                               max_iterations=j + 1, stop_threshold_change_pct=0,
                               print_logs=False)
        recons.append(np.asarray(recon))
        init_recon = recon
    return recons


def self_check(sinogram):
    """The copied updater with masks=None must reproduce the library solver exactly."""
    lib_model = build_model('cone', cls=mj.ConeBeamModel)
    par_model = build_model('cone')
    par_model.parity_masks = None
    outs = []
    for model in (lib_model, par_model):
        model.set_params(partition_sequence=[7])
        np.random.seed(SEED)
        recon, _ = model.recon(sinogram, max_iterations=2, stop_threshold_change_pct=0,
                               print_logs=False)
        outs.append(np.asarray(recon))
    err = np.max(np.abs(outs[0] - outs[1])) / max(np.max(np.abs(outs[0])), 1e-30)
    print(f'[self-check] copied-updater vs library rel max err: {err:.3g}')
    if err > 1e-6:      # CPU in-process is deterministic; any real drift shows up large
        raise AssertionError('copied updater does not reproduce the library solver')


def per_slice_norms(vol):
    return np.linalg.norm(vol, axis=(0, 1))


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summaries = {}

    for geometry in ('cone', 'parallel'):
        # Shared data + converged reference per geometry (baseline solver — all variants
        # minimize the same objective, so they share one fixed point).
        model = build_model(geometry, cls=(mj.ConeBeamModel if geometry == 'cone'
                                           else mj.ParallelBeamModel))
        phantom = mj.gen_cube_phantom(model.get_params('recon_shape'))
        sinogram = np.asarray(model.forward_project(phantom))
        ref_path = os.path.join(RESULTS_DIR, f'converged_{geometry}_sharp{SHARPNESS}.npy')
        if os.path.exists(ref_path):
            reference = np.load(ref_path)
        else:
            print(f'[{geometry}] computing {REFERENCE_ITERATIONS}-iteration reference...')
            np.random.seed(SEED)
            reference, _ = model.recon(sinogram, max_iterations=REFERENCE_ITERATIONS,
                                       stop_threshold_change_pct=1e-6, print_logs=False)
            reference = np.asarray(reference)
            np.save(ref_path, reference)

        if geometry == 'cone' and SELF_CHECK:
            self_check(sinogram)

        for name, geom, pseq, phases in VARIANTS:
            if geom != geometry:
                continue
            t0 = time.perf_counter()
            recons = run_variant(build_model(geometry), sinogram, pseq, phases,
                                 NUM_ITERATIONS)
            wall = time.perf_counter() - t0

            errs = np.stack([per_slice_norms(r - reference) for r in recons])     # (it, z)
            incs = np.stack([per_slice_norms(recons[j + 1] - recons[j])
                             for j in range(len(recons) - 1)])                    # (it-1, z)
            cosines = {}
            for lag in (1, 2, 3):
                rows = []
                for j in range(len(recons) - 1 - lag):
                    d0 = recons[j + 1] - recons[j]
                    d1 = recons[j + 1 + lag] - recons[j + lag]
                    denom = (np.linalg.norm(d0, axis=(0, 1))
                             * np.linalg.norm(d1, axis=(0, 1)) + 1e-30)
                    rows.append(np.sum(d0 * d1, axis=(0, 1)) / denom)
                if rows:
                    cosines[lag] = np.stack(rows)

            n_subsets = 2 ** pseq[0]
            summaries[name] = dict(
                geometry=geometry, partition_sequence=pseq, num_phases=phases,
                sub_updates_per_iteration=n_subsets * phases, wall_s=round(wall, 1),
                final_nrmse=float(np.linalg.norm(recons[-1] - reference)
                                  / np.linalg.norm(reference)),
                slice19_log_err=[float(np.log10(e[19] + 1e-30)) for e in errs]
                                 if errs.shape[1] > 19 else None,
            )
            np.savez_compressed(os.path.join(RESULTS_DIR, f'{name}.npz'),
                                errs=errs, incs=incs,
                                **{f'cos_lag{k}': v for k, v in cosines.items()},
                                final_recon=recons[-1])
            print(f'[{name}] wall {wall:.1f}s  final NRMSE {summaries[name]["final_nrmse"]:.5f}  '
                  f'sub-updates/iter {n_subsets * phases}')

    # ── Figures: per-slice error curves for the diagnostic slices, per variant ──
    for geometry in ('cone', 'parallel'):
        names = [v[0] for v in VARIANTS if v[1] == geometry]
        fig, axes = plt.subplots(1, len(DIAG_SLICES), figsize=(4 * len(DIAG_SLICES), 3.4),
                                 sharey=True)
        for ax, z in zip(axes, DIAG_SLICES):
            for name in names:
                errs = np.load(os.path.join(RESULTS_DIR, f'{name}.npz'))['errs']
                if z < errs.shape[1]:
                    ax.plot(np.log10(errs[:, z] + 1e-30), label=name)
            ax.set_title(f'slice {z}')
            ax.set_xlabel('iteration')
        axes[0].set_ylabel('log10 ||x_j - x_inf|| (slice)')
        axes[-1].legend(fontsize=7)
        fig.suptitle(f'{geometry}: per-slice error vs iteration (sharpness {SHARPNESS})')
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, f'errors_{geometry}.png'), dpi=140)

        fig2, axes2 = plt.subplots(1, len(names), figsize=(4 * len(names), 3.4), sharey=True)
        axes2 = np.atleast_1d(axes2)
        for ax, name in zip(axes2, names):
            incs = np.load(os.path.join(RESULTS_DIR, f'{name}.npz'))['incs']
            for j in range(incs.shape[0]):
                ax.plot(incs[j], alpha=0.7)
            ax.set_title(name)
            ax.set_xlabel('slice')
        axes2[0].set_ylabel('||increment|| per slice')
        fig2.suptitle(f'{geometry}: increment norms by slice (one curve per iteration)')
        fig2.tight_layout()
        fig2.savefig(os.path.join(RESULTS_DIR, f'increments_{geometry}.png'), dpi=140)

    with open(os.path.join(RESULTS_DIR, 'summary.json'), 'w') as f:
        json.dump(summaries, f, indent=2)
    print(f'\n[results + figures in {RESULTS_DIR}]')
    for name, s in summaries.items():
        print(f'  {name:15s} final NRMSE {s["final_nrmse"]:.5f}  '
              f'({s["sub_updates_per_iteration"]} sub-updates/iter, {s["wall_s"]}s)')


if __name__ == '__main__':
    main()
