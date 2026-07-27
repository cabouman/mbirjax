"""Beam-hardening on the synthetic testbed: does hardening alone produce the
ray-aligned streak family seen on the padded real scan?

Forward model: two-metal polychromatic Beer-Lambert (driver/spectrum.py) --
slab w_pe 0.15, metals w_pe (0.8, 0.55) with reference values (6.0, 4.5),
checkerboard-assigned (driver/phantom.py materials mode).  Severity dial
s in {0, 0.5, 1}: s = 0 is the exact monochromatic identity (gate).

Cases (17-iteration protocol, seeds 1-2, shp=1.5 / snr=35, ball layer 0.35,
cone 11.3 deg):
  contained_s{0,0.5,1}    -- object fully in view: hardening is the ONLY
                             pathology.  The attribution test.
  truncpad_s{0,0.5,1}     -- truncated slab + padded (1.5x) recon, crop-scored:
                             mirrors the padded real scan.
  contained_s1_rot        -- view angles offset by half a step (seed 1 only):
                             object-anchored (hardening) streaks stay put,
                             view-anchored (aliasing) streaks rotate.

In-job instruments per run: the usual hook metrics vs the reference-energy
ground truth; the final residual sinogram e = y - A x_hat (saved); transfer
curves of e binned by the TRUE per-metal path sinograms and by the MAR
segmentation coordinates (mbirjax.preprocess.mar, num_metal=2 -- the metals
differ at reference energy, so intensity segmentation can separate them);
the signed deposit ledger.

Run on gautschi:  python -u hardening_bh.py
"""

import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'driver'))

import mbirjax as mj  # mbirjax must be imported before jax (sets XLA env vars)
import jax.numpy as jnp
from mbirjax.preprocess import mar

from segmented_driver import run_segmented, compute_targets   # noqa: E402
from phantom import ball_grid_phantom                         # noqa: E402
from noise import add_transmission_noise                      # noqa: E402
import synthetic_hardening as sh                              # noqa: E402
import spectrum as spx                                        # noqa: E402
import metrics                                                # noqa: E402
import run_io                                                 # noqa: E402
from sweep_sharpness_mass import mass_ledger                  # noqa: E402

# ---------------------------------------------------------------- configuration
ITERATIONS = 17
SEEDS = (1, 2)
BALL_LAYER_Z_FRAC = 0.35
SDD_MULT = 2.5
PAD_SCALE = 1.5
SEVERITIES = (0.0, 0.5, 1.0)
W_PE_SLAB = 0.15
W_PE_METALS = (0.8, 0.55)
BALL_VALUES = (6.0, 4.5)
IMAGE_ITS = (0, 2, 5, 14, 16)
N_BINS = 20
OUTPUT_ROOT = '/scratch/gautschi/buzzard/sharpness_schedule/hardening_bh'
SPECTRUM = spx.make_spectrum()
# -------------------------------------------------------------------------------


def make_models(geometry, rotated=False):
    """(generation model, recon model, crop_rc or None) for a geometry."""
    sino_shape = (sh.NUM_VIEWS, sh.SIZE, sh.SIZE)
    step = 2 * np.pi / sh.NUM_VIEWS
    offset = 0.5 * step if rotated else 0.0
    angles = jnp.linspace(0, 2 * np.pi, sh.NUM_VIEWS, endpoint=False) + offset
    sdd = SDD_MULT * sh.SIZE

    def cone():
        m = mj.ConeBeamModel(sino_shape, angles, source_detector_dist=sdd,
                             source_iso_dist=sdd / 2.0)
        m.set_params(verbose=0)
        return m

    if geometry == 'contained':
        m = cone()
        return m, m, None
    gen = cone()
    gen.scale_recon_shape(sh.TRUNC_GRID_SCALE, sh.TRUNC_GRID_SCALE)
    rec = cone()
    rec.scale_recon_shape(PAD_SCALE, PAD_SCALE)
    big = gen.get_params('recon_shape')
    small = (sh.SIZE, sh.SIZE, big[2])
    r0 = (big[0] - small[0]) // 2
    c0 = (big[1] - small[1]) // 2
    return gen, rec, (r0, c0, small[0], small[1])


def build_case(geometry, severity, rotated=False, phantom_kwargs=None):
    """Materials phantom -> polychromatic noisy sinogram + reference-energy GT.

    phantom_kwargs: optional overrides for ball_grid_phantom (e.g. a denser
    lattice via ball_pitch_frac / ball_radius_frac for real-scan parity).
    """
    gen, rec, crop_rc = make_models(geometry, rotated)
    gshape = gen.get_params('recon_shape')
    kwargs = dict(ball_layer_z_frac=BALL_LAYER_Z_FRAC, return_materials=True,
                  num_metal_types=len(W_PE_METALS), ball_value=BALL_VALUES)
    if geometry != 'contained':
        kwargs['slab_xy_frac'] = sh.TRUNC_SLAB_FRAC
    kwargs.update(phantom_kwargs or {})
    vol, slab_map, metal_maps = ball_grid_phantom(gshape, **kwargs)

    t_slab = np.asarray(gen.forward_project(slab_map), dtype=np.float64)
    t_metals = [np.asarray(gen.forward_project(m), dtype=np.float64)
                for m in metal_maps]
    mono = t_slab + sum(t_metals)
    scale = sh.TARGET_MAX_SINO / float(mono.max())
    t_all = [t_slab * scale] + [t * scale for t in t_metals]

    # Gate: mono identity vs the single-volume projection.
    y_mono = np.asarray(sum(t_all), dtype=np.float32)
    y_direct = np.asarray(gen.forward_project(vol), dtype=np.float64) * scale
    assert np.allclose(y_mono, y_direct, rtol=1e-4, atol=1e-5), \
        'materials decomposition does not reproduce the single-volume sinogram'

    w_pe = [W_PE_SLAB] + list(W_PE_METALS)
    y_poly = spx.poly_sinogram(t_all, w_pe, SPECTRUM, severity=severity)
    dr = spx.deficit_report(t_all, w_pe, SPECTRUM, severity=severity) \
        if severity > 0 else dict(rel_deficit_p99=0.0)
    sino_noisy, weights = add_transmission_noise(y_poly, i0=sh.I0,
                                                 noise_seed=sh.NOISE_SEED)

    if crop_rc is None:
        gt = (vol * scale).astype(np.float32)
    else:
        r0, c0, nr, nc = crop_rc
        gt = (vol[r0:r0 + nr, c0:c0 + nc, :] * scale).astype(np.float32)
    t_metals_scaled = [np.asarray(t * scale, dtype=np.float32)
                       for t in t_metals]
    return rec, gt, sino_noisy, weights, crop_rc, t_metals_scaled, dr


def bin_curve(e, c, n_bins=N_BINS):
    pos = c > 1e-6
    if not pos.any():
        return dict(bins=[])
    edges = np.quantile(c[pos], np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-6
    idx = np.digitize(c[pos], edges) - 1
    ep, cp = e[pos], c[pos]
    rows = []
    for b in range(n_bins):
        sel = idx == b
        if sel.any():
            rows.append(dict(coord_mean=float(cp[sel].mean()),
                             mean=float(ep[sel].mean()),
                             std=float(ep[sel].std()), count=int(sel.sum())))
    return dict(zero_ray_mean=float(e[~pos].mean()) if (~pos).any() else None,
                bins=rows)


def main():
    t0 = time.time()
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    summary = {}
    cases = [('contained', s, False) for s in SEVERITIES] + \
            [('truncpad', s, False) for s in SEVERITIES] + \
            [('contained', 1.0, True)]
    for geometry, severity, rotated in cases:
        name = f'{geometry}_s{severity:g}' + ('_rot' if rotated else '')
        model, gt, sino, weights, crop_rc, t_metals, dr = build_case(
            geometry, severity, rotated)
        model.set_params(sharpness=sh.CENTER_S, snr_db=sh.CENTER_DB)
        mask = metrics.interior_mask(gt.shape)
        seeds = (1,) if rotated else SEEDS
        print(f'=== {name}: recon {model.get_params("recon_shape")}, '
              f'rel deficit p99 {dr["rel_deficit_p99"]:.3f} ===', flush=True)
        case_sum = dict(rel_deficit_p99=dr['rel_deficit_p99'], per_seed={})
        for seed in seeds:
            run_dir = os.path.join(OUTPUT_ROOT, name, f'seed{seed}')
            if not run_io.run_is_complete(run_dir):
                if crop_rc is None:
                    targets = compute_targets(model, sino, weights)
                    hook = run_io.make_hook(
                        model, gt, mask, run_dir, targets=targets,
                        weights_device=jnp.asarray(weights), z_step=1,
                        snapshot_iterations=(0, 5, 14, 16), prior_loss=True,
                        image_iterations=IMAGE_ITS,
                        real_sino_size=int(np.prod(sino.shape)))
                else:
                    hook = run_io.make_crop_hook(
                        model, gt, mask, run_dir, crop_rc=crop_rc, z_step=1,
                        snapshot_iterations=(0, 5, 14, 16),
                        image_iterations=IMAGE_ITS, label=name)
                rec = run_segmented(model, sino, weights=weights,
                                    max_iterations=ITERATIONS, seed=seed,
                                    per_iteration_hook=hook)
                full = rec['final_recon']
                if crop_rc is not None:
                    np.save(os.path.join(run_dir, 'final_recon_full.npy'),
                            full.astype(np.float32))
                    r0, c0, nr, nc = crop_rc
                    rec['final_recon'] = full[r0:r0 + nr, c0:c0 + nc, :]
                run_io.save_run(run_dir, rec, dict(
                    experiment='hardening_bh', case=name, geometry=geometry,
                    severity=severity, rotated=rotated, seed=seed,
                    iterations=ITERATIONS, w_pe_metals=list(W_PE_METALS),
                    ball_values=list(BALL_VALUES), w_pe_slab=W_PE_SLAB))

            # Instruments from the finished run.
            final_name = ('final_recon_full.npy' if crop_rc is not None
                          else 'final_recon.npy')
            final = np.load(os.path.join(run_dir, final_name))
            e = sino - np.asarray(model.forward_project(final))
            np.save(os.path.join(run_dir, 'final_residual_sino.npy'),
                    e.astype(np.float32))
            inst = dict(residual_rms=float(np.sqrt(np.mean(e ** 2))))
            for k, tm in enumerate(t_metals):
                inst[f'true_m{k}'] = bin_curve(e, tm)
            try:
                _, mar_metals = mar._est_plastic_metal_sinos_from_recon(
                    final, num_metal=len(W_PE_METALS), ct_model=model)
                for k, ms in enumerate(mar_metals):
                    inst[f'mar_m{k}'] = bin_curve(e, np.asarray(ms))
            except Exception as exc:                       # noqa: BLE001
                inst['mar_error'] = repr(exc)
            crop_final = np.load(os.path.join(run_dir, 'final_recon.npy'))
            inst['ledger'] = mass_ledger(crop_final, gt)
            rec_npz = np.load(os.path.join(run_dir, 'records.npz'),
                              allow_pickle=True)
            inst['S_low_final'] = float(rec_npz['S_low'][-1])
            case_sum['per_seed'][f'seed{seed}'] = inst
            tm0 = inst['true_m0']['bins']
            print(f'  [{name} seed{seed}] res rms={inst["residual_rms"]:.5f} '
                  f'S_low={inst["S_low_final"]:.4g} '
                  f'top-m0 mean={tm0[-1]["mean"] if tm0 else float("nan"):+.5f} '
                  f'ledger total {inst["ledger"]["total_mass_frac"]:+.3f} '
                  f'({(time.time() - t0) / 60:.1f} min)', flush=True)
        summary[name] = case_sum
        with open(os.path.join(OUTPUT_ROOT, 'bh_summary.json'), 'w') as f:
            json.dump(summary, f, indent=1)
    print(f'hardening_bh complete in {(time.time() - t0) / 60:.1f} min',
          flush=True)


if __name__ == '__main__':
    main()
