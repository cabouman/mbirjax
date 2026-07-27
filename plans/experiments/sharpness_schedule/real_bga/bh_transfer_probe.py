"""Beam-hardening transfer-curve probe on the real scan (instrument pre-test).

Question: does the converged residual sinogram e = y - A x_hat concentrate on
metal-heavy rays with a concave profile -- the beam-hardening fingerprint --
once the truncation deficit is out of the way (padded run)?

Instrument: bin the final-iteration residual by the MAR metal sinogram
coordinate m (mbirjax.preprocess.mar._est_plastic_metal_sinos_from_recon:
Otsu segmentation of the reconstruction, metal-masked forward projection --
the same coordinate the MAR H-model uses, so the measured curve is directly
commensurable with that machinery).  The plastic coordinate p serves as the
control: hardening should organize the residual by m, not by p.

Cases: the padded long-pair final (scale 1.502, iteration 59) and, for
contrast, the unpadded long-pair final (whose residual is truncation-
dominated).  Outputs per case: binned curves (mean/median/std/count of e vs
m and vs p), headline scalars, and a figure.

Run on gautschi:  python -u bh_transfer_probe.py
"""

import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'driver'))

import mbirjax as mj  # mbirjax must be imported before jax (sets XLA env vars)
from mbirjax.preprocess import mar
import a2_bga                                                 # noqa: E402

# ---------------------------------------------------------------- configuration
PAD_SCALE = 1.502
CENTER_S, CENTER_DB = 1.5, 35.0
PADDED_FINAL = ('/scratch/gautschi/buzzard/sharpness_schedule/e4_pad_long/'
                'bga_s1.502/seed1/final_recon_full.npy')
UNPADDED_FINAL = ('/scratch/gautschi/buzzard/sharpness_schedule/e1_longtail/'
                  'baseline/seed1/final_recon.npy')
N_BINS = 24
OUTPUT_ROOT = '/scratch/gautschi/buzzard/sharpness_schedule/bh_probe'
# -------------------------------------------------------------------------------


def transfer_curves(name, model, sino, recon, out):
    """Residual vs the MAR plastic/metal sinogram coordinates for one final."""
    t0 = time.time()
    ax = np.asarray(model.forward_project(recon))
    e = sino - ax
    plastic_sino, metal_sinos = mar._est_plastic_metal_sinos_from_recon(
        recon, num_metal=1, ct_model=model)
    p = np.asarray(plastic_sino)
    m = np.asarray(metal_sinos[0])
    print(f'[{name}] residual rms={np.sqrt(np.mean(e ** 2)):.5f}  '
          f'm>0 ray fraction={float(np.mean(m > 0)):.3f}  '
          f'({(time.time() - t0) / 60:.1f} min)', flush=True)

    case = dict(residual_rms=float(np.sqrt(np.mean(e ** 2))))
    for coord_name, c in (('m', m), ('p', p)):
        pos = c > 0
        zero_mean = float(e[~pos].mean()) if (~pos).any() else float('nan')
        edges = np.quantile(c[pos], np.linspace(0, 1, N_BINS + 1))
        edges[-1] += 1e-6
        idx = np.digitize(c[pos], edges) - 1
        ep = e[pos]
        cp = c[pos]
        rows = []
        for b in range(N_BINS):
            sel = idx == b
            if not sel.any():
                continue
            rows.append(dict(coord_mean=float(cp[sel].mean()),
                             mean=float(ep[sel].mean()),
                             median=float(np.median(ep[sel])),
                             std=float(ep[sel].std()),
                             count=int(sel.sum())))
        case[coord_name] = dict(zero_ray_mean=zero_mean,
                                zero_ray_count=int((~pos).sum()),
                                bins=rows)
        # Energy bookkeeping: residual power on coord-positive rays vs share.
        case[coord_name]['pos_energy_frac'] = float(
            (e[pos] ** 2).sum() / (e ** 2).sum())
        case[coord_name]['pos_ray_frac'] = float(np.mean(pos))
    out[name] = case

    # Figure: mean +/- std vs coordinate, m and p side by side.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.3))
    for axp, coord_name, ttl in ((axes[0], 'm', 'metal coordinate m'),
                                 (axes[1], 'p', 'plastic coordinate p (control)')):
        rows = case[coord_name]['bins']
        x = [r['coord_mean'] for r in rows]
        mu = [r['mean'] for r in rows]
        sd = [r['std'] for r in rows]
        axp.axhline(0, color='#888', lw=0.8)
        axp.axhline(case[coord_name]['zero_ray_mean'], color='#888', lw=0.8,
                    ls=':', label='coord=0 rays (floor)')
        axp.errorbar(x, mu, yerr=sd, fmt='o-', ms=3, lw=1.2, color='#1d4ed8',
                     ecolor='#9db7f5', capsize=2)
        axp.set_xlabel(ttl)
        axp.set_ylabel('residual  y \N{MINUS SIGN} A x\N{COMBINING CIRCUMFLEX ACCENT}')
        axp.grid(alpha=0.3)
        axp.legend(fontsize=8)
    fig.suptitle(f'Residual transfer curves \N{EM DASH} {name} '
                 f'(BGA ds3, iteration 59, shp=1.5, snr=35)', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(os.path.join(OUTPUT_ROOT, f'transfer_{name}.png'), dpi=130)
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    out = {}

    # Padded case: rebuild the even-delta padded model.
    model, sino, _ = a2_bga.load_case()
    model.scale_recon_shape(PAD_SCALE, PAD_SCALE)
    model.set_params(sharpness=CENTER_S, snr_db=CENTER_DB, verbose=0)
    recon = np.load(PADDED_FINAL)
    assert tuple(model.get_params('recon_shape')) == recon.shape, \
        (model.get_params('recon_shape'), recon.shape)
    transfer_curves('padded', model, sino, recon, out)

    # Unpadded case for contrast (truncation-dominated residual).
    model2, sino2, _ = a2_bga.load_case()
    model2.set_params(sharpness=CENTER_S, snr_db=CENTER_DB, verbose=0)
    recon2 = np.load(UNPADDED_FINAL)
    assert tuple(model2.get_params('recon_shape')) == recon2.shape
    transfer_curves('unpadded', model2, sino2, recon2, out)

    with open(os.path.join(OUTPUT_ROOT, 'bh_probe.json'), 'w') as f:
        json.dump(out, f, indent=1)
    for name, case in out.items():
        top = case['m']['bins'][-1] if case['m']['bins'] else {}
        print(f'[{name}] m=0 floor {case["m"]["zero_ray_mean"]:+.5f} | '
              f'top-m bin mean {top.get("mean", float("nan")):+.5f} '
              f'(m~{top.get("coord_mean", float("nan")):.3g}) | '
              f'residual energy on m>0 rays '
              f'{case["m"]["pos_energy_frac"] * 100:.1f}% '
              f'(ray share {case["m"]["pos_ray_frac"] * 100:.1f}%)',
              flush=True)
    print('bh transfer probe complete', flush=True)


if __name__ == '__main__':
    main()
