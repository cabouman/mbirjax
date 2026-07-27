"""Shared run machinery for the schedule study (Phase B onward).

Consolidates what a2_bga.py / a3_fullres.py each carried separately (panel finding:
the hook/save/two-seed machinery was triplicated and had diverged -- the downsampled
two-seed path lacked the v2 metrics that gate Phase B).  Everything Phase B needs in
one place:

  make_hook        -- per-iteration in-stream metrics (v1 + v2 + es/NRMSE + the
                      target-objective terms + preconditioner share), disk snapshots,
                      and per-iteration (x,z) error images for visual evaluation.
  save_run         -- records.npz + final volume + config, one layout for all runs.
  save_run_images  -- the permanent per-run PNG panels (kept even after snapshot
                      volumes are cleaned up -- Greg's visual-evaluation requirement).
  two_seed_curves  -- disk-based, ALL seed pairs, v1 + v2 at the snapshot grid.
  family_offsets   -- the balance-matched schedule families (D / S / J at depth b).

Conventions: 17-iteration Phase B protocol, snapshot grid {0,1,2,3,4,5,9,14,16},
metrics vs a fixed converged reference, z_step for full-resolution cost control.
"""

import glob
import itertools
import json
import os

import numpy as np

import mbirjax as mj  # mbirjax must be imported before jax (sets XLA env vars)
import jax
import jax.numpy as jnp

import metrics

PHASE_B_ITERATIONS = 17
PHASE_B_SNAPSHOTS = (0, 1, 2, 3, 4, 5, 9, 14, 16)
BALANCE_DB_PER_SHARPNESS = 6.02          # 20*log10(2): Δs=+1 ≈ Δdb=+6.02 (plan)

# Jitted helper so the per-iteration weighted reduction is one cached executable
# (an eager reduce on sharded arrays would allocate fresh collective buffers).
@jax.jit
def _weighted_sq_sum(weights, err_sino):
    return jnp.sum(weights * err_sino * err_sino)


def family_offsets(family, b):
    """offsets_by_entry for one schedule variant (phase_b_plan.md 'Variant space').

    family: 'D' (data-side), 'S' (prior-side), 'J' (joint); b: balance decrement per
    granularity level in dB.  Entries 2/4/6 (4/16/64 subsets) carry d = 3/2/1; the
    128-subset entries default to (0, 0) in the driver.
    """
    def off(d):
        if family == 'D':
            return (0.0, -b * d)
        if family == 'S':
            return (-(b / BALANCE_DB_PER_SHARPNESS) * d, 0.0)
        if family == 'J':
            return (-(b / 2.0) / BALANCE_DB_PER_SHARPNESS * d, -(b / 2.0) * d)
        raise ValueError(family)
    return {2: off(3), 4: off(2), 6: off(1)}


def make_hook(model, reference, mask, run_dir, *, targets, weights_device=None,
              z_step=1, snapshot_iterations=PHASE_B_SNAPSHOTS, prior_loss=False,
              image_iterations=(), sample_pixels=512, real_sino_size=None):
    """Phase B per-iteration hook.

    Records (per iteration): v1 S/control; v2 S_low/S_high/Rz + power; interior
    NRMSE vs the reference (REPORTED only -- biased toward smoother variants);
    data_term_target = (1/(2 sigma_y*^2)) sum(w e^2) / N_real (the data half of the
    target objective, from the checkpoint's error sinogram); prior_target =
    qGGMRF loss at target sigma_x per real voxel (prior_loss=True; downsampled only
    -- impractical at full volume scale); precond_prior_share = the prior's share of
    the preconditioner diagonal at the SCHEDULED sigmas on a fixed pixel sample.
    Also writes snapshot volumes and per-iteration (x,z) error images to run_dir.
    """
    snap_dir = os.path.join(run_dir, 'snapshots')
    img_dir = os.path.join(run_dir, 'images')
    os.makedirs(snap_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    sigma_x_t, sigma_y_t = targets
    recon_shape = model.get_params('recon_shape')
    n_real_voxels = float(np.prod([int(v) for v in recon_shape]))
    qggmrf_nbr_wts, p, q, T = model.get_params(['qggmrf_nbr_wts', 'p', 'q', 'T'])
    b_wts = mj.get_b_from_nbr_wts(qggmrf_nbr_wts)
    params_target = (b_wts, float(sigma_x_t), float(p), float(q), float(T))
    # Fixed in-mask pixel sample for the preconditioner-share diagnostic.
    rng = np.random.default_rng(0)
    mask_idx = np.flatnonzero(np.asarray(mask).ravel())
    sample_idx = jnp.asarray(rng.choice(mask_idx, size=min(sample_pixels,
                                                           mask_idx.size),
                                        replace=False))
    interior3 = np.asarray(mask)

    def hook(i, recon_device, ckpt, seg_record):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        vol = np.asarray(model._gather_recon(recon_device))
        if i in snapshot_iterations:
            np.save(os.path.join(snap_dir, f'it_{i:03d}.npy'),
                    vol.astype(np.float32))
        err = vol - reference
        del vol
        sc = metrics.streak_score(err, mask=mask, z_step=z_step)
        freqs, power = metrics.axial_power_spectrum(err, mask=mask, z_step=z_step)
        v2 = metrics.zcoherence_summary(freqs, power)
        smap = metrics.streak_map(err, z_step=z_step).astype(np.float32)

        # Interior NRMSE vs the reference (central 70% of slices; reported only).
        ns = err.shape[2]
        lo = int(round(ns * 0.15))
        core = err[interior3, lo:ns - lo]
        ref_core = reference[interior3, lo:ns - lo]
        nrmse = float(np.sqrt(np.mean(core.astype(np.float64) ** 2))
                      / np.sqrt(np.mean(ref_core.astype(np.float64) ** 2)))

        # Target-objective data term from the fresh checkpoint (device reduction).
        es = ckpt['error_sinogram']
        if weights_device is not None:
            wss = float(_weighted_sq_sum(weights_device, es))
        else:
            wss = float(jnp.sum(es * es))
        n_sino = real_sino_size or es.size
        data_term = wss / (2.0 * sigma_y_t ** 2) / float(n_sino)

        prior_term = float('nan')
        if prior_loss:
            prior_term = float(mj.qggmrf_loss(recon_device, params_target)) \
                / n_real_voxels

        # Preconditioner share at the SCHEDULED sigmas on the fixed sample:
        # prior_hess / (prior_hess + fm_constant_i * fm_hessian), sample means.
        params_i = (b_wts, float(seg_record['sigma_x']), float(p), float(q),
                    float(T))
        flat = recon_device.reshape((-1, recon_device.shape[-1]))
        _, prior_hess = mj.qggmrf_gradient_and_hessian_at_indices(
            flat, (int(recon_shape[0]), int(recon_shape[1]), flat.shape[-1]),
            sample_idx, params_i)
        fm_c = 1.0 / float(seg_record['sigma_y']) ** 2
        ph = float(jnp.mean(prior_hess))
        fh = fm_c * float(jnp.mean(ckpt['fm_hessian'][sample_idx]))
        share = ph / (ph + fh)

        if i in image_iterations:
            xz_err = err[err.shape[0] // 2, :, :].T
            emax = float(np.percentile(np.abs(xz_err), 99.5))
            fig, ax = plt.subplots(figsize=(6.5, 4.2))
            im = ax.imshow(xz_err, vmin=-emax, vmax=emax, cmap='seismic',
                           aspect='equal')
            ax.set_title(f'(x,z) error vs reference, iteration {i}', fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, shrink=0.85)
            fig.tight_layout()
            fig.savefig(os.path.join(img_dir, f'err_xz_it_{i:03d}.png'), dpi=120)
            plt.close(fig)
        del err

        print(f'    it {i:3d}: S_low={v2["S_low"]:.4g} Rz={v2["Rz"]:.1f} '
              f'S={sc["S"]:.4g} ctrl={sc["control"]:.4g} nrmse={nrmse:.5f} '
              f'obj_d={data_term:.6g} share={share:.3f} '
              f'alpha={seg_record["alpha"]:.3f} ({seg_record["wall_s"]:.1f}s)',
              flush=True)
        return dict(S=sc['S'], control=sc['control'], streak_map=smap,
                    S_low=v2['S_low'], S_high=v2['S_high'], Rz=v2['Rz'],
                    power=power.astype(np.float32), nrmse=nrmse,
                    data_term_target=data_term, prior_target=prior_term,
                    precond_prior_share=share)
    return hook


HOOK_SCALARS = ('S', 'control', 'S_low', 'S_high', 'Rz', 'nrmse',
                'data_term_target', 'prior_target', 'precond_prior_share')


def make_crop_hook(model, reference, mask, run_dir, *, crop_rc=None, z_step=1,
                   snapshot_iterations=(), image_iterations=(), label=''):
    """Per-iteration hook for runs on an ENLARGED (laterally padded) grid,
    scored on a central crop so the numbers share a ruler with unpadded runs.

    crop_rc = (r0, c0, rows, cols): the central region to crop the gathered
    volume to before scoring (None = no crop).  reference and mask live on the
    CROP grid.  Snapshots are the CROPPED volumes, so two_seed_curves /
    two_seed_powers work unchanged.  Objective terms are not recorded (the
    padded problem's objective is not comparable to the unpadded one): the
    make_hook keys are filled with NaN so save_run's layout is unchanged.
    """
    snap_dir = os.path.join(run_dir, 'snapshots')
    img_dir = os.path.join(run_dir, 'images')
    os.makedirs(snap_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    interior3 = np.asarray(mask)

    def hook(i, recon_device, ckpt, seg_record):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        vol = np.asarray(model._gather_recon(recon_device))
        if crop_rc is not None:
            r0, c0, nr, nc = crop_rc
            vol = vol[r0:r0 + nr, c0:c0 + nc, :]
        if i in snapshot_iterations:
            np.save(os.path.join(snap_dir, f'it_{i:03d}.npy'),
                    vol.astype(np.float32))
        err = vol - reference
        del vol
        sc = metrics.streak_score(err, mask=mask, z_step=z_step)
        freqs, power = metrics.axial_power_spectrum(err, mask=mask, z_step=z_step)
        v2 = metrics.zcoherence_summary(freqs, power)
        smap = metrics.streak_map(err, z_step=z_step).astype(np.float32)
        ns = err.shape[2]
        lo = int(round(ns * 0.15))
        core = err[interior3, lo:ns - lo]
        ref_core = reference[interior3, lo:ns - lo]
        nrmse = float(np.sqrt(np.mean(core.astype(np.float64) ** 2))
                      / np.sqrt(np.mean(ref_core.astype(np.float64) ** 2)))
        if i in image_iterations:
            xz_err = err[err.shape[0] // 2, :, :].T
            emax = float(np.percentile(np.abs(xz_err), 99.5))
            fig, ax = plt.subplots(figsize=(6.5, 4.2))
            im = ax.imshow(xz_err, vmin=-emax, vmax=emax, cmap='seismic',
                           aspect='equal')
            ax.set_title(f'{label} (x,z) crop error, iteration {i}', fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, shrink=0.85)
            fig.tight_layout()
            fig.savefig(os.path.join(img_dir, f'err_xz_it_{i:03d}.png'), dpi=120)
            plt.close(fig)
        del err
        print(f'    it {i:3d}: S_low={v2["S_low"]:.4g} Rz={v2["Rz"]:.1f} '
              f'S={sc["S"]:.4g} ctrl={sc["control"]:.4g} nrmse={nrmse:.5f} '
              f'alpha={seg_record["alpha"]:.3f} ({seg_record["wall_s"]:.1f}s)',
              flush=True)
        return dict(S=sc['S'], control=sc['control'], streak_map=smap,
                    S_low=v2['S_low'], S_high=v2['S_high'], Rz=v2['Rz'],
                    power=power.astype(np.float32), nrmse=nrmse,
                    data_term_target=float('nan'), prior_target=float('nan'),
                    precond_prior_share=float('nan'))
    return hook


def two_seed_powers(run_dirs, mask, out_path, z_step=1):
    """Per-iteration two-seed P(f_z) vectors (per-bin, complementing
    two_seed_curves' scalars) from the on-disk snapshots of exactly two runs."""
    assert len(run_dirs) == 2
    snaps_a = _snapshots(run_dirs[0])
    snaps_b = _snapshots(run_dirs[1])
    its, powers, freqs = [], [], None
    for i in sorted(set(snaps_a) & set(snaps_b)):
        va, vb = np.load(snaps_a[i]), np.load(snaps_b[i])
        freqs, p2 = metrics.two_seed_spectrum(va, vb, mask=mask, z_step=z_step)
        del va, vb
        its.append(int(i))
        powers.append(p2.astype(np.float32))
    np.savez_compressed(out_path, iterations=np.asarray(its),
                        freqs=freqs.astype(np.float32), powers=np.stack(powers))
    return len(its)


def save_run(run_dir, records, run_config):
    """records.npz + final volume + config.json (snapshots/images written by the
    hook).  One layout for every Phase B run."""
    os.makedirs(run_dir, exist_ok=True)
    series = {k: np.asarray(records[k]) for k in
              ('entry', 'num_subsets', 'sigma_x', 'sigma_y', 'fm_rmse_raw',
               'fm_rmse', 'es_rmse', 'nmae', 'alpha', 'wall_s', 'perm_verified')}
    for key in HOOK_SCALARS:
        series[key] = np.asarray([h[key] for h in records['hook']])
    series['streak_maps'] = np.stack([h['streak_map'] for h in records['hook']])
    series['powers'] = np.stack([h['power'] for h in records['hook']])
    perms = np.empty(len(records['perm']), dtype=object)
    for j, prm in enumerate(records['perm']):
        perms[j] = np.asarray(prm)
    series['perms'] = perms
    seq = records['seq']
    for e in sorted({int(v) for v in seq[:3]} | {int(seq[-1])}):
        series[f'partition_entry{e}'] = records['partitions_host'][e]
    np.savez_compressed(os.path.join(run_dir, 'records.npz'), **series)
    np.save(os.path.join(run_dir, 'final_recon.npy'),
            records['final_recon'].astype(np.float32))
    run_config = dict(run_config)
    run_config.update(targets=[float(v) for v in records['targets']],
                      seq=[int(v) for v in records['seq']],
                      mbirjax_version=getattr(mj, '__version__', 'unknown'))
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(run_config, f, indent=1)


def save_run_images(run_dir, reference, label=''):
    """The permanent per-run visual panel: final reconstruction and error, mid
    axial slice and (x,z) mid-plane.  Written from final_recon.npy; survives
    snapshot cleanup."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    vol = np.load(os.path.join(run_dir, 'final_recon.npy'))
    err = vol - reference
    zc, yc = vol.shape[2] // 2, vol.shape[0] // 2
    vmax = float(np.percentile(vol, 99.9))
    emax = float(np.percentile(np.abs(err), 99.5))
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.6), constrained_layout=True)
    panels = ((axes[0, 0], vol[:, :, zc], 'final, axial mid-slice',
               dict(vmin=0, vmax=vmax, cmap='gray')),
              (axes[0, 1], err[:, :, zc], 'error, axial',
               dict(vmin=-emax, vmax=emax, cmap='seismic')),
              (axes[1, 0], vol[yc, :, :].T, 'final, (x,z) mid-plane',
               dict(vmin=0, vmax=vmax, cmap='gray')),
              (axes[1, 1], err[yc, :, :].T, 'error, (x,z)',
               dict(vmin=-emax, vmax=emax, cmap='seismic')))
    for ax, img, ttl, kw in panels:
        im = ax.imshow(img, **kw, aspect='equal')
        ax.set_title(ttl, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.75)
    fig.suptitle(label or os.path.basename(run_dir), fontsize=11)
    img_dir = os.path.join(run_dir, 'images')
    os.makedirs(img_dir, exist_ok=True)
    fig.savefig(os.path.join(img_dir, 'final_panels.png'), dpi=130)
    plt.close(fig)


def _snapshots(run_dir):
    return {int(os.path.basename(pth)[3:6]): pth for pth in
            sorted(glob.glob(os.path.join(run_dir, 'snapshots', 'it_*.npy')))}


def two_seed_curves(run_dirs, mask, z_step=1):
    """Disk-based two-seed scores over ALL seed pairs (panel: C1 gates on every
    pair), at the common snapshot iterations + final."""
    def both(path_a, path_b):
        va, vb = np.load(path_a), np.load(path_b)
        sc = metrics.two_seed_score(va, vb, mask=mask, z_step=z_step)
        freqs, power = metrics.two_seed_spectrum(va, vb, mask=mask, z_step=z_step)
        del va, vb
        v2 = metrics.zcoherence_summary(freqs, power)
        return dict(S2=float(sc['S']), control2=float(sc['control']),
                    S2_low=float(v2['S_low']), Rz2=float(v2['Rz']))

    out = {'pairs': []}
    for dir_a, dir_b in itertools.combinations(sorted(run_dirs), 2):
        sa, sb = _snapshots(dir_a), _snapshots(dir_b)
        entry = {'runs': [os.path.basename(dir_a), os.path.basename(dir_b)],
                 'iterations': [], 'points': []}
        for i in sorted(set(sa) & set(sb)):
            entry['iterations'].append(int(i))
            entry['points'].append(both(sa[i], sb[i]))
        entry['final'] = both(os.path.join(dir_a, 'final_recon.npy'),
                              os.path.join(dir_b, 'final_recon.npy'))
        out['pairs'].append(entry)
    return out


def delete_snapshots(run_dir):
    """Free the snapshot volumes AFTER two-seed + digest + images are done (the
    permanent artifacts: records.npz, final_recon.npy, images/)."""
    for pth in glob.glob(os.path.join(run_dir, 'snapshots', 'it_*.npy')):
        os.remove(pth)


def run_is_complete(run_dir):
    """Idempotence check: a run is complete when its records and final exist."""
    return (os.path.exists(os.path.join(run_dir, 'records.npz'))
            and os.path.exists(os.path.join(run_dir, 'final_recon.npy')))
