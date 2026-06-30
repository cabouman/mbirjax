"""Validation + visual evidence for the MultiAxisParallelModel elevation-projector corrections.

WHAT THIS IS ABOUT
------------------
The vertical (elevation) fan of MultiAxisParallelModel had two separate problems:

  ELEVATION PATHLENGTH  (a normalization BUG).  The old code deposited each voxel with amplitude
      1.0, so a voxel's total projected mass fell off as cos(el).  A line integral (OPL) cannot
      lose mass with viewing angle -- the total projection of a fixed object must be
      elevation-independent.  Fix: a mass-conserving amplitude (= 1/cos(el) in the simple case),
      capped via the footprint floor so it never blows up, even at 90 deg.

  ELEVATION FOOTPRINT  (a SEPARABILITY APPROXIMATION).  The vertical pass is the (t,z) -> v
      projection, i.e. 2-D parallel beam at angle el.  Its footprint should be the same
      max-of-edges amplitude-method footprint ParallelBeamModel already uses.  The old code kept
      only the foreshortened z-edge cos(el); the corrected one also picks up the in-plane sin(el)
      edge.  Below ~45 deg the z-edge dominates so this changes nothing; above 45 deg it matters.

The two are independent: the pathlength problem is a normalization bug (fixable to EXACT); the
footprint problem is a shape approximation we accept for the efficiency of a separable projector.

These are exposed as temporary A/B flags, and three modes are compared throughout (named for the
corrections they apply):
  uncorrected             : correct_elevation_pathlength=False, broaden_elevation_footprint=False
  pathlength              : correct_elevation_pathlength=True,  broaden_elevation_footprint=False
  pathlength + footprint  : correct_elevation_pathlength=True,  broaden_elevation_footprint=True

WHY INVERSE CRIME MATTERS
-------------------------
If you SIMULATE and RECONSTRUCT with the same (buggy) projector, the error cancels and you see
nothing -- that is why this survived.  Every recon test below is run BOTH ways: "inverse crime"
(data made by the same mode) and "crime-free" (data made by an independent oversampled reference).

RUN (on a GPU node with the editable mbirjax env):
  python validate_elevation_fix.py
Prints tables to stdout and writes labeled figures to ./figures/.  Read alongside README.md.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")                       # headless: save PNGs, no display
import matplotlib.pyplot as plt
import jax, jax.numpy as jnp
from mbirjax import MultiAxisParallelModel

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

UNCORRECTED = "uncorrected"
PATHLENGTH = "pathlength"
PATH_FOOT = "pathlength + footprint"
MODES = {
    UNCORRECTED: dict(correct_elevation_pathlength=False, broaden_elevation_footprint=False),
    PATHLENGTH:  dict(correct_elevation_pathlength=True,  broaden_elevation_footprint=False),
    PATH_FOOT:   dict(correct_elevation_pathlength=True,  broaden_elevation_footprint=True),
}
COLOR = {UNCORRECTED: "tab:red", PATHLENGTH: "tab:blue", PATH_FOOT: "tab:green"}


def build(mode, ang, shp, recon_shape, dv=1.0):
    m = MultiAxisParallelModel(shp, ang)
    m.set_elevation_correction(**MODES[mode])
    m.set_params(recon_shape=recon_shape, delta_voxel=dv, delta_det_row=1.0, delta_det_channel=1.0)
    return m


def nrmse(a, b):
    return float(jnp.linalg.norm(a - b) / jnp.linalg.norm(b))


def sphere_volume(n, R):
    g = np.arange(n) - (n - 1) / 2.0
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    return ((X**2 + Y**2 + Z**2) <= R**2).astype(np.float32)


def structured_volume(n, seed=0, nblobs=6):
    rng = np.random.default_rng(seed)
    v = np.zeros((n, n, n), np.float32)
    for _ in range(nblobs):
        c = rng.integers(5, n - 10, size=3); s = rng.integers(4, 8)
        v[c[0]:c[0]+s, c[1]:c[1]+s, c[2]:c[2]+s] += rng.uniform(0.5, 1.5)
    return v


# =====================================================================================
# Figure 1: mass conservation -- the clearest single picture of the pathlength bug.
# =====================================================================================
def fig_mass_conservation():
    NS, R = 61, 20.0
    vol = jnp.asarray(sphere_volume(NS, R))
    true_mass = float(np.asarray(vol).sum())          # voxel volume 1 => OPL mass == voxel count
    els = np.arange(0, 71, 5)
    series = {m: [] for m in MODES}
    for eldeg in els:
        ang = jnp.array([[0.0, np.deg2rad(eldeg)]]); shp = (1, 81, 81)
        for m in MODES:
            tot = float(jnp.sum(build(m, ang, shp, (NS, NS, NS)).forward_project(vol)))
            series[m].append(tot)
    plt.figure(figsize=(7, 4.5))
    plt.axhline(true_mass, color="k", ls="--", lw=1, label="true mass (angle-independent)")
    for m in MODES:
        plt.plot(els, series[m], "o-", color=COLOR[m], label=m)
    plt.plot(els, true_mass * np.cos(np.deg2rad(els)), ":", color="gray", label="true_mass x cos(el)")
    plt.xlabel("elevation (deg)"); plt.ylabel("total projected mass of a fixed sphere")
    plt.title("Elevation pathlength: uncorrected mass decays as cos(el); \n corrected is conserved")
    plt.legend(fontsize=8); plt.tight_layout()
    p = os.path.join(FIGDIR, "fig1_mass_conservation.png"); plt.savefig(p, dpi=130); plt.close()
    print("  wrote", p)
    i60 = list(els).index(60)
    print("  uncorrected mass @60deg = {:.0f} ({:.0f}% low);  pathlength = {:.0f}".format(
        series[UNCORRECTED][i60], 100 * (1 - series[UNCORRECTED][i60] / true_mass), series[PATHLENGTH][i60]))


# =====================================================================================
# Figure 2: forward-projection accuracy vs elevation, against EXACT/independent truth.
# =====================================================================================
def fig_projection_accuracy():
    # (a) sphere vs ANALYTIC truth   (b) structured phantom vs OVERSAMPLED reference
    NS, R = 61, 20.0
    sph = jnp.asarray(sphere_volume(NS, R))
    g = np.arange(NS) - (NS - 1) / 2.0
    DU, DV = np.meshgrid(g, g, indexing="ij")
    analytic = jnp.asarray((2.0 * np.sqrt(np.clip(R**2 - DU**2 - DV**2, 0, None))).astype(np.float32))

    N, F, NDET = 40, 4, 141
    vol = structured_volume(N); vol_j = jnp.asarray(vol)
    vol_fine = jnp.asarray(np.repeat(np.repeat(np.repeat(vol, F, 0), F, 1), F, 2))

    els = [0, 20, 30, 40, 45, 50, 60, 70]
    sph_err = {m: [] for m in MODES}; pha_err = {m: [] for m in MODES}
    for eldeg in els:
        a1 = jnp.array([[0.0, np.deg2rad(eldeg)]])
        for m in MODES:
            sph_err[m].append(nrmse(build(m, a1, (1, NS, NS), (NS, NS, NS)).forward_project(sph)[0], analytic))
        a2 = jnp.array([[np.pi / 4, np.deg2rad(eldeg)]])
        # independent reference: finest footprint at high resolution (separability error ~1/F)
        ref = build(PATH_FOOT, a2, (1, NDET, NDET), (N*F, N*F, N*F), 1.0/F).forward_project(vol_fine)
        for m in MODES:
            pha_err[m].append(nrmse(build(m, a2, (1, NDET, NDET), (N, N, N)).forward_project(vol_j), ref))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    for m in MODES:
        ax[0].plot(els, sph_err[m], "o-", color=COLOR[m], label=m)
        ax[1].plot(els, pha_err[m], "o-", color=COLOR[m], label=m)
    ax[0].set_title("Sphere vs ANALYTIC truth"); ax[1].set_title("Structured phantom vs oversampled ref (az=45)")
    for a in ax:
        a.axvline(45, color="gray", ls=":", lw=1); a.set_xlabel("elevation (deg)")
        a.set_ylabel("forward-projection NRMSE"); a.legend(fontsize=8)
    fig.suptitle("Forward accuracy: uncorrected diverges with elevation; pathlength fixes it; footprint helps only >45deg")
    fig.tight_layout()
    p = os.path.join(FIGDIR, "fig2_projection_accuracy.png"); fig.savefig(p, dpi=130); plt.close(fig)
    print("  wrote", p)


# =====================================================================================
# Figure 3: SEE the projection -- detector images at high elevation.
# =====================================================================================
def fig_projection_images(eldeg=60):
    N, F, NDET = 48, 4, 121
    vol = structured_volume(N, seed=2, nblobs=8); vol_j = jnp.asarray(vol)
    vol_fine = jnp.asarray(np.repeat(np.repeat(np.repeat(vol, F, 0), F, 1), F, 2))
    ang = jnp.array([[np.pi / 4, np.deg2rad(eldeg)]])
    truth = np.asarray(build(PATH_FOOT, ang, (1, NDET, NDET), (N*F, N*F, N*F), 1.0/F).forward_project(vol_fine)[0])
    imgs = {m: np.asarray(build(m, ang, (1, NDET, NDET), (N, N, N)).forward_project(vol_j)[0]) for m in MODES}

    vmax = truth.max()
    fig, ax = plt.subplots(2, 4, figsize=(13, 6.5))
    panels = [("oversampled truth", truth)] + [(m, imgs[m]) for m in MODES]
    for j, (name, img) in enumerate(panels):
        ax[0, j].imshow(img, vmin=0, vmax=vmax, cmap="viridis")
        ax[0, j].set_title(f"{name}\nsum={img.sum():.0f}", fontsize=9); ax[0, j].axis("off")
        if j == 0:
            ax[1, j].axis("off"); ax[1, j].text(0.5, 0.5, "difference vs truth ->", ha="center", va="center")
        else:
            d = img - truth
            mx = np.abs(truth).max() * 0.5
            ax[1, j].imshow(d, vmin=-mx, vmax=mx, cmap="RdBu_r")
            ax[1, j].set_title(f"{name} - truth\nNRMSE={nrmse(jnp.asarray(img), jnp.asarray(truth)):.3f}", fontsize=9)
            ax[1, j].axis("off")
    fig.suptitle(f"Detector image at elevation={eldeg} deg (az=45): uncorrected is globally too dim "
                 f"(mass loss); corrected modes match truth")
    fig.tight_layout()
    p = os.path.join(FIGDIR, "fig3_projection_images.png"); fig.savefig(p, dpi=120); plt.close(fig)
    print("  wrote", p)


# =====================================================================================
# Figure 4 + table: RECON quality, inverse-crime vs crime-free, at low and high elevation.
# =====================================================================================
def fig_recon():
    def run_geometry(label, el_amp_deg, N=32, F=4, NDET=57, n_views=36, iters=40):
        az = np.linspace(0, np.pi, n_views, endpoint=False)
        el = np.deg2rad(el_amp_deg) * np.cos(2 * az)
        angles = jnp.asarray(np.stack([az, el], 1).astype(np.float32)); shp = (n_views, NDET, NDET)
        truth = structured_volume(N, seed=1, nblobs=7); truth_j = jnp.asarray(truth)
        truth_fine = jnp.asarray(np.repeat(np.repeat(np.repeat(truth, F, 0), F, 1), F, 2))

        def model(mode, recon_shape, dv):
            m = MultiAxisParallelModel(shp, angles); m.set_elevation_correction(**MODES[mode])
            m.set_params(recon_shape=recon_shape, delta_voxel=dv, delta_det_row=1.0, delta_det_channel=1.0)
            return m

        # crime-free reference data: independent oversampled projector
        sino_ref = model(PATHLENGTH, (N*F, N*F, N*F), 1.0/F).forward_project(truth_fine)

        def recon(mode, sino):
            m = model(mode, (N, N, N), 1.0)
            rec, _ = m.recon(sino, weights=jnp.ones_like(sino),
                             init_recon=jnp.zeros((N, N, N), jnp.float32),
                             max_iterations=iters, stop_threshold_change_pct=0.05, print_logs=False)
            return nrmse(rec, truth_j)

        out = {}
        for mode in MODES:
            sino_crime = model(mode, (N, N, N), 1.0).forward_project(truth_j)
            out[mode] = (recon(mode, sino_crime), recon(mode, sino_ref))   # (inverse-crime, crime-free)
        print(f"\n  [{label}]  RECON NRMSE to truth (init_recon=0)")
        print(f"  {'mode':>24} {'inverse-crime':>14} {'crime-free':>12}")
        for mode in MODES:
            print(f"  {mode:>24} {out[mode][0]:14.4f} {out[mode][1]:12.4f}")
        return out

    res_low = run_geometry("low elevation +/-20deg", 20)
    res_high = run_geometry("high elevation +/-55deg", 55)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    x = np.arange(len(MODES)); w = 0.38
    for a, (res, ttl) in zip(ax, [(res_low, "elevation +/-20 deg"), (res_high, "elevation +/-55 deg")]):
        crime = [res[m][0] for m in MODES]; nocrime = [res[m][1] for m in MODES]
        a.bar(x - w/2, crime, w, label="inverse crime", color="lightgray", edgecolor="k")
        a.bar(x + w/2, nocrime, w, label="crime-free (independent ref)",
              color=[COLOR[m] for m in MODES], edgecolor="k")
        a.set_xticks(x); a.set_xticklabels(list(MODES), rotation=15, fontsize=8)
        a.set_title(ttl); a.set_ylabel("recon NRMSE to truth"); a.legend(fontsize=8)
    fig.suptitle("Recon: inverse crime (gray) is flat across modes -- it HIDES the bug; "
                 "crime-free reveals uncorrected worst, corrected best")
    fig.tight_layout()
    p = os.path.join(FIGDIR, "fig4_recon_crime_vs_nocrime.png"); fig.savefig(p, dpi=130); plt.close(fig)
    print("  wrote", p)


# =====================================================================================
# Reference trust: prove the oversampled "truth" is trustworthy, not just asserted.
# =====================================================================================
def check_reference_trust():
    """Demonstrate WHY the crime-free reference can be trusted, with numbers:
      (a) the oversampled sphere projection converges to the projector-FREE analytic truth as F grows;
      (b) at fine resolution the reference is mode-independent (footprint choice stops mattering);
      (c) the reference is converged (successive F agree), so F=4 is in the converged regime.
    """
    eldeg = 40
    # (a) oversampled sphere -> ANALYTIC (closed-form, no projector) truth
    NS, R = 61, 20.0
    sph = sphere_volume(NS, R)
    g = np.arange(NS) - (NS - 1) / 2.0; DU, DV = np.meshgrid(g, g, indexing="ij")
    analytic = jnp.asarray((2.0 * np.sqrt(np.clip(R**2 - DU**2 - DV**2, 0, None))).astype(np.float32))
    a1 = jnp.array([[0.0, np.deg2rad(eldeg)]])
    print(f"  (a) projection vs ANALYTIC (projector-free) sphere at el={eldeg} deg, sphere re-voxelized at each F:")
    print(f"      pathlength converges to the detector-pixel discretization floor and matches analytic.")
    un1 = nrmse(build(UNCORRECTED, a1, (1, NS, NS), (NS, NS, NS), 1.0).forward_project(jnp.asarray(sphere_volume(NS, R)))[0], analytic)
    print(f"      For contrast, uncorrected at its native resolution misses analytic by {un1:.4f} (the cos(el) mass error).")
    for F in [1, 2, 4, 8]:
        nf = NS * F
        sf = jnp.asarray(sphere_volume(nf, R * F))           # same physical sphere, finer voxelization
        pl = nrmse(build(PATHLENGTH, a1, (1, NS, NS), (nf, nf, nf), 1.0/F).forward_project(sf)[0], analytic)
        print(f"        pathlength F={F}: NRMSE={pl:.4f}")

    # (b) reference is mode-independent at fine resolution; (c) reference self-converges
    N, NDET = 40, 141
    vol = structured_volume(N)
    a2 = jnp.array([[np.pi / 4, np.deg2rad(eldeg)]])
    refs = {}
    print(f"  (b) reference mode-independence (NRMSE between pathlength-fine and +footprint-fine; ~0 => no favoritism):")
    for F in [4, 8]:
        vf = jnp.asarray(np.repeat(np.repeat(np.repeat(vol, F, 0), F, 1), F, 2))
        pl = build(PATHLENGTH, a2, (1, NDET, NDET), (N*F, N*F, N*F), 1.0/F).forward_project(vf)
        pf = build(PATH_FOOT, a2, (1, NDET, NDET), (N*F, N*F, N*F), 1.0/F).forward_project(vf)
        refs[F] = pl
        print(f"        F={F}: NRMSE={nrmse(pl, pf):.4f}")
    vf2 = jnp.asarray(np.repeat(np.repeat(np.repeat(vol, 2, 0), 2, 1), 2, 2))
    refs[2] = build(PATHLENGTH, a2, (1, NDET, NDET), (N*2, N*2, N*2), 1.0/2).forward_project(vf2)
    print(f"  (c) reference self-convergence (NRMSE vs the F=8 reference; small => F=4 already converged):")
    print(f"        F=2 vs F=8: {nrmse(refs[2], refs[8]):.4f};  F=4 vs F=8: {nrmse(refs[4], refs[8]):.4f}")


if __name__ == "__main__":
    print("jax devices:", jax.devices())
    print("\n[1/5] reference trust (why the crime-free 'truth' is trustworthy) ..."); check_reference_trust()
    print("\n[2/5] mass conservation ..."); fig_mass_conservation()
    print("\n[3/5] forward accuracy ...");   fig_projection_accuracy()
    print("\n[4/5] projection images ...");  fig_projection_images()
    print("\n[5/5] recon (crime vs crime-free, low & high el) ..."); fig_recon()
    print("\nDONE -- see", FIGDIR)
