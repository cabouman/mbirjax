# Multiaxis elevation projector — corrections and evidence

A human-readable record of two corrections to the **vertical (elevation) fan** of
`MultiAxisParallelModel`, with the before/after figures that motivated them. Both corrections are
now **unconditional** in the projector; the durable gates live in
[`tests/geometries/test_multiaxis_elevation.py`](../../tests/geometries/test_multiaxis_elevation.py)
(mass conservation across elevation, `det_channel_offset` sign, FBP permutation-invariance) plus
the adjoint / reduce-to-parallel gates in `tests/geometries/test_projectors.py`.

## The two corrections (independent of each other)

**Elevation pathlength (a normalization bug).** A parallel-beam projection is a line integral, so
the total projected mass of a fixed object must not depend on the viewing angle. The old vertical
fan deposited each voxel with amplitude `1.0`, so a voxel's projected mass fell off as `cos(el)`
(≈23 % low at 40°, 50 % low at 60°). Fix: a mass-conserving amplitude `(Δ_slice/Δ_row)/W_p_r`
(= `1/cos(el)` in the simple case), applied to the voxel values like `ConeBeamModel`'s `1/cos φ`.
This is the dominant, must-have fix.

**Elevation footprint (a separability approximation).** The vertical pass is 2-D parallel beam in
the `(t,z)→v` plane at angle `el`, so its footprint uses the same max-of-edges rule
`ParallelBeamModel` already uses (`max` over the z-edge and the in-plane `sin(el)` edges). Below
~45° the z-edge dominates; above 45° the in-plane edge takes over. Because the max never collapses
to zero, **no footprint floor is needed** (matching parallel/cone).

Why both went unnoticed: under **inverse crime** (simulate and reconstruct with the same
projector) both errors cancel, so they only surfaced against independent or real data.

## How the crime-free "truth" in the figures was made

The recon figures need data the model under test did *not* produce. It was made by **oversampling**:
the object was placed on a grid `F=4×` finer and projected at that resolution onto the same
detector. The separable projector's only approximation is per-voxel (intra-voxel shear
`~ Δ_voxel·sin el`), so at `F×` finer resolution that error shrinks `~1/F` and the fine projection
converges to the true line integral — reconstructing it with the coarse model is not an inverse
crime. It was checked to be converged (successive `F` agree), mode-independent, and — against a
uniform sphere's closed-form projection `2·√(R²−ρ²)` — to reproduce the projector-free analytic
truth to the detector-pixel floor (while the uncorrected projector was ~14× further off).

## The figures (`figures/`)

- **fig1_mass_conservation.png** — total projected mass of a fixed sphere vs elevation: the
  uncorrected mode rides the `cos(el)` curve; the corrected mode sits flat on the true mass.
- **fig2_projection_accuracy.png** — forward-projection NRMSE vs elevation against analytic and
  oversampled references: uncorrected diverges with elevation; the pathlength correction flattens
  it to the discretization floor; the footprint term matters only above 45°.
- **fig3_projection_images.png** — the 60° detector image: uncorrected is globally too dim (the
  visible mass loss); corrected matches the oversampled truth.
- **fig4_recon_crime_vs_nocrime.png** — reconstruction NRMSE at ±20° and ±55° elevation: the
  inverse-crime bars are flat across modes (the bug hidden); the crime-free bars show uncorrected
  worst, corrected best.

## What the evidence showed

- The **pathlength correction is decisive on honest data**, catastrophically so at high elevation:
  in the crime-free recon at ±55°, NRMSE dropped from `0.33` (uncorrected) to `0.05` (≈6.5×).
- The **footprint correction is a small (~1 %) bonus that only acts above 45°**, and it lets the
  non-standard footprint floor be removed (max-of-edges is self-bounding like parallel/cone).
- **Inverse crime completely hid both** — the sharpest methodological lesson here.
