"""Parametric polychromatic forward model for beam-hardening experiments.

Design goal: the correct STRUCTURE of the nonlinearity (concave per-ray
response; metals much harder than plastic; multiple metals with different
hardening factors), with a single severity dial -- not spectroscopic fidelity.

Components:
  - a filtered-bremsstrahlung source spectrum (Kramers' law x Al filtration),
    discretized over energy bins;
  - per-material energy dependence via the standard two-basis form
        mu(E) / mu(E_ref) = w_pe (E/E_ref)^-3 + (1 - w_pe) KN(E)/KN(E_ref)
    (photoelectric + Klein-Nishina Compton), where w_pe is the material's
    hardening factor at the reference energy;
  - a severity dial s in [0, 1] applied as
        r_s(E) = (1 - s) + s * r(E),
    so s = 0 is EXACTLY the monochromatic (linear) model -- the identity gate
    -- and s = 1 the full parametric nonlinearity;
  - poly_sinogram: y = -ln sum_k w_k exp(-sum_i r_i(E_k) t_i) for reference-
    energy path sinograms t_i (attenuation units, mu_ref folded in), any
    number of materials.

Everything is plain numpy on host arrays.
"""

import numpy as np

E_REF_KEV = 60.0     # reference energy: t_i are attenuation path integrals here


def klein_nishina(e_kev):
    """Klein-Nishina total cross-section (relative units)."""
    k = np.asarray(e_kev, dtype=np.float64) / 511.0
    t1 = (1 + k) / k ** 2 * (2 * (1 + k) / (1 + 2 * k) - np.log(1 + 2 * k) / k)
    t2 = np.log(1 + 2 * k) / (2 * k)
    t3 = -(1 + 3 * k) / (1 + 2 * k) ** 2
    return t1 + t2 + t3


def mu_ratio(e_kev, w_pe, e_ref=E_REF_KEV):
    """mu(E)/mu(E_ref) for a material with photoelectric weight w_pe at E_ref."""
    e = np.asarray(e_kev, dtype=np.float64)
    pe = (e / e_ref) ** -3
    kn = klein_nishina(e) / klein_nishina(e_ref)
    return w_pe * pe + (1.0 - w_pe) * kn


def make_spectrum(kvp=140.0, n_bins=24, filtration_mm_al=2.0, e_min=20.0):
    """Detected-spectrum weights: Kramers' law filtered by aluminum.

    Returns (energies_keV, weights) with weights normalized to sum to 1.
    Aluminum's attenuation uses the same two-basis form (w_pe ~ 0.35 at 60 keV,
    mu ~ 0.075 /mm there) -- adequate for shaping, not spectroscopy.
    """
    edges = np.linspace(e_min, kvp, n_bins + 1)
    e = 0.5 * (edges[:-1] + edges[1:])
    kramers = np.maximum(kvp - e, 0.0) / e
    mu_al = 0.075 * mu_ratio(e, w_pe=0.35)          # per mm
    s = kramers * np.exp(-mu_al * filtration_mm_al)
    return e, s / s.sum()


def poly_sinogram(path_sinos, w_pe_list, spectrum, severity=1.0):
    """Polychromatic log sinogram for reference-energy path sinograms.

    Args:
        path_sinos: list of arrays t_i -- REFERENCE-ENERGY attenuation path
            integrals per material (mu_ref folded in; any common shape).
        w_pe_list: photoelectric weight per material (same length).
        spectrum: (energies_keV, weights) from make_spectrum.
        severity: the dial s in [0, 1]; s = 0 returns sum(t_i) exactly
            (monochromatic identity).

    Returns:
        float32 array, y_poly = -ln sum_k w_k exp(-sum_i r_i,s(E_k) t_i).
    """
    assert len(path_sinos) == len(w_pe_list)
    assert all(np.shape(t) == np.shape(path_sinos[0]) for t in path_sinos)
    if severity == 0.0:
        return sum(np.asarray(t, dtype=np.float32) for t in path_sinos)
    energies, weights = spectrum
    # Per-material severity-dialed ratio curves, normalized so the spectrum-
    # weighted mean is exactly 1: the linear (monochromatic) model is then the
    # TANGENT of the polychromatic response at t -> 0 for every severity --
    # the dial bends the curve without recalibrating the small-path slope,
    # and the deficit y_mono - y_poly is nonnegative and monotone in s.
    ratios = []
    for w_pe in w_pe_list:
        rs = (1.0 - severity) + severity * mu_ratio(energies, w_pe)
        ratios.append(rs / float(np.sum(weights * rs)))
    total = np.zeros(np.shape(path_sinos[0]), dtype=np.float64)
    for k, wk in enumerate(weights):
        expo = np.zeros_like(total)
        for t, rs in zip(path_sinos, ratios):
            expo += float(rs[k]) * np.asarray(t, dtype=np.float64)
        total += wk * np.exp(-expo)
    return (-np.log(total)).astype(np.float32)


def deficit_report(path_sinos, w_pe_list, spectrum, severity=1.0):
    """Summary of the injected nonlinearity: y_mono - y_poly percentiles."""
    y_mono = sum(np.asarray(t, dtype=np.float64) for t in path_sinos)
    y_poly = poly_sinogram(path_sinos, w_pe_list, spectrum, severity)
    d = np.asarray(y_mono - y_poly)
    rel = d[y_mono > 0.1] / y_mono[y_mono > 0.1]
    return dict(deficit_max=float(d.max()),
                deficit_p99=float(np.percentile(d, 99)),
                rel_deficit_max=float(rel.max()) if rel.size else 0.0,
                rel_deficit_p99=float(np.percentile(rel, 99)) if rel.size else 0.0)
