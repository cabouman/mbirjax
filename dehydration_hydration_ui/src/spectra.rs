//! The MCP/TPX `*_Spectra.txt` file that sits next to a VENUS TOF image
//! stack: one row per image, `shutter_time,counts` with the shutter time in
//! seconds. It provides the physical time-of-flight axis for the profile
//! plots, and its wavelength conversion (same formula and default
//! source–detector distance as the VENUS marimo notebooks).

use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};

/// Default source–detector distance at VENUS, in meters.
pub const DEFAULT_DISTANCE_M: f64 = 25.0;

/// The first `*_Spectra.txt` directly inside `dir`, if any.
pub fn find_spectra_file(dir: &Path) -> Option<PathBuf> {
    let mut found: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.is_file()
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.ends_with("_Spectra.txt"))
        })
        .collect();
    found.sort();
    found.into_iter().next()
}

/// Read the TOF axis from a spectra file: one value per image, in µs.
pub fn load_tof_us(path: &Path) -> Result<Vec<f64>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("read spectra file {}", path.display()))?;
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        // Header line ("shutter_time,counts").
        if i == 0 && line.to_lowercase().starts_with("shutter") {
            continue;
        }
        let first = line.split([',', '\t', ' ']).next().unwrap_or("");
        let seconds: f64 = first
            .parse()
            .with_context(|| format!("bad TOF value '{first}' on line {} of {}", i + 1, path.display()))?;
        out.push(seconds * 1e6);
    }
    if out.is_empty() {
        bail!("no TOF values in {}", path.display());
    }
    Ok(out)
}

/// λ[Å] = h·t / (mₙ·L) — same conversion as the VENUS notebooks.
pub fn tof_us_to_lambda_angstroms(tof_us: f64, distance_m: f64) -> f64 {
    const H: f64 = 6.626_070_15e-34; // Planck constant (J s)
    const M_N: f64 = 1.674_927_498_04e-27; // neutron mass (kg)
    H * (tof_us * 1e-6) / (M_N * distance_m) * 1e10
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("dehydration_spectra_test_{tag}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn parses_header_and_seconds_to_microseconds() {
        let dir = tmp_dir("parse");
        let path = dir.join("Run_1_Spectra.txt");
        std::fs::write(&path, "shutter_time,counts\n6.08e-06,8115484\n1.12e-05,8217989\n").unwrap();
        let tof = load_tof_us(&path).unwrap();
        assert_eq!(tof.len(), 2);
        assert!((tof[0] - 6.08).abs() < 1e-9);
        assert!((tof[1] - 11.2).abs() < 1e-9);
    }

    #[test]
    fn finds_the_spectra_file_in_a_folder() {
        let dir = tmp_dir("find");
        std::fs::write(dir.join("image_0000.tif"), b"x").unwrap();
        std::fs::write(dir.join("Run_9_Spectra.txt"), "1e-6,1\n").unwrap();
        let found = find_spectra_file(&dir).unwrap();
        assert!(found.ends_with("Run_9_Spectra.txt"));
        assert!(find_spectra_file(&tmp_dir("empty")).is_none());
    }

    #[test]
    fn wavelength_conversion_matches_notebook_formula() {
        // 10000 µs at 25 m → λ = h·t/(m·L)·1e10 ≈ 1.5824 Å
        let lambda = tof_us_to_lambda_angstroms(10_000.0, 25.0);
        assert!((lambda - 1.5824).abs() < 1e-3, "{lambda}");
    }
}
