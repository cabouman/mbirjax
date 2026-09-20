//! Native port of `mbirjax.hsnt.hyper_denoise` / `dehydrate` / `rehydrate`:
//!
//! M. S. N. Chowdhury et al., "Fast Hyperspectral Neutron Tomography," IEEE
//! Transactions on Computational Imaging, vol. 11, pp. 663–677, 2025.
//!
//! The data is a 2-D matrix of hyperspectral points: one row per pixel, one
//! column per spectral band (here: one column per image of the stack — the
//! image index plays the role of the spectral axis, exactly like the
//! notebook's `np.swapaxes(raw, 0, 2)` + reshape). Dehydration projects the
//! spectra onto a low-dimensional non-negative subspace estimated by NMF;
//! rehydration multiplies back, discarding the noise that lives outside the
//! subspace.
//!
//! Differences from the Python original (all inconsequential for the
//! notebook's use):
//! * `num_materials` is always provided by the GUI, so the automatic
//!   subspace-dimension estimation is not ported;
//! * the batch row permutation and the SVD test matrix are seeded, so runs
//!   are reproducible;
//! * the final `W·H` product is computed in f64 and cast to f32 (Python
//!   casts to f32 first).

use crate::linalg::{randomized_svd, Rng};
use crate::nmf::{nmf, BetaLoss, NmfConfig};
use anyhow::Result;
use ndarray::{s, Array2, ArrayView2, Axis};
use std::sync::atomic::AtomicBool;

/// Version of the mbirjax library this module is a native port of. Bump
/// after diffing the denoising functions of `mbirjax/hsnt.py` against the
/// newer release (the HDF5 utilities in that file are not part of the port).
pub const MBIRJAX_VERSION: &str = "0.7.2";
pub const MBIRJAX_COMMIT: &str = "7bb2009, 2026-07-24";

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DatasetType {
    /// attenuation = −log(transmission)
    Attenuation,
    Transmission,
}

impl DatasetType {
    pub fn label(self) -> &'static str {
        match self {
            DatasetType::Attenuation => "attenuation",
            DatasetType::Transmission => "transmission",
        }
    }
}

#[derive(Clone)]
pub struct HsntParams {
    pub dataset_type: DatasetType,
    pub num_materials: usize,
    /// Multiplier (≥ 1) on `num_materials` giving the subspace dimension.
    pub safety_factor: f64,
    pub beta_loss: BetaLoss,
    pub max_iter: usize,
    pub tolerance: f64,
    /// Elements (points × bands) processed per NMF batch.
    pub batch_size: usize,
}

impl Default for HsntParams {
    fn default() -> Self {
        Self {
            dataset_type: DatasetType::Attenuation,
            num_materials: 2,
            safety_factor: 2.0,
            beta_loss: BetaLoss::Frobenius,
            max_iter: 300,
            tolerance: 1e-10,
            batch_size: 1 << 27,
        }
    }
}

impl HsntParams {
    pub fn subspace_dimension(&self) -> usize {
        (self.safety_factor * self.num_materials as f64).ceil() as usize
    }
}

const SEED: u64 = 0x5EED_D411;

/// Denoise `x` (points × bands, consumed): dehydrate then rehydrate.
/// `progress(stage, fraction)` is called from the solver's outer iterations;
/// setting `cancel` makes the call return an error at the next iteration.
pub fn hyper_denoise(
    x: Array2<f64>,
    params: &HsntParams,
    cancel: &AtomicBool,
    progress: &mut dyn FnMut(&str, f32),
) -> Result<Array2<f32>> {
    let (w, ht, transmission) = dehydrate(x, params, cancel, progress)?;
    progress("Rehydrating", 0.0);
    let out = rehydrate(&w, &ht, transmission);
    progress("Rehydrating", 1.0);
    Ok(out)
}

/// Project every spectrum onto the non-negative subspace. Returns
/// `(subspace_data W (points×k), basis-transposed Ht (bands×k), was_transmission)`.
fn dehydrate(
    mut x: Array2<f64>,
    params: &HsntParams,
    cancel: &AtomicBool,
    progress: &mut dyn FnMut(&str, f32),
) -> Result<(Array2<f64>, Array2<f64>, bool)> {
    let epsilon = 1e-3;
    let transmission = params.dataset_type == DatasetType::Transmission;

    if transmission {
        // Initial cleanup in the attenuation domain (safety factor ×3) to
        // remove defective measurements, then convert to attenuation.
        let pre = HsntParams {
            dataset_type: DatasetType::Attenuation,
            safety_factor: params.safety_factor * 3.0,
            ..params.clone()
        };
        let mut pre_progress = |stage: &str, f: f32| {
            progress(&format!("Pre-clean: {stage}"), f);
        };
        let cleaned = hyper_denoise(x, &pre, cancel, &mut pre_progress)?;
        x = cleaned.mapv(|v| -(f64::from(v).max(epsilon)).ln());
    }

    x.mapv_inplace(|v| v.max(0.0)); // enforce non-negativity

    let (num_points, num_bands) = x.dim();
    let subspace_dimension = params.subspace_dimension().min(num_bands).max(1);

    let config = NmfConfig {
        n_components: subspace_dimension,
        beta_loss: params.beta_loss,
        tol: params.tolerance,
        max_iter: params.max_iter,
        seed: SEED,
    };

    let num_points_batch = (params.batch_size / num_bands.max(1)).max(1);
    let num_batches = num_points.div_ceil(num_points_batch);

    if num_batches <= 1 {
        // Single batch: one NMF solves both factors.
        let mut p = |f: f32| progress("Solving NMF", f);
        let res = nmf(x.view(), &config, None, cancel, &mut p)?;
        return Ok((res.w, res.ht, transmission));
    }

    // ---- Multi-batch: estimate a shared basis from shuffled batches …
    let mut row_idx: Vec<usize> = (0..num_points).collect();
    Rng::new(SEED ^ 0xBA7C).shuffle(&mut row_idx);

    let batch_config = NmfConfig {
        max_iter: (params.max_iter / num_batches).max(50),
        ..config.clone()
    };
    let mut basis_stack = Array2::<f64>::zeros((0, num_bands));
    for batch in 0..num_batches {
        let start = batch * num_points_batch;
        let stop = ((batch + 1) * num_points_batch).min(num_points);
        let mut batch_data = Array2::<f64>::zeros((stop - start, num_bands));
        for (r, &src) in row_idx[start..stop].iter().enumerate() {
            batch_data.row_mut(r).assign(&x.row(src));
        }
        let label = format!("Estimating basis (batch {}/{num_batches})", batch + 1);
        let mut p = |f: f32| progress(&label, f);
        let res = nmf(batch_data.view(), &batch_config, None, cancel, &mut p)?;
        // Stack the basis spectra (rows of H = columns of Ht) of every batch.
        basis_stack.append(Axis(0), res.ht.t()).expect("same band count");
    }

    // … combine the per-batch bases into the final one …
    let mut p = |f: f32| progress("Combining batch bases", f);
    let combined = nmf(basis_stack.view(), &config, None, cancel, &mut p)?;
    let basis_ht = combined.ht; // bands × k

    // … then transform every batch with the basis held fixed.
    let mut w = Array2::<f64>::zeros((num_points, subspace_dimension));
    for batch in 0..num_batches {
        let start = batch * num_points_batch;
        let stop = ((batch + 1) * num_points_batch).min(num_points);
        let label = format!("Projecting data (batch {}/{num_batches})", batch + 1);
        let mut p = |f: f32| progress(&label, f);
        let res = nmf(
            x.slice(s![start..stop, ..]),
            &config,
            Some(&basis_ht),
            cancel,
            &mut p,
        )?;
        w.slice_mut(s![start..stop, ..]).assign(&res.w);
    }

    Ok((w, basis_ht, transmission))
}

/// Estimate the number of materials in a sample of pixel spectra — port of
/// `_estimate_subspace_dimension` without the safety-factor scaling (the
/// caller applies it): fit a log-linear noise model to the singular values
/// over the 25–75 percentile window and count the leading singular values
/// exceeding 1.5× the model's prediction.
///
/// `sample` is (pixels × bands); pass a few hundred randomly chosen pixel
/// spectra (the Python original uses up to `num_bands` rows — sampling a
/// few hundred gives the same estimate at a fraction of the cost).
pub fn estimate_num_materials(sample: ArrayView2<f64>) -> usize {
    let (m, n_bands) = sample.dim();
    let k = m.min(n_bands);
    if k < 4 {
        return 1;
    }

    let svd = randomized_svd(sample, k, SEED ^ 0xE571);
    let s = &svd.s;
    let size = s.len();

    // Percentile window [25%, 75%] of the singular value indices.
    let start = ((0.25 * size as f64).floor() as usize).min(size.saturating_sub(2));
    let stop = ((0.75 * size as f64).ceil() as usize).clamp(start + 2, size);

    // Least-squares fit of log(s) ≈ a·n + b over the window.
    let n_fit = (stop - start) as f64;
    let (mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0);
    for i in start..stop {
        let x = i as f64;
        let y = (s[i] + 1e-12).ln();
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
    }
    let denom = n_fit * sxx - sx * sx;
    if denom.abs() < 1e-12 {
        return 1;
    }
    let a = (n_fit * sxy - sx * sy) / denom;
    let b = (sy - a * sx) / n_fit;

    // Leading singular values above 1.5× the noise prediction are signal.
    let threshold = 1.5;
    (0..start)
        .filter(|&i| s[i] > threshold * (a * i as f64 + b).exp())
        .count()
        .max(1)
}

/// Multiply the subspace data back onto the basis; convert back to
/// transmission when the input was transmission.
fn rehydrate(w: &Array2<f64>, ht: &Array2<f64>, transmission: bool) -> Array2<f32> {
    let out = crate::linalg::par_matmul(w.view(), ht.t());
    if transmission {
        out.mapv(|v| (-v).exp() as f32)
    } else {
        out.mapv(|v| v as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::Rng;

    /// Noisy low-rank "hyperspectral" data: 2 materials × 30 bands over a
    /// 16×16 image grid, Gaussian noise on top.
    fn synthetic(noise: f64) -> (Array2<f64>, Array2<f64>) {
        let points = 256;
        let bands = 30;
        let mut rng = Rng::new(99);
        // Two smooth non-negative spectra.
        let spec = |t: usize, j: usize| -> f64 {
            match t {
                0 => 1.0 + (j as f64 * 0.3).sin().abs(),
                _ => 0.5 + 0.1 * j as f64,
            }
        };
        let mut clean = Array2::<f64>::zeros((points, bands));
        for p in 0..points {
            let a = ((p % 16) as f64 / 16.0) * 2.0;
            let b = ((p / 16) as f64 / 16.0) * 1.5;
            for j in 0..bands {
                clean[[p, j]] = a * spec(0, j) + b * spec(1, j);
            }
        }
        let noisy = clean.mapv(|v| (v + noise * rng.normal()).max(0.0));
        (clean, noisy)
    }

    #[test]
    fn denoising_reduces_error_to_clean_data() {
        let (clean, noisy) = synthetic(0.2);
        let params = HsntParams {
            num_materials: 2,
            max_iter: 200,
            ..HsntParams::default()
        };
        let cancel = AtomicBool::new(false);
        let out = hyper_denoise(noisy.clone(), &params, &cancel, &mut |_, _| {}).unwrap();

        let err_before: f64 = (&noisy - &clean).iter().map(|v| v * v).sum::<f64>();
        let err_after: f64 = clean
            .indexed_iter()
            .map(|((i, j), &c)| {
                let d = f64::from(out[[i, j]]) - c;
                d * d
            })
            .sum();
        assert!(
            err_after < err_before * 0.5,
            "denoising should at least halve the squared error: {err_after} vs {err_before}"
        );
    }

    #[test]
    fn batched_path_matches_shape_and_stays_finite() {
        let (_, noisy) = synthetic(0.1);
        let params = HsntParams {
            num_materials: 2,
            max_iter: 120,
            batch_size: 30 * 60, // force ~5 batches for 256 points × 30 bands
            ..HsntParams::default()
        };
        let cancel = AtomicBool::new(false);
        let out = hyper_denoise(noisy.clone(), &params, &cancel, &mut |_, _| {}).unwrap();
        assert_eq!(out.dim(), noisy.dim());
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn material_estimation_finds_two_materials() {
        let (_, noisy) = synthetic(0.05);
        let estimate = estimate_num_materials(noisy.view());
        assert_eq!(estimate, 2, "synthetic data has exactly 2 materials");
    }

    #[test]
    fn transmission_roundtrip_stays_in_unit_range() {
        let (_, noisy_att) = synthetic(0.05);
        // Build a transmission dataset from the attenuation one.
        let trans = noisy_att.mapv(|v| (-v).exp());
        let params = HsntParams {
            dataset_type: DatasetType::Transmission,
            num_materials: 2,
            max_iter: 80,
            ..HsntParams::default()
        };
        let cancel = AtomicBool::new(false);
        let out = hyper_denoise(trans, &params, &cancel, &mut |_, _| {}).unwrap();
        assert!(out.iter().all(|&v| v.is_finite() && v > 0.0 && v <= 1.0 + 1e-3));
    }
}
