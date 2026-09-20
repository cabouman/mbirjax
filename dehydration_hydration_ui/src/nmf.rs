//! Non-negative matrix factorization, equivalent to scikit-learn's
//! `non_negative_factorization` as used by `mbirjax.hsnt`:
//!
//! * initialization: NNDSVD (Boutsidis & Gallopoulos) from a randomized SVD,
//! * `beta_loss='frobenius'` → coordinate-descent / HALS updates (sklearn's
//!   `cd` solver),
//! * `beta_loss='kullback-leibler'` → multiplicative updates (sklearn's `mu`
//!   solver),
//! * optional fixed basis (`update_H=False`) for the transform-only calls of
//!   the batched dehydration path.
//!
//! `X (m×n) ≈ W (m×k) · H (k×n)`. The basis is stored transposed (`ht`,
//! n×k) so both HALS half-steps run through the same row-parallel kernel.

use crate::linalg::{par_gram, par_matmul, randomized_svd, ROW_CHUNK};
use anyhow::{bail, Result};
use ndarray::parallel::prelude::*;
use ndarray::{Array1, Array2, ArrayView2, Axis};
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BetaLoss {
    Frobenius,
    KullbackLeibler,
}

impl BetaLoss {
    pub fn label(self) -> &'static str {
        match self {
            BetaLoss::Frobenius => "frobenius",
            BetaLoss::KullbackLeibler => "kullback-leibler",
        }
    }
}

#[derive(Clone)]
pub struct NmfConfig {
    pub n_components: usize,
    pub beta_loss: BetaLoss,
    /// Convergence tolerance (sklearn semantics: relative violation for the
    /// cd solver, relative loss decrease checked every 10 iterations for mu).
    pub tol: f64,
    pub max_iter: usize,
    pub seed: u64,
}

pub struct NmfResult {
    pub w: Array2<f64>,
    /// Basis, transposed: `ht` is n×k (row j of H is column j of `ht`).
    pub ht: Array2<f64>,
    pub n_iter: usize,
}

/// Factorize `x`. With `fixed_ht = Some(basis)` only `w` is solved
/// (sklearn's `update_H=False`: `W` starts at zeros for the cd solver, at
/// `sqrt(mean(X)/k)` for mu); otherwise both factors start from NNDSVD.
pub fn nmf(
    x: ArrayView2<f64>,
    config: &NmfConfig,
    fixed_ht: Option<&Array2<f64>>,
    cancel: &AtomicBool,
    progress: &mut dyn FnMut(f32),
) -> Result<NmfResult> {
    let (m, n) = x.dim();
    let k = config.n_components.min(m).min(n).max(1);

    let (mut w, mut ht, update_h) = match fixed_ht {
        Some(basis) => {
            assert_eq!(basis.nrows(), n, "fixed basis must be n×k");
            let w0 = match config.beta_loss {
                BetaLoss::Frobenius => Array2::<f64>::zeros((m, k)),
                BetaLoss::KullbackLeibler => {
                    let avg = (x.mean().unwrap_or(0.0).max(0.0) / k as f64).sqrt();
                    Array2::<f64>::from_elem((m, k), avg)
                }
            };
            (w0, basis.clone(), false)
        }
        None => {
            let (w0, ht0) = nndsvd_init(x, k, config.seed);
            (w0, ht0, true)
        }
    };

    let n_iter = match config.beta_loss {
        BetaLoss::Frobenius => solve_cd(
            x, &mut w, &mut ht, update_h, config.tol, config.max_iter, cancel, progress,
        )?,
        BetaLoss::KullbackLeibler => solve_mu_kl(
            x, &mut w, &mut ht, update_h, config.tol, config.max_iter, cancel, progress,
        )?,
    };

    Ok(NmfResult { w, ht, n_iter })
}

// ---------------------------------------------------------------- NNDSVD --

/// NNDSVD initialization (sklearn `_initialize_nmf(init='nndsvd')`): split
/// each singular-vector pair into its positive/negative parts and keep the
/// dominant one; entries below 1e-6 are zeroed.
pub fn nndsvd_init(x: ArrayView2<f64>, k: usize, seed: u64) -> (Array2<f64>, Array2<f64>) {
    let (m, n) = x.dim();
    let svd = randomized_svd(x, k, seed);
    let kk = svd.s.len();

    let mut w = Array2::<f64>::zeros((m, k));
    let mut ht = Array2::<f64>::zeros((n, k));

    if kk == 0 {
        return (w, ht);
    }

    let s0 = svd.s[0].max(0.0).sqrt();
    w.column_mut(0)
        .assign(&svd.u.column(0).mapv(|v| s0 * v.abs()));
    ht.column_mut(0)
        .assign(&svd.vt.row(0).mapv(|v| s0 * v.abs()));

    for j in 1..kk {
        let xj = svd.u.column(j);
        let yj = svd.vt.row(j);
        let xp = xj.mapv(|v| v.max(0.0));
        let xn = xj.mapv(|v| (-v).max(0.0));
        let yp = yj.mapv(|v| v.max(0.0));
        let yn = yj.mapv(|v| (-v).max(0.0));

        let xpn = xp.dot(&xp).sqrt();
        let xnn = xn.dot(&xn).sqrt();
        let ypn = yp.dot(&yp).sqrt();
        let ynn = yn.dot(&yn).sqrt();

        let (u, v, unorm, vnorm, sigma) = if xpn * ypn > xnn * ynn {
            (xp, yp, xpn, ypn, xpn * ypn)
        } else {
            (xn, yn, xnn, ynn, xnn * ynn)
        };
        if unorm <= 1e-12 || vnorm <= 1e-12 {
            continue; // leave the component at zero
        }
        let lbd = (svd.s[j].max(0.0) * sigma).sqrt();
        w.column_mut(j).assign(&u.mapv(|t| lbd * t / unorm));
        ht.column_mut(j).assign(&v.mapv(|t| lbd * t / vnorm));
    }

    let eps = 1e-6;
    w.mapv_inplace(|v| if v < eps { 0.0 } else { v });
    ht.mapv_inplace(|v| if v < eps { 0.0 } else { v });
    (w, ht)
}

// ---------------------------------------------- coordinate descent (HALS) --

/// One HALS pass over every component of `w` (rows are independent →
/// row-parallel). `xht = X·Ht` (m×k), `hht = HtᵀHt` (k×k). Returns the
/// accumulated projected-gradient violation (sklearn's stopping metric).
fn hals_pass(w: &mut Array2<f64>, xht: &Array2<f64>, hht: &Array2<f64>) -> f64 {
    let k = w.ncols();
    w.axis_chunks_iter_mut(Axis(0), ROW_CHUNK)
        .into_par_iter()
        .zip(xht.axis_chunks_iter(Axis(0), ROW_CHUNK).into_par_iter())
        .map(|(mut wc, xc)| {
            let mut violation = 0.0;
            for (mut wrow, xrow) in wc.axis_iter_mut(Axis(0)).zip(xc.axis_iter(Axis(0))) {
                for t in 0..k {
                    // gradient of 0.5‖X−WH‖² w.r.t. w[t]
                    let mut wh = 0.0;
                    for s in 0..k {
                        wh += wrow[s] * hht[[s, t]];
                    }
                    let grad = wh - xrow[t];
                    let pg = if wrow[t] == 0.0 { grad.min(0.0) } else { grad };
                    violation += pg.abs();
                    let hess = hht[[t, t]];
                    if hess != 0.0 {
                        wrow[t] = (wrow[t] - grad / hess).max(0.0);
                    }
                }
            }
            violation
        })
        .sum()
}

#[allow(clippy::too_many_arguments)]
fn solve_cd(
    x: ArrayView2<f64>,
    w: &mut Array2<f64>,
    ht: &mut Array2<f64>,
    update_h: bool,
    tol: f64,
    max_iter: usize,
    cancel: &AtomicBool,
    progress: &mut dyn FnMut(f32),
) -> Result<usize> {
    let mut violation_init: Option<f64> = None;
    let mut n_iter = 0;
    for it in 0..max_iter {
        if cancel.load(Ordering::Relaxed) {
            bail!("cancelled");
        }
        let mut violation = 0.0;

        let xht = par_matmul(x, ht.view());
        let hht = par_gram(ht.view());
        violation += hals_pass(w, &xht, &hht);

        if update_h {
            let xtw = par_matmul(x.t(), w.view());
            let wtw = par_gram(w.view());
            violation += hals_pass(ht, &xtw, &wtw);
        }

        n_iter = it + 1;
        progress(n_iter as f32 / max_iter as f32);

        let vinit = *violation_init.get_or_insert(violation);
        if vinit == 0.0 || violation / vinit <= tol {
            break;
        }
    }
    Ok(n_iter)
}

// ------------------------------------------- multiplicative updates (KL) --

/// KL-divergence multiplicative-update solver (sklearn `solver='mu'`,
/// `beta_loss='kullback-leibler'`), fused over row chunks so the full m×n
/// `W·H` product is never materialized.
#[allow(clippy::too_many_arguments)]
fn solve_mu_kl(
    x: ArrayView2<f64>,
    w: &mut Array2<f64>,
    ht: &mut Array2<f64>,
    update_h: bool,
    tol: f64,
    max_iter: usize,
    cancel: &AtomicBool,
    progress: &mut dyn FnMut(f32),
) -> Result<usize> {
    let eps = f32::EPSILON as f64; // sklearn's EPSILON guard
    let k = w.ncols();
    let n = ht.nrows();

    let mut error_init: Option<f64> = None;
    let mut previous_error: Option<f64> = None;
    let mut n_iter = 0;

    for it in 0..max_iter {
        if cancel.load(Ordering::Relaxed) {
            bail!("cancelled");
        }

        // ---- W ← W ⊙ ((X ⊘ WH)·H) / rowsum(H). The pass also returns the
        // KL divergence of the pre-update factors for the convergence check.
        let h_sums = ht.sum_axis(Axis(0)); // Σ_j H[t,j], length k
        let kl = mu_update_w(x, w, ht, &h_sums, eps);

        // ---- H ← H ⊙ (Wᵀ(X ⊘ WH)) / colsum(W), with the just-updated W.
        if update_h {
            let w_sums = w.sum_axis(Axis(0)); // length k
            let numer_ht: Array2<f64> = x
                .axis_chunks_iter(Axis(0), ROW_CHUNK)
                .into_par_iter()
                .zip(w.axis_chunks_iter(Axis(0), ROW_CHUNK).into_par_iter())
                .map(|(xc, wc)| {
                    let mut wh = wc.dot(&ht.t());
                    ndarray::Zip::from(&mut wh)
                        .and(&xc)
                        .for_each(|p, &xv| *p = xv / p.max(eps));
                    wh.t().dot(&wc) // n×k
                })
                .reduce(|| Array2::<f64>::zeros((n, k)), |a, b| a + b);
            ndarray::Zip::indexed(&mut *ht).for_each(|(j, t), hv| {
                *hv *= numer_ht[[j, t]] / w_sums[t].max(eps);
            });
        }

        n_iter = it + 1;
        progress(n_iter as f32 / max_iter as f32);

        let einit = *error_init.get_or_insert(kl.max(eps));
        if it % 10 == 9 {
            if let Some(prev) = previous_error {
                if (prev - kl) / einit < tol {
                    break;
                }
            }
            previous_error = Some(kl);
        }
    }
    Ok(n_iter)
}

/// W ← W ⊙ ((X ⊘ WH)·H) / rowsum(H), row-chunk parallel and fused so the
/// full m×n product is never materialized. Returns the KL divergence
/// `Σ x·ln(x/wh) − x + wh` of the factors *before* the update.
fn mu_update_w(
    x: ArrayView2<f64>,
    w: &mut Array2<f64>,
    ht: &Array2<f64>,
    h_sums: &Array1<f64>,
    eps: f64,
) -> f64 {
    w.axis_chunks_iter_mut(Axis(0), ROW_CHUNK)
        .into_par_iter()
        .zip(x.axis_chunks_iter(Axis(0), ROW_CHUNK).into_par_iter())
        .map(|(mut wc, xc)| {
            let mut wh = wc.dot(&ht.t()); // c×n
            let mut kl_part = 0.0;
            ndarray::Zip::from(&mut wh).and(&xc).for_each(|p, &xv| {
                let pv = p.max(eps);
                kl_part += if xv > 0.0 {
                    xv * (xv / pv).ln() - xv + pv
                } else {
                    pv
                };
                *p = xv / pv; // reuse the buffer as X ⊘ WH
            });
            let numer = wh.dot(ht); // c×k
            ndarray::Zip::indexed(&mut wc).for_each(|(i, t), wv| {
                *wv *= numer[[i, t]] / h_sums[t].max(eps);
            });
            kl_part
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;

    /// Random-ish nonnegative rank-2 matrix.
    fn low_rank(m: usize, n: usize) -> Array2<f64> {
        let mut x = Array2::<f64>::zeros((m, n));
        for i in 0..m {
            for j in 0..n {
                let a = ((i as f64 * 0.31).sin() + 1.1) * ((j as f64 * 0.17).cos() + 1.2);
                let b = ((i as f64 * 0.05).cos() + 1.05) * ((j as f64 * 0.4).sin() + 1.3);
                x[[i, j]] = 2.0 * a + 0.7 * b;
            }
        }
        x
    }

    fn rel_err(x: &Array2<f64>, w: &Array2<f64>, ht: &Array2<f64>) -> f64 {
        let recon = w.dot(&ht.t());
        let num: f64 = (x - &recon).iter().map(|v| v * v).sum::<f64>().sqrt();
        let den: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        num / den
    }

    #[test]
    fn nndsvd_is_nonnegative() {
        let x = low_rank(40, 15);
        let (w, ht) = nndsvd_init(x.view(), 4, 7);
        assert!(w.iter().all(|&v| v >= 0.0));
        assert!(ht.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn frobenius_nmf_reconstructs_low_rank_data() {
        let x = low_rank(60, 25);
        let cancel = AtomicBool::new(false);
        let cfg = NmfConfig {
            n_components: 3,
            beta_loss: BetaLoss::Frobenius,
            tol: 1e-10,
            max_iter: 300,
            seed: 1,
        };
        let res = nmf(x.view(), &cfg, None, &cancel, &mut |_| {}).unwrap();
        assert!(res.w.iter().all(|&v| v >= 0.0));
        assert!(res.ht.iter().all(|&v| v >= 0.0));
        let err = rel_err(&x, &res.w, &res.ht);
        assert!(err < 1e-3, "relative error {err}");
    }

    #[test]
    fn kl_nmf_reconstructs_low_rank_data() {
        let x = low_rank(50, 20);
        let cancel = AtomicBool::new(false);
        let cfg = NmfConfig {
            n_components: 3,
            beta_loss: BetaLoss::KullbackLeibler,
            tol: 1e-10,
            max_iter: 400,
            seed: 1,
        };
        let res = nmf(x.view(), &cfg, None, &cancel, &mut |_| {}).unwrap();
        let err = rel_err(&x, &res.w, &res.ht);
        assert!(err < 5e-2, "relative error {err}");
    }

    #[test]
    fn fixed_basis_transform_fits_w_only() {
        let x = low_rank(60, 25);
        let cancel = AtomicBool::new(false);
        let cfg = NmfConfig {
            n_components: 3,
            beta_loss: BetaLoss::Frobenius,
            tol: 1e-10,
            max_iter: 300,
            seed: 1,
        };
        // Learn a basis on the data, then transform with it held fixed.
        let learned = nmf(x.view(), &cfg, None, &cancel, &mut |_| {}).unwrap();
        let res = nmf(x.view(), &cfg, Some(&learned.ht), &cancel, &mut |_| {}).unwrap();
        assert_eq!(res.ht, learned.ht, "fixed basis must not change");
        let err = rel_err(&x, &res.w, &res.ht);
        assert!(err < 1e-2, "relative error {err}");
    }

    #[test]
    fn cancellation_stops_the_solver() {
        let x = low_rank(30, 10);
        let cancel = AtomicBool::new(true);
        let cfg = NmfConfig {
            n_components: 2,
            beta_loss: BetaLoss::Frobenius,
            tol: 1e-10,
            max_iter: 100,
            seed: 1,
        };
        assert!(nmf(x.view(), &cfg, None, &cancel, &mut |_| {}).is_err());
    }
}
