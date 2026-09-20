//! Minimal dense linear algebra for the NMF solver: rayon-parallel matrix
//! products, modified Gram-Schmidt orthonormalization, a Jacobi symmetric
//! eigensolver, and a seeded randomized truncated SVD (used by the NNDSVD
//! initialization). No BLAS/LAPACK dependency — `ndarray`'s built-in
//! `matrixmultiply` kernels do the per-chunk work.

use ndarray::parallel::prelude::*;
use ndarray::{Array1, Array2, ArrayView2, Axis};

/// Row-chunk size for the parallel products: big enough that each chunk's
/// matmul amortizes the scheduling, small enough to load-balance.
pub const ROW_CHUNK: usize = 2048;

/// `A · B`, parallelized over row chunks of `A`.
pub fn par_matmul(a: ArrayView2<f64>, b: ArrayView2<f64>) -> Array2<f64> {
    let (m, n) = (a.nrows(), b.ncols());
    let mut c = Array2::<f64>::zeros((m, n));
    c.axis_chunks_iter_mut(Axis(0), ROW_CHUNK)
        .into_par_iter()
        .zip(a.axis_chunks_iter(Axis(0), ROW_CHUNK).into_par_iter())
        .for_each(|(mut cc, aa)| cc.assign(&aa.dot(&b)));
    c
}

/// Gram matrix `Aᵀ · A` (k×k for a tall m×k input), parallelized over row
/// chunks with a sum reduction.
pub fn par_gram(a: ArrayView2<f64>) -> Array2<f64> {
    let k = a.ncols();
    a.axis_chunks_iter(Axis(0), ROW_CHUNK)
        .into_par_iter()
        .map(|c| c.t().dot(&c))
        .reduce(|| Array2::<f64>::zeros((k, k)), |x, y| x + y)
}

/// Orthonormalize the columns of `y` in place (modified Gram-Schmidt).
/// Columns that become numerically zero are left as zeros.
pub fn orthonormalize_columns(y: &mut Array2<f64>) {
    let l = y.ncols();
    for j in 0..l {
        for i in 0..j {
            let dot = y.column(i).dot(&y.column(j));
            let col_i = y.column(i).to_owned();
            y.column_mut(j).zip_mut_with(&col_i, |a, &b| *a -= dot * b);
        }
        let norm = y.column(j).dot(&y.column(j)).sqrt();
        if norm > 1e-12 {
            y.column_mut(j).mapv_inplace(|v| v / norm);
        } else {
            y.column_mut(j).fill(0.0);
        }
    }
}

/// Eigendecomposition of a small symmetric matrix by cyclic Jacobi rotations.
/// Returns `(eigenvalues, eigenvectors-as-columns)` sorted by descending
/// eigenvalue.
pub fn symmetric_eigen(a: &Array2<f64>) -> (Array1<f64>, Array2<f64>) {
    let n = a.nrows();
    assert_eq!(n, a.ncols(), "symmetric_eigen needs a square matrix");
    let mut a = a.clone();
    let mut v = Array2::<f64>::eye(n);

    for _sweep in 0..100 {
        let mut off = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                off += a[[i, j]] * a[[i, j]];
            }
        }
        if off.sqrt() < 1e-14 {
            break;
        }
        for p in 0..n.saturating_sub(1) {
            for q in (p + 1)..n {
                let apq = a[[p, q]];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let theta = (a[[q, q]] - a[[p, p]]) / (2.0 * apq);
                let sign = if theta >= 0.0 { 1.0 } else { -1.0 };
                let t = sign / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for i in 0..n {
                    let aip = a[[i, p]];
                    let aiq = a[[i, q]];
                    a[[i, p]] = c * aip - s * aiq;
                    a[[i, q]] = s * aip + c * aiq;
                }
                for i in 0..n {
                    let api = a[[p, i]];
                    let aqi = a[[q, i]];
                    a[[p, i]] = c * api - s * aqi;
                    a[[q, i]] = s * api + c * aqi;
                }
                for i in 0..n {
                    let vip = v[[i, p]];
                    let viq = v[[i, q]];
                    v[[i, p]] = c * vip - s * viq;
                    v[[i, q]] = s * vip + c * viq;
                }
            }
        }
    }

    // Sort by descending eigenvalue, permuting the eigenvector columns along.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| a[[j, j]].partial_cmp(&a[[i, i]]).unwrap());
    let evals = Array1::from_iter(order.iter().map(|&i| a[[i, i]]));
    let mut evecs = Array2::<f64>::zeros((n, n));
    for (dst, &src) in order.iter().enumerate() {
        evecs.column_mut(dst).assign(&v.column(src));
    }
    (evals, evecs)
}

/// Truncated SVD result: `x ≈ u · diag(s) · vt`.
pub struct Svd {
    pub u: Array2<f64>,
    pub s: Array1<f64>,
    pub vt: Array2<f64>,
}

/// Randomized truncated SVD of `x` (m×n) with `k` components, seeded so the
/// NNDSVD initialization (and therefore the whole NMF) is reproducible.
/// Two power iterations with re-orthonormalization — plenty for an
/// initialization and matches sklearn's `randomized_svd` accuracy regime.
pub fn randomized_svd(x: ArrayView2<f64>, k: usize, seed: u64) -> Svd {
    let (m, n) = x.dim();
    let k = k.min(m).min(n).max(1);
    let l = (k + 10).min(m).min(n);

    let mut rng = Rng::new(seed);
    let omega = Array2::from_shape_fn((n, l), |_| rng.normal());

    let mut q = par_matmul(x, omega.view()); // m×l
    orthonormalize_columns(&mut q);
    for _ in 0..2 {
        let mut z = par_matmul(x.t(), q.view()); // n×l
        orthonormalize_columns(&mut z);
        q = par_matmul(x, z.view());
        orthonormalize_columns(&mut q);
    }

    let b = par_matmul(q.t(), x); // l×n
    let bbt = b.dot(&b.t()); // l×l
    let (evals, evecs) = symmetric_eigen(&bbt);

    let s_full = evals.mapv(|v| v.max(0.0).sqrt());
    let u_full = q.dot(&evecs); // m×l
    let mut vt_full = evecs.t().dot(&b); // l×n
    for (i, mut row) in vt_full.axis_iter_mut(Axis(0)).enumerate() {
        let si = s_full[i];
        if si > 1e-12 {
            row.mapv_inplace(|v| v / si);
        } else {
            row.fill(0.0);
        }
    }

    Svd {
        u: u_full.slice(ndarray::s![.., ..k]).to_owned(),
        s: s_full.slice(ndarray::s![..k]).to_owned(),
        vt: vt_full.slice(ndarray::s![..k, ..]).to_owned(),
    }
}

/// Small deterministic RNG (xorshift64*) with a Box-Muller normal sampler and
/// a Fisher-Yates shuffle — enough randomness for the SVD test matrix and the
/// batch row permutation without pulling in a dependency.
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed | 1, // never zero
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in [0, 1).
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal (Box-Muller).
    pub fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-300);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    pub fn shuffle(&mut self, v: &mut [usize]) {
        for i in (1..v.len()).rev() {
            let j = (self.next_u64() % (i as u64 + 1)) as usize;
            v.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn par_matmul_matches_dot() {
        let a = Array2::from_shape_fn((37, 5), |(i, j)| (i * 5 + j) as f64 * 0.1);
        let b = Array2::from_shape_fn((5, 4), |(i, j)| (i as f64) - (j as f64) * 0.5);
        let c = par_matmul(a.view(), b.view());
        let d = a.dot(&b);
        assert!((&c - &d).iter().all(|v| v.abs() < 1e-12));
    }

    #[test]
    fn gram_matches_transpose_product() {
        let a = Array2::from_shape_fn((100, 3), |(i, j)| ((i + j) as f64).sin());
        let g = par_gram(a.view());
        let d = a.t().dot(&a);
        assert!((&g - &d).iter().all(|v| v.abs() < 1e-10));
    }

    #[test]
    fn jacobi_eigen_recovers_known_eigenvalues() {
        let a = array![[2.0, 1.0], [1.0, 2.0]];
        let (vals, vecs) = symmetric_eigen(&a);
        assert!((vals[0] - 3.0).abs() < 1e-10);
        assert!((vals[1] - 1.0).abs() < 1e-10);
        // A v = λ v for the leading pair.
        let av = a.dot(&vecs.column(0));
        let lv = vecs.column(0).mapv(|v| v * vals[0]);
        assert!((&av - &lv).iter().all(|v| v.abs() < 1e-10));
    }

    #[test]
    fn randomized_svd_reconstructs_low_rank_matrix() {
        // Rank-2 matrix: outer products of two fixed vectors.
        let u1 = Array1::from_shape_fn(50, |i| (i as f64 * 0.3).sin() + 2.0);
        let u2 = Array1::from_shape_fn(50, |i| (i as f64 * 0.11).cos());
        let v1 = Array1::from_shape_fn(20, |j| (j as f64 * 0.7).cos() + 1.5);
        let v2 = Array1::from_shape_fn(20, |j| (j as f64 * 0.2).sin());
        let mut x = Array2::<f64>::zeros((50, 20));
        for i in 0..50 {
            for j in 0..20 {
                x[[i, j]] = 3.0 * u1[i] * v1[j] + 0.5 * u2[i] * v2[j];
            }
        }
        let svd = randomized_svd(x.view(), 2, 42);
        let recon = svd.u.dot(&Array2::from_diag(&svd.s)).dot(&svd.vt);
        let err: f64 = (&x - &recon).iter().map(|v| v * v).sum::<f64>().sqrt();
        let norm: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(err / norm < 1e-8, "relative error {}", err / norm);
    }
}
