//! Gaussian-process mechanism fit.

use std::sync::Arc;

use super::{FitFrame, MechanismSlot};

#[cfg(feature = "gaussian-process")]
use crate::error::ModelError;
#[cfg(feature = "gaussian-process")]
use antecedent_stats::{chol_log_det, chol_solve, cholesky_spd};

/// Largest sample the exact [`super::MechanismFamily::GaussianProcess`] grid search fits.
///
/// The exact fit runs a 25-cell `(ℓ, σ)` grid where every cell builds an O(n²) dense
/// Gram matrix and factors it with an O(n³) Cholesky, and family scoring refits it on every
/// training fold. At n = 1 000 that is an 8 MB Gram and about six full grid searches of
/// ≈ 25 · n³/3 flops each: seconds on current hardware and the last point where the exact GP
/// is a reasonable candidate. Above it the fit switches to the inducing-point (deterministic
/// training conditional) approximation of [`GP_INDUCING_POINTS`] points chosen by
/// farthest-point sampling, which costs O(n · m²) per length scale and O(m²) memory, and
/// produces the same fitted slot (inducing points as the training set).
#[cfg(feature = "gaussian-process")]
pub const GP_EXACT_MAX_ROWS: usize = 1_000;

/// Inducing points of the sparse Gaussian-process fit used above [`GP_EXACT_MAX_ROWS`].
#[cfg(feature = "gaussian-process")]
pub const GP_INDUCING_POINTS: usize = 256;

/// Flop budget of the sparse fit, `5 · n · m²` (one Gram pass per length scale). A request
/// beyond it is refused with [`ModelError::Unsupported`] rather than running for hours.
#[cfg(feature = "gaussian-process")]
const GP_SPARSE_WORK_BUDGET: f64 = 4.0e11;

/// Relative diagonal jitter of the inducing-point kernel matrix.
#[cfg(feature = "gaussian-process")]
const INDUCING_JITTER: f64 = 1e-6;

/// Hyperparameter grid of the GP fit, on the data's own scale.
///
/// Length scales are multiples of the parents' RMS spread and noise levels are
/// multiples of the outcome's SD, so the grid — and therefore the fit — is
/// equivariant to a change of units of the outcome and of every parent. The
/// signal variance is the outcome's variance and the prior mean its sample mean
/// (the fitted surface reverts to that mean off-support, not to zero).
#[cfg(feature = "gaussian-process")]
#[derive(Clone, Debug)]
pub(crate) struct GpGrid {
    /// Sample mean of the outcome.
    pub mean: f64,
    /// Population variance of the outcome.
    pub variance: f64,
    /// Length scales, as multiples of the parents' RMS spread.
    pub length_scales: [f64; 5],
    /// Noise standard deviations, as multiples of the outcome SD.
    pub noise_stds: [f64; 5],
}

/// Unit-equivariant `(length scale, noise)` grid for one Gaussian-process fit.
#[cfg(feature = "gaussian-process")]
pub(crate) fn gp_grid(parents: &[&[f64]], y: &[f64]) -> Result<GpGrid, ModelError> {
    let n = y.len().max(1) as f64;
    let mean = y.iter().sum::<f64>() / n;
    let variance = y.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
    if !(variance.is_finite() && variance > 0.0) {
        return Err(ModelError::Unsupported {
            message: "GaussianProcess needs an outcome with positive variance".into(),
        });
    }
    let x_var = parents
        .iter()
        .map(|c| {
            let m = c.iter().sum::<f64>() / n;
            c.iter().map(|v| (v - m).powi(2)).sum::<f64>() / n
        })
        .sum::<f64>()
        / parents.len().max(1) as f64;
    if !(x_var.is_finite() && x_var > 0.0) {
        return Err(ModelError::Unsupported {
            message: "GaussianProcess needs at least one parent that varies".into(),
        });
    }
    let (s_x, s_y) = (x_var.sqrt(), variance.sqrt());
    Ok(GpGrid {
        mean,
        variance,
        length_scales: [0.25, 0.5, 1.0, 2.0, 4.0].map(|m| m * s_x),
        noise_stds: [0.05, 0.1, 0.2, 0.5, 1.0].map(|m| m * s_y),
    })
}

#[cfg(feature = "gaussian-process")]
/// Grid-search RBF GP hyperparameters by exact Cholesky NLML.
///
/// For each `(ℓ, σ)` cell of [`gp_grid`], form `K = k_RBF + σ²I`, factor once with
/// [`cholesky_spd`], reuse that factor for both `log|K|` ([`chol_log_det`]) and
/// `α = K⁻¹(y − ȳ)` ([`chol_solve`]). Do not proxy the determinant by `Σ log Kᵢᵢ`
/// (MM-015).
pub(super) fn fit_gaussian_process(
    frame: &FitFrame<'_>,
    y: &[f64],
) -> Result<MechanismSlot, ModelError> {
    let n = y.len();
    let p = frame.n_parents();
    if p == 0 {
        return Err(ModelError::Unsupported {
            message: "GaussianProcess requires at least one parent".into(),
        });
    }
    let mut x_train = vec![0.0; n * p];
    for r in 0..n {
        for c in 0..p {
            x_train[r * p + c] = frame.parents[c][r];
        }
    }
    let refs: Vec<&[f64]> = frame.parents.iter().map(|c| &c[..n]).collect();
    let grid = gp_grid(&refs, y)?;
    let yc: Vec<f64> = y.iter().map(|v| v - grid.mean).collect();
    let variance = grid.variance;
    if n > GP_EXACT_MAX_ROWS {
        return fit_inducing_gaussian_process(&x_train, &yc, n, p, &grid);
    }
    let mut best = None::<(f64, f64, f64, Vec<f64>)>; // (nlml, ℓ, σ, α)
    for &length_scale in &grid.length_scales {
        for &noise_std in &grid.noise_stds {
            let mut k = vec![0.0; n * n];
            let inv_l2 = 1.0 / (length_scale * length_scale);
            for i in 0..n {
                for j in i..n {
                    let mut d2 = 0.0;
                    for c in 0..p {
                        let d = x_train[i * p + c] - x_train[j * p + c];
                        d2 += d * d;
                    }
                    let kij = variance * (-0.5 * d2 * inv_l2).exp();
                    k[i * n + j] = kij;
                    k[j * n + i] = kij;
                }
                k[i * n + i] += noise_std * noise_std;
            }
            let Some(chol) = cholesky_spd(&k, n) else {
                continue;
            };
            let Some(alpha) = chol_solve(&chol, n, &yc) else {
                continue;
            };
            let mut y_alpha = 0.0;
            for i in 0..n {
                y_alpha += yc[i] * alpha[i];
            }
            let nlml = 0.5 * y_alpha
                + 0.5 * chol_log_det(&chol, n)
                + 0.5 * n as f64 * (2.0 * std::f64::consts::PI).ln();
            match &best {
                Some((best_nlml, ..)) if nlml >= *best_nlml => {}
                _ => best = Some((nlml, length_scale, noise_std, alpha)),
            }
        }
    }
    let (_nlml, length_scale, noise_std, alpha) = best.ok_or_else(|| ModelError::Numerical {
        message: "GP hyperparameter search failed".into(),
    })?;
    Ok(MechanismSlot::GaussianProcess {
        length_scale,
        variance,
        noise_std,
        mean: grid.mean,
        x_train: Arc::from(x_train),
        n_train: n,
        n_parents: p,
        alpha: Arc::from(alpha),
    })
}

/// Squared Euclidean distance between rows `i` and `j` of the row-major `n x p` matrix.
#[cfg(feature = "gaussian-process")]
fn row_distance_sq(x: &[f64], p: usize, i: usize, j: usize) -> f64 {
    let (a, b) = (&x[i * p..(i + 1) * p], &x[j * p..(j + 1) * p]);
    a.iter().zip(b).map(|(u, v)| (u - v) * (u - v)).sum()
}

/// Deterministic farthest-point selection of up to `m` distinct rows, starting at row 0.
#[cfg(feature = "gaussian-process")]
fn farthest_point_rows(x: &[f64], n: usize, p: usize, m: usize) -> Vec<usize> {
    let mut chosen = vec![0usize];
    let mut nearest: Vec<f64> = (0..n).map(|i| row_distance_sq(x, p, i, 0)).collect();
    while chosen.len() < m {
        let (mut best, mut best_d2) = (0usize, 0.0_f64);
        for (i, &d2) in nearest.iter().enumerate() {
            if d2 > best_d2 {
                (best, best_d2) = (i, d2);
            }
        }
        if best_d2 <= 0.0 {
            break;
        }
        chosen.push(best);
        for (i, slot) in nearest.iter_mut().enumerate() {
            *slot = slot.min(row_distance_sq(x, p, i, best));
        }
    }
    chosen
}

/// Inducing-point (deterministic training conditional) Gaussian-process fit.
///
/// With inducing set `Z` (`m` rows), `K_mm = k(Z, Z)` and `K_mn = k(Z, X)`, the marginal
/// likelihood is that of `y ~ N(0, K_nm K_mm⁻¹ K_mn + σ²I)`. By the matrix-determinant lemma
/// and Woodbury, with `A = σ²K_mm + K_mn K_nm`:
/// `log|Q + σ²I| = (n − m) ln σ² + log|A| − log|K_mm|` and
/// `yᵀ(Q + σ²I)⁻¹y = (yᵀy − bᵀA⁻¹b) / σ²` for `b = K_mn y`. The predictive mean is
/// `Σⱼ βⱼ k(x, zⱼ)` with `β = A⁻¹b`, which is the dual form of the fitted slot with the
/// inducing points as its training set. `K_mn K_nm` and `b` are accumulated row by row, so
/// memory is O(m²).
#[cfg(feature = "gaussian-process")]
fn fit_inducing_gaussian_process(
    x_train: &[f64],
    yc: &[f64],
    n: usize,
    p: usize,
    grid: &GpGrid,
) -> Result<MechanismSlot, ModelError> {
    let rows = farthest_point_rows(x_train, n, p, GP_INDUCING_POINTS.min(n));
    let m = rows.len();
    let work = 5.0 * n as f64 * (m as f64) * (m as f64);
    if work > GP_SPARSE_WORK_BUDGET {
        return Err(ModelError::Unsupported {
            message: format!(
                "GaussianProcess inducing-point fit of {n} rows needs about {work:.2e} flops, \
                 above the {GP_SPARSE_WORK_BUDGET:.0e} budget; use another mechanism family (or \
                 subsample) for data this large"
            ),
        });
    }
    let variance = grid.variance;
    let mut z = vec![0.0; m * p];
    for (a, &row) in rows.iter().enumerate() {
        z[a * p..(a + 1) * p].copy_from_slice(&x_train[row * p..(row + 1) * p]);
    }
    let yy: f64 = yc.iter().map(|v| v * v).sum();
    let mut best = None::<(f64, f64, f64, Vec<f64>)>; // (nlml, ℓ, σ, β)
    let mut k_row = vec![0.0; m];
    for &length_scale in &grid.length_scales {
        let inv_l2 = 1.0 / (length_scale * length_scale);
        let mut k_mm = vec![0.0; m * m];
        for a in 0..m {
            for b in a..m {
                let kab = variance * (-0.5 * row_distance_sq(&z, p, a, b) * inv_l2).exp();
                k_mm[a * m + b] = kab;
                k_mm[b * m + a] = kab;
            }
            k_mm[a * m + a] += variance * INDUCING_JITTER;
        }
        let Some(chol_mm) = cholesky_spd(&k_mm, m) else {
            continue;
        };
        let log_det_mm = chol_log_det(&chol_mm, m);
        let mut gram = vec![0.0; m * m];
        let mut rhs = vec![0.0; m];
        for i in 0..n {
            let xi = &x_train[i * p..(i + 1) * p];
            for (a, slot) in k_row.iter_mut().enumerate() {
                let za = &z[a * p..(a + 1) * p];
                let d2: f64 = xi.iter().zip(za).map(|(u, v)| (u - v) * (u - v)).sum();
                *slot = variance * (-0.5 * d2 * inv_l2).exp();
            }
            for a in 0..m {
                let ka = k_row[a];
                rhs[a] += ka * yc[i];
                let row = &mut gram[a * m..(a + 1) * m];
                for (slot, &kb) in row[a..].iter_mut().zip(&k_row[a..]) {
                    *slot += ka * kb;
                }
            }
        }
        for a in 0..m {
            for b in 0..a {
                gram[a * m + b] = gram[b * m + a];
            }
        }
        for &noise_std in &grid.noise_stds {
            let noise_var = noise_std * noise_std;
            let a_matrix: Vec<f64> =
                gram.iter().zip(&k_mm).map(|(g, k)| g + noise_var * k).collect();
            let Some(chol_a) = cholesky_spd(&a_matrix, m) else {
                continue;
            };
            let Some(beta) = chol_solve(&chol_a, m, &rhs) else {
                continue;
            };
            let b_beta: f64 = rhs.iter().zip(&beta).map(|(b, w)| b * w).sum();
            let quad = (yy - b_beta) / noise_var;
            let log_det = (n - m) as f64 * noise_var.ln() + chol_log_det(&chol_a, m) - log_det_mm;
            let nlml = 0.5 * quad
                + 0.5 * log_det
                + 0.5 * n as f64 * (2.0 * std::f64::consts::PI).ln();
            if !nlml.is_finite() {
                continue;
            }
            match &best {
                Some((best_nlml, ..)) if nlml >= *best_nlml => {}
                _ => best = Some((nlml, length_scale, noise_std, beta)),
            }
        }
    }
    let (_nlml, length_scale, noise_std, beta) = best.ok_or_else(|| ModelError::Numerical {
        message: "GP inducing-point hyperparameter search failed".into(),
    })?;
    Ok(MechanismSlot::GaussianProcess {
        length_scale,
        variance,
        noise_std,
        mean: grid.mean,
        x_train: Arc::from(z),
        n_train: m,
        n_parents: p,
        alpha: Arc::from(beta),
    })
}
