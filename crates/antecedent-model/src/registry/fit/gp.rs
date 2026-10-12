//! Gaussian-process mechanism fit.

#[cfg(feature = "gaussian-process")]
use crate::error::ModelError;
#[cfg(feature = "gaussian-process")]
use antecedent_stats::{chol_log_det, chol_solve, cholesky_spd};

/// Row cap for the [`MechanismFamily::GaussianProcess`] grid search.
///
/// The fit runs a 25-cell `(ℓ, σ)` grid where every cell builds an O(n²) dense
/// Gram matrix and factors it with an O(n³) Cholesky, and family scoring refits
/// it on every training fold. At n = 1 000 that is an 8 MB Gram and about six
/// full grid searches of ≈ 25 · n³/3 flops each — seconds on current hardware and
/// the last point where the exact GP is a reasonable candidate; n = 10 000
/// would already need 800 MB and minutes per node. Above the cap
/// `fit_family(GaussianProcess)` refuses with [`ModelError::Unsupported`]
/// instead of silently hanging; because model selection records failed
/// families and picks among the rest, the refusal is surfaced in
/// [`MechanismAssignment::failed_families`] and another family is selected.
#[cfg(feature = "gaussian-process")]
pub const GP_FAMILY_MAX_ROWS: usize = 1_000;

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
    if n > GP_FAMILY_MAX_ROWS {
        return Err(ModelError::Unsupported {
            message: format!(
                "GaussianProcess is limited to {GP_FAMILY_MAX_ROWS} rows (got {n}): the exact \
                 Cholesky grid search is O(n³) per cell with an O(n²) Gram matrix; use another \
                 mechanism family (or subsample) for larger data"
            ),
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
