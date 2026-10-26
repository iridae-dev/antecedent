//! Extracted stage of the parent estimator module.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

/// Coefficients and raw residuals of one linear solve, before the variance model is attached.
pub(super) struct SolvedLinear {
    pub(super) coefficients: Vec<f64>,
    pub(super) residuals: Vec<f64>,
    pub(super) rss: f64,
    /// Huber IRLS scale. Absent for OLS, ridge, and lasso.
    pub(super) huber_scale: Option<f64>,
}

/// One solver body for the point fit and for bootstrap resamples.
pub(super) fn solve_linear(
    kind: LinearFitKind,
    backend: FaerBackend,
    x: &[f64],
    n: usize,
    p: usize,
    y: &[f64],
    workspace: &mut LeastSquaresWorkspace,
) -> Result<SolvedLinear, EstimationError> {
    match kind {
        LinearFitKind::Ols => {
            let fit =
                backend.least_squares(x, n, p, y, workspace).map_err(EstimationError::from)?;
            Ok(SolvedLinear {
                coefficients: fit.coefficients,
                residuals: fit.residuals,
                rss: fit.rss,
                huber_scale: None,
            })
        }
        LinearFitKind::Ridge { lambda } => {
            let fit = fit_ridge_with(x, n, p, y, lambda, RidgeIntercept::sniff(x, n, p))
                .map_err(EstimationError::from)?;
            Ok(SolvedLinear {
                coefficients: fit.coefficients,
                residuals: fit.residuals,
                rss: fit.rss,
                huber_scale: None,
            })
        }
        LinearFitKind::Lasso { lambda } => {
            let fit = fit_lasso_with_ones_column(
                x,
                n,
                p,
                y,
                &LassoOptions { lambda, fit_intercept: true, ..LassoOptions::default() },
            )
            .map_err(EstimationError::from)?;
            // An unconverged replicate is a snapshot, not a draw from the estimator's
            // sampling distribution; the caller counts it as a failed replicate.
            require_converged(fit.converged, LASSO_UNCONVERGED)?;
            // Design keeps an all-ones column; stitch intercept back for g-computation.
            let mut coefficients = Vec::with_capacity(p);
            coefficients.push(fit.intercept);
            coefficients.extend_from_slice(&fit.coefficients);
            let pred = if fit.coefficients.is_empty() {
                vec![fit.intercept; n]
            } else {
                predict_lasso(&fit, &x[n..], n, p - 1).map_err(EstimationError::from)?
            };
            let mut residuals = vec![0.0; n];
            let mut rss = 0.0;
            for r in 0..n {
                let e = y[r] - pred[r];
                residuals[r] = e;
                rss += e * e;
            }
            Ok(SolvedLinear { coefficients, residuals, rss, huber_scale: None })
        }
        LinearFitKind::Huber { c } => {
            let opts = MEstimateOptions { c, ..MEstimateOptions::default() };
            let fit = fit_huber_m(x, n, p, y, &opts, &backend, workspace)
                .map_err(EstimationError::from)?;
            // An unconverged IRLS run is a snapshot of an unfinished reweighting
            // sequence, not the M-estimator's fixed point; publishing it — and an
            // analytic SE derived from it — would attach robust-estimand semantics
            // to a number that has none. Mirrors the GAM backfit gate in
            // `response.rs::fit_additive`.
            require_converged(fit.converged, HUBER_UNCONVERGED)?;
            let mut residuals = vec![0.0; n];
            let mut rss = 0.0;
            for r in 0..n {
                let mut pred = 0.0;
                for col in 0..p {
                    pred += x[col * n + r] * fit.coefficients[col];
                }
                let e = y[r] - pred;
                residuals[r] = e;
                rss += e * e;
            }
            Ok(SolvedLinear {
                coefficients: fit.coefficients,
                residuals,
                rss,
                huber_scale: Some(fit.scale),
            })
        }
    }
}

pub(super) fn linear_fit_from_solved(
    kind: LinearFitKind,
    solved: SolvedLinear,
) -> Result<LinearFit, EstimationError> {
    let variance = match kind {
        LinearFitKind::Ols => FitVariance::Ols,
        LinearFitKind::Ridge { lambda } => FitVariance::Ridge { lambda },
        // Permanent policy: no analytic SE for Lasso (bootstrap only).
        LinearFitKind::Lasso { .. } => FitVariance::Lasso,
        LinearFitKind::Huber { c } => {
            // `solve_linear` records the IRLS scale on the Huber arm before returning.
            let scale = solved
                .huber_scale
                .ok_or_else(|| EstimationError::stats_msg("Huber fit is missing its IRLS scale"))?;
            let bound = c * scale;
            // Estimating equation `Σ x_i · s·ψ_c(r_i/s) = 0`: the score is the residual
            // clipped at `c·s`, and its derivative in the linear predictor is the
            // indicator of the unclipped region.
            let scores: Vec<f64> =
                solved.residuals.iter().map(|r| r.clamp(-bound, bound)).collect();
            let curvature: Vec<f64> =
                solved.residuals.iter().map(|r| f64::from(u8::from(r.abs() <= bound))).collect();
            FitVariance::Huber { scores, curvature }
        }
    };
    Ok(LinearFit {
        coefficients: solved.coefficients,
        residuals: solved.residuals,
        rss: solved.rss,
        variance,
    })
}

/// Fit failures and an empty target arm are failed replicates. A resample can be
/// singular, unconverged, or contain none of the target arm; those continue.
/// Shape errors in the caller's buffers propagate.
pub(super) fn bootstrap_fit_failed(error: &EstimationError) -> bool {
    match error {
        EstimationError::Stats(_) => true,
        EstimationError::Unsupported { message }
            if *message == LASSO_UNCONVERGED || *message == HUBER_UNCONVERGED =>
        {
            true
        }
        EstimationError::Data(antecedent_data::DataError::InvalidArgument { message })
            if message == "target population left no rows for g-computation" =>
        {
            true
        }
        _ => false,
    }
}

/// G-computation of μ(active,Z)−μ(control,Z) averaged under the target arm’s covariate law.
///
/// Under a linear main-effects model this equals `β_T · Δ` for every target, including ATT/ATC.
pub(super) fn gcomp_or_coef_ate(
    problem: &PreparedEstimationProblem,
    coefficients: &[f64],
    t_col: usize,
) -> Result<f64, EstimationError> {
    match problem.target_population {
        TargetPopulation::AllObserved | TargetPopulation::Predicate(_) => {
            Ok(coefficients[t_col] * problem.treatment_delta)
        }
        // The design is the weighted transform, which g-computation cannot average over. Under
        // the main-effects model every target has the effect `β_T · Δ`.
        TargetPopulation::Treated | TargetPopulation::Untreated
            if problem.sampling_weights.is_some() =>
        {
            Ok(coefficients[t_col] * problem.treatment_delta)
        }
        TargetPopulation::Treated | TargetPopulation::Untreated => {
            let n = problem.design.nrows;
            let ncols = problem.design.ncols;
            let want_treated = matches!(problem.target_population, TargetPopulation::Treated);
            let mut sum = 0.0;
            let mut count = 0usize;
            for r in 0..n {
                let treated = TreatmentArm::is_treated_indicator(problem.treatment[r]);
                if treated != want_treated {
                    continue;
                }
                let mut pred_a = 0.0;
                let mut pred_c = 0.0;
                for (c, &coef) in coefficients.iter().enumerate().take(ncols) {
                    let x = if c == t_col {
                        (problem.active, problem.control)
                    } else {
                        let v = problem.design.matrix[c * n + r];
                        (v, v)
                    };
                    pred_a += coef * x.0;
                    pred_c += coef * x.1;
                }
                sum += pred_a - pred_c;
                count += 1;
            }
            if count == 0 {
                return Err(EstimationError::data_msg(
                    "target population left no rows for g-computation",
                ));
            }
            Ok(sum / count as f64)
        }
        _ => Err(EstimationError::TargetPopulation),
    }
}

pub(crate) fn intervention_f64(intervention: &Intervention) -> Result<f64, EstimationError> {
    match intervention {
        Intervention::Set { value, .. } => value.as_f64().ok_or_else(|| {
            EstimationError::unsupported(" linear adjustment requires numeric treatment levels")
        }),
        _ => Err(EstimationError::unsupported(" linear adjustment requires Set interventions")),
    }
}

/// Fitted coefficients plus the variance model of the estimator that produced them.
pub(super) struct LinearFit {
    pub(super) coefficients: Vec<f64>,
    /// Raw residuals `y − Xβ̂`.
    pub(super) residuals: Vec<f64>,
    pub(super) rss: f64,
    pub(super) variance: FitVariance,
}

/// Which sampling-variance formula describes a [`LinearFit`].
pub(super) enum FitVariance {
    Ols,
    /// Bread `(XᵀX + λP)⁻¹`; no analytic SE (shrinkage bias is outside a sampling variance).
    Ridge {
        lambda: f64,
    },
    /// No analytic SE, no influence function (post-selection).
    Lasso,
    /// `scores[i] = s·ψ(r_i/s)`, `curvature[i] = ψ′(r_i/s)`.
    Huber {
        scores: Vec<f64>,
        curvature: Vec<f64>,
    },
}

/// Refuse an iterative fit that stopped at its iteration cap.
pub(super) fn require_converged(
    converged: bool,
    refusal: &'static str,
) -> Result<(), EstimationError> {
    if converged { Ok(()) } else { Err(EstimationError::unsupported(refusal)) }
}

pub(super) const LASSO_UNCONVERGED: &str =
    "Lasso coordinate descent did not converge; refuse rather than publish an unfinished fit";

pub(super) const HUBER_UNCONVERGED: &str =
    "Huber M-estimator did not converge; refuse rather than publish an unfinished fit";

/// `(Σ d_i x_i x_iᵀ)⁻¹` (row-major) for nonnegative row weights `d`; `None` when singular.
pub(super) fn curvature_gram_inverse(
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    d: &[f64],
) -> Option<Vec<f64>> {
    let rows = nrows.min(d.len());
    let mut gram = vec![0.0; ncols * ncols];
    for a in 0..ncols {
        let col_a = &x_colmajor[a * nrows..a * nrows + rows];
        for b in a..ncols {
            let col_b = &x_colmajor[b * nrows..b * nrows + rows];
            let value: f64 = col_a.iter().zip(col_b).zip(d).map(|((xa, xb), w)| w * xa * xb).sum();
            gram[a * ncols + b] = value;
            gram[b * ncols + a] = value;
        }
    }
    antecedent_stats::invert_square(&gram, ncols)
}

/// Huber (1981, §7.10) homoskedastic-form treatment SE, used when the sandwich is
/// unavailable and the mean curvature `m = mean ψ′` is positive.
pub(super) fn huber_homoskedastic_se(
    x: &[f64],
    nrows: usize,
    ncols: usize,
    t_col: usize,
    scores: &[f64],
    mean_curvature: f64,
) -> f64 {
    let n = nrows as f64;
    let p = ncols as f64;
    let kappa = 1.0 + (p / n) * (1.0 - mean_curvature) / mean_curvature;
    let tau2 =
        scores.iter().map(|e| e * e).sum::<f64>() / (n - p) / (mean_curvature * mean_curvature);
    crate::util::xtx_inverse(x, nrows, ncols)
        .map_or(f64::NAN, |inv| (kappa * kappa * tau2 * inv[t_col * ncols + t_col].max(0.0)).sqrt())
}

/// Frisch–Waugh influence of the treatment coefficient, scaled by `delta`.
///
/// `ψ_i = δ · n · [(XᵀX)⁻¹ x_i]_T · e_i`, which equals
/// `δ · n · t̃_i e_i / Σ t̃²` with `t̃` the treatment residualized on every
/// other design column. Centering the treatment alone (`t − t̄`) is only the
/// unadjusted IF: with covariates correlated with the treatment it understates
/// the variance by `1 − R²(T | Z)`. Non-invertible designs (`xtx_inv` is
/// `None`) return NaN so a downstream mixture SE fails closed instead of
/// reporting zero.
///
/// The leverage `[(XᵀX)⁻¹ x_i]_T` accumulates one design column at a time
/// (contiguous, vectorizable) rather than striding across columns per row.
pub(super) fn treatment_coef_influence(
    matrix: &[f64],
    nrows: usize,
    ncols: usize,
    t_col: usize,
    residuals: &[f64],
    delta: f64,
    xtx_inv: Option<&[f64]>,
) -> Vec<f64> {
    if nrows == 0 || residuals.len() != nrows || t_col >= ncols {
        return vec![0.0; residuals.len()];
    }
    let Some(inv) = xtx_inv else {
        return vec![f64::NAN; nrows];
    };
    let row = &inv[t_col * ncols..t_col * ncols + ncols];
    let scale = delta * nrows as f64;
    let mut psi = vec![0.0; nrows];
    for (c, &w) in row.iter().enumerate() {
        let column = &matrix[c * nrows..(c + 1) * nrows];
        for (slot, &x) in psi.iter_mut().zip(column) {
            *slot += w * x;
        }
    }
    for (slot, &e) in psi.iter_mut().zip(residuals) {
        *slot = scale * *slot * e;
    }
    psi
}
