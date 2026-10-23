//! Pooling of per-imputation fits: Rubin's rules with the Barnard–Rubin degrees of freedom
//! for estimators that report a point and a variance, and an equal-weight posterior mixture
//! for Bayesian fits.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::MultipleImputationError;

/// Rubin's-rules combination of `m` complete-data estimates and their variances.
///
/// With `Q̄ = mean(q)`, within variance `Ū = mean(u)`, between variance
/// `B = Σ(q − Q̄)² / (m − 1)` and total variance `T = Ū + (1 + 1/m) B`
/// (Rubin 1987). The large-sample degrees of freedom are `ν_m = (m − 1) / λ²` with
/// `λ = (1 + 1/m) B / T`; with a finite complete-data `ν_com` the Barnard–Rubin (1999)
/// adjustment is `ν = (1/ν_m + 1/ν_obs)⁻¹`, `ν_obs = (ν_com + 1)/(ν_com + 3) · ν_com · (1 − λ)`.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct RubinPooled {
    /// Number of imputations `m`.
    pub imputations: usize,
    /// Pooled point `Q̄`.
    pub point: f64,
    /// Within-imputation variance `Ū`.
    pub within: f64,
    /// Between-imputation variance `B`.
    pub between: f64,
    /// Total variance `T = Ū + (1 + 1/m) B`.
    pub total: f64,
    /// Relative increase in variance due to nonresponse, `r = (1 + 1/m) B / Ū`.
    pub relative_increase: f64,
    /// Proportion of the total variance due to missingness, `λ = (1 + 1/m) B / T`.
    pub lambda: f64,
    /// Rubin's large-sample degrees of freedom `ν_m`; infinite when `B = 0`.
    pub df_rubin: f64,
    /// Complete-data degrees of freedom used for the Barnard–Rubin adjustment; `None` for
    /// a large-sample (normal-reference) estimator.
    pub df_complete: Option<f64>,
    /// Degrees of freedom of the Student-t reference: Barnard–Rubin when `df_complete` is
    /// known, else `ν_m`. Infinite means the normal reference.
    pub df: f64,
    /// Fraction of missing information `γ = (r + 2/(ν + 3)) / (r + 1)`.
    pub fraction_missing_information: f64,
}

impl RubinPooled {
    /// Pooled standard error `√T`.
    #[must_use]
    pub fn standard_error(&self) -> f64 {
        self.total.sqrt()
    }

    /// Two-sided `level` interval `Q̄ ± t_{ν, (1+level)/2} √T` (normal quantile when `ν`
    /// is infinite). The interval is an unmeasured construction until a coverage record
    /// measures the whole impute-fit-pool procedure.
    #[must_use]
    pub fn interval(&self, level: f64) -> (f64, f64) {
        let p = 0.5 + 0.5 * level;
        let quantile = if self.df.is_finite() {
            antecedent_stats::student_t_ppf(p, self.df)
        } else {
            antecedent_kernels::norm_inv(p)
        };
        let half = quantile * self.standard_error();
        (self.point - half, self.point + half)
    }
}

/// Combine `points` and their `variances` by Rubin's rules.
///
/// # Errors
///
/// Fewer than two imputations, lengths that differ, a non-finite point, a negative or
/// non-finite variance, or a complete-data degrees of freedom that is not positive.
pub fn rubin_pool(
    points: &[f64],
    variances: &[f64],
    df_complete: Option<f64>,
) -> Result<RubinPooled, MultipleImputationError> {
    let m = points.len();
    if m < 2 {
        return Err(MultipleImputationError::invalid(
            "multiple_imputation.too_few_imputations",
            format!("Rubin's rules need at least two imputations, got {m}"),
        ));
    }
    if variances.len() != m {
        return Err(MultipleImputationError::invalid(
            "multiple_imputation.pool_shape",
            format!("{m} points and {} variances", variances.len()),
        ));
    }
    if points.iter().any(|q| !q.is_finite()) {
        return Err(MultipleImputationError::invalid(
            "multiple_imputation.pool_non_finite",
            "an imputation's point estimate is not finite",
        ));
    }
    if variances.iter().any(|u| !(u.is_finite() && *u >= 0.0)) {
        return Err(MultipleImputationError::invalid(
            "multiple_imputation.pool_non_finite",
            "an imputation's variance is negative or not finite",
        ));
    }
    if let Some(df) = df_complete {
        if !(df.is_finite() && df > 0.0) {
            return Err(MultipleImputationError::invalid(
                "multiple_imputation.pool_df",
                format!("complete-data degrees of freedom must be positive, got {df}"),
            ));
        }
    }
    let m_f = count(m);
    let point = points.iter().sum::<f64>() / m_f;
    let within = variances.iter().sum::<f64>() / m_f;
    let between = points.iter().map(|q| (q - point).powi(2)).sum::<f64>() / (m_f - 1.0);
    let inflated = (1.0 + 1.0 / m_f) * between;
    let total = within + inflated;
    let relative_increase = if within > 0.0 { inflated / within } else { f64::INFINITY };
    let lambda = if total > 0.0 { inflated / total } else { 0.0 };
    let df_rubin = if lambda > 0.0 { (m_f - 1.0) / (lambda * lambda) } else { f64::INFINITY };
    let df = match df_complete {
        Some(df_com) => {
            let df_obs = (df_com + 1.0) / (df_com + 3.0) * df_com * (1.0 - lambda);
            if df_obs <= 0.0 {
                // All of the variance is between imputations: the observed-data information
                // is nil and the reference has no degrees of freedom to borrow.
                df_rubin.min(m_f - 1.0)
            } else if df_rubin.is_finite() {
                1.0 / (1.0 / df_rubin + 1.0 / df_obs)
            } else {
                df_obs
            }
        }
        None => df_rubin,
    };
    let fraction_missing_information = if relative_increase.is_finite() {
        let tail = if df.is_finite() { 2.0 / (df + 3.0) } else { 0.0 };
        (relative_increase + tail) / (relative_increase + 1.0)
    } else {
        1.0
    };
    Ok(RubinPooled {
        imputations: m,
        point,
        within,
        between,
        total,
        relative_increase,
        lambda,
        df_rubin,
        df_complete,
        df,
        fraction_missing_information,
    })
}

/// The posterior draws of `m` imputed datasets stacked into one equal-weight mixture.
///
/// Each completed dataset contributes the same number of draws, so the mixture weights are
/// equal. The between-imputation variance of the per-imputation posterior means is kept as
/// its own uncertainty axis.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct StackedPosterior {
    /// Number of imputations.
    pub imputations: usize,
    /// Draws per imputation.
    pub draws_per_imputation: usize,
    /// Stacked draws, imputation-major: imputation `j` occupies
    /// `draws[j * draws_per_imputation..(j + 1) * draws_per_imputation]`.
    pub draws: Vec<f64>,
    /// Mixture mean.
    pub mean: f64,
    /// Mixture variance (sample variance of the stacked draws).
    pub variance: f64,
    /// Mean of the per-imputation posterior variances.
    pub within: f64,
    /// Sample variance of the per-imputation posterior means.
    pub between: f64,
    /// Posterior mean of each imputation.
    pub imputation_means: Vec<f64>,
}

impl StackedPosterior {
    /// Share of the mixture variance carried by the between-imputation axis.
    #[must_use]
    pub fn between_share(&self) -> f64 {
        if self.variance > 0.0 { (self.between / self.variance).min(1.0) } else { 0.0 }
    }
}

/// Stack per-imputation posterior draws into one equal-weight mixture.
///
/// # Errors
///
/// Fewer than two imputations, imputations with different draw counts, fewer than two draws
/// per imputation, or a non-finite draw.
pub fn stack_posterior(
    per_imputation: &[&[f64]],
) -> Result<StackedPosterior, MultipleImputationError> {
    let m = per_imputation.len();
    if m < 2 {
        return Err(MultipleImputationError::invalid(
            "multiple_imputation.too_few_imputations",
            format!("a stacked posterior needs at least two imputations, got {m}"),
        ));
    }
    let k = per_imputation[0].len();
    if k < 2 || per_imputation.iter().any(|draws| draws.len() != k) {
        return Err(MultipleImputationError::invalid(
            "multiple_imputation.stack_shape",
            "every imputation must contribute the same number (at least two) of posterior draws, \
             so the mixture weights are equal",
        ));
    }
    if per_imputation.iter().any(|draws| draws.iter().any(|v| !v.is_finite())) {
        return Err(MultipleImputationError::invalid(
            "multiple_imputation.pool_non_finite",
            "a posterior draw is not finite",
        ));
    }
    let k_f = count(k);
    let m_f = count(m);
    let mut draws = Vec::with_capacity(m * k);
    let mut imputation_means = Vec::with_capacity(m);
    let mut within = 0.0;
    for column in per_imputation {
        let mean = column.iter().sum::<f64>() / k_f;
        let var = column.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (k_f - 1.0);
        within += var / m_f;
        imputation_means.push(mean);
        draws.extend_from_slice(column);
    }
    let total = count(draws.len());
    let mean = draws.iter().sum::<f64>() / total;
    let variance = draws.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (total - 1.0);
    let between = imputation_means.iter().map(|q| (q - mean).powi(2)).sum::<f64>() / (m_f - 1.0);
    Ok(StackedPosterior {
        imputations: m,
        draws_per_imputation: k,
        draws,
        mean,
        variance,
        within,
        between,
        imputation_means,
    })
}

#[allow(clippy::cast_precision_loss, reason = "imputation and draw counts are far below 2^52")]
fn count(n: usize) -> f64 {
    n as f64
}
