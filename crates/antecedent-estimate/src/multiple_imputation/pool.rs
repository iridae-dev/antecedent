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

/// Rubin's-rules combination of `m` vector-valued complete-data estimates and their
/// covariance matrices.
///
/// This is the pooling for an estimand with several coordinates that are reported together
/// (the cells of a conditional effect, the points of a response curve, the components of a
/// mediation decomposition, the levels of a distribution): `Q̄ = mean(q)`, within covariance
/// `Ū = mean(U)`, between covariance `B = Σ(q − Q̄)(q − Q̄)ᵀ / (m − 1)` and total covariance
/// `T = Ū + (1 + 1/m) B`. Each coordinate is also pooled as a scalar
/// ([`Self::coordinates`]), so its degrees of freedom, fraction of missing information and
/// interval are those of [`rubin_pool`]. A joint hypothesis uses [`Self::wald`].
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct RubinVectorPooled {
    /// Number of imputations `m`.
    pub imputations: usize,
    /// Number of coordinates `p`.
    pub dim: usize,
    /// Pooled point `Q̄`, length `p`.
    pub point: Vec<f64>,
    /// Within-imputation covariance `Ū`, row-major `p × p`.
    pub within: Vec<f64>,
    /// Between-imputation covariance `B`, row-major `p × p`.
    pub between: Vec<f64>,
    /// Total covariance `T = Ū + (1 + 1/m) B`, row-major `p × p`.
    pub total: Vec<f64>,
    /// Each coordinate pooled as a scalar, in coordinate order.
    pub coordinates: Vec<RubinPooled>,
    /// Average relative increase in variance `r̄ = (1 + 1/m) tr(B Ū⁻¹) / p`; infinite when
    /// `Ū` is singular.
    pub average_relative_increase: f64,
}

/// A pooled multivariate Wald statistic (Li, Raghunathan and Rubin 1991, the `D₁` statistic).
///
/// `statistic` is referred to an `F` law with `df_numerator` and `df_denominator` degrees of
/// freedom; an infinite denominator is the chi-square reference over `df_numerator`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct PooledWald {
    /// `D₁ = (Q̄ − θ₀)ᵀ Ū⁻¹ (Q̄ − θ₀) / (p (1 + r̄))`.
    pub statistic: f64,
    /// Numerator degrees of freedom `p`.
    pub df_numerator: f64,
    /// Denominator degrees of freedom `ν₂`.
    pub df_denominator: f64,
}

fn symmetric_enough(matrix: &[f64], dim: usize) -> bool {
    (0..dim).all(|i| {
        (0..i).all(|j| {
            let (upper, lower) = (matrix[j * dim + i], matrix[i * dim + j]);
            (upper - lower).abs() <= 1e-8 * (1.0 + upper.abs().max(lower.abs()))
        })
    })
}

fn multiply_square(a: &[f64], b: &[f64], dim: usize) -> Vec<f64> {
    let mut out = vec![0.0; dim * dim];
    for i in 0..dim {
        for k in 0..dim {
            let left = a[i * dim + k];
            for j in 0..dim {
                out[i * dim + j] += left * b[k * dim + j];
            }
        }
    }
    out
}

/// Combine vector estimates `points` and their covariance matrices by Rubin's rules.
///
/// # Errors
///
/// Fewer than two imputations, points or covariances of differing shape, a non-finite
/// entry, a covariance that is not symmetric with a nonnegative diagonal, or a complete-data
/// degrees of freedom that is not positive.
pub fn rubin_pool_vector(
    points: &[Vec<f64>],
    covariances: &[Vec<f64>],
    df_complete: Option<f64>,
) -> Result<RubinVectorPooled, MultipleImputationError> {
    let m = points.len();
    if m < 2 {
        return Err(MultipleImputationError::invalid(
            "multiple_imputation.too_few_imputations",
            format!("Rubin's rules need at least two imputations, got {m}"),
        ));
    }
    let dim = points[0].len();
    if dim == 0 || covariances.len() != m || points.iter().any(|q| q.len() != dim) {
        return Err(MultipleImputationError::invalid(
            "multiple_imputation.pool_shape",
            format!(
                "{m} points of {dim} coordinates need {m} covariance matrices of {dim} by {dim}"
            ),
        ));
    }
    if covariances.iter().any(|u| u.len() != dim * dim) {
        return Err(MultipleImputationError::invalid(
            "multiple_imputation.pool_shape",
            format!("every covariance must hold {dim} by {dim} entries"),
        ));
    }
    if points.iter().flatten().any(|q| !q.is_finite())
        || covariances.iter().flatten().any(|u| !u.is_finite())
    {
        return Err(MultipleImputationError::invalid(
            "multiple_imputation.pool_non_finite",
            "an imputation's point or covariance entry is not finite",
        ));
    }
    for covariance in covariances {
        if !symmetric_enough(covariance, dim) || (0..dim).any(|i| covariance[i * dim + i] < 0.0) {
            return Err(MultipleImputationError::invalid(
                "multiple_imputation.pool_non_finite",
                "an imputation's covariance is not symmetric with a nonnegative diagonal",
            ));
        }
    }
    let m_f = count(m);
    let point: Vec<f64> =
        (0..dim).map(|i| points.iter().map(|q| q[i]).sum::<f64>() / m_f).collect();
    let mut within = vec![0.0; dim * dim];
    for covariance in covariances {
        for (slot, value) in within.iter_mut().zip(covariance) {
            *slot += value / m_f;
        }
    }
    let mut between = vec![0.0; dim * dim];
    for q in points {
        for i in 0..dim {
            for j in 0..dim {
                between[i * dim + j] += (q[i] - point[i]) * (q[j] - point[j]) / (m_f - 1.0);
            }
        }
    }
    let inflate = 1.0 + 1.0 / m_f;
    let total: Vec<f64> = within.iter().zip(&between).map(|(u, b)| u + inflate * b).collect();
    let coordinates = (0..dim)
        .map(|i| {
            let column: Vec<f64> = points.iter().map(|q| q[i]).collect();
            let variances: Vec<f64> = covariances.iter().map(|u| u[i * dim + i]).collect();
            rubin_pool(&column, &variances, df_complete)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let average_relative_increase =
        antecedent_stats::invert_square(&within, dim).map_or(f64::INFINITY, |inverse| {
            let product = multiply_square(&between, &inverse, dim);
            let trace: f64 = (0..dim).map(|i| product[i * dim + i]).sum();
            inflate * trace / count(dim)
        });
    Ok(RubinVectorPooled {
        imputations: m,
        dim,
        point,
        within,
        between,
        total,
        coordinates,
        average_relative_increase,
    })
}

impl RubinVectorPooled {
    /// Pooled standard error of coordinate `index`, `√T[index, index]`.
    #[must_use]
    pub fn standard_error(&self, index: usize) -> Option<f64> {
        (index < self.dim).then(|| self.total[index * self.dim + index].sqrt())
    }

    /// The pooled `D₁` Wald statistic of the hypothesis `θ = null`.
    ///
    /// # Errors
    ///
    /// `null` of the wrong length, or a within-imputation covariance that is singular (the
    /// statistic needs its inverse).
    pub fn wald(&self, null: &[f64]) -> Result<PooledWald, MultipleImputationError> {
        if null.len() != self.dim {
            return Err(MultipleImputationError::invalid(
                "multiple_imputation.pool_shape",
                format!("the hypothesis has {} coordinates, the estimate {}", null.len(), self.dim),
            ));
        }
        let inverse = antecedent_stats::invert_square(&self.within, self.dim).ok_or_else(|| {
            MultipleImputationError::invalid(
                "multiple_imputation.within_singular",
                "the within-imputation covariance is singular, so the joint Wald statistic is \
                 not defined; test the coordinates one at a time",
            )
        })?;
        let gap: Vec<f64> = self.point.iter().zip(null).map(|(q, n)| q - n).collect();
        let mut quadratic = 0.0;
        for i in 0..self.dim {
            for j in 0..self.dim {
                quadratic += gap[i] * inverse[i * self.dim + j] * gap[j];
            }
        }
        let k = count(self.dim);
        let r = self.average_relative_increase;
        let t = k * (count(self.imputations) - 1.0);
        let df_denominator = if r == 0.0 {
            f64::INFINITY
        } else if t > 4.0 {
            4.0 + (t - 4.0) * (1.0 + (1.0 - 2.0 / t) / r).powi(2)
        } else {
            t * (1.0 + 1.0 / k) * (1.0 + 1.0 / r).powi(2) / 2.0
        };
        Ok(PooledWald { statistic: quadratic / (k * (1.0 + r)), df_numerator: k, df_denominator })
    }
}

/// The posterior draws of several quantities, stacked over `m` imputations.
///
/// Each quantity is stacked as in [`stack_posterior`]; `covariance` is the sample covariance
/// of the stacked mixture across quantities, which keeps the within- and between-imputation
/// dependence between them (draw `d` of every quantity in one imputation is one joint draw).
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct StackedQuantities {
    /// One stacked posterior per quantity, in quantity order.
    pub quantities: Vec<StackedPosterior>,
    /// Row-major `p × p` covariance of the stacked mixture.
    pub covariance: Vec<f64>,
}

/// Stack the joint posterior draws of several quantities over imputations.
///
/// `per_imputation[j][q]` is the draw vector of quantity `q` under imputation `j`.
///
/// # Errors
///
/// The refusals of [`stack_posterior`], or imputations that report different quantity counts.
pub fn stack_posterior_quantities(
    per_imputation: &[Vec<Vec<f64>>],
) -> Result<StackedQuantities, MultipleImputationError> {
    let Some(first) = per_imputation.first() else {
        return Err(MultipleImputationError::invalid(
            "multiple_imputation.too_few_imputations",
            "a stacked posterior needs at least two imputations, got 0",
        ));
    };
    let dim = first.len();
    if dim == 0 || per_imputation.iter().any(|quantities| quantities.len() != dim) {
        return Err(MultipleImputationError::invalid(
            "multiple_imputation.stack_shape",
            "every imputation must report the same, nonzero number of quantities",
        ));
    }
    let mut quantities = Vec::with_capacity(dim);
    for q in 0..dim {
        let columns: Vec<&[f64]> = per_imputation.iter().map(|j| j[q].as_slice()).collect();
        quantities.push(stack_posterior(&columns)?);
    }
    let total = quantities[0].draws.len();
    let mut covariance = vec![0.0; dim * dim];
    for i in 0..dim {
        for j in i..dim {
            let value = quantities[i]
                .draws
                .iter()
                .zip(&quantities[j].draws)
                .map(|(a, b)| (a - quantities[i].mean) * (b - quantities[j].mean))
                .sum::<f64>()
                / (count(total) - 1.0);
            covariance[i * dim + j] = value;
            covariance[j * dim + i] = value;
        }
    }
    Ok(StackedQuantities { quantities, covariance })
}

#[allow(clippy::cast_precision_loss, reason = "imputation and draw counts are far below 2^52")]
fn count(n: usize) -> f64 {
    n as f64
}
