//! Binary and multicategory measurement-error corrections.

use crate::gram::invert_square;

use super::{MAX_MEASUREMENT_CATEGORIES, MeasurementErrorError, matvec};

/// Sensitivity and specificity with the sampling variances of their estimates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BinaryErrorRates {
    /// `P(observed = 1 | true = 1)`.
    pub sensitivity: f64,
    /// `P(observed = 0 | true = 0)`.
    pub specificity: f64,
    /// Variance of the sensitivity estimate; zero when declared exact.
    pub sensitivity_variance: f64,
    /// Variance of the specificity estimate; zero when declared exact.
    pub specificity_variance: f64,
}

impl BinaryErrorRates {
    /// Validated rates.
    ///
    /// # Errors
    /// Rates outside `[0, 1]` or invalid variances refuse.
    pub fn new(
        sensitivity: f64,
        specificity: f64,
        sensitivity_variance: f64,
        specificity_variance: f64,
    ) -> Result<Self, MeasurementErrorError> {
        let unit = |v: f64| v.is_finite() && (0.0..=1.0).contains(&v);
        let var = |v: f64| v.is_finite() && v >= 0.0;
        if !(unit(sensitivity) && unit(specificity)) {
            return Err(MeasurementErrorError::InvalidParameter(
                "sensitivity and specificity must lie in [0, 1]",
            ));
        }
        if !(var(sensitivity_variance) && var(specificity_variance)) {
            return Err(MeasurementErrorError::InvalidParameter(
                "rate variances must be finite and non-negative",
            ));
        }
        Ok(Self { sensitivity, specificity, sensitivity_variance, specificity_variance })
    }

    /// Rates estimated from a validation sample against a reference standard.
    ///
    /// `true_positive + false_negative` units are truly positive and
    /// `true_negative + false_positive` truly negative. Variances are the
    /// binomial `p (1 - p) / n`.
    ///
    /// # Errors
    /// An empty reference class refuses.
    pub fn from_validation_counts(
        true_positive: u64,
        false_negative: u64,
        true_negative: u64,
        false_positive: u64,
    ) -> Result<Self, MeasurementErrorError> {
        let positives = true_positive + false_negative;
        let negatives = true_negative + false_positive;
        if positives == 0 || negatives == 0 {
            return Err(MeasurementErrorError::InvalidParameter(
                "validation sample needs both truly positive and truly negative units",
            ));
        }
        let se = true_positive as f64 / positives as f64;
        let sp = true_negative as f64 / negatives as f64;
        Self::new(se, sp, se * (1.0 - se) / positives as f64, sp * (1.0 - sp) / negatives as f64)
    }

    /// Youden's index `sensitivity + specificity - 1`.
    #[must_use]
    pub fn youden(&self) -> f64 {
        self.sensitivity + self.specificity - 1.0
    }

    fn require_youden(&self, minimum: f64) -> Result<f64, MeasurementErrorError> {
        if !(minimum.is_finite() && minimum > 0.0 && minimum < 1.0) {
            return Err(MeasurementErrorError::InvalidParameter(
                "minimum Youden index must lie in (0, 1)",
            ));
        }
        let j = self.youden();
        if j.is_nan() || j <= minimum {
            return Err(MeasurementErrorError::YoudenTooSmall { youden: j, minimum });
        }
        Ok(j)
    }
}

/// A corrected proportion with its variance components.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CorrectedProportion {
    /// Observed (misclassified) proportion.
    pub observed: f64,
    /// Rogan-Gladen corrected proportion, not truncated to `[0, 1]`.
    pub corrected: f64,
    /// Youden's index used.
    pub youden: f64,
    /// Total delta-method variance.
    pub variance: f64,
    /// Part due to sampling of the observed proportion.
    pub variance_sampling: f64,
    /// Part due to the sensitivity and specificity estimates.
    pub variance_rates: f64,
    /// The raw correction left `[0, 1]`: a sign of sampling noise or a wrong
    /// declared rate; the value is reported unclipped.
    pub outside_unit_interval: bool,
}

fn check_count(events: u64, total: u64) -> Result<(), MeasurementErrorError> {
    if total == 0 || events > total {
        return Err(MeasurementErrorError::InvalidParameter(
            "events must not exceed a positive total",
        ));
    }
    Ok(())
}

/// Corrected prevalence (Rogan and Gladen 1978) with delta-method variance
/// including the uncertainty of the sensitivity and specificity.
///
/// # Errors
/// Invalid counts, or Youden's index at or below `min_youden`, refuse.
pub fn corrected_prevalence(
    events: u64,
    total: u64,
    rates: &BinaryErrorRates,
    min_youden: f64,
) -> Result<CorrectedProportion, MeasurementErrorError> {
    check_count(events, total)?;
    let j = rates.require_youden(min_youden)?;
    let n = total as f64;
    let p = events as f64 / n;
    let pi = (p + rates.specificity - 1.0) / j;
    let var_p = p * (1.0 - p) / n;
    let sampling = var_p / (j * j);
    let from_rates = ((1.0 - pi) * (1.0 - pi) * rates.specificity_variance
        + pi * pi * rates.sensitivity_variance)
        / (j * j);
    Ok(CorrectedProportion {
        observed: p,
        corrected: pi,
        youden: j,
        variance: sampling + from_rates,
        variance_sampling: sampling,
        variance_rates: from_rates,
        outside_unit_interval: !(0.0..=1.0).contains(&pi),
    })
}

/// Outcome events by exposure arm.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TwoByTwoCounts {
    /// Observed events among the exposed.
    pub exposed_events: u64,
    /// Exposed units.
    pub exposed_total: u64,
    /// Observed events among the unexposed.
    pub unexposed_events: u64,
    /// Unexposed units.
    pub unexposed_total: u64,
}

/// A corrected risk difference with its variance components.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CorrectedRiskDifference {
    /// Observed risk difference.
    pub naive: f64,
    /// Corrected risk difference.
    pub corrected: f64,
    /// Corrected exposed-arm risk.
    pub risk_exposed: f64,
    /// Corrected unexposed-arm risk.
    pub risk_unexposed: f64,
    /// Total delta-method variance.
    pub variance: f64,
    /// Part due to sampling of the observed arm proportions.
    pub variance_sampling: f64,
    /// Part due to the error-rate estimates.
    pub variance_rates: f64,
    /// Youden's index of the exposed arm (the shared one when non-differential).
    pub youden_exposed: f64,
    /// Youden's index of the unexposed arm (equal to the exposed one when non-differential).
    pub youden_unexposed: f64,
    /// Either corrected arm risk left `[0, 1]`.
    pub outside_unit_interval: bool,
}

/// Risk difference under **non-differential** outcome misclassification: the
/// same sensitivity and specificity, from one validation sample, apply to both
/// exposure arms.
///
/// `RD = (p1* - p0*) / J`; the variance is
/// `[Var(p1*) + Var(p0*)] / J^2 + RD^2 (Var(Se) + Var(Sp)) / J^2`, which keeps
/// the covariance that the shared rates induce between the two arms.
///
/// # Errors
/// Invalid counts or Youden's index at or below `min_youden` refuse.
pub fn corrected_risk_difference_nondifferential(
    table: &TwoByTwoCounts,
    rates: &BinaryErrorRates,
    min_youden: f64,
) -> Result<CorrectedRiskDifference, MeasurementErrorError> {
    check_count(table.exposed_events, table.exposed_total)?;
    check_count(table.unexposed_events, table.unexposed_total)?;
    let j = rates.require_youden(min_youden)?;
    let n1 = table.exposed_total as f64;
    let n0 = table.unexposed_total as f64;
    let p1 = table.exposed_events as f64 / n1;
    let p0 = table.unexposed_events as f64 / n0;
    let naive = p1 - p0;
    let corrected = naive / j;
    let risk1 = (p1 + rates.specificity - 1.0) / j;
    let risk0 = (p0 + rates.specificity - 1.0) / j;
    let sampling = (p1 * (1.0 - p1) / n1 + p0 * (1.0 - p0) / n0) / (j * j);
    let from_rates =
        corrected * corrected * (rates.sensitivity_variance + rates.specificity_variance) / (j * j);
    Ok(CorrectedRiskDifference {
        naive,
        corrected,
        risk_exposed: risk1,
        risk_unexposed: risk0,
        variance: sampling + from_rates,
        variance_sampling: sampling,
        variance_rates: from_rates,
        youden_exposed: j,
        youden_unexposed: j,
        outside_unit_interval: !(0.0..=1.0).contains(&risk1) || !(0.0..=1.0).contains(&risk0),
    })
}

/// Risk difference under **differential** outcome misclassification: each
/// exposure arm has its own sensitivity and specificity, estimated from its
/// own independent validation sample.
///
/// # Errors
/// Invalid counts or either arm's Youden index at or below `min_youden` refuse.
pub fn corrected_risk_difference_differential(
    table: &TwoByTwoCounts,
    rates_exposed: &BinaryErrorRates,
    rates_unexposed: &BinaryErrorRates,
    min_youden: f64,
) -> Result<CorrectedRiskDifference, MeasurementErrorError> {
    check_count(table.exposed_events, table.exposed_total)?;
    check_count(table.unexposed_events, table.unexposed_total)?;
    let exposed =
        corrected_prevalence(table.exposed_events, table.exposed_total, rates_exposed, min_youden)?;
    let unexposed = corrected_prevalence(
        table.unexposed_events,
        table.unexposed_total,
        rates_unexposed,
        min_youden,
    )?;
    Ok(CorrectedRiskDifference {
        naive: exposed.observed - unexposed.observed,
        corrected: exposed.corrected - unexposed.corrected,
        risk_exposed: exposed.corrected,
        risk_unexposed: unexposed.corrected,
        variance: exposed.variance + unexposed.variance,
        variance_sampling: exposed.variance_sampling + unexposed.variance_sampling,
        variance_rates: exposed.variance_rates + unexposed.variance_rates,
        youden_exposed: exposed.youden,
        youden_unexposed: unexposed.youden,
        outside_unit_interval: exposed.outside_unit_interval || unexposed.outside_unit_interval,
    })
}

// ---------------------------------------------------------------------------
// Categorical misclassification
// ---------------------------------------------------------------------------

/// Column-stochastic confusion matrix `C[observed, true] = P(observed | true)`.
#[derive(Clone, Debug, PartialEq)]
pub struct ConfusionMatrix {
    levels: usize,
    matrix: Vec<f64>,
    column_counts: Option<Vec<u64>>,
}

impl ConfusionMatrix {
    /// A matrix from row-major `[observed][true]` probabilities.
    ///
    /// `column_counts[k]` is the number of validation units whose reference
    /// category is `k` (the sample behind column `k`); `None` declares the
    /// matrix exact.
    ///
    /// # Errors
    /// A wrong shape, entries outside `[0, 1]`, columns not summing to one
    /// (tolerance `1e-9`), or zero validation counts refuse.
    pub fn new(
        levels: usize,
        matrix: Vec<f64>,
        column_counts: Option<Vec<u64>>,
    ) -> Result<Self, MeasurementErrorError> {
        if !(2..=MAX_MEASUREMENT_CATEGORIES).contains(&levels) {
            return Err(MeasurementErrorError::Shape("category count outside the declared bound"));
        }
        if matrix.len() != levels * levels {
            return Err(MeasurementErrorError::Shape("confusion matrix is not square"));
        }
        if !matrix.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)) {
            return Err(MeasurementErrorError::InvalidParameter(
                "confusion entries must be probabilities",
            ));
        }
        for k in 0..levels {
            let sum: f64 = (0..levels).map(|j| matrix[j * levels + k]).sum();
            if (sum - 1.0).abs() > 1e-9 {
                return Err(MeasurementErrorError::InvalidParameter(
                    "each confusion column must sum to one",
                ));
            }
        }
        if let Some(counts) = &column_counts {
            if counts.len() != levels || counts.contains(&0) {
                return Err(MeasurementErrorError::InvalidParameter(
                    "validation counts need one positive count per true category",
                ));
            }
        }
        Ok(Self { levels, matrix, column_counts })
    }

    /// A matrix estimated from validation counts `[observed][true]`, row-major.
    ///
    /// # Errors
    /// An empty reference category refuses.
    pub fn from_validation_counts(
        levels: usize,
        counts: &[u64],
    ) -> Result<Self, MeasurementErrorError> {
        if !(2..=MAX_MEASUREMENT_CATEGORIES).contains(&levels) || counts.len() != levels * levels {
            return Err(MeasurementErrorError::Shape("validation counts are not square"));
        }
        let mut column_totals = vec![0_u64; levels];
        for j in 0..levels {
            for k in 0..levels {
                column_totals[k] += counts[j * levels + k];
            }
        }
        if column_totals.contains(&0) {
            return Err(MeasurementErrorError::InvalidParameter(
                "a true category has no validation units",
            ));
        }
        let matrix = (0..levels * levels)
            .map(|idx| counts[idx] as f64 / column_totals[idx % levels] as f64)
            .collect();
        Self::new(levels, matrix, Some(column_totals))
    }

    /// Number of categories.
    #[must_use]
    pub fn levels(&self) -> usize {
        self.levels
    }

    /// Row-major `[observed][true]` probabilities.
    #[must_use]
    pub fn matrix(&self) -> &[f64] {
        &self.matrix
    }

    /// Validation units per true category, when the matrix is estimated.
    #[must_use]
    pub fn column_counts(&self) -> Option<&[u64]> {
        self.column_counts.as_deref()
    }
}

/// Observed-category summary of the outcome.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ObservedClassSummary {
    /// Units observed in the category.
    pub count: u64,
    /// Mean outcome among them.
    pub mean: f64,
    /// Sample variance of the outcome among them.
    pub variance: f64,
}

/// Corrected category proportions and per-category outcome means.
#[derive(Clone, Debug, PartialEq)]
pub struct CorrectedCategorical {
    /// Observed category proportions.
    pub observed_proportions: Vec<f64>,
    /// Corrected true-category proportions (not projected onto the simplex).
    pub proportions: Vec<f64>,
    /// Covariance of the corrected proportions, row-major, including the
    /// uncertainty of the confusion columns.
    pub proportion_covariance: Vec<f64>,
    /// Corrected expected counts `n * proportions`.
    pub expected_counts: Vec<f64>,
    /// Mean outcome among units in each true category.
    pub outcome_means: Vec<f64>,
    /// Covariance of the outcome means, row-major.
    pub outcome_mean_covariance: Vec<f64>,
    /// 1-norm condition number of the confusion matrix.
    pub condition_number: f64,
    /// A corrected proportion left `[0, 1]`.
    pub outside_unit_interval: bool,
}

fn one_norm(a: &[f64], d: usize) -> f64 {
    (0..d).map(|j| (0..d).map(|i| a[i * d + j].abs()).sum::<f64>()).fold(0.0, f64::max)
}

/// Outcome means by true category from the model map. `theta` packs the
/// observed-class means (K), observed proportions (K) and confusion entries
/// (K x K, row-major).
fn category_outcome_means(theta: &[f64], k: usize) -> Option<Vec<f64>> {
    let means = &theta[..k];
    let p_star = &theta[k..2 * k];
    let c = &theta[2 * k..];
    let c_inv = invert_square(c, k)?;
    let pi = matvec(&c_inv, p_star, k);
    // Q[j][t] = C[j][t] pi_t / p*_j
    let mut q = vec![0.0; k * k];
    for j in 0..k {
        if p_star[j].is_nan() || p_star[j] <= 0.0 {
            return None;
        }
        for t in 0..k {
            q[j * k + t] = c[j * k + t] * pi[t] / p_star[j];
        }
    }
    let q_inv = invert_square(&q, k)?;
    Some(matvec(&q_inv, means, k))
}

/// Correct category proportions and outcome means for declared, known
/// misclassification.
///
/// The observed-class means obey `ybar = Q mu` with
/// `Q[j][k] = P(true = k | observed = j) = C[j][k] pi_k / p*_j`, so
/// `mu = Q^{-1} ybar`. The covariance is delta-method over the observed
/// proportions (multinomial), the observed-class means (independent) and the
/// confusion columns (independent multinomials from the validation counts);
/// the outcome-mean Jacobian is a central finite difference of the exact map.
///
/// # Errors
/// An empty observed class, a singular or ill-conditioned confusion matrix
/// (1-norm condition above `max_condition`) or an empty latent category refuse.
#[allow(
    clippy::too_many_lines,
    reason = "a flat sequence of independent validity checks followed by the delta-method assembly of one corrected categorical result"
)]
pub fn corrected_categories(
    classes: &[ObservedClassSummary],
    confusion: &ConfusionMatrix,
    max_condition: f64,
) -> Result<CorrectedCategorical, MeasurementErrorError> {
    let k = confusion.levels;
    if classes.len() != k {
        return Err(MeasurementErrorError::Shape("observed classes differ from confusion levels"));
    }
    if !(max_condition.is_finite() && max_condition >= 1.0) {
        return Err(MeasurementErrorError::InvalidParameter(
            "maximum condition number must be at least one",
        ));
    }
    if classes.iter().any(|c| c.count < 2) {
        return Err(MeasurementErrorError::TooFewRows {
            rows: classes
                .iter()
                .map(|c| usize::try_from(c.count).unwrap_or(usize::MAX))
                .min()
                .unwrap_or(0),
            minimum: 2,
        });
    }
    if classes.iter().any(|c| !(c.mean.is_finite() && c.variance.is_finite() && c.variance >= 0.0))
    {
        return Err(MeasurementErrorError::NonFinite("observed class summaries"));
    }
    let total: u64 = classes
        .iter()
        .try_fold(0_u64, |sum, c| sum.checked_add(c.count))
        .ok_or(MeasurementErrorError::InvalidParameter("observed class counts overflow"))?;
    let n = total as f64;
    let p_star: Vec<f64> = classes.iter().map(|c| c.count as f64 / n).collect();
    let c = &confusion.matrix;
    let c_inv = invert_square(c, k).ok_or(MeasurementErrorError::ConfusionIllConditioned {
        condition: f64::INFINITY,
        maximum: max_condition,
    })?;
    let condition = one_norm(c, k) * one_norm(&c_inv, k);
    if !(condition.is_finite() && condition <= max_condition) {
        return Err(MeasurementErrorError::ConfusionIllConditioned {
            condition,
            maximum: max_condition,
        });
    }
    let pi = matvec(&c_inv, &p_star, k);

    // Covariance of theta = (ybar, p*, C).
    let dim = 2 * k + k * k;
    let mut cov = vec![0.0; dim * dim];
    for j in 0..k {
        cov[j * dim + j] = classes[j].variance / classes[j].count as f64;
    }
    for a in 0..k {
        for b in 0..k {
            let value = if a == b { p_star[a] * (1.0 - p_star[a]) } else { -p_star[a] * p_star[b] };
            cov[(k + a) * dim + (k + b)] = value / n;
        }
    }
    if let Some(counts) = &confusion.column_counts {
        for t in 0..k {
            let nt = counts[t] as f64;
            for a in 0..k {
                for b in 0..k {
                    let ca = c[a * k + t];
                    let cb = c[b * k + t];
                    let value = if a == b { ca * (1.0 - ca) } else { -ca * cb };
                    cov[(2 * k + a * k + t) * dim + (2 * k + b * k + t)] = value / nt;
                }
            }
        }
    }

    // Proportions: pi = C^{-1} p*; d pi = C^{-1} d p* - C^{-1} (d C) pi.
    let mut jac_pi = vec![0.0; k * dim];
    for i in 0..k {
        for j in 0..k {
            jac_pi[i * dim + k + j] = c_inv[i * k + j];
            for t in 0..k {
                // d pi_i / d C[j][t] = - Cinv[i][j] pi_t
                jac_pi[i * dim + 2 * k + j * k + t] = -c_inv[i * k + j] * pi[t];
            }
        }
    }
    let proportion_covariance = congruence(&jac_pi, &cov, k, dim);

    // Outcome means via central differences of the exact map.
    let mut theta = Vec::with_capacity(dim);
    theta.extend(classes.iter().map(|c| c.mean));
    theta.extend_from_slice(&p_star);
    theta.extend_from_slice(c);
    let mu = category_outcome_means(&theta, k).ok_or(MeasurementErrorError::EmptyLatentCategory)?;
    let mut jac_mu = vec![0.0; k * dim];
    for idx in 0..dim {
        if (0..dim).all(|other| cov[idx * dim + other] == 0.0 && cov[other * dim + idx] == 0.0) {
            continue;
        }
        let h = 1e-6 * theta[idx].abs().max(1.0);
        let mut up = theta.clone();
        up[idx] += h;
        let mut down = theta.clone();
        down[idx] -= h;
        let f_up =
            category_outcome_means(&up, k).ok_or(MeasurementErrorError::EmptyLatentCategory)?;
        let f_down =
            category_outcome_means(&down, k).ok_or(MeasurementErrorError::EmptyLatentCategory)?;
        for i in 0..k {
            jac_mu[i * dim + idx] = (f_up[i] - f_down[i]) / (2.0 * h);
        }
    }
    let outcome_mean_covariance = congruence(&jac_mu, &cov, k, dim);

    if (0..k).any(|i| {
        !(proportion_covariance[i * k + i] >= 0.0 && outcome_mean_covariance[i * k + i] >= 0.0)
    }) {
        return Err(MeasurementErrorError::NonPositiveVariance("corrected category summary"));
    }
    let expected_counts = pi.iter().map(|v| v * n).collect();
    let outside = pi.iter().any(|v| !(0.0..=1.0).contains(v));
    Ok(CorrectedCategorical {
        observed_proportions: p_star,
        proportions: pi,
        proportion_covariance,
        expected_counts,
        outcome_means: mu,
        outcome_mean_covariance,
        condition_number: condition,
        outside_unit_interval: outside,
    })
}

/// `J Sigma J'` for `J` of shape `rows x cols` and `Sigma` `cols x cols`.
fn congruence(jac: &[f64], sigma: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut tmp = vec![0.0; rows * cols];
    for i in 0..rows {
        for c in 0..cols {
            let mut acc = 0.0;
            for m in 0..cols {
                acc += jac[i * cols + m] * sigma[m * cols + c];
            }
            tmp[i * cols + c] = acc;
        }
    }
    let mut out = vec![0.0; rows * rows];
    for i in 0..rows {
        for j in 0..rows {
            let mut acc = 0.0;
            for c in 0..cols {
                acc += tmp[i * cols + c] * jac[j * cols + c];
            }
            out[i * rows + j] = acc;
        }
    }
    out
}
