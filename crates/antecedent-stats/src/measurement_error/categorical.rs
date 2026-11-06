//! Categorical misclassification correction.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{MAX_MEASUREMENT_CATEGORIES, MeasurementErrorError, invert_square, matvec};

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
