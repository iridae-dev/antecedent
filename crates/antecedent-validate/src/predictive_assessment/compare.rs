//! Score differences between named predictors.
//!
//! Two predictors assessed on the same realized observations are *paired*: the
//! difference of their case scores `d_i = S_A,i - S_B,i` is the object of
//! interest, and its variance is formed from the differences, never from the sum
//! of the two predictors' variances (their errors share the same observations
//! and are generally correlated).
//!
//! The standard error of the mean difference uses the declared dependence:
//!
//! * `iid`: `s^2 / n`;
//! * `serial`: the Bartlett-kernel long-run variance
//!   `(gamma_0 + 2 sum_{k=1}^{L} (1 - k / (L + 1)) gamma_k) / n` with the declared
//!   truncation lag `L` (Newey and West 1987; the Diebold-Mariano (1995) statistic
//!   with a positive-definite kernel);
//! * `cluster`: the cluster-robust variance
//!   `G / (G - 1) * sum_g (S_g - n_g mean)^2 / n^2` over `G >= 2` clusters.
//!
//! With no declared dependence the standard error is withheld. The Diebold-Mariano
//! statistic `mean / se` is a labelled diagnostic: no p-value, interval or
//! coverage statement is attached, because none has measured calibration.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::reason_code;

use super::design::DependenceDeclaration;
use super::refusal::{PredictiveAssessmentError, detail, refuse};
use super::score::ScoreKind;

/// The difference of one score between a candidate and a reference predictor.
#[derive(Clone, Debug, PartialEq)]
pub struct ScoreDifference {
    /// The score compared.
    pub score: ScoreKind,
    /// The candidate predictor's name.
    pub candidate: String,
    /// The reference predictor's name.
    pub reference: String,
    /// Observations both predictors were scored on.
    pub shared_observations: usize,
    /// Mean of `score(candidate) - score(reference)`; negative favours the candidate.
    pub mean_difference: f64,
    /// Standard error of the mean difference under the declared dependence.
    pub standard_error: Option<f64>,
    /// The Diebold-Mariano style statistic `mean / standard_error`, a diagnostic.
    pub dm_statistic: Option<f64>,
    /// The dependence the standard error is under: `iid`, `serial`, `cluster` or
    /// `undeclared` (then no standard error is reported).
    pub dependence: &'static str,
    /// Always `paired_on_shared_observations`: the variance is that of the
    /// per-case differences, with the predictors' dependence from the shared
    /// observations left in, not assumed away.
    pub pairing: &'static str,
    /// Observations both predictors were fitted on, when both declare their fit.
    pub fit_overlap: Option<u64>,
    /// Always `diagnostic`.
    pub label: &'static str,
    /// Always `unmeasured`.
    pub calibration: &'static str,
    /// Always `not_reported`: a score-difference interval is closed.
    pub interval: &'static str,
}

/// Mean of `values`.
fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

/// Variance of the mean of `d` under the declared dependence, with `d` in
/// canonical case order (`cluster` is in the same order).
///
/// # Errors
/// Fewer than two cases, fewer than two clusters, or a truncation lag that is
/// not shorter than the number of cases.
pub fn variance_of_mean(
    d: &[f64],
    dependence: &DependenceDeclaration,
    cluster: Option<&[u32]>,
) -> Result<f64, PredictiveAssessmentError> {
    let n = d.len();
    if n < 2 {
        return Err(refuse(
            reason_code!("route_not_supported"),
            detail::BOUNDS_EXCEEDED,
            "a standard error of a mean difference needs at least two cases",
        ));
    }
    let nf = n as f64;
    let center = mean(d);
    match dependence {
        DependenceDeclaration::Iid => {
            let s2 = d.iter().map(|x| (x - center).powi(2)).sum::<f64>() / (nf - 1.0);
            Ok(s2 / nf)
        }
        DependenceDeclaration::Serial { truncation_lag, .. } => {
            let lag = usize::try_from(*truncation_lag).unwrap_or(usize::MAX);
            if lag >= n {
                return Err(refuse(
                    reason_code!("sampling_dependence_unknown"),
                    detail::SERIAL_LAG_BELOW_HORIZON,
                    format!(
                        "the declared truncation lag {truncation_lag} is not shorter than the \
                         {n} cases; declare a shorter lag or assess more cases"
                    ),
                ));
            }
            let centered: Vec<f64> = d.iter().map(|x| x - center).collect();
            let gamma = |k: usize| -> f64 {
                centered[k..].iter().zip(&centered).map(|(a, b)| a * b).sum::<f64>() / nf
            };
            let mut long_run = gamma(0);
            for k in 1..=lag {
                let weight = 1.0 - k as f64 / (lag as f64 + 1.0);
                long_run += 2.0 * weight * gamma(k);
            }
            // The Bartlett kernel is positive semidefinite, so this is nonnegative up to rounding.
            Ok(long_run.max(0.0) / nf)
        }
        DependenceDeclaration::Cluster { .. } => {
            let labels = cluster.unwrap_or(&[]);
            // One pass: clusters in order of first appearance, so each cluster's sum
            // accumulates in case order exactly as a scan would.
            let mut position: std::collections::HashMap<u32, usize> =
                std::collections::HashMap::new();
            let mut sums: Vec<(u32, f64, f64)> = Vec::new();
            for (value, label) in d.iter().zip(labels) {
                if let Some(&at) = position.get(label) {
                    sums[at].1 += value - center;
                    sums[at].2 += 1.0;
                } else {
                    position.insert(*label, sums.len());
                    sums.push((*label, value - center, 1.0));
                }
            }
            let groups = sums.len();
            if groups < 2 {
                return Err(refuse(
                    reason_code!("too_few_clusters"),
                    detail::TOO_FEW_CLUSTERS,
                    format!(
                        "{groups} cluster(s) declared; a cluster-robust variance needs at least \
                         two. Declare finer clusters or assess more clusters"
                    ),
                ));
            }
            let g = groups as f64;
            let sum_squares: f64 = sums.iter().map(|entry| entry.1 * entry.1).sum();
            Ok(g / (g - 1.0) * sum_squares / (nf * nf))
        }
    }
}

/// One difference from two predictors' per-case scores in canonical order.
pub(crate) fn difference(
    score: ScoreKind,
    candidate: (&str, &[f64]),
    reference: (&str, &[f64]),
    dependence: Option<&DependenceDeclaration>,
    cluster: Option<&[u32]>,
    fit_overlap: Option<u64>,
) -> Result<ScoreDifference, PredictiveAssessmentError> {
    let d: Vec<f64> = candidate.1.iter().zip(reference.1).map(|(a, b)| a - b).collect();
    let mean_difference = mean(&d);
    let (standard_error, dm_statistic) = match dependence {
        None => (None, None),
        Some(declared) => {
            let variance = variance_of_mean(&d, declared, cluster)?;
            let se = variance.sqrt();
            if se > 0.0 && se.is_finite() {
                (Some(se), Some(mean_difference / se))
            } else {
                (Some(se), None)
            }
        }
    };
    Ok(ScoreDifference {
        score,
        candidate: candidate.0.to_owned(),
        reference: reference.0.to_owned(),
        shared_observations: d.len(),
        mean_difference,
        standard_error,
        dm_statistic,
        dependence: super::design::dependence_label(dependence),
        pairing: "paired_on_shared_observations",
        fit_overlap,
        label: super::calibration::DIAGNOSTIC_LABEL,
        calibration: super::calibration::DIAGNOSTIC_CALIBRATION,
        interval: "not_reported",
    })
}
