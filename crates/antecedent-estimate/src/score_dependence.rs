//! Dependence-aware variance of a cross-fitted score table.
//!
//! A score table publishes the means of per-row influence-function columns. With independent
//! rows the variance of a mean is the sample variance of its influence over `n`. When the rows
//! are declared to be a clustered panel or a dependent series, that variance is too small, and
//! the table is held to the declaration with one of two sandwich estimators on the same
//! influence values:
//!
//! - **Clusters.** With `psi_i = (w_i / mean w)(phi_i - theta)` the weighted influence of row
//!   `i`, the variance of the mean is `G/(G-1) * sum_g (sum_{i in g} psi_i)^2 / n^2` over the `G`
//!   declared clusters, the Arellano small-sample factor the average-effect influence error
//!   uses. With one row per cluster it equals the independent-row variance.
//! - **Newey-West.** The same `psi` taken in table order as unit-spaced time, with the Bartlett
//!   kernel of the declared lag: `(gamma_0 + 2 sum_{l=1}^{L} (1 - l/(L+1)) gamma_l) / n^2`.
//!
//! Both are quadratic forms in `psi`, so the covariance between two columns is the polarized
//! form `(V(psi_i + psi_j) - V(psi_i) - V(psi_j)) / 2` of the same meat, and a linear contrast
//! of columns keeps the variance of its combined influence.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_stats::{cluster_meat_scalar, newey_west_meat_scalar};

use crate::error::EstimationError;
use crate::joint_if::JointCovariance;

/// The dependence a score table's variance must read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ScoreDependence {
    /// One cluster label per row of the table the scores were fitted on (source-row order).
    Clusters(Arc<[u32]>),
    /// Rows are a time-ordered series; the Bartlett kernel reaches `lag` rows.
    NeweyWest {
        /// Kernel bandwidth in rows.
        lag: usize,
    },
}

impl ScoreDependence {
    /// Clusters from per-source-row labels.
    #[must_use]
    pub fn clusters(labels: impl Into<Arc<[u32]>>) -> Self {
        Self::Clusters(labels.into())
    }

    /// A Newey-West kernel of `lag` rows.
    #[must_use]
    pub const fn newey_west(lag: usize) -> Self {
        Self::NeweyWest { lag }
    }

    /// The same dependence restricted to the retained rows `rows` (indices into the source
    /// rows). A Newey-West lag is unchanged.
    ///
    /// # Errors
    ///
    /// A row index beyond the declared labels.
    pub fn restricted(&self, rows: &[usize]) -> Result<Self, EstimationError> {
        match self {
            Self::NeweyWest { lag } => Ok(Self::NeweyWest { lag: *lag }),
            Self::Clusters(labels) => {
                let mut kept = Vec::with_capacity(rows.len());
                for &row in rows {
                    kept.push(*labels.get(row).ok_or_else(|| {
                        EstimationError::data_msg(
                            "a cluster label was declared for fewer rows than the table has",
                        )
                    })?);
                }
                Ok(Self::Clusters(kept.into()))
            }
        }
    }

    /// Fingerprint of the declaration, for identities that carry it.
    #[must_use]
    pub fn digest(&self) -> u64 {
        let mut hash = antecedent_core::Fnv1a64::new();
        match self {
            Self::Clusters(labels) => {
                hash.word(1);
                hash.word(labels.len() as u64);
                for &label in labels.iter() {
                    hash.word(u64::from(label));
                }
            }
            Self::NeweyWest { lag } => {
                hash.word(2);
                hash.word(*lag as u64);
            }
        }
        hash.finish()
    }

    /// Variance of the mean whose per-row weighted influence is `psi` (already centred).
    fn variance_of_mean(&self, psi: &[f64]) -> Result<f64, EstimationError> {
        let n = psi.len();
        let nf = n as f64;
        match self {
            Self::Clusters(labels) => {
                if labels.len() != n {
                    return Err(EstimationError::data_msg(
                        "cluster labels must cover exactly the rows of the score table",
                    ));
                }
                let (meat, groups) = cluster_meat_scalar(psi, labels, 0.0).map_err(|_| {
                    EstimationError::data_msg("cluster influence variance failed to form meat")
                })?;
                if groups < 2 {
                    return Err(crate::se::few_clusters(groups));
                }
                let g = groups as f64;
                Ok((g / (g - 1.0)) * meat / (nf * nf))
            }
            Self::NeweyWest { lag } => Ok(newey_west_meat_scalar(psi, *lag) / (nf * nf)),
        }
    }
}

/// Joint covariance of the weighted means of `scores` under `dependence`.
///
/// `weights` are the sampling weights of the rows (`None`: equal), `means` their weighted means.
///
/// # Errors
///
/// Shape mismatch, fewer than two rows, fewer than two clusters, or a non-finite result.
pub(crate) fn dependent_influence_covariance(
    scores: &[&[f64]],
    weights: Option<&[f64]>,
    means: &[f64],
    dependence: &ScoreDependence,
) -> Result<JointCovariance, EstimationError> {
    let dim = scores.len();
    if dim == 0 || means.len() != dim {
        return Err(EstimationError::data_msg("joint IF requires aligned means"));
    }
    let n = scores[0].len();
    if n < 2 || scores.iter().any(|col| col.len() != n) {
        return Err(EstimationError::data_msg(
            "score columns must share a row count of at least two",
        ));
    }
    if weights.is_some_and(|w| w.len() != n) {
        return Err(EstimationError::data_msg("weight length does not match scores"));
    }
    // Weights enter only through their ratio to the mean weight, so scale cannot matter.
    let mean_weight = match weights {
        Some(w) => {
            let total: f64 = w.iter().sum();
            if !(total.is_finite() && total > 0.0) {
                return Err(EstimationError::data_msg("sampling weights have no positive mass"));
            }
            total / n as f64
        }
        None => 1.0,
    };
    let psi: Vec<Vec<f64>> = scores
        .iter()
        .zip(means)
        .map(|(col, &mean)| {
            col.iter()
                .enumerate()
                .map(|(row, &value)| {
                    let relative = weights.map_or(1.0, |w| w[row] / mean_weight);
                    relative * (value - mean)
                })
                .collect()
        })
        .collect();
    let mut values = vec![0.0; dim * dim];
    let variances: Vec<f64> =
        psi.iter().map(|p| dependence.variance_of_mean(p)).collect::<Result<_, _>>()?;
    for j in 0..dim {
        values[j * dim + j] = variances[j];
        for i in 0..j {
            let sum: Vec<f64> = psi[i].iter().zip(&psi[j]).map(|(a, b)| a + b).collect();
            let cov = 0.5 * (dependence.variance_of_mean(&sum)? - variances[i] - variances[j]);
            values[j * dim + i] = cov;
            values[i * dim + j] = cov;
        }
    }
    if values.iter().any(|v| !v.is_finite()) {
        return Err(EstimationError::data_msg("dependent joint covariance overflowed"));
    }
    Ok(JointCovariance { dim, values: Arc::from(values) })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The covariance between two columns is the polarized meat, so the variance of a linear
    /// contrast read from the joint matrix equals the variance of the contrast's own influence.
    #[test]
    fn a_contrast_variance_from_the_joint_matrix_is_the_variance_of_the_combined_influence() {
        let a: Vec<f64> = (0..23).map(|i| ((i * 7 % 11) as f64) - 4.5).collect();
        let b: Vec<f64> = (0..23).map(|i| ((i * 5 % 13) as f64) * 0.5 - 1.0).collect();
        let w: Vec<f64> = (0..23).map(|i| 1.0 + (i % 4) as f64).collect();
        let labels: Arc<[u32]> = (0..23u32).map(|i| i / 3).collect();
        for dependence in [ScoreDependence::Clusters(labels), ScoreDependence::NeweyWest { lag: 2 }]
        {
            let mean = |col: &[f64]| {
                col.iter().zip(&w).map(|(x, w)| x * w).sum::<f64>() / w.iter().sum::<f64>()
            };
            let means = [mean(&a), mean(&b)];
            let cov =
                dependent_influence_covariance(&[&a, &b], Some(&w), &means, &dependence).unwrap();
            let combined: Vec<f64> = a.iter().zip(&b).map(|(x, y)| 2.0 * x - 3.0 * y).collect();
            let combined_mean = 2.0 * means[0] - 3.0 * means[1];
            let direct = dependent_influence_covariance(
                &[&combined],
                Some(&w),
                &[combined_mean],
                &dependence,
            )
            .unwrap();
            let joint = 4.0 * cov.get(0, 0) - 12.0 * cov.get(0, 1) + 9.0 * cov.get(1, 1);
            assert!((joint - direct.get(0, 0)).abs() < 1e-12 * joint.abs().max(1.0), "{joint}");
        }
    }

    #[test]
    fn one_cluster_per_row_is_the_independent_row_variance_and_one_cluster_is_refused() {
        let x: Vec<f64> = (0..9).map(|i| (i * i % 7) as f64).collect();
        let mean = x.iter().sum::<f64>() / 9.0;
        let singletons: Arc<[u32]> = (0..9u32).collect();
        let independent = crate::joint_if::joint_influence_covariance(&[&x], None).unwrap();
        let clustered = dependent_influence_covariance(
            &[&x],
            None,
            &[mean],
            &ScoreDependence::Clusters(singletons),
        )
        .unwrap();
        assert!((independent.get(0, 0) - clustered.get(0, 0)).abs() < 1e-14);
        let one: Arc<[u32]> = vec![0u32; 9].into();
        assert!(
            dependent_influence_covariance(&[&x], None, &[mean], &ScoreDependence::Clusters(one))
                .is_err()
        );
        // A lag of zero is the lag-zero meat over n^2.
        let lag_zero = dependent_influence_covariance(
            &[&x],
            None,
            &[mean],
            &ScoreDependence::NeweyWest { lag: 0 },
        )
        .unwrap();
        let expected = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / 81.0;
        assert!((lag_zero.get(0, 0) - expected).abs() < 1e-14);
    }
}
