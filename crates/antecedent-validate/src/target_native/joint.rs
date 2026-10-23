//! Joint statistics over a whole target vector.
//!
//! A refuter on a `k`-coordinate target compares the replicate distribution of the refuted
//! vector with an expected vector `e` (zero for a null refuter, the original estimate for a
//! stability refuter) through the squared Mahalanobis distance
//! `D² = (e − m)ᵀ S⁻¹ (e − m)` of `e` from the replicate mean `m` under the replicate
//! covariance `S`, referred to `χ²_k`. On one coordinate this is the squared z-score of the
//! scalar refuters' "null value under the replicate distribution" p-value; on `k`
//! coordinates it keeps their correlation instead of `k` marginal tests.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_stats::chi_square_sf;

use crate::ValidationError;

/// How a [`JointStatistic`] was formed.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum JointStatisticKind {
    /// Squared Mahalanobis distance under a replicate covariance, referred to `χ²_k`.
    MahalanobisChiSquare,
    /// Squared Mahalanobis shift under the baseline posterior covariance (no reference law;
    /// compared with a declared tolerance).
    PosteriorShift,
    /// Posterior-predictive tail probability of a discrepancy.
    PredictiveTail,
    /// Largest standardized discrepancy across evaluation points.
    MaxStandardizedDiscrepancy,
    /// Break-even value of a sensitivity parameter (compared with a declared robustness bar).
    BreakEven,
}

impl JointStatisticKind {
    /// Stable name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::MahalanobisChiSquare => "mahalanobis_chi_square",
            Self::PosteriorShift => "posterior_shift",
            Self::PredictiveTail => "predictive_tail",
            Self::MaxStandardizedDiscrepancy => "max_standardized_discrepancy",
            Self::BreakEven => "break_even",
        }
    }
}

/// One joint statistic over a target vector.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JointStatistic {
    /// How the value was formed.
    pub kind: JointStatisticKind,
    /// Statistic value.
    pub value: f64,
    /// Degrees of freedom (the target dimension) for a `χ²` reference.
    pub df: usize,
    /// Tail probability, where the statistic has a reference law.
    pub p_value: Option<f64>,
}

/// Ridge added to the covariance diagonal, relative to its mean diagonal, when the replicate
/// covariance is numerically singular (a coordinate that never varies).
const RELATIVE_RIDGE: f64 = 1e-10;

/// Squared Mahalanobis distance of `expected` from the mean of `replicates`.
///
/// `replicates` holds `r` vectors of dimension `k` (row-major, `r × k`).
///
/// # Errors
///
/// Fewer than `k + 2` replicates (the covariance would be singular), a non-finite replicate,
/// or a covariance whose every coordinate is constant.
pub fn joint_mahalanobis(
    expected: &[f64],
    replicates: &[f64],
    k: usize,
) -> Result<JointStatistic, ValidationError> {
    if k == 0 || expected.len() != k || replicates.len() % k != 0 {
        return Err(ValidationError::NotApplicable {
            message: "joint statistic: expected vector and replicate rows disagree in dimension",
        });
    }
    let r = replicates.len() / k;
    if r < k + 2 {
        return Err(ValidationError::NotApplicable {
            message: "joint statistic needs at least target dimension + 2 successful replicates; \
                      next attempt: raise the replicate count",
        });
    }
    if replicates.iter().any(|v| !v.is_finite()) {
        return Err(ValidationError::estimation_msg("a refuted target vector is not finite"));
    }
    let (mean, mut cov) = mean_and_covariance(replicates, k);
    let diff: Vec<f64> = expected.iter().zip(&mean).map(|(e, m)| e - m).collect();
    let value = mahalanobis_squared(&diff, &mut cov, k)?;
    Ok(JointStatistic {
        kind: JointStatisticKind::MahalanobisChiSquare,
        value,
        df: k,
        p_value: Some(chi_square_sf(value, k)),
    })
}

/// Column means and the unbiased covariance (`k × k`, row-major) of `r × k` rows.
pub(crate) fn mean_and_covariance(rows: &[f64], k: usize) -> (Vec<f64>, Vec<f64>) {
    let r = rows.len() / k;
    let mut mean = vec![0.0; k];
    for row in rows.chunks_exact(k) {
        for (m, v) in mean.iter_mut().zip(row) {
            *m += v;
        }
    }
    for m in &mut mean {
        *m /= r as f64;
    }
    let mut cov = vec![0.0; k * k];
    for row in rows.chunks_exact(k) {
        for i in 0..k {
            let di = row[i] - mean[i];
            for j in 0..=i {
                cov[i * k + j] += di * (row[j] - mean[j]);
            }
        }
    }
    let denom = (r.max(2) - 1) as f64;
    for i in 0..k {
        for j in 0..=i {
            let value = cov[i * k + j] / denom;
            cov[i * k + j] = value;
            cov[j * k + i] = value;
        }
    }
    (mean, cov)
}

/// `dᵀ C⁻¹ d` through a Cholesky factor of `cov` (overwritten). A numerically singular
/// covariance is regularized by a ridge of [`RELATIVE_RIDGE`] times its mean diagonal.
pub(crate) fn mahalanobis_squared(
    diff: &[f64],
    cov: &mut [f64],
    k: usize,
) -> Result<f64, ValidationError> {
    let scale = (0..k).map(|i| cov[i * k + i]).sum::<f64>() / k as f64;
    if !(scale.is_finite() && scale > 0.0) {
        return Err(ValidationError::NotApplicable {
            message: "joint statistic: every coordinate of the refuted target is constant across \
                      replicates",
        });
    }
    let original = cov.to_vec();
    // A pivot at rounding level is a direction the replicates do not vary in (a target whose
    // coordinates are affine in fewer parameters); treat it as singular rather than divide
    // by rounding error.
    if !cholesky_in_place(cov, k, RELATIVE_RIDGE * scale) {
        cov.copy_from_slice(&original);
        for i in 0..k {
            cov[i * k + i] += RELATIVE_RIDGE * scale;
        }
        if !cholesky_in_place(cov, k, 0.0) {
            return Err(ValidationError::estimation_msg(
                "joint statistic: replicate covariance is not positive definite",
            ));
        }
    }
    // Forward solve L y = d; the statistic is |y|².
    let mut y = diff.to_vec();
    for i in 0..k {
        let mut s = y[i];
        for j in 0..i {
            s -= cov[i * k + j] * y[j];
        }
        y[i] = s / cov[i * k + i];
    }
    Ok(y.iter().map(|v| v * v).sum())
}

/// Lower Cholesky factor in place (`k × k`, row-major); `false` when a squared pivot is at
/// most `min_pivot` (not numerically positive definite).
fn cholesky_in_place(a: &mut [f64], k: usize, min_pivot: f64) -> bool {
    for i in 0..k {
        for j in 0..=i {
            let mut s = a[i * k + j];
            for p in 0..j {
                s -= a[i * k + p] * a[j * k + p];
            }
            if i == j {
                if !(s.is_finite() && s > min_pivot) {
                    return false;
                }
                a[i * k + i] = s.sqrt();
            } else {
                a[i * k + j] = s / a[j * k + j];
            }
        }
    }
    for i in 0..k {
        for j in (i + 1)..k {
            a[i * k + j] = 0.0;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_coordinate_reduces_to_the_squared_z_score() {
        let reps = [1.0, 2.0, 3.0, 4.0, 5.0];
        // mean 3, variance 2.5; distance of 0 is 9 / 2.5.
        let stat = joint_mahalanobis(&[0.0], &reps, 1).unwrap();
        assert!((stat.value - 9.0 / 2.5).abs() < 1e-12);
        assert_eq!(stat.df, 1);
    }

    #[test]
    fn correlated_coordinates_are_not_two_marginal_tests() {
        // Two perfectly-correlated-but-for-noise coordinates: a shift along the correlation is
        // cheap, a shift against it is expensive.
        let mut reps = Vec::new();
        for i in 0..50 {
            let u = f64::from(i) / 10.0 - 2.5;
            let w = if i % 2 == 0 { 0.05 } else { -0.05 };
            reps.push(u);
            reps.push(u + w);
        }
        let along = joint_mahalanobis(&[1.0, 1.0], &reps, 2).unwrap();
        let against = joint_mahalanobis(&[1.0, -1.0], &reps, 2).unwrap();
        assert!(against.value > 50.0 * along.value);
    }

    #[test]
    fn too_few_replicates_are_refused() {
        assert!(joint_mahalanobis(&[0.0, 0.0], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2).is_err());
    }
}
