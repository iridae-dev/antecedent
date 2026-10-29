//! Joint discrepancy over a whole target vector.
//!
//! # What is claimed
//!
//! A refuter on a `k`-coordinate target compares the replicate distribution of the refuted
//! vector with an expected vector `e` (zero for a null refuter, the original estimate for a
//! stability refuter) through the squared Mahalanobis distance
//! `D² = (e − m)ᵀ S⁻¹ (e − m)` of `e` from the replicate mean `m` under the replicate
//! covariance `S` (the spread of one replicate). On one coordinate this is the squared
//! standardised shift `((e − m) / s)²`; on `k` coordinates it keeps their correlation instead of
//! `k` marginal checks.
//!
//! This is a **diagnostic with a fixed tolerance, not a hypothesis test**. `p_value` is the
//! `χ²_k` tail at `D²`, kept as a monotone yardstick: the check passes when it is at least the
//! configured `alpha`, which is the same as `D²` staying below the `(1 − alpha)` quantile of
//! `χ²_k`. Three facts keep it from being a calibrated test:
//!
//! * For a stability refuter, `m − e` estimates the resampling **bias** of the procedure
//!   (bootstrap mean minus the original estimate); the null "the population value is `e`" is
//!   not what is tested.
//! * The covariance is that of one replicate, while `m` is a mean of `R` replicates whose own
//!   covariance is `S / R`. For an unbiased procedure `R · D²` is a Hotelling `T²` on `k` and
//!   `R − 1` degrees of freedom, so `D²` is about `k / R` and a clean case rejects far less
//!   often than `alpha`; a declared violation is detected only when its bias exceeds about
//!   `sqrt(χ²_{k, 1−alpha})` replicate standard deviations.
//! * Nothing here is a calibration claim. The clean-case rejection rate and the power against
//!   planted violations are separate measurements.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_stats::chi_square_sf;

use crate::ValidationError;

/// How a [`JointStatistic`] was formed.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum JointStatisticKind {
    /// Squared Mahalanobis distance of the expectation from the replicate mean under the
    /// replicate covariance, with the `χ²_k` tail as a fixed yardstick (not a calibrated
    /// test; see the module documentation).
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
    /// Reference tail where the statistic has a reference yardstick. For
    /// [`JointStatisticKind::MahalanobisChiSquare`] this is the `χ²_k` tail of one replicate's
    /// spread, not the p-value of a test with a nominal rejection rate.
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
/// covariance is regularized by a ridge of [`RELATIVE_RIDGE`] times each coordinate's own
/// variance (the mean diagonal for a coordinate that never varies).
///
/// The singularity test and the ridge are relative to each coordinate's own variance, not to
/// the mean diagonal, so the statistic does not depend on the units of the coordinates: a
/// target mixing coordinates whose variances differ by more than `1 / RELATIVE_RIDGE` is a
/// well-conditioned correlation matrix, not a singular one.
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
    let floors: Vec<f64> = (0..k)
        .map(|i| {
            let own = original[i * k + i];
            RELATIVE_RIDGE * if own > 0.0 { own } else { scale }
        })
        .collect();
    // A pivot at rounding level is a direction the replicates do not vary in (a target whose
    // coordinates are affine in fewer parameters); treat it as singular rather than divide
    // by rounding error.
    if !cholesky_in_place(cov, k, &floors) {
        cov.copy_from_slice(&original);
        for i in 0..k {
            cov[i * k + i] += floors[i];
        }
        if !cholesky_in_place(cov, k, &vec![0.0; k]) {
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
/// most `min_pivot[i]` (not numerically positive definite).
fn cholesky_in_place(a: &mut [f64], k: usize, min_pivot: &[f64]) -> bool {
    for i in 0..k {
        for j in 0..=i {
            let mut s = a[i * k + j];
            for p in 0..j {
                s -= a[i * k + p] * a[j * k + p];
            }
            if i == j {
                if !(s.is_finite() && s > min_pivot[i]) {
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

    /// Deterministic standard normals (xorshift64*, Box-Muller).
    struct Normals(u64);

    impl Normals {
        fn uniform(&mut self) -> f64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            let bits = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11;
            (bits as f64 + 0.5) / (1u64 << 53) as f64
        }

        fn normal(&mut self) -> f64 {
            let (u, v) = (self.uniform(), self.uniform());
            (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
        }

        fn index(&mut self, n: usize) -> usize {
            ((self.uniform() * n as f64) as usize).min(n - 1)
        }
    }

    /// One data set of an iid problem with an exactly unbiased bootstrap mean (the estimator is
    /// the sample mean of `x` and, for `k = 2`, of `x²`), its `r` row-bootstrap replicates and
    /// the original estimate. Conditional on the data the replicate mean is the original
    /// estimate, so the exact law of the replicate mean around it is the Monte Carlo law of
    /// `S / r`, which is what the reference check below uses.
    fn bootstrap_problem(draw: &mut Normals, n: usize, r: usize, k: usize) -> (Vec<f64>, Vec<f64>) {
        let x: Vec<f64> = (0..n).map(|_| draw.normal()).collect();
        let stat = |rows: &[usize]| -> Vec<f64> {
            let m = rows.len() as f64;
            let mut out = vec![rows.iter().map(|&i| x[i]).sum::<f64>() / m];
            if k == 2 {
                out.push(rows.iter().map(|&i| x[i] * x[i]).sum::<f64>() / m);
            }
            out
        };
        let original = stat(&(0..n).collect::<Vec<_>>());
        let mut replicates = Vec::with_capacity(r * k);
        for _ in 0..r {
            let rows: Vec<usize> = (0..n).map(|_| draw.index(n)).collect();
            replicates.extend(stat(&rows));
        }
        (original, replicates)
    }

    /// The null law of the discrepancy on a case with known answer: an unbiased procedure
    /// (iid mean, bootstrap mean equal to the original). The discrepancy is the Monte Carlo
    /// error of the replicate mean over one replicate's spread, so `R · D²` is a Hotelling
    /// `T²` (mean `k (R − 1) / (R − k − 2)`) and `D²` itself is about `k / R`. Referred to the
    /// declared `χ²_k` yardstick it essentially never fails, so the yardstick is a tolerance on
    /// the bias, not a level-`alpha` test; referred to `χ²_k` after the `R` scaling it fails at
    /// about `alpha`, which documents what a calibrated test of the bootstrap mean would be.
    #[test]
    fn unbiased_bootstrap_discrepancy_is_the_monte_carlo_error_not_chi_square() {
        const REPS: usize = 300;
        const R: usize = 30;
        for k in [1_usize, 2] {
            let mut draw = Normals(0x9E37_79B9_7F4A_7C15 ^ k as u64);
            let (mut sum_scaled, mut reject_tolerance, mut reject_scaled) = (0.0, 0, 0);
            for _ in 0..REPS {
                let (original, replicates) = bootstrap_problem(&mut draw, 40, R, k);
                let stat = joint_mahalanobis(&original, &replicates, k).unwrap();
                sum_scaled += R as f64 * stat.value;
                reject_tolerance += usize::from(stat.p_value.unwrap() < 0.05);
                reject_scaled += usize::from(chi_square_sf(R as f64 * stat.value, k) < 0.05);
            }
            let mean_scaled = sum_scaled / REPS as f64;
            let hotelling = k as f64 * (R as f64 - 1.0) / (R as f64 - k as f64 - 2.0);
            assert!(
                (mean_scaled - hotelling).abs() < 0.3 * hotelling,
                "k = {k}: mean R·D² {mean_scaled} against the Hotelling mean {hotelling}"
            );
            // The tolerance yardstick rejects an unbiased procedure essentially never.
            assert!(reject_tolerance <= 2, "k = {k}: {reject_tolerance} of {REPS}");
            // Scaled by R the same statistic rejects near alpha (Hotelling vs chi-square
            // heavy tail: between 2% and 15%).
            let rate = reject_scaled as f64 / REPS as f64;
            assert!((0.02..0.15).contains(&rate), "k = {k}: scaled rejection rate {rate}");
        }
    }

    /// A bias of `z` replicate standard deviations is detected exactly when `z²` exceeds the
    /// yardstick: the tolerance is on standardised bias, whatever the replicate count.
    #[test]
    fn the_yardstick_is_a_tolerance_on_standardised_bias() {
        let mut draw = Normals(5);
        let replicates: Vec<f64> = (0..200).map(|_| draw.normal()).collect();
        let (mean, cov) = mean_and_covariance(&replicates, 1);
        let sd = cov[0].sqrt();
        let within = joint_mahalanobis(&[mean[0] + 1.5 * sd], &replicates, 1).unwrap();
        let beyond = joint_mahalanobis(&[mean[0] + 2.5 * sd], &replicates, 1).unwrap();
        assert!((within.value - 2.25).abs() < 1e-9 && within.p_value.unwrap() > 0.05);
        assert!((beyond.value - 6.25).abs() < 1e-9 && beyond.p_value.unwrap() < 0.05);
    }

    #[test]
    fn a_rank_deficient_replicate_law_is_regularised_and_still_detects_a_shift_off_its_span() {
        let mut draw = Normals(11);
        let mut replicates = Vec::new();
        for _ in 0..40 {
            let a = draw.normal();
            replicates.extend([a, 3.0 * a]);
        }
        let (mean, _) = mean_and_covariance(&replicates, 2);
        // On the span of the replicates: a finite, small discrepancy.
        let on = joint_mahalanobis(&[mean[0] + 0.1, mean[1] + 0.3], &replicates, 2).unwrap();
        assert!(on.value.is_finite() && on.value < 1.0, "{}", on.value);
        // Off the span: no replicate ever moved that way, so the discrepancy is enormous.
        let off = joint_mahalanobis(&[mean[0], mean[1] + 0.5], &replicates, 2).unwrap();
        assert!(off.value.is_finite() && off.value > 1e6, "{}", off.value);
        // A coordinate that never varies, expected where it sits, is not a discrepancy.
        let mut flat = Vec::new();
        for i in 0..40 {
            flat.extend([f64::from(i % 7), 5.0]);
        }
        let (flat_mean, _) = mean_and_covariance(&flat, 2);
        let at = joint_mahalanobis(&[flat_mean[0], 5.0], &flat, 2).unwrap();
        assert!(at.value < 1e-9, "{}", at.value);
    }

    #[test]
    fn the_discrepancy_does_not_depend_on_the_units_of_a_coordinate() {
        let mut draw = Normals(23);
        let mut base = Vec::new();
        let mut scaled = Vec::new();
        for _ in 0..60 {
            let (a, b) = (draw.normal(), draw.normal());
            base.extend([a, a + b]);
            scaled.extend([a * 1e6, (a + b) * 1e-6]);
        }
        let one = joint_mahalanobis(&[0.3, -0.2], &base, 2).unwrap();
        let two = joint_mahalanobis(&[0.3e6, -0.2e-6], &scaled, 2).unwrap();
        assert!((one.value - two.value).abs() < 1e-6 * one.value.max(1.0));
    }

    #[test]
    fn too_few_replicates_are_refused() {
        assert!(joint_mahalanobis(&[0.0, 0.0], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2).is_err());
    }
}
