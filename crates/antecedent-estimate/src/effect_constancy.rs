//! F18 `EffectConstancy`: a global heterogeneity test of one effect across a declared
//! partition (time periods or regions), under a frozen null and a shared effect estimand.
//!
//! # Contract
//!
//! * **Frozen null:** [`EFFECT_CONSTANCY_NULL`], the effect is equal across *all* declared
//!   partitions. The null is not configurable.
//! * **Shared estimand:** every partition must carry the *same* [`EffectEstimandIdentity`]
//!   (effect, units, regime, population). A partition that changes any of them refuses the
//!   global test (`route_not_supported`, `effect_constancy.incompatible_partitions`); the
//!   statistic of estimates of different quantities has no meaning.
//! * **Partition dependence is declared, never assumed from silence:**
//!   [`PartitionDependence::Independent`] runs Cochran's `Q`; a supplied full covariance of
//!   the partition estimates runs the Wald chi-square of the contrasts against the first
//!   partition. With a diagonal covariance both are the same number.
//! * **Multiplicity:** per-pair (or per-reference) contrasts with Holm adjustment, and the
//!   declared [`ContrastFamily`] is reported with them.
//! * **Coordinate support:** each partition carries a typed coordinate label and a
//!   [`PartitionSupport`] status; an unsupported partition refuses.
//! * **Order independence:** partitions are put in canonical (label) order before any
//!   arithmetic, so relabelling or permuting the input gives bit-identical output.
//!
//! # What a result does not say
//!
//! Non-rejection does **not** prove constancy ([`ConstancyConclusion::NotRejected`] is the
//! absence of evidence against the null at the stated level). The test's Type I error and
//! power are *unmeasured* ([`CalibrationStatus::Unmeasured`]); no coverage or false-positive
//! claim is made here, and the result grants no causal or interval license. The per-partition
//! effects are exposed so transport diagnostics, prior transfer and policy generalization can
//! consume them directly.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::cmp::Ordering;
use std::collections::BTreeSet;

use antecedent_core::reason_code;
use antecedent_kernels::erfc;
use antecedent_stats::gamma_q;

use crate::error::EstimationError;

/// The frozen null of the global test.
pub const EFFECT_CONSTANCY_NULL: &str = "effect equal across all declared partitions";

/// Inference claim of the route.
pub const EFFECT_CONSTANCY_INFERENCE_CLAIM: &str = "point_only";

/// Most partitions one test accepts.
pub const EFFECT_CONSTANCY_MAX_PARTITIONS: usize = 1024;

/// Statement attached to every result: non-rejection is not evidence of constancy.
pub const NON_REJECTION_CAVEAT: &str = "failing to reject the null does not prove the effect is \
                                        constant across partitions";

/// Statement attached to every result about the unmeasured error rates.
pub const POWER_CAVEAT: &str = "Type I error and power of this test are unmeasured; the test may \
                                lack power to detect heterogeneity of a size that matters";

/// Relative tolerance for covariance symmetry and its agreement with the standard errors.
const COVARIANCE_TOLERANCE: f64 = 1e-9;

/// The effect every partition must estimate, identically.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EffectEstimandIdentity {
    /// The effect definition (for example `ate_difference`).
    pub estimand: String,
    /// Units / scale of the effect.
    pub units: String,
    /// Regime or intervention contrast the effect is defined under.
    pub regime: String,
    /// Population (and horizon) the effect is defined for.
    pub population: String,
}

/// Coordinate support of one partition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PartitionSupport {
    /// The partition's coordinate is fully supported by the evidence.
    Supported,
    /// Only part of the coordinate is supported; the estimate is reported with that status.
    Partial,
    /// The coordinate is unsupported: the partition cannot enter the test.
    Unsupported,
}

/// One partition's effect estimate.
#[derive(Clone, Debug, PartialEq)]
pub struct PartitionEstimate {
    /// Unique partition label.
    pub label: String,
    /// Typed coordinate of the partition (a period or region identifier).
    pub coordinate: String,
    /// Coordinate support status.
    pub support: PartitionSupport,
    /// The estimand identity the effect estimates.
    pub estimand: EffectEstimandIdentity,
    /// The effect estimate.
    pub effect: f64,
    /// Its standard error (positive).
    pub standard_error: f64,
}

/// Dependence between partition estimates.
#[derive(Clone, Debug, PartialEq)]
pub enum PartitionDependence {
    /// The partitions use disjoint, independent evidence: Cochran's `Q`.
    Independent,
    /// Full row-major `k x k` covariance of the estimates, in the order the partitions are
    /// supplied: the Wald chi-square.
    Covariance(Vec<f64>),
}

/// Declared family of contrasts that the multiplicity adjustment covers.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ContrastFamily {
    /// Every unordered pair of partitions.
    AllPairs,
    /// Every other partition against the named reference partition.
    AgainstReference(String),
}

/// Which global statistic was computed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HeterogeneityStatistic {
    /// Cochran's `Q` for independent partitions.
    CochranQ,
    /// Wald chi-square with the supplied full covariance.
    WaldChiSquare,
}

/// Decision of the global test at the declared level.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConstancyConclusion {
    /// The null was not rejected. This does **not** establish constancy.
    NotRejected,
    /// The null was rejected: the effects differ across partitions.
    Rejected,
}

/// State of the test's error-rate calibration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CalibrationStatus {
    /// Null Type I error and power have not been measured; no claim is made.
    Unmeasured,
}

/// Global heterogeneity test output.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeterogeneityTest {
    /// Which statistic.
    pub statistic_kind: HeterogeneityStatistic,
    /// The statistic.
    pub statistic: f64,
    /// Degrees of freedom (`k - 1`).
    pub degrees_of_freedom: usize,
    /// Chi-square tail probability.
    pub p_value: f64,
}

/// One partition as reported (canonical order).
#[derive(Clone, Debug, PartialEq)]
pub struct PartitionReport {
    /// Partition label.
    pub label: String,
    /// Typed coordinate.
    pub coordinate: String,
    /// Coordinate support.
    pub support: PartitionSupport,
    /// Effect estimate.
    pub effect: f64,
    /// Standard error.
    pub standard_error: f64,
}

/// One multiplicity-adjusted contrast.
#[derive(Clone, Debug, PartialEq)]
pub struct ContrastReport {
    /// First partition label (effect minuend).
    pub left: String,
    /// Second partition label (effect subtrahend).
    pub right: String,
    /// `effect(left) - effect(right)`.
    pub difference: f64,
    /// Standard error of the difference.
    pub standard_error: f64,
    /// Standardized difference.
    pub z: f64,
    /// Two-sided normal p-value, unadjusted.
    pub p_value: f64,
    /// Holm-adjusted p-value over the declared family.
    pub p_holm: f64,
    /// Whether the Holm-adjusted p-value is below the level.
    pub rejected: bool,
}

/// Result of an `EffectConstancy` test.
#[derive(Clone, Debug, PartialEq)]
pub struct EffectConstancyResult {
    /// The frozen null ([`EFFECT_CONSTANCY_NULL`]).
    pub null: &'static str,
    /// The shared estimand.
    pub estimand: EffectEstimandIdentity,
    /// Partitions in canonical (label) order, with their effects for downstream consumers.
    pub partitions: Vec<PartitionReport>,
    /// Global test.
    pub test: HeterogeneityTest,
    /// Level of the decisions.
    pub alpha: f64,
    /// Decision of the global test; non-rejection is never a proof of constancy.
    pub conclusion: ConstancyConclusion,
    /// [`NON_REJECTION_CAVEAT`].
    pub non_rejection_caveat: &'static str,
    /// [`POWER_CAVEAT`].
    pub power_caveat: &'static str,
    /// Calibration state: unmeasured.
    pub calibration: CalibrationStatus,
    /// The declared multiplicity family.
    pub family: ContrastFamily,
    /// Contrasts of the family with Holm adjustment, in canonical order.
    pub contrasts: Vec<ContrastReport>,
    /// Inverse-variance pooled effect; present only for independent partitions.
    pub pooled_effect: Option<f64>,
    /// [`EFFECT_CONSTANCY_INFERENCE_CLAIM`].
    pub inference_claim: &'static str,
}

fn refuse(code: &'static str, detail: &str, message: &str) -> EstimationError {
    EstimationError::refused(code, format!("{detail}: {message}"))
}

fn count(n: usize) -> f64 {
    f64::from(u32::try_from(n).unwrap_or(u32::MAX))
}

/// Chi-square survival function through the regularized upper incomplete gamma.
fn chi_square_sf(statistic: f64, df: usize) -> f64 {
    if statistic <= 0.0 {
        return 1.0;
    }
    gamma_q(count(df) * 0.5, statistic * 0.5)
}

/// Holm step-down adjustment of raw p-values; the result is in the input order.
#[must_use]
pub fn holm_adjust(p_values: &[f64]) -> Vec<f64> {
    let m = p_values.len();
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| p_values[a].total_cmp(&p_values[b]).then(a.cmp(&b)));
    let mut adjusted = vec![0.0; m];
    let mut running = 0.0_f64;
    for (rank, &index) in order.iter().enumerate() {
        let scaled = (count(m - rank) * p_values[index]).min(1.0);
        running = running.max(scaled);
        adjusted[index] = running;
    }
    adjusted
}

/// Lower Cholesky factor of a row-major symmetric `n x n` matrix; `None` unless positive
/// definite (every pivot strictly positive and finite).
fn cholesky(matrix: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut lower = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = matrix[i * n + j];
            for k in 0..j {
                sum -= lower[i * n + k] * lower[j * n + k];
            }
            if i == j {
                if sum.is_nan() || sum <= 0.0 || !sum.is_finite() {
                    return None;
                }
                lower[i * n + i] = sum.sqrt();
            } else {
                lower[i * n + j] = sum / lower[j * n + j];
            }
        }
    }
    Some(lower)
}

/// `x' V^-1 x` from the Cholesky factor `lower` of `V`.
fn quadratic_form(lower: &[f64], n: usize, x: &[f64]) -> f64 {
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut sum = x[i];
        for k in 0..i {
            sum -= lower[i * n + k] * y[k];
        }
        y[i] = sum / lower[i * n + i];
    }
    y.iter().map(|v| v * v).sum()
}

fn validate_inputs(partitions: &[PartitionEstimate], alpha: f64) -> Result<(), EstimationError> {
    let invalid = reason_code!("invalid_argument");
    if !(alpha > 0.0 && alpha < 1.0) {
        return Err(refuse(invalid, "effect_constancy.invalid_alpha", "alpha must lie in (0, 1)"));
    }
    if partitions.len() < 2 {
        return Err(refuse(
            invalid,
            "effect_constancy.too_few_partitions",
            "a constancy test needs at least two partitions",
        ));
    }
    if partitions.len() > EFFECT_CONSTANCY_MAX_PARTITIONS {
        return Err(refuse(
            invalid,
            "effect_constancy.too_many_partitions",
            "the partition count exceeds the supported bound",
        ));
    }
    for p in partitions {
        if !p.effect.is_finite() || !p.standard_error.is_finite() {
            return Err(refuse(
                invalid,
                "effect_constancy.non_finite_estimate",
                "an effect or standard error is not finite",
            ));
        }
        if p.standard_error <= 0.0 {
            return Err(refuse(
                invalid,
                "effect_constancy.invalid_standard_error",
                "a partition standard error is not positive",
            ));
        }
    }
    Ok(())
}

fn validate_comparability(partitions: &[PartitionEstimate]) -> Result<(), EstimationError> {
    let incompatible = reason_code!("route_not_supported");
    let first = &partitions[0].estimand;
    if partitions.iter().any(|p| &p.estimand != first) {
        return Err(refuse(
            incompatible,
            "effect_constancy.incompatible_partitions",
            "partitions estimate different effects (estimand, units, regime or population \
             differ); the global test is refused",
        ));
    }
    let mut labels = BTreeSet::new();
    if !partitions.iter().all(|p| labels.insert(p.label.as_str())) {
        return Err(refuse(
            incompatible,
            "effect_constancy.incompatible_partitions",
            "partition labels are not unique",
        ));
    }
    if partitions.iter().any(|p| p.support == PartitionSupport::Unsupported) {
        return Err(refuse(
            incompatible,
            "effect_constancy.unsupported_partition",
            "a partition's coordinate is unsupported and cannot enter the test",
        ));
    }
    Ok(())
}

/// Validate a full covariance against the (original-order) standard errors.
fn validate_covariance(
    covariance: &[f64],
    partitions: &[PartitionEstimate],
) -> Result<(), EstimationError> {
    let invalid = reason_code!("invalid_argument");
    let k = partitions.len();
    let bad = |message: &str| refuse(invalid, "effect_constancy.invalid_covariance", message);
    if covariance.len() != k * k {
        return Err(bad("the covariance is not k x k for the supplied partitions"));
    }
    if covariance.iter().any(|v| !v.is_finite()) {
        return Err(bad("the covariance has a non-finite entry"));
    }
    for i in 0..k {
        for j in 0..i {
            let (a, b) = (covariance[i * k + j], covariance[j * k + i]);
            if (a - b).abs() > COVARIANCE_TOLERANCE * a.abs().max(b.abs()).max(1.0) {
                return Err(bad("the covariance is not symmetric"));
            }
        }
        let variance = partitions[i].standard_error * partitions[i].standard_error;
        if (covariance[i * k + i] - variance).abs() > COVARIANCE_TOLERANCE * variance {
            return Err(bad(
                "a covariance diagonal entry does not equal the squared standard error",
            ));
        }
    }
    if cholesky(covariance, k).is_none() {
        return Err(bad("the covariance is not positive definite"));
    }
    Ok(())
}

/// The canonical order: partition indices sorted by label.
fn canonical_order(partitions: &[PartitionEstimate]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..partitions.len()).collect();
    order.sort_by(|&a, &b| match partitions[a].label.cmp(&partitions[b].label) {
        Ordering::Equal => a.cmp(&b),
        other => other,
    });
    order
}

/// Wald chi-square of the contrasts against the first (canonical) partition.
fn wald_statistic(effects: &[f64], covariance: &[f64]) -> Result<f64, EstimationError> {
    let k = effects.len();
    let m = k - 1;
    let d: Vec<f64> = (1..k).map(|i| effects[i] - effects[0]).collect();
    let mut v = vec![0.0; m * m];
    for i in 0..m {
        for j in 0..m {
            v[i * m + j] =
                covariance[(i + 1) * k + (j + 1)] - covariance[(i + 1) * k] - covariance[j + 1]
                    + covariance[0];
        }
    }
    let lower = cholesky(&v, m).ok_or_else(|| {
        refuse(
            reason_code!("invalid_argument"),
            "effect_constancy.invalid_covariance",
            "the covariance of the contrasts is not positive definite",
        )
    })?;
    Ok(quadratic_form(&lower, m, &d))
}

/// Cochran's `Q` and the inverse-variance pooled effect.
fn cochran_q(effects: &[f64], ses: &[f64]) -> (f64, f64) {
    let weights: Vec<f64> = ses.iter().map(|s| 1.0 / (s * s)).collect();
    let total: f64 = weights.iter().sum();
    let pooled = weights.iter().zip(effects).map(|(w, e)| w * e).sum::<f64>() / total;
    let q = weights.iter().zip(effects).map(|(w, e)| w * (e - pooled) * (e - pooled)).sum();
    (q, pooled)
}

fn contrast_pairs(
    labels: &[&str],
    family: &ContrastFamily,
) -> Result<Vec<(usize, usize)>, EstimationError> {
    let k = labels.len();
    match family {
        ContrastFamily::AllPairs => {
            Ok((0..k).flat_map(|i| ((i + 1)..k).map(move |j| (i, j))).collect())
        }
        ContrastFamily::AgainstReference(reference) => {
            let at = labels.iter().position(|l| *l == reference.as_str()).ok_or_else(|| {
                refuse(
                    reason_code!("invalid_argument"),
                    "effect_constancy.unknown_reference",
                    "the reference partition is not among the declared partitions",
                )
            })?;
            Ok((0..k).filter(|i| *i != at).map(|i| (i, at)).collect())
        }
    }
}

fn contrast_reports(
    partitions: &[PartitionReport],
    covariance: &[f64],
    family: &ContrastFamily,
    alpha: f64,
) -> Result<Vec<ContrastReport>, EstimationError> {
    let k = partitions.len();
    let labels: Vec<&str> = partitions.iter().map(|p| p.label.as_str()).collect();
    let pairs = contrast_pairs(&labels, family)?;
    let mut rows = Vec::with_capacity(pairs.len());
    for (a, b) in pairs {
        let variance = covariance[a * k + a] + covariance[b * k + b] - 2.0 * covariance[a * k + b];
        if !(variance.is_finite() && variance > 0.0) {
            return Err(refuse(
                reason_code!("invalid_argument"),
                "effect_constancy.invalid_covariance",
                "a contrast has a non-positive variance",
            ));
        }
        let difference = partitions[a].effect - partitions[b].effect;
        let se = variance.sqrt();
        let z = difference / se;
        rows.push((a, b, difference, se, z, erfc(z.abs() / std::f64::consts::SQRT_2)));
    }
    let raw: Vec<f64> = rows.iter().map(|r| r.5).collect();
    let adjusted = holm_adjust(&raw);
    Ok(rows
        .into_iter()
        .zip(adjusted)
        .map(|((a, b, difference, standard_error, z, p_value), p_holm)| ContrastReport {
            left: partitions[a].label.clone(),
            right: partitions[b].label.clone(),
            difference,
            standard_error,
            z,
            p_value,
            p_holm,
            rejected: p_holm < alpha,
        })
        .collect())
}

/// Run the F18 constancy test.
///
/// Partitions are put in label order first, so the result is invariant to input order and
/// relabelling that keeps the label order.
///
/// # Errors
/// `route_not_supported` with `effect_constancy.incompatible_partitions` (differing estimand,
/// units, regime or population, or duplicate labels) or `effect_constancy.unsupported_partition`;
/// `invalid_argument` with `effect_constancy.too_few_partitions`,
/// `effect_constancy.too_many_partitions`, `effect_constancy.invalid_covariance`,
/// `effect_constancy.non_finite_estimate`, `effect_constancy.invalid_standard_error`,
/// `effect_constancy.invalid_alpha` or `effect_constancy.unknown_reference`.
pub fn test_effect_constancy(
    partitions: &[PartitionEstimate],
    dependence: &PartitionDependence,
    family: &ContrastFamily,
    alpha: f64,
) -> Result<EffectConstancyResult, EstimationError> {
    validate_inputs(partitions, alpha)?;
    validate_comparability(partitions)?;
    let k = partitions.len();
    if let PartitionDependence::Covariance(covariance) = dependence {
        validate_covariance(covariance, partitions)?;
    }
    let order = canonical_order(partitions);
    let reports: Vec<PartitionReport> = order
        .iter()
        .map(|&i| {
            let p = &partitions[i];
            PartitionReport {
                label: p.label.clone(),
                coordinate: p.coordinate.clone(),
                support: p.support,
                effect: p.effect,
                standard_error: p.standard_error,
            }
        })
        .collect();
    let effects: Vec<f64> = reports.iter().map(|p| p.effect).collect();
    let ses: Vec<f64> = reports.iter().map(|p| p.standard_error).collect();
    let (kind, statistic, covariance, pooled_effect) = match dependence {
        PartitionDependence::Independent => {
            let (q, pooled) = cochran_q(&effects, &ses);
            let mut diagonal = vec![0.0; k * k];
            for (i, s) in ses.iter().enumerate() {
                diagonal[i * k + i] = s * s;
            }
            (HeterogeneityStatistic::CochranQ, q, diagonal, Some(pooled))
        }
        PartitionDependence::Covariance(supplied) => {
            let mut permuted = vec![0.0; k * k];
            for (i, &a) in order.iter().enumerate() {
                for (j, &b) in order.iter().enumerate() {
                    permuted[i * k + j] = supplied[a * k + b];
                }
            }
            let w = wald_statistic(&effects, &permuted)?;
            (HeterogeneityStatistic::WaldChiSquare, w, permuted, None)
        }
    };
    let df = k - 1;
    let p_value = chi_square_sf(statistic, df);
    let contrasts = contrast_reports(&reports, &covariance, family, alpha)?;
    Ok(EffectConstancyResult {
        null: EFFECT_CONSTANCY_NULL,
        estimand: partitions[0].estimand.clone(),
        partitions: reports,
        test: HeterogeneityTest {
            statistic_kind: kind,
            statistic,
            degrees_of_freedom: df,
            p_value,
        },
        alpha,
        conclusion: if p_value < alpha {
            ConstancyConclusion::Rejected
        } else {
            ConstancyConclusion::NotRejected
        },
        non_rejection_caveat: NON_REJECTION_CAVEAT,
        power_caveat: POWER_CAVEAT,
        calibration: CalibrationStatus::Unmeasured,
        family: family.clone(),
        contrasts,
        pooled_effect,
        inference_claim: EFFECT_CONSTANCY_INFERENCE_CLAIM,
    })
}
