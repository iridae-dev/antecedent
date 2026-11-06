//! Binary misclassification correction.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::MeasurementErrorError;

// ---------------------------------------------------------------------------
// Binary misclassification
// ---------------------------------------------------------------------------

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
