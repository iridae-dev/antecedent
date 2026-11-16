//! Misclassification of a binary exposure, including error that differs by outcome.
//!
//! The recorded exposure `X*` differs from the true exposure `X` by a sensitivity
//! `Se = P(X* = 1 | X = 1)` and a specificity `Sp = P(X* = 0 | X = 0)`. When the error
//! does not depend on the outcome `Y` one pair of rates serves both outcome levels; when
//! it does (recall that differs between cases and controls, detection that follows the
//! disease) each outcome level has its own pair, `Se_y` and `Sp_y`.
//!
//! Within outcome level `y` the recorded exposure share is
//! `p_y = Se_y pi_y + (1 - Sp_y)(1 - pi_y)`, so the true exposure share is
//! `pi_y = (p_y + Sp_y - 1) / J_y` with `J_y = Se_y + Sp_y - 1`. The exposure odds
//! ratio is `OR = [pi_1 / (1 - pi_1)] / [pi_0 / (1 - pi_0)]`; it is the outcome odds
//! ratio of the exposure under case-control and cohort sampling alike.
//!
//! The variance is a delta-method variance in the observed shares and the rate
//! estimates. Rates estimated from independent validation samples are independent of
//! each other and of the main sample. One pair shared by both outcome levels is one
//! estimate, so its derivatives add before they are squared.

use super::{BinaryErrorRates, MeasurementErrorError};

/// Recorded exposure counts by outcome level.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExposureOutcomeCounts {
    /// Units recorded as exposed among those with the outcome.
    pub exposed_cases: u64,
    /// Units with the outcome.
    pub cases: u64,
    /// Units recorded as exposed among those without the outcome.
    pub exposed_controls: u64,
    /// Units without the outcome.
    pub controls: u64,
}

/// How the exposure's error depends on the outcome.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ExposureErrorStructure {
    /// One sensitivity and specificity for both outcome levels (non-differential).
    Shared(BinaryErrorRates),
    /// Outcome-specific rates, from independent samples or declarations (differential).
    ByOutcome {
        /// Rates among units without the outcome.
        controls: BinaryErrorRates,
        /// Rates among units with the outcome.
        cases: BinaryErrorRates,
    },
}

/// A corrected exposure association with its variance components.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CorrectedExposureAssociation {
    /// Recorded exposure share among cases.
    pub observed_prevalence_cases: f64,
    /// Recorded exposure share among controls.
    pub observed_prevalence_controls: f64,
    /// Corrected exposure share among cases.
    pub prevalence_cases: f64,
    /// Corrected exposure share among controls.
    pub prevalence_controls: f64,
    /// Log odds ratio of the recorded exposure; `None` when a recorded cell is empty.
    pub naive_log_odds_ratio: Option<f64>,
    /// Corrected log odds ratio.
    pub log_odds_ratio: f64,
    /// Corrected odds ratio.
    pub odds_ratio: f64,
    /// Delta-method variance of the corrected log odds ratio.
    pub variance: f64,
    /// Part due to sampling of the recorded shares.
    pub variance_sampling: f64,
    /// Part due to the estimated sensitivities and specificities.
    pub variance_rates: f64,
    /// Youden index applied to cases.
    pub youden_cases: f64,
    /// Youden index applied to controls.
    pub youden_controls: f64,
    /// Whether the outcome-specific rates were applied.
    pub differential: bool,
}

fn youden_of(rates: &BinaryErrorRates, minimum: f64) -> Result<f64, MeasurementErrorError> {
    if !(minimum.is_finite() && minimum > 0.0 && minimum < 1.0) {
        return Err(MeasurementErrorError::InvalidParameter(
            "minimum Youden index must lie in (0, 1)",
        ));
    }
    let j = rates.sensitivity + rates.specificity - 1.0;
    if j.is_nan() || j <= minimum {
        return Err(MeasurementErrorError::YoudenTooSmall { youden: j, minimum });
    }
    Ok(j)
}

fn check_stratum(events: u64, total: u64) -> Result<f64, MeasurementErrorError> {
    if total == 0 || events > total {
        return Err(MeasurementErrorError::InvalidParameter(
            "recorded exposed counts must not exceed a positive stratum total",
        ));
    }
    Ok(events as f64 / total as f64)
}

fn logit(p: f64) -> f64 {
    (p / (1.0 - p)).ln()
}

/// The corrected exposure shares, log odds ratio and odds ratio of a misclassified
/// binary exposure.
///
/// # Errors
/// Invalid counts, a Youden index at or below `min_youden` in either outcome level, or
/// a corrected exposure share outside `(0, 1)` (the odds are then undefined) refuse.
pub fn corrected_exposure_association(
    counts: &ExposureOutcomeCounts,
    structure: &ExposureErrorStructure,
    min_youden: f64,
) -> Result<CorrectedExposureAssociation, MeasurementErrorError> {
    let p1 = check_stratum(counts.exposed_cases, counts.cases)?;
    let p0 = check_stratum(counts.exposed_controls, counts.controls)?;
    let (rates_cases, rates_controls, differential) = match structure {
        ExposureErrorStructure::Shared(rates) => (*rates, *rates, false),
        ExposureErrorStructure::ByOutcome { controls, cases } => (*cases, *controls, true),
    };
    let j1 = youden_of(&rates_cases, min_youden)?;
    let j0 = youden_of(&rates_controls, min_youden)?;
    let pi1 = (p1 + rates_cases.specificity - 1.0) / j1;
    let pi0 = (p0 + rates_controls.specificity - 1.0) / j0;
    let interior = |v: f64| v > 0.0 && v < 1.0;
    if !(interior(pi1) && interior(pi0)) {
        return Err(MeasurementErrorError::InvalidParameter(
            "a corrected exposure share lies outside (0, 1), so the odds ratio is undefined",
        ));
    }
    let g1 = 1.0 / (pi1 * (1.0 - pi1));
    let g0 = 1.0 / (pi0 * (1.0 - pi0));
    let n1 = counts.cases as f64;
    let n0 = counts.controls as f64;
    let sampling =
        g1 * g1 * p1 * (1.0 - p1) / n1 / (j1 * j1) + g0 * g0 * p0 * (1.0 - p0) / n0 / (j0 * j0);
    let from_rates = match structure {
        ExposureErrorStructure::ByOutcome { .. } => {
            g1 * g1
                * ((1.0 - pi1) * (1.0 - pi1) * rates_cases.specificity_variance
                    + pi1 * pi1 * rates_cases.sensitivity_variance)
                / (j1 * j1)
                + g0 * g0
                    * ((1.0 - pi0) * (1.0 - pi0) * rates_controls.specificity_variance
                        + pi0 * pi0 * rates_controls.sensitivity_variance)
                    / (j0 * j0)
        }
        ExposureErrorStructure::Shared(rates) => {
            // One estimate serves both strata: the derivatives of the log odds ratio in the
            // shared specificity and sensitivity add before they are squared.
            let d_sp = (g1 * (1.0 - pi1) - g0 * (1.0 - pi0)) / j1;
            let d_se = (-g1 * pi1 + g0 * pi0) / j1;
            d_sp * d_sp * rates.specificity_variance + d_se * d_se * rates.sensitivity_variance
        }
    };
    let log_or = logit(pi1) - logit(pi0);
    let recorded = |p: f64| if p > 0.0 && p < 1.0 { Some(logit(p)) } else { None };
    let naive = match (recorded(p1), recorded(p0)) {
        (Some(a), Some(b)) => Some(a - b),
        _ => None,
    };
    Ok(CorrectedExposureAssociation {
        observed_prevalence_cases: p1,
        observed_prevalence_controls: p0,
        prevalence_cases: pi1,
        prevalence_controls: pi0,
        naive_log_odds_ratio: naive,
        log_odds_ratio: log_or,
        odds_ratio: log_or.exp(),
        variance: sampling + from_rates,
        variance_sampling: sampling,
        variance_rates: from_rates,
        youden_cases: j1,
        youden_controls: j0,
        differential,
    })
}

/// Outcome-specific rates estimated from independent validation samples against a
/// reference standard, one sample per outcome level.
///
/// Each argument is `(true_positive, false_negative, true_negative, false_positive)`.
///
/// # Errors
/// An empty reference class in either sample refuses.
pub fn exposure_rates_from_validation(
    controls: (u64, u64, u64, u64),
    cases: (u64, u64, u64, u64),
) -> Result<ExposureErrorStructure, MeasurementErrorError> {
    let rates =
        |c: (u64, u64, u64, u64)| BinaryErrorRates::from_validation_counts(c.0, c.1, c.2, c.3);
    Ok(ExposureErrorStructure::ByOutcome { controls: rates(controls)?, cases: rates(cases)? })
}
