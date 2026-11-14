//! Wire section of a sequentially randomized unit-period design or regime survival result.
//!
//! The section is optional on the result body and absent on every other result, so their bodies
//! and digests are unchanged. A `TemporalDesign` query requires it. A loader checks that the
//! section's design is the query's, that a claimed support is the one its graphless support row
//! grants, that it claims no calibration and publishes no scalar interval beside it, that every
//! coordinate array is aligned and finite, that each
//! pointwise interval is ordered, that a band names coordinates in range and contains their
//! pointwise intervals, and that Holm-adjusted p-values are probabilities.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use serde::{Deserialize, Serialize};

use super::AnalysisResultWire;
use crate::temporal_design_query_wire::{TemporalAssignmentWire, TemporalDesignQueryWire};
use crate::{CausalQueryWire, IoError};

/// Relative slack of the covariance's symmetry and Cauchy-Schwarz checks.
const COVARIANCE_SLACK: f64 = 1e-9;

/// A temporal-design result with its uncertainty (calibration `not_measured`).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TemporalDesignWire {
    /// Design name.
    pub design: String,
    /// Outcome functional.
    pub functional: String,
    /// Coordinate names.
    pub coordinates: Vec<String>,
    /// Estimate (frequentist) or posterior mean (Bayesian).
    pub estimate: Vec<f64>,
    /// Standard error or posterior standard deviation.
    pub standard_error: Vec<f64>,
    /// Pointwise lower endpoints.
    pub lower: Vec<f64>,
    /// Pointwise upper endpoints.
    pub upper: Vec<f64>,
    /// Row-major covariance across coordinates.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub covariance: Option<Vec<f64>>,
    /// Coordinates a simultaneous band covers.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub band_coordinates: Vec<u64>,
    /// Critical value of the band.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub band_critical_value: Option<f64>,
    /// Lower band envelope.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub band_lower: Option<Vec<f64>>,
    /// Upper band envelope.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub band_upper: Option<Vec<f64>>,
    /// Holm-adjusted p-values of the contrast family.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub holm_adjusted_p: Option<Vec<f64>>,
    /// Worst-case smoothing bias per coordinate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub smoothing_bias_bound: Option<Vec<f64>>,
    /// Decision-outcome pairs (person-periods for regime survival).
    pub pairs: u64,
    /// Independent units or clusters.
    pub clusters: u64,
    /// Decisions whose outcome period was not observed.
    pub unpaired: u64,
    /// Smallest realized assignment probability, when the design reports one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub minimum_probability: Option<f64>,
    /// Successful and failed whole-procedure bootstrap replicates.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub replicates: Option<[u64; 2]>,
    /// Posterior draws.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub draws: Option<u64>,
    /// Uncertainty construction.
    pub uncertainty: String,
    /// Always `not_measured`.
    pub calibration: String,
    /// Whether the fit matched a row of the graphless support table.
    pub supported: bool,
}

/// Check a temporal-design section against its query.
pub(in crate::analysis_result_artifact) fn validate_temporal_design(
    result: &AnalysisResultWire,
) -> Result<(), IoError> {
    let fail = |message: &str| Err(IoError::Convert(format!("temporal design section: {message}")));
    let query = match &result.query {
        CausalQueryWire::TemporalDesign(query) => Some(query),
        _ => None,
    };
    let (query, section) = match (query, &result.temporal_design) {
        (None, None) => return Ok(()),
        (Some(_), None) => return fail("a temporal-design artifact is missing its section"),
        (None, Some(_)) => return fail("the section is attached to another query"),
        (Some(query), Some(section)) => (query, section),
    };
    if section.design != query.design_name() {
        return fail("the design does not belong to the query");
    }
    // A stored support is recomputed from the query and the section; a section that states
    // none is accepted as written.
    if (section.supported && !design_row_admits(query, section))
        || section.calibration != "not_measured"
        || result.standard_error.is_some()
        || result.interval_lower.is_some()
        || result.interval_upper.is_some()
    {
        return fail("a temporal-design result claims no support, calibration or scalar interval");
    }
    let k = section.coordinates.len();
    if k == 0
        || [
            section.estimate.len(),
            section.standard_error.len(),
            section.lower.len(),
            section.upper.len(),
        ]
        .iter()
        .any(|len| *len != k)
        || section.covariance.as_ref().is_some_and(|values| values.len() != k * k)
        || section.smoothing_bias_bound.as_ref().is_some_and(|values| values.len() != k)
    {
        return fail("the coordinate arrays are not aligned");
    }
    if section
        .estimate
        .iter()
        .chain(&section.standard_error)
        .chain(&section.lower)
        .chain(&section.upper)
        .any(|value| !value.is_finite())
        || section.standard_error.iter().any(|value| *value < 0.0)
        || section.lower.iter().zip(&section.upper).any(|(low, high)| low > high)
    {
        return fail("a coordinate is not finite, or an interval is reversed");
    }
    match (&section.band_lower, &section.band_upper, section.band_critical_value) {
        (None, None, None) if section.band_coordinates.is_empty() => {}
        (Some(lower), Some(upper), Some(critical)) => {
            let n = section.band_coordinates.len();
            if n == 0 || lower.len() != n || upper.len() != n || !critical.is_finite() {
                return fail("the band is not aligned with its coordinates");
            }
            if lower.iter().chain(upper).any(|value| !value.is_finite())
                || section.band_coordinates.windows(2).any(|pair| pair[0] >= pair[1])
            {
                return fail("a band is not finite or its coordinates are not strictly ascending");
            }
            for (j, coordinate) in section.band_coordinates.iter().enumerate() {
                let Some(index) = usize::try_from(*coordinate).ok().filter(|index| *index < k)
                else {
                    return fail("a band coordinate is out of range");
                };
                let slack = 1e-9 * (1.0 + section.lower[index].abs() + section.upper[index].abs());
                if lower[j] > section.lower[index] + slack
                    || upper[j] < section.upper[index] - slack
                {
                    return fail("a band does not contain its pointwise interval");
                }
            }
        }
        _ => return fail("the band fields are incomplete"),
    }
    check_linked_fields(query, section, k)?;
    if section
        .holm_adjusted_p
        .as_ref()
        .is_some_and(|values| values.iter().any(|value| !(0.0..=1.0).contains(value)))
    {
        return fail("a Holm-adjusted p-value is not a probability");
    }
    if section.replicates.is_some() && section.draws.is_some() {
        return fail("a result is either a bootstrap or a posterior, not both");
    }
    Ok(())
}

/// Fewest posterior draws a posterior row is stated for (`DESIGN_ROW_REFIT_DRAWS` for the dose
/// grid and regime survival), fewest completed bootstrap replicates of a percentile row, and
/// the longest regime-survival horizon: the measured settings.
const DESIGN_ROW_DRAWS: u64 = 1000;
const DESIGN_ROW_REFIT_DRAWS: u64 = 500;
const DESIGN_ROW_REPLICATES: u64 = 99;
const DESIGN_ROW_HORIZON: u32 = 3;
/// Fewest validation units per true-outcome margin, and the smallest validated Youden index,
/// a misclassification-corrected row is stated for: the measured validation conditions.
const DESIGN_ROW_VALIDATION: u64 = 100;
const DESIGN_ROW_YOUDEN: f64 = 0.6;

/// Whether the section is the misclassification-corrected two-arm mean inside the measured
/// validation conditions; the stored counts are true positive, false negative, true negative,
/// false positive.
#[allow(clippy::cast_precision_loss, reason = "validation counts are far below 2^53")]
fn design_row_validated_correction(query: &TemporalDesignQueryWire, functional: &str) -> bool {
    let Some([true_positive, false_negative, true_negative, false_positive]) =
        query.outcome_misclassification
    else {
        return false;
    };
    let positives = true_positive.saturating_add(false_negative);
    let negatives = true_negative.saturating_add(false_positive);
    if !matches!(query.assignment, TemporalAssignmentWire::TwoArm { .. })
        || query.quantile_level.is_some()
        || positives < DESIGN_ROW_VALIDATION
        || negatives < DESIGN_ROW_VALIDATION
    {
        return false;
    }
    let youden =
        true_positive as f64 / positives as f64 + true_negative as f64 / negatives as f64 - 1.0;
    let corrected = "mean; outcome misclassification corrected";
    youden >= DESIGN_ROW_YOUDEN
        && functional.strip_prefix(corrected).is_some_and(|rest| {
            rest.is_empty() || (rest.starts_with(" (Youden index ") && rest.ends_with(')'))
        })
}

/// Design name of the graphless support key: the route manifest's, which splits the quantile
/// functional and the dose targets. An observational design has no key, and neither has a
/// setting outside the measured ones: a quantile level other than the median, a
/// local-quadratic derivative or a positive curvature bound, or a longer survival horizon.
#[allow(
    clippy::float_cmp,
    reason = "the measured settings are the exact declared values 0.5 and 0.0"
)]
fn design_row_name(query: &TemporalDesignQueryWire) -> Option<&'static str> {
    use crate::temporal_design_query_wire::TemporalDoseTargetWire;
    if let Some(level) = query.quantile_level {
        let two_arm = matches!(query.assignment, TemporalAssignmentWire::TwoArm { .. });
        return (two_arm && level == 0.5).then_some("quantile");
    }
    match &query.assignment {
        TemporalAssignmentWire::TwoArm { .. } => Some("two_arm"),
        TemporalAssignmentWire::MultiArm { categorical: true, .. } => Some("categorical"),
        TemporalAssignmentWire::MultiArm { .. } => Some("multi_arm"),
        TemporalAssignmentWire::Factorial { .. } => Some("factorial"),
        TemporalAssignmentWire::Interference { .. } => Some("interference"),
        TemporalAssignmentWire::Dose { target, .. } => match target {
            TemporalDoseTargetWire::SmoothedContrast { .. } => Some("continuous_dose"),
            TemporalDoseTargetWire::SmoothedCurve { .. } => Some("smoothed_dose"),
            TemporalDoseTargetWire::LocalQuadratic { derivative, curvature_bound, .. } => {
                (!*derivative && *curvature_bound == Some(0.0)).then_some("dose_grid")
            }
        },
        TemporalAssignmentWire::RegimeSurvival { horizon, .. } => {
            (*horizon <= DESIGN_ROW_HORIZON).then_some("regime_survival")
        }
        TemporalAssignmentWire::ObservationalExcursion { .. }
        | TemporalAssignmentWire::ObservationalRegime { .. } => None,
    }
}

/// Positivity the section does not carry, read from the query: the smallest realized arm
/// probability of a two-arm design, and for a dose design the expected number of analysed
/// decisions within one bandwidth of dose at the smallest recorded density.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "a saturating count of expected decisions"
)]
fn design_row_positivity(query: &TemporalDesignQueryWire, pairs: u64) -> (f64, usize) {
    match &query.assignment {
        TemporalAssignmentWire::TwoArm { assignment, probability } => {
            let realized = assignment
                .iter()
                .zip(probability)
                .map(|(treated, p)| if *treated { *p } else { 1.0 - *p })
                .fold(1.0_f64, f64::min);
            (realized, 0)
        }
        TemporalAssignmentWire::Dose { density, bandwidth, .. } => {
            let thinnest = density.iter().copied().fold(f64::INFINITY, f64::min);
            (0.0, (pairs as f64 * thinnest * *bandwidth).floor() as usize)
        }
        _ => (0.0, 0),
    }
}

/// Whether the section's design, method and observed support match a supported row of the
/// graphless table. Recomputed here from the stored query and section, never taken from the
/// producer.
fn design_row_admits(query: &TemporalDesignQueryWire, section: &TemporalDesignWire) -> bool {
    let Some(design) = design_row_name(query) else {
        return false;
    };
    // The claim names the kind of pointwise 95% interval the section publishes; a posterior
    // or a bootstrap smaller than the measured ones has no row.
    let inference_claim = if let Some(draws) = section.draws {
        let floor = if matches!(design, "dose_grid" | "regime_survival") {
            DESIGN_ROW_REFIT_DRAWS
        } else {
            DESIGN_ROW_DRAWS
        };
        if draws < floor {
            return false;
        }
        "pointwise_95_posterior_quantiles"
    } else if let Some([completed, _]) = section.replicates {
        if completed < DESIGN_ROW_REPLICATES {
            return false;
        }
        "pointwise_95_percentile_interval"
    } else {
        "pointwise_95_normal_interval"
    };
    // A weighted or misclassification-corrected arm functional is no longer the plain mean,
    // and a dropped pair leaves the counts short of the rows.
    let arm_mean = query.quantile_level.is_none()
        && matches!(
            query.assignment,
            TemporalAssignmentWire::TwoArm { .. }
                | TemporalAssignmentWire::MultiArm { .. }
                | TemporalAssignmentWire::Factorial { .. }
                | TemporalAssignmentWire::Interference { .. }
        );
    if (arm_mean
        && section.functional != "mean"
        && !design_row_validated_correction(query, &section.functional))
        || section.pairs.checked_add(section.unpaired) != u64::try_from(query.units.len()).ok()
    {
        return false;
    }
    let published = section
        .lower
        .iter()
        .zip(&section.upper)
        .filter(|(lower, upper)| lower.is_finite() && upper.is_finite() && lower < upper)
        .count();
    let all_published = published > 0 && published == section.coordinates.len();
    let (declared_probability, local_rows) = design_row_positivity(query, section.pairs);
    crate::graphless_support::graphless_admits(
        crate::graphless_support::GraphlessSupportKey {
            family: "temporal_design",
            design,
            method: section.uncertainty.as_str(),
            inference_claim,
        },
        crate::graphless_support::GraphlessAssignmentSupport {
            assignment_unit: "sequence",
            rows: usize::try_from(section.pairs).unwrap_or(usize::MAX),
            blocks: usize::try_from(section.clusters).unwrap_or(usize::MAX),
            min_probability: section.minimum_probability.unwrap_or(declared_probability),
            min_local_rows: local_rows,
            interval_95_published: all_published,
            reported_intervals: published,
            all_reported_intervals: all_published,
            ..Default::default()
        },
    )
}

/// Check the section's fields against the query that produced it and against each other: the
/// functional names the query's functional, no cluster or unpaired count exceeds the table the
/// query carries, the assignment floor is a probability, the covariance is a symmetric covariance matrix, the
/// smoothing bound is a finite magnitude and a posterior has draws.
fn check_linked_fields(
    query: &TemporalDesignQueryWire,
    section: &TemporalDesignWire,
    k: usize,
) -> Result<(), IoError> {
    let fail = |message: &str| Err(IoError::Convert(format!("temporal design section: {message}")));
    let functional = section.functional.as_str();
    let functional_ok = match &query.assignment {
        TemporalAssignmentWire::RegimeSurvival { .. } => functional.starts_with("survival"),
        TemporalAssignmentWire::TwoArm { .. }
        | TemporalAssignmentWire::MultiArm { .. }
        | TemporalAssignmentWire::Factorial { .. }
        | TemporalAssignmentWire::Interference { .. } => {
            functional.starts_with("quantile:") == query.quantile_level.is_some()
                && !functional.trim().is_empty()
        }
        TemporalAssignmentWire::Dose { .. }
        | TemporalAssignmentWire::ObservationalExcursion { .. }
        | TemporalAssignmentWire::ObservationalRegime { .. } => !functional.trim().is_empty(),
    };
    if !functional_ok {
        return fail("the functional is not the query's");
    }
    let rows = query.units.len() as u64;
    if section.clusters > rows || section.unpaired > rows {
        return fail("a count exceeds the rows of the query");
    }
    if section
        .minimum_probability
        .is_some_and(|floor| !(0.0..=1.0).contains(&floor) || floor == 0.0)
    {
        return fail("the minimum assignment probability is not in (0, 1]");
    }
    if section.draws == Some(0) {
        return fail("a posterior has draws");
    }
    if section
        .smoothing_bias_bound
        .as_ref()
        .is_some_and(|values| values.iter().any(|value| !value.is_finite() || *value < 0.0))
    {
        return fail("a smoothing bias bound is not a finite magnitude");
    }
    if let Some(covariance) = &section.covariance {
        let at = |i: usize, j: usize| covariance[i * k + j];
        let invalid = covariance.iter().any(|value| !value.is_finite())
            || (0..k).any(|i| {
                at(i, i) < 0.0
                    || (0..i).any(|j| {
                        let (a, b) = (at(i, j), at(j, i));
                        (a - b).abs() > COVARIANCE_SLACK * (1.0 + a.abs() + b.abs())
                            || a * a
                                > at(i, i) * at(j, j) * (1.0 + COVARIANCE_SLACK) + f64::MIN_POSITIVE
                    })
            });
        if invalid {
            return fail("the covariance is not a finite symmetric covariance matrix");
        }
    }
    Ok(())
}
