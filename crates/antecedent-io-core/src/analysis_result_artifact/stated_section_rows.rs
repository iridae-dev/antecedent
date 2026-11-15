//! Support rows a result states beside a design section, and the two constructions of the
//! history-match period interval.
//!
//! A section of an association, sharp instrument bounds, a sequential contrast, a history
//! match, a temporal design or an augmented panel difference in differences keeps its own
//! `supported` field `false`. A fit that meets its graphless support row says so in a coded
//! diagnostic (the codes in [`crate::graphless_support`]). The loader recomputes the row from
//! the stored query and section for every such diagnostic and refuses a statement the section
//! does not meet, so a statement is never taken from the producer.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    AnalysisResultWire, AssociationWire, HistoryMatchWire, IV_BOUNDS_BAYESIAN_METHOD,
    IV_BOUNDS_FREQUENTIST_METHOD, IvBoundsWire, PanelDidWire, SequentialContrastWire,
    TemporalDesignWire,
};
use crate::graphless_support::{
    ASSOCIATION_SLOPE_ROW_CODE, AUGMENTED_DID_ROW_CODE, CROSSFIT_CONTRAST_ROW_CODE,
    DESIGN_POSTERIOR_ROW_CODE, GraphlessAssignmentSupport, GraphlessResampleSupport,
    GraphlessSupportKey, IV_BOUNDS_ROW_CODE, PERIOD_HISTORY_ROW_CODE, TEMPORAL_DESIGN_ROW_CODE,
    graphless_admits, graphless_admits_resampled,
};
use crate::replay_tolerance::{ReplayNonFinite, ReplayScale, ReplayTolerance};
use crate::temporal_design_query_wire::{TemporalAssignmentWire, TemporalDesignQueryWire};
use crate::{CausalQueryWire, IoError};

/// Whether `result` carries a diagnostic with `code`.
#[must_use]
pub fn section_row_stated(result: &AnalysisResultWire, code: &str) -> bool {
    result.diagnostics.iter().any(|diagnostic| diagnostic.code == code)
}

/// Refuse a support row a result states that its stored query and section do not meet.
///
/// A result that states no row is accepted as written. A sequential-contrast posterior states
/// its row in the design-posterior diagnostic, whose stated row the generic check admits; here
/// the design, the rows and the smallest fitted probability it states must be the section's.
pub(super) fn validate_stated_section_rows(result: &AnalysisResultWire) -> Result<(), IoError> {
    let refuse = |family: &str| {
        Err(IoError::Convert(format!(
            "{family} states a graphless support row its query and section do not meet"
        )))
    };
    if section_row_stated(result, ASSOCIATION_SLOPE_ROW_CODE)
        && !matches!(
            (&result.query, &result.association),
            (CausalQueryWire::Association(query), Some(section))
                if association_slope_row_admits(query, section)
        )
    {
        return refuse("an association");
    }
    if section_row_stated(result, IV_BOUNDS_ROW_CODE)
        && !matches!(
            (&result.query, &result.iv_bounds),
            (CausalQueryWire::IvBounds(query), Some(section))
                if bounds_endpoint_row_admits(query, section)
        )
    {
        return refuse("a bounds result");
    }
    if let (CausalQueryWire::SequentialContrast(query), Some(fit)) =
        (&result.query, &result.sequential_contrast)
    {
        if section_row_stated(result, CROSSFIT_CONTRAST_ROW_CODE)
            && !sequential_contrast_wald_row_admits(result, query, fit)
        {
            return refuse("a sequential contrast");
        }
        if !sequential_contrast_posterior_row_is_own(result, query, fit) {
            return refuse("a sequential contrast posterior");
        }
    } else if section_row_stated(result, CROSSFIT_CONTRAST_ROW_CODE) {
        return refuse("a sequential contrast");
    }
    if section_row_stated(result, PERIOD_HISTORY_ROW_CODE)
        && !matches!(
            (&result.query, &result.history_match),
            (CausalQueryWire::HistoryMatch(query), Some(fit))
                if history_match_row_admits(query, fit)
        )
    {
        return refuse("a history match");
    }
    if section_row_stated(result, TEMPORAL_DESIGN_ROW_CODE)
        && !matches!(
            (&result.query, &result.temporal_design),
            (CausalQueryWire::TemporalDesign(query), Some(section))
                if design_row_admits(query, section)
        )
    {
        return refuse("a temporal design");
    }
    if section_row_stated(result, AUGMENTED_DID_ROW_CODE)
        && !matches!(
            (&result.query, &result.panel_did),
            (CausalQueryWire::PanelDid(query), Some(section))
                if augmented_panel_row_admits(query, section)
        )
    {
        return refuse("an augmented panel difference in differences");
    }
    Ok(())
}

/// Whether the section's slope interval, rows and resamples match a supported row of the
/// graphless table. Recomputed from the stored query and section, never taken from the
/// producer. Rows and resamples are thresholds of the row; a linear regression of a real
/// outcome on one numeric predictor, nothing held fixed, unweighted, no dropped row, the 95%
/// level, every requested replicate fitted and a conjugate prior that moves the slope by at
/// most a hundredth of its posterior standard deviation are checked here.
fn association_slope_row_admits(
    query: &crate::AssociationQueryWire,
    section: &AssociationWire,
) -> bool {
    if section.analysis != "linear_regression"
        || section.outcome_kind != "continuous"
        || query.predictors.len() != 1
        || !query.conditioning.is_empty()
        || query.weight.is_some()
        || section.rows_dropped != 0
        || section.terms.len() != 2
        || section.terms[0].label != "intercept"
        || section.terms[1].label.contains('=')
        || section.terms[1].scale != "outcome_units"
    {
        return false;
    }
    let slope = &section.terms[1];
    // One series is one dependent sequence; table rows are independent units.
    let (design, assignment_unit, blocks) = match section.modality.as_str() {
        "cross_section" => ("cross_section_linear_slope", "unit", 0),
        "series" => ("series_linear_slope", "sequence", 1),
        _ => return false,
    };
    let at_95 = |level: f64| (level - 0.95).abs() <= 1e-12;
    let (method, inference_claim, band, resamples) = if let Some(record) = &section.posterior {
        let (Some(lower), Some(upper)) = (slope.posterior_lower, slope.posterior_upper) else {
            return false;
        };
        let moved = slope
            .posterior_mean
            .zip(slope.posterior_sd)
            .is_none_or(|(mean, sd)| (mean - slope.estimate).abs() > 0.01 * sd);
        if !at_95(record.level) || (record.method == "conjugate_normal_inverse_gamma" && moved) {
            return false;
        }
        (
            record.method.as_str(),
            "slope_95_posterior_equal_tailed_interval",
            (lower, upper),
            record.draws,
        )
    } else if let Some(record) = &section.resampling {
        let (Some(lower), Some(upper)) = (slope.spread_lower, slope.spread_upper) else {
            return false;
        };
        let (method, inference_claim) = match record.unit.as_str() {
            "row" => ("bootstrap_row", "slope_95_studentized_spread"),
            "calendar_block" => ("bootstrap_circular_block", "slope_95_fixed_b_percentile_spread"),
            _ => return false,
        };
        if !at_95(record.spread_level)
            || record.replicates_ok != record.replicates_requested
            || slope.standard_error_method.as_deref() != Some(method)
        {
            return false;
        }
        (method, inference_claim, (lower, upper), record.replicates_ok)
    } else {
        let (Some(error), Some(method)) =
            (slope.standard_error, slope.standard_error_method.as_deref())
        else {
            return false;
        };
        (
            method,
            "slope_standard_error_95_normal_reference_interval",
            (slope.estimate - error, slope.estimate + error),
            0,
        )
    };
    graphless_admits_resampled(
        GraphlessSupportKey { family: "association", design, method, inference_claim },
        GraphlessAssignmentSupport {
            assignment_unit,
            rows: usize::try_from(section.rows).unwrap_or(usize::MAX),
            blocks,
            interval_95_published: band.0.is_finite() && band.1.is_finite() && band.0 < band.1,
            reported_intervals: 1,
            all_reported_intervals: true,
            ..Default::default()
        },
        GraphlessResampleSupport {
            interval_resamples: usize::try_from(resamples).unwrap_or(usize::MAX),
            ..Default::default()
        },
    )
}

/// Whether the section's interval, rows and resamples match a supported row of the graphless
/// table. Recomputed from the stored query and section, never taken from the producer. Rows
/// and resamples are thresholds of the row; unweighted rows, no dropped row, the 95% level,
/// no resample left out for an infeasible law and a retained width are checked here.
fn bounds_endpoint_row_admits(query: &crate::IvBoundsQueryWire, section: &IvBoundsWire) -> bool {
    let Some(interval) = &section.interval else {
        return false;
    };
    if query.weight.is_some()
        || section.rows_dropped != 0
        || interval.infeasible != 0
        || !interval.width_retained
        || (interval.level - 0.95).abs() > 1e-12
        || section.uncertainty != interval.method
    {
        return false;
    }
    let inference_claim = match interval.method.as_str() {
        IV_BOUNDS_FREQUENTIST_METHOD => "identified_set_endpoints_95_imbens_manski_interval",
        IV_BOUNDS_BAYESIAN_METHOD => "identified_set_endpoints_95_posterior_quantiles",
        _ => return false,
    };
    graphless_admits_resampled(
        GraphlessSupportKey {
            family: "iv_bounds",
            design: "binary_instrument",
            method: interval.method.as_str(),
            inference_claim,
        },
        GraphlessAssignmentSupport {
            assignment_unit: "unit",
            rows: usize::try_from(section.rows).unwrap_or(usize::MAX),
            interval_95_published: interval.lower.is_finite()
                && interval.upper.is_finite()
                && interval.lower < interval.upper,
            reported_intervals: 1,
            all_reported_intervals: true,
            ..Default::default()
        },
        GraphlessResampleSupport {
            interval_resamples: usize::try_from(interval.replicates).unwrap_or(usize::MAX),
            ..Default::default()
        },
    )
}

/// Whether a sequential contrast query is the design its support rows were measured on: the
/// always-treat against never-treat contrast over at least five folds, none above a fifth of
/// the rows (rounded up).
fn sequential_contrast_row_design(query: &crate::query_wire::SequentialContrastQueryWire) -> bool {
    let rows = query.fold_of_row.len();
    let mut sizes = std::collections::BTreeMap::<u32, usize>::new();
    for fold in &query.fold_of_row {
        *sizes.entry(*fold).or_default() += 1;
    }
    let largest = sizes.values().copied().max().unwrap_or(rows);
    query.active == [1, 1]
        && query.control == [0, 0]
        && sizes.len() >= 5
        && largest.saturating_mul(5) <= rows.saturating_add(4)
}

/// Whether a sequential contrast that states the Wald row meets it: the measured design, no
/// posterior, a published interval and the row's thresholds.
fn sequential_contrast_wald_row_admits(
    result: &AnalysisResultWire,
    query: &crate::query_wire::SequentialContrastQueryWire,
    fit: &SequentialContrastWire,
) -> bool {
    sequential_contrast_row_design(query)
        && result.posterior_artifact.is_none()
        && fit.interval.is_some()
        && graphless_admits(
            GraphlessSupportKey {
                family: "sequential_contrast",
                design: "two_decision_always_versus_never",
                method: "crossfit_augmented_scores_wald",
                inference_claim: "pointwise_95_normal_interval",
            },
            GraphlessAssignmentSupport {
                assignment_unit: "unit",
                rows: fit.contrast_scores.len(),
                min_probability: fit.minimum_action_probability,
                interval_95_published: true,
                reported_intervals: 1,
                all_reported_intervals: true,
                ..Default::default()
            },
        )
}

/// Whether every design-posterior row a sequential contrast states is its own: the measured
/// design, the family, the section's rows and a probability floor the section's fitted
/// probabilities meet. True when none is stated.
fn sequential_contrast_posterior_row_is_own(
    result: &AnalysisResultWire,
    query: &crate::query_wire::SequentialContrastQueryWire,
    fit: &SequentialContrastWire,
) -> bool {
    let design = sequential_contrast_row_design(query);
    let rows = fit.contrast_scores.len();
    result.diagnostics.iter().filter(|diagnostic| diagnostic.code == DESIGN_POSTERIOR_ROW_CODE).all(
        |diagnostic| {
            let stated = |name: &str| {
                diagnostic
                    .fields
                    .iter()
                    .find(|(field, _)| field == name)
                    .map(|(_, value)| value.as_str())
            };
            design
                && stated("family") == Some("sequential_contrast")
                && stated("rows").and_then(|value| value.parse::<usize>().ok()) == Some(rows)
                && stated("min_probability")
                    .and_then(|value| value.parse::<f64>().ok())
                    .is_some_and(|value| value <= fit.minimum_action_probability)
        },
    )
}

/// Whether a history match that claims its support row meets it: calendar periods declared, no
/// caliper, no sampling weights, every treated unit matched, a published period interval, and
/// the row's matched periods and pairs in each.
fn history_match_row_admits(
    query: &crate::query_wire::HistoryMatchQueryWire,
    fit: &HistoryMatchWire,
) -> bool {
    let mut pairs_in_period = std::collections::BTreeMap::<i64, usize>::new();
    for period in &fit.pair_periods {
        *pairs_in_period.entry(*period).or_default() += 1;
    }
    let published = fit.interval_lower.is_some() && fit.interval_upper.is_some();
    fit.uncertainty == HISTORY_MATCH_FIXED_B_UNCERTAINTY
        && query.period.is_some()
        && query.caliper.is_none()
        && query.weight.is_none()
        && fit.caliper.is_none()
        && fit.unmatched_treated.is_empty()
        && graphless_admits(
            GraphlessSupportKey {
                family: "history_match",
                design: "within_period_nearest_history",
                method: "period_newey_west_fixed_b",
                inference_claim: "pointwise_95_fixed_b_interval",
            },
            GraphlessAssignmentSupport {
                assignment_unit: "cluster",
                rows: fit.matched_queries.len(),
                blocks: pairs_in_period.len(),
                min_block_arm: pairs_in_period.values().copied().min().unwrap_or(0),
                interval_95_published: published,
                reported_intervals: usize::from(published),
                all_reported_intervals: published,
                ..Default::default()
            },
        )
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
    graphless_admits(
        GraphlessSupportKey {
            family: "temporal_design",
            design,
            method: section.uncertainty.as_str(),
            inference_claim,
        },
        GraphlessAssignmentSupport {
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

/// Uncertainty label of the augmented panel's whole-cluster resample interval.
const AUGMENTED_ROW_UNCERTAINTY: &str =
    "whole_cluster_resample_supplied_nuisance_held_fixed_calibration_not_measured";

/// Whether an augmented panel section that states its row meets it: the whole-cluster
/// bootstrap interval of a declared resample, every subject its own cluster, the declared
/// refits, and the supplied propensity range the section records. The interval itself is
/// replayed by the section's own check.
fn augmented_panel_row_admits(
    query: &crate::query_wire::PanelDidQueryWire,
    did: &PanelDidWire,
) -> bool {
    let mut treated_of = std::collections::BTreeMap::<&str, bool>::new();
    for (subject, treated) in query.subjects.iter().zip(&query.treated) {
        treated_of.insert(subject.as_str(), *treated);
    }
    let subjects = treated_of.len();
    let treated = treated_of.values().filter(|treated| **treated).count();
    let room = did.augmented.map_or(0.0, |(lowest, highest, _, _)| lowest.min(1.0 - highest));
    query.augmented.is_some()
        && did.interval_95.is_some()
        && did.uncertainty == AUGMENTED_ROW_UNCERTAINTY
        && did.treated_subjects == treated
        && did.clusters == subjects
        && query.resampling.as_ref().is_some_and(|resampling| {
            resampling.scheme == "cluster_bootstrap"
                && graphless_admits_resampled(
                    GraphlessSupportKey {
                        family: "difference_in_differences",
                        design: "augmented_panel",
                        method: "whole_cluster_bootstrap_supplied_nuisance_held_fixed",
                        inference_claim: "pointwise_95_normal_interval",
                    },
                    GraphlessAssignmentSupport {
                        assignment_unit: "cluster",
                        treated,
                        control: subjects - treated,
                        interval_95_published: true,
                        rows: subjects,
                        blocks: subjects,
                        min_probability: room,
                        reported_intervals: 1,
                        ..GraphlessAssignmentSupport::default()
                    },
                    GraphlessResampleSupport {
                        interval_resamples: resampling.replicates as usize,
                        ..GraphlessResampleSupport::default()
                    },
                )
        })
}

/// Uncertainty label of a history match whose period interval is the fixed-b interval: the
/// Newey-West standard error with the Kiefer-Vogelsang critical value at the bandwidth's share
/// of the matched periods.
pub const HISTORY_MATCH_FIXED_B_UNCERTAINTY: &str =
    "abadie_imbens_and_newey_west_fixed_b_period_interval_calibration_not_measured";

/// Uncertainty label of a history match whose period interval is the normal interval of the
/// same standard error: the construction of artifacts written before the fixed-b interval.
const HISTORY_MATCH_NORMAL_UNCERTAINTY: &str =
    "abadie_imbens_and_newey_west_period_interval_calibration_not_measured";

/// Replay policy of the fixed-b endpoints against the section's own standard error.
const FIXED_B_REPLAY: ReplayTolerance =
    ReplayTolerance::new(1e-9, ReplayScale::OnePlusReplayed, ReplayNonFinite::Refuse);

/// Check a history-match result under the construction its uncertainty label names.
///
/// A section with the earlier label is checked by the earlier check unchanged: its interval is
/// the normal interval of the Newey-West standard error. A section with the fixed-b label
/// carries `contrast -+ z * cv(b) / 1.96 * se`, `cv(b) = 1.96 + 2.9694 b + 0.4160 b^2 -
/// 0.5324 b^3` at the bandwidth share `b = (lag + 1) / T`; that interval is recomputed here
/// from the section's standard error, lag and matched periods, and everything else (the
/// pairs, the contrast, both standard errors and the lag) is checked by the earlier check on
/// the same section restated with the normal interval of that standard error. The
/// outcome-model posterior of the same query carries no matched section. `earlier_check` is
/// the earlier check of the section.
#[allow(clippy::cast_precision_loss, reason = "period counts are far below 2^53")]
pub(super) fn validate_history_match_section(
    result: &AnalysisResultWire,
    earlier_check: impl Fn(&AnalysisResultWire) -> Result<(), IoError>,
) -> Result<(), IoError> {
    let Some(fit) = &result.history_match else {
        return if matches!(result.query, CausalQueryWire::HistoryMatch(_))
            && result.posterior_artifact.is_some()
        {
            Ok(())
        } else {
            earlier_check(result)
        };
    };
    if fit.uncertainty != HISTORY_MATCH_FIXED_B_UNCERTAINTY {
        return earlier_check(result);
    }
    let mut earlier = result.clone();
    let Some(restated) = earlier.history_match.as_mut() else {
        return earlier_check(result);
    };
    HISTORY_MATCH_NORMAL_UNCERTAINTY.clone_into(&mut restated.uncertainty);
    let stated = (fit.serial_standard_error, fit.interval_lower, fit.interval_upper);
    if let (Some(standard_error), Some(_), Some(_)) = stated {
        let normal = antecedent_stats::NORMAL_Q975 * standard_error;
        restated.interval_lower = Some(fit.contrast - normal);
        restated.interval_upper = Some(fit.contrast + normal);
    }
    // The summary is checked first, so a fabricated one is named as such.
    earlier_check(&earlier)?;
    if let (Some(standard_error), Some(lower), Some(upper)) = stated {
        let mut periods = fit.pair_periods.clone();
        periods.sort_unstable();
        periods.dedup();
        let share =
            ((fit.serial_bandwidth + 1) as f64 / periods.len().max(1) as f64).clamp(0.0, 1.0);
        let fixed_b = (1.96 + 2.9694 * share + 0.4160 * share * share
            - 0.5324 * share * share * share)
            / 1.96;
        let critical = antecedent_stats::NORMAL_Q975 * fixed_b;
        if !(FIXED_B_REPLAY.close(lower, fit.contrast - critical * standard_error)
            && FIXED_B_REPLAY.close(upper, fit.contrast + critical * standard_error))
        {
            return Err(IoError::Convert(
                "history match period interval is not the fixed-b Newey-West interval of its \
                 pairs"
                    .into(),
            ));
        }
    }
    Ok(())
}
