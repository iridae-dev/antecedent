//! Support rows of the design sections: an association, sharp instrument bounds, a history
//! match, a sequential contrast and a temporal design.
//!
//! Each predicate reads the row key and the observed support from the query and the published
//! estimate alone, so the artifact check recomputes the same answer from the stored section.
//! A fit that meets its row states it in a coded diagnostic (the codes of
//! [`antecedent_io::graphless_support`]); the section's own `supported` field stays `false`.
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::result::StudyResult;
use antecedent_core::{
    AssociationQuery, Diagnostic, DiagnosticKind, DiagnosticSeverity, TemporalAssignment,
    TemporalDesignFunctional, TemporalDesignQuery, TemporalDoseTarget,
};
use antecedent_estimate::AssociationEstimate;

impl StudyResult {
    /// Whether the result states the graphless support row its design section meets: the
    /// coded diagnostic of an association, sharp instrument bounds, a sequential contrast, a
    /// history match, a temporal design or an augmented panel difference in differences.
    #[must_use]
    pub fn states_section_row(&self) -> bool {
        self.diagnostics.iter().any(|diagnostic| {
            antecedent_io::graphless_support::SECTION_ROW_CODES.contains(&diagnostic.code.as_ref())
        })
    }
}

/// The diagnostic by which a fit states the row `code` names.
pub(crate) fn section_row_note(code: &'static str, message: String) -> Diagnostic {
    Diagnostic::new(code, DiagnosticKind::Scientific, DiagnosticSeverity::Info, message)
}

/// Whether the fit runs under a row of the graphless support table. The key and the observed
/// support are read from the query and the published estimate alone, so the artifact check
/// recomputes the same answer. Rows and resamples are thresholds of the row; the measured
/// conditions the table has no field for are checked here: a linear regression of a real
/// outcome on one numeric predictor, nothing held fixed, unweighted, no row dropped, and every
/// requested replicate fitted, and a conjugate prior that moves the slope by at most a
/// hundredth of its posterior standard deviation. The claim is the one interval the slope term
/// publishes: its posterior interval, else its resampling spread, else its standard error.
pub(crate) fn association_slope_row(query: &AssociationQuery, fit: &AssociationEstimate) -> bool {
    if fit.selection.analysis.as_str() != "linear_regression"
        || fit.selection.outcome_kind.as_str() != "continuous"
        || query.predictors.len() != 1
        || !query.conditioning.is_empty()
        || query.weight.is_some()
        || fit.rows_dropped != 0
        || fit.terms.len() != 2
        || fit.terms[0].label != "intercept"
        || fit.terms[1].label.contains('=')
        || fit.terms[1].scale != "outcome_units"
    {
        return false;
    }
    let slope = &fit.terms[1];
    // One series is one dependent sequence; table rows are independent units.
    let (design, assignment_unit, blocks) = match fit.selection.modality.as_str() {
        "cross_section" => ("cross_section_linear_slope", "unit", 0),
        "series" => ("series_linear_slope", "sequence", 1),
        _ => return false,
    };
    let (method, inference_claim, band, resamples) = if let Some(record) = fit.posterior {
        let Some(summary) = slope.posterior else {
            return false;
        };
        // Measured with a prior the data swamp; a prior that moves the slope has no row.
        if record.method == "conjugate_normal_inverse_gamma"
            && (summary.mean - slope.estimate).abs() > 0.01 * summary.sd
        {
            return false;
        }
        (
            record.method,
            "slope_95_posterior_equal_tailed_interval",
            (summary.lower, summary.upper),
            record.draws,
        )
    } else if let Some(record) = fit.resampling {
        let Some(spread) = slope.spread else {
            return false;
        };
        let (method, inference_claim) = match record.unit {
            "row" => ("bootstrap_row", "slope_95_studentized_spread"),
            "calendar_block" => ("bootstrap_circular_block", "slope_95_fixed_b_percentile_spread"),
            _ => return false,
        };
        if record.replicates_ok != record.replicates_requested
            || slope.standard_error_method != Some(method)
        {
            return false;
        }
        (
            method,
            inference_claim,
            spread,
            usize::try_from(record.replicates_ok).unwrap_or(usize::MAX),
        )
    } else {
        let (Some(error), Some(method)) = (slope.standard_error, slope.standard_error_method)
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
    antecedent_io::graphless_support::graphless_admits_resampled(
        crate::support::GraphlessSupportKey {
            family: "association",
            design,
            method,
            inference_claim,
        },
        crate::support::GraphlessAssignmentSupport {
            assignment_unit,
            rows: fit.rows,
            blocks,
            interval_95_published: band.0.is_finite() && band.1.is_finite() && band.0 < band.1,
            reported_intervals: 1,
            all_reported_intervals: true,
            ..Default::default()
        },
        antecedent_io::graphless_support::GraphlessResampleSupport {
            interval_resamples: resamples,
            ..Default::default()
        },
    )
}

/// Prior scale the conjugate slope row was measured with: the built-in scale.
pub(crate) const CONJUGATE_SLOPE_ROW_PRIOR_SCALE: f64 = 10.0;

/// Whether a fit's prior is the one its association row was measured with. Only the conjugate
/// Normal-Inverse-Gamma posterior reads the prior scale: its row was measured at the built-in
/// scale [`CONJUGATE_SLOPE_ROW_PRIOR_SCALE`], and a fit at any other scale states no row, however
/// little that prior moves the slope. `prior_scale` is the scale of the study's Bayesian
/// inference, absent under frequentist inference. The stored result does not carry the scale,
/// so this condition is the executor's alone; the artifact check keeps the bound on the
/// prior's pull that [`association_slope_row`] also applies.
pub(crate) fn conjugate_slope_row_prior(
    prior_scale: Option<f64>,
    fit: &AssociationEstimate,
) -> bool {
    let conjugate =
        fit.posterior.is_some_and(|record| record.method == "conjugate_normal_inverse_gamma");
    !conjugate
        || prior_scale.is_some_and(|scale| (scale - CONJUGATE_SLOPE_ROW_PRIOR_SCALE).abs() <= 1e-9)
}

/// Whether the fit runs under a row of the graphless support table. The key and the observed
/// support are read from the query and the published estimate alone, so the artifact check
/// recomputes the same answer. Rows and resamples are thresholds of the row; the measured
/// conditions the table has no field for are checked here: unweighted rows, no row dropped,
/// the 95% level, no resample or draw left out for an infeasible law, and a set whose width
/// passed the moment-selection threshold (wider than its sampling noise).
pub(crate) fn bounds_endpoint_row(
    query: &antecedent_core::IvBoundsQuery,
    fit: &antecedent_estimate::IvBoundsEstimate,
) -> bool {
    let Some(interval) = fit.interval else {
        return false;
    };
    if query.weight.is_some()
        || fit.rows_dropped != 0
        || interval.infeasible != 0
        || !interval.width_retained
        || (interval.level - 0.95).abs() > 1e-12
    {
        return false;
    }
    // The claim is an interval for the endpoints of the identified set, not for a point effect.
    let inference_claim = match interval.method.as_str() {
        "imbens_manski_row_bootstrap" => "identified_set_endpoints_95_imbens_manski_interval",
        "posterior_endpoint_quantiles" => "identified_set_endpoints_95_posterior_quantiles",
        _ => return false,
    };
    antecedent_io::graphless_support::graphless_admits_resampled(
        crate::support::GraphlessSupportKey {
            family: "iv_bounds",
            design: "binary_instrument",
            method: interval.method.as_str(),
            inference_claim,
        },
        crate::support::GraphlessAssignmentSupport {
            assignment_unit: "unit",
            rows: fit.rows,
            interval_95_published: interval.lower.is_finite()
                && interval.upper.is_finite()
                && interval.lower < interval.upper,
            reported_intervals: 1,
            all_reported_intervals: true,
            ..Default::default()
        },
        antecedent_io::graphless_support::GraphlessResampleSupport {
            interval_resamples: interval.replicates,
            ..Default::default()
        },
    )
}

/// Code of the standing note of a matched contrast that states no support row.
const HISTORY_NO_ROW_CODE: &str = "estimate.history_match.unsupported";

/// Prior scale the outcome-model support row was measured with.
pub(crate) const OUTCOME_MODEL_ROW_PRIOR_SCALE: f64 = 10.0;

/// Whether the query is the design both support rows were measured on: calendar periods
/// declared, no caliper and no sampling weights.
pub(crate) fn period_history_row_design(query: &antecedent_core::HistoryMatchQuery) -> bool {
    query.period.is_some() && query.caliper.is_none() && query.weight.is_none()
}

/// Whether the matched contrast meets its support row: the measured design, every treated
/// unit matched, a published period interval, and the row's matched periods and pairs.
pub(crate) fn period_history_row_support(
    query: &antecedent_core::HistoryMatchQuery,
    estimate: &antecedent_estimate::history_match::PeriodHistoryEstimate,
) -> bool {
    let mut pairs_in_period = std::collections::BTreeMap::<i64, usize>::new();
    for period in &estimate.matched_periods {
        *pairs_in_period.entry(*period).or_default() += 1;
    }
    period_history_row_design(query)
        && estimate.unmatched_treated.is_empty()
        && crate::support::support_if_graphless(
            crate::support::GraphlessSupportKey {
                family: "history_match",
                design: "within_period_nearest_history",
                method: "period_newey_west_fixed_b",
                inference_claim: "pointwise_95_fixed_b_interval",
            },
            crate::support::GraphlessAssignmentSupport {
                assignment_unit: "cluster",
                rows: estimate.matched_queries.len(),
                blocks: pairs_in_period.len(),
                min_block_arm: pairs_in_period.values().copied().min().unwrap_or(0),
                interval_95_published: estimate.serial.is_some(),
                reported_intervals: usize::from(estimate.serial.is_some()),
                all_reported_intervals: estimate.serial.is_some(),
                ..Default::default()
            },
        )
        .is_some()
}

/// Replace the standing note of a supported matched contrast with one naming its row.
pub(crate) fn note_period_history_row(result: &mut StudyResult) {
    result.diagnostics.retain(|diagnostic| diagnostic.code.as_ref() != HISTORY_NO_ROW_CODE);
    result.diagnostics.push(Diagnostic::new(
        antecedent_io::graphless_support::PERIOD_HISTORY_ROW_CODE,
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        "supported graphless row history_match/within_period_nearest_history/\
         period_newey_west_fixed_b: the fixed-b period interval is for the expected matched \
         contrast with calendar periods as the sampling units; the mean history carrying the \
         confounding stays the caller's assumption",
    ));
}

/// Code of the standing note of a sequential contrast that states no support row.
pub(crate) const CONTRAST_NO_ROW_CODE: &str = "estimate.sequential_contrast.unsupported";

/// Whether the query is the design the support rows were measured on: the always-treat
/// against never-treat contrast, cross-fit over exactly five folds (the measured count) none
/// of which holds more than a fifth of the rows (rounded up), so every nuisance model trains on
/// four fifths.
pub(crate) fn crossfit_contrast_row_design(
    query: &antecedent_core::SequentialContrastQuery,
) -> bool {
    let rows = query.fold_of_row.len();
    let mut sizes = std::collections::BTreeMap::<u32, usize>::new();
    for fold in &*query.fold_of_row {
        *sizes.entry(*fold).or_default() += 1;
    }
    let largest = sizes.values().copied().max().unwrap_or(rows);
    query.active == [1, 1]
        && query.control == [0, 0]
        && sizes.len() == 5
        && largest.saturating_mul(5) <= rows.saturating_add(4)
}

/// Support row key of the contrast under the inference that ran.
pub(crate) fn crossfit_contrast_row_key(
    posterior: bool,
) -> crate::support::GraphlessSupportKey<'static> {
    crate::support::GraphlessSupportKey {
        family: "sequential_contrast",
        design: "two_decision_always_versus_never",
        method: if posterior {
            "crossfit_augmented_scores_bayesian_bootstrap"
        } else {
            "crossfit_augmented_scores_wald"
        },
        inference_claim: if posterior {
            "pointwise_95_posterior_quantiles"
        } else {
            "pointwise_95_normal_interval"
        },
    }
}

/// Fewest posterior draws a posterior row is stated for: the count its measured intervals
/// were formed from.
const EXCURSION_ROW_DRAWS: u64 = 1000;
/// The same floor for the dose grid and regime survival, which were measured at fewer draws.
const EXCURSION_ROW_REFIT_DRAWS: u64 = 500;
/// Completed bootstrap replicates behind the measured percentile intervals.
const EXCURSION_ROW_REPLICATES: u64 = 99;
/// The regime-survival horizon the rows were measured at; no other horizon has a row.
const EXCURSION_ROW_HORIZON: u32 = 3;
/// Fewest validation units per true-outcome margin a misclassification-corrected row is stated
/// for: the smallest validation sample its measured intervals were formed from.
const EXCURSION_ROW_VALIDATION: u64 = 100;
/// Smallest validated Youden index of such a row: the low end of what the measured design
/// realizes (sensitivity 0.9, specificity 0.85, 100 units per margin).
const EXCURSION_ROW_YOUDEN: f64 = 0.6;

/// Whether the fit is the misclassification-corrected two-arm mean inside the measured
/// validation conditions: both validation margins hold at least [`EXCURSION_ROW_VALIDATION`]
/// units, the validated Youden index is at least [`EXCURSION_ROW_YOUDEN`], and the functional
/// carries the correction and nothing else (a sampling-weighted fit names its weights first).
#[allow(clippy::cast_precision_loss, reason = "validation counts are far below 2^53")]
fn validated_correction_row(query: &TemporalDesignQuery, functional: &str) -> bool {
    let Some(counts) = query.outcome_misclassification else {
        return false;
    };
    let positives = counts.true_positive.saturating_add(counts.false_negative);
    let negatives = counts.true_negative.saturating_add(counts.false_positive);
    if !matches!(query.assignment, TemporalAssignment::TwoArm { .. })
        || !matches!(query.functional, TemporalDesignFunctional::Mean)
        || positives < EXCURSION_ROW_VALIDATION
        || negatives < EXCURSION_ROW_VALIDATION
    {
        return false;
    }
    let youden = counts.true_positive as f64 / positives as f64
        + counts.true_negative as f64 / negatives as f64
        - 1.0;
    let corrected = "mean; outcome misclassification corrected";
    youden >= EXCURSION_ROW_YOUDEN
        && functional.strip_prefix(corrected).is_some_and(|rest| {
            rest.is_empty() || (rest.starts_with(" (Youden index ") && rest.ends_with(')'))
        })
}

/// Design name of the graphless support key: the design of the route manifest, which splits
/// the quantile functional and the dose targets that [`TemporalDesignQuery::design_name`]
/// groups. An observational design has no key, and neither has a setting outside the measured
/// ones: a quantile level other than the median, a local-quadratic derivative or a positive
/// curvature bound, or a survival horizon other than [`EXCURSION_ROW_HORIZON`].
#[allow(
    clippy::float_cmp,
    reason = "the measured settings are the exact declared values 0.5 and 0.0"
)]
fn excursion_row_design(query: &TemporalDesignQuery) -> Option<&'static str> {
    if let TemporalDesignFunctional::Quantile { level } = query.functional {
        let two_arm = matches!(query.assignment, TemporalAssignment::TwoArm { .. });
        return (two_arm && level == 0.5).then_some("quantile");
    }
    match &query.assignment {
        TemporalAssignment::TwoArm { .. } => Some("two_arm"),
        TemporalAssignment::MultiArm { categorical: true, .. } => Some("categorical"),
        TemporalAssignment::MultiArm { .. } => Some("multi_arm"),
        TemporalAssignment::Factorial { .. } => Some("factorial"),
        TemporalAssignment::Interference { .. } => Some("interference"),
        TemporalAssignment::Dose { target, .. } => match target {
            TemporalDoseTarget::SmoothedContrast { .. } => Some("continuous_dose"),
            TemporalDoseTarget::SmoothedCurve { .. } => Some("smoothed_dose"),
            TemporalDoseTarget::LocalQuadratic { derivative, curvature_bound, .. } => {
                (!*derivative && *curvature_bound == Some(0.0)).then_some("dose_grid")
            }
        },
        TemporalAssignment::RegimeSurvival { horizon, .. } => {
            (*horizon == EXCURSION_ROW_HORIZON).then_some("regime_survival")
        }
        TemporalAssignment::ObservationalExcursion { .. }
        | TemporalAssignment::ObservationalRegime { .. } => None,
    }
}

/// Inference claim of the graphless support key, from the kind of pointwise 95% interval the
/// section publishes: posterior quantiles, bootstrap percentiles or a normal interval. A
/// posterior with fewer draws, or a bootstrap with fewer completed replicates, than the
/// measured intervals were formed from has no key.
fn excursion_row_claim(
    design: &str,
    fit: &crate::result::TemporalDesignEstimate,
) -> Option<&'static str> {
    if let Some(draws) = fit.draws {
        let floor = if matches!(design, "dose_grid" | "regime_survival") {
            EXCURSION_ROW_REFIT_DRAWS
        } else {
            EXCURSION_ROW_DRAWS
        };
        return (draws >= floor).then_some("pointwise_95_posterior_quantiles");
    }
    if let Some((completed, _)) = fit.replicates {
        return (completed >= EXCURSION_ROW_REPLICATES)
            .then_some("pointwise_95_percentile_interval");
    }
    Some("pointwise_95_normal_interval")
}

/// Positivity the section does not carry, read from the query: the smallest realized arm
/// probability of a two-arm design (the quantile section reports none), and for a dose design
/// the expected number of analysed decisions within one bandwidth of dose at the smallest
/// recorded density (`pairs * density * bandwidth`, rounded down).
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "a saturating count of expected decisions"
)]
fn excursion_row_positivity(query: &TemporalDesignQuery, pairs: u64) -> (f64, usize) {
    match &query.assignment {
        TemporalAssignment::TwoArm { assignment, probability } => {
            let realized = assignment
                .iter()
                .zip(probability.iter())
                .map(|(treated, p)| if *treated { *p } else { 1.0 - *p })
                .fold(1.0_f64, f64::min);
            (realized, 0)
        }
        TemporalAssignment::Dose { density, bandwidth, .. } => {
            let thinnest = density.iter().copied().fold(f64::INFINITY, f64::min);
            (0.0, (pairs as f64 * thinnest * *bandwidth).floor() as usize)
        }
        _ => (0.0, 0),
    }
}

/// Whether a temporal-design fit runs under a graphless support row: its exact key and
/// observed support match one. The key and the observed support are read from the published
/// section and the query alone, so the artifact check recomputes the same answer. `arm_mean`
/// says the design is an arm design with the mean functional.
pub(crate) fn excursion_row_met(
    query: &TemporalDesignQuery,
    arm_mean: bool,
    fit: &crate::result::TemporalDesignEstimate,
) -> bool {
    let Some(design) = excursion_row_design(query) else {
        return false;
    };
    let Some(inference_claim) = excursion_row_claim(design, fit) else {
        return false;
    };
    // A weighted or misclassification-corrected arm functional is no longer the plain mean,
    // and a missing-outcome policy that dropped a pair leaves the counts short of the rows.
    // The corrected two-arm mean under a measured validation sample keeps a key: its method
    // names the correction, so it derives a row of its own.
    if (arm_mean && &*fit.functional != "mean" && !validated_correction_row(query, &fit.functional))
        || fit.pairs.checked_add(fit.unpaired) != u64::try_from(query.rows()).ok()
    {
        return false;
    }
    let published = fit
        .lower
        .iter()
        .zip(fit.upper.iter())
        .filter(|(lower, upper)| lower.is_finite() && upper.is_finite() && lower < upper)
        .count();
    let all_published = published > 0 && published == fit.coordinates.len();
    let (declared_probability, local_rows) = excursion_row_positivity(query, fit.pairs);
    crate::support::support_if_graphless(
        crate::support::GraphlessSupportKey {
            family: "temporal_design",
            design,
            method: fit.uncertainty.as_ref(),
            inference_claim,
        },
        crate::support::GraphlessAssignmentSupport {
            assignment_unit: "sequence",
            rows: usize::try_from(fit.pairs).unwrap_or(usize::MAX),
            blocks: usize::try_from(fit.clusters).unwrap_or(usize::MAX),
            min_probability: if fit.minimum_probability.is_finite() {
                fit.minimum_probability
            } else {
                declared_probability
            },
            min_local_rows: local_rows,
            interval_95_published: all_published,
            reported_intervals: published,
            all_reported_intervals: all_published,
            ..Default::default()
        },
    )
    .is_some()
}
