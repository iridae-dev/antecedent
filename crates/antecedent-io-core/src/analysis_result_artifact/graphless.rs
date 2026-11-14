//! Graphless-support checks for an analysis-result body.
//!
//! Each family's stored status is compared with that family's own eligibility
//! recompute. The recompute stays in this module so a forged status cannot be
//! laundered through another classifier.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{AnalysisResultWire, HistoryMatchWire};
use crate::IoError;
use crate::graphless_support::{GraphlessAssignmentSupport, GraphlessSupportKey, graphless_admits};
use crate::replay_tolerance::{ReplayNonFinite, ReplayScale, ReplayTolerance};

/// Replay policy of a graphless section's published estimates against the values recomputed from
/// its stored scores: a relative `1e-9` of the recomputed value, absolute below one; non-finite
/// is refused.
const SECTION_REPLAY: ReplayTolerance =
    ReplayTolerance::new(1e-9, ReplayScale::OnePlusReplayed, ReplayNonFinite::Refuse);

/// Replay policy of a recomputed quantity that is a closed-form function of other stored
/// numbers (a ratio, a half-sum, a variance identity): absolute to `1e-8`, non-finite refused.
const REPLAY_ABSOLUTE_LOOSE: ReplayTolerance =
    ReplayTolerance::new(1e-8, ReplayScale::Absolute, ReplayNonFinite::Refuse);
/// Replay policy of a quantity recomputed by the same arithmetic as the stored one: absolute to
/// `1e-10`, non-finite refused.
const REPLAY_ABSOLUTE: ReplayTolerance =
    ReplayTolerance::new(1e-10, ReplayScale::Absolute, ReplayNonFinite::Refuse);
/// Replay policy of a quantity that is exact up to one rounding (a probability or a stored copy
/// of a variance): absolute to `1e-12`, non-finite refused.
const REPLAY_ABSOLUTE_TIGHT: ReplayTolerance =
    ReplayTolerance::new(1e-12, ReplayScale::Absolute, ReplayNonFinite::Refuse);
/// Replay policy of two numbers that are both recomputed: a relative `1e-10` of one plus the sum
/// of their magnitudes; non-finite is refused.
const REPLAY_PAIR: ReplayTolerance =
    ReplayTolerance::new(1e-10, ReplayScale::OnePlusSum, ReplayNonFinite::Refuse);

/// Relative tolerance of an interval bound against `center -/+ span`.
const INTERVAL_BOUND_RELATIVE: f64 = 1e-10;
/// Relative tolerance of an interval bound whose half-width is recomputed from a stored variance.
const INTERVAL_BOUND_RELATIVE_LOOSE: f64 = 1e-8;
/// Relative tolerance of a pooled interval bound.
const INTERVAL_BOUND_RELATIVE_POOLED: f64 = 1e-9;

/// Replay policy of an interval bound against `center -/+ span`: `relative` of one plus the
/// magnitudes of the center and the half-width.
fn interval_bound_replay(relative: f64, center: f64, span: f64) -> ReplayTolerance {
    ReplayTolerance::new(
        relative * (1.0 + center.abs() + span.abs()),
        ReplayScale::Absolute,
        ReplayNonFinite::Refuse,
    )
}

/// Expected rows in each arm of an independent Bernoulli intention-to-treat interval (the support
/// row's `min_expected_arm_units`, restated for this self-contained recompute).
const MIN_EXPECTED_ARM_UNITS: f64 = 40.0;
/// Fewest observed rows in each arm of a small-arm Student interval.
const MIN_SMALL_ARM_UNITS: usize = 2;
/// Fewest rows in all of a trial that publishes a small-arm Student interval. Support
/// boundary, not a parser bound: the interval's coverage was only measured from this size, so
/// a stored interval claiming it below the boundary is a forgery whatever its bytes.
const MIN_SMALL_ARM_ROWS: usize = 100;

/// Whether a stored graphless support status matches the independently
/// recomputed support for a row. This is only the status comparison shared
/// across the families; each family's eligibility recompute (`*_supported`) is
/// deliberately kept self-contained as a forgery check across the trust
/// boundary and is never routed through the execute/contract classifier.
/// `allow_legacy_missing` permits a `None` status to stand in for an
/// unsupported row on artifacts written before the field existed.
pub(in crate::analysis_result_artifact) fn graphless_status_ok(
    actual: Option<&str>,
    supported: bool,
    allow_legacy_missing: bool,
) -> bool {
    actual == supported.then_some("supported") || (allow_legacy_missing && actual.is_none())
}

/// `C(n, k)` with the same multiply-then-divide recurrence the artifact
/// validator used inline. Overflow, or `k > n`, refuses instead of panicking
/// or wrapping.
fn combination_count(n: u64, k: u64) -> Option<u64> {
    let mut count = 1u64;
    for i in 0..k {
        let term = n.checked_sub(i)?;
        count = count.checked_mul(term)?.checked_div(i.checked_add(1)?)?;
    }
    Some(count)
}

/// Whether a non-ITT randomized effect equals its intention-to-treat over the
/// first stage. Missing or non-finite components do not match.
fn late_ratio_matches(effect: f64, intention: Option<f64>, first_stage: Option<f64>) -> bool {
    match (intention, first_stage) {
        (Some(intention), Some(first_stage))
            if intention.is_finite() && first_stage.is_finite() && first_stage > f64::EPSILON =>
        {
            REPLAY_ABSOLUTE.close(effect, intention / first_stage)
        }
        _ => false,
    }
}

mod panel_survival;
mod policy;
mod randomized;

pub(in crate::analysis_result_artifact) use panel_survival::{
    validate_did_graphless, validate_survival_graphless,
};
pub(in crate::analysis_result_artifact) use policy::{
    validate_continuous_dose_graphless, validate_longitudinal_graphless, validate_policy_graphless,
};
pub(in crate::analysis_result_artifact) use randomized::{
    validate_randomized_graphless, validate_ratio_graphless,
};

/// Uncertainty label of a sampling-weighted two-arm intention-to-treat fit: the linearized
/// variance of the weighted Hajek arm means, with no interval.
pub(crate) const SAMPLING_WEIGHTED_RANDOMIZED_UNCERTAINTY: &str =
    "sampling_weighted_hajek_linearized_variance_no_interval";

/// Uncertainty label of a sampling-weighted balanced two-period panel `DiD`: the cluster score
/// sandwich over weighted scores, with no interval.
pub(crate) const SAMPLING_WEIGHTED_DID_UNCERTAINTY: &str =
    "sampling_weighted_cluster_robust_standard_error_no_interval";

/// Whether `label` is the sampling-weighted uncertainty, which only the unadjusted Bernoulli
/// and complete designs carry and which publishes no interval.
fn sampling_weighted_uncertainty(
    label: &str,
    design: &crate::RandomizationDesignWire,
    has_interval: bool,
) -> bool {
    label == SAMPLING_WEIGHTED_RANDOMIZED_UNCERTAINTY
        && !has_interval
        && matches!(
            design,
            crate::RandomizationDesignWire::Bernoulli
                | crate::RandomizationDesignWire::Complete { .. }
        )
}

/// Uncertainty label of the normal interval of a sequential contrast.
const SEQUENTIAL_CONTRAST_INTERVAL: &str = "normal_interval_calibration_not_measured";
/// Uncertainty label of the modular posterior of a sequential contrast.
const SEQUENTIAL_CONTRAST_POSTERIOR: &str = "modular_bayesian_bootstrap_posterior_no_interval";

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

/// Whether the support a sequential contrast claims is the support its query and scores have.
/// A section's `supported` flag is the Wald row: the measured design, a published interval and
/// the row's thresholds. A posterior states its row in the design-posterior diagnostic, whose
/// stated row the generic check admits; here the design, the rows and the smallest fitted
/// probability it states must be the section's own.
fn sequential_contrast_row_claim_ok(
    result: &AnalysisResultWire,
    query: &crate::query_wire::SequentialContrastQueryWire,
    fit: &super::SequentialContrastWire,
) -> bool {
    let design = sequential_contrast_row_design(query);
    let rows = fit.contrast_scores.len();
    let wald = !fit.supported
        || (design
            && result.posterior_artifact.is_none()
            && fit.interval.is_some()
            && crate::graphless_support::graphless_admits(
                crate::graphless_support::GraphlessSupportKey {
                    family: "sequential_contrast",
                    design: "two_decision_always_versus_never",
                    method: "crossfit_augmented_scores_wald",
                    inference_claim: "pointwise_95_normal_interval",
                },
                crate::graphless_support::GraphlessAssignmentSupport {
                    assignment_unit: "unit",
                    rows,
                    min_probability: fit.minimum_action_probability,
                    interval_95_published: true,
                    reported_intervals: 1,
                    all_reported_intervals: true,
                    ..Default::default()
                },
            ));
    let posterior = result
        .diagnostics
        .iter()
        .filter(|diagnostic| diagnostic.code == crate::graphless_support::DESIGN_POSTERIOR_ROW_CODE)
        .all(|diagnostic| {
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
        });
    wald && posterior
}

/// Recompute an observational sequential contrast from its retained scores.
///
/// Every reported summary must be the summary of the retained scores, and a support row the
/// result claims must be the one its query and scores meet.
#[allow(
    clippy::float_cmp,
    reason = "the declared floor is carried through the artifact unchanged, so equality is the check"
)]
pub(in crate::analysis_result_artifact) fn validate_sequential_contrast_graphless(
    result: &AnalysisResultWire,
) -> Result<(), IoError> {
    let query = match &result.query {
        crate::CausalQueryWire::SequentialContrast(query) => Some(query),
        _ => None,
    };
    let fail = |message: &str| Err(IoError::Convert(message.into()));
    let (Some(query), Some(fit)) = (query, &result.sequential_contrast) else {
        return if query.is_some() || result.sequential_contrast.is_some() {
            fail("a sequential contrast section needs its query, and its query its section")
        } else {
            Ok(())
        };
    };
    let rows = fit.contrast_scores.len();
    let n = rows as f64;
    let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len() as f64;
    let near = |a: f64, b: f64| SECTION_REPLAY.close(a, b);
    let scores_ok = rows >= 2
        && fit.active_scores.len() == rows
        && fit.control_scores.len() == rows
        && fit.evaluated_rows == rows as u64
        && rows == query.fold_of_row.len()
        && fit
            .active_scores
            .iter()
            .chain(&fit.control_scores)
            .chain(&fit.contrast_scores)
            .all(|value| value.is_finite())
        && fit
            .active_scores
            .iter()
            .zip(&fit.control_scores)
            .zip(&fit.contrast_scores)
            .all(|((a, c), d)| near(*d, a - c));
    if !scores_ok
        || fit.active_regime != query.active
        || fit.control_regime != query.control
        || fit.min_probability != query.min_probability
        || !near(fit.contrast, mean(&fit.contrast_scores))
        || !near(fit.active_mean, mean(&fit.active_scores))
        || !near(fit.control_mean, mean(&fit.control_scores))
        || !fit.minimum_action_probability.is_finite()
        || fit.minimum_action_probability < fit.min_probability
        || !sequential_contrast_row_claim_ok(result, query, fit)
        || fit.calibration != "not_measured"
        || result.estimate.is_some()
        || result.standard_error.is_some()
        || result.interval_lower.is_some()
        || result.interval_upper.is_some()
    {
        return fail("invalid sequential contrast payload or fabricated summary");
    }
    if result.posterior_artifact.is_some() {
        if fit.uncertainty != SEQUENTIAL_CONTRAST_POSTERIOR
            || fit.standard_error.is_some()
            || fit.interval.is_some()
        {
            return fail("a sequential contrast posterior reports no frequentist interval");
        }
        return Ok(());
    }
    let (Some(se), Some(bounds)) = (fit.standard_error, fit.interval) else {
        return fail("a sequential contrast interval needs its standard error and bounds");
    };
    if fit.uncertainty != SEQUENTIAL_CONTRAST_INTERVAL
        || !(fit.coverage > 0.0 && fit.coverage < 1.0)
    {
        return fail("invalid sequential contrast interval label or coverage");
    }
    let centre = mean(&fit.contrast_scores);
    let squares: f64 = fit.contrast_scores.iter().map(|s| (s - centre) * (s - centre)).sum();
    let expected_se = (squares / ((n - 1.0) * n)).sqrt();
    let span = antecedent_stats::normal_ppf(0.5 + fit.coverage / 2.0) * expected_se;
    let tolerance = interval_bound_replay(INTERVAL_BOUND_RELATIVE_POOLED, centre, span);
    if expected_se.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater)
        || !near(se, expected_se)
        || !tolerance.close(bounds[0], centre - span)
        || !tolerance.close(bounds[1], centre + span)
    {
        return fail("sequential contrast interval is not the normal interval of its scores");
    }
    Ok(())
}

/// Uncertainty label of the history-matched contrast.
const HISTORY_MATCH_UNCERTAINTY: &str =
    "abadie_imbens_and_newey_west_period_interval_calibration_not_measured";

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
    query.period.is_some()
        && query.caliper.is_none()
        && query.weight.is_none()
        && fit.caliper.is_none()
        && fit.unmatched_treated.is_empty()
        && crate::graphless_support::graphless_admits(
            crate::graphless_support::GraphlessSupportKey {
                family: "history_match",
                design: "within_period_nearest_history",
                method: "period_newey_west_fixed_b",
                inference_claim: "pointwise_95_fixed_b_interval",
            },
            crate::graphless_support::GraphlessAssignmentSupport {
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

/// Recompute a within-period history-matched contrast from its retained pairs.
///
/// The result carries no top-level summary, so the check is that it claims none, that the
/// contrast, the weight total, the Abadie-Imbens standard error and the Newey-West standard
/// error and interval over the matched periods are those of the retained pairs, their periods
/// and their within-arm variance estimates, and that a support row it claims is one it meets.
#[allow(clippy::float_cmp, reason = "the check requires exact equality of the retained values")]
pub(in crate::analysis_result_artifact) fn validate_history_match_graphless(
    result: &AnalysisResultWire,
) -> Result<(), IoError> {
    let query = match &result.query {
        crate::CausalQueryWire::HistoryMatch(query) => Some(query),
        _ => None,
    };
    let is_query = query.is_some();
    let Some(fit) = &result.history_match else {
        // The outcome-model posterior of the same query carries no matched section.
        return if is_query && result.posterior_artifact.is_none() {
            Err(IoError::Convert("history match artifact is missing its result section".into()))
        } else {
            Ok(())
        };
    };
    let fail = |message: &str| Err(IoError::Convert(message.into()));
    if !is_query {
        return fail("a history match section needs its query");
    }
    let near = |a: f64, b: f64| SECTION_REPLAY.close(a, b);
    let pairs = fit.matched_queries.len();
    let total: f64 = fit.pair_weights.iter().sum();
    let weighted: f64 = fit.pair_weights.iter().zip(&fit.differences).map(|(w, d)| w * d).sum();
    let structure_ok = pairs > 0
        && fit.matched_donors.len() == pairs
        && fit.differences.len() == pairs
        && fit.pair_weights.len() == pairs
        && fit.matched_queries.windows(2).all(|w| w.first() < w.get(1))
        && fit.pair_weights.iter().chain(&fit.differences).all(|v| v.is_finite())
        && total > 0.0
        && near(fit.weight_total, total)
        && near(fit.contrast, weighted / total);
    if !structure_ok
        || fit.calibration != "not_measured"
        || fit.uncertainty != HISTORY_MATCH_UNCERTAINTY
        || result.estimate.is_some()
        || result.standard_error.is_some()
        || result.interval_lower.is_some()
        || result.interval_upper.is_some()
    {
        return fail("invalid history match payload or fabricated summary");
    }
    // Abadie-Imbens conditional variance of a weighted mean of one-match differences, with
    // replacement: squared shares times the within-arm variance of each query plus squared donated
    // shares times that of each used donor.
    let variances_ok = fit.query_variances.len() == fit.donor_variances.len()
        && fit
            .query_variances
            .iter()
            .chain(&fit.donor_variances)
            .all(|v| v.is_finite() && *v >= 0.0)
        && (fit.query_variances.is_empty() || fit.query_variances.len() == pairs)
        && (fit.query_variances.is_empty() == fit.standard_error.is_none());
    if !variances_ok {
        return fail("invalid history match variance estimates");
    }
    let expected = if fit.query_variances.is_empty() || fit.pair_weights.iter().any(|w| *w <= 0.0) {
        None
    } else {
        let mut variance = 0.0_f64;
        let mut donated: std::collections::BTreeMap<usize, (f64, f64)> =
            std::collections::BTreeMap::new();
        for k in 0..pairs {
            let share = fit.pair_weights[k] / total;
            variance += share * share * fit.query_variances[k];
            let entry =
                donated.entry(fit.matched_donors[k]).or_insert((0.0, fit.donor_variances[k]));
            if entry.1 != fit.donor_variances[k] {
                return fail("history match donor variance differs between its pairs");
            }
            entry.0 += share;
        }
        variance += donated.values().map(|(share, noise)| share * share * noise).sum::<f64>();
        (variance.is_finite() && variance >= 0.0).then_some(variance.sqrt())
    };
    let consistent = match (fit.standard_error, expected) {
        (None, None) => true,
        (Some(reported), Some(recomputed)) => near(reported, recomputed),
        _ => false,
    };
    if !consistent {
        return fail("history match standard error is not the Abadie-Imbens form of its pairs");
    }
    validate_history_match_serial(fit, total, near)?;
    if fit.supported && !query.is_some_and(|query| history_match_row_admits(query, fit)) {
        return fail("history match claims a support row its query and pairs do not meet");
    }
    Ok(())
}

/// Newey-West standard error and interval of the weighted mean of the pair differences, with the
/// pairs aggregated by calendar period (periods enter by label, so a missing period invents no
/// adjacency). Bartlett truncation lag `floor(4 (T / 100)^(2/9))` clamped to `[1, T - 1]`, scale
/// `T / (T - 1) / total^2`. The interval is the Kiefer-Vogelsang fixed-b interval: the normal
/// 97.5% point times `cv(b) / 1.96`, `cv(b) = 1.96 + 2.9694 b + 0.4160 b^2 - 0.5324 b^3`, at the
/// bandwidth share `b = (lag + 1) / T`.
#[allow(
    clippy::cast_sign_loss,
    reason = "the value is bounded by its validated numerical or fixture domain"
)]
#[allow(
    clippy::cast_possible_truncation,
    reason = "the value is bounded by its validated numerical or fixture domain"
)]
fn validate_history_match_serial(
    fit: &HistoryMatchWire,
    total: f64,
    near: impl Fn(f64, f64) -> bool,
) -> Result<(), IoError> {
    let fail = |message: &str| Err(IoError::Convert(message.into()));
    let pairs = fit.matched_queries.len();
    let reported = (fit.serial_standard_error, fit.interval_lower, fit.interval_upper);
    if fit.pair_periods.is_empty() {
        return if reported == (None, None, None) && fit.serial_bandwidth == 0 {
            Ok(())
        } else {
            fail("history match reports a period interval without its pair periods")
        };
    }
    if fit.pair_periods.len() != pairs {
        return fail("history match pair periods differ in length from its pairs");
    }
    let mut labels = fit.pair_periods.clone();
    labels.sort_unstable();
    labels.dedup();
    if labels.len() < 2 || fit.pair_weights.iter().any(|w| *w <= 0.0) {
        return if reported == (None, None, None) && fit.serial_bandwidth == 0 {
            Ok(())
        } else {
            fail("history match reports a period interval that needs two matched periods")
        };
    }
    let mut residual = vec![0.0_f64; labels.len()];
    for k in 0..pairs {
        let at = labels.binary_search(&fit.pair_periods[k]).unwrap_or_default();
        residual[at] += fit.pair_weights[k] * (fit.differences[k] - fit.contrast);
    }
    let t = labels.len();
    let bandwidth = {
        let rule = (4.0 * (t as f64 / 100.0).powf(2.0 / 9.0)).floor().max(1.0) as usize;
        rule.min(t - 1)
    };
    let mut meat: f64 = residual.iter().map(|r| r * r).sum();
    for lag in 1..=bandwidth {
        let weight = 1.0 - lag as f64 / (bandwidth as f64 + 1.0);
        let Ok(step) = i64::try_from(lag) else { continue };
        for (index, label) in labels.iter().enumerate() {
            if let Some(earlier) =
                label.checked_sub(step).and_then(|p| labels.binary_search(&p).ok())
            {
                meat += 2.0 * weight * residual[index] * residual[earlier];
            }
        }
    }
    let variance = meat * t as f64 / (t as f64 - 1.0) / (total * total);
    let expected_se = variance.max(0.0).sqrt();
    let share = ((bandwidth + 1) as f64 / t as f64).clamp(0.0, 1.0);
    let fixed_b =
        (1.96 + 2.9694 * share + 0.4160 * share * share - 0.5324 * share * share * share) / 1.96;
    let critical = antecedent_stats::NORMAL_Q975 * fixed_b;
    let ok = matches!(reported, (Some(se), Some(lo), Some(hi))
            if expected_se.is_finite()
                && near(se, expected_se)
                && near(lo, fit.contrast - critical * expected_se)
                && near(hi, fit.contrast + critical * expected_se))
        && fit.serial_bandwidth == bandwidth;
    if ok {
        Ok(())
    } else {
        fail("history match period interval is not the Newey-West interval of its pairs")
    }
}
