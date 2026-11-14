//! Wire section of a rollout design result: heterogeneity-robust or covariate-adjusted
//! difference in differences, conformal or multi-unit synthetic control, and the local ratio
//! with its design layer.
//!
//! The section is optional on the result body and absent on every other result, so their bodies
//! and digests are unchanged. A loader checks that the section belongs to its query, claims
//! support only where a support row admits what the section reports, claims no calibration,
//! publishes no scalar interval beside it, that each pointwise interval with a standard error
//! is centred and at least as wide as that error's normal interval, that each band contains
//! it, and re-derives the
//! scalar summary from the reported effects wherever the design's aggregation defines it from
//! them alone.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::cmp::Ordering;

use serde::{Deserialize, Serialize};

use super::AnalysisResultWire;
use crate::replay_tolerance::{ReplayNonFinite, ReplayScale, ReplayTolerance};
use crate::{CausalQueryWire, IoError};

/// Replay policy of a rollout's published numbers against its own effects: a relative `1e-9`
/// of the larger value, absolute below one; non-finite is refused.
const ROLLOUT_AGREEMENT: ReplayTolerance =
    ReplayTolerance::new(1e-9, ReplayScale::OnePlusLarger, ReplayNonFinite::Refuse);

/// Normal 97.5% quantile of the pointwise intervals.
pub(super) const NORMAL_Q975: f64 = 1.959_963_984_540_054;

/// One reported effect of a rollout design.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DesignEffectWire {
    /// `overall`, `group_time`, `event_time`, `group`, `calendar`, `pre_trend`, `period` or
    /// `unit`.
    pub kind: String,
    /// Adoption cohort.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group: Option<i64>,
    /// Calendar period.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub period: Option<i64>,
    /// Event time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub event_time: Option<i64>,
    /// Treated unit label.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
    /// Point estimate.
    pub estimate: f64,
    /// Standard error of the uncertainty construction.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub standard_error: Option<f64>,
    /// Pointwise 95% interval.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub interval: Option<[f64; 2]>,
    /// Simultaneous 95% band.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub band: Option<[f64; 2]>,
    /// Treated units, rows or post periods behind the estimate.
    pub treated: usize,
    /// Comparison units or rows behind the estimate.
    pub comparison: usize,
}

/// A rollout design result with its whole-procedure uncertainty (calibration `not_measured`;
/// supported only where a support row admits the reported design).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct QuasiRolloutWire {
    /// Design name.
    pub design: String,
    /// Estimator, comparison, adjustment and aggregation words.
    pub method: String,
    /// Reported effects; the first is the scalar summary.
    pub effects: Vec<DesignEffectWire>,
    /// How the uncertainty was computed.
    pub uncertainty: String,
    /// Always `not_measured`.
    pub calibration: String,
    /// Requested replicates.
    pub replicates_requested: u32,
    /// Successful replicates.
    pub replicates_ok: u32,
    /// Failed replicates.
    pub replicates_failed: u32,
    /// Sup-t critical value of the simultaneous band.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub critical_value: Option<f64>,
    /// Named numeric diagnostics.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub diagnostics: Vec<(String, f64)>,
    /// Whether a support row admits the reported design, method and observed support.
    pub supported: bool,
}

impl QuasiRolloutWire {
    /// The named numeric diagnostic, when present.
    #[must_use]
    pub fn diagnostic(&self, name: &str) -> Option<f64> {
        self.diagnostics.iter().find(|(key, _)| key == name).map(|(_, value)| *value)
    }
}

/// The scalar summary a fresh consumer re-derives from the reported effects alone, when the
/// design's aggregation defines it from them: the mean of the post-adoption event-time effects
/// (dynamic), the mean of the calendar effects (calendar), the mean of the per-period effects
/// (conformal synthetic control) or the post-period-weighted mean of the per-unit effects
/// (multi-unit synthetic control). `None` for aggregations that need cohort sizes the section
/// does not carry.
#[must_use]
pub fn rederive_rollout_summary(wire: &QuasiRolloutWire) -> Option<f64> {
    fn kind_of<'a>(
        wire: &'a QuasiRolloutWire,
        kind: &'a str,
    ) -> impl Iterator<Item = &'a DesignEffectWire> + 'a {
        wire.effects.iter().filter(move |effect| effect.kind == kind)
    }
    let of = |kind: &'static str| kind_of(wire, kind);
    let mean = |values: Vec<f64>| {
        (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
    };
    match wire.design.as_str() {
        "staggered_did" if wire.method.contains(".dynamic") => mean(
            of("event_time")
                .filter(|effect| effect.event_time.is_some_and(|event| event >= 0))
                .map(|effect| effect.estimate)
                .collect(),
        ),
        "staggered_did" if wire.method.contains(".calendar") => {
            mean(of("calendar").map(|effect| effect.estimate).collect())
        }
        "conformal_synthetic_control" => mean(of("period").map(|effect| effect.estimate).collect()),
        "multi_unit_synthetic_control" | "multi_unit_synthetic_did" => {
            let units: Vec<&DesignEffectWire> = of("unit").collect();
            let total: usize = units.iter().map(|unit| unit.treated).sum();
            (total > 0).then(|| {
                units.iter().map(|unit| unit.estimate * unit.treated as f64).sum::<f64>()
                    / total as f64
            })
        }
        _ => None,
    }
}

/// Whether a support row admits a local-ratio design section, from the section alone and the
/// query's bandwidth rule (`selected_bandwidth`: the MSE-optimal rule was declared).
///
/// The key is the section's design and interval procedure; the observed support is the window
/// rows on each side, the analysed rows, the first-stage estimate over its standard error
/// (infinite for a first stage without error) and the replicates or draws behind the interval.
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a count diagnostic is checked finite and non-negative before it is read as a count"
)]
pub fn local_design_row_admits(selected_bandwidth: bool, section: &QuasiRolloutWire) -> bool {
    let (design, polynomial) = match section.design.as_str() {
        "fuzzy_rd" => ("fuzzy_jump", "local_quadratic_cubic"),
        "regression_kink" => ("regression_kink", "local_quartic_slope"),
        "temporal_fuzzy_rd" => ("temporal_fuzzy_jump", "local_quadratic_cubic"),
        "temporal_regression_kink" => ("temporal_regression_kink", "local_quartic_slope"),
        _ => return false,
    };
    let (procedure, inference_claim) = match (section.uncertainty.as_str(), selected_bandwidth) {
        ("rbc_hc0_delta_normal_at_selected_bandwidth", true) => {
            ("rbc_hc0_delta_mse_optimal_bandwidth", "pointwise_95_normal_interval")
        }
        ("rbc_hc0_delta_normal_fixed_bandwidth", false) => {
            ("rbc_hc0_delta_declared_bandwidth_with_diagnostics", "pointwise_95_normal_interval")
        }
        ("within_side_moving_block_bootstrap_refit_iqr_se_student_t_interval", false) => {
            ("rbc_prewhitened_block_bootstrap", "pointwise_95_student_interval")
        }
        ("bayesian_within_side_block_bootstrap_credible_interval", false) => {
            ("rbc_prewhitened_block_bayesian_bootstrap", "pointwise_95_posterior_quantiles")
        }
        _ => return false,
    };
    let method = format!("{polynomial}_{procedure}");
    let count = |name: &str| {
        section
            .diagnostic(name)
            .filter(|value| value.is_finite() && *value >= 0.0)
            .map_or(0, |value| value as usize)
    };
    let first_stage_t =
        match (section.diagnostic("first_stage"), section.diagnostic("first_stage_standard_error"))
        {
            (Some(first_stage), Some(error)) if first_stage.is_finite() && first_stage != 0.0 => {
                if error == 0.0 {
                    f64::INFINITY
                } else if error > 0.0 {
                    first_stage.abs() / error
                } else {
                    0.0
                }
            }
            _ => 0.0,
        };
    let published = section.effects.first().is_some_and(|effect| effect.interval.is_some());
    crate::graphless_support::graphless_admits(
        crate::graphless_support::GraphlessSupportKey {
            family: "local_polynomial_ratio",
            design,
            method: &method,
            inference_claim,
        },
        crate::graphless_support::GraphlessAssignmentSupport {
            assignment_unit: "unit",
            treated: count("rows_right"),
            control: count("rows_left"),
            interval_95_published: published,
            rows: count("rows"),
            reported_intervals: usize::from(published),
            first_stage_t,
            interval_resamples: section.replicates_ok as usize,
            ..Default::default()
        },
    )
}

/// A named count diagnostic of a section; zero when absent or not a count.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "a count diagnostic is checked finite and non-negative before it is read as a count"
)]
fn rollout_count(section: &QuasiRolloutWire, name: &str) -> usize {
    section
        .diagnostic(name)
        .filter(|value| value.is_finite() && *value >= 0.0)
        .map_or(0, |value| value as usize)
}

/// The declared specification a difference-in-differences rollout row is keyed on.
#[derive(Clone, Copy, Debug)]
pub struct DidRolloutRowSpec<'a> {
    /// The covariate-adjusted two-period design; otherwise the staggered rollout.
    pub two_period: bool,
    /// `callaway_santanna`, `sun_abraham` or `imputation`.
    pub estimator: &'a str,
    /// `never_treated` or `not_yet_treated`.
    pub comparison: &'a str,
    /// `outcome_regression`, `inverse_probability_weighting` or `doubly_robust`.
    pub nuisance: &'a str,
    /// `cluster_bootstrap` or `cluster_multiplier`; absent without a resample.
    pub scheme: Option<&'a str>,
    /// Declared baseline covariates.
    pub covariates: usize,
    /// The simple aggregate with no anticipation, the varying base period and no declared
    /// observation process.
    pub measured_options: bool,
}

impl<'a> DidRolloutRowSpec<'a> {
    fn of(
        spec: &'a crate::rollout_query_wire::RolloutDidWire,
        resampling: Option<&'a crate::rollout_query_wire::DidResamplingWire>,
    ) -> Self {
        Self {
            two_period: spec.design == "covariate_adjusted_two_period",
            estimator: &spec.estimator,
            comparison: &spec.comparison,
            nuisance: &spec.nuisance,
            scheme: resampling.map(|resampling| resampling.scheme.as_str()),
            covariates: spec.covariates.len(),
            measured_options: spec.aggregation == "simple"
                && spec.anticipation == 0
                && spec.base_period == "varying"
                && spec.observation.is_none(),
        }
    }
}

/// Whether a support row admits a frequentist heterogeneity-robust or covariate-adjusted
/// difference-in-differences section under its declared specification and resampling scheme.
///
/// A row is keyed on the declared estimator, comparison (or nuisance method) and scheme, and
/// is stated only for what was measured: the simple aggregate of a balanced panel read whole,
/// one unit per inference cluster, no anticipation, the varying base period and no declared
/// observation process. The observed support is the clusters, the smallest treated cohort and
/// the never-treated units, the completed refits, and for a covariate-adjusted comparison
/// the covariate count and the room left above the largest fitted comparison propensity.
#[must_use]
pub fn did_rollout_row_admits(spec: DidRolloutRowSpec<'_>, section: &QuasiRolloutWire) -> bool {
    let scheme = match (spec.scheme, section.uncertainty.as_str()) {
        (
            Some("cluster_bootstrap"),
            "whole_cluster_bootstrap_refit_iqr_se_pointwise_and_sup_t_band",
        ) => "cluster_bootstrap",
        (
            Some("cluster_multiplier"),
            "whole_cluster_multiplier_refit_iqr_se_pointwise_and_sup_t_band",
        ) => "cluster_multiplier",
        _ => return false,
    };
    let units = rollout_count(section, "units");
    let clusters = rollout_count(section, "clusters");
    let rows = rollout_count(section, "rows_read");
    if !spec.measured_options
        || section.diagnostic("balanced_panel") != Some(1.0)
        || units == 0
        || clusters != units
        || rows != rollout_count(section, "rows_supplied")
    {
        return false;
    }
    let treated_units = rollout_count(section, "treated_units");
    let never_treated = rollout_count(section, "never_treated_units");
    let (design, method, inference_claim, treated, min_probability) =
        match (section.design.as_str(), spec.two_period) {
            ("staggered_did", false) if spec.covariates == 0 => {
                let smallest_cohort = section
                    .diagnostics
                    .iter()
                    .filter(|(name, value)| {
                        name.starts_with("cohort_")
                            && name.ends_with("_units")
                            && value.is_finite()
                            && *value >= 0.0
                    })
                    .map(|(name, _)| rollout_count(section, name))
                    .min()
                    .unwrap_or(0);
                (
                    "staggered_heterogeneity_robust",
                    format!("{}_{}_{scheme}_refit", spec.estimator, spec.comparison),
                    "simple_aggregate_pointwise_95_student_interval",
                    smallest_cohort,
                    0.0,
                )
            }
            ("covariate_adjusted_did", true)
                if spec.covariates > 0 && spec.estimator == "callaway_santanna" =>
            {
                let room = section
                    .diagnostic("comparison_propensity_max")
                    .filter(|largest| largest.is_finite() && *largest > 0.0 && *largest < 1.0)
                    .map_or(0.0, |largest| 1.0 - largest);
                (
                    "covariate_adjusted_two_period",
                    format!("{}_{scheme}_refit", spec.nuisance),
                    "pointwise_95_student_interval",
                    treated_units,
                    room,
                )
            }
            _ => return false,
        };
    let published = section
        .effects
        .first()
        .is_some_and(|effect| effect.interval.is_some() && effect.standard_error.is_some());
    crate::graphless_support::graphless_admits(
        crate::graphless_support::GraphlessSupportKey {
            family: "difference_in_differences",
            design,
            method: &method,
            inference_claim,
        },
        crate::graphless_support::GraphlessAssignmentSupport {
            assignment_unit: "cluster",
            treated,
            control: never_treated,
            interval_95_published: published,
            rows,
            blocks: clusters,
            min_probability,
            reported_intervals: usize::from(published),
            covariates: spec.covariates,
            interval_resamples: section.replicates_ok as usize,
            ..Default::default()
        },
    )
}

/// Whether a support row admits a frequentist synthetic-control section with intervals: the
/// conformal sets of one treated unit, or the placebo-variance interval of a staggered
/// multi-unit rollout.
///
/// The conformal row counts the pre-intervention periods and the donor series and needs every
/// set bounded. The placebo row counts the adopting units, the never-treated donor series, the
/// fewest pre-adoption periods of an adopting unit and the completed placebo refits.
#[must_use]
pub fn synthetic_rollout_row_admits(section: &QuasiRolloutWire) -> bool {
    let intervals = section.effects.iter().filter(|effect| effect.interval.is_some()).count();
    let overall = section.effects.first().is_some_and(|effect| effect.interval.is_some());
    let (design, method, inference_claim, observed) =
        match (section.design.as_str(), section.uncertainty.as_str()) {
            ("conformal_synthetic_control", "conformal_test_inversion_refit_under_each_null") => (
                "single_treated_unit",
                "simplex_donor_weights_conformal_test_inversion",
                "per_period_and_constant_effect_95_conformal_sets",
                crate::graphless_support::GraphlessAssignmentSupport {
                    assignment_unit: "sequence",
                    rows: rollout_count(section, "pre_periods"),
                    blocks: rollout_count(section, "donors"),
                    interval_95_published: overall,
                    reported_intervals: intervals,
                    all_reported_intervals: intervals == section.effects.len()
                        && section.diagnostic("unbounded_intervals") == Some(0.0),
                    ..Default::default()
                },
            ),
            (
                "multi_unit_synthetic_control",
                "placebo_variance_refit_normal_interval_and_per_unit_sup_t_band",
            ) => (
                "staggered_multi_unit",
                "simplex_donor_weights_placebo_variance",
                "post_weighted_average_pointwise_95_normal_interval",
                crate::graphless_support::GraphlessAssignmentSupport {
                    assignment_unit: "sequence",
                    treated: rollout_count(section, "adopting_units"),
                    control: rollout_count(section, "never_treated_donors"),
                    rows: rollout_count(section, "fewest_pre_periods"),
                    blocks: rollout_count(section, "never_treated_donors"),
                    interval_95_published: overall,
                    reported_intervals: usize::from(overall),
                    interval_resamples: section.replicates_ok as usize,
                    ..Default::default()
                },
            ),
            _ => return false,
        };
    crate::graphless_support::graphless_admits(
        crate::graphless_support::GraphlessSupportKey {
            family: "synthetic_control",
            design,
            method,
            inference_claim,
        },
        observed,
    )
}

/// Whether a support row admits the section under its query. Only the designs with a row
/// have an arm.
fn rollout_row_admits(query: &CausalQueryWire, section: &QuasiRolloutWire) -> bool {
    match query {
        CausalQueryWire::LocalPolynomialRatio(query) => local_design_row_admits(
            query.options.as_ref().is_some_and(|options| options.bandwidth_rule == "mse_optimal"),
            section,
        ),
        CausalQueryWire::PanelDid(query) => query.rollout.as_ref().is_some_and(|spec| {
            did_rollout_row_admits(DidRolloutRowSpec::of(spec, query.resampling.as_ref()), section)
        }),
        CausalQueryWire::SyntheticControl(_) => synthetic_rollout_row_admits(section),
        _ => false,
    }
}

fn design_matches(design: &str, query: &CausalQueryWire) -> bool {
    match query {
        CausalQueryWire::PanelDid(query) => {
            query.rollout.as_ref().is_some_and(|rollout| {
                (design == "covariate_adjusted_did")
                    == (rollout.design == "covariate_adjusted_two_period")
            }) && matches!(design, "staggered_did" | "covariate_adjusted_did")
        }
        CausalQueryWire::SyntheticControl(query) => {
            query.rollout.is_some()
                && matches!(
                    design,
                    "conformal_synthetic_control"
                        | "posterior_predictive_synthetic_control"
                        | "multi_unit_synthetic_control"
                        | "multi_unit_synthetic_did"
                )
        }
        CausalQueryWire::LocalPolynomialRatio(query) => {
            query.options.as_ref().is_some_and(|options| {
                let temporal = options.temporal_replicates.is_some();
                let label = if query.kink { "regression_kink" } else { "fuzzy_rd" };
                if temporal { design == format!("temporal_{label}") } else { design == label }
            })
        }
        _ => false,
    }
}

/// Check a rollout section against its query and re-derive what the section determines.
pub(in crate::analysis_result_artifact) fn validate_quasi_rollout(
    result: &AnalysisResultWire,
) -> Result<(), IoError> {
    let Some(rollout) = &result.quasi_rollout else {
        return Ok(());
    };
    let fail = |message: &str| Err(IoError::Convert(format!("quasi rollout section: {message}")));
    if !design_matches(&rollout.design, &result.query) {
        return fail("the design does not belong to the query");
    }
    // A Bayesian rollout section rides beside its posterior; a frequentist one has none and
    // publishes no scalar standard error.
    let bayesian = rollout.uncertainty.starts_with("bayesian_");
    if bayesian != result.posterior_artifact.is_some() {
        return fail("a Bayesian rollout section needs its posterior, a frequentist one none");
    }
    if (rollout.supported && !rollout_row_admits(&result.query, rollout))
        || rollout.calibration != "not_measured"
        || (!bayesian && result.standard_error.is_some())
        || result.interval_lower.is_some()
        || result.interval_upper.is_some()
    {
        return fail(
            "a rollout result claims support no row admits, a calibration or a scalar interval",
        );
    }
    if bayesian && rollout.effects.iter().any(|effect| effect.standard_error.is_some()) {
        return fail("a credible interval carries no normal-interval standard error");
    }
    let Some(overall) = rollout.effects.first() else {
        return fail("the scalar summary is missing");
    };
    if overall.kind != "overall"
        || rollout.effects[1..].iter().any(|effect| effect.kind == "overall")
    {
        return fail("exactly the first effect is the scalar summary");
    }
    if result.estimate.is_some_and(|estimate| !ROLLOUT_AGREEMENT.close(estimate, overall.estimate))
    {
        return fail("the published estimate differs from the scalar summary");
    }
    if rollout.replicates_ok.checked_add(rollout.replicates_failed)
        != Some(rollout.replicates_requested)
    {
        return fail("replicate counts do not add up");
    }
    if rollout.critical_value.is_some_and(|critical| {
        !critical.is_finite()
            || critical.partial_cmp(&(NORMAL_Q975 - 1e-12)).is_none_or(Ordering::is_lt)
    }) {
        return fail("a critical value is not finite or is below the pointwise quantile");
    }
    if rollout.diagnostics.iter().any(|(_, value)| !value.is_finite()) {
        return fail("a diagnostic is not finite");
    }
    let near = |a: f64, b: f64| ROLLOUT_AGREEMENT.close(a, b);
    for effect in &rollout.effects {
        if !effect.estimate.is_finite() {
            return fail("an effect estimate is not finite");
        }
        if effect.standard_error.is_some_and(|se| !se.is_finite()) {
            return fail("a standard error is not finite");
        }
        if effect.interval.iter().chain(effect.band.iter()).flatten().any(|v| !v.is_finite()) {
            return fail("a pointwise interval or band is not finite");
        }
        if let Some(interval) = effect.interval {
            if interval[0] > interval[1] {
                return fail("a pointwise interval is reversed");
            }
            // A conformal set is the hull of the accepted effects and need not contain the
            // pre-period point; a standard-error interval is centred on its estimate, except
            // the robust bias-corrected local-ratio interval, which is centred on the
            // bias-corrected estimate and sits beside the conventional standard error.
            // A centred interval is symmetric about its estimate and never narrower than the
            // normal interval of its standard error: a Student-t reference only widens it.
            if let Some(se) = effect.standard_error {
                let centred = !rollout.uncertainty.starts_with("rbc_");
                let (below, above) = (effect.estimate - interval[0], interval[1] - effect.estimate);
                let normal = NORMAL_Q975 * se;
                if se.partial_cmp(&0.0) != Some(Ordering::Greater)
                    || (centred
                        && (!near(below, above) || (below < normal && !near(below, normal))))
                {
                    return fail(
                        "a pointwise interval is not centred on its estimate or is narrower \
                         than the normal interval of its standard error",
                    );
                }
            }
            if let Some(band) = effect.band
                && (band[0] > interval[0] + 1e-12 || band[1] < interval[1] - 1e-12)
            {
                return fail("a simultaneous band does not contain its pointwise interval");
            }
        } else if effect.band.is_some() {
            return fail("a band without its pointwise interval");
        }
    }
    if let Some(summary) = rederive_rollout_summary(rollout)
        && !near(summary, overall.estimate)
    {
        return fail("the scalar summary does not re-derive from the reported effects");
    }
    Ok(())
}
