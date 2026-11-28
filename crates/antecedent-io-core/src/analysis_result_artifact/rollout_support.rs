//! Support rows of a rollout design section and the Student-t reference of its pointwise
//! intervals.
//!
//! The executor and the artifact loader decide a rollout section's support through the same
//! functions here, each reading the section in its artifact form, so the two cannot drift. A
//! pointwise interval wider than the normal interval of its standard error is accepted by the
//! loader only when it is the Student-t interval of the section's own procedure.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::QuasiRolloutWire;
use crate::CausalQueryWire;
use crate::replay_tolerance::{ReplayNonFinite, ReplayScale, ReplayTolerance};

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
    crate::graphless_support::graphless_admits_resampled(
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
            ..Default::default()
        },
        crate::graphless_support::GraphlessResampleSupport {
            first_stage_t,
            interval_resamples: section.replicates_ok as usize,
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
    did_rollout_row(spec, section).is_some_and(|row| row.admitted)
}

/// The support row a rollout section is decided under and whether that row admits it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RolloutRowDecision {
    /// The row's family.
    pub family: &'static str,
    /// The row's design.
    pub design: &'static str,
    /// The row's method.
    pub method: String,
    /// The row's inference claim.
    pub inference_claim: &'static str,
    /// Whether the row admits the section.
    pub admitted: bool,
}

/// [`did_rollout_row_admits`] with the row it decided under; `None` when the section's
/// estimator, scheme or design has no row.
#[must_use]
pub fn did_rollout_row(
    spec: DidRolloutRowSpec<'_>,
    section: &QuasiRolloutWire,
) -> Option<RolloutRowDecision> {
    let scheme = match (spec.scheme, section.uncertainty.as_str()) {
        (
            Some("cluster_bootstrap"),
            "whole_cluster_bootstrap_refit_iqr_se_pointwise_and_sup_t_band",
        ) => "cluster_bootstrap",
        (
            Some("cluster_multiplier"),
            "whole_cluster_multiplier_refit_iqr_se_pointwise_and_sup_t_band",
        ) => "cluster_multiplier",
        _ => return None,
    };
    let units = rollout_count(section, "units");
    let clusters = rollout_count(section, "clusters");
    let rows = rollout_count(section, "rows_read");
    let measured = spec.measured_options
        && section.diagnostic("balanced_panel") == Some(1.0)
        && units != 0
        && clusters == units
        && rows == rollout_count(section, "rows_supplied");
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
            _ => return None,
        };
    let published = section
        .effects
        .first()
        .is_some_and(|effect| effect.interval.is_some() && effect.standard_error.is_some());
    let admitted = measured
        && crate::graphless_support::graphless_admits_resampled(
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
            ..Default::default()
        },
        crate::graphless_support::GraphlessResampleSupport {
            interval_resamples: section.replicates_ok as usize,
            ..Default::default()
        },
    );
    Some(RolloutRowDecision {
        family: "difference_in_differences",
        design,
        method,
        inference_claim,
        admitted,
    })
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
    synthetic_rollout_row(section).is_some_and(|row| row.admitted)
}

/// [`synthetic_rollout_row_admits`] with the row it decided under; `None` when the section's
/// design has no row.
#[must_use]
pub fn synthetic_rollout_row(section: &QuasiRolloutWire) -> Option<RolloutRowDecision> {
    let intervals = section.effects.iter().filter(|effect| effect.interval.is_some()).count();
    let overall = section.effects.first().is_some_and(|effect| effect.interval.is_some());
    let (design, method, inference_claim, observed, interval_resamples) =
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
                0,
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
                    ..Default::default()
                },
                section.replicates_ok as usize,
            ),
            _ => return None,
        };
    let admitted = crate::graphless_support::graphless_admits_resampled(
        crate::graphless_support::GraphlessSupportKey {
            family: "synthetic_control",
            design,
            method,
            inference_claim,
        },
        observed,
        crate::graphless_support::GraphlessResampleSupport {
            interval_resamples,
            ..Default::default()
        },
    );
    Some(RolloutRowDecision {
        family: "synthetic_control",
        design,
        method: method.to_owned(),
        inference_claim,
        admitted,
    })
}

/// Whether a support row admits the section under its query. Only the designs with a row
/// have an arm.
pub(super) fn rollout_row_admits(query: &CausalQueryWire, section: &QuasiRolloutWire) -> bool {
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

/// Replay policy of a recomputed interval endpoint: the rollout section's own agreement.
const STUDENT_ENDPOINT_AGREEMENT: ReplayTolerance =
    ReplayTolerance::new(1e-9, ReplayScale::OnePlusLarger, ReplayNonFinite::Refuse);

/// Named diagnostic by which a calendar-time local ratio carries the Satterthwaite degrees of
/// freedom of its Student-t interval.
pub const STUDENT_REFERENCE_DIAGNOSTIC: &str = "interval_degrees_of_freedom";

/// Equivalent degrees of freedom of an interquartile scale estimate per replicate, and the
/// fewest replicates counted: the constants of the cluster-resample Student-t reference.
const CLUSTER_REFERENCE_EFFICIENCY: f64 = 0.3675;
const CLUSTER_REFERENCE_MIN_REPLICATES: u32 = 20;

/// Critical value of a whole-cluster resample's pointwise interval: the Student-t 97.5% point
/// at the Satterthwaite combination of `clusters - 1` and `0.3675 x replicates` degrees of
/// freedom, never below the normal point.
#[allow(clippy::cast_precision_loss, reason = "cluster counts are far below 2^53")]
fn cluster_reference_critical(clusters: usize, replicates_ok: u32) -> f64 {
    let normal = antecedent_stats::NORMAL_Q975;
    let sampling = clusters.saturating_sub(1) as f64;
    let monte_carlo = CLUSTER_REFERENCE_EFFICIENCY
        * f64::from(replicates_ok.max(CLUSTER_REFERENCE_MIN_REPLICATES));
    if sampling <= 0.0 {
        return normal;
    }
    let degrees_of_freedom = 1.0 / (1.0 / sampling + 1.0 / monte_carlo);
    let critical = antecedent_stats::student_t_ppf(0.975, degrees_of_freedom);
    if critical.is_finite() { critical.max(normal) } else { normal }
}

/// Whether `interval` is the Student-t interval the section's procedure builds around
/// `estimate` with `standard_error`, recomputed from what the section carries.
///
/// A whole-cluster resample of a difference in differences refers to the clusters and the
/// completed replicates of the block the effect was summarized in; the section carries the
/// clusters and the first block's count, so an effect of a later block is accepted at any
/// completed count up to the requested replicates. A calendar-time local ratio carries its
/// degrees of freedom as the diagnostic [`STUDENT_REFERENCE_DIAGNOSTIC`]. Every other
/// procedure has no Student-t interval.
pub(super) fn student_interval_holds(
    section: &QuasiRolloutWire,
    estimate: f64,
    standard_error: f64,
    interval: [f64; 2],
) -> bool {
    let built_with = |critical: f64| {
        critical.is_finite()
            && STUDENT_ENDPOINT_AGREEMENT.close(interval[0], estimate - critical * standard_error)
            && STUDENT_ENDPOINT_AGREEMENT.close(interval[1], estimate + critical * standard_error)
    };
    match section.uncertainty.as_str() {
        "whole_cluster_bootstrap_refit_iqr_se_pointwise_and_sup_t_band"
        | "whole_cluster_multiplier_refit_iqr_se_pointwise_and_sup_t_band" => {
            let clusters = rollout_count(section, "clusters");
            let critical = |replicates: u32| cluster_reference_critical(clusters, replicates);
            if built_with(critical(section.replicates_ok)) {
                return true;
            }
            // The critical value falls as replicates complete: find the count whose value is
            // the interval's own, then require the endpoints to be built with it.
            let implied = (interval[1] - interval[0]) / (2.0 * standard_error);
            let mut low = CLUSTER_REFERENCE_MIN_REPLICATES;
            let mut high = section.replicates_requested.max(low);
            while low < high {
                let middle = low + (high - low) / 2;
                if critical(middle) > implied {
                    low = middle + 1;
                } else {
                    high = middle;
                }
            }
            [low.saturating_sub(1), low, low.saturating_add(1)]
                .into_iter()
                .filter(|count| *count <= section.replicates_requested)
                .any(|count| built_with(critical(count)))
        }
        "within_side_moving_block_bootstrap_refit_iqr_se_student_t_interval" => section
            .diagnostic(STUDENT_REFERENCE_DIAGNOSTIC)
            .filter(|degrees| degrees.is_finite() && *degrees > 0.0)
            .is_some_and(|degrees| built_with(antecedent_stats::student_t_ppf(0.975, degrees))),
        _ => false,
    }
}
