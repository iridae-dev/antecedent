//! Exact graphless design-family support predicate.
//!
//! The support table is generated from `parity/support_graphless.toml` by
//! `scripts/generate_graphless_support.py`. This module is the single owner of
//! the predicate: the facade's checked execution and artifact verification
//! both classify through [`classify_graphless`] / [`graphless_admits`], so the
//! two cannot drift apart.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "support_graphless_data.rs"]
mod data;

use data::GraphlessSupportRow;
use data::SUPPORTS;

/// Exact graphless design, method, and inference claim to classify.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GraphlessSupportKey<'a> {
    /// Query family, such as `randomized_effect`.
    pub family: &'a str,
    /// Assignment design, such as `complete` or `cluster`.
    pub design: &'a str,
    /// Concrete estimator method; similar methods cannot inherit support.
    pub method: &'a str,
    /// Exact inference claim, including confidence level and pointwise scope.
    pub inference_claim: &'a str,
}

/// Graphless key for a sequential-DR interval at a declared decision horizon.
#[must_use]
pub const fn sequential_dr_key(periods: usize) -> Option<GraphlessSupportKey<'static>> {
    let design = match periods {
        2 => "known_sequential_randomized_two_period",
        3 => "known_sequential_randomized_three_period",
        4 => "known_sequential_randomized_four_period",
        _ => return None,
    };
    Some(GraphlessSupportKey {
        family: "longitudinal_regime",
        design,
        method: "subject_excluded_q_sequential_dr_scores",
        inference_claim: "conditional_q_pointwise_95_normal_interval",
    })
}

/// Observed design and inference support for an exact graphless support.
#[allow(
    clippy::struct_excessive_bools,
    reason = "each flag records a distinct observed support condition; they are not a state enum"
)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GraphlessAssignmentSupport {
    /// `unit`, `cluster`, or independent `sequence`, matched to the supported design.
    pub assignment_unit: &'static str,
    /// Number assigned to the treatment arm.
    pub treated: usize,
    /// Number assigned to the reference arm.
    pub control: usize,
    /// True only after the estimator actually published its 95% interval.
    pub interval_95_published: bool,
    /// Number of analyzed outcome rows.
    pub rows: usize,
    /// Number of separate randomization blocks or independent sequences.
    pub blocks: usize,
    /// Smallest realized assignment count in any block arm.
    pub min_block_arm: usize,
    /// Smallest realized count in any factorial cell.
    pub min_factorial_cell: usize,
    /// Smallest realized count across every named action.
    pub min_action_rows: usize,
    /// Smallest declared probability across every action and row.
    pub min_probability: f64,
    /// Smaller of the expected treated and expected control rows: the arm's summed known
    /// assignment probabilities. Zero when the design declares no per-row probabilities.
    pub min_expected_arm_units: f64,
    /// Number of distinct published pointwise intervals.
    pub reported_intervals: usize,
    /// Number of fitted pre-assignment ANCOVA covariates.
    pub covariates: usize,
    /// Whether every contrast named by the query has an interval.
    pub all_reported_intervals: bool,
    /// Realized matches for the evaluated policy and reference.
    pub policy_matches: usize,
    /// Realized reference matches.
    pub reference_matches: usize,
    /// Smallest held-out rank bin and observed arm count within a bin.
    pub min_bin_rows: usize,
    /// Smallest observed arm count within a held-out rank bin.
    pub min_bin_arm_rows: usize,
    /// Smallest fixed baseline group and observed compared-arm count.
    pub min_group_rows: usize,
    /// Smallest compared-arm count within a fixed baseline group.
    pub min_group_arm_rows: usize,
    /// Smallest local kernel window across fixed dose-policy targets.
    pub min_local_rows: usize,
    /// Smallest effective sample size across fixed dose-policy windows.
    pub min_effective_sample_size: f64,
    /// Largest normalized kernel weight across fixed dose-policy windows.
    pub max_normalized_weight: f64,
    /// Smallest observed-dose density across fixed dose-policy windows.
    pub min_dose_density: f64,
    /// Caller declares the supplied observed-dose density known, not fitted here.
    pub known_density: bool,
    /// Global action constraints do not couple subject recommendations.
    pub uncoupled_constraints: bool,
    /// Outcome nuisance predictions use declared disjoint training subjects.
    pub disjoint_nuisance_training: bool,
    /// Ranking training IDs are declared and disjoint from evaluation IDs.
    pub rank_ownership: bool,
    /// Every independent sequence has the same observed number of periods.
    pub balanced_sequences: bool,
}

/// Runtime status for one exact graphless support key.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GraphlessSupportStatus {
    /// Exact support and assignment support matched.
    Supported {
        /// Scope restrictions published with the support.
        limitations: &'static str,
        /// Known-truth calibration test used for the row.
        known_truth_test: &'static str,
        /// Retained result/artifact route test used for the row.
        retained_route_test: &'static str,
    },
    /// No exact row or too little independent assignment support.
    Refused,
}

/// Derived support status of the graphless rows (`data/unit_support_status.json`), written by
/// `scripts/generate_support_matrix_docs.py` from the evidence registries and the
/// scientific-records ledger. A row whose status is not `supported` is not admitted.
#[doc(hidden)]
pub const UNIT_STATUS: &str = include_str!("../data/unit_support_status.json");

#[derive(serde::Deserialize)]
struct UnitStatusFile {
    graphless: Vec<GraphlessUnitStatus>,
}

#[derive(serde::Deserialize)]
struct GraphlessUnitStatus {
    family: String,
    design: String,
    method: String,
    inference_claim: String,
    status: String,
    #[serde(default)]
    missing_evidence: Vec<String>,
}

type GraphlessKey = (String, String, String, String);

/// The `(family, design, method, inference_claim)` keys whose derived status is not
/// `supported`, each with the evidence the snapshot lists as missing.
fn missing_rows(status: &str) -> std::collections::HashMap<GraphlessKey, &'static [&'static str]> {
    let file: UnitStatusFile = serde_json::from_str(status).expect("graphless unit support status");
    file.graphless
        .into_iter()
        .filter(|row| row.status != "supported")
        .map(|row| {
            let evidence: &'static [&'static str] = Box::leak(
                row.missing_evidence
                    .into_iter()
                    .map(|item| &*Box::leak(item.into_boxed_str()))
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            );
            ((row.family, row.design, row.method, row.inference_claim), evidence)
        })
        .collect()
}

fn missing_table() -> &'static std::collections::HashMap<GraphlessKey, &'static [&'static str]> {
    static TABLE: std::sync::OnceLock<
        std::collections::HashMap<GraphlessKey, &'static [&'static str]>,
    > = std::sync::OnceLock::new();
    TABLE.get_or_init(|| missing_rows(UNIT_STATUS))
}

/// The snapshot's missing evidence for a graphless row it marks `not_supported`.
#[must_use]
pub fn graphless_missing_evidence(key: GraphlessSupportKey<'_>) -> &'static [&'static str] {
    missing_table()
        .get(&(
            key.family.to_owned(),
            key.design.to_owned(),
            key.method.to_owned(),
            key.inference_claim.to_owned(),
        ))
        .copied()
        .unwrap_or(&[])
}

/// The supported row that admits `key` under `observed`, if any.
fn find_support(
    key: GraphlessSupportKey<'_>,
    observed: GraphlessAssignmentSupport,
) -> Option<&'static GraphlessSupportRow> {
    let owed = missing_table();
    SUPPORTS.iter().find(|row| {
        row.admits(key, observed)
            && !owed.contains_key(&(
                row.family.to_owned(),
                row.design.to_owned(),
                row.method.to_owned(),
                row.inference_claim.to_owned(),
            ))
    })
}

/// Whether some supported row matches `key` and is satisfied by `observed`.
#[must_use]
pub fn graphless_admits(
    key: GraphlessSupportKey<'_>,
    observed: GraphlessAssignmentSupport,
) -> bool {
    find_support(key, observed).is_some()
}

/// Classify a graphless design against the separate machine-readable table.
/// This lookup cannot grant a geometric support-matrix cell by analogy.
#[must_use]
pub fn classify_graphless(
    key: GraphlessSupportKey<'_>,
    observed: GraphlessAssignmentSupport,
) -> GraphlessSupportStatus {
    find_support(key, observed).map_or(GraphlessSupportStatus::Refused, |row| {
        GraphlessSupportStatus::Supported {
            limitations: row.limitations,
            known_truth_test: row.known_truth_test,
            retained_route_test: row.retained_route_test,
        }
    })
}

/// Support key of the independent-factor Horvitz–Thompson 2×2 reading of a four-cell multi-arm
/// randomized design: the factor-A and factor-B main effects and their interaction with
/// Young-inequality variance bounds. Execution, contract emission and artifact verification
/// all classify that result under this key.
pub const FACTORIAL_BERNOULLI_KEY: GraphlessSupportKey<'static> = GraphlessSupportKey {
    family: "randomized_effect",
    design: "factorial_2x2_bernoulli",
    method: "independent_factor_ht_young_bound_contrasts",
    inference_claim: "three_pointwise_95_normal_intervals",
};

/// Assignment-design label of a randomized result read as independent-factor 2×2 contrasts.
pub const FACTORIAL_BERNOULLI_DESIGN: &str = "factorial_2x2_bernoulli";

/// Smallest observed risk set at any event in the delayed-entry
/// target-cause cumulative incidence support rows.
#[doc(hidden)]
pub const INCIDENCE_MIN_EVENT_RISK_SET: usize = 25;

/// Minimum observed subjects per arm for a delayed-entry incidence interval.
#[doc(hidden)]
#[must_use]
pub const fn incidence_min_arm(fixed_known_censoring: bool) -> usize {
    if fixed_known_censoring { 200 } else { 160 }
}

/// Release scope of the standalone delayed-entry pointwise incidence interval.
///
/// The registered designs use 299 arm-stratified draws. Requests that reach the
/// band threshold enter the separate whole-grid procedure, even if its band is
/// later withheld. Both observed event and restriction-horizon risk sets must
/// reach the delayed-entry risk-set floor.
#[doc(hidden)]
#[must_use]
pub fn delayed_entry_incidence_support_profile(
    event_risk: Option<usize>,
    horizon_risk: Option<usize>,
    requested: Option<u32>,
    completed: Option<u32>,
) -> bool {
    event_risk.is_some_and(|risk| risk >= INCIDENCE_MIN_EVENT_RISK_SET)
        && horizon_risk.is_some_and(|risk| risk >= INCIDENCE_MIN_EVENT_RISK_SET)
        && requested.is_some_and(|requested| {
            requested < antecedent_stats::survival::SURVIVAL_BAND_MIN_REPLICATES
        })
        && completed.is_some_and(|completed| completed >= 299)
}

/// Calibration target for the audited delayed-entry incidence study.
///
/// The benchmark uses 299 requested and completed draws at three balanced arm
/// sizes. The weighted benchmark uses the fixed six-knot censoring grid and
/// its recorded positivity floor. These are narrower than the executable
/// graphless support conditions. The study covers all attempted draws of the
/// data-generating law, with withheld publications counted as misses. It does
/// not measure coverage conditional on a particular realized risk-set size.
#[doc(hidden)]
#[must_use]
pub fn incidence_calibration_functional(
    query: &crate::query_wire::SurvivalQueryWire,
    minimum_event_risk_set: Option<usize>,
    minimum_horizon_risk_set: Option<usize>,
    replicates_requested: Option<u32>,
    replicates_ok: Option<u32>,
    assignment_counts: Option<[usize; 2]>,
) -> Option<String> {
    if query.stratified_censoring.is_some() {
        return None;
    }
    let cause = query.target_cause?;
    query.delayed_entry?;
    let fixed_known = !query.censoring_columns.is_empty();
    let censoring = if fixed_known { "fixed_known" } else { "unweighted" };
    let minimum_arm = incidence_min_arm(fixed_known);
    let supported = delayed_entry_incidence_support_profile(
        minimum_event_risk_set,
        minimum_horizon_risk_set,
        replicates_requested,
        replicates_ok,
    ) && assignment_counts
        .is_some_and(|[control, treated]| control.min(treated) >= minimum_arm);
    let measured_design = cause == 1
        && query.tau.to_bits() == 3.0_f64.to_bits()
        && query.independent_observation
        && query.independent_given.is_empty()
        && replicates_requested == Some(299)
        && replicates_ok == Some(299)
        && benchmark_balanced_arms(assignment_counts, minimum_arm)
        && (!fixed_known || benchmark_fixed_censoring_grid(query));
    let scope = if !supported {
        "+scope=insufficient"
    } else if measured_design {
        ""
    } else {
        "+scope=unmeasured_design"
    };
    Some(format!(
        "incidence_difference+entry=delayed+censoring={censoring}+cause={cause}+tau={:016x}{scope}",
        query.tau.to_bits(),
    ))
}

const SURVIVAL_BENCHMARK_EVENT_GRID: [f64; 4] = [0.0, 1.0, 2.0, 3.0];
const SURVIVAL_BENCHMARK_CENSORING_GRID: [f64; 6] = [0.0, 1.0, 1.5, 2.0, 2.5, 3.0];

fn benchmark_balanced_arms(assignment_counts: Option<[usize; 2]>, minimum_arm: usize) -> bool {
    assignment_counts.is_some_and(|[control, treated]| {
        control == treated && [minimum_arm, 2 * minimum_arm, 4 * minimum_arm].contains(&control)
    })
}

fn benchmark_fixed_censoring_grid(query: &crate::query_wire::SurvivalQueryWire) -> bool {
    query.censoring_times == SURVIVAL_BENCHMARK_CENSORING_GRID
        && query.censoring_probability_floor.map(f64::to_bits) == Some(0.01_f64.to_bits())
}

/// Calibration functional of a full-grid survival or incidence band. The bare
/// key locates repeated-study evidence on the registered benchmark geometry.
/// That evidence is about the simulated law, which an observed table cannot establish.
#[doc(hidden)]
#[must_use]
pub fn survival_band_calibration_functional(
    query: &crate::query_wire::SurvivalQueryWire,
    assignment_counts: Option<[usize; 2]>,
    requested: Option<u32>,
    completed: u32,
    times: &[f64],
) -> Option<String> {
    let measured_draws = requested == Some(399) && completed == 399;
    if let Some(fit) = &query.stratified_censoring {
        let strata = [fit.entry_stratum, fit.censoring_stratum];
        let measured_design = measured_draws
            && benchmark_balanced_arms(assignment_counts, 1_000)
            && times == SURVIVAL_BENCHMARK_EVENT_GRID
            && query.independent_observation
            && strata[0] != strata[1]
            && query.independent_given.len() == strata.len()
            && strata.iter().all(|id| query.independent_given.contains(id))
            && conditional_survival_band_key(
                query.delayed_entry.is_some(),
                query.target_cause.is_some(),
                query.tau,
                fit.population_weights,
                fit.minimum_probability,
            )
            .is_some();
        let scope = if assignment_counts.is_none_or(|[a, b]| a.min(b) < 1_000) {
            "+scope=insufficient"
        } else if measured_design {
            ""
        } else {
            "+scope=unmeasured_design"
        };
        let entry = if query.delayed_entry.is_some() { "standardized" } else { "none" };
        let endpoint = query
            .target_cause
            .map_or_else(|| "survival".to_owned(), |cause| format!("incidence:{cause}"));
        return Some(format!(
            "difference_band+entry={entry}+censoring=stratified_refit+endpoint={endpoint}+tau={:016x}+population={:016x}/{:016x}+floor={:016x}{scope}",
            query.tau.to_bits(),
            fit.population_weights[0].to_bits(),
            fit.population_weights[1].to_bits(),
            fit.minimum_probability.to_bits(),
        ));
    }
    query.delayed_entry?;
    let fixed_known = !query.censoring_columns.is_empty();
    let censoring = if fixed_known { "fixed_known" } else { "unweighted" };
    let endpoint =
        query.target_cause.map_or_else(|| "survival".into(), |cause| format!("incidence:{cause}"));
    let minimum_arm = incidence_min_arm(fixed_known);
    let measured_design = measured_draws
        && benchmark_balanced_arms(assignment_counts, minimum_arm)
        && query.tau.to_bits() == 3.0_f64.to_bits()
        && matches!(query.target_cause, None | Some(1))
        && query.independent_observation
        && query.independent_given.is_empty()
        && if fixed_known {
            times == SURVIVAL_BENCHMARK_CENSORING_GRID && benchmark_fixed_censoring_grid(query)
        } else {
            times == SURVIVAL_BENCHMARK_EVENT_GRID
        };
    let scope = if assignment_counts.is_none_or(|[a, b]| a.min(b) < minimum_arm) {
        "+scope=insufficient"
    } else if measured_design {
        ""
    } else {
        "+scope=unmeasured_design"
    };
    Some(format!(
        "difference_band+entry=delayed+censoring={censoring}+endpoint={endpoint}+tau={:016x}{scope}",
        query.tau.to_bits(),
    ))
}

/// Coordinate earned for the pre-entry standardized survival band with an owned censoring fit.
#[doc(hidden)]
pub const STANDARDIZED_CONDITIONAL_SURVIVAL_BAND: GraphlessSupportKey<'static> =
    GraphlessSupportKey {
        family: "survival",
        design: "binary_entry_and_censoring_strata_two_arm_individual_randomized",
        method: "arm_stratified_subject_bootstrap_stratified_censoring_refit",
        inference_claim: "simultaneous_95_standardized_survival_difference_band",
    };

/// The initial standardized release scope with a fixed horizon, target and censoring floor.
#[doc(hidden)]
pub fn conditional_survival_band_key(
    delayed_entry: bool,
    incidence: bool,
    tau: f64,
    population_weights: [f64; 2],
    censoring_floor: f64,
) -> Option<GraphlessSupportKey<'static>> {
    (delayed_entry
        && !incidence
        && tau.to_bits() == 3.0_f64.to_bits()
        && population_weights.map(f64::to_bits) == [0.5_f64.to_bits(); 2]
        && censoring_floor.to_bits() == 0.05_f64.to_bits())
    .then_some(STANDARDIZED_CONDITIONAL_SURVIVAL_BAND)
}

/// Exact scalar-interval claim of a randomized survival query.
#[doc(hidden)]
#[must_use]
pub const fn survival_pointwise_key(
    delayed_entry: bool,
    fixed_known_censoring: bool,
    competing_incidence: bool,
) -> Option<GraphlessSupportKey<'static>> {
    if competing_incidence && !delayed_entry {
        return None;
    }
    Some(GraphlessSupportKey {
        family: "survival",
        design: if delayed_entry {
            "delayed_entry_two_arm_individual_randomized"
        } else {
            "two_arm_individual_randomized"
        },
        method: if fixed_known_censoring {
            "arm_stratified_subject_bootstrap_ipcw_product_limit"
        } else {
            "arm_stratified_subject_bootstrap_product_limit"
        },
        inference_claim: if competing_incidence {
            "cumulative_incidence_difference_at_horizon_pointwise_95_percentile_interval"
        } else {
            "rmst_and_horizon_survival_pointwise_95_percentile_intervals"
        },
    })
}

/// Exact full-grid band claim of a randomized survival query.
#[doc(hidden)]
#[must_use]
pub const fn survival_band_key(
    delayed_entry: bool,
    fixed_known_censoring: bool,
    competing_incidence: bool,
) -> Option<GraphlessSupportKey<'static>> {
    if fixed_known_censoring && !delayed_entry {
        return None;
    }
    Some(GraphlessSupportKey {
        family: "survival",
        design: if delayed_entry {
            "delayed_entry_two_arm_individual_randomized"
        } else {
            "two_arm_individual_randomized"
        },
        method: if fixed_known_censoring {
            "arm_stratified_subject_bootstrap_ipcw_supremum_band"
        } else {
            "arm_stratified_subject_bootstrap_supremum_band"
        },
        inference_claim: if competing_incidence {
            "simultaneous_95_cumulative_incidence_difference_band"
        } else {
            "simultaneous_95_survival_difference_band"
        },
    })
}

/// Fewest observed rows in every cell before the three pointwise intervals are published.
pub const FACTORIAL_BERNOULLI_MIN_CELL_ROWS: usize = 30;

/// Uncertainty label of the independent-factor 2×2 reading with its three published intervals.
pub const FACTORIAL_BERNOULLI_INTERVAL_UNCERTAINTY: &str =
    "factorial_bernoulli_ht_young_bound_pointwise_normal_intervals";

/// Uncertainty label of the independent-factor 2×2 reading reported as points and bounds only.
pub const FACTORIAL_BERNOULLI_POINT_UNCERTAINTY: &str =
    "factorial_bernoulli_ht_young_variance_bound_no_interval";

/// Code of the diagnostic a same-sample (estimated-slope) CUPED result carries: the
/// estimator's own diagnostic, whose structured fields state the covariate the adjustment read
/// (a scientific premise: it must be pre-assignment). Execution writes the fields and artifact
/// verification requires them.
pub const ESTIMATED_CUPED_DIAGNOSTIC: &str = "estimate.randomized.estimated_cuped_ht_itt";

/// Field of the [`ESTIMATED_CUPED_DIAGNOSTIC`] that holds the covariate's variable id, in
/// decimal.
pub const ESTIMATED_CUPED_COVARIATE_FIELD: &str = "covariate";

/// The structured fields of the [`ESTIMATED_CUPED_DIAGNOSTIC`] of an adjustment by the
/// covariate with variable id `covariate`.
#[must_use]
pub fn estimated_cuped_fields(covariate: u32) -> Vec<(String, String)> {
    vec![(ESTIMATED_CUPED_COVARIATE_FIELD.to_owned(), covariate.to_string())]
}

/// Code of the diagnostic by which a Bayesian design posterior states the support row it ran
/// under. Such a result has no design section of its own, so the row key and the observed
/// support ride in the diagnostic's fields.
pub const DESIGN_POSTERIOR_ROW_CODE: &str = "estimate.design_posterior.graphless_support";

/// Codes of the diagnostics by which a result with a design section states the support row
/// the section meets. The section's own `supported` field stays `false`; the artifact check
/// recomputes the row from the stored query and section for every such diagnostic.
pub const ASSOCIATION_SLOPE_ROW_CODE: &str = "estimate.association.graphless_support";
/// Sharp instrument bounds: the interval for the endpoints of the identified set.
pub const IV_BOUNDS_ROW_CODE: &str = "estimate.iv_bounds.graphless_support";
/// Sequential contrast: the normal interval of the cross-fit scores.
pub const CROSSFIT_CONTRAST_ROW_CODE: &str = "estimate.sequential_contrast.graphless_support";
/// History match: the fixed-b period interval.
pub const PERIOD_HISTORY_ROW_CODE: &str = "estimate.history_match.graphless_support";
/// Temporal design: the pointwise intervals of the design's coordinates.
pub const TEMPORAL_DESIGN_ROW_CODE: &str = "estimate.temporal_design.graphless_support";
/// Augmented panel difference in differences: the whole-cluster bootstrap interval.
pub const AUGMENTED_DID_ROW_CODE: &str = "estimate.panel_did.augmented.graphless_support";
/// Every code by which a design section's support row is stated.
pub const SECTION_ROW_CODES: [&str; 6] = [
    ASSOCIATION_SLOPE_ROW_CODE,
    IV_BOUNDS_ROW_CODE,
    CROSSFIT_CONTRAST_ROW_CODE,
    PERIOD_HISTORY_ROW_CODE,
    TEMPORAL_DESIGN_ROW_CODE,
    AUGMENTED_DID_ROW_CODE,
];

/// The fields of the [`DESIGN_POSTERIOR_ROW_CODE`] diagnostic for `key` under `observed`.
#[must_use]
pub fn design_posterior_row_statement(
    key: GraphlessSupportKey<'_>,
    observed: GraphlessAssignmentSupport,
    resampled: GraphlessResampleSupport,
) -> Vec<(String, String)> {
    let text = |name: &str, value: &str| (name.to_owned(), value.to_owned());
    let count = |name: &str, value: usize| (name.to_owned(), value.to_string());
    let real = |name: &str, value: f64| (name.to_owned(), format!("{value:?}"));
    let flag = |name: &str, value: bool| (name.to_owned(), value.to_string());
    let mut fields = vec![
        text("family", key.family),
        text("design", key.design),
        text("method", key.method),
        text("inference_claim", key.inference_claim),
        text("assignment_unit", observed.assignment_unit),
        count("treated", observed.treated),
        count("control", observed.control),
        count("rows", observed.rows),
        count("blocks", observed.blocks),
        count("min_action_rows", observed.min_action_rows),
        count("min_group_rows", observed.min_group_rows),
        count("min_local_rows", observed.min_local_rows),
        count("reported_intervals", observed.reported_intervals),
        real("min_probability", observed.min_probability),
        real("min_expected_arm_units", observed.min_expected_arm_units),
        flag("all_reported_intervals", observed.all_reported_intervals),
        flag("known_density", observed.known_density),
        flag("balanced_sequences", observed.balanced_sequences),
        real("first_stage_t", resampled.first_stage_t),
        count("interval_resamples", resampled.interval_resamples),
    ];
    // Policy and reference matches are stated only by a design that counts them.
    if observed.policy_matches > 0 || observed.reference_matches > 0 {
        fields.push(count("policy_matches", observed.policy_matches));
        fields.push(count("reference_matches", observed.reference_matches));
    }
    fields
}

/// Whether the row and observed support a [`DESIGN_POSTERIOR_ROW_CODE`] diagnostic states are
/// admitted by the support table. A missing or unreadable field is not admitted.
#[must_use]
pub fn design_posterior_row_stated(fields: &[(String, String)]) -> bool {
    let get = |name: &str| {
        fields.iter().find(|(field, _)| field == name).map(|(_, value)| value.as_str())
    };
    let count = |name: &str| get(name).and_then(|value| value.parse::<usize>().ok());
    let real = |name: &str| {
        get(name).and_then(|value| value.parse::<f64>().ok()).filter(|value| value.is_finite())
    };
    let flag = |name: &str| get(name).and_then(|value| value.parse::<bool>().ok());
    let assignment_unit = match get("assignment_unit") {
        Some("unit") => "unit",
        Some("cluster") => "cluster",
        Some("sequence") => "sequence",
        _ => return false,
    };
    let (Some(family), Some(design), Some(method), Some(inference_claim)) =
        (get("family"), get("design"), get("method"), get("inference_claim"))
    else {
        return false;
    };
    let observed = (|| {
        Some(GraphlessAssignmentSupport {
            assignment_unit,
            treated: count("treated")?,
            control: count("control")?,
            rows: count("rows")?,
            blocks: count("blocks")?,
            min_action_rows: count("min_action_rows")?,
            min_group_rows: count("min_group_rows")?,
            min_local_rows: count("min_local_rows")?,
            reported_intervals: count("reported_intervals")?,
            min_probability: real("min_probability")?,
            min_expected_arm_units: real("min_expected_arm_units")?,
            all_reported_intervals: flag("all_reported_intervals")?,
            known_density: flag("known_density")?,
            balanced_sequences: flag("balanced_sequences")?,
            policy_matches: count("policy_matches").unwrap_or(0),
            reference_matches: count("reference_matches").unwrap_or(0),
            interval_95_published: true,
            ..Default::default()
        })
    })();
    // Absent on a statement of a row without these thresholds.
    let resampled = GraphlessResampleSupport {
        first_stage_t: real("first_stage_t").unwrap_or(0.0),
        interval_resamples: count("interval_resamples").unwrap_or(0),
    };
    observed.is_some_and(|observed| {
        graphless_admits_resampled(
            GraphlessSupportKey { family, design, method, inference_claim },
            observed,
            resampled,
        )
    })
}

impl GraphlessSupportRow {
    /// Whether this row matches `key` and every threshold is met by `observed`.
    fn admits(&self, key: GraphlessSupportKey<'_>, observed: GraphlessAssignmentSupport) -> bool {
        let row = self;
        row.family == key.family
            && row.design == key.design
            && row.method == key.method
            && row.inference_claim == key.inference_claim
            && row.assignment_unit == observed.assignment_unit
            && observed.treated >= row.min_assignment_units_per_arm
            && observed.control >= row.min_assignment_units_per_arm
            && observed.interval_95_published
            && observed.rows >= row.min_rows
            && observed.blocks >= row.min_blocks
            && observed.min_block_arm >= row.min_block_arm
            && observed.min_factorial_cell >= row.min_factorial_cell
            && observed.min_action_rows >= row.min_action_rows
            && observed.min_probability + 1e-12 >= row.min_probability
            && observed.min_expected_arm_units + 1e-9 >= row.min_expected_arm_units
            && observed.reported_intervals >= row.min_reported_intervals
            && (row.max_covariates == 0 || observed.covariates <= row.max_covariates)
            && (!row.all_reported_intervals || observed.all_reported_intervals)
            && observed.policy_matches >= row.min_policy_matches
            && observed.reference_matches >= row.min_reference_matches
            && observed.min_bin_rows >= row.min_bin_rows
            && observed.min_bin_arm_rows >= row.min_bin_arm_rows
            && observed.min_group_rows >= row.min_group_rows
            && observed.min_group_arm_rows >= row.min_group_arm_rows
            && observed.min_local_rows >= row.min_local_rows
            && observed.min_effective_sample_size + 1e-12 >= row.min_effective_sample_size
            && (row.max_normalized_weight == 0.0
                || observed.max_normalized_weight <= row.max_normalized_weight + 1e-12)
            && observed.min_dose_density + 1e-12 >= row.min_dose_density
            && (!row.requires_uncoupled_constraints || observed.uncoupled_constraints)
            && (!row.requires_disjoint_nuisance_training || observed.disjoint_nuisance_training)
            && (!row.requires_rank_ownership || observed.rank_ownership)
            && (!row.requires_balanced_sequences || observed.balanced_sequences)
            && (!row.requires_known_density || observed.known_density)
    }
}

/// Observed resampling support of the rows that carry a resampling threshold: a first-stage
/// strength or a count of interval resamples. It travels beside
/// [`GraphlessAssignmentSupport`]; such rows are looked up through
/// [`graphless_admits_resampled`] only.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GraphlessResampleSupport {
    /// Absolute first-stage estimate over its standard error (infinite for a first stage
    /// without error); zero when the design has no first stage.
    pub first_stage_t: f64,
    /// Bootstrap replicates or posterior draws that produced the published interval.
    pub interval_resamples: usize,
}

/// Whether the resampling thresholds of `row` are met by `resampled`.
fn row_meets_resampling(row: &GraphlessSupportRow, resampled: GraphlessResampleSupport) -> bool {
    resampled.first_stage_t >= row.min_first_stage_t
        && resampled.interval_resamples >= row.min_interval_resamples
}

/// Whether some supported row matches `key`, is satisfied by `observed`, and has its
/// resampling thresholds met by `resampled`.
#[must_use]
pub fn graphless_admits_resampled(
    key: GraphlessSupportKey<'_>,
    observed: GraphlessAssignmentSupport,
    resampled: GraphlessResampleSupport,
) -> bool {
    find_support(key, observed).is_some_and(|row| row_meets_resampling(row, resampled))
}

#[cfg(test)]
mod survival_calibration_tests {
    use super::{
        SURVIVAL_BENCHMARK_CENSORING_GRID, SURVIVAL_BENCHMARK_EVENT_GRID,
        incidence_calibration_functional, survival_band_calibration_functional,
    };
    use crate::query_wire::{StratifiedCensoringWire, SurvivalQueryWire};

    fn query() -> SurvivalQueryWire {
        SurvivalQueryWire {
            duration: 0,
            event: 1,
            treatment: 2,
            tau: 3.0,
            delayed_entry: Some(3),
            target_cause: Some(1),
            independent_observation: true,
            independent_given: Vec::new(),
            censoring_times: Vec::new(),
            censoring_columns: Vec::new(),
            censoring_probability_floor: None,
            stratified_censoring: None,
        }
    }

    #[test]
    fn survival_bands_calibrate_only_at_the_exact_benchmark_geometry() {
        let mut query = query();
        let event_grid = &SURVIVAL_BENCHMARK_EVENT_GRID;
        let censoring_grid = &SURVIVAL_BENCHMARK_CENSORING_GRID;
        let band = |query: &SurvivalQueryWire, counts, requested, completed, times: &[f64]| {
            survival_band_calibration_functional(
                query,
                Some(counts),
                Some(requested),
                completed,
                times,
            )
            .unwrap()
        };
        for n in [160, 320, 640] {
            assert!(!band(&query, [n, n], 399, 399, event_grid).contains("+scope="));
        }
        for function in [
            band(&query, [159, 161], 399, 399, event_grid),
            band(&query, [200, 200], 399, 399, event_grid),
            band(&query, [319, 321], 399, 399, event_grid),
            band(&query, [320, 320], 400, 399, event_grid),
            band(&query, [320, 320], 399, 398, event_grid),
            band(&query, [320, 320], 399, 399, &[0.0, 1.0, 1.75, 3.0]),
        ] {
            assert!(function.contains("+scope="), "{function}");
        }

        query.target_cause = Some(2);
        let wrong_cause = incidence_calibration_functional(
            &query,
            Some(25),
            Some(25),
            Some(299),
            Some(299),
            Some([320; 2]),
        );
        assert!(wrong_cause.unwrap().ends_with("+scope=unmeasured_design"));
        query.target_cause = Some(1);
        query.tau = 2.0;
        let wrong_horizon = incidence_calibration_functional(
            &query,
            Some(25),
            Some(25),
            Some(299),
            Some(299),
            Some([320; 2]),
        );
        assert!(wrong_horizon.unwrap().ends_with("+scope=unmeasured_design"));
        query.tau = 3.0;

        query.censoring_times = censoring_grid.to_vec();
        query.censoring_columns = (4..10).collect();
        query.censoring_probability_floor = Some(0.01);
        assert!(!band(&query, [200; 2], 399, 399, censoring_grid).contains("+scope="));
        query.censoring_probability_floor = Some(0.05);
        assert!(
            band(&query, [200; 2], 399, 399, censoring_grid).ends_with("+scope=unmeasured_design")
        );

        query.censoring_times.clear();
        query.censoring_columns.clear();
        query.censoring_probability_floor = None;
        query.target_cause = None;
        query.independent_given = vec![4, 5];
        query.stratified_censoring = Some(StratifiedCensoringWire {
            entry_stratum: 4,
            censoring_stratum: 5,
            population_weights: [0.5; 2],
            minimum_probability: 0.05,
        });
        for n in [1_000, 2_000, 4_000] {
            assert!(!band(&query, [n; 2], 399, 399, event_grid).contains("+scope="));
        }
        for function in [
            band(&query, [1_999, 2_001], 399, 399, event_grid),
            band(&query, [2_000; 2], 400, 400, event_grid),
        ] {
            assert!(function.ends_with("+scope=unmeasured_design"), "{function}");
        }
        query.independent_given = vec![4];
        assert!(
            band(&query, [2_000; 2], 399, 399, event_grid).ends_with("+scope=unmeasured_design")
        );
    }
}

#[cfg(test)]
mod status_tests {
    use super::missing_rows;

    /// A row that release-mode data marks `not_supported` is excluded from admission;
    /// a `supported` row is not.
    #[test]
    fn a_not_supported_graphless_row_in_the_status_data_is_excluded() {
        let row = |method: &str, status: &str| {
            format!(
                r#"{{"family":"f","design":"d","method":"{method}","inference_claim":"c","status":"{status}","missing_evidence":[]}}"#
            )
        };
        let named = row("named", "not_supported")
            .replace(r#""missing_evidence":[]"#, r#""missing_evidence":["flagged: x"]"#);
        let json = format!(
            r#"{{"graphless":[{},{},{named}],"capabilities":[]}}"#,
            row("owed", "not_supported"),
            row("current", "supported")
        );
        let owed = missing_rows(&json);
        assert!(owed.contains_key(&("f".into(), "d".into(), "owed".into(), "c".into())));
        assert!(!owed.contains_key(&("f".into(), "d".into(), "current".into(), "c".into())));
        assert_eq!(owed[&("f".into(), "d".into(), "named".into(), "c".into())], ["flagged: x"]);
    }
}
