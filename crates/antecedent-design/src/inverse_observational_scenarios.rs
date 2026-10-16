//! Observational scenarios compatible with an outcome: `P(X in scenario | Y event)`.
//!
//! The finite-action inverse query asks which *interventions* reach a target.
//! A different question is what the units that reach an outcome look like: of
//! the observed units whose outcome meets an event, which declared covariate
//! scenarios do they fall in, and how likely is the event inside each scenario.
//! That is Bayes inversion of an observational joint law, and it is coherent
//! and answerable from a supplied aligned law; it is not an action, and this
//! module never presents it as one.
//!
//! # Contract
//!
//! - **Source.** An aligned joint law over the outcome and the scenario
//!   coordinates whose meaning is observational (`PosteriorPredictive` or
//!   `EmpiricalOutcome`). A law with interventional meaning answers an
//!   intervention, not an observed population, and is refused; independent
//!   marginals carry no pairing between the outcome and a covariate and are
//!   refused (`joint_law_required`).
//! - **Scenarios.** A finite list of declared cells, each a conjunction of
//!   closed intervals on named coordinates (a discrete value is a degenerate
//!   interval). Scenarios may overlap; none is required to cover the support;
//!   a scenario may not condition on the outcome itself.
//! - **Event.** `Y >= t` or `Y <= t`, the atom at `t` included.
//! - **Values.** For each scenario `S` and event `E`, with normalized draw
//!   weights: `P(S)`, `P(S and E)`, `P(E | S)`, `P(S | E)` and the lift
//!   `P(E | S) / P(E)`. The mass of the event no declared scenario explains is
//!   reported, never renormalized away.
//!
//! # What this is not
//!
//! The result is associational. `P(X | Y)` is not the effect of changing `X`,
//! it ranks no action and recommends none (`ranks_actions` is `false`), and it
//! carries no interval: a sampled law gives a Monte-Carlo ratio with no
//! calibrated error, an exact finite law gives exact arithmetic that supplies
//! no coverage claim. The effective number of event draws (Kish) is reported so
//! a thin conditioning set is visible.
//!
//! Every request is refused with a typed reason when the source cannot answer
//! it, and in particular when the event has no mass in the supplied law (the
//! conditional is then undefined).
//!
//! # Ownership
//!
//! The law is borrowed for one call; the result owns its numbers.
//!
//! # Refresh
//!
//! Refresh with outcome-only data. The scenarios are a function of the law
//! supplied at the call. New outcome-only data is a new aligned law (a new
//! distribution artifact) and the scenarios are recomputed from it; the
//! declared scenario cells and the event are kept. A law of another population
//! or meaning is refused (an interventional law, or independent marginals under
//! `joint_law_required`), and
//! nothing is cached or refreshed in place.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashSet;

use antecedent_contract::{DistributionMeaning, ExternalRefusal, ScientificQuantity};
use antecedent_core::CanonicalWriter;
use antecedent_io_core::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DrawAlignment,
};

use crate::signal::RefusalBuilder;

/// Most scenarios one query may declare.
pub const MAX_OBSERVATIONAL_SCENARIOS: usize = 1024;
/// Version word of the result identity encoding.
pub const OBSERVATIONAL_SCENARIO_VERSION: u64 = 1;

/// What a result is and is not, carried on every result.
pub const OBSERVATIONAL_SCENARIO_SCOPE_NOTE: &str = "Associational: P(scenario | outcome event) and P(event | scenario) under the supplied observational law. Not the effect of changing a covariate, not an action, ranked or recommended by nothing here; point values with no interval and no calibration claim.";

const STAGE: &str = "inverse_observational_scenarios";
const TAG: &str = "antecedent.inverse_observational_scenarios.result";

/// Detail: no scenario was declared, or a scenario identity is blank or repeated.
pub const DETAIL_SCENARIOS: &str = "inverse_observational_scenarios.invalid_scenarios";
/// Detail: a threshold, bound or probability is not finite or is inverted.
pub const DETAIL_PARAMETER: &str = "inverse_observational_scenarios.invalid_parameter";
/// Detail: a scenario conditions on the outcome.
pub const DETAIL_CONDITIONS_ON_OUTCOME: &str =
    "inverse_observational_scenarios.condition_on_outcome";
/// Detail: a coordinate is not in the law.
pub const DETAIL_COORDINATE_MISSING: &str = "inverse_observational_scenarios.coordinate_missing";
/// Detail: a coordinate is masked as unsupported.
pub const DETAIL_COORDINATE_UNSUPPORTED: &str =
    "inverse_observational_scenarios.coordinate_unsupported";
/// Detail: the law's rows are not paired joint realizations.
pub const DETAIL_JOINT_LAW: &str = "inverse_observational_scenarios.joint_law_required";
/// Detail: the law is not observational.
pub const DETAIL_MEANING: &str = "inverse_observational_scenarios.meaning_not_observational";
/// Detail: the event has no mass in the supplied law.
pub const DETAIL_EVENT_MASS: &str = "inverse_observational_scenarios.event_has_no_mass";
/// Detail: a draw is not finite.
pub const DETAIL_DRAWS: &str = "inverse_observational_scenarios.non_finite_draw";
/// Detail: a stored result does not replay from its law and query.
pub const DETAIL_REPLAY: &str = "inverse_observational_scenarios.result_mismatch";

/// Which tail of the outcome is the event.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OutcomeEvent {
    /// `Y >= threshold`.
    AtLeast(f64),
    /// `Y <= threshold`.
    AtMost(f64),
}

impl OutcomeEvent {
    fn threshold(self) -> f64 {
        match self {
            Self::AtLeast(t) | Self::AtMost(t) => t,
        }
    }

    fn holds(self, value: f64) -> bool {
        match self {
            Self::AtLeast(t) => value >= t,
            Self::AtMost(t) => value <= t,
        }
    }

    const fn tag(self) -> u64 {
        match self {
            Self::AtLeast(_) => 1,
            Self::AtMost(_) => 2,
        }
    }
}

/// One closed-interval condition on a coordinate.
#[derive(Clone, Debug, PartialEq)]
pub struct ScenarioCondition {
    /// The coordinate the condition constrains.
    pub coordinate: ScientificQuantity,
    /// Inclusive lower bound.
    pub lower: f64,
    /// Inclusive upper bound; equal to `lower` for a discrete value.
    pub upper: f64,
}

/// One declared scenario: a conjunction of conditions.
#[derive(Clone, Debug, PartialEq)]
pub struct ScenarioCell {
    /// Stable scenario identity.
    pub id: String,
    /// Conditions that must all hold.
    pub conditions: Vec<ScenarioCondition>,
}

/// A declared observational-scenario question.
#[derive(Clone, Debug, PartialEq)]
pub struct ObservationalScenarioQuery {
    /// The outcome coordinate.
    pub outcome: ScientificQuantity,
    /// The outcome event the scenarios are compatible with.
    pub event: OutcomeEvent,
    /// The finite scenarios.
    pub scenarios: Vec<ScenarioCell>,
}

/// One scenario's values.
#[derive(Clone, Debug, PartialEq)]
pub struct ScenarioReport {
    /// Scenario identity.
    pub id: String,
    /// `P(S)`.
    pub mass: f64,
    /// `P(S and E)`.
    pub joint_mass: f64,
    /// `P(E | S)`; `None` when `P(S) = 0`.
    pub event_given_scenario: Option<f64>,
    /// `P(S | E)`.
    pub scenario_given_event: f64,
    /// `P(E | S) / P(E)`; `None` when `P(S) = 0`.
    pub lift: Option<f64>,
    /// Draws with positive weight inside the scenario.
    pub scenario_draws: u64,
    /// Draws with positive weight inside the scenario and the event.
    pub joint_draws: u64,
}

/// The answer to an observational-scenario question.
#[derive(Clone, Debug, PartialEq)]
pub struct ObservationalScenarioResult {
    /// `P(E)` under the supplied law.
    pub event_mass: f64,
    /// Mass of the event inside at least one declared scenario.
    pub covered_event_mass: f64,
    /// Mass of the event outside every declared scenario: not explained, not
    /// renormalized away.
    pub uncovered_event_mass: f64,
    /// Kish effective number of draws inside the event.
    pub effective_event_draws: f64,
    /// Draws in the law.
    pub n_draws: u64,
    /// Per-scenario values, in declaration order.
    pub scenarios: Vec<ScenarioReport>,
    /// Whether the law is an exact finite law (arithmetic is exact).
    pub exact: bool,
    /// `Some(0.0)` for an exact law; `None` for a sampled one (no calibrated
    /// Monte-Carlo error is claimed).
    pub standard_error: Option<f64>,
    /// Identity of the law the values were computed from.
    pub snapshot_id: String,
    /// Provider object of the law.
    pub provider_id: String,
    /// Always `false`: an observational scenario is not an action.
    pub ranks_actions: bool,
    /// Always `false`: an association is not an intervention effect.
    pub is_intervention_effect: bool,
    /// What this result is and is not.
    pub scope_note: &'static str,
    /// Canonical identity of the query, the law's identities and every value.
    pub identity: String,
}

/// Why an observational-scenario question was refused.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum ObservationalScenarioError {
    /// No scenario, a blank or repeated identity, or too many.
    InvalidScenarios(&'static str),
    /// A threshold or bound is not finite, or a bound is inverted.
    InvalidParameter(&'static str),
    /// A scenario conditions on the outcome coordinate.
    ConditionsOnOutcome(String),
    /// A coordinate is not in the law.
    CoordinateMissing(String),
    /// A coordinate is masked unsupported.
    CoordinateUnsupported(String),
    /// The rows are not paired joint realizations.
    JointLawRequired(&'static str),
    /// The law is not an observational law.
    MeaningNotObservational(&'static str),
    /// The event has no mass in the law.
    EventHasNoMass,
    /// A draw of a used column is not finite.
    NonFiniteDraw,
    /// A stored result does not replay.
    ResultMismatch,
}

impl ObservationalScenarioError {
    /// Registered reason code.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::InvalidScenarios(_)
            | Self::InvalidParameter(_)
            | Self::EventHasNoMass
            | Self::NonFiniteDraw
            | Self::ResultMismatch => antecedent_core::reason_code!("invalid_argument"),
            Self::ConditionsOnOutcome(_)
            | Self::CoordinateMissing(_)
            | Self::CoordinateUnsupported(_) => {
                antecedent_core::reason_code!("quantity_semantics_mismatch")
            }
            Self::JointLawRequired(_) => antecedent_core::reason_code!("joint_law_required"),
            Self::MeaningNotObservational(_) => {
                antecedent_core::reason_code!("distribution_meaning_mismatch")
            }
        }
    }

    /// Namespaced detail.
    #[must_use]
    pub const fn detail(&self) -> &'static str {
        match self {
            Self::InvalidScenarios(_) => DETAIL_SCENARIOS,
            Self::InvalidParameter(_) => DETAIL_PARAMETER,
            Self::ConditionsOnOutcome(_) => DETAIL_CONDITIONS_ON_OUTCOME,
            Self::CoordinateMissing(_) => DETAIL_COORDINATE_MISSING,
            Self::CoordinateUnsupported(_) => DETAIL_COORDINATE_UNSUPPORTED,
            Self::JointLawRequired(_) => DETAIL_JOINT_LAW,
            Self::MeaningNotObservational(_) => DETAIL_MEANING,
            Self::EventHasNoMass => DETAIL_EVENT_MASS,
            Self::NonFiniteDraw => DETAIL_DRAWS,
            Self::ResultMismatch => DETAIL_REPLAY,
        }
    }

    /// Structured refusal with a stated remedy.
    #[must_use]
    pub fn to_refusal(&self) -> ExternalRefusal {
        let build = || RefusalBuilder::new(self.code(), STAGE, self.detail());
        match self {
            Self::InvalidScenarios(what) | Self::InvalidParameter(what) => {
                build().offending(*what).build()
            }
            Self::ConditionsOnOutcome(coordinate) => build()
                .offending(coordinate.clone())
                .remedy("condition scenarios on covariates; the outcome is the event, not a scenario coordinate")
                .build(),
            Self::CoordinateMissing(coordinate) => build()
                .offending(coordinate.clone())
                .remedy("supply a law whose columns include every outcome and scenario coordinate exactly")
                .build(),
            Self::CoordinateUnsupported(coordinate) => build()
                .offending(coordinate.clone())
                .remedy("a coordinate masked as unsupported is not read; drop the scenario or supply support")
                .build(),
            Self::JointLawRequired(alignment) => build()
                .supplied(*alignment)
                .remedy("supply aligned joint draws of the outcome and the scenario coordinates; independent marginals carry no pairing")
                .build(),
            Self::MeaningNotObservational(meaning) => build()
                .supplied(*meaning)
                .remedy("supply an observational predictive or empirical law; an interventional or parameter law does not describe an observed population")
                .build(),
            Self::EventHasNoMass => build()
                .remedy("relax the event or supply a law (or more draws) with mass in it; the conditional is undefined at zero mass")
                .build(),
            Self::NonFiniteDraw => build().build(),
            Self::ResultMismatch => build()
                .remedy("recompute the result from the law and the query")
                .build(),
        }
    }
}

impl std::fmt::Display for ObservationalScenarioError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}: {}", antecedent_core::reason_code::PREFIX, self.code(), self.detail())
    }
}

impl std::error::Error for ObservationalScenarioError {}

fn column_of(
    parsed: &[Option<ScientificQuantity>],
    mask: Option<&[bool]>,
    wanted: &ScientificQuantity,
) -> Result<usize, ObservationalScenarioError> {
    let column = parsed
        .iter()
        .position(|candidate| {
            candidate.as_ref().is_some_and(|q| q.require_same_coordinate(wanted).is_ok())
        })
        .ok_or_else(|| ObservationalScenarioError::CoordinateMissing(wanted.variable_id.clone()))?;
    if mask.is_some_and(|m| m.get(column).is_some_and(|ok| !*ok)) {
        return Err(ObservationalScenarioError::CoordinateUnsupported(wanted.variable_id.clone()));
    }
    Ok(column)
}

fn validate(query: &ObservationalScenarioQuery) -> Result<(), ObservationalScenarioError> {
    if query.scenarios.is_empty() {
        return Err(ObservationalScenarioError::InvalidScenarios("at least one scenario"));
    }
    if query.scenarios.len() > MAX_OBSERVATIONAL_SCENARIOS {
        return Err(ObservationalScenarioError::InvalidScenarios(
            "scenario count exceeds the bound",
        ));
    }
    if !query.event.threshold().is_finite() {
        return Err(ObservationalScenarioError::InvalidParameter("event threshold is finite"));
    }
    let mut ids = HashSet::new();
    for scenario in &query.scenarios {
        if scenario.id.trim().is_empty() || !ids.insert(scenario.id.as_str()) {
            return Err(ObservationalScenarioError::InvalidScenarios(
                "scenario identities are distinct and non-blank",
            ));
        }
        if scenario.conditions.is_empty() {
            return Err(ObservationalScenarioError::InvalidScenarios(
                "a scenario has at least one condition",
            ));
        }
        for condition in &scenario.conditions {
            if !condition.lower.is_finite()
                || !condition.upper.is_finite()
                || condition.lower > condition.upper
            {
                return Err(ObservationalScenarioError::InvalidParameter(
                    "condition bounds are finite and ordered",
                ));
            }
            if condition.coordinate.require_same_coordinate(&query.outcome).is_ok() {
                return Err(ObservationalScenarioError::ConditionsOnOutcome(
                    condition.coordinate.variable_id.clone(),
                ));
            }
        }
    }
    Ok(())
}

fn meaning_label(meaning: DistributionMeaning) -> &'static str {
    match meaning {
        DistributionMeaning::ParameterPosterior => "parameter_posterior",
        DistributionMeaning::CausalFunctionalPosterior => "causal_functional_posterior",
        DistributionMeaning::PosteriorPredictive => "posterior_predictive",
        DistributionMeaning::InterventionalPredictive => "interventional_predictive",
        DistributionMeaning::EstimatorSampling => "estimator_sampling",
        DistributionMeaning::Bootstrap => "bootstrap",
        DistributionMeaning::EmpiricalOutcome => "empirical_outcome",
    }
}

fn identity_of(
    query: &ObservationalScenarioQuery,
    law: &DistributionArtifact,
    values: &Values,
) -> String {
    let mut writer = CanonicalWriter::new(TAG);
    let identity = &law.metadata().identity;
    writer
        .word(OBSERVATIONAL_SCENARIO_VERSION)
        .text(&identity.source_id)
        .text(&identity.provider_id)
        .text(&identity.snapshot_id)
        .text(&identity.causal_contract_id)
        .text(&identity.rng_id);
    query.outcome.write_canonical(&mut writer);
    writer.word(query.event.tag()).real(query.event.threshold());
    writer.word(query.scenarios.len() as u64);
    for scenario in &query.scenarios {
        writer.text(&scenario.id).word(scenario.conditions.len() as u64);
        for condition in &scenario.conditions {
            condition.coordinate.write_canonical(&mut writer);
            writer.real(condition.lower).real(condition.upper);
        }
    }
    writer
        .real(values.event_mass)
        .real(values.covered)
        .real(values.effective_event_draws)
        .word(values.n_draws);
    for report in &values.reports {
        writer
            .text(&report.id)
            .real(report.mass)
            .real(report.joint_mass)
            .opt_real(report.event_given_scenario)
            .real(report.scenario_given_event)
            .opt_real(report.lift)
            .word(report.scenario_draws)
            .word(report.joint_draws);
    }
    writer.finish_hex()
}

struct Values {
    event_mass: f64,
    covered: f64,
    effective_event_draws: f64,
    n_draws: u64,
    reports: Vec<ScenarioReport>,
}

/// One scenario's `(column, lower, upper)` conditions, resolved against the law.
type ScenarioConditions = Vec<(usize, f64, f64)>;

/// The (column, lower, upper) conditions of every scenario, resolved against the law.
fn scenario_columns(
    query: &ObservationalScenarioQuery,
    parsed: &[Option<ScientificQuantity>],
    mask: Option<&[bool]>,
) -> Result<Vec<ScenarioConditions>, ObservationalScenarioError> {
    let mut scenario_columns: Vec<ScenarioConditions> = Vec::with_capacity(query.scenarios.len());
    for scenario in &query.scenarios {
        let mut columns = Vec::with_capacity(scenario.conditions.len());
        for condition in &scenario.conditions {
            columns.push((
                column_of(parsed, mask, &condition.coordinate)?,
                condition.lower,
                condition.upper,
            ));
        }
        scenario_columns.push(columns);
    }
    Ok(scenario_columns)
}

/// Per-scenario masses and draw counts accumulated over the law's rows.
struct ScenarioTotals<'a> {
    event_mass: f64,
    masses: &'a [f64],
    joint: &'a [f64],
    scenario_draws: &'a [u64],
    joint_draws: &'a [u64],
}

fn scenario_reports(
    query: &ObservationalScenarioQuery,
    totals: &ScenarioTotals<'_>,
) -> Vec<ScenarioReport> {
    let ScenarioTotals { event_mass, masses, joint, scenario_draws, joint_draws } = *totals;
    query
        .scenarios
        .iter()
        .enumerate()
        .map(|(index, scenario)| {
            let event_given = (masses[index] > 0.0).then(|| joint[index] / masses[index]);
            ScenarioReport {
                id: scenario.id.clone(),
                mass: masses[index],
                joint_mass: joint[index],
                event_given_scenario: event_given,
                scenario_given_event: joint[index] / event_mass,
                lift: event_given.map(|p| p / event_mass),
                scenario_draws: scenario_draws[index],
                joint_draws: joint_draws[index],
            }
        })
        .collect()
}

/// Where the outcome sits in the law's row-major draw matrix.
#[derive(Clone, Copy)]
struct RowLayout {
    width: usize,
    outcome_column: usize,
}

/// Weighted event and scenario masses and draw counts accumulated over the law's rows.
struct RowAccumulation {
    event_mass: f64,
    event_square: f64,
    covered: f64,
    masses: Vec<f64>,
    joint: Vec<f64>,
    scenario_draws: Vec<u64>,
    joint_draws: Vec<u64>,
}

fn accumulate_rows(
    query: &ObservationalScenarioQuery,
    scenario_columns: &[ScenarioConditions],
    weights: &[f64],
    draws: &[f64],
    layout: RowLayout,
) -> Result<RowAccumulation, ObservationalScenarioError> {
    let RowLayout { width, outcome_column } = layout;
    let rows = weights.len();
    let mut event_mass = 0.0;
    let mut event_square = 0.0;
    let mut covered = 0.0;
    let mut masses = vec![0.0; query.scenarios.len()];
    let mut joint = vec![0.0; query.scenarios.len()];
    let mut scenario_draws = vec![0_u64; query.scenarios.len()];
    let mut joint_draws = vec![0_u64; query.scenarios.len()];
    for row in 0..rows {
        let weight = weights[row];
        let y = draws[row * width + outcome_column];
        if !y.is_finite() {
            return Err(ObservationalScenarioError::NonFiniteDraw);
        }
        let in_event = query.event.holds(y);
        let mut in_any = false;
        for (index, columns) in scenario_columns.iter().enumerate() {
            let mut inside = true;
            for (column, lower, upper) in columns {
                let x = draws[row * width + column];
                if !x.is_finite() {
                    return Err(ObservationalScenarioError::NonFiniteDraw);
                }
                if x < *lower || x > *upper {
                    inside = false;
                }
            }
            if inside && weight > 0.0 {
                masses[index] += weight;
                scenario_draws[index] += 1;
                if in_event {
                    joint[index] += weight;
                    joint_draws[index] += 1;
                    in_any = true;
                }
            }
        }
        if in_event && weight > 0.0 {
            event_mass += weight;
            event_square += weight * weight;
            if in_any {
                covered += weight;
            }
        }
    }
    Ok(RowAccumulation {
        event_mass,
        event_square,
        covered,
        masses,
        joint,
        scenario_draws,
        joint_draws,
    })
}

/// Answer an observational-scenario question on an aligned observational law.
///
/// # Errors
/// An empty, repeated or oversized scenario list, a non-finite threshold or
/// bound, a scenario that conditions on the outcome, a coordinate that is
/// missing or masked, non-joint alignment, a law that is not observational, a
/// non-finite draw, and an event with no mass.
pub fn evaluate_observational_scenarios(
    query: &ObservationalScenarioQuery,
    law: &DistributionArtifact,
) -> Result<ObservationalScenarioResult, ObservationalScenarioError> {
    validate(query)?;
    let metadata = law.metadata();
    if metadata.identity.alignment != DrawAlignment::Joint {
        return Err(ObservationalScenarioError::JointLawRequired("independent_marginals"));
    }
    let meaning = DistributionMeaning::from(law.semantic());
    if !matches!(
        meaning,
        DistributionMeaning::PosteriorPredictive | DistributionMeaning::EmpiricalOutcome
    ) {
        return Err(ObservationalScenarioError::MeaningNotObservational(meaning_label(meaning)));
    }
    let parsed: Vec<Option<ScientificQuantity>> = law
        .quantities()
        .iter()
        .map(|wire| ScientificQuantity::try_from(wire.clone()).ok())
        .collect();
    let mask = metadata.supported.as_deref();
    let outcome_column = column_of(&parsed, mask, &query.outcome)?;
    let scenario_columns = scenario_columns(query, &parsed, mask)?;
    let [rows, width] = law.shape();
    let draws = law.draws();
    let raw: Vec<f64> = metadata.weights.clone().unwrap_or_else(|| vec![1.0; rows]);
    let total: f64 = raw.iter().sum();
    let weights: Vec<f64> = raw.iter().map(|w| w / total).collect();

    let RowAccumulation {
        event_mass,
        event_square,
        covered,
        masses,
        joint,
        scenario_draws,
        joint_draws,
    } = accumulate_rows(
        query,
        &scenario_columns,
        &weights,
        draws,
        RowLayout { width, outcome_column },
    )?;
    if event_mass.is_nan() || event_mass <= 0.0 {
        return Err(ObservationalScenarioError::EventHasNoMass);
    }
    let effective_event_draws = event_mass * event_mass / event_square;
    let reports = scenario_reports(
        query,
        &ScenarioTotals {
            event_mass,
            masses: &masses,
            joint: &joint,
            scenario_draws: &scenario_draws,
            joint_draws: &joint_draws,
        },
    );
    let values =
        Values { event_mass, covered, effective_event_draws, n_draws: rows as u64, reports };
    let identity = identity_of(query, law, &values);
    let exact = metadata.calibration == DistributionCalibration::Exact;
    Ok(ObservationalScenarioResult {
        event_mass,
        covered_event_mass: covered,
        uncovered_event_mass: (event_mass - covered).max(0.0),
        effective_event_draws,
        n_draws: values.n_draws,
        scenarios: values.reports,
        exact,
        standard_error: exact.then_some(0.0),
        snapshot_id: metadata.identity.snapshot_id.clone(),
        provider_id: metadata.identity.provider_id.clone(),
        ranks_actions: false,
        is_intervention_effect: false,
        scope_note: OBSERVATIONAL_SCENARIO_SCOPE_NOTE,
        identity,
    })
}

/// Re-derive a result from its law and query and compare, rather than trusting
/// the stored values.
///
/// # Errors
/// Every refusal of [`evaluate_observational_scenarios`], and
/// [`ObservationalScenarioError::ResultMismatch`] when the stored identity or
/// any stored value differs from the recomputed one.
pub fn verify_observational_scenarios(
    result: &ObservationalScenarioResult,
    query: &ObservationalScenarioQuery,
    law: &DistributionArtifact,
) -> Result<(), ObservationalScenarioError> {
    let fresh = evaluate_observational_scenarios(query, law)?;
    if fresh != *result {
        return Err(ObservationalScenarioError::ResultMismatch);
    }
    Ok(())
}
