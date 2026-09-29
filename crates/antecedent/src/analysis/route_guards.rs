//! Predicates and small helpers shared by the sealed-route sealers, the
//! one-shot dispatch guards, and the checked operations they retain.
//!
//! Every helper here is a pure restatement of a fragment that several routes
//! spelled out inline; none widens or narrows what any route admits.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{ObservationSpec, OutcomeFunctional, ResponseQuery, TargetPopulation};
use antecedent_data::TimeSeriesData;
use antecedent_identify::IdentificationEnvelope;

use super::builder::{DataInput, RefuteSuite};
use super::execute::Study;
use crate::AcceptedGraph;
use crate::error::CausalError;
use crate::inference::InferenceMode;
use crate::planner::PhysicalExecutionPlan;
use crate::strategy_table::{EstimatorId, IdentifierId};
use crate::support::StructureSource;

impl Study {
    /// Structure was supplied explicitly or accepted from review, not
    /// discovered or drawn from a graph posterior.
    #[must_use]
    pub(crate) fn fixed_structure(&self) -> bool {
        matches!(self.structure_source, StructureSource::Explicit | StructureSource::Accepted)
    }

    /// Validation suite is one of the built-in point suites (`none`, `cheap`,
    /// or `full`).
    #[must_use]
    pub(crate) fn point_validation(&self) -> bool {
        matches!(self.refute, RefuteSuite::None | RefuteSuite::Cheap | RefuteSuite::Full)
    }

    /// [`Self::point_validation`] widened to the placebo-and-RCC suite.
    #[must_use]
    pub(crate) fn point_validation_or_placebo(&self) -> bool {
        matches!(
            self.refute,
            RefuteSuite::None | RefuteSuite::Cheap | RefuteSuite::PlaceboAndRcc | RefuteSuite::Full
        )
    }
}

/// Mean outcome functional over the all-observed population.
#[must_use]
pub(crate) fn mean_all_observed(
    outcome_functional: &OutcomeFunctional,
    target_population: &TargetPopulation,
) -> bool {
    matches!(outcome_functional, OutcomeFunctional::Mean)
        && *target_population == TargetPopulation::AllObserved
}

/// Complete-observation, all-observed, mean response query.
#[must_use]
pub(crate) fn complete_mean_response(query: &ResponseQuery) -> bool {
    query.observation == ObservationSpec::Complete
        && mean_all_observed(&query.outcome_functional, &query.target_population)
}

/// How a plan-record name is compared against the name a route expects.
#[derive(Clone, Copy)]
pub(crate) enum PlanRecordMode<'a> {
    /// The record must name exactly the expected id.
    Exact,
    /// A missing record name reads as this default before comparison.
    OrDefault(&'a str),
    /// A missing record name is accepted; a present one must match.
    IfPresent,
}

fn plan_name_is(recorded: Option<&str>, expected: &str, mode: PlanRecordMode<'_>) -> bool {
    match mode {
        PlanRecordMode::Exact => recorded == Some(expected),
        PlanRecordMode::OrDefault(default) => recorded.unwrap_or(default) == expected,
        PlanRecordMode::IfPresent => recorded.is_none_or(|name| name == expected),
    }
}

/// Whether the logical plan record names `identifier` and `estimator`, each
/// compared under its own [`PlanRecordMode`].
#[must_use]
pub(crate) fn plan_record_is(
    plan: &PhysicalExecutionPlan,
    identifier: IdentifierId,
    identifier_mode: PlanRecordMode<'_>,
    estimator: EstimatorId,
    estimator_mode: PlanRecordMode<'_>,
) -> bool {
    let record = &plan.logical.record;
    plan_name_is(record.identifier.as_deref(), identifier.as_str(), identifier_mode)
        && plan_name_is(record.estimator.as_deref(), estimator.as_str(), estimator_mode)
}

/// Whether two accepted graphs are the same class, version, and structure.
#[must_use]
pub(crate) fn same_accepted_graph(retained: &AcceptedGraph, current: &AcceptedGraph) -> bool {
    retained.class() == current.class()
        && retained.version() == current.version()
        && format!("{retained:?}") == format!("{current:?}")
}

/// Completion count and identified / unidentified mass of an envelope.
#[must_use]
pub(crate) fn envelope_summary<G>(envelope: &IdentificationEnvelope<G>) -> (usize, f64, f64) {
    (envelope.cases.len(), envelope.identified_weight.0, envelope.unidentified_weight.0)
}

/// Stable inspector label for a frozen inference mode.
#[must_use]
pub(crate) fn inference_label(inference: &InferenceMode) -> Arc<str> {
    match inference {
        InferenceMode::Bayesian(config) => Arc::from(format!("bayesian:{:?}", config.backend)),
        InferenceMode::Frequentist => Arc::from("frequentist"),
    }
}

/// Series data in the variant `retained` was prepared with: event studies
/// keep the event modality through estimate and refresh.
#[must_use]
pub(crate) fn series_input(retained: &DataInput, data: TimeSeriesData) -> DataInput {
    match retained {
        DataInput::Event(_) => DataInput::Event(data),
        _ => DataInput::Temporal(data),
    }
}

/// A compile error carrying `message`.
#[must_use]
pub(crate) fn compile_error(message: &str) -> CausalError {
    CausalError::Compile { message: message.into() }
}
