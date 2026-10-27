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
use crate::error::CausalError;
use crate::inference::InferenceMode;
use crate::support::StructureSource;
use crate::{AcceptedGraph, GraphRef};

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
        matches!(self.settings.refute, RefuteSuite::None | RefuteSuite::Cheap | RefuteSuite::Full)
    }

    /// [`Self::point_validation`] widened to the placebo-and-RCC suite.
    #[must_use]
    pub(crate) fn point_validation_or_placebo(&self) -> bool {
        matches!(
            self.settings.refute,
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

/// The query contrasts two distinct numeric `Set` levels of its treatment.
///
/// Propensity and AIPW preparation recode the declared active and control levels to a 0/1
/// arm indicator, so the contrast is not limited to treatment coded `1` against `0`.
#[must_use]
#[allow(
    clippy::float_cmp,
    reason = "treatment arms are exact category labels, not computed values"
)]
pub(crate) fn distinct_numeric_arms(query: &antecedent_core::AverageEffectQuery) -> bool {
    let level = |intervention: &antecedent_core::Intervention| match intervention {
        antecedent_core::Intervention::Set { variable, value } if *variable == query.treatment => {
            value.as_f64()
        }
        _ => None,
    };
    matches!((level(&query.active), level(&query.control)),
        (Some(active), Some(control)) if active != control)
}

/// Complete-observation, all-observed, mean response query.
#[must_use]
pub(crate) fn complete_mean_response(query: &ResponseQuery) -> bool {
    query.observation == ObservationSpec::Complete
        && mean_all_observed(&query.outcome_functional, &query.target_population)
}

/// Whether two accepted graphs are the same class, version, and structure.
#[must_use]
pub(crate) fn same_accepted_graph(retained: &AcceptedGraph, current: &AcceptedGraph) -> bool {
    retained.class() == current.class()
        && retained.version() == current.version()
        && retained.algorithm_id() == current.algorithm_id()
        && retained.variable_names() == current.variable_names()
        && match (retained.graph_ref(), current.graph_ref()) {
            (GraphRef::Dag(a), GraphRef::Dag(b)) => a == b,
            (GraphRef::Admg(a), GraphRef::Admg(b)) => a == b,
            (GraphRef::Cpdag(a), GraphRef::Cpdag(b)) => a == b,
            (GraphRef::Pag(a), GraphRef::Pag(b)) => a == b,
            (GraphRef::TemporalDag(a), GraphRef::TemporalDag(b)) => a == b,
            (GraphRef::TemporalCpdag(a), GraphRef::TemporalCpdag(b)) => a == b,
            (GraphRef::TemporalPag(a), GraphRef::TemporalPag(b)) => a == b,
            (GraphRef::TemporalAdmg(a), GraphRef::TemporalAdmg(b)) => same_temporal_admg(a, b),
            (GraphRef::RandomizedTrial, GraphRef::RandomizedTrial) => true,
            _ => false,
        }
}

fn same_temporal_admg(
    a: &antecedent_graph::TemporalAdmg,
    b: &antecedent_graph::TemporalAdmg,
) -> bool {
    a.nodes() == b.nodes()
        && (0..a.node_count()).all(|i| {
            let id = antecedent_graph::DenseNodeId::from_raw(u32::try_from(i).unwrap_or(u32::MAX));
            a.children(id) == b.children(id)
                && a.parents(id) == b.parents(id)
                && a.bidirected_neighbors(id) == b.bidirected_neighbors(id)
        })
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

#[cfg(test)]
mod tests {
    use super::same_accepted_graph;
    use crate::AcceptedGraph;
    use antecedent_graph::{Dag, DenseNodeId};

    #[test]
    fn same_accepted_graph_compares_edges() {
        let mut left = Dag::with_variables(2);
        left.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let mut right = Dag::with_variables(2);
        right.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        assert!(same_accepted_graph(&AcceptedGraph::dag(left), &AcceptedGraph::dag(right)));
        let mut reversed = Dag::with_variables(2);
        reversed.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(0)).unwrap();
        let mut again = Dag::with_variables(2);
        again.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        assert!(!same_accepted_graph(&AcceptedGraph::dag(again), &AcceptedGraph::dag(reversed)));
    }
}
