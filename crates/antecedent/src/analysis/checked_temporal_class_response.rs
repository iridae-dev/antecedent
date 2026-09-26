//! Retained completion proofs and execution settings for an incomplete temporal
//! graph response (mean curve or single-step intervention response).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{
    CausalQuery, ObservationSpec, OutcomeFunctional, ResponseQuery, TargetPopulation,
    TemporalEffectQuery, TemporalNodeKey,
};

use crate::error::CausalError;
use crate::planner::PhysicalExecutionPlan;
use crate::strategy_table::{EstimatorId, IdentifierId};
use crate::{AcceptedGraph, ClassPrior, GraphClass, InferenceMode};

use super::builder::RefuteSuite;
use super::checked_temporal_response::temporal_response_is_direct;
use super::prepared::CachedTemporalClassIdentification;
use super::route_guards::same_accepted_graph;

/// A complete, checked completion envelope per requested horizon for a
/// TemporalCpdag or TemporalPag response. The per-horizon envelopes and their
/// aligned unfolding indexers stay together with the source graph, so no
/// completion can be evaluated against another completion's time window and no
/// execution can re-derive the class from a different graph.
#[derive(Clone)]
pub(crate) struct CheckedTemporalClassResponseOperation {
    source_graph: AcceptedGraph,
    max_completions: Option<usize>,
    query: ResponseQuery,
    bundle: CachedTemporalClassIdentification,
    identifier: IdentifierId,
    estimator: EstimatorId,
    inference: InferenceMode,
    bootstrap_replicates: u32,
    class_prior: Option<ClassPrior>,
    validation: RefuteSuite,
    physical: PhysicalExecutionPlan,
}

impl std::fmt::Debug for CheckedTemporalClassResponseOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CheckedTemporalClassResponseOperation")
            .field("graph_class", &self.source_graph.class())
            .field("structure_version", &self.source_graph.version())
            .field("query", &self.query)
            .field("horizon_count", &self.bundle.by_horizon.len())
            .field("identifier", &self.identifier)
            .field("estimator", &self.estimator)
            .field("bootstrap_replicates", &self.bootstrap_replicates)
            .field("validation", &self.validation)
            .finish_non_exhaustive()
    }
}

impl CheckedTemporalClassResponseOperation {
    /// Check and freeze the per-horizon completion proofs and selected procedure.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn checked(
        source_graph: AcceptedGraph,
        query: &ResponseQuery,
        bundle: &CachedTemporalClassIdentification,
        max_completions: Option<usize>,
        inference: &InferenceMode,
        bootstrap_replicates: u32,
        class_prior: Option<ClassPrior>,
        validation: RefuteSuite,
        physical: PhysicalExecutionPlan,
    ) -> Result<Self, CausalError> {
        query.validate().map_err(|error| CausalError::Compile { message: error.to_string() })?;
        if !matches!(source_graph.class(), GraphClass::TemporalCpdag | GraphClass::TemporalPag) {
            return Err(CausalError::Compile {
                message: "temporal class response requires a TemporalCpdag or TemporalPag proof"
                    .into(),
            });
        }
        let spec = query.temporal.as_ref().ok_or_else(|| CausalError::Compile {
            message: "temporal class response requires a temporal response specification".into(),
        })?;
        if !temporal_response_is_direct(query) {
            return Err(CausalError::Unsupported {
                message: "checked temporal class response supports MeanCurve and single-step InterventionResponse only",
            });
        }
        if query.observation != ObservationSpec::Complete
            || query.target_population != TargetPopulation::AllObserved
            || !matches!(query.outcome_functional, OutcomeFunctional::Mean)
        {
            return Err(CausalError::Unsupported {
                message: "checked temporal class response requires complete observation, the all-observed population, and the mean outcome functional",
            });
        }
        if !spec.policy.is_single_step() {
            return Err(CausalError::Unsupported {
                message: "checked temporal class response supports Pulse or single-step Sustained policies",
            });
        }
        if validation != RefuteSuite::None {
            return Err(CausalError::Unsupported {
                message: "checked temporal class response supports validation suite none only",
            });
        }
        let (treatment, outcome) =
            query.functional.primary_pair().ok_or_else(|| CausalError::Compile {
                message: "temporal class response has no treatment/outcome pair".into(),
            })?;
        let identifier = IdentifierId::GeneralizedAdjustment;
        let estimator = EstimatorId::temporal_response_for(inference);
        if physical.logical.query != CausalQuery::Response(query.clone())
            || physical
                .logical
                .record
                .identifier
                .as_deref()
                .is_some_and(|selected| selected != identifier.as_str())
            || physical
                .logical
                .record
                .estimator
                .as_deref()
                .is_some_and(|selected| selected != estimator.as_str())
        {
            return Err(CausalError::Compile {
                message:
                    "temporal class response plan disagrees with its retained query or procedure"
                        .into(),
            });
        }
        for &horizon in spec.horizons.iter() {
            let (_, envelope) = bundle
                .by_horizon
                .iter()
                .find(|(cached, _)| *cached == horizon)
                .ok_or_else(|| CausalError::Compile {
                    message: format!(
                        "temporal class response proof lacks a completion envelope for horizon {horizon}"
                    ),
                })?;
            if envelope.envelope.cases.is_empty() {
                return Err(CausalError::Compile {
                    message: format!(
                        "temporal class response proof has no completion cases at horizon {horizon}"
                    ),
                });
            }
            if envelope.indexers.len() != envelope.envelope.cases.len() {
                return Err(CausalError::Compile {
                    message: "temporal class response proof does not bind one unfolding indexer per completion".into(),
                });
            }
            // Every completion proof is an unfolded (dense) adjustment certificate;
            // bind its treatment/outcome nodes back through that completion's own
            // indexer to the requested variables, policy origin, and horizon.
            let witness = TemporalEffectQuery::pulse(treatment, outcome, 1.0)
                .with_policy(spec.policy.clone())
                .with_horizon_steps(horizon)
                .with_max_history_lag(spec.max_history_lag)
                .with_target_population(query.target_population.clone());
            let treatment_key = TemporalNodeKey {
                variable: treatment,
                offset: witness
                    .try_treatment_offset()
                    .map_err(|e| CausalError::Compile { message: e.to_string() })?,
            };
            let outcome_key =
                TemporalNodeKey { variable: outcome, offset: witness.outcome_offset() };
            let witness_binds =
                envelope.envelope.cases.iter().zip(&envelope.indexers).all(|(case, indexer)| {
                    case.result.average_effect().is_none_or(|proof| {
                        Some(proof.treatment.raw()) == indexer.dense_id(treatment_key).ok()
                            && Some(proof.outcome.raw()) == indexer.dense_id(outcome_key).ok()
                    })
                });
            if !witness_binds {
                return Err(CausalError::Compile {
                    message: "temporal class response proof was certified for a different treatment, outcome, policy, or horizon".into(),
                });
            }
        }
        Ok(Self {
            source_graph,
            max_completions,
            query: query.clone(),
            bundle: bundle.clone(),
            identifier,
            estimator,
            inference: inference.clone(),
            bootstrap_replicates,
            class_prior,
            validation,
            physical,
        })
    }

    #[must_use]
    pub(crate) fn source_graph(&self) -> &AcceptedGraph {
        &self.source_graph
    }

    #[must_use]
    pub(crate) fn graph_class(&self) -> GraphClass {
        self.source_graph.class()
    }

    #[must_use]
    pub(crate) fn structure_version(&self) -> u32 {
        self.source_graph.version()
    }

    /// Verify that a click or refresh still carries the source graph the proof
    /// was enumerated from.
    #[must_use]
    pub(crate) fn matches_source_graph(&self, graph: &AcceptedGraph) -> bool {
        same_accepted_graph(&self.source_graph, graph)
    }

    #[must_use]
    pub(crate) fn max_completions(&self) -> Option<usize> {
        self.max_completions
    }

    #[must_use]
    pub(crate) fn query(&self) -> &ResponseQuery {
        &self.query
    }

    #[must_use]
    pub(crate) fn bundle(&self) -> &CachedTemporalClassIdentification {
        &self.bundle
    }

    #[must_use]
    pub(crate) const fn inference(&self) -> &InferenceMode {
        &self.inference
    }

    #[must_use]
    pub(crate) fn procedure(&self) -> (IdentifierId, EstimatorId, u32, RefuteSuite) {
        (self.identifier, self.estimator, self.bootstrap_replicates, self.validation)
    }

    #[must_use]
    pub(crate) fn class_prior(&self) -> Option<&ClassPrior> {
        self.class_prior.as_ref()
    }

    #[must_use]
    pub(crate) fn physical(&self) -> &PhysicalExecutionPlan {
        &self.physical
    }

    /// Ordered requested horizons.
    #[must_use]
    pub(crate) fn horizons(&self) -> &[u32] {
        self.query.temporal.as_ref().map_or(&[], |spec| spec.horizons.as_ref())
    }

    /// Number of completion cases retained for the first requested horizon.
    #[must_use]
    pub(crate) fn completion_count(&self) -> usize {
        self.horizons()
            .first()
            .and_then(|horizon| self.bundle.by_horizon.iter().find(|(cached, _)| cached == horizon))
            .map_or(0, |(_, envelope)| envelope.envelope.cases.len())
    }

    /// Identified and unidentified enumeration mass retained for the first horizon.
    #[must_use]
    pub(crate) fn enumeration_mass(&self) -> (f64, f64) {
        self.horizons()
            .first()
            .and_then(|horizon| self.bundle.by_horizon.iter().find(|(cached, _)| cached == horizon))
            .map_or((0.0, 0.0), |(_, envelope)| {
                (envelope.envelope.identified_weight.0, envelope.envelope.unidentified_weight.0)
            })
    }
}
