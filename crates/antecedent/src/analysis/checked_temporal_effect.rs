//! Retained proof and procedure for a fixed-graph temporal effect.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{CausalQuery, IdentificationStatus, TemporalEffectQuery};
use antecedent_data::TemporalIndexer;
use antecedent_expr::IdentifiedEstimand;
use antecedent_graph::{NodeRef, TemporalDag};
use antecedent_identify::result::IdentificationResult;

use crate::error::CausalError;
use crate::strategy_table::{EstimatorId, IdentifierId};

use super::builder::RefuteSuite;

/// A temporal effect's full selected identification and estimation route.
///
/// The finite-unfolding indexer is part of the proof binding: dense IDs in the
/// estimand are meaningful only under this exact indexer and graph.
#[derive(Clone, Debug)]
pub(crate) struct CheckedTemporalEffectOperation {
    graph: TemporalDag,
    graph_nodes: Vec<NodeRef>,
    graph_edges: Vec<(u32, u32)>,
    query: TemporalEffectQuery,
    identification: IdentificationResult,
    estimand: IdentifiedEstimand,
    indexer: TemporalIndexer,
    identifier: IdentifierId,
    estimator: EstimatorId,
    bootstrap_replicates: u32,
    refute: RefuteSuite,
}

impl CheckedTemporalEffectOperation {
    /// Check and freeze the graph, query, selected proof, and estimator family.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn checked(
        graph: &TemporalDag,
        query: &TemporalEffectQuery,
        identification: IdentificationResult,
        estimand: IdentifiedEstimand,
        indexer: TemporalIndexer,
        identifier: IdentifierId,
        estimator: EstimatorId,
        bootstrap_replicates: u32,
        refute: RefuteSuite,
    ) -> Result<Self, CausalError> {
        query.validate().map_err(|error| CausalError::Compile { message: error.to_string() })?;
        if identifier != IdentifierId::TemporalBackdoorUnfolded {
            return Err(CausalError::Compile {
                message: "temporal effect requires the retained unfolded backdoor proof".into(),
            });
        }
        let expected = if query.is_multi_step_sustained() {
            EstimatorId::TemporalSequentialGcomp
        } else {
            EstimatorId::TemporalLinearAdjustment
        };
        if estimator != expected {
            return Err(CausalError::Compile {
                message: "temporal effect estimator does not match its policy window".into(),
            });
        }
        if !matches!(
            identification.status,
            IdentificationStatus::NonparametricallyIdentified
                | IdentificationStatus::IdentifiedUnderParametricRestrictions
        ) {
            return Err(CausalError::Unsupported {
                message: "checked temporal effect requires point identification",
            });
        }
        if !identification.estimands.iter().any(|candidate| {
            candidate.functional == estimand.functional
                && candidate.method == estimand.method
                && candidate.adjustment_set == estimand.adjustment_set
        }) {
            return Err(CausalError::Compile {
                message: "temporal effect estimand is absent from its retained proof".into(),
            });
        }
        temporal_proof_binds_query(&identification, query, &indexer)?;
        let mut graph_edges =
            graph.edges().map(|edge| (edge.a.raw(), edge.b.raw())).collect::<Vec<_>>();
        graph_edges.sort_unstable();
        Ok(Self {
            graph: graph.clone(),
            graph_nodes: graph.nodes().to_vec(),
            graph_edges,
            query: query.clone(),
            identification,
            estimand,
            indexer,
            identifier,
            estimator,
            bootstrap_replicates,
            refute,
        })
    }

    #[must_use]
    pub(crate) fn query(&self) -> &TemporalEffectQuery {
        &self.query
    }

    #[must_use]
    pub(crate) fn identification(&self) -> &IdentificationResult {
        &self.identification
    }

    #[must_use]
    pub(crate) fn estimand(&self) -> &IdentifiedEstimand {
        &self.estimand
    }

    #[must_use]
    pub(crate) fn indexer(&self) -> &TemporalIndexer {
        &self.indexer
    }

    #[must_use]
    pub(crate) fn graph(&self) -> &TemporalDag {
        &self.graph
    }

    #[must_use]
    pub(crate) fn procedure(&self) -> (IdentifierId, EstimatorId, RefuteSuite, u32) {
        (self.identifier, self.estimator, self.refute, self.bootstrap_replicates)
    }

    #[must_use]
    pub(crate) fn matches_graph(&self, graph: &TemporalDag) -> bool {
        if graph.nodes() != self.graph_nodes {
            return false;
        }
        let mut edges = graph.edges().map(|edge| (edge.a.raw(), edge.b.raw())).collect::<Vec<_>>();
        edges.sort_unstable();
        edges == self.graph_edges
    }

    /// Canonical structural signature of the retained temporal template.
    #[must_use]
    pub(crate) fn graph_signature(&self) -> String {
        format!("nodes={:?};edges={:?}", self.graph_nodes, self.graph_edges)
    }

    #[must_use]
    pub(crate) fn matches_query(&self, query: &CausalQuery) -> bool {
        matches!(query, CausalQuery::TemporalEffect(candidate) if candidate == &self.query)
    }
}

/// Confirm an unfolded temporal proof targets `query`: the identifier's query
/// is an average-effect query over the unfolded graph, so the semantic
/// variables and both intervention levels are checked after mapping them into
/// that graph's dense ID space. Multi-step schedules retain the joint query.
pub(crate) fn temporal_proof_binds_query(
    identification: &IdentificationResult,
    query: &TemporalEffectQuery,
    indexer: &TemporalIndexer,
) -> Result<(), CausalError> {
    // The identifier's query is an average-effect query over the unfolded
    // graph. Check the semantic variables and both intervention levels after
    // mapping them into that graph's dense ID space.
    let expected_treatment = indexer
        .dense_id(antecedent_core::TemporalNodeKey {
            variable: query.treatment,
            offset: query
                .try_treatment_offset()
                .map_err(|error| CausalError::Compile { message: error.to_string() })?,
        })
        .map_err(|error| CausalError::Compile { message: error.to_string() })?;
    let expected_outcome = indexer
        .dense_id(antecedent_core::TemporalNodeKey {
            variable: query.outcome,
            offset: query.outcome_offset(),
        })
        .map_err(|error| CausalError::Compile { message: error.to_string() })?;
    if query.is_multi_step_sustained() {
        // The schedule identifier retains the complete temporal query and
        // its joint contrast. A scalar average-effect projection would lose
        // the other intervened time nodes.
        if identification.query != CausalQuery::TemporalEffect(query.clone()) {
            return Err(CausalError::Compile {
                message: "temporal schedule proof disagrees on the joint intervention regime"
                    .into(),
            });
        }
    } else {
        let Some(proof_query) = identification.average_effect() else {
            return Err(CausalError::Compile {
                message: "temporal effect proof has no average-effect target".into(),
            });
        };
        if proof_query.treatment.raw() != expected_treatment
            || proof_query.outcome.raw() != expected_outcome
            || proof_query.target_population != query.target_population
            || !intervention_matches(
                &proof_query.control,
                &query.control,
                expected_treatment,
                query.treatment,
            )
            || !intervention_matches(
                &proof_query.active,
                &query.active,
                expected_treatment,
                query.treatment,
            )
        {
            return Err(CausalError::Compile {
                message:
                    "temporal effect proof disagrees on target, population, or intervention levels"
                        .into(),
            });
        }
    }
    Ok(())
}

fn intervention_matches(
    proof: &antecedent_core::Intervention,
    requested: &antecedent_core::Intervention,
    dense_treatment: u32,
    semantic_treatment: antecedent_core::VariableId,
) -> bool {
    matches!(
        (proof, requested),
        (
            antecedent_core::Intervention::Set { variable: proof_var, value: proof_value },
            antecedent_core::Intervention::Set { variable: requested_var, value: requested_value },
        ) if proof_var.raw() == dense_treatment && proof_value == requested_value
            && *requested_var == semantic_treatment
    )
}
