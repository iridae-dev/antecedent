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
use super::proof_support::{Procedure, proof_supplies};

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

/// The frozen premises of [`CheckedTemporalEffectOperation::checked`], carried together.
pub(crate) struct CheckedTemporalEffectOperationPremises<'a> {
    pub(crate) graph: &'a TemporalDag,
    pub(crate) query: &'a TemporalEffectQuery,
    pub(crate) identification: IdentificationResult,
    pub(crate) estimand: IdentifiedEstimand,
    pub(crate) indexer: TemporalIndexer,
    pub(crate) identifier: IdentifierId,
    pub(crate) estimator: EstimatorId,
    pub(crate) bootstrap_replicates: u32,
    pub(crate) refute: RefuteSuite,
}

impl CheckedTemporalEffectOperation {
    /// Check and freeze the graph, query, selected proof, and estimator family.
    pub(crate) fn checked(
        premises: CheckedTemporalEffectOperationPremises<'_>,
    ) -> Result<Self, CausalError> {
        let CheckedTemporalEffectOperationPremises {
            graph,
            query,
            identification,
            estimand,
            indexer,
            identifier,
            estimator,
            bootstrap_replicates,
            refute,
        } = premises;
        query.validate()?;
        if identifier != IdentifierId::TemporalBackdoorUnfolded {
            return Err(CausalError::Compile {
                message: "temporal effect requires the retained unfolded backdoor proof".into(),
            });
        }
        let expected = EstimatorId::temporal_effect_procedure(query, false);
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
        if !proof_supplies(&identification, &estimand) {
            return Err(CausalError::Compile {
                message: "temporal effect estimand is absent from its retained proof".into(),
            });
        }
        temporal_proof_binds_query(&identification, query, &indexer)?;
        Ok(Self {
            graph: graph.clone(),
            graph_nodes: graph.nodes().to_vec(),
            graph_edges: graph.directed_edge_pairs_sorted(),
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

    crate::analysis::checked_ref! {
        #[must_use]
        pub(crate) fn query(&self) -> &TemporalEffectQuery { &self.query }
        #[must_use]
        pub(crate) fn identification(&self) -> &IdentificationResult { &self.identification }
        #[must_use]
        pub(crate) fn estimand(&self) -> &IdentifiedEstimand { &self.estimand }
        #[must_use]
        pub(crate) fn indexer(&self) -> &TemporalIndexer { &self.indexer }
        #[must_use]
        pub(crate) fn graph(&self) -> &TemporalDag { &self.graph }
    }

    #[must_use]
    pub(crate) fn procedure(&self) -> Procedure {
        Procedure {
            identifier: self.identifier,
            estimator: self.estimator,
            bootstrap_replicates: self.bootstrap_replicates,
            validation: self.refute,
        }
    }

    #[must_use]
    pub(crate) fn matches_graph(&self, graph: &TemporalDag) -> bool {
        if graph.nodes() != self.graph_nodes {
            return false;
        }
        graph.directed_edge_pairs_sorted() == self.graph_edges
    }

    /// Canonical structural signature of the retained temporal template.
    ///
    /// Node order is the template order. Edges are sorted endpoints, written
    /// as integers, so the string does not follow `Debug`.
    #[must_use]
    pub(crate) fn graph_signature(&self) -> String {
        temporal_structure(&self.graph_nodes, &self.graph_edges)
    }

    #[must_use]
    pub(crate) fn matches_query(&self, query: &CausalQuery) -> bool {
        matches!(query, CausalQuery::TemporalEffect(candidate) if candidate == &self.query)
    }
}

/// Structural signature of a temporal DAG: template node order, then sorted
/// directed endpoints. The text is the identity, not `Debug` output.
#[must_use]
pub(crate) fn temporal_dag_signature(graph: &TemporalDag) -> String {
    temporal_structure(graph.nodes(), &graph.directed_edge_pairs_sorted())
}

/// `edges` are ascending `(parent, child)` dense-id pairs.
fn temporal_structure(nodes: &[NodeRef], edges: &[(u32, u32)]) -> String {
    let mut out = String::from("nodes=");
    for (index, node) in nodes.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        match node {
            NodeRef::Static(variable) => {
                out.push('s');
                push_u32(&mut out, variable.raw());
            }
            NodeRef::Lagged { variable, lag } => {
                out.push('l');
                push_u32(&mut out, variable.raw());
                out.push('@');
                push_u32(&mut out, lag.raw());
            }
            NodeRef::Context { variable, environment } => {
                out.push('c');
                push_u32(
                    &mut out,
                    environment.map_or(u32::MAX, antecedent_core::EnvironmentId::raw),
                );
                out.push(':');
                push_u32(&mut out, variable.raw());
            }
            NodeRef::Unfolded { variable, offset } => {
                out.push('u');
                push_u32(&mut out, variable.raw());
                out.push('@');
                out.push_str(&offset.to_string());
            }
        }
    }
    out.push_str(";edges=");
    for (index, (from, to)) in edges.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        push_u32(&mut out, *from);
        out.push('>');
        push_u32(&mut out, *to);
    }
    out
}

fn push_u32(out: &mut String, value: u32) {
    out.push_str(&value.to_string());
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
    let expected_treatment = indexer.dense_id(antecedent_core::TemporalNodeKey {
        variable: query.treatment,
        offset: query.try_treatment_offset()?,
    })?;
    let expected_outcome = indexer.dense_id(antecedent_core::TemporalNodeKey {
        variable: query.outcome,
        offset: query.outcome_offset()?,
    })?;
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

#[cfg(test)]
mod signature_tests {
    use super::temporal_dag_signature;
    use antecedent_core::{Lag, VariableId};
    use antecedent_graph::TemporalDag;

    #[test]
    fn temporal_dag_signature_records_endpoints() {
        let mut graph = TemporalDag::empty();
        let from = graph.add_lagged(VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
        let to = graph.add_lagged(VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
        graph.insert_directed(from, to).unwrap();
        let signature = temporal_dag_signature(&graph);
        assert!(signature.starts_with("nodes=l0@1,l1@0;edges="), "{signature}");
        assert!(signature.contains(&format!("{}>{}", from.raw(), to.raw())), "{signature}");
        assert!(!signature.contains("TemporalDag"), "{signature}");
    }
}
