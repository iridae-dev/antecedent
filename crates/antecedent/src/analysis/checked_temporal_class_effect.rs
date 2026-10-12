//! Retained proof and execution settings for an incomplete temporal graph effect.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{CausalQuery, TemporalEffectQuery};
use antecedent_data::DiscoveryEstimationSplit;
use antecedent_identify::result::IdentificationResult;
use antecedent_validate::CustomEffectValidator;
use std::sync::Arc;

use crate::AcceptedGraph;
use crate::GraphClass;
use crate::GraphRef;
use crate::error::CausalError;
use crate::strategy_table::{EstimatorId, IdentifierId};

use super::builder::RefuteSuite;
use super::prepared::CachedTemporalClassIdentification;
use super::proof_support::Procedure;
use super::route_guards::same_accepted_graph;

/// A complete, checked completion envelope for a `TemporalCpdag` or `TemporalPag`
/// pulse or sustained effect. The envelope and its aligned unfolding indexers
/// remain together so a completion can never be evaluated against another
/// completion's time window.
#[derive(Clone)]
pub(crate) struct CheckedTemporalClassEffectOperation {
    source_graph: AcceptedGraph,
    max_completions: Option<usize>,
    query: TemporalEffectQuery,
    bundle: Arc<CachedTemporalClassIdentification>,
    identification: IdentificationResult,
    identifier: IdentifierId,
    estimator: EstimatorId,
    bootstrap_replicates: u32,
    split: Option<DiscoveryEstimationSplit>,
    refute: RefuteSuite,
    custom_validators: Vec<Arc<dyn CustomEffectValidator>>,
}

/// The frozen premises of [`CheckedTemporalClassEffectOperation::checked`], carried together.
pub(crate) struct CheckedTemporalClassEffectOperationPremises<'a> {
    pub(crate) source_graph: AcceptedGraph,
    pub(crate) query: &'a TemporalEffectQuery,
    pub(crate) bundle: &'a CachedTemporalClassIdentification,
    pub(crate) max_completions: Option<usize>,
    pub(crate) identifier: IdentifierId,
    pub(crate) estimator: EstimatorId,
    pub(crate) bootstrap_replicates: u32,
    pub(crate) split: Option<DiscoveryEstimationSplit>,
    pub(crate) refute: RefuteSuite,
    pub(crate) custom_validators: Vec<Arc<dyn CustomEffectValidator>>,
}

impl std::fmt::Debug for CheckedTemporalClassEffectOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CheckedTemporalClassEffectOperation")
            .field("graph_class", &self.source_graph.class())
            .field("structure_version", &self.source_graph.version())
            .field("query", &self.query)
            .field("completion_count", &self.bundle.envelope.envelope.cases.len())
            .field("identifier", &self.identifier)
            .field("estimator", &self.estimator)
            .field("bootstrap_replicates", &self.bootstrap_replicates)
            .field("custom_validator_count", &self.custom_validators.len())
            .finish_non_exhaustive()
    }
}

impl CheckedTemporalClassEffectOperation {
    /// Replace the caller validators; names are checked by the prepared handle.
    pub(crate) fn set_custom_validators(
        &mut self,
        validators: Vec<Arc<dyn CustomEffectValidator>>,
    ) {
        self.custom_validators = validators;
    }

    /// Check and freeze the full completion proof and selected temporal method.
    pub(crate) fn checked(
        premises: CheckedTemporalClassEffectOperationPremises<'_>,
    ) -> Result<Self, CausalError> {
        let CheckedTemporalClassEffectOperationPremises {
            source_graph,
            query,
            bundle,
            max_completions,
            identifier,
            estimator,
            bootstrap_replicates,
            split,
            refute,
            custom_validators,
        } = premises;
        query.validate()?;
        let wrong_class = || CausalError::Compile {
            message: "temporal class effect requires a TemporalCpdag or TemporalPag proof".into(),
        };
        if !matches!(
            source_graph.graph_ref(),
            GraphRef::TemporalCpdag(_) | GraphRef::TemporalPag(_)
        ) {
            return Err(wrong_class());
        }
        if identifier != IdentifierId::GeneralizedAdjustment {
            return Err(CausalError::Compile {
                message: "temporal class effect requires the generalized-adjustment envelope"
                    .into(),
            });
        }
        let expected_estimator = if query.is_multi_step_sustained() {
            EstimatorId::TemporalSequentialGcomp
        } else {
            EstimatorId::TemporalLinearAdjustment
        };
        if estimator != expected_estimator {
            return Err(CausalError::Unsupported {
                message: "checked temporal class effect procedure does not match the retained pulse or sustained schedule",
            });
        }
        if bundle.envelope.indexers.len() != bundle.envelope.envelope.cases.len() {
            return Err(CausalError::Compile {
                message: "temporal class proof does not bind one unfolding indexer per completion"
                    .into(),
            });
        }
        let mut config = antecedent_identify::GeneralizedAdjustmentConfig::default();
        if let Some(max) = max_completions {
            config.max_completions = max;
        }
        let expected = match source_graph.graph_ref() {
            GraphRef::TemporalCpdag(graph) => {
                crate::strategy_table::identify_temporal_cpdag_configured(
                    identifier, graph, query, config,
                )?
            }
            GraphRef::TemporalPag(graph) => {
                crate::strategy_table::identify_temporal_pag_configured(
                    identifier, graph, query, config,
                )?
            }
            _ => return Err(wrong_class()),
        };
        if !same_temporal_class_proof(&expected, &bundle.envelope) {
            return Err(CausalError::Compile {
                message:
                    "cached temporal class proof does not match the retained source graph and query"
                        .into(),
            });
        }
        let bundle = Arc::new(CachedTemporalClassIdentification {
            envelope: Arc::new(expected),
            by_horizon: Vec::new(),
        });
        let envelope = &bundle.envelope.envelope;
        if envelope.cases.is_empty() {
            return Err(CausalError::Compile {
                message: "temporal class proof has no completion cases".into(),
            });
        }
        let identification = super::execute::envelope_to_identification_result_for(
            envelope,
            CausalQuery::TemporalEffect(query.clone()),
        );
        Ok(Self {
            source_graph,
            max_completions,
            query: query.clone(),
            bundle,
            identification,
            identifier,
            estimator,
            bootstrap_replicates,
            split,
            refute,
            custom_validators,
        })
    }

    #[must_use]
    pub(crate) fn graph_class(&self) -> GraphClass {
        self.source_graph.class()
    }

    #[must_use]
    pub(crate) fn structure_version(&self) -> u32 {
        self.source_graph.version()
    }

    #[must_use]
    pub(crate) fn matches_source_graph(&self, graph: &AcceptedGraph) -> bool {
        same_accepted_graph(&self.source_graph, graph)
    }

    #[must_use]
    pub(crate) fn max_completions(&self) -> Option<usize> {
        self.max_completions
    }

    crate::analysis::checked_ref! {
        #[must_use]
        pub(crate) fn query(&self) -> &TemporalEffectQuery { &self.query }
        #[must_use]
        pub(crate) fn bundle(&self) -> &Arc<CachedTemporalClassIdentification> { &self.bundle }
        #[must_use]
        pub(crate) fn identification(&self) -> &IdentificationResult { &self.identification }
    }

    #[must_use]
    pub(crate) fn procedure(&self) -> TemporalClassProcedure<'_> {
        TemporalClassProcedure {
            procedure: Procedure {
                identifier: self.identifier,
                estimator: self.estimator,
                bootstrap_replicates: self.bootstrap_replicates,
                validation: self.refute,
            },
            split: self.split.as_ref(),
            custom_validators: &self.custom_validators,
        }
    }

    #[must_use]
    pub(crate) fn matches_query(&self, query: &CausalQuery) -> bool {
        matches!(query, CausalQuery::TemporalEffect(candidate) if candidate == &self.query)
    }
}

/// The sealed [`Procedure`] together with the split and caller validators the
/// class effect retains beside it; dereferences to the procedure.
pub(crate) struct TemporalClassProcedure<'a> {
    pub(crate) procedure: Procedure,
    pub(crate) split: Option<&'a DiscoveryEstimationSplit>,
    pub(crate) custom_validators: &'a [Arc<dyn CustomEffectValidator>],
}

impl std::ops::Deref for TemporalClassProcedure<'_> {
    type Target = Procedure;

    fn deref(&self) -> &Procedure {
        &self.procedure
    }
}

pub(crate) fn same_temporal_class_proof(
    expected: &antecedent_identify::TemporalClassEnvelope,
    supplied: &antecedent_identify::TemporalClassEnvelope,
) -> bool {
    let left = &expected.envelope;
    let right = &supplied.envelope;
    expected.indexers == supplied.indexers
        && left.cases.len() == right.cases.len()
        && left.status == right.status
        && left.identified_weight.0.to_bits() == right.identified_weight.0.to_bits()
        && left.unidentified_weight.0.to_bits() == right.unidentified_weight.0.to_bits()
        && left.truncated_completions == right.truncated_completions
        && left.cases.iter().zip(&right.cases).all(|(a, b)| {
            a.graph.fingerprint() == b.graph.fingerprint()
                && a.weight.0.to_bits() == b.weight.0.to_bits()
                && a.result.query == b.result.query
                && a.result.status == b.result.status
                && a.result.estimands.len() == b.result.estimands.len()
                && a.result.estimands.iter().zip(&b.result.estimands).all(|(x, y)| {
                    x.method == y.method
                        && x.adjustment_set == y.adjustment_set
                        && x.instruments == y.instruments
                        && x.mediators == y.mediators
                        && x.rd_design == y.rd_design
                        && format!("{:?}", a.result.arena.node(x.functional))
                            == format!("{:?}", b.result.arena.node(y.functional))
                })
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::VariableId;
    use antecedent_identify::{IdentificationEnvelope, TemporalClassEnvelope};

    fn empty_bundle() -> CachedTemporalClassIdentification {
        CachedTemporalClassIdentification {
            envelope: Arc::new(TemporalClassEnvelope {
                envelope: IdentificationEnvelope::from_cases(Vec::new()),
                indexers: Vec::new(),
            }),
            by_horizon: Vec::new(),
        }
    }

    #[test]
    fn class_effect_rejects_a_procedure_that_collapses_a_sustained_schedule() {
        let treatment = VariableId::from_raw(0);
        let outcome = VariableId::from_raw(1);
        let query = TemporalEffectQuery::sustained(treatment, outcome, 1, 1.0);
        let error = CheckedTemporalClassEffectOperation::checked(
            CheckedTemporalClassEffectOperationPremises {
                source_graph: AcceptedGraph::temporal_cpdag(
                    antecedent_graph::TemporalCpdag::empty(),
                )
                .unwrap(),
                query: &query,
                bundle: &empty_bundle(),
                max_completions: None,
                identifier: IdentifierId::GeneralizedAdjustment,
                estimator: EstimatorId::TemporalLinearAdjustment,
                bootstrap_replicates: 0,
                split: None,
                refute: RefuteSuite::None,
                custom_validators: Vec::new(),
            },
        )
        .expect_err("multi-step schedule must retain its sequential estimator");
        assert!(matches!(error, CausalError::Unsupported { .. }));
    }

    #[test]
    fn class_effect_rejects_a_dag_shaped_source_class() {
        let treatment = VariableId::from_raw(0);
        let outcome = VariableId::from_raw(1);
        let query = TemporalEffectQuery::pulse(treatment, outcome, 1.0);
        let error = CheckedTemporalClassEffectOperation::checked(
            CheckedTemporalClassEffectOperationPremises {
                source_graph: AcceptedGraph::temporal_dag(antecedent_graph::TemporalDag::empty()),
                query: &query,
                bundle: &empty_bundle(),
                max_completions: None,
                identifier: IdentifierId::GeneralizedAdjustment,
                estimator: EstimatorId::TemporalLinearAdjustment,
                bootstrap_replicates: 0,
                split: None,
                refute: RefuteSuite::None,
                custom_validators: Vec::new(),
            },
        )
        .expect_err("an incomplete-class operation must retain its source class");
        assert!(matches!(error, CausalError::Compile { .. }));
    }
}
