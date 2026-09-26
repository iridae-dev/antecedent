//! Sealed explicit/accepted CPDAG and PAG response execution.
//!
//! The completion envelope identified at prepare time, the exact response
//! query, the selected procedure and the inference mode are retained together,
//! so a click or refresh mixes the same completions the plan was sealed on and
//! cannot re-identify against another class member. Unresolved completion mass
//! keeps its legacy treatment: the envelope executor still withholds a scalar
//! or refuses when no completion identifies the response.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::response_path::class_aware_response_supported;
use super::*;
use crate::analysis::route_guards::mean_all_observed;
use crate::analysis::{StaticClassGraph, StaticClassIdentification};

/// Frozen class response route over a supplied CPDAG or PAG.
#[derive(Clone, Debug)]
pub(crate) struct CheckedStaticClassResponse {
    graph: StaticClassGraph,
    query: ResponseQuery,
    identification: StaticClassIdentification,
    physical: PhysicalExecutionPlan,
    identifier: IdentifierId,
    estimator: EstimatorId,
    inference: InferenceMode,
    validation: RefuteSuite,
}

impl CheckedStaticClassResponse {
    /// Whether `study` is a coordinate this operation seals: a tabular
    /// complete-observation mean class response on a fixed CPDAG or PAG with
    /// validation `none`. The one-shot facade and the prepared sealer share
    /// this predicate.
    #[must_use]
    pub(crate) fn admits(study: &Study) -> bool {
        let CausalQuery::Response(query) = &study.query else {
            return false;
        };
        study.graph_posterior.is_none()
            && study.tiered.is_none()
            && matches!(study.graph.class(), GraphClass::Cpdag | GraphClass::Pag)
            && study.fixed_structure()
            && study.custom_validators.is_empty()
            && study.refute == RefuteSuite::None
            && class_aware_response_supported(query)
            && mean_all_observed(&query.outcome_functional, &query.target_population)
            && study.estimator.is_none_or(|id| {
                id == EstimatorId::static_response_for(&query.functional, &study.inference)
            })
    }

    /// Seal a class response over its prepare-time completion envelope.
    ///
    /// # Errors
    ///
    /// Refuses a query outside the complete-observation mean response family,
    /// a validation suite other than `none`, a procedure that does not match the
    /// inference mode, or an envelope identified for another target.
    pub(crate) fn prepare(
        graph: StaticClassGraph,
        query: ResponseQuery,
        identification: StaticClassIdentification,
        physical: PhysicalExecutionPlan,
        identifier: IdentifierId,
        estimator: EstimatorId,
        inference: InferenceMode,
        validation: RefuteSuite,
    ) -> Result<Self, CausalError> {
        if !class_aware_response_supported(&query)
            || query.target_population != antecedent_core::TargetPopulation::AllObserved
            || !matches!(query.outcome_functional, antecedent_core::OutcomeFunctional::Mean)
        {
            return Err(CausalError::Unsupported {
                message: "checked class response requires a complete-observation all-observed mean MeanCurve or InterventionResponse",
            });
        }
        if validation != RefuteSuite::None {
            return Err(CausalError::Unsupported {
                message: "checked class response supports validation none only",
            });
        }
        let expected_estimator = EstimatorId::static_response_for(&query.functional, &inference);
        if estimator != expected_estimator
            || identifier != DEFAULT_PAG_IDENTIFIER_ID
            || physical.logical.record.identifier.as_deref() != Some(identifier.as_str())
            || physical.logical.record.estimator.as_deref() != Some(estimator.as_str())
        {
            return Err(CausalError::Unsupported {
                message: "checked class response procedure does not match the sealed plan or inference mode",
            });
        }
        // The prepare-time envelope is identified for the response itself
        // (class-aware response identification), not for a scalar witness. An
        // envelope with no identified completion mass is retained as is: the
        // envelope executor refuses it as not identified, exactly as before.
        let response_query = CausalQuery::Response(query.clone());
        let bound = match (&graph, &identification) {
            (StaticClassGraph::Cpdag(_), StaticClassIdentification::Cpdag(cache)) => {
                cache.identification.query == response_query
            }
            (StaticClassGraph::Pag(_), StaticClassIdentification::Pag(cache)) => {
                cache.identification.query == response_query
            }
            _ => false,
        };
        if !bound {
            return Err(CausalError::Unsupported {
                message: "checked class response envelope does not bind the response witness on the sealed graph class",
            });
        }
        Ok(Self {
            graph,
            query,
            identification,
            physical,
            identifier,
            estimator,
            inference,
            validation,
        })
    }

    pub(crate) fn graph_class(&self) -> GraphClass {
        match self.graph {
            StaticClassGraph::Cpdag(_) => GraphClass::Cpdag,
            StaticClassGraph::Pag(_) => GraphClass::Pag,
        }
    }
    pub(crate) fn query(&self) -> &ResponseQuery {
        &self.query
    }
    pub(crate) fn identification(&self) -> &StaticClassIdentification {
        &self.identification
    }
    pub(crate) fn physical(&self) -> &PhysicalExecutionPlan {
        &self.physical
    }
    pub(crate) const fn procedure(&self) -> (IdentifierId, EstimatorId) {
        (self.identifier, self.estimator)
    }
    pub(crate) fn inference(&self) -> &InferenceMode {
        &self.inference
    }
    pub(crate) const fn validation(&self) -> RefuteSuite {
        self.validation
    }

    /// Completion count and identified mass of the frozen envelope.
    pub(crate) fn envelope_summary(&self) -> (usize, f64, f64) {
        self.identification.envelope_summary()
    }
}

impl super::Study {
    /// Execute the retained class response through the envelope executor,
    /// binding every retained field before dispatch so the click cannot read a
    /// different query, inference, validation suite, or completion cache from
    /// the study snapshot.
    pub(in crate::analysis) fn execute_checked_static_class_response(
        &self,
        data: &TabularData,
        operation: &CheckedStaticClassResponse,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        if self.graph.class() != operation.graph_class()
            || self.graph_posterior.is_some()
            || self.tiered.is_some()
        {
            return Err(CausalError::Compile {
                message:
                    "prepared class response operation no longer matches the study graph class"
                        .into(),
            });
        }
        let mut bound = self.clone();
        bound.data = DataInput::Tabular(data.clone());
        bound.query = CausalQuery::Response(operation.query().clone());
        bound.inference = operation.inference().clone();
        bound.refute = operation.validation();
        bound.estimator = Some(operation.procedure().1);
        bound.identifier = Some(operation.procedure().0);
        bound.custom_validators = Vec::new();
        match operation.identification() {
            StaticClassIdentification::Cpdag(cache) => {
                bound.cpdag_identification_cache = Some(Arc::new(cache.clone()));
                bound.pag_identification_cache = None;
                if bound.graph.as_cpdag().is_none() {
                    return Err(CausalError::Compile {
                        message: "checked CPDAG response requires its retained CPDAG".into(),
                    });
                }
            }
            StaticClassIdentification::Pag(cache) => {
                bound.pag_identification_cache = Some(Arc::new(cache.clone()));
                bound.cpdag_identification_cache = None;
                if operation.physical().static_pag().is_none() {
                    return Err(CausalError::Compile {
                        message: "checked PAG response requires its resolved static PAG".into(),
                    });
                }
            }
        }
        bound.execute_class_response(data, operation.query(), operation.physical(), ctx)
    }
}
