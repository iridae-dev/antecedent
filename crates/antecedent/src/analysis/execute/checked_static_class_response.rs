//! Sealed explicit/accepted CPDAG and PAG response execution.
//!
//! The completion envelope identified at prepare time, the exact response
//! query, the selected procedure and the inference mode are retained together,
//! so a click or refresh mixes the same completions the plan was sealed on and
//! cannot re-identify against another class member. Unresolved completion mass
//! keeps the envelope executor's treatment: the envelope executor still withholds a scalar
//! or refuses when no completion identifies the response.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::response_path::class_aware_response_supported;
use super::{
    Arc, CausalError, CausalQuery, DEFAULT_PAG_IDENTIFIER_ID, DataInput, EstimatorId,
    ExecutionContext, GraphClass, IdentifierId, InferenceMode, PhysicalExecutionPlan, RefuteSuite,
    ResponseQuery, Study, StudyResult, TabularData,
};
use crate::analysis::prepared::Inspect;
use crate::analysis::route_guards::inference_label;
use crate::analysis::route_guards::mean_all_observed;
use crate::analysis::{StaticClassGraph, StaticClassIdentification};

/// Frozen class response route over a supplied CPDAG or PAG.
#[derive(Clone, Debug)]
pub(crate) struct CheckedStaticClassResponse {
    graph: StaticClassGraph,
    query: ResponseQuery,
    identification: StaticClassIdentification,
    physical: std::sync::Arc<PhysicalExecutionPlan>,
    identifier: IdentifierId,
    estimator: EstimatorId,
    inference: InferenceMode,
    validation: RefuteSuite,
}

/// The frozen premises of [`CheckedStaticClassResponse::prepare`], carried together.
pub(crate) struct CheckedStaticClassResponsePremises {
    pub(crate) graph: StaticClassGraph,
    pub(crate) query: ResponseQuery,
    pub(crate) identification: StaticClassIdentification,
    pub(crate) physical: std::sync::Arc<PhysicalExecutionPlan>,
    pub(crate) identifier: IdentifierId,
    pub(crate) estimator: EstimatorId,
    pub(crate) inference: InferenceMode,
    pub(crate) validation: RefuteSuite,
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
        study.plain_fixed_route()
            && matches!(study.graph.class(), GraphClass::Cpdag | GraphClass::Pag)
            && study.settings.refute == RefuteSuite::None
            && class_aware_response_supported(query)
            && mean_all_observed(&query.outcome_functional, &query.target_population)
            && study.declared.estimator.is_none_or(|id| {
                id == EstimatorId::static_response_for(&query.functional, &study.settings.inference)
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
        premises: CheckedStaticClassResponsePremises,
    ) -> Result<Self, CausalError> {
        let CheckedStaticClassResponsePremises {
            graph,
            query,
            identification,
            physical,
            identifier,
            estimator,
            inference,
            validation,
        } = premises;
        if !class_aware_response_supported(&query)
            || !mean_all_observed(&query.outcome_functional, &query.target_population)
        {
            return Err(CausalError::Unsupported {
                message: "checked class response requires a complete-observation all-observed mean MeanCurve, InterventionResponse or derivative functional",
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
            || !physical.binding().names(identifier, estimator)
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
            || self.declared.tiered.is_some()
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
        bound.settings.inference = operation.inference().clone();
        bound.settings.refute = operation.validation();
        bound.declared.estimator = Some(operation.procedure().1);
        bound.declared.identifier = Some(operation.procedure().0);
        bound.declared.custom_validators = Vec::new();
        match operation.identification() {
            StaticClassIdentification::Cpdag(cache) => {
                bound.install_identification(crate::analysis::IdentificationCache::Cpdag(
                    Arc::clone(cache),
                ));
                if bound.graph.as_cpdag().is_none() {
                    return Err(CausalError::Compile {
                        message: "checked CPDAG response requires its retained CPDAG".into(),
                    });
                }
            }
            StaticClassIdentification::Pag(cache) => {
                bound.install_identification(crate::analysis::IdentificationCache::Pag(
                    Arc::clone(cache),
                ));
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

/// Frozen explicit/accepted CPDAG or PAG response route.
#[derive(Clone, Debug, PartialEq)]
pub struct CheckedStaticClassResponseInfo {
    /// Response target retained at preparation.
    pub query: antecedent_core::ResponseQuery,
    /// Graph class the completion envelope was enumerated for.
    pub graph_class: crate::accepted::GraphClass,
    /// Identifier fixed at preparation.
    pub identifier: crate::strategy_table::IdentifierId,
    /// Estimator fixed at preparation.
    pub estimator: crate::strategy_table::EstimatorId,
    /// Inference procedure label (`frequentist` or `bayesian:<backend>`).
    pub inference: std::sync::Arc<str>,
    /// Validation suite fixed at preparation.
    pub validation: crate::analysis::builder::RefuteSuite,
    /// Completions retained in the frozen envelope.
    pub completion_count: usize,
    /// Identified completion mass retained in the frozen envelope.
    pub identified_mass: f64,
    /// Unidentified completion mass retained in the frozen envelope.
    pub unidentified_mass: f64,
}

impl Inspect for CheckedStaticClassResponse {
    type Info = CheckedStaticClassResponseInfo;

    fn info(&self) -> CheckedStaticClassResponseInfo {
        let (identifier, estimator) = self.procedure();
        let (completion_count, identified_mass, unidentified_mass) = self.envelope_summary();
        CheckedStaticClassResponseInfo {
            query: self.query().clone(),
            graph_class: self.graph_class(),
            identifier,
            estimator,
            inference: inference_label(self.inference()),
            validation: self.validation(),
            completion_count,
            identified_mass,
            unidentified_mass,
        }
    }
}
