//! Sealed DAG, CPDAG and PAG graph-posterior response execution.
//!
//! The frozen posterior atoms, their prepare-time identification cache, the
//! exact response query, the inference mode and the validation suite are
//! retained together. Execution mixes the same atoms the plan was sealed on,
//! including the unidentified, failed and subsampled-out mass, without
//! re-identifying or re-reading the procedure from the builder.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::response_path::{graph_posterior_response_supported, response_witness_ate};
use super::{
    Arc, CausalError, CausalQuery, DEFAULT_PAG_IDENTIFIER_ID, DEFAULT_RESPONSE_IDENTIFIER_ID,
    DataInput, EstimatorId, ExecutionContext, GraphPosterior, IdentifierId, InferenceMode,
    LatencyMode, PhysicalExecutionPlan, RefuteSuite, ResponseFunctional, ResponseQuery,
    StudyResult, TabularData,
};
use crate::analysis::posterior_binding::{CacheKeys, validate_posterior_cache};
use crate::analysis::prepared::CachedGraphPosteriorIdentification;
use crate::analysis::prepared::Inspect;
use crate::analysis::route_guards::inference_label;

/// Frozen response route over a DAG, CPDAG or PAG graph posterior.
#[derive(Clone, Debug)]
pub(crate) struct CheckedGraphPosteriorResponse {
    posterior: Arc<GraphPosterior>,
    query: ResponseQuery,
    identification: Arc<CachedGraphPosteriorIdentification>,
    physical: std::sync::Arc<PhysicalExecutionPlan>,
    identifier: IdentifierId,
    estimator: EstimatorId,
    inference: InferenceMode,
    validation: RefuteSuite,
    latency_mode: Option<LatencyMode>,
}

/// The frozen premises of [`CheckedGraphPosteriorResponse::prepare`], carried together.
pub(crate) struct CheckedGraphPosteriorResponsePremises {
    pub(crate) posterior: GraphPosterior,
    pub(crate) query: ResponseQuery,
    pub(crate) identification: Arc<CachedGraphPosteriorIdentification>,
    pub(crate) physical: std::sync::Arc<PhysicalExecutionPlan>,
    pub(crate) identifier: IdentifierId,
    pub(crate) estimator: EstimatorId,
    pub(crate) inference: InferenceMode,
    pub(crate) validation: RefuteSuite,
    pub(crate) latency_mode: Option<LatencyMode>,
}

impl CheckedGraphPosteriorResponse {
    /// Seal a graph-posterior response over its prepare-time atom cache.
    ///
    /// # Errors
    ///
    /// Refuses ADMG atoms (they have their own sealed family), a query outside
    /// the complete-observation mean response family, a validation suite the
    /// family does not license, a procedure that does not match the inference
    /// mode, or a cache that does not describe the frozen posterior.
    pub(crate) fn prepare(
        premises: CheckedGraphPosteriorResponsePremises,
    ) -> Result<Self, CausalError> {
        let CheckedGraphPosteriorResponsePremises {
            posterior,
            query,
            identification,
            physical,
            identifier,
            estimator,
            inference,
            validation,
            latency_mode,
        } = premises;
        let class_atoms = match posterior.atom_kind {
            antecedent_discovery::GraphPosteriorAtomKind::Dag => false,
            antecedent_discovery::GraphPosteriorAtomKind::Cpdag
            | antecedent_discovery::GraphPosteriorAtomKind::Pag => true,
            _ => {
                return Err(CausalError::Unsupported {
                    message: "checked graph-posterior response requires DAG, CPDAG, or PAG atoms",
                });
            }
        };
        graph_posterior_response_supported(&query)?;
        if query.target_population != antecedent_core::TargetPopulation::AllObserved
            || !matches!(query.outcome_functional, antecedent_core::OutcomeFunctional::Mean)
            || !matches!(
                query.functional,
                ResponseFunctional::MeanCurve { .. }
                    | ResponseFunctional::InterventionResponse { .. }
            )
        {
            return Err(CausalError::Unsupported {
                message: "checked graph-posterior response requires an all-observed mean MeanCurve or InterventionResponse",
            });
        }
        let intervention_response =
            matches!(query.functional, ResponseFunctional::InterventionResponse { .. });
        if (!intervention_response && validation != RefuteSuite::None) || !validation.is_point() {
            return Err(CausalError::Unsupported {
                message: "checked graph-posterior response supports no refuters for curves and none/cheap/full for intervention responses",
            });
        }
        let expected_estimator = EstimatorId::static_response_for(&query.functional, &inference);
        let expected_identifier =
            if class_atoms { DEFAULT_PAG_IDENTIFIER_ID } else { DEFAULT_RESPONSE_IDENTIFIER_ID };
        if estimator != expected_estimator
            || identifier != expected_identifier
            || !physical.binding().names(identifier, estimator)
        {
            return Err(CausalError::Unsupported {
                message: "checked graph-posterior response procedure does not match the sealed plan or inference mode",
            });
        }
        let witness = CausalQuery::AverageEffect(response_witness_ate(&query)?);
        validate_posterior_cache(
            &posterior,
            &*identification,
            CacheKeys::PosteriorKeys,
            "graph-posterior response identification cache does not match frozen posterior atoms",
        )?;
        let atoms_bound = if class_atoms {
            identification.atoms.is_empty()
                && identification
                    .class_atoms
                    .iter()
                    .all(|atom| atom.identification.query == witness)
        } else {
            identification.class_atoms.is_empty()
                && identification.atoms.iter().all(|atom| atom.identification.query == witness)
        };
        if !atoms_bound {
            return Err(CausalError::Unsupported {
                message: "graph-posterior response identification cache binds another target or atom kind",
            });
        }
        Ok(Self {
            posterior: Arc::new(posterior),
            query,
            identification,
            physical,
            identifier,
            estimator,
            inference,
            validation,
            latency_mode,
        })
    }

    pub(crate) fn posterior(&self) -> &GraphPosterior {
        &self.posterior
    }
    pub(crate) fn query(&self) -> &ResponseQuery {
        &self.query
    }
    pub(crate) fn identification(&self) -> &Arc<CachedGraphPosteriorIdentification> {
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
    pub(crate) const fn latency_mode(&self) -> Option<LatencyMode> {
        self.latency_mode
    }
}

impl super::Study {
    /// Execute the retained graph-posterior response through the atom mixing
    /// executor for its atom kind, binding every retained field before
    /// dispatch so the click cannot read a different query, posterior, cache,
    /// inference, or validation suite from the study snapshot.
    pub(in crate::analysis) fn execute_checked_graph_posterior_response(
        &self,
        data: &TabularData,
        operation: &CheckedGraphPosteriorResponse,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let mut bound = self.clone();
        bound.data = DataInput::Tabular(data.clone());
        bound.query = CausalQuery::Response(operation.query().clone());
        bound.graph_posterior = Some(operation.posterior().clone());
        bound.install_identification(crate::analysis::IdentificationCache::GraphPosterior(Arc::clone(operation.identification())));
        bound.inference = operation.inference().clone();
        bound.refute = operation.validation();
        bound.latency_mode = operation.latency_mode();
        bound.estimator = Some(operation.procedure().1);
        bound.identifier = Some(operation.procedure().0);
        bound.custom_validators = Vec::new();
        match operation.posterior().atom_kind {
            antecedent_discovery::GraphPosteriorAtomKind::Dag => bound
                .execute_graph_posterior_response(
                    data,
                    operation.posterior(),
                    operation.query(),
                    operation.physical(),
                    ctx,
                ),
            antecedent_discovery::GraphPosteriorAtomKind::Cpdag
            | antecedent_discovery::GraphPosteriorAtomKind::Pag => bound
                .execute_class_graph_posterior_response(
                    data,
                    operation.posterior(),
                    operation.query(),
                    operation.physical(),
                    ctx,
                ),
            _ => Err(CausalError::Compile {
                message: "checked graph-posterior response retained an unsupported atom kind"
                    .into(),
            }),
        }
    }
}

/// Frozen static or temporal graph-posterior response: atom keys, weights,
/// identification flags, and the selected procedure, inference and validation.
#[derive(Clone, Debug, PartialEq)]
pub struct CheckedGraphPosteriorResponseInfo {
    /// Response target retained at preparation.
    pub query: antecedent_core::ResponseQuery,
    /// Atom kind of the frozen posterior.
    pub atom_kind: antecedent_discovery::GraphPosteriorAtomKind,
    /// Identifier fixed at preparation.
    pub identifier: crate::strategy_table::IdentifierId,
    /// Estimator fixed at preparation.
    pub estimator: crate::strategy_table::EstimatorId,
    /// Inference procedure label (`frequentist` or `bayesian:<backend>`).
    pub inference: std::sync::Arc<str>,
    /// Frozen graph identity per posterior atom.
    pub graph_keys: std::sync::Arc<[u64]>,
    /// Frozen posterior weight per atom.
    pub weights: std::sync::Arc<[f64]>,
    /// Identified and unidentified atom status, preserving unresolved mass.
    pub identified: std::sync::Arc<[antecedent_prob::GraphIdentFlag]>,
    /// Validation suite fixed at preparation.
    pub validation: crate::analysis::builder::RefuteSuite,
}

impl Inspect for CheckedGraphPosteriorResponse {
    type Info = CheckedGraphPosteriorResponseInfo;

    fn info(&self) -> CheckedGraphPosteriorResponseInfo {
        let (identifier, estimator) = self.procedure();
        CheckedGraphPosteriorResponseInfo {
            query: self.query().clone(),
            atom_kind: self.posterior().atom_kind,
            identifier,
            estimator,
            inference: inference_label(self.inference()),
            graph_keys: Arc::clone(&self.identification().graphs.graph_keys),
            weights: Arc::clone(&self.identification().graphs.weights),
            identified: Arc::clone(&self.identification().graphs.identified),
            validation: self.validation(),
        }
    }
}
