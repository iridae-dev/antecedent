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
use super::*;
use crate::analysis::prepared::CachedGraphPosteriorIdentification;

/// Frozen response route over a DAG, CPDAG or PAG graph posterior.
#[derive(Clone, Debug)]
pub(crate) struct CheckedGraphPosteriorResponse {
    posterior: Arc<GraphPosterior>,
    query: ResponseQuery,
    identification: Arc<CachedGraphPosteriorIdentification>,
    physical: PhysicalExecutionPlan,
    identifier: IdentifierId,
    estimator: EstimatorId,
    inference: InferenceMode,
    validation: RefuteSuite,
    latency_mode: Option<LatencyMode>,
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
        posterior: GraphPosterior,
        query: ResponseQuery,
        identification: CachedGraphPosteriorIdentification,
        physical: PhysicalExecutionPlan,
        identifier: IdentifierId,
        estimator: EstimatorId,
        inference: InferenceMode,
        validation: RefuteSuite,
        latency_mode: Option<LatencyMode>,
    ) -> Result<Self, CausalError> {
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
        if (!intervention_response && validation != RefuteSuite::None)
            || !matches!(validation, RefuteSuite::None | RefuteSuite::Cheap | RefuteSuite::Full)
        {
            return Err(CausalError::Unsupported {
                message: "checked graph-posterior response supports no refuters for curves and none/cheap/full for intervention responses",
            });
        }
        let expected_estimator = EstimatorId::static_response_for(&query.functional, &inference);
        let expected_identifier =
            if class_atoms { DEFAULT_PAG_IDENTIFIER_ID } else { DEFAULT_RESPONSE_IDENTIFIER_ID };
        if estimator != expected_estimator
            || identifier != expected_identifier
            || physical
                .logical
                .record
                .identifier
                .as_deref()
                .is_some_and(|name| name != identifier.as_str())
            || physical
                .logical
                .record
                .estimator
                .as_deref()
                .is_some_and(|name| name != estimator.as_str())
        {
            return Err(CausalError::Unsupported {
                message: "checked graph-posterior response procedure does not match the sealed plan or inference mode",
            });
        }
        let witness = CausalQuery::AverageEffect(response_witness_ate(&query)?);
        if posterior.weights.as_ref() != identification.graphs.weights.as_ref()
            || posterior.graph_keys.as_ref() != identification.graphs.graph_keys.as_ref()
            || posterior.n_graphs != identification.graphs.n_samples
            || identification.graphs.identified.len() != posterior.n_graphs
        {
            return Err(CausalError::Unsupported {
                message: "graph-posterior response identification cache does not match frozen posterior atoms",
            });
        }
        let atoms_bound = if class_atoms {
            identification.atoms.is_empty()
                && identification.class_atoms.iter().all(|atom| {
                    posterior.graph_keys.contains(&atom.key) && atom.identification.query == witness
                })
        } else {
            identification.class_atoms.is_empty()
                && identification.atoms.iter().all(|atom| {
                    posterior.graph_keys.contains(&atom.key) && atom.identification.query == witness
                })
        };
        if !atoms_bound {
            return Err(CausalError::Unsupported {
                message: "graph-posterior response identification cache binds another target or atom kind",
            });
        }
        Ok(Self {
            posterior: Arc::new(posterior),
            query,
            identification: Arc::new(identification),
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
    pub(crate) fn identification(&self) -> &CachedGraphPosteriorIdentification {
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
        bound.graph_posterior_identification_cache =
            Some(Arc::new(operation.identification().clone()));
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
