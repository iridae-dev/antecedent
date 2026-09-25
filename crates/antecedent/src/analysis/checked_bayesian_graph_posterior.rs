//! Sealed Bayesian ATE composition over a static DAG posterior.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::AverageEffectQuery;
use antecedent_discovery::{GraphPosterior, GraphPosteriorAtomKind};
use antecedent_estimate::OverlapPolicy;

use crate::{
    CausalError, EstimatorId, EstimatorSpec, InferenceMode, RefuteSuite,
    analysis::prepared::CachedGraphPosteriorIdentification, planner::PhysicalExecutionPlan,
};

/// Frozen graph family and per-atom Bayesian procedure for a mean ATE.
#[derive(Clone, Debug)]
pub(crate) struct CheckedBayesianGraphPosteriorAte {
    posterior: Arc<GraphPosterior>,
    query: AverageEffectQuery,
    identification: Arc<CachedGraphPosteriorIdentification>,
    inference: InferenceMode,
    procedure: EstimatorSpec,
    physical: PhysicalExecutionPlan,
    validation: RefuteSuite,
    overlap: OverlapPolicy,
    latency_mode: Option<super::latency::LatencyMode>,
}

impl CheckedBayesianGraphPosteriorAte {
    pub(crate) fn prepare(
        posterior: GraphPosterior,
        query: AverageEffectQuery,
        identification: CachedGraphPosteriorIdentification,
        inference: InferenceMode,
        procedure: EstimatorSpec,
        physical: PhysicalExecutionPlan,
        validation: RefuteSuite,
        overlap: OverlapPolicy,
        latency_mode: Option<super::latency::LatencyMode>,
    ) -> Result<Self, CausalError> {
        let InferenceMode::Bayesian(_) = inference else {
            return Err(CausalError::Unsupported {
                message: "checked Bayesian graph-posterior ATE requires Bayesian inference",
            });
        };
        if posterior.atom_kind != GraphPosteriorAtomKind::Dag
            || identification.graphs.graph_keys != posterior.graph_keys
            || identification.graphs.weights != posterior.weights
            || identification.atoms.iter().any(|atom| {
                atom.identification.query
                    != antecedent_core::CausalQuery::AverageEffect(query.clone())
            })
            || query.treatment == query.outcome
            || !matches!(query.outcome_functional, antecedent_core::OutcomeFunctional::Mean)
            || query.target_population != antecedent_core::TargetPopulation::AllObserved
            || !matches!(validation, RefuteSuite::None | RefuteSuite::Cheap | RefuteSuite::Full)
            || procedure.id() != EstimatorId::BayesianGcomp
            || physical.logical.record.estimator.as_deref()
                != Some(EstimatorId::BayesianGcomp.as_str())
            || physical.logical.record.identifier.as_deref()
                != Some(crate::strategy_table::IdentifierId::BackdoorAdjustment.as_str())
        {
            return Err(CausalError::Unsupported {
                message: "checked Bayesian graph-posterior ATE has incompatible graph, target, identification, or procedure",
            });
        }
        Ok(Self {
            posterior: Arc::new(posterior),
            query,
            identification: Arc::new(identification),
            inference,
            procedure,
            physical,
            validation,
            overlap,
            latency_mode,
        })
    }

    pub(crate) fn posterior(&self) -> &GraphPosterior {
        &self.posterior
    }
    pub(crate) fn query(&self) -> &AverageEffectQuery {
        &self.query
    }
    pub(crate) fn identification(&self) -> &CachedGraphPosteriorIdentification {
        &self.identification
    }
    pub(crate) fn inference(&self) -> &InferenceMode {
        &self.inference
    }
    pub(crate) fn procedure(&self) -> &EstimatorSpec {
        &self.procedure
    }
    pub(crate) fn physical(&self) -> &PhysicalExecutionPlan {
        &self.physical
    }
    pub(crate) const fn validation(&self) -> RefuteSuite {
        self.validation
    }
    pub(crate) const fn overlap(&self) -> OverlapPolicy {
        self.overlap
    }
    pub(crate) const fn latency_mode(&self) -> Option<super::latency::LatencyMode> {
        self.latency_mode
    }
}
