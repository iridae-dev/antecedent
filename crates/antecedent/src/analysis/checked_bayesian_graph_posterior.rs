//! Sealed Bayesian average or conditional effect composition over a static
//! DAG posterior.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::AverageEffectQuery;
use antecedent_discovery::{GraphPosterior, GraphPosteriorAtomKind};
use antecedent_estimate::OverlapPolicy;

use crate::{
    CausalError, EstimatorSpec, InferenceMode, RefuteSuite,
    analysis::prepared::CachedGraphPosteriorIdentification, planner::PhysicalExecutionPlan,
};

use super::GraphPosteriorEffectTarget;

/// Frozen graph family and per-atom Bayesian procedure for a mean average or
/// conditional effect. The target records which query kind the click executes.
#[derive(Clone, Debug)]
pub(crate) struct CheckedBayesianGraphPosteriorAte {
    posterior: Arc<GraphPosterior>,
    target: GraphPosteriorEffectTarget,
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
        target: GraphPosteriorEffectTarget,
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
        let query = target.inner();
        let atom_query = target.atom_identification_query();
        let licensed = target.bayesian_estimator();
        if posterior.atom_kind != GraphPosteriorAtomKind::Dag
            || identification.graphs.graph_keys != posterior.graph_keys
            || identification.graphs.weights != posterior.weights
            || identification.atoms.iter().any(|atom| atom.identification.query != atom_query)
            || query.treatment == query.outcome
            || !matches!(query.outcome_functional, antecedent_core::OutcomeFunctional::Mean)
            || query.target_population != antecedent_core::TargetPopulation::AllObserved
            || !matches!(validation, RefuteSuite::None | RefuteSuite::Cheap | RefuteSuite::Full)
            || procedure.id() != licensed
            || physical.logical.record.estimator.as_deref() != Some(licensed.as_str())
            || physical.logical.record.identifier.as_deref()
                != Some(crate::strategy_table::IdentifierId::BackdoorAdjustment.as_str())
        {
            return Err(CausalError::Unsupported {
                message: "checked Bayesian graph-posterior ATE has incompatible graph, target, identification, or procedure",
            });
        }
        Ok(Self {
            posterior: Arc::new(posterior),
            target,
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
    /// The average-effect query every posterior atom was identified for.
    pub(crate) const fn query(&self) -> &AverageEffectQuery {
        self.target.inner()
    }
    /// The sealed query kind, including conditional modifiers.
    pub(crate) const fn target(&self) -> &GraphPosteriorEffectTarget {
        &self.target
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
