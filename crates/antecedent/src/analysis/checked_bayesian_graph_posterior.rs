//! Sealed Bayesian average or conditional effect composition over a static
//! DAG posterior.
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::analysis::prepared::Inspect;
use crate::analysis::route_guards::inference_label;
use std::sync::Arc;

use antecedent_core::AverageEffectQuery;
use antecedent_discovery::{GraphPosterior, GraphPosteriorAtomKind};
use antecedent_estimate::OverlapPolicy;

use crate::{
    CausalError, EstimatorSpec, InferenceMode, RefuteSuite,
    analysis::prepared::CachedGraphPosteriorIdentification, planner::PhysicalExecutionPlan,
};

use super::GraphPosteriorEffectTarget;
use super::posterior_binding::{CacheKeys, posterior_cache_matches};

/// Frozen graph family and per-atom Bayesian procedure for a mean average or
/// conditional effect. The target records which query kind the click executes.
#[derive(Clone, Debug)]
pub(crate) struct CheckedBayesianGraphPosteriorAte {
    posterior: Arc<GraphPosterior>,
    target: GraphPosteriorEffectTarget,
    identification: Arc<CachedGraphPosteriorIdentification>,
    inference: InferenceMode,
    procedure: EstimatorSpec,
    physical: std::sync::Arc<PhysicalExecutionPlan>,
    validation: RefuteSuite,
    overlap: OverlapPolicy,
    latency_mode: Option<super::latency::LatencyMode>,
}

/// The frozen premises of [`CheckedBayesianGraphPosteriorAte::prepare`], carried together.
pub(crate) struct CheckedBayesianGraphPosteriorAtePremises {
    pub(crate) posterior: GraphPosterior,
    pub(crate) target: GraphPosteriorEffectTarget,
    pub(crate) identification: Arc<CachedGraphPosteriorIdentification>,
    pub(crate) inference: InferenceMode,
    pub(crate) procedure: EstimatorSpec,
    pub(crate) physical: std::sync::Arc<PhysicalExecutionPlan>,
    pub(crate) validation: RefuteSuite,
    pub(crate) overlap: OverlapPolicy,
    pub(crate) latency_mode: Option<super::latency::LatencyMode>,
}

impl CheckedBayesianGraphPosteriorAte {
    pub(crate) fn prepare(
        premises: CheckedBayesianGraphPosteriorAtePremises,
    ) -> Result<Self, CausalError> {
        let CheckedBayesianGraphPosteriorAtePremises {
            posterior,
            target,
            identification,
            inference,
            procedure,
            physical,
            validation,
            overlap,
            latency_mode,
        } = premises;
        let InferenceMode::Bayesian(_) = inference else {
            return Err(CausalError::Unsupported {
                message: "checked Bayesian graph-posterior ATE requires Bayesian inference",
            });
        };
        let query = target.inner();
        let atom_query = target.atom_identification_query();
        let licensed = target.bayesian_estimator();
        if posterior.atom_kind != GraphPosteriorAtomKind::Dag
            || !posterior_cache_matches(&posterior, &*identification, CacheKeys::PosteriorKeys)
            || identification.atoms.iter().any(|atom| atom.identification.query != atom_query)
            || query.treatment == query.outcome
            || !matches!(query.outcome_functional, antecedent_core::OutcomeFunctional::Mean)
            || query.target_population != antecedent_core::TargetPopulation::AllObserved
            || !validation.is_point()
            || procedure.id() != licensed
            || !physical
                .binding()
                .names(crate::strategy_table::IdentifierId::BackdoorAdjustment, licensed)
        {
            return Err(CausalError::Unsupported {
                message: "checked Bayesian graph-posterior ATE has incompatible graph, target, identification, or procedure",
            });
        }
        Ok(Self {
            posterior: Arc::new(posterior),
            target,
            identification,
            inference,
            procedure,
            physical,
            validation,
            overlap,
            latency_mode,
        })
    }

    crate::analysis::checked_ref! {
        pub(crate) fn posterior(&self) -> &GraphPosterior { &self.posterior }
    }
    /// The average-effect query every posterior atom was identified for.
    pub(crate) const fn query(&self) -> &AverageEffectQuery {
        self.target.inner()
    }
    /// The sealed query kind, including conditional modifiers.
    pub(crate) const fn target(&self) -> &GraphPosteriorEffectTarget {
        &self.target
    }
    crate::analysis::checked_ref! {
        pub(crate) fn identification(&self) -> &Arc<CachedGraphPosteriorIdentification> { &self.identification }
        pub(crate) fn inference(&self) -> &InferenceMode { &self.inference }
        pub(crate) fn procedure(&self) -> &EstimatorSpec { &self.procedure }
        pub(crate) fn physical(&self) -> &PhysicalExecutionPlan { &self.physical }
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

/// Read-only target and atom identities retained for Bayesian DAG graph-posterior ATE.
#[derive(Clone, Debug, PartialEq)]
pub struct CheckedBayesianGraphPosteriorAteInfo {
    /// Average-effect query every posterior atom was identified for.
    pub query: antecedent_core::AverageEffectQuery,
    /// Whether the sealed click estimates a conditional effect over the atoms.
    pub conditional: bool,
    /// Bayesian estimator applied to each identified atom.
    pub estimator: crate::strategy_table::EstimatorId,
    /// Validation suite frozen at preparation.
    pub validation: crate::analysis::builder::RefuteSuite,
    /// Frozen graph identity per posterior sample.
    pub graph_keys: std::sync::Arc<[u64]>,
    /// Posterior weight per graph sample.
    pub weights: std::sync::Arc<[f64]>,
    /// Identification state per sample, retaining failed atom mass.
    pub identified: std::sync::Arc<[antecedent_prob::GraphIdentFlag]>,
    /// Backend label for the frozen Bayesian configuration.
    pub inference: std::sync::Arc<str>,
}

impl Inspect for CheckedBayesianGraphPosteriorAte {
    type Info = CheckedBayesianGraphPosteriorAteInfo;

    fn info(&self) -> CheckedBayesianGraphPosteriorAteInfo {
        CheckedBayesianGraphPosteriorAteInfo {
            query: self.query().clone(),
            conditional: self.target().is_conditional(),
            estimator: self.procedure().id(),
            validation: self.validation(),
            graph_keys: Arc::clone(&self.identification().graphs.graph_keys),
            weights: Arc::clone(&self.identification().graphs.weights),
            identified: Arc::clone(&self.identification().graphs.identified),
            inference: inference_label(self.inference()),
        }
    }
}
