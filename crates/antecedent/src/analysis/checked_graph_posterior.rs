//! Checked execution contract for graph-posterior average effects.
//!
//! Frequentist DAG atoms keep linear adjustment for average effects and
//! conditional linear adjustment for conditional effects. ADMG atoms keep
//! general ID and `functional.effect`, including the Bayesian evaluator. Static
//! CPDAG and PAG envelopes keep either linear adjustment or Bayesian
//! g-computation together with the inference settings that selected the
//! procedure.
// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::accepted::GraphClass;
use crate::analysis::prepared::Inspect;
use crate::strategy_table::IdentifierId;
use std::sync::Arc;

use antecedent_core::{AverageEffectQuery, PopulationRegistry, ResponseQuery};
use antecedent_discovery::{GraphPosterior, GraphPosteriorAtomKind};
use antecedent_estimate::OverlapPolicy;
use antecedent_prob::WeightedGraphSamples;

use crate::{
    CausalError, EstimatorId, EstimatorSpec, RefuteSuite,
    analysis::prepared::CachedGraphPosteriorIdentification,
};

use super::GraphPosteriorEffectTarget;
use super::posterior_binding::{CacheKeys, PosteriorProof, validate_posterior_cache};
use super::route_guards::complete_mean_response;

/// Sealed average effect over a static CPDAG or PAG completion envelope.
/// The graph and prepare-time completion cache are retained together so
/// execution cannot silently select a different class member or identifier.
/// Frequentist inference binds `linear.adjustment.ate`; Bayesian inference
/// binds `bayesian.gcomp` together with its prior and backend configuration.
#[derive(Clone, Debug)]
pub(crate) struct CheckedStaticClassEffect {
    graph: StaticClassGraph,
    query: AverageEffectQuery,
    identification: StaticClassIdentification,
    physical: std::sync::Arc<crate::planner::PhysicalExecutionPlan>,
    procedure: EstimatorSpec,
    inference: crate::InferenceMode,
    bootstrap_replicates: u32,
    overlap: OverlapPolicy,
    validation: RefuteSuite,
}

#[derive(Clone, Debug)]
pub(crate) enum StaticClassGraph {
    Cpdag(antecedent_graph::Cpdag),
    Pag(antecedent_graph::Pag),
}

#[derive(Clone, Debug)]
pub(crate) enum StaticClassIdentification {
    Cpdag(Arc<crate::analysis::prepared::CachedCpdagIdentification>),
    Pag(Arc<crate::analysis::prepared::CachedPagIdentification>),
}

impl StaticClassIdentification {
    /// Completion count and identified / unidentified mass of the frozen envelope.
    #[must_use]
    pub(crate) fn envelope_summary(&self) -> (usize, f64, f64) {
        match self {
            Self::Cpdag(cache) => super::route_guards::envelope_summary(&cache.envelope),
            Self::Pag(cache) => super::route_guards::envelope_summary(&cache.envelope),
        }
    }
}

/// The frozen premises of [`CheckedStaticClassEffect::prepare`], carried together.
pub(crate) struct CheckedStaticClassEffectPremises {
    pub(crate) graph: StaticClassGraph,
    pub(crate) query: AverageEffectQuery,
    pub(crate) identification: StaticClassIdentification,
    pub(crate) physical: std::sync::Arc<crate::planner::PhysicalExecutionPlan>,
    pub(crate) procedure: EstimatorSpec,
    pub(crate) inference: crate::InferenceMode,
    pub(crate) bootstrap_replicates: u32,
    pub(crate) overlap: OverlapPolicy,
    pub(crate) validation: RefuteSuite,
}

impl CheckedStaticClassEffect {
    pub(crate) fn prepare(premises: CheckedStaticClassEffectPremises) -> Result<Self, CausalError> {
        let CheckedStaticClassEffectPremises {
            graph,
            query,
            identification,
            physical,
            procedure,
            inference,
            bootstrap_replicates,
            overlap,
            validation,
        } = premises;
        let expected_estimator = Self::expected_estimator(&inference);
        if procedure.id() != expected_estimator {
            return Err(CausalError::Unsupported {
                message: "checked static class effects require linear.adjustment.ate under frequentist inference or bayesian.gcomp under Bayesian inference",
            });
        }
        let matching_class = matches!(
            (&graph, &identification),
            (StaticClassGraph::Cpdag(_), StaticClassIdentification::Cpdag(_))
                | (StaticClassGraph::Pag(_), StaticClassIdentification::Pag(_))
        );
        let identification_binds_query = match &identification {
            StaticClassIdentification::Cpdag(cache) => {
                cache.identification.query
                    == antecedent_core::CausalQuery::AverageEffect(query.clone())
            }
            StaticClassIdentification::Pag(cache) => {
                cache.identification.query
                    == antecedent_core::CausalQuery::AverageEffect(query.clone())
            }
        };
        if !matching_class
            || !identification_binds_query
            || query.treatment == query.outcome
            || !validation.is_point_or_placebo()
            || !physical.binding().names(
                crate::strategy_table::IdentifierId::GeneralizedAdjustment,
                expected_estimator,
            )
        {
            return Err(CausalError::Unsupported {
                message: "checked static class effect has incompatible graph, query, procedure, or validation binding",
            });
        }
        Ok(Self {
            graph,
            query,
            identification,
            physical,
            procedure,
            inference,
            bootstrap_replicates,
            overlap,
            validation,
        })
    }

    /// The single estimator each inference mode may bind on a class envelope.
    pub(crate) const fn expected_estimator(inference: &crate::InferenceMode) -> EstimatorId {
        match inference {
            crate::InferenceMode::Frequentist => EstimatorId::LinearAdjustmentAte,
            crate::InferenceMode::Bayesian(_) => EstimatorId::BayesianGcomp,
        }
    }

    pub(crate) fn graph(&self) -> &StaticClassGraph {
        &self.graph
    }
    pub(crate) fn query(&self) -> &AverageEffectQuery {
        &self.query
    }
    pub(crate) fn identification(&self) -> &StaticClassIdentification {
        &self.identification
    }
    pub(crate) fn physical(&self) -> &crate::planner::PhysicalExecutionPlan {
        &self.physical
    }
    pub(crate) fn procedure(&self) -> &EstimatorSpec {
        &self.procedure
    }
    pub(crate) const fn inference(&self) -> &crate::InferenceMode {
        &self.inference
    }
    pub(crate) const fn bootstrap_replicates(&self) -> u32 {
        self.bootstrap_replicates
    }
    pub(crate) const fn overlap(&self) -> OverlapPolicy {
        self.overlap
    }
    pub(crate) const fn validation(&self) -> RefuteSuite {
        self.validation
    }
}

/// A frozen frequentist DAG graph-posterior composition.
///
/// This retains the original graph atoms and posterior weights together with
/// prepare-time atom identifications. Execution can therefore aggregate the
/// same atom set and preserve unidentified mass without consulting a builder.
/// The target records whether the click estimates the average or a
/// conditional effect over those atoms.
#[derive(Clone, Debug)]
pub(crate) struct CheckedGraphPosteriorEffect {
    posterior: Arc<GraphPosterior>,
    target: GraphPosteriorEffectTarget,
    identification: Arc<CachedGraphPosteriorIdentification>,
    procedure: EstimatorSpec,
    bootstrap_replicates: u32,
    overlap: OverlapPolicy,
    population_registry: Option<PopulationRegistry>,
    latency_mode: Option<super::latency::LatencyMode>,
    validation: RefuteSuite,
}

/// Frozen ADMG posterior response execution, including the atom-wise general-ID cache.
#[derive(Clone, Debug)]
pub(crate) struct CheckedAdmgGraphPosteriorResponse {
    posterior: Arc<GraphPosterior>,
    query: ResponseQuery,
    identification: Arc<CachedGraphPosteriorIdentification>,
    physical: std::sync::Arc<crate::planner::PhysicalExecutionPlan>,
    inference: crate::InferenceMode,
    validation: RefuteSuite,
}

impl CheckedAdmgGraphPosteriorResponse {
    pub(crate) fn prepare(
        posterior: GraphPosterior,
        query: ResponseQuery,
        identification: Arc<CachedGraphPosteriorIdentification>,
        physical: std::sync::Arc<crate::planner::PhysicalExecutionPlan>,
        inference: crate::InferenceMode,
        validation: RefuteSuite,
    ) -> Result<Self, CausalError> {
        if !matches!(
            posterior.atom_kind,
            GraphPosteriorAtomKind::Admg | GraphPosteriorAtomKind::TemporalAdmg
        ) {
            return Err(CausalError::Unsupported {
                message: "checked graph-posterior response requires ADMG atoms",
            });
        }
        if query.temporal.is_some()
            || !complete_mean_response(&query)
            || !matches!(
                query.functional,
                antecedent_core::ResponseFunctional::MeanCurve { .. }
                    | antecedent_core::ResponseFunctional::InterventionResponse { .. }
            )
            || query.functional.primary_pair().is_none()
        {
            return Err(CausalError::Unsupported {
                message: "checked ADMG graph-posterior response requires a complete-observation mean response with a treatment/outcome pair",
            });
        }
        if !matches!(
            inference,
            crate::InferenceMode::Frequentist | crate::InferenceMode::Bayesian(_)
        ) {
            return Err(CausalError::Unsupported {
                message: "checked ADMG graph-posterior response requires frequentist or Bayesian inference",
            });
        }
        let intervention_response = matches!(
            query.functional,
            antecedent_core::ResponseFunctional::InterventionResponse { .. }
        );
        if (!intervention_response && validation != RefuteSuite::None) || !validation.is_point() {
            return Err(CausalError::Unsupported {
                message: "checked ADMG graph-posterior response supports no refuters for curves and none/cheap/full for intervention responses",
            });
        }
        validate_posterior_cache(
            &posterior,
            &*identification,
            CacheKeys::PosteriorKeys,
            "ADMG response identification cache does not match frozen posterior atoms",
        )?;
        Ok(Self {
            posterior: Arc::new(posterior),
            query,
            identification,
            physical,
            inference,
            validation,
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
    pub(crate) fn physical(&self) -> &crate::planner::PhysicalExecutionPlan {
        &self.physical
    }
    pub(crate) fn inference(&self) -> &crate::InferenceMode {
        &self.inference
    }
    pub(crate) const fn validation(&self) -> RefuteSuite {
        self.validation
    }
}

/// The frozen premises of [`CheckedGraphPosteriorEffect::prepare`], carried together.
pub(crate) struct CheckedGraphPosteriorEffectPremises {
    pub(crate) posterior: GraphPosterior,
    pub(crate) target: GraphPosteriorEffectTarget,
    pub(crate) identification: Arc<CachedGraphPosteriorIdentification>,
    pub(crate) procedure: EstimatorSpec,
    pub(crate) bootstrap_replicates: u32,
    pub(crate) overlap: OverlapPolicy,
    pub(crate) population_registry: Option<PopulationRegistry>,
    pub(crate) latency_mode: Option<super::latency::LatencyMode>,
    pub(crate) validation: RefuteSuite,
}

impl CheckedGraphPosteriorEffect {
    /// Seal a frequentist static DAG effect over a previously identified graph
    /// posterior.
    ///
    /// # Errors
    ///
    /// Rejects unsupported graph classes, estimator choices, malformed query
    /// bindings, and identification caches from another atom/weight ensemble.
    pub(crate) fn prepare(
        premises: CheckedGraphPosteriorEffectPremises,
    ) -> Result<Self, CausalError> {
        let CheckedGraphPosteriorEffectPremises {
            posterior,
            target,
            identification,
            procedure,
            bootstrap_replicates,
            overlap,
            population_registry,
            latency_mode,
            validation,
        } = premises;
        let licensed = match posterior.atom_kind {
            GraphPosteriorAtomKind::Dag => procedure.id() == target.frequentist_estimator(),
            GraphPosteriorAtomKind::Admg | GraphPosteriorAtomKind::TemporalAdmg => {
                !target.is_conditional() && procedure.id() == EstimatorId::FunctionalEffect
            }
            _ => false,
        };
        if !licensed {
            return Err(CausalError::Unsupported {
                message: "checked graph-posterior effect supports frequentist DAG (conditional) linear adjustment or ADMG average functional.effect",
            });
        }
        let query = target.inner();
        if query.treatment == query.outcome
            || usize::try_from(query.treatment.raw()).map_or(true, |v| v >= posterior.n_vars)
            || usize::try_from(query.outcome.raw()).map_or(true, |v| v >= posterior.n_vars)
        {
            return Err(CausalError::Unsupported {
                message: "graph-posterior effect query variables must be distinct and in range",
            });
        }
        validate_posterior_cache(
            &posterior,
            &*identification,
            CacheKeys::PosteriorKeys,
            "graph-posterior identification cache does not match frozen atoms, weights, and flags",
        )?;
        Ok(Self {
            posterior: Arc::new(posterior),
            target,
            identification,
            procedure,
            bootstrap_replicates,
            overlap,
            population_registry,
            latency_mode,
            validation,
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

    pub(crate) fn identification(&self) -> &Arc<CachedGraphPosteriorIdentification> {
        &self.identification
    }

    pub(crate) fn estimator(&self) -> EstimatorId {
        self.procedure.id()
    }

    pub(crate) fn procedure(&self) -> &EstimatorSpec {
        &self.procedure
    }

    pub(crate) const fn bootstrap_replicates(&self) -> u32 {
        self.bootstrap_replicates
    }

    pub(crate) const fn overlap(&self) -> OverlapPolicy {
        self.overlap
    }

    pub(crate) fn population_registry(&self) -> Option<&PopulationRegistry> {
        self.population_registry.as_ref()
    }

    pub(crate) const fn latency_mode(&self) -> Option<super::latency::LatencyMode> {
        self.latency_mode
    }

    pub(crate) const fn validation(&self) -> RefuteSuite {
        self.validation
    }
}

impl PosteriorProof for CheckedGraphPosteriorEffect {
    fn graphs(&self) -> &WeightedGraphSamples {
        &self.identification.graphs
    }

    fn atom_keys(&self) -> Vec<u64> {
        self.identification.atom_keys()
    }
}

/// Read-only checked target and procedure for a static CPDAG/PAG average effect.
#[derive(Clone, Debug)]
pub struct CheckedStaticClassEffectInfo {
    /// Exact average-effect target sealed at preparation.
    pub query: antecedent_core::AverageEffectQuery,
    /// Source graph class retained with the completion envelope.
    pub graph_class: crate::accepted::GraphClass,
    /// Selected identification procedure.
    pub identifier: crate::strategy_table::IdentifierId,
    /// Selected estimator identity.
    pub estimator: crate::strategy_table::EstimatorId,
    /// Inference settings (including any Bayesian prior and backend) retained by preparation.
    pub inference: crate::inference::InferenceMode,
    /// Prepared refutation suite.
    pub validation: crate::analysis::builder::RefuteSuite,
    /// Number of class completions in the retained envelope.
    pub completion_count: usize,
    /// Identified completion mass in the retained envelope.
    pub identified_mass: f64,
    /// Unresolved completion mass in the retained envelope.
    pub unresolved_mass: f64,
}

impl Inspect for CheckedStaticClassEffect {
    type Info = CheckedStaticClassEffectInfo;

    fn info(&self) -> CheckedStaticClassEffectInfo {
        let graph_class = match self.identification() {
            StaticClassIdentification::Cpdag(_) => GraphClass::Cpdag,
            StaticClassIdentification::Pag(_) => GraphClass::Pag,
        };
        let (completion_count, identified_mass, unresolved_mass) =
            self.identification().envelope_summary();
        CheckedStaticClassEffectInfo {
            query: self.query().clone(),
            graph_class,
            identifier: IdentifierId::GeneralizedAdjustment,
            estimator: self.procedure().id(),
            inference: self.inference().clone(),
            validation: self.validation(),
            completion_count,
            identified_mass,
            unresolved_mass,
        }
    }
}

/// Read-only graph atoms, frozen weights and procedure retained for a
/// frequentist DAG graph-posterior effect.
#[derive(Clone, Debug, PartialEq)]
pub struct CheckedGraphPosteriorEffectInfo {
    /// Average-effect query every posterior atom was identified for.
    pub query: antecedent_core::AverageEffectQuery,
    /// Whether the sealed click estimates a conditional effect over the atoms.
    pub conditional: bool,
    /// Static estimator used for every identified atom.
    pub estimator: crate::strategy_table::EstimatorId,
    /// Validation suite applied to the atom-wise estimates.
    pub validation: crate::analysis::builder::RefuteSuite,
    /// Opaque graph identity per posterior sample.
    pub graph_keys: std::sync::Arc<[u64]>,
    /// Frozen posterior weight per graph sample.
    pub weights: std::sync::Arc<[f64]>,
    /// Identification status per graph sample, preserving unknown graph mass.
    pub identified: std::sync::Arc<[antecedent_prob::GraphIdentFlag]>,
    /// Procedure-specific bootstrap count retained by preparation.
    pub bootstrap_replicates: u32,
}

impl Inspect for CheckedGraphPosteriorEffect {
    type Info = CheckedGraphPosteriorEffectInfo;

    fn info(&self) -> CheckedGraphPosteriorEffectInfo {
        let (graph_keys, weights, identified) = self.sample_mass();
        CheckedGraphPosteriorEffectInfo {
            query: self.query().clone(),
            conditional: self.target().is_conditional(),
            estimator: self.estimator(),
            validation: self.validation(),
            graph_keys: Arc::from(graph_keys),
            weights: Arc::from(weights),
            identified: Arc::from(identified),
            bootstrap_replicates: self.bootstrap_replicates(),
        }
    }
}

/// Frozen ADMG response posterior, including atom weights and identification flags.
#[derive(Clone, Debug, PartialEq)]
pub struct CheckedAdmgGraphPosteriorResponseInfo {
    /// Response target retained at preparation.
    pub query: antecedent_core::ResponseQuery,
    /// Frozen graph identity per posterior atom.
    pub graph_keys: std::sync::Arc<[u64]>,
    /// Frozen posterior weight per atom.
    pub weights: std::sync::Arc<[f64]>,
    /// Identified and unidentified atom status, preserving unresolved mass.
    pub identified: std::sync::Arc<[antecedent_prob::GraphIdentFlag]>,
    /// Validation suite fixed at preparation.
    pub validation: crate::analysis::builder::RefuteSuite,
}

impl Inspect for CheckedAdmgGraphPosteriorResponse {
    type Info = CheckedAdmgGraphPosteriorResponseInfo;

    fn info(&self) -> CheckedAdmgGraphPosteriorResponseInfo {
        CheckedAdmgGraphPosteriorResponseInfo {
            query: self.query().clone(),
            graph_keys: Arc::clone(&self.identification().graphs.graph_keys),
            weights: Arc::clone(&self.identification().graphs.weights),
            identified: Arc::clone(&self.identification().graphs.identified),
            validation: self.validation(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::VariableId;
    use antecedent_discovery::SamplingSummary;
    use antecedent_prob::{GraphIdentFlag, InferenceDiagnostics};

    fn posterior() -> GraphPosterior {
        GraphPosterior::new(
            2,
            vec![0.25, 0.75],
            vec![0, 0],
            vec![0.0; 4],
            vec![0.0; 4],
            SamplingSummary {
                ess: 1.6,
                diagnostics: InferenceDiagnostics::analytic("checked_plan_test"),
                rejected_invalid: 0,
            },
        )
        .unwrap()
    }

    fn cache(graphs: &GraphPosterior) -> Arc<CachedGraphPosteriorIdentification> {
        Arc::new(CachedGraphPosteriorIdentification {
            graphs: WeightedGraphSamples::new(
                graphs.weights.to_vec(),
                vec![GraphIdentFlag::Unidentified; graphs.n_graphs],
                graphs.graph_keys.to_vec(),
            )
            .unwrap(),
            atoms: Arc::from([]),
            class_atoms: Arc::from([]),
        })
    }

    #[test]
    fn plan_freezes_query_weights_unidentified_flags_procedure_and_validation() {
        let graphs = posterior();
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let plan = CheckedGraphPosteriorEffect::prepare(CheckedGraphPosteriorEffectPremises {
            posterior: graphs.clone(),
            target: GraphPosteriorEffectTarget::Average(query.clone()),
            identification: cache(&graphs),
            procedure: EstimatorSpec::Default(EstimatorId::LinearAdjustmentAte),
            bootstrap_replicates: 0,
            overlap: OverlapPolicy::ExplicitOverride,
            population_registry: None,
            latency_mode: Some(crate::analysis::LatencyMode::Standard),
            validation: RefuteSuite::Full,
        })
        .unwrap();
        assert_eq!(plan.posterior().weights.as_ref(), &[0.25, 0.75]);
        assert_eq!(plan.query(), &query);
        assert_eq!(plan.estimator(), EstimatorId::LinearAdjustmentAte);
        assert_eq!(plan.validation(), RefuteSuite::Full);
        assert_eq!(plan.sample_mass().2, &[GraphIdentFlag::Unidentified; 2]);
    }

    #[test]
    fn plan_refuses_procedure_or_weight_mismatch() {
        let graphs = posterior();
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        assert!(
            CheckedGraphPosteriorEffect::prepare(CheckedGraphPosteriorEffectPremises {
                posterior: graphs.clone(),
                target: GraphPosteriorEffectTarget::Average(query.clone()),
                identification: cache(&graphs),
                procedure: EstimatorSpec::Default(EstimatorId::Aipw),
                bootstrap_replicates: 0,
                overlap: OverlapPolicy::ExplicitOverride,
                population_registry: None,
                latency_mode: Some(crate::analysis::LatencyMode::Standard),
                validation: RefuteSuite::None
            })
            .is_err()
        );

        let mut wrong_cache = cache(&graphs);
        Arc::make_mut(&mut wrong_cache).graphs.weights = Arc::from([0.5, 0.5]);
        assert!(
            CheckedGraphPosteriorEffect::prepare(CheckedGraphPosteriorEffectPremises {
                posterior: graphs,
                target: GraphPosteriorEffectTarget::Average(query),
                identification: wrong_cache,
                procedure: EstimatorSpec::Default(EstimatorId::LinearAdjustmentAte),
                bootstrap_replicates: 0,
                overlap: OverlapPolicy::ExplicitOverride,
                population_registry: None,
                latency_mode: Some(crate::analysis::LatencyMode::Standard),
                validation: RefuteSuite::None
            })
            .is_err()
        );
    }

    #[test]
    fn conditional_target_requires_conditional_linear_adjustment() {
        let graphs = posterior();
        let conditional = GraphPosteriorEffectTarget::Conditional(
            antecedent_core::ConditionalEffectQuery::try_new(
                AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
                    .with_effect_modifiers([VariableId::from_raw(2)]),
            )
            .unwrap(),
        );
        let prepare = |procedure: EstimatorId| {
            CheckedGraphPosteriorEffect::prepare(CheckedGraphPosteriorEffectPremises {
                posterior: graphs.clone(),
                target: conditional.clone(),
                identification: cache(&graphs),
                procedure: EstimatorSpec::Default(procedure),
                bootstrap_replicates: 0,
                overlap: OverlapPolicy::ExplicitOverride,
                population_registry: None,
                latency_mode: Some(crate::analysis::LatencyMode::Standard),
                validation: RefuteSuite::Cheap,
            })
        };
        let plan = prepare(EstimatorId::ConditionalLinearAdjustment).unwrap();
        assert!(plan.target().is_conditional());
        assert_eq!(plan.query(), conditional.inner());
        assert_eq!(plan.estimator(), EstimatorId::ConditionalLinearAdjustment);
        assert!(prepare(EstimatorId::LinearAdjustmentAte).is_err());
    }
}
