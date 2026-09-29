//! Checked execution contract for graph-posterior average effects.
//!
//! Frequentist DAG atoms keep linear adjustment for average effects and
//! conditional linear adjustment for conditional effects. ADMG atoms keep
//! general ID and `functional.effect`, including the Bayesian evaluator. Static
//! CPDAG and PAG envelopes keep either linear adjustment or Bayesian
//! g-computation together with the inference settings that selected the
//! procedure.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{AverageEffectQuery, PopulationRegistry, ResponseQuery};
use antecedent_discovery::{GraphPosterior, GraphPosteriorAtomKind};
use antecedent_estimate::OverlapPolicy;
use antecedent_prob::GraphIdentFlag;

use crate::{
    CausalError, EstimatorId, EstimatorSpec, RefuteSuite,
    analysis::prepared::CachedGraphPosteriorIdentification,
};

use super::GraphPosteriorEffectTarget;

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
    physical: crate::planner::PhysicalExecutionPlan,
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
    Cpdag(crate::analysis::prepared::CachedCpdagIdentification),
    Pag(crate::analysis::prepared::CachedPagIdentification),
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

impl CheckedStaticClassEffect {
    pub(crate) fn prepare(
        graph: StaticClassGraph,
        query: AverageEffectQuery,
        identification: StaticClassIdentification,
        physical: crate::planner::PhysicalExecutionPlan,
        procedure: EstimatorSpec,
        inference: crate::InferenceMode,
        bootstrap_replicates: u32,
        overlap: OverlapPolicy,
        validation: RefuteSuite,
    ) -> Result<Self, CausalError> {
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
            || !matches!(
                validation,
                RefuteSuite::None
                    | RefuteSuite::Cheap
                    | RefuteSuite::PlaceboAndRcc
                    | RefuteSuite::Full
            )
            || physical.logical.record.identifier.as_deref()
                != Some(crate::strategy_table::IdentifierId::GeneralizedAdjustment.as_str())
            || physical.logical.record.estimator.as_deref() != Some(expected_estimator.as_str())
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
    physical: crate::planner::PhysicalExecutionPlan,
    inference: crate::InferenceMode,
    validation: RefuteSuite,
}

impl CheckedAdmgGraphPosteriorResponse {
    pub(crate) fn prepare(
        posterior: GraphPosterior,
        query: ResponseQuery,
        identification: CachedGraphPosteriorIdentification,
        physical: crate::planner::PhysicalExecutionPlan,
        inference: crate::InferenceMode,
        validation: RefuteSuite,
    ) -> Result<Self, CausalError> {
        if posterior.atom_kind != GraphPosteriorAtomKind::Admg {
            return Err(CausalError::Unsupported {
                message: "checked graph-posterior response requires ADMG atoms",
            });
        }
        if query.temporal.is_some()
            || query.observation != antecedent_core::ObservationSpec::Complete
            || query.target_population != antecedent_core::TargetPopulation::AllObserved
            || !matches!(query.outcome_functional, antecedent_core::OutcomeFunctional::Mean)
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
        if (!intervention_response && validation != RefuteSuite::None)
            || !matches!(validation, RefuteSuite::None | RefuteSuite::Cheap | RefuteSuite::Full)
        {
            return Err(CausalError::Unsupported {
                message: "checked ADMG graph-posterior response supports no refuters for curves and none/cheap/full for intervention responses",
            });
        }
        if posterior.weights.as_ref() != identification.graphs.weights.as_ref()
            || posterior.graph_keys.as_ref() != identification.graphs.graph_keys.as_ref()
            || posterior.n_graphs != identification.graphs.n_samples
            || identification.graphs.identified.len() != posterior.n_graphs
            || identification.atoms.iter().any(|atom| !posterior.graph_keys.contains(&atom.key))
        {
            return Err(CausalError::Unsupported {
                message: "ADMG response identification cache does not match frozen posterior atoms",
            });
        }
        Ok(Self {
            posterior: Arc::new(posterior),
            query,
            identification: Arc::new(identification),
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
    pub(crate) fn identification(&self) -> &CachedGraphPosteriorIdentification {
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

impl CheckedGraphPosteriorEffect {
    /// Seal a frequentist static DAG effect over a previously identified graph
    /// posterior.
    ///
    /// # Errors
    ///
    /// Rejects unsupported graph classes, estimator choices, malformed query
    /// bindings, and identification caches from another atom/weight ensemble.
    pub(crate) fn prepare(
        posterior: GraphPosterior,
        target: GraphPosteriorEffectTarget,
        identification: CachedGraphPosteriorIdentification,
        procedure: EstimatorSpec,
        bootstrap_replicates: u32,
        overlap: OverlapPolicy,
        population_registry: Option<PopulationRegistry>,
        latency_mode: Option<super::latency::LatencyMode>,
        validation: RefuteSuite,
    ) -> Result<Self, CausalError> {
        let licensed = match posterior.atom_kind {
            GraphPosteriorAtomKind::Dag => procedure.id() == target.frequentist_estimator(),
            GraphPosteriorAtomKind::Admg => {
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
        let cache_graphs = &identification.graphs;
        if posterior.weights.as_ref() != cache_graphs.weights.as_ref()
            || posterior.graph_keys.as_ref() != cache_graphs.graph_keys.as_ref()
            || posterior.n_graphs != cache_graphs.n_samples
            || identification
                .atoms
                .iter()
                .any(|atom| !posterior.graph_keys.iter().any(|key| *key == atom.key))
        {
            return Err(CausalError::Unsupported {
                message: "graph-posterior identification cache does not match frozen atoms and weights",
            });
        }
        if cache_graphs.identified.len() != posterior.n_graphs {
            return Err(CausalError::Unsupported {
                message: "graph-posterior identification flags do not match frozen atoms",
            });
        }
        Ok(Self {
            posterior: Arc::new(posterior),
            target,
            identification: Arc::new(identification),
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

    pub(crate) fn identification(&self) -> &CachedGraphPosteriorIdentification {
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

    /// Read-only per-sample atom identities and posterior weights, including
    /// unidentified samples retained in the frozen envelope.
    pub(crate) fn atom_weights(&self) -> (&[u64], &[f64], &[GraphIdentFlag]) {
        (
            &self.posterior.graph_keys,
            &self.posterior.weights,
            &self.identification.graphs.identified,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::VariableId;
    use antecedent_prob::{InferenceDiagnostics, WeightedGraphSamples};

    fn posterior() -> GraphPosterior {
        GraphPosterior::new(
            2,
            vec![0.25, 0.75],
            vec![0, 0],
            vec![0.0; 4],
            vec![0.0; 4],
            1.6,
            InferenceDiagnostics::analytic("checked_plan_test"),
            0,
        )
        .unwrap()
    }

    fn cache(graphs: &GraphPosterior) -> CachedGraphPosteriorIdentification {
        CachedGraphPosteriorIdentification {
            graphs: WeightedGraphSamples::new(
                graphs.weights.to_vec(),
                vec![GraphIdentFlag::Unidentified; graphs.n_graphs],
                graphs.graph_keys.to_vec(),
            )
            .unwrap(),
            atoms: Arc::from([]),
            class_atoms: Arc::from([]),
        }
    }

    #[test]
    fn plan_freezes_query_weights_unidentified_flags_procedure_and_validation() {
        let graphs = posterior();
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        let plan = CheckedGraphPosteriorEffect::prepare(
            graphs.clone(),
            GraphPosteriorEffectTarget::Average(query.clone()),
            cache(&graphs),
            EstimatorSpec::Default(EstimatorId::LinearAdjustmentAte),
            0,
            OverlapPolicy::ExplicitOverride,
            None,
            Some(crate::analysis::LatencyMode::Standard),
            RefuteSuite::Full,
        )
        .unwrap();
        assert_eq!(plan.posterior().weights.as_ref(), &[0.25, 0.75]);
        assert_eq!(plan.query(), &query);
        assert_eq!(plan.estimator(), EstimatorId::LinearAdjustmentAte);
        assert_eq!(plan.validation(), RefuteSuite::Full);
        assert_eq!(plan.atom_weights().2, &[GraphIdentFlag::Unidentified; 2]);
    }

    #[test]
    fn plan_refuses_procedure_or_weight_mismatch() {
        let graphs = posterior();
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
        assert!(
            CheckedGraphPosteriorEffect::prepare(
                graphs.clone(),
                GraphPosteriorEffectTarget::Average(query.clone()),
                cache(&graphs),
                EstimatorSpec::Default(EstimatorId::Aipw),
                0,
                OverlapPolicy::ExplicitOverride,
                None,
                Some(crate::analysis::LatencyMode::Standard),
                RefuteSuite::None,
            )
            .is_err()
        );

        let mut wrong_cache = cache(&graphs);
        wrong_cache.graphs.weights = Arc::from([0.5, 0.5]);
        assert!(
            CheckedGraphPosteriorEffect::prepare(
                graphs,
                GraphPosteriorEffectTarget::Average(query),
                wrong_cache,
                EstimatorSpec::Default(EstimatorId::LinearAdjustmentAte),
                0,
                OverlapPolicy::ExplicitOverride,
                None,
                Some(crate::analysis::LatencyMode::Standard),
                RefuteSuite::None,
            )
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
            CheckedGraphPosteriorEffect::prepare(
                graphs.clone(),
                conditional.clone(),
                cache(&graphs),
                EstimatorSpec::Default(procedure),
                0,
                OverlapPolicy::ExplicitOverride,
                None,
                Some(crate::analysis::LatencyMode::Standard),
                RefuteSuite::Cheap,
            )
        };
        let plan = prepare(EstimatorId::ConditionalLinearAdjustment).unwrap();
        assert!(plan.target().is_conditional());
        assert_eq!(plan.query(), conditional.inner());
        assert_eq!(plan.estimator(), EstimatorId::ConditionalLinearAdjustment);
        assert!(prepare(EstimatorId::LinearAdjustmentAte).is_err());
    }
}
