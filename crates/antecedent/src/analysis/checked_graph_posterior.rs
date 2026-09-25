//! Checked execution contract for graph-posterior average effects.
//!
//! Frequentist DAG atoms keep linear adjustment. ADMG atoms keep general ID
//! and `functional.effect`, including the Bayesian evaluator.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{AverageEffectQuery, PopulationRegistry};
use antecedent_discovery::{GraphPosterior, GraphPosteriorAtomKind};
use antecedent_estimate::OverlapPolicy;
use antecedent_prob::GraphIdentFlag;

use crate::{
    analysis::prepared::CachedGraphPosteriorIdentification, CausalError, EstimatorId,
    EstimatorSpec, RefuteSuite,
};

/// A frozen frequentist DAG graph-posterior composition.
///
/// This retains the original graph atoms and posterior weights together with
/// prepare-time atom identifications. Execution can therefore aggregate the
/// same atom set and preserve unidentified mass without consulting a builder.
#[derive(Clone, Debug)]
pub(crate) struct CheckedGraphPosteriorEffect {
    posterior: Arc<GraphPosterior>,
    query: AverageEffectQuery,
    identification: Arc<CachedGraphPosteriorIdentification>,
    procedure: EstimatorSpec,
    bootstrap_replicates: u32,
    overlap: OverlapPolicy,
    population_registry: Option<PopulationRegistry>,
    latency_mode: Option<super::latency::LatencyMode>,
    validation: RefuteSuite,
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
        query: AverageEffectQuery,
        identification: CachedGraphPosteriorIdentification,
        procedure: EstimatorSpec,
        bootstrap_replicates: u32,
        overlap: OverlapPolicy,
        population_registry: Option<PopulationRegistry>,
        latency_mode: Option<super::latency::LatencyMode>,
        validation: RefuteSuite,
    ) -> Result<Self, CausalError> {
        let licensed = matches!(
            (posterior.atom_kind, procedure.id()),
            (GraphPosteriorAtomKind::Dag, EstimatorId::LinearAdjustmentAte)
                | (GraphPosteriorAtomKind::Admg, EstimatorId::FunctionalEffect)
        );
        if !licensed {
            return Err(CausalError::Unsupported {
                message: "checked graph-posterior effect supports frequentist DAG linear adjustment or ADMG functional.effect",
            });
        }
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
                message:
                    "graph-posterior identification cache does not match frozen atoms and weights",
            });
        }
        if cache_graphs.identified.len() != posterior.n_graphs {
            return Err(CausalError::Unsupported {
                message: "graph-posterior identification flags do not match frozen atoms",
            });
        }
        Ok(Self {
            posterior: Arc::new(posterior),
            query,
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

    pub(crate) fn query(&self) -> &AverageEffectQuery {
        &self.query
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
            query.clone(),
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
        assert!(CheckedGraphPosteriorEffect::prepare(
            graphs.clone(),
            query.clone(),
            cache(&graphs),
            EstimatorSpec::Default(EstimatorId::Aipw),
            0,
            OverlapPolicy::ExplicitOverride,
            None,
            Some(crate::analysis::LatencyMode::Standard),
            RefuteSuite::None,
        )
        .is_err());

        let mut wrong_cache = cache(&graphs);
        wrong_cache.graphs.weights = Arc::from([0.5, 0.5]);
        assert!(CheckedGraphPosteriorEffect::prepare(
            graphs,
            query,
            wrong_cache,
            EstimatorSpec::Default(EstimatorId::LinearAdjustmentAte),
            0,
            OverlapPolicy::ExplicitOverride,
            None,
            Some(crate::analysis::LatencyMode::Standard),
            RefuteSuite::None,
        )
        .is_err());
    }
}
