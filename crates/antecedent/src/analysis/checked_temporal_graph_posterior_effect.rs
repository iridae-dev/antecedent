//! Checked pulse and sustained effect operation over a temporal graph posterior.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{CausalQuery, TemporalEffectQuery};
use antecedent_data::DiscoveryEstimationSplit;
use antecedent_discovery::{GraphPosterior, GraphPosteriorAtomKind};
use antecedent_prob::GraphIdentFlag;

use super::builder::RefuteSuite;
use super::prepared::{
    CachedDbnPosteriorIdentification, CachedTemporalClassPosteriorIdentification,
};
use crate::error::CausalError;
use crate::planner::PhysicalExecutionPlan;
use crate::strategy_table::{EstimatorId, IdentifierId};
use crate::{EstimatorSpec, InferenceMode};

/// Per-atom proof retained for a temporal graph posterior. DBN atoms carry one
/// unfolded backdoor identification and indexer each; TemporalCpdag/TemporalPag
/// atoms carry one completion envelope each. Unidentified sample mass stays in
/// the weighted samples of either proof.
#[derive(Clone, Debug)]
pub(crate) enum CheckedTemporalGraphPosteriorProof {
    Dbn(Arc<CachedDbnPosteriorIdentification>),
    Class(Arc<CachedTemporalClassPosteriorIdentification>),
}

/// A fixed pulse or sustained effect procedure over posterior-weighted
/// TemporalDag, TemporalCpdag, or TemporalPag atoms under frequentist or
/// Bayesian inference. The frozen posterior samples, the per-atom proofs, the
/// inference settings, and the physical plan are sealed together.
#[derive(Clone, Debug)]
pub(crate) struct CheckedTemporalGraphPosteriorEffect {
    posterior: Arc<GraphPosterior>,
    query: TemporalEffectQuery,
    proof: CheckedTemporalGraphPosteriorProof,
    inference: InferenceMode,
    identifier: IdentifierId,
    estimator: EstimatorId,
    bootstrap_replicates: u32,
    split: Option<DiscoveryEstimationSplit>,
    max_completions: Option<usize>,
    validation: RefuteSuite,
    physical: PhysicalExecutionPlan,
}

impl CheckedTemporalGraphPosteriorEffect {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare(
        posterior: GraphPosterior,
        query: TemporalEffectQuery,
        proof: CheckedTemporalGraphPosteriorProof,
        inference: InferenceMode,
        estimator_spec: Option<&EstimatorSpec>,
        bootstrap_replicates: u32,
        split: Option<DiscoveryEstimationSplit>,
        max_completions: Option<usize>,
        validation: RefuteSuite,
        physical: PhysicalExecutionPlan,
    ) -> Result<Self, CausalError> {
        query.validate().map_err(|error| CausalError::Compile { message: error.to_string() })?;
        if !matches!(
            query.policy,
            antecedent_core::TemporalPolicy::Pulse { .. }
                | antecedent_core::TemporalPolicy::Sustained { .. }
        ) {
            return Err(CausalError::Unsupported {
                message: "checked temporal graph-posterior effect supports pulse and sustained schedules",
            });
        }
        if !matches!(validation, RefuteSuite::None | RefuteSuite::Cheap | RefuteSuite::Full) {
            return Err(CausalError::Unsupported {
                message: "checked temporal graph-posterior effect supports none, cheap, or full validation",
            });
        }
        match (&proof, posterior.atom_kind) {
            (CheckedTemporalGraphPosteriorProof::Dbn(_), GraphPosteriorAtomKind::Dag)
            | (
                CheckedTemporalGraphPosteriorProof::Class(_),
                GraphPosteriorAtomKind::Cpdag | GraphPosteriorAtomKind::Pag,
            ) => {}
            _ => {
                return Err(CausalError::Unsupported {
                    message: "checked temporal graph-posterior proof does not match the posterior atom kind",
                });
            }
        }
        if posterior.lag_masks.is_none() || posterior.max_lag.is_none() {
            return Err(CausalError::Unsupported {
                message: "checked temporal graph-posterior effect requires per-atom lag masks",
            });
        }
        let identifier = IdentifierId::TemporalBackdoorUnfolded;
        let estimator = expected_estimator(&query, &inference);
        if estimator_spec.is_some_and(|spec| spec.id() != estimator) {
            return Err(CausalError::Unsupported {
                message: "checked temporal graph-posterior effect procedure does not match the sealed schedule",
            });
        }
        let target = CausalQuery::TemporalEffect(query.clone());
        if physical.logical.query != target
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
            return Err(CausalError::Compile {
                message:
                    "temporal graph-posterior physical plan disagrees with its retained procedure"
                        .into(),
            });
        }
        validate_proof_binding(&posterior, &target, &proof)?;
        Ok(Self {
            posterior: Arc::new(posterior),
            query,
            proof,
            inference,
            identifier,
            estimator,
            bootstrap_replicates,
            split,
            max_completions,
            validation,
            physical,
        })
    }

    #[must_use]
    pub(crate) fn posterior(&self) -> &GraphPosterior {
        &self.posterior
    }

    #[must_use]
    pub(crate) fn query(&self) -> &TemporalEffectQuery {
        &self.query
    }

    #[must_use]
    pub(crate) fn proof(&self) -> &CheckedTemporalGraphPosteriorProof {
        &self.proof
    }

    #[must_use]
    pub(crate) fn inference(&self) -> &InferenceMode {
        &self.inference
    }

    #[must_use]
    pub(crate) const fn identifier(&self) -> IdentifierId {
        self.identifier
    }

    #[must_use]
    pub(crate) const fn estimator(&self) -> EstimatorId {
        self.estimator
    }

    #[must_use]
    pub(crate) const fn bootstrap_replicates(&self) -> u32 {
        self.bootstrap_replicates
    }

    #[must_use]
    pub(crate) fn split(&self) -> Option<&DiscoveryEstimationSplit> {
        self.split.as_ref()
    }

    #[must_use]
    pub(crate) const fn max_completions(&self) -> Option<usize> {
        self.max_completions
    }

    #[must_use]
    pub(crate) const fn validation(&self) -> RefuteSuite {
        self.validation
    }

    #[must_use]
    pub(crate) fn physical(&self) -> &PhysicalExecutionPlan {
        &self.physical
    }

    #[must_use]
    pub(crate) fn matches_query(&self, query: &CausalQuery) -> bool {
        matches!(query, CausalQuery::TemporalEffect(candidate) if candidate == &self.query)
    }

    /// Whether a study's posterior is the frozen sample set sealed here.
    #[must_use]
    pub(crate) fn matches_posterior(&self, posterior: &GraphPosterior) -> bool {
        let sealed = &*self.posterior;
        sealed.atom_kind == posterior.atom_kind
            && sealed.n_vars == posterior.n_vars
            && sealed.n_graphs == posterior.n_graphs
            && sealed.max_lag == posterior.max_lag
            && sealed
                .weights
                .iter()
                .zip(posterior.weights.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits())
            && sealed.adjacency == posterior.adjacency
            && sealed.graph_keys == posterior.graph_keys
            && sealed.lag_masks == posterior.lag_masks
    }

    /// Posterior sample keys, weights, and identification flags retain the
    /// sample-level mass, including unidentified atoms.
    #[must_use]
    pub(crate) fn sample_mass(&self) -> (&[u64], &[f64], &[GraphIdentFlag]) {
        let graphs = match &self.proof {
            CheckedTemporalGraphPosteriorProof::Dbn(cache) => &cache.graphs,
            CheckedTemporalGraphPosteriorProof::Class(cache) => &cache.graphs,
        };
        (&graphs.graph_keys, &graphs.weights, &graphs.identified)
    }

    /// Number of identified atoms carrying a retained proof.
    #[must_use]
    pub(crate) fn identified_atom_count(&self) -> usize {
        match &self.proof {
            CheckedTemporalGraphPosteriorProof::Dbn(cache) => cache.atoms.len(),
            CheckedTemporalGraphPosteriorProof::Class(cache) => cache.class_atoms.len(),
        }
    }
}

fn expected_estimator(query: &TemporalEffectQuery, inference: &InferenceMode) -> EstimatorId {
    if query.is_multi_step_sustained() {
        EstimatorId::TemporalSequentialGcomp
    } else if matches!(inference, InferenceMode::Bayesian(_)) {
        EstimatorId::BayesianTemporalGcomp
    } else {
        EstimatorId::TemporalLinearAdjustment
    }
}

fn validate_proof_binding(
    posterior: &GraphPosterior,
    query: &CausalQuery,
    proof: &CheckedTemporalGraphPosteriorProof,
) -> Result<(), CausalError> {
    let (graphs, atom_keys): (_, Vec<u64>) = match proof {
        CheckedTemporalGraphPosteriorProof::Dbn(cache) => {
            // DBN atoms retain the identifier's average-effect query over each
            // atom's own unfolded graph; bind it through that atom's indexer.
            let CausalQuery::TemporalEffect(target) = query else {
                return Err(CausalError::Compile {
                    message: "temporal graph-posterior proof requires a temporal effect query"
                        .into(),
                });
            };
            for atom in cache.atoms.iter() {
                super::checked_temporal_effect::temporal_proof_binds_query(
                    &atom.identification,
                    target,
                    &atom.indexer,
                )?;
            }
            (&cache.graphs, cache.atoms.iter().map(|atom| atom.key).collect())
        }
        CheckedTemporalGraphPosteriorProof::Class(cache) => {
            if cache.class_atoms.iter().any(|atom| &atom.identification.query != query) {
                return Err(CausalError::Unsupported {
                    message: "temporal graph-posterior proof identifies a different temporal query",
                });
            }
            (&cache.graphs, cache.class_atoms.iter().map(|atom| atom.key).collect())
        }
    };
    if graphs.n_samples != posterior.n_graphs
        || graphs.weights.len() != posterior.n_graphs
        || graphs.identified.len() != posterior.n_graphs
        || graphs
            .weights
            .iter()
            .zip(posterior.weights.iter())
            .any(|(a, b)| a.to_bits() != b.to_bits())
    {
        return Err(CausalError::Unsupported {
            message: "temporal graph-posterior proof does not match the frozen posterior samples",
        });
    }
    let identified_keys: std::collections::HashSet<u64> = graphs
        .graph_keys
        .iter()
        .zip(graphs.identified.iter())
        .filter(|(_, flag)| matches!(flag, GraphIdentFlag::Identified))
        .map(|(key, _)| *key)
        .collect();
    let mut seen = std::collections::HashSet::with_capacity(atom_keys.len());
    for key in &atom_keys {
        if !seen.insert(*key) || !identified_keys.contains(key) {
            return Err(CausalError::Unsupported {
                message: "temporal graph-posterior proof has an atom without identified sample mass",
            });
        }
    }
    if identified_keys.iter().any(|key| !seen.contains(key)) {
        return Err(CausalError::Unsupported {
            message: "identified temporal posterior sample has no retained atom proof",
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::{TemporalPolicy, VariableId};

    #[test]
    fn expected_estimator_follows_schedule_and_inference() {
        let treatment = VariableId::from_raw(0);
        let outcome = VariableId::from_raw(1);
        let pulse = TemporalEffectQuery::pulse(treatment, outcome, 1.0)
            .with_policy(TemporalPolicy::pulse(-1));
        let multi = TemporalEffectQuery::sustained(treatment, outcome, -2, 1.0)
            .with_policy(TemporalPolicy::sustained(-2, -1));
        let bayesian = InferenceMode::Bayesian(crate::BayesianConfig::conjugate());
        assert_eq!(
            expected_estimator(&pulse, &InferenceMode::Frequentist),
            EstimatorId::TemporalLinearAdjustment
        );
        assert_eq!(expected_estimator(&pulse, &bayesian), EstimatorId::BayesianTemporalGcomp);
        assert_eq!(
            expected_estimator(&multi, &InferenceMode::Frequentist),
            EstimatorId::TemporalSequentialGcomp
        );
        assert_eq!(expected_estimator(&multi, &bayesian), EstimatorId::TemporalSequentialGcomp);
    }
}
