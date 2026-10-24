//! Checked pulse and sustained effect operation over a temporal graph posterior.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::analysis::prepared::Inspect;
use crate::analysis::route_guards::inference_label;
use std::sync::Arc;

use antecedent_core::{CausalQuery, TemporalEffectQuery};
use antecedent_data::DiscoveryEstimationSplit;
use antecedent_discovery::{GraphPosterior, GraphPosteriorAtomKind};
use antecedent_prob::WeightedGraphSamples;

use super::builder::{DataInput, RefuteSuite};
use super::execute::Study;
use super::posterior_binding::{
    CacheKeys, PosteriorProof, identified_key_set, validate_posterior_cache,
};
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
/// `TemporalDag`, `TemporalCpdag`, or `TemporalPag` atoms under frequentist or
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
    physical: std::sync::Arc<PhysicalExecutionPlan>,
}

/// The frozen premises of [`CheckedTemporalGraphPosteriorEffect::prepare`], carried together.
pub(crate) struct CheckedTemporalGraphPosteriorEffectPremises<'a> {
    pub(crate) posterior: GraphPosterior,
    pub(crate) query: TemporalEffectQuery,
    pub(crate) proof: CheckedTemporalGraphPosteriorProof,
    pub(crate) inference: InferenceMode,
    pub(crate) estimator_spec: Option<&'a EstimatorSpec>,
    pub(crate) bootstrap_replicates: u32,
    pub(crate) split: Option<DiscoveryEstimationSplit>,
    pub(crate) max_completions: Option<usize>,
    pub(crate) validation: RefuteSuite,
    pub(crate) physical: std::sync::Arc<PhysicalExecutionPlan>,
}

impl CheckedTemporalGraphPosteriorEffect {
    /// Whether `study` is a coordinate this operation seals: a series pulse or
    /// sustained effect over `TemporalDag`, `TemporalCpdag`, or `TemporalPag`
    /// posterior atoms with the default point-estimate settings. The one-shot
    /// facade and the prepared sealer share this predicate.
    #[must_use]
    pub(crate) fn admits(study: &Study) -> bool {
        let (Some(posterior), CausalQuery::TemporalEffect(query)) =
            (study.graph_posterior.as_ref(), &study.query)
        else {
            return false;
        };
        matches!(study.data, DataInput::Temporal(_) | DataInput::Event(_))
            && matches!(
                posterior.atom_kind,
                GraphPosteriorAtomKind::Dag
                    | GraphPosteriorAtomKind::Cpdag
                    | GraphPosteriorAtomKind::Pag
            )
            && study.tiered.is_none()
            && study.custom_validators.is_empty()
            && study.point_validation()
            && query.policy.is_pulse_or_sustained()
            // A conditional effect over TemporalDag atoms is answered atom by atom with the
            // interval of the whole procedure, not by the pulse design.
            && !(posterior.atom_kind == GraphPosteriorAtomKind::Dag
                && !query.effect_modifiers.is_empty())
            && study
                .estimator
                .is_none_or(|id| id == EstimatorId::temporal_effect_for(query, &study.inference))
    }

    pub(crate) fn prepare(
        premises: CheckedTemporalGraphPosteriorEffectPremises<'_>,
    ) -> Result<Self, CausalError> {
        let CheckedTemporalGraphPosteriorEffectPremises {
            posterior,
            query,
            proof,
            inference,
            estimator_spec,
            bootstrap_replicates,
            split,
            max_completions,
            validation,
            physical,
        } = premises;
        query.validate()?;
        if !query.policy.is_pulse_or_sustained() {
            return Err(CausalError::Unsupported {
                message: "checked temporal graph-posterior effect supports pulse and sustained schedules",
            });
        }
        if !validation.is_point() {
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
        let estimator = EstimatorId::temporal_effect_for(&query, &inference);
        if estimator_spec.is_some_and(|spec| spec.id() != estimator) {
            return Err(CausalError::Unsupported {
                message: "checked temporal graph-posterior effect procedure does not match the sealed schedule",
            });
        }
        let target = CausalQuery::TemporalEffect(query.clone());
        if physical.logical.query != target || !physical.binding().names(identifier, estimator) {
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
            && sealed.atoms == posterior.atoms
            && sealed.graph_keys == posterior.graph_keys
            && sealed.lag_masks == posterior.lag_masks
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

impl PosteriorProof for CheckedTemporalGraphPosteriorEffect {
    fn graphs(&self) -> &WeightedGraphSamples {
        self.proof.graphs()
    }

    fn atom_keys(&self) -> Vec<u64> {
        self.proof.atom_keys()
    }
}

fn validate_proof_binding(
    posterior: &GraphPosterior,
    query: &CausalQuery,
    proof: &CheckedTemporalGraphPosteriorProof,
) -> Result<(), CausalError> {
    match proof {
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
        }
        CheckedTemporalGraphPosteriorProof::Class(cache) => {
            if cache.class_atoms.iter().any(|atom| &atom.identification.query != query) {
                return Err(CausalError::Unsupported {
                    message: "temporal graph-posterior proof identifies a different temporal query",
                });
            }
        }
    }
    validate_posterior_cache(
        posterior,
        proof,
        CacheKeys::CacheLocal,
        "temporal graph-posterior proof does not match the frozen posterior samples",
    )?;
    let identified_keys = identified_key_set(proof.graphs());
    let atom_keys = proof.atom_keys();
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

/// Read-only frozen samples, per-atom proof counts, and procedure retained for
/// a pulse or sustained effect over a temporal graph posterior.
#[derive(Clone, Debug, PartialEq)]
pub struct CheckedTemporalGraphPosteriorEffectInfo {
    /// Exact policy, intervention levels, horizon, and target population.
    pub query: antecedent_core::TemporalEffectQuery,
    /// Atom kind of the frozen posterior (`TemporalDag`, `TemporalCpdag`, or `TemporalPag`).
    pub atom_kind: antecedent_discovery::GraphPosteriorAtomKind,
    /// Identifier fixed during preparation.
    pub identifier: crate::strategy_table::IdentifierId,
    /// Estimator fixed during preparation.
    pub estimator: crate::strategy_table::EstimatorId,
    /// Validation suite fixed during preparation.
    pub validation: crate::analysis::builder::RefuteSuite,
    /// Inference label frozen at preparation.
    pub inference: std::sync::Arc<str>,
    /// Posterior sample key, including repeated and unidentified samples.
    pub graph_keys: std::sync::Arc<[u64]>,
    /// Frozen posterior sample weights.
    pub weights: std::sync::Arc<[f64]>,
    /// Sample-level identification flags, preserving unidentified mass.
    pub identified: std::sync::Arc<[antecedent_prob::GraphIdentFlag]>,
    /// Number of identified atoms carrying a retained proof.
    pub identified_atom_count: usize,
    /// Circular-block bootstrap budget fixed during preparation.
    pub bootstrap_replicates: u32,
}

impl Inspect for CheckedTemporalGraphPosteriorEffect {
    type Info = CheckedTemporalGraphPosteriorEffectInfo;

    fn info(&self) -> CheckedTemporalGraphPosteriorEffectInfo {
        let (graph_keys, weights, identified) = self.sample_mass();
        CheckedTemporalGraphPosteriorEffectInfo {
            query: self.query().clone(),
            atom_kind: self.posterior().atom_kind,
            identifier: self.identifier(),
            estimator: self.estimator(),
            validation: self.validation(),
            inference: inference_label(self.inference()),
            graph_keys: Arc::from(graph_keys),
            weights: Arc::from(weights),
            identified: Arc::from(identified),
            identified_atom_count: self.identified_atom_count(),
            bootstrap_replicates: self.bootstrap_replicates(),
        }
    }
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
            EstimatorId::temporal_effect_for(&pulse, &InferenceMode::Frequentist),
            EstimatorId::TemporalLinearAdjustment
        );
        assert_eq!(
            EstimatorId::temporal_effect_for(&pulse, &bayesian),
            EstimatorId::BayesianTemporalGcomp
        );
        assert_eq!(
            EstimatorId::temporal_effect_for(&multi, &InferenceMode::Frequentist),
            EstimatorId::TemporalSequentialGcomp
        );
        assert_eq!(
            EstimatorId::temporal_effect_for(&multi, &bayesian),
            EstimatorId::TemporalSequentialGcomp
        );
    }
}
