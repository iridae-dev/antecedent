//! Frozen temporal graph-posterior response: retained atoms, weights, per-atom
//! identification, and the selected inference, validation, and physical plan.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    CausalQuery, ObservationSpec, OutcomeFunctional, ResponseFunctional, ResponseQuery,
    TargetPopulation,
};
use antecedent_discovery::{GraphPosterior, GraphPosteriorAtomKind};
use antecedent_prob::GraphIdentFlag;

use crate::error::CausalError;
use crate::planner::PhysicalExecutionPlan;
use crate::strategy_table::{EstimatorId, IdentifierId};
use crate::{ClassPrior, InferenceMode};

use super::builder::{DataInput, RefuteSuite};
use super::checked_temporal_response::temporal_response_is_direct;
use super::execute::Study;
use super::prepared::{
    CachedDbnPosteriorIdentification, CachedTemporalClassPosteriorIdentification,
};
use super::route_guards::complete_mean_response;

/// Per-atom identification retained for a temporal graph posterior.
#[derive(Clone, Debug)]
pub(crate) enum TemporalPosteriorResponseProof {
    /// TemporalDag atoms: per-horizon temporal backdoor proofs.
    Dbn(Arc<CachedDbnPosteriorIdentification>),
    /// TemporalCpdag/TemporalPag atoms: one completion envelope per atom.
    Class(Arc<CachedTemporalClassPosteriorIdentification>),
}

impl TemporalPosteriorResponseProof {
    fn graphs(&self) -> &antecedent_prob::WeightedGraphSamples {
        match self {
            Self::Dbn(cache) => &cache.graphs,
            Self::Class(cache) => &cache.graphs,
        }
    }
}

/// Frozen temporal graph-posterior response execution.
///
/// Atoms, weights, and identification flags are retained together with the
/// per-atom proofs, so a click cannot re-enumerate the posterior, and
/// unidentified mass is preserved exactly as the prepare-time envelope
/// recorded it.
#[derive(Clone, Debug)]
pub(crate) struct CheckedTemporalGraphPosteriorResponse {
    posterior: Arc<GraphPosterior>,
    query: ResponseQuery,
    proof: TemporalPosteriorResponseProof,
    physical: PhysicalExecutionPlan,
    inference: InferenceMode,
    identifier: IdentifierId,
    estimator: EstimatorId,
    validation: RefuteSuite,
    bootstrap_replicates: u32,
    max_completions: Option<usize>,
    class_prior: Option<ClassPrior>,
}

impl CheckedTemporalGraphPosteriorResponse {
    /// Whether `study` is a coordinate this operation seals: a series
    /// complete-observation mean temporal response over TemporalDag,
    /// TemporalCpdag, or TemporalPag posterior atoms with the default
    /// point-estimate settings. The one-shot facade and the prepared sealer
    /// share this predicate.
    #[must_use]
    pub(crate) fn admits(study: &Study) -> bool {
        let (Some(posterior), CausalQuery::Response(query)) =
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
            && study.observation_delayed_entry.is_none()
            && query.is_temporal()
            && complete_mean_response(query)
            && temporal_response_is_direct(query)
            && matches!(
                (&query.functional, study.refute),
                (_, RefuteSuite::None)
                    | (
                        ResponseFunctional::InterventionResponse { .. },
                        RefuteSuite::Cheap | RefuteSuite::Full
                    )
            )
            && study.identifier.is_none_or(|id| match posterior.atom_kind {
                GraphPosteriorAtomKind::Dag => id == IdentifierId::TemporalBackdoorUnfolded,
                _ => id == IdentifierId::GeneralizedAdjustment,
            })
            && study
                .estimator
                .is_none_or(|id| id == EstimatorId::temporal_response_for(&study.inference))
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare(
        posterior: GraphPosterior,
        query: ResponseQuery,
        proof: TemporalPosteriorResponseProof,
        physical: PhysicalExecutionPlan,
        inference: InferenceMode,
        validation: RefuteSuite,
        bootstrap_replicates: u32,
        max_completions: Option<usize>,
        class_prior: Option<ClassPrior>,
    ) -> Result<Self, CausalError> {
        let (identifier, kind_matches) = match (posterior.atom_kind, &proof) {
            (GraphPosteriorAtomKind::Dag, TemporalPosteriorResponseProof::Dbn(_)) => {
                (IdentifierId::TemporalBackdoorUnfolded, true)
            }
            (
                GraphPosteriorAtomKind::Cpdag | GraphPosteriorAtomKind::Pag,
                TemporalPosteriorResponseProof::Class(_),
            ) => (IdentifierId::GeneralizedAdjustment, true),
            _ => (IdentifierId::TemporalBackdoorUnfolded, false),
        };
        if !kind_matches {
            return Err(CausalError::Unsupported {
                message: "checked temporal graph-posterior response requires TemporalDag atoms with per-horizon proofs or TemporalCpdag/TemporalPag atoms with completion envelopes",
            });
        }
        super::execute::dbn_posterior_response_supported(&query)?;
        if !temporal_response_is_direct(&query)
            || query.observation != ObservationSpec::Complete
            || query.target_population != TargetPopulation::AllObserved
            || !matches!(query.outcome_functional, OutcomeFunctional::Mean)
            || query.functional.primary_pair().is_none()
        {
            return Err(CausalError::Unsupported {
                message: "checked temporal graph-posterior response requires a complete-observation mean curve or single-step intervention response with a treatment/outcome pair",
            });
        }
        let intervention_response =
            matches!(query.functional, ResponseFunctional::InterventionResponse { .. });
        if (!intervention_response && validation != RefuteSuite::None)
            || !matches!(validation, RefuteSuite::None | RefuteSuite::Cheap | RefuteSuite::Full)
        {
            return Err(CausalError::Unsupported {
                message: "checked temporal graph-posterior response supports no refuters for curves and none/cheap/full for intervention responses",
            });
        }
        let estimator = EstimatorId::temporal_response_for(&inference);
        if physical.logical.query != CausalQuery::Response(query.clone())
            || physical
                .logical
                .record
                .estimator
                .as_deref()
                .is_some_and(|selected| selected != estimator.as_str())
        {
            return Err(CausalError::Compile {
                message: "temporal graph-posterior response plan disagrees with its retained query or estimator".into(),
            });
        }
        let graphs = proof.graphs();
        if graphs.n_samples != posterior.n_graphs
            || graphs.weights.as_ref() != posterior.weights.as_ref()
            || graphs.identified.len() != posterior.n_graphs
        {
            return Err(CausalError::Unsupported {
                message: "temporal graph-posterior identification cache does not match the frozen atoms and weights",
            });
        }
        match &proof {
            TemporalPosteriorResponseProof::Dbn(cache) => {
                if cache.atoms.iter().any(|atom| !graphs.graph_keys.contains(&atom.key)) {
                    return Err(CausalError::Unsupported {
                        message: "temporal graph-posterior proof names an atom outside the frozen posterior",
                    });
                }
            }
            TemporalPosteriorResponseProof::Class(cache) => {
                if cache.class_atoms.iter().any(|atom| !graphs.graph_keys.contains(&atom.key)) {
                    return Err(CausalError::Unsupported {
                        message: "temporal graph-posterior proof names an atom outside the frozen posterior",
                    });
                }
            }
        }
        Ok(Self {
            posterior: Arc::new(posterior),
            query,
            proof,
            physical,
            inference,
            identifier,
            estimator,
            validation,
            bootstrap_replicates,
            max_completions,
            class_prior,
        })
    }

    pub(crate) fn posterior(&self) -> &GraphPosterior {
        &self.posterior
    }
    pub(crate) fn query(&self) -> &ResponseQuery {
        &self.query
    }
    pub(crate) fn proof(&self) -> &TemporalPosteriorResponseProof {
        &self.proof
    }
    pub(crate) fn physical(&self) -> &PhysicalExecutionPlan {
        &self.physical
    }
    pub(crate) const fn inference(&self) -> &InferenceMode {
        &self.inference
    }
    pub(crate) const fn identifier(&self) -> IdentifierId {
        self.identifier
    }
    pub(crate) const fn estimator(&self) -> EstimatorId {
        self.estimator
    }
    pub(crate) const fn validation(&self) -> RefuteSuite {
        self.validation
    }
    pub(crate) const fn bootstrap_replicates(&self) -> u32 {
        self.bootstrap_replicates
    }
    pub(crate) const fn max_completions(&self) -> Option<usize> {
        self.max_completions
    }
    pub(crate) fn class_prior(&self) -> Option<&ClassPrior> {
        self.class_prior.as_ref()
    }

    /// Read-only per-sample atom identities, posterior weights, and
    /// identification flags, including unidentified samples.
    pub(crate) fn atom_weights(&self) -> (&[u64], &[f64], &[GraphIdentFlag]) {
        let graphs = self.proof.graphs();
        (&graphs.graph_keys, &graphs.weights, &graphs.identified)
    }

    /// Verify that a refresh still carries the posterior the proofs were built for.
    #[must_use]
    pub(crate) fn matches_posterior(&self, posterior: Option<&GraphPosterior>) -> bool {
        posterior.is_some_and(|candidate| {
            candidate.atom_kind == self.posterior.atom_kind
                && candidate.n_graphs == self.posterior.n_graphs
                && candidate.weights.as_ref() == self.posterior.weights.as_ref()
                && candidate.adjacency.as_ref() == self.posterior.adjacency.as_ref()
                && candidate.lag_masks.as_deref() == self.posterior.lag_masks.as_deref()
                && candidate.mark_masks.as_deref() == self.posterior.mark_masks.as_deref()
        })
    }
}
