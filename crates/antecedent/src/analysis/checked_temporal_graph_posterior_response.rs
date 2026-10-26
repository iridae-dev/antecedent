//! Frozen temporal graph-posterior response: retained atoms, weights, per-atom
//! identification, and the selected inference, validation, and physical plan.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::analysis::execute::CheckedGraphPosteriorResponseInfo;
use crate::analysis::prepared::Inspect;
use crate::analysis::route_guards::inference_label;
use std::sync::Arc;

use antecedent_core::{CausalQuery, ResponseFunctional, ResponseQuery};
use antecedent_discovery::{GraphPosterior, GraphPosteriorAtomKind};
use antecedent_prob::WeightedGraphSamples;

use crate::error::CausalError;
use crate::planner::PhysicalExecutionPlan;
use crate::strategy_table::{EstimatorId, IdentifierId};
use crate::{ClassPrior, InferenceMode};

use super::builder::{DataInput, RefuteSuite};
use super::checked_route_frozen::{CheckedRouteFrozen, PlanBindingScope};
use super::checked_temporal_graph_posterior_effect::CheckedTemporalGraphPosteriorProof;
use super::checked_temporal_response::temporal_response_is_direct;
use super::execute::Study;
use super::posterior_binding::{CacheKeys, PosteriorProof, validate_posterior_cache};
use super::route_guards::complete_mean_response;

/// Per-atom identification retained for a temporal graph posterior response:
/// the same proof shape the temporal effect route retains (`Dbn` per-horizon
/// backdoor proofs for `TemporalDag` atoms, `Class` completion envelopes for
/// TemporalCpdag/TemporalPag atoms).
pub(crate) type TemporalPosteriorResponseProof = CheckedTemporalGraphPosteriorProof;

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
    class_prior: Option<ClassPrior>,
    frozen: CheckedRouteFrozen,
}

/// The frozen premises of [`CheckedTemporalGraphPosteriorResponse::prepare`], carried together.
pub(crate) struct CheckedTemporalGraphPosteriorResponsePremises {
    pub(crate) posterior: GraphPosterior,
    pub(crate) query: ResponseQuery,
    pub(crate) proof: TemporalPosteriorResponseProof,
    pub(crate) physical: std::sync::Arc<PhysicalExecutionPlan>,
    pub(crate) inference: InferenceMode,
    pub(crate) validation: RefuteSuite,
    pub(crate) bootstrap_replicates: u32,
    pub(crate) max_completions: Option<usize>,
    pub(crate) class_prior: Option<ClassPrior>,
}

impl CheckedTemporalGraphPosteriorResponse {
    /// Whether `study` is a coordinate this operation seals: a series
    /// complete-observation mean temporal response over `TemporalDag`,
    /// `TemporalCpdag`, or `TemporalPag` posterior atoms with the default
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
            // A single-horizon derivative over TemporalDag atoms is answered atom by atom with
            // the interval of the whole procedure, not by the response design.
            && !(posterior.atom_kind == GraphPosteriorAtomKind::Dag
                && super::execute::dbn_route_query(&study.query))
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

    pub(crate) fn prepare(
        premises: CheckedTemporalGraphPosteriorResponsePremises,
    ) -> Result<Self, CausalError> {
        let CheckedTemporalGraphPosteriorResponsePremises {
            posterior,
            query,
            proof,
            physical,
            inference,
            validation,
            bootstrap_replicates,
            max_completions,
            class_prior,
        } = premises;
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
            || !complete_mean_response(&query)
            || query.functional.primary_pair().is_none()
        {
            return Err(CausalError::Unsupported {
                message: "checked temporal graph-posterior response requires a complete-observation mean curve or single-step intervention response with a treatment/outcome pair",
            });
        }
        let intervention_response =
            matches!(query.functional, ResponseFunctional::InterventionResponse { .. });
        if (!intervention_response && validation != RefuteSuite::None) || !validation.is_point() {
            return Err(CausalError::Unsupported {
                message: "checked temporal graph-posterior response supports no refuters for curves and none/cheap/full for intervention responses",
            });
        }
        let estimator = EstimatorId::temporal_response_for(&inference);
        let frozen = CheckedRouteFrozen::new(
            identifier,
            estimator,
            inference,
            bootstrap_replicates,
            validation,
            physical,
            max_completions,
        );
        frozen.validate_plan_binding(
            &CausalQuery::Response(query.clone()),
            PlanBindingScope::Estimator,
            "temporal graph-posterior response plan disagrees with its retained query or estimator",
        )?;
        validate_posterior_cache(
            &posterior,
            &proof,
            CacheKeys::CacheLocal,
            "temporal graph-posterior identification cache does not match the frozen atoms and weights",
        )?;
        Ok(Self {
            posterior: Arc::new(posterior),
            query,
            proof,
            class_prior,
            frozen,
        })
    }

    /// The frozen procedure settings sealed with the proof.
    pub(crate) fn frozen(&self) -> &CheckedRouteFrozen {
        &self.frozen
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
        self.frozen.physical()
    }
    pub(crate) const fn inference(&self) -> &InferenceMode {
        self.frozen.inference()
    }
    pub(crate) const fn identifier(&self) -> IdentifierId {
        self.frozen.identifier()
    }
    pub(crate) const fn estimator(&self) -> EstimatorId {
        self.frozen.estimator()
    }
    pub(crate) const fn validation(&self) -> RefuteSuite {
        self.frozen.validation()
    }
    pub(crate) fn class_prior(&self) -> Option<&ClassPrior> {
        self.class_prior.as_ref()
    }

    /// Verify that a refresh still carries the posterior the proofs were built for.
    #[must_use]
    pub(crate) fn matches_posterior(&self, posterior: Option<&GraphPosterior>) -> bool {
        posterior.is_some_and(|candidate| {
            candidate.atom_kind == self.posterior.atom_kind
                && candidate.n_graphs == self.posterior.n_graphs
                && candidate.weights.as_ref() == self.posterior.weights.as_ref()
                && candidate.atoms.as_ref() == self.posterior.atoms.as_ref()
                && candidate.lag_masks.as_deref() == self.posterior.lag_masks.as_deref()
                && candidate.mark_atoms.as_deref() == self.posterior.mark_atoms.as_deref()
        })
    }
}

impl PosteriorProof for CheckedTemporalGraphPosteriorResponse {
    fn graphs(&self) -> &WeightedGraphSamples {
        self.proof.graphs()
    }

    fn atom_keys(&self) -> Vec<u64> {
        self.proof.atom_keys()
    }
}

/// Frozen temporal response posterior: atom keys, weights, identification
/// flags, and the selected inference and validation. The retained fields are
/// those of [`CheckedGraphPosteriorResponseInfo`].
pub type CheckedTemporalGraphPosteriorResponseInfo = CheckedGraphPosteriorResponseInfo;

impl Inspect for CheckedTemporalGraphPosteriorResponse {
    type Info = CheckedTemporalGraphPosteriorResponseInfo;

    fn info(&self) -> CheckedTemporalGraphPosteriorResponseInfo {
        let (graph_keys, weights, identified) = self.sample_mass();
        CheckedGraphPosteriorResponseInfo {
            query: self.query().clone(),
            atom_kind: self.posterior().atom_kind,
            graph_keys: Arc::from(graph_keys),
            weights: Arc::from(weights),
            identified: Arc::from(identified),
            identifier: self.identifier(),
            estimator: self.estimator(),
            inference: inference_label(self.inference()),
            validation: self.validation(),
        }
    }
}
