//! Prepare-time identification caches retained by a prepared handle.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_data::TemporalIndexer;
use antecedent_expr::IdentifiedEstimand;
use antecedent_graph::Pag;
use antecedent_identify::{IdentificationEnvelope, IdentificationResult};
use std::sync::Arc;

/// Prepare-time identification products for the static ATE / response path.
///
/// Everything identification reads — identifier, graph, query, RD config — is
/// frozen when the handle is built, and identification is deterministic, so an
/// estimate click reuses these instead of re-running identification. Results
/// carry an `exec.identify.cached` diagnostic so reuse is observable.
#[derive(Clone, Debug)]
pub struct CachedStaticIdentification {
    /// Identification result computed at prepare time.
    pub identification: IdentificationResult,
    /// Estimand selected for the prepared estimator.
    pub estimand: IdentifiedEstimand,
}

/// Prepare-time generalized-adjustment envelope for a supplied PAG.
#[derive(Clone, Debug)]
pub(crate) struct CachedPagIdentification {
    /// Per-completion identification results and their probability mass.
    pub envelope: Arc<IdentificationEnvelope<Pag>>,
    /// Public aggregate identification result derived from [`Self::envelope`].
    pub identification: IdentificationResult,
}

/// Prepare-time MEC envelope for a supplied CPDAG.
#[derive(Clone, Debug)]
pub(crate) struct CachedCpdagIdentification {
    /// Per-completion identification results and their probability mass.
    pub envelope: Arc<IdentificationEnvelope<antecedent_graph::Dag>>,
    /// Public aggregate identification result derived from [`Self::envelope`].
    pub identification: IdentificationResult,
}

/// A class conditional effect is sealed only when every enumerated completion
/// has full nonparametric identification with the same adjustment target.
pub(in crate::analysis) fn conditional_class_proof_is_complete<G>(
    envelope: &IdentificationEnvelope<G>,
) -> bool {
    let Some(invariant) = envelope.invariant.as_ref() else { return false };
    envelope.status == antecedent_identify::IdentificationStatus::NonparametricallyIdentified
        && envelope.truncated_completions == 0
        && envelope.unidentified_weight.0 <= f64::EPSILON
        && envelope.identified_weight.0 >= 1.0 - f64::EPSILON
        && !envelope.cases.is_empty()
        && envelope.cases.iter().all(|case| {
            case.result.status
                == antecedent_identify::IdentificationStatus::NonparametricallyIdentified
                && case.result.estimands.iter().any(|candidate| {
                    candidate.is_adjustment_shaped()
                        && candidate.method == invariant.method
                        && candidate.adjustment_set == invariant.adjustment_set
                })
        })
}

/// Prepare-time TemporalCpdag/Pag envelope (completions + unfold indexers).
#[derive(Clone, Debug)]
pub(crate) struct CachedTemporalClassIdentification {
    /// Generalized-adjustment envelope over `TemporalDag` completions.
    pub envelope: Arc<antecedent_identify::TemporalClassEnvelope>,
    /// Functional-specific certificates for every requested response or mediation horizon.
    pub by_horizon: Vec<(u32, Arc<antecedent_identify::TemporalClassEnvelope>)>,
}

/// Prepare-time identification products for one temporal horizon.
#[derive(Clone, Debug)]
pub struct CachedTemporalHorizonIdentification {
    /// Requested horizon these products were identified for.
    pub horizon: u32,
    /// Unfolded backdoor identification at [`Self::horizon`].
    pub identification: IdentificationResult,
    /// Estimand selected for this horizon.
    pub estimand: IdentifiedEstimand,
    /// Finite-unfolding indexer paired with [`Self::identification`].
    pub indexer: TemporalIndexer,
}

/// Prepare-time temporal-backdoor identification (ADR 0021).
///
/// For temporal [`CausalQuery::Response`], identification + lag indexer are
/// frozen once per unique requested horizon. For scalar
/// [`CausalQuery::TemporalEffect`] (Pulse / single-step Sustained), they are
/// frozen for that query's horizon. For [`CausalQuery::Mediation`]
/// (`TemporalMediationEffect`), identification is frozen independently for
/// every requested horizon.
#[derive(Clone, Debug)]
pub struct CachedTemporalIdentification {
    /// One entry per unique requested horizon, in query order.
    pub by_horizon: Arc<[CachedTemporalHorizonIdentification]>,
}

impl CachedTemporalIdentification {
    /// Identification products for `horizon`, if prepared.
    #[must_use]
    pub fn get(&self, horizon: u32) -> Option<&CachedTemporalHorizonIdentification> {
        self.by_horizon.iter().find(|entry| entry.horizon == horizon)
    }
}
