//! Decision inputs from a cohort-mean treatment-contrast source.
//!
//! The source ([`CohortDecisionSource`]) is built and replayed in the artifact layer from a
//! retained fitted effect model and a declared cohort. This module is the thin native
//! binding to the decision boundary: it turns that source into one [`MeanSource`] with one
//! coordinate (the cohort-mean `mean_difference` of the declared contrast) and into a
//! [`DecisionInput`] whose trust is resolved from evidence, never from a label.
//!
//! A mean answers the expectation of an affine utility of that coordinate and nothing
//! else. The existing evaluators refuse a probability, a quantile, a nonlinear utility,
//! a hard constraint and an outcome-law input on it; this module adds no route around
//! that. No utility standard error is attached: the model-based variance diagnostic of a
//! linear compact model stays on the source and is never inherited by a decision.
//!
//! Trust. A source built from a verified native analysis carries a native execution
//! record whose identity is the source identity, so `TrustRequirement::Native` accepts it.
//! A source built from a caller-declared compact model is an external attestation by that
//! model's identity and is refused where a native source is required.

use antecedent_contract::ExternalTrustState;
use antecedent_io_core::cohort_decision::{CohortDecisionSource, CohortRefusal};

use crate::composition_boundary::{
    BoundaryError, DecisionInput, NativeExecutionRecord, TrustEvidence, TrustRequirement,
};
use crate::decision_eval::MeanSource;
use crate::inverse_query::ForwardClaim;

/// The cohort-mean contrast as a mean source with exactly one coordinate.
///
/// The snapshot identity is the cohort digest and the causal-contract identity is the
/// parent claim, so a decision result names both.
#[must_use]
pub fn cohort_mean_source(source: &CohortDecisionSource) -> MeanSource {
    MeanSource {
        coordinates: vec![source.quantity().clone()],
        means: vec![source.mean_contrast()],
        provider_id: source.provider_id(),
        snapshot_id: source.cohort().digest(),
        causal_contract_id: source.model().parent().claim_id.clone(),
        rng_id: "none:cohort_mean".to_owned(),
    }
}

/// Trust evidence for `source`: a native execution record for a source built from a
/// verified native analysis, otherwise an attestation by the declared model's identity.
#[must_use]
pub fn cohort_trust_evidence(source: &CohortDecisionSource) -> TrustEvidence {
    if source.is_native() {
        TrustEvidence::NativeExecution(NativeExecutionRecord {
            execution_id: source.identity().to_owned(),
            provider_id: source.provider_id(),
            snapshot_id: source.cohort().digest(),
        })
    } else {
        TrustEvidence::External(ExternalTrustState::ExternallyAttested {
            attestor: source.model().identity().to_owned(),
        })
    }
}

/// A decision input for `source`.
///
/// The coordinate's support is `Supported` inside the model's own population and
/// `Extrapolative` under a declared covariate shift, so the default support policy admits
/// only the former and a transported decision needs an explicit weaker policy.
///
/// # Errors
/// A blank input id, or `requirement` is [`TrustRequirement::Native`] and the source is a
/// declared model.
pub fn cohort_decision_input(
    id: &str,
    source: &CohortDecisionSource,
    requirement: TrustRequirement,
) -> Result<DecisionInput, BoundaryError> {
    DecisionInput::from_mean_source(
        id,
        cohort_mean_source(source),
        &[source.support_status()],
        &cohort_trust_evidence(source),
        requirement,
    )
}

/// The cohort-mean contrast as a licensed forward source of an inverse query.
///
/// An inverse query asks which finite action meets a target mean, so a mean source
/// answers a target on an affine utility and the existing engine refuses a probability,
/// quantile or nonlinear utility on it. This entry takes no acceptance policy: a cohort
/// mean transported under a declared covariate shift is a transported claim and is read
/// through [`crate::inverse_forward_source::cohort_forward_source`] with
/// `AcceptsTransported`, which carries the transport identity into the result.
///
/// # Errors
/// A source built under a declared covariate shift: the shape every consumer that does
/// not accept transported claims returns (`cell_not_licensed`,
/// `inverse_forward_source.transported_not_accepted`).
pub fn cohort_forward_claim(source: &CohortDecisionSource) -> Result<ForwardClaim, CohortRefusal> {
    if source.shift().is_some() {
        return Err(CohortRefusal {
            code: antecedent_core::reason_code!("cell_not_licensed"),
            detail: crate::inverse_forward_source::DETAIL_TRANSPORTED_NOT_ACCEPTED,
            subject: source.cohort().population_id().to_owned(),
            remedy: "accept transported claims explicitly (AcceptsTransported) so the result \
                     carries the transport identity, or supply native target data",
        });
    }
    Ok(ForwardClaim::Means(cohort_mean_source(source)))
}
