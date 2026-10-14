//! Typed constructors of a transported response from the library transport routes.
//!
//! A transported response ([`TransportedResponse`]) is the one object a decision
//! (`BoundDecisionContract::evaluate_transported`), an inverse query ([`super::inverse_sources::transported_source`]) and the
//! assumption-range report read. Each constructor here takes the typed object a
//! transport route consumes, runs the route, and builds the response from the
//! route's own result, so the populations, the transport identities, the
//! assumptions and the support label are copied from the producing result and
//! never typed by the caller.
//!
//! | Route | Constructor | Transport identity |
//! |---|---|---|
//! | Policy value in another population | [`transported_policy_response`] | selection-diagram digest |
//! | Policy value in another period | [`temporal_policy_response`] | time-invariance digest |
//! | Adaptive-trial arm mean or contrast | [`adaptive_trial_response`] | trial selection-diagram digest |
//! | Transported synthesis | [`transported_synthesis_response`] | ledger transport digests |
//! | Surrogate bridge to another population | [`surrogate_bridge_response`] | bridge selection-diagram digest |
//!
//! # What a constructed response is
//!
//! A point, labelled `extrapolative`, whose receipt names the populations the
//! evidence was learned in, the population it was carried to and the premises
//! it is conditional on. It is never native target data and carries no interval
//! or calibration claim: the route's own standard error (a synthesis's model
//! variance) is an uncalibrated diagnostic only. The coordinate the value is
//! stated at must be in the target population of the transport; a coordinate in
//! the source population is refused.
//!
//! The period-shift route has a period, not a population, as its target. A
//! quantity in the target period is stated in the period-qualified population
//! [`period_population_label`] (`population@period`), which is also the source
//! label of the model's own period, so source and target stay distinct.
//!
//! Each route carries its own receipt route (`policy`, `temporal_policy`,
//! `adaptive_trial`, `synthesis`, `surrogate_bridge`), so two routes never
//! share a receipt even over the same populations, and a typed route cannot be
//! hand-declared: its identities are the producing result's digests.
//!
//! # Lifecycle
//!
//! * **Execute.** Every constructor re-runs its route; there is no trusted
//!   stored value. A refused route returns the route's own refusal unchanged.
//! * **Atomic.** [`transported_synthesis_response`] registers the synthesis in
//!   a copy of the ledger and replaces the caller's ledger only when the
//!   response was also built; a refusal leaves the ledger as it was.
//! * **Refresh.** New data changes the model, contract or ledger, which is a
//!   new evaluation and a new response identity. The transport declaration
//!   (diagram, invariance, premises) is kept only by being passed again; a
//!   response is never refreshed in place.
//! * **Borrowing.** Requests borrow their model, policy, contract and
//!   declarations for the call only; the response owns copies of everything it
//!   keeps.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::result_large_err,
    reason = "ExternalRefusal is the shared structured refusal returned by value on a cold once-per-construction path; boxing it would change the public error contract"
)]

use antecedent_contract::bridge::BridgedClaim;
use antecedent_contract::evidence_ledger::EvidenceLedger;
use antecedent_contract::{ClaimEnvelope, ExternalRefusal, ScientificQuantity};
use antecedent_core::ExecutionContext;
use antecedent_design::decision_transported_source::TransportedRefusal;
use antecedent_design::sequential_evaluation::SequentialEvaluationError;

pub use antecedent_design::adaptive_trial_transport::{
    AdaptiveTransportEstimate, AdaptiveTransportPremises, AdaptiveTransportRequest, TrialMechanism,
    TrialSelectionDiagram, transport_adaptive_trial, verify_adaptive_transport,
};
pub use antecedent_design::decision_transported_source::{
    TransportDeclaration, TransportReceipt, TransportedResponse, TransportedRoute,
};
pub use antecedent_design::temporal_policy_transport::{
    ActionAlphabet, HistoryAlphabet, TemporalPolicyEvaluation, TemporalPolicyRequest,
    TemporalTarget, TimeInvariance, evaluate_temporal_policy, verify_temporal_policy,
};
pub use antecedent_design::transported_bridge::{
    BridgeSelectionDiagram, SURROGATE_TRANSPORT_PREMISE_ID, TransportedSurrogateBridge,
};
pub use antecedent_design::transported_policy::{
    MechanismStatus, ModelMechanism, SelectionDiagram, TransportPremises,
    TransportedPolicyEvaluation, TransportedPolicyRequest, evaluate_transported_policy,
    verify_transported_policy,
};
pub use antecedent_design::transported_synthesis::{
    TransportedContribution, TransportedSynthesis, TransportedSynthesisMethod,
    TransportedSynthesisRefusal,
};

use antecedent_design::transported_synthesis::{
    synthesize_transported, synthesize_transported_in_population,
};

use super::composition_verification::{CalibrationDeclaration, ClaimStanding, CompositionNode};
use super::transport_revalidation::{
    TransportRevalidationRefusal, transported_claim_node, transported_claim_standing,
};

/// The stage a constructor refusal is reported under.
const STAGE: &str = "bind";

/// A transported response with the route result it was built from.
#[derive(Clone, Debug, PartialEq)]
pub struct TransportedConstruction<E> {
    /// The producing route's own result (the evaluation, estimate, synthesis or
    /// bridged claim), kept so its values and identity can be inspected.
    pub evaluation: E,
    /// The response a decision, an inverse query or a sensitivity range reads.
    pub response: TransportedResponse,
}

pub use antecedent_design::decision_transported_source::{
    AdaptiveTransportedValue, period_population_label,
};

impl<E> TransportedConstruction<E> {
    /// The standing this response states as a claim in a composition graph.
    ///
    /// The route is one of the library's own, so the claim rests on no provider
    /// (trust not applicable); its support follows the receipt's label; its
    /// uncertainty guarantee is `calibration`, which is
    /// [`CalibrationDeclaration::PointOnly`] unless a record established at
    /// exactly this target and quantity is declared. A record established at
    /// another target is withdrawn by revalidation, never inherited.
    ///
    /// # Errors
    /// A value whose metadata does not encode.
    pub fn claim_standing(
        &self,
        calibration: CalibrationDeclaration,
    ) -> Result<ClaimStanding, TransportRevalidationRefusal> {
        transported_claim_standing(&self.response, calibration, None)
    }

    /// The composition-graph node of this response under `key`, carrying
    /// [`Self::claim_standing`].
    ///
    /// # Errors
    /// A value whose metadata does not encode.
    pub fn composition_node(
        &self,
        key: &str,
        calibration: CalibrationDeclaration,
    ) -> Result<CompositionNode, TransportRevalidationRefusal> {
        transported_claim_node(key, &self.response, calibration, None)
    }
}

fn sequential(error: &SequentialEvaluationError) -> ExternalRefusal {
    error.to_refusal()
}

fn response_refusal(refusal: &TransportedRefusal) -> ExternalRefusal {
    refusal.to_refusal()
}

/// The refusal of a transported synthesis as the shared structured shape.
#[must_use]
pub fn synthesis_refusal(refusal: &TransportedSynthesisRefusal) -> ExternalRefusal {
    let base = ExternalRefusal::new(refusal.code(), STAGE, refusal.detail());
    match refusal {
        TransportedSynthesisRefusal::Ledger(error) => {
            base.with_offending(error.message.clone()).with_remedy(
                "register each observation once, declare the overlap between sources, and \
                 contribute a source either as native or as transported, never both",
            )
        }
        TransportedSynthesisRefusal::Numeric(error) => {
            base.with_offending(format!("{error:?}")).with_remedy(
                "supply finite variances, a covariance that is symmetric positive definite and, \
             for a transport shared by several claims, the sampling covariance",
            )
        }
        TransportedSynthesisRefusal::Declaration { source, remedy, .. } => {
            let base = match source {
                Some(source) => base.with_offending(format!("{source:?}")),
                None => base,
            };
            base.with_remedy(remedy)
        }
        _ => base,
    }
}

/// A policy's value in another population, stated at `coordinate`.
///
/// Runs [`evaluate_transported_policy`] on the request (a missing selection
/// diagram or target initial law, a shifted transition, observation or
/// utility mechanism, a target outside the source's support, and every
/// sequential-evaluation refusal are the route's), then builds the response
/// from the evaluation: the source population is the model's, the target is
/// the request's, the identity is the selection-diagram digest.
///
/// # Errors
/// The route's refusal, or [`TransportedRefusal`] as a structured refusal for a
/// coordinate stated in another population than the target.
pub fn transported_policy_response(
    request: &TransportedPolicyRequest<'_>,
    coordinate: ScientificQuantity,
    provider_id: &str,
    ctx: &ExecutionContext,
) -> Result<TransportedConstruction<TransportedPolicyEvaluation>, ExternalRefusal> {
    let evaluation = evaluate_transported_policy(request, ctx).map_err(|e| sequential(&e))?;
    let response =
        TransportedResponse::from_transported_policy(&evaluation, coordinate, provider_id)
            .map_err(|e| response_refusal(&e))?;
    Ok(TransportedConstruction { evaluation, response })
}

/// A policy's value in another period, stated at `coordinate`.
///
/// Runs [`evaluate_temporal_policy`] and builds the response from the
/// evaluation. The source population is `population@source_period`, the
/// target `population@target_period` ([`period_population_label`]), the
/// identity the digest of the declared time invariance; the fixed horizon and
/// every mechanism's declared status are assumptions of the receipt.
///
/// # Errors
/// The route's refusal (a missing invariance declaration or target law, a
/// horizon or alphabet change, a time-varying mechanism, ...), or the
/// response refusal for a coordinate not stated in the target period.
pub fn temporal_policy_response(
    request: &TemporalPolicyRequest<'_>,
    coordinate: ScientificQuantity,
    provider_id: &str,
    ctx: &ExecutionContext,
) -> Result<TransportedConstruction<TemporalPolicyEvaluation>, ExternalRefusal> {
    let evaluation = evaluate_temporal_policy(request, ctx).map_err(|e| sequential(&e))?;
    // The route accepted the request, so the invariance declaration is present.
    let Some(invariance) = request.invariance else {
        return Err(ExternalRefusal::new(
            antecedent_core::reason_code!("transport_missing_evidence"),
            STAGE,
            "temporal_policy.invariance_missing",
        )
        .with_remedy("declare the status of every mechanism across the two periods"));
    };
    let response = TransportedResponse::from_temporal_policy(
        &evaluation,
        &request.source.population,
        invariance,
        coordinate,
        provider_id,
    )
    .map_err(|e| response_refusal(&e))?;
    Ok(TransportedConstruction { evaluation, response })
}

/// An adaptive trial's transported arm mean or contrast, stated at
/// `coordinate`.
///
/// Runs [`transport_adaptive_trial`] (the retained assignment and monitoring
/// contract is required; a fixed design, a shifted outcome mechanism, an arm
/// under the probability floor and an unobserved target stratum are the
/// route's refusals) and builds the response from the estimate: the source
/// population is the trial's, the target the diagram's, the identity the
/// diagram digest, and the declared status of both mechanisms and the five
/// route premises are the assumptions. Several values of one estimate merge
/// into one grid with [`TransportedResponse::merge_points`].
///
/// # Errors
/// The route's refusal, or the response refusal for a coordinate not stated
/// in the target population.
pub fn adaptive_trial_response(
    request: &AdaptiveTransportRequest<'_>,
    value: AdaptiveTransportedValue,
    coordinate: ScientificQuantity,
    provider_id: &str,
) -> Result<TransportedConstruction<AdaptiveTransportEstimate>, ExternalRefusal> {
    let evaluation = transport_adaptive_trial(request).map_err(|e| sequential(&e))?;
    // The route accepted the request, so the diagram is present.
    let Some(diagram) = request.diagram else {
        return Err(ExternalRefusal::new(
            antecedent_core::reason_code!("transport_missing_evidence"),
            STAGE,
            "adaptive_transport.selection_diagram_missing",
        )
        .with_remedy("supply a selection diagram marking the stratum law and the outcome mechanism invariant or shifted"));
    };
    let response = TransportedResponse::from_adaptive_trial(
        &evaluation,
        value,
        request.trial_population,
        diagram,
        coordinate,
        provider_id,
    )
    .map_err(|e| response_refusal(&e))?;
    Ok(TransportedConstruction { evaluation, response })
}

/// The terminal-outcome value a surrogate bridge derives for a population the
/// validation trials were not run in, stated at the bridge's terminal quantity.
///
/// Derives the bridged claim from `claim` under the bridge's contract (a
/// selection node into the terminal outcome or a relation modifier, a diagram
/// between other populations, a surrogate effect outside the validated range
/// and an unlicensed trial-level fit are the bridge's refusals), then builds
/// the response from it: the source population is where the relation was
/// validated, the target the claim's population, the identity the bridge
/// diagram digest and the premise, its modifiers and the validation evidence
/// are assumptions.
///
/// # Errors
/// The bridge's refusal; [`TransportedRefusal::NotTransported`] (as a
/// structured refusal) when the claim is about the validated scope, which is a
/// native bridged claim; the response refusal for a validation and claim in
/// one population under different regimes (no source population distinct from
/// the target).
pub fn surrogate_bridge_response(
    bridge: &TransportedSurrogateBridge,
    claim: &ClaimEnvelope,
    provider_id: &str,
) -> Result<TransportedConstruction<BridgedClaim>, ExternalRefusal> {
    let bridged = bridge.derive(claim)?;
    let response = TransportedResponse::from_surrogate_bridge(bridge, &bridged, provider_id)
        .map_err(|e| response_refusal(&e))?;
    Ok(TransportedConstruction { evaluation: bridged, response })
}

/// A transported synthesis registered in the evidence ledger, stated at
/// `coordinate`.
///
/// Registers the combination of `contributions` under `method` (double
/// counting, unknown overlap, an undeclared transport variance and transported
/// contributions carried to different populations are the route's refusals)
/// and builds the response from the result: the source populations and the
/// target are the ones the ledger's transport identities recorded, never
/// caller text. With `native_population` the native contributions' population
/// is checked against the transports' target. The synthesis's model variance
/// is retained as an uncalibrated standard error and is not inherited by a
/// decision.
///
/// The ledger is replaced only when both the synthesis and the response were
/// built.
///
/// # Errors
/// The synthesis refusal, or the response refusal
/// ([`TransportedRefusal::NotTransported`] for native sources only, a
/// coordinate in another population than the target).
pub fn transported_synthesis_response(
    ledger: &mut EvidenceLedger,
    label: &str,
    contributions: &[TransportedContribution],
    method: &TransportedSynthesisMethod,
    native_population: Option<&str>,
    coordinate: ScientificQuantity,
    provider_id: &str,
) -> Result<TransportedConstruction<TransportedSynthesis>, ExternalRefusal> {
    let mut staged = ledger.clone();
    let synthesis = match native_population {
        Some(population) => synthesize_transported_in_population(
            &mut staged,
            label,
            contributions,
            method,
            population,
        ),
        None => synthesize_transported(&mut staged, label, contributions, method),
    }
    .map_err(|e| synthesis_refusal(&e))?;
    let response =
        TransportedResponse::from_transported_synthesis(&synthesis, coordinate, provider_id)
            .map_err(|e| response_refusal(&e))?;
    *ledger = staged;
    Ok(TransportedConstruction { evaluation: synthesis, response })
}
