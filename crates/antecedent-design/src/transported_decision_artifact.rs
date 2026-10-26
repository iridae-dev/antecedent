//! Portable transported decision artifact.
//!
//! A decision made on a value carried to a target population
//! ([`evaluate_transported_decision`]) is exported as the contract it was made
//! under, the carried point with its [`TransportReceipt`], the support policy
//! that accepted it, and the values the decision produced. A consumer never
//! trusts those values: [`consume_transported_decision`] rebuilds the contract
//! from its declaration (its identity is recomputed), validates the receipt as
//! at construction, re-evaluates the decision and requires the decision
//! identity and every action value to match bit for bit. A resealed edit of
//! the receipt, the means, the contract or the policy changes what the
//! re-evaluation produces and is refused.
//!
//! # Scope
//!
//! Only a transported point is exported here; a transported law is a
//! [`antecedent_io_core::distribution_artifact::DistributionArtifact`] with its
//! own export, and is refused with `law_not_exportable`. The artifact claims
//! point-only values under declared premises and nothing about coverage.
//!
//! # Refresh
//!
//! The artifact is a record of one decision, not a prepared object. New
//! outcome-only data at the source population is a new transported value and
//! therefore a new artifact: nothing here is refreshed in place. What a
//! consumer keeps across such a refresh is the contract declaration and the
//! support policy; it recomputes the carried value and the receipt's source
//! identity; it refuses data of another population, quantity or arm (the
//! receipt's target population and the contract's coordinates must still
//! agree).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::ScientificQuantity;
use antecedent_core::SupportStatus;
use antecedent_io_core::convert::to_cbor;
use antecedent_io_core::error::IoError;
use antecedent_io_core::quantity_wire::ScientificQuantityWire;
use serde::{Deserialize, Serialize};

use crate::composition_boundary::{SupportPolicy, UnsupportedActionPolicy};
use crate::decision_artifact::{
    contract_from_json, contract_to_json, decode_section, encode_section,
};
use crate::decision_contract::DecisionContract;
use crate::decision_eval::MeanSource;
use crate::decision_transported_source::{
    TransportReceipt, TransportedDecision, TransportedDemand, TransportedRefusal,
    TransportedResponse, TransportedValue, evaluate_transported_decision,
};

/// Container kind of the artifact.
pub const TRANSPORTED_DECISION_ARTIFACT_KIND: &str = "transported_decision_v1";
/// Body version this reader accepts.
pub const TRANSPORTED_DECISION_ARTIFACT_VERSION: u16 = 1;
/// Most coordinates a transported point may carry.
pub const MAX_TRANSPORTED_DECISION_COORDINATES: usize = 1024;

const BODY_SECTION: &str = "transported_decision_body";
const MAX_LIST_ITEMS: usize = 1024;
const MAX_TEXT_BYTES: usize = 8 * 1024;

const DETAIL_LAW_NOT_EXPORTABLE: &str = "transported_decision_artifact.law_not_exportable";
const DETAIL_BOUNDS: &str = "transported_decision_artifact.bounds_exceeded";
const DETAIL_REPLAY: &str = "transported_decision_artifact.replay_mismatch";
const DETAIL_IDENTITY: &str = "transported_decision_artifact.identity_expected";
const DETAIL_INVALID: &str = "transported_decision_artifact.invalid";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct OutcomeWire {
    action: String,
    expected_utility_bits: u64,
    value_bits: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Body {
    version: u16,
    contract_json: String,
    coordinates: Vec<ScientificQuantityWire>,
    mean_bits: Vec<u64>,
    provider_id: String,
    snapshot_id: String,
    causal_contract_id: String,
    rng_id: String,
    /// Posterior mean or frequentist point estimate. Absent on an older artifact,
    /// which is read as a frequentist point estimate.
    #[serde(default = "frequentist_estimate")]
    estimate: crate::decision_eval::MeanEstimate,
    receipt: TransportReceipt,
    unsupported: String,
    weakest_support: String,
    decision_identity: String,
    outcomes: Vec<OutcomeWire>,
}

fn frequentist_estimate() -> crate::decision_eval::MeanEstimate {
    crate::decision_eval::MeanEstimate::Frequentist
}

fn refused(code: &'static str, detail: &str, message: impl std::fmt::Display) -> IoError {
    crate::plan_common::refused_with(code, detail, message)
}

fn from_transported(error: &TransportedRefusal) -> IoError {
    IoError::Refused { code: error.code(), message: format!("{}: {error}", error.detail()) }
}

fn invalid(message: impl std::fmt::Display) -> IoError {
    refused(antecedent_core::reason_code!("invalid_argument"), DETAIL_INVALID, message)
}

const fn unsupported_name(policy: UnsupportedActionPolicy) -> &'static str {
    match policy {
        UnsupportedActionPolicy::RequireAllActions => "require_all_actions",
        UnsupportedActionPolicy::CompareSupported => "compare_supported",
    }
}

fn unsupported_from(name: &str) -> Result<UnsupportedActionPolicy, IoError> {
    match name {
        "require_all_actions" => Ok(UnsupportedActionPolicy::RequireAllActions),
        "compare_supported" => Ok(UnsupportedActionPolicy::CompareSupported),
        other => Err(invalid(format!("unknown unsupported-action policy `{other}`"))),
    }
}

fn outcome_wires(decision: &TransportedDecision) -> Vec<OutcomeWire> {
    decision
        .decision
        .outcomes
        .iter()
        .map(|o| OutcomeWire {
            action: o.id.clone(),
            expected_utility_bits: o.expected_utility.to_bits(),
            value_bits: o.value.to_bits(),
        })
        .collect()
}

/// Decide on a transported point and export the decision with what it stands on.
///
/// # Errors
/// A transported law ([`DETAIL_LAW_NOT_EXPORTABLE`] under
/// `route_not_supported`), any refusal of [`evaluate_transported_decision`], an
/// invalid contract, an oversized declaration or an encoding failure.
pub fn export_transported_decision(
    contract: &DecisionContract,
    response: &TransportedResponse,
    policy: &SupportPolicy,
    artifact_id: &str,
) -> Result<Vec<u8>, IoError> {
    let TransportedValue::Means(mean) = response.value() else {
        return Err(refused(
            antecedent_core::reason_code!("route_not_supported"),
            DETAIL_LAW_NOT_EXPORTABLE,
            "a transported law is exported as its distribution artifact; this artifact carries a point",
        ));
    };
    let decision = evaluate_transported_decision(
        contract,
        response,
        TransportedDemand::ExpectedUtility,
        policy,
    )
    .map_err(|error| from_transported(&error))?;
    let body = Body {
        version: TRANSPORTED_DECISION_ARTIFACT_VERSION,
        contract_json: contract_to_json(contract)?,
        coordinates: mean.coordinates.iter().map(ScientificQuantityWire::from).collect(),
        mean_bits: mean.means.iter().map(|m| m.to_bits()).collect(),
        provider_id: mean.provider_id.clone(),
        snapshot_id: mean.snapshot_id.clone(),
        causal_contract_id: mean.causal_contract_id.clone(),
        rng_id: mean.rng_id.clone(),
        estimate: mean.estimate,
        receipt: response.receipt().clone(),
        unsupported: unsupported_name(policy.unsupported).to_owned(),
        weakest_support: policy.weakest_support.as_str().to_owned(),
        decision_identity: decision.identity.clone(),
        outcomes: outcome_wires(&decision),
    };
    encode_section(
        TRANSPORTED_DECISION_ARTIFACT_KIND,
        BODY_SECTION,
        artifact_id,
        "transported_decision",
        &to_cbor(&body)?,
    )
}

/// A consumed transported decision: the re-derived decision and the response
/// it was re-derived from.
#[derive(Clone, Debug, PartialEq)]
pub struct ConsumedTransportedDecision {
    /// The decision, recomputed by the consumer.
    pub decision: TransportedDecision,
    /// The transported point rebuilt from the artifact.
    pub response: TransportedResponse,
}

fn check_text(text: &str, what: &str) -> Result<(), IoError> {
    if text.len() > MAX_TEXT_BYTES {
        return Err(refused(
            antecedent_core::reason_code!("invalid_argument"),
            DETAIL_BOUNDS,
            format!("{what} exceeds {MAX_TEXT_BYTES} bytes"),
        ));
    }
    Ok(())
}

fn check_bounds(body: &Body) -> Result<(), IoError> {
    let over = |what: &str| {
        refused(
            antecedent_core::reason_code!("invalid_argument"),
            DETAIL_BOUNDS,
            format!("{what} exceeds its item bound"),
        )
    };
    if body.coordinates.len() > MAX_TRANSPORTED_DECISION_COORDINATES {
        return Err(over("coordinates"));
    }
    if body.outcomes.len() > MAX_LIST_ITEMS
        || body.receipt.source_populations.len() > MAX_LIST_ITEMS
        || body.receipt.transport_identities.len() > MAX_LIST_ITEMS
        || body.receipt.assumptions.len() > MAX_LIST_ITEMS
    {
        return Err(over("receipt or outcome lists"));
    }
    for text in [
        &body.contract_json,
        &body.provider_id,
        &body.snapshot_id,
        &body.causal_contract_id,
        &body.rng_id,
        &body.decision_identity,
        &body.receipt.source_identity,
        &body.receipt.target_population,
    ] {
        check_text(text, "a text field")?;
    }
    for text in body
        .receipt
        .source_populations
        .iter()
        .chain(&body.receipt.transport_identities)
        .chain(&body.receipt.assumptions)
    {
        check_text(text, "a receipt text")?;
    }
    Ok(())
}

/// Decode and re-derive a transported decision.
///
/// `expected_identity` is the decision identity the consumer retained
/// independently of the bytes; when present the artifact must reproduce it.
///
/// # Errors
/// Corruption, an unsupported version (refused before any field is read), an
/// item or text bound, an invalid contract or receipt, a refusal of the
/// re-evaluated decision, and a stored identity or action value that the
/// re-evaluation does not reproduce bit for bit.
pub fn consume_transported_decision(
    bytes: &[u8],
    expected_identity: Option<&str>,
) -> Result<ConsumedTransportedDecision, IoError> {
    let body: Body = decode_section(bytes, TRANSPORTED_DECISION_ARTIFACT_KIND, BODY_SECTION)?;
    check_bounds(&body)?;
    if body.coordinates.len() != body.mean_bits.len() {
        return Err(invalid("one mean per coordinate"));
    }
    let contract = contract_from_json(&body.contract_json)?;
    let coordinates: Vec<ScientificQuantity> = body
        .coordinates
        .iter()
        .cloned()
        .map(ScientificQuantity::try_from)
        .collect::<Result<_, _>>()
        .map_err(|why| invalid(format!("coordinate: {why}")))?;
    let mean = MeanSource {
        coordinates,
        means: body.mean_bits.iter().map(|b| f64::from_bits(*b)).collect(),
        provider_id: body.provider_id.clone(),
        snapshot_id: body.snapshot_id.clone(),
        causal_contract_id: body.causal_contract_id.clone(),
        rng_id: body.rng_id.clone(),
        estimate: body.estimate,
    };
    let response = TransportedResponse::from_stored_point(mean, body.receipt.clone())
        .map_err(|error| from_transported(&error))?;
    let weakest = SupportStatus::from_name(&body.weakest_support)
        .ok_or_else(|| invalid("unknown weakest support label"))?;
    let policy = SupportPolicy {
        unsupported: unsupported_from(&body.unsupported)?,
        weakest_support: weakest,
    };
    let decision = evaluate_transported_decision(
        &contract,
        &response,
        TransportedDemand::ExpectedUtility,
        &policy,
    )
    .map_err(|error| from_transported(&error))?;
    let replayed = outcome_wires(&decision);
    let same_outcomes = replayed.len() == body.outcomes.len()
        && replayed.iter().zip(&body.outcomes).all(|(a, b)| {
            a.action == b.action
                && a.expected_utility_bits == b.expected_utility_bits
                && a.value_bits == b.value_bits
        });
    if decision.identity != body.decision_identity || !same_outcomes {
        return Err(refused(
            antecedent_core::reason_code!("invalid_argument"),
            DETAIL_REPLAY,
            "the stored decision does not replay from its contract, carried value, receipt and policy",
        ));
    }
    if let Some(expected) = expected_identity {
        if expected != decision.identity {
            return Err(refused(
                antecedent_core::reason_code!("decision_contract_unsatisfied"),
                DETAIL_IDENTITY,
                "the decision identity differs from the consumer's retained identity",
            ));
        }
    }
    Ok(ConsumedTransportedDecision { decision, response })
}
