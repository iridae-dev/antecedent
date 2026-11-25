//! Structure claims a decision is evaluated across, as Python declares them: one claim wire
//! and its parsing into the supplied claim set the compiled decision programs take. Ordinary
//! decision contracts and decisions across a preference set read structures through it.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::claim_decision_programs::SuppliedClaimSet;
use antecedent::analysis::inverse_query::AtomEvidence;
use antecedent::contract::ExternalRefusal;
use antecedent::design::decision_adapters::{ClaimProbability, SuppliedClaim};
use antecedent::design::decision_robustness::{AtomSupport, InputSupport};
use pyo3::prelude::*;

use crate::bridge::external_refusal_json;
use crate::distribution_api::PyJointDistributionArtifact;

/// Most draw bytes the claims of one decision may copy between them.
const MAX_CLAIM_DRAW_BYTES: usize = 512 * 1024 * 1024;

/// A refusal of malformed claim JSON, at the adapt stage.
pub(crate) fn invalid_json(detail: &str, offending: Option<String>) -> String {
    external_refusal_json(&ExternalRefusal {
        code: antecedent::reason::reason_code!("invalid_argument"),
        stage: "adapt",
        detail: detail.to_owned(),
        offending,
        expected: None,
        supplied: None,
        capability: None,
        remedy: None,
    })
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ProbabilityWire {
    kind: String,
    #[serde(default)]
    value: Option<f64>,
    #[serde(default)]
    count: Option<u64>,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct InputSupportWire {
    action_id: String,
    input: usize,
    status: String,
}

/// The support of one claim.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SupportWire {
    overall: String,
    #[serde(default)]
    per_input: Vec<InputSupportWire>,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ClaimWire {
    id: String,
    #[serde(default)]
    probability: Option<ProbabilityWire>,
    status: String,
    #[serde(default)]
    reason: Option<String>,
    #[serde(default)]
    support: Option<SupportWire>,
}

fn support_status(name: &str) -> Result<antecedent::query::SupportStatus, String> {
    antecedent::query::SupportStatus::from_name(name).ok_or_else(|| {
        invalid_json(
            antecedent::reason::reason_details::decision_adapters::UNKNOWN_SUPPORT_STATUS,
            Some(name.into()),
        )
    })
}

/// The support a claim declares; unassessed when it declares none.
pub(crate) fn support_from(wire: Option<SupportWire>) -> Result<AtomSupport, String> {
    let Some(wire) = wire else {
        return Ok(AtomSupport::unassessed());
    };
    Ok(AtomSupport {
        overall: support_status(&wire.overall)?,
        per_input: wire
            .per_input
            .into_iter()
            .map(|p| {
                Ok(InputSupport {
                    action_id: p.action_id,
                    input: p.input,
                    status: support_status(&p.status)?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?,
    })
}

fn probability_from(wire: Option<ProbabilityWire>, id: &str) -> Result<ClaimProbability, String> {
    Ok(match wire {
        None => ClaimProbability::Unspecified,
        Some(p) => match (p.kind.as_str(), p.value, p.count) {
            ("unspecified", None, None) => ClaimProbability::Unspecified,
            ("genuine", Some(value), None) => ClaimProbability::Genuine(value),
            ("completion_count", None, Some(count)) => ClaimProbability::CompletionCount(count),
            _ => {
                return Err(invalid_json(
                    antecedent::reason::reason_details::decision_adapters::INVALID_PROBABILITY,
                    Some(id.to_owned()),
                ));
            }
        },
    })
}

fn claim_from(
    wire: ClaimWire,
    source: Option<&PyJointDistributionArtifact>,
) -> Result<SuppliedClaim, String> {
    let bad = |slot: &str| invalid_json(slot, Some(wire.id.clone()));
    let evidence = match (wire.status.as_str(), source) {
        ("evaluated", Some(source)) => AtomEvidence::Evaluated(Box::new(source.inner().clone())),
        ("evaluated", None) => {
            return Err(bad(
                antecedent::reason::reason_details::decision_adapters::EVALUATED_NEEDS_DRAWS,
            ));
        }
        ("unidentified", _) => AtomEvidence::Unidentified,
        ("unevaluated", _) => {
            match wire.reason.as_deref() {
                Some(reason) if !reason.trim().is_empty() => {
                    AtomEvidence::Unevaluated(reason.to_owned())
                }
                _ => return Err(bad(
                    antecedent::reason::reason_details::decision_adapters::UNEVALUATED_NEEDS_REASON,
                )),
            }
        }
        _ => {
            return Err(bad(
                antecedent::reason::reason_details::decision_adapters::UNKNOWN_CLAIM_STATUS,
            ));
        }
    };
    let probability = probability_from(wire.probability, &wire.id)?;
    Ok(SuppliedClaim { id: wire.id, probability, evidence, support: support_from(wire.support)? })
}

/// Parse the claims of `kind`, one per source, into the supplied claim set a compiled
/// decision program takes. `subject` names the claims in the refusal of draws too large to
/// copy. The error is a refusal JSON.
pub(crate) fn supplied_claims(
    kind: &str,
    claims_json: &str,
    sources: &[Option<PyRef<'_, PyJointDistributionArtifact>>],
    subject: &str,
) -> Result<SuppliedClaimSet, String> {
    let wires: Vec<ClaimWire> = serde_json::from_str(claims_json).map_err(|e| {
        invalid_json(
            antecedent::reason::reason_details::decision_adapters::INVALID_CLAIMS,
            Some(e.to_string()),
        )
    })?;
    if wires.len() != sources.len() {
        return Err(invalid_json(
            antecedent::reason::reason_details::decision_adapters::CLAIMS_SOURCES_MISMATCH,
            None,
        ));
    }
    // Each evaluated claim owns a copy of its draws: bound the total before cloning.
    let cloned_bytes = wires
        .iter()
        .zip(sources)
        .filter(|(wire, _)| wire.status == "evaluated")
        .filter_map(|(_, source)| source.as_deref())
        .try_fold(0usize, |total, source| {
            total.checked_add(source.inner().draws().len().checked_mul(size_of::<f64>())?)
        });
    if cloned_bytes.is_none_or(|bytes| bytes > MAX_CLAIM_DRAW_BYTES) {
        return Err(invalid_json(
            antecedent::reason::reason_details::decision_adapters::CLAIM_DRAWS_TOO_LARGE,
            Some(format!("the {subject} copy more than {MAX_CLAIM_DRAW_BYTES} bytes of draws")),
        ));
    }
    let claims = wires
        .into_iter()
        .zip(sources)
        .map(|(wire, source)| claim_from(wire, source.as_deref()))
        .collect::<Result<Vec<_>, String>>()?;
    Ok(match kind {
        "point" => {
            let Ok([claim]) = <[SuppliedClaim; 1]>::try_from(claims) else {
                return Err(invalid_json(
                    antecedent::reason::reason_details::decision_adapters::POINT_CLAIM_IS_SINGLE,
                    None,
                ));
            };
            SuppliedClaimSet::Point {
                id: claim.id,
                evidence: claim.evidence,
                support: claim.support,
            }
        }
        "graph_dependent" => SuppliedClaimSet::GraphDependent(claims),
        "weighted_graph_posterior" => SuppliedClaimSet::WeightedGraphAtoms(claims),
        "finite_scenarios" => SuppliedClaimSet::FiniteScenarios(claims),
        other => {
            return Err(invalid_json(
                antecedent::reason::reason_details::decision_adapters::UNKNOWN_CLAIM_KIND,
                Some(other.into()),
            ));
        }
    })
}
