//! Bounded Python bridge for 2.3 decision contracts.
//!
//! Python builds the contract declaration; the identity, validation, evaluation,
//! artifacts and refusals are Rust's. A refusal comes back as structured JSON
//! for the Python layer to raise as its own exception type.

use antecedent_design::decision_adapters::{
    AdaptedClaims, AdaptedDecision, ClaimProbability, IdentifiedSetDecision, IdentifiedUtility,
    IdentifiedVerdict, SuppliedClaim, adapt_finite_scenarios, adapt_graph_dependent_claims,
    adapt_point_claim, adapt_weighted_graph_atoms, evaluate_adapted, evaluate_identified_sets,
    utility_interval,
};
use antecedent_design::decision_artifact::{
    DecisionContractArtifact, DecisionResultArtifact, MAX_DECISION_ARTIFACT_BYTES,
    contract_from_json, contract_from_json_refusal, contract_to_json, mean_result_to_json,
    result_to_json, source_digest,
};
use antecedent_design::decision_contract::{AdmissibleDecisionContract, StructuralPolicy};
use antecedent_design::decision_eval::{
    DecisionEvalError, MeanSource, evaluate_contract, evaluate_contract_on_means,
};
use antecedent_design::decision_robust_artifact::{
    AdmissibleContractArtifact, ExternalCallbackReceipt, ExternalTrustLimit, RobustResultArtifact,
    admissible_contract_from_json, admissible_contract_from_json_refusal,
    admissible_contract_to_json, atom_digests_of,
};
use antecedent_design::decision_robustness::{AtomSupport, InputSupport};
use antecedent_design::decision_structural::{
    AtomEvidence, AtomStatus, StructuralDecisionResult, StructuralVerdict,
};
use antecedent_io::error::IoError;
use antecedent_io::quantity_wire::ScientificQuantityWire;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use antecedent_core::{ExternalRefusal, ScientificQuantity};

use crate::CausalSerializationError;
use crate::distribution_api::PyJointDistributionArtifact;

fn serialization(error: impl std::fmt::Display) -> PyErr {
    CausalSerializationError::new_err(error.to_string())
}

fn refusal_json(value: &ExternalRefusal) -> String {
    serde_json::json!({
        "code": value.code,
        "stage": value.stage,
        "detail": value.detail,
        "offending": value.offending,
        "expected": value.expected,
        "supplied": value.supplied,
        "remedy": value.remedy,
    })
    .to_string()
}

fn check_size(json: &str) -> PyResult<()> {
    if json.len() > MAX_DECISION_ARTIFACT_BYTES {
        return Err(PyValueError::new_err("decision declaration is too large"));
    }
    Ok(())
}

fn io_refusal(error: &IoError, stage: &'static str) -> String {
    let (code, detail, offending) = match error {
        IoError::Refused { code, message } => {
            (*code, message.split(':').next().unwrap_or("decision_artifact").to_owned(), None)
        }
        other => (
            antecedent_core::reason_code!("decision_contract_unsatisfied"),
            "decision_contract.invalid_declaration".to_owned(),
            Some(other.to_string()),
        ),
    };
    refusal_json(&ExternalRefusal {
        code,
        stage,
        detail,
        offending,
        expected: None,
        supplied: None,
        capability: None,
        remedy: None,
    })
}

fn eval_refusal(error: &DecisionEvalError) -> String {
    refusal_json(&error.to_refusal())
}

/// Validate a contract declaration; returns its normalized JSON (with the
/// recomputed identity) or a structured refusal.
#[pyfunction]
fn decision_contract_normalize(contract_json: &str) -> PyResult<(Option<String>, Option<String>)> {
    check_size(contract_json)?;
    match contract_from_json_refusal(contract_json) {
        Ok(contract) => Ok((Some(contract_to_json(&contract).map_err(serialization)?), None)),
        Err(refusal) => Ok((None, Some(refusal_json(&refusal)))),
    }
}

/// Evaluate a contract on aligned joint draws; returns the result JSON or a refusal.
#[pyfunction]
fn evaluate_decision(
    contract_json: &str,
    source: &PyJointDistributionArtifact,
) -> PyResult<(Option<String>, Option<String>)> {
    check_size(contract_json)?;
    let contract = match contract_from_json_refusal(contract_json) {
        Ok(contract) => contract,
        Err(refusal) => return Ok((None, Some(refusal_json(&refusal)))),
    };
    match evaluate_contract(&contract, source.inner()) {
        Ok(result) => {
            Ok((Some(result_to_json(&result, source.inner()).map_err(serialization)?), None))
        }
        Err(error) => Ok((None, Some(eval_refusal(&error)))),
    }
}

/// Evaluate a contract on a mean-only source (one mean per coordinate, such as an
/// external response grid); returns the result JSON or a refusal. The result's
/// `lineage` is empty: the caller supplies the derivation of the means.
#[pyfunction]
fn evaluate_decision_means(
    contract_json: &str,
    coordinates_json: &str,
    means: Vec<f64>,
    provider_id: &str,
    snapshot_id: &str,
    causal_contract_id: &str,
) -> PyResult<(Option<String>, Option<String>)> {
    check_size(contract_json)?;
    check_size(coordinates_json)?;
    let contract = match contract_from_json_refusal(contract_json) {
        Ok(contract) => contract,
        Err(refusal) => return Ok((None, Some(refusal_json(&refusal)))),
    };
    let wires: Vec<ScientificQuantityWire> = serde_json::from_str(coordinates_json)
        .map_err(|e| PyValueError::new_err(format!("invalid coordinates: {e}")))?;
    let coordinates = wires
        .into_iter()
        .map(|wire| {
            ScientificQuantity::try_from(wire).map_err(|e| PyValueError::new_err(String::from(e)))
        })
        .collect::<PyResult<Vec<_>>>()?;
    let source = MeanSource {
        coordinates,
        means,
        provider_id: provider_id.to_owned(),
        snapshot_id: snapshot_id.to_owned(),
        causal_contract_id: causal_contract_id.to_owned(),
        rng_id: "none:mean_grid".to_owned(),
    };
    match evaluate_contract_on_means(&contract, &source) {
        Ok(result) => {
            Ok((Some(mean_result_to_json(&result, &source).map_err(serialization)?), None))
        }
        Err(error) => Ok((None, Some(eval_refusal(&error)))),
    }
}

/// Export a contract artifact.
#[pyfunction]
fn export_decision_contract<'py>(
    py: Python<'py>,
    contract_json: &str,
    artifact_id: &str,
) -> PyResult<Bound<'py, PyBytes>> {
    check_size(contract_json)?;
    let contract = contract_from_json(contract_json).map_err(serialization)?;
    let artifact = DecisionContractArtifact::new(contract).map_err(serialization)?;
    Ok(PyBytes::new(py, &artifact.to_bytes(artifact_id).map_err(serialization)?))
}

/// Load a contract artifact under the consumer's retained identity.
#[pyfunction]
fn load_decision_contract(data: &[u8], expected_identity: &str) -> PyResult<String> {
    let artifact =
        DecisionContractArtifact::from_bytes(data, expected_identity).map_err(serialization)?;
    contract_to_json(artifact.contract()).map_err(serialization)
}

/// Evaluate a contract on a source and export the bound result artifact.
#[pyfunction]
fn export_decision_result<'py>(
    py: Python<'py>,
    contract_json: &str,
    source: &PyJointDistributionArtifact,
    artifact_id: &str,
) -> PyResult<Bound<'py, PyBytes>> {
    check_size(contract_json)?;
    let contract = contract_from_json(contract_json).map_err(serialization)?;
    let result = evaluate_contract(&contract, source.inner()).map_err(serialization_eval)?;
    let artifact = DecisionResultArtifact::new(result, source.inner());
    Ok(PyBytes::new(py, &artifact.to_bytes(artifact_id).map_err(serialization)?))
}

fn serialization_eval(error: DecisionEvalError) -> PyErr {
    serialization(format!("{error:?}"))
}

/// Replay a stored result: load it under this contract and source, recompute,
/// and require an exact match. Returns the result JSON, or a refusal.
#[pyfunction]
fn replay_decision_result(
    data: &[u8],
    contract_json: &str,
    source: &PyJointDistributionArtifact,
) -> PyResult<(Option<String>, Option<String>)> {
    check_size(contract_json)?;
    let contract = match contract_from_json_refusal(contract_json) {
        Ok(contract) => contract,
        Err(refusal) => return Ok((None, Some(refusal_json(&refusal)))),
    };
    let identity = contract.identity().map_err(|e| serialization(format!("{e:?}")))?;
    let outcome =
        DecisionResultArtifact::from_bytes(data, &identity, &source_digest(source.inner()))
            .and_then(|artifact| artifact.replay(&contract, source.inner()).map(|()| artifact));
    match outcome {
        Ok(artifact) => Ok((
            Some(result_to_json(artifact.result(), source.inner()).map_err(serialization)?),
            None,
        )),
        Err(error) => Ok((None, Some(io_refusal(&error, "replay")))),
    }
}

/// Validate a derivation chain given as `[id, stage, [parent ids]]` rows and
/// return every link with its Merkle digest and its parents' digests, as JSON.
///
/// Python cannot hash, so a chain it extends (a decision over an external claim)
/// gets its digests here.
#[pyfunction]
fn composition_lineage(links_json: &str) -> PyResult<String> {
    check_size(links_json)?;
    let rows: Vec<(String, String, Vec<String>)> =
        serde_json::from_str(links_json).map_err(serialization)?;
    let parents: Vec<Vec<&str>> =
        rows.iter().map(|(_, _, parents)| parents.iter().map(String::as_str).collect()).collect();
    let borrowed: Vec<(&str, &str, &[&str])> = rows
        .iter()
        .zip(&parents)
        .map(|((id, stage, _), parents)| (id.as_str(), stage.as_str(), parents.as_slice()))
        .collect();
    let wire =
        antecedent_io::external_claim_artifact::lineage_wire(&borrowed).map_err(serialization)?;
    serde_json::to_string(&wire).map_err(serialization)
}

/// BLAKE3 digest of a source's aligned draws, for retaining alongside a result.
#[pyfunction]
fn decision_source_digest(source: &PyJointDistributionArtifact) -> String {
    source_digest(source.inner())
}

// ---------------------------------------------------------------------------
// Admissible contracts, robustness, adapters and identified sets (2.3 B0).
// ---------------------------------------------------------------------------

/// Refusal JSON for a malformed bridge argument that never reached Rust validation.
fn invalid_json(detail: &str, offending: Option<String>) -> String {
    refusal_json(&ExternalRefusal {
        code: antecedent_core::reason_code!("invalid_argument"),
        stage: "adapt",
        detail: detail.to_owned(),
        offending,
        expected: None,
        supplied: None,
        capability: None,
        remedy: None,
    })
}

fn policy_name(policy: StructuralPolicy) -> &'static str {
    match policy {
        StructuralPolicy::RequireInvariantBestAction => "require_invariant_best_action",
        StructuralPolicy::Maximin => "maximin",
        StructuralPolicy::BayesOverStructures => "bayes_over_structures",
        StructuralPolicy::ReportOnly => "report_only",
    }
}

/// Validate an admissible declaration `{"contract": ..., "rules": ...}`; returns its
/// normalized JSON (with the recomputed identity) or a structured refusal.
#[pyfunction]
fn admissible_contract_normalize(
    declaration_json: &str,
) -> PyResult<(Option<String>, Option<String>)> {
    check_size(declaration_json)?;
    match admissible_contract_from_json_refusal(declaration_json) {
        Ok(contract) => {
            Ok((Some(admissible_contract_to_json(&contract).map_err(serialization)?), None))
        }
        Err(refusal) => Ok((None, Some(refusal_json(&refusal)))),
    }
}

/// Export an admissible contract artifact (base contract plus every rule).
#[pyfunction]
fn export_admissible_contract<'py>(
    py: Python<'py>,
    declaration_json: &str,
    artifact_id: &str,
) -> PyResult<Bound<'py, PyBytes>> {
    check_size(declaration_json)?;
    let contract = admissible_contract_from_json(declaration_json).map_err(serialization)?;
    let artifact = AdmissibleContractArtifact::new(contract).map_err(serialization)?;
    Ok(PyBytes::new(py, &artifact.to_bytes(artifact_id).map_err(serialization)?))
}

/// Load an admissible contract artifact under the consumer's retained identity.
#[pyfunction]
fn load_admissible_contract(data: &[u8], expected_identity: &str) -> PyResult<String> {
    let artifact =
        AdmissibleContractArtifact::from_bytes(data, expected_identity).map_err(serialization)?;
    admissible_contract_to_json(artifact.contract()).map_err(serialization)
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

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct SupportWire {
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

fn parse_status(name: &str) -> Result<antecedent_core::SupportStatus, String> {
    antecedent_core::SupportStatus::from_name(name)
        .ok_or_else(|| invalid_json("decision_adapters.unknown_support_status", Some(name.into())))
}

fn support_from(wire: Option<SupportWire>) -> Result<AtomSupport, String> {
    let Some(wire) = wire else {
        return Ok(AtomSupport::unassessed());
    };
    Ok(AtomSupport {
        overall: parse_status(&wire.overall)?,
        per_input: wire
            .per_input
            .into_iter()
            .map(|p| {
                Ok(InputSupport {
                    action_id: p.action_id,
                    input: p.input,
                    status: parse_status(&p.status)?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?,
    })
}

fn probability_from(wire: Option<ProbabilityWire>, id: &str) -> Result<ClaimProbability, String> {
    let bad = || invalid_json("decision_adapters.invalid_probability", Some(id.to_owned()));
    Ok(match wire {
        None => ClaimProbability::Unspecified,
        Some(p) => match (p.kind.as_str(), p.value, p.count) {
            ("unspecified", None, None) => ClaimProbability::Unspecified,
            ("genuine", Some(value), None) => ClaimProbability::Genuine(value),
            ("completion_count", None, Some(count)) => ClaimProbability::CompletionCount(count),
            _ => return Err(bad()),
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
        ("unidentified", _) => AtomEvidence::Unidentified,
        ("unevaluated", _) => match wire.reason.as_deref() {
            Some(reason) if !reason.trim().is_empty() => {
                AtomEvidence::Unevaluated(reason.to_owned())
            }
            _ => return Err(bad("decision_adapters.unevaluated_needs_reason")),
        },
        ("evaluated", None) => return Err(bad("decision_adapters.evaluated_needs_draws")),
        _ => return Err(bad("decision_adapters.unknown_claim_status")),
    };
    let probability = probability_from(wire.probability, &wire.id)?;
    Ok(SuppliedClaim { id: wire.id, probability, evidence, support: support_from(wire.support)? })
}

struct Prepared {
    contract: AdmissibleDecisionContract,
    adapted: AdaptedClaims,
}

/// Parse the declaration and claims and adapt them under the contract's policy.
/// The error is a refusal JSON.
fn prepare(
    declaration_json: &str,
    kind: &str,
    claims_json: &str,
    sources: &[Option<PyRef<'_, PyJointDistributionArtifact>>],
) -> Result<Prepared, String> {
    let contract = admissible_contract_from_json_refusal(declaration_json)
        .map_err(|refusal| refusal_json(&refusal))?;
    let wires: Vec<ClaimWire> = serde_json::from_str(claims_json)
        .map_err(|e| invalid_json("decision_adapters.invalid_claims", Some(e.to_string())))?;
    if wires.len() != sources.len() {
        return Err(invalid_json("decision_adapters.claims_sources_mismatch", None));
    }
    let claims = wires
        .into_iter()
        .zip(sources)
        .map(|(wire, source)| claim_from(wire, source.as_deref()))
        .collect::<Result<Vec<_>, String>>()?;
    let policy = contract.contract.structural_policy;
    let adapted = match kind {
        "point" => {
            let Ok([claim]) = <[SuppliedClaim; 1]>::try_from(claims) else {
                return Err(invalid_json("decision_adapters.point_claim_is_single", None));
            };
            adapt_point_claim(&claim.id, claim.evidence, claim.support, policy)
        }
        "graph_dependent" => adapt_graph_dependent_claims(claims, policy),
        "weighted_graph_posterior" => adapt_weighted_graph_atoms(claims, policy),
        "finite_scenarios" => adapt_finite_scenarios(claims, policy),
        other => {
            return Err(invalid_json("decision_adapters.unknown_claim_kind", Some(other.into())));
        }
    }
    .map_err(|e| refusal_json(&e.to_refusal()))?;
    Ok(Prepared { contract, adapted })
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct TrustJson {
    kind: String,
    #[serde(default)]
    attestor: Option<String>,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ReceiptJson {
    atom_id: String,
    provider_id: String,
    snapshot_id: String,
    request_fingerprint: String,
    attested_value: f64,
    trust: TrustJson,
}

fn receipts_from(json: &str) -> Result<Vec<ExternalCallbackReceipt>, String> {
    let wires: Vec<ReceiptJson> = serde_json::from_str(json)
        .map_err(|e| invalid_json("decision_robust.invalid_receipts", Some(e.to_string())))?;
    wires
        .into_iter()
        .map(|w| {
            let trust = match (w.trust.kind.as_str(), w.trust.attestor) {
                ("externally_attested", Some(attestor)) => {
                    ExternalTrustLimit::ExternallyAttested { attestor }
                }
                ("verified_extension", None) => ExternalTrustLimit::VerifiedExtension,
                (other, _) => {
                    return Err(invalid_json(
                        "decision_robust.external_trust_never_native",
                        Some(other.to_owned()),
                    ));
                }
            };
            Ok(ExternalCallbackReceipt {
                atom_id: w.atom_id,
                provider_id: w.provider_id,
                snapshot_id: w.snapshot_id,
                request_fingerprint: w.request_fingerprint,
                attested_value: w.attested_value,
                trust,
            })
        })
        .collect()
}

fn structural_verdict_json(verdict: &StructuralVerdict) -> serde_json::Value {
    use serde_json::json;
    match verdict {
        StructuralVerdict::InvariantBest(id) => json!({"kind": "invariant_best", "action": id}),
        StructuralVerdict::NoInvariantBest(leaders) => {
            json!({"kind": "no_invariant_best", "leaders": leaders})
        }
        StructuralVerdict::WorstCaseChoice(id) => {
            json!({"kind": "worst_case_choice", "action": id})
        }
        StructuralVerdict::BayesChoice { action, evaluated_mass } => {
            json!({"kind": "bayes_choice", "action": action, "evaluated_mass": evaluated_mass})
        }
        StructuralVerdict::ReportOnly => json!({"kind": "report_only"}),
        StructuralVerdict::InsufficientScience(why) => {
            json!({"kind": "insufficient_science", "reason": why})
        }
        StructuralVerdict::NoAdmissibleAction => json!({"kind": "no_admissible_action"}),
    }
}

fn structural_json(result: &StructuralDecisionResult) -> serde_json::Value {
    use serde_json::json;
    let atoms: Vec<serde_json::Value> = result
        .atoms
        .iter()
        .map(|atom| {
            let (status, reason, actions) = match &atom.status {
                AtomStatus::Evaluated(inner) => (
                    "evaluated",
                    None,
                    Some(
                        inner
                            .actions
                            .iter()
                            .map(|o| {
                                json!({"id": o.id, "admissible": o.admissible, "value": o.value})
                            })
                            .collect::<Vec<_>>(),
                    ),
                ),
                AtomStatus::Unidentified => ("unidentified", None, None),
                AtomStatus::Unevaluated(why) => ("unevaluated", Some(why.clone()), None),
            };
            json!({
                "id": atom.id,
                "probability": atom.probability,
                "status": status,
                "reason": reason,
                "actions": actions,
            })
        })
        .collect();
    let actions: Vec<serde_json::Value> = result
        .actions
        .iter()
        .map(|a| {
            json!({
                "id": a.id,
                "per_atom": a.per_atom,
                "range": a.range,
                "weighted_value": a.weighted_value,
                "mass_where_best": a.mass_where_best,
                "excluded_in": a.excluded_in,
            })
        })
        .collect();
    json!({
        "contract_identity": result.contract_identity,
        "policy": policy_name(result.policy),
        "atoms": atoms,
        "actions": actions,
        "unidentified_mass": result.unidentified_mass,
        "unevaluated_mass": result.unevaluated_mass,
        "evaluated_mass": result.evaluated_mass,
        "verdict": structural_verdict_json(&result.verdict),
    })
}

fn seal(
    prepared: &Prepared,
    decision: &AdaptedDecision,
    receipts_json: &str,
) -> Result<RobustResultArtifact, String> {
    let receipts = receipts_from(receipts_json)?;
    RobustResultArtifact::new(
        &prepared.contract,
        decision.robust.clone(),
        &prepared.adapted.profile,
        &prepared.adapted.atoms,
        receipts,
    )
    .map_err(|e| io_refusal(&e, "seal"))
}

fn evaluate_prepared(prepared: &Prepared) -> Result<AdaptedDecision, String> {
    evaluate_adapted(&prepared.contract, &prepared.adapted)
        .map_err(|e| refusal_json(&e.to_refusal()))
}

fn robust_value(artifact: &RobustResultArtifact) -> Result<serde_json::Value, String> {
    let json = artifact.to_json().map_err(|e| io_refusal(&e, "seal"))?;
    serde_json::from_str(&json)
        .map_err(|e| invalid_json("decision_robust.json", Some(e.to_string())))
}

/// Evaluate claims under an admissible contract and report the structural result
/// and robustness verdict. `kind` is `point`, `graph_dependent`,
/// `weighted_graph_posterior` or `finite_scenarios`; `sources` aligns with the
/// claims (`None` for a claim without draws). Returns the result JSON or a refusal.
#[pyfunction]
fn evaluate_robust_decision(
    declaration_json: &str,
    kind: &str,
    claims_json: &str,
    sources: Vec<Option<PyRef<'_, PyJointDistributionArtifact>>>,
    receipts_json: &str,
) -> PyResult<(Option<String>, Option<String>)> {
    check_size(declaration_json)?;
    check_size(claims_json)?;
    check_size(receipts_json)?;
    let run = || -> Result<String, String> {
        let prepared = prepare(declaration_json, kind, claims_json, &sources)?;
        let decision = evaluate_prepared(&prepared)?;
        let artifact = seal(&prepared, &decision, receipts_json)?;
        Ok(serde_json::json!({
            "kind": kind,
            "structural": structural_json(&decision.structural),
            "robust": robust_value(&artifact)?,
            "completion_counts": decision.completion_counts,
        })
        .to_string())
    };
    Ok(match run() {
        Ok(json) => (Some(json), None),
        Err(refusal) => (None, Some(refusal)),
    })
}

/// Evaluate and export the robust result artifact. Returns the bytes or a refusal.
#[pyfunction]
fn export_robust_decision<'py>(
    py: Python<'py>,
    declaration_json: &str,
    kind: &str,
    claims_json: &str,
    sources: Vec<Option<PyRef<'_, PyJointDistributionArtifact>>>,
    receipts_json: &str,
    artifact_id: &str,
) -> PyResult<(Option<Bound<'py, PyBytes>>, Option<String>)> {
    check_size(declaration_json)?;
    check_size(claims_json)?;
    check_size(receipts_json)?;
    let run = || -> Result<Vec<u8>, String> {
        let prepared = prepare(declaration_json, kind, claims_json, &sources)?;
        let decision = evaluate_prepared(&prepared)?;
        let artifact = seal(&prepared, &decision, receipts_json)?;
        artifact.to_bytes(artifact_id).map_err(|e| io_refusal(&e, "export"))
    };
    Ok(match run() {
        Ok(bytes) => (Some(PyBytes::new(py, &bytes)), None),
        Err(refusal) => (None, Some(refusal)),
    })
}

/// Replay a stored robust result: load it under this contract and these claims,
/// recompute, and require an exact match. Returns `{robust, replay}` JSON or a
/// refusal. The replay says whether the result is natively verified.
#[pyfunction]
fn replay_robust_decision(
    data: &[u8],
    declaration_json: &str,
    kind: &str,
    claims_json: &str,
    sources: Vec<Option<PyRef<'_, PyJointDistributionArtifact>>>,
) -> PyResult<(Option<String>, Option<String>)> {
    check_size(declaration_json)?;
    check_size(claims_json)?;
    let run = || -> Result<String, String> {
        let prepared = prepare(declaration_json, kind, claims_json, &sources)?;
        let identity = prepared.contract.identity().map_err(|e| refusal_json(&e.to_refusal()))?;
        let artifact = RobustResultArtifact::from_bytes(
            data,
            &identity,
            &atom_digests_of(&prepared.adapted.atoms),
        )
        .map_err(|e| io_refusal(&e, "replay"))?;
        let receipt = artifact
            .replay(&prepared.contract, &prepared.adapted.atoms, &prepared.adapted.profile)
            .map_err(|e| io_refusal(&e, "replay"))?;
        Ok(serde_json::json!({
            "robust": robust_value(&artifact)?,
            "replay": {
                "recomputed": receipt.recomputed,
                "native_verified": receipt.native_verified,
                "external_atoms": receipt.external_atoms,
            },
        })
        .to_string())
    };
    Ok(match run() {
        Ok(json) => (Some(json), None),
        Err(refusal) => (None, Some(refusal)),
    })
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct UtilityWire {
    action_id: String,
    lower: f64,
    upper: f64,
    #[serde(default)]
    hard_exclusions: Vec<String>,
    #[serde(default)]
    support: Option<SupportWire>,
}

fn identified_verdict_json(verdict: &IdentifiedVerdict) -> serde_json::Value {
    use serde_json::json;
    match verdict {
        IdentifiedVerdict::NecessarilyBest(id) => {
            json!({"kind": "necessarily_best", "action": id})
        }
        IdentifiedVerdict::NoNecessarilyBest { possibly_optimal } => {
            json!({"kind": "no_necessarily_best", "actions": possibly_optimal})
        }
        IdentifiedVerdict::WorstCaseChoice(id) => {
            json!({"kind": "worst_case_choice", "action": id})
        }
        IdentifiedVerdict::MinimaxRegretChoice(id) => {
            json!({"kind": "minimax_regret_choice", "action": id})
        }
        IdentifiedVerdict::Tied(ids) => json!({"kind": "tied", "actions": ids}),
        IdentifiedVerdict::ReportOnly => json!({"kind": "report_only"}),
        IdentifiedVerdict::UnsupportedExtrapolation => json!({"kind": "unsupported_extrapolation"}),
        IdentifiedVerdict::InsufficientClaims(why) => {
            json!({"kind": "insufficient_claims", "reason": why})
        }
        IdentifiedVerdict::NoAdmissibleAction => json!({"kind": "no_admissible_action"}),
    }
}

fn identified_json(decision: &IdentifiedSetDecision) -> serde_json::Value {
    use serde_json::json;
    let actions: Vec<serde_json::Value> = decision
        .actions
        .iter()
        .map(|a| {
            let shortfalls: Vec<serde_json::Value> = a
                .support_shortfalls
                .iter()
                .map(|(input, status)| json!({"input": input, "status": status.as_str()}))
                .collect();
            json!({
                "id": a.id,
                "utility": [a.utility.0, a.utility.1],
                "hard_exclusions": a.hard_exclusions,
                "declared_exclusion": a.declared_exclusion,
                "support_shortfalls": shortfalls,
                "eligible": a.eligible,
                "dominated_by": a.dominated_by,
                "possibly_optimal": a.possibly_optimal,
                "necessarily_optimal": a.necessarily_optimal,
                "max_regret": a.max_regret,
            })
        })
        .collect();
    json!({
        "contract_identity": decision.contract_identity,
        "policy": policy_name(decision.policy),
        "actions": actions,
        "lower_leader": decision.lower_leader,
        "upper_leader": decision.upper_leader,
        "conflicting_leaders": decision.conflicting_leaders,
        "verdict": identified_verdict_json(&decision.verdict),
    })
}

/// Decide over per-action identified sets of utility (partial identification).
/// Returns the decision JSON or a refusal.
#[pyfunction]
fn evaluate_identified_set_decision(
    declaration_json: &str,
    utilities_json: &str,
) -> PyResult<(Option<String>, Option<String>)> {
    check_size(declaration_json)?;
    check_size(utilities_json)?;
    let run = || -> Result<String, String> {
        let contract = admissible_contract_from_json_refusal(declaration_json)
            .map_err(|refusal| refusal_json(&refusal))?;
        let wires: Vec<UtilityWire> = serde_json::from_str(utilities_json).map_err(|e| {
            invalid_json("decision_adapters.invalid_utilities", Some(e.to_string()))
        })?;
        let utilities = wires
            .into_iter()
            .map(|w| {
                Ok(IdentifiedUtility {
                    action_id: w.action_id,
                    utility: antecedent_core::IdentifiedSet { lower: w.lower, upper: w.upper },
                    hard_exclusions: w.hard_exclusions,
                    support: support_from(w.support)?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let decision = evaluate_identified_sets(&contract, &utilities)
            .map_err(|e| refusal_json(&e.to_refusal()))?;
        Ok(identified_json(&decision).to_string())
    };
    Ok(match run() {
        Ok(json) => (Some(json), None),
        Err(refusal) => (None, Some(refusal)),
    })
}

/// An optional `(lower, upper)` interval paired with an optional refusal document.
type IntervalOrRefusal = (Option<(f64, f64)>, Option<String>);

/// Interval enclosure of an action's utility from one interval per input.
/// Returns `(lower, upper)` or a refusal.
#[pyfunction]
fn identified_utility_interval(
    declaration_json: &str,
    action_id: &str,
    intervals: Vec<(f64, f64)>,
) -> PyResult<IntervalOrRefusal> {
    check_size(declaration_json)?;
    let contract = match admissible_contract_from_json_refusal(declaration_json) {
        Ok(contract) => contract,
        Err(refusal) => return Ok((None, Some(refusal_json(&refusal)))),
    };
    let inputs: Vec<antecedent_core::IdentifiedSet<f64>> = intervals
        .into_iter()
        .map(|(lower, upper)| antecedent_core::IdentifiedSet { lower, upper })
        .collect();
    Ok(match utility_interval(&contract.contract, action_id, &inputs) {
        Ok(set) => (Some((set.lower, set.upper)), None),
        Err(error) => (None, Some(refusal_json(&error.to_refusal()))),
    })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(admissible_contract_normalize, m)?)?;
    m.add_function(wrap_pyfunction!(export_admissible_contract, m)?)?;
    m.add_function(wrap_pyfunction!(load_admissible_contract, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_robust_decision, m)?)?;
    m.add_function(wrap_pyfunction!(export_robust_decision, m)?)?;
    m.add_function(wrap_pyfunction!(replay_robust_decision, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_identified_set_decision, m)?)?;
    m.add_function(wrap_pyfunction!(identified_utility_interval, m)?)?;
    m.add_function(wrap_pyfunction!(decision_contract_normalize, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_decision, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_decision_means, m)?)?;
    m.add_function(wrap_pyfunction!(export_decision_contract, m)?)?;
    m.add_function(wrap_pyfunction!(load_decision_contract, m)?)?;
    m.add_function(wrap_pyfunction!(export_decision_result, m)?)?;
    m.add_function(wrap_pyfunction!(replay_decision_result, m)?)?;
    m.add_function(wrap_pyfunction!(composition_lineage, m)?)?;
    m.add_function(wrap_pyfunction!(decision_source_digest, m)?)
}
