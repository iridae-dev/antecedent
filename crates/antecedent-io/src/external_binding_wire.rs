//! JSON wire for binding a typed external response, shared by every surface.
//!
//! Rust owns the identities, checks and refusal rules; a host language only
//! builds these declarations. Binding is stateless: a verified provider sends
//! its independent probe values and verification is recomputed here, so no
//! opaque receipt crosses the boundary.

// Refusals are the cold path of a once-per-binding call, so the large `Err`
// variant is not worth boxing at every `?`.
#![allow(clippy::result_large_err)]

use antecedent_core::{
    BoundExternalClaim, CheckedCausalContract, CheckedEquivalence, EquivalenceScope,
    ExternalBindingError, ExternalContractError, ExternalProgramClaim, ExternalResponse,
    ExternalResult, ExternalResultHeader, ExternalScientificObject, ExternalTrustState,
    ExternalUncertaintyMeaning, IdentificationStatus, LawProviderContract, ProgramBinding,
    ProviderObjectIdentity, ScientificQuantity, SupportStatus, VerificationProbe,
    bind_external_result, bind_external_result_to_program, capability_from_name, probe_from_name,
    verify_external_object,
};
use serde::{Deserialize, Serialize};

use crate::external_claim_artifact::ExternalClaimArtifact;
use crate::quantity_wire::{DistributionMeaningWire, ScientificQuantityWire};

/// Maximum accepted JSON declaration bytes.
pub const MAX_BINDING_WIRE_BYTES: usize = 4 * 1024 * 1024;

/// `P(Y | T = v)` read as `P(Y | do(T = v))` for each listed value.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConditionedTreatmentWire {
    /// Treatment variable whose conditioning is replaced by intervention.
    pub variable_id: String,
    /// `(conditioning value id, interventional regime id)` pairs.
    pub value_to_regime: Vec<(String, String)>,
}

/// A checked equivalence on the wire.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EquivalenceWire {
    /// Graph the equivalence was checked on.
    pub graph_id: String,
    /// Interventional regime it licenses, for a coordinate differing only in regime.
    pub interventional_regime_id: Option<String>,
    /// Or: a conditional law on the treatment standing for its intervention.
    /// Exactly one of the two scopes must be given.
    pub conditioned_treatment: Option<ConditionedTreatmentWire>,
    /// Stable identity of the check.
    pub justification_id: String,
}

/// What the identified contract requires.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContractWire {
    /// Graph identity.
    pub graph_id: String,
    /// `IdentificationStatus` name.
    pub identification: String,
    /// Ordered requested coordinates.
    pub estimand: Vec<ScientificQuantityWire>,
    /// Accepted distribution meanings.
    pub accepted_meanings: Vec<DistributionMeaningWire>,
    /// Required evidence factor IDs.
    pub required_evidence_ids: Vec<String>,
    /// Required assumption IDs.
    pub required_assumption_ids: Vec<String>,
    /// Checked equivalences.
    pub equivalences: Vec<EquivalenceWire>,
}

/// The provider object declaration.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderWire {
    /// Provider service.
    pub provider_id: String,
    /// Object within the provider.
    pub object_id: String,
    /// Provider version.
    pub version_id: String,
    /// Input or model snapshot.
    pub snapshot_id: String,
    /// Exact request fingerprint.
    pub request_id: String,
    /// What one draw or value means.
    pub meaning: DistributionMeaningWire,
    /// Declared operation names.
    pub capabilities: Vec<String>,
}

/// One independent verification probe.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProbeWire {
    /// Property name.
    pub kind: String,
    /// Provider value.
    pub observed: f64,
    /// Independent expected value.
    pub expected: f64,
    /// Predeclared absolute tolerance.
    pub tolerance: f64,
}

/// A typed external response grid.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResponseWire {
    /// Provider object.
    pub provider: ProviderWire,
    /// Graph identity the provider used.
    pub graph_id: String,
    /// Ordered coordinates, one per value.
    pub quantities: Vec<ScientificQuantityWire>,
    /// Values in coordinate order.
    pub values: Vec<f64>,
    /// Evidence factors used.
    pub evidence_ids: Vec<String>,
    /// Assumptions relied on.
    pub assumption_ids: Vec<String>,
    /// Party attesting the provider; required unless probes are given.
    pub attestor: Option<String>,
    /// Independent probes; when present the exact contract is verified.
    pub probes: Option<Vec<ProbeWire>>,
    /// Provider-declared uncertainty method.
    pub uncertainty_method: Option<String>,
    /// Provider-declared support labels per coordinate.
    pub point_support: Option<Vec<String>>,
}

/// A structured refusal on the wire; mirrors `ExternalRefusal`.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RefusalWire {
    /// Registered runtime reason code.
    pub code: String,
    /// Refusing stage.
    pub stage: String,
    /// Namespaced detail.
    pub detail: String,
    /// Offending coordinate or probe.
    pub offending: Option<String>,
    /// Expected semantics.
    pub expected: Option<String>,
    /// Supplied semantics.
    pub supplied: Option<String>,
    /// Missing operation.
    pub capability: Option<String>,
    /// Known remedy.
    pub remedy: Option<String>,
}

impl From<antecedent_core::ExternalRefusal> for RefusalWire {
    fn from(value: antecedent_core::ExternalRefusal) -> Self {
        Self {
            code: value.code.to_owned(),
            stage: value.stage.to_owned(),
            detail: value.detail,
            offending: value.offending,
            expected: value.expected,
            supplied: value.supplied,
            capability: value.capability.map(|c| antecedent_core::capability_name(c).to_owned()),
            remedy: value.remedy.map(str::to_owned),
        }
    }
}

fn malformed(detail: &'static str, message: &str) -> RefusalWire {
    RefusalWire {
        code: antecedent_core::reason_code!("invalid_argument").to_owned(),
        stage: "declare".into(),
        detail: detail.to_owned(),
        offending: Some(message.to_owned()),
        expected: None,
        supplied: None,
        capability: None,
        remedy: None,
    }
}

fn quantities(wire: &[ScientificQuantityWire]) -> Result<Vec<ScientificQuantity>, RefusalWire> {
    wire.iter()
        .map(|q| {
            ScientificQuantity::try_from(q.clone())
                .map_err(|error| malformed("external_response_binding.malformed_quantity", error))
        })
        .collect()
}

/// Convert the original contract declaration without performing new identification.
/// # Errors
/// Original identification, equivalence or quantity declaration refusal.
pub fn core_contract(wire: &ContractWire) -> Result<CheckedCausalContract, RefusalWire> {
    Ok(CheckedCausalContract {
        graph_id: wire.graph_id.clone(),
        identification: IdentificationStatus::from_name(&wire.identification).ok_or_else(|| {
            malformed("external_response_binding.malformed_identification", &wire.identification)
        })?,
        estimand: quantities(&wire.estimand)?,
        accepted_meanings: wire.accepted_meanings.iter().copied().map(Into::into).collect(),
        required_evidence_ids: wire.required_evidence_ids.clone(),
        required_assumption_ids: wire.required_assumption_ids.clone(),
        equivalences: wire
            .equivalences
            .iter()
            .map(|e| {
                let scope = match (&e.interventional_regime_id, &e.conditioned_treatment) {
                    (Some(regime), None) => {
                        EquivalenceScope::Regime { interventional_regime_id: regime.clone() }
                    }
                    (None, Some(treatment)) => EquivalenceScope::ConditionedTreatment {
                        variable_id: treatment.variable_id.clone(),
                        value_to_regime: treatment.value_to_regime.clone(),
                    },
                    _ => {
                        return Err(malformed(
                            "external_response_binding.malformed_equivalence",
                            "give exactly one of interventional_regime_id or conditioned_treatment",
                        ));
                    }
                };
                Ok(CheckedEquivalence {
                    graph_id: e.graph_id.clone(),
                    justification_id: e.justification_id.clone(),
                    scope,
                })
            })
            .collect::<Result<Vec<_>, RefusalWire>>()?,
    })
}

/// Build the checked contract and the typed external result from their wires.
/// Convert existing binding wire declarations for an executing callback bridge.
/// The resulting values retain their original declared trust; the callback executor
/// separately restricts them to attested mean outputs.
/// # Errors
/// Original typed declaration, trust, capability or quantity refusal.
pub fn prepare(
    contract: &ContractWire,
    response: &ResponseWire,
) -> Result<(CheckedCausalContract, ExternalResult), RefusalWire> {
    let contract_core = core_contract(contract)?;
    Ok((contract_core, ExternalResult::Response(response_from_wire(response)?)))
}

/// Convert the original external response declaration using the existing typed parser.
/// This preserves declared trust; executing mean callbacks separately force attestation.
/// # Errors
/// Original quantity, capability, trust or support declaration refusal.
pub fn response_from_wire(response: &ResponseWire) -> Result<ExternalResponse, RefusalWire> {
    let coordinates = quantities(&response.quantities)?;
    let capabilities = response
        .provider
        .capabilities
        .iter()
        .map(|name| {
            capability_from_name(name)
                .ok_or_else(|| malformed("external_response_binding.malformed_capability", name))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let object = ExternalScientificObject::Law(LawProviderContract {
        identity: ProviderObjectIdentity {
            provider_id: response.provider.provider_id.clone(),
            object_id: response.provider.object_id.clone(),
            version_id: response.provider.version_id.clone(),
            snapshot_id: response.provider.snapshot_id.clone(),
            request_id: response.provider.request_id.clone(),
        },
        quantities: coordinates.clone(),
        meaning: response.provider.meaning.into(),
        capabilities,
    });
    object.validate().map_err(|e: ExternalContractError| RefusalWire::from(e.to_refusal()))?;
    let trust = match (&response.probes, &response.attestor) {
        (Some(probes), _) => {
            let probes = probes
                .iter()
                .map(|p| {
                    Ok(VerificationProbe {
                        kind: probe_from_name(&p.kind).ok_or_else(|| {
                            malformed("external_response_binding.malformed_probe", &p.kind)
                        })?,
                        observed: p.observed,
                        expected: p.expected,
                        tolerance: p.tolerance,
                    })
                })
                .collect::<Result<Vec<_>, RefusalWire>>()?;
            let receipt = verify_external_object(&object, &probes)
                .map_err(|e| RefusalWire::from(e.to_refusal()))?;
            ExternalTrustState::ExactRequestVerified(Box::new(receipt))
        }
        (None, Some(attestor)) => ExternalTrustState::attest(&object, attestor)
            .map_err(|e| RefusalWire::from(e.to_refusal()))?,
        (None, None) => {
            return Err(malformed(
                "external_response_binding.malformed_trust",
                "attestor or probes required",
            ));
        }
    };
    let point_support = response
        .point_support
        .as_ref()
        .map(|labels| {
            labels
                .iter()
                .map(|l| {
                    SupportStatus::from_name(l)
                        .ok_or_else(|| malformed("external_response_binding.malformed_support", l))
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?;
    Ok(ExternalResponse {
        header: ExternalResultHeader {
            object,
            graph_id: response.graph_id.clone(),
            quantities: coordinates,
            evidence_ids: response.evidence_ids.clone(),
            assumption_ids: response.assumption_ids.clone(),
            trust,
        },
        values: response.values.clone(),
        uncertainty: response
            .uncertainty_method
            .clone()
            .map_or(ExternalUncertaintyMeaning::None, |method_id| {
                ExternalUncertaintyMeaning::ProviderDeclared { method_id }
            }),
        point_support,
    })
}

fn artifact_of(
    claim: &BoundExternalClaim,
    causal_contract_id: &str,
) -> Result<ExternalClaimArtifact, RefusalWire> {
    ExternalClaimArtifact::from_bound_claim(claim, causal_contract_id).map_err(|error| {
        malformed("external_response_binding.malformed_artifact", &error.to_string())
    })
}

/// Bind a response under a contract and return its exportable claim.
///
/// # Errors
/// Any malformed declaration, failed verification or binding mismatch returns
/// a structured refusal.
pub fn bind_response(
    contract: &ContractWire,
    response: &ResponseWire,
    causal_contract_id: &str,
) -> Result<ExternalClaimArtifact, RefusalWire> {
    let (contract_core, result) = prepare(contract, response)?;
    let claim =
        bind_external_result(&contract_core, &result).map_err(|e: ExternalBindingError| {
            RefusalWire::from(e.to_refusal(&contract_core, &result))
        })?;
    artifact_of(&claim, causal_contract_id)
}

/// Bind a response under a contract and to the identified program it answers.
///
/// Runs the program checks of [`bind_external_result_to_program`] (treatment and
/// outcome, population, dose grid, any `quantities` override and the durable
/// identity) before the ordinary binding. The claim stays external.
///
/// # Errors
/// Any malformed declaration, program mismatch, failed verification or binding
/// mismatch returns a structured refusal.
pub fn bind_response_to_program(
    contract: &ContractWire,
    response: &ResponseWire,
    causal_contract_id: &str,
    binding: &ProgramBinding,
    declared: &ExternalProgramClaim,
) -> Result<ExternalClaimArtifact, RefusalWire> {
    let (contract_core, result) = prepare(contract, response)?;
    let bound = bind_external_result_to_program(binding, declared, &contract_core, &result)
        .map_err(RefusalWire::from)?;
    artifact_of(bound.claim(), causal_contract_id)
}

/// Original native scientific contract identity, shared with producer-issued Python claims.
/// This does not establish native execution authority.
/// # Errors
/// Malformed or oversized premise declarations.
// Match the existing external scientific-contract canonical JSON (sorted keys,
// Python's default comma/colon spaces and UTF-16 escapes for non-ASCII strings).
struct ContractFormatter;
impl serde_json::ser::Formatter for ContractFormatter {
    fn write_string_fragment<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
        fragment: &str,
    ) -> std::io::Result<()> {
        for ch in fragment.chars() {
            if ch.is_ascii() {
                writer.write_all(&[u8::try_from(u32::from(ch)).expect("ASCII character")])?;
            } else {
                let mut units = [0u16; 2];
                for unit in ch.encode_utf16(&mut units) {
                    write!(writer, "\\u{unit:04x}")?;
                }
            }
        }
        Ok(())
    }
    fn begin_array_value<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if !first {
            writer.write_all(b", ")?;
        }
        Ok(())
    }
    fn begin_object_key<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if !first {
            writer.write_all(b", ")?;
        }
        Ok(())
    }
    fn begin_object_value<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
    ) -> std::io::Result<()> {
        writer.write_all(b": ")
    }
}
/// Match the original native scientific contract canonical JSON identity.
/// # Errors
/// Oversized or malformed premise declarations.
pub fn native_contract_identity(
    graph: &str,
    status: &str,
    premises: Option<&str>,
) -> Result<String, RefusalWire> {
    use serde::Serialize;
    use sha2::{Digest, Sha256};
    let default = std::collections::BTreeMap::from([
        ("graph_id", serde_json::json!(graph)),
        ("identification", serde_json::json!(status)),
        ("accepted_meanings", serde_json::json!(["interventional_predictive"])),
        ("required_evidence_ids", serde_json::json!([])),
        ("required_assumption_ids", serde_json::json!([])),
        ("equivalences", serde_json::json!([])),
    ]);
    let wire: serde_json::Value = if let Some(text) = premises {
        if text.len() > 1_000_000 {
            return Err(malformed("native_claims.contract_mismatch", "premises exceed bound"));
        }
        serde_json::from_str(text)
            .map_err(|error| malformed("native_claims.contract_mismatch", &error.to_string()))?
    } else {
        serde_json::to_value(default)
            .map_err(|error| malformed("native_claims.contract_mismatch", &error.to_string()))?
    };
    if wire.get("graph_id").and_then(serde_json::Value::as_str) != Some(graph)
        || wire.get("identification").and_then(serde_json::Value::as_str) != Some(status)
    {
        return Ok("invalid:premise_substitution".into());
    }
    let mut typed = wire.clone();
    typed["estimand"] = serde_json::json!([]);
    let _: ContractWire = serde_json::from_value(typed)
        .map_err(|error| malformed("native_claims.contract_mismatch", &error.to_string()))?;
    if wire.as_object().is_none_or(|object| object.len() != 6) {
        return Ok("invalid:premise_fields".into());
    }
    let mut bytes = Vec::new();
    wire.serialize(&mut serde_json::Serializer::with_formatter(&mut bytes, ContractFormatter))
        .map_err(|error| malformed("native_claims.contract_mismatch", &error.to_string()))?;
    Ok(format!("contract:{:x}", Sha256::digest(bytes)))
}
