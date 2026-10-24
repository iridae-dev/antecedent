//! Python bridge for typed scientific bridges.
//!
//! Python declares a bridge contract and a source claim as JSON; the licensing
//! checks, the mapping, the identities, the artifact and every refusal are
//! Rust's. A refusal comes back as structured JSON for the Python layer to
//! raise as a typed exception; a malformed declaration is an input error.
//!
//! A bridge of a mean is point-only: no interval, posterior, predictive law or
//! joint law is produced, and a request for one is refused by the library. A
//! bridge of the aligned draws of a posterior or a predictive law derives the same
//! draw table with the bridged column in place of the source column, read as an
//! uncalibrated spread; its interval is published only through the measured-coverage
//! gate.

use std::sync::Arc;

use antecedent::{
    ConsumedScientificBridge, PreparedScientificBridge, PreparedScientificBridgeDistribution,
    ScientificBridgeDistributionResult, ScientificBridgeRefusal, ScientificBridgeResult,
    consume_scientific_bridge_artifact, consume_scientific_bridge_distribution_artifact,
    scientific_bridge_artifact_shape,
};
use antecedent_contract::bridge::{
    AssumptionKind, BridgeAssumption, BridgeContract, BridgeEndpoint, BridgeKind, BridgeMapping,
    BridgeScope, BridgeSource, BridgedClaim, EstimandBridge, ExposureSetting, MappingUncertainty,
    NonlinearMap, PointPropagation, QuantityClass, SourceMappingDependence,
    TransportabilityPremise, TrialEffectPair, TrialLevelValidation, ValueKind,
    point_propagation_unavailable,
};
use antecedent_contract::evidence_ledger::parse_digest_hex;
use antecedent_contract::{
    ClaimDomains, ClaimEnvelope, ClaimKind, DomainStatus, ReasoningView, ScientificQuantity,
    SlotAvailability,
};
use antecedent_core::{ContractIdentities, SemanticDigest};
use antecedent_io::quantity_wire::ScientificQuantityWire;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde_json::{Map, Value as Json, json};

use crate::distribution_api::PyJointDistributionArtifact;
use crate::measurement_model_api::{
    list, member, model_json, object, opt_text, parse, parse_model, present, real, small, text,
    texts,
};

// ---------------------------------------------------------------------------
// Refusals as data.
// ---------------------------------------------------------------------------

fn refusal_json(refusal: &ScientificBridgeRefusal) -> String {
    json!({
        "code": refusal.code,
        "stage": "scientific_bridge",
        "detail": refusal.detail,
        "message": refusal.message,
        "offending": Json::Null,
        "remedy": refusal.remedy,
    })
    .to_string()
}

fn external_refusal_json(refusal: &antecedent_contract::ExternalRefusal) -> String {
    json!({
        "code": refusal.code,
        "stage": "scientific_bridge",
        "detail": refusal.detail,
        "message": refusal.remedy.unwrap_or_default(),
        "offending": refusal.offending,
        "remedy": refusal.remedy,
    })
    .to_string()
}

fn freshness_of(
    source: &ClaimEnvelope,
    bridged: &BridgedClaim,
    withdrawn: bool,
    superseded: bool,
) -> String {
    let snapshot = source.identities.data_snapshot;
    let standing = source.freshness(
        Some(snapshot),
        Some(snapshot),
        superseded.then_some(source.claim_id),
        withdrawn,
        true,
    );
    let derived = bridged.freshness(&standing);
    json!({
        "historically_valid": derived.historically_valid,
        "currently_applicable": derived.currently_applicable,
    })
    .to_string()
}

// ---------------------------------------------------------------------------
// Declarations.
// ---------------------------------------------------------------------------

fn digest(value: &str, what: &str) -> PyResult<SemanticDigest> {
    parse_digest_hex(value)
        .ok_or_else(|| crate::value_err(format!("{what} must be 64 hexadecimal characters")))
}

fn parse_quantity(value: &Json) -> PyResult<ScientificQuantity> {
    let wire: ScientificQuantityWire = serde_json::from_value(value.clone())
        .map_err(|error| crate::value_err(format!("invalid quantity declaration: {error}")))?;
    ScientificQuantity::try_from(wire).map_err(crate::value_err)
}

fn quantity_json(quantity: &ScientificQuantity) -> Json {
    serde_json::to_value(ScientificQuantityWire::from(quantity)).unwrap_or(Json::Null)
}

fn parse_scope(value: &Json, what: &str) -> PyResult<BridgeScope> {
    let map = object(value, what)?;
    Ok(BridgeScope {
        population_id: text(map, "population_id")?,
        regime_id: text(map, "regime_id")?,
    })
}

fn scope_json(scope: &BridgeScope) -> Json {
    json!({"population_id": scope.population_id, "regime_id": scope.regime_id})
}

fn parse_endpoint(value: &Json, what: &str) -> PyResult<BridgeEndpoint> {
    let map = object(value, what)?;
    let name = text(map, "class")?;
    let class = QuantityClass::from_name(&name)
        .ok_or_else(|| crate::value_err(format!("unknown quantity class `{name}`")))?;
    Ok(BridgeEndpoint { quantity: parse_quantity(member(map, "quantity")?)?, class })
}

fn endpoint_json(endpoint: &BridgeEndpoint) -> Json {
    json!({"quantity": quantity_json(&endpoint.quantity), "class": endpoint.class.as_str()})
}

fn parse_value_kind(map: &Map<String, Json>, key: &str) -> PyResult<ValueKind> {
    let name = text(map, key)?;
    ValueKind::from_name(&name)
        .ok_or_else(|| crate::value_err(format!("unknown value kind `{name}`")))
}

fn parse_mapping(value: &Json) -> PyResult<BridgeMapping> {
    let map = object(value, "bridge mapping")?;
    Ok(match text(map, "kind")?.as_str() {
        "identity" => BridgeMapping::Identity,
        "affine" => {
            BridgeMapping::Affine { intercept: real(map, "intercept")?, slope: real(map, "slope")? }
        }
        "trial_level_linear" => {
            let mut trials = Vec::new();
            for entry in list(map, "trials")? {
                let pair = entry.as_array().filter(|pair| pair.len() == 2).ok_or_else(|| {
                    crate::value_err("a validation trial is a [surrogate, terminal] pair")
                })?;
                let number = |index: usize| {
                    pair[index]
                        .as_f64()
                        .ok_or_else(|| crate::value_err("a trial effect must be a number"))
                };
                trials.push(TrialEffectPair {
                    surrogate_effect: number(0)?,
                    terminal_effect: number(1)?,
                });
            }
            BridgeMapping::TrialLevelLinear(TrialLevelValidation {
                trials,
                min_trials: small(map, "min_trials")?,
                min_r_squared: real(map, "min_r_squared")?,
            })
        }
        "measurement_correction" => {
            BridgeMapping::MeasurementCorrection(Box::new(parse_model(member(map, "model")?)?))
        }
        other => return Err(crate::value_err(format!("unknown bridge mapping `{other}`"))),
    })
}

fn mapping_json(mapping: &BridgeMapping) -> Json {
    match mapping {
        BridgeMapping::Identity => json!({"kind": "identity"}),
        BridgeMapping::Affine { intercept, slope } => {
            json!({"kind": "affine", "intercept": intercept, "slope": slope})
        }
        BridgeMapping::TrialLevelLinear(validation) => json!({
            "kind": "trial_level_linear",
            "trials": validation
                .trials
                .iter()
                .map(|t| json!([t.surrogate_effect, t.terminal_effect]))
                .collect::<Vec<_>>(),
            "min_trials": validation.min_trials,
            "min_r_squared": validation.min_r_squared,
        }),
        BridgeMapping::MeasurementCorrection(model) => json!({
            "kind": "measurement_correction",
            "model": model_json(model),
            "model_identity": model.identities().model.to_hex(),
        }),
        _ => json!({"kind": "unrecognized"}),
    }
}

fn parse_assumption(value: &Json) -> PyResult<BridgeAssumption> {
    let map = object(value, "bridge assumption")?;
    let name = text(map, "kind")?;
    let kind = AssumptionKind::from_name(&name)
        .ok_or_else(|| crate::value_err(format!("unknown assumption kind `{name}`")))?;
    Ok(BridgeAssumption {
        id: text(map, "id")?,
        kind,
        statement: text(map, "statement")?,
        evidence: opt_text(map, "evidence")?,
    })
}

pub(crate) fn parse_contract(value: &Json) -> PyResult<BridgeContract> {
    let map = object(value, "bridge contract")?;
    let name = text(map, "kind")?;
    let kind = BridgeKind::from_name(&name)
        .ok_or_else(|| crate::value_err(format!("unknown bridge kind `{name}`")))?;
    let assumptions =
        list(map, "assumptions")?.iter().map(parse_assumption).collect::<PyResult<Vec<_>>>()?;
    let transport = match present(map, "transport") {
        None => None,
        Some(value) => {
            let premise = object(value, "transportability premise")?;
            Some(TransportabilityPremise {
                id: text(premise, "id")?,
                statement: text(premise, "statement")?,
                established_in: parse_scope(member(premise, "established_in")?, "established_in")?,
                applies_to: parse_scope(member(premise, "applies_to")?, "applies_to")?,
                evidence: opt_text(premise, "evidence")?,
            })
        }
    };
    Ok(BridgeContract {
        kind,
        source: parse_endpoint(member(map, "source")?, "source endpoint")?,
        target: parse_endpoint(member(map, "target")?, "target endpoint")?,
        source_value: parse_value_kind(map, "source_value")?,
        target_value: parse_value_kind(map, "target_value")?,
        evidence_scope: parse_scope(member(map, "evidence_scope")?, "evidence_scope")?,
        mapping: parse_mapping(member(map, "mapping")?)?,
        assumptions,
        transport,
        mapping_uncertainty: present(map, "mapping_uncertainty")
            .map(parse_mapping_uncertainty)
            .transpose()?,
        source_dependence: present(map, "source_dependence")
            .map(parse_source_dependence)
            .transpose()?,
        estimand_bridge: present(map, "estimand_bridge").map(parse_estimand_bridge).transpose()?,
    })
}

fn parse_mapping_uncertainty(value: &Json) -> PyResult<MappingUncertainty> {
    let map = object(value, "mapping uncertainty")?;
    Ok(MappingUncertainty {
        id: text(map, "id")?,
        intercept_variance: real(map, "intercept_variance")?,
        slope_variance: real(map, "slope_variance")?,
        intercept_slope_covariance: real(map, "intercept_slope_covariance")?,
    })
}

fn parse_source_dependence(value: &Json) -> PyResult<SourceMappingDependence> {
    let map = object(value, "source dependence")?;
    Ok(match text(map, "kind")?.as_str() {
        "independent" => SourceMappingDependence::Independent { evidence: text(map, "evidence")? },
        "slope_covariance" => SourceMappingDependence::SlopeCovariance {
            covariance: real(map, "covariance")?,
            evidence: text(map, "evidence")?,
        },
        other => {
            return Err(crate::value_err(format!("unknown source dependence `{other}`")));
        }
    })
}

fn parse_exposure(value: &Json, what: &str) -> PyResult<ExposureSetting> {
    let map = object(value, what)?;
    Ok(ExposureSetting {
        population_id: text(map, "population_id")?,
        time_start: text(map, "time_start")?,
        time_end: text(map, "time_end")?,
        dose: real(map, "dose")?,
        dose_units: text(map, "dose_units")?,
        duration: real(map, "duration")?,
        duration_units: text(map, "duration_units")?,
        exposure_regime: text(map, "exposure_regime")?,
    })
}

fn parse_estimand_bridge(value: &Json) -> PyResult<EstimandBridge> {
    let map = object(value, "estimand bridge")?;
    Ok(EstimandBridge {
        id: text(map, "id")?,
        experiment_id: text(map, "experiment_id")?,
        experiment: parse_exposure(member(map, "experiment")?, "experiment setting")?,
        target: parse_exposure(member(map, "target")?, "target setting")?,
        recency_delta: real(map, "recency_delta")?,
        recency_units: text(map, "recency_units")?,
        spend_delta: real(map, "spend_delta")?,
        spend_units: text(map, "spend_units")?,
    })
}

fn exposure_json(setting: &ExposureSetting) -> Json {
    json!({
        "population_id": setting.population_id,
        "time_start": setting.time_start,
        "time_end": setting.time_end,
        "dose": setting.dose,
        "dose_units": setting.dose_units,
        "duration": setting.duration,
        "duration_units": setting.duration_units,
        "exposure_regime": setting.exposure_regime,
    })
}

fn contract_json(contract: &BridgeContract) -> Json {
    let mut declared = json!({
        "kind": contract.kind.as_str(),
        "source": endpoint_json(&contract.source),
        "target": endpoint_json(&contract.target),
        "source_value": contract.source_value.as_str(),
        "target_value": contract.target_value.as_str(),
        "evidence_scope": scope_json(&contract.evidence_scope),
        "mapping": mapping_json(&contract.mapping),
        "assumptions": contract.assumptions.iter().map(|a| json!({
            "id": a.id,
            "kind": a.kind.as_str(),
            "statement": a.statement,
            "evidence": a.evidence,
        })).collect::<Vec<_>>(),
        "transport": contract.transport.as_ref().map(|p| json!({
            "id": p.id,
            "statement": p.statement,
            "established_in": scope_json(&p.established_in),
            "applies_to": scope_json(&p.applies_to),
            "evidence": p.evidence,
        })),
    });
    // The extended declarations are listed only where the contract makes them, so a
    // contract that declares none reads exactly as before.
    if let Some(uncertainty) = &contract.mapping_uncertainty {
        declared["mapping_uncertainty"] = json!({
            "id": uncertainty.id,
            "intercept_variance": uncertainty.intercept_variance,
            "slope_variance": uncertainty.slope_variance,
            "intercept_slope_covariance": uncertainty.intercept_slope_covariance,
        });
    }
    match &contract.source_dependence {
        Some(SourceMappingDependence::Independent { evidence }) => {
            declared["source_dependence"] = json!({"kind": "independent", "evidence": evidence});
        }
        Some(SourceMappingDependence::SlopeCovariance { covariance, evidence }) => {
            declared["source_dependence"] = json!({
                "kind": "slope_covariance",
                "covariance": covariance,
                "evidence": evidence,
            });
        }
        _ => {}
    }
    if let Some(bridge) = &contract.estimand_bridge {
        declared["estimand_bridge"] = json!({
            "id": bridge.id,
            "experiment_id": bridge.experiment_id,
            "experiment": exposure_json(&bridge.experiment),
            "target": exposure_json(&bridge.target),
            "recency_delta": bridge.recency_delta,
            "recency_units": bridge.recency_units,
            "spend_delta": bridge.spend_delta,
            "spend_units": bridge.spend_units,
        });
    }
    declared
}

/// A declared source claim. The envelope carries only what a derivation
/// depends on; the reasoning slots are unavailable.
pub(crate) fn parse_claim(value: &Json) -> PyResult<ClaimEnvelope> {
    let map = object(value, "source claim")?;
    let kind_name = text(map, "kind")?;
    let kind = ClaimKind::from_name(&kind_name)
        .ok_or_else(|| crate::value_err(format!("unknown claim kind `{kind_name}`")))?;
    let identities = object(member(map, "identities")?, "claim identities")?;
    let hex = |key: &str| -> PyResult<SemanticDigest> {
        digest(&text(identities, key)?, &format!("identity `{key}`"))
    };
    let product = match opt_text(identities, "identification_product")? {
        Some(text) => Some(digest(&text, "identity `identification_product`")?),
        None => None,
    };
    let identities = ContractIdentities::new(
        hex("target")?,
        hex("identification")?,
        product,
        None,
        hex("inference_binding")?,
        hex("observation")?,
        hex("data_snapshot")?,
    );
    let status = text(map, "evaluated")?;
    let evaluated = DomainStatus::from_name(&status)
        .ok_or_else(|| crate::value_err(format!("unknown domain status `{status}`")))?;
    let execution = match opt_text(map, "execution")? {
        Some(text) => Some(digest(&text, "the execution identity")?),
        None => None,
    };
    let value = match present(map, "value") {
        None => None,
        Some(number) => Some(
            number.as_f64().ok_or_else(|| crate::value_err("claim `value` must be a number"))?,
        ),
    };
    let evidence: Vec<Arc<str>> =
        texts(map, "evidence")?.iter().map(|item| Arc::from(item.as_str())).collect();
    Ok(ClaimEnvelope::new(
        digest(&text(map, "claim_id")?, "the claim identity")?,
        identities,
        kind,
        value,
        opt_text(map, "units")?.as_deref().map(Arc::from),
        ReasoningView::new(
            SlotAvailability::unavailable("declared_source"),
            SlotAvailability::unavailable("declared_source"),
            SlotAvailability::unavailable("declared_source"),
            SlotAvailability::unavailable("declared_source"),
        ),
        ClaimDomains::new(DomainStatus::Unknown, DomainStatus::Unknown, evaluated),
        execution,
        evidence,
    ))
}

// ---------------------------------------------------------------------------
// Results as JSON.
// ---------------------------------------------------------------------------

fn propagation_json(propagation: &PointPropagation) -> Json {
    json!({
        "construction": propagation.construction,
        "source_value": propagation.source_value,
        "source_standard_error": propagation.source_standard_error,
        "plug_in_value": propagation.plug_in_value,
        "mean": propagation.mean,
        "variance": propagation.variance,
        "propagated_mean": propagation.propagated_mean(),
        "reading": propagation.reading(),
        "calibration": propagation.calibration(),
        "mapping_uncertainty": propagation.mapping_uncertainty.as_ref().map(|u| u.id.as_str()),
        "dependence": match &propagation.dependence {
            SourceMappingDependence::Independent { evidence } => json!({
                "kind": "independent", "evidence": evidence, "covariance": 0.0
            }),
            SourceMappingDependence::SlopeCovariance { covariance, evidence } => json!({
                "kind": "slope_covariance", "evidence": evidence, "covariance": covariance
            }),
            _ => Json::Null,
        },
        "nonlinear": propagation.nonlinear.map(|n| json!({
            "map": n.map.as_str(),
            "map_of_mean": n.map_of_mean,
            "expectation": n.expectation,
        })),
    })
}

fn components_json(claim: &ClaimEnvelope) -> Json {
    match &claim.reasoning.uncertainty {
        SlotAvailability::Available(slot) => Json::Array(
            slot.components
                .iter()
                .map(|c| {
                    json!({
                        "source": c.source.as_str(),
                        "target": c.target.to_string(),
                        "omitted": c.omitted,
                    })
                })
                .collect(),
        ),
        _ => Json::Array(Vec::new()),
    }
}

fn parse_nonlinear(map: &Map<String, Json>) -> PyResult<Option<NonlinearMap>> {
    match present(map, "nonlinear") {
        None => Ok(None),
        Some(value) => match value.as_str() {
            Some("square") => Ok(Some(NonlinearMap::Square)),
            Some("exponential") => Ok(Some(NonlinearMap::Exponential)),
            _ => Err(crate::value_err("source `nonlinear` must be `square` or `exponential`")),
        },
    }
}

fn bridged_json(bridged: &BridgedClaim) -> Json {
    let claim = &bridged.claim;
    json!({
        "propagation": bridged.propagation.as_ref().map(propagation_json),
        "uncertainty_components": components_json(claim),
        "kind": bridged.kind().as_str(),
        "source_value": bridged.source_value,
        "value": bridged.value,
        "value_kind": bridged.value_kind.as_str(),
        "mapping": bridged.mapping.as_str(),
        "fit": bridged.fit.map(|fit| json!({
            "intercept": fit.intercept,
            "slope": fit.slope,
            "r_squared": fit.r_squared,
            "trials": fit.trials,
            "surrogate_min": fit.surrogate_min,
            "surrogate_max": fit.surrogate_max,
        })),
        "source_claim_id": bridged.source_claim_id.to_hex(),
        "claim_id": claim.claim_id.to_hex(),
        "contract_identity": bridged.contract_identity.to_hex(),
        "transported": bridged.transported,
        "units": claim.outcome_units.as_deref(),
        "domains": {
            "identification": claim.domains.identification.as_str(),
            "support": claim.domains.support.as_str(),
            "evaluated": claim.domains.evaluated.as_str(),
        },
        "identification_available": claim.reasoning.identification.is_available(),
        "support_available": claim.reasoning.support.is_available(),
        "uncertainty_available": claim.reasoning.uncertainty.is_available(),
        "evidence": claim.evidence.iter().map(ToString::to_string).collect::<Vec<_>>(),
        "unresolved": bridged.receipt.unresolved.iter().map(ToString::to_string).collect::<Vec<_>>(),
        "unavailable": bridged
            .receipt
            .unavailable_operations
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>(),
        "provenance": bridged.provenance.nodes.iter().map(|node| json!({
            "artifact_id": node.artifact_id.to_string(),
            "operation": node.operation.to_string(),
            "parents": node.parents.iter().map(ToString::to_string).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
        "contract": contract_json(&bridged.contract),
    })
}

fn source_json(claim: &ClaimEnvelope) -> Json {
    json!({
        "claim_id": claim.claim_id.to_hex(),
        "kind": claim.kind.as_str(),
        "value": claim.value,
        "units": claim.outcome_units.as_deref(),
        "evaluated": claim.domains.evaluated.as_str(),
        "identification_domain": claim.domains.identification.as_str(),
        "support_domain": claim.domains.support.as_str(),
        "program_identity": claim.identities.program.map(|p| p.to_hex()),
        "data_snapshot": claim.identities.data_snapshot.to_hex(),
        "evidence": claim.evidence.iter().map(ToString::to_string).collect::<Vec<_>>(),
    })
}

// ---------------------------------------------------------------------------
// Handles.
// ---------------------------------------------------------------------------

/// A checked bridge contract with its licensing verdict. Every method that
/// changes it returns a new handle and leaves this one untouched.
#[pyclass(name = "PreparedBridgeHandle", skip_from_py_object)]
pub(crate) struct PreparedBridgeHandle {
    inner: PreparedScientificBridge,
}

/// A derived claim and the source claim it was derived from.
#[pyclass(name = "BridgedClaimHandle", skip_from_py_object)]
pub(crate) struct BridgedClaimHandle {
    inner: ScientificBridgeResult,
}

/// A consumed artifact: the replayed derivation and the reduced source claim.
#[pyclass(name = "ConsumedBridgeHandle", skip_from_py_object)]
pub(crate) struct ConsumedBridgeHandle {
    inner: ConsumedScientificBridge,
}

type Derived = (Option<BridgedClaimHandle>, Option<String>);
type Prepared = (Option<PreparedBridgeHandle>, Option<String>);
type Exported = (Option<(Py<PyBytes>, String)>, Option<String>);

#[pymethods]
impl PreparedBridgeHandle {
    /// The checked contract as a declaration.
    fn contract_json(&self) -> String {
        contract_json(self.inner.contract()).to_string()
    }

    /// The identity of the contract.
    fn contract_identity(&self) -> String {
        self.inner.contract_identity().to_hex()
    }

    /// Whether the bridge carries evidence from another scope under a premise.
    fn transported(&self) -> bool {
        self.inner.transported()
    }

    /// Derive the claim for a declared source claim, or the refusal.
    fn derive(&self, source_json: &str) -> PyResult<Derived> {
        let parsed = parse(source_json, "source claim")?;
        let map = object(&parsed, "source")?;
        let claim = parse_claim(member(map, "claim")?)?;
        let quantity = parse_quantity(member(map, "quantity")?)?;
        let source = BridgeSource { claim: &claim, quantity: &quantity };
        let standard_error = match present(map, "standard_error") {
            None => None,
            Some(number) => Some(
                number
                    .as_f64()
                    .ok_or_else(|| crate::value_err("source `standard_error` must be a number"))?,
            ),
        };
        let nonlinear = parse_nonlinear(map)?;
        Ok(match self.inner.derive_propagating(&source, standard_error, nonlinear) {
            Ok(inner) => (Some(BridgedClaimHandle { inner }), None),
            Err(refusal) => (None, Some(refusal_json(&refusal))),
        })
    }

    /// A prepared bridge for a replacement contract, re-checked in full.
    fn with_contract(&self, contract_json: &str) -> PyResult<Prepared> {
        let contract = parse_contract(&parse(contract_json, "bridge contract")?)?;
        Ok(match self.inner.with_contract(contract) {
            Ok(inner) => (Some(PreparedBridgeHandle { inner }), None),
            Err(refusal) => (None, Some(refusal_json(&refusal))),
        })
    }
}

#[pymethods]
impl BridgedClaimHandle {
    /// The derivation, its identities, assumptions and provenance, as JSON.
    fn summary_json(&self) -> String {
        bridged_json(self.inner.bridged()).to_string()
    }

    /// The source claim as the derivation saw it, as JSON.
    fn source_json(&self) -> String {
        source_json(self.inner.source_claim()).to_string()
    }

    /// The refusal when a consumer requires another kind of object.
    fn require_value_kind(&self, kind: &str) -> PyResult<Option<String>> {
        let kind = ValueKind::from_name(kind)
            .ok_or_else(|| crate::value_err(format!("unknown value kind `{kind}`")))?;
        Ok(self.inner.require_value_kind(kind).err().map(|r| refusal_json(&r)))
    }

    /// The propagated variance and mean of the bridged value as JSON, or the typed
    /// refusal naming the declaration that is missing.
    fn propagated_uncertainty(&self) -> (Option<String>, Option<String>) {
        match self.inner.propagated_uncertainty() {
            Ok(propagation) => (Some(propagation_json(propagation).to_string()), None),
            Err(refusal) => (None, Some(refusal_json(&refusal))),
        }
    }

    /// The typed refusal of an unscoped request: no level named, or a posterior or
    /// joint law.
    fn interval(&self) -> String {
        match self.inner.interval() {
            Err(refusal) => refusal_json(&refusal),
            Ok(()) => String::new(),
        }
    }

    /// The interval of the bridged value at `level` as JSON, or the typed refusal.
    fn bridged_interval(&self, level: f64) -> (Option<String>, Option<String>) {
        match self.inner.bridged_interval(level) {
            Ok(interval) => (
                Some(
                    json!({
                        "lower": interval.lower,
                        "upper": interval.upper,
                        "level": interval.level,
                        "source_value": interval.source_value,
                        "value": interval.value,
                        "source_standard_error": interval.source_standard_error,
                        "standard_error": interval.standard_error,
                        "degrees_of_freedom": interval.degrees_of_freedom,
                        "mapping": interval.mapping.as_str(),
                        "construction": interval.construction,
                        "licensed_by": interval.licensed_by,
                        "coverage_record": interval.coverage_record,
                        "calibration_sha": interval.calibration_sha,
                        "assumptions": interval.assumptions,
                    })
                    .to_string(),
                ),
                None,
            ),
            Err(refusal) => (None, Some(refusal_json(&refusal))),
        }
    }

    /// Applicability of the derived claim given the source's standing.
    fn freshness_json(&self, source_withdrawn: bool, source_superseded: bool) -> String {
        freshness_of(
            self.inner.source_claim(),
            self.inner.bridged(),
            source_withdrawn,
            source_superseded,
        )
    }

    /// The artifact bytes and identity, or the refusal.
    fn export(&self, py: Python<'_>, artifact_id: &str) -> Exported {
        match self.inner.export(artifact_id) {
            Ok(done) => (Some((PyBytes::new(py, &done.bytes).unbind(), done.identity)), None),
            Err(refusal) => (None, Some(refusal_json(&refusal))),
        }
    }
}

#[pymethods]
impl ConsumedBridgeHandle {
    /// The replayed derivation, the reduced source claim and the artifact
    /// identity, as JSON.
    fn summary_json(&self) -> String {
        let mut summary = bridged_json(&self.inner.bridged);
        summary["source"] = source_json(&self.inner.source);
        summary["identity"] = json!(self.inner.identity);
        summary.to_string()
    }

    /// The source claim as the bridge saw it, as JSON.
    fn source_json(&self) -> String {
        source_json(&self.inner.source).to_string()
    }

    /// The refusal when a consumer requires another kind of object.
    fn require_value_kind(&self, kind: &str) -> PyResult<Option<String>> {
        let kind = ValueKind::from_name(kind)
            .ok_or_else(|| crate::value_err(format!("unknown value kind `{kind}`")))?;
        Ok(self
            .inner
            .bridged
            .require_value_kind(kind)
            .err()
            .map(|refusal| external_refusal_json(&refusal)))
    }

    /// The propagated variance and mean the replayed derivation carried as JSON, or the
    /// typed refusal naming what the stored bridge lacks.
    fn propagated_uncertainty(&self) -> (Option<String>, Option<String>) {
        match &self.inner.bridged.propagation {
            Some(propagation) => (Some(propagation_json(propagation).to_string()), None),
            None => (
                None,
                Some(external_refusal_json(&point_propagation_unavailable(
                    &self.inner.bridged.contract,
                    false,
                ))),
            ),
        }
    }

    /// Always the typed refusal: no interval, posterior or joint law.
    fn interval(&self) -> String {
        match self.inner.bridged.request_interval() {
            Err(refusal) => external_refusal_json(&refusal),
            Ok(()) => String::new(),
        }
    }

    /// Applicability of the derived claim given the source's standing.
    fn freshness_json(&self, source_withdrawn: bool, source_superseded: bool) -> String {
        freshness_of(&self.inner.source, &self.inner.bridged, source_withdrawn, source_superseded)
    }
}

// ---------------------------------------------------------------------------
// Entry points.
// ---------------------------------------------------------------------------

/// Check a bridge contract before any number is formed.
#[pyfunction]
fn scientific_bridge_prepare(contract_json: &str) -> PyResult<Prepared> {
    let contract = parse_contract(&parse(contract_json, "bridge contract")?)?;
    Ok(match PreparedScientificBridge::prepare(contract) {
        Ok(inner) => (Some(PreparedBridgeHandle { inner }), None),
        Err(refusal) => (None, Some(refusal_json(&refusal))),
    })
}

/// The identity of a contract declaration without checking its licensing.
#[pyfunction]
fn scientific_bridge_contract_identity(contract_json: &str) -> PyResult<String> {
    let contract = parse_contract(&parse(contract_json, "bridge contract")?)?;
    Ok(contract.identity().to_hex())
}

/// Independently consume a scientific bridge artifact by replaying its
/// derivation, against an artifact identity the consumer retained.
#[pyfunction]
#[pyo3(signature = (artifact, expected_identity=None))]
fn scientific_bridge_consume(
    artifact: &[u8],
    expected_identity: Option<&str>,
) -> (Option<ConsumedBridgeHandle>, Option<String>) {
    match consume_scientific_bridge_artifact(artifact, expected_identity) {
        Ok(inner) => (Some(ConsumedBridgeHandle { inner }), None),
        Err(refusal) => (None, Some(refusal_json(&refusal))),
    }
}

// ---------------------------------------------------------------------------
// Bridges of distributions.
// ---------------------------------------------------------------------------

/// A bridge of draws whose contract was checked once.
#[pyclass(name = "PreparedDistributionBridgeHandle", skip_from_py_object)]
pub(crate) struct PreparedDistributionBridgeHandle {
    inner: PreparedScientificBridgeDistribution,
}

/// A bridged distribution: the derived claim and the aligned draw table.
#[pyclass(name = "BridgedDistributionHandle", skip_from_py_object)]
pub(crate) struct BridgedDistributionHandle {
    inner: ScientificBridgeDistributionResult,
    identity: Option<String>,
}

type DerivedDistribution = (Option<BridgedDistributionHandle>, Option<String>);
type PreparedDistribution = (Option<PreparedDistributionBridgeHandle>, Option<String>);

fn claim_kind_of(name: Option<&str>) -> PyResult<Option<ClaimKind>> {
    name.map(|name| {
        ClaimKind::from_name(name)
            .ok_or_else(|| crate::value_err(format!("unknown claim kind `{name}`")))
    })
    .transpose()
}

fn distribution_summary(result: &ScientificBridgeDistributionResult) -> Json {
    let inspection = result.inspect();
    let claim = result.claim();
    let envelope = result.envelope();
    json!({
        "kind": inspection.kind.as_str(),
        "mapping": inspection.mapping.as_str(),
        "value_kind": inspection.value_kind.as_str(),
        "n_draws": inspection.n_draws,
        "source_mean": inspection.source_mean,
        "source_variance": inspection.source_variance,
        "mean": inspection.mean,
        "variance": inspection.variance,
        "stream": inspection.stream,
        "alignment_identity": inspection.alignment_identity,
        "draws_digest": inspection.draws_digest,
        "source_claim_id": inspection.source_claim_id,
        "claim_id": inspection.claim_id,
        "contract_identity": inspection.contract_identity,
        "transported": inspection.transported,
        "reading": inspection.reading,
        "calibration": "unmeasured",
        "units": envelope.outcome_units.as_deref(),
        "domains": {
            "identification": envelope.domains.identification.as_str(),
            "support": envelope.domains.support.as_str(),
            "evaluated": envelope.domains.evaluated.as_str(),
        },
        "uncertainty_available": envelope.reasoning.uncertainty.is_available(),
        "evidence": envelope.evidence.iter().map(ToString::to_string).collect::<Vec<_>>(),
        "unresolved": claim.receipt.unresolved.iter().map(ToString::to_string).collect::<Vec<_>>(),
        "unavailable": inspection.unavailable,
        "provenance": claim.provenance.nodes.iter().map(|node| json!({
            "artifact_id": node.artifact_id.to_string(),
            "operation": node.operation.to_string(),
            "parents": node.parents.iter().map(ToString::to_string).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
        "contract": contract_json(&claim.contract),
        "bridged_column": result.bridged().column(),
    })
}

fn distribution_freshness(
    result: &ScientificBridgeDistributionResult,
    withdrawn: bool,
    superseded: bool,
) -> String {
    let source = result.bridged().source_envelope();
    let snapshot = source.identities.data_snapshot;
    let standing = source.freshness(
        Some(snapshot),
        Some(snapshot),
        superseded.then_some(source.claim_id),
        withdrawn,
        true,
    );
    let derived = result.freshness(&standing);
    json!({
        "historically_valid": derived.historically_valid,
        "currently_applicable": derived.currently_applicable,
    })
    .to_string()
}

#[pymethods]
impl PreparedDistributionBridgeHandle {
    /// The checked contract as a declaration.
    fn contract_json(&self) -> String {
        contract_json(self.inner.contract()).to_string()
    }

    /// The identity of the contract.
    fn contract_identity(&self) -> String {
        self.inner.contract_identity().to_hex()
    }

    /// Whether the bridge carries evidence from another scope under a premise.
    fn transported(&self) -> bool {
        self.inner.transported()
    }

    /// Bridge the draws of an aligned draw table, or the refusal.
    #[pyo3(signature = (distribution, seed, claim_kind=None))]
    fn derive(
        &self,
        distribution: PyRef<'_, PyJointDistributionArtifact>,
        seed: u64,
        claim_kind: Option<&str>,
    ) -> PyResult<DerivedDistribution> {
        let kind = claim_kind_of(claim_kind)?;
        Ok(match self.inner.derive(distribution.inner(), kind, seed) {
            Ok(inner) => (Some(BridgedDistributionHandle { inner, identity: None }), None),
            Err(refusal) => (None, Some(refusal_json(&refusal))),
        })
    }

    /// A prepared bridge for a replacement contract, re-checked in full.
    fn with_contract(&self, contract_json: &str) -> PyResult<PreparedDistribution> {
        let contract = parse_contract(&parse(contract_json, "bridge contract")?)?;
        Ok(match self.inner.with_contract(contract) {
            Ok(inner) => (Some(PreparedDistributionBridgeHandle { inner }), None),
            Err(refusal) => (None, Some(refusal_json(&refusal))),
        })
    }
}

#[pymethods]
impl BridgedDistributionHandle {
    /// The derivation, its identities, assumptions and provenance, as JSON.
    fn summary_json(&self) -> String {
        let mut summary = distribution_summary(&self.inner);
        if let Some(identity) = &self.identity {
            summary["identity"] = json!(identity);
        }
        summary.to_string()
    }

    /// The bridged draw table: the source table with the bridged column in place
    /// of the source column, with the source table's alignment identity.
    fn distribution(&self) -> PyJointDistributionArtifact {
        PyJointDistributionArtifact {
            artifact: self.inner.distribution().clone(),
            source_claim: None,
        }
    }

    /// The source draw table, unchanged.
    fn source_distribution(&self) -> PyJointDistributionArtifact {
        PyJointDistributionArtifact { artifact: self.inner.source().clone(), source_claim: None }
    }

    /// The refusal when a consumer requires another kind of object.
    fn require_value_kind(&self, kind: &str) -> PyResult<Option<String>> {
        let kind = ValueKind::from_name(kind)
            .ok_or_else(|| crate::value_err(format!("unknown value kind `{kind}`")))?;
        Ok(self.inner.require_value_kind(kind).err().map(|r| refusal_json(&r)))
    }

    /// The typed refusal of an unscoped request for an interval.
    fn interval(&self) -> String {
        match self.inner.interval() {
            Err(refusal) => refusal_json(&refusal),
            Ok(()) => String::new(),
        }
    }

    /// The interval of the bridged value at `level` as JSON, or the typed refusal.
    fn bridged_interval(&self, level: f64) -> (Option<String>, Option<String>) {
        match self.inner.bridged_interval(level) {
            Ok(interval) => (
                Some(
                    json!({
                        "lower": interval.lower,
                        "upper": interval.upper,
                        "level": interval.level,
                        "source_value": interval.source_value,
                        "value": interval.value,
                        "source_standard_error": interval.source_standard_error,
                        "standard_error": interval.standard_error,
                        "degrees_of_freedom": interval.degrees_of_freedom,
                        "mapping": interval.mapping.as_str(),
                        "construction": interval.construction,
                        "licensed_by": interval.licensed_by,
                        "coverage_record": interval.coverage_record,
                        "calibration_sha": interval.calibration_sha,
                        "assumptions": interval.assumptions,
                    })
                    .to_string(),
                ),
                None,
            ),
            Err(refusal) => (None, Some(refusal_json(&refusal))),
        }
    }

    /// The bridge as a propagation-chain measurement-mapping stage: the sources it
    /// reads and the stage, as JSON in the chain record's wire shape.
    fn chain_stage_json(&self, stage_id: &str) -> (Option<String>, Option<String>) {
        match self.inner.chain_stage(stage_id) {
            Ok(parts) => (
                serde_json::to_string(&json!({"sources": parts.sources, "stage": parts.stage}))
                    .ok(),
                None,
            ),
            Err(refusal) => (None, Some(refusal_json(&refusal))),
        }
    }

    /// Applicability of the derived claim given the source's standing.
    fn freshness_json(&self, source_withdrawn: bool, source_superseded: bool) -> String {
        distribution_freshness(&self.inner, source_withdrawn, source_superseded)
    }

    /// The artifact bytes and identity, or the refusal.
    fn export(&self, py: Python<'_>, artifact_id: &str) -> Exported {
        match self.inner.export(artifact_id) {
            Ok(done) => (Some((PyBytes::new(py, &done.bytes).unbind(), done.identity)), None),
            Err(refusal) => (None, Some(refusal_json(&refusal))),
        }
    }
}

/// Check a bridge contract whose source is the draws of a posterior or a
/// predictive law.
#[pyfunction]
fn scientific_bridge_distribution_prepare(contract_json: &str) -> PyResult<PreparedDistribution> {
    let contract = parse_contract(&parse(contract_json, "bridge contract")?)?;
    Ok(match PreparedScientificBridgeDistribution::prepare(contract) {
        Ok(inner) => (Some(PreparedDistributionBridgeHandle { inner }), None),
        Err(refusal) => (None, Some(refusal_json(&refusal))),
    })
}

/// Independently consume a bridge-of-draws artifact by mapping its stored draws
/// again, against an artifact identity the consumer retained.
#[pyfunction]
#[pyo3(signature = (artifact, expected_identity=None))]
fn scientific_bridge_distribution_consume(
    artifact: &[u8],
    expected_identity: Option<&str>,
) -> DerivedDistribution {
    match consume_scientific_bridge_distribution_artifact(artifact, expected_identity) {
        Ok(consumed) => (
            Some(BridgedDistributionHandle {
                inner: consumed.bridged.into(),
                identity: Some(consumed.identity),
            }),
            None,
        ),
        Err(refusal) => (None, Some(refusal_json(&refusal))),
    }
}

/// Which shape a scientific bridge artifact has, `point` or `draws`.
#[pyfunction]
fn scientific_bridge_shape(artifact: &[u8]) -> (Option<String>, Option<String>) {
    match scientific_bridge_artifact_shape(artifact) {
        Ok(shape) => (Some(shape.to_owned()), None),
        Err(refusal) => (None, Some(refusal_json(&refusal))),
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PreparedBridgeHandle>()?;
    module.add_class::<BridgedClaimHandle>()?;
    module.add_class::<ConsumedBridgeHandle>()?;
    module.add_class::<PreparedDistributionBridgeHandle>()?;
    module.add_class::<BridgedDistributionHandle>()?;
    module.add_function(wrap_pyfunction!(scientific_bridge_prepare, module)?)?;
    module.add_function(wrap_pyfunction!(scientific_bridge_contract_identity, module)?)?;
    module.add_function(wrap_pyfunction!(scientific_bridge_consume, module)?)?;
    module.add_function(wrap_pyfunction!(scientific_bridge_distribution_prepare, module)?)?;
    module.add_function(wrap_pyfunction!(scientific_bridge_distribution_consume, module)?)?;
    module.add_function(wrap_pyfunction!(scientific_bridge_shape, module)?)?;
    Ok(())
}
