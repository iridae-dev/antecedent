//! Transported callback handoffs and provider source admission.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

// -- transported claim handoff -----------------------------------------------------------------

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TransportIdentityJson {
    source_populations: Vec<String>,
    target_population: String,
    selection_diagram: String,
    assumptions: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OfferJson {
    component: String,
    port: String,
    quantity: ScientificQuantityWire,
    semantic_identity: String,
    support: SupportJson,
    operations: Vec<String>,
    uncertainty: Option<String>,
    transport: TransportIdentityJson,
    evidence: Vec<String>,
}

fn offer_of(text: &str) -> PyResult<TransportedClaimOffer> {
    let value = parse(text, "transported claim offer")?;
    let json: OfferJson =
        serde_json::from_value(value).map_err(|error| bad("transported claim offer", error))?;
    let mut operations = std::collections::BTreeSet::new();
    for name in &json.operations {
        operations
            .insert(ContractOperation::from_name(name).ok_or_else(|| bad("operation", name))?);
    }
    let uncertainty = match &json.uncertainty {
        None => None,
        Some(name) => {
            Some(UncertaintyMeaning::from_name(name).ok_or_else(|| bad("meaning", name))?.meaning)
        }
    };
    let selection_diagram = parse_digest_hex(&json.transport.selection_diagram)
        .ok_or_else(|| bad("selection diagram", "not a 64 hexadecimal digest"))?;
    let mut evidence = std::collections::BTreeSet::new();
    for hex in &json.evidence {
        evidence.insert(
            SourceId::from_hex(hex).ok_or_else(|| bad("evidence", "a source is not a digest"))?,
        );
    }
    Ok(TransportedClaimOffer {
        component: json.component,
        port: json.port,
        quantity: quantity_of(&json.quantity)?,
        semantic_identity: json.semantic_identity,
        support: support_of(&json.support)?,
        operations,
        uncertainty,
        transport: TransportIdentity {
            source_populations: json.transport.source_populations,
            target_population: json.transport.target_population,
            selection_diagram,
            assumptions: json.transport.assumptions,
        },
        evidence,
    })
}

fn handoff_summary(
    identity: &str,
    transport: &TransportIdentity,
    calibration: &CalibrationStatus,
    standing: &str,
) -> Json {
    let calibration = match calibration {
        CalibrationStatus::None => "none",
        CalibrationStatus::PointOnly => "point_only",
        CalibrationStatus::NotMeasured => "not_measured",
        CalibrationStatus::Measured { .. } => "measured",
        CalibrationStatus::Inherited { .. } => "inherited",
    };
    json!({
        "identity": identity,
        "standing": standing,
        "native_target_data": false,
        "transport": {
            "source_populations": transport.source_populations,
            "target_population": transport.target_population,
            "selection_diagram": transport.selection_diagram.to_hex(),
            "assumptions": transport.assumptions,
        },
        "calibration": calibration,
    })
}

fn handoff_refusal_json(error: &TransportHandoffRefusal) -> String {
    RefusalJson::new(error.code, "transport_handoff", &error.detail, &error.message).json()
}

/// Seal a transported claim to one input of a callback and verify it against the callback's
/// declaration: `((bytes, identity, summary json), refusal json)`.
#[pyfunction]
fn transport_handoff_seal(
    py: Python<'_>,
    offer_json: &str,
    callback_json: &str,
    input: &str,
    accepts_transported: bool,
    artifact_id: &str,
) -> PyResult<Refused<(Py<PyBytes>, String, String)>> {
    crate::catch_ffi(|| {
        let offer = offer_of(offer_json)?;
        let callback = declaration_of(callback_json)?;
        let policy = if accepts_transported {
            TransportedInputPolicy::AcceptsTransported
        } else {
            TransportedInputPolicy::NativeTargetOnly
        };
        let handoff = match hand_to_callback(&offer, &callback, input, policy) {
            Ok(handoff) => handoff,
            Err(error) => return Ok((None, Some(handoff_refusal_json(&error)))),
        };
        let bytes = match handoff.to_bytes(artifact_id) {
            Ok(bytes) => bytes,
            Err(error) => return Ok((None, Some(handoff_refusal_json(&error)))),
        };
        let calibration = handoff.artifact().handoff().connection().calibration.clone();
        let summary = handoff_summary(
            handoff.identity(),
            handoff.transport(),
            &calibration,
            handoff.artifact().handoff().standing(),
        );
        Ok((
            Some((
                PyBytes::new(py, &bytes).unbind(),
                handoff.identity().to_owned(),
                summary.to_string(),
            )),
            None,
        ))
    })
}

/// Consume a handoff's bytes and verify them against the callback declaration the consumer
/// holds: `(summary json, refusal json)`.
#[pyfunction]
#[pyo3(signature = (artifact, callback_json, expected_identity=None))]
fn transport_handoff_consume(
    artifact: &[u8],
    callback_json: &str,
    expected_identity: Option<&str>,
) -> PyResult<Refused<String>> {
    crate::catch_ffi(|| {
        let callback = declaration_of(callback_json)?;
        Ok(match consume_for_callback(artifact, expected_identity, &callback) {
            Ok(consumed) => {
                let summary = handoff_summary(
                    consumed.identity(),
                    consumed.transport(),
                    &consumed.handoff().connection().calibration,
                    consumed.handoff().standing(),
                );
                (Some(summary.to_string()), None)
            }
            Err(error) => (None, Some(handoff_refusal_json(&error))),
        })
    })
}

// -- provider output as source evidence -----------------------------------------------------------

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProviderSourceJson {
    component: String,
    quantity: ScientificQuantityWire,
    semantic_identity: String,
    support: SupportJson,
    assumptions: Vec<String>,
    /// The ledger sources the output rests on; `None` is undeclared evidence.
    evidence: Option<Vec<String>>,
    requirement: String,
    trust: String,
    attestor: Option<String>,
}

/// The source a declaration and its JSON describe, or the refusal of a trust claim that names no
/// receipt. A malformed document is an input error.
fn provider_source_of(
    declaration_json: &str,
    source_json: &str,
    receipt: Option<&PyProviderVerificationReceipt>,
) -> PyResult<Result<ProviderTransportSource, String>> {
    let declaration = declaration_of(declaration_json)?;
    let wire: ProviderSourceJson = serde_json::from_value(parse(source_json, "provider source")?)
        .map_err(|error| bad("provider source", error))?;
    let trust = match wire.trust.as_str() {
        "externally_attested" => ExternalTrustState::ExternallyAttested {
            attestor: wire.attestor.clone().unwrap_or_default(),
        },
        "exact_request_verified" => match receipt {
            Some(receipt) => {
                ExternalTrustState::ExactRequestVerified(Box::new(receipt.inner.clone()))
            }
            None => {
                return Ok(Err(RefusalJson::new(
                    antecedent::reason::reason_code!("external_verification_failed"),
                    "verify",
                    antecedent::reason::reason_details::provider_transport::VERIFICATION_REQUIRED,
                    "exact-request verification needs the verification receipt the library issued; a label is not a receipt",
                )
                .json()));
            }
        },
        "native_supported" => ExternalTrustState::NativeSupported,
        other => return Err(bad("trust", other)),
    };
    let requirement = SourceTrustRequirement::from_name(&wire.requirement)
        .ok_or_else(|| bad("requirement", &wire.requirement))?;
    let evidence = match &wire.evidence {
        None => EvidenceBasis::Undeclared,
        Some(sources) => {
            let mut declared = std::collections::BTreeSet::new();
            for hex in sources {
                declared.insert(
                    SourceId::from_hex(hex)
                        .ok_or_else(|| bad("evidence", "a source is not a digest"))?,
                );
            }
            EvidenceBasis::Declared(declared)
        }
    };
    Ok(Ok(ProviderTransportSource {
        component: wire.component,
        object: declaration.provider_object(),
        trust,
        quantity: quantity_of(&wire.quantity)?,
        semantic_identity: wire.semantic_identity,
        support: support_of(&wire.support)?,
        assumptions: wire.assumptions,
        evidence,
        requirement,
    }))
}

fn admitted_json(admitted: &AdmittedProviderSource) -> PyResult<String> {
    let contract = serde_json::to_value(contract_to_wire(&admitted.interface_contract()))
        .map_err(|error| crate::serialization_err(error.to_string()))?;
    Ok(json!({
        "identity": admitted.canonical_identity().to_hex(),
        "trust": admitted.trust().as_wire(),
        "population": admitted.population(),
        "regime": admitted.regime(),
        "operations": admitted.operations().iter().map(|o| o.as_str()).collect::<Vec<_>>(),
        "native_target_data": admitted.is_native_target_data(),
        "contract": contract,
    })
    .to_string())
}

/// Admit a provider's output as source evidence for a transported claim, from declarations
/// alone: the provider object is the callback declaration's, the trust is an attestation or the
/// library-issued receipt. The admission JSON, or the refusal. Nothing runs.
#[pyfunction]
#[pyo3(signature = (declaration_json, source_json, receipt=None))]
fn provider_source_admit(
    declaration_json: &str,
    source_json: &str,
    receipt: Option<PyRef<'_, PyProviderVerificationReceipt>>,
) -> PyResult<Checked> {
    crate::catch_ffi(|| {
        let source = match provider_source_of(declaration_json, source_json, receipt.as_deref())? {
            Ok(source) => source,
            Err(refusal) => return Ok((None, Some(refusal))),
        };
        Ok(match source.admit() {
            Ok(admitted) => (Some(admitted_json(&admitted)?), None),
            Err(refusal) => (None, Some(crate::bridge::external_refusal_json(&refusal))),
        })
    })
}

/// Connect an admitted provider source to a transport route's consumer, after checking that the
/// source belongs to the transport the route carries (its population is a source population and
/// every assumption it relies on is carried). The connection JSON, or a JSON array of every
/// refusal.
#[pyfunction]
#[pyo3(signature = (declaration_json, source_json, transport_json, consumer_json, consumed_port, receipt=None))]
fn provider_source_connect(
    declaration_json: &str,
    source_json: &str,
    transport_json: &str,
    consumer_json: &str,
    consumed_port: &str,
    receipt: Option<PyRef<'_, PyProviderVerificationReceipt>>,
) -> PyResult<Checked> {
    crate::catch_ffi(|| {
        let source = match provider_source_of(declaration_json, source_json, receipt.as_deref())? {
            Ok(source) => source,
            Err(refusal) => return Ok((None, Some(format!("[{refusal}]")))),
        };
        let admitted = match source.admit() {
            Ok(admitted) => admitted,
            Err(refusal) => {
                return Ok((None, Some(Json::Array(vec![refusal_json(&refusal)]).to_string())));
            }
        };
        let transport_wire: TransportIdentityJson =
            serde_json::from_value(parse(transport_json, "transport identity")?)
                .map_err(|error| bad("transport identity", error))?;
        let transport = TransportIdentity {
            selection_diagram: parse_digest_hex(&transport_wire.selection_diagram)
                .ok_or_else(|| bad("selection diagram", "not a 64 hexadecimal digest"))?,
            source_populations: transport_wire.source_populations,
            target_population: transport_wire.target_population,
            assumptions: transport_wire.assumptions,
        };
        let consumer = contract_of(consumer_json, "consumer contract")?;
        Ok(match connect_bound_provider_source(&admitted, &transport, &consumer, consumed_port) {
            Ok(connection) => (Some(connection_json(&connection)), None),
            Err(refusals) => {
                (None, Some(Json::Array(refusals.iter().map(refusal_json).collect()).to_string()))
            }
        })
    })
}

pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(transport_handoff_seal, module)?)?;
    module.add_function(wrap_pyfunction!(transport_handoff_consume, module)?)?;
    module.add_function(wrap_pyfunction!(provider_source_admit, module)?)?;
    module.add_function(wrap_pyfunction!(provider_source_connect, module)?)?;
    Ok(())
}
