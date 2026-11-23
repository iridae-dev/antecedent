//! Callback wire declarations and native conversions.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    BTreeMap, CallbackDeclaration, CallbackDependency, CallbackExecutionPolicy, CallbackInput,
    CallbackKind, CallbackOperation, CallbackOutput, CallbackPhase, CallbackQuery, CallbackRequest,
    CallbackTrust, DependencyDeclaration, DependencyKind, Deserialize, EvidenceBasis,
    InputSupplier, Json, ProviderObjectIdentity, PyErr, PyResult, ReplayContract, RetryContract,
    ScientificQuantity, ScientificQuantityWire, Serialize, SourceId, SupportDomain,
    UncertaintyMeaning, UpdateDeclaration, ValidatedOutput, json, parse,
};

// -- JSON shapes -------------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(
    clippy::struct_field_names,
    reason = "the JSON field names are the serialized identity names of a provider object"
)]
pub(super) struct IdentityJson {
    pub(super) provider_id: String,
    pub(super) object_id: String,
    pub(super) version_id: String,
    pub(super) snapshot_id: String,
    pub(super) request_id: String,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct SupportJson {
    pub(super) kind: String,
    pub(super) lower: Option<f64>,
    pub(super) upper: Option<f64>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct EvidenceJson {
    pub(super) declared: bool,
    pub(super) sources: Vec<String>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct DependencyJson {
    pub(super) kind: String,
    pub(super) name: String,
    pub(super) version: String,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct DependenciesJson {
    pub(super) status: String,
    pub(super) items: Vec<DependencyJson>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ReplayJson {
    pub(super) kind: String,
    pub(super) stream_id: Option<String>,
    pub(super) exposes_state_identity: Option<bool>,
    pub(super) retry: Option<String>,
    pub(super) max_attempts: Option<u32>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct TrustJson {
    pub(super) level: String,
    pub(super) attestor: Option<String>,
    pub(super) receipt_digest: Option<String>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct UpdateJson {
    pub(super) candidate_id: String,
    pub(super) prior_id: String,
    pub(super) observation: ScientificQuantityWire,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct InputJson {
    pub(super) name: String,
    pub(super) quantity: ScientificQuantityWire,
    pub(super) semantic_identity: String,
    pub(super) support: SupportJson,
    pub(super) operation: String,
    pub(super) uncertainty: Option<String>,
    pub(super) supplier: String,
    pub(super) callback_identity: Option<String>,
    pub(super) evidence: EvidenceJson,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct DeclarationJson {
    pub(super) component: String,
    pub(super) kind: String,
    pub(super) identity: IdentityJson,
    pub(super) quantity: ScientificQuantityWire,
    pub(super) semantic_identity: String,
    pub(super) meaning: String,
    pub(super) support: SupportJson,
    pub(super) operations: Vec<String>,
    pub(super) update: Option<UpdateJson>,
    pub(super) inputs: Vec<InputJson>,
    pub(super) dependencies: DependenciesJson,
    pub(super) replay: ReplayJson,
    pub(super) trust: TrustJson,
    pub(super) evidence: EvidenceJson,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct QueryJson {
    pub(super) operation: String,
    pub(super) draws: Option<u32>,
    pub(super) points: Vec<f64>,
    pub(super) observation: Option<f64>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RequestJson {
    pub(super) query: QueryJson,
    pub(super) seed: u64,
    pub(super) input_digests: BTreeMap<String, String>,
    pub(super) idempotency_key: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct PolicyJson {
    pub(super) max_invocations: u64,
    pub(super) max_output_values: usize,
    pub(super) wall_limit_ms: Option<u64>,
    pub(super) allow_side_effects: bool,
    pub(super) phase: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct OutputBitsJson {
    pub(super) operation: String,
    pub(super) values_bits: Vec<u64>,
}

// -- conversions -------------------------------------------------------------------------------

pub(super) fn bad(what: &str, detail: impl std::fmt::Display) -> PyErr {
    crate::value_err(format!("invalid {what}: {detail}"))
}

pub(super) fn evidence_of(json: &EvidenceJson) -> PyResult<EvidenceBasis> {
    if !json.declared {
        if json.sources.is_empty() {
            return Ok(EvidenceBasis::Undeclared);
        }
        return Err(bad("evidence", "undeclared evidence lists sources"));
    }
    let mut set = std::collections::BTreeSet::new();
    for hex in &json.sources {
        set.insert(
            SourceId::from_hex(hex).ok_or_else(|| bad("evidence", "a source is not a digest"))?,
        );
    }
    Ok(EvidenceBasis::Declared(set))
}

pub(super) fn evidence_json(evidence: &EvidenceBasis) -> EvidenceJson {
    match evidence {
        EvidenceBasis::Undeclared => EvidenceJson { declared: false, sources: Vec::new() },
        EvidenceBasis::Declared(sources) => {
            EvidenceJson { declared: true, sources: sources.iter().map(SourceId::to_hex).collect() }
        }
    }
}

pub(super) fn support_of(json: &SupportJson) -> PyResult<SupportDomain> {
    match (json.kind.as_str(), json.lower, json.upper) {
        ("real", None, None) => Ok(SupportDomain::Real),
        ("non_negative", None, None) => Ok(SupportDomain::NonNegative),
        ("binary", None, None) => Ok(SupportDomain::Binary),
        ("interval", Some(lower), Some(upper)) => Ok(SupportDomain::interval(lower, upper)),
        _ => Err(bad("support", "a numeric support is real, non_negative, binary or an interval")),
    }
}

pub(super) fn support_json(support: &SupportDomain) -> SupportJson {
    match support {
        SupportDomain::Interval { lower_bits, upper_bits } => SupportJson {
            kind: "interval".into(),
            lower: Some(f64::from_bits(*lower_bits)),
            upper: Some(f64::from_bits(*upper_bits)),
        },
        other => SupportJson { kind: other.kind().to_owned(), lower: None, upper: None },
    }
}

pub(super) fn quantity_of(wire: &ScientificQuantityWire) -> PyResult<ScientificQuantity> {
    ScientificQuantity::try_from(wire.clone()).map_err(|reason| bad("quantity", reason))
}

pub(super) fn operation_of(name: &str) -> PyResult<CallbackOperation> {
    CallbackOperation::from_name(name).ok_or_else(|| bad("operation", name))
}

#[allow(
    clippy::too_many_lines,
    reason = "one flat conversion of every declared field between the JSON and typed forms"
)]
pub(super) fn declaration_of(text: &str) -> PyResult<CallbackDeclaration> {
    let value = parse(text, "callback declaration")?;
    let json: DeclarationJson =
        serde_json::from_value(value).map_err(|error| bad("callback declaration", error))?;
    let kind = CallbackKind::from_name(&json.kind).ok_or_else(|| bad("kind", &json.kind))?;
    let meaning = UncertaintyMeaning::from_name(&json.meaning)
        .ok_or_else(|| bad("meaning", &json.meaning))?
        .meaning;
    let mut operations = std::collections::BTreeSet::new();
    for name in &json.operations {
        operations.insert(operation_of(name)?);
    }
    let update = match &json.update {
        None => None,
        Some(update) => Some(UpdateDeclaration {
            candidate_id: update.candidate_id.clone(),
            prior_id: update.prior_id.clone(),
            observation: quantity_of(&update.observation)?,
        }),
    };
    let mut inputs = Vec::with_capacity(json.inputs.len());
    for input in &json.inputs {
        let supplier = match (input.supplier.as_str(), &input.callback_identity) {
            ("external", None) => InputSupplier::External,
            ("callback", Some(identity)) => InputSupplier::Callback {
                identity: identity.clone(),
                evidence: evidence_of(&input.evidence)?,
            },
            _ => return Err(bad("input supplier", &input.supplier)),
        };
        let uncertainty = match &input.uncertainty {
            None => None,
            Some(name) => Some(
                UncertaintyMeaning::from_name(name)
                    .ok_or_else(|| bad("input meaning", name))?
                    .meaning,
            ),
        };
        inputs.push(CallbackInput {
            name: input.name.clone(),
            quantity: quantity_of(&input.quantity)?,
            semantic_identity: input.semantic_identity.clone(),
            support: support_of(&input.support)?,
            operation: operation_of(&input.operation)?,
            uncertainty,
            supplier,
        });
    }
    let dependencies = match json.dependencies.status.as_str() {
        "complete" => {
            let mut items = Vec::new();
            for item in &json.dependencies.items {
                let kind = DependencyKind::from_name(&item.kind)
                    .ok_or_else(|| bad("dependency kind", &item.kind))?;
                items.push(CallbackDependency::new(kind, item.name.clone(), item.version.clone()));
            }
            DependencyDeclaration::Complete(items)
        }
        "opaque" => DependencyDeclaration::Opaque,
        "undeclared" => DependencyDeclaration::Undeclared,
        other => return Err(bad("dependency status", other)),
    };
    let replay = match (json.replay.kind.as_str(), &json.replay) {
        ("deterministic_pure", _) => ReplayContract::DeterministicPure,
        ("seeded_stochastic", r) => {
            ReplayContract::SeededStochastic { stream_id: r.stream_id.clone().unwrap_or_default() }
        }
        ("stateful", r) => ReplayContract::Stateful {
            exposes_state_identity: r.exposes_state_identity.unwrap_or(false),
        },
        ("side_effecting", r) => ReplayContract::SideEffecting {
            retry: match (r.retry.as_deref(), r.max_attempts) {
                (Some("idempotent_key"), Some(max_attempts)) => {
                    RetryContract::IdempotentKey { max_attempts }
                }
                (Some("no_retry") | None, _) => RetryContract::NoRetry,
                _ => return Err(bad("retry contract", "unknown")),
            },
        },
        (other, _) => return Err(bad("replay contract", other)),
    };
    let trust = match json.trust.level.as_str() {
        "unverified" => CallbackTrust::Unverified,
        "externally_attested" => CallbackTrust::ExternallyAttested {
            attestor: json.trust.attestor.clone().unwrap_or_default(),
        },
        "verified_extension" => CallbackTrust::ExactRequestVerified {
            receipt_digest: json.trust.receipt_digest.clone().unwrap_or_default(),
        },
        // Metadata never confers native support; there is no such level to build.
        other => return Err(bad("trust level", other)),
    };
    Ok(CallbackDeclaration {
        component: json.component.clone(),
        kind,
        identity: ProviderObjectIdentity {
            provider_id: json.identity.provider_id.clone(),
            object_id: json.identity.object_id.clone(),
            version_id: json.identity.version_id.clone(),
            snapshot_id: json.identity.snapshot_id.clone(),
            request_id: json.identity.request_id.clone(),
        },
        quantity: quantity_of(&json.quantity)?,
        semantic_identity: json.semantic_identity.clone(),
        meaning,
        support: support_of(&json.support)?,
        operations,
        update,
        inputs,
        dependencies,
        replay,
        trust,
        evidence: evidence_of(&json.evidence)?,
    })
}

#[allow(
    clippy::too_many_lines,
    reason = "one flat conversion of every declared field between the JSON and typed forms"
)]
pub(super) fn declaration_json(declaration: &CallbackDeclaration) -> DeclarationJson {
    let dependencies = match &declaration.dependencies {
        DependencyDeclaration::Complete(items) => DependenciesJson {
            status: "complete".into(),
            items: items
                .iter()
                .map(|d| DependencyJson {
                    kind: d.kind.as_str().into(),
                    name: d.name.clone(),
                    version: d.version.clone(),
                })
                .collect(),
        },
        DependencyDeclaration::Opaque => {
            DependenciesJson { status: "opaque".into(), items: Vec::new() }
        }
        DependencyDeclaration::Undeclared => {
            DependenciesJson { status: "undeclared".into(), items: Vec::new() }
        }
    };
    let replay = match &declaration.replay {
        ReplayContract::DeterministicPure => ReplayJson {
            kind: "deterministic_pure".into(),
            stream_id: None,
            exposes_state_identity: None,
            retry: None,
            max_attempts: None,
        },
        ReplayContract::SeededStochastic { stream_id } => ReplayJson {
            kind: "seeded_stochastic".into(),
            stream_id: Some(stream_id.clone()),
            exposes_state_identity: None,
            retry: None,
            max_attempts: None,
        },
        ReplayContract::Stateful { exposes_state_identity } => ReplayJson {
            kind: "stateful".into(),
            stream_id: None,
            exposes_state_identity: Some(*exposes_state_identity),
            retry: None,
            max_attempts: None,
        },
        ReplayContract::SideEffecting { retry } => ReplayJson {
            kind: "side_effecting".into(),
            stream_id: None,
            exposes_state_identity: None,
            retry: Some(
                match retry {
                    RetryContract::NoRetry => "no_retry",
                    RetryContract::IdempotentKey { .. } => "idempotent_key",
                }
                .into(),
            ),
            max_attempts: match retry {
                RetryContract::NoRetry => None,
                RetryContract::IdempotentKey { max_attempts } => Some(*max_attempts),
            },
        },
    };
    let trust = match &declaration.trust {
        CallbackTrust::Unverified => {
            TrustJson { level: "unverified".into(), attestor: None, receipt_digest: None }
        }
        CallbackTrust::ExternallyAttested { attestor } => TrustJson {
            level: "externally_attested".into(),
            attestor: Some(attestor.clone()),
            receipt_digest: None,
        },
        CallbackTrust::ExactRequestVerified { receipt_digest } => TrustJson {
            level: "verified_extension".into(),
            attestor: None,
            receipt_digest: Some(receipt_digest.clone()),
        },
    };
    DeclarationJson {
        component: declaration.component.clone(),
        kind: declaration.kind.as_str().into(),
        identity: IdentityJson {
            provider_id: declaration.identity.provider_id.clone(),
            object_id: declaration.identity.object_id.clone(),
            version_id: declaration.identity.version_id.clone(),
            snapshot_id: declaration.identity.snapshot_id.clone(),
            request_id: declaration.identity.request_id.clone(),
        },
        quantity: ScientificQuantityWire::from(&declaration.quantity),
        semantic_identity: declaration.semantic_identity.clone(),
        meaning: UncertaintyMeaning::new(declaration.meaning).name().into(),
        support: support_json(&declaration.support),
        operations: declaration.operations.iter().map(|o| o.as_str().to_owned()).collect(),
        update: declaration.update.as_ref().map(|u| UpdateJson {
            candidate_id: u.candidate_id.clone(),
            prior_id: u.prior_id.clone(),
            observation: ScientificQuantityWire::from(&u.observation),
        }),
        inputs: declaration
            .inputs
            .iter()
            .map(|input| {
                let (supplier, callback_identity, evidence) = match &input.supplier {
                    InputSupplier::External => {
                        ("external", None, EvidenceJson { declared: false, sources: Vec::new() })
                    }
                    InputSupplier::Callback { identity, evidence } => {
                        ("callback", Some(identity.clone()), evidence_json(evidence))
                    }
                };
                InputJson {
                    name: input.name.clone(),
                    quantity: ScientificQuantityWire::from(&input.quantity),
                    semantic_identity: input.semantic_identity.clone(),
                    support: support_json(&input.support),
                    operation: input.operation.as_str().into(),
                    uncertainty: input
                        .uncertainty
                        .map(|m| UncertaintyMeaning::new(m).name().into()),
                    supplier: supplier.into(),
                    callback_identity,
                    evidence,
                }
            })
            .collect(),
        dependencies,
        replay,
        trust,
        evidence: evidence_json(&declaration.evidence),
    }
}

pub(super) fn query_of(json: &QueryJson) -> PyResult<CallbackQuery> {
    let operation = operation_of(&json.operation)?;
    Ok(match operation {
        CallbackOperation::Mean => CallbackQuery::Mean,
        CallbackOperation::Sample => {
            CallbackQuery::Sample { draws: json.draws.ok_or_else(|| bad("query", "draws"))? }
        }
        CallbackOperation::Cdf => CallbackQuery::Cdf { thresholds: json.points.clone() },
        CallbackOperation::Quantile => CallbackQuery::Quantile { levels: json.points.clone() },
        CallbackOperation::PosteriorUpdate => CallbackQuery::PosteriorUpdate {
            observation: json.observation.ok_or_else(|| bad("query", "observation"))?,
            draws: json.draws.unwrap_or(0),
        },
    })
}

pub(super) fn query_json(query: &CallbackQuery) -> QueryJson {
    let blank = |operation: &str| QueryJson {
        operation: operation.into(),
        draws: None,
        points: Vec::new(),
        observation: None,
    };
    match query {
        CallbackQuery::Mean => blank("mean"),
        CallbackQuery::Sample { draws } => QueryJson { draws: Some(*draws), ..blank("sample") },
        CallbackQuery::Cdf { thresholds } => {
            QueryJson { points: thresholds.clone(), ..blank("cdf") }
        }
        CallbackQuery::Quantile { levels } => {
            QueryJson { points: levels.clone(), ..blank("quantile") }
        }
        CallbackQuery::PosteriorUpdate { observation, draws } => QueryJson {
            draws: Some(*draws),
            observation: Some(*observation),
            ..blank("posterior_update")
        },
    }
}

pub(super) fn request_of(text: &str) -> PyResult<CallbackRequest> {
    let value = parse(text, "callback request")?;
    let json: RequestJson =
        serde_json::from_value(value).map_err(|error| bad("callback request", error))?;
    Ok(CallbackRequest {
        query: query_of(&json.query)?,
        seed: json.seed,
        input_digests: json.input_digests,
        idempotency_key: json.idempotency_key,
    })
}

pub(super) fn policy_of(text: &str) -> PyResult<CallbackExecutionPolicy> {
    let value = parse(text, "execution policy")?;
    let json: PolicyJson =
        serde_json::from_value(value).map_err(|error| bad("execution policy", error))?;
    let phase = match json.phase.as_str() {
        "planning" => CallbackPhase::Planning,
        "verification" => CallbackPhase::Verification,
        "execution" => CallbackPhase::Execution,
        other => return Err(bad("phase", other)),
    };
    Ok(CallbackExecutionPolicy {
        max_invocations: json.max_invocations,
        max_output_values: json.max_output_values,
        wall_limit_ms: json.wall_limit_ms,
        allow_side_effects: json.allow_side_effects,
        phase,
    })
}

pub(super) fn output_json(output: &ValidatedOutput) -> Json {
    json!({
        "operation": output.output().operation().as_str(),
        "values": output.output().values(),
    })
}

pub(super) fn output_of(parsed: &OutputBitsJson) -> Result<CallbackOutput, String> {
    // The output is built for the operation it names; a mismatch with the query is refused by
    // the Rust output check, not here.
    let values: Vec<f64> = parsed.values_bits.iter().map(|b| f64::from_bits(*b)).collect();
    let operation = CallbackOperation::from_name(&parsed.operation)
        .ok_or_else(|| "the output names an unknown operation".to_owned())?;
    Ok(match operation {
        CallbackOperation::Mean => match values.as_slice() {
            [value] => CallbackOutput::Mean { value: *value },
            _ => return Err("a mean is one value".into()),
        },
        CallbackOperation::Sample => CallbackOutput::Sample { draws: values },
        CallbackOperation::Cdf => CallbackOutput::Cdf { values },
        CallbackOperation::Quantile => CallbackOutput::Quantile { values },
        CallbackOperation::PosteriorUpdate => match values.split_first() {
            Some((mean, draws)) => {
                CallbackOutput::PosteriorUpdate { mean: *mean, draws: draws.to_vec() }
            }
            None => return Err("a posterior update has at least its mean".into()),
        },
    })
}
