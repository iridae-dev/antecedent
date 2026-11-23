//! Artifact-specific decoding and fact inspection for composition nodes.

use super::{
    AdmissibleContractArtifact, BTreeMap, BundleStage, ConsumeExpectation, DecisionContract,
    DecisionContractArtifact, DecisionEvalError, DecisionResultArtifact,
    DesignRankingArtifactError, DistributionArtifact, DistributionMetadata, DrawAlignment,
    EvidenceRelationship, ExecutionContext, ExternalClaimArtifact, ExternalClaimMetadata,
    ExternalClaimTrust, ExternalTrustLimit, FACT_EVIDENCE_DIGEST, FACT_LAW, FACT_MEAN_SOURCE,
    FACT_PROVIDER_BINDING, FACT_REQUEST, FACT_REQUIRES_LAW, FACT_SNAPSHOT, FACT_SOURCE_DIGEST,
    FACT_TRUST, Inspected, InverseQueryArtifact, IoError, KIND_ADMISSIBLE, KIND_CONTRACT,
    KIND_RESULT, KIND_ROBUST, KIND_STRUCTURAL, LAW_JOINT, LAW_MARGINAL, LAW_MEAN_ONLY,
    MAX_DECISION_ARTIFACT_BYTES, MAX_DESIGN_RANKING_ARTIFACT_BYTES,
    MAX_DISTRIBUTION_ARTIFACT_BYTES, MAX_EXTERNAL_CLAIM_ARTIFACT_BYTES,
    MAX_INVERSE_QUERY_ARTIFACT_BYTES, MAX_SENSITIVITY_ARTIFACT_BYTES, MAX_VALUES, MeanSource,
    NodeFailure, NodeKind, RELATIONSHIP_MAGIC, ReceiptInfo, RepairArtifactError,
    RepairConsumeLimits, RepairReportArtifact, RobustResultArtifact, ScientificQuantity,
    ScientificQuantityWire, SensitivityArtifact, SignalTrustLabel, StructuralResultArtifact,
    TRUST_ATTESTED, TRUST_NATIVE, TRUST_VERIFIED, Value, blake3_hex, consume_ranking,
    container_label, content_identity, contract_coordinates, coordinate_ids, digest_parts,
    evaluate_contract_on_means, failure, first_section, io_failure, mean_result_to_bytes,
    mean_source_digest, publish, publish_coordinates, publish_requirement, push_value,
    source_digest, to_cbor, trust_label,
};

pub(super) fn inspect_distribution(bytes: &[u8]) -> Result<Inspected, NodeFailure> {
    let kind = NodeKind::Distribution;
    let meta: DistributionMetadata =
        first_section(bytes, MAX_DISTRIBUTION_ARTIFACT_BYTES).map_err(|e| io_failure(kind, &e))?;
    let artifact = DistributionArtifact::from_bytes(bytes, &meta.identity)
        .map_err(|e| io_failure(kind, &e))?;
    let metadata = artifact.metadata();
    let identity = &metadata.identity;
    let draws = source_digest(&artifact);
    let metadata_bytes = to_cbor(metadata).map_err(|e| io_failure(kind, &e))?;
    let metadata_digest = blake3_hex(&metadata_bytes);
    let mut facts = BTreeMap::new();
    let law = match identity.alignment {
        DrawAlignment::Joint => LAW_JOINT,
        DrawAlignment::IndependentMarginals => LAW_MARGINAL,
    };
    publish(&mut facts, FACT_LAW, law);
    publish(&mut facts, FACT_TRUST, trust_label(metadata.trust));
    publish(
        &mut facts,
        FACT_SNAPSHOT,
        format!("{}|{}", identity.snapshot_id, identity.causal_contract_id),
    );
    publish(&mut facts, FACT_SOURCE_DIGEST, draws.clone());
    publish_coordinates(&mut facts, &coordinate_ids(artifact.quantities()), true);
    let mut values = Vec::new();
    push_value(&mut values, "n_draws".to_owned(), artifact.n_draws() as f64);
    for coordinate in 0..artifact.shape()[1].min(MAX_VALUES) {
        if let Ok(mean) = artifact.mean(coordinate) {
            push_value(&mut values, format!("mean.{coordinate}"), mean);
        }
    }
    Ok(Inspected {
        identity: digest_parts(&["distribution", &draws, &metadata_digest]),
        facts,
        values,
        ..Inspected::default()
    })
}

/// The mean source an external claim yields: its coordinates and values under the
/// provider's snapshot and the claim's causal contract. A point-only decision
/// computed from this source retains [`mean_source_digest`] of it.
///
/// # Errors
/// A coordinate that does not convert refuses.
pub fn mean_source_of(claim: &ExternalClaimArtifact) -> Result<MeanSource, IoError> {
    let identity = &claim.metadata().identity;
    let coordinates = identity
        .quantities
        .iter()
        .map(|wire| {
            ScientificQuantity::try_from(wire.clone())
                .map_err(|reason| IoError::Convert(reason.into()))
        })
        .collect::<Result<Vec<_>, IoError>>()?;
    Ok(MeanSource {
        coordinates,
        means: claim.values().to_vec(),
        provider_id: identity.provider_id.clone(),
        snapshot_id: identity.snapshot_id.clone(),
        causal_contract_id: identity.causal_contract_id.clone(),
        rng_id: "none:mean_grid".to_owned(),
        estimate: crate::decision_eval::MeanEstimate::Frequentist,
    })
}

/// Evaluate a decision contract on the means of an external claim's container
/// bytes and return the point-only result as container bytes, ready to embed as a
/// decision-result node beneath the claim.
///
/// The claim is decoded through its own consumer first. A contract whose functional
/// a mean cannot answer (a quantile, a probability or a nonlinear utility) is
/// refused as `unsupported_law`: a mean never yields an outcome distribution.
///
/// # Errors
/// A claim its own consumer refuses, a contract that cannot be evaluated on the
/// means, or an encoding failure.
pub fn mean_decision_bytes(
    contract: &DecisionContract,
    claim_bytes: &[u8],
    artifact_id: &str,
) -> Result<Vec<u8>, NodeFailure> {
    let kind = NodeKind::ExternalClaim;
    let meta: ExternalClaimMetadata = first_section(claim_bytes, MAX_EXTERNAL_CLAIM_ARTIFACT_BYTES)
        .map_err(|e| io_failure(kind, &e))?;
    let claim = ExternalClaimArtifact::from_bytes(claim_bytes, &meta.identity)
        .map_err(|e| io_failure(kind, &e))?;
    let source = mean_source_of(&claim).map_err(|e| io_failure(kind, &e))?;
    let result = evaluate_contract_on_means(contract, &source).map_err(|e| {
        let stage = match e {
            DecisionEvalError::MeanSourceInsufficient { .. } => BundleStage::UnsupportedLaw,
            _ => BundleStage::TamperedQuantity,
        };
        failure(stage, format!("{e:?}"))
    })?;
    mean_result_to_bytes(&result, &source, artifact_id)
        .map_err(|e| io_failure(NodeKind::DecisionResult, &e))
}

pub(super) fn inspect_external_claim(bytes: &[u8]) -> Result<Inspected, NodeFailure> {
    let kind = NodeKind::ExternalClaim;
    let meta: ExternalClaimMetadata = first_section(bytes, MAX_EXTERNAL_CLAIM_ARTIFACT_BYTES)
        .map_err(|e| io_failure(kind, &e))?;
    let artifact = ExternalClaimArtifact::from_bytes(bytes, &meta.identity)
        .map_err(|e| io_failure(kind, &e))?;
    let identity = &artifact.metadata().identity;
    let trust = match (identity.trust, identity.verification.is_some()) {
        (ExternalClaimTrust::ExactRequestVerified, true) => TRUST_VERIFIED,
        _ => TRUST_ATTESTED,
    };
    let mut facts = BTreeMap::new();
    publish(&mut facts, FACT_LAW, LAW_MEAN_ONLY);
    publish(&mut facts, FACT_TRUST, trust);
    publish(
        &mut facts,
        FACT_SNAPSHOT,
        format!("{}|{}", identity.snapshot_id, identity.causal_contract_id),
    );
    publish(&mut facts, FACT_REQUEST, identity.request_id.clone());
    publish(
        &mut facts,
        FACT_PROVIDER_BINDING,
        format!("{}|{}|{}", identity.provider_id, identity.snapshot_id, identity.request_id),
    );
    let sorted = |ids: &[String]| {
        let mut ids = ids.to_vec();
        ids.sort();
        ids.join(",")
    };
    publish(
        &mut facts,
        FACT_EVIDENCE_DIGEST,
        digest_parts(&[
            "evidence",
            &sorted(&identity.evidence_ids),
            &sorted(&identity.assumption_ids),
            &sorted(&identity.equivalence_ids),
        ]),
    );
    let mean_source = mean_source_of(&artifact).map_err(|e| io_failure(kind, &e))?;
    publish(&mut facts, FACT_MEAN_SOURCE, mean_source_digest(&mean_source));
    publish_coordinates(&mut facts, &coordinate_ids(&identity.quantities), true);
    let identity_bytes = to_cbor(identity).map_err(|e| io_failure(kind, &e))?;
    let identity_digest = blake3_hex(&identity_bytes);
    let mut values = Vec::new();
    for (index, value) in artifact.values().iter().enumerate() {
        push_value(&mut values, format!("value.{index}"), *value);
    }
    Ok(Inspected {
        identity: digest_parts(&["external_claim", &identity_digest]),
        facts,
        values,
        ..Inspected::default()
    })
}

fn body_str(body: &Value, path: &[&str]) -> String {
    let mut node = body;
    for key in path {
        node = &node[*key];
    }
    node.as_str().unwrap_or_default().to_owned()
}

pub(super) fn inspect_contract(bytes: &[u8]) -> Result<Inspected, NodeFailure> {
    let kind = NodeKind::DecisionContract;
    let io = |e: &IoError| io_failure(kind, e);
    let label = container_label(bytes, MAX_DECISION_ARTIFACT_BYTES).map_err(|e| io(&e))?;
    let body: Value = first_section(bytes, MAX_DECISION_ARTIFACT_BYTES).map_err(|e| io(&e))?;
    let stored = body_str(&body, &["identity"]);
    let (identity, contract) = match label.as_str() {
        KIND_CONTRACT => {
            let artifact =
                DecisionContractArtifact::from_bytes(bytes, &stored).map_err(|e| io(&e))?;
            (artifact.identity().to_owned(), artifact.contract().clone())
        }
        KIND_ADMISSIBLE => {
            let artifact =
                AdmissibleContractArtifact::from_bytes(bytes, &stored).map_err(|e| io(&e))?;
            (artifact.identity().to_owned(), artifact.contract().contract.clone())
        }
        other => {
            return Err(failure(
                BundleStage::SwappedEvidence,
                format!("`{other}` is not a decision contract artifact"),
            ));
        }
    };
    let mut facts = BTreeMap::new();
    publish_coordinates(&mut facts, &contract_coordinates(&contract), false);
    publish_requirement(&contract, &mut facts)?;
    Ok(Inspected { identity, facts, ..Inspected::default() })
}

fn atom_digests(body: &Value) -> Result<Vec<(String, Option<String>)>, NodeFailure> {
    let invalid = || failure(BundleStage::TamperedQuantity, "the structure records are malformed");
    let atoms = body["atoms"].as_array().ok_or_else(invalid)?;
    atoms
        .iter()
        .map(|atom| {
            let id = atom["id"].as_str().ok_or_else(invalid)?.to_owned();
            let digest = match &atom["source_digest"] {
                Value::Null => None,
                Value::String(text) => Some(text.clone()),
                _ => return Err(invalid()),
            };
            Ok((id, digest))
        })
        .collect()
}

// One arm per stored result kind: splitting them would only scatter the shared tail.
#[allow(
    clippy::too_many_lines,
    reason = "one decode-and-check per field of the stored result, in wire order"
)]
pub(super) fn inspect_result(bytes: &[u8]) -> Result<Inspected, NodeFailure> {
    let kind = NodeKind::DecisionResult;
    let io = |e: &IoError| io_failure(kind, e);
    let label = container_label(bytes, MAX_DECISION_ARTIFACT_BYTES).map_err(|e| io(&e))?;
    let body: Value = first_section(bytes, MAX_DECISION_ARTIFACT_BYTES).map_err(|e| io(&e))?;
    let mut inspected =
        Inspected { contract_required: true, bound_to_contract: true, ..Inspected::default() };
    match label.as_str() {
        KIND_RESULT => {
            let contract = body_str(&body, &["contract_identity"]);
            let source = body_str(&body, &["source_digest"]);
            let artifact = DecisionResultArtifact::from_bytes(bytes, &contract, &source)
                .map_err(|e| io(&e))?;
            let result = artifact.result();
            publish(
                &mut inspected.facts,
                FACT_SNAPSHOT,
                format!("{}|{}", result.source.snapshot_id, result.source.causal_contract_id),
            );
            for action in &result.actions {
                push_value(
                    &mut inspected.values,
                    format!("{}.expected_utility", action.id),
                    action.expected_utility,
                );
                push_value(&mut inspected.values, format!("{}.value", action.id), action.value);
            }
            if let Some(evpi) = result.evpi {
                push_value(&mut inspected.values, "evpi".to_owned(), evpi);
            }
            inspected.contract = Some(contract);
            inspected.sources = vec![source];
            inspected.sources_must_be_upstream = true;
        }
        KIND_ROBUST => {
            let contract = body_str(&body, &["result", "contract_identity"]);
            let atoms = atom_digests(&body)?;
            let artifact =
                RobustResultArtifact::from_bytes(bytes, &contract, &atoms).map_err(|e| io(&e))?;
            for action in &artifact.result().actions {
                if let Some((low, high)) = action.range {
                    push_value(&mut inspected.values, format!("{}.range_low", action.id), low);
                    push_value(&mut inspected.values, format!("{}.range_high", action.id), high);
                }
            }
            push_value(
                &mut inspected.values,
                "native_verified".to_owned(),
                f64::from(u8::from(artifact.native_verified())),
            );
            let receipts = artifact.receipts();
            if !receipts.is_empty() {
                let all_verified = receipts
                    .iter()
                    .all(|r| matches!(r.trust, ExternalTrustLimit::VerifiedExtension));
                publish(
                    &mut inspected.facts,
                    FACT_TRUST,
                    if all_verified { TRUST_VERIFIED } else { TRUST_ATTESTED },
                );
            }
            inspected.receipts = receipts
                .iter()
                .map(|r| ReceiptInfo {
                    provider_id: r.provider_id.clone(),
                    snapshot_id: r.snapshot_id.clone(),
                    request: r.request_fingerprint.clone(),
                    verified: matches!(r.trust, ExternalTrustLimit::VerifiedExtension),
                })
                .collect();
            inspected.contract = Some(contract);
            inspected.sources = atoms.into_iter().filter_map(|(_, digest)| digest).collect();
        }
        KIND_STRUCTURAL => {
            let contract = body_str(&body, &["contract_identity"]);
            let atoms = atom_digests(&body)?;
            let artifact = StructuralResultArtifact::from_bytes(bytes, &contract, &atoms)
                .map_err(|e| io(&e))?;
            for action in &artifact.result().actions {
                if let Some(weighted) = action.weighted_value {
                    push_value(
                        &mut inspected.values,
                        format!("{}.weighted_value", action.id),
                        weighted,
                    );
                }
                if let Some((low, high)) = action.range {
                    push_value(&mut inspected.values, format!("{}.range_low", action.id), low);
                    push_value(&mut inspected.values, format!("{}.range_high", action.id), high);
                }
            }
            inspected.contract = Some(contract);
            inspected.sources = atoms.into_iter().filter_map(|(_, digest)| digest).collect();
        }
        other => {
            return Err(failure(
                BundleStage::TamperedQuantity,
                format!("`{other}` is not a decision result artifact"),
            ));
        }
    }
    inspected.identity =
        content_identity(&label, bytes, MAX_DECISION_ARTIFACT_BYTES).map_err(|e| io(&e))?;
    Ok(inspected)
}

pub(super) fn inspect_sensitivity(bytes: &[u8]) -> Result<Inspected, NodeFailure> {
    let kind = NodeKind::Sensitivity;
    if bytes.len() > MAX_SENSITIVITY_ARTIFACT_BYTES {
        return Err(failure(BundleStage::Oversized, "the sensitivity artifact exceeds its bound"));
    }
    let artifact =
        SensitivityArtifact::from_bytes(bytes, None).map_err(|e| io_failure(kind, &e))?;
    let provenance = artifact.provenance();
    let mut facts = BTreeMap::new();
    publish(
        &mut facts,
        FACT_SNAPSHOT,
        format!("{}|{}", provenance.provider_snapshot, provenance.causal_contract_id),
    );
    let wires: Vec<ScientificQuantityWire> =
        artifact.quantities().iter().map(|q| q.quantity.clone()).collect();
    publish_coordinates(&mut facts, &coordinate_ids(&wires), false);
    let mut values = Vec::new();
    push_value(&mut values, "grid_points".to_owned(), artifact.grid().len() as f64);
    Ok(Inspected {
        identity: artifact.identity().digest.clone(),
        facts,
        values,
        ..Inspected::default()
    })
}

pub(super) fn inspect_ranking(bytes: &[u8]) -> Result<Inspected, NodeFailure> {
    if bytes.len() > MAX_DESIGN_RANKING_ARTIFACT_BYTES {
        return Err(failure(BundleStage::Oversized, "the ranking artifact exceeds its bound"));
    }
    let consumed = consume_ranking(bytes, &ConsumeExpectation::default()).map_err(|e| {
        let stage = match &e {
            DesignRankingArtifactError::UnsupportedVersion(_) => BundleStage::IncompatibleVersion,
            DesignRankingArtifactError::Bounds(_) => BundleStage::Oversized,
            _ => BundleStage::TamperedQuantity,
        };
        failure(stage, e.to_string())
    })?;
    let mut facts = BTreeMap::new();
    let wire = crate::design_ranking_artifact::DesignRankingArtifactWire::from_bytes(bytes)
        .map_err(|e| failure(BundleStage::TamperedQuantity, e.to_string()))?;
    publish(
        &mut facts,
        "rollout_terminal_decision",
        crate::rollout_artifact::decision_digest(&wire.decision)
            .map_err(|e| io_failure(NodeKind::StudyRanking, &e))?,
    );
    let trust =
        if consumed.candidates.iter().any(|c| c.trust == SignalTrustLabel::ExternallyAttested) {
            TRUST_ATTESTED
        } else if consumed.candidates.iter().all(|c| c.trust == SignalTrustLabel::NativeSupported) {
            TRUST_NATIVE
        } else {
            TRUST_VERIFIED
        };
    publish(&mut facts, FACT_TRUST, trust);
    let mut values = Vec::new();
    for entry in &consumed.ranking.entries {
        push_value(&mut values, format!("{}.evsi", entry.semantic_id), entry.evsi);
        if let Some(net) = entry.net_value {
            push_value(&mut values, format!("{}.net_value", entry.semantic_id), net);
        }
        push_value(&mut values, format!("{}.rank", entry.semantic_id), entry.rank as f64);
    }
    Ok(Inspected {
        identity: consumed.identity.clone(),
        facts,
        values,
        contract: consumed.ranking.decision_contract_identity.clone(),
        sources: consumed.ranking.source_digests.clone(),
        ..Inspected::default()
    })
}

pub(super) fn inspect_rollout(bytes: &[u8]) -> Result<Inspected, NodeFailure> {
    let artifact = crate::rollout_artifact::RolloutArtifact::from_bytes(bytes)
        .map_err(|e| io_failure(NodeKind::Rollout, &e))?;
    artifact.verify(&artifact.binding).map_err(|e| io_failure(NodeKind::Rollout, &e))?;
    let source = inspect_distribution(&artifact.source_artifact)?;
    let mut facts = BTreeMap::new();
    publish(&mut facts, FACT_TRUST, trust_label(artifact.binding.source.trust));
    publish(&mut facts, FACT_REQUIRES_LAW, LAW_JOINT);
    publish(&mut facts, "rollout_source_identity", source.identity);
    publish(&mut facts, "rollout_ranking_identity", artifact.binding.ranking_identity.clone());
    publish(
        &mut facts,
        "rollout_terminal_decision",
        artifact.decision_digest().map_err(|e| io_failure(NodeKind::Rollout, &e))?,
    );
    publish_coordinates(
        &mut facts,
        &coordinate_ids(std::slice::from_ref(&artifact.binding.state)),
        false,
    );
    Ok(Inspected {
        identity: artifact.digest,
        facts,
        contract: Some(artifact.binding.decision.contract_identity),
        sources: vec![artifact.binding.source_digest],
        ..Inspected::default()
    })
}

pub(super) fn inspect_inverse(bytes: &[u8]) -> Result<Inspected, NodeFailure> {
    let kind = NodeKind::InverseQuery;
    if bytes.len() > MAX_INVERSE_QUERY_ARTIFACT_BYTES {
        return Err(failure(BundleStage::Oversized, "the inverse query exceeds its bound"));
    }
    let artifact =
        InverseQueryArtifact::from_bytes(bytes, None).map_err(|e| io_failure(kind, &e))?;
    let mut facts = BTreeMap::new();
    publish_coordinates(&mut facts, &contract_coordinates(&artifact.query().contract), false);
    let mut values = Vec::new();
    push_value(
        &mut values,
        "feasible_actions".to_owned(),
        artifact.result().feasible_actions.len() as f64,
    );
    Ok(Inspected {
        identity: artifact.identity().digest.clone(),
        facts,
        values,
        contract: Some(artifact.result().contract_identity.clone()),
        ..Inspected::default()
    })
}

fn repair_failure(error: &RepairArtifactError) -> NodeFailure {
    let stage = match error.detail {
        antecedent_core::reason_details::repair_artifact::UNSUPPORTED_VERSION => {
            BundleStage::IncompatibleVersion
        }
        antecedent_core::reason_details::repair_artifact::BOUNDS_EXCEEDED => BundleStage::Oversized,
        antecedent_core::reason_details::repair_artifact::BUDGET => {
            BundleStage::CallbackUnavailable
        }
        _ => BundleStage::SwappedEvidence,
    };
    failure(stage, error.to_string())
}

pub(super) fn inspect_repair(bytes: &[u8]) -> Result<Inspected, NodeFailure> {
    let artifact = RepairReportArtifact::from_bytes(bytes).map_err(|e| repair_failure(&e))?;
    let ctx = ExecutionContext::production(1, 1);
    artifact.consume(RepairConsumeLimits::default(), &ctx).map_err(|e| repair_failure(&e))?;
    let mut values = Vec::new();
    push_value(&mut values, "outcomes".to_owned(), artifact.report.outcomes.len() as f64);
    push_value(
        &mut values,
        "sufficient".to_owned(),
        artifact.report.ranked_sufficient.len() as f64,
    );
    Ok(Inspected { identity: artifact.report_digest.clone(), values, ..Inspected::default() })
}

/// The bytes of an evidence-relationship node: how the evidence behind the
/// bundle nodes `left` and `right` depends on each other. Connect both nodes to the
/// relationship node so the verifier can check the declaration against them.
#[must_use]
pub fn evidence_relationship_bytes(
    relationship: EvidenceRelationship,
    left: &str,
    right: &str,
) -> Vec<u8> {
    format!(
        "{RELATIONSHIP_MAGIC}\nrelationship={}\nleft={left}\nright={right}\n",
        relationship.as_str()
    )
    .into_bytes()
}

pub(super) fn inspect_relationship(bytes: &[u8]) -> Result<Inspected, NodeFailure> {
    let invalid = |why: &str| failure(BundleStage::SwappedEvidence, why.to_owned());
    if bytes.len() > 4096 {
        return Err(failure(BundleStage::Oversized, "an evidence relationship is a short record"));
    }
    let text = std::str::from_utf8(bytes).map_err(|_| invalid("the relationship is not text"))?;
    let mut lines = text.lines();
    if lines.next() != Some(RELATIONSHIP_MAGIC) {
        return Err(invalid("not an evidence relationship record"));
    }
    let field = |lines: &mut std::str::Lines<'_>, key: &str| -> Result<String, NodeFailure> {
        lines
            .next()
            .and_then(|line| line.strip_prefix(key))
            .and_then(|rest| rest.strip_prefix('='))
            .map(str::to_owned)
            .ok_or_else(|| invalid("a relationship field is missing"))
    };
    let name = field(&mut lines, "relationship")?;
    let left = field(&mut lines, "left")?;
    let right = field(&mut lines, "right")?;
    let relationship = EvidenceRelationship::from_name(&name)
        .ok_or_else(|| invalid("the relationship is not a known dependence"))?;
    if left.is_empty() || right.is_empty() || left == right || lines.next().is_some() {
        return Err(invalid("a relationship names two distinct nodes"));
    }
    Ok(Inspected {
        identity: digest_parts(&["evidence_relationship", &name, &left, &right]),
        relationship: Some((relationship, left, right)),
        ..Inspected::default()
    })
}
