//! Callback result artifact: a fresh consumer rebuilds the declaration as an
//! instance of the interface-contract schema, re-validates the request and the
//! output, recomputes every identity and refuses edits, native trust claims,
//! malformed laws and over-large collections. Decoding never executes anything:
//! the artifact is data and the consumer has no provider.

use std::collections::BTreeSet;

use antecedent_contract::callback_protocol::{
    CallbackDeclaration, CallbackDependency, CallbackKind, CallbackOperation, CallbackOutput,
    CallbackQuery, CallbackRequest, CallbackTrust, DependencyDeclaration, DependencyKind,
    MAX_CALLBACK_INPUTS, ReplayContract, validate_output,
};
use antecedent_contract::interface_contract::{EvidenceBasis, SupportDomain};
use antecedent_contract::{
    DistributionMeaning, ProviderObjectIdentity, QuantityRole, ScientificQuantity,
};
use antecedent_io_core::callback_result_artifact::{
    CALLBACK_RESULT_ARTIFACT_FEATURE, CallbackOutcome, CallbackResultArtifact,
    CallbackResultArtifactError, CallbackResultMeta, CallbackResultRecord,
    MAX_CALLBACK_RESULT_ARTIFACT_BYTES, output_values_digest,
};

fn quantity() -> ScientificQuantity {
    ScientificQuantity {
        variable_id: "schema:systolic".into(),
        variable_name: "systolic".into(),
        role: QuantityRole::Outcome,
        units: "mmHg".into(),
        population_id: "trial".into(),
        regime_id: "do(a=1)".into(),
        horizon: 0,
        functional_id: "outcome".into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    }
}

fn declaration() -> CallbackDeclaration {
    CallbackDeclaration {
        component: "gaussian_law".into(),
        kind: CallbackKind::Law,
        identity: ProviderObjectIdentity {
            provider_id: "lab".into(),
            object_id: "gaussian".into(),
            version_id: "1.0".into(),
            snapshot_id: "snap-1".into(),
            request_id: "req-1".into(),
        },
        quantity: quantity(),
        semantic_identity: "estimand:outcome_law".into(),
        meaning: DistributionMeaning::InterventionalPredictive,
        support: SupportDomain::Real,
        operations: [
            CallbackOperation::Mean,
            CallbackOperation::Sample,
            CallbackOperation::Cdf,
            CallbackOperation::Quantile,
        ]
        .into_iter()
        .collect(),
        update: None,
        inputs: Vec::new(),
        dependencies: DependencyDeclaration::Complete(vec![
            CallbackDependency::new(DependencyKind::Model, "parameters", "mu=2;sd=1.5"),
            CallbackDependency::new(DependencyKind::File, "weights", "sha:abc"),
        ]),
        replay: ReplayContract::SeededStochastic { stream_id: "outcome_draws".into() },
        trust: CallbackTrust::ExternallyAttested { attestor: "lab".into() },
        evidence: EvidenceBasis::Declared(BTreeSet::new()),
    }
}

fn computed(
    declaration: CallbackDeclaration,
    query: CallbackQuery,
    output: CallbackOutput,
) -> CallbackResultRecord {
    let output = validate_output(&query, output, &declaration.support, 1_000).expect("valid");
    let request = CallbackRequest::new(query, 9);
    CallbackResultRecord {
        declaration,
        request,
        outcome: CallbackOutcome::Computed { output, invocations: 1, state_identity: None },
    }
}

fn cdf_record() -> CallbackResultRecord {
    computed(
        declaration(),
        CallbackQuery::Cdf { thresholds: vec![0.5, 2.0, 3.5] },
        CallbackOutput::Cdf { values: vec![0.158_655_253_931_457_1, 0.5, 0.841_344_746_068_542_9] },
    )
}

fn values(record: &CallbackResultRecord) -> Vec<f64> {
    match &record.outcome {
        CallbackOutcome::Computed { output, .. } => output.output().values(),
        CallbackOutcome::Unavailable { .. } => Vec::new(),
    }
}

fn sealed_meta(record: &CallbackResultRecord) -> CallbackResultMeta {
    CallbackResultArtifact::seal(record).expect("seals").1
}

fn resealed(
    mut meta: CallbackResultMeta,
    values: &[f64],
    edit: impl FnOnce(&mut CallbackResultMeta),
) -> Result<CallbackResultArtifact, CallbackResultArtifactError> {
    edit(&mut meta);
    CallbackResultArtifact::reseal_identity(&mut meta);
    CallbackResultArtifact::from_meta(&meta, values, None)
}

#[test]
fn a_sealed_result_round_trips_with_stable_identities() {
    let record = cdf_record();
    let bytes = CallbackResultArtifact::to_bytes(&record, "result-1").expect("bytes");
    assert_eq!(bytes, CallbackResultArtifact::to_bytes(&record, "result-1").unwrap());
    let artifact = CallbackResultArtifact::from_bytes(&bytes, None).expect("consumes");
    assert_eq!(artifact.record(), &record);
    assert_eq!(artifact.declaration_identity(), record.declaration.canonical_identity().to_hex());
    // A consumer that retained the identity independently accepts the same artifact.
    let again =
        CallbackResultArtifact::from_bytes(&bytes, Some(artifact.identity())).expect("retained");
    assert_eq!(again.identity(), artifact.identity());
    // The contract inside is the declaration's projection into the one schema.
    let meta = CallbackResultArtifact::inspect(&bytes).expect("inspect");
    assert_eq!(meta.feature, CALLBACK_RESULT_ARTIFACT_FEATURE);
    assert_eq!(meta.contract.component, "gaussian_law");
    assert_eq!(meta.contract.produces.len(), 1);
    assert_eq!(meta.contract.produces[0].name, "output");
    assert_eq!(meta.protocol.operations, ["cdf", "mean", "quantile", "sample"]);
    assert_eq!(meta.stage.status, "computed");
    assert_eq!(meta.standing, "licensed_external");
    // Different values are a different artifact.
    let other = computed(
        declaration(),
        CallbackQuery::Cdf { thresholds: vec![0.5, 2.0, 3.5] },
        CallbackOutput::Cdf { values: vec![0.158_655_253_931_457_1, 0.5, 0.85] },
    );
    let (other_artifact, _) = CallbackResultArtifact::seal(&other).unwrap();
    assert_ne!(other_artifact.identity(), artifact.identity());
}

#[test]
fn an_unavailable_stage_round_trips_and_carries_no_output() {
    let record = CallbackResultRecord {
        declaration: declaration(),
        request: CallbackRequest::new(CallbackQuery::Mean, 1),
        outcome: CallbackOutcome::Unavailable {
            reason: "callback_protocol.provider_unavailable".into(),
        },
    };
    let bytes = CallbackResultArtifact::to_bytes(&record, "unavailable").expect("bytes");
    let artifact = CallbackResultArtifact::from_bytes(&bytes, None).expect("consumes");
    assert_eq!(artifact.record(), &record);
    // An edit that gives the unavailable stage an output refuses.
    let meta = sealed_meta(&record);
    let error = resealed(meta, &[1.0], |_| {}).expect_err("an unavailable stage has no values");
    assert!(matches!(error, CallbackResultArtifactError::Malformed(_)));
}

#[test]
fn resealed_semantic_edits_are_refused_by_recomputed_identities() {
    let record = cdf_record();
    let vals = values(&record);
    let meta = sealed_meta(&record);
    CallbackResultArtifact::from_meta(&meta, &vals, None).expect("the unedited meta consumes");
    let field = |error: CallbackResultArtifactError| match error {
        CallbackResultArtifactError::IdentityMismatch { field } => field,
        other => panic!("expected an identity mismatch, got {other:?}"),
    };
    // A dependency version.
    let error = resealed(meta.clone(), &vals, |m| {
        m.protocol.dependencies[0].version = "mu=3;sd=1.5".into();
    })
    .expect_err("dependency edited");
    assert_eq!(field(error), "declaration_identity");
    // The replay contract.
    let error = resealed(meta.clone(), &vals, |m| {
        m.protocol.replay.kind = "deterministic_pure".into();
        m.protocol.replay.stream_id = None;
    })
    .expect_err("replay edited");
    assert_eq!(field(error), "declaration_identity");
    // The provider version.
    let error = resealed(meta.clone(), &vals, |m| m.protocol.version_id = "2.0".into())
        .expect_err("provider edited");
    assert_eq!(field(error), "declaration_identity");
    // The population of the quantity is the declaration's quantity: a different callback.
    let error = resealed(meta.clone(), &vals, |m| {
        m.contract.produces[0].quantity.population_id = "target".into();
    })
    .expect_err("population edited");
    assert_eq!(field(error), "declaration_identity");
    // An operation added to the interface contract only: the contract is not
    // the declaration's own projection.
    let error = resealed(meta.clone(), &vals, |m| {
        m.contract.produces[0].operations.insert(0, "point".into());
    })
    .expect_err("contract edited");
    assert_eq!(field(error), "contract");
    // The seed.
    let error = resealed(meta.clone(), &vals, |m| m.request.seed = 10).expect_err("seed edited");
    assert_eq!(field(error), "request_identity");
    // A value, even one that is still a law.
    let mut moved = vals.clone();
    moved[2] = 0.9;
    let error = CallbackResultArtifact::from_meta(&meta, &moved, None).expect_err("values edited");
    assert_eq!(field(error), "output_digest");
    // A standing that the declaration does not earn.
    let mut unverified = record.clone();
    unverified.declaration.trust = CallbackTrust::Unverified;
    let unverified_meta = sealed_meta(&unverified);
    assert_eq!(unverified_meta.standing, "diagnostic");
    let error = resealed(unverified_meta, &vals, |m| m.standing = "licensed_external".into())
        .expect_err("standing edited");
    assert_eq!(field(error), "standing");
    // An unequal retained identity.
    let error = CallbackResultArtifact::from_meta(&meta, &vals, Some(&"0".repeat(64)))
        .expect_err("another result");
    assert_eq!(field(error), "retained_identity");
}

#[test]
fn metadata_cannot_mint_native_trust() {
    let record = cdf_record();
    let vals = values(&record);
    let error = resealed(sealed_meta(&record), &vals, |m| {
        m.protocol.trust.level = "native_licensed".into();
        m.protocol.trust.attestor = None;
    })
    .expect_err("native trust is refused, not downgraded");
    let (code, detail, _) = error.refusal().expect("a registered refusal");
    assert_eq!(code, antecedent_core::reason_code!("callback_protocol_invalid"));
    assert_eq!(detail, "callback_protocol.native_trust_claimed");
    // An unknown level is unsupported semantics, never silently accepted.
    let error =
        resealed(sealed_meta(&record), &vals, |m| m.protocol.trust.level = "licensed".into())
            .expect_err("unknown level");
    assert!(matches!(error, CallbackResultArtifactError::UnsupportedSemantics(_)));
}

#[test]
fn a_stored_output_is_revalidated_as_a_law() {
    let record = cdf_record();
    let meta = sealed_meta(&record);
    // A decreasing distribution function, sealed consistently: refused as not a law.
    let decreasing = [0.9, 0.5, 0.1];
    let error = resealed(meta.clone(), &decreasing, |m| {
        m.output_digest = output_values_digest(&decreasing);
    })
    .expect_err("not a law");
    let (code, detail, _) = error.refusal().expect("a registered refusal");
    assert_eq!(code, antecedent_core::reason_code!("callback_output_invalid"));
    assert_eq!(detail, "callback_protocol.output_not_a_law");
    // A non-finite value.
    let nan = [0.1, f64::NAN, 0.9];
    let error = resealed(meta.clone(), &nan, |m| m.output_digest = output_values_digest(&nan))
        .expect_err("not finite");
    assert_eq!(error.refusal().unwrap().1, "callback_protocol.output_not_finite");
    // The wrong number of values for the query.
    let short = [0.1, 0.5];
    let error = resealed(meta, &short, |m| {
        m.stage.value_count = 2;
        m.output_digest = output_values_digest(&short);
    })
    .expect_err("short");
    assert_eq!(error.refusal().unwrap().1, "callback_protocol.output_shape");
}

#[test]
fn collections_and_versions_are_bounded_and_checked_before_use() {
    let record = cdf_record();
    let vals = values(&record);
    let meta = sealed_meta(&record);
    // More inputs than the bound, before any is interpreted.
    let error = resealed(meta.clone(), &vals, |m| {
        let template = antecedent_io_core::callback_result_artifact::InputWire {
            name: String::new(),
            operation: "mean".into(),
            supplier: "external".into(),
            callback_identity: None,
            evidence_declared: false,
            evidence_sources: Vec::new(),
        };
        m.protocol.inputs = (0..=MAX_CALLBACK_INPUTS)
            .map(|i| antecedent_io_core::callback_result_artifact::InputWire {
                name: format!("input-{i:04}"),
                ..template.clone()
            })
            .collect();
    })
    .expect_err("too many inputs");
    assert!(matches!(error, CallbackResultArtifactError::LimitsExceeded(_)));
    // More operations than exist.
    let error = resealed(meta.clone(), &vals, |m| {
        m.protocol.operations =
            ["a", "b", "c", "d", "e", "f"].into_iter().map(str::to_owned).collect();
    })
    .expect_err("too many operations");
    assert!(matches!(error, CallbackResultArtifactError::LimitsExceeded(_)));
    // Too many points in the query.
    let error = resealed(meta.clone(), &vals, |m| {
        m.request.query.points_bits =
            vec![0; antecedent_contract::callback_protocol::MAX_CALLBACK_POINTS + 1];
    })
    .expect_err("too many points");
    assert!(matches!(error, CallbackResultArtifactError::LimitsExceeded(_)));
    // Another version and another feature are refused before the payload is read.
    let error = resealed(meta.clone(), &vals, |m| m.version = 2).expect_err("version");
    assert!(matches!(error, CallbackResultArtifactError::UnsupportedVersion { version: 2 }));
    let error = resealed(meta.clone(), &vals, |m| m.feature = "callback_result_v2".into())
        .expect_err("feature");
    assert!(matches!(error, CallbackResultArtifactError::UnsupportedSemantics(_)));
    let error = resealed(meta, &vals, |m| m.required_features = vec!["streaming".into()])
        .expect_err("required feature");
    assert!(matches!(error, CallbackResultArtifactError::UnsupportedSemantics(_)));
    // Bytes beyond the artifact bound are not decoded; arbitrary bytes are not an artifact.
    let oversized = vec![0_u8; MAX_CALLBACK_RESULT_ARTIFACT_BYTES + 1];
    assert!(CallbackResultArtifact::from_bytes(&oversized, None).is_err());
    assert!(CallbackResultArtifact::from_bytes(b"not an artifact", None).is_err());
    assert!(CallbackResultArtifact::inspect(b"\x00\x01\x02").is_err());
}

#[test]
fn a_corrupted_container_is_refused() {
    let bytes = CallbackResultArtifact::to_bytes(&cdf_record(), "result-1").unwrap();
    for offset in [bytes.len() / 3, bytes.len() / 2, bytes.len() - 5] {
        let mut corrupt = bytes.clone();
        corrupt[offset] ^= 0x01;
        assert!(
            CallbackResultArtifact::from_bytes(&corrupt, None).is_err(),
            "flipping byte {offset} must be detected"
        );
    }
    let truncated = &bytes[..bytes.len() - 17];
    assert!(CallbackResultArtifact::from_bytes(truncated, None).is_err());
}

#[test]
fn a_nested_callback_declaration_survives_the_wire_with_its_inner_evidence() {
    use antecedent_contract::callback_protocol::{CallbackInput, InputSupplier};
    use antecedent_contract::evidence_ledger::SourceId;
    use antecedent_core::SemanticDigest;
    let inner = declaration();
    let source = SourceId::from_digest(SemanticDigest::from_bytes([5; 32]));
    let mut outer = declaration();
    outer.component = "shifted".into();
    outer.kind = CallbackKind::Nested;
    outer.replay = ReplayContract::DeterministicPure;
    outer.operations = [CallbackOperation::Mean].into_iter().collect();
    outer.inputs = vec![CallbackInput {
        name: "inner".into(),
        quantity: inner.quantity.clone(),
        semantic_identity: inner.semantic_identity.clone(),
        support: SupportDomain::Real,
        operation: CallbackOperation::Mean,
        uncertainty: None,
        supplier: InputSupplier::Callback {
            identity: inner.canonical_identity().to_hex(),
            evidence: EvidenceBasis::Declared([source].into_iter().collect()),
        },
    }];
    outer.evidence = EvidenceBasis::Declared(BTreeSet::new());
    let mut request = CallbackRequest::new(CallbackQuery::Mean, 0);
    request.input_digests.insert("inner".into(), output_values_digest(&[2.0]));
    let output =
        validate_output(&request.query, CallbackOutput::Mean { value: 5.0 }, &outer.support, 10)
            .unwrap();
    let record = CallbackResultRecord {
        declaration: outer,
        request,
        outcome: CallbackOutcome::Computed { output, invocations: 1, state_identity: None },
    };
    let bytes = CallbackResultArtifact::to_bytes(&record, "nested").expect("bytes");
    let artifact = CallbackResultArtifact::from_bytes(&bytes, None).expect("consumes");
    assert_eq!(artifact.record(), &record);
    assert_eq!(
        artifact.record().declaration.effective_evidence(),
        EvidenceBasis::Declared([source].into_iter().collect())
    );
}
