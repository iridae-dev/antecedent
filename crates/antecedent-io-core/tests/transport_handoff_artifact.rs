//! Transport handoff artifact: a fresh consumer rebuilds the offer and the
//! consumer, re-derives the connection and refuses edits, relabelled
//! calibration, a consumer that needs native target data and another
//! callback declaration.

use std::collections::BTreeSet;

use antecedent_contract::callback_protocol::{
    CallbackDeclaration, CallbackDependency, CallbackInput, CallbackKind, CallbackOperation,
    CallbackTrust, DependencyDeclaration, DependencyKind, InputSupplier, ReplayContract,
};
use antecedent_contract::evidence_ledger::{SourceId, TransportIdentity};
use antecedent_contract::interface_contract::{
    CalibrationStatus, ContractOperation, EvidenceBasis, SupportDomain,
};
use antecedent_contract::provider_transport::{TransportedClaimOffer, TransportedInputPolicy};
use antecedent_contract::{
    DistributionMeaning, ProviderObjectIdentity, QuantityRole, ScientificQuantity,
};
use antecedent_core::SemanticDigest;
use antecedent_io_core::transport_handoff_artifact::{
    TRANSPORT_HANDOFF_ARTIFACT_FEATURE, TransportHandoffArtifact, TransportHandoffArtifactError,
};

fn quantity(population: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: "schema:recovery".into(),
        variable_name: "recovery".into(),
        role: QuantityRole::Outcome,
        units: "days".into(),
        population_id: population.into(),
        regime_id: "do(a=1)".into(),
        horizon: 0,
        functional_id: "outcome".into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    }
}

fn digest(byte: u8) -> SemanticDigest {
    SemanticDigest::from_bytes([byte; 32])
}

fn offer() -> TransportedClaimOffer {
    TransportedClaimOffer {
        component: "transport_result".into(),
        port: "effect".into(),
        quantity: quantity("target"),
        semantic_identity: "estimand:recovery_under_treatment".into(),
        support: SupportDomain::Real,
        operations: [ContractOperation::Mean].into_iter().collect(),
        uncertainty: None,
        transport: TransportIdentity {
            source_populations: vec!["trial_a".into(), "trial_b".into()],
            target_population: "target".into(),
            selection_diagram: digest(7),
            assumptions: vec!["recovery mechanism invariant across populations".into()],
        },
        evidence: [SourceId::from_digest(digest(5))].into_iter().collect(),
    }
}

fn callback(supplier: InputSupplier) -> CallbackDeclaration {
    CallbackDeclaration {
        component: "downstream_model".into(),
        kind: CallbackKind::Law,
        identity: ProviderObjectIdentity {
            provider_id: "lab".into(),
            object_id: "downstream".into(),
            version_id: "1.0".into(),
            snapshot_id: "snap-1".into(),
            request_id: "req-1".into(),
        },
        quantity: quantity("target"),
        semantic_identity: "estimand:downstream_outcome".into(),
        meaning: DistributionMeaning::InterventionalPredictive,
        support: SupportDomain::Real,
        operations: [CallbackOperation::Mean].into_iter().collect(),
        update: None,
        inputs: vec![CallbackInput {
            name: "effect".into(),
            quantity: quantity("target"),
            semantic_identity: "estimand:recovery_under_treatment".into(),
            support: SupportDomain::Real,
            operation: CallbackOperation::Mean,
            uncertainty: None,
            supplier,
        }],
        dependencies: DependencyDeclaration::Complete(vec![CallbackDependency::new(
            DependencyKind::Model,
            "parameters",
            "beta=0.5",
        )]),
        replay: ReplayContract::DeterministicPure,
        trust: CallbackTrust::ExternallyAttested { attestor: "lab".into() },
        evidence: EvidenceBasis::Declared(BTreeSet::new()),
    }
}

fn sealed()
-> (TransportHandoffArtifact, antecedent_io_core::transport_handoff_artifact::TransportHandoffMeta)
{
    TransportHandoffArtifact::seal_for_callback(
        &offer(),
        &callback(InputSupplier::External),
        "effect",
        TransportedInputPolicy::AcceptsTransported,
    )
    .expect("seals")
}

#[test]
fn a_sealed_handoff_round_trips_and_a_fresh_consumer_rederives_it() {
    let declaration = callback(InputSupplier::External);
    let (artifact, meta) = sealed();
    assert_eq!(meta.feature, TRANSPORT_HANDOFF_ARTIFACT_FEATURE);
    assert_eq!(meta.policy, "accepts_transported");
    assert_eq!(meta.transport.source_populations, vec!["trial_a", "trial_b"]);
    assert_eq!(meta.transport.target_population, "target");
    assert_eq!(meta.transport.selection_diagram, digest(7).to_hex());
    assert!(!artifact.is_native_target_data());

    let bytes = TransportHandoffArtifact::meta_to_bytes(&meta, "handoff-1").expect("encodes");
    let consumed = TransportHandoffArtifact::from_bytes(&bytes, Some(artifact.identity()))
        .expect("a fresh consumer accepts the sealed record");
    assert_eq!(consumed.identity(), artifact.identity());
    assert_eq!(consumed.handoff_identity(), artifact.handoff_identity());
    assert_eq!(consumed.offer(), &offer());
    assert_eq!(consumed.transport(), &offer().transport);
    assert_eq!(consumed.handoff().standing(), "transported_claim");
    assert_eq!(consumed.handoff().connection().calibration, CalibrationStatus::PointOnly);
    consumed.verify_against_callback(&declaration).expect("same declaration");

    // Another declaration (a changed dependency version at unchanged shape).
    let mut other = declaration;
    other.dependencies = DependencyDeclaration::Complete(vec![CallbackDependency::new(
        DependencyKind::Model,
        "parameters",
        "beta=0.6",
    )]);
    let error = consumed.verify_against_callback(&other).expect_err("another declaration");
    assert_eq!(
        error,
        TransportHandoffArtifactError::IdentityMismatch { field: "callback_identity" }
    );
}

#[test]
fn an_edited_or_resealed_handoff_is_refused() {
    let (artifact, meta) = sealed();

    // A changed selection diagram without resealing: the artifact identity differs.
    let mut edited = meta.clone();
    edited.transport.selection_diagram = digest(8).to_hex();
    let error = TransportHandoffArtifact::from_meta(&edited, None).expect_err("edited");
    assert!(matches!(
        error,
        TransportHandoffArtifactError::IdentityMismatch { field: "handoff_identity" }
    ));

    // A resealed edit of the policy to the native-only policy is refused by the
    // library, not believed.
    let mut native_only = meta.clone();
    native_only.policy = "native_target_only".into();
    TransportHandoffArtifact::reseal_identity(&mut native_only);
    let error = TransportHandoffArtifact::from_meta(&native_only, None).expect_err("native");
    let (code, detail, _) = error.refusal().expect("a registered refusal");
    assert_eq!(code, "interface_contract_assumption_mismatch");
    assert_eq!(detail, "provider_transport.native_target_required");

    // A relabelled calibration on the offered port is a different declaration.
    let mut relabelled = meta.clone();
    relabelled.offer_contract.produces[0].calibration.status = "measured".into();
    relabelled.offer_contract.produces[0].calibration.record_id = Some("cal-1".into());
    TransportHandoffArtifact::reseal_identity(&mut relabelled);
    let error = TransportHandoffArtifact::from_meta(&relabelled, None).expect_err("relabelled");
    assert_eq!(error, TransportHandoffArtifactError::IdentityMismatch { field: "offer_contract" });

    // A resealed artifact of another handoff does not match the identity the
    // consumer retained.
    let mut other = meta.clone();
    other.transport.assumptions.push("zz additional assumption".into());
    TransportHandoffArtifact::reseal_identity(&mut other);
    let error = TransportHandoffArtifact::from_meta(&other, Some(artifact.identity()));
    assert!(error.is_err());

    // An unknown required feature and a newer version are refused.
    let mut feature = meta.clone();
    feature.required_features.push("future".into());
    TransportHandoffArtifact::reseal_identity(&mut feature);
    assert_eq!(
        TransportHandoffArtifact::from_meta(&feature, None).expect_err("feature"),
        TransportHandoffArtifactError::UnsupportedSemantics("required feature")
    );
    let mut version = meta;
    version.version = 2;
    assert!(matches!(
        TransportHandoffArtifact::from_meta(&version, None).expect_err("version"),
        TransportHandoffArtifactError::UnsupportedVersion { version: 2 }
    ));
}

#[test]
fn a_handoff_that_the_library_refuses_does_not_seal() {
    // The consumer needs native target data.
    let error = TransportHandoffArtifact::seal_for_callback(
        &offer(),
        &callback(InputSupplier::External),
        "effect",
        TransportedInputPolicy::NativeTargetOnly,
    )
    .expect_err("native target data required");
    assert_eq!(error.refusal().expect("registered").1, "provider_transport.native_target_required");

    // The claim is not in the target population of its own transport identity.
    let mut elsewhere = offer();
    elsewhere.quantity.population_id = "trial_a".into();
    let error = TransportHandoffArtifact::seal_for_callback(
        &elsewhere,
        &callback(InputSupplier::External),
        "effect",
        TransportedInputPolicy::AcceptsTransported,
    )
    .expect_err("not about the target");
    assert_eq!(error.refusal().expect("registered").1, "provider_transport.claim_not_in_target");
}
