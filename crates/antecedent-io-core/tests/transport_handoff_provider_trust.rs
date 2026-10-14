//! A transport handoff records which external trust level stood behind the
//! claim, so a reader of the artifact alone can tell attested from exact-request
//! verified. The label is part of the identity and is never native.

use antecedent_contract::ProviderTrust;
use antecedent_contract::evidence_ledger::{SourceId, TransportIdentity};
use antecedent_contract::interface_contract::{
    ConsumedPort, ContractOperation, InterfaceContract, SupportDomain,
};
use antecedent_contract::provider_transport::{TransportedClaimOffer, TransportedInputPolicy};
use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_core::SemanticDigest;
use antecedent_io_core::transport_handoff_artifact::{
    TransportHandoffArtifact, TransportHandoffArtifactError,
};

fn quantity() -> ScientificQuantity {
    ScientificQuantity {
        variable_id: "schema:recovery".into(),
        variable_name: "recovery".into(),
        role: QuantityRole::Outcome,
        units: "days".into(),
        population_id: "target".into(),
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
        quantity: quantity(),
        semantic_identity: "estimand:recovery_under_treatment".into(),
        support: SupportDomain::Real,
        operations: [ContractOperation::Mean].into_iter().collect(),
        uncertainty: None,
        transport: TransportIdentity {
            source_populations: vec!["trial_a".into()],
            target_population: "target".into(),
            selection_diagram: digest(7),
            assumptions: vec!["recovery mechanism invariant across populations".into()],
        },
        evidence: [SourceId::from_digest(digest(5))].into_iter().collect(),
    }
}

fn consumer() -> InterfaceContract {
    InterfaceContract {
        component: "downstream".into(),
        produces: Vec::new(),
        consumes: vec![ConsumedPort::new(
            "effect",
            quantity(),
            "estimand:recovery_under_treatment",
            SupportDomain::Real,
            [ContractOperation::Mean],
        )],
    }
}

fn seal(trust: ProviderTrust) -> Result<
    (TransportHandoffArtifact, antecedent_io_core::transport_handoff_artifact::TransportHandoffMeta),
    TransportHandoffArtifactError,
> {
    TransportHandoffArtifact::seal_with_provider_trust(
        &offer(),
        &consumer(),
        "effect",
        TransportedInputPolicy::AcceptsTransported,
        trust,
    )
}

#[test]
fn the_trust_level_is_recoverable_from_the_artifact_and_part_of_its_identity() {
    let (attested, attested_meta) = seal(ProviderTrust::ExternallyAttested).expect("attested");
    let (verified, verified_meta) = seal(ProviderTrust::ExactRequestVerified).expect("verified");
    let (plain, plain_meta) = TransportHandoffArtifact::seal(
        &offer(),
        &consumer(),
        "effect",
        TransportedInputPolicy::AcceptsTransported,
    )
    .expect("unlabelled");

    assert_eq!(attested_meta.provider_trust.as_deref(), Some("externally_attested"));
    assert_eq!(verified_meta.provider_trust.as_deref(), Some("verified_extension"));
    assert_eq!(plain_meta.provider_trust, None);
    assert_eq!(attested.provider_trust(), Some(ProviderTrust::ExternallyAttested));
    assert_eq!(verified.provider_trust(), Some(ProviderTrust::ExactRequestVerified));
    assert_eq!(plain.provider_trust(), None);

    // The same claim at a different trust level is a different artifact.
    assert_ne!(attested.identity(), verified.identity());
    assert_ne!(attested.identity(), plain.identity());
    // Neither is native target data.
    assert!(!verified.is_native_target_data());

    // Round trip through bytes keeps the label.
    let bytes = TransportHandoffArtifact::to_bytes_with_provider_trust(
        &offer(),
        &consumer(),
        "effect",
        TransportedInputPolicy::AcceptsTransported,
        ProviderTrust::ExactRequestVerified,
        "handoff",
    )
    .expect("bytes");
    let consumed = TransportHandoffArtifact::from_bytes(&bytes, Some(verified.identity()))
        .expect("consumes");
    assert_eq!(consumed.provider_trust(), Some(ProviderTrust::ExactRequestVerified));
}

#[test]
fn a_native_or_unverified_label_is_refused_and_an_edited_label_changes_identity() {
    for trust in [ProviderTrust::NativeLicensed, ProviderTrust::Unverified] {
        assert_eq!(
            seal(trust).expect_err("not an external level"),
            TransportHandoffArtifactError::UnsupportedSemantics("provider trust label")
        );
    }
    let (artifact, meta) = seal(ProviderTrust::ExternallyAttested).expect("attested");

    // Relabelling attested as verified without resealing is an identity mismatch.
    let mut upgraded = meta.clone();
    upgraded.provider_trust = Some("verified_extension".into());
    assert_eq!(
        TransportHandoffArtifact::from_meta(&upgraded, None).expect_err("identity changed"),
        TransportHandoffArtifactError::IdentityMismatch { field: "identity" }
    );
    // A resealed upgrade is a different artifact the consumer's retained identity refuses.
    TransportHandoffArtifact::reseal_identity(&mut upgraded);
    assert_eq!(
        TransportHandoffArtifact::from_meta(&upgraded, Some(artifact.identity()))
            .expect_err("retained identity differs"),
        TransportHandoffArtifactError::IdentityMismatch { field: "retained_identity" }
    );
    // Native is not a representable label.
    let mut native = meta;
    native.provider_trust = Some("native_licensed".into());
    TransportHandoffArtifact::reseal_identity(&mut native);
    assert_eq!(
        TransportHandoffArtifact::from_meta(&native, None).expect_err("native label"),
        TransportHandoffArtifactError::UnsupportedSemantics("provider trust label")
    );
}
