//! A transport identity must state its assumptions and carry its claim to a
//! population that is not one of its sources, and a provider's source must
//! belong to the transport it stands behind: the ledger, the connection and the
//! handoff all refuse the same defects with the same detail.

use antecedent_contract::evidence_ledger::{
    EvidenceLedger, SourceDeclaration, SourceKind, TransportIdentity,
};
use antecedent_contract::interface_contract::{ContractOperation, EvidenceBasis, SupportDomain};
use antecedent_contract::provider_transport::{
    ProviderTransportSource, SourceTrustRequirement, TRANSPORT_SOURCE_PORT, TransportedClaimOffer,
    TransportedInputPolicy, connect_bound_provider_source, hand_to_consumer_from_sources,
    transport_source_consumer,
};
use antecedent_contract::{
    DistributionMeaning, ExternalCapability, ExternalScientificObject, ExternalTrustState,
    LawProviderContract, ProviderObjectIdentity, QuantityRole, ScientificQuantity,
};
use antecedent_core::SemanticDigest;

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

fn transport(sources: &[&str], target: &str, assumptions: &[&str]) -> TransportIdentity {
    TransportIdentity {
        source_populations: sources.iter().map(|s| (*s).to_owned()).collect(),
        target_population: target.to_owned(),
        selection_diagram: digest(7),
        assumptions: assumptions.iter().map(|s| (*s).to_owned()).collect(),
    }
}

const INVARIANT: &str = "recovery mechanism invariant across populations";

fn provider(population: &str, assumption: &str) -> ProviderTransportSource {
    ProviderTransportSource {
        component: "provider".into(),
        object: ExternalScientificObject::Law(LawProviderContract {
            identity: ProviderObjectIdentity {
                provider_id: "lab".into(),
                object_id: "law".into(),
                version_id: "1.0".into(),
                snapshot_id: "snap".into(),
                request_id: "req".into(),
            },
            quantities: vec![quantity(population)],
            meaning: DistributionMeaning::InterventionalPredictive,
            capabilities: vec![ExternalCapability::Mean],
        }),
        trust: ExternalTrustState::ExternallyAttested { attestor: "lab".into() },
        quantity: quantity(population),
        semantic_identity: "estimand:recovery_under_treatment".into(),
        support: SupportDomain::Real,
        assumptions: vec![assumption.into()],
        evidence: EvidenceBasis::Declared(std::collections::BTreeSet::new()),
        requirement: SourceTrustRequirement::AttestationAccepted,
    }
}

fn claim_declaration(identity: TransportIdentity) -> SourceDeclaration {
    SourceDeclaration::new("claim", SourceKind::ExternalClaim).with_transport(identity)
}

#[test]
fn the_ledger_refuses_a_transport_without_assumptions() {
    let mut ledger = EvidenceLedger::new();
    let error = ledger
        .add_source(claim_declaration(transport(&["trial_a"], "target", &[])))
        .expect_err("no assumption licenses nothing");
    assert_eq!(error.code, "invalid_argument");
    assert_eq!(error.detail, "evidence_ledger.invalid_declaration");
    // The same identity with one assumption registers.
    ledger
        .add_source(claim_declaration(transport(&["trial_a"], "target", &[INVARIANT])))
        .expect("a stated assumption is accepted");
}

#[test]
fn the_ledger_refuses_a_target_that_is_one_of_its_sources() {
    let mut ledger = EvidenceLedger::new();
    let error = ledger
        .add_source(claim_declaration(transport(&["trial_a", "target"], "target", &[INVARIANT])))
        .expect_err("a population is not transported to itself");
    assert_eq!(error.code, "invalid_argument");
    assert_eq!(error.detail, "evidence_ledger.invalid_declaration");
    assert!(error.message.contains("source populations"), "{}", error.message);
}

#[test]
fn a_provider_of_another_population_is_not_connected_to_the_transport() {
    let consumer = transport_source_consumer(
        "route",
        quantity("trial_c"),
        "estimand:recovery_under_treatment",
        SupportDomain::Real,
        [ContractOperation::Mean],
    );
    let identity = transport(&["trial_a", "trial_b"], "target", &[INVARIANT]);

    // trial_c is not a source population of the transport: refused before connecting.
    let admitted = provider("trial_c", INVARIANT).admit().expect("admits on its own");
    let refusals =
        connect_bound_provider_source(&admitted, &identity, &consumer, TRANSPORT_SOURCE_PORT)
            .expect_err("population not a source");
    assert_eq!(refusals[0].detail, "provider_transport.population_not_a_source");
    assert_eq!(refusals[0].expected.as_deref(), Some("trial_a,trial_b"));
    assert_eq!(refusals[0].supplied.as_deref(), Some("trial_c"));

    // The right population relying on an assumption the transport does not carry.
    let consumer_a = transport_source_consumer(
        "route",
        quantity("trial_a"),
        "estimand:recovery_under_treatment",
        SupportDomain::Real,
        [ContractOperation::Mean],
    );
    let uncarried = provider("trial_a", "treatment effect homogeneous").admit().expect("admits");
    let refusals =
        connect_bound_provider_source(&uncarried, &identity, &consumer_a, TRANSPORT_SOURCE_PORT)
            .expect_err("assumption not carried");
    assert_eq!(refusals[0].detail, "provider_transport.assumptions_not_carried");
    assert_eq!(refusals[0].offending.as_deref(), Some("treatment effect homogeneous"));

    // A matching provider connects.
    let bound = provider("trial_a", INVARIANT).admit().expect("admits");
    connect_bound_provider_source(&bound, &identity, &consumer_a, TRANSPORT_SOURCE_PORT)
        .expect("a source population carrying its assumptions connects");
}

#[test]
fn a_handoff_names_only_providers_of_its_transport() {
    let identity = transport(&["trial_a", "trial_b"], "target", &[INVARIANT]);
    let offer = TransportedClaimOffer {
        component: "transport_result".into(),
        port: "effect".into(),
        quantity: quantity("target"),
        semantic_identity: "estimand:recovery_under_treatment".into(),
        support: SupportDomain::Real,
        operations: [ContractOperation::Mean].into_iter().collect(),
        uncertainty: None,
        transport: identity,
        evidence: [antecedent_contract::evidence_ledger::SourceId::from_digest(digest(5))]
            .into_iter()
            .collect(),
    };
    let consumer = transport_source_consumer(
        "downstream",
        quantity("target"),
        "estimand:recovery_under_treatment",
        SupportDomain::Real,
        [ContractOperation::Mean],
    );
    // A transported claim is a claim about the target; the downstream consumer's
    // port is a different shape, so only the provider binding is asserted here.
    let stranger = provider("trial_c", INVARIANT).admit().expect("admits");
    let refusals = hand_to_consumer_from_sources(
        &offer,
        &[&stranger],
        &consumer,
        TRANSPORT_SOURCE_PORT,
        TransportedInputPolicy::AcceptsTransported,
    )
    .expect_err("a provider of another population stands behind no claim of this transport");
    assert_eq!(refusals[0].detail, "provider_transport.population_not_a_source");
}
