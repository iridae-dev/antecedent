//! Provider output as source evidence for transport, and a transported claim
//! handed to an external consumer, decided from declarations alone.
//!
//! Every expectation is a hand-enumerated outcome of the rule it names. No
//! provider and no callback runs here.

use std::collections::BTreeSet;

use antecedent_contract::callback_protocol::{
    CallbackDeclaration, CallbackDependency, CallbackInput, CallbackKind, CallbackOperation,
    CallbackTrust, DependencyDeclaration, DependencyKind, InputSupplier, ReplayContract,
};
use antecedent_contract::evidence_ledger::{
    CombinationRequest, EvidenceLedger, ObservationId, SourceDeclaration, SourceId, SourceKind,
    SynthesisOperation, TransportIdentity,
};
use antecedent_contract::interface_contract::{
    CalibrationRequirement, CalibrationStatus, ContractOperation, EvidenceBasis, SupportDomain,
};
use antecedent_contract::provider_transport::{
    ProviderTransportSource, SourceTrustRequirement, TRANSPORT_SOURCE_PORT, TransportedClaimOffer,
    TransportedHandoff, TransportedInputPolicy, connect_provider_source, hand_to_callback,
    hand_to_consumer, transport_source_consumer,
};
use antecedent_contract::{
    DistributionMeaning, ExternalCapability, ExternalRefusal, ExternalScientificObject,
    ExternalTrustState, LawProviderContract, ProviderObjectIdentity, ProviderTrust, QuantityRole,
    ScientificQuantity, VerificationProbe, VerificationProbeKind, verify_external_object,
};
use antecedent_core::SemanticDigest;

fn quantity(population: &str, regime: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: "schema:recovery".into(),
        variable_name: "recovery".into(),
        role: QuantityRole::Outcome,
        units: "days".into(),
        population_id: population.into(),
        regime_id: regime.into(),
        horizon: 0,
        functional_id: "outcome".into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    }
}

fn identity(request: &str) -> ProviderObjectIdentity {
    ProviderObjectIdentity {
        provider_id: "lab".into(),
        object_id: "trial-a-law".into(),
        version_id: "1.0".into(),
        snapshot_id: "snap-1".into(),
        request_id: request.into(),
    }
}

fn law(population: &str, regime: &str, request: &str) -> ExternalScientificObject {
    ExternalScientificObject::Law(LawProviderContract {
        identity: identity(request),
        quantities: vec![quantity(population, regime)],
        meaning: DistributionMeaning::InterventionalPredictive,
        capabilities: vec![ExternalCapability::Mean, ExternalCapability::Sample],
    })
}

/// Probes whose observed values equal independently known truth.
fn probes() -> Vec<VerificationProbe> {
    use VerificationProbeKind as P;
    [P::KnownTruth, P::Shape, P::Support, P::Moments, P::SeededBehavior]
        .into_iter()
        .map(|kind| VerificationProbe { kind, observed: 2.0, expected: 2.0, tolerance: 1e-9 })
        .collect()
}

fn verified(object: &ExternalScientificObject) -> ExternalTrustState {
    ExternalTrustState::ExactRequestVerified(Box::new(
        verify_external_object(object, &probes()).expect("probes agree with the known truth"),
    ))
}

fn attested() -> ExternalTrustState {
    ExternalTrustState::ExternallyAttested { attestor: "lab".into() }
}

fn digest(byte: u8) -> SemanticDigest {
    SemanticDigest::from_bytes([byte; 32])
}

fn transport(sources: &[&str], target: &str) -> TransportIdentity {
    TransportIdentity {
        source_populations: sources.iter().map(|s| (*s).to_owned()).collect(),
        target_population: target.to_owned(),
        selection_diagram: digest(7),
        assumptions: vec!["recovery mechanism invariant across populations".to_owned()],
    }
}

fn source(
    object: ExternalScientificObject,
    trust: ExternalTrustState,
    requirement: SourceTrustRequirement,
    evidence: EvidenceBasis,
) -> ProviderTransportSource {
    ProviderTransportSource {
        component: "trial_a_provider".into(),
        object,
        trust,
        quantity: quantity("trial_a", "do(a=1)"),
        semantic_identity: "estimand:recovery_under_treatment".into(),
        support: SupportDomain::Real,
        assumptions: vec!["recovery mechanism invariant across populations".into()],
        evidence,
        requirement,
    }
}

fn verified_source(evidence: EvidenceBasis) -> ProviderTransportSource {
    let object = law("trial_a", "do(a=1)", "req-1");
    let trust = verified(&object);
    source(object, trust, SourceTrustRequirement::ExactRequestVerified, evidence)
}

fn observations(range: std::ops::Range<u32>) -> Vec<ObservationId> {
    range.map(|i| ObservationId::from_key("cohort", &i.to_le_bytes())).collect()
}

fn check(refusal: &ExternalRefusal, code: &str, detail: &str) {
    assert_eq!(refusal.code, code, "{refusal:?}");
    assert_eq!(refusal.detail, detail, "{refusal:?}");
    assert!(refusal.remedy.is_some(), "every refusal states a remedy: {refusal:?}");
}

// -- provider output as source evidence ---------------------------------------------------------------

/// A verified provider's law of the trial-A recovery is admitted, registered
/// in the ledger as an external claim over the trial data it rests on,
/// connected to a transport route's source consumer, and carried to the target
/// as a claim with its transport identity next to disjoint native target data.
#[test]
fn a_verified_provider_output_is_source_evidence_that_reaches_a_transported_claim() {
    let mut ledger = EvidenceLedger::new();
    let trial = ledger
        .add_source(
            SourceDeclaration::new("trial_a_rows", SourceKind::ObservationSet)
                .with_observations(observations(100..160)),
        )
        .expect("trial data");
    let native = ledger
        .add_source(
            SourceDeclaration::new("target_cohort", SourceKind::ObservationSet)
                .with_observations(observations(0..30)),
        )
        .expect("native target data");

    let declared = verified_source(EvidenceBasis::Declared([trial].into_iter().collect()));
    let admitted = declared.admit().expect("a verified, declared provider is admitted");
    assert_eq!(admitted.trust(), ProviderTrust::ExactRequestVerified);
    assert!(!admitted.is_native_target_data());
    assert_eq!(admitted.population(), "trial_a");
    assert_eq!(admitted.regime(), "do(a=1)");
    // Mean -> mean, Sample -> draws; sampling does not supply a law.
    let expected: BTreeSet<ContractOperation> =
        [ContractOperation::Mean, ContractOperation::Draws].into_iter().collect();
    assert_eq!(admitted.operations(), expected);

    // The projection is an interface contract; it is unmeasured, never measured.
    let contract = admitted.interface_contract();
    contract.validate().expect("the projection is a valid interface contract");
    let port = contract.produced("provider_source").expect("one produced port");
    assert_eq!(port.calibration, CalibrationStatus::Unmeasured);
    assert!(port.provenance.contains("external_provider"));

    // Connection to the route's consumer: same population, regime, operations.
    let consumer = transport_source_consumer(
        "transport_route",
        quantity("trial_a", "do(a=1)"),
        "estimand:recovery_under_treatment",
        SupportDomain::Real,
        [ContractOperation::Mean],
    );
    let connection = connect_provider_source(&admitted, &consumer, TRANSPORT_SOURCE_PORT)
        .expect("population, regime, operations and provenance agree");
    assert_eq!(connection.operations, [ContractOperation::Mean].into_iter().collect());
    assert_eq!(connection.calibration, CalibrationStatus::Unmeasured);
    assert_eq!(connection.evidence, EvidenceBasis::Declared([trial].into_iter().collect()));

    // Ledger: external claim, derived from the trial rows, no transport identity yet.
    let provider = ledger.add_source(admitted.ledger_declaration()).expect("provider source");
    let record = ledger.source(provider).expect("registered");
    assert_eq!(record.declaration.kind, SourceKind::ExternalClaim);
    assert!(record.declaration.transport.is_none());

    // Bound to the transport: trial_a is a declared source population.
    let identity = transport(&["trial_a", "trial_b"], "target");
    admitted.require_bound_to(&identity).expect("source population and assumptions carried");

    // The claim carried to the target rests on the provider source.
    let offer = TransportedClaimOffer {
        component: "transport_result".into(),
        port: "effect".into(),
        quantity: quantity("target", "do(a=1)"),
        semantic_identity: "estimand:recovery_under_treatment".into(),
        support: SupportDomain::Real,
        operations: [ContractOperation::Mean].into_iter().collect(),
        uncertainty: None,
        transport: identity.clone(),
        evidence: [provider].into_iter().collect(),
    };
    offer.validate().expect("a well formed transported claim");
    let claim =
        ledger.add_source(offer.ledger_declaration("effect_in_target")).expect("claim source");
    let result = ledger
        .combine(CombinationRequest {
            label: "transport".to_owned(),
            operation: SynthesisOperation::CausalTransport {
                transported: vec![claim],
                native: vec![native],
            },
            replaces: None,
        })
        .expect("the claim is derived from trial data disjoint from the native cohort");
    assert_eq!(
        ledger.result(result).expect("record").transport_identities,
        vec![identity.digest()]
    );
}

/// The same provider output offered as native target data double counts: the
/// transported claim derived from the trial rows cannot be listed as native.
#[test]
fn a_provider_source_is_not_native_target_data() {
    let mut ledger = EvidenceLedger::new();
    let trial = ledger
        .add_source(
            SourceDeclaration::new("trial_a_rows", SourceKind::ObservationSet)
                .with_observations(observations(100..160)),
        )
        .expect("trial data");
    let admitted = verified_source(EvidenceBasis::Declared([trial].into_iter().collect()))
        .admit()
        .expect("admitted");
    let provider = ledger.add_source(admitted.ledger_declaration()).expect("provider source");
    let offer = TransportedClaimOffer {
        component: "transport_result".into(),
        port: "effect".into(),
        quantity: quantity("target", "do(a=1)"),
        semantic_identity: "estimand:recovery_under_treatment".into(),
        support: SupportDomain::Real,
        operations: [ContractOperation::Mean].into_iter().collect(),
        uncertainty: None,
        transport: transport(&["trial_a"], "target"),
        evidence: [provider].into_iter().collect(),
    };
    let claim = ledger.add_source(offer.ledger_declaration("claim")).expect("claim");
    let error = ledger
        .combine(CombinationRequest {
            label: "transport".to_owned(),
            operation: SynthesisOperation::CausalTransport {
                transported: vec![claim],
                native: vec![trial],
            },
            replaces: None,
        })
        .expect_err("trial rows are both native and the claim's basis");
    assert_eq!(error.code, "evidence_double_counted");
}

#[test]
fn a_missing_population_or_regime_refuses_before_any_connection() {
    let mut no_population = verified_source(EvidenceBasis::Undeclared);
    no_population.quantity.population_id = "  ".into();
    check(
        &no_population.admit().expect_err("population"),
        "interface_contract_invalid",
        "provider_transport.population_missing",
    );
    let mut no_regime = verified_source(EvidenceBasis::Undeclared);
    no_regime.quantity.regime_id = String::new();
    check(
        &no_regime.admit().expect_err("regime"),
        "interface_contract_invalid",
        "provider_transport.regime_missing",
    );
    let mut no_assumptions = verified_source(EvidenceBasis::Undeclared);
    no_assumptions.assumptions.clear();
    check(
        &no_assumptions.admit().expect_err("assumptions"),
        "interface_contract_invalid",
        "provider_transport.assumptions_missing",
    );
}

#[test]
fn attestation_does_not_stand_in_for_required_verification() {
    let object = law("trial_a", "do(a=1)", "req-1");
    let required = source(
        object.clone(),
        attested(),
        SourceTrustRequirement::ExactRequestVerified,
        EvidenceBasis::Undeclared,
    );
    let refusal = required.admit().expect_err("attestation where verification is required");
    check(&refusal, "external_verification_failed", "provider_transport.verification_required");
    assert_eq!(refusal.expected.as_deref(), Some("verified_extension"));
    assert_eq!(refusal.supplied.as_deref(), Some("externally_attested"));

    // Where attestation is accepted the source stays attested, never verified.
    let accepted = source(
        object,
        attested(),
        SourceTrustRequirement::AttestationAccepted,
        EvidenceBasis::Undeclared,
    )
    .admit()
    .expect("attestation accepted");
    assert_eq!(accepted.trust(), ProviderTrust::ExternallyAttested);
}

#[test]
fn a_receipt_for_another_request_or_a_native_licence_is_refused() {
    let first = law("trial_a", "do(a=1)", "req-1");
    let receipt_for_first = verified(&first);
    let second = law("trial_a", "do(a=1)", "req-2");
    let wrong = source(
        second,
        receipt_for_first,
        SourceTrustRequirement::ExactRequestVerified,
        EvidenceBasis::Undeclared,
    );
    check(
        &wrong.admit().expect_err("receipt covers another request"),
        "external_verification_failed",
        "provider_transport.receipt_does_not_cover",
    );
    let native = source(
        first,
        ExternalTrustState::NativeLicensed,
        SourceTrustRequirement::AttestationAccepted,
        EvidenceBasis::Undeclared,
    );
    check(
        &native.admit().expect_err("native licence"),
        "interface_contract_invalid",
        "provider_transport.native_trust_claimed",
    );
}

#[test]
fn a_quantity_the_provider_does_not_declare_or_a_foreign_population_is_refused() {
    let mut other_regime = verified_source(EvidenceBasis::Undeclared);
    other_regime.quantity.regime_id = "do(a=0)".into();
    check(
        &other_regime.admit().expect_err("regime not declared by the object"),
        "external_binding_mismatch",
        "provider_transport.quantity_not_declared",
    );

    // A source whose population is not a declared source population of the
    // selection diagram does not belong to that transport.
    let admitted = verified_source(EvidenceBasis::Undeclared).admit().expect("admitted");
    check(
        &admitted
            .require_bound_to(&transport(&["trial_b"], "target"))
            .expect_err("trial_a is not a source of this transport"),
        "interface_contract_quantity_mismatch",
        "provider_transport.population_not_a_source",
    );
    let mut without_assumption = transport(&["trial_a"], "target");
    without_assumption.assumptions = vec!["something else".to_owned()];
    check(
        &admitted
            .require_bound_to(&without_assumption)
            .expect_err("the provider's assumption is not carried"),
        "interface_contract_assumption_mismatch",
        "provider_transport.assumptions_not_carried",
    );

    // A route that needs another population is not served by this source.
    let consumer = transport_source_consumer(
        "transport_route",
        quantity("trial_b", "do(a=1)"),
        "estimand:recovery_under_treatment",
        SupportDomain::Real,
        [ContractOperation::Mean],
    );
    let refusals = connect_provider_source(&admitted, &consumer, TRANSPORT_SOURCE_PORT)
        .expect_err("population differs");
    assert!(refusals.iter().any(|r| r.code == "interface_contract_quantity_mismatch"));
}

#[test]
fn a_changed_request_regime_trust_or_assumption_changes_the_source_identity() {
    let base =
        verified_source(EvidenceBasis::Undeclared).admit().expect("admitted").canonical_identity();
    let same = verified_source(EvidenceBasis::Undeclared).admit().expect("admitted");
    assert_eq!(base, same.canonical_identity());

    let mut assumption = verified_source(EvidenceBasis::Undeclared);
    assumption.assumptions = vec!["recovery mechanism invariant across populations!".into()];
    assert_ne!(base, assumption.admit().expect("admitted").canonical_identity());

    let object = law("trial_a", "do(a=1)", "req-9");
    let other_request = source(
        object.clone(),
        verified(&object),
        SourceTrustRequirement::ExactRequestVerified,
        EvidenceBasis::Undeclared,
    );
    assert_ne!(base, other_request.admit().expect("admitted").canonical_identity());

    let attested_same_shape = source(
        law("trial_a", "do(a=1)", "req-1"),
        attested(),
        SourceTrustRequirement::AttestationAccepted,
        EvidenceBasis::Undeclared,
    );
    assert_ne!(base, attested_same_shape.admit().expect("admitted").canonical_identity());
}

// -- a transported claim handed to an external consumer ------------------------------------------------

fn evidence_id(byte: u8) -> SourceId {
    SourceId::from_digest(digest(byte))
}

fn offer() -> TransportedClaimOffer {
    TransportedClaimOffer {
        component: "transport_result".into(),
        port: "effect".into(),
        quantity: quantity("target", "do(a=1)"),
        semantic_identity: "estimand:recovery_under_treatment".into(),
        support: SupportDomain::Real,
        operations: [ContractOperation::Mean].into_iter().collect(),
        uncertainty: None,
        transport: transport(&["trial_a", "trial_b"], "target"),
        evidence: [evidence_id(5)].into_iter().collect(),
    }
}

fn consumer_callback(input_population: &str, supplier: InputSupplier) -> CallbackDeclaration {
    CallbackDeclaration {
        component: "downstream_model".into(),
        kind: CallbackKind::Law,
        identity: identity("req-consumer"),
        quantity: quantity("target", "do(a=1)"),
        semantic_identity: "estimand:downstream_outcome".into(),
        meaning: DistributionMeaning::InterventionalPredictive,
        support: SupportDomain::Real,
        operations: [CallbackOperation::Mean].into_iter().collect(),
        update: None,
        inputs: vec![CallbackInput {
            name: "effect".into(),
            quantity: quantity(input_population, "do(a=1)"),
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

/// The consumer receives the claim with its source populations, selection
/// diagram and assumptions, and the record never calls it native target data.
#[test]
fn a_transported_claim_reaches_a_callback_input_with_its_transport_identity() {
    let offer = offer();
    let callback = consumer_callback("target", InputSupplier::External);
    let handoff =
        hand_to_callback(&offer, &callback, "effect", TransportedInputPolicy::AcceptsTransported)
            .expect("population, regime, units, operation and provenance agree");
    assert_eq!(handoff.standing(), "transported_claim");
    assert_eq!(handoff.standing(), TransportedHandoff::STANDING);
    assert!(!handoff.is_native_target_data());
    assert_eq!(handoff.transport(), &offer.transport);
    assert_eq!(
        handoff.transport().source_populations,
        vec!["trial_a".to_owned(), "trial_b".to_owned()]
    );
    assert_eq!(handoff.transport().selection_diagram, digest(7));
    assert_eq!(handoff.connection().consumer, "downstream_model");
    assert_eq!(handoff.connection().consumed_port, "effect");
    // A mean alone is point-only; the handoff does not raise it.
    assert_eq!(handoff.connection().calibration, CalibrationStatus::PointOnly);
    assert_eq!(
        handoff.connection().evidence,
        EvidenceBasis::Declared([evidence_id(5)].into_iter().collect())
    );

    // Deterministic identity; a changed selection diagram at unchanged shape
    // changes it.
    let again =
        hand_to_callback(&offer, &callback, "effect", TransportedInputPolicy::AcceptsTransported)
            .expect("same");
    assert_eq!(handoff.identity(), again.identity());
    let mut changed = offer.clone();
    changed.transport.selection_diagram = digest(8);
    let other =
        hand_to_callback(&changed, &callback, "effect", TransportedInputPolicy::AcceptsTransported)
            .expect("same shape");
    assert_ne!(handoff.identity(), other.identity());
}

#[test]
fn a_consumer_that_needs_native_target_data_refuses_a_transported_claim() {
    let callback = consumer_callback("target", InputSupplier::External);
    let refusals =
        hand_to_callback(&offer(), &callback, "effect", TransportedInputPolicy::NativeTargetOnly)
            .expect_err("native target data required");
    assert_eq!(refusals.len(), 1);
    check(
        &refusals[0],
        "interface_contract_assumption_mismatch",
        "provider_transport.native_target_required",
    );
    assert_eq!(refusals[0].supplied.as_deref(), Some("transported_claim"));
}

#[test]
fn a_claim_without_its_identity_or_outside_the_target_is_refused() {
    let callback = consumer_callback("target", InputSupplier::External);
    let policy = TransportedInputPolicy::AcceptsTransported;

    let mut no_sources = offer();
    no_sources.transport.source_populations.clear();
    let refusals =
        hand_to_callback(&no_sources, &callback, "effect", policy).expect_err("no sources");
    check(
        &refusals[0],
        "interface_contract_invalid",
        "provider_transport.transport_identity_incomplete",
    );

    let mut target_is_source = offer();
    target_is_source.transport.source_populations.push("target".into());
    let refusals = hand_to_callback(&target_is_source, &callback, "effect", policy)
        .expect_err("nothing is transported");
    check(
        &refusals[0],
        "interface_contract_invalid",
        "provider_transport.transport_identity_incomplete",
    );

    let mut native_coordinate = offer();
    native_coordinate.quantity.population_id = "trial_a".into();
    let refusals = hand_to_callback(&native_coordinate, &callback, "effect", policy)
        .expect_err("the claim is not about the target");
    check(
        &refusals[0],
        "interface_contract_quantity_mismatch",
        "provider_transport.claim_not_in_target",
    );

    let mut no_evidence = offer();
    no_evidence.evidence.clear();
    let refusals =
        hand_to_callback(&no_evidence, &callback, "effect", policy).expect_err("no evidence");
    check(&refusals[0], "interface_contract_invalid", "provider_transport.evidence_missing");
}

#[test]
fn the_connection_still_refuses_a_different_population_a_calibration_demand_and_a_callback_input() {
    let policy = TransportedInputPolicy::AcceptsTransported;

    // The consumer's input is about another population than the claim's target.
    let elsewhere = consumer_callback("trial_b", InputSupplier::External);
    let refusals =
        hand_to_callback(&offer(), &elsewhere, "effect", policy).expect_err("population");
    assert!(refusals.iter().any(|r| r.code == "interface_contract_quantity_mismatch"));

    // A consumer that relies on a measured calibration record is not served by
    // a transported claim, which never holds one.
    let callback = consumer_callback("target", InputSupplier::External);
    let mut contract = callback.interface_contract();
    // A measured requirement is only well formed beside an interval or posterior.
    contract.consumes[0].calibration = CalibrationRequirement::Measured;
    contract.consumes[0].operations.insert(ContractOperation::Interval);
    contract.consumes[0].uncertainty = Some(
        antecedent_contract::interface_contract::UncertaintyMeaning::new(
            DistributionMeaning::EstimatorSampling,
        ),
    );
    let refusals = hand_to_consumer(&offer(), &contract, "effect", policy)
        .expect_err("transport supplies no measured calibration");
    assert!(refusals.iter().any(|r| r.code == "interface_contract_calibration_not_held"));

    // An input another callback supplies is not a claim from outside the graph.
    let mut nested = consumer_callback(
        "target",
        InputSupplier::Callback {
            identity: "a".repeat(64),
            evidence: EvidenceBasis::Declared(BTreeSet::new()),
        },
    );
    nested.kind = CallbackKind::Nested;
    let refusals = hand_to_callback(&offer(), &nested, "effect", policy)
        .expect_err("the input is supplied by a callback");
    check(
        &refusals[0],
        "interface_contract_assumption_mismatch",
        "provider_transport.input_not_external",
    );

    // An unknown consumed port is reported by the connection check.
    let refusals = hand_to_callback(&offer(), &callback, "missing", policy).expect_err("port");
    assert_eq!(refusals.len(), 1);
}
