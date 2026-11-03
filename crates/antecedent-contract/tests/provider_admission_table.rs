//! Provider-source admission against the full factorial table of its specification.
//!
//! The specification: an external object is admitted as source evidence only if its quantity
//! names a population and a regime, the assumptions it relies on are named, and its trust meets
//! the route's requirement. Trust meets the requirement when the object holds an exact-request
//! verification receipt covering it (always), or is merely attested and attestation is accepted;
//! native support is never conferred; a receipt for another request never covers. The table
//! below crosses population (named or blank) x regime (named or blank) x assumptions (named or
//! empty) with seven trust/requirement pairs (56 declarations); the expected outcome of each is
//! the conjunction written out in `expected`, and an admitted source reports the trust it
//! actually has.

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


#[derive(Clone, Copy, Debug)]
enum Trust {
    VerifiedOwn,
    VerifiedOtherRequest,
    Attested,
    Native,
}

fn build(
    population: &str,
    regime: &str,
    assumptions: bool,
    trust: Trust,
    requirement: SourceTrustRequirement,
) -> ProviderTransportSource {
    // The object declares the trial-A quantity in its own named population and regime; the
    // source's quantity is that same quantity, possibly with the population or regime blanked.
    let object = law("trial_a", "do(a=1)", "req-1");
    let trust_state = match trust {
        Trust::VerifiedOwn => verified(&object),
        Trust::VerifiedOtherRequest => verified(&law("trial_a", "do(a=1)", "req-2")),
        Trust::Attested => attested(),
        Trust::Native => ExternalTrustState::NativeSupported,
    };
    let mut s = source(object, trust_state, requirement, EvidenceBasis::Undeclared);
    s.quantity = quantity(population, regime);
    if !assumptions {
        s.assumptions.clear();
    }
    s
}

/// The specification's conjunction.
fn expected(
    population_named: bool,
    regime_named: bool,
    assumptions: bool,
    trust: Trust,
    requirement: SourceTrustRequirement,
) -> bool {
    let trust_meets = match (trust, requirement) {
        (Trust::VerifiedOwn, _) => true,
        (Trust::Attested, SourceTrustRequirement::AttestationAccepted) => true,
        _ => false,
    };
    population_named && regime_named && assumptions && trust_meets
}

#[test]
fn admission_equals_the_specification_conjunction_over_the_full_factorial_table() {
    use SourceTrustRequirement::{AttestationAccepted as Att, ExactRequestVerified as Exact};
    let pairs = [
        (Trust::VerifiedOwn, Att),
        (Trust::VerifiedOwn, Exact),
        (Trust::Attested, Att),
        (Trust::Attested, Exact),
        (Trust::Native, Att),
        (Trust::Native, Exact),
        (Trust::VerifiedOtherRequest, Exact),
    ];
    let (mut admitted, mut refused) = (0, 0);
    for population_named in [true, false] {
        for regime_named in [true, false] {
            for assumptions in [true, false] {
                for (trust, requirement) in pairs {
                    let population = if population_named { "trial_a" } else { "  " };
                    let regime = if regime_named { "do(a=1)" } else { "" };
                    let result =
                        build(population, regime, assumptions, trust, requirement).admit();
                    let want = expected(
                        population_named,
                        regime_named,
                        assumptions,
                        trust,
                        requirement,
                    );
                    assert_eq!(
                        result.is_ok(),
                        want,
                        "population {population_named} regime {regime_named} assumptions \
                         {assumptions} trust {trust:?} requirement {requirement:?}: {:?}",
                        result.as_ref().err()
                    );
                    if let Ok(source) = result {
                        let reported = match trust {
                            Trust::VerifiedOwn => ProviderTrust::ExactRequestVerified,
                            _ => ProviderTrust::ExternallyAttested,
                        };
                        assert_eq!(source.trust(), reported, "{trust:?}");
                        assert!(!source.is_native_target_data());
                        admitted += 1;
                    } else {
                        refused += 1;
                    }
                }
            }
        }
    }
    // By hand: the all-named cell admits 3 of the 7 trust/requirement pairs (verified under
    // either requirement, attested under accepted attestation); every other cell admits none.
    assert_eq!((admitted, refused), (3, 53));
}
