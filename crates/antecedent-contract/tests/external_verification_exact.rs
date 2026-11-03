//! External-object verification against hand-set probe tolerances.
//!
//! A probe passes when |observed - expected| <= tolerance. Every case below fixes the three
//! numbers by hand, using values exact in binary so the boundary is not a rounding question
//! (0.25 and 0.75 differ by exactly 0.5). A receipt covers only the exact request it verified:
//! changing any one of the five identity fields, or the quantity, leaves it uncovering.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::{
    DistributionMeaning, ExternalCapability, ExternalScientificObject, LawProviderContract,
    ProviderObjectIdentity, QuantityRole, ScientificQuantity, VerificationProbe,
    VerificationProbeKind, verify_external_object,
};

const KINDS: [VerificationProbeKind; 5] = [
    VerificationProbeKind::KnownTruth,
    VerificationProbeKind::Shape,
    VerificationProbeKind::Support,
    VerificationProbeKind::Moments,
    VerificationProbeKind::SeededBehavior,
];

fn quantity(regime: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: "y".into(),
        variable_name: "y".into(),
        role: QuantityRole::Outcome,
        units: "units".into(),
        population_id: "target".into(),
        regime_id: regime.into(),
        horizon: 0,
        functional_id: "outcome".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

fn identity() -> ProviderObjectIdentity {
    ProviderObjectIdentity {
        provider_id: "lab".into(),
        object_id: "law-1".into(),
        version_id: "v1".into(),
        snapshot_id: "snap-1".into(),
        request_id: "req-1".into(),
    }
}

fn law(identity: ProviderObjectIdentity, regime: &str) -> ExternalScientificObject {
    ExternalScientificObject::Law(LawProviderContract {
        identity,
        quantities: vec![quantity(regime)],
        meaning: DistributionMeaning::InterventionalPredictive,
        capabilities: vec![ExternalCapability::Sample, ExternalCapability::Mean],
    })
}

fn probes(observed: f64, expected: f64, tolerance: f64) -> Vec<VerificationProbe> {
    KINDS
        .into_iter()
        .map(|kind| VerificationProbe { kind, observed, expected, tolerance })
        .collect()
}

/// Probes at the tolerance boundary pass (0.25 vs 0.75 within 0.5, exactly), and a tolerance
/// just under the gap refuses, for each of the five probe kinds taken alone as the deviant.
#[test]
fn a_probe_passes_exactly_when_its_gap_is_within_its_tolerance() {
    let object = law(identity(), "do(a=1)");
    assert!(verify_external_object(&object, &probes(0.25, 0.75, 0.5)).is_ok());
    assert!(verify_external_object(&object, &probes(0.75, 0.25, 0.5)).is_ok());
    assert!(verify_external_object(&object, &probes(0.25, 0.75, 0.4999)).is_err());
    assert!(verify_external_object(&object, &probes(1.0, 1.0, 0.0)).is_ok());
    for deviant in 0..KINDS.len() {
        let mut set = probes(1.0, 1.0, 0.0);
        set[deviant] = VerificationProbe {
            kind: KINDS[deviant],
            observed: 1.0,
            expected: 1.25,
            tolerance: 0.125,
        };
        assert!(
            verify_external_object(&object, &set).is_err(),
            "a gap of 0.25 against 0.125 in probe {deviant} must refuse"
        );
        set[deviant].tolerance = 0.25;
        assert!(verify_external_object(&object, &set).is_ok(), "gap 0.25 within 0.25: {deviant}");
    }
}

/// A receipt covers the verified request and none of the six neighbours that differ from it in
/// exactly one identity field or in the quantity.
#[test]
fn a_receipt_covers_only_the_exact_request_it_verified() {
    let object = law(identity(), "do(a=1)");
    let receipt = verify_external_object(&object, &probes(1.0, 1.0, 0.0)).unwrap();
    assert!(receipt.covers(&object));
    let edits: [fn(&mut ProviderObjectIdentity); 5] = [
        |i| i.provider_id = "other".into(),
        |i| i.object_id = "law-2".into(),
        |i| i.version_id = "v2".into(),
        |i| i.snapshot_id = "snap-2".into(),
        |i| i.request_id = "req-2".into(),
    ];
    for (position, edit) in edits.into_iter().enumerate() {
        let mut changed = identity();
        edit(&mut changed);
        assert!(!receipt.covers(&law(changed, "do(a=1)")), "identity field {position}");
    }
    assert!(!receipt.covers(&law(identity(), "do(a=0)")), "another quantity");
}
