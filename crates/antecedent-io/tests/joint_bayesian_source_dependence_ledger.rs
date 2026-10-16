//! Source dependence of the joint Bayesian transport model derived from the
//! evidence ledger: disjoint declared observations are independent; shared
//! observations and undeclared sources are never independent.

use antecedent_contract::evidence_ledger::{
    EvidenceLedger, ObservationId, SourceDeclaration, SourceId, SourceKind,
};
use antecedent_estimate::joint_bayesian_transport::SourceDependence;
use antecedent_io::joint_bayesian_transport_artifact::{
    prior_data_overlap_from_ledger, source_dependence_from_ledger,
};

fn add(ledger: &mut EvidenceLedger, label: &str, rows: Option<std::ops::Range<u32>>) -> SourceId {
    let mut declaration = SourceDeclaration::new(label, SourceKind::ObservationSet);
    if let Some(range) = rows {
        declaration = declaration
            .with_observations(range.map(|i| ObservationId::from_key("units", &i.to_le_bytes())));
    }
    ledger.add_source(declaration).unwrap()
}

#[test]
fn ledger_derives_independence_sharing_and_unknown_for_sources() {
    let mut ledger = EvidenceLedger::new();
    let a = add(&mut ledger, "a", Some(0..100));
    let b = add(&mut ledger, "b", Some(1_000..1_100));
    let target = add(&mut ledger, "target", Some(5_000..5_050));
    assert_eq!(
        source_dependence_from_ledger(&ledger, &[a, b, target]).unwrap(),
        SourceDependence::IndependentSamples
    );

    // Rows 50..150 are in both `a` and `shared`: counted twice if treated as independent.
    let shared = add(&mut ledger, "shared", Some(50..150));
    assert_eq!(
        source_dependence_from_ledger(&ledger, &[a, b, shared]).unwrap(),
        SourceDependence::OverlappingUnits
    );

    // A source with undeclared observations is unknown, never independent.
    let undeclared = add(&mut ledger, "undeclared", None);
    assert_eq!(
        source_dependence_from_ledger(&ledger, &[a, b, undeclared]).unwrap(),
        SourceDependence::Unknown
    );
    // Sharing outranks unknown.
    assert_eq!(
        source_dependence_from_ledger(&ledger, &[a, shared, undeclared]).unwrap(),
        SourceDependence::OverlappingUnits
    );
}

#[test]
fn ledger_prior_bank_overlap_is_derived_and_empty_groups_refuse() {
    let mut ledger = EvidenceLedger::new();
    let bank = add(&mut ledger, "bank", Some(0..100));
    let disjoint = add(&mut ledger, "likelihood", Some(500..600));
    let reused = add(&mut ledger, "reused", Some(90..190));
    let undeclared = add(&mut ledger, "undeclared", None);
    assert_eq!(
        prior_data_overlap_from_ledger(&ledger, &[bank], &[disjoint]).unwrap(),
        SourceDependence::IndependentSamples
    );
    assert_eq!(
        prior_data_overlap_from_ledger(&ledger, &[bank], &[disjoint, reused]).unwrap(),
        SourceDependence::OverlappingUnits
    );
    assert_eq!(
        prior_data_overlap_from_ledger(&ledger, &[bank], &[undeclared]).unwrap(),
        SourceDependence::Unknown
    );
    assert!(prior_data_overlap_from_ledger(&ledger, &[], &[disjoint]).is_err());
    assert!(source_dependence_from_ledger(&ledger, &[]).is_err());
}
