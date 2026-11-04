//! Ledger-decided dependence against hand-counted row sets.
//!
//! Four sources hold the rows a = {1,2,3}, b = {3,4}, c = {7,8}, d = {4,9}. By hand the only
//! pairs that share a row are (a, b) on row 3 and (b, d) on row 4; c shares nothing with anyone.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::evidence_ledger::{
    EvidenceLedger, ObservationId, SourceDeclaration, SourceId, SourceKind,
};
use antecedent_contract::interface_contract::{
    ComponentConnection, ConsumedPort, ContractOperation, DependenceRequirement, InterfaceContract,
    ProducedPort, SupportDomain, check_composition_with_ledger,
};
use antecedent_contract::{
    InputAlignment, InputRole, PropagationDependence, PropagationInput, PropagationOperation,
    PropagationRequest, QuantityRole, RequestedReading, ScientificQuantity,
    check_propagation_with_ledger,
};

fn source(ledger: &mut EvidenceLedger, label: &str, rows: &[u8]) -> SourceId {
    ledger
        .add_source(SourceDeclaration::new(label, SourceKind::ObservationSet).with_observations(
            rows.iter().copied().map(|r| ObservationId::from_key("trial", &[r])),
        ))
        .expect("source")
}

struct World {
    ledger: EvidenceLedger,
    a: SourceId,
    b: SourceId,
    c: SourceId,
    d: SourceId,
}

fn world() -> World {
    let mut ledger = EvidenceLedger::new();
    let a = source(&mut ledger, "a", &[1, 2, 3]);
    let b = source(&mut ledger, "b", &[3, 4]);
    let c = source(&mut ledger, "c", &[7, 8]);
    let d = source(&mut ledger, "d", &[4, 9]);
    World { ledger, a, b, c, d }
}

fn input(id: &str, evidence: SourceId) -> PropagationInput {
    PropagationInput {
        id: id.into(),
        evidence_id: evidence.to_hex(),
        role: InputRole::Estimate,
        alignment: InputAlignment::MomentsOnly,
    }
}

fn request(inputs: Vec<PropagationInput>, dependence: PropagationDependence) -> PropagationRequest {
    PropagationRequest {
        operation: PropagationOperation::Sum,
        inputs,
        dependence,
        reading: RequestedReading::Point,
    }
}

/// Under a declared covariance the plan reports exactly the evidence that is shared: the
/// sources of the overlapping pairs (a, b) and (b, d), and not c.
#[test]
fn the_plan_names_exactly_the_sources_that_share_rows() {
    let w = world();
    let plan = check_propagation_with_ledger(
        &request(
            vec![input("x", w.a), input("y", w.b), input("z", w.c), input("w", w.d)],
            PropagationDependence::Covariance { id: "cov-1".into() },
        ),
        &w.ledger,
    )
    .expect("a declared covariance carries the overlaps");
    let mut expected = vec![w.a.to_hex(), w.b.to_hex(), w.d.to_hex()];
    expected.sort();
    assert_eq!(plan.shared_evidence, expected);
    assert!(!plan.shared_evidence.contains(&w.c.to_hex()));
}

/// Declared independence is decided pair by pair from the rows: (a, c) is accepted with nothing
/// shared, and adding b (which shares row 3 with a) or pairing b with d (row 4) refuses.
#[test]
fn declared_independence_is_accepted_only_for_the_row_disjoint_inputs() {
    let w = world();
    let ok = check_propagation_with_ledger(
        &request(
            vec![input("x", w.a), input("z", w.c)],
            PropagationDependence::DeclaredIndependent,
        ),
        &w.ledger,
    )
    .expect("a and c share no row");
    assert!(ok.shared_evidence.is_empty());
    for (first, second) in [(w.a, w.b), (w.b, w.d)] {
        let error = check_propagation_with_ledger(
            &request(
                vec![input("x", first), input("y", second)],
                PropagationDependence::DeclaredIndependent,
            ),
            &w.ledger,
        )
        .expect_err("a row is shared");
        assert_eq!(
            error.detail,
            antecedent_core::reason_details::joint_law_propagation::SHARED_EVIDENCE_DECLARED_INDEPENDENT
        );
    }
    // Three-way: a, c and d are pairwise row-disjoint, so independence holds for the triple.
    let triple = check_propagation_with_ledger(
        &request(
            vec![input("x", w.a), input("z", w.c), input("w", w.d)],
            PropagationDependence::DeclaredIndependent,
        ),
        &w.ledger,
    )
    .expect("a, c, d are pairwise disjoint");
    assert!(triple.shared_evidence.is_empty());
}

fn q(name: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: format!("schema:{name}"),
        variable_name: name.into(),
        role: QuantityRole::Outcome,
        units: "mmHg".into(),
        population_id: "adults".into(),
        regime_id: "observational".into(),
        horizon: 0,
        functional_id: "realization".into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    }
}

fn pooled_pair(
    first: SourceId,
    second: SourceId,
) -> (Vec<InterfaceContract>, Vec<ComponentConnection>) {
    let producer = |name: &str, evidence: SourceId| InterfaceContract {
        component: name.into(),
        produces: vec![
            ProducedPort::new(
                "effect",
                q(name),
                format!("schema:{name}"),
                SupportDomain::Real,
                [ContractOperation::Mean],
            )
            .with_evidence([evidence]),
        ],
        consumes: Vec::new(),
    };
    let port = |name: &str| {
        ConsumedPort::new(
            name,
            q(name),
            format!("schema:{name}"),
            SupportDomain::Real,
            [ContractOperation::Mean],
        )
        .with_dependence(DependenceRequirement::DisjointEvidence)
    };
    let pooled = InterfaceContract {
        component: "pooled".into(),
        produces: Vec::new(),
        consumes: vec![port("a"), port("b")],
    };
    (
        vec![producer("a", first), producer("b", second), pooled],
        vec![
            ComponentConnection::new("a", "effect", "pooled", "a"),
            ComponentConnection::new("b", "effect", "pooled", "b"),
        ],
    )
}

/// A disjoint-evidence port is met exactly by the source pairs with no common row: of the six
/// pairs of the four sources only (a,b) and (b,d) fail; their shared rows number one each.
#[test]
fn disjoint_evidence_ports_are_met_by_exactly_the_row_disjoint_pairs() {
    let w = world();
    let named = [("a", w.a), ("b", w.b), ("c", w.c), ("d", w.d)];
    let mut failing = Vec::new();
    for i in 0..named.len() {
        for j in i + 1..named.len() {
            let (components, connections) = pooled_pair(named[i].1, named[j].1);
            let shared = w.ledger.shared_observations(named[i].1, named[j].1).expect("overlap");
            let met = check_composition_with_ledger(&components, &connections, &w.ledger).is_ok();
            assert_eq!(met, shared.is_empty(), "{}-{}", named[i].0, named[j].0);
            if !met {
                failing.push((named[i].0, named[j].0, shared.len()));
            }
        }
    }
    assert_eq!(failing, vec![("a", "b", 1), ("b", "d", 1)]);
}
