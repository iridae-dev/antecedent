//! Evidence obligations of a failed identification on graphs with irrelevant nodes.
//!
//! The expected proof steps and variable sets are the textbook back-door and front-door sets of
//! each graph, worked out by hand, so a repair that simply asked for every variable fails.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::EvidenceObligationKind;
use antecedent_core::{
    AssumptionSet, AverageEffectQuery, CausalQuery, VariableId,
};
use antecedent_design::obligation_adapters::identification_failure_obligations;
use antecedent_graph::{Admg, Dag, DenseNodeId};
use antecedent_identify::{
    DerivationTrace, IdentificationPerformanceRecord, IdentificationResult,
};

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn graph(nodes: u32, edges: &[(u32, u32)]) -> Admg {
    let mut dag = Dag::with_variables(nodes);
    for (a, b) in edges {
        dag.insert_directed(DenseNodeId::from_raw(*a), DenseNodeId::from_raw(*b)).unwrap();
    }
    Admg::from_dag(&dag).unwrap()
}

fn failed(treatment: u32, outcome: u32) -> IdentificationResult {
    IdentificationResult::not_identified(
        CausalQuery::average_effect(AverageEffectQuery::binary_ate(v(treatment), v(outcome))),
        DerivationTrace::default(),
        AssumptionSet::new(),
        IdentificationPerformanceRecord::default(),
    )
}

/// T=0, Y=1; Z=2 and W=3 each cause both T and Y; I=4 only causes Z; D=5 is a child of T.
/// The minimal back-door set is {Z, W}: I is upstream of an already-blocked path and D is a
/// descendant of the treatment, which must never be adjusted for.
#[test]
fn the_backdoor_obligation_names_the_minimal_adjustment_set_and_not_the_irrelevant_nodes() {
    let g = graph(
        6,
        &[(0, 1), (2, 0), (2, 1), (3, 0), (3, 1), (4, 2), (0, 5)],
    );
    let obligations =
        identification_failure_obligations(&failed(0, 1), &g, &[v(0), v(1)], "pop", None).unwrap();
    let backdoor: Vec<_> =
        obligations.iter().filter(|o| o.provenance.family.as_ref() == "backdoor").collect();
    assert_eq!(backdoor.len(), 1);
    let joint = backdoor[0];
    assert_eq!(joint.kind, EvidenceObligationKind::ProvideJointLaw);
    assert_eq!(joint.provenance.proof_step.as_deref(), Some("backdoor.adjustment_set:0,1,2,3"));
    let raw: Vec<u32> = joint.variables.iter().map(|x| x.raw()).collect();
    assert_eq!(raw, vec![0, 1, 2, 3]);
}

/// T=0 -> M=1 -> Y=2 with U=3 confounding T and Y; I=4 causes M only and N=5 is isolated.
/// Only T and Y are observed. The front-door mediator set is {M}: the joint law owed is over
/// (T, M, Y) and mentions neither I nor N.
#[test]
fn the_frontdoor_obligation_names_the_mediator_and_not_the_irrelevant_nodes() {
    let g = graph(6, &[(0, 1), (1, 2), (3, 0), (3, 2), (4, 1)]);
    let obligations =
        identification_failure_obligations(&failed(0, 2), &g, &[v(0), v(2)], "pop", None).unwrap();
    let front: Vec<_> =
        obligations.iter().filter(|o| o.provenance.family.as_ref() == "frontdoor").collect();
    assert_eq!(front.len(), 1);
    assert_eq!(front[0].kind, EvidenceObligationKind::ProvideJointLaw);
    assert_eq!(front[0].provenance.proof_step.as_deref(), Some("frontdoor.mediator_set:1"));
    let raw: Vec<u32> = front[0].variables.iter().map(|x| x.raw()).collect();
    assert_eq!(raw, vec![0, 1, 2]);
}
