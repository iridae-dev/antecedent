//! Transport under graph uncertainty against hand-enumerated equivalence classes.
//!
//! The oracle is the enumeration of the Markov equivalence class by hand (a chain of three
//! variables has exactly three DAGs: no orientation of its two edges creates a collider) and
//! the structural verdicts the transport identifier's own unit tests fix for the same DAGs
//! with the same query.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    ContinuousDomain, GridSpec, ResponseFunctional, ResponseQuery, TransportQuery, VariableId,
};
use antecedent_graph::{Cpdag, DenseNodeId};
use antecedent_identify::TransportFormula;
use antecedent_identify::discovered_transport::{
    DiscoveredTransportError, DiscoveredTransportRequest, EnvelopeStatus, MemberVerdict,
    transport_across_class,
};

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn d(raw: u32) -> DenseNodeId {
    DenseNodeId::from_raw(raw)
}

/// `P(v2 | do(v0))` from an experiment on `v0`.
fn query() -> TransportQuery {
    TransportQuery::new(
        ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: v(2),
            treatment: ContinuousDomain::new(v(0), GridSpec::Values(Arc::from([0.0, 1.0]))),
        }),
        "trial",
        "target",
        [v(0)],
    )
}

fn chain(n: u32) -> Cpdag {
    let mut class = Cpdag::with_variables(n);
    for i in 0..n - 1 {
        class.insert_undirected(d(i), d(i + 1)).unwrap();
    }
    class
}

fn request<'a>(class: &'a Cpdag, targets: &'a [VariableId]) -> DiscoveredTransportRequest<'a> {
    DiscoveredTransportRequest {
        class,
        selection_targets: targets,
        exhausted_search_bounds: 0,
        member_bound: 64,
    }
}

fn refusal(
    error: DiscoveredTransportError,
) -> antecedent_identify::discovered_transport::DiscoveredTransportRefusal {
    match error {
        DiscoveredTransportError::Refused(refusal) => refusal,
        other => panic!("expected a scientific refusal, got {other}"),
    }
}

#[test]
fn chain_class_yields_its_three_members_with_the_verdicts_fixed_for_each_dag() {
    let class = chain(3);
    let targets = [v(1)];
    let envelope = transport_across_class(&request(&class, &targets), &query()).unwrap();

    assert_eq!(envelope.kind(), "structural_envelope");
    assert!(envelope.enumeration_complete());
    assert_eq!(envelope.members().len(), 3);
    assert_eq!(envelope.unresolved().len(), 2);
    assert_eq!(envelope.status(), EnvelopeStatus::IdentifiedInEveryMember);
    // The class does not determine the expression: Direct, recursive factorization and a third
    // identified expression are all present, so the formulas are not one formula.
    assert!(!envelope.formulas_agree());

    let by_edges = |edges: &[(u32, u32)]| {
        let want: Vec<(VariableId, VariableId)> =
            edges.iter().map(|&(p, c)| (v(p), v(c))).collect();
        envelope
            .members()
            .iter()
            .find(|m| m.directed_edges == want)
            .unwrap_or_else(|| panic!("member {edges:?} missing"))
    };
    // 0 -> 1 -> 2 with the mediator varying: recursive population factors.
    match &by_edges(&[(0, 1), (1, 2)]).verdict {
        MemberVerdict::Identified {
            formula: TransportFormula::RecursiveFactorization { .. },
            ..
        } => {}
        other => panic!("0->1->2: {other:?}"),
    }
    // 0 <- 1 -> 2: a Markovian graph, so the truncated factorization identifies it.
    assert!(matches!(by_edges(&[(1, 0), (1, 2)]).verdict, MemberVerdict::Identified { .. }));
    // 0 <- 1 <- 2: the varying mechanism is not an ancestor of the outcome: direct transport.
    match &by_edges(&[(1, 0), (2, 1)]).verdict {
        MemberVerdict::Identified { formula: TransportFormula::Direct(_), rule } => {
            assert_eq!(rule, "transport.sid.direct");
        }
        other => panic!("0<-1<-2: {other:?}"),
    }
}

#[test]
fn a_fully_directed_class_is_one_member_and_still_labelled_an_envelope() {
    // 0 -> 2 <- 1: the unshielded collider compels both edges.
    let mut class = Cpdag::with_variables(3);
    class.insert_directed(d(0), d(2)).unwrap();
    class.insert_directed(d(1), d(2)).unwrap();
    let targets = [v(1)];
    let envelope = transport_across_class(&request(&class, &targets), &query()).unwrap();
    assert_eq!(envelope.kind(), "structural_envelope");
    assert_eq!(envelope.members().len(), 1);
    assert!(envelope.unresolved().is_empty());
    assert_eq!(envelope.status(), EnvelopeStatus::IdentifiedInEveryMember);
    assert!(envelope.formulas_agree());
    // Pre-treatment exogenous selection node: standardization over it.
    match &envelope.members()[0].verdict {
        MemberVerdict::Identified {
            formula: TransportFormula::Standardize { over, .. }, ..
        } => assert_eq!(&over[..], &[v(1)]),
        other => panic!("{other:?}"),
    }
}

#[test]
fn member_bound_stops_the_enumeration_and_no_every_member_statement_is_made() {
    // A chain of four variables has four members (one per choice of the source of the chain).
    let class = chain(4);
    let targets = [v(1)];
    let unbounded = transport_across_class(&request(&class, &targets), &query()).unwrap();
    assert_eq!(unbounded.members().len(), 4);
    assert!(unbounded.enumeration_complete());

    let mut bounded_request = request(&class, &targets);
    bounded_request.member_bound = 2;
    let bounded = transport_across_class(&bounded_request, &query()).unwrap();
    assert_eq!(bounded.members().len(), 2);
    assert!(!bounded.enumeration_complete());
    assert_eq!(bounded.status(), EnvelopeStatus::Partial);
    assert_eq!(bounded.member_bound(), 2);
}

#[test]
fn exhausted_search_bound_refuses_and_names_the_unresolved_pairs() {
    let class = chain(3);
    let targets = [v(1)];
    let mut r = request(&class, &targets);
    r.exhausted_search_bounds = 1;
    let refused = refusal(transport_across_class(&r, &query()).unwrap_err());
    assert_eq!(refused.code, "discovered_structure_unresolved");
    assert_eq!(refused.detail, "discovered_transport.search_incomplete");
    assert_eq!(refused.exhausted_search_bounds, 1);
    let pairs: Vec<(u32, u32)> =
        refused.unresolved.iter().map(|p| (p.a.raw(), p.b.raw())).collect();
    assert_eq!(pairs, vec![(0, 1), (1, 2)]);
    assert!(refused.unresolved.iter().all(|p| p.reason == "undirected_in_equivalence_class"));
    assert!(refused.remedy.contains("declare the graph"));
}

#[test]
fn orientation_conflict_unknown_target_and_zero_bound_are_refused() {
    let mut conflicted = chain(3);
    conflicted.mark_conflict(d(0), d(1)).unwrap();
    let targets = [v(1)];
    let refused =
        refusal(transport_across_class(&request(&conflicted, &targets), &query()).unwrap_err());
    assert_eq!(refused.detail, "discovered_transport.orientation_conflict");
    assert!(refused.unresolved.iter().any(|p| p.reason == "orientation_conflict"));

    let class = chain(3);
    let unknown = [v(9)];
    let refused =
        refusal(transport_across_class(&request(&class, &unknown), &query()).unwrap_err());
    assert_eq!(refused.detail, "discovered_transport.selection_target_unknown");
    assert_eq!(refused.unknown_selection_targets, vec![v(9)]);

    let mut zero = request(&class, &targets);
    zero.member_bound = 0;
    let refused = refusal(transport_across_class(&zero, &query()).unwrap_err());
    assert_eq!(refused.detail, "discovered_transport.member_bound_zero");
}

#[test]
fn pag_route_is_refused_with_its_reason() {
    let error = antecedent_identify::discovered_transport::refuse_pag_transport(Vec::new(), 0);
    let refused = refusal(error);
    assert_eq!(refused.code, "discovered_structure_unresolved");
    assert_eq!(refused.detail, "discovered_transport.pag_members_not_causal_graphs");
    assert!(refused.remedy.contains("declared causal graph"));
}
