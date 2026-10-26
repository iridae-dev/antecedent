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
    ContinuousDomain, GridSpec, Lag, ResponseFunctional, ResponseQuery, TransportQuery, VariableId,
};
use antecedent_graph::{Cpdag, DenseNodeId, Pag, TemporalCpdag, TemporalDag, TemporalPag};
use antecedent_identify::TransportFormula;
use antecedent_identify::discovered_transport::{
    DiscoveredTransportError, DiscoveredTransportRequest, EnvelopeStatus, MemberVerdict,
    PagTransportRequest, transport_across_class, transport_across_pag,
};
use antecedent_identify::discovered_transport_temporal::{
    LaggedEdge, TemporalClass, TemporalTransportRequest, transport_across_temporal_class,
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

/// `E[outcome | do(treatment)]` from an experiment on `treatment`.
fn effect_query(outcome: u32, treatment: u32) -> TransportQuery {
    TransportQuery::new(
        ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: v(outcome),
            treatment: ContinuousDomain::new(v(treatment), GridSpec::Values(Arc::from([0.0, 1.0]))),
        }),
        "trial",
        "target",
        [v(treatment)],
    )
}

fn pag_request<'a>(class: &'a Pag, targets: &'a [VariableId]) -> PagTransportRequest<'a> {
    PagTransportRequest {
        class,
        selection_targets: targets,
        exhausted_search_bounds: 0,
        member_bound: 64,
        completion_operations: antecedent_graph::DEFAULT_COMPLETION_OPERATIONS,
    }
}

fn kind(formula: &TransportFormula) -> &'static str {
    match formula {
        TransportFormula::Direct(_) => "direct",
        TransportFormula::Standardize { .. } => "standardize",
        TransportFormula::RecursiveFactorization { .. } => "recursive_factorization",
    }
}

/// `c -> z -> y <-o x` with `z` selected. `z -> y` is visible (`c` is not adjacent to `y` and
/// has an edge into `z`), so no latent common cause of `z` and `y` is compatible with the class;
/// `x o-> y` leaves both `x -> y` and `x <-> y`.
#[test]
fn identified_pag_class_transports_through_every_circle_completion() {
    let mut class = Pag::with_variables(4);
    class.insert_circle_arrow(d(0), d(1)).unwrap();
    class.insert_directed(d(2), d(1)).unwrap();
    class.insert_directed(d(3), d(2)).unwrap();
    let targets = [v(2)];
    let envelope =
        transport_across_pag(&pag_request(&class, &targets), &effect_query(1, 0)).unwrap();

    assert_eq!(envelope.kind(), "structural_envelope");
    assert!(envelope.enumeration_complete());
    assert_eq!(envelope.members().len(), 2);
    assert_eq!(envelope.status(), EnvelopeStatus::IdentifiedInEveryMember);
    assert!(envelope.formulas_agree());
    let by_x_edge = |bidirected: bool| {
        envelope
            .members()
            .iter()
            .find(|m| m.bidirected_edges.contains(&(v(0), v(1))) == bidirected)
            .expect("member")
    };
    // The circle at `x` is two members: one with the latent common cause as an edge, one
    // where `x -> y` is not visible and so is also read as confounded. Neither reads the
    // circle as no confounding.
    assert_eq!(by_x_edge(true).confounded_edges, vec![(v(3), v(2))]);
    assert!(by_x_edge(false).confounded_edges.contains(&(v(0), v(1))));
    for member in envelope.members() {
        assert!(member.directed_edges.contains(&(v(2), v(1))));
        assert!(!member.confounded_edges.contains(&(v(2), v(1))), "z -> y is visible");
        let MemberVerdict::Identified { formula, .. } = &member.verdict else {
            panic!("{:?}", member.verdict);
        };
        assert_eq!(kind(formula), "standardize");
    }
    assert!(!envelope.unresolved().is_empty());
}

/// `x o-> y <-o z` with `z` selected: `z -> y` and `z <-> y` are both completions. Where `z`
/// is a cause of `y` it can also share a latent cause with it (the edge is not visible), and
/// shifting the mechanism of a confounded cause moves the outcome law in a way no standardizer
/// captures. Where `z` only shares a cause with `y`, the shift never reaches the outcome.
#[test]
fn pag_members_disagree_when_a_selected_cause_may_share_a_latent_cause_with_the_outcome() {
    let mut class = Pag::with_variables(3);
    class.insert_circle_arrow(d(0), d(1)).unwrap();
    class.insert_circle_arrow(d(2), d(1)).unwrap();
    let targets = [v(2)];
    let envelope =
        transport_across_pag(&pag_request(&class, &targets), &effect_query(1, 0)).unwrap();

    assert_eq!(envelope.members().len(), 4);
    assert_eq!(envelope.status(), EnvelopeStatus::MembersDisagree);
    assert!(!envelope.formulas_agree());
    for member in envelope.members() {
        let z_is_a_cause = member.directed_edges.contains(&(v(2), v(1)));
        match (&member.verdict, z_is_a_cause) {
            (MemberVerdict::NotCertified { .. }, true) => {}
            (MemberVerdict::Identified { formula, rule }, false) => {
                assert_eq!(kind(formula), "direct");
                assert_eq!(rule, "transport.sid.direct");
            }
            other => panic!("{other:?}"),
        }
    }
}

#[test]
fn pag_search_bound_target_and_bound_refusals_name_the_unresolved_marks() {
    let mut class = Pag::with_variables(3);
    class.insert_circle_arrow(d(0), d(1)).unwrap();
    class.insert_circle_arrow(d(2), d(1)).unwrap();
    let targets = [v(2)];

    let mut exhausted = pag_request(&class, &targets);
    exhausted.exhausted_search_bounds = 2;
    let refused = refusal(transport_across_pag(&exhausted, &effect_query(1, 0)).unwrap_err());
    assert_eq!(refused.detail, "discovered_transport.search_incomplete");
    assert_eq!(refused.exhausted_search_bounds, 2);
    assert!(refused.unresolved.iter().all(|p| p.reason == "circle_endpoint"));
    assert_eq!(refused.unresolved.len(), 2);

    let unknown = [v(7)];
    let refused = refusal(
        transport_across_pag(&pag_request(&class, &unknown), &effect_query(1, 0)).unwrap_err(),
    );
    assert_eq!(refused.detail, "discovered_transport.selection_target_unknown");
    assert_eq!(refused.unknown_selection_targets, vec![v(7)]);

    let mut zero = pag_request(&class, &targets);
    zero.member_bound = 0;
    let refused = refusal(transport_across_pag(&zero, &effect_query(1, 0)).unwrap_err());
    assert_eq!(refused.detail, "discovered_transport.member_bound_zero");

    let mut conflicted = class.clone();
    conflicted.mark_conflict(d(0), d(1)).unwrap();
    let refused = refusal(
        transport_across_pag(&pag_request(&conflicted, &targets), &effect_query(1, 0)).unwrap_err(),
    );
    assert_eq!(refused.detail, "discovered_transport.orientation_conflict");
}

#[test]
fn a_pag_member_bound_or_walk_budget_leaves_the_envelope_partial() {
    let mut class = Pag::with_variables(3);
    class.insert_circle_arrow(d(0), d(1)).unwrap();
    class.insert_circle_arrow(d(2), d(1)).unwrap();
    let targets = [v(2)];

    let mut bounded = pag_request(&class, &targets);
    bounded.member_bound = 2;
    let envelope = transport_across_pag(&bounded, &effect_query(1, 0)).unwrap();
    assert_eq!(envelope.members().len(), 2);
    assert!(!envelope.enumeration_complete());
    assert_eq!(envelope.status(), EnvelopeStatus::Partial);

    // A walk budget too small to classify the assignments is a typed refusal when it finds no
    // member, never a claim about the class.
    let mut starved = pag_request(&class, &targets);
    starved.completion_operations = 1;
    match transport_across_pag(&starved, &effect_query(1, 0)) {
        Ok(envelope) => assert_eq!(envelope.status(), EnvelopeStatus::Partial),
        Err(error) => {
            assert_eq!(refusal(error).detail, "discovered_transport.class_too_large");
        }
    }
}

#[test]
fn a_fully_oriented_pag_without_latent_structure_matches_its_dag() {
    // 0 -> 2 <- 1 with every edge visible is the PAG of the collider DAG; the class has one
    // member, the DAG's own, and its verdict is the DAG's: a standardization over the exogenous
    // selected cause.
    let mut class = Pag::with_variables(4);
    class.insert_directed(d(0), d(2)).unwrap();
    class.insert_directed(d(1), d(2)).unwrap();
    class.insert_directed(d(3), d(0)).unwrap();
    class.insert_directed(d(3), d(1)).unwrap();
    let targets = [v(1)];
    let envelope =
        transport_across_pag(&pag_request(&class, &targets), &effect_query(2, 0)).unwrap();
    assert_eq!(envelope.members().len(), 1);
    let member = &envelope.members()[0];
    assert!(member.bidirected_edges.is_empty());
    assert_eq!(envelope.status(), EnvelopeStatus::IdentifiedInEveryMember);
}

fn lagged(graph: &mut TemporalDag, variable: u32, lag: u32) -> DenseNodeId {
    graph.add_lagged(v(variable), Lag::from_raw(lag)).unwrap()
}

fn temporal_request<'a>(
    class: TemporalClass<'a>,
    targets: &'a [VariableId],
    horizon_steps: u32,
) -> TemporalTransportRequest<'a> {
    TemporalTransportRequest {
        class,
        selection_targets: targets,
        exhausted_search_bounds: 0,
        member_bound: 64,
        completion_operations: antecedent_graph::DEFAULT_COMPLETION_OPERATIONS,
        horizon_steps,
        history: None,
    }
}

/// `y_t <- x_{t-1}, z_{t-1}` with autoregressive `x` and `z`; `z` is selected.
fn autoregressive_template() -> TemporalDag {
    let mut graph = TemporalDag::empty();
    let x1 = lagged(&mut graph, 0, 1);
    let x0 = lagged(&mut graph, 0, 0);
    let y0 = lagged(&mut graph, 1, 0);
    let z1 = lagged(&mut graph, 2, 1);
    let z0 = lagged(&mut graph, 2, 0);
    for (from, to) in [(x1, x0), (x1, y0), (z1, z0), (z1, y0)] {
        graph.insert_directed(from, to).unwrap();
    }
    graph
}

#[test]
fn a_lag_one_effect_with_autoregressive_history_is_identified_at_horizon_one() {
    let template = autoregressive_template();
    let targets = [v(2)];
    let envelope = transport_across_temporal_class(
        &temporal_request(TemporalClass::Dag(&template), &targets, 1),
        &effect_query(1, 0),
    )
    .unwrap();
    assert_eq!(envelope.members().len(), 1);
    assert_eq!(envelope.status(), EnvelopeStatus::IdentifiedInEveryMember);
    assert_eq!(envelope.max_lag(), 1);
    assert_eq!(envelope.history(), 2);
    assert_eq!(envelope.horizon_steps(), 1);
    let member = &envelope.members()[0];
    assert!(member.directed_edges.contains(&LaggedEdge { from: v(0), to: v(1), lag: 1 }));
    assert!(member.bidirected_edges.is_empty());
    let MemberVerdict::Identified { formula, .. } = &member.verdict else {
        panic!("{:?}", member.verdict);
    };
    assert_eq!(kind(formula), "standardize");
    // The unrolled question names the slots: the treatment at the origin, the outcome one
    // step later, and the selection target at every slice that the outcome depends on.
    let window = envelope.window();
    assert_ne!(window.treatment, window.outcome);
    assert!(member.window_graph().dense_of_static(window.treatment).is_some());
    assert!(member.window_graph().dense_of_static(window.outcome).is_some());
}

#[test]
fn the_effect_horizon_decides_whether_a_selected_history_matters() {
    // `y_t <- x_{t-1}` only, and `z` selected but unrelated to `y`: at horizon one the
    // outcome's ancestors never meet a population-varying mechanism, so transport is direct;
    // the same template at horizon zero has no path from the treatment to the outcome at all.
    let mut template = TemporalDag::empty();
    let x1 = lagged(&mut template, 0, 1);
    let y0 = lagged(&mut template, 1, 0);
    let z1 = lagged(&mut template, 2, 1);
    let z0 = lagged(&mut template, 2, 0);
    template.insert_directed(x1, y0).unwrap();
    template.insert_directed(z1, z0).unwrap();
    let targets = [v(2)];
    let envelope = transport_across_temporal_class(
        &temporal_request(TemporalClass::Dag(&template), &targets, 1),
        &effect_query(1, 0),
    )
    .unwrap();
    match &envelope.members()[0].verdict {
        MemberVerdict::Identified { rule, formula } => {
            assert_eq!(kind(formula), "direct");
            assert_eq!(rule, "transport.sid.direct");
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_temporal_cpdag_orients_contemporaneous_edges_both_ways_and_keeps_lagged_edges_forward() {
    // `a_t - b_t` is undirected: two completions, one per direction, the same at every slice.
    // `c_{t-1} -> a_t` is lagged and so stays forward in time; orienting `b_t -> a_t` would
    // create the collider `c_{t-1} -> a_t <- b_t` the class does not have, so only `a -> b`
    // completes it.
    let mut free = TemporalCpdag::empty();
    let a0 = free.add_lagged(v(0), Lag::from_raw(0)).unwrap();
    let b0 = free.add_lagged(v(1), Lag::from_raw(0)).unwrap();
    free.insert_undirected(a0, b0).unwrap();
    let envelope = transport_across_temporal_class(
        &temporal_request(TemporalClass::Cpdag(&free), &[], 0),
        &effect_query(1, 0),
    )
    .unwrap();
    assert_eq!(envelope.members().len(), 2);
    assert!(envelope.enumeration_complete());
    let oriented: Vec<Vec<(VariableId, VariableId)>> = envelope
        .members()
        .iter()
        .map(|m| m.directed_edges.iter().map(|e| (e.from, e.to)).collect())
        .collect();
    assert!(oriented.contains(&vec![(v(0), v(1))]));
    assert!(oriented.contains(&vec![(v(1), v(0))]));
    assert_eq!(envelope.unresolved().len(), 1);
    assert_eq!(envelope.unresolved()[0].reason, "undirected_in_equivalence_class");

    let mut shielded = TemporalCpdag::empty();
    let c1 = shielded.add_lagged(v(2), Lag::from_raw(1)).unwrap();
    let a0 = shielded.add_lagged(v(0), Lag::from_raw(0)).unwrap();
    let b0 = shielded.add_lagged(v(1), Lag::from_raw(0)).unwrap();
    shielded.insert_directed(c1, a0).unwrap();
    shielded.insert_undirected(a0, b0).unwrap();
    let envelope = transport_across_temporal_class(
        &temporal_request(TemporalClass::Cpdag(&shielded), &[], 0),
        &effect_query(1, 0),
    )
    .unwrap();
    assert_eq!(envelope.members().len(), 1);
    let member = &envelope.members()[0];
    assert!(member.directed_edges.contains(&LaggedEdge { from: v(2), to: v(0), lag: 1 }));
    assert!(member.directed_edges.contains(&LaggedEdge { from: v(0), to: v(1), lag: 0 }));
}

/// `x_{t-1} o-> y_t <-o w_{t-1}` with `w` selected: where `w` is a cause of `y` it may also share
/// a latent cause with it, which no standardizer repairs.
#[test]
fn temporal_pag_members_disagree_when_a_selected_lagged_cause_may_be_confounded() {
    let mut class = TemporalPag::empty();
    let x1 = class.add_lagged(v(0), Lag::from_raw(1)).unwrap();
    let y0 = class.add_lagged(v(1), Lag::from_raw(0)).unwrap();
    let w1 = class.add_lagged(v(2), Lag::from_raw(1)).unwrap();
    class.insert_circle_arrow(x1, y0).unwrap();
    class.insert_circle_arrow(w1, y0).unwrap();
    let targets = [v(2)];
    let envelope = transport_across_temporal_class(
        &temporal_request(TemporalClass::Pag(&class), &targets, 1),
        &effect_query(1, 0),
    )
    .unwrap();
    assert_eq!(envelope.members().len(), 4);
    assert_eq!(envelope.status(), EnvelopeStatus::MembersDisagree);
    for member in envelope.members() {
        let w_is_a_cause =
            member.directed_edges.contains(&LaggedEdge { from: v(2), to: v(1), lag: 1 });
        match (&member.verdict, w_is_a_cause) {
            (MemberVerdict::NotCertified { .. }, true) => {}
            (MemberVerdict::Identified { formula, .. }, false) => {
                assert_eq!(kind(formula), "direct");
            }
            other => panic!("{other:?}"),
        }
    }
}

#[test]
fn temporal_requests_refuse_unknown_variables_and_unroll_only_the_declared_treatment() {
    let template = autoregressive_template();
    let targets = [v(2)];
    let refused = refusal(
        transport_across_temporal_class(
            &temporal_request(TemporalClass::Dag(&template), &targets, 1),
            &effect_query(9, 0),
        )
        .unwrap_err(),
    );
    assert_eq!(refused.detail, "discovered_transport.temporal_query_invalid");

    let unknown = [v(8)];
    let refused = refusal(
        transport_across_temporal_class(
            &temporal_request(TemporalClass::Dag(&template), &unknown, 1),
            &effect_query(1, 0),
        )
        .unwrap_err(),
    );
    assert_eq!(refused.detail, "discovered_transport.selection_target_unknown");

    // An experiment on a variable other than the treatment is not unrolled.
    let mut other_experiment = effect_query(1, 0);
    other_experiment.source_experiments = Arc::from([v(2)]);
    let refused = refusal(
        transport_across_temporal_class(
            &temporal_request(TemporalClass::Dag(&template), &targets, 1),
            &other_experiment,
        )
        .unwrap_err(),
    );
    assert_eq!(refused.detail, "discovered_transport.temporal_query_invalid");
}

#[test]
fn static_class_transport_preserves_context_and_typed_completion_stops() {
    use antecedent_core::ExecutionContext;
    use antecedent_identify::discovered_transport::{
        transport_across_class_with_context, transport_across_pag_with_context,
    };
    use antecedent_identify::{IdentificationBudget, IdentificationError};
    let cpdag = chain(3);
    let targets = [v(1)];
    let cpdag_request = request(&cpdag, &targets);
    let mut pag = Pag::with_variables(3);
    pag.insert_circle_arrow(d(0), d(1)).unwrap();
    pag.insert_circle_arrow(d(2), d(1)).unwrap();
    let pag_targets = [v(2)];
    let pag_request = pag_request(&pag, &pag_targets);
    let pag_query = effect_query(1, 0);
    let ctx = ExecutionContext::for_tests(17).with_memory_limit(Some(1 << 24));
    let before = ctx.admission().available();
    assert_eq!(
        transport_across_class_with_context(&cpdag_request, &query(), &ctx).unwrap(),
        transport_across_class(&cpdag_request, &query()).unwrap()
    );
    assert_eq!(
        transport_across_pag_with_context(&pag_request, &pag_query, &ctx).unwrap(),
        transport_across_pag(&pag_request, &pag_query).unwrap()
    );
    assert_eq!(ctx.admission().available(), before);

    let tiny = ExecutionContext::for_tests(17).with_memory_limit(Some(1));
    let before = tiny.admission().available();
    for result in [
        transport_across_class_with_context(&cpdag_request, &query(), &tiny),
        transport_across_pag_with_context(&pag_request, &pag_query, &tiny),
    ] {
        assert!(matches!(
            result,
            Err(DiscoveredTransportError::Identification(IdentificationError::Budget {
                budget: IdentificationBudget::Memory
            }))
        ));
    }
    assert_eq!(tiny.admission().available(), before);
    ctx.cancellation.cancel();
    for result in [
        transport_across_class_with_context(&cpdag_request, &query(), &ctx),
        transport_across_pag_with_context(&pag_request, &pag_query, &ctx),
    ] {
        assert!(matches!(
            result,
            Err(DiscoveredTransportError::Identification(IdentificationError::Cancelled))
        ));
    }
}
