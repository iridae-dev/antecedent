//! Claim-level invalidation graph: classification of downstream claims
//! against hand-computable dependency graphs, calibration inheritance,
//! lineage and shared-data views, and order invariance.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use antecedent_contract::composition_verification::{
    CalibrationDeclaration, CalibrationState, CompositionGraph, CompositionNode, CompositionView,
    DependencyKind, GraphError, GraphRevalidation, NodeKind, NumericalCause, NumericalReason,
    ObservedEffect, PriorAuthority, TrustStanding, ViewError, WithdrawalKind, ledger_effects,
    revalidate, shared_data_relations,
};
use antecedent_contract::evidence_ledger::{
    CombinationRequest, EvidenceLedger, ObservationId, PoolingDependence, Revalidation,
    SourceDeclaration, SourceKind, SynthesisOperation,
};
use antecedent_core::{CancellationToken, SemanticDigest};

fn d(byte: u8) -> SemanticDigest {
    SemanticDigest::from_bytes([byte; 32])
}

#[derive(Clone)]
struct Params {
    cohort: u8,
    result: u8,
    trust: TrustStanding,
    target_unsupported: bool,
    result_calibration: CalibrationDeclaration,
}

fn params() -> Params {
    Params {
        cohort: 1,
        result: 30,
        trust: TrustStanding::Held,
        target_unsupported: false,
        result_calibration: CalibrationDeclaration::Held { record: d(90), coordinate: d(100) },
    }
}

fn node(key: &str, kind: NodeKind, id: u8) -> CompositionNode {
    CompositionNode::new(key, kind, d(id))
}

/// ```text
/// cohort ──data──► measurement ──model_version──► mechanisms ──structural──► result
///    │  ▲provider_trust                              ▲                       │ │
///    └─────evidence────────────────────────────────┘     cohort─data──►────┘ │
///                                                        result ─data/calibration─► reported_effect ─data─► decision
///                                                        result ─data─► transported ◄─support─ target
/// sibling_data ─data─► sibling_claim
/// ```
fn build(p: &Params) -> CompositionGraph {
    let mut g = CompositionGraph::new();
    let mut add = |n: CompositionNode| g.add_node(n).expect("node is accepted");
    add(node("cohort", NodeKind::CohortSource, p.cohort));
    add(node("indicator_provider", NodeKind::ExternalProvider, 50).with_trust(p.trust));
    add(node("measurement", NodeKind::MeasurementModel, 10)
        .with_structure(d(11))
        .with_representation("measurement_model/1")
        .with_assumptions(vec!["nondifferential error".to_owned()]));
    add(node("mechanisms", NodeKind::MechanismComposition, 20)
        .with_structure(d(21))
        .with_assumptions(vec!["causal sufficiency".to_owned()]));
    add(node("result", NodeKind::LedgerResult, p.result)
        .with_structure(d(31))
        .with_coordinate(d(100))
        .with_calibration(p.result_calibration.clone()));
    let target = node("target", NodeKind::TargetPopulation, 60);
    add(if p.target_unsupported {
        target.with_unsupported(vec!["income above 200k".to_owned()])
    } else {
        target
    });
    add(node("reported_effect", NodeKind::Claim, 40).with_coordinate(d(100)).with_calibration(
        CalibrationDeclaration::Inherited {
            from: "result".to_owned(),
            record: d(90),
            coordinate: d(100),
        },
    ));
    add(node("transported", NodeKind::TransportedResult, 70)
        .with_coordinate(d(101))
        .with_transformation("transport_to_target")
        .with_assumptions(vec!["s-admissible adjustment set".to_owned()]));
    add(node("decision", NodeKind::Decision, 80));
    add(node("sibling_data", NodeKind::Data, 200));
    add(node("sibling_claim", NodeKind::Claim, 201));
    let mut edge = |down: &str, up: &str, kind| g.add_edge(down, up, kind).expect("edge");
    edge("measurement", "cohort", DependencyKind::Data);
    edge("measurement", "indicator_provider", DependencyKind::ProviderTrust);
    edge("mechanisms", "measurement", DependencyKind::ModelVersion);
    edge("mechanisms", "cohort", DependencyKind::Evidence);
    edge("result", "mechanisms", DependencyKind::StructuralDerivation);
    edge("result", "cohort", DependencyKind::Data);
    edge("reported_effect", "result", DependencyKind::Data);
    edge("reported_effect", "result", DependencyKind::Calibration);
    edge("transported", "result", DependencyKind::Data);
    edge("transported", "target", DependencyKind::Support);
    edge("decision", "reported_effect", DependencyKind::Data);
    edge("sibling_claim", "sibling_data", DependencyKind::Data);
    g
}

fn no_effects() -> BTreeMap<String, ObservedEffect> {
    BTreeMap::new()
}

fn keys(r: &GraphRevalidation, class: Revalidation) -> Vec<&str> {
    r.keys_of(class)
}

const ALL: [&str; 11] = [
    "cohort",
    "decision",
    "indicator_provider",
    "measurement",
    "mechanisms",
    "reported_effect",
    "result",
    "sibling_claim",
    "sibling_data",
    "target",
    "transported",
];

#[test]
fn an_unchanged_graph_is_still_valid_and_keeps_its_authority() {
    let g = build(&params());
    let r = revalidate(&g, &g, &no_effects()).expect("revalidate");
    assert_eq!(keys(&r, Revalidation::StillValid), ALL.to_vec());
    assert!(r.claims().all(|c| c.authority == PriorAuthority::Retained));
    assert!(r.claims().all(|c| c.numerical.is_empty() && c.withdrawals.is_empty()));
    assert_eq!(
        r.claim("reported_effect").expect("claim").calibration.state,
        CalibrationState::Inherited
    );
    assert_eq!(r.claim("result").expect("claim").calibration.state, CalibrationState::Held);
    assert!(r.require_valid("decision").is_ok());
}

#[test]
fn a_change_leaves_the_sibling_valid_and_recomputes_every_dependent() {
    let before = build(&params());
    let after = build(&Params { cohort: 2, ..params() });
    let r = revalidate(&before, &after, &no_effects()).expect("revalidate");
    assert_eq!(
        keys(&r, Revalidation::Recompute),
        vec![
            "cohort",
            "decision",
            "measurement",
            "mechanisms",
            "reported_effect",
            "result",
            "transported"
        ]
    );
    assert_eq!(
        keys(&r, Revalidation::StillValid),
        vec!["indicator_provider", "sibling_claim", "sibling_data", "target"]
    );
    assert!(keys(&r, Revalidation::Withdrawn).is_empty());
    // numerical invalidation only: no license is withdrawn and none is retained for old numbers
    for key in keys(&r, Revalidation::Recompute) {
        let claim = r.claim(key).expect("claim");
        assert!(claim.withdrawals.is_empty(), "{key}");
        assert_eq!(claim.authority, PriorAuthority::NotRetained, "{key}");
        assert!(claim.numerical.iter().all(|reason| reason.origin == "cohort"), "{key}");
    }
    // a refusal names the changed origin and does not call it a withdrawal
    let refusal = r.require_valid("decision").expect_err("stale decision");
    assert_eq!(refusal.code, "claim_revalidation_required");
    assert_eq!(refusal.detail, "composition_verification.recompute_required");
    assert_eq!(refusal.expected.as_deref(), Some("origin cohort"));
}

#[test]
fn reusing_a_structural_derivation_after_data_replacement_keeps_no_fit_authority() {
    let before = build(&params());
    let after = build(&Params { cohort: 2, ..params() });
    let r = revalidate(&before, &after, &no_effects()).expect("revalidate");
    let result = r.claim("result").expect("claim");
    // the structure of mechanisms is unchanged, but the result recomputes
    assert!(result.numerical.contains(&NumericalReason {
        origin: "cohort".to_owned(),
        via: Some(("mechanisms".to_owned(), DependencyKind::StructuralDerivation)),
        cause: NumericalCause::StructureReusedFitNotRetained,
    }));
    assert!(result.numerical.contains(&NumericalReason {
        origin: "cohort".to_owned(),
        via: Some(("cohort".to_owned(), DependencyKind::Data)),
        cause: NumericalCause::IdentityChanged,
    }));
    assert_eq!(result.class, Revalidation::Recompute);
    assert_eq!(result.authority, PriorAuthority::NotRetained);
}

#[test]
fn a_refit_under_an_unchanged_structure_is_a_recomputation_with_no_old_authority() {
    let before = build(&params());
    let after = build(&Params { result: 31, ..params() });
    let r = revalidate(&before, &after, &no_effects()).expect("revalidate");
    let result = r.claim("result").expect("claim");
    assert_eq!(result.class, Revalidation::Recompute);
    assert_eq!(
        result.numerical,
        vec![NumericalReason {
            origin: "result".to_owned(),
            via: None,
            cause: NumericalCause::StructureReusedFitNotRetained
        }]
    );
    assert_eq!(result.authority, PriorAuthority::NotRetained);
    // upstream of the refit and unrelated claims stay valid
    for key in ["cohort", "measurement", "mechanisms", "target", "sibling_claim"] {
        assert_eq!(r.claim(key).expect("claim").class, Revalidation::StillValid, "{key}");
    }
    for key in ["reported_effect", "transported", "decision"] {
        assert_eq!(r.claim(key).expect("claim").class, Revalidation::Recompute, "{key}");
    }
}

#[test]
fn a_withdrawn_provider_trust_withdraws_dependents_without_stale_numbers() {
    let before = build(&params());
    let after = build(&Params { trust: TrustStanding::Withdrawn, ..params() });
    let r = revalidate(&before, &after, &no_effects()).expect("revalidate");
    assert_eq!(
        keys(&r, Revalidation::Withdrawn),
        vec![
            "decision",
            "indicator_provider",
            "measurement",
            "mechanisms",
            "reported_effect",
            "result",
            "transported"
        ]
    );
    assert_eq!(
        keys(&r, Revalidation::StillValid),
        vec!["cohort", "sibling_claim", "sibling_data", "target"]
    );
    // withdrawal of a license is not numerical invalidation
    assert!(r.claims().all(|c| c.numerical.is_empty()));
    let decision = r.claim("decision").expect("claim");
    assert!(decision.withdrawn_for(WithdrawalKind::ProviderTrust));
    let refusal = r.require_valid("decision").expect_err("withdrawn");
    assert_eq!(refusal.code, "claim_license_withdrawn");
    assert_eq!(refusal.detail, "composition_verification.provider_trust_withdrawn");
    assert_eq!(refusal.offending.as_deref(), Some("decision"));
}

#[test]
fn an_unsupported_target_withdraws_only_what_rests_on_its_support() {
    let before = build(&params());
    let after = build(&Params { target_unsupported: true, ..params() });
    let r = revalidate(&before, &after, &no_effects()).expect("revalidate");
    assert_eq!(keys(&r, Revalidation::Withdrawn), vec!["target", "transported"]);
    assert!(r.claim("decision").expect("claim").class == Revalidation::StillValid);
    assert!(r.claim("result").expect("claim").class == Revalidation::StillValid);
    let transported = r.claim("transported").expect("claim");
    assert_eq!(transported.withdrawals.len(), 1);
    let w = &transported.withdrawals[0];
    assert_eq!(w.kind, WithdrawalKind::Support);
    assert_eq!(w.origin, "target");
    assert_eq!(w.detail, "income above 200k");
    assert_eq!(w.via, Some(("target".to_owned(), DependencyKind::Support)));
}

#[test]
fn calibration_inherited_from_an_object_that_no_longer_holds_it_is_refused() {
    let before = build(&params());
    // the upstream result no longer states a guarantee: point value only
    let after =
        build(&Params { result_calibration: CalibrationDeclaration::PointOnly, ..params() });
    let r = revalidate(&before, &after, &no_effects()).expect("revalidate");
    let reported = r.claim("reported_effect").expect("claim");
    assert_eq!(reported.class, Revalidation::Withdrawn);
    assert_eq!(reported.calibration.state, CalibrationState::Lost);
    assert_eq!(reported.calibration.from.as_deref(), Some("result"));
    assert!(reported.calibration.newly_required);
    assert_eq!(reported.calibration.detail, Some("upstream_calibration_not_held"));
    // the point value was never touched: the claim is withdrawn for calibration only
    assert!(reported.numerical.is_empty());
    assert!(reported.withdrawn_for(WithdrawalKind::Calibration));
    let refusal = r.require_valid("reported_effect").expect_err("refused");
    assert_eq!(refusal.code, "cell_not_licensed");
    assert_eq!(refusal.detail, "composition_verification.calibration_not_held");
    // a point-only dependent does not lose its value to a calibration withdrawal
    assert_eq!(r.claim("decision").expect("claim").class, Revalidation::StillValid);
    assert_eq!(r.claim("result").expect("claim").class, Revalidation::StillValid);
}

#[test]
fn a_different_calibration_record_is_not_inherited() {
    let before = build(&params());
    let after = build(&Params {
        result_calibration: CalibrationDeclaration::Held { record: d(91), coordinate: d(100) },
        ..params()
    });
    let r = revalidate(&before, &after, &no_effects()).expect("revalidate");
    let reported = r.claim("reported_effect").expect("claim");
    assert_eq!(reported.class, Revalidation::Withdrawn);
    assert_eq!(reported.calibration.detail, Some("record_differs"));
}

fn transport_pair(
    held_coordinate: SemanticDigest,
    transported_coordinate: SemanticDigest,
    inherited_coordinate: SemanticDigest,
    inherited_node_coordinate: SemanticDigest,
) -> CompositionGraph {
    let mut g = CompositionGraph::new();
    g.add_node(
        node("transported", NodeKind::TransportedResult, 70)
            .with_coordinate(transported_coordinate)
            .with_calibration(CalibrationDeclaration::Held {
                record: d(90),
                coordinate: held_coordinate,
            }),
    )
    .expect("node");
    g.add_node(
        node("retargeted", NodeKind::TransportedResult, 71)
            .with_coordinate(inherited_node_coordinate)
            .with_calibration(CalibrationDeclaration::Inherited {
                from: "transported".to_owned(),
                record: d(90),
                coordinate: inherited_coordinate,
            }),
    )
    .expect("node");
    g.add_edge("retargeted", "transported", DependencyKind::Calibration).expect("edge");
    g
}

#[test]
fn transported_calibration_is_never_inherited_by_a_different_target() {
    let (t1, t2) = (d(101), d(102));
    // the same target: inherited
    let same = transport_pair(t1, t1, t1, t1);
    let r = revalidate(&same, &same, &no_effects()).expect("revalidate");
    assert_eq!(
        r.claim("retargeted").expect("claim").calibration.state,
        CalibrationState::Inherited
    );

    // a second claim about target t2 asks to carry the t1 record: refused both ways
    let other_target = transport_pair(t1, t1, t1, t2);
    let r = revalidate(&other_target, &other_target, &no_effects()).expect("revalidate");
    let retargeted = r.claim("retargeted").expect("claim");
    assert_eq!(retargeted.class, Revalidation::Withdrawn);
    assert_eq!(retargeted.calibration.detail, Some("coordinate_mismatch"));
    let claims_t2 = transport_pair(t1, t1, t2, t2);
    let r = revalidate(&claims_t2, &claims_t2, &no_effects()).expect("revalidate");
    assert_eq!(r.claim("retargeted").expect("claim").calibration.detail, Some("record_differs"));

    // the target of a held record changes: the record is lost and newly required there
    let before = transport_pair(t1, t1, t1, t1);
    let after = transport_pair(t1, t2, t1, t1);
    let r = revalidate(&before, &after, &no_effects()).expect("revalidate");
    let transported = r.claim("transported").expect("claim");
    assert_eq!(transported.class, Revalidation::Withdrawn);
    assert_eq!(transported.calibration.state, CalibrationState::Lost);
    assert!(transported.calibration.newly_required);
    assert!(transported.numerical.iter().any(|n| n.cause == NumericalCause::CoordinateChanged));
    // the claim that inherited from it is refused too
    assert_eq!(r.claim("retargeted").expect("claim").class, Revalidation::Withdrawn);
}

#[test]
fn a_removed_input_is_a_retired_license_not_a_stale_number() {
    let before = build(&params());
    let mut after = CompositionGraph::new();
    for n in before.nodes().filter(|n| n.key != "target") {
        after.add_node(n.clone()).expect("node");
    }
    for e in before.edges().filter(|e| e.upstream != "target") {
        after.add_edge(&e.downstream, &e.upstream, e.kind).expect("edge");
    }
    let r = revalidate(&before, &after, &no_effects()).expect("revalidate");
    assert_eq!(r.removed(), ["target".to_owned()]);
    let transported = r.claim("transported").expect("claim");
    assert_eq!(transported.class, Revalidation::Withdrawn);
    assert_eq!(transported.withdrawals[0].kind, WithdrawalKind::Retired);
    assert_eq!(transported.withdrawals[0].detail, "removed_from_graph");
    assert!(transported.numerical.is_empty());
}

#[test]
fn a_node_added_after_the_earlier_snapshot_has_nothing_to_reuse() {
    let before = build(&params());
    let mut after = build(&params());
    after.add_node(node("late", NodeKind::Claim, 99)).expect("node");
    after.add_edge("late", "result", DependencyKind::Data).expect("edge");
    let r = revalidate(&before, &after, &no_effects()).expect("revalidate");
    let late = r.claim("late").expect("claim");
    assert_eq!(late.class, Revalidation::Recompute);
    assert_eq!(late.numerical[0].cause, NumericalCause::Added);
    assert_eq!(r.claim("decision").expect("claim").class, Revalidation::StillValid);
}

// -- observed effects: ledger, plan, freshness -----------------------------------------------------

fn obs(range: std::ops::Range<u32>) -> Vec<ObservationId> {
    range.map(|i| ObservationId::from_key("cohort", &i.to_le_bytes())).collect()
}

fn pool(
    label: &str,
    sources: &[antecedent_contract::evidence_ledger::SourceId],
) -> CombinationRequest {
    CombinationRequest {
        label: label.to_owned(),
        operation: SynthesisOperation::StatisticalPooling {
            sources: sources.to_vec(),
            dependence: PoolingDependence::Independent,
        },
        replaces: None,
    }
}

#[test]
fn ledger_standings_drive_the_graph_classification() {
    let mut ledger = EvidenceLedger::new();
    let mut source = |label: &str, range| {
        ledger
            .add_source(
                SourceDeclaration::new(label, SourceKind::ObservationSet)
                    .with_observations(obs(range)),
            )
            .expect("source")
    };
    let (s1, s2, s3) = (source("s1", 0..10), source("s2", 10..20), source("s3", 20..30));
    let r_12 = ledger.combine(pool("s1_s2", &[s1, s2])).expect("pool");
    let r_23 = ledger.combine(pool("s2_s3", &[s2, s3])).expect("pool");
    let r_all = ledger.combine(pool("all", &[s1, s2, s3])).expect("pool");

    let mut g = CompositionGraph::new();
    for (key, kind, id) in [
        ("pooled_12", NodeKind::LedgerResult, 1),
        ("pooled_23", NodeKind::LedgerResult, 2),
        ("pooled_all", NodeKind::LedgerResult, 3),
        ("decision_12", NodeKind::Decision, 4),
        ("decision_23", NodeKind::Decision, 5),
        ("decision_all", NodeKind::Decision, 6),
    ] {
        g.add_node(node(key, kind, id)).expect("node");
    }
    for (down, up) in
        [("decision_12", "pooled_12"), ("decision_23", "pooled_23"), ("decision_all", "pooled_all")]
    {
        g.add_edge(down, up, DependencyKind::Evidence).expect("edge");
    }
    let bindings = [("pooled_12", r_12), ("pooled_23", r_23), ("pooled_all", r_all)];
    let fresh = ledger_effects(&ledger, &bindings).expect("effects");
    assert!(fresh.is_empty());

    ledger.withdraw_source(s1, "consent withdrawn").expect("withdraw");
    let effects = ledger_effects(&ledger, &bindings).expect("effects");
    let r = revalidate(&g, &g, &effects).expect("revalidate");
    // pooling s1 and s2 loses a member of two; three studies minus one recompute; one untouched
    assert_eq!(r.claim("pooled_12").expect("c").class, Revalidation::Withdrawn);
    assert_eq!(r.claim("decision_12").expect("c").class, Revalidation::Withdrawn);
    assert_eq!(r.claim("pooled_all").expect("c").class, Revalidation::Recompute);
    assert_eq!(r.claim("decision_all").expect("c").class, Revalidation::Recompute);
    assert_eq!(r.claim("pooled_23").expect("c").class, Revalidation::StillValid);
    assert_eq!(r.claim("decision_23").expect("c").class, Revalidation::StillValid);
    // the graph agrees with the ledger's own classification of the results
    for (key, id) in bindings {
        assert_eq!(r.claim(key).expect("c").class, ledger.standing(id).expect("standing"), "{key}");
    }
}

// -- lineage ---------------------------------------------------------------------------------------

#[test]
fn the_view_reports_lineage_boundaries_versions_and_unsupported_coordinates() {
    let g = build(&Params { target_unsupported: true, ..params() });
    let r = revalidate(&g, &g, &no_effects()).expect("revalidate");
    let view = CompositionView::of(&g, &r, "transported", &[]).expect("view");
    let upstream: Vec<&str> = view.upstream.iter().map(|e| e.key.as_str()).collect();
    assert_eq!(
        upstream,
        vec!["cohort", "indicator_provider", "measurement", "mechanisms", "result", "target"]
    );
    assert_eq!(view.claim.key, "transported");
    assert_eq!(
        view.transformations.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(),
        vec!["measurement_model", "mechanism_composition", "transport_to_target"]
    );
    assert_eq!(
        view.assumptions.iter().map(|a| a.statement.as_str()).collect::<Vec<_>>(),
        vec!["nondifferential error", "causal sufficiency", "s-admissible adjustment set"]
    );
    assert_eq!(view.versions(), vec![("measurement", "measurement_model/1")]);
    assert_eq!(view.unsupported.len(), 1);
    assert_eq!(view.unsupported[0].node, "target");
    assert_eq!(view.unsupported[0].coordinate, "income above 200k");
    // sibling claims are not in the lineage
    assert!(view.upstream.iter().all(|e| !e.key.starts_with("sibling")));
    // the calibration trail names where the record came from
    let reported = CompositionView::of(&g, &r, "reported_effect", &[]).expect("view");
    assert_eq!(reported.claim.calibration.state, CalibrationState::Inherited);
    assert_eq!(reported.claim.calibration.from.as_deref(), Some("result"));
    assert!(matches!(
        CompositionView::of(&g, &r, "absent", &[]),
        Err(ViewError::Graph(GraphError::UnknownNode(key))) if key == "absent"
    ));
}

#[test]
fn shared_data_comes_from_the_ledger_overlap_and_unknown_is_never_independent() {
    let mut ledger = EvidenceLedger::new();
    let a = ledger
        .add_source(
            SourceDeclaration::new("a", SourceKind::ObservationSet).with_observations(obs(0..100)),
        )
        .expect("a");
    let b = ledger
        .add_source(
            SourceDeclaration::new("b", SourceKind::ObservationSet).with_observations(obs(50..150)),
        )
        .expect("b");
    let c = ledger
        .add_source(
            SourceDeclaration::new("c", SourceKind::ObservationSet)
                .with_observations(obs(200..210)),
        )
        .expect("c");
    let vendor = ledger
        .add_source(SourceDeclaration::new("vendor", SourceKind::ExternalClaim))
        .expect("vendor");
    let mut g = CompositionGraph::new();
    for (key, source) in [("na", a), ("nb", b), ("nc", c), ("nv", vendor)] {
        g.add_node(node(key, NodeKind::CohortSource, 1).with_evidence(vec![source])).expect("node");
    }
    g.add_node(node("pooled", NodeKind::LedgerResult, 9)).expect("node");
    for up in ["na", "nb", "nc", "nv"] {
        g.add_edge("pooled", up, DependencyKind::Evidence).expect("edge");
    }
    let cancel = CancellationToken::new();
    let relations = shared_data_relations(&g, &ledger, &cancel).expect("relations");
    let listed: Vec<(&str, &str, &str, u64)> = relations
        .iter()
        .map(|r| (r.first.as_str(), r.second.as_str(), r.relation, r.shared_observations))
        .collect();
    // rows 50..100 are shared by a and b; c is disjoint from both; the vendor's rows are unknown
    assert_eq!(
        listed,
        vec![
            ("na", "nb", "shared_data", 50),
            ("na", "nv", "unknown_dependence", 0),
            ("nb", "nv", "unknown_dependence", 0),
            ("nc", "nv", "unknown_dependence", 0),
        ]
    );
    let r = revalidate(&g, &g, &no_effects()).expect("revalidate");
    let view = CompositionView::of(&g, &r, "pooled", &relations).expect("view");
    assert_eq!(view.shared_data.len(), 4);
    assert_eq!(view.evidence.len(), 4);

    cancel.cancel();
    assert_eq!(shared_data_relations(&g, &ledger, &cancel), Err(ViewError::Cancelled));
}

// -- declaration checks and order invariance ----------------------------------------------------------

#[test]
fn malformed_graphs_are_refused() {
    let mut g = CompositionGraph::new();
    g.add_node(node("a", NodeKind::Claim, 1)).expect("node");
    g.add_node(node("b", NodeKind::Claim, 2)).expect("node");
    assert!(matches!(g.add_node(node("a", NodeKind::Claim, 3)), Err(GraphError::DuplicateNode(_))));
    assert!(matches!(
        g.add_edge("a", "missing", DependencyKind::Data),
        Err(GraphError::UnknownNode(_))
    ));
    assert!(matches!(
        g.add_edge("a", "a", DependencyKind::Data),
        Err(GraphError::SelfDependency(_))
    ));
    g.add_edge("a", "b", DependencyKind::Data).expect("edge");
    assert!(matches!(g.add_edge("b", "a", DependencyKind::Data), Err(GraphError::Cycle { .. })));
    assert!(matches!(g.add_node(node(" ", NodeKind::Claim, 1)), Err(GraphError::InvalidText(_))));
    assert!(matches!(
        g.add_node(node("c", NodeKind::Claim, 1).with_unsupported(Vec::new())),
        Err(GraphError::EmptyUnsupported(_))
    ));
    // inherited calibration without the calibration dependency does not validate
    g.add_node(node("c", NodeKind::Claim, 1).with_calibration(CalibrationDeclaration::Inherited {
        from: "a".to_owned(),
        record: d(1),
        coordinate: d(0),
    }))
    .expect("node");
    assert!(matches!(
        revalidate(&g, &g, &no_effects()),
        Err(GraphError::InheritedWithoutEdge { .. })
    ));
}

fn permutations(n: usize) -> Vec<Vec<usize>> {
    // Heap's algorithm
    fn heap(k: usize, items: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if k == 1 {
            out.push(items.clone());
            return;
        }
        for i in 0..k {
            heap(k - 1, items, out);
            let j = if k % 2 == 0 { i } else { 0 };
            items.swap(j, k - 1);
        }
    }
    let mut out = Vec::new();
    heap(n, &mut (0..n).collect(), &mut out);
    out
}

#[test]
fn declaration_order_never_changes_identity_order_or_classification() {
    let nodes = || {
        vec![
            node("data", NodeKind::Data, 1),
            node("model", NodeKind::MeasurementModel, 2).with_structure(d(9)),
            node("fit", NodeKind::LedgerResult, 3).with_structure(d(8)),
            node("claim", NodeKind::Claim, 4),
            node("other", NodeKind::Claim, 5),
            node("decision", NodeKind::Decision, 6),
        ]
    };
    let edges = [
        ("model", "data", DependencyKind::Data),
        ("fit", "model", DependencyKind::StructuralDerivation),
        ("fit", "data", DependencyKind::Data),
        ("claim", "fit", DependencyKind::Data),
        ("decision", "claim", DependencyKind::Evidence),
    ];
    let make = |order: &[usize], flip: bool, data_id: u8| {
        let mut g = CompositionGraph::new();
        let mut declared = nodes();
        declared[0] = node("data", NodeKind::Data, data_id);
        for &i in order {
            g.add_node(declared[i].clone()).expect("node");
        }
        let mut list = edges.to_vec();
        if flip {
            list.reverse();
        }
        for (down, up, kind) in list {
            g.add_edge(down, up, kind).expect("edge");
        }
        g
    };
    let reference_before = make(&[0, 1, 2, 3, 4, 5], false, 1);
    let reference_after = make(&[0, 1, 2, 3, 4, 5], false, 7);
    let reference =
        revalidate(&reference_before, &reference_after, &no_effects()).expect("revalidate");
    assert_eq!(
        keys(&reference, Revalidation::Recompute),
        vec!["claim", "data", "decision", "fit", "model"]
    );
    assert_eq!(keys(&reference, Revalidation::StillValid), vec!["other"]);
    let order =
        reference_after.topological_order().iter().map(ToString::to_string).collect::<Vec<_>>();
    assert_eq!(order, ["data", "model", "fit", "claim", "decision", "other"].map(String::from));
    let perms = permutations(6);
    assert_eq!(perms.len(), 720);
    for permutation in &perms {
        for flip in [false, true] {
            let before = make(permutation, flip, 1);
            let after = make(permutation, flip, 7);
            assert_eq!(after.identity(), reference_after.identity());
            assert_eq!(after, reference_after);
            let got = revalidate(&before, &after, &no_effects()).expect("revalidate");
            assert_eq!(got.identity(), reference.identity());
        }
    }
    assert_ne!(reference_before.identity(), reference_after.identity());
}

#[test]
fn the_classification_identity_binds_every_reported_field() {
    let before = build(&params());
    let after = build(&Params { cohort: 2, ..params() });
    let stale = revalidate(&before, &after, &no_effects()).expect("revalidate");
    let same = revalidate(&before, &before, &no_effects()).expect("revalidate");
    assert_ne!(stale.identity(), same.identity());
    let withdrawn = revalidate(
        &before,
        &build(&Params { trust: TrustStanding::Withdrawn, ..params() }),
        &no_effects(),
    )
    .expect("revalidate");
    assert_ne!(withdrawn.identity(), stale.identity());
}

#[test]
fn a_reused_claim_does_not_keep_authority_across_a_trust_standing_that_moved() {
    // every identity, structure and edge is unchanged: only the provider's stated standing
    // moved from held to not stated as applicable
    let before = build(&params());
    let after = build(&Params { trust: TrustStanding::NotApplicable, ..params() });
    let r = revalidate(&before, &after, &no_effects()).expect("revalidate");
    assert_eq!(
        keys(&r, Revalidation::Recompute),
        vec![
            "decision",
            "indicator_provider",
            "measurement",
            "mechanisms",
            "reported_effect",
            "result",
            "transported"
        ]
    );
    assert_eq!(
        keys(&r, Revalidation::StillValid),
        vec!["cohort", "sibling_claim", "sibling_data", "target"]
    );
    let provider = r.claim("indicator_provider").expect("claim");
    assert_eq!(provider.numerical[0].cause, NumericalCause::TrustChanged);
    assert_eq!(provider.authority, PriorAuthority::NotRetained);
    // a numerical reason, not a lost license
    assert!(r.claims().all(|c| c.withdrawals.is_empty()));
    assert!(r.claim("sibling_claim").expect("claim").authority == PriorAuthority::Retained);
}

#[test]
fn a_replaced_calibration_record_recomputes_its_owner_and_refuses_its_heirs() {
    let before = build(&params());
    let after = build(&Params {
        result_calibration: CalibrationDeclaration::Held { record: d(91), coordinate: d(100) },
        ..params()
    });
    let r = revalidate(&before, &after, &no_effects()).expect("revalidate");
    let result = r.claim("result").expect("claim");
    assert_eq!(result.class, Revalidation::Recompute);
    assert!(result.numerical.iter().any(|n| n.cause == NumericalCause::CalibrationChanged));
    assert_eq!(result.authority, PriorAuthority::NotRetained);
    // the claim that inherited the old record is refused: it states a record no one holds
    let heir = r.claim("reported_effect").expect("claim");
    assert_eq!(heir.class, Revalidation::Withdrawn);
    assert_eq!(heir.calibration.detail, Some("record_differs"));
    // claims that merely read the value do not inherit the calibration change
    assert_eq!(r.claim("decision").expect("claim").class, Revalidation::StillValid);
    assert_eq!(r.claim("transported").expect("claim").class, Revalidation::StillValid);
    assert_eq!(r.claim("sibling_claim").expect("claim").class, Revalidation::StillValid);
}

#[test]
fn a_provider_standing_nobody_stated_is_not_a_license() {
    let g = build(&Params { trust: TrustStanding::Undeclared, ..params() });
    let r = revalidate(&g, &g, &no_effects()).expect("revalidate");
    let provider = r.claim("indicator_provider").expect("claim");
    assert_eq!(provider.class, Revalidation::Withdrawn);
    assert_eq!(provider.withdrawals[0].kind, WithdrawalKind::ProviderTrust);
    assert_eq!(provider.withdrawals[0].detail, "provider_trust_undeclared");
    assert!(r.claim("decision").expect("claim").withdrawn_for(WithdrawalKind::ProviderTrust));
    assert_eq!(r.claim("cohort").expect("claim").class, Revalidation::StillValid);
    assert_eq!(TrustStanding::from_name("undeclared"), Some(TrustStanding::Undeclared));
}

#[test]
fn restored_support_recomputes_the_claims_that_were_never_licensed_there() {
    let before = build(&Params { target_unsupported: true, ..params() });
    let after = build(&params());
    let r = revalidate(&before, &after, &no_effects()).expect("revalidate");
    let target = r.claim("target").expect("claim");
    assert_eq!(target.class, Revalidation::Recompute);
    assert_eq!(target.numerical[0].cause, NumericalCause::SupportRestored);
    assert_eq!(r.claim("transported").expect("claim").class, Revalidation::Recompute);
    assert!(keys(&r, Revalidation::Withdrawn).is_empty());
}

#[test]
fn a_standing_is_stated_as_one_value_and_unstated_standing_is_refused() {
    use antecedent_contract::composition_verification::ClaimStanding;
    let mut g = CompositionGraph::new();
    g.add_node(node("native", NodeKind::Claim, 1).with_standing(ClaimStanding::native_point()))
        .expect("node");
    g.add_node(node("unstated", NodeKind::Claim, 2).with_standing(ClaimStanding::undeclared()))
        .expect("node");
    g.add_node(node("outside", NodeKind::Claim, 3).with_standing(
        ClaimStanding::native_point().with_unsupported(vec!["income above 200k".to_owned()]),
    ))
    .expect("node");
    let r = revalidate(&g, &g, &no_effects()).expect("revalidate");
    assert_eq!(keys(&r, Revalidation::StillValid), vec!["native"]);
    assert_eq!(keys(&r, Revalidation::Withdrawn), vec!["outside", "unstated"]);
    assert_eq!(g.node("native").expect("node").standing(), ClaimStanding::native_point());
}

#[test]
fn another_vocabulary_adapts_onto_the_one_class() {
    assert_eq!(ObservedEffect::from_revalidation(Revalidation::StillValid, "x"), None);
    assert_eq!(
        ObservedEffect::from_revalidation(Revalidation::Recompute, "stage"),
        Some(ObservedEffect::Recomputed { cause: "stage".to_owned() })
    );
    assert_eq!(
        ObservedEffect::from_revalidation(Revalidation::Withdrawn, "stage"),
        Some(ObservedEffect::Withdrawn { cause: "stage".to_owned() })
    );
}

#[test]
fn replaced_scientific_premises_withdraw_old_authority_and_descendants() {
    let mut ledger = EvidenceLedger::new();
    let a =
        ledger.add_source(SourceDeclaration::new("premise-a", SourceKind::ExternalClaim)).unwrap();
    let b =
        ledger.add_source(SourceDeclaration::new("premise-b", SourceKind::ExternalClaim)).unwrap();
    let base = node("premise", NodeKind::Data, 7)
        .with_assumptions(vec!["one".into(), "two".into()])
        .with_transformation("map-a")
        .with_evidence(vec![a, b]);
    let graph = |premise: CompositionNode| {
        let mut graph = CompositionGraph::new();
        graph.add_node(premise).unwrap();
        graph.add_node(node("derived", NodeKind::Claim, 8)).unwrap();
        graph.add_node(node("sibling", NodeKind::Claim, 9)).unwrap();
        graph.add_edge("derived", "premise", DependencyKind::Data).unwrap();
        graph
    };
    let before = graph(base.clone());
    let mut variants = Vec::new();
    let mut kind = base.clone();
    kind.kind = NodeKind::CohortSource;
    variants.push((kind, "node_kind_changed"));
    let mut assumptions = base.clone();
    assumptions.assumptions.push("three".into());
    variants.push((assumptions, "assumptions_changed"));
    let mut transformation = base.clone();
    transformation.transformation = Some("map-b".into());
    variants.push((transformation, "transformation_changed"));
    let mut evidence = base.clone();
    evidence.evidence.pop();
    variants.push((evidence, "evidence_changed"));
    for (variant, detail) in variants {
        let mut effects = no_effects();
        effects
            .insert("premise".into(), ObservedEffect::Recomputed { cause: "fresh-numbers".into() });
        let result = revalidate(&before, &graph(variant), &effects).unwrap();
        let premise = result.claim("premise").unwrap();
        assert_eq!(premise.class, Revalidation::Withdrawn);
        assert_eq!(premise.authority, PriorAuthority::NotRetained);
        assert!(
            premise
                .withdrawals
                .iter()
                .any(|w| w.kind == WithdrawalKind::Retired && w.detail == detail)
        );
        let downstream = result.claim("derived").unwrap();
        assert_eq!(downstream.class, Revalidation::Withdrawn);
        assert!(downstream.withdrawals.iter().any(|w| w.origin == "premise" && w.detail == detail));
        assert_eq!(result.claim("sibling").unwrap().class, Revalidation::StillValid);
    }
    let mut equivalent = base;
    equivalent.assumptions = vec!["two".into(), "one".into(), "one".into()];
    equivalent.evidence = vec![b, a, a];
    let result = revalidate(&before, &graph(equivalent), &no_effects()).unwrap();
    assert_eq!(result.claim("premise").unwrap().class, Revalidation::StillValid);
    assert_eq!(result.claim("derived").unwrap().class, Revalidation::StillValid);
}
