//! Bounded multi-source limited-experiment (`TR^mz`) identification.
//!
//! The positive fixture is Bareinboim & Pearl (`NeurIPS` 2014, R-443) Figure 1(c,d):
//! neither source transports `P*(y | do(x))` alone, but experiments on `{Z2}` in
//! one source and `{Z1}` in the other do, through
//! `Σ_z2 P^b(z2 | x, do(z1)) P^a(y | do(z2))`.

use std::sync::Arc;

use antecedent_core::{
    DependenceGroup, DistributionAvailability, EvidenceCatalog, EvidenceKind, EvidenceRegime,
    ExecutionContext, InterventionAssignment, RegimeBinding, RegimeId, RegimeKind, SamplingDesign,
    SearchLimits, SearchStop, Value, VariableId,
};
use antecedent_graph::{Admg, DenseNodeId};
use antecedent_identify::{
    MZ_TRANSPORT_DEFAULT_LIMITS, MzTransportDecision, MzTransportQuery, MzTransportRoute,
    SidLimits, ZTransportSourceSpec, decide_mz_transport, verify_mz_transport_obstruction,
};

const Z1: u32 = 0;
const X: u32 = 1;
const Z2: u32 = 2;
const Y: u32 = 3;

fn v(i: u32) -> VariableId {
    VariableId::from_raw(i)
}

/// Shared causal graph of R-443 Figure 1(c,d).
fn figure_1_graph() -> Admg {
    let mut g = Admg::with_variables(4);
    for (a, b) in [(Z1, X), (X, Z2), (Z2, Y)] {
        g.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    for (a, b) in [(Z1, X), (Z1, Z2), (Z1, Y)] {
        g.insert_bidirected(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    g
}

fn source(population: &str, controllable: &[u32], selection: &[u32]) -> ZTransportSourceSpec {
    ZTransportSourceSpec {
        population: Arc::from(population),
        controllable: controllable.iter().copied().map(v).collect::<Vec<_>>().into(),
        experiment_assignment: Arc::from([]),
        selection_targets: selection.iter().copied().map(v).collect::<Vec<_>>().into(),
    }
}

/// Figure 1(c): source `a`, selection on {Z1, Z2}, experiments on {Z2}.
fn source_a() -> ZTransportSourceSpec {
    source("a", &[Z2], &[Z1, Z2])
}

/// Figure 1(d): source `b`, selection on {Z1, Y}, experiments on {Z1}. The formula
/// is constant in `z1`, so `do(Z1)` is cited at one declared level.
fn source_b() -> ZTransportSourceSpec {
    ZTransportSourceSpec {
        experiment_assignment: Arc::from([InterventionAssignment {
            variable: v(Z1),
            value: Value::f64(0.0),
        }]),
        ..source("b", &[Z1], &[Z1, Y])
    }
}

fn query(sources: Vec<ZTransportSourceSpec>) -> MzTransportQuery {
    MzTransportQuery {
        outcomes: Arc::from([v(Y)]),
        treatments: Arc::from([v(X)]),
        target: Arc::from("target"),
        sources: sources.into(),
    }
}

/// An available joint regime over every coordinate it does not intervene on.
fn regime(id: u32, population: &str, on: &[u32]) -> EvidenceRegime {
    let measured = (0..4).filter(|i| !on.contains(i)).map(v).collect::<Vec<_>>();
    EvidenceRegime::try_new(
        RegimeId::from_raw(id),
        if on.is_empty() { RegimeKind::Observational } else { RegimeKind::Experimental },
        EvidenceKind::Available,
        on.iter().copied().map(v).collect::<Vec<_>>(),
        [],
        measured,
        population,
        DistributionAvailability::Joint,
    )
    .unwrap()
}

fn catalog(regimes: Vec<EvidenceRegime>) -> EvidenceCatalog {
    let bindings = regimes
        .iter()
        .map(|r| RegimeBinding {
            dataset_identity: None,
            regime: r.id,
            snapshot_identity: Arc::from(format!("snapshot-{}", r.id.raw())),
            schema_names: Arc::from([]),
            sampling: SamplingDesign::Independent,
            weights: None,
            dependence: DependenceGroup::IndependentStudies,
        })
        .collect::<Vec<_>>();
    EvidenceCatalog::try_new([], regimes, bindings, None).unwrap()
}

/// Target observational law, `do(Z2)` in `a`, `do(Z1)` in `b`.
fn complementary_catalog() -> EvidenceCatalog {
    catalog(vec![regime(0, "target", &[]), regime(1, "a", &[Z2]), regime(2, "b", &[Z1])])
}

fn decide(query: &MzTransportQuery, catalog: &EvidenceCatalog) -> MzTransportDecision {
    decide_mz_transport(
        &figure_1_graph(),
        query,
        catalog,
        MZ_TRANSPORT_DEFAULT_LIMITS,
        &ExecutionContext::for_tests(1),
    )
    .unwrap()
}

#[test]
fn combined_catalog_identifies_what_no_single_source_can() {
    let decision = decide(&query(vec![source_a(), source_b()]), &complementary_catalog());
    let MzTransportDecision::Identified { derivation, cited } = decision else {
        panic!("expected identification, got {decision:?}");
    };
    assert_eq!(
        derivation.route(),
        &MzTransportRoute::Combined { populations: Arc::from([Arc::from("a"), Arc::from("b")]) }
    );
    // Q[Z2] is exchanged into b at the declared Z1 level; Q[Y] into a under do(Z2).
    let exchanges = derivation
        .rules()
        .iter()
        .filter(|rule| rule.starts_with("ztr.line10.source_exchange"))
        .cloned()
        .collect::<Vec<_>>();
    assert_eq!(
        exchanges,
        [
            "ztr.line10.source_exchange:b:[(0, Some(0.0))]",
            "ztr.line10.source_exchange:a:[(2, None)]"
        ]
    );
    assert_eq!(&*cited, [RegimeId::from_raw(1), RegimeId::from_raw(2)]);
}

/// A source that can never exchange: its selection nodes touch every variable.
fn unhelpful() -> ZTransportSourceSpec {
    source("unhelpful", &[X], &[Z1, X, Z2, Y])
}

#[test]
fn each_figure_1_source_alone_is_a_checked_obstruction() {
    for (alone, c0) in [(source_a(), [Z2]), (source_b(), [Y])] {
        let decision = decide(&query(vec![alone.clone(), unhelpful()]), &complementary_catalog());
        let MzTransportDecision::ProvenNonTransportable(obstruction) = decision else {
            panic!("{} alone: expected an obstruction, got {decision:?}", alone.population);
        };
        assert_eq!(obstruction.c0(), c0);
        // At the forced terminal no source can exchange.
        assert!(
            obstruction
                .sources()
                .iter()
                .all(|(_, active, separated)| active.is_empty() || !separated)
        );
        verify_mz_transport_obstruction(
            &figure_1_graph(),
            &obstruction,
            SidLimits::default(),
            &ExecutionContext::for_tests(1),
        )
        .unwrap();
    }
}

#[test]
fn source_declaration_order_does_not_change_the_result() {
    let forward = decide(&query(vec![source_a(), source_b()]), &complementary_catalog());
    let reverse = decide(&query(vec![source_b(), source_a()]), &complementary_catalog());
    let (
        MzTransportDecision::Identified { derivation: f, cited: fc },
        MzTransportDecision::Identified { derivation: r, cited: rc },
    ) = (forward, reverse)
    else {
        panic!("both orders must identify");
    };
    assert_eq!(f.query(), r.query());
    assert_eq!((f.arena(), f.root(), f.rules()), (r.arena(), r.root(), r.rules()));
    assert_eq!(fc, rc);
}

#[test]
fn a_certified_formula_with_an_unsupplied_regime_is_missing_evidence() {
    // b's do(Z1) experiment is declared possible but was never supplied.
    let partial = catalog(vec![regime(0, "target", &[]), regime(1, "a", &[Z2])]);
    let decision = decide(&query(vec![source_a(), source_b()]), &partial);
    let MzTransportDecision::MissingEvidence { derivation: Some(derivation), detail } = decision
    else {
        panic!("expected missing evidence with a certified formula, got {decision:?}");
    };
    assert!(matches!(derivation.route(), MzTransportRoute::Combined { .. }));
    assert!(detail.contains('b'), "{detail}");
}

/// R-443 Figure 1(e,f): Z1 -> X, Z2 -> X, X -> Y, Z1 <-> X, Z1 <-> Y, Z2 <-> X, Z2 <-> Y.
fn figure_1ef_graph() -> Admg {
    let mut g = Admg::with_variables(4);
    for (a, b) in [(Z1, X), (Z2, X), (X, Y)] {
        g.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    for (a, b) in [(Z1, X), (Z1, Y), (Z2, X), (Z2, Y)] {
        g.insert_bidirected(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    g
}

fn assigned(mut spec: ZTransportSourceSpec, levels: &[(u32, f64)]) -> ZTransportSourceSpec {
    spec.experiment_assignment = levels
        .iter()
        .map(|(var, level)| InterventionAssignment { variable: v(*var), value: Value::f64(*level) })
        .collect::<Vec<_>>()
        .into();
    spec
}

fn decide_1ef(query: &MzTransportQuery, catalog: &EvidenceCatalog) -> MzTransportDecision {
    decide_mz_transport(
        &figure_1ef_graph(),
        query,
        catalog,
        MZ_TRANSPORT_DEFAULT_LIMITS,
        &ExecutionContext::for_tests(1),
    )
    .unwrap()
}

#[test]
fn separate_experiments_never_stand_in_for_the_joint_regime() {
    // Figure 1(e): {Z1, Z2} in one source identifies via P^a(y | x, do(z1, z2)).
    let joint =
        query(vec![assigned(source("a", &[Z1, Z2], &[Z1]), &[(Z1, 0.0), (Z2, 0.0)]), unhelpful()]);
    let separate =
        catalog(vec![regime(0, "target", &[]), regime(1, "a", &[Z1]), regime(2, "a", &[Z2])]);
    let MzTransportDecision::MissingEvidence { derivation: Some(_), .. } =
        decide_1ef(&joint, &separate)
    else {
        panic!("do(Z1) and do(Z2) must not bind a do(Z1, Z2) factor");
    };
    let together = catalog(vec![regime(0, "target", &[]), regime(1, "a", &[Z1, Z2])]);
    let MzTransportDecision::Identified { derivation, cited } = decide_1ef(&joint, &together)
    else {
        panic!("the supplied joint regime identifies");
    };
    assert_eq!(derivation.route(), &MzTransportRoute::SingleSource { population: Arc::from("a") });
    assert_eq!(&*cited, [RegimeId::from_raw(1)]);
}

#[test]
fn splitting_the_joint_experiment_across_sources_is_never_identified() {
    // Figure 1(e,f): {Z2} in a and {Z1} in b. One c-factor would need both
    // sources' interventions; the search refuses to fabricate that joint.
    let split = query(vec![
        assigned(source("a", &[Z2], &[Z1]), &[(Z2, 0.0)]),
        assigned(source("b", &[Z1], &[Z2]), &[(Z1, 0.0)]),
    ]);
    let evidence =
        catalog(vec![regime(0, "target", &[]), regime(1, "a", &[Z2]), regime(2, "b", &[Z1])]);
    let MzTransportDecision::NotCertified(inspection) = decide_1ef(&split, &evidence) else {
        panic!("a split joint must stay unresolved, not identified");
    };
    let stages =
        inspection.stages.iter().map(|s| (s.stage.as_str(), s.outcome)).collect::<Vec<_>>();
    assert_eq!(
        stages,
        [
            ("target_only", "not_certified"),
            ("source:a", "obstruction"),
            ("source:b", "obstruction"),
            ("multi_source", "not_certified"),
        ]
    );
    // Both sources were explored by the combined search.
    for population in ["a", "b"] {
        assert!(
            inspection
                .explored_rules
                .iter()
                .any(|r| r.contains(&format!("source_exchange:{population}:")))
        );
    }
}

#[test]
fn the_target_alone_is_tried_before_any_source_experiment() {
    // Without confounding the target identifies P*(y | do(x)) by itself.
    let mut g = Admg::with_variables(4);
    for (a, b) in [(Z1, X), (X, Z2), (Z2, Y)] {
        g.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    let decision = decide_mz_transport(
        &g,
        &query(vec![source_a(), source_b()]),
        &complementary_catalog(),
        MZ_TRANSPORT_DEFAULT_LIMITS,
        &ExecutionContext::for_tests(1),
    )
    .unwrap();
    let MzTransportDecision::Identified { derivation, cited } = decision else {
        panic!("expected target-only identification, got {decision:?}");
    };
    assert_eq!(derivation.route(), &MzTransportRoute::TargetOnly);
    assert_eq!(&*cited, [RegimeId::from_raw(0)]);
}

#[test]
fn exhausted_search_returns_a_receipt_never_a_verdict() {
    let q = query(vec![source_a(), source_b()]);
    let run = |limits: SearchLimits, ctx: &ExecutionContext| {
        let MzTransportDecision::Exhausted(receipt) =
            decide_mz_transport(&figure_1_graph(), &q, &complementary_catalog(), limits, ctx)
                .unwrap()
        else {
            panic!("expected a limits receipt");
        };
        receipt
    };
    let ctx = ExecutionContext::for_tests(1);
    let steps = run(SearchLimits { operations: 3, depth: 24 }, &ctx);
    assert_eq!(steps.stop, SearchStop::Operations);
    assert!(steps.operations_consumed.is_some() && !steps.explored.is_empty());
    assert!(steps.unevaluated.contains(&"multi_source".to_owned()));
    // A zero limit stops before the search is entered, with no fabricated accounting.
    let zero = run(SearchLimits { operations: 0, depth: 24 }, &ctx);
    assert_eq!((zero.stop, zero.operations_consumed), (SearchStop::Operations, None));
    assert_eq!(zero.unevaluated.len(), 4);
    let cancelled = ExecutionContext::for_tests(1);
    cancelled.cancellation.cancel();
    assert_eq!(run(MZ_TRANSPORT_DEFAULT_LIMITS, &cancelled).stop, SearchStop::Cancelled);
}
