//! Bounded multi-source limited-experiment (`TR^mz`) identification.
//!
//! The positive fixture is Bareinboim & Pearl (`NeurIPS` 2014, R-443) Figure 1(c,d):
//! neither source transports `P*(y | do(x))` alone, but experiments on `{Z2}` in
//! one source and `{Z1}` in the other do, through
//! `Σ_z2 P^b(z2 | x, do(z1)) P^a(y | do(z2))` (the paper's Eq. 2). Source `b`'s
//! diagram here also puts a selection node into `Z1`, which the paper's (d) lacks
//! (its only selection node is into `Y`): a stricter diagram that `do(Z1)` cuts,
//! so the same formula transports.

use std::sync::Arc;

use antecedent_core::{
    DependenceGroup, DistributionAvailability, EvidenceCatalog, EvidenceKind, EvidenceRegime,
    ExecutionContext, InterventionAssignment, MemoryBudget, RegimeBinding, RegimeId, RegimeKind,
    SamplingDesign, SearchLimits, SearchReceipt, SearchStop, Value, VariableId,
};
use antecedent_graph::{Admg, DenseNodeId};
use antecedent_identify::{
    IdentificationError, MZ_TRANSPORT_DEFAULT_LIMITS, MzTransportDecision, MzTransportQuery,
    MzTransportRoute, SidLimits, ZTransportSourceSpec, decide_mz_transport,
    verify_mz_transport_obstruction,
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

/// Figure 1(d) with an extra selection node into `Z1`: source `b`, selection on
/// {Z1, Y}, experiments on {Z1}. The formula is constant in `z1`, so `do(Z1)` is
/// cited at one declared level.
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
    regime_over(4, id, population, on)
}

/// [`regime`] over the first `n` coordinates.
fn regime_over(n: u32, id: u32, population: &str, on: &[u32]) -> EvidenceRegime {
    let measured = (0..n).filter(|i| !on.contains(i)).map(v).collect::<Vec<_>>();
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
    assert_eq!(inspection.detail, "mz_transport.fabricated_joint");
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

/// Decide under `limits` and `ctx`, requiring a limits receipt.
fn exhausted(
    graph: &Admg,
    query: &MzTransportQuery,
    limits: SearchLimits,
    ctx: &ExecutionContext,
) -> SearchReceipt {
    match decide_mz_transport(graph, query, &complementary_catalog(), limits, ctx).unwrap() {
        MzTransportDecision::Exhausted(receipt) => receipt,
        other => panic!("expected a limits receipt, never a verdict; got {other:?}"),
    }
}

/// Operations one unbounded decision charges, read from the receipt of a
/// decision stopped one operation short of it.
fn operations_needed(graph: &Admg, query: &MzTransportQuery) -> usize {
    let ctx = ExecutionContext::for_tests(1);
    (1..=MZ_TRANSPORT_DEFAULT_LIMITS.operations)
        .find(|&operations| {
            !matches!(
                decide_mz_transport(
                    graph,
                    query,
                    &complementary_catalog(),
                    SearchLimits { operations, depth: MZ_TRANSPORT_DEFAULT_LIMITS.depth },
                    &ctx,
                )
                .unwrap(),
                MzTransportDecision::Exhausted(_)
            )
        })
        .unwrap()
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
    // One SearchBudget for the whole decision: the receipt reports its totals.
    assert_eq!((steps.operations_limit, steps.operations_consumed), (3, Some(3)));
    assert!(steps.explored.iter().all(|r| r.starts_with("stage:") || r.starts_with("rule:")));
    assert!(steps.unevaluated.contains(&"stage:multi_source".to_owned()));
    // A zero limit stops before the search is entered, with no fabricated accounting.
    let zero = run(SearchLimits { operations: 0, depth: 24 }, &ctx);
    assert_eq!((zero.stop, zero.operations_consumed), (SearchStop::Operations, None));
    assert_eq!(zero.unevaluated.len(), 4);
    let cancelled = ExecutionContext::for_tests(1);
    cancelled.cancellation.cancel();
    assert_eq!(run(MZ_TRANSPORT_DEFAULT_LIMITS, &cancelled).stop, SearchStop::Cancelled);
}

#[test]
fn depth_exhaustion_is_a_receipt_with_search_budget_depth_semantics() {
    let q = query(vec![source_a(), source_b()]);
    let receipt = exhausted(
        &figure_1_graph(),
        &q,
        SearchLimits { operations: 4096, depth: 1 },
        &ExecutionContext::for_tests(1),
    );
    assert_eq!(receipt.stop, SearchStop::Depth);
    // SearchBudget stops only past the limit: depth 1 is charged, depth 2 is not.
    assert_eq!((receipt.depth_limit, receipt.depth_reached), (1, Some(2)));
    assert!(receipt.operations_consumed.is_some_and(|n| n >= 2));
}

#[test]
fn memory_exhaustion_is_a_receipt_and_is_checked_on_every_charge() {
    let q = query(vec![source_a(), source_b()]);
    // One live engine step on the four-node graph is estimated at 1536 bytes.
    for (limit, consumed) in [(1024, None), (3 * 1536, Some(3))] {
        let mut ctx = ExecutionContext::for_tests(1);
        ctx.memory = MemoryBudget { soft_limit_bytes: None, hard_limit_bytes: Some(limit) };
        let receipt = exhausted(&figure_1_graph(), &q, MZ_TRANSPORT_DEFAULT_LIMITS, &ctx);
        assert_eq!(receipt.stop, SearchStop::Memory);
        assert_eq!(receipt.memory_limit_bytes, Some(limit));
        assert_eq!(receipt.operations_consumed, consumed);
    }
}

#[test]
fn one_budget_is_shared_by_every_stage() {
    // The obstruction of `a` alone runs every stage: target, both sources, the
    // combined search and its replay. Measure what each stage costs by the
    // smallest limit under which it finishes.
    let (graph, q) = (figure_1_graph(), query(vec![source_a(), unhelpful()]));
    let ctx = ExecutionContext::for_tests(1);
    let total = operations_needed(&graph, &q);
    let mut finished_at = vec![0];
    for operations in 1..total {
        let receipt = exhausted(&graph, &q, SearchLimits { operations, depth: 24 }, &ctx);
        let stages = receipt.explored.iter().filter(|r| r.starts_with("stage:")).count();
        if stages >= finished_at.len() {
            finished_at.push(operations);
        }
    }
    finished_at.push(total);
    let largest_stage = finished_at.windows(2).map(|w| w[1] - w[0]).max().unwrap();
    assert!(finished_at.len() >= 4 && largest_stage < total, "{finished_at:?}");
    // A limit every stage fits inside on its own still exhausts the decision,
    // and the receipt reports the operations charged across all stages.
    let limits = SearchLimits { operations: largest_stage, depth: 24 };
    let receipt = exhausted(&graph, &q, limits, &ctx);
    assert_eq!(receipt.stop, SearchStop::Operations);
    assert_eq!(receipt.operations_consumed, Some(largest_stage));
    assert!(!receipt.unevaluated.is_empty());
}

#[test]
fn limits_above_the_declared_maxima_refuse_as_bounds_exceeded() {
    let q = query(vec![source_a(), source_b()]);
    let exceeded = IdentificationError::UnsupportedInput { code: "mz_transport.bounds_exceeded" };
    for limits in [
        SearchLimits { operations: MZ_TRANSPORT_DEFAULT_LIMITS.operations + 1, depth: 24 },
        SearchLimits { operations: 4096, depth: MZ_TRANSPORT_DEFAULT_LIMITS.depth + 1 },
    ] {
        let refused = decide_mz_transport(
            &figure_1_graph(),
            &q,
            &complementary_catalog(),
            limits,
            &ExecutionContext::for_tests(1),
        );
        assert_eq!(refused.unwrap_err(), exceeded);
    }
}

#[test]
fn graph_and_controllable_bounds_refuse_as_bounds_exceeded() {
    let exceeded = IdentificationError::UnsupportedInput { code: "mz_transport.bounds_exceeded" };
    let ctx = ExecutionContext::for_tests(1);
    // Thirteen observed variables.
    let mut large = Admg::with_variables(13);
    for (a, b) in [(Z1, X), (X, Z2), (Z2, Y)] {
        large.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    let q = query(vec![source_a(), source_b()]);
    let refused =
        decide_mz_transport(&large, &q, &catalog(vec![]), MZ_TRANSPORT_DEFAULT_LIMITS, &ctx);
    assert_eq!(refused.unwrap_err(), exceeded);
    // Five controllables in one source.
    let mut wide = Admg::with_variables(6);
    for (a, b) in [(Z1, X), (X, Z2), (Z2, Y)] {
        wide.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    let q = query(vec![source("a", &[0, 2, 3, 4, 5], &[]), source_b()]);
    let refused =
        decide_mz_transport(&wide, &q, &catalog(vec![]), MZ_TRANSPORT_DEFAULT_LIMITS, &ctx);
    assert_eq!(refused.unwrap_err(), exceeded);
}

/// The observable content of a decision, which must not depend on source order.
fn outcome_signature(decision: &MzTransportDecision) -> String {
    match decision {
        MzTransportDecision::Identified { derivation, cited } => {
            format!("identified {:?} {:?} {cited:?}", derivation.route(), derivation.rules())
        }
        MzTransportDecision::ProvenNonTransportable(obstruction) => format!(
            "obstruction {:?} {:?} {:?}",
            obstruction.query(),
            obstruction.c0(),
            obstruction.sources()
        ),
        MzTransportDecision::MissingEvidence { derivation, detail } => format!(
            "missing {detail} {:?}",
            derivation.as_ref().map(|d| (
                d.route().clone(),
                d.rules().to_vec(),
                d.stages().to_vec()
            ))
        ),
        MzTransportDecision::NotCertified(inspection) => format!("not_certified {inspection:?}"),
        MzTransportDecision::Exhausted(receipt) => format!("exhausted {receipt:?}"),
    }
}

#[test]
fn every_outcome_is_invariant_to_source_declaration_order() {
    let partial = catalog(vec![regime(0, "target", &[]), regime(1, "a", &[Z2])]);
    let split_evidence =
        catalog(vec![regime(0, "target", &[]), regime(1, "a", &[Z2]), regime(2, "b", &[Z1])]);
    let split = [
        assigned(source("a", &[Z2], &[Z1]), &[(Z2, 0.0)]),
        assigned(source("b", &[Z1], &[Z2]), &[(Z1, 0.0)]),
    ];
    let cases: Vec<(Admg, [ZTransportSourceSpec; 2], EvidenceCatalog, SearchLimits, &str)> = vec![
        (
            figure_1_graph(),
            [source_a(), unhelpful()],
            complementary_catalog(),
            MZ_TRANSPORT_DEFAULT_LIMITS,
            "transport_proven_non_transportable",
        ),
        (
            figure_1_graph(),
            [source_a(), source_b()],
            partial,
            MZ_TRANSPORT_DEFAULT_LIMITS,
            "transport_missing_evidence",
        ),
        (
            figure_1ef_graph(),
            split,
            split_evidence,
            MZ_TRANSPORT_DEFAULT_LIMITS,
            "transport_not_certified",
        ),
        (
            figure_1_graph(),
            [source_a(), source_b()],
            complementary_catalog(),
            SearchLimits {
                operations: operations_needed(
                    &figure_1_graph(),
                    &query(vec![source_a(), source_b()]),
                ) - 1,
                depth: 24,
            },
            "transport_budget_cancel",
        ),
    ];
    let ctx = ExecutionContext::for_tests(1);
    for (graph, [first, second], evidence, limits, reason) in cases {
        let decide = |sources: Vec<ZTransportSourceSpec>| {
            decide_mz_transport(&graph, &query(sources), &evidence, limits, &ctx).unwrap()
        };
        let forward = decide(vec![first.clone(), second.clone()]);
        let reverse = decide(vec![second, first]);
        assert_eq!(forward.reason_code(), Some(reason), "{forward:?}");
        assert_eq!(outcome_signature(&forward), outcome_signature(&reverse));
    }
}

#[test]
fn a_budget_stop_in_the_combined_search_names_untried_source_branches() {
    // `a` and its twin `c` can both exchange Q[Y] at the same line-10 state;
    // a stop inside `a`'s branch leaves `c`'s untried, named by population.
    let q = query(vec![source_a(), source_b(), source("c", &[Z2], &[Z1, Z2])]);
    let total = operations_needed(&figure_1_graph(), &q);
    let ctx = ExecutionContext::for_tests(1);
    let mut named = Vec::new();
    for operations in 1..total {
        let receipt =
            exhausted(&figure_1_graph(), &q, SearchLimits { operations, depth: 24 }, &ctx);
        named.extend(receipt.unevaluated.into_iter().filter(|r| r.starts_with("multi:line10:")));
    }
    assert!(!named.is_empty(), "no stop left a line-10 source branch untried");
    assert!(named.iter().all(|r| r.starts_with("multi:line10:source=c@")), "{named:?}");
}

#[test]
fn outcomes_carry_the_frozen_detail_codes() {
    let identified = decide(&query(vec![source_a(), source_b()]), &complementary_catalog());
    assert_eq!((identified.reason_code(), identified.detail_code()), (None, None));
    let obstruction = decide(&query(vec![source_a(), unhelpful()]), &complementary_catalog());
    assert_eq!(obstruction.detail_code(), Some("mz_transport.checked_obstruction"));
    let partial = catalog(vec![regime(0, "target", &[]), regime(1, "a", &[Z2])]);
    let missing = decide(&query(vec![source_a(), source_b()]), &partial);
    assert_eq!(missing.detail_code(), Some("mz_transport.missing_joint_regime"));
    let stopped = decide_mz_transport(
        &figure_1_graph(),
        &query(vec![source_a(), source_b()]),
        &complementary_catalog(),
        SearchLimits { operations: 1, depth: 24 },
        &ExecutionContext::for_tests(1),
    )
    .unwrap();
    assert_eq!(stopped.detail_code(), Some("mz_transport.budget"));
}

fn identified(decision: MzTransportDecision) -> Box<antecedent_identify::MzTransportDerivation> {
    let MzTransportDecision::Identified { derivation, .. } = decision else {
        panic!("expected identification, got {decision:?}");
    };
    derivation
}

#[test]
fn the_success_path_receipt_is_deterministic_and_matches_the_operations_charged() {
    let q = query(vec![source_a(), source_b()]);
    let derivation = identified(decide(&q, &complementary_catalog()));
    let receipt = derivation.search_record();
    assert_eq!(
        (receipt.operations_limit, receipt.depth_limit),
        (MZ_TRANSPORT_DEFAULT_LIMITS.operations, MZ_TRANSPORT_DEFAULT_LIMITS.depth)
    );
    // The memory bound is never absent: the default cap when the context sets none.
    assert_eq!(receipt.memory_limit_bytes, antecedent_identify::MZ_TRANSPORT_MEMORY_BYTES);
    // Every stage ran up to the identifying combined search.
    assert_eq!(
        receipt.explored,
        ["stage:target_only", "stage:source:a", "stage:source:b", "stage:multi_source"]
    );
    assert!(receipt.unevaluated.is_empty());
    // The smallest limit under which the decision still identifies is exactly
    // what the receipt reports as consumed: one budget across all stages.
    assert_eq!(receipt.operations_consumed, operations_needed(&figure_1_graph(), &q));
    assert!(receipt.depth_reached >= 1);
    // Deterministic: another context and the other declaration order agree.
    let again = identified(
        decide_mz_transport(
            &figure_1_graph(),
            &query(vec![source_b(), source_a()]),
            &complementary_catalog(),
            MZ_TRANSPORT_DEFAULT_LIMITS,
            &ExecutionContext::for_tests(4),
        )
        .unwrap(),
    );
    assert_eq!(again.search_record(), receipt);
    // A stage that identifies early leaves later stages unevaluated.
    let mut g = Admg::with_variables(4);
    for (a, b) in [(Z1, X), (X, Z2), (Z2, Y)] {
        g.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    let early = identified(
        decide_mz_transport(
            &g,
            &q,
            &complementary_catalog(),
            MZ_TRANSPORT_DEFAULT_LIMITS,
            &ExecutionContext::for_tests(1),
        )
        .unwrap(),
    );
    assert_eq!(early.search_record().explored, ["stage:target_only"]);
    assert_eq!(early.search_record().unevaluated.len(), 3);
    assert!(early.search_record().operations_consumed < receipt.operations_consumed);
}

type Edit = Box<dyn Fn(&mut antecedent_identify::MzTransportDerivationRecord)>;

#[test]
fn a_checked_replay_reproduces_the_stored_search_receipt_and_refuses_any_edit() {
    use antecedent_identify::MzTransportDerivation;
    let (graph, q, catalog) =
        (figure_1_graph(), query(vec![source_a(), source_b()]), complementary_catalog());
    let derivation = identified(decide(&q, &catalog));
    let record = derivation.to_record();
    let ctx = ExecutionContext::for_tests(1);
    let replay = |record: &antecedent_identify::MzTransportDerivationRecord,
                  limits: SearchLimits,
                  ctx: &ExecutionContext| {
        MzTransportDerivation::from_record_checked(
            &graph,
            &q,
            &catalog,
            record,
            derivation.arena(),
            limits,
            ctx,
        )
    };
    replay(&record, MZ_TRANSPORT_DEFAULT_LIMITS, &ctx).unwrap();
    let invalid = |e: IdentificationError| e.to_string();
    let bounds = IdentificationError::UnsupportedInput { code: "mz_transport.bounds_exceeded" };
    let edits: Vec<(&str, Edit)> = vec![
        ("operations consumed", Box::new(|r| r.search.operations_consumed += 1)),
        ("depth reached", Box::new(|r| r.search.depth_reached += 1)),
        ("explored", Box::new(|r| r.search.explored.pop().map(drop).unwrap_or_default())),
        ("unevaluated", Box::new(|r| r.search.unevaluated.push("stage:x".into()))),
        // The replay runs under the stored limits, so a limit the decision cannot
        // fit inside never replays to the stored receipt.
        (
            "operation limit",
            Box::new(|r| r.search.operations_limit = r.search.operations_consumed - 1),
        ),
        ("depth limit", Box::new(|r| r.search.depth_limit = r.search.depth_reached - 1)),
        ("memory limit", Box::new(|r| r.search.memory_limit_bytes = 1)),
    ];
    for (name, edit) in &edits {
        let mut edited = record.clone();
        edit(&mut edited);
        let error = replay(&edited, MZ_TRANSPORT_DEFAULT_LIMITS, &ctx)
            .expect_err(&format!("the edited {name} must not replay"));
        assert!(
            matches!(error, IdentificationError::InvalidDerivation { .. }),
            "{name}: {}",
            invalid(error)
        );
    }
    // A lower stored limit that still suffices replays to its own, consistent
    // receipt (the artifact binds these limits into its premises digest).
    let mut lower = record.clone();
    lower.search.operations_limit -= 1;
    replay(&lower, MZ_TRANSPORT_DEFAULT_LIMITS, &ctx).unwrap();
    // A stored limit above the consumer's own maximum refuses, never replays.
    let mut wide = record.clone();
    wide.search.memory_limit_bytes += 1;
    assert_eq!(replay(&wide, MZ_TRANSPORT_DEFAULT_LIMITS, &ctx).unwrap_err(), bounds);
    let tight = SearchLimits { operations: record.search.operations_limit - 1, depth: 24 };
    assert_eq!(replay(&record, tight, &ctx).unwrap_err(), bounds);
    let mut limited = ExecutionContext::for_tests(1);
    limited.memory = MemoryBudget { soft_limit_bytes: None, hard_limit_bytes: Some(1 << 20) };
    assert_eq!(replay(&record, MZ_TRANSPORT_DEFAULT_LIMITS, &limited).unwrap_err(), bounds);
}

#[test]
fn every_identification_detail_pairs_with_its_recorded_reason_code() {
    use antecedent_identify::{mz_transport_refusal, validate_mz_transport_query};
    let ctx = ExecutionContext::for_tests(1);
    let (graph, q, full) =
        (figure_1_graph(), query(vec![source_a(), source_b()]), complementary_catalog());
    let pair = |error: &IdentificationError| mz_transport_refusal(error).unwrap();
    // A bound: `route_not_supported`.
    let bound = decide_mz_transport(
        &graph,
        &q,
        &full,
        SearchLimits { operations: MZ_TRANSPORT_DEFAULT_LIMITS.operations + 1, depth: 24 },
        &ctx,
    )
    .unwrap_err();
    assert_eq!(pair(&bound), ("route_not_supported", "mz_transport.bounds_exceeded"));
    // An invalid query: `invalid_argument`.
    let same_population = query(vec![source("target", &[Z2], &[Z1]), source_b()]);
    let invalid_query =
        decide_mz_transport(&graph, &same_population, &full, MZ_TRANSPORT_DEFAULT_LIMITS, &ctx)
            .unwrap_err();
    assert_eq!(pair(&invalid_query), ("invalid_argument", "mz_transport.invalid_query"));
    // An invalid catalog (the target supplies an experiment): `invalid_argument`.
    let experimental_target = catalog(vec![regime(0, "target", &[Z2])]);
    let invalid_catalog =
        validate_mz_transport_query(&graph, &q, &experimental_target).unwrap_err();
    assert_eq!(pair(&invalid_catalog), ("invalid_argument", "mz_transport.invalid_catalog"));
    // An obstruction that does not verify: `transport_not_certified`.
    let MzTransportDecision::ProvenNonTransportable(obstruction) =
        decide(&query(vec![source_a(), unhelpful()]), &full)
    else {
        panic!("expected an obstruction");
    };
    let mut other_graph = figure_1_graph();
    other_graph.insert_directed(DenseNodeId::from_raw(Z1), DenseNodeId::from_raw(Y)).unwrap();
    let invalid_obstruction =
        verify_mz_transport_obstruction(&other_graph, &obstruction, SidLimits::default(), &ctx)
            .unwrap_err();
    assert_eq!(
        pair(&invalid_obstruction),
        ("transport_not_certified", "mz_transport.invalid_obstruction")
    );
    // A derivation that does not replay: `transport_not_certified`.
    let derivation = identified(decide(&q, &full));
    let mut record = derivation.to_record();
    record.search.operations_consumed += 1;
    let invalid_derivation = antecedent_identify::MzTransportDerivation::from_record_checked(
        &graph,
        &q,
        &full,
        &record,
        derivation.arena(),
        MZ_TRANSPORT_DEFAULT_LIMITS,
        &ctx,
    )
    .unwrap_err();
    assert_eq!(
        pair(&invalid_derivation),
        ("transport_not_certified", "mz_transport.invalid_derivation")
    );
    // A cited regime the catalog lacks: `transport_missing_evidence`.
    let missing = antecedent_identify::bind_mz_transport_catalog(
        &graph,
        &derivation,
        &catalog(vec![regime(0, "target", &[]), regime(1, "a", &[Z2])]),
    )
    .unwrap_err();
    assert_eq!(pair(&missing), ("transport_missing_evidence", "mz_transport.missing_joint_regime"));
    // A stop: `transport_budget_cancel`.
    assert_eq!(
        pair(&IdentificationError::Cancelled),
        ("transport_budget_cancel", "mz_transport.budget")
    );
    // Errors the route does not own carry no pair.
    assert_eq!(mz_transport_refusal(&IdentificationError::msg("other")), None);
    // Every decision outcome carries its own frozen pair.
    for (decision, expected) in [
        (decide(&query(vec![source_a(), unhelpful()]), &full), "mz_transport.checked_obstruction"),
        (
            decide(&q, &catalog(vec![regime(0, "target", &[]), regime(1, "a", &[Z2])])),
            "mz_transport.missing_joint_regime",
        ),
    ] {
        assert_eq!(decision.detail_code(), Some(expected));
    }
}

/// A regime of `population` under `do(Z1 = 0)`, declaring the level.
fn at_z1_zero(id: u32, population: &str) -> EvidenceRegime {
    let measured = (0..4).filter(|i| *i != Z1).map(v).collect::<Vec<_>>();
    EvidenceRegime::try_new(
        RegimeId::from_raw(id),
        RegimeKind::Experimental,
        EvidenceKind::Available,
        [v(Z1)],
        [InterventionAssignment { variable: v(Z1), value: Value::f64(0.0) }],
        measured,
        population,
        DistributionAvailability::Joint,
    )
    .unwrap()
}

/// The combined route binds duplicate evidence by the single-source rule: a
/// family regime (no declared levels) before a concrete one, otherwise the
/// lowest regime id, whatever the catalog order. Supplying a factor twice is
/// never refused, and the same rule binds the published functional.
#[test]
fn combined_route_binds_duplicate_evidence_by_the_single_source_rule() {
    let q = query(vec![source_a(), source_b()]);
    let cases: [(Vec<EvidenceRegime>, [u32; 2]); 2] = [
        // Two family regimes per source: the lowest id of each.
        (
            vec![
                regime(0, "target", &[]),
                regime(5, "a", &[Z2]),
                regime(3, "a", &[Z2]),
                regime(7, "b", &[Z1]),
                regime(4, "b", &[Z1]),
            ],
            [3, 4],
        ),
        // b's concrete do(Z1 = 0) regime has the lower id; its family regime wins.
        (
            vec![
                regime(0, "target", &[]),
                regime(1, "a", &[Z2]),
                at_z1_zero(2, "b"),
                regime(8, "b", &[Z1]),
            ],
            [1, 8],
        ),
    ];
    for (regimes, expected) in cases {
        let expected = expected.map(RegimeId::from_raw);
        for reversed in [false, true] {
            let mut ordered = regimes.clone();
            if reversed {
                ordered.reverse();
            }
            let supplied = catalog(ordered);
            let MzTransportDecision::Identified { derivation, cited } = decide(&q, &supplied)
            else {
                panic!("duplicate evidence must still identify (reversed: {reversed})");
            };
            assert_eq!(&*cited, expected, "reversed: {reversed}");
            let bound = antecedent_identify::bind_mz_transport_catalog(
                &figure_1_graph(),
                &derivation,
                &supplied,
            )
            .unwrap();
            assert_eq!(bound.cited_regimes(), expected, "reversed: {reversed}");
        }
    }
}

/// The decision of the Figure 1(e,f) graph, whatever the catalog supplies.
fn decide_split_1ef(sources: Vec<ZTransportSourceSpec>) -> MzTransportDecision {
    let evidence =
        catalog(vec![regime(0, "target", &[]), regime(1, "a", &[Z2]), regime(2, "b", &[Z1])]);
    decide_1ef(&query(sources), &evidence)
}

/// DELIBERATE INCOMPLETENESS (paper-vs-code departure 1). R-443 states that
/// Figure 1(e,f), with an experiment on {Z2} in one source and on {Z1} in the
/// other, is NOT mz-transportable. The search reaches that state only after a
/// line-10 exchange (an experiment is active), where the forced line-11 terminal
/// that certifies an obstruction is unavailable, so it reports `not_certified`
/// with the detail `mz_transport.fabricated_joint` and never claims the paper's
/// verdict. The sound direction is kept: nothing is identified and nothing is
/// proven impossible by this route.
#[test]
fn fig_1ef_split_is_non_transportable_in_the_paper_but_deliberately_not_certified_here() {
    let split = || {
        vec![
            assigned(source("a", &[Z2], &[Z1]), &[(Z2, 0.0)]),
            assigned(source("b", &[Z1], &[Z2]), &[(Z1, 0.0)]),
        ]
    };
    let mut reversed = split();
    reversed.reverse();
    for sources in [split(), reversed] {
        let decision = decide_split_1ef(sources);
        // The paper would answer "not transportable"; this contract does not.
        assert!(
            !matches!(decision, MzTransportDecision::ProvenNonTransportable(_)),
            "an obstruction after an exchange must never be claimed: {decision:?}"
        );
        assert_ne!(decision.reason_code(), Some("transport_proven_non_transportable"));
        assert_ne!(decision.detail_code(), Some("mz_transport.checked_obstruction"));
        assert!(!matches!(decision, MzTransportDecision::Identified { .. }));
        let MzTransportDecision::NotCertified(inspection) = &decision else {
            panic!("expected not_certified, got {decision:?}");
        };
        assert_eq!(inspection.detail, "mz_transport.fabricated_joint");
        assert_eq!(decision.reason_code(), Some("transport_not_certified"));
        // The combined search ended not_certified, not in an obstruction.
        assert_eq!(inspection.stages.last().map(|s| s.outcome), Some("not_certified"));
        // The non-claim is a failure reached after an exchange: an experiment was
        // active at the line-11 state, which is exactly what disqualifies it.
        let rules = &inspection.explored_rules;
        let first_exchange = rules.iter().position(|r| r.starts_with("ztr.line10.source_exchange"));
        let first_fail = rules.iter().position(|r| r == "ztr.line11.fail");
        assert!(first_exchange.is_some() && first_fail > first_exchange, "{rules:?}");
    }
}

/// A three-node analogue of R-443 Figure 2(a,b) (`Z -> X -> Y`, `X <-> Y`): source
/// `a` experiments on `X` but its selection node points into `Y`, source `b`
/// experiments on `Z` and its selection node points into `X`. Neither experiment
/// set transports `P*(y | do(x))`, and no combination helps: the obstruction is a
/// forced line-11 terminal with no active experiment. The paper's own Figure 2
/// graph is `fig_2_graph_exact` below; this analogue is kept as a second
/// obstruction of the same shape. Both two-model counterexamples are executed in
/// `antecedent-estimate`'s `mz_transport_execution`.
const FIG2_Z: u32 = 0;
const FIG2_X: u32 = 1;
const FIG2_Y: u32 = 2;

fn fig_2_graph() -> Admg {
    let mut g = Admg::with_variables(3);
    for (a, b) in [(FIG2_Z, FIG2_X), (FIG2_X, FIG2_Y)] {
        g.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    g.insert_bidirected(DenseNodeId::from_raw(FIG2_X), DenseNodeId::from_raw(FIG2_Y)).unwrap();
    g
}

fn fig_2_sources() -> Vec<ZTransportSourceSpec> {
    vec![
        assigned(source("a", &[FIG2_X], &[FIG2_Y]), &[(FIG2_X, 0.0)]),
        assigned(source("b", &[FIG2_Z], &[FIG2_X]), &[(FIG2_Z, 0.0)]),
    ]
}

fn fig_2_query(sources: Vec<ZTransportSourceSpec>) -> MzTransportQuery {
    MzTransportQuery {
        outcomes: Arc::from([v(FIG2_Y)]),
        treatments: Arc::from([v(FIG2_X)]),
        target: Arc::from("target"),
        sources: sources.into(),
    }
}

#[test]
fn fig_2_analogue_with_an_x_experiment_and_a_z_experiment_is_a_replayed_obstruction() {
    let on = |id, population: &str, on: &[u32]| regime_over(3, id, population, on);
    // Every regime the paper's information family supplies: the target's
    // observational law, source a's do(X) and source b's do(Z).
    let evidence =
        catalog(vec![on(0, "target", &[]), on(1, "a", &[FIG2_X]), on(2, "b", &[FIG2_Z])]);
    let graph = fig_2_graph();
    let ctx = ExecutionContext::for_tests(1);
    let mut reversed = fig_2_sources();
    reversed.reverse();
    let mut signatures = Vec::new();
    for sources in [fig_2_sources(), reversed] {
        let decision = decide_mz_transport(
            &graph,
            &fig_2_query(sources),
            &evidence,
            MZ_TRANSPORT_DEFAULT_LIMITS,
            &ctx,
        )
        .unwrap();
        let MzTransportDecision::ProvenNonTransportable(obstruction) = &decision else {
            panic!("the Figure 2 analogue is non-transportable, got {decision:?}");
        };
        assert_eq!(decision.reason_code(), Some("transport_proven_non_transportable"));
        assert_eq!(decision.detail_code(), Some("mz_transport.checked_obstruction"));
        // The failing c-component is {Y}, inside the confounded {X, Y}.
        assert_eq!(obstruction.c0(), [FIG2_Y]);
        // No source can exchange at the terminal, each for its own reason: a
        // holds the active X but S_a -> Y is not separated from Y; b's selection
        // node is separated (S_b -> X is cut by the edges-into-X removal) but its
        // Z never meets the treatments.
        let mut per_source = obstruction.sources();
        per_source.sort();
        assert_eq!(per_source, [("a", &[FIG2_X][..], false), ("b", &[][..], true)]);
        assert!(
            obstruction
                .sources()
                .iter()
                .all(|(_, active, separated)| active.is_empty() || !separated)
        );
        // The independent checker replays the same terminal.
        verify_mz_transport_obstruction(&graph, obstruction, SidLimits::default(), &ctx).unwrap();
        signatures.push(outcome_signature(&decision));
    }
    assert_eq!(signatures[0], signatures[1]);
}

/// R-443 Figure 2(a,b) exactly: `X -> Y <- Z` with `X <-> Y` and `Z <-> Y`;
/// source `a` (diagram (a), `S_a -> Z`) experiments on `{X}`, source `b`
/// (diagram (b), `S_b -> Y`) experiments on `{Z}`. The paper states that
/// `P*(y | do(x))` is not mz-transportable here (Theorem 3's worked example:
/// `F' = {Y, Z}`, `F = F' ∪ {X}` with a selection node into `F'` in both
/// domains), and the search certifies exactly that as a forced line-11 terminal.
const FIG2_EXACT_X: u32 = 0;
const FIG2_EXACT_Z: u32 = 1;
const FIG2_EXACT_Y: u32 = 2;

fn fig_2_graph_exact() -> Admg {
    let mut g = Admg::with_variables(3);
    for (a, b) in [(FIG2_EXACT_X, FIG2_EXACT_Y), (FIG2_EXACT_Z, FIG2_EXACT_Y)] {
        g.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    for (a, b) in [(FIG2_EXACT_X, FIG2_EXACT_Y), (FIG2_EXACT_Z, FIG2_EXACT_Y)] {
        g.insert_bidirected(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    g
}

fn fig_2_sources_exact() -> Vec<ZTransportSourceSpec> {
    vec![
        assigned(source("a", &[FIG2_EXACT_X], &[FIG2_EXACT_Z]), &[(FIG2_EXACT_X, 0.0)]),
        assigned(source("b", &[FIG2_EXACT_Z], &[FIG2_EXACT_Y]), &[(FIG2_EXACT_Z, 0.0)]),
    ]
}

#[test]
fn fig_2_with_an_x_experiment_and_a_z_experiment_is_a_replayed_obstruction() {
    let on = |id, population: &str, on: &[u32]| regime_over(3, id, population, on);
    // The paper's information family: the target's observational law, source a's
    // do(X) and source b's do(Z).
    let evidence = catalog(vec![
        on(0, "target", &[]),
        on(1, "a", &[FIG2_EXACT_X]),
        on(2, "b", &[FIG2_EXACT_Z]),
    ]);
    let graph = fig_2_graph_exact();
    let ctx = ExecutionContext::for_tests(1);
    let mut reversed = fig_2_sources_exact();
    reversed.reverse();
    let mut signatures = Vec::new();
    for sources in [fig_2_sources_exact(), reversed] {
        let q = MzTransportQuery {
            outcomes: Arc::from([v(FIG2_EXACT_Y)]),
            treatments: Arc::from([v(FIG2_EXACT_X)]),
            target: Arc::from("target"),
            sources: sources.into(),
        };
        let decision =
            decide_mz_transport(&graph, &q, &evidence, MZ_TRANSPORT_DEFAULT_LIMITS, &ctx).unwrap();
        let MzTransportDecision::ProvenNonTransportable(obstruction) = &decision else {
            panic!("R-443 Figure 2 is non-transportable, got {decision:?}");
        };
        assert_eq!(decision.reason_code(), Some("transport_proven_non_transportable"));
        assert_eq!(decision.detail_code(), Some("mz_transport.checked_obstruction"));
        // The failing c-component is the paper's F' = {Y, Z}, inside F = {X, Y, Z}.
        assert_eq!(obstruction.c0(), [FIG2_EXACT_Z, FIG2_EXACT_Y]);
        // No source can exchange at the terminal: a holds the active X but
        // S_a -> Z -> Y stays open given X; b's Z never meets the treatments and
        // S_b -> Y is not separated either.
        let mut per_source = obstruction.sources();
        per_source.sort();
        assert_eq!(per_source, [("a", &[FIG2_EXACT_X][..], false), ("b", &[][..], false)]);
        // The independent checker replays the same terminal.
        verify_mz_transport_obstruction(&graph, obstruction, SidLimits::default(), &ctx).unwrap();
        signatures.push(outcome_signature(&decision));
    }
    assert_eq!(signatures[0], signatures[1]);
}

/// Target experiments (`Z* != {}`) are supported by R-443 and refused here by
/// design: the contract gives the target observational evidence only. The
/// refusal is the catalog validation refusal `mz_transport.invalid_catalog`
/// (reason code `invalid_argument`), raised before any search, never a
/// verdict, even when the target experiment would identify the effect at once.
#[test]
fn target_experiments_are_refused_by_design_though_the_paper_supports_them() {
    use antecedent_identify::{mz_transport_refusal, validate_mz_transport_query};
    let ctx = ExecutionContext::for_tests(1);
    let q = query(vec![source_a(), source_b()]);
    // The target ran do(X): P*(y | do(x)) is directly observed, yet refused.
    let with_target_experiment = catalog(vec![
        regime(0, "target", &[]),
        regime(1, "target", &[X]),
        regime(2, "a", &[Z2]),
        regime(3, "b", &[Z1]),
    ]);
    for refused in [
        decide_mz_transport(
            &figure_1_graph(),
            &q,
            &with_target_experiment,
            MZ_TRANSPORT_DEFAULT_LIMITS,
            &ctx,
        )
        .unwrap_err(),
        validate_mz_transport_query(&figure_1_graph(), &q, &with_target_experiment).unwrap_err(),
    ] {
        assert!(matches!(refused, IdentificationError::InvalidCatalog { .. }), "{refused:?}");
        assert!(refused.to_string().contains("target regime 1 is experimental"), "{refused}");
        assert_eq!(
            mz_transport_refusal(&refused),
            Some(("invalid_argument", "mz_transport.invalid_catalog"))
        );
    }
    // The same query with only observational target evidence decides normally.
    let observational =
        catalog(vec![regime(0, "target", &[]), regime(2, "a", &[Z2]), regime(3, "b", &[Z1])]);
    assert!(matches!(decide(&q, &observational), MzTransportDecision::Identified { .. }));
    // A target experiment on any other variable is refused the same way.
    for on in [&[Z1][..], &[Z2], &[Z1, Z2]] {
        let evidence = catalog(vec![regime(0, "target", &[]), regime(1, "target", on)]);
        let refused = decide_mz_transport(
            &figure_1_graph(),
            &q,
            &evidence,
            MZ_TRANSPORT_DEFAULT_LIMITS,
            &ctx,
        )
        .unwrap_err();
        assert!(matches!(refused, IdentificationError::InvalidCatalog { .. }), "{on:?}");
    }
}

/// A second source `c` that certifies exactly what `a` certifies: same
/// controllable {Z2} and selection on {Z1, Z2}.
fn twin_of_a() -> ZTransportSourceSpec {
    source("c", &[Z2], &[Z1, Z2])
}

/// DEPARTURE FROM FIG. 3 LINE 11 (paper-vs-code departure 5). When two sources
/// both certify the factor `Q[Y]`, the paper returns a weighted combination of
/// the certifying sources. This search returns only the first certifying source
/// in canonical (population-sorted) order and does not retain the other. Either
/// is a valid formula for exact laws, so the result is the same with either
/// source alone (its numeric agreement with enumerated truth is executed in
/// `mz_transport_execution::two_sources_certifying_one_factor_give_the_same_truth_either_way`).
#[test]
fn two_sources_certifying_one_factor_yield_the_first_canonical_source_in_any_order() {
    let all = catalog(vec![
        regime(0, "target", &[]),
        regime(1, "a", &[Z2]),
        regime(2, "b", &[Z1]),
        regime(3, "c", &[Z2]),
    ]);
    let ids = |cited: &[RegimeId]| cited.iter().map(|r| r.raw()).collect::<Vec<_>>();
    let mut results = Vec::new();
    for order in [[0, 1, 2], [2, 1, 0], [1, 2, 0], [2, 0, 1]] {
        let pool = [source_a(), source_b(), twin_of_a()];
        let sources = order.iter().map(|i| pool[*i].clone()).collect::<Vec<_>>();
        let MzTransportDecision::Identified { derivation, cited } = decide(&query(sources), &all)
        else {
            panic!("a twin source must not change identifiability");
        };
        // First certifying source in canonical order: a (not c); c is not retained.
        assert_eq!(
            derivation.route(),
            &MzTransportRoute::Combined {
                populations: Arc::from([Arc::from("a"), Arc::from("b")])
            }
        );
        assert_eq!(ids(&cited), [1, 2]);
        assert!(!derivation.rules().iter().any(|r| r.contains(":c:")));
        results.push(outcome_signature(&MzTransportDecision::Identified { derivation, cited }));
    }
    assert!(results.windows(2).all(|w| w[0] == w[1]), "source order changed the result");

    // `c` alone certifies the same factor: the formula is the same up to the
    // population label, so either certifying source is a valid answer.
    let only_c =
        catalog(vec![regime(0, "target", &[]), regime(2, "b", &[Z1]), regime(3, "c", &[Z2])]);
    let with_a = identified(decide(&query(vec![source_a(), source_b()]), &complementary_catalog()));
    let with_c = identified(decide(&query(vec![source_b(), twin_of_a()]), &only_c));
    // Only the first certifying source is kept: when its experiment was never
    // supplied the decision is missing evidence for `a`, although `c` could have
    // supplied the same factor (the paper's weighted sum would have used it).
    let c_supplied_only =
        catalog(vec![regime(0, "target", &[]), regime(2, "b", &[Z1]), regime(3, "c", &[Z2])]);
    let decision = decide(&query(vec![source_a(), source_b(), twin_of_a()]), &c_supplied_only);
    let MzTransportDecision::MissingEvidence { derivation: Some(kept), detail } = decision else {
        panic!("expected missing evidence for the first certifying source, got {decision:?}");
    };
    assert!(detail.contains(": a ") && !detail.contains(": c "), "{detail}");
    assert!(!kept.rules().iter().any(|r| r.contains(":c:")));
    let relabel = |rules: &[String]| {
        rules.iter().map(|r| r.replace("exchange:c:", "exchange:a:")).collect::<Vec<_>>()
    };
    assert_eq!(with_a.rules(), relabel(with_c.rules()));
    assert_eq!(
        with_c.route(),
        &MzTransportRoute::Combined { populations: Arc::from([Arc::from("b"), Arc::from("c")]) }
    );
}
