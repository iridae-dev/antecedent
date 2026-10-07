//! Exact execution of the DAG completions of a CPDAG against an enumerated SCM (2.3A X2).
//!
//! The chain `a - b - c` has three completions: `a -> b -> c`, `a <- b -> c` and
//! `a <- b <- c`. One target joint law is enumerated from the finite SCM
//! `a -> b -> c`, and the query is `P(c | do(b = 1))`. The expected value of each
//! completion is computed here from the law: the first two completions give
//! `P(c = 1 | b = 1)` (the SCM makes `c` independent of `a` given `b`), the third
//! gives `P(c = 1)` because `c` is not a descendant of `b`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    DistributionAvailability, EvidenceCatalog, EvidenceKind, EvidenceRegime, ExecutionContext,
    RegimeId, RegimeKind, SearchLimits, SearchStop, Value, VariableDomain, VariableId,
};
use antecedent_estimate::cpdag_scenarios::{
    CpdagScenarioReport, PreparedCpdagScenarios, prepare_cpdag_scenarios,
};
use antecedent_estimate::transport_scenarios::STRUCTURAL_ENVELOPE_INTERPRETATION;
use antecedent_expr::{
    Assignment, DiscreteAxis, ExactDiscreteLaw, ExactEvaluationLimits, ExactTransportData,
    LawTolerance,
};
use antecedent_graph::{Cpdag, DenseNodeId, NodeRef};
use antecedent_identify::ClassicalTransportQuery;
use antecedent_identify::sid::cpdag_completion::{
    CompletionEvidenceBinding, CpdagCompletionInput, CpdagEvidence, decide_cpdag_completions,
};
use antecedent_identify::sid::scenarios::ScenarioCoordinate;

const SEARCH: SearchLimits = SearchLimits { operations: 100_000, depth: 256 };

const FORWARD: [(u32, u32); 2] = [(0, 1), (1, 2)];
const FORK: [(u32, u32); 2] = [(1, 0), (1, 2)];
const BACKWARD: [(u32, u32); 2] = [(1, 0), (2, 1)];

fn v(i: u32) -> VariableId {
    VariableId::from_raw(i)
}

fn d(i: u32) -> DenseNodeId {
    DenseNodeId::from_raw(i)
}

/// `P(a = 1) = 0.4`, `P(b = 1 | a) = 0.2, 0.7`, `P(c = 1 | b) = 0.1, 0.8`.
fn joint(a: usize, b: usize, c: usize) -> f64 {
    let p = |one: bool, p_one: f64| if one { p_one } else { 1.0 - p_one };
    p(a == 1, 0.4) * p(b == 1, [0.2, 0.7][a]) * p(c == 1, [0.1, 0.8][b])
}

fn prob(keep: impl Fn(usize, usize, usize) -> bool) -> f64 {
    let mut total = 0.0;
    for a in 0..2 {
        for b in 0..2 {
            for c in 0..2 {
                if keep(a, b, c) {
                    total += joint(a, b, c);
                }
            }
        }
    }
    total
}

/// Truth of `P(c = 1 | do(b = 1))` under one completion.
fn truth(edges: &[(u32, u32)]) -> f64 {
    if edges == FORWARD {
        // Backdoor adjustment over a: sum_a P(a) P(c = 1 | b = 1, a).
        (0..2)
            .map(|a| {
                prob(|x, _, _| x == a) * prob(|x, y, z| x == a && y == 1 && z == 1)
                    / prob(|x, y, _| x == a && y == 1)
            })
            .sum()
    } else if edges == FORK {
        // b has no parent: P(c = 1 | b = 1).
        prob(|_, y, z| y == 1 && z == 1) / prob(|_, y, _| y == 1)
    } else if edges == BACKWARD {
        // c is a parent of b: do(b) does not move c.
        prob(|_, _, z| z == 1)
    } else {
        panic!("unexpected completion {edges:?}")
    }
}

fn axis(variable: u32) -> DiscreteAxis {
    DiscreteAxis { variable: v(variable), values: Arc::from([Value::f64(0.0), Value::f64(1.0)]) }
}

fn laws() -> ExactTransportData {
    let mut table = [0.0; 8];
    for a in 0..2 {
        for b in 0..2 {
            for c in 0..2 {
                table[a * 4 + b * 2 + c] = joint(a, b, c);
            }
        }
    }
    let law = ExactDiscreteLaw::try_new(
        "target",
        RegimeId::from_raw(0),
        [],
        [axis(0), axis(1), axis(2)],
        table,
        "target",
        LawTolerance::default(),
    )
    .unwrap();
    ExactTransportData::try_new(vec![law], 1000).unwrap()
}

fn catalog(measured: &[u32]) -> EvidenceCatalog {
    let regime = EvidenceRegime::try_new(
        RegimeId::from_raw(0),
        RegimeKind::Observational,
        EvidenceKind::Available,
        Vec::<VariableId>::new(),
        [],
        measured.iter().copied().map(v).collect::<Vec<_>>(),
        "target",
        DistributionAvailability::Joint,
    )
    .unwrap();
    EvidenceCatalog::try_new([], vec![regime], [], None).unwrap()
}

fn coordinates() -> Arc<[ScenarioCoordinate]> {
    (0..3)
        .map(|i| ScenarioCoordinate {
            variable: v(i),
            name: Arc::from(format!("v{i}")),
            domain: VariableDomain::Binary,
            unit: None,
        })
        .collect()
}

fn query() -> ClassicalTransportQuery {
    ClassicalTransportQuery {
        outcomes: Arc::from([v(2)]),
        treatments: Arc::from([v(1)]),
        source: Arc::from("source"),
        target: Arc::from("target"),
    }
}

fn chain() -> Cpdag {
    let mut graph = Cpdag::with_variables(3);
    graph.insert_undirected(d(0), d(1)).unwrap();
    graph.insert_undirected(d(1), d(2)).unwrap();
    graph
}

/// The same chain with the nodes added in a different order and the edges
/// inserted reversed.
fn permuted_chain() -> Cpdag {
    let mut graph = Cpdag::empty();
    for variable in [2, 0, 1] {
        graph.add_node(NodeRef::Static(v(variable))).unwrap();
    }
    graph.insert_undirected(d(0), d(2)).unwrap(); // v2 - v1
    graph.insert_undirected(d(2), d(1)).unwrap(); // v1 - v0
    graph
}

fn shared() -> CpdagEvidence {
    CpdagEvidence::Shared {
        evidence_identity: Arc::from("target-law"),
        catalog: catalog(&[0, 1, 2]),
    }
}

fn prepare(
    graph: Cpdag,
    evidence: &CpdagEvidence,
    budget: SearchLimits,
    ctx: &ExecutionContext,
) -> PreparedCpdagScenarios {
    let decision = decide_cpdag_completions(
        &CpdagCompletionInput::new(graph),
        &coordinates(),
        &query(),
        evidence,
        budget,
        ctx,
    )
    .unwrap();
    prepare_cpdag_scenarios(
        decision,
        laws(),
        Assignment::from_pairs([(v(1), Value::f64(1.0))]),
        ExactEvaluationLimits::default(),
        ctx,
    )
    .unwrap()
}

fn report(graph: Cpdag, evidence: &CpdagEvidence) -> CpdagScenarioReport {
    let ctx = ExecutionContext::for_tests(1);
    prepare(graph, evidence, SEARCH, &ctx).evaluate(&ctx).unwrap()
}

fn raw(edges: &[(VariableId, VariableId)]) -> Vec<(u32, u32)> {
    edges.iter().map(|(a, b)| (a.raw(), b.raw())).collect()
}

fn id_of(report: &CpdagScenarioReport, edges: &[(u32, u32)]) -> Arc<str> {
    let found = report.completions.iter().find(|c| raw(&c.edges) == edges).unwrap();
    Arc::clone(&found.id)
}

fn mean_of(report: &CpdagScenarioReport, id: &str) -> Option<f64> {
    let scenario = report.report.as_ref()?.scenarios.iter().find(|s| &*s.name == id)?;
    scenario.distribution.as_ref().map(|dist| dist.mean(v(2)).unwrap())
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-12
}

#[test]
fn x2_cpdag_enumerated_truth_effects_and_envelope() {
    let report = report(chain(), &shared());
    assert_eq!(report.completions.len(), 3);
    assert_eq!((report.identified, report.unidentified, report.unevaluated), (3, 0, 0));
    assert!(report.is_exportable() && report.not_enumerated == 0 && report.total() == 3);
    assert_eq!(report.status_count("identified"), 3);
    for entry in &report.completions {
        assert_eq!(entry.status, "identified");
        assert_eq!(entry.evidence_identity.as_deref(), Some("target-law"));
        assert_eq!(entry.id.len(), 64);
        // Each completion's own effect equals the SCM truth for its graph.
        let expected = truth(&raw(&entry.edges));
        let got = mean_of(&report, &entry.id).unwrap();
        assert!(close(got, expected), "{:?}: {got} vs {expected}", entry.edges);
    }
    // The constants behind the truth: 0.8 for the first two completions, 0.38 for the third.
    assert!(
        close(truth(&FORWARD), 0.8) && close(truth(&FORK), 0.8) && close(truth(&BACKWARD), 0.38)
    );
    // The completions disagree, and the envelope spans all of them.
    let inner = report.report.as_ref().unwrap();
    let envelope = inner.envelope.as_ref().unwrap();
    assert_eq!(envelope.interpretation, STRUCTURAL_ENVELOPE_INTERPRETATION);
    assert_eq!(envelope.scenarios.len(), 3);
    let mean = &envelope.means[0];
    assert_eq!(mean.outcome, v(2));
    assert!(close(mean.lower, 0.38) && close(mean.upper, 0.8));
    assert_eq!(mean.lower_scenario, id_of(&report, &BACKWARD));
    assert!([id_of(&report, &FORWARD), id_of(&report, &FORK)].contains(&mean.upper_scenario));
    // Unweighted: no graph probability, no weighted report, no mass.
    assert!(inner.weighted.is_none() && inner.residual_mass.is_none());
    assert!(inner.masses.iter().all(|m| m.mass.is_none()));
    assert_eq!(report.masses().iter().find(|m| m.status == "identified").unwrap().count, 3);
}

#[test]
fn x2_cpdag_missing_evidence_status() {
    // The backward completion is bound to a catalog that lacks the outcome.
    let ids = report(chain(), &shared());
    let (a, b, c) = (id_of(&ids, &FORWARD), id_of(&ids, &FORK), id_of(&ids, &BACKWARD));
    let bind = |id: &Arc<str>, name: &str, measured: &[u32]| CompletionEvidenceBinding {
        completion: Arc::clone(id),
        certified_for: Arc::clone(id),
        evidence_identity: Arc::from(name),
        catalog: catalog(measured),
    };
    let evidence = CpdagEvidence::PerCompletion(vec![
        bind(&a, "ev-forward", &[0, 1, 2]),
        bind(&b, "ev-fork", &[0, 1, 2]),
        bind(&c, "ev-backward", &[0, 1]),
    ]);
    let report = report(chain(), &evidence);
    let status = |id: &Arc<str>| report.completions.iter().find(|e| &e.id == id).unwrap().status;
    assert_eq!(
        (status(&a), status(&b), status(&c)),
        ("identified", "identified", "missing_evidence")
    );
    // The missing completion is counted, not redistributed or dropped.
    assert_eq!((report.identified, report.unidentified, report.unevaluated), (2, 1, 0));
    assert_eq!(report.status_count("missing_evidence"), 1);
    assert_eq!(report.completions.len(), 3);
    assert!(mean_of(&report, &c).is_none());
    let backward = report.completions.iter().find(|e| e.id == c).unwrap();
    assert_eq!(backward.evidence_identity.as_deref(), Some("ev-backward"));
    assert!(backward.detail.is_some());
    // The envelope ranges over the two identified completions only: 0.38 is absent.
    let envelope = report.report.as_ref().unwrap().envelope.as_ref().unwrap();
    assert_eq!(envelope.scenarios.len(), 2);
    assert!(close(envelope.means[0].lower, 0.8) && close(envelope.means[0].upper, 0.8));
    // A completion with no evidence bound to it is missing_evidence as well.
    let only_one = CpdagEvidence::PerCompletion(vec![bind(&a, "ev-forward", &[0, 1, 2])]);
    let report = self::report(chain(), &only_one);
    assert_eq!((report.identified, report.unidentified), (1, 2));
    assert_eq!(report.status_count("missing_evidence"), 2);
    assert_eq!(report.completions.iter().filter(|e| e.evidence_identity.is_none()).count(), 2);
    assert!(report.report.unwrap().envelope.is_some());
}

#[test]
fn x2_cpdag_permutation_envelope() {
    let canonical = report(chain(), &shared());
    let permuted = report(permuted_chain(), &shared());
    assert_eq!(canonical.cpdag_identity, permuted.cpdag_identity);
    assert_eq!(canonical.completions, permuted.completions);
    let (a, b) = (canonical.report.unwrap(), permuted.report.unwrap());
    assert_eq!(a.envelope, b.envelope);
    let names = |r: &antecedent_estimate::transport_scenarios::ScenarioSetReport| {
        r.scenarios.iter().map(|s| (s.name.to_string(), s.status)).collect::<Vec<_>>()
    };
    assert_eq!(names(&a), names(&b));
    assert_eq!(a.masses, b.masses);
}

#[test]
fn x2_cpdag_budget_stop_partial_report() {
    // Enumeration stops after one of three completions (three orientation
    // attempts); the other two are counted separately, never as identified.
    let ctx = ExecutionContext::for_tests(1);
    let limits = SearchLimits { operations: 3, depth: 256 };
    let prepared = prepare(chain(), &shared(), limits, &ctx);
    let partial = prepared.evaluate(&ctx).unwrap();
    assert!(!partial.is_exportable());
    assert_eq!(partial.completions.len(), 1);
    assert_eq!(partial.not_enumerated, 2);
    assert_eq!(partial.total(), 3);
    assert_eq!(partial.completions[0].status, "unevaluated");
    assert_eq!((partial.identified, partial.unidentified, partial.unevaluated), (0, 0, 3));
    assert_eq!(partial.receipt.as_ref().unwrap().stop, SearchStop::Operations);
    assert!(partial.report.as_ref().unwrap().envelope.is_none());

    // Enumeration finishes in six attempts; the sixth operation is spent, so the
    // first decision cannot be entered and every completion is unevaluated.
    let limits = SearchLimits { operations: 6, depth: 256 };
    let stopped = prepare(chain(), &shared(), limits, &ctx).evaluate(&ctx).unwrap();
    assert!(!stopped.is_exportable());
    assert_eq!((stopped.completions.len(), stopped.not_enumerated), (3, 0));
    assert!(stopped.completions.iter().all(|e| e.status == "unevaluated"));
    assert_eq!((stopped.identified, stopped.unidentified, stopped.unevaluated), (0, 0, 3));
    let receipt = stopped.receipt.as_ref().unwrap();
    assert_eq!(receipt.stop, SearchStop::Operations);
    assert_eq!(receipt.unevaluated.len(), 3);
    assert!(stopped.report.as_ref().unwrap().envelope.is_none());

    // Cancelled before any work: nothing enumerated, everything unevaluated, not exportable.
    let cancelled = ExecutionContext::for_tests(1);
    cancelled.cancellation.cancel();
    let none = prepare(chain(), &shared(), SEARCH, &cancelled).evaluate(&ctx).unwrap();
    assert!(!none.is_exportable());
    assert!(none.completions.is_empty() && none.report.is_none());
    assert_eq!((none.not_enumerated, none.unevaluated, none.identified), (3, 3, 0));
    assert_eq!(none.receipt.as_ref().unwrap().stop, SearchStop::Cancelled);

    // The unconstrained run is complete and exportable.
    let whole = report(chain(), &shared());
    assert!(whole.is_exportable() && whole.identified == 3);
}
