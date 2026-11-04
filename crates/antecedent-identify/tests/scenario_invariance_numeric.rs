//! The invariance report's factor lists, evaluated numerically on enumerated source and target
//! structural models, against the target's own interventional truth.
//!
//! The graph is `z -> x`, `z -> y`, `x -> y` with a latent `u` into `x` and `y`; the source and
//! target populations are the same binary SCM except for the mechanism(s) of the selected
//! node(s): `z` (prior 0.3 in the source, 0.7 in the target), `x` (its table) or `y` (its
//! table). For each selection placement the answer is rebuilt from nothing but the factors the
//! report lists (a source `P_s(y | do(x), conditioned)` and, when the report lists a target
//! factor over the conditioning variable, its target marginal), with every number enumerated
//! from the source or the target SCM; it must equal the target's own `P*(y = 1 | do(x))`. When
//! the outcome's mechanism is selected the report carries an obstruction and the source
//! experiment's answer is demonstrably wrong.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    DistributionAvailability, EvidenceCatalog, EvidenceKind, EvidenceRegime, ExecutionContext,
    RegimeId, RegimeKind, SearchLimits, VariableDomain, VariableId,
};
use antecedent_graph::{Admg, DenseNodeId, SelectionDiagram};
use antecedent_identify::ClassicalTransportQuery;
use antecedent_identify::sid::scenario_invariance::{InvarianceReport, invariance_report};
use antecedent_identify::sid::scenarios::{
    ScenarioCoordinate, ScenarioSetDecision, TransportScenario, TransportScenarioSet,
    decide_transport_scenarios,
};

const Z: u32 = 0;
const X: u32 = 1;
const Y: u32 = 2;

fn v(i: u32) -> VariableId {
    VariableId::from_raw(i)
}

fn schema() -> Arc<[ScenarioCoordinate]> {
    [(Z, "z"), (X, "x"), (Y, "y")]
        .into_iter()
        .map(|(id, name)| ScenarioCoordinate {
            variable: v(id),
            name: Arc::from(name),
            domain: VariableDomain::Binary,
            unit: None,
        })
        .collect()
}

/// The shared graph with edges inserted in the given order, and selections in
/// the given order.
fn scenario_in_order(
    name: &str,
    directed: &[(u32, u32)],
    bidirected: (u32, u32),
    selections: &[u32],
) -> TransportScenario {
    let mut graph = Admg::with_variables(3);
    for (a, b) in directed {
        graph.insert_directed(DenseNodeId::from_raw(*a), DenseNodeId::from_raw(*b)).unwrap();
    }
    graph
        .insert_bidirected(DenseNodeId::from_raw(bidirected.0), DenseNodeId::from_raw(bidirected.1))
        .unwrap();
    let selections = selections.iter().copied().map(v).collect::<Vec<_>>();
    TransportScenario {
        name: Arc::from(name),
        diagram: SelectionDiagram::try_new(graph, Arc::<[VariableId]>::from(selections)).unwrap(),
        weight: None,
        coordinates: schema(),
    }
}

fn scenario(name: &str, selections: &[u32]) -> TransportScenario {
    scenario_in_order(name, &[(Z, X), (Z, Y), (X, Y)], (X, Y), selections)
}

fn query() -> ClassicalTransportQuery {
    ClassicalTransportQuery {
        outcomes: Arc::from([v(Y)]),
        treatments: Arc::from([v(X)]),
        source: Arc::from("source"),
        target: Arc::from("target"),
    }
}

fn regime(id: u32, population: &str, interventions: &[u32], measured: &[u32]) -> EvidenceRegime {
    EvidenceRegime::try_new(
        RegimeId::from_raw(id),
        if interventions.is_empty() { RegimeKind::Observational } else { RegimeKind::Experimental },
        EvidenceKind::Available,
        interventions.iter().copied().map(v).collect::<Vec<_>>(),
        [],
        measured.iter().copied().map(v).collect::<Vec<_>>(),
        population,
        DistributionAvailability::Joint,
    )
    .unwrap()
}

fn catalog(with_target: bool) -> EvidenceCatalog {
    let mut regimes = vec![regime(1, "source", &[X], &[Z, Y])];
    if with_target {
        regimes.push(regime(0, "target", &[], &[Z, X, Y]));
    }
    EvidenceCatalog::try_new([], regimes, [], None).unwrap()
}

fn decide(scenarios: Vec<TransportScenario>, with_target: bool) -> ScenarioSetDecision {
    let set = TransportScenarioSet::try_new(scenarios).unwrap();
    decide_transport_scenarios(
        &set,
        &query(),
        &catalog(with_target),
        SearchLimits { operations: 100_000, depth: 256 },
        &ExecutionContext::for_tests(1),
    )
    .unwrap()
}

fn report(decision: &ScenarioSetDecision, name: &str) -> InvarianceReport {
    let d = decision.decisions.iter().find(|d| &*d.scenario.name == name).unwrap();
    invariance_report(&d.scenario, &d.outcome).unwrap()
}

/// Binary SCM: `z ~ Bern(pz)`, `u ~ Bern(pu)` independent; `x | z, u ~ Bern(px[z][u])`;
/// `y | x, z, u ~ Bern(py[x][z][u])`.
#[derive(Clone, Copy)]
struct Scm {
    pz: f64,
    pu: f64,
    px: [[f64; 2]; 2],
    py: [[[f64; 2]; 2]; 2],
}

fn base() -> Scm {
    Scm {
        pz: 0.3,
        pu: 0.4,
        px: [[0.2, 0.7], [0.5, 0.9]],
        py: [[[0.1, 0.4], [0.3, 0.6]], [[0.5, 0.8], [0.35, 0.95]]],
    }
}

/// The target population: the source with the selected nodes' mechanisms changed.
fn target_of(selected: &[u32]) -> Scm {
    let mut t = base();
    if selected.contains(&Z) {
        t.pz = 0.7;
    }
    if selected.contains(&X) {
        t.px = [[0.6, 0.1], [0.3, 0.45]];
    }
    if selected.contains(&Y) {
        t.py = [[[0.7, 0.2], [0.55, 0.15]], [[0.25, 0.5], [0.65, 0.05]]];
    }
    t
}

fn pr(p: f64, one: usize) -> f64 {
    if one == 1 { p } else { 1.0 - p }
}

/// `P(y = 1 | do(x), z)` in the SCM.
fn y_do_x_given_z(s: &Scm, x: usize, z: usize) -> f64 {
    (0..2).map(|u| pr(s.pu, u) * s.py[x][z][u]).sum()
}

/// `P(y = 1 | do(x))` in the SCM.
fn y_do_x(s: &Scm, x: usize) -> f64 {
    (0..2).map(|z| pr(s.pz, z) * y_do_x_given_z(s, x, z)).sum()
}

/// The answer rebuilt from the factors the report lists: the source factor conditioned on
/// the listed variables, weighted by the target marginal of the conditioning variable when the
/// report lists one, and by nothing otherwise.
fn answer_from_report(report: &InvarianceReport, source: &Scm, target: &Scm, x: usize) -> f64 {
    let invariances = report.invariances().expect("an identified report lists its invariances");
    assert_eq!(invariances.len(), 1);
    let target_factors = report.target_factors().unwrap();
    match invariances[0].conditioned_on.as_slice() {
        [] => {
            assert!(target_factors.is_empty());
            y_do_x(source, x)
        }
        [only] if *only == v(Z) => {
            assert_eq!(target_factors.len(), 1);
            assert_eq!(target_factors[0].variables, [v(Z)]);
            (0..2).map(|z| pr(target.pz, z) * y_do_x_given_z(source, x, z)).sum()
        }
        other => panic!("unexpected conditioning {other:?}"),
    }
}

#[test]
fn the_listed_factors_rebuild_the_target_interventional_truth_where_they_exist() {
    let placements: [(&str, &[u32]); 3] = [("direct", &[]), ("on_x", &[X]), ("on_z", &[Z])];
    let decision =
        decide(placements.iter().map(|(name, selected)| scenario(name, selected)).collect(), true);
    for (name, selected) in placements {
        let report = report(&decision, name);
        assert_eq!(report.status(), "identified", "{name}");
        let target = target_of(selected);
        for x in 0..2 {
            let rebuilt = answer_from_report(&report, &base(), &target, x);
            let truth = y_do_x(&target, x);
            assert!((rebuilt - truth).abs() < 1e-12, "{name} do(x={x}): {rebuilt} vs {truth}");
        }
    }
    // Selection on z matters: the source population's own answer is a different number.
    let target = target_of(&[Z]);
    assert!((y_do_x(&base(), 0) - y_do_x(&target, 0)).abs() > 0.05);
}

#[test]
fn selection_on_the_outcome_is_an_obstruction_because_the_source_answer_is_wrong() {
    let decision = decide(vec![scenario("on_y", &[Y])], true);
    let report = report(&decision, "on_y");
    assert_eq!(report.status(), "structurally_unidentified");
    assert!(report.obstruction().is_some());
    let target = target_of(&[Y]);
    for x in 0..2 {
        let source_answer = y_do_x(&base(), x);
        let truth = y_do_x(&target, x);
        assert!(
            (source_answer - truth).abs() > 0.02,
            "do(x={x}): the source experiment must not answer: {source_answer} vs {truth}"
        );
    }
}
