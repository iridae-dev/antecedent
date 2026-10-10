//! The edges, selections and invariances behind each scenario answer and each
//! envelope extreme (2.3.0 A1 remainder).
//!
//! Every scenario shares `z -> x`, `z -> y`, `x -> y`, `x <-> y`; the source
//! holds `do(x)` over `(z, y)` (regime 1) and the target its observational joint
//! (regime 0). By hand: selection on `z` standardizes,
//! `sum_z P_s(y | do(x), z) P*(z) = 0.75 * 0.2 + 0.25 * 0.8 = 0.35`, relying on
//! the source conditional of `y` given `z` and the target marginal of `z`; no
//! selection (and selection on the intervened `x`) answers directly,
//! `P_s(y = 1 | do(x)) = 0.08 + 0.48 = 0.56`, relying on the source experiment
//! alone; selection on `y` is an s-hedge and relies on nothing.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    DistributionAvailability, EvidenceCatalog, EvidenceKind, EvidenceRegime, ExecutionContext,
    RegimeId, RegimeKind, SearchLimits, Value, VariableDomain, VariableId,
};
use antecedent_estimate::scenario_invariance_report::{EnvelopeSide, ScenarioSetInvarianceReport};
use antecedent_estimate::transport_scenarios::{
    PreparedScenarioSet, ScenarioSetReport, prepare_transport_scenarios,
};
use antecedent_expr::{
    Assignment, DiscreteAxis, ExactDiscreteLaw, ExactEvaluationLimits, ExactTransportData,
    InterventionAssignment, LawTolerance,
};
use antecedent_graph::{Admg, DenseNodeId, SelectionDiagram};
use antecedent_identify::ClassicalTransportQuery;
use antecedent_identify::sid::scenario_invariance::{InvarianceBody, SourceInvariance};
use antecedent_identify::sid::scenarios::{
    ScenarioCoordinate, TransportScenario, TransportScenarioSet, decide_transport_scenarios,
};

const Z: u32 = 0;
const X: u32 = 1;
const Y: u32 = 2;

const SOURCE: [f64; 4] = [0.32, 0.08, 0.12, 0.48];
const TARGET: [f64; 8] = [0.3, 0.15, 0.2, 0.1, 0.05, 0.05, 0.05, 0.1];
const STANDARDIZED: f64 = 0.75 * 0.2 + 0.25 * 0.8;
const DIRECT: f64 = 0.08 + 0.48;

fn v(i: u32) -> VariableId {
    VariableId::from_raw(i)
}

fn scenario(name: &str, selections: &[u32]) -> TransportScenario {
    let mut graph = Admg::with_variables(3);
    for (a, b) in [(Z, X), (Z, Y), (X, Y)] {
        graph.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    graph.insert_bidirected(DenseNodeId::from_raw(X), DenseNodeId::from_raw(Y)).unwrap();
    let selections = selections.iter().copied().map(v).collect::<Vec<_>>();
    TransportScenario {
        name: Arc::from(name),
        diagram: SelectionDiagram::try_new(graph, Arc::<[VariableId]>::from(selections)).unwrap(),
        weight: None,
        coordinates: [(Z, "z"), (X, "x"), (Y, "y")]
            .into_iter()
            .map(|(id, name)| ScenarioCoordinate {
                variable: v(id),
                name: Arc::from(name),
                domain: VariableDomain::Binary,
                unit: None,
            })
            .collect(),
    }
}

fn four() -> Vec<TransportScenario> {
    vec![
        scenario("standardize", &[Z]),
        scenario("direct", &[]),
        scenario("outcome_shift", &[Y]),
        scenario("on_treatment", &[X]),
    ]
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

fn axis(variable: u32) -> DiscreteAxis {
    DiscreteAxis { variable: v(variable), values: Arc::from([Value::f64(0.0), Value::f64(1.0)]) }
}

fn laws() -> ExactTransportData {
    let source = ExactDiscreteLaw::try_new(
        "source",
        RegimeId::from_raw(1),
        [InterventionAssignment::concrete(v(X), Value::f64(1.0))],
        [axis(Z), axis(Y)],
        SOURCE,
        "trial",
        LawTolerance::default(),
    )
    .unwrap();
    let target = ExactDiscreteLaw::try_new(
        "target",
        RegimeId::from_raw(0),
        [],
        [axis(Z), axis(X), axis(Y)],
        TARGET,
        "target",
        LawTolerance::default(),
    )
    .unwrap();
    ExactTransportData::try_new(vec![source, target], 1000).unwrap()
}

fn prepared(scenarios: Vec<TransportScenario>) -> PreparedScenarioSet {
    let set = TransportScenarioSet::try_new(scenarios).unwrap();
    let ctx = ExecutionContext::for_tests(1);
    let query = ClassicalTransportQuery {
        outcomes: Arc::from([v(Y)]),
        treatments: Arc::from([v(X)]),
        source: Arc::from("source"),
        target: Arc::from("target"),
    };
    let catalog = EvidenceCatalog::try_new(
        [],
        vec![regime(1, "source", &[X], &[Z, Y]), regime(0, "target", &[], &[Z, X, Y])],
        [],
        None,
    )
    .unwrap();
    let decision = decide_transport_scenarios(
        &set,
        &query,
        &catalog,
        SearchLimits { operations: 100_000, depth: 256 },
        &ctx,
    )
    .unwrap();
    prepare_transport_scenarios(
        decision,
        laws(),
        Assignment::from_pairs([(v(X), Value::f64(1.0))]),
        ExactEvaluationLimits::default(),
        &ctx,
    )
    .unwrap()
}

fn run(scenarios: Vec<TransportScenario>) -> (PreparedScenarioSet, ScenarioSetReport) {
    let prepared = prepared(scenarios);
    let report = prepared.evaluate(&ExecutionContext::for_tests(1)).unwrap();
    (prepared, report)
}

fn y_given_do_x(conditioned_on: &[u32]) -> SourceInvariance {
    SourceInvariance {
        population: Arc::from("source"),
        variables: vec![v(Y)],
        conditioned_on: conditioned_on.iter().copied().map(v).collect(),
        do_set: vec![v(X)],
        regime: Some(1),
        invariant_mechanisms: vec![v(Y)],
        district_selection_targets: vec![],
        // Nothing conditioned on: the source experiment transports directly.
        rule: if conditioned_on.is_empty() {
            "transport.direct"
        } else {
            "transport.pretreatment_standardize"
        },
    }
}

fn invariances(report: &ScenarioSetInvarianceReport, name: &str) -> Vec<SourceInvariance> {
    report.scenario(name).unwrap().report.invariances().unwrap().to_vec()
}

#[test]
fn a1_invariance_each_envelope_extreme_lists_the_edges_selections_and_invariances_behind_it() {
    let (prepared, report) = run(four());
    let envelope = report.envelope.as_ref().unwrap();
    let mean = &envelope.means[0];
    assert!((mean.lower - STANDARDIZED).abs() < 1e-12 && (mean.upper - DIRECT).abs() < 1e-12);
    assert_eq!((&*mean.lower_scenario, &*mean.upper_scenario), ("standardize", "direct"));

    let invariance = prepared.invariance_report(&report).unwrap();
    let lower = invariance.extreme(v(Y), EnvelopeSide::Lower).unwrap();
    let upper = invariance.extreme(v(Y), EnvelopeSide::Upper).unwrap();
    assert_eq!(invariance.extremes.len(), 2);
    assert_eq!((&*lower.scenario, &*upper.scenario), ("standardize", "direct"));
    assert!((lower.value - STANDARDIZED).abs() < 1e-12 && (upper.value - DIRECT).abs() < 1e-12);

    // The lower extreme: selection on z; z -> x, z -> y, x -> y, x <-> y; it
    // relies on y's mechanism given z under do(x) (source regime 1) and on P*(z).
    let selection = lower.report.selection();
    assert_eq!(selection.selection_targets, [v(Z)]);
    assert_eq!(selection.shared_mechanisms, [v(X), v(Y)]);
    assert_eq!(selection.directed_edges, [(v(Z), v(X)), (v(Z), v(Y)), (v(X), v(Y))]);
    assert_eq!(selection.bidirected_edges, [(v(X), v(Y))]);
    assert_eq!(lower.report.invariances().unwrap(), [y_given_do_x(&[Z])]);
    let target = &lower.report.target_factors().unwrap()[0];
    assert_eq!((&target.variables, target.regime), (&vec![v(Z)], Some(0)));

    // The upper extreme: no selection; it relies on the source experiment alone.
    assert!(upper.report.selection().selection_targets.is_empty());
    assert_eq!(upper.report.selection().shared_mechanisms, [v(Z), v(X), v(Y)]);
    assert_eq!(upper.report.invariances().unwrap(), [y_given_do_x(&[])]);
    assert!(upper.report.target_factors().unwrap().is_empty());

    // Scenarios with disagreeing answers rest on different invariance sets.
    assert_ne!(lower.report.invariances(), upper.report.invariances());
    assert_ne!(lower.report.identity(), upper.report.identity());
}

#[test]
fn a1_invariance_every_scenario_is_reported_and_an_unidentified_one_has_no_invariances() {
    let (prepared, report) = run(four());
    let invariance = prepared.invariance_report(&report).unwrap();
    let names = invariance.scenarios.iter().map(|s| &*s.name).collect::<Vec<_>>();
    assert_eq!(names, ["direct", "on_treatment", "outcome_shift", "standardize"]);
    let shift = invariance.scenario("outcome_shift").unwrap();
    assert_eq!(shift.result_status, "structurally_unidentified");
    assert!(shift.report.invariances().is_none());
    assert!(matches!(shift.report.body(), InvarianceBody::Obstructed { .. }));
    let witness = shift.report.obstruction().unwrap();
    assert_eq!(witness.kind, "s_hedge");
    assert!(witness.larger_nodes.contains(&v(Y)));
    assert_eq!(witness.selection_targets_in_larger, [v(Y)]);
    // It is not an envelope extreme: the envelope ranges over identified ones.
    assert!(invariance.extremes.iter().all(|e| &*e.scenario != "outcome_shift"));
    // Selection on the intervened treatment leaves the direct invariance intact.
    assert_eq!(invariances(&invariance, "on_treatment"), invariances(&invariance, "direct"));
    assert_eq!(
        invariance.scenario("on_treatment").unwrap().report.selection().selection_targets,
        [v(X)]
    );
    assert_eq!(invariance.scenario("direct").unwrap().result_status, "identified");
}

#[test]
fn a1_invariance_the_report_is_independent_of_scenario_supply_order() {
    let (prepared, report) = run(four());
    let mut reversed = four();
    reversed.reverse();
    let (other_prepared, other_report) = run(reversed);
    let first = prepared.invariance_report(&report).unwrap();
    let second = other_prepared.invariance_report(&other_report).unwrap();
    assert_eq!(first, second);
    assert_eq!(first.canonical_text(), second.canonical_text());
    // Re-evaluating does not move it, and reporting leaves the results alone.
    let again = prepared.evaluate(&ExecutionContext::for_tests(1)).unwrap();
    assert_eq!(prepared.invariance_report(&again).unwrap(), first);
    assert_eq!(again.scenarios.len(), report.scenarios.len());
}

#[test]
fn a1_invariance_a_report_for_another_set_is_refused() {
    let (prepared, _) = run(four());
    let (_, small) = run(vec![scenario("standardize", &[Z]), scenario("direct", &[])]);
    assert!(prepared.invariance_report(&small).is_err());
}
