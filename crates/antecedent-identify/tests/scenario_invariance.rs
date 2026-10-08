//! Per-scenario selection differences and invariances (2.3.0 A1 remainder).
//!
//! Every scenario shares `z -> x`, `z -> y`, `x -> y` and `x <-> y` (so `z`
//! confounds `x` and `y`, and a latent confounder joins `x` and `y`), with the
//! source holding the experiment `do(x)` over `(z, y)` (regime 1) and the target
//! its observational joint over `(z, x, y)` (regime 0). The factors each answer
//! uses are derived by hand:
//!
//! * selection on `z`: no source law answers `y` directly (`S -> z -> y`), so
//!   the answer standardizes over `z`, `sum_z P_s(y | do(x), z) P*(z)`: one
//!   source factor (`y` given `z` under `do(x)`, regime 1) and one target factor
//!   (`z`, regime 0);
//! * no selection: the experiment answers directly, `P_s(y | do(x))`: one source
//!   factor and no target factor;
//! * selection on `x`: the treatment is intervened, so its mechanism is
//!   irrelevant and the answer is the same single source factor;
//! * selection on `y`: the outcome's own mechanism differs, an s-hedge, so there
//!   is an obstruction and no invariance list.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    DistributionAvailability, EvidenceCatalog, EvidenceKind, EvidenceRegime, ExecutionContext,
    RegimeId, RegimeKind, SearchLimits, VariableDomain, VariableId,
};
use antecedent_graph::{Admg, DenseNodeId, SelectionDiagram};
use antecedent_identify::ClassicalTransportQuery;
use antecedent_identify::sid::scenario_invariance::{
    InvarianceBody, InvarianceReport, SourceInvariance, TargetFactor, invariance_report,
};
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

fn y_given_do_x(conditioned_on: &[u32]) -> SourceInvariance {
    SourceInvariance {
        population: Arc::from("source"),
        variables: vec![v(Y)],
        conditioned_on: conditioned_on.iter().copied().map(v).collect(),
        do_set: vec![v(X)],
        regime: Some(1),
        invariant_mechanisms: vec![v(Y)],
        district_selection_targets: vec![],
        // With nothing conditioned on, the source experiment transports directly; the
        // standardization rule is the one that conditions on the pre-treatment variable.
        rule: if conditioned_on.is_empty() {
            "transport.direct"
        } else {
            "transport.pretreatment_standardize"
        },
    }
}

#[test]
fn a1_invariance_standardize_relies_on_the_source_conditional_and_the_target_marginal() {
    let decision = decide(vec![scenario("standardize", &[Z])], true);
    let report = report(&decision, "standardize");
    assert_eq!(report.status(), "identified");
    // Selection differences: z's mechanism may differ; x and y are shared.
    let selection = report.selection();
    assert_eq!(selection.selection_targets, [v(Z)]);
    assert_eq!(selection.shared_mechanisms, [v(X), v(Y)]);
    assert_eq!(selection.directed_edges, [(v(Z), v(X)), (v(Z), v(Y)), (v(X), v(Y))]);
    assert_eq!(selection.bidirected_edges, [(v(X), v(Y))]);
    // sum_z P_s(y | do(x), z) P*(z): y's mechanism is the invariance (z is only
    // conditioned on, because it differs), read from the experiment regime 1.
    assert_eq!(report.invariances().unwrap(), [y_given_do_x(&[Z])]);
    // P*(z) comes from the target, regime 0: a target law, not an invariance.
    assert_eq!(
        report.target_factors().unwrap(),
        [TargetFactor { variables: vec![v(Z)], conditioned_on: vec![], regime: Some(0) }]
    );
    let InvarianceBody::Identified { rules, conditional, .. } = report.body() else {
        panic!("identified body expected");
    };
    assert_eq!(rules, &["transport.pretreatment_standardize"]);
    assert!(conditional.is_none());
    assert!(report.obstruction().is_none());
}

#[test]
fn a1_invariance_no_selection_relies_on_the_direct_source_experiment_alone() {
    let decision = decide(vec![scenario("direct", &[])], true);
    let report = report(&decision, "direct");
    assert_eq!(report.status(), "identified");
    assert!(report.selection().selection_targets.is_empty());
    assert_eq!(report.selection().shared_mechanisms, [v(Z), v(X), v(Y)]);
    // P_s(y | do(x)): one source factor, nothing conditioned on, no target factor.
    assert_eq!(report.invariances().unwrap(), [y_given_do_x(&[])]);
    assert!(report.target_factors().unwrap().is_empty());
}

#[test]
fn a1_invariance_selection_on_the_treatment_changes_the_difference_not_the_invariance() {
    let decision = decide(
        vec![scenario("direct", &[]), scenario("on_treatment", &[X]), scenario("on_z", &[Z])],
        true,
    );
    let direct = report(&decision, "direct");
    let on_treatment = report(&decision, "on_treatment");
    assert_eq!(on_treatment.status(), "identified");
    // x is intervened, so its own mechanism is irrelevant: the same factor.
    assert_eq!(on_treatment.invariances(), direct.invariances());
    assert_eq!(on_treatment.target_factors(), direct.target_factors());
    // The recorded selection difference is reported all the same, and the
    // identity (which covers it) tells the two scenarios apart.
    assert_eq!(on_treatment.selection().selection_targets, [v(X)]);
    assert_eq!(on_treatment.selection().shared_mechanisms, [v(Z), v(Y)]);
    assert_ne!(on_treatment.identity(), direct.identity());
}

#[test]
fn a1_invariance_selection_on_the_outcome_gives_an_obstruction_and_no_invariance_list() {
    let decision = decide(vec![scenario("outcome_shift", &[Y])], true);
    let report = report(&decision, "outcome_shift");
    assert_eq!(report.status(), "structurally_unidentified");
    assert!(report.invariances().is_none());
    assert!(report.target_factors().is_none());
    assert_eq!(report.selection().selection_targets, [v(Y)]);
    let witness = report.obstruction().expect("an s-hedge obstruction");
    assert_eq!(witness.kind, "s_hedge");
    // The hedge is rooted at the outcome, whose own mechanism is selected.
    assert!(witness.larger_nodes.contains(&v(Y)));
    assert_eq!(witness.selection_targets_in_larger, [v(Y)]);
    assert!(matches!(report.body(), InvarianceBody::Obstructed { .. }));
}

#[test]
fn a1_invariance_a_scenario_without_its_evidence_reports_no_invariances() {
    // Without the target law the standardization cannot bind: a derivation
    // exists but its evidence is missing, which is neither an invariance list
    // nor an obstruction.
    let decision = decide(vec![scenario("standardize", &[Z]), scenario("direct", &[])], false);
    let missing = report(&decision, "standardize");
    assert_eq!(missing.status(), "missing_evidence");
    assert!(missing.invariances().is_none());
    assert!(missing.obstruction().is_none());
    let InvarianceBody::Undecided { status, obligations, candidate } = missing.body() else {
        panic!("undecided body expected");
    };
    assert_eq!(*status, "missing_evidence");
    assert!(!obligations.is_empty());
    assert!(candidate.is_none());
    // The direct scenario still identifies from the source experiment alone.
    assert_eq!(report(&decision, "direct").invariances().unwrap(), [y_given_do_x(&[])]);
}

#[test]
fn a1_invariance_scenarios_with_different_answers_report_different_invariance_sets() {
    let decision = decide(vec![scenario("standardize", &[Z]), scenario("direct", &[])], true);
    let standardize = report(&decision, "standardize");
    let direct = report(&decision, "direct");
    assert_ne!(standardize.invariances(), direct.invariances());
    assert_ne!(standardize.target_factors(), direct.target_factors());
    assert_ne!(standardize.identity(), direct.identity());
    // Only the conditioning of the source factor and the target marginal differ.
    let (a, b) = (&standardize.invariances().unwrap()[0], &direct.invariances().unwrap()[0]);
    assert_eq!((&a.variables, &a.do_set, a.regime), (&b.variables, &b.do_set, b.regime));
    assert_ne!(a.conditioned_on, b.conditioned_on);
}

#[test]
fn a1_invariance_the_identity_ignores_edge_selection_and_scenario_order() {
    let forward =
        decide(vec![scenario_in_order("a", &[(Z, X), (Z, Y), (X, Y)], (X, Y), &[Z, X])], true);
    let reversed =
        decide(vec![scenario_in_order("b", &[(X, Y), (Z, Y), (Z, X)], (Y, X), &[X, Z])], true);
    let (first, second) = (report(&forward, "a"), report(&reversed, "b"));
    assert_eq!(first.status(), "identified");
    assert_eq!(first, second);
    assert_eq!(first.identity(), second.identity());
    assert_eq!(first.canonical_text(), second.canonical_text());
    assert!(first.identity().starts_with("inv.v1."));
    // Selections on z and x: z is standardized over, x is intervened. The
    // targets are listed by variable id whatever order they were supplied in.
    assert_eq!(first.selection().selection_targets, [v(Z), v(X)]);
    assert_eq!(first.invariances().unwrap(), [y_given_do_x(&[Z])]);
    // Rebuilding from the same decision is stable, and a different graph moves it.
    assert_eq!(report(&forward, "a").identity(), first.identity());
    let other = decide(vec![scenario("c", &[Z])], true);
    assert_ne!(report(&other, "c").identity(), first.identity());
}
