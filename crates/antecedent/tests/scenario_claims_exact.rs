//! Scenario claims on a second, independently derived set of laws.
//!
//! Source experiment cells over (z, y): (0,0) 0.15, (0,1) 0.25, (1,0) 0.30, (1,1) 0.30.
//! Target cells over (z, x, y) have z-margin 0.9 / 0.1. Scenario `a_direct` (no selection)
//! reads the source law directly: P(y=1) = 0.25 + 0.30 = 0.55. Scenario `b_standardize`
//! (selection on z) standardizes: P(y=1|z=0) = 0.25 / 0.40 = 0.625, P(y=1|z=1) = 0.5, so
//! 0.9 * 0.625 + 0.1 * 0.5 = 0.6125. With `treat = y - 0.29` and `hold = 0.5 y`:
//! a: treat 0.26, hold 0.275 (hold leads); b: treat 0.3225, hold 0.30625 (treat leads).
//! Maximin: treat min 0.26 against hold min 0.275, so hold. Envelope intervals:
//! treat [0.26, 0.3225], hold [0.275, 0.30625]. With declared weights 0.3, 0.2 on the two
//! identified scenarios, the weighted values are treat 0.3*0.26 + 0.2*0.3225 = 0.1425 and
//! hold 0.3*0.275 + 0.2*0.30625 = 0.14375; the 0.5 of unaccounted mass moves each to
//! treat [0.1425 - 0.145, 0.1425 + 0.355], hold [0.14375, 0.14375 + 0.25].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::analysis::decision_claims::{
    OutcomeBinding, ScenarioClaimBinding, evaluate_scenario_decision, report_identified_utilities,
    scenario_claims,
};
use antecedent::{
    AdmissibilityRules, AdmissibleDecisionContract, AtomSupport, BoundDecisionContract,
    IdentifiedVerdict, PreparedTransportScenarios, Study, StudyBuilder,
};
use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_core::{
    DistributionAvailability, EvidenceCatalog, EvidenceKind, EvidenceRegime, ExecutionContext,
    RegimeId, RegimeKind, SearchLimits, Value, VariableDomain, VariableId,
};
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, StructuralPolicy, UtilityExpr,
};
use antecedent_design::decision_structural::StructuralVerdict;
use antecedent_estimate::transport_scenarios::ScenarioSetReport;
use antecedent_expr::{
    Assignment, DiscreteAxis, ExactDiscreteLaw, ExactEvaluationLimits, ExactTransportData,
    InterventionAssignment, LawTolerance,
};
use antecedent_graph::{Admg, DenseNodeId, SelectionDiagram};
use antecedent_identify::{
    ClassicalTransportQuery,
    sid::scenarios::{
        ScenarioCoordinate, TransportScenario, TransportScenarioSet, decide_transport_scenarios,
    },
};

#[path = "fixtures_common/mod.rs"]
mod common;

use antecedent_test_support::within;

const NEAR_TOLERANCE: f64 = 1e-9;

fn near(a: f64, b: f64) -> bool {
    within(a, b, NEAR_TOLERANCE)
}

use common::fixtures::confounded_scm;

const Z: u32 = 0;
const X: u32 = 1;
const Y: u32 = 2;

/// `P(y = 1)` under `a_direct`: the source cells with y = 1, 0.25 + 0.30.
const DIRECT: f64 = 0.55;
/// `P(y = 1)` under `b_standardize`: P(y=1|z=0) = 0.25/0.40 = 0.625 and P(y=1|z=1) = 0.30/0.60 = 0.5
/// weighted by the target margin of z, 0.9 and 0.1.
const STANDARDIZED: f64 = 0.9 * 0.625 + 0.1 * 0.5;
const COST: f64 = 0.29;
const BUDGET: SearchLimits = SearchLimits { operations: 100_000, depth: 256 };

fn v(i: u32) -> VariableId {
    VariableId::from_raw(i)
}

fn scenario(name: &str, selections: &[u32], weight: Option<f64>) -> TransportScenario {
    let coordinates: Arc<[ScenarioCoordinate]> = [(Z, "z"), (X, "x"), (Y, "y")]
        .into_iter()
        .map(|(id, label)| ScenarioCoordinate {
            variable: v(id),
            name: Arc::from(label),
            domain: VariableDomain::Binary,
            unit: None,
        })
        .collect();
    let mut graph = Admg::with_variables(3);
    for (a, b) in [(Z, X), (Z, Y), (X, Y)] {
        graph.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    graph.insert_bidirected(DenseNodeId::from_raw(X), DenseNodeId::from_raw(Y)).unwrap();
    let selections = selections.iter().copied().map(v).collect::<Vec<_>>();
    TransportScenario {
        name: Arc::from(name),
        diagram: SelectionDiagram::try_new(graph, Arc::<[VariableId]>::from(selections)).unwrap(),
        weight,
        coordinates,
    }
}

fn two() -> Vec<TransportScenario> {
    vec![scenario("a_direct", &[], None), scenario("b_standardize", &[Z], None)]
}

fn three() -> Vec<TransportScenario> {
    let mut scenarios = two();
    scenarios.push(scenario("c_shift", &[Y], None));
    scenarios
}

fn four(weights: Option<[f64; 4]>) -> Vec<TransportScenario> {
    let w = |k: usize| weights.map(|w| w[k]);
    vec![
        scenario("a_direct", &[], w(0)),
        scenario("b_standardize", &[Z], w(1)),
        scenario("c_shift", &[Y], w(2)),
        scenario("d_late", &[Z, Y], w(3)),
    ]
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

fn catalog() -> EvidenceCatalog {
    EvidenceCatalog::try_new(
        [],
        vec![regime(1, "source", &[X], &[Z, Y]), regime(0, "target", &[], &[Z, X, Y])],
        [],
        None,
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
        [0.15, 0.25, 0.30, 0.30],
        "trial",
        LawTolerance::default(),
    )
    .unwrap();
    let target = ExactDiscreteLaw::try_new(
        "target",
        RegimeId::from_raw(0),
        [],
        [axis(Z), axis(X), axis(Y)],
        [0.30, 0.20, 0.20, 0.20, 0.02, 0.02, 0.03, 0.03],
        "target",
        LawTolerance::default(),
    )
    .unwrap();
    ExactTransportData::try_new(vec![source, target], 1000).unwrap()
}

fn report_under(scenarios: Vec<TransportScenario>, budget: SearchLimits) -> ScenarioSetReport {
    let set = TransportScenarioSet::try_new(scenarios).unwrap();
    let ctx = ExecutionContext::for_tests(1);
    let prepared: PreparedTransportScenarios = StudyBuilder::transport_scenarios(
        &set,
        query(),
        antecedent::EvidenceSearch { catalog: catalog(), budget },
        laws(),
        Assignment::from_pairs([(v(X), Value::f64(1.0))]),
        ExactEvaluationLimits::default(),
        &ctx,
    )
    .unwrap();
    prepared.estimate(&ctx).unwrap()
}

/// The fewest shared operations that decide `a_direct`, `b_standardize` and
/// `c_shift`; `d_late` is then never entered.
fn operations_for_three() -> usize {
    (1..10_000)
        .find(|operations| {
            decide_transport_scenarios(
                &TransportScenarioSet::try_new(three()).unwrap(),
                &query(),
                &catalog(),
                SearchLimits { operations: *operations, depth: 256 },
                &ExecutionContext::for_tests(1),
            )
            .unwrap()
            .receipt
            .is_none()
        })
        .expect("a small graph decides within 10k operations")
}

fn status_of<'a>(report: &'a ScenarioSetReport, name: &str) -> &'a str {
    report.scenarios.iter().find(|s| &*s.name == name).unwrap().status
}

/// Two identified scenarios that disagree, one unidentified, one unevaluated.
fn mixed_report(weights: Option<[f64; 4]>) -> ScenarioSetReport {
    let budget = SearchLimits { operations: operations_for_three(), depth: 256 };
    let report = report_under(four(weights), budget);
    assert_eq!(status_of(&report, "a_direct"), "identified");
    assert_eq!(status_of(&report, "b_standardize"), "identified");
    assert_eq!(status_of(&report, "c_shift"), "structurally_unidentified");
    assert_eq!(status_of(&report, "d_late"), "unevaluated");
    report
}

fn two_report() -> ScenarioSetReport {
    let report = report_under(two(), BUDGET);
    assert_eq!(status_of(&report, "a_direct"), "identified");
    assert_eq!(status_of(&report, "b_standardize"), "identified");
    report
}

fn y_quantity(variable: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: variable.into(),
        variable_name: variable.into(),
        role: QuantityRole::Outcome,
        units: "units".into(),
        population_id: "target".into(),
        regime_id: "do(x=1)".into(),
        horizon: 0,
        functional_id: "outcome".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

fn boxed(expr: UtilityExpr) -> Box<UtilityExpr> {
    Box::new(expr)
}

/// `treat` earns `y - cost`; `hold` earns `0.5 * y`.
fn decision(cost: f64, policy: StructuralPolicy) -> AdmissibleDecisionContract {
    let treat = DecisionAction {
        id: "treat".into(),
        kind: ActionKind::Intervention,
        inputs: vec![y_quantity("y")],
        utility: UtilityExpr::Sub(boxed(UtilityExpr::Input(0)), boxed(UtilityExpr::Const(cost))),
    };
    let hold = DecisionAction {
        id: "hold".into(),
        kind: ActionKind::Regime,
        inputs: vec![y_quantity("y")],
        utility: UtilityExpr::Mul(boxed(UtilityExpr::Const(0.5)), boxed(UtilityExpr::Input(0))),
    };
    AdmissibleDecisionContract {
        contract: DecisionContract {
            actions: vec![treat, hold],
            utility_units: "units".into(),
            criterion: DecisionCriterion::PosteriorExpectedUtility,
            constraints: vec![],
            target_population: "target".into(),
            horizon: 0,
            structural_policy: policy,
        },
        rules: AdmissibilityRules::default(),
    }
}

fn bound(cost: f64, policy: StructuralPolicy) -> BoundDecisionContract {
    let (data, dag, query) = confounded_scm(64, 5);
    let causal = Study::tabular(data).query(query).graph(dag).inspect().unwrap();
    BoundDecisionContract::bind(&causal, decision(cost, policy)).unwrap()
}

fn binding_for(bound: &BoundDecisionContract, variable: &str) -> ScenarioClaimBinding {
    ScenarioClaimBinding::new(
        vec![OutcomeBinding { outcome: v(Y), quantity: y_quantity(variable) }],
        "scenario-study",
        "transport.exact_supplied_laws",
        bound.causal_identification(),
        "premises-1",
        "data-1",
    )
    .with_default_support(AtomSupport::supported())
}

fn binding(bound: &BoundDecisionContract) -> ScenarioClaimBinding {
    binding_for(bound, "y")
}

#[test]
fn per_scenario_leaders_and_policies_follow_the_hand_derived_utilities() {
    let report = two_report();
    let leaders = vec![
        ("a_direct".to_owned(), vec!["hold".to_owned()]),
        ("b_standardize".to_owned(), vec!["treat".to_owned()]),
    ];
    let invariant = bound(COST, StructuralPolicy::RequireInvariantBestAction);
    let decided = evaluate_scenario_decision(
        invariant.decision(),
        invariant.causal_identification(),
        &report,
        &binding(&invariant),
        StructuralPolicy::RequireInvariantBestAction,
        None,
    )
    .unwrap();
    assert_eq!(decided.leaders, leaders);
    assert_eq!(decided.decision.structural.verdict, StructuralVerdict::NoInvariantBest(leaders));
    let range = |id: &str| {
        decided.decision.structural.actions.iter().find(|a| a.id == id).unwrap().range.unwrap()
    };
    let (lo, hi) = range("treat");
    assert!(near(lo, DIRECT - COST) && near(hi, STANDARDIZED - COST), "{lo} {hi}");
    let (lo, hi) = range("hold");
    assert!(near(lo, 0.275) && near(hi, 0.30625), "{lo} {hi}");

    let maximin = bound(COST, StructuralPolicy::Maximin);
    let decided = evaluate_scenario_decision(
        maximin.decision(),
        maximin.causal_identification(),
        &report,
        &binding(&maximin),
        StructuralPolicy::Maximin,
        None,
    )
    .unwrap();
    assert_eq!(decided.decision.structural.verdict, StructuralVerdict::WorstCaseChoice("hold".into()));
}

#[test]
fn unrenormalized_weighted_values_and_intervals_match_the_hand_arithmetic() {
    let report = mixed_report(Some([0.3, 0.2, 0.25, 0.15]));
    let bound = bound(COST, StructuralPolicy::ReportOnly);
    let claims =
        scenario_claims(&report, &binding(&bound), StructuralPolicy::ReportOnly, None).unwrap();
    let decided = bound.evaluate_claims(&claims.adapted).unwrap();
    let structural = &decided.structural;
    assert!(near(structural.evaluated_mass.unwrap(), 0.5));
    let weighted =
        |id: &str| structural.actions.iter().find(|a| a.id == id).unwrap().weighted_value.unwrap();
    assert!(near(weighted("treat"), 0.1425), "{}", weighted("treat"));
    assert!(near(weighted("hold"), 0.14375), "{}", weighted("hold"));

    let utilities = report_identified_utilities(
        bound.decision(),
        &report,
        &binding(&bound),
        &AtomSupport::supported(),
    )
    .unwrap();
    assert!(near(utilities[0].utility.lower, -0.0025) && near(utilities[0].utility.upper, 0.4975));
    assert!(near(utilities[1].utility.lower, 0.14375) && near(utilities[1].utility.upper, 0.39375));
}

#[test]
fn unweighted_envelope_intervals_are_the_hand_derived_ranges() {
    let report = two_report();
    let bound = bound(COST, StructuralPolicy::RequireInvariantBestAction);
    let utilities = report_identified_utilities(
        bound.decision(),
        &report,
        &binding(&bound),
        &AtomSupport::supported(),
    )
    .unwrap();
    assert!(near(utilities[0].utility.lower, 0.26) && near(utilities[0].utility.upper, 0.3225));
    assert!(near(utilities[1].utility.lower, 0.275) && near(utilities[1].utility.upper, 0.30625));
    let decided = bound.evaluate_identified_sets(&utilities).unwrap();
    // The lower endpoints favor hold, the upper endpoints favor treat.
    assert_eq!(decided.lower_leader.as_deref(), Some("hold"));
    assert_eq!(decided.upper_leader.as_deref(), Some("treat"));
    assert!(matches!(decided.verdict, IdentifiedVerdict::NoNecessarilyBest { .. }));
}
