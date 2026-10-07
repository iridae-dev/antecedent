//! Facade-level adapters from the real 2.2 scenario claims into decision atoms
//! and per-action intervals (2.3 B). Every expected value is derived by hand
//! from the laws below.
//!
//! The scenarios share `z -> x`, `z -> y`, `x -> y`, `x <-> y` and differ in the
//! mechanisms that may change. `a_direct` gives `P(y = 1) = 0.56`;
//! `b_standardize` gives `0.75 * 0.2 + 0.25 * 0.8 = 0.35`; `c_shift` selects on
//! `y` and is not transportable; `d_late` is left unevaluated by the shared
//! budget. Two actions read the mean of `y`: `treat` has utility `y - cost` and
//! `hold` has utility `0.5 * y`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::analysis::decision_claims::{
    ClaimError, NOT_ENUMERATED_ATOM, OutcomeBinding, ScenarioClaimBinding, atom_leaders,
    cpdag_claims, evaluate_cpdag_decision, evaluate_scenario_decision, report_identified_utilities,
    scenario_claims,
};
use antecedent::{
    AdmissibilityRules, AdmissibleDecisionContract, AtomSupport, BoundDecisionContract,
    IdentifiedVerdict, PreparedTransportScenarios, Study, StudyBuilder,
};
use antecedent_core::{
    DistributionAvailability, EvidenceCatalog, EvidenceKind, EvidenceRegime, ExecutionContext,
    QuantityRole, RegimeId, RegimeKind, ScientificQuantity, SearchLimits, SupportStatus, Value,
    VariableDomain, VariableId,
};
use antecedent_design::decision_adapters::ClaimKind;
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, StructuralPolicy, UtilityExpr,
};
use antecedent_design::decision_structural::{AtomEvidence, StructuralVerdict};
use antecedent_estimate::cpdag_scenarios::{CompletionReceiptEntry, CpdagScenarioReport};
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

mod common;

use common::fixtures::confounded_scm;

const Z: u32 = 0;
const X: u32 = 1;
const Y: u32 = 2;

/// `P(y = 1)` under `a_direct`.
const DIRECT: f64 = 0.08 + 0.48;
/// `P(y = 1)` under `b_standardize`.
const STANDARDIZED: f64 = 0.75 * 0.2 + 0.25 * 0.8;
const BUDGET: SearchLimits = SearchLimits { operations: 100_000, depth: 256 };

fn v(i: u32) -> VariableId {
    VariableId::from_raw(i)
}

fn near(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-9
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
        [0.32, 0.08, 0.12, 0.48],
        "trial",
        LawTolerance::default(),
    )
    .unwrap();
    let target = ExactDiscreteLaw::try_new(
        "target",
        RegimeId::from_raw(0),
        [],
        [axis(Z), axis(X), axis(Y)],
        [0.3, 0.15, 0.2, 0.1, 0.05, 0.05, 0.05, 0.1],
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
        catalog(),
        budget,
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

fn cpdag_of(report: ScenarioSetReport, not_enumerated: usize) -> CpdagScenarioReport {
    let completions = report
        .scenarios
        .iter()
        .map(|s| CompletionReceiptEntry {
            id: Arc::clone(&s.name),
            edges: Arc::from(Vec::<(VariableId, VariableId)>::new()),
            status: s.status,
            detail: s.detail.clone(),
            evidence_identity: None,
        })
        .collect::<Vec<_>>();
    CpdagScenarioReport {
        cpdag_identity: Arc::from("cpdag-1"),
        identified: completions.len(),
        unidentified: 0,
        unevaluated: not_enumerated,
        completions,
        not_enumerated,
        report: Some(report),
        receipt: None,
    }
}

#[test]
fn b_claims_scenario_atoms_keep_status_weight_support_and_unrenormalized_mass() {
    let weights = [0.3, 0.2, 0.25, 0.15];
    let report = mixed_report(Some(weights));
    let bound = bound(0.2, StructuralPolicy::ReportOnly);
    let binding = binding(&bound);
    let claims = scenario_claims(&report, &binding, StructuralPolicy::ReportOnly, None).unwrap();

    let statuses: Vec<(&str, &str)> =
        claims.atoms.iter().map(|a| (a.id.as_str(), a.status)).collect();
    assert_eq!(
        statuses,
        [
            ("a_direct", "identified"),
            ("b_standardize", "identified"),
            ("c_shift", "structurally_unidentified"),
            ("d_late", "unevaluated"),
        ]
    );
    assert!(claims.declared_weights);
    assert_eq!(claims.adapted.kind, ClaimKind::FiniteScenarios);
    for (record, weight) in claims.atoms.iter().zip(weights) {
        assert!(near(record.weight.unwrap(), weight));
    }
    // Identified scenarios carry the supplied support; the rest have none to assess.
    let support: Vec<SupportStatus> = claims.atoms.iter().map(|a| a.support).collect();
    assert_eq!(
        support,
        [
            SupportStatus::Supported,
            SupportStatus::Supported,
            SupportStatus::MissingEvidence,
            SupportStatus::MissingEvidence,
        ]
    );
    let atoms = &claims.adapted.atoms;
    assert!(matches!(atoms[0].evidence, AtomEvidence::Evaluated(_)));
    assert!(matches!(atoms[2].evidence, AtomEvidence::Unidentified));
    assert!(matches!(atoms[3].evidence, AtomEvidence::Unevaluated(_)));
    assert!(near(atoms[0].probability.unwrap(), 0.3) && near(atoms[3].probability.unwrap(), 0.15));

    // The report's own masses are retained: 0.25 unidentified, 0.1 undeclared.
    let mass = |name: &str| claims.masses.iter().find(|m| m.status == name).unwrap().mass.unwrap();
    assert!(near(mass("structurally_unidentified"), 0.25) && near(mass("unevaluated"), 0.15));
    assert!(near(claims.residual_mass.unwrap(), 0.1));
    assert_eq!(claims.masses, report.masses);

    // Evaluated mass 0.5; unidentified 0.25; unevaluated 0.15 + 0.1 undeclared.
    // Nothing is renormalized over the two identified scenarios.
    let decided = bound.evaluate_claims(&claims.adapted).unwrap();
    let structural = &decided.structural;
    assert!(near(structural.evaluated_mass.unwrap(), 0.5));
    assert!(near(structural.unidentified_mass.unwrap(), 0.25));
    assert!(near(structural.unevaluated_mass.unwrap(), 0.25));
    assert_eq!(structural.verdict, StructuralVerdict::ReportOnly);
    // treat: a 0.56 - 0.2 = 0.36, b 0.35 - 0.2 = 0.15. hold: a 0.28, b 0.175.
    let range = |id: &str| structural.actions.iter().find(|a| a.id == id).unwrap().range.unwrap();
    let (lo, hi) = range("treat");
    assert!(near(lo, STANDARDIZED - 0.2) && near(hi, DIRECT - 0.2));
    let (lo, hi) = range("hold");
    assert!(near(lo, 0.175) && near(hi, 0.28));
    // Weighted values are sums over the evaluated mass only: 0.3 * 0.36 + 0.2 * 0.15.
    let weighted =
        |id: &str| structural.actions.iter().find(|a| a.id == id).unwrap().weighted_value.unwrap();
    assert!(near(weighted("treat"), 0.138) && near(weighted("hold"), 0.119));
}

#[test]
fn b_claims_unweighted_report_gives_unweighted_atoms() {
    let report = mixed_report(None);
    let bound = bound(0.2, StructuralPolicy::ReportOnly);
    let claims =
        scenario_claims(&report, &binding(&bound), StructuralPolicy::ReportOnly, None).unwrap();
    assert!(!claims.declared_weights);
    assert!(claims.adapted.atoms.iter().all(|a| a.probability.is_none()));
    assert!(claims.residual_mass.is_none());
    let decided = bound.evaluate_claims(&claims.adapted).unwrap();
    assert!(decided.structural.unidentified_mass.is_none());
}

#[test]
fn b_claims_policies_reach_the_hand_derived_choices() {
    let report = two_report();
    // Cost 0.05: treat beats hold in both scenarios (0.51 > 0.28, 0.30 > 0.175).
    let invariant = bound(0.05, StructuralPolicy::RequireInvariantBestAction);
    let decided = evaluate_scenario_decision(
        &invariant,
        &report,
        &binding(&invariant),
        StructuralPolicy::RequireInvariantBestAction,
        None,
    )
    .unwrap();
    assert_eq!(
        decided.decision.structural.verdict,
        StructuralVerdict::InvariantBest("treat".into())
    );
    assert_eq!(
        decided.leaders,
        [
            ("a_direct".to_owned(), vec!["treat".to_owned()]),
            ("b_standardize".to_owned(), vec!["treat".to_owned()])
        ]
    );

    // Cost 0.2: treat leads under a_direct (0.36 > 0.28), hold under b_standardize
    // (0.175 > 0.15), so nothing is invariant.
    let leaders = vec![
        ("a_direct".to_owned(), vec!["treat".to_owned()]),
        ("b_standardize".to_owned(), vec!["hold".to_owned()]),
    ];
    let invariant = bound(0.2, StructuralPolicy::RequireInvariantBestAction);
    let decided = evaluate_scenario_decision(
        &invariant,
        &report,
        &binding(&invariant),
        StructuralPolicy::RequireInvariantBestAction,
        None,
    )
    .unwrap();
    assert_eq!(
        decided.decision.structural.verdict,
        StructuralVerdict::NoInvariantBest(leaders.clone())
    );
    assert_eq!(decided.leaders, leaders);
    assert_eq!(atom_leaders(&decided.decision), leaders);

    // Worst cases: treat min(0.36, 0.15) = 0.15 < hold min(0.28, 0.175) = 0.175.
    let maximin = bound(0.2, StructuralPolicy::Maximin);
    let decided = evaluate_scenario_decision(
        &maximin,
        &report,
        &binding(&maximin),
        StructuralPolicy::Maximin,
        None,
    )
    .unwrap();
    assert_eq!(
        decided.decision.structural.verdict,
        StructuralVerdict::WorstCaseChoice("hold".into())
    );

    let report_only = bound(0.2, StructuralPolicy::ReportOnly);
    let decided = evaluate_scenario_decision(
        &report_only,
        &report,
        &binding(&report_only),
        StructuralPolicy::ReportOnly,
        None,
    )
    .unwrap();
    assert_eq!(decided.decision.structural.verdict, StructuralVerdict::ReportOnly);
}

#[test]
fn b_claims_unresolved_scenarios_leave_invariance_and_worst_case_unchecked() {
    let report = mixed_report(None);
    for policy in [StructuralPolicy::RequireInvariantBestAction, StructuralPolicy::Maximin] {
        let bound = bound(0.05, policy);
        let decided =
            evaluate_scenario_decision(&bound, &report, &binding(&bound), policy, None).unwrap();
        assert!(matches!(
            decided.decision.structural.verdict,
            StructuralVerdict::InsufficientScience(_)
        ));
        // The two identified scenarios still name their leaders.
        assert_eq!(decided.leaders.len(), 2);
    }
}

#[test]
fn b_claims_identified_sets_give_one_interval_per_action() {
    let report = two_report();
    let supported = AtomSupport::supported();
    // Mean of y ranges over [0.35, 0.56]. treat = y - 0.2 in [0.15, 0.36];
    // hold = 0.5 * y in [0.175, 0.28]. Neither interval lies wholly above the
    // other, the lower bound favors hold and the upper bound favors treat.
    let invariant = bound(0.2, StructuralPolicy::RequireInvariantBestAction);
    let utilities = report_identified_utilities(
        invariant.decision(),
        &report,
        &binding(&invariant),
        &supported,
    )
    .unwrap();
    assert_eq!(
        utilities.iter().map(|u| u.action_id.as_str()).collect::<Vec<_>>(),
        ["treat", "hold"]
    );
    assert!(near(utilities[0].utility.lower, 0.15) && near(utilities[0].utility.upper, 0.36));
    assert!(near(utilities[1].utility.lower, 0.175) && near(utilities[1].utility.upper, 0.28));
    let decided = invariant.evaluate_identified_sets(&utilities).unwrap();
    assert_eq!(
        decided.verdict,
        IdentifiedVerdict::NoNecessarilyBest {
            possibly_optimal: vec!["treat".into(), "hold".into()]
        }
    );
    assert_eq!(decided.lower_leader.as_deref(), Some("hold"));
    assert_eq!(decided.upper_leader.as_deref(), Some("treat"));
    assert!(decided.conflicting_leaders);

    let maximin = bound(0.2, StructuralPolicy::Maximin);
    let utilities =
        report_identified_utilities(maximin.decision(), &report, &binding(&maximin), &supported)
            .unwrap();
    let decided = maximin.evaluate_identified_sets(&utilities).unwrap();
    assert_eq!(decided.verdict, IdentifiedVerdict::WorstCaseChoice("hold".into()));

    // Cost 0.05: treat in [0.30, 0.51] lies wholly above hold in [0.175, 0.28].
    let cheap = bound(0.05, StructuralPolicy::RequireInvariantBestAction);
    let utilities =
        report_identified_utilities(cheap.decision(), &report, &binding(&cheap), &supported)
            .unwrap();
    let decided = cheap.evaluate_identified_sets(&utilities).unwrap();
    assert_eq!(decided.verdict, IdentifiedVerdict::NecessarilyBest("treat".into()));
}

#[test]
fn b_claims_weighted_identified_sets_keep_unaccounted_mass_at_the_domain_limits() {
    // Identified weight 0.3 + 0.2 = 0.5; its weighted sum is 0.3 * 0.56 + 0.2 * 0.35
    // = 0.238. The other 0.5 may sit anywhere in {0, 1}: the mean lies in
    // [0.238, 0.738], never the renormalized 0.476.
    let report = mixed_report(Some([0.3, 0.2, 0.25, 0.15]));
    let bound = bound(0.2, StructuralPolicy::ReportOnly);
    let utilities = report_identified_utilities(
        bound.decision(),
        &report,
        &binding(&bound),
        &AtomSupport::supported(),
    )
    .unwrap();
    assert!(near(utilities[0].utility.lower, 0.038) && near(utilities[0].utility.upper, 0.538));
    assert!(near(utilities[1].utility.lower, 0.119) && near(utilities[1].utility.upper, 0.369));
}

#[test]
fn b_claims_an_unweighted_envelope_refuses_while_scenarios_are_unresolved() {
    let report = mixed_report(None);
    let bound = bound(0.2, StructuralPolicy::ReportOnly);
    let error = report_identified_utilities(
        bound.decision(),
        &report,
        &binding(&bound),
        &AtomSupport::supported(),
    )
    .unwrap_err();
    assert_eq!(error, ClaimError::UnresolvedScenarios(2));
    let refusal = error.to_refusal();
    assert_eq!(refusal.detail, "decision_claims.unresolved_scenarios");
    assert_eq!(refusal.offending.as_deref(), Some("2"));
}

#[test]
fn b_claims_an_unbound_action_input_refuses_with_missing_scenario_utility() {
    let report = two_report();
    let bound = bound(0.2, StructuralPolicy::ReportOnly);
    // The binding answers a different quantity than the actions read.
    let error = report_identified_utilities(
        bound.decision(),
        &report,
        &binding_for(&bound, "w"),
        &AtomSupport::supported(),
    )
    .unwrap_err();
    let refusal = error.to_refusal();
    assert_eq!(refusal.detail, "decision_claims.missing_scenario_utility");
    assert_eq!(refusal.offending.as_deref(), Some("treat.inputs[0]"));
}

#[test]
fn b_claims_bayes_needs_declared_weights_and_never_reads_completion_counts() {
    let bayes = bound(0.2, StructuralPolicy::BayesOverStructures);
    let binding = binding(&bayes);

    // An unweighted scenario set declares no probabilities.
    let error =
        scenario_claims(&two_report(), &binding, StructuralPolicy::BayesOverStructures, None)
            .unwrap_err();
    assert_eq!(error, ClaimError::ProbabilitiesNotDeclared { completion_counts: false });
    let refusal = error.to_refusal();
    assert_eq!(refusal.detail, "decision_claims.probabilities_not_declared");
    assert_eq!(refusal.supplied.as_deref(), Some("none"));

    // Completions of a CPDAG carry counts, never probabilities.
    let completions = cpdag_of(two_report(), 2);
    let error = cpdag_claims(&completions, &binding, StructuralPolicy::BayesOverStructures, None)
        .unwrap_err();
    assert_eq!(error, ClaimError::ProbabilitiesNotDeclared { completion_counts: true });
    assert_eq!(
        error.to_refusal().supplied.as_deref(),
        Some("completion counts (never probabilities)")
    );
    let refusal = evaluate_cpdag_decision(
        &bayes,
        &completions,
        &binding,
        StructuralPolicy::BayesOverStructures,
        None,
    )
    .unwrap_err();
    assert_eq!(refusal.detail, "decision_claims.probabilities_not_declared");

    // Declared weights are read; unresolved mass then blocks a Bayes choice.
    let report = mixed_report(Some([0.3, 0.2, 0.25, 0.15]));
    let decided = evaluate_scenario_decision(
        &bayes,
        &report,
        &binding,
        StructuralPolicy::BayesOverStructures,
        None,
    )
    .unwrap();
    let structural = &decided.decision.structural;
    assert!(near(structural.unidentified_mass.unwrap(), 0.25));
    assert!(matches!(structural.verdict, StructuralVerdict::InsufficientScience(_)));
}

#[test]
fn b_claims_cpdag_completions_are_graph_dependent_unweighted_atoms() {
    let completions = cpdag_of(two_report(), 2);
    let maximin = bound(0.2, StructuralPolicy::Maximin);
    let binding = binding(&maximin);
    let claims = cpdag_claims(&completions, &binding, StructuralPolicy::Maximin, None).unwrap();
    assert_eq!(claims.adapted.kind, ClaimKind::GraphDependent);
    assert_eq!(claims.source.kind, "cpdag_completions");
    assert_eq!(claims.source.cpdag_identity.as_deref(), Some("cpdag-1"));
    assert!(!claims.declared_weights);
    assert!(claims.adapted.atoms.iter().all(|a| a.probability.is_none()));
    // The two completions a stop never enumerated are one unevaluated atom that
    // carries the count, and the count is retained, not used as a weight.
    assert_eq!(claims.adapted.atoms.len(), 3);
    assert_eq!(claims.atoms[2].id, NOT_ENUMERATED_ATOM);
    assert_eq!(claims.atoms[2].status, "unevaluated");
    assert!(matches!(claims.adapted.atoms[2].evidence, AtomEvidence::Unevaluated(_)));
    assert_eq!(claims.adapted.completion_counts, [(NOT_ENUMERATED_ATOM.to_owned(), 2)]);

    // Each graph names its own leader, and the unread completions block a worst case.
    let decided =
        evaluate_cpdag_decision(&maximin, &completions, &binding, StructuralPolicy::Maximin, None)
            .unwrap();
    assert_eq!(
        decided.leaders,
        [
            ("a_direct".to_owned(), vec!["treat".to_owned()]),
            ("b_standardize".to_owned(), vec!["hold".to_owned()])
        ]
    );
    assert!(matches!(
        decided.decision.structural.verdict,
        StructuralVerdict::InsufficientScience(_)
    ));

    // A receipt that disagrees with the scenario results is refused.
    let mut broken = completions.clone();
    broken.completions.pop();
    let error = cpdag_claims(&broken, &binding, StructuralPolicy::Maximin, None).unwrap_err();
    assert_eq!(error, ClaimError::IdentityMismatch("completions".into()));
}

#[test]
fn b_claims_a_changed_scenario_digest_is_refused() {
    let report = two_report();
    let bound = bound(0.2, StructuralPolicy::Maximin);
    let binding = binding(&bound);
    let policy = StructuralPolicy::Maximin;
    let source = scenario_claims(&report, &binding, policy, None).unwrap().source;
    assert_eq!(source.scenarios.len(), 2);
    assert_eq!(source.premises_digest, "premises-1");

    // The same report reads back to the same identity.
    let again = scenario_claims(&report, &binding, policy, Some(&source)).unwrap();
    assert_eq!(again.source, source);

    // A changed law under one scenario changes its digest.
    let mut changed = report.clone();
    changed.scenarios[1].distribution.as_mut().unwrap().probabilities = Arc::from(vec![0.5, 0.5]);
    let error = scenario_claims(&changed, &binding, policy, Some(&source)).unwrap_err();
    assert_eq!(error, ClaimError::IdentityMismatch("b_standardize".into()));
    let refusal =
        evaluate_scenario_decision(&bound, &changed, &binding, policy, Some(&source)).unwrap_err();
    assert_eq!(refusal.detail, "decision_claims.identity_mismatch");
    assert_eq!(refusal.offending.as_deref(), Some("b_standardize"));

    // A dropped scenario, changed premises and changed data are refused likewise.
    let mut dropped = report.clone();
    dropped.scenarios.pop();
    let error = scenario_claims(&dropped, &binding, policy, Some(&source)).unwrap_err();
    assert_eq!(error, ClaimError::IdentityMismatch("scenario_set".into()));
    let premises = ScenarioClaimBinding { premises_digest: "premises-2".into(), ..binding.clone() };
    let error = scenario_claims(&report, &premises, policy, Some(&source)).unwrap_err();
    assert_eq!(error, ClaimError::IdentityMismatch("premises_digest".into()));
    let data = ScenarioClaimBinding { data_digest: "data-2".into(), ..binding.clone() };
    let error = scenario_claims(&report, &data, policy, Some(&source)).unwrap_err();
    assert_eq!(error, ClaimError::IdentityMismatch("data_digest".into()));

    // A binding for another causal contract is not the bound decision's.
    let foreign = ScenarioClaimBinding { causal_contract_id: "other".into(), ..binding.clone() };
    let refusal = evaluate_scenario_decision(&bound, &report, &foreign, policy, None).unwrap_err();
    assert_eq!(refusal.detail, "decision_claims.identity_mismatch");
    assert_eq!(refusal.offending.as_deref(), Some("causal_contract_id"));
}

#[test]
fn b_claims_a_policy_other_than_the_contracts_is_refused() {
    let report = two_report();
    let bound = bound(0.2, StructuralPolicy::Maximin);
    let refusal = evaluate_scenario_decision(
        &bound,
        &report,
        &binding(&bound),
        StructuralPolicy::ReportOnly,
        None,
    )
    .unwrap_err();
    assert_eq!(refusal.detail, "decision_adapters.policy_mismatch");
}
