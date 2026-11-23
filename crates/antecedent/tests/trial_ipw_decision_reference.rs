//! Independent analytic checks of the trial-to-target IPW Bayesian-bootstrap decisions.
//!
//! Forty trial units and forty target units, known selection and treatment probabilities.
//! With Dirichlet(1, ..., 1) row weights `pi` over the `m` trial rows, the effect draw is
//! `(m / n_target) * sum_i pi_i * score_i` with the inverse-odds score
//! `sign * (1 - s) / s * y / P(arm)` (Dahabreh et al. 2020, equation 2, for the nonparticipant
//! target). The Dirichlet moments give the exact law of that sum: its mean is
//! `sum_i score_i / n_target` and its variance `(m / n_target)^2 * var_pop(score) / (m + 1)`.
//! For the predictive worlds, one draw selects an arm row with probability proportional to its
//! Dirichlet mass times a weight that is constant within the arm, so each arm row is selected
//! with probability `1 / (rows in the arm)` and the two arms, which use disjoint Dirichlet
//! blocks, are independent. All of that is computed here without any code of the library.

use std::sync::Arc;

use antecedent::analysis::trial_ipw_predictive_decision::{
    TrialIpwPredictiveDecisionRequest, compile_trial_ipw_predictive_decision,
};
use antecedent::analysis::trial_ipw_transport_decision::{
    TrialIpwRows, TrialIpwTransportDecisionRequest, compile_trial_ipw_transport_decision,
};
use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_core::{
    ContinuousDomain, ExecutionContext, GridSpec, ResponseFunctional, ResponseQuery,
    TransportQuery, VariableId,
};
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, StructuralPolicy, UtilityExpr,
};
use antecedent_design::decision_eval::Verdict;
use antecedent_graph::{Admg, SelectionDiagram};

const TRIAL: usize = 40;
const TARGET: usize = 40;
const DRAWS: usize = 20_000;
const SELECTION: f64 = 0.25;
const ASSIGNMENT: f64 = 0.4;

fn diagram_and_query() -> (SelectionDiagram, TransportQuery) {
    let treatment = VariableId::from_raw(0);
    let outcome = VariableId::from_raw(2);
    let mut graph = Admg::with_variables(3);
    let from = graph.dense_of_static(treatment).unwrap();
    let to = graph.dense_of_static(outcome).unwrap();
    graph.insert_directed(from, to).unwrap();
    let diagram = SelectionDiagram::try_new(graph, [VariableId::from_raw(1)]).unwrap();
    let query = TransportQuery::new(
        ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome,
            treatment: ContinuousDomain::new(treatment, GridSpec::Values(Arc::from([0.0, 1.0]))),
        }),
        "trial",
        "target",
        [treatment],
    );
    (diagram, query)
}

fn quantity(regime: &str, functional: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: "2".into(),
        variable_name: "outcome".into(),
        role: QuantityRole::Outcome,
        units: "score".into(),
        population_id: "target".into(),
        regime_id: regime.into(),
        horizon: 0,
        functional_id: functional.into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    }
}

fn rows(outcomes: Vec<f64>, selection: f64, assignment: f64) -> TrialIpwRows {
    let n = TRIAL + TARGET;
    TrialIpwRows {
        unit_ids: (0..n).map(|i| format!("unit-{i}")).collect(),
        outcomes,
        treatment: (0..n).map(|i| i % 2 == 0).collect(),
        trial: (0..n).map(|i| i < TRIAL).collect(),
        selection_probability: vec![selection; n],
        treatment_probability: vec![assignment; n],
        draws: DRAWS,
        seed: 31,
    }
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn variance(values: &[f64]) -> f64 {
    let centre = mean(values);
    values.iter().map(|v| (v - centre).powi(2)).sum::<f64>() / (values.len() - 1) as f64
}

fn effect_decision(cost: f64) -> (DecisionContract, ScientificQuantity) {
    let coordinate = quantity("do(0=1)-do(0=0)", "average_effect");
    let decision = DecisionContract {
        actions: vec![
            DecisionAction {
                id: "treat".into(),
                kind: ActionKind::External,
                inputs: vec![coordinate.clone()],
                utility: UtilityExpr::difference(UtilityExpr::Input(0), UtilityExpr::Const(cost)),
            },
            DecisionAction {
                id: "hold".into(),
                kind: ActionKind::External,
                inputs: vec![coordinate.clone()],
                utility: UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Const(0.0)),
            },
        ],
        utility_units: "score".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: Vec::new(),
        target_population: "target".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::RequireInvariantBestAction,
    };
    (decision, coordinate)
}

#[test]
fn the_effect_draws_follow_the_exact_dirichlet_moments_of_the_inverse_odds_scores() {
    // Treated trial rows have outcomes 0..3 cycling, control trial rows 0 and 1 alternating;
    // target rows carry placeholders the producer ignores.
    let outcomes: Vec<f64> = (0..TRIAL + TARGET)
        .map(|i| {
            if i >= TRIAL {
                0.0
            } else if i % 2 == 0 {
                f64::from(((i / 2) % 4) as u32)
            } else {
                f64::from((((i - 1) / 2) % 2) as u32)
            }
        })
        .collect();
    // Inverse-odds score of each trial row, written from the estimand.
    let odds = (1.0 - SELECTION) / SELECTION;
    let scores: Vec<f64> = (0..TRIAL)
        .map(|i| {
            if i % 2 == 0 {
                odds * outcomes[i] / ASSIGNMENT
            } else {
                -odds * outcomes[i] / (1.0 - ASSIGNMENT)
            }
        })
        .collect();
    let m = TRIAL as f64;
    let n_target = TARGET as f64;
    let exact_mean = scores.iter().sum::<f64>() / n_target;
    let score_mean = scores.iter().sum::<f64>() / m;
    let var_pop = scores.iter().map(|s| (s - score_mean).powi(2)).sum::<f64>() / m;
    let exact_variance = (m / n_target).powi(2) * var_pop / (m + 1.0);

    let verdicts = [(1.0, "treat"), (exact_mean + 1.0, "hold")];
    for (cost, optimal) in verdicts {
        let (diagram, query) = diagram_and_query();
        let (decision, coordinate) = effect_decision(cost);
        let request = TrialIpwTransportDecisionRequest {
            diagram,
            query,
            rows: rows(outcomes.clone(), SELECTION, ASSIGNMENT),
            quantity: coordinate,
            decision,
        };
        let ctx = ExecutionContext::for_tests(31);
        let result = compile_trial_ipw_transport_decision(request).unwrap().execute(&ctx).unwrap();
        let draws = &result.effect_draws;
        assert_eq!(draws.len(), DRAWS);
        let tolerance = 5.0 * (exact_variance / DRAWS as f64).sqrt();
        assert!(
            (mean(draws) - exact_mean).abs() < tolerance,
            "mean {} vs exact {exact_mean}",
            mean(draws)
        );
        let ratio = variance(draws) / exact_variance;
        assert!((ratio - 1.0).abs() < 0.1, "variance ratio {ratio}");
        // Every draw is a convex-combination of the scores, scaled by m / n_target.
        let (lowest, highest) =
            scores.iter().fold((f64::MAX, f64::MIN), |(lo, hi), s| (lo.min(*s), hi.max(*s)));
        assert!(draws.iter().all(|d| *d >= m / n_target * lowest - 1e-9));
        assert!(draws.iter().all(|d| *d <= m / n_target * highest + 1e-9));

        // The decision scores those draws.
        let actions = &result.decision.decision.actions;
        let treat = actions.iter().find(|a| a.id == "treat").unwrap();
        assert!((treat.expected_utility - (mean(draws) - cost)).abs() < 1e-9);
        assert_eq!(result.decision.decision.verdict, Verdict::UniquelyOptimal(optimal.into()));
        assert_eq!(result.calibration, "not_measured");
    }
}

#[test]
fn the_predictive_worlds_are_the_arm_empirical_laws_and_independent_given_the_declared_premise() {
    // Treated trial rows (even) have outcomes 0 and 2, control trial rows (odd) 1 and 3, ten of
    // each value; every selection and assignment probability is constant within an arm.
    let outcomes: Vec<f64> = (0..TRIAL + TARGET)
        .map(|i| if i < TRIAL { f64::from((i % 4) as u32) } else { 0.0 })
        .collect();
    let (diagram, query) = diagram_and_query();
    let worlds = [quantity("do(0=0)", "outcome"), quantity("do(0=1)", "outcome")];
    let decision = DecisionContract {
        actions: vec![
            DecisionAction {
                id: "hold".into(),
                kind: ActionKind::Intervention,
                inputs: vec![worlds[0].clone()],
                utility: UtilityExpr::Input(0),
            },
            DecisionAction {
                id: "treat".into(),
                kind: ActionKind::Intervention,
                inputs: vec![worlds[1].clone()],
                utility: UtilityExpr::difference(
                    UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Input(0)),
                    UtilityExpr::Const(1.0),
                ),
            },
        ],
        utility_units: "score".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: Vec::new(),
        target_population: "target".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::RequireInvariantBestAction,
    };
    let request = TrialIpwPredictiveDecisionRequest {
        diagram,
        query,
        rows: rows(outcomes, 0.3, 0.6),
        worlds,
        decision,
        independent_potential_outcomes_given_mass: true,
    };
    // The cross-world premise is part of the declaration, never defaulted.
    let mut undeclared = request.clone();
    undeclared.independent_potential_outcomes_given_mass = false;
    assert!(compile_trial_ipw_predictive_decision(undeclared).is_err());

    let ctx = ExecutionContext::for_tests(31);
    let result = compile_trial_ipw_predictive_decision(request).unwrap().execute(&ctx).unwrap();
    let [control, treated] = &result.predictive_draws;
    assert_eq!(control.len(), DRAWS);
    // Each arm's draws sit on that arm's empirical support, with the exact frequencies 1/2.
    let frequency = |draws: &[f64], value: f64| {
        draws.iter().filter(|d| **d == value).count() as f64 / draws.len() as f64
    };
    assert!(control.iter().all(|d| *d == 1.0 || *d == 3.0));
    assert!(treated.iter().all(|d| *d == 0.0 || *d == 2.0));
    for (draws, value) in [(control, 1.0), (control, 3.0), (treated, 0.0), (treated, 2.0)] {
        assert!((frequency(draws, value) - 0.5).abs() < 0.02, "frequency of {value}");
    }
    // Distinct Dirichlet blocks and the declared conditional independence leave no covariance.
    let (mc, mt) = (mean(control), mean(treated));
    let covariance = control.iter().zip(treated).map(|(c, t)| (c - mc) * (t - mt)).sum::<f64>()
        / (DRAWS - 1) as f64;
    assert!(covariance.abs() < 0.04, "covariance {covariance}");

    // hold = E[Y0] = 2, treat = E[Y1^2] - 1 = 1: the decision scores the rows, not their mean.
    let actions = &result.decision.decision.actions;
    let utility = |id: &str| actions.iter().find(|a| a.id == id).unwrap().expected_utility;
    assert!((utility("hold") - mc).abs() < 1e-9);
    let treat = treated.iter().map(|y| y * y - 1.0).sum::<f64>() / DRAWS as f64;
    assert!((utility("treat") - treat).abs() < 1e-9);
    assert!((utility("hold") - 2.0).abs() < 0.07 && (utility("treat") - 1.0).abs() < 0.07);
    assert_eq!(result.decision.decision.verdict, Verdict::UniquelyOptimal("hold".into()));
    assert_eq!(result.calibration, "not_measured");
}

#[test]
fn both_trial_ipw_routes_report_and_consult_their_support_unit() {
    let check = |plan_view: antecedent::planner::PlanView,
                 refused: Vec<antecedent_core::support_gate::WouldRefuse>| {
        let view = plan_view.unit_support().expect("the route is bound to its capability record");
        assert_eq!(view.unit(), "transport.trial_ipw_decision");
        assert_eq!(view.supported(), view.status().as_str() == "supported");
        assert_eq!(view.supported(), view.missing_evidence().is_empty());
        if view.supported() {
            assert!(refused.is_empty(), "{refused:?}");
        }
        assert!(refused.iter().all(|entry| entry.code == "not_supported"), "{refused:?}");
    };
    let outcomes: Vec<f64> = (0..TRIAL + TARGET)
        .map(|i| if i < TRIAL { f64::from((i % 4) as u32) } else { 0.0 })
        .collect();

    let (diagram, query) = diagram_and_query();
    let (decision, coordinate) = effect_decision(1.0);
    let plan = compile_trial_ipw_transport_decision(TrialIpwTransportDecisionRequest {
        diagram,
        query,
        rows: rows(outcomes.clone(), SELECTION, ASSIGNMENT),
        quantity: coordinate,
        decision,
    })
    .unwrap();
    let ctx = ExecutionContext::for_tests(31);
    plan.execute(&ctx).unwrap();
    check(plan.plan_view(), ctx.would_refuse());

    let (diagram, query) = diagram_and_query();
    let worlds = [quantity("do(0=0)", "outcome"), quantity("do(0=1)", "outcome")];
    let decision = DecisionContract {
        actions: vec![
            DecisionAction {
                id: "hold".into(),
                kind: ActionKind::Intervention,
                inputs: vec![worlds[0].clone()],
                utility: UtilityExpr::Input(0),
            },
            DecisionAction {
                id: "treat".into(),
                kind: ActionKind::Intervention,
                inputs: vec![worlds[1].clone()],
                utility: UtilityExpr::Input(0),
            },
        ],
        utility_units: "score".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: Vec::new(),
        target_population: "target".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::RequireInvariantBestAction,
    };
    let plan = compile_trial_ipw_predictive_decision(TrialIpwPredictiveDecisionRequest {
        diagram,
        query,
        rows: rows(outcomes, 0.3, 0.6),
        worlds,
        decision,
        independent_potential_outcomes_given_mass: true,
    })
    .unwrap();
    let ctx = ExecutionContext::for_tests(31);
    plan.execute(&ctx).unwrap();
    check(plan.plan_view(), ctx.would_refuse());
}

#[test]
fn a_decision_off_the_effect_coordinate_and_an_undeclared_premise_refuse() {
    let outcomes: Vec<f64> = (0..TRIAL + TARGET)
        .map(|i| if i < TRIAL { f64::from((i % 4) as u32) } else { 0.0 })
        .collect();
    let (diagram, query) = diagram_and_query();
    let (decision, mut coordinate) = effect_decision(1.0);
    // The decision reads the exact effect coordinate of the target; another regime is refused.
    coordinate.regime_id = "do(0=1)".into();
    let request = TrialIpwTransportDecisionRequest {
        diagram,
        query,
        rows: rows(outcomes.clone(), SELECTION, ASSIGNMENT),
        quantity: coordinate,
        decision,
    };
    assert!(compile_trial_ipw_transport_decision(request).is_err());
    // A single trial arm has no inverse-odds contrast: the producer refuses to draw.
    let (diagram, query) = diagram_and_query();
    let (decision, coordinate) = effect_decision(1.0);
    let mut one_arm = rows(outcomes, SELECTION, ASSIGNMENT);
    for (treated, trial) in one_arm.treatment.iter_mut().zip(&one_arm.trial) {
        if *trial {
            *treated = true;
        }
    }
    let plan = compile_trial_ipw_transport_decision(TrialIpwTransportDecisionRequest {
        diagram,
        query,
        rows: one_arm,
        quantity: coordinate,
        decision,
    })
    .unwrap();
    assert!(plan.execute(&ExecutionContext::for_tests(31)).is_err());
}
