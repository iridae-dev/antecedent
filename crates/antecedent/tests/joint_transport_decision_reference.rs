//! Independent analytic checks of the joint source-target transport decisions.
//!
//! One randomized trial of 40 units (20 treated), no covariates, a known noise variance and
//! independent zero-mean Gaussian priors on the treatment effect `tau` and the source baseline
//! `b`: `y = b + tau * A + noise`. The posterior of `(tau, b)` is then a two-dimensional Gaussian
//! whose precision, mean and covariance are written out here from the textbook conjugate
//! formulas on hand-computed sufficient statistics, without any code of the library. The effect
//! decision must score the posterior of `tau`; the predictive decision must score outcome draws
//! `b + noise` and `b + tau + noise`, whose paired-difference variance is
//! `Var(tau) + 2 sigma^2 (1 - rho)` for a declared disturbance correlation `rho`.

use std::sync::Arc;

use antecedent::analysis::joint_transport_effect_decision::{
    JointTransportEffectDecisionRequest, compile_joint_transport_effect_decision,
};
use antecedent::analysis::joint_transport_predictive_decision::{
    JointTransportPredictiveDecisionRequest, compile_joint_transport_predictive_decision,
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
use antecedent_estimate::joint_bayesian_transport::{
    DataIdentity, GaussianPrior, JointPriors, JointTransportModel, JointTransportOptions,
    PriorProvenance, SourceData, SourceDependence, SourceSharing, TargetData, TransportGraphClass,
    VaryingBlock,
};
use antecedent_graph::{Admg, SelectionDiagram};

const ROWS: usize = 40;
const NOISE: f64 = 0.1;
const PRIOR_VARIANCE: f64 = 10.0;
const DRAWS: usize = 1024;

/// The exact posterior of `(tau, b)` for `ROWS` units, half of them treated, with outcomes
/// `base` for controls and `base + effect` for treated units, from the conjugate formulas.
struct Posterior {
    mean: [f64; 2],
    covariance: [f64; 4],
}

fn analytic_posterior(base: f64, effect: f64) -> Posterior {
    let treated = (ROWS / 2) as f64;
    let total = ROWS as f64;
    let sum_treated = treated * (base + effect);
    let sum_all = sum_treated + (total - treated) * base;
    let weight = 1.0 / NOISE;
    // Precision = prior precision + Z'Z / sigma^2 with columns [A, 1].
    let a = weight * treated + 1.0 / PRIOR_VARIANCE;
    let b = weight * treated;
    let d = weight * total + 1.0 / PRIOR_VARIANCE;
    let moment = [weight * sum_treated, weight * sum_all];
    let det = a * d - b * b;
    let covariance = [d / det, -b / det, -b / det, a / det];
    let mean = [
        covariance[0] * moment[0] + covariance[1] * moment[1],
        covariance[2] * moment[0] + covariance[3] * moment[1],
    ];
    Posterior { mean, covariance }
}

fn quantity(variable_id: &str, regime: &str, functional: &str, name: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: variable_id.into(),
        variable_name: name.into(),
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

fn data(base: f64, effect: f64) -> (SourceData, TargetData) {
    let source = SourceData {
        id: "trial-a".into(),
        identity: DataIdentity {
            snapshot_digest: "trial-snapshot".into(),
            datum_ids: (0..ROWS).map(|i| format!("trial-{i}")).collect(),
        },
        treatment: (0..ROWS).map(|i| i % 2 == 0).collect(),
        outcome: (0..ROWS).map(|i| base + if i % 2 == 0 { effect } else { 0.0 }).collect(),
        covariates: Vec::new(),
        noise_variance: NOISE,
    };
    let target = TargetData {
        identity: DataIdentity {
            snapshot_digest: "target-snapshot".into(),
            datum_ids: (0..ROWS).map(|i| format!("target-{i}")).collect(),
        },
        rows: ROWS,
        covariates: Vec::new(),
    };
    (source, target)
}

fn model(sharing: SourceSharing) -> JointTransportModel {
    JointTransportModel {
        graph: TransportGraphClass::FixedDag,
        features: Vec::new(),
        varying: VaryingBlock::Intercept,
        sharing,
        dependence: SourceDependence::IndependentSamples,
        priors: JointPriors {
            invariant: GaussianPrior::isotropic(1, 0.0, PRIOR_VARIANCE, PriorProvenance::Declared),
            varying: GaussianPrior::isotropic(1, 0.0, PRIOR_VARIANCE, PriorProvenance::Declared),
        },
        max_unsupported_mass: 0.0,
        conflict_z_threshold: 3.0,
    }
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn variance(values: &[f64]) -> f64 {
    let centre = mean(values);
    values.iter().map(|v| (v - centre).powi(2)).sum::<f64>() / (values.len() - 1) as f64
}

/// Five Monte Carlo standard errors of a sample mean of `DRAWS` independent draws.
fn mean_tolerance(variance: f64) -> f64 {
    5.0 * (variance / DRAWS as f64).sqrt()
}

fn effect_request(base: f64, effect: f64, cost: f64) -> JointTransportEffectDecisionRequest {
    let (diagram, query) = diagram_and_query();
    let coordinate = quantity("2", "do(0=1)-do(0=0)", "average_effect", "outcome effect");
    let (source, target) = data(base, effect);
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
    JointTransportEffectDecisionRequest {
        diagram,
        query,
        model: model(SourceSharing::IndependentVaryingBlocks),
        sources: vec![source],
        target,
        options: JointTransportOptions { draws: DRAWS, seed: 91 },
        quantity: coordinate,
        decision,
    }
}

fn expected_utility(actions: &[antecedent_design::decision_eval::ActionOutcome], id: &str) -> f64 {
    actions.iter().find(|a| a.id == id).unwrap().expected_utility
}

#[test]
fn the_effect_posterior_and_its_decision_equal_the_hand_solved_conjugate_posterior() {
    for (base, effect) in [(3.0, 2.0), (30.0, 2.0), (3.0, 0.5)] {
        let truth = analytic_posterior(base, effect);
        let ctx = ExecutionContext::for_tests(91);
        let plan =
            compile_joint_transport_effect_decision(effect_request(base, effect, 1.0)).unwrap();
        let result = plan.execute(&ctx).unwrap();
        let fit = &result.fit;

        // The exact moments are the closed-form conjugate moments.
        for (got, want) in fit.posterior_mean.iter().zip(truth.mean) {
            assert!((got - want).abs() < 1e-9, "posterior mean {got} vs {want}");
        }
        for (got, want) in fit.posterior_covariance.iter().zip(truth.covariance) {
            assert!((got - want).abs() < 1e-9, "posterior covariance {got} vs {want}");
        }
        // Without covariates the target average effect is the treatment coefficient.
        assert!((fit.target_effect_mean - truth.mean[0]).abs() < 1e-9);
        assert!((fit.target_effect_variance - truth.covariance[0]).abs() < 1e-9);

        // The draws that the decision reads follow that posterior.
        let column = fit.draws.column("effect.target").unwrap();
        let draws = fit.draws.coordinate(column);
        assert_eq!(draws.len(), DRAWS);
        assert!((mean(&draws) - truth.mean[0]).abs() < mean_tolerance(truth.covariance[0]));
        let relative = variance(&draws) / truth.covariance[0];
        assert!((relative - 1.0).abs() < 0.3, "draw variance ratio {relative}");

        // The utility scores the effect draws: the treatment utility is the mean draw minus the
        // cost, the hold utility is zero, and the verdict follows the sign of the planted effect
        // net of the cost.
        let actions = &result.decision.decision.actions;
        assert!((expected_utility(actions, "treat") - (mean(&draws) - 1.0)).abs() < 1e-9);
        assert!(expected_utility(actions, "hold").abs() < 1e-12);
        let optimal = if effect > 1.0 { "treat" } else { "hold" };
        assert_eq!(result.decision.decision.verdict, Verdict::UniquelyOptimal(optimal.into()));
        assert_eq!(result.calibration, "not_measured");
    }
}

#[test]
fn the_effect_decision_is_the_decision_of_the_planted_effect_not_of_the_baseline() {
    // A baseline shift of 27 units leaves the planted effect, hence the verdict, unchanged,
    // and moves the posterior of the effect only by the prior's shrinkage, written out here.
    let low = analytic_posterior(3.0, 2.0);
    let high = analytic_posterior(30.0, 2.0);
    assert!((low.mean[0] - 2.0).abs() < 0.05 && (high.mean[0] - 2.0).abs() < 0.05);
    let ctx = ExecutionContext::for_tests(91);
    let result = compile_joint_transport_effect_decision(effect_request(30.0, 2.0, 1.0))
        .unwrap()
        .execute(&ctx)
        .unwrap();
    assert!((result.fit.target_effect_mean - high.mean[0]).abs() < 1e-9);
    assert_eq!(result.decision.decision.verdict, Verdict::UniquelyOptimal("treat".into()));
}

fn predictive_request(
    base: f64,
    effect: f64,
    rho: f64,
) -> (JointTransportPredictiveDecisionRequest, [ScientificQuantity; 2]) {
    let (diagram, query) = diagram_and_query();
    let worlds = [
        quantity("2", "do(0=0)", "outcome", "target outcome"),
        quantity("2", "do(0=1)", "outcome", "target outcome"),
    ];
    let (source, target) = data(base, effect);
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
                utility: UtilityExpr::difference(UtilityExpr::Input(0), UtilityExpr::Const(1.0)),
            },
        ],
        utility_units: "score".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: Vec::new(),
        target_population: "target".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::RequireInvariantBestAction,
    };
    let request = JointTransportPredictiveDecisionRequest {
        diagram,
        query,
        model: model(SourceSharing::SharedVaryingBlock),
        sources: vec![source],
        target,
        options: JointTransportOptions { draws: DRAWS, seed: 92 },
        target_noise_variance: NOISE,
        target_noise_correlation: rho,
        worlds: worlds.clone(),
        decision,
    };
    (request, worlds)
}

fn chain_columns(
    result: &antecedent::analysis::joint_transport_predictive_decision::JointTransportPredictiveDecisionResult,
) -> (Vec<f64>, Vec<f64>) {
    let sources = &result.decision.chain.record().sources;
    let column = |i: usize| sources[i].values.iter().map(|bits| f64::from_bits(*bits)).collect();
    (column(0), column(1))
}

#[test]
fn the_predictive_worlds_follow_the_hand_solved_posterior_predictive_law() {
    let (base, effect) = (3.0, 2.0);
    let truth = analytic_posterior(base, effect);
    let (var_tau, cov, var_b) = (truth.covariance[0], truth.covariance[1], truth.covariance[3]);
    for rho in [1.0, 0.0] {
        let (request, _) = predictive_request(base, effect, rho);
        let ctx = ExecutionContext::for_tests(92);
        let result =
            compile_joint_transport_predictive_decision(request).unwrap().execute(&ctx).unwrap();
        assert!((result.fit.posterior_mean[0] - truth.mean[0]).abs() < 1e-9);
        assert!((result.fit.posterior_mean[1] - truth.mean[1]).abs() < 1e-9);

        let (untreated, treated) = chain_columns(&result);
        assert_eq!(untreated.len(), DRAWS);
        // Y0 = b + e0 and Y1 = b + tau + e1, with e the declared target disturbance.
        let y0_variance = var_b + NOISE;
        let y1_variance = var_b + var_tau + 2.0 * cov + NOISE;
        assert!((mean(&untreated) - truth.mean[1]).abs() < mean_tolerance(y0_variance));
        assert!(
            (mean(&treated) - (truth.mean[0] + truth.mean[1])).abs() < mean_tolerance(y1_variance)
        );
        let difference: Vec<f64> = untreated.iter().zip(&treated).map(|(y0, y1)| y1 - y0).collect();
        // The disturbances' correlation fixes the spread of the paired difference.
        let expected = var_tau + 2.0 * NOISE * (1.0 - rho);
        let ratio = variance(&difference) / expected;
        assert!((ratio - 1.0).abs() < 0.3, "paired-difference variance ratio {ratio} at rho {rho}");

        // The decision scores those draws: hold is the mean untreated draw, treat the mean
        // treated draw minus the cost; treating wins because the planted effect (2) exceeds it.
        let actions = &result.decision.decision.actions;
        assert!((expected_utility(actions, "hold") - mean(&untreated)).abs() < 1e-9);
        assert!((expected_utility(actions, "treat") - (mean(&treated) - 1.0)).abs() < 1e-9);
        assert_eq!(result.decision.decision.verdict, Verdict::UniquelyOptimal("treat".into()));
        assert_eq!(result.calibration, "not_measured");
    }
}

/// A route's program reports the support decision of its unit; a run records what a release
/// build would refuse only while that unit is not supported (a development build runs the route
/// anyway, a release build with the unit supported records nothing).
fn assert_support_is_reported_and_consulted(
    plan_view: &antecedent::planner::PlanView,
    refused: &[antecedent_core::support_gate::WouldRefuse],
) {
    let view = plan_view.unit_support().expect("the route is bound to its capability record");
    assert_eq!(view.unit(), "transport.joint_transport_decision");
    assert_eq!(view.supported(), view.status().as_str() == "supported");
    assert_eq!(view.supported(), view.missing_evidence().is_empty());
    if view.supported() {
        assert!(refused.is_empty(), "{refused:?}");
    }
    assert!(refused.iter().all(|entry| entry.code == "not_supported"), "{refused:?}");
}

#[test]
fn both_joint_transport_routes_report_and_consult_their_support_unit() {
    let ctx = ExecutionContext::for_tests(91);
    let plan = compile_joint_transport_effect_decision(effect_request(3.0, 2.0, 1.0)).unwrap();
    plan.execute(&ctx).unwrap();
    assert_support_is_reported_and_consulted(&plan.plan_view(), &ctx.would_refuse());

    let (request, _) = predictive_request(3.0, 2.0, 1.0);
    let ctx = ExecutionContext::for_tests(92);
    let plan = compile_joint_transport_predictive_decision(request).unwrap();
    plan.execute(&ctx).unwrap();
    assert_support_is_reported_and_consulted(&plan.plan_view(), &ctx.would_refuse());
}

#[test]
fn a_decision_off_the_effect_coordinate_and_an_impossible_correlation_refuse() {
    // The effect decision reads exactly the target average-effect coordinate: a quantity that
    // names another regime is not that coordinate.
    let mut request = effect_request(3.0, 2.0, 1.0);
    request.quantity.regime_id = "do(0=1)".into();
    assert!(compile_joint_transport_effect_decision(request).is_err());
    // A correlation of two disturbances lies in [-1, 1].
    let (mut request, _) = predictive_request(3.0, 2.0, 1.0);
    request.target_noise_correlation = 1.5;
    assert!(compile_joint_transport_predictive_decision(request).is_err());
    // A non-positive residual variance is no variance.
    let (mut request, _) = predictive_request(3.0, 2.0, 1.0);
    request.target_noise_variance = 0.0;
    assert!(compile_joint_transport_predictive_decision(request).is_err());
}
