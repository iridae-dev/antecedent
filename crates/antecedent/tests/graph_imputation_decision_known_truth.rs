//! Known-truth checks of the graph-imputation decision routes against their generating truth.
//!
//! Every source is generated from a declared outcome mechanism, so the effect it implies is known
//! in closed form before any library code runs:
//!
//! * the outcome is `1 + 2 t + x + 1.5 t x` plus a small residual, so the treatment effect at
//!   covariate value `x` is `2 + 1.5 x` and the mean outcome under `do(t = a)` is
//!   `1 + 2 a + (1 + 1.5 a) mean(x)`;
//! * the effect transported to a target frame is therefore `2 + 1.5 * (target mean of x)`, which
//!   differs from the source effect (mean of `x` one half) by more than every tolerance below,
//!   so a route that ignored its target frame fails;
//! * the binary-outcome cells have declared success rates, so the success probability under
//!   `do(t = a)` is the target mix of them;
//! * two Gaussian predictive worlds that share a declared residual law differ with variance
//!   `2 variance (1 - correlation)` plus the (negligible) posterior variance of the mean
//!   difference.
//!
//! In the binary designs the residuals sum to zero within each cell, and the modifier enters the
//! fitted outcome model with its treatment interaction, so the least-squares coefficients are the
//! cell means whatever weights a target frame induces; a weak prior (scale 10) keeps the ridge
//! shrinkage far below every tolerance. The action utilities are linear in the quantity they
//! read, so a decision's expected utility is the generating truth of that quantity less the
//! declared threshold. Tolerances are several Monte-Carlo standard errors of the quantity they
//! bound, and a bootstrap route (whose draws are one per replicate) uses a wider one. No
//! calibration is asserted anywhere.

use std::sync::Arc;

use antecedent::analysis::graph_imputation::{
    CompiledGraphImputation, GraphImputationSpec, GraphImputationSupport,
};
use antecedent::analysis::graph_imputation_effect_decision::{
    GaussianPredictivePolicy, compile_graph_imputation_bernoulli_predictive_world_decision,
    compile_graph_imputation_effect_decision,
    compile_graph_imputation_gaussian_target_predictive_world_decision,
    compile_graph_imputation_world_decision,
};
use antecedent::analysis::graph_imputation_kernel_transport_decision::compile_graph_imputation_kernel_transport_decision;
use antecedent::analysis::graph_imputation_mar_transport_decision::compile_graph_imputation_mar_transport_decision;
use antecedent::analysis::graph_imputation_transport_decision::{
    TargetClusterBootstrap, compile_graph_imputation_confounded_transport_decision,
    compile_graph_imputation_transport_cluster_bootstrap_decision,
    compile_graph_imputation_transport_decision,
};
use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_core::{ExecutionContext, VariableId};
use antecedent_data::{ImputationSpec, TabularData};
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, StructuralPolicy, UtilityExpr,
};
use antecedent_design::decision_eval::{DecisionResult, Verdict};
use antecedent_io::measured_inference::graph_candidate::{
    KernelTransportStandardization, MarTargetStandardization, RawTargetFrame,
    TransportStandardization,
};

const NOISE: [f64; 10] = [0.1, -0.1, 0.05, -0.05, 0.2, -0.2, 0.15, -0.15, 0.0, 0.0];
const PRIOR_SCALE: f64 = 10.0;
const THRESHOLD: f64 = 3.0;

fn outcome(treated: f64, modifier: f64, noise: f64) -> f64 {
    1.0 + 2.0 * treated + modifier + 1.5 * treated * modifier + noise
}

fn effect_at(modifier: f64) -> f64 {
    2.0 + 1.5 * modifier
}

fn world_at(arm: u8, modifier: f64) -> f64 {
    let arm = f64::from(arm);
    1.0 + 2.0 * arm + (1.0 + 1.5 * arm) * modifier
}

/// Every (treatment, modifier) cell holds 40 rows and its residuals sum to zero.
fn factorial() -> [Vec<f64>; 3] {
    let rows = 160;
    let modifier: Vec<f64> = (0..rows).map(|row| f64::from(u8::from(row % 2 == 1))).collect();
    let treated: Vec<f64> = (0..rows).map(|row| f64::from(u8::from((row / 2) % 2 == 1))).collect();
    let y = (0..rows)
        .map(|row| outcome(treated[row], modifier[row], NOISE[(row / 4) % NOISE.len()]))
        .collect();
    [treated, y, modifier]
}

/// A binary confounder of treatment: 40 rows in the cells that agree, 10 in those that differ.
fn confounded() -> [Vec<f64>; 3] {
    let (mut treated, mut modifier, mut y) = (Vec::new(), Vec::new(), Vec::new());
    for (cell, (arm, level, count)) in
        [(0.0, 0.0, 40), (1.0, 0.0, 10), (0.0, 1.0, 10), (1.0, 1.0, 40)].into_iter().enumerate()
    {
        for index in 0..count {
            treated.push(arm);
            modifier.push(level);
            y.push(outcome(arm, level, NOISE[(index + cell) % NOISE.len()]));
        }
    }
    [treated, y, modifier]
}

/// A modifier on a twenty-point grid in both arms with a small fixed residual pattern.
fn continuous() -> [Vec<f64>; 3] {
    let rows = 160;
    let modifier: Vec<f64> = (0..rows).map(|row| (row % 20) as f64 / 19.0).collect();
    let treated: Vec<f64> = (0..rows).map(|row| f64::from(u8::from((row / 20) % 2 == 1))).collect();
    let y = (0..rows)
        .map(|row| {
            outcome(treated[row], modifier[row], 0.05 * (((row * 37) % 11) as f64 - 5.0) / 5.0)
        })
        .collect();
    [treated, y, modifier]
}

/// Cell success rates 0.2 and 0.4 (x = 0) and 0.4 and 0.8 (x = 1) for control and treated.
fn binary_outcome() -> [Vec<f64>; 3] {
    let (mut treated, mut modifier, mut y) = (Vec::new(), Vec::new(), Vec::new());
    for (arm, level, successes) in [(0.0, 0.0, 8), (1.0, 0.0, 16), (0.0, 1.0, 16), (1.0, 1.0, 32)] {
        for index in 0..40 {
            treated.push(arm);
            modifier.push(level);
            y.push(f64::from(u8::from(index < successes)));
        }
    }
    [treated, y, modifier]
}

fn producer(
    columns: &[Vec<f64>; 3],
    domains: &[u8],
    support: GraphImputationSupport,
    ctx: &ExecutionContext,
) -> CompiledGraphImputation {
    let table = TabularData::from_f64_columns([
        ("t", columns[0].as_slice()),
        ("y", columns[1].as_slice()),
        ("x", columns[2].as_slice()),
    ])
    .unwrap();
    CompiledGraphImputation::compile(
        table,
        &GraphImputationSpec {
            treatment: VariableId::from_raw(0),
            outcome: VariableId::from_raw(1),
            support,
            imputation: ImputationSpec::new().with_imputations(2).with_iterations(2).with_seed(17),
            domains: Arc::from(domains),
            posterior_draws: 400,
            prior_scale: PRIOR_SCALE,
            execution_seed: 23,
        },
        ctx,
    )
    .unwrap()
}

fn quantity(population: &str, regime: &str, functional: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: "y".into(),
        variable_name: "outcome".into(),
        role: QuantityRole::Outcome,
        units: "score".into(),
        population_id: population.into(),
        regime_id: regime.into(),
        horizon: 0,
        functional_id: functional.into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    }
}

fn effect_quantity(population: &str) -> ScientificQuantity {
    quantity(population, "do(t=1)-do(t=0)", "average_effect")
}

fn effect_contract(population: &str, threshold: f64) -> DecisionContract {
    let effect = effect_quantity(population);
    DecisionContract {
        actions: vec![
            DecisionAction {
                id: "treat".into(),
                kind: ActionKind::External,
                inputs: vec![effect.clone()],
                utility: UtilityExpr::difference(
                    UtilityExpr::Input(0),
                    UtilityExpr::Const(threshold),
                ),
            },
            DecisionAction {
                id: "hold".into(),
                kind: ActionKind::External,
                inputs: vec![effect],
                utility: UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Const(0.0)),
            },
        ],
        utility_units: "score".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: Vec::new(),
        target_population: population.into(),
        horizon: 0,
        structural_policy: StructuralPolicy::RequireInvariantBestAction,
    }
}

fn worlds(population: &str, functional: &str) -> [ScientificQuantity; 2] {
    [0, 1].map(|arm| quantity(population, &format!("do(t={arm})"), functional))
}

fn world_contract(
    population: &str,
    functional: &str,
) -> (DecisionContract, [ScientificQuantity; 2]) {
    let quantities = worlds(population, functional);
    let contract = DecisionContract {
        actions: quantities
            .iter()
            .enumerate()
            .map(|(arm, world)| DecisionAction {
                id: format!("action-{arm}"),
                kind: ActionKind::Intervention,
                inputs: vec![world.clone()],
                utility: UtilityExpr::Input(0),
            })
            .collect(),
        utility_units: "score".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: Vec::new(),
        target_population: population.into(),
        horizon: 0,
        structural_policy: StructuralPolicy::RequireInvariantBestAction,
    };
    (contract, quantities)
}

fn utility(result: &DecisionResult, action: &str) -> f64 {
    result.actions.iter().find(|outcome| outcome.id == action).unwrap().expected_utility
}

fn chosen(result: &DecisionResult) -> &str {
    match &result.verdict {
        Verdict::UniquelyOptimal(id) => id,
        other => panic!("expected a unique best action, got {other:?}"),
    }
}

fn near(actual: f64, truth: f64, tolerance: f64, what: &str) {
    assert!((actual - truth).abs() < tolerance, "{what}: {actual} differs from {truth}");
}

fn ids(prefix: &str, rows: usize) -> Vec<String> {
    (0..rows).map(|row| format!("{prefix}-{row}")).collect()
}

fn binary_target(
    producer: &CompiledGraphImputation,
    ones: usize,
    rows: usize,
) -> TransportStandardization {
    TransportStandardization {
        columns: vec![(2, (0..rows).map(|row| f64::from(u8::from(row < ones))).collect())],
        selection_targets: vec![2],
        source_population: "source".into(),
        target_population: "target".into(),
        source_sample: producer.source_identity().to_hex(),
        target_sample: "independent-target-sample".into(),
        source_unit_ids: ids("source", 160),
        target_unit_ids: ids("target", rows),
    }
}

fn clusters(rows: usize) -> TargetClusterBootstrap {
    TargetClusterBootstrap {
        replicates: 40,
        seed: 29,
        cluster_ids: (0..rows).map(|row| format!("cluster-{}", row / 2)).collect(),
    }
}

const BINARY: [u8; 3] = [1, 0, 1];
const ROOT: GraphImputationSupport = GraphImputationSupport::TreatmentRootPreTreatmentCovariates;

#[test]
fn a_source_effect_decision_matches_the_true_effect_and_a_threshold_decides_it() {
    let ctx = ExecutionContext::for_tests(23);
    // The effect in the source is 2 + 1.5 * 0.5 = 2.75.
    let plan = compile_graph_imputation_effect_decision(
        producer(&factorial(), &BINARY, ROOT, &ctx),
        effect_quantity("source"),
        effect_contract("source", 2.0),
    )
    .unwrap();
    let result = plan.execute(&ctx).unwrap();
    near(result.producer.mean, effect_at(0.5), 0.05, "source effect");
    near(utility(&result.decision.decision, "treat"), effect_at(0.5) - 2.0, 0.05, "treat utility");
    assert!(utility(&result.decision.decision, "hold").abs() < 1e-9);
    assert_eq!(chosen(&result.decision.decision), "treat");
    // A cost above the true effect holds.
    let plan = compile_graph_imputation_effect_decision(
        producer(&factorial(), &BINARY, ROOT, &ctx),
        effect_quantity("source"),
        effect_contract("source", THRESHOLD),
    )
    .unwrap();
    let result = plan.execute(&ctx).unwrap();
    near(utility(&result.decision.decision, "treat"), effect_at(0.5) - THRESHOLD, 0.05, "treat");
    assert_eq!(chosen(&result.decision.decision), "hold");
}

#[test]
fn source_treatment_worlds_have_the_true_mean_outcomes() {
    let ctx = ExecutionContext::for_tests(23);
    let (contract, quantities) = world_contract("source", "mean");
    let plan = compile_graph_imputation_world_decision(
        producer(&factorial(), &BINARY, ROOT, &ctx),
        quantities,
        contract,
    )
    .unwrap();
    let result = plan.execute(&ctx).unwrap();
    near(utility(&result.decision.decision, "action-0"), world_at(0, 0.5), 0.05, "control world");
    near(utility(&result.decision.decision, "action-1"), world_at(1, 0.5), 0.05, "active world");
    assert_eq!(chosen(&result.decision.decision), "action-1");
}

#[test]
fn binary_transport_matches_the_effect_at_the_target_frame_and_changes_the_action() {
    let ctx = ExecutionContext::for_tests(23);
    // 80% of the target frame carries x = 1: the effect is 2 + 1.5 * 0.8 = 3.2, above the cost,
    // while the source effect 2.75 is below it.
    let build = || producer(&factorial(), &BINARY, ROOT, &ctx);
    let source = build();
    let plan = compile_graph_imputation_transport_decision(
        build(),
        binary_target(&source, 80, 100),
        effect_quantity("target"),
        effect_contract("target", THRESHOLD),
    )
    .unwrap();
    let result = plan.execute(&ctx).unwrap();
    near(result.producer.mean, effect_at(0.8), 0.05, "transported effect");
    near(utility(&result.decision.decision, "treat"), effect_at(0.8) - THRESHOLD, 0.05, "treat");
    assert_eq!(chosen(&result.decision.decision), "treat");
    // The same cluster-resampled target: one draw per replicate around the same truth.
    let plan = compile_graph_imputation_transport_cluster_bootstrap_decision(
        build(),
        binary_target(&source, 80, 100),
        effect_quantity("target"),
        effect_contract("target", THRESHOLD),
        clusters(100),
        false,
    )
    .unwrap();
    let result = plan.execute(&ctx).unwrap();
    assert_eq!(result.producer.draws.len(), 40);
    near(result.producer.mean, effect_at(0.8), 0.2, "cluster-resampled transported effect");
}

#[test]
fn confounded_transport_matches_the_effect_at_the_target_frame() {
    let ctx = ExecutionContext::for_tests(23);
    let source =
        producer(&confounded(), &BINARY, GraphImputationSupport::PreTreatmentCovariates, &ctx);
    let target = TransportStandardization {
        source_unit_ids: ids("source", 100),
        ..binary_target(&source, 80, 100)
    };
    let plan = compile_graph_imputation_confounded_transport_decision(
        source,
        target,
        effect_quantity("target"),
        effect_contract("target", THRESHOLD),
    )
    .unwrap();
    let result = plan.execute(&ctx).unwrap();
    near(result.producer.mean, effect_at(0.8), 0.05, "confounded transported effect");
    assert_eq!(chosen(&result.decision.decision), "treat");
}

#[test]
fn kernel_transport_matches_the_effect_at_the_continuous_target_frame() {
    let ctx = ExecutionContext::for_tests(23);
    let source = producer(&continuous(), &[1, 0, 0], ROOT, &ctx);
    let rows = 80;
    // The target modifier spans 0.6 to 1.0 with mean 0.8.
    let target = KernelTransportStandardization {
        columns: vec![(2, (0..rows).map(|row| 0.6 + 0.4 * (row % 20) as f64 / 19.0).collect())],
        selection_targets: vec![2],
        source_population: "source".into(),
        target_population: "target".into(),
        source_sample: source.source_identity().to_hex(),
        target_sample: "independent-continuous-target".into(),
        source_unit_ids: ids("source", 160),
        target_unit_ids: ids("target", rows),
        bandwidths: vec![(2, 0.3)],
        min_kernel_overlap: 0.01,
    };
    let plan = compile_graph_imputation_kernel_transport_decision(
        source,
        target,
        effect_quantity("target"),
        effect_contract("target", THRESHOLD),
    )
    .unwrap();
    let result = plan.execute(&ctx).unwrap();
    near(result.producer.mean, effect_at(0.8), 0.1, "kernel transported effect");
    assert_eq!(chosen(&result.decision.decision), "treat");
}

#[test]
fn missing_at_random_transport_matches_the_weighted_complete_case_target() {
    let ctx = ExecutionContext::for_tests(23);
    let source = producer(&factorial(), &BINARY, ROOT, &ctx);
    let rows = 100;
    let values: Vec<f64> = (0..rows).map(|row| f64::from(u8::from(row % 5 != 0))).collect();
    let cells: Vec<Option<f64>> =
        (0..rows).map(|row| (row % 7 != 0).then_some(values[row])).collect();
    // Complete rows count by weight over the observation probability (constant here), and a
    // treated-modifier row weighs three times a control-modifier row.
    let weights: Vec<f64> = values.iter().map(|&value| 1.0 + 2.0 * value).collect();
    let complete: Vec<usize> = (0..rows).filter(|&row| cells[row].is_some()).collect();
    let mean = complete.iter().map(|&row| weights[row] * values[row]).sum::<f64>()
        / complete.iter().map(|&row| weights[row]).sum::<f64>();
    let target = MarTargetStandardization {
        observation_mechanism_id: "mar-v1".into(),
        sampling_design_id: "cluster-design-v1".into(),
        frame: RawTargetFrame {
            columns: vec![(2, cells)],
            row_ids: ids("target", rows),
            cluster_ids: (0..rows).map(|row| format!("cluster-{}", row / 4)).collect(),
            weights,
            observation_probabilities: vec![0.8; rows],
            always_observed: Vec::new(),
        },
        selection_targets: vec![2],
        source_population: "source".into(),
        target_population: "target".into(),
        source_sample: source.source_identity().to_hex(),
        target_sample: "target-sample".into(),
        source_unit_ids: ids("source", 160),
        bandwidths: Vec::new(),
        min_kernel_overlap: 0.0,
        kernel: false,
        confounded: false,
    };
    let plan = compile_graph_imputation_mar_transport_decision(
        source,
        target,
        effect_quantity("target"),
        effect_contract("target", THRESHOLD),
    )
    .unwrap();
    let result = plan.execute(&ctx).unwrap();
    near(result.producer.mean, effect_at(mean), 0.08, "weighted complete-case effect");
}

#[test]
fn gaussian_predictive_worlds_have_the_true_mean_and_the_declared_residual_spread() {
    let ctx = ExecutionContext::for_tests(23);
    let source = producer(&factorial(), &BINARY, ROOT, &ctx);
    let target = binary_target(&source, 80, 100);
    let (contract, quantities) = world_contract("target", "outcome");
    let plan = compile_graph_imputation_gaussian_target_predictive_world_decision(
        source,
        target,
        quantities,
        contract,
        GaussianPredictivePolicy { residual_variance: 0.4, cross_world_correlation: 0.3, seed: 5 },
    )
    .unwrap();
    let result = plan.execute(&ctx).unwrap();
    near(utility(&result.decision.decision, "action-0"), world_at(0, 0.8), 0.1, "control world");
    near(utility(&result.decision.decision, "action-1"), world_at(1, 0.8), 0.1, "active world");
    assert_eq!(chosen(&result.decision.decision), "action-1");
    // The two residuals have correlation 0.3, so the difference has variance
    // 2 * 0.4 * (1 - 0.3) = 0.56 plus the negligible posterior variance of the mean difference.
    let differences: Vec<f64> =
        result.producer.worlds.iter().map(|pair| pair[1] - pair[0]).collect();
    let centre = differences.iter().sum::<f64>() / differences.len() as f64;
    let variance = differences.iter().map(|d| (d - centre).powi(2)).sum::<f64>()
        / (differences.len() - 1) as f64;
    near(variance, 0.56, 0.15, "difference variance");
}

#[test]
#[allow(clippy::float_cmp, reason = "the predictive worlds are exactly zero or one")]
fn bernoulli_predictive_worlds_have_the_true_success_probabilities() {
    let ctx = ExecutionContext::for_tests(23);
    let source = producer(&binary_outcome(), &[1, 1, 1], ROOT, &ctx);
    let target = binary_target(&source, 128, 160);
    let (contract, quantities) = world_contract("target", "outcome");
    let plan = compile_graph_imputation_bernoulli_predictive_world_decision(
        source, target, quantities, contract,
    )
    .unwrap();
    let result = plan.execute(&ctx).unwrap();
    // The target mix is 80% x = 1: success under control is 0.2 * 0.2 + 0.8 * 0.4 = 0.36 and under
    // treatment 0.2 * 0.4 + 0.8 * 0.8 = 0.72.
    assert!(result.producer.worlds.iter().flatten().all(|&value| value == 0.0 || value == 1.0));
    let draws = result.producer.probabilities.len() as f64;
    let control = result.producer.probabilities.iter().map(|pair| pair[0]).sum::<f64>() / draws;
    let active = result.producer.probabilities.iter().map(|pair| pair[1]).sum::<f64>() / draws;
    near(control, 0.36, 0.05, "control success probability");
    near(active, 0.72, 0.05, "active success probability");
    near(utility(&result.decision.decision, "action-0"), 0.36, 0.06, "control utility");
    near(utility(&result.decision.decision, "action-1"), 0.72, 0.06, "active utility");
    assert_eq!(chosen(&result.decision.decision), "action-1");
}
