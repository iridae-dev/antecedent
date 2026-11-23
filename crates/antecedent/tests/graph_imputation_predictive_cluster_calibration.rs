//! Predictive coverage with independent two-row target clusters. Each
//! replicate resamples whole clusters and refits the complete predictive
//! graph-imputation model. Gaussian truth is a future target-population
//! individual's outcome; Bernoulli truth is that individual's success
//! probability under the known population law, because the interval of a 0/1
//! outcome is its support and covers every outcome.

mod common;

use antecedent::analysis::graph_imputation::{CompiledGraphImputation, GraphImputationSpec};
use antecedent::analysis::graph_imputation_effect_decision::{
    GaussianPredictivePolicy, compile_graph_imputation_bernoulli_predictive_world_decision,
    compile_graph_imputation_gaussian_target_predictive_world_decision,
    compile_graph_imputation_predictive_cluster_bootstrap_decision,
};
use antecedent::analysis::graph_imputation_transport_decision::TargetClusterBootstrap;
use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_core::graph_imputation_contract::GraphSupport;
use antecedent_core::{CalibrationBasis, ExecutionContext, SemanticDigest, VariableId};
use antecedent_data::{ImputationSpec, TabularData};
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, StructuralPolicy, UtilityExpr,
};
use antecedent_design::decision_eval::Verdict;
use antecedent_io::measured_inference::graph_candidate::TransportStandardization;
use common::calibration::{
    Construction, CoverageTally, REPORTED_LEVEL, RecordKey, ScopeFacts, gaussian, grid_n,
    grid_seed, map_replicates, n_sim, uniform,
};
use std::sync::Arc;

fn logistic(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

type Observation = ((f64, f64), f64, CalibrationBasis, SemanticDigest, bool);

#[allow(clippy::too_many_lines, reason = "one replicate builds the whole simulated study")]
fn replicate(seed: u64, n: usize, binary: bool, world: usize) -> Option<Observation> {
    let mut source = uniform(seed, 0xA51C_0001);
    let mut target_rng = uniform(seed, 0xA51C_0002);
    let mut future_rng = uniform(seed, 0xA51C_0003);
    let mut normal = gaussian(seed ^ 0xA51C_0004);
    let mut treatment = Vec::with_capacity(n);
    let mut outcome = Vec::with_capacity(n);
    let mut covariate = Vec::with_capacity(n);
    for _ in 0..n {
        let x = f64::from(u8::from(source() < 0.35));
        let a = f64::from(u8::from(source() < 0.5));
        let y = if binary {
            f64::from(u8::from(source() < logistic(-1.0 + 0.4 * x + a * (1.0 + 0.4 * x))))
        } else {
            0.2 * x + a * (0.5 + 0.5 * x) + 0.3 * normal()
        };
        treatment.push(a);
        outcome.push(if source() < 0.06 + 0.06 * a { f64::NAN } else { y });
        covariate.push(x);
    }
    let target_x: Vec<f64> = (0..n.div_ceil(2))
        .flat_map(|_| {
            let x = f64::from(u8::from(target_rng() < 0.75));
            [x, x]
        })
        .take(n)
        .collect();
    let x = f64::from(u8::from(future_rng() < 0.75));
    let a = world as f64;
    let truth = if binary {
        0.75 * logistic(-1.0 + 0.4 + a * 1.4) + 0.25 * logistic(-1.0 + a)
    } else {
        0.2 * x + a * (0.5 + 0.5 * x) + 0.3 * normal()
    };
    let table = TabularData::from_f64_columns([
        ("t", treatment.as_slice()),
        ("y", outcome.as_slice()),
        ("x", covariate.as_slice()),
    ])
    .ok()?;
    let ctx = ExecutionContext::for_tests(grid_seed(seed));
    let producer = CompiledGraphImputation::compile(
        table,
        &GraphImputationSpec {
            treatment: VariableId::from_raw(0),
            outcome: VariableId::from_raw(1),
            support: GraphSupport::TreatmentRootPreTreatmentCovariates,
            imputation: ImputationSpec::new()
                .with_imputations(2)
                .with_iterations(2)
                .with_seed(grid_seed(seed ^ 0xA51C_0005)),
            domains: Arc::from(vec![1, u8::from(binary), 1]),
            posterior_draws: 100,
            prior_scale: 1.0,
            execution_seed: grid_seed(seed ^ 0xA51C_0006),
        },
        &ctx,
    )
    .ok()?;
    let target = TransportStandardization {
        columns: vec![(2, target_x)],
        selection_targets: vec![2],
        source_population: "source".into(),
        target_population: "target".into(),
        source_sample: producer.source_identity().to_hex(),
        target_sample: format!("target-{seed}"),
        source_unit_ids: (0..n).map(|row| format!("source-{row}")).collect(),
        target_unit_ids: (0..n).map(|row| format!("target-{row}")).collect(),
    };
    let quantities = [0, 1].map(|arm| ScientificQuantity {
        variable_id: "y".into(),
        variable_name: "individual outcome".into(),
        role: QuantityRole::Outcome,
        units: "score".into(),
        population_id: "target".into(),
        regime_id: format!("do(t={arm})"),
        horizon: 0,
        functional_id: "outcome".into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    });
    let decision = DecisionContract {
        actions: quantities
            .iter()
            .enumerate()
            .map(|(arm, q)| DecisionAction {
                id: format!("action-{arm}"),
                kind: ActionKind::Intervention,
                inputs: vec![q.clone()],
                utility: UtilityExpr::Input(0),
            })
            .collect(),
        utility_units: "score".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: Vec::new(),
        target_population: "target".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::RequireInvariantBestAction,
    };
    let fixed = if binary {
        compile_graph_imputation_bernoulli_predictive_world_decision(
            producer, target, quantities, decision,
        )
        .ok()?
    } else {
        compile_graph_imputation_gaussian_target_predictive_world_decision(
            producer,
            target,
            quantities,
            decision,
            GaussianPredictivePolicy {
                residual_variance: 0.09,
                cross_world_correlation: 0.0,
                seed: 91,
            },
        )
        .ok()?
    };
    let plan = compile_graph_imputation_predictive_cluster_bootstrap_decision(
        fixed,
        TargetClusterBootstrap {
            replicates: 40,
            seed: 91,
            cluster_ids: (0..n).map(|row| format!("cluster-{}", row / 2)).collect(),
        },
    )
    .ok()?;
    let candidate = plan.predictive_calibration_candidate(&ctx).ok()?;
    assert_eq!(candidate.coverage_bases[world].replicates_ok, Some(40));
    assert_eq!(candidate.result.calibration, "not_measured");
    let basis = candidate.coverage_bases[world].clone();
    assert_eq!(
        basis.estimator.as_ref(),
        if binary {
            "graph_imputation_bernoulli_target_predictive_cluster_bootstrap_decision_v1"
        } else {
            "graph_imputation_gaussian_target_predictive_cluster_bootstrap_decision_v1"
        }
    );
    let correct = matches!(&candidate.result.decision.decision.verdict,
        Verdict::UniquelyOptimal(id) if id == "action-1");
    Some((candidate.world_intervals[world], truth, basis, candidate.action_choice_basis, correct))
}

fn run_grid(test: &'static str, binary: bool, world: usize) {
    let n = grid_n(120);
    let interval =
        if binary { "predictive_binary_success_probability" } else { "predictive_gaussian_worlds" };
    let mut tally = CoverageTally::for_record(
        RecordKey {
            test,
            dgp: if binary {
                "mar_logit_binary_target_population_success_probability"
            } else {
                "mar_gaussian_binary_target_future"
            },
            interval,
        },
        REPORTED_LEVEL,
    );
    let mut action_scope = None;
    let (mut choices, mut correct) = (0_u32, 0_u32);
    for result in map_replicates(n_sim(), |rep| replicate(0xA51C_9001 + rep, n, binary, world)) {
        if let Some((interval, truth, basis, scope, optimal)) = result {
            if let Some(previous) = action_scope {
                assert_eq!(previous, scope);
            } else {
                action_scope = Some(scope);
            }
            choices += 1;
            correct += u32::from(optimal);
            let construction = Construction {
                query: basis.query.to_string(),
                graph_class: basis.graph_class.to_string(),
                structure: basis.structure.to_string(),
                modality: basis.modality.to_string(),
                inference: basis.inference.to_string(),
                estimator: basis.estimator.to_string(),
                interval_method: basis.interval_method.to_string(),
                se_kind: basis.se_kind.to_string(),
                dependence: basis.dependence.to_string(),
                posterior: basis.posterior.to_string(),
                functional: basis.functional.to_string(),
                identification: basis.identification.to_string(),
                reported_level: basis.level,
            };
            tally.bind_validation(
                &construction,
                ScopeFacts {
                    row_count: basis.row_count,
                    replicates_ok: basis.replicates_ok,
                    posterior_draws: basis.posterior_draws,
                    unidentified_mass: basis.unidentified_mass,
                },
                &basis.validation,
            );
            tally.record(Some(interval), truth);
        } else {
            tally.skip();
        }
    }
    eprintln!("prospective action choices: {correct}/{choices} correct; no decision record earned");
    tally.assert();
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn graph_imputation_bernoulli_predictive_cluster_control_l95() {
    run_grid("graph_imputation_bernoulli_predictive_cluster_control_l95", true, 0);
}
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn graph_imputation_bernoulli_predictive_cluster_active_l95() {
    run_grid("graph_imputation_bernoulli_predictive_cluster_active_l95", true, 1);
}
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn graph_imputation_gaussian_predictive_cluster_control_l95() {
    run_grid("graph_imputation_gaussian_predictive_cluster_control_l95", false, 0);
}
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn graph_imputation_gaussian_predictive_cluster_active_l95() {
    run_grid("graph_imputation_gaussian_predictive_cluster_active_l95", false, 1);
}
