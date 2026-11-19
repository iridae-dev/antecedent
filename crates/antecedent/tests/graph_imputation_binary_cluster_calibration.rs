//! Prospective target-population coverage under independent two-row target
//! clusters. Every replicate generates new source rows and target clusters,
//! recompiles the complete cluster-bootstrap graph, imputation, transport and
//! decision schedule, and compares its interval with analytic population truth.
//! This ignored grid earns no calibration or decision reliability.

mod common;

use std::sync::Arc;

use antecedent::analysis::graph_imputation::{CompiledGraphImputation, GraphImputationSpec};
use antecedent::analysis::graph_imputation_transport_decision::{
    compile_graph_imputation_transport_cluster_bootstrap_decision, TargetClusterBootstrap,
};
use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_core::graph_imputation_contract::GraphSupport;
use antecedent_core::{CalibrationBasis, ExecutionContext, VariableId};
use antecedent_data::{ImputationSpec, TabularData};
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion,
    StructuralPolicy, UtilityExpr,
};
use antecedent_io::measured_inference::graph_candidate::TransportStandardization;
use common::calibration::{
    Construction, CoverageTally, REPORTED_LEVEL, RecordKey, ScopeFacts,
    gaussian, grid_n, grid_seed, map_replicates, n_sim, uniform,
};

const ROOT_TEST: &str = "graph_imputation_binary_cluster_target_l95";
const CONFOUNDED_TEST: &str = "graph_imputation_confounded_binary_cluster_target_l95";
const DGP: &str = "binary_modifier_mar_target_cluster_shift";
const INTERVAL: &str = "nearest_rank_95";

fn quantity() -> ScientificQuantity {
    ScientificQuantity {
        variable_id: "y".into(), variable_name: "target-frame effect".into(),
        role: QuantityRole::Outcome, units: "score".into(),
        population_id: "target-frame".into(),
        regime_id: "do(t=1)-do(t=0)".into(), horizon: 0,
        functional_id: "average_effect".into(), conditioning: Vec::new(),
        transform_id: "identity".into(),
    }
}

fn decision(effect: &ScientificQuantity) -> DecisionContract {
    DecisionContract {
        actions: vec![
            DecisionAction {
                id: "treat".into(), kind: ActionKind::External,
                inputs: vec![effect.clone()],
                utility: UtilityExpr::difference(
                    UtilityExpr::Input(0), UtilityExpr::Const(0.7)),
            },
            DecisionAction {
                id: "hold".into(), kind: ActionKind::External,
                inputs: vec![effect.clone()], utility: UtilityExpr::Const(0.0),
            },
        ],
        utility_units: "score".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: Vec::new(), target_population: "target-frame".into(),
        horizon: 0, structural_policy: StructuralPolicy::RequireInvariantBestAction,
    }
}

fn replicate(seed: u64, n: usize, confounded: bool) -> Option<((f64, f64), f64, CalibrationBasis)> {
    let mut source_random = uniform(seed, 0xA01C_E001);
    let mut target_random = uniform(seed, 0xA01C_E002);
    let mut noise = gaussian(seed ^ 0xA01C_E003);
    let (mut treatment, mut outcome, mut covariate) =
        (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
    for _ in 0..n {
        let x = f64::from(u8::from(source_random() < 0.35));
        let a = f64::from(u8::from(source_random() < (if confounded { 0.3 + 0.4 * x } else { 0.5 })));
        let y = 0.2 * x + a * (0.5 + 0.5 * x) + 0.3 * noise();
        let missing = source_random() < 0.06 + 0.06 * a;
        treatment.push(a);
        outcome.push(if missing { f64::NAN } else { y });
        covariate.push(x);
    }
    let target_x: Vec<f64> = (0..n.div_ceil(2))
        .flat_map(|_| { let x = f64::from(u8::from(target_random() < 0.75)); [x, x] })
        .take(n).collect();
    let truth = 0.5 + 0.5 * 0.75;
    let source = TabularData::from_f64_columns([
        ("t", treatment.as_slice()), ("y", outcome.as_slice()),
        ("x", covariate.as_slice()),
    ]).ok()?;
    let ctx = ExecutionContext::for_tests(grid_seed(seed));
    let producer = CompiledGraphImputation::compile(source, &GraphImputationSpec {
        treatment: VariableId::from_raw(0), outcome: VariableId::from_raw(1),
        support: if confounded { GraphSupport::PreTreatmentCovariates }
            else { GraphSupport::TreatmentRootPreTreatmentCovariates },
        imputation: ImputationSpec::new().with_imputations(2)
            .with_iterations(2).with_seed(grid_seed(seed ^ 0xA01C_E004)),
        domains: Arc::<[u8]>::from(vec![1, 0, 1]), posterior_draws: 100,
        prior_scale: 3.0, execution_seed: grid_seed(seed ^ 0xA01C_E005),
    }, &ctx).ok()?;
    let target = TransportStandardization {
        columns: vec![(2, target_x)], selection_targets: vec![2],
        source_population: "source".into(), target_population: "target-frame".into(),
        source_sample: producer.source_identity().to_hex(),
        target_sample: format!("target-frame-{seed}"),
        source_unit_ids: (0..n).map(|row| format!("source-{row}")).collect(),
        target_unit_ids: (0..n).map(|row| format!("target-{row}")).collect(),
    };
    let effect = quantity();
    let plan = compile_graph_imputation_transport_cluster_bootstrap_decision(
        producer, target, effect.clone(), decision(&effect),
        TargetClusterBootstrap { replicates: 40, seed: 91,
            cluster_ids: (0..n).map(|row| format!("cluster-{}", row / 2)).collect() }, confounded,
    ).ok()?;
    let candidate = plan.calibration_candidate(&ctx).ok()?;
    assert_eq!(candidate.result.calibration, "not_measured");
    assert_eq!(candidate.basis.interval_method.as_ref(), INTERVAL);
    assert_eq!(candidate.basis.estimator.as_ref(), if confounded {
        "graph_imputation_confounded_transport_target_cluster_bootstrap_decision_v1"
    } else { "graph_imputation_transport_target_cluster_bootstrap_decision_v1" });
    Some((candidate.interval, truth, candidate.basis))
}

fn run_grid(test: &'static str, confounded: bool) {
    let n = grid_n(120);
    let mut tally = CoverageTally::for_record(RecordKey {
        test, dgp: DGP, interval: INTERVAL,
    }, REPORTED_LEVEL);
    for result in map_replicates(n_sim(), |rep| {
        replicate(0xD0C0_5001 + rep, n, confounded)
    }) {
        if let Some((interval, truth, basis)) = result {
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
            tally.bind_validation(&construction, ScopeFacts {
                row_count: basis.row_count,
                replicates_ok: basis.replicates_ok,
                posterior_draws: basis.posterior_draws,
                unidentified_mass: basis.unidentified_mass,
            }, &basis.validation);
            tally.record(Some(interval), truth);
        } else {
            tally.skip();
        }
    }
    tally.assert();
}

#[test]
#[ignore = "prospective target cluster population coverage; no record earned"]
fn graph_imputation_binary_cluster_target_l95() { run_grid(ROOT_TEST, false); }

#[test]
#[ignore = "prospective confounded target cluster population coverage; no record earned"]
fn graph_imputation_confounded_binary_cluster_target_l95() { run_grid(CONFOUNDED_TEST, true); }
