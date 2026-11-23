//! Whole-route coverage of a mixed target graph-imputation effect. Each
//! replicate draws an independent continuous target frame, samples a new
//! source trial with missing outcomes, reruns graph selection per imputation,
//! kernel standardizes each graph and scores the final effect interval, Rubin's
//! rules across the completed tables, against analytic truth conditional on
//! that target frame.

mod common;

use std::sync::Arc;

use antecedent::analysis::graph_imputation::{CompiledGraphImputation, GraphImputationSpec};
use antecedent::analysis::graph_imputation_kernel_transport_decision::compile_graph_imputation_kernel_transport_decision;
use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_core::graph_imputation_contract::GraphSupport;
use antecedent_core::{CalibrationBasis, ExecutionContext, VariableId};
use antecedent_data::{ImputationSpec, TabularData};
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, StructuralPolicy, UtilityExpr,
};
use antecedent_io::measured_inference::graph_candidate::KernelTransportStandardization;
use common::calibration::{
    Construction, CoverageTally, REPORTED_LEVEL, RecordKey, ScopeFacts, gaussian, grid_n,
    grid_seed, map_replicates, n_sim, uniform,
};

const TEST: &str = "graph_imputation_kernel_target_frame_l95";
const DGP: &str = "mar_continuous_modifier_independent_target_frame";
const INTERVAL: &str = "rubin_barnard_t_95";

fn replicate(seed: u64, n: usize) -> Option<((f64, f64), f64, CalibrationBasis)> {
    let mut source_x = uniform(seed, 0x6B31_1001);
    let mut source_a = uniform(seed, 0x6B31_1002);
    let mut source_miss = uniform(seed, 0x6B31_1003);
    let mut outcome_noise = gaussian(seed ^ 0x6B31_1004);
    let mut target_x = uniform(seed, 0x6B31_1005);
    let mut treatment = Vec::with_capacity(n);
    let mut outcome = Vec::with_capacity(n);
    let mut covariate = Vec::with_capacity(n);
    for _ in 0..n {
        let x = source_x();
        let a = if source_a() < 0.5 { 0.0 } else { 1.0 };
        let y = 0.2 * x + a * (0.5 + 0.5 * x) + 0.1 * outcome_noise();
        treatment.push(a);
        outcome.push(if source_miss() < 0.1 { f64::NAN } else { y });
        covariate.push(x);
    }
    let target_values: Vec<f64> = (0..n).map(|_| 0.2 + 0.8 * target_x()).collect();
    let truth = 0.5 + 0.5 * target_values.iter().sum::<f64>() / n as f64;
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
                .with_seed(grid_seed(seed ^ 0x6B31_2001)),
            domains: Arc::<[u8]>::from(vec![1, 0, 0]),
            posterior_draws: 100,
            // The conjugate prior is beta | sigma2 ~ N(0, sigma2 * scale^2). With residual SD
            // 0.1 a unit scale puts the 0.8 effect eight prior SDs from zero and shrinks every
            // posterior toward it; the weakly informative scale lets the data dominate.
            prior_scale: 10.0,
            execution_seed: grid_seed(seed ^ 0x6B31_2002),
        },
        &ctx,
    )
    .ok()?;
    let target = KernelTransportStandardization {
        columns: vec![(2, target_values)],
        selection_targets: vec![2],
        source_population: "source".into(),
        target_population: "target".into(),
        source_sample: producer.source_identity().to_hex(),
        target_sample: "independent-target-frame".into(),
        source_unit_ids: (0..n).map(|row| format!("source-{row}")).collect(),
        target_unit_ids: (0..n).map(|row| format!("target-{row}")).collect(),
        bandwidths: vec![(2, 0.12)],
        min_kernel_overlap: 0.01,
    };
    let quantity = ScientificQuantity {
        variable_id: "y".into(),
        variable_name: "target effect".into(),
        role: QuantityRole::Outcome,
        units: "score".into(),
        population_id: "target".into(),
        regime_id: "do(t=1)-do(t=0)".into(),
        horizon: 0,
        functional_id: "average_effect".into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    };
    let decision = DecisionContract {
        actions: vec![
            DecisionAction {
                id: "treat".into(),
                kind: ActionKind::External,
                inputs: vec![quantity.clone()],
                utility: UtilityExpr::Input(0),
            },
            DecisionAction {
                id: "hold".into(),
                kind: ActionKind::External,
                inputs: vec![quantity.clone()],
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
    let plan =
        compile_graph_imputation_kernel_transport_decision(producer, target, quantity, decision)
            .ok()?;
    let candidate = plan.calibration_candidate(&ctx).ok()?;
    assert_eq!(candidate.result.calibration, "not_measured");
    assert_eq!(candidate.basis.interval_method.as_ref(), INTERVAL);
    Some((candidate.interval, truth, candidate.basis))
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn graph_imputation_kernel_target_frame_l95() {
    let n = grid_n(120);
    let mut tally = CoverageTally::for_record(
        RecordKey { test: TEST, dgp: DGP, interval: INTERVAL },
        REPORTED_LEVEL,
    );
    for result in map_replicates(n_sim(), |rep| replicate(0x6B31_7001 + rep, n)) {
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
    tally.assert();
}
