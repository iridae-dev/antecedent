//! Prospective coverage of a fresh treated target outcome. Each replicate
//! reruns source fitting, target-frame predictive sampling and paired utility
//! through one compiled program. The source and target mechanisms are equal
//! by the stated shared-baseline premise. This ignored harness has earned no
//! predictive coverage or decision-reliability record.

mod common;

use std::sync::Arc;

use antecedent::analysis::joint_transport_predictive_decision::{
    compile_joint_transport_predictive_decision, JointTransportPredictiveDecisionRequest,
};
use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_core::{
    CalibrationBasis, ContinuousDomain, ExecutionContext, GridSpec,
    ResponseFunctional, ResponseQuery, TransportQuery, VariableId,
};
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion,
    StructuralPolicy, UtilityExpr,
};
use antecedent_estimate::joint_bayesian_transport::{
    DataIdentity, GaussianPrior, JointPriors, JointTransportModel,
    JointTransportOptions, PriorProvenance, SourceData, SourceDependence,
    SourceSharing, TargetData, TransportGraphClass, VaryingBlock,
};
use antecedent_graph::{Admg, SelectionDiagram};
use common::calibration::{
    Construction, CoverageTally, REPORTED_LEVEL, RecordKey, ScopeFacts,
    gaussian, grid_n, grid_seed, map_replicates, n_sim, uniform,
};

const TEST: &str = "joint_transport_predictive_treated_outcome_l95";
const DGP: &str = "shared_source_target_gaussian_baseline";
const INTERVAL: &str = "nearest_rank_95";
const NOISE_VARIANCE: f64 = 0.1;

fn world(value: u8) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: "2".into(), variable_name: "target outcome".into(),
        role: QuantityRole::Outcome, units: "score".into(),
        population_id: "target".into(), regime_id: format!("do(0={value})"),
        horizon: 0, functional_id: "outcome".into(),
        conditioning: Vec::new(), transform_id: "identity".into(),
    }
}

fn replicate(seed: u64, n: usize) -> Option<((f64, f64), f64, CalibrationBasis)> {
    let mut assignment = uniform(seed, 0xA01C_F101);
    let mut source_noise = gaussian(seed ^ 0xA01C_F102);
    let mut future_noise = gaussian(seed ^ 0xA01C_F103);
    let treatment = VariableId::from_raw(0);
    let outcome = VariableId::from_raw(2);
    let mut graph = Admg::with_variables(3);
    let from = graph.dense_of_static(treatment)?;
    let to = graph.dense_of_static(outcome)?;
    graph.insert_directed(from, to).ok()?;
    let diagram = SelectionDiagram::try_new(graph, [VariableId::from_raw(1)]).ok()?;
    let query = TransportQuery::new(
        ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome,
            treatment: ContinuousDomain::new(treatment,
                GridSpec::Values(Arc::from([0.0, 1.0]))),
        }),
        "trial", "target", [treatment],
    );
    let worlds = [world(0), world(1)];
    let decision = DecisionContract {
        actions: vec![
            DecisionAction {
                id: "hold".into(), kind: ActionKind::Intervention,
                inputs: vec![worlds[0].clone()], utility: UtilityExpr::Input(0),
            },
            DecisionAction {
                id: "treat".into(), kind: ActionKind::Intervention,
                inputs: vec![worlds[1].clone()],
                utility: UtilityExpr::difference(UtilityExpr::Input(0), UtilityExpr::Const(1.0)),
            },
        ],
        utility_units: "score".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: Vec::new(), target_population: "target".into(), horizon: 0,
        structural_policy: StructuralPolicy::RequireInvariantBestAction,
    };
    let source_treatment: Vec<bool> = (0..n).map(|_| assignment() < 0.5).collect();
    let source_outcome: Vec<f64> = source_treatment.iter()
        .map(|treated| 3.0 + (if *treated { 2.0 } else { 0.0 })
            + NOISE_VARIANCE.sqrt() * source_noise())
        .collect();
    let source = SourceData {
        id: "trial-a".into(),
        identity: DataIdentity {
            snapshot_digest: format!("source-{seed}"),
            datum_ids: (0..n).map(|row| format!("source-{row}")).collect(),
        },
        treatment: source_treatment, outcome: source_outcome,
        covariates: Vec::new(), noise_variance: NOISE_VARIANCE,
    };
    let target = TargetData {
        identity: DataIdentity {
            snapshot_digest: format!("target-{seed}"),
            datum_ids: (0..n).map(|row| format!("target-{row}")).collect(),
        },
        rows: n, covariates: Vec::new(),
    };
    let model = JointTransportModel {
        graph: TransportGraphClass::FixedDag, features: Vec::new(),
        varying: VaryingBlock::Intercept, sharing: SourceSharing::SharedVaryingBlock,
        dependence: SourceDependence::IndependentSamples,
        priors: JointPriors {
            invariant: GaussianPrior::isotropic(1, 0.0, 10.0, PriorProvenance::Declared),
            varying: GaussianPrior::isotropic(1, 0.0, 10.0, PriorProvenance::Declared),
        },
        max_unsupported_mass: 0.0, conflict_z_threshold: 3.0,
    };
    let ctx = ExecutionContext::for_tests(grid_seed(seed));
    let plan = compile_joint_transport_predictive_decision(JointTransportPredictiveDecisionRequest {
        diagram, query, model, sources: vec![source], target,
        options: JointTransportOptions { draws: 1024, seed: grid_seed(seed) },
        target_noise_variance: NOISE_VARIANCE, worlds, decision,
    }).ok()?;
    let candidate = plan.calibration_candidate(&ctx).ok()?;
    assert_eq!(candidate.result.calibration, "not_measured");
    assert_eq!(candidate.basis.interval_method.as_ref(), INTERVAL);
    let future_treated_outcome = 5.0 + NOISE_VARIANCE.sqrt() * future_noise();
    Some((candidate.interval, future_treated_outcome, candidate.basis))
}

#[test]
#[ignore = "source-only prospective predictive coverage; no record earned"]
fn joint_transport_predictive_treated_outcome_l95() {
    let n = grid_n(120);
    let mut tally = CoverageTally::for_record(RecordKey {
        test: TEST, dgp: DGP, interval: INTERVAL,
    }, REPORTED_LEVEL);
    for result in map_replicates(n_sim(), |rep| replicate(0xD0C0_6001 + rep, n)) {
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
