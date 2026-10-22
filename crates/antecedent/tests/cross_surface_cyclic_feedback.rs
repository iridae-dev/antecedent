//! Cyclic feedback on the prepared handle, with the chain left on the DAG.
//!
//! A directed cycle is refused by the DAG and solved as an equilibrium. Compilation
//! performs no fit and does not certify a recovered draw. The recovered law stays
//! conditional on the clamp-experiment recovery. Open means `1.25` and `0.625`,
//! and the clamp at `2`, are the equilibrium of the declared coefficients.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::{
    CyclicExecutionRequest, CyclicFamilyAnswer, CyclicFamilyRequest, consume_cyclic_artifact,
    consume_cyclic_family,
};
use antecedent::{AverageEffectQuery, Dag, RefuteSuite, Study};
use antecedent_core::{CausalSchemaBuilder, ExecutionContext};
use antecedent_data::TabularData;
use antecedent_estimate::cyclic_gradient::CyclicGradientPolicy;
use antecedent_graph::GraphError;
use antecedent_io::cyclic_family_artifact::{
    RecoveredCoefficientWire, RecoveredDrawAnswer, RecoveredDrawWire, RecoveredInterventionWire,
    RecoveredSolvedBy,
};
use antecedent_io::cyclic_system_artifact::CyclicArtifactError;
use antecedent_model::compile::{ExecutionBlock, SystemSemantics};
use antecedent_model::nonlinear_equilibrium::{
    Activation, Clamp, Equation, EquilibriumModel, SolvePolicy,
};
use antecedent_test_support::slice_cases;

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(1)
}

fn model() -> EquilibriumModel {
    EquilibriumModel {
        equations: vec![
            Equation {
                id: "a".into(),
                intercept: 0.0,
                weights: vec![0.0, 0.4],
                activation: Activation::Identity,
                expression: None,
                parameters: vec![],
            },
            Equation {
                id: "b".into(),
                intercept: 0.0,
                weights: vec![0.5, 0.0],
                activation: Activation::Identity,
                expression: None,
                parameters: vec![],
            },
        ],
        noise_covariance: vec![1.0, 0.0, 0.0, 1.0],
    }
}

fn draw(into_from: (f64, f64), weight: f64, marked: bool) -> RecoveredDrawWire {
    RecoveredDrawWire {
        coefficients: vec![
            RecoveredCoefficientWire { into: 0, from: 1, value: into_from.0 },
            RecoveredCoefficientWire { into: 1, from: 0, value: into_from.1 },
        ],
        intercept: vec![0.0, 0.0],
        noise_mean: vec![1.0, 0.0],
        noise_covariance: vec![1.0, 0.0, 0.0, 1.0],
        unresolved: Vec::new(),
        weight,
        marked_inadmissible: marked,
    }
}

fn recovered_request() -> CyclicFamilyRequest {
    CyclicFamilyRequest::RecoveredLinear {
        nodes: vec![0, 1],
        draws: vec![
            draw((0.4, 0.5), 0.5, false),
            draw((0.4, 0.5), 0.25, true),
            draw((1.0, 1.0), 0.25, false),
        ],
        worlds: vec![
            RecoveredInterventionWire { clamp_nodes: vec![], clamp_values: vec![] },
            RecoveredInterventionWire { clamp_nodes: vec![0], clamp_values: vec![2.0] },
        ],
        outcome: 1,
        source_id: "clamp-experiments".into(),
        certifying_evidence: vec!["held-out-regime".into()],
    }
}

#[test]
fn a_chain_uses_the_dag_and_a_cycle_is_refused() {
    let gaussian = slice_cases::load().gaussian;
    let schema = CausalSchemaBuilder::new()
        .continuous("t")
        .treatment()
        .continuous("y")
        .outcome()
        .build()
        .unwrap();
    let chain = Dag::from_named_edges(&schema, &[("t", "y")]).unwrap();
    let data = TabularData::try_from_schema_f64(
        schema.clone(),
        [("t", gaussian.treatment.as_slice()), ("y", gaussian.outcome.as_slice())],
    )
    .unwrap();
    let query =
        AverageEffectQuery::binary_ate(schema.id_of("t").unwrap(), schema.id_of("y").unwrap());
    let prepared = Study::tabular(data)
        .graph(chain)
        .query(query)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ctx())
        .unwrap();
    let result = prepared.estimate_retained(&ctx()).unwrap();
    let effect = result.estimate.as_effect().unwrap().ate;
    assert!((effect - gaussian.effect).abs() < 1e-9);

    let cycle = Dag::from_named_edges(&schema, &[("t", "y"), ("y", "t")]).unwrap_err();
    assert!(matches!(cycle, GraphError::Cycle { .. }));
}

#[test]
fn specified_and_recovered_cycles_execute_and_reload() {
    let specified = CyclicExecutionRequest {
        model: model(),
        fit: None,
        worlds: vec![vec![], vec![Clamp { variable: 0, value: 2.0 }]],
        noise: vec![vec![0.0, 0.0], vec![0.0, 0.0]],
        initial: vec![0.0, 0.0],
        policy: SolvePolicy::default(),
        factual: None,
    };
    let prepared = Study::prepare_cyclic(specified, &ctx()).unwrap();
    let inspection = prepared.inspect().unwrap();
    assert_eq!(inspection.fits_performed, 0);
    assert!(
        inspection.execution_plans[0]
            .blocks
            .iter()
            .any(|block| { matches!(block, ExecutionBlock::SimultaneousEquations { .. }) })
    );
    assert!(inspection.solve_blocks[0].iter().any(|block| block.len() == 2));
    let (report, bytes) = prepared.export("equilibrium-intervention", &ctx()).unwrap();
    assert_eq!(prepared.inspect().unwrap().fits_performed, 0);
    assert_eq!(report.claim, "specified_or_fitted_model_conditional");
    assert_eq!(report.calibration, "unmeasured");
    assert!(report.fit.is_none());
    assert!(report.worlds[0][0].values[0].abs() < 1e-8);
    assert!(report.worlds[0][0].values[1].abs() < 1e-8);
    assert!((report.worlds[1][0].values[0] - 2.0).abs() < 1e-8);
    assert!((report.worlds[1][0].values[1] - 1.0).abs() < 1e-8);
    let identity = report.identity.clone();
    drop(prepared);
    let consumed = consume_cyclic_artifact(&bytes, &identity, &ctx()).unwrap();
    assert_eq!(consumed, report);

    let request = recovered_request();
    let wrong = CyclicFamilyRequest::GradientFit {
        model: model(),
        parameters: vec![],
        rows: vec![],
        policy: CyclicGradientPolicy::default(),
    };
    let refused = Study::prepare_recovered_linear(&wrong, &ctx()).unwrap_err();
    assert!(matches!(
        refused,
        CyclicArtifactError::Invalid(message) if message.contains("recovered linear")
    ));
    let CyclicFamilyRequest::RecoveredLinear { source_id, certifying_evidence, .. } = &request
    else {
        panic!("recovered linear request");
    };
    assert_eq!(source_id, "clamp-experiments");
    assert_eq!(certifying_evidence.as_slice(), ["held-out-regime"]);

    let prepared = Study::prepare_recovered_linear(&request, &ctx()).unwrap();
    let inspection = prepared.inspect();
    assert_eq!(inspection.preparation_evaluations, 0);
    assert_eq!(inspection.family, "recovered_linear");
    assert_eq!(inspection.execution_plans[0].semantics, SystemSemantics::EquilibriumScm);
    assert!(
        inspection.execution_plans[0]
            .blocks
            .iter()
            .any(|block| { matches!(block, ExecutionBlock::SimultaneousEquations { .. }) })
    );
    assert_eq!(inspection.execution_plans[1].semantics, SystemSemantics::AcyclicScm);
    let (exported, bytes) = prepared.export(&ctx()).unwrap();
    assert_eq!(prepared.inspect().preparation_evaluations, 0);
    let CyclicFamilyAnswer::RecoveredLinear(answer) = &exported.answer else {
        panic!("recovered linear answer");
    };
    match &answer.worlds[0].draws[0].outcome {
        RecoveredDrawAnswer::Equilibrium {
            solved_by: RecoveredSolvedBy::Cyclic { next_attempt },
            mean,
            ..
        } => {
            assert!(next_attempt.contains("condensation"), "{next_attempt}");
            assert!((mean[0] - 1.25).abs() < 1e-8);
            assert!((mean[1] - 0.625).abs() < 1e-8);
        }
        other => panic!("open draw was {other:?}"),
    }
    match &answer.worlds[1].draws[0].outcome {
        RecoveredDrawAnswer::Equilibrium {
            solved_by: RecoveredSolvedBy::Acyclic, mean, ..
        } => {
            assert!((mean[0] - 2.0).abs() < 1e-8);
            assert!((mean[1] - 1.0).abs() < 1e-8);
        }
        other => panic!("clamped draw was {other:?}"),
    }
    assert!(matches!(
        &answer.worlds[0].draws[1].outcome,
        RecoveredDrawAnswer::Inadmissible { reason } if reason == "marked"
    ));
    assert!(matches!(&answer.worlds[0].draws[2].outcome, RecoveredDrawAnswer::NotCertified { .. }));
    assert!((answer.worlds[0].withheld_mass - 0.5).abs() < 1e-12);
    assert!(answer.worlds[0].mixture_variance.is_some());
    let identity = exported.identity.clone();
    drop(prepared);
    assert_eq!(consume_cyclic_family(&bytes, &identity, &ctx()).unwrap(), exported);
}
