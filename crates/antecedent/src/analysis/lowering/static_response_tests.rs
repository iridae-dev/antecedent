//! Lowering checks for the static response cells.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;
use std::sync::Arc;

use antecedent_core::{
    CausalQuery, ContinuousDomain, DerivativeScale, ExecutionContext, GraphFamily, GridSpec,
    Intervention, ResponseFunctional, ResponseQuery, TargetForm, Value, VariableId,
};
use antecedent_data::{TableView, TabularData};
use antecedent_graph::{Admg, Cpdag, Dag, DenseNodeId, Pag, TieredBackground, WithinTier};

use crate::analysis::lowering::registry::REGISTERED;
use crate::planner::lowered::{
    FunctionalKind, LoweredProgram, LoweringDepth, ProblemTarget, UncertaintyReading, WidthSource,
};
use crate::planner::statistical::{ModelRole, Realization};
use crate::{BayesianConfig, EstimatorId, InferenceMode, RefuteSuite, Study};

const T: VariableId = VariableId::from_raw(0);
const Y: VariableId = VariableId::from_raw(1);

fn d(raw: u32) -> DenseNodeId {
    DenseNodeId::from_raw(raw)
}

/// Columns `t, y, z`: `z` confounds `t` and `y`.
fn data() -> TabularData {
    let z: Vec<f64> = (0..240_u32).map(|i| (f64::from(i) * 0.37).sin()).collect();
    let t: Vec<f64> =
        z.iter().zip(0_u32..).map(|(z, i)| 0.8 * z + (f64::from(i) * 1.13).cos()).collect();
    let y: Vec<f64> = t
        .iter()
        .zip(&z)
        .zip(0_u32..)
        .map(|((t, z), i)| 0.4 + 1.5 * t + 0.6 * z + 0.05 * (f64::from(i) * 0.71).sin())
        .collect();
    TabularData::from_f64_columns([("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())])
        .unwrap()
}

fn dag(data: &TabularData) -> Dag {
    Dag::from_named_edges(data.schema(), &[("z", "t"), ("z", "y"), ("t", "y")]).unwrap()
}

fn mean_curve(grid: &[f64]) -> ResponseQuery {
    ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: Y,
        treatment: ContinuousDomain::new(T, GridSpec::Values(Arc::from(grid))),
    })
}

fn intervention() -> ResponseQuery {
    ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: Y,
        interventions: Arc::from([Intervention::set(T, Value::f64(0.5))]),
    })
}

fn dag_study(query: ResponseQuery, inference: InferenceMode) -> Study {
    let data = data();
    let graph = dag(&data);
    Study::tabular(data)
        .graph(graph)
        .query(CausalQuery::Response(query))
        .inference(inference)
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
}

fn lowered(study: &Study) -> LoweredProgram {
    let plan = study.compile(&ExecutionContext::for_tests(7)).unwrap();
    crate::planner::plan_view::checks::assert_view_agrees("static response", &plan);
    plan.lowered().cloned().expect("compile attaches a lowered program")
}

fn bayesian() -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(48))
}

#[test]
fn mean_curve_lowers_one_world_per_grid_level_and_its_cross_fit() {
    let study = dag_study(mean_curve(&[-0.5, 0.0, 0.5]), InferenceMode::Frequentist);
    let program = lowered(&study);
    assert_eq!(program.route(), "StaticResponseDag");
    assert_eq!(program.depth(), LoweringDepth::Complete);
    assert!(program.identification_retained(), "compile retains the response identification");
    let ProblemTarget::Target(target) = program.scientific().target() else {
        panic!("a mean curve lowers a world-indexed target");
    };
    let TargetForm::Worlds(worlds) = target.form() else { panic!("one world per grid level") };
    assert_eq!(worlds.len(), 3);
    assert_eq!(target.assumptions().graph(), GraphFamily::Dag);
    let statistical = program.statistical();
    assert_eq!(statistical.estimator(), Some(EstimatorId::ResponseKennedyDr));
    assert!(matches!(statistical.realization(), Realization::Frequentist(_)));
    let options = antecedent_estimate::ContinuousResponseOptions::default();
    assert_eq!(statistical.folds().map(|folds| folds.folds()), Some(options.folds));
    assert!(statistical.resampling().is_none(), "the curve's uncertainty is its influence");
    let roles: Vec<_> = statistical.models().iter().map(|model| model.role()).collect();
    assert_eq!(roles, [ModelRole::Outcome, ModelRole::Propensity]);
    assert_eq!(statistical.models()[0].output(), Some(Y));
    let design = program.physical().design();
    assert!(matches!(design.source(), WidthSource::NotDesignBased(_)));
    assert_eq!((design.rows(), design.columns()), (None, None), "no design is fabricated");
    assert_eq!(program.claim().kind(), Some(antecedent_contract::ClaimKind::Response));
    assert_eq!(program.claim().uncertainty(), UncertaintyReading::Sampling);

    assert_eq!(lowered(&study).identity(), program.identity(), "the identity is stable");
    let regridded = dag_study(mean_curve(&[-0.5, 0.5]), InferenceMode::Frequentist);
    assert_ne!(lowered(&regridded).identity(), program.identity(), "the target changed");
    let posterior = lowered(&dag_study(mean_curve(&[-0.5, 0.0, 0.5]), bayesian()));
    assert_eq!(posterior.statistical().estimator(), Some(EstimatorId::ResponseBayesian));
    assert!(matches!(posterior.statistical().realization(), Realization::Bayesian(_)));
    assert_eq!(posterior.statistical().posterior().map(|plan| plan.draws()), Some(48));
    assert_ne!(posterior.identity(), program.identity(), "the inference changed");
}

#[test]
fn derivative_lowers_the_response_derivative_functional() {
    let query = ResponseQuery::new(ResponseFunctional::PointDerivative {
        outcome: Y,
        treatment: T,
        at: 0.0,
        order: 1,
        scale: DerivativeScale::Identity,
    });
    let program = lowered(&dag_study(query, InferenceMode::Frequentist));
    assert_eq!(program.depth(), LoweringDepth::Complete);
    assert!(matches!(
        program.scientific().target(),
        ProblemTarget::QueryFunctional {
            kind: FunctionalKind::ResponseDerivative,
            outcome: Some(Y),
            ..
        }
    ));
}

#[test]
fn intervention_response_prepare_and_estimate_stamp_the_compiled_identity() {
    std::thread::Builder::new()
        .name("static-response-lowering".into())
        .stack_size(16 * 1024 * 1024)
        .spawn(|| {
            let ctx = ExecutionContext::for_tests(11);
            let study = dag_study(intervention(), InferenceMode::Frequentist);
            let program = lowered(&study);
            let ProblemTarget::Target(target) = program.scientific().target() else {
                panic!("an intervention response lowers its world");
            };
            assert!(matches!(target.form(), TargetForm::World(_)));
            assert_eq!(
                program.statistical().estimator(),
                Some(EstimatorId::ResponseInterventionGcomp)
            );
            let prepared = study.prepare(&ctx).unwrap();
            assert_eq!(prepared.plan().lowered_program_identity(), Some(program.identity()));
            let result = prepared.estimate(&data(), &ctx).unwrap();
            assert_eq!(result.lowered_program_identity(), Some(program.identity()));
        })
        .unwrap()
        .join()
        .unwrap();
}

/// Columns `z, t1, t2, y` on a balanced binary design.
fn codetermined_study() -> Study {
    let z: Vec<f64> = (0..480_u32).map(|i| f64::from(i % 2)).collect();
    let t1: Vec<f64> = (0..480_u32).map(|i| f64::from(i / 2 % 2)).collect();
    let t2: Vec<f64> = (0..480_u32).map(|i| f64::from(i / 4 % 2)).collect();
    let y: Vec<f64> = (0..480_usize)
        .map(|i| 0.2 + 0.4 * t1[i] + 0.3 * t2[i] + 0.5 * t1[i] * t2[i] + 0.2 * z[i])
        .collect();
    let data = TabularData::from_f64_columns([
        ("z", z.as_slice()),
        ("t1", t1.as_slice()),
        ("t2", t2.as_slice()),
        ("y", y.as_slice()),
    ])
    .unwrap();
    let background = TieredBackground::from_named(
        data.schema(),
        &[vec!["z"], vec!["t1", "t2"], vec!["y"]],
        WithinTier::CoDetermined,
    )
    .unwrap();
    let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(3),
        interventions: Arc::from([
            Intervention::set(VariableId::from_raw(1), Value::f64(1.0)),
            Intervention::set(VariableId::from_raw(2), Value::f64(1.0)),
        ]),
    });
    Study::tabular(data)
        .tiered_background(background)
        .unwrap()
        .query(CausalQuery::Response(query))
        .estimator(EstimatorId::CellAipw)
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
}

/// `t -> m -> y` with `t <-> y`: the front-door response.
fn admg_study() -> Study {
    let t: Vec<f64> = (0..240_u32).map(|i| f64::from(i % 2)).collect();
    let m: Vec<f64> = (0..240_u32).map(|i| f64::from((i / 2 + i / 3) % 2)).collect();
    let y: Vec<f64> = t.iter().zip(&m).map(|(t, m)| 0.3 + 0.5 * m + 0.2 * t).collect();
    let data = TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("m", m.as_slice()),
        ("y", y.as_slice()),
    ])
    .unwrap();
    let mut graph = Admg::with_variables(3);
    graph.insert_directed(d(0), d(1)).unwrap();
    graph.insert_directed(d(1), d(2)).unwrap();
    graph.insert_bidirected(d(0), d(2)).unwrap();
    let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: VariableId::from_raw(2),
        interventions: Arc::from([Intervention::set(T, Value::f64(1.0))]),
    });
    Study::tabular(data)
        .graph(graph)
        .query(CausalQuery::Response(query))
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
}

fn class_study(graph_class: &str) -> Study {
    let builder = Study::tabular(data());
    let builder = if graph_class == "pag" {
        // `z -> t -> y`: one visible completion.
        let mut graph = Pag::with_variables(3);
        graph.insert_directed(d(2), d(0)).unwrap();
        graph.insert_directed(d(0), d(1)).unwrap();
        builder.graph(graph)
    } else {
        // `z — t`, `z -> y`, `t -> y`: two completions.
        let mut graph = Cpdag::with_variables(3);
        graph.insert_directed(d(2), d(1)).unwrap();
        graph.insert_directed(d(0), d(1)).unwrap();
        graph.insert_undirected(d(2), d(0)).unwrap();
        builder.graph(graph)
    };
    builder
        .query(CausalQuery::Response(mean_curve(&[-0.5, 0.0, 0.5])))
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
}

#[test]
fn every_static_response_cell_lowers_completely() {
    let mut covered = BTreeSet::new();
    let studies = [
        dag_study(mean_curve(&[-0.5, 0.0, 0.5]), InferenceMode::Frequentist),
        class_study("pag"),
        class_study("cpdag"),
        codetermined_study(),
        admg_study(),
    ];
    for study in &studies {
        let program = lowered(study);
        assert_eq!(program.depth(), LoweringDepth::Complete, "{}", program.route());
        assert_eq!(lowered(study).identity(), program.identity(), "{}", program.route());
        covered.insert(program.route());
        match program.route() {
            "StaticResponsePag" | "StaticResponseCpdag" => {
                assert_eq!(program.claim().uncertainty(), UncertaintyReading::Envelope);
            }
            "CodeterminedJoint" => {
                let ProblemTarget::Target(target) = program.scientific().target() else {
                    panic!("the joint cell lowers its world");
                };
                assert_eq!(target.assumptions().graph(), GraphFamily::CoDetermined);
                let statistical = program.statistical();
                assert_eq!(statistical.estimator(), Some(EstimatorId::CellAipw));
                assert_eq!(
                    statistical.folds().map(|folds| folds.folds()),
                    Some(antecedent_estimate::CellSaturatedAipw::new().folds)
                );
            }
            _ => {}
        }
    }
    let family: BTreeSet<_> = REGISTERED
        .iter()
        .filter(|(_, family)| *family == "static_response")
        .map(|(cell, _)| cell.as_str())
        .collect();
    assert_eq!(covered, family, "a static response cell has no complete lowering");
}
