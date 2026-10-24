//! Lowering checks for the static mediation and structural-model cells.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

use antecedent_core::{
    AnomalyAttributionQuery, CausalQuery, ChangeAttributionQuery, CounterfactualQuery,
    ExecutionContext, Intervention, Lag, MechanismChangeQuery, MediationContrast, MediationQuery,
    PopulationSelector, TargetForm, UnitChangeQuery, Value, VariableId,
};
use antecedent_data::{TabularData, TimeSeriesData};
use antecedent_graph::{Cpdag, Dag, DenseNodeId, TemporalCpdag, TemporalDag, ensure_lagged};

use crate::analysis::lowering::registry::REGISTERED;
use crate::planner::lowered::{
    FunctionalKind, LoweredProgram, LoweringDepth, ProblemTarget, UncertaintyReading,
};
use crate::planner::statistical::{ModelRole, Realization};
use crate::{BayesianConfig, EstimatorId, InferenceMode, RefuteSuite, Study};

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn lowered(study: &Study) -> LoweredProgram {
    let plan = study.compile(&ExecutionContext::for_tests(5)).unwrap();
    crate::planner::plan_view::checks::assert_view_agrees("structural", &plan);
    plan.lowered().cloned().expect("compile attaches a lowered program")
}

fn on_large_stack(name: &str, body: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .name(name.into())
        .stack_size(16 * 1024 * 1024)
        .spawn(body)
        .unwrap()
        .join()
        .unwrap();
}

/// Columns `a, m, y, c`: `c` confounds every pair, `a -> m -> y` and `a -> y`.
fn mediation_data() -> TabularData {
    let c: Vec<f64> = (0..320_u32).map(|i| (f64::from(i) * 0.37).sin()).collect();
    let a: Vec<f64> =
        c.iter().zip(0_u32..).map(|(c, i)| 0.5 * c + (f64::from(i) * 0.71).cos()).collect();
    let m: Vec<f64> = a
        .iter()
        .zip(&c)
        .zip(0_u32..)
        .map(|((a, c), i)| 0.6 * a + 0.3 * c + 0.2 * (f64::from(i) * 1.13).sin())
        .collect();
    let y: Vec<f64> = a
        .iter()
        .zip(&m)
        .zip(&c)
        .zip(0_u32..)
        .map(|(((a, m), c), i)| 0.4 * a + 0.5 * m + 0.2 * c + 0.1 * (f64::from(i) * 0.53).cos())
        .collect();
    TabularData::from_f64_columns([
        ("a", a.as_slice()),
        ("m", m.as_slice()),
        ("y", y.as_slice()),
        ("c", c.as_slice()),
    ])
    .unwrap()
}

fn mediation_graph() -> Dag {
    let mut graph = Dag::with_variables(4);
    for (from, to) in [(3, 0), (3, 1), (3, 2), (0, 1), (0, 2), (1, 2)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    graph
}

fn mediation_study(inference: InferenceMode, replicates: u32) -> Study {
    let mut query = MediationQuery::binary(v(0), v(2), [v(1)], MediationContrast::NaturalDirect);
    query.control = Intervention::set(v(0), Value::f64(0.2));
    query.active = Intervention::set(v(0), Value::f64(0.8));
    Study::tabular(mediation_data())
        .graph(mediation_graph())
        .query(CausalQuery::Mediation(query))
        .inference(inference)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(replicates)
        .build()
        .unwrap()
}

#[test]
fn mediation_lowers_the_theorem_target_the_executor_checks() {
    on_large_stack("static-mediation-lowering", || {
        let ctx = ExecutionContext::for_tests(17);
        let study = mediation_study(InferenceMode::Frequentist, 0);
        let program = lowered(&study);
        assert_eq!(program.route(), "StaticMediationDag");
        assert_eq!(program.depth(), LoweringDepth::Complete);
        assert!(program.identification_retained());
        let ProblemTarget::Target(target) = program.scientific().target() else {
            panic!("mediation lowers its natural-effect target");
        };
        assert!(matches!(target.form(), TargetForm::Mediation(_)));
        let statistical = program.statistical();
        assert_eq!(statistical.estimator(), Some(EstimatorId::StaticMediationLinear));
        assert!(matches!(statistical.realization(), Realization::Frequentist(_)));
        assert!(statistical.resampling().is_none());
        let roles: Vec<_> =
            statistical.models().iter().map(crate::planner::statistical::ModelSlot::role).collect();
        assert_eq!(roles, [ModelRole::Mediator, ModelRole::Outcome]);
        assert_eq!(statistical.models()[1].output(), Some(v(2)));

        let prepared = study.prepare(&ctx).unwrap();
        assert_eq!(prepared.plan().lowered_program_identity(), Some(program.identity()));
        let crate::analysis::RetainedOperation::StaticMediation(info) =
            prepared.retained_operation()
        else {
            panic!("sealed mediation")
        };
        let view = prepared.plan_view().unwrap();
        assert_eq!(view.identity(), program.identity(), "the prepared view reads the same program");
        assert_eq!(view.operation(), Some("static_mediation"));
        assert_eq!(info.scientific_target.identity(), target.identity());
        let result = prepared.estimate(&mediation_data(), &ctx).unwrap();
        assert_eq!(result.lowered_program_identity(), Some(program.identity()));

        assert_eq!(lowered(&study).identity(), program.identity(), "the identity is stable");
        let resampled = lowered(&mediation_study(InferenceMode::Frequentist, 25));
        assert_eq!(resampled.statistical().resampling().map(|plan| plan.replicates()), Some(25));
        assert_ne!(resampled.identity(), program.identity(), "the resampling changed");
        let posterior = lowered(&mediation_study(
            InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(32)),
            25,
        ));
        assert!(matches!(posterior.statistical().realization(), Realization::Bayesian(_)));
        assert!(posterior.statistical().resampling().is_none());
        assert_ne!(posterior.identity(), program.identity(), "the inference changed");
    });
}

fn xy(y_of: impl Fn(u32, f64) -> f64, rows: u32) -> (Vec<f64>, Vec<f64>) {
    let x: Vec<f64> = (0..rows).map(|i| f64::from(i % 40) * 0.1).collect();
    let y: Vec<f64> = x.iter().zip(0_u32..).map(|(x, i)| y_of(i, *x)).collect();
    (x, y)
}

fn tabular(x: &[f64], y: &[f64]) -> TabularData {
    TabularData::from_f64_columns([("x", x), ("y", y)]).unwrap()
}

fn chain() -> Dag {
    let mut graph = Dag::with_variables(2);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    graph
}

fn gcm_study(data: TabularData, query: CausalQuery, inference: InferenceMode) -> Study {
    Study::tabular(data)
        .graph(chain())
        .query(query)
        .inference(inference)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
}

fn shifted() -> TabularData {
    let (x, y) = xy(|i, x| if i < 40 { 1.0 + 2.0 * x } else { 6.0 + 2.0 * x }, 80);
    tabular(&x, &y)
}

fn first_half() -> PopulationSelector {
    PopulationSelector::TimeRange { start: 0, end: 40 }
}

fn second_half() -> PopulationSelector {
    PopulationSelector::TimeRange { start: 40, end: 80 }
}

fn counterfactual() -> CausalQuery {
    CausalQuery::Counterfactual(CounterfactualQuery::new(
        v(1),
        [Intervention::set(v(0), Value::f64(1.0))],
    ))
}

#[test]
fn counterfactual_lowers_its_functional_mechanisms_and_row_weight_posterior() {
    let frequentist = lowered(&gcm_study(shifted(), counterfactual(), InferenceMode::Frequentist));
    assert_eq!(frequentist.route(), "GcmDag");
    assert_eq!(frequentist.depth(), LoweringDepth::Complete);
    assert!(matches!(
        frequentist.scientific().target(),
        ProblemTarget::QueryFunctional {
            kind: FunctionalKind::Counterfactual,
            outcome: Some(y),
            ..
        }
            if *y == v(1)
    ));
    let statistical = frequentist.statistical();
    assert!(matches!(statistical.realization(), Realization::ExactPoint { .. }));
    assert_eq!(frequentist.claim().uncertainty(), UncertaintyReading::None);
    let mechanisms: Vec<_> = statistical
        .models()
        .iter()
        .map(|model| (model.role(), model.inputs().to_vec(), model.output()))
        .collect();
    assert_eq!(
        mechanisms,
        [
            (ModelRole::Mechanism, vec![], Some(v(0))),
            (ModelRole::Mechanism, vec![v(0)], Some(v(1)))
        ]
    );
    assert_eq!(statistical.models()[0].family(), "gcm.registry.heterogeneity.best_score");

    let bayesian = lowered(&gcm_study(
        shifted(),
        counterfactual(),
        InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(48)),
    ));
    let posterior = bayesian.statistical().posterior().expect("row-weight posterior");
    assert_eq!(posterior.draws(), 48);
    assert_eq!(posterior.prior().name(), "dirichlet_row_weights(1)");
    assert!(matches!(bayesian.statistical().realization(), Realization::Bayesian(_)));
    assert_eq!(bayesian.claim().uncertainty(), UncertaintyReading::Posterior);
    assert_ne!(bayesian.identity(), frequentist.identity());
}

/// A fitted-model attribution lowers to the Dirichlet row-weight posterior over the mechanisms
/// under Bayesian inference, whatever its target, and stays a point answer under Frequentist.
#[test]
fn a_fitted_model_attribution_lowers_its_posterior_over_the_mechanisms() {
    let change = |rooted: bool| {
        let change = ChangeAttributionQuery::new(v(1), first_half(), second_half());
        let estimator = antecedent_core::ChangeEstimator::Sampled {
            measure: antecedent_core::ChangeMeasure::MeanDifference,
            n_samples: 100,
            seed: 1,
        };
        CausalQuery::Attribute(antecedent_core::AttributeQuery::new(if rooted {
            antecedent_core::AttributeTarget::RootCauses { change, estimator }
        } else {
            antecedent_core::AttributeTarget::DistributionChange { change, estimator }
        }))
    };
    let paths = CausalQuery::Attribute(antecedent_core::AttributeQuery::new(
        antecedent_core::AttributeTarget::Paths {
            sources: std::sync::Arc::from([v(0)]),
            outcome: v(1),
            max_paths: 8,
            max_len: 4,
            through: std::sync::Arc::from([]),
        },
    ));
    let bayesian = |draws| InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(draws));
    for rooted in [false, true] {
        let point = lowered(&gcm_study(shifted(), change(rooted), InferenceMode::Frequentist));
        assert_eq!(point.claim().uncertainty(), UncertaintyReading::None);
        let program = lowered(&gcm_study(shifted(), change(rooted), bayesian(32)));
        let posterior = program.statistical().posterior().expect("row-weight posterior");
        assert_eq!(posterior.draws(), 32);
        assert_eq!(posterior.prior().name(), "dirichlet_row_weights(1)");
        assert!(matches!(program.statistical().realization(), Realization::Bayesian(_)));
        assert_eq!(program.claim().uncertainty(), UncertaintyReading::Posterior);
        assert_ne!(program.identity(), point.identity());
    }
    let program = lowered(&gcm_study(shifted(), paths, bayesian(32)));
    let posterior = program.statistical().posterior().expect("paths have a posterior too");
    assert_eq!(posterior.draws(), 32);
    assert_eq!(program.claim().uncertainty(), UncertaintyReading::Posterior);
}

#[test]
fn anomaly_prepare_and_estimate_stamp_the_compiled_identity() {
    on_large_stack("anomaly-lowering", || {
        let ctx = ExecutionContext::for_tests(13);
        let (x, y) = xy(|i, x| if i == 19 { 200.0 } else { 2.0 * x }, 40);
        let data = tabular(&x, &y);
        let query = CausalQuery::AnomalyAttribution(AnomalyAttributionQuery::new([v(1)], 100));
        let study = gcm_study(data.clone(), query, InferenceMode::Frequentist);
        let program = lowered(&study);
        assert!(matches!(
            program.scientific().target(),
            ProblemTarget::QueryFunctional { kind: FunctionalKind::AnomalyAttribution, .. }
        ));
        assert_eq!(program.statistical().models()[1].family(), "gcm.registry.standard.best_score");
        let prepared = study.prepare(&ctx).unwrap();
        assert_eq!(prepared.plan().lowered_program_identity(), Some(program.identity()));
        let result = prepared.estimate(&data, &ctx).unwrap();
        assert_eq!(result.lowered_program_identity(), Some(program.identity()));
    });
}

fn temporal_study() -> Study {
    let mut graph = TemporalDag::empty();
    let parent = ensure_lagged(&mut graph, v(0), Lag::from_raw(1)).unwrap();
    let child = ensure_lagged(&mut graph, v(1), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(parent, child).unwrap();
    let (x, y) = xy(|i, x| if i == 19 { 200.0 } else { 2.0 * x }, 40);
    let series =
        TimeSeriesData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())], 1).unwrap();
    Study::series(series)
        .graph(graph)
        .query(CausalQuery::AnomalyAttribution(AnomalyAttributionQuery::new([v(1)], 100)))
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
}

#[test]
fn every_structural_cell_lowers_completely() {
    let kinds = [
        (counterfactual(), FunctionalKind::Counterfactual),
        (
            CausalQuery::ChangeAttribution(ChangeAttributionQuery::new(
                v(1),
                first_half(),
                second_half(),
            )),
            FunctionalKind::ChangeAttribution,
        ),
        (
            CausalQuery::MechanismChange(MechanismChangeQuery::new(
                [v(0), v(1)],
                first_half(),
                second_half(),
                0.05,
                10,
            )),
            FunctionalKind::MechanismChange,
        ),
        (CausalQuery::UnitChange(UnitChangeQuery::new(v(1), 20)), FunctionalKind::UnitChange),
        (
            CausalQuery::Attribute(antecedent_core::AttributeQuery::new(
                antecedent_core::AttributeTarget::Paths {
                    sources: std::sync::Arc::from([v(0)]),
                    outcome: v(1),
                    max_paths: 8,
                    max_len: 4,
                    through: std::sync::Arc::from([]),
                },
            )),
            FunctionalKind::Attribution,
        ),
        (
            CausalQuery::Simulate(antecedent_core::SimulateQuery::new(
                antecedent_core::SimulationTarget::PolicyValue {
                    interventions: std::sync::Arc::from([Intervention::set(v(0), Value::f64(1.0))]),
                    outcome: v(1),
                    draws: 64,
                },
            )),
            FunctionalKind::Simulation,
        ),
    ];
    let mut covered = BTreeSet::new();
    for (query, expected) in kinds {
        let program = lowered(&gcm_study(shifted(), query, InferenceMode::Frequentist));
        assert_eq!(program.depth(), LoweringDepth::Complete);
        assert!(matches!(
            program.scientific().target(),
            ProblemTarget::QueryFunctional { kind, .. } if *kind == expected
        ));
        covered.insert(program.route());
    }
    let temporal = lowered(&temporal_study());
    assert_eq!(temporal.route(), "GcmTemporal");
    assert_eq!(temporal.depth(), LoweringDepth::Complete);
    let mechanism = temporal
        .statistical()
        .models()
        .iter()
        .find(|model| model.output() == Some(v(1)))
        .expect("the contemporaneous outcome has a mechanism");
    assert_eq!(mechanism.inputs(), &[v(0)], "the lagged parent is read by its variable");
    covered.insert(temporal.route());
    // The class cells lower the attribution their completions share.
    let mut undirected = Cpdag::with_variables(2);
    undirected.insert_undirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let class = lowered(
        &Study::tabular(shifted())
            .graph(undirected)
            .query(CausalQuery::AnomalyAttribution(AnomalyAttributionQuery::new([v(1)], 100)))
            .inference(InferenceMode::Frequentist)
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap(),
    );
    assert_eq!(class.route(), "GcmClass");
    assert_eq!(class.depth(), LoweringDepth::Complete);
    covered.insert(class.route());
    let mut lagged = TemporalCpdag::empty();
    let parent = lagged.add_lagged(v(0), Lag::from_raw(1)).unwrap();
    let child = lagged.add_lagged(v(1), Lag::CONTEMPORANEOUS).unwrap();
    lagged.insert_directed(parent, child).unwrap();
    let (x, y) = xy(|i, x| if i == 19 { 200.0 } else { 2.0 * x }, 40);
    let series =
        TimeSeriesData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())], 1).unwrap();
    let temporal_class = lowered(
        &Study::series(series)
            .graph(lagged)
            .query(CausalQuery::AnomalyAttribution(AnomalyAttributionQuery::new([v(1)], 100)))
            .inference(InferenceMode::Frequentist)
            .refute(RefuteSuite::None)
            .build()
            .unwrap(),
    );
    assert_eq!(temporal_class.route(), "GcmTemporalClass");
    assert_eq!(temporal_class.depth(), LoweringDepth::Complete);
    covered.insert(temporal_class.route());
    let mediation = lowered(&mediation_study(InferenceMode::Frequentist, 0));
    assert_eq!(mediation.depth(), LoweringDepth::Complete);
    covered.insert(mediation.route());
    let family: BTreeSet<_> = REGISTERED
        .iter()
        .filter(|(_, family)| *family == "structural")
        .map(|(cell, _)| cell.as_str())
        .collect();
    assert_eq!(covered, family, "a structural cell has no complete lowering");
}
