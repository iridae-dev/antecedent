//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    CausalSchemaBuilder, ContinuousDomain, ExecutionContext, GridSpec, MeasurementSpec,
    ResponseFunctional, ResponseQuery, RoleHint, SmallRoleSet, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, TableView, TabularData, ValidityBitmap,
};
use antecedent_graph::Dag;

use crate::analysis::execute::Study;

fn data(shift: f64) -> TabularData {
    let mut schema = CausalSchemaBuilder::new();
    for (name, hint) in [
        ("x", RoleHint::Context),
        ("a", RoleHint::TreatmentCandidate),
        ("y", RoleHint::OutcomeCandidate),
    ] {
        schema
            .add_variable(
                name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(hint),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
    }
    let schema = schema.build().unwrap();
    let n = 120;
    let x: Vec<_> = (0..n).map(|i| (i as f64 * 0.13).sin()).collect();
    let a: Vec<_> = (0..n).map(|i| 0.6 * x[i] + (i as f64 * 0.31).cos()).collect();
    let y: Vec<_> =
        (0..n).map(|i| shift + 1.7 * a[i] + 0.8 * x[i] + (i as f64 * 0.23).sin() * 0.1).collect();
    let cols = [x, a, y]
        .into_iter()
        .enumerate()
        .map(|(i, values)| {
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(u32::try_from(i).unwrap()),
                    Arc::from(values),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            )
        })
        .collect();
    TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap())
}

fn study(data: TabularData, grid: &[f64]) -> Study {
    let graph =
        Dag::from_named_edges(data.schema(), &[("x", "a"), ("x", "y"), ("a", "y")]).unwrap();
    let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(2),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(1),
            GridSpec::Values(Arc::from(grid)),
        ),
    });
    Study::tabular(data)
        .graph(graph)
        .query(query)
        .bootstrap_replicates(0)
        .refute(crate::analysis::builder::RefuteSuite::None)
        .build()
        .unwrap()
}

#[test]
fn prepared_curve_refresh_matches_independent_numeric_estimate() {
    let grid = [-0.5, 0.0, 0.5];
    let context = ExecutionContext::for_tests(17);
    let mut prepared = study(data(0.0), &grid).prepare(&context).unwrap();
    assert!(matches!(
        prepared.execution,
        crate::analysis::prepared::PreparedExecution::StaticDagResponse(_)
    ));
    let refreshed = data(0.4);
    let actual = prepared.refresh(refreshed.clone(), &context).unwrap().into_response().unwrap();
    let expected = study(refreshed, &grid).run(&context).unwrap().into_response().unwrap();
    assert_eq!(actual, expected);
    let antecedent_core::ResponseIdentification::PointIdentified(
        antecedent_core::ResponseValue::Surface { mean, .. },
    ) = actual.estimate
    else {
        panic!("expected a point-identified curve")
    };
    for (&dose, &estimate) in grid.iter().zip(mean.iter()) {
        let truth = 0.4 + 1.7 * dose;
        assert!((estimate - truth).abs() < 0.15, "dose={dose}: estimate={estimate}, truth={truth}");
    }
}

#[test]
fn prepared_curve_refuses_a_tampered_grid() {
    let context = ExecutionContext::for_tests(17);
    let mut prepared = study(data(0.0), &[-0.5, 0.0, 0.5]).prepare(&context).unwrap();
    let changed = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(2),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(1),
            GridSpec::Values(Arc::from([-0.5, 0.25, 0.5])),
        ),
    });
    prepared.study_mut().unwrap().query = changed.into();
    let err = prepared.estimate(&data(0.2), &context).unwrap_err();
    assert!(
        err.to_string().contains(
            "prepared static response operation no longer matches the study query or graph"
        )
    );
}

/// The legacy mean-curve operation admits every study the static DAG response
/// admits at the default response procedure. The table ranks the DAG response
/// first and declares the overlap; both must admit this study for that
/// declaration to be needed, and the DAG response must be the one retained.
#[test]
fn legacy_mean_curve_overlaps_the_static_dag_response_and_yields_to_it() {
    use crate::analysis::prepared::ExecutionKind;
    use crate::analysis::prepared::sealers::{SEALERS, SealContext, seal_execution};

    let context = ExecutionContext::for_tests(17);
    let prepared = study(data(0.0), &[-0.5, 0.0, 0.5]).prepare(&context).unwrap();
    let source = study(data(0.0), &[-0.5, 0.0, 0.5]);
    let binding = prepared.plan.binding();
    let cx = SealContext {
        source: &source,
        analysis: &prepared.analysis,
        plan: &prepared.plan,
        binding: &binding,
        ctx: &context,
        route: source.route_input_classified(),
        counterfactual: None,
        nested_counterfactual: None,
    };
    let admitting: Vec<_> = SEALERS
        .iter()
        .filter(|sealer| (sealer.seal)(&cx).unwrap().is_some())
        .map(|sealer| sealer.kind)
        .collect();
    assert_eq!(admitting, [ExecutionKind::StaticDagResponse, ExecutionKind::StaticResponseCurve]);
    assert_eq!(seal_execution(&cx).unwrap().kind(), ExecutionKind::StaticDagResponse);
}

#[test]
fn deferred_static_response_inspection_retains_checked_metadata_without_materialization() {
    use crate::analysis::prepared::{PreparedExecution, RetainedOperation};
    use crate::strategy_table::EstimatorId;
    use std::sync::Mutex;

    #[derive(Default)]
    struct Stages(Mutex<Vec<String>>);
    impl antecedent_core::ProgressSink for Stages {
        fn report(&self, _fraction: f64, stage: &str) {
            self.0.lock().unwrap().push(stage.to_owned());
        }
    }

    let grid = [-0.5, 0.0, 0.5];
    for intervention in [false, true] {
        let mut source = study(data(0.0), &grid);
        if intervention {
            source.query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
                outcome: VariableId::from_raw(2),
                interventions: Arc::from([antecedent_core::Intervention::set(
                    VariableId::from_raw(1),
                    antecedent_core::Value::f64(0.25),
                )]),
            })
            .into();
        }
        let stages = Arc::new(Stages::default());
        let mut context = ExecutionContext::for_tests(17);
        context.progress = Some(stages.clone());
        let prepared = source.prepare_deferred(&context).unwrap();
        assert!(!prepared.materialized);
        assert!(prepared.score_table.is_none());
        assert!(matches!(prepared.execution, PreparedExecution::StaticDagResponse(_)));
        let stages_before_inspection = stages.0.lock().unwrap().clone();
        let RetainedOperation::StaticDagResponse(info) = prepared.retained_operation() else {
            panic!("deferred response must retain its checked declaration");
        };
        let antecedent_core::CausalQuery::Response(expected_query) = prepared.query() else {
            panic!("expected the frozen response query");
        };
        assert_eq!(info.query, *expected_query);
        assert_eq!(info.identifier.as_str(), "response.backdoor");
        assert_eq!(
            info.estimator,
            if intervention {
                EstimatorId::ResponseInterventionGcomp
            } else {
                EstimatorId::ResponseKennedyDr
            }
        );
        assert_eq!(info.validation, crate::analysis::builder::RefuteSuite::None);
        assert_eq!(info.grid_members.as_ref(), if intervention { &[][..] } else { &grid[..] });
        let before = prepared.plan_view().unwrap().identity();
        assert_eq!(before, prepared.plan_view().unwrap().identity());
        assert!(!prepared.materialized);
        assert!(prepared.score_table.is_none());
        assert_eq!(*stages.0.lock().unwrap(), stages_before_inspection);
        let ready = prepared.materialize(&context).unwrap();
        let RetainedOperation::StaticDagResponse(materialized_info) = ready.retained_operation()
        else {
            panic!("materialization must preserve the checked response declaration");
        };
        assert_eq!(info, materialized_info);
        assert_eq!(before, ready.plan_view().unwrap().identity());
        assert!(!prepared.materialized);
        assert!(prepared.score_table.is_none());
    }
}
