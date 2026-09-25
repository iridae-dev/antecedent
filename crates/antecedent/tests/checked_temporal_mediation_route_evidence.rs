//! Builder-independent evidence for checked fixed-TemporalDag mediation.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{AcceptedGraph, BayesianConfig, EstimatorId, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, CausalSchemaBuilder, ExecutionContext, Lag, MeasurementSpec, MediationContrast,
    MediationQuery, RoleHint, SmallRoleSet, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
    TimeSeriesData, ValidityBitmap,
};
use antecedent_graph::{TemporalDag, ensure_lagged};

const N: usize = 241;
const MEDIATED_TRUTH: f64 = 0.8 * 0.55;

fn series(outcome_shift: f64, extra_column: bool) -> TimeSeriesData {
    let mut schema = CausalSchemaBuilder::new();
    for (name, hint) in [
        ("t", RoleHint::TreatmentCandidate),
        ("m", RoleHint::Context),
        ("y", RoleHint::OutcomeCandidate),
        ("z", RoleHint::Context),
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
    if extra_column {
        schema
            .add_variable(
                "extra",
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
    }
    let schema = schema.build().unwrap();
    let z: Vec<f64> = (0..N).map(|i| if i % 4 == 0 || i % 4 == 1 { 1.0 } else { -1.0 }).collect();
    let t: Vec<f64> =
        (0..N).map(|i| z[i] + if i % 4 == 0 || i % 4 == 2 { 1.0 } else { -1.0 }).collect();
    let mut m = vec![0.0; N];
    let mut y = vec![1.0 + outcome_shift; N];
    for i in 1..N {
        let noise = (i % 5) as f64 - 2.0;
        m[i] = 0.8 * t[i - 1] + 0.6 * z[i - 1] + 0.15 * noise;
        y[i] = 1.0 + outcome_shift + 0.25 * t[i - 1] + 0.55 * m[i] + 5.0 * z[i - 1];
    }
    let mut columns = vec![t, m, y, z];
    if extra_column {
        columns.push((0..N).map(|i| (i as f64).sin()).collect());
    }
    let columns = columns
        .into_iter()
        .enumerate()
        .map(|(id, values)| {
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(id as u32),
                    Arc::from(values),
                    ValidityBitmap::all_valid(N),
                )
                .unwrap(),
            )
        })
        .collect();
    let storage = OwnedColumnarStorage::try_new(schema, columns, None, None).unwrap();
    TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: N },
    )
    .unwrap()
}

fn graph() -> TemporalDag {
    let mut graph = TemporalDag::empty();
    let t0 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::CONTEMPORANEOUS).unwrap();
    let t1 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let m0 = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    let y0 = ensure_lagged(&mut graph, VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
    let z0 = ensure_lagged(&mut graph, VariableId::from_raw(3), Lag::CONTEMPORANEOUS).unwrap();
    let z1 = ensure_lagged(&mut graph, VariableId::from_raw(3), Lag::from_raw(1)).unwrap();
    for (from, to) in [(z0, t0), (z1, y0), (z1, m0), (t1, y0), (t1, m0), (m0, y0)] {
        graph.insert_directed(from, to).unwrap();
    }
    graph
}

fn changed_graph() -> TemporalDag {
    let mut graph = graph();
    // Adding an unused older node changes the prepared temporal structure while
    // leaving the fixture's causal target and observed schema intact.
    ensure_lagged(&mut graph, VariableId::from_raw(3), Lag::from_raw(2)).unwrap();
    graph
}

fn mediation_query(horizons: &[u32]) -> MediationQuery {
    MediationQuery::binary(
        VariableId::from_raw(0),
        VariableId::from_raw(2),
        [VariableId::from_raw(1)],
        MediationContrast::Mediated,
    )
    .with_horizons(horizons.to_vec())
    .unwrap()
}

#[test]
fn checked_temporal_mediation_lifecycle_covers_all_fixed_dag_coordinates() {
    let initial = series(0.0, false);
    let graph = graph();
    let query = mediation_query(&[1]);
    let context = ExecutionContext::for_tests(817);

    for accepted in [false, true] {
        for (inference, expected_estimator) in [
            (InferenceMode::Frequentist, EstimatorId::TemporalMediation),
            (
                InferenceMode::Bayesian(
                    BayesianConfig::conjugate().n_draws(2048).prior_scale(1_000.0),
                ),
                EstimatorId::BayesianTemporalMediation,
            ),
        ] {
            for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
                let base = Study::series(initial.clone());
                let base = if accepted {
                    base.graph(AcceptedGraph::temporal_dag(graph.clone()))
                } else {
                    base.graph(graph.clone())
                };
                let builder = base
                    .query(CausalQuery::Mediation(query.clone()))
                    .inference(inference.clone())
                    .refute(suite)
                    .bootstrap_replicates(0)
                    .build()
                    .unwrap();
                let mut prepared = builder.prepare(&context).unwrap();
                let info = prepared
                    .checked_temporal_mediation_info()
                    .expect("checked temporal mediation plan retained");
                assert_eq!(info.query, query);
                assert_eq!(info.horizons.as_ref(), &[1]);
                assert_eq!(info.validation, suite);
                assert_eq!(info.estimator, expected_estimator);
                assert!(!info.graph_signature.is_empty());
                drop(builder);

                let first = prepared.estimate_series(&initial, &context).unwrap();
                let tolerance =
                    if expected_estimator == EstimatorId::TemporalMediation { 1e-6 } else { 0.08 };
                assert!(
                    (first.estimate.ate - MEDIATED_TRUTH).abs() < tolerance,
                    "accepted={accepted} estimator={expected_estimator} suite={suite:?}: {}",
                    first.estimate.ate
                );
                assert_eq!(
                    first.posterior.is_some(),
                    expected_estimator == EstimatorId::BayesianTemporalMediation
                );
                assert_eq!(first.refutations.is_empty(), suite == RefuteSuite::None);
                assert_eq!(first.mediation_grid.as_ref().unwrap().joint_posterior, false);

                let refreshed = prepared.refresh_series(series(0.4, false), &context).unwrap();
                assert!(
                    (refreshed.estimate.ate - MEDIATED_TRUTH).abs() < tolerance,
                    "intercept shift must not change the mediated contrast"
                );
                assert_eq!(
                    prepared.checked_temporal_mediation_info().unwrap().graph_signature,
                    info.graph_signature
                );

                let artifact = prepared
                    .encode_contracted_result(&refreshed, "checked-temporal-mediation", &context)
                    .unwrap();
                let consumed = antecedent_io::consume_analysis_result(&artifact).unwrap();
                assert!(consumed.acceptance.unresolved.iter().any(|dependency| {
                    dependency.as_ref() == "dependencies.checked_temporal_mediation_operation"
                }));
                assert!(!consumed.acceptance.accepts_as_verified_program());
            }
        }
    }
}

#[test]
fn checked_temporal_mediation_refuses_schema_changes_and_makes_no_joint_horizon_claim() {
    let initial = series(0.0, false);
    let graph = graph();
    let context = ExecutionContext::for_tests(818);
    let query = mediation_query(&[1]);
    let mut prepared = Study::series(initial.clone())
        .graph(graph.clone())
        .query(CausalQuery::Mediation(query))
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .prepare(&context)
        .unwrap();
    assert!(prepared.refresh_series(series(0.0, true), &context).is_err());

    let other_graph = changed_graph();
    let other = Study::series(initial.clone())
        .graph(other_graph)
        .query(CausalQuery::Mediation(mediation_query(&[1])))
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .prepare(&context)
        .unwrap();
    assert_ne!(
        prepared.checked_temporal_mediation_info().unwrap().graph_signature,
        other.checked_temporal_mediation_info().unwrap().graph_signature,
        "the prepared graph identity must include disconnected lagged nodes"
    );
    let other_result = other.estimate_series(&initial, &context).unwrap();
    assert!(
        prepared.encode_contracted_result(&other_result, "wrong-temporal-graph", &context).is_err(),
        "a result prepared under another graph must not bind to this handle"
    );

    let multi_query = mediation_query(&[1, 2]);
    let multi = Study::series(initial)
        .graph(graph)
        .query(CausalQuery::Mediation(multi_query.clone()))
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(2048)))
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .prepare(&context)
        .unwrap();
    let info = multi.checked_temporal_mediation_info().unwrap();
    assert_eq!(info.query, multi_query);
    assert_eq!(info.horizons.as_ref(), &[1, 2]);
    let result = multi.estimate_series(&series(0.0, false), &context).unwrap();
    let grid = result.mediation_grid.as_ref().expect("horizon-wise results retained");
    assert_eq!(grid.slices.len(), 2);
    assert!(!grid.joint_posterior);
    assert!(
        result.posterior.is_none(),
        "a pointwise horizon posterior must not be presented as joint"
    );
    assert!(result.estimate.ate.is_nan(), "no single horizon is promoted to the scalar result");
}
