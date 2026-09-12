//! Per-horizon DBN mediation eligibility and prepared numerical reuse.
#![allow(clippy::cast_precision_loss)]

use super::*;
use crate::{BayesianConfig, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalSchemaBuilder, IdentificationStatus, MeasurementSpec, MediationContrast, RoleHint,
    SmallRoleSet, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex, ValidityBitmap,
};
use antecedent_prob::InferenceDiagnostics;
use std::sync::Arc;

fn mediation_series(rows: usize) -> (TimeSeriesData, MediationQuery) {
    let mut builder = CausalSchemaBuilder::new();
    for name in ["treatment", "mediator", "outcome"] {
        builder
            .add_variable(
                name,
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
    }
    let schema = builder.build().unwrap();
    let mut treatment = vec![0.0; rows];
    let mut mediator = vec![0.0; rows];
    let mut outcome = vec![0.0; rows];
    for (i, slot) in treatment.iter_mut().enumerate() {
        *slot = (0.071 * i as f64).sin() + 0.35 * (0.137 * i as f64).cos();
    }
    for i in 1..rows {
        mediator[i] = 0.8 * treatment[i - 1] + 0.12 * (0.43 * i as f64).sin();
        outcome[i] = 0.25 * treatment[i - 1] + 0.55 * mediator[i] + 0.09 * (0.29 * i as f64).cos();
    }
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(
                VariableId::from_raw(0),
                Arc::from(treatment),
                ValidityBitmap::all_valid(rows),
            )
            .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(
                VariableId::from_raw(1),
                Arc::from(mediator),
                ValidityBitmap::all_valid(rows),
            )
            .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(
                VariableId::from_raw(2),
                Arc::from(outcome),
                ValidityBitmap::all_valid(rows),
            )
            .unwrap(),
        ),
    ];
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    let series = TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: rows },
    )
    .unwrap();
    let query = MediationQuery::binary(
        VariableId::from_raw(0),
        VariableId::from_raw(2),
        [VariableId::from_raw(1)],
        MediationContrast::Mediated,
    );
    (series, query)
}

fn graph_posterior() -> GraphPosterior {
    GraphPosterior::new(
        3,
        vec![0.7, 0.3],
        vec![8, 8],
        vec![0.0; 9],
        vec![0.0; 9],
        1.0 / (0.7 * 0.7 + 0.3 * 0.3),
        InferenceDiagnostics::analytic("horizon-cache-test"),
        0,
    )
    .unwrap()
    .with_lagged_marginals(1, vec![0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    .unwrap()
    .with_lag_masks(vec![6, 6])
    .unwrap()
}

#[test]
fn mixed_horizon_certificates_keep_mass_and_prepared_estimates_independent() {
    let (series, mut query) = mediation_series(320);
    query.horizons = Arc::from([1, 2]);
    let posterior = graph_posterior();
    let ctx = ExecutionContext::for_tests(7);
    let variables = [VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2)];
    let mut calls = 0;
    // The current three-variable identifier certifies these atoms at both
    // horizons. Inject one explicitly labeled certificate failure to test
    // horizon eligibility independently of that narrower theory surface.
    let cache = build_dbn_mediation_cache_with_identifier(
        &posterior,
        &variables,
        &query,
        &ctx,
        |atom, horizon, graph, horizon_query| {
            calls += 1;
            if atom == 1 && horizon == 2 {
                return Err(CausalError::Compile {
                    message: "test-only missing I(2) certificate".into(),
                });
            }
            identify_temporal_mediation_horizons(
                graph,
                horizon_query,
                crate::strategy_table::EstimatorId::BayesianTemporalMediation,
            )
        },
    )
    .unwrap();
    assert_eq!(calls, 4);
    assert_eq!(cache.atoms.len(), 2, "I(2) failure must not remove the atom's I(1)");
    let first = cache.mediation_horizon(1).unwrap();
    let second = cache.mediation_horizon(2).unwrap();
    assert!(first.graphs.unidentified_mass().abs() < 1e-12);
    assert!((second.graphs.unidentified_mass() - 0.3).abs() < 1e-12);
    assert_eq!(first.identify_demotion.identify_failed, 0);
    assert_eq!(second.identify_demotion.identify_failed, 1);
    assert!(cache.mediation_horizon(3).is_err());

    let study = Study::series(series.clone())
        .graph_posterior(posterior)
        .query(CausalQuery::Mediation(query))
        .inference(InferenceMode::Bayesian(
            BayesianConfig::conjugate().n_draws(1024).prior_scale(10.0),
        ))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let fresh = study.clone().run(&ctx).unwrap();
    let mut prepared = study.prepare(&ctx).unwrap();
    let baseline_click = prepared.estimate_series(&series, &ctx).unwrap();
    let fresh_grid = fresh.mediation_grid.as_ref().unwrap();
    let baseline_grid = baseline_click.mediation_grid.as_ref().unwrap();
    for (fresh, cached) in fresh_grid.slices.iter().zip(baseline_grid.slices.iter()) {
        assert!((fresh.estimate.effect.ate - cached.estimate.effect.ate).abs() < 1e-12);
    }
    assert!((fresh_grid.slices[0].estimate.effect.ate - 0.44).abs() < 0.04);
    prepared.analysis.dbn_posterior_identification_cache = Some(Arc::new(cache));
    let mixed = prepared.estimate_series(&series, &ctx).unwrap();
    let repeated = prepared.refresh_series(series.clone(), &ctx).unwrap();
    let grid = mixed.mediation_grid.as_ref().unwrap();
    assert!(
        (grid.slices[0].estimate.effect.ate - fresh_grid.slices[0].estimate.effect.ate).abs()
            < 1e-12
    );
    assert!(
        (grid.slices[1].estimate.effect.ate - fresh_grid.slices[1].estimate.effect.ate).abs()
            < 0.02
    );
    assert_ne!(grid.slices[0].identification_status, IdentificationStatus::GraphDependent);
    assert_eq!(grid.slices[1].identification_status, IdentificationStatus::GraphDependent);
    for (slice, repeated) in
        grid.slices.iter().zip(repeated.mediation_grid.as_ref().unwrap().slices.iter())
    {
        assert!((slice.estimate.effect.ate - repeated.estimate.effect.ate).abs() < 1e-12);
    }
    for (slice, expected_mass) in grid.slices.iter().zip([0.0, 0.3]) {
        let mass = slice
            .diagnostics
            .iter()
            .find_map(|d| {
                if d.code.as_ref() != "estimate.dbn_posterior.envelope" {
                    return None;
                }
                d.message
                    .strip_prefix("unidentified_mass=")
                    .and_then(|value| value.parse::<f64>().ok())
            })
            .expect("each slice retains its unidentified mass");
        assert!((mass - expected_mass).abs() < 1e-12, "mass={mass} expected={expected_mass}");
    }
    let all_missing = build_dbn_mediation_cache_with_identifier(
        &graph_posterior(),
        &variables,
        &MediationQuery { horizons: Arc::from([1, 2]), ..mediation_series(8).1 },
        &ctx,
        |_, horizon, graph, horizon_query| {
            if horizon == 2 {
                return Err(CausalError::Compile {
                    message: "test-only unavailable horizon".into(),
                });
            }
            identify_temporal_mediation_horizons(
                graph,
                horizon_query,
                crate::strategy_table::EstimatorId::BayesianTemporalMediation,
            )
        },
    )
    .unwrap();
    prepared.analysis.dbn_posterior_identification_cache = Some(Arc::new(all_missing));
    let partial_grid = prepared.estimate_series(&series, &ctx).unwrap();
    let slices = &partial_grid.mediation_grid.as_ref().unwrap().slices;
    assert!(
        (slices[0].estimate.effect.ate - fresh_grid.slices[0].estimate.effect.ate).abs() < 1e-12
    );
    assert_eq!(slices[1].identification_status, IdentificationStatus::NotIdentified);
    assert!(slices[1].estimate.effect.ate.is_nan());
    assert!(matches!(
        slices[1].uncertainty,
        antecedent_estimate::TemporalMediationUncertainty::Unavailable
    ));
}
