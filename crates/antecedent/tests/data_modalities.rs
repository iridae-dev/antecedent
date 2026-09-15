//! Event align + Panel stacked estimate facade paths.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use std::sync::Arc;

use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, CausalSchemaBuilder, ContinuousDomain, DataClassification, ExecutionContext,
    GridSpec, IdentificationStatus, Lag, MeasurementSpec, ResponseFunctional, ResponseQuery,
    RoleHint, SmallRoleSet, TemporalEffectQuery, TemporalPolicy, TemporalResponseSpec, ValueType,
    VariableId,
};
use antecedent_data::{
    EventData, Float64Column, OwnedColumn, OwnedColumnarStorage, PanelData, PanelUnit,
    SamplingRegularity, TableView, TimeIndex, TimeSeriesData, ValidityBitmap,
};
use antecedent_estimate::{
    BayesianGCompWorkspace, BayesianGComputationAte, TemporalLinearAdjustment,
};
use antecedent_graph::{TemporalCpdag, TemporalDag, ensure_lagged};
use antecedent_identify::TemporalBackdoorIdentifier;

fn xy_series(n: usize, seed: f64) -> TimeSeriesData {
    xy_series_regular(n, seed, 1)
}

fn xy_series_regular(n: usize, seed: f64, interval_ns: u64) -> TimeSeriesData {
    let mut b = CausalSchemaBuilder::new();
    b.add_variable(
        "x",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    b.add_variable(
        "y",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    let schema = b.build().unwrap();
    let mut x = vec![0.0; n];
    let mut y = vec![0.0; n];
    for t in 1..n {
        x[t] = ((t as f64) * 0.07 + seed).sin();
        y[t] = 0.8 * x[t - 1];
    }
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(0), Arc::from(x), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(1), Arc::from(y), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
    ];
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns }, length: n },
    )
    .unwrap()
}

fn lagged_xy_graph() -> TemporalDag {
    let mut g = TemporalDag::empty();
    let x1 = ensure_lagged(&mut g, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y0 = ensure_lagged(&mut g, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    g.insert_directed(x1, y0).unwrap();
    g
}

#[test]
fn event_align_then_temporal_effect() {
    // Dense regular events at every ns → align_to_grid(1) recovers the series.
    let series = xy_series(200, 0.0);
    let n = series.row_count();
    let times: Vec<i64> =
        (0..n).map(|i| i64::try_from(i).expect("test row count fits i64")).collect();
    let event = EventData::try_new(series.storage().clone(), Arc::from(times)).unwrap();
    let q = TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
        .with_policy(TemporalPolicy::pulse(-1))
        .with_horizon_steps(1)
        .with_max_history_lag(Some(1));
    let analysis = Study::events(&event, 1)
        .unwrap()
        .graph(lagged_xy_graph())
        .temporal_query(q)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let result = analysis.run(&ExecutionContext::for_tests(1)).unwrap();
    assert!((result.estimate.ate - 0.8).abs() < 0.08, "ate={}", result.estimate.ate);
    assert_eq!(result.logical_plan.data_classification, DataClassification::Event);
}

#[test]
fn panel_stacked_estimate_with_cluster_se() {
    let panel = PanelData::try_new(Arc::from([
        PanelUnit { unit_id: 0, series: xy_series(180, 0.1) },
        PanelUnit { unit_id: 1, series: xy_series(180, 0.4) },
        PanelUnit { unit_id: 2, series: xy_series(180, 0.7) },
    ]))
    .unwrap();
    let q = TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
        .with_policy(TemporalPolicy::pulse(-1))
        .with_horizon_steps(1)
        .with_max_history_lag(Some(1));
    let analysis = Study::panel(panel)
        .graph(lagged_xy_graph())
        .temporal_query(q)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let result = analysis.run(&ExecutionContext::for_tests(2)).unwrap();
    assert!((result.estimate.ate - 0.8).abs() < 0.08, "ate={}", result.estimate.ate);
    assert_eq!(result.logical_plan.data_classification, DataClassification::Panel);
    assert!(result.estimate.se_analytic.is_finite());
}

fn xy_series_with_unit_effect(n: usize, seed: f64, treat: f64, unit_eff: f64) -> TimeSeriesData {
    let mut b = CausalSchemaBuilder::new();
    b.add_variable(
        "x",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    b.add_variable(
        "y",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    let schema = b.build().unwrap();
    let x = vec![treat; n];
    let mut y = vec![0.0; n];
    // Treatment is constant within unit. Stacked iid treats T repeats as
    // independent looks at the same contrast; random-intercept GLS must widen.
    for t in 1..n {
        y[t] = 0.8 * x[t - 1] + unit_eff + 0.2 * ((t as f64) * 0.19 + seed).cos();
    }
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(0), Arc::from(x), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(1), Arc::from(y), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
    ];
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap()
}

#[test]
fn bayesian_panel_pulse_posterior_wider_than_stacked_iid() {
    let panel = PanelData::try_new(
        (0..20u32)
            .map(|u| {
                let treat = if u % 2 == 0 { 1.0 } else { 0.0 };
                let unit_eff = 1.6 * (f64::from(u) * 0.31).sin();
                PanelUnit {
                    unit_id: u,
                    series: xy_series_with_unit_effect(16, f64::from(u) * 0.37, treat, unit_eff),
                }
            })
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let q = TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
        .with_policy(TemporalPolicy::pulse(-1))
        .with_horizon_steps(1)
        .with_max_history_lag(Some(1));
    let ctx = ExecutionContext::for_tests(3);
    let study = Study::panel(panel.clone())
        .graph(lagged_xy_graph())
        .temporal_query(q.clone())
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(400)))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let result = study.run(&ctx).unwrap();
    let post = result.posterior.as_ref().expect("bayesian panel posterior");
    let eq = post.effect_column().expect("effect column");
    let study_sd = post.summaries.sd[eq];

    let id = TemporalBackdoorIdentifier::new().identify_temporal(&lagged_xy_graph(), &q).unwrap();
    let estimand = id.result.estimands.first().unwrap();
    let (prep, cluster_ids, _) = TemporalLinearAdjustment::new()
        .prepare_panel(&panel, estimand, &q, &id.indexer, None, &ctx.kernel_policy)
        .unwrap();
    let stacked = BayesianGComputationAte::from_prepared_estimation(&prep);
    let mut hierarchical = stacked.clone();
    hierarchical.unit_ids = Some(cluster_ids);
    let bayes =
        BayesianGComputationAte { n_draws: 400, seed: 3, ..BayesianGComputationAte::conjugate() };
    let mut ws = BayesianGCompWorkspace::default();
    let post_s = bayes
        .fit(&stacked, IdentificationStatus::NonparametricallyIdentified, &mut ws, &ctx)
        .unwrap();
    let post_h = bayes
        .fit(&hierarchical, IdentificationStatus::NonparametricallyIdentified, &mut ws, &ctx)
        .unwrap();
    let eq_s = post_s.effect_column().unwrap();
    let sd_s = post_s.summaries.sd[eq_s];
    let sd_h = post_h.summaries.sd[eq_s];
    let pin: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/estimate/panel_hierarchical_vs_stacked/expected.json"
    ))
    .unwrap();
    let min_ratio = pin["min_posterior_sd_ratio_hierarchical_over_stacked"].as_f64().unwrap();
    assert!(sd_s.is_finite() && sd_h.is_finite() && study_sd.is_finite());
    assert!(
        sd_h > sd_s * min_ratio,
        "hierarchical GLS must be wider than stacked iid (stacked={sd_s}, hierarchical={sd_h})"
    );
    assert!(
        study_sd > sd_s * min_ratio,
        "execute_panel Bayesian posterior must use hierarchical unit effects (study={study_sd}, stacked={sd_s})"
    );
    assert!(
        (study_sd - sd_h).abs() / sd_h.max(1e-12) < 0.35,
        "execute_panel posterior must match the GLS fit, not stacked iid (study={study_sd}, hierarchical={sd_h}, stacked={sd_s})"
    );
}

#[test]
fn prepared_panel_pulse_reuses_identification() {
    let panel = PanelData::try_new(Arc::from([
        PanelUnit { unit_id: 0, series: xy_series(180, 0.1) },
        PanelUnit { unit_id: 1, series: xy_series(180, 0.4) },
        PanelUnit { unit_id: 2, series: xy_series(180, 0.7) },
    ]))
    .unwrap();
    let q = TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
        .with_policy(TemporalPolicy::pulse(-1))
        .with_horizon_steps(1)
        .with_max_history_lag(Some(1));
    let ctx = ExecutionContext::for_tests(2);
    let study = Study::panel(panel.clone())
        .graph(lagged_xy_graph())
        .temporal_query(q)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let fresh = study.clone().run(&ctx).unwrap();
    let mut prepared = study.prepare(&ctx).unwrap();
    let first = prepared.estimate_panel(&panel, &ctx).unwrap();
    let second = prepared.estimate_panel(&panel, &ctx).unwrap();
    assert_eq!(
        fresh.diagnostics.iter().filter(|d| d.code.as_ref() == "exec.identify.cached").count(),
        0
    );
    assert_eq!(
        first.diagnostics.iter().filter(|d| d.code.as_ref() == "exec.identify.cached").count(),
        1
    );
    assert_eq!(
        second.diagnostics.iter().filter(|d| d.code.as_ref() == "exec.identify.cached").count(),
        1
    );
    assert!((first.estimate.ate - fresh.estimate.ate).abs() < 1e-12);
    assert!((second.estimate.ate - first.estimate.ate).abs() < 1e-12);

    let mismatch = PanelData::try_new(Arc::from([PanelUnit {
        unit_id: 0,
        series: xy_series_regular(80, 0.2, 2),
    }]))
    .unwrap();
    let prior_ate = first.estimate.ate;
    let err = prepared.refresh_panel(mismatch, &ctx).unwrap_err();
    assert!(err.to_string().contains("regularity"), "{err}");
    let still = prepared.estimate_panel(&panel, &ctx).unwrap();
    assert!((still.estimate.ate - prior_ate).abs() < 1e-12);
}

fn pulse_query() -> TemporalEffectQuery {
    TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
        .with_policy(TemporalPolicy::pulse(-1))
        .with_horizon_steps(1)
        .with_max_history_lag(Some(1))
}

#[test]
fn prepared_panel_handle_refuses_series_and_tabular_data() {
    let panel = PanelData::try_new(Arc::from([
        PanelUnit { unit_id: 0, series: xy_series(120, 0.1) },
        PanelUnit { unit_id: 1, series: xy_series(120, 0.4) },
    ]))
    .unwrap();
    let ctx = ExecutionContext::for_tests(2);
    let mut prepared = Study::panel(panel.clone())
        .graph(lagged_xy_graph())
        .temporal_query(pulse_query())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let before = prepared.estimate_panel(&panel, &ctx).unwrap();

    // Same schema and regularity: only the modality differs.
    let series = xy_series(120, 0.1);
    for err in [
        prepared.estimate_series(&series, &ctx).unwrap_err(),
        prepared.refresh_series(series.clone(), &ctx).unwrap_err(),
        prepared
            .estimate(&antecedent_data::TabularData::new(series.storage().clone()), &ctx)
            .unwrap_err(),
    ] {
        let text = err.to_string();
        assert!(text.contains("prepared panel analysis requires panel data"), "{text}");
        assert!(text.contains("estimate_panel"), "{text}");
    }
    let after = prepared.estimate_panel(&panel, &ctx).unwrap();
    assert!((after.estimate.ate - before.estimate.ate).abs() < 1e-12);
}

#[test]
fn prepared_series_handle_refuses_panel_data() {
    let series = xy_series(180, 0.1);
    let ctx = ExecutionContext::for_tests(2);
    let mut prepared = Study::series(series.clone())
        .graph(lagged_xy_graph())
        .temporal_query(pulse_query())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let before = prepared.estimate_series(&series, &ctx).unwrap();
    let panel = PanelData::try_new(Arc::from([
        PanelUnit { unit_id: 0, series: xy_series(180, 0.1) },
        PanelUnit { unit_id: 1, series: xy_series(180, 0.4) },
    ]))
    .unwrap();
    for err in [
        prepared.estimate_panel(&panel, &ctx).unwrap_err(),
        prepared.refresh_panel(panel.clone(), &ctx).unwrap_err(),
    ] {
        let text = err.to_string();
        assert!(text.contains("prepared series analysis requires series data"), "{text}");
        assert!(text.contains("estimate_series"), "{text}");
    }
    let after = prepared.estimate_series(&series, &ctx).unwrap();
    assert!((after.estimate.ate - before.estimate.ate).abs() < 1e-12);
}

#[test]
fn prepared_panel_refuses_units_with_mixed_regularity() {
    // Refresh requires every unit to match the frozen regularity, so prepare
    // must not freeze unit 0's grid for a panel whose units disagree.
    let panel = PanelData::try_new(Arc::from([
        PanelUnit { unit_id: 0, series: xy_series_regular(120, 0.1, 1) },
        PanelUnit { unit_id: 1, series: xy_series_regular(120, 0.4, 2) },
    ]))
    .unwrap();
    let err = Study::panel(panel)
        .graph(lagged_xy_graph())
        .temporal_query(pulse_query())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ExecutionContext::for_tests(2))
        .unwrap_err();
    assert!(
        err.to_string().contains("every panel unit to share one time-index regularity"),
        "{err}"
    );
}

fn panel_mean_curve_query() -> ResponseQuery {
    ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(vec![0.0, 1.0].into()),
        ),
    })
    .with_temporal(
        TemporalResponseSpec::new(vec![1u32], TemporalPolicy::pulse(-1), Some(1)).unwrap(),
    )
}

#[test]
fn panel_response_curve_uses_unit_cluster_bands() {
    let panel = PanelData::try_new(Arc::from([
        PanelUnit { unit_id: 0, series: xy_series(80, 0.1) },
        PanelUnit { unit_id: 1, series: xy_series(80, 0.4) },
        PanelUnit { unit_id: 2, series: xy_series(80, 0.7) },
    ]))
    .unwrap();
    let query = panel_mean_curve_query();
    let result = Study::panel(panel)
        .graph(lagged_xy_graph())
        .query(CausalQuery::Response(query))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(2))
        .unwrap();
    assert_eq!(result.logical_plan.data_classification, DataClassification::Panel);
    let response = result.response.as_ref().expect("panel response surface");
    let antecedent_core::ResponseIdentification::PointIdentified(
        antecedent_core::ResponseValue::Surface { mean, .. },
    ) = &response.estimate
    else {
        panic!("expected a point-identified surface, got {:?}", response.estimate);
    };
    assert!(mean.iter().all(|value| value.is_finite()));
    assert!(
        (mean[1] - mean[0] - 0.8).abs() < 0.15,
        "unit-average pulse contrast should stay near 0.8, got {:?}",
        mean
    );
    match &response.uncertainty {
        antecedent_core::ResponseUncertainty::PointwiseBand { lower, upper, .. } => {
            assert_eq!(lower.len(), mean.len());
            assert!(lower.iter().zip(upper.iter()).all(|(lo, hi)| lo < hi));
        }
        other => panic!("panel response must publish between-unit pointwise bands, got {other:?}"),
    }
    assert!(
        result
            .diagnostics
            .iter()
            .any(|item| item.code.as_ref() == "estimate.temporal_response.panel.cluster_units"),
        "missing cluster-unit diagnostic"
    );
}

#[test]
fn bayesian_panel_response_stays_refused() {
    let panel = PanelData::try_new(Arc::from([
        PanelUnit { unit_id: 0, series: xy_series(80, 0.1) },
        PanelUnit { unit_id: 1, series: xy_series(80, 0.4) },
    ]))
    .unwrap();
    let err = Study::panel(panel)
        .graph(lagged_xy_graph())
        .query(CausalQuery::Response(panel_mean_curve_query()))
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate()))
        .refute(RefuteSuite::None)
        .build()
        .unwrap_err();
    assert!(err.to_string().contains("single-series response likelihood"), "{err}");
}

#[test]
fn panel_class_pulse_uses_completion_masses() {
    let panel = PanelData::try_new(Arc::from([
        PanelUnit { unit_id: 0, series: xy_series(180, 0.1) },
        PanelUnit { unit_id: 1, series: xy_series(180, 0.4) },
        PanelUnit { unit_id: 2, series: xy_series(180, 0.7) },
    ]))
    .unwrap();
    let result = Study::panel(panel)
        .graph(TemporalCpdag::from_temporal_dag(&lagged_xy_graph()))
        .temporal_query(pulse_query())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(2))
        .unwrap();
    assert_eq!(result.logical_plan.data_classification, DataClassification::Panel);
    assert!((result.estimate.ate - 0.8).abs() < 0.08, "ate={}", result.estimate.ate);
    assert!(result.structural_response.is_some(), "class Pulse must publish completion masses");
    assert!(
        result.diagnostics.iter().any(|item| {
            item.code.as_ref() == "estimate.temporal_effect.panel.class.cluster_units"
        }),
        "missing panel class diagnostic"
    );
}

#[test]
fn bayesian_panel_class_pulse_stays_refused() {
    let panel = PanelData::try_new(Arc::from([
        PanelUnit { unit_id: 0, series: xy_series(80, 0.1) },
        PanelUnit { unit_id: 1, series: xy_series(80, 0.4) },
    ]))
    .unwrap();
    let err = Study::panel(panel)
        .graph(TemporalCpdag::from_temporal_dag(&lagged_xy_graph()))
        .temporal_query(pulse_query())
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate()))
        .refute(RefuteSuite::None)
        .build()
        .unwrap_err();
    assert!(err.to_string().contains("series class-posterior is not a panel class model"), "{err}");
}

#[test]
fn incomplete_class_panel_response_stays_refused() {
    let panel = PanelData::try_new(Arc::from([
        PanelUnit { unit_id: 0, series: xy_series(80, 0.1) },
        PanelUnit { unit_id: 1, series: xy_series(80, 0.4) },
    ]))
    .unwrap();
    let err = Study::panel(panel)
        .graph(TemporalCpdag::from_temporal_dag(&lagged_xy_graph()))
        .query(CausalQuery::Response(panel_mean_curve_query()))
        .refute(RefuteSuite::None)
        .build()
        .unwrap_err();
    assert!(err.to_string().contains("incomplete temporal classes"), "{err}");
}

#[test]
fn contract_snapshot_binds_panel_labels_order_and_time_metadata() {
    let a = PanelUnit { unit_id: 7, series: xy_series_regular(80, 0.1, 1) };
    let b = PanelUnit { unit_id: 9, series: xy_series_regular(80, 0.4, 1) };
    let inspect = |units: Vec<PanelUnit>| {
        Study::panel(PanelData::try_new(Arc::from(units)).unwrap())
            .graph(lagged_xy_graph())
            .temporal_query(pulse_query())
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
            .inspect()
            .unwrap()
    };
    let base = inspect(vec![a.clone(), b.clone()]);
    assert_eq!(base.identities, inspect(vec![a.clone(), b.clone()]).identities);
    for units in [
        vec![b.clone(), a.clone()],
        vec![PanelUnit { unit_id: 8, ..a.clone() }, b.clone()],
        vec![a, PanelUnit { unit_id: 9, series: xy_series_regular(80, 0.4, 2) }],
    ] {
        let changed = inspect(units);
        assert_eq!(base.identities.target, changed.identities.target);
        assert_eq!(base.identities.program, changed.identities.program);
        assert_ne!(base.identities.data_snapshot, changed.identities.data_snapshot);
    }
}
