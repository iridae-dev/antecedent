//! Event align + Panel stacked estimate facade paths.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use std::sync::Arc;

use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalSchemaBuilder, DataClassification, ExecutionContext, IdentificationStatus, Lag,
    MeasurementSpec, RoleHint, SmallRoleSet, TemporalEffectQuery, TemporalPolicy, ValueType,
    VariableId,
};
use antecedent_data::{
    EventData, Float64Column, OwnedColumn, OwnedColumnarStorage, PanelData, PanelUnit,
    SamplingRegularity, TableView, TimeIndex, TimeSeriesData, ValidityBitmap,
};
use antecedent_estimate::{
    BayesianGCompWorkspace, BayesianGComputationAte, TemporalLinearAdjustment,
};
use antecedent_graph::{TemporalDag, ensure_lagged};
use antecedent_identify::TemporalBackdoorIdentifier;

fn xy_series(n: usize, seed: f64) -> TimeSeriesData {
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
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
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
