//! Event align + Panel stacked estimate facade paths.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names,
    clippy::uninlined_format_args
)]

mod common;

use std::sync::Arc;

use antecedent::{BayesianConfig, ClassPrior, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, CausalSchemaBuilder, ContinuousDomain, DataClassification, ExecutionContext,
    GridSpec, IdentificationStatus, Lag, MeasurementSpec, ResponseFunctional, ResponseQuery,
    RoleHint, SmallRoleSet, TemporalEffectQuery, TemporalPolicy, TemporalResponseSpec, ValueType,
    VariableId,
};
use antecedent_data::{
    DiscoveryEstimationSplit, EventData, Float64Column, OwnedColumn, OwnedColumnarStorage,
    PanelData, PanelUnit, SamplingRegularity, TableView, TimeIndex, TimeSeriesData, ValidityBitmap,
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
fn panel_refuses_units_with_mixed_regularity() {
    // A horizon of h steps must mean one duration in every unit, so the panel
    // license refuses mixed sampling intervals on every route, not only on
    // prepare (a refresh also checks each unit against the frozen regularity).
    let panel = PanelData::try_new(Arc::from([
        PanelUnit { unit_id: 0, series: xy_series_regular(120, 0.1, 1) },
        PanelUnit { unit_id: 1, series: xy_series_regular(120, 0.4, 2) },
    ]))
    .unwrap();
    for query in [
        CausalQuery::TemporalEffect(pulse_query()),
        CausalQuery::Response(panel_mean_curve_query()),
    ] {
        for builder in [
            Study::panel(panel.clone()).graph(lagged_xy_graph()),
            Study::panel(panel.clone()).graph(TemporalCpdag::from_temporal_dag(&lagged_xy_graph())),
        ] {
            let err = builder
                .query(query.clone())
                .refute(RefuteSuite::None)
                .bootstrap_replicates(0)
                .build()
                .unwrap_err();
            assert!(
                err.to_string().contains("every panel unit to share one time-index regularity"),
                "{err}"
            );
        }
    }
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
fn bayesian_panel_response_uses_unit_cluster_bands() {
    let panel = PanelData::try_new(Arc::from([
        PanelUnit { unit_id: 0, series: xy_series(80, 0.1) },
        PanelUnit { unit_id: 1, series: xy_series(80, 0.4) },
        PanelUnit { unit_id: 2, series: xy_series(80, 0.7) },
    ]))
    .unwrap();
    let result = Study::panel(panel)
        .graph(lagged_xy_graph())
        .query(CausalQuery::Response(panel_mean_curve_query()))
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(32)))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(3))
        .unwrap();
    let response = result.response.as_ref().expect("Bayesian panel response surface");
    let antecedent_core::ResponseIdentification::PointIdentified(
        antecedent_core::ResponseValue::Surface { mean, .. },
    ) = &response.estimate
    else {
        panic!("expected a point-identified surface, got {:?}", response.estimate);
    };
    assert!(mean.iter().all(|value| value.is_finite()));
    assert!(
        (mean[1] - mean[0] - 0.8).abs() < 0.25,
        "Bayesian unit-average pulse contrast should stay near 0.8, got {:?}",
        mean
    );
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
fn bayesian_panel_class_pulse_uses_completion_masses() {
    // The series class-prior contract: completion-enumeration weights are not a
    // class prior, so without one the Bayesian class effect is NaN and each
    // completion's panel posterior stays an atom; a caller class prior mixes the
    // completion posteriors' draws.
    let panel = PanelData::try_new(Arc::from([
        PanelUnit { unit_id: 0, series: xy_series(180, 0.1) },
        PanelUnit { unit_id: 1, series: xy_series(180, 0.4) },
        PanelUnit { unit_id: 2, series: xy_series(180, 0.7) },
    ]))
    .unwrap();
    let run = |prior: Option<ClassPrior>| {
        let mut builder = Study::panel(panel.clone())
            .graph(TemporalCpdag::from_temporal_dag(&lagged_xy_graph()))
            .temporal_query(pulse_query())
            .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(32)))
            .refute(RefuteSuite::None)
            .bootstrap_replicates(199);
        if let Some(prior) = prior {
            builder = builder.class_prior(prior);
        }
        builder.build().unwrap().run(&ExecutionContext::for_tests(4)).unwrap()
    };
    let has = |result: &antecedent::StudyResult, code: &str| {
        result.diagnostics.iter().any(|item| item.code.as_ref() == code)
    };
    let enumeration = run(None);
    assert!(enumeration.estimate.ate.is_nan(), "ate={}", enumeration.estimate.ate);
    assert!(enumeration.posterior.is_none());
    assert!(enumeration.estimate.se_bootstrap.is_none(), "no frequentist bootstrap under Bayes");
    assert!(has(&enumeration, "estimate.temporal_class.enumeration_not_probability"));
    assert!(has(&enumeration, "estimate.temporal_effect.panel.class.bayesian_units"));
    let mixture = enumeration.structural_response.as_ref().expect("completion masses");
    assert_eq!(mixture.weight_basis, antecedent::StructuralWeightBasis::CompletionEnumeration);
    let atom = mixture.atoms.iter().find(|atom| atom.posterior.is_some()).expect("atom posterior");
    assert!((atom_scalar(atom) - 0.8).abs() < 0.15, "atom={:?}", atom.value);

    let prior = run(Some(ClassPrior::from_ordered([1.0]).unwrap()));
    assert!((prior.estimate.ate - 0.8).abs() < 0.15, "ate={}", prior.estimate.ate);
    assert!(prior.posterior.is_some(), "a class prior mixes completion posteriors");
    assert_eq!(
        prior.structural_response.as_ref().unwrap().weight_basis,
        antecedent::StructuralWeightBasis::CallerSuppliedClassPrior
    );
    assert_eq!(
        prior.interval.as_ref().map(|interval| interval.method),
        Some(antecedent_core::IntervalMethod::PosteriorQuantile)
    );
}

#[test]
fn bayesian_panel_class_pulse_mixes_draws_by_the_caller_class_prior() {
    use common::panel_dgp::{confounded_unit, cpdag_two, panel, pulse_query};
    // Two completions that disagree (adjusting z@1 or not). The mixture follows the
    // class prior, and its variance includes the between-completion spread.
    let units: Vec<_> = (0..4).map(|i| confounded_unit(200, 11 + i)).collect();
    let run = |masses: [f64; 2]| {
        Study::panel(panel(units.clone()))
            .graph(cpdag_two())
            .temporal_query(pulse_query())
            .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(256)))
            .class_prior(ClassPrior::from_ordered(masses).unwrap())
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
            .run(&ExecutionContext::for_tests(9))
            .unwrap()
    };
    let low = run([0.1, 0.9]);
    let high = run([0.9, 0.1]);
    for (result, masses) in [(&low, [0.1, 0.9]), (&high, [0.9, 0.1])] {
        let mixture = result.structural_response.as_ref().unwrap();
        let values: Vec<f64> = mixture.atoms.iter().map(atom_scalar).collect();
        assert_eq!(values.len(), 2);
        assert!((values[0] - values[1]).abs() > 0.3, "completions must disagree: {values:?}");
        let expected = masses[0] * values[0] + masses[1] * values[1];
        assert!(
            (result.estimate.ate - expected).abs() < 0.05,
            "mixture {} vs class-prior mean {expected} ({values:?})",
            result.estimate.ate
        );
        let posterior = result.posterior.as_ref().expect("mixed posterior");
        let sd = posterior.summaries.sd[posterior.effect_column().unwrap()];
        let between = (masses[0] * masses[1]).sqrt() * (values[0] - values[1]).abs();
        assert!(
            sd >= 0.9 * between,
            "mixture sd {sd} omits the between-completion spread {between}"
        );
    }
    assert!((low.estimate.ate - high.estimate.ate).abs() > 0.2);
}

#[test]
fn panel_class_response_uses_completion_surfaces() {
    // Like the series class response, the panel class response publishes the
    // identified set as a pointwise envelope over the completions' full surfaces,
    // not an enumeration-weighted point surface.
    let panel = PanelData::try_new(Arc::from([
        PanelUnit { unit_id: 0, series: xy_series(80, 0.1) },
        PanelUnit { unit_id: 1, series: xy_series(80, 0.4) },
        PanelUnit { unit_id: 2, series: xy_series(80, 0.7) },
    ]))
    .unwrap();
    let result = Study::panel(panel)
        .graph(TemporalCpdag::from_temporal_dag(&lagged_xy_graph()))
        .query(CausalQuery::Response(panel_mean_curve_query()))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(5))
        .unwrap();
    let response = result.response.as_ref().expect("panel class response surface");
    let antecedent_core::ResponseIdentification::PartiallyIdentified(
        antecedent_core::ResponseValue::Envelope(envelope),
    ) = &response.estimate
    else {
        panic!("expected a partially identified envelope, got {:?}", response.estimate);
    };
    assert_eq!(response.identification_status, IdentificationStatus::PartiallyIdentified);
    assert_eq!(result.identification.status, response.identification_status);
    assert!(envelope.lower.iter().zip(envelope.upper.iter()).all(|(lo, hi)| lo <= hi));
    // One completion identifies here, so the identified set is a singleton.
    assert!(
        envelope.lower.iter().zip(envelope.upper.iter()).all(|(lo, hi)| (hi - lo).abs() < 1e-12)
    );
    assert!(
        (envelope.lower[1] - envelope.lower[0] - 0.8).abs() < 0.15,
        "class unit-average pulse contrast should stay near 0.8, got {:?}",
        envelope.lower
    );
    assert!(result.structural_response.is_some(), "class response must publish completion masses");
    let mixture = result.structural_response.as_ref().expect("masses");
    assert!(
        (mixture.identified_mass + mixture.unidentified_mass + mixture.unevaluable_mass - 1.0)
            .abs()
            < 1e-12
    );
    assert_eq!(mixture.identified_set.as_ref().map(|set| set.lower.len()), Some(2));
}

#[test]
fn panel_class_response_identified_set_spans_disagreeing_completions() {
    use common::panel_dgp::{confounded_unit, cpdag_two, curve_query, panel};
    // Two completions that disagree: the published set must span both surfaces, and
    // each completion's own unit-average surface stays on its structural atom.
    let units: Vec<_> = (0..3).map(|i| confounded_unit(200, 140 + i)).collect();
    let result = Study::panel(panel(units.clone()))
        .graph(cpdag_two())
        .query(CausalQuery::Response(curve_query(vec![1])))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(9))
        .unwrap();
    let response = result.response.as_ref().unwrap();
    let antecedent_core::ResponseIdentification::PartiallyIdentified(
        antecedent_core::ResponseValue::Envelope(envelope),
    ) = &response.estimate
    else {
        panic!("expected a partially identified envelope, got {:?}", response.estimate);
    };
    let atoms: Vec<Vec<f64>> = result
        .structural_response
        .as_ref()
        .unwrap()
        .atoms
        .iter()
        .filter_map(|atom| match &atom.value {
            Some(antecedent_core::ResponseValue::Surface { mean, .. }) => Some(mean.to_vec()),
            _ => None,
        })
        .collect();
    assert_eq!(atoms.len(), 2, "both completions keep their full surface");
    for cell in 0..envelope.lower.len() {
        let values: Vec<f64> = atoms.iter().map(|atom| atom[cell]).collect();
        let lo = values.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert!((envelope.lower[cell] - lo).abs() < 1e-12, "cell {cell}");
        assert!((envelope.upper[cell] - hi).abs() < 1e-12, "cell {cell}");
    }
    assert!(
        envelope.upper[1] - envelope.lower[1] > 0.3,
        "the completions disagree at dose 1: {envelope:?}"
    );
    let identified_set = result.structural_response.as_ref().unwrap().identified_set.clone();
    assert_eq!(identified_set.as_ref().map(|set| set.lower.len()), Some(2));
}

#[test]
fn panel_class_response_covers_several_horizons() {
    use common::panel_dgp::{UnitSpec, curve_query, lagged_ty_dag, panel, unit_series};
    // Two horizons on one panel: the grid comes from the query (dose x horizon
    // pairs), every horizon carries an identification record, and the surface
    // matches the DAG panel route's cell layout.
    let spec = UnitSpec { n: 200, beta: 0.8, gz: 0.0, phi: 0.5, rho: 0.0 };
    let units: Vec<_> = (0..3).map(|i| unit_series(spec, 150 + i)).collect();
    let run = |builder: antecedent::StudyBuilder| {
        builder
            .query(CausalQuery::Response(curve_query(vec![1, 2])))
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
            .run(&ExecutionContext::for_tests(5))
            .unwrap()
    };
    let class = run(Study::panel(panel(units.clone()))
        .graph(TemporalCpdag::from_temporal_dag(&lagged_ty_dag())));
    let dag = run(Study::panel(panel(units)).graph(lagged_ty_dag()));
    let response = class.response.as_ref().unwrap();
    let antecedent_core::ResponseIdentification::PartiallyIdentified(
        antecedent_core::ResponseValue::Envelope(envelope),
    ) = &response.estimate
    else {
        panic!("expected a partially identified envelope, got {:?}", response.estimate);
    };
    let (dag_mean, dag_grid) = match &dag.response.as_ref().unwrap().estimate {
        antecedent_core::ResponseIdentification::PointIdentified(
            antecedent_core::ResponseValue::Surface { mean, grid, .. },
        ) => (mean.to_vec(), grid.to_vec()),
        other => panic!("{other:?}"),
    };
    assert_eq!(envelope.grid.to_vec(), dag_grid, "the grid is the query's dose x horizon grid");
    assert_eq!(envelope.lower.len(), 4);
    assert_eq!(
        response.horizon_identification.as_ref().map(|records| records.len()),
        Some(2),
        "every horizon carries an identification record"
    );
    for (cell, value) in dag_mean.iter().enumerate() {
        assert!(
            (envelope.lower[cell] - value).abs() < 1e-9,
            "single identified completion must match the DAG route at cell {cell}"
        );
    }
}

#[test]
fn panel_class_multi_step_sustained_uses_completion_masses() {
    let panel = PanelData::try_new(Arc::from([
        PanelUnit { unit_id: 0, series: xy_series(180, 0.1) },
        PanelUnit { unit_id: 1, series: xy_series(180, 0.4) },
        PanelUnit { unit_id: 2, series: xy_series(180, 0.7) },
    ]))
    .unwrap();
    let query = TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
        .with_policy(TemporalPolicy::sustained(-1, 0))
        .with_horizon_steps(1)
        .with_max_history_lag(Some(1));
    let result = Study::panel(panel)
        .graph(TemporalCpdag::from_temporal_dag(&lagged_xy_graph()))
        .temporal_query(query)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(6))
        .unwrap();
    assert!(result.estimate.ate.is_finite(), "ate={}", result.estimate.ate);
    assert!(result.structural_response.is_some(), "multi-step class Sustained must publish masses");
    assert!(
        result
            .diagnostics
            .iter()
            .any(|item| item.code.as_ref() == "estimate.temporal.sustained_window"),
        "missing sequential window diagnostic"
    );
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
        vec![PanelUnit { unit_id: 8, ..a }, b],
        vec![
            PanelUnit { unit_id: 7, series: xy_series_regular(80, 0.1, 2) },
            PanelUnit { unit_id: 9, series: xy_series_regular(80, 0.4, 2) },
        ],
    ] {
        let changed = inspect(units);
        assert_eq!(base.identities.target, changed.identities.target);
        assert_eq!(base.identities.program, changed.identities.program);
        assert_ne!(base.identities.data_snapshot, changed.identities.data_snapshot);
    }
}

#[test]
fn panel_class_multi_step_sustained_refuses_discovery_split() {
    use common::panel_dgp::{UnitSpec, lagged_ty_dag, panel, sustained_query, unit_series};
    // The per-unit sequential fit reads every unit's full series; a split's
    // discovery rows would silently be reused for estimation.
    let spec = UnitSpec { n: 200, beta: 0.8, gz: 0.0, phi: 0.5, rho: 0.0 };
    let units = (0..3).map(|i| unit_series(spec, 13 + i)).collect();
    let err = Study::panel(panel(units))
        .graph(TemporalCpdag::from_temporal_dag(&lagged_ty_dag()))
        .temporal_query(sustained_query())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .split(DiscoveryEstimationSplit::from_sizes(200, 100, 0, 100).unwrap())
        .build()
        .and_then(|study| study.run(&ExecutionContext::for_tests(6)))
        .unwrap_err();
    assert!(err.to_string().contains("requires no discovery-estimation split"), "{err}");
}

#[test]
fn panel_class_multi_step_sustained_refuses_a_caller_prior() {
    use common::panel_dgp::{UnitSpec, lagged_ty_dag, panel, sustained_query, unit_series};
    let spec = UnitSpec { n: 200, beta: 0.8, gz: 0.0, phi: 0.5, rho: 0.0 };
    let units: Vec<_> = (0..3).map(|i| unit_series(spec, 13 + i)).collect();
    let prior = antecedent_prob::PriorSet::weakly_informative(3);
    let err = Study::panel(panel(units))
        .graph(TemporalCpdag::from_temporal_dag(&lagged_ty_dag()))
        .temporal_query(sustained_query())
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(32).prior(prior)))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .and_then(|study| study.run(&ExecutionContext::for_tests(6)))
        .unwrap_err();
    assert!(err.to_string().contains("transfer stays refused on incomplete classes"), "{err}");
}

#[test]
fn panel_class_pulse_unit_bootstrap_resamples_fresh_clusters() {
    use common::panel_dgp::{
        UnitSpec, confounded_unit, cpdag_two, lagged_ty_dag, panel, pulse_query, unit_series,
    };
    // A unit drawn twice must form two clusters. Keeping the original id made every
    // replicate with a repeated unit fail the panel SE, so only permutations of the
    // panel survived and their SD was floating-point noise (about 1e-16).
    let spec = UnitSpec { n: 150, beta: 0.8, gz: 0.0, phi: 0.5, rho: 0.0 };
    let units: Vec<_> = (0..4).map(|i| unit_series(spec, 21 + i)).collect();
    let run = |builder: antecedent::StudyBuilder| {
        builder
            .temporal_query(pulse_query())
            .refute(RefuteSuite::None)
            .bootstrap_replicates(199)
            .build()
            .unwrap()
            .run(&ExecutionContext::for_tests(3))
            .unwrap()
    };
    let class = run(Study::panel(panel(units.clone()))
        .graph(TemporalCpdag::from_temporal_dag(&lagged_ty_dag())));
    let dag = run(Study::panel(panel(units)).graph(lagged_ty_dag()));
    for result in [&class, &dag] {
        assert_eq!(result.estimate.bootstrap_replicates_ok, Some(199));
        assert_eq!(result.estimate.bootstrap_replicates_failed, Some(0));
    }
    let (class_se, dag_se) =
        (class.estimate.se_bootstrap.unwrap(), dag.estimate.se_bootstrap.unwrap());
    eprintln!("class boot={class_se} dag boot={dag_se} analytic={}", class.estimate.se_analytic);
    assert!((class_se / dag_se - 1.0).abs() < 0.2, "class={class_se} dag={dag_se}");
    let analytic = class.estimate.se_analytic;
    assert!(
        class_se > 0.5 * analytic && class_se < 2.0 * analytic,
        "bootstrap={class_se} analytic={analytic}"
    );

    // Two completions: the mixture bootstrap is the only uncertainty, and every
    // replicate refits both completions.
    let units: Vec<_> = (0..4).map(|i| confounded_unit(200, 11 + i)).collect();
    let two = run(Study::panel(panel(units)).graph(cpdag_two()));
    assert_eq!(two.estimate.bootstrap_replicates_ok, Some(199));
    let se = two.estimate.se_bootstrap.unwrap();
    eprintln!("two-completion boot={se}");
    assert!(se > 1e-3, "two-completion bootstrap SE={se}");
}

fn point_surface(response: &antecedent_core::CausalResponse) -> Vec<f64> {
    match &response.estimate {
        antecedent_core::ResponseIdentification::PointIdentified(
            antecedent_core::ResponseValue::Surface { mean, .. },
        ) => mean.to_vec(),
        other => panic!("expected a point surface, got {other:?}"),
    }
}

/// Unbiased sample SD, the one the code under test uses.
fn sample_sd(values: &[f64]) -> f64 {
    antecedent_stats::sample_std(values)
}

/// `t_2(0.975)`.
const T2_975: f64 = 4.302_652_729_911_275;

#[test]
fn panel_response_band_is_the_between_unit_t_interval() {
    use common::panel_dgp::{UnitSpec, curve_query, lagged_ty_dag, panel, unit_series};
    // Three units with different slopes: the band is mean ± t_2(0.975)·sd/sqrt(3)
    // over the unit surfaces (a z critical value covered about 0.82 at N = 3), and
    // requesting bootstrap replicates neither changes it nor relabels it.
    let units: Vec<_> = [0.4, 0.8, 1.2]
        .iter()
        .zip(40..)
        .map(|(&beta, seed)| unit_series(UnitSpec::iid(120, beta), seed))
        .collect();
    let per_unit: Vec<Vec<f64>> = units
        .iter()
        .map(|series| {
            let result = Study::series(series.clone())
                .graph(common::panel_dgp::lagged_ty_dag())
                .query(CausalQuery::Response(curve_query(vec![1])))
                .refute(RefuteSuite::None)
                .bootstrap_replicates(0)
                .build()
                .unwrap()
                .run(&ExecutionContext::for_tests(2))
                .unwrap();
            point_surface(result.response.as_ref().unwrap())
        })
        .collect();
    for boot in [0, 199] {
        let result = Study::panel(panel(units.clone()))
            .graph(lagged_ty_dag())
            .query(CausalQuery::Response(curve_query(vec![1])))
            .refute(RefuteSuite::None)
            .bootstrap_replicates(boot)
            .build()
            .unwrap()
            .run(&ExecutionContext::for_tests(2))
            .unwrap();
        let response = result.response.as_ref().unwrap();
        let mean = point_surface(response);
        let antecedent_core::ResponseUncertainty::PointwiseBand { level, lower, upper } =
            &response.uncertainty
        else {
            panic!("expected a pointwise band, got {:?}", response.uncertainty);
        };
        assert!((level - 0.95).abs() < 1e-12);
        for cell in 0..mean.len() {
            let column: Vec<f64> = per_unit.iter().map(|unit| unit[cell]).collect();
            let expected_mean = column.iter().sum::<f64>() / 3.0;
            let half = T2_975 * sample_sd(&column) / 3.0_f64.sqrt();
            assert!((mean[cell] - expected_mean).abs() < 1e-9, "{mean:?} vs {per_unit:?}");
            assert!((upper[cell] - mean[cell] - half).abs() < 1e-9, "cell {cell} boot {boot}");
            assert!((mean[cell] - lower[cell] - half).abs() < 1e-9, "cell {cell} boot {boot}");
        }
        assert_eq!(
            result.interval.as_ref().map(|interval| interval.method),
            Some(antecedent_core::IntervalMethod::AnalyticSe),
            "the band is analytic whatever the replicate request"
        );
        let has = |code: &str| result.diagnostics.iter().any(|d| d.code.as_ref() == code);
        assert!(has("estimate.temporal_response.panel.between_unit_band"));
        assert_eq!(has("estimate.temporal_response.panel.bootstrap_not_used"), boot > 0);
    }
}

#[test]
fn panel_class_sequential_se_is_the_between_unit_t_se() {
    use common::panel_dgp::{UnitSpec, lagged_ty_dag, panel, sustained_query, unit_series};
    let units: Vec<_> = [0.5, 0.8, 1.1]
        .iter()
        .zip(60..)
        .map(|(&beta, seed)| {
            unit_series(UnitSpec { n: 180, beta, gz: 0.0, phi: 0.5, rho: 0.0 }, seed)
        })
        .collect();
    let run = |builder: antecedent::StudyBuilder| {
        builder
            .graph(TemporalCpdag::from_temporal_dag(&lagged_ty_dag()))
            .temporal_query(sustained_query())
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
            .run(&ExecutionContext::for_tests(6))
            .unwrap()
    };
    let ates: Vec<f64> =
        units.iter().map(|series| run(Study::series(series.clone())).estimate.ate).collect();
    let pooled = run(Study::panel(panel(units)));
    let mean = ates.iter().sum::<f64>() / 3.0;
    let expected_se = sample_sd(&ates) / 3.0_f64.sqrt() * T2_975 / 1.959_963_984_540_054;
    assert!((pooled.estimate.ate - mean).abs() < 1e-9, "{} vs {ates:?}", pooled.estimate.ate);
    assert!(
        (pooled.estimate.se_analytic - expected_se).abs() < 1e-9,
        "se={} expected={expected_se}",
        pooled.estimate.se_analytic
    );
}

#[test]
fn panel_response_support_and_assumptions_describe_every_unit() {
    use antecedent_core::{Assumption, SupportStatus};
    use common::panel_dgp::{
        UnitSpec, curve_query, lagged_ty_dag, narrow_treatment_unit, panel, unit_series,
    };
    // Unit 0 covers dose 1; unit 2's treatment never reaches it. The panel average
    // needs every unit at every cell, so its support must say dose 1 is not
    // supported, whichever unit comes first, and must carry no unit's simultaneous
    // credible band or per-unit band assumptions.
    let units = vec![
        unit_series(UnitSpec::iid(120, 0.4), 70),
        unit_series(UnitSpec::iid(120, 0.8), 71),
        narrow_treatment_unit(120, 72),
    ];
    let series_support = |series: &TimeSeriesData| {
        Study::series(series.clone())
            .graph(lagged_ty_dag())
            .query(CausalQuery::Response(curve_query(vec![1])))
            .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(64)))
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
            .run(&ExecutionContext::for_tests(3))
            .unwrap()
            .response
            .unwrap()
            .support
    };
    let unit_supports: Vec<_> = units.iter().map(series_support).collect();
    let dose1 =
        |support: &antecedent_core::SupportReport| support.point_status.as_ref().unwrap()[1];
    assert_eq!(dose1(&unit_supports[0]), SupportStatus::Supported);
    assert_ne!(dose1(&unit_supports[2]), SupportStatus::Supported);

    let result = Study::panel(panel(units))
        .graph(lagged_ty_dag())
        .query(CausalQuery::Response(curve_query(vec![1])))
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(64)))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(3))
        .unwrap();
    let response = result.response.as_ref().unwrap();
    let support = &response.support;
    assert_eq!(dose1(support), dose1(&unit_supports[2]), "{support:?}");
    assert_ne!(support.status, SupportStatus::Supported);
    assert!(
        support.diagnostics.iter().all(|d| !d.id.starts_with("response.simultaneous_band")),
        "a unit's simultaneous credible band leaked into the panel support: {support:?}"
    );
    let range = |support: &antecedent_core::SupportReport| {
        support
            .diagnostics
            .iter()
            .find(|d| d.id.as_ref() == "response.temporal.horizon_treatment_range")
            .unwrap()
            .values
            .to_vec()
    };
    let expected_lo = unit_supports.iter().map(|s| range(s)[0]).fold(f64::NEG_INFINITY, f64::max);
    let expected_hi = unit_supports.iter().map(|s| range(s)[1]).fold(f64::INFINITY, f64::min);
    assert_eq!(range(support), vec![expected_lo, expected_hi]);
    // Every unit's support warning reaches the result.
    for warning in &unit_supports[2].warnings {
        if warning.code.as_ref().starts_with("response.simultaneous_band") {
            continue;
        }
        assert!(
            result.diagnostics.iter().any(|d| d.code == warning.code),
            "unit warning {} was dropped",
            warning.code
        );
    }
    let ids: Vec<&str> = response
        .assumptions
        .entries
        .iter()
        .filter_map(|record| match &record.assumption {
            Assumption::ParametricRestriction(p) => Some(p.id.as_ref()),
            _ => None,
        })
        .collect();
    assert!(ids.contains(&"panel.response.between_unit_band"), "{ids:?}");
    assert!(ids.contains(&"bayesian.temporal_response.panel_unit_posterior_means"), "{ids:?}");
    assert!(!ids.contains(&"bayesian.temporal_response.linear_additive"), "{ids:?}");
    assert!(!ids.contains(&"bayes.temporal.long_run_tempering"), "{ids:?}");
}

#[test]
fn bayesian_panel_response_applies_the_caller_prior_to_every_unit() {
    use antecedent_prob::{GaussianCoefficientPrior, PriorSet, PriorSpec};
    use common::panel_dgp::{UnitSpec, curve_query, lagged_ty_dag, panel, unit_series};
    // A tight prior at slope 5 against data at slope 0.8: when the prior is in
    // force every unit surface, and so the panel average, moves to it.
    let units: Vec<_> = (0..3).map(|i| unit_series(UnitSpec::iid(120, 0.8), 80 + i)).collect();
    let run = |cfg: BayesianConfig| {
        let result = Study::panel(panel(units.clone()))
            .graph(lagged_ty_dag())
            .query(CausalQuery::Response(curve_query(vec![1])))
            .inference(InferenceMode::Bayesian(cfg))
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
            .run(&ExecutionContext::for_tests(3))
            .unwrap();
        let mean = point_surface(result.response.as_ref().unwrap());
        mean[1] - mean[0]
    };
    let prior = PriorSet {
        specs: vec![PriorSpec::GaussianCoefficients(GaussianCoefficientPrior {
            mean: [0.0, 5.0].into(),
            variance: [100.0, 1e-6].into(),
        })],
        contrast: None,
        categorical: Vec::new(),
        restrictions: Vec::new(),
    };
    let flat = run(BayesianConfig::conjugate().n_draws(64));
    let informed = run(BayesianConfig::conjugate().n_draws(64).prior(prior));
    assert!((flat - 0.8).abs() < 0.3, "flat contrast {flat}");
    assert!((informed - 5.0).abs() < 0.2, "the caller prior was dropped: contrast {informed}");
}

fn atom_scalar(atom: &antecedent::result::StructuralResponseAtom) -> f64 {
    match atom.value {
        Some(antecedent_core::ResponseValue::Scalar(value)) => value,
        ref other => panic!("expected a scalar atom, got {other:?}"),
    }
}

#[test]
fn panel_class_multi_step_sustained_reports_unevaluable_bidirected_completions() {
    use antecedent_graph::TemporalPag;
    // w@1 o-o z@1 beside a directed t -> y design: the completions orient the edge
    // either way or make it bidirected. Every completion identifies the t effect
    // (adjust z@1), but sequential g-computation cannot evaluate a bidirected one;
    // its mass must be reported as unevaluable, with the series diagnostic, rather
    // than silently renormalized away.
    let v = VariableId::from_raw;
    let mut pag = TemporalPag::empty();
    let t1 = pag.add_lagged(v(0), Lag::from_raw(1)).unwrap();
    let t2 = pag.add_lagged(v(0), Lag::from_raw(2)).unwrap();
    let y0 = pag.add_lagged(v(1), Lag::CONTEMPORANEOUS).unwrap();
    let z1 = pag.add_lagged(v(2), Lag::from_raw(1)).unwrap();
    let w1 = pag.add_lagged(v(3), Lag::from_raw(1)).unwrap();
    let u1 = pag.add_lagged(v(4), Lag::from_raw(1)).unwrap();
    pag.insert_directed(t1, y0).unwrap();
    pag.insert_directed(t2, y0).unwrap();
    pag.insert_directed(z1, t1).unwrap();
    pag.insert_directed(z1, y0).unwrap();
    pag.insert_directed(u1, t1).unwrap();
    pag.insert_circle_circle_with_middle(w1, z1, antecedent_graph::MiddleMark::Unknown).unwrap();
    let mut query = TemporalEffectQuery::sustained(v(0), v(1), 0, 1.0);
    query.policy = TemporalPolicy::sustained(-2, -1);
    query.horizon_steps = 1;
    let unit = |seed: u64| {
        let mut draw = common::calibration::gaussian(seed);
        let n = 240;
        let (mut t, mut y, mut z, mut w, mut u) =
            (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        for i in 0..n {
            z[i] = draw();
            w[i] = 0.5 * z[i] + draw();
            u[i] = draw();
            t[i] = 0.4 * z[i] + 0.3 * u[i] + draw();
            y[i] = if i > 1 { 1.0 + 0.8 * t[i - 1] + 0.3 * t[i - 2] + 0.5 * z[i - 1] } else { 1.0 }
                + draw();
        }
        TimeSeriesData::from_f64_columns(
            [
                ("t", t.as_slice()),
                ("y", y.as_slice()),
                ("z", z.as_slice()),
                ("w", w.as_slice()),
                ("u", u.as_slice()),
            ],
            1,
        )
        .unwrap()
    };
    let units: Vec<_> = (0..3).map(|i| unit(90 + i)).collect();
    let run = |builder: antecedent::StudyBuilder| {
        builder
            .graph(pag.clone())
            .temporal_query(query.clone())
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
            .run(&ExecutionContext::for_tests(6))
    };
    let series = run(Study::series(units[0].clone())).unwrap();
    let series_unevaluable = series.structural_response.as_ref().unwrap().unevaluable_mass;
    eprintln!("series unevaluable={series_unevaluable}");
    assert!(series_unevaluable > 0.0, "fixture must carry a bidirected identified completion");
    let pooled = run(Study::panel(common::panel_dgp::panel(units))).unwrap();
    let structural = pooled.structural_response.as_ref().unwrap();
    assert!((structural.unevaluable_mass - series_unevaluable).abs() < 1e-12);
    assert!(
        pooled
            .diagnostics
            .iter()
            .any(|d| d.code.as_ref() == "estimate.temporal_class.mag_sequential_unevaluable"),
        "missing unevaluable-completion diagnostic"
    );
}

#[test]
fn bayesian_panel_class_multi_step_sustained_follows_the_class_prior_contract() {
    use common::panel_dgp::{UnitSpec, lagged_ty_dag, panel, sustained_query, unit_series};
    let spec = UnitSpec { n: 200, beta: 0.8, gz: 0.0, phi: 0.5, rho: 0.0 };
    let units: Vec<_> = (0..3).map(|i| unit_series(spec, 100 + i)).collect();
    let run = |prior: Option<ClassPrior>| {
        let mut builder = Study::panel(panel(units.clone()))
            .graph(TemporalCpdag::from_temporal_dag(&lagged_ty_dag()))
            .temporal_query(sustained_query())
            .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(64)))
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0);
        if let Some(prior) = prior {
            builder = builder.class_prior(prior);
        }
        builder.build().unwrap().run(&ExecutionContext::for_tests(8)).unwrap()
    };
    let enumeration = run(None);
    assert!(enumeration.estimate.ate.is_nan(), "ate={}", enumeration.estimate.ate);
    assert!(enumeration.posterior.is_none());
    assert!(
        enumeration
            .diagnostics
            .iter()
            .any(|d| d.code.as_ref() == "estimate.temporal_class.enumeration_not_probability")
    );
    let mixed = run(Some(ClassPrior::from_ordered([1.0]).unwrap()));
    let atom = mixed.structural_response.as_ref().unwrap().atoms[0].clone();
    assert!(mixed.posterior.is_some());
    assert!((mixed.estimate.ate - atom_scalar(&atom)).abs() < 0.05, "{}", mixed.estimate.ate);
    assert!(mixed.estimate.ate.is_finite() && (mixed.estimate.ate - 0.8).abs() < 0.3);
}

#[derive(Default)]
struct RecordingProgress(std::sync::Mutex<Vec<String>>);

impl antecedent_core::ProgressSink for RecordingProgress {
    fn report(&self, _fraction: f64, stage: &str) {
        self.0.lock().unwrap().push(stage.to_owned());
    }
}

#[test]
fn panel_class_multi_step_sustained_identifies_the_class_once() {
    use common::panel_dgp::{UnitSpec, lagged_ty_dag, panel, sustained_query, unit_series};
    let spec = UnitSpec { n: 160, beta: 0.8, gz: 0.0, phi: 0.5, rho: 0.0 };
    let units: Vec<_> = (0..3).map(|i| unit_series(spec, 110 + i)).collect();
    let sink = Arc::new(RecordingProgress::default());
    let mut ctx = ExecutionContext::for_tests(6);
    ctx.progress = Some(Arc::clone(&sink) as Arc<dyn antecedent_core::ProgressSink>);
    Study::panel(panel(units))
        .graph(TemporalCpdag::from_temporal_dag(&lagged_ty_dag()))
        .temporal_query(sustained_query())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap();
    let computes =
        sink.0.lock().unwrap().iter().filter(|stage| stage.as_str() == "identify.compute").count();
    assert_eq!(computes, 1, "the multi-step route must not identify the class twice");
}

#[test]
fn panel_class_routes_run_the_requested_refuters() {
    use common::panel_dgp::{
        UnitSpec, confounded_unit, cpdag_two, lagged_ty_dag, panel, pulse_query, sustained_query,
        unit_series,
    };
    // A requested refute suite runs on every completion (Pulse: the panel refuters on
    // the stacked units; multi-step: the sequential validator on each unit) instead
    // of returning no refutations.
    let units: Vec<_> = (0..3).map(|i| confounded_unit(200, 120 + i)).collect();
    let pulse = Study::panel(panel(units))
        .graph(cpdag_two())
        .temporal_query(pulse_query())
        .refute(RefuteSuite::PlaceboAndRcc)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(12))
        .unwrap();
    assert!(!pulse.refutations.is_empty());
    assert!(pulse.refutations.iter().all(|report| report.refuter.starts_with("completion.")));
    let keys: std::collections::BTreeSet<_> = pulse
        .refutations
        .iter()
        .map(|report| report.refuter.split('.').nth(1).unwrap().to_owned())
        .collect();
    assert_eq!(keys.len(), 2, "both completions are refuted: {keys:?}");

    let spec = UnitSpec { n: 200, beta: 0.8, gz: 0.0, phi: 0.5, rho: 0.0 };
    let units: Vec<_> = (0..2).map(|i| unit_series(spec, 130 + i)).collect();
    let sustained = Study::panel(panel(units))
        .graph(TemporalCpdag::from_temporal_dag(&lagged_ty_dag()))
        .temporal_query(sustained_query())
        .refute(RefuteSuite::PlaceboAndRcc)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(12))
        .unwrap();
    assert!(!sustained.refutations.is_empty());
    for unit in [0, 1] {
        assert!(
            sustained
                .refutations
                .iter()
                .any(|report| report.refuter.starts_with(&format!("unit.{unit}.completion."))),
            "unit {unit} unrefuted"
        );
    }
}

#[test]
fn panel_routes_name_their_estimand_and_refuse_short_units() {
    use antecedent_core::Assumption;
    use common::panel_dgp::{
        UnitSpec, curve_query, lagged_ty_dag, panel, pulse_query, unit_series,
    };
    // Long unit A and short unit B with different slopes: the pooled Pulse and the
    // equal-weight response contrast answer different questions, and each result
    // says which one it answers.
    let a = unit_series(UnitSpec::iid(400, 0.2), 160);
    let b = unit_series(UnitSpec::iid(40, 1.4), 161);
    let ids = |assumptions: &antecedent_core::AssumptionSet| {
        assumptions
            .entries
            .iter()
            .filter_map(|record| match &record.assumption {
                Assumption::ParametricRestriction(p) => Some(p.id.to_string()),
                _ => None,
            })
            .collect::<Vec<_>>()
    };
    let pulse = Study::panel(panel(vec![a.clone(), b.clone()]))
        .graph(lagged_ty_dag())
        .temporal_query(pulse_query())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(1))
        .unwrap();
    assert!(
        pulse
            .diagnostics
            .iter()
            .any(|d| d.code.as_ref() == "estimate.temporal_effect.panel.estimand"
                && d.message.contains("pooled common-coefficient"))
    );
    assert!(
        ids(&pulse.estimate.assumptions)
            .contains(&"panel.estimand.pooled_common_coefficient".to_owned())
    );

    let response = Study::panel(panel(vec![a.clone(), b.clone()]))
        .graph(lagged_ty_dag())
        .query(CausalQuery::Response(curve_query(vec![1])))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(1))
        .unwrap();
    let surface = point_surface(response.response.as_ref().unwrap());
    let contrast = surface[1] - surface[0];
    assert!(response.diagnostics.iter().any(|d| {
        d.code.as_ref() == "estimate.temporal_response.panel.estimand"
            && d.message.contains("equal-weight average")
    }));
    assert!(
        ids(&response.response.as_ref().unwrap().assumptions)
            .contains(&"panel.estimand.equal_weight_unit_average".to_owned())
    );
    // The two estimands really differ on this panel: the pooled effect leans to the
    // long unit, the unit average sits midway.
    assert!((pulse.estimate.ate - 0.2).abs() < 0.2, "pooled ate={}", pulse.estimate.ate);
    assert!((contrast - 0.8).abs() < 0.3, "unit-average contrast={contrast}");

    // A unit too short to fit on its own is refused, not averaged with weight 1/N.
    let short = unit_series(UnitSpec::iid(8, 0.8), 162);
    let err = Study::panel(panel(vec![a, b, short]))
        .graph(lagged_ty_dag())
        .query(CausalQuery::Response(curve_query(vec![1])))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(1))
        .unwrap_err();
    assert!(err.to_string().contains("lag-aligned rows, below the"), "{err}");
}

#[test]
fn multi_env_pulse_pools_every_environment() {
    use antecedent_data::MultiEnvironmentData;
    use common::panel_dgp::{UnitSpec, lagged_ty_dag, pulse_query, unit_series};
    // Two environments with different slopes: the effect must not be environment 0's,
    // and it must not depend on the argument order.
    let first = unit_series(UnitSpec::iid(300, 0.3), 170);
    let second = unit_series(UnitSpec::iid(300, 2.1), 171);
    let run = |series: [TimeSeriesData; 2]| {
        Study::series_multi(MultiEnvironmentData::try_new(Arc::from(series)).unwrap())
            .graph(lagged_ty_dag())
            .temporal_query(pulse_query())
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
            .run(&ExecutionContext::for_tests(2))
            .unwrap()
    };
    let one = |series: TimeSeriesData| {
        Study::series(series)
            .graph(lagged_ty_dag())
            .temporal_query(pulse_query())
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
            .run(&ExecutionContext::for_tests(2))
            .unwrap()
            .estimate
            .ate
    };
    let (a, b) = (one(first.clone()), one(second.clone()));
    assert!((b - a).abs() > 1.0, "the environments must disagree: {a} vs {b}");
    let forward = run([first.clone(), second.clone()]);
    let reversed = run([second, first]);
    assert!(
        (forward.estimate.ate - reversed.estimate.ate).abs() < 1e-9,
        "argument order changed the answer: {} vs {}",
        forward.estimate.ate,
        reversed.estimate.ate
    );
    assert!(
        forward.estimate.ate > a.min(b) + 0.2 && forward.estimate.ate < a.max(b) - 0.2,
        "the pooled effect must use both environments: {} (envs {a}, {b})",
        forward.estimate.ate
    );
    assert_eq!(forward.logical_plan.data_classification, DataClassification::MultiEnvironment);
    assert!(
        forward.diagnostics.iter().any(|d| {
            d.code.as_ref() == "estimate.temporal_effect.panel.estimand"
                && d.message.contains("environment")
        }),
        "the pooled multi-environment estimand must be named"
    );
    assert!(forward.estimate.se_analytic.is_finite());
}
