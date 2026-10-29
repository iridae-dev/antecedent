// SPDX-License-Identifier: MIT OR Apache-2.0
//! Declarations the sustained-window route cannot honour are refused or named, never ignored.
//!
//! The sustained contrast is indexed by a lag and a horizon, so a static graph class has no
//! time index for it; its treatment is a numeric series, so a categorical-typed column is
//! refused; and an intervention level outside the observed treatment range is reached only by
//! linear extrapolation, which the support slot names. On a dependent series with AR(2)
//! treatment and residual the single-step sustained posterior equals the independent
//! tempered reference of the same estimand.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study, StudyResult};
use antecedent_core::{
    CategoryDomainId, CausalQuery, CausalSchemaBuilder, ExecutionContext, Lag, MeasurementSpec,
    RoleHint, SmallRoleSet, TemporalEffectQuery, TemporalPolicy, ValueType, VariableId,
};
use antecedent_data::{
    CategoricalColumn, CategoryCode, CategoryDomain, CategoryLevel, Float64Column, OwnedColumn,
    OwnedColumnarStorage, SamplingRegularity, TimeIndex, TimeSeriesData, UnknownCategoryPolicy,
    ValidityBitmap,
};
use antecedent_graph::{Admg, Cpdag, Dag, DenseNodeId, TemporalDag, ensure_lagged};

const N: usize = 200;

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn wiggle(i: usize, rate: f64) -> f64 {
    (i as f64 * rate).sin()
}

fn lag1_graph() -> TemporalDag {
    let mut graph = TemporalDag::empty();
    let t1 = ensure_lagged(&mut graph, v(0), Lag::from_raw(1)).unwrap();
    let y0 = ensure_lagged(&mut graph, v(1), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(t1, y0).unwrap();
    graph
}

fn lag12_graph() -> TemporalDag {
    let mut graph = lag1_graph();
    let t2 = ensure_lagged(&mut graph, v(0), Lag::from_raw(2)).unwrap();
    let y0 = ensure_lagged(&mut graph, v(1), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(t2, y0).unwrap();
    graph
}

fn outcome_of(t: &[f64], second_lag: f64) -> Vec<f64> {
    let mut y = vec![0.0; t.len()];
    for i in 2..t.len() {
        y[i] = 2.0 * t[i - 1] + second_lag * t[i - 2] + 0.3 * wiggle(i, 0.13);
    }
    y
}

fn series(t: &[f64], second_lag: f64) -> TimeSeriesData {
    let y = outcome_of(t, second_lag);
    TimeSeriesData::from_f64_columns([("t", t), ("y", y.as_slice())], 1).unwrap()
}

fn sustained(level: f64, until: i32) -> CausalQuery {
    CausalQuery::TemporalEffect(
        TemporalEffectQuery::sustained(v(0), v(1), until, level)
            .with_policy(TemporalPolicy::sustained(until, -1))
            .with_horizon_steps(1),
    )
}

fn run(
    data: TimeSeriesData,
    graph: TemporalDag,
    query: CausalQuery,
) -> Result<StudyResult, String> {
    Study::series(data)
        .graph(graph)
        .query(query)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .map_err(|e| e.to_string())
        .and_then(|study| study.run(&ExecutionContext::for_tests(5)).map_err(|e| e.to_string()))
}

#[test]
fn cells_00_sustained_contrast_is_refused_on_every_static_graph_class() {
    let t: Vec<f64> = (0..N).map(|i| wiggle(i, 0.31)).collect();
    // The same query on a temporal graph runs: the refusal is of a route that works.
    run(series(&t, 0.5), lag1_graph(), sustained(1.0, -1)).unwrap();
    let node = DenseNodeId::from_raw;
    for class in ["dag", "admg", "cpdag"] {
        let builder = Study::series(series(&t, 0.5)).query(sustained(1.0, -1));
        let builder = match class {
            "dag" => {
                let mut graph = Dag::with_variables(2);
                graph.insert_directed(node(0), node(1)).unwrap();
                builder.graph(graph)
            }
            "admg" => {
                let mut graph = Admg::with_variables(2);
                graph.insert_directed(node(0), node(1)).unwrap();
                builder.graph(graph)
            }
            _ => {
                let mut graph = Cpdag::with_variables(2);
                graph.insert_directed(node(0), node(1)).unwrap();
                builder.graph(graph)
            }
        };
        let error = builder
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .map_err(|e| e.to_string())
            .and_then(|study| study.run(&ExecutionContext::for_tests(5)).map_err(|e| e.to_string()))
            .expect_err(&format!("{class}: a sustained contrast needs a temporal graph"));
        assert!(error.contains("not_applicable"), "{class}: {error}");
        assert!(error.contains("static graph class has no time index"), "{class}: {error}");
    }
}

fn categorical_treatment_series() -> TimeSeriesData {
    let codes: Vec<u32> = (0..N).map(|i| (i % 3) as u32).collect();
    let numeric: Vec<f64> = codes.iter().map(|&c| f64::from(c)).collect();
    let y = outcome_of(&numeric, 0.5);
    let mut builder = CausalSchemaBuilder::new();
    builder
        .add_variable(
            "t",
            ValueType::Categorical,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            Some(CategoryDomainId::from_raw(1)),
            MeasurementSpec::default(),
        )
        .unwrap();
    builder
        .add_variable(
            "y",
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    let domain = CategoryDomain::try_new(
        CategoryDomainId::from_raw(1),
        ["none", "low", "high"]
            .into_iter()
            .map(|label| CategoryLevel { label: Arc::from(label) })
            .collect::<Vec<_>>(),
        false,
        Some(CategoryCode::from_raw(0)),
        UnknownCategoryPolicy::Fail,
    )
    .unwrap();
    let columns = vec![
        OwnedColumn::Categorical(
            CategoricalColumn::try_new(
                v(0),
                codes.iter().map(|&c| CategoryCode::from_raw(c)).collect::<Vec<_>>(),
                ValidityBitmap::all_valid(N),
                Arc::new(domain),
            )
            .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(v(1), Arc::<[f64]>::from(y), ValidityBitmap::all_valid(N)).unwrap(),
        ),
    ];
    let storage =
        OwnedColumnarStorage::try_new(builder.build().unwrap(), columns, None, None).unwrap();
    TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: N },
    )
    .unwrap()
}

#[test]
fn cells_00_sustained_window_refuses_a_categorical_typed_treatment_column() {
    for until in [-1, -2] {
        let graph = if until == -1 { lag1_graph() } else { lag12_graph() };
        let error = run(categorical_treatment_series(), graph, sustained(1.0, until))
            .expect_err("a categorical treatment column must be refused, not read as a dose");
        assert!(error.contains("not of type float64"), "until {until}: {error}");
    }
}

fn level_support_warnings(result: &StudyResult) -> Vec<String> {
    result
        .support_diagnostics()
        .filter(|d| &*d.code == "estimate.temporal.window.level_outside_support")
        .map(|d| d.message.to_string())
        .collect()
}

#[test]
fn cells_00_sustained_level_beyond_the_observed_treatment_range_is_named_in_the_support_slot() {
    // The treatment is observed on [-0.25, 0.25]; the lag coefficients are 2 and 0.5.
    let t: Vec<f64> = (0..N).map(|i| 0.25 * wiggle(i, 0.31)).collect();
    for (until, graph_of, second_lag) in [
        (-1, lag1_graph as fn() -> TemporalDag, 0.0),
        (-2, lag12_graph as fn() -> TemporalDag, 0.5),
    ] {
        let slope = 2.0 + second_lag;
        let inside = run(series(&t, second_lag), graph_of(), sustained(0.2, until)).unwrap();
        assert!(level_support_warnings(&inside).is_empty(), "until {until}: inside the range");
        let held_out = run(series(&t, second_lag), graph_of(), sustained(5.0, until)).unwrap();
        let warnings = level_support_warnings(&held_out);
        assert_eq!(warnings.len(), 1, "until {until}: {warnings:?}");
        assert!(warnings[0].contains("active intervention level 5"), "{}", warnings[0]);
        // The extrapolated contrast is the fitted lag coefficients times the level.
        let ate = held_out.estimate().as_effect().unwrap().ate;
        assert!((ate - 5.0 * slope).abs() < 0.05 * 5.0 * slope, "until {until}: {ate}");
    }
}

#[test]
fn cells_00_single_step_sustained_posterior_on_a_dependent_series_matches_the_tempered_reference() {
    let fixture: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../conformance/bayesian/posterior_moments/expected.json"
        ))
        .unwrap(),
    )
    .unwrap();
    let column = |name: &str| -> Vec<f64> {
        fixture["data"][name].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect()
    };
    let data = TimeSeriesData::from_f64_columns(
        [("x", column("td").as_slice()), ("y", column("yd").as_slice())],
        1,
    )
    .unwrap();
    let pinned = &fixture["tempering"]["dependent_pulse"][0];
    for scale in [0.1, 10.0] {
        let query = CausalQuery::TemporalEffect(
            TemporalEffectQuery::sustained(v(0), v(1), -1, 1.0)
                .with_policy(TemporalPolicy::sustained(-1, -1))
                .with_horizon_steps(1),
        );
        let result = Study::series(data.clone())
            .graph(lag1_graph())
            .query(query)
            .inference(InferenceMode::Bayesian(
                BayesianConfig::conjugate().n_draws(8192).prior_scale(scale),
            ))
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
            .run(&ExecutionContext::for_tests(1202))
            .unwrap();
        let post = result.posterior.as_ref().unwrap();
        let expected = &fixture["posterior"][format!("{scale:.1}")]["dependent_pulse"];
        let (mean, var) = (expected[0].as_f64().unwrap(), expected[1].as_f64().unwrap());
        let draws = post.draws.column(post.effect_column().unwrap()).unwrap();
        let n = draws.len() as f64;
        let draw_mean = draws.iter().sum::<f64>() / n;
        let draw_var = draws.iter().map(|d| (d - draw_mean).powi(2)).sum::<f64>() / (n - 1.0);
        assert!(
            (draw_mean - mean).abs() < 0.08 * var.sqrt(),
            "scale {scale}: mean {draw_mean} vs {mean}"
        );
        assert!((draw_var / var - 1.0).abs() < 0.06, "scale {scale}: variance {draw_var} vs {var}");
        let kappa = antecedent_estimate::tempering_kappa_from_notes(&post.diagnostics.notes)
            .expect("tempering note");
        let pinned_kappa = pinned["kappa"].as_f64().unwrap();
        assert!((kappa - pinned_kappa).abs() < 1e-5, "kappa {kappa} vs pinned {pinned_kappa}");
        assert!(kappa > 4.0, "the AR(2) design must be tempered well beyond the AR(1) value");
    }
}
