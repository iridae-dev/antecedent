#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::manual_map,
    clippy::match_wildcard_for_single_variants,
    clippy::doc_markdown,
    clippy::map_unwrap_or
)]
//! Sales spreadsheet E2E: Bayesian ATE → path → ITE + temporal pulse.
//!
//! Mirrors the interactive UX spine (ADR 0011):
//!   accepted DAG → Bayesian ATE → path-specific decompose → unit ITE
//!   plus a temporal pulse Bayesian block on a held `TemporalDag`.
//!
//! Run: `cargo run -p antecedent --example sales_spreadsheet_e2e`

use std::sync::Arc;

use antecedent::RefuteSuite;
use antecedent::gcm::{attribute_path_specific, counterfactual_ite, fit_gcm};
use antecedent::prelude::*;
use antecedent_core::{
    CausalRng, Lag, MeasurementSpec, PathSpecificEffectQuery, RoleHint, SmallRoleSet,
    TemporalPolicy, ValueType,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex, ValidityBitmap,
};
use antecedent_graph::{DenseNodeId, ensure_lagged};

fn sales_static(n: usize, seed: u64) -> (TabularData, Dag) {
    let mut rng = CausalRng::from_seed(seed);
    let mut gauss = || {
        let u1 = rng.next_f64().max(1e-12);
        let u2 = rng.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };
    let mut z = vec![0.0; n];
    let mut t = vec![0.0; n];
    let mut m = vec![0.0; n];
    let mut y = vec![0.0; n];
    for i in 0..n {
        let zi = gauss();
        let ti = 0.7 * zi + gauss();
        let mi = 0.6 * ti + 0.3 * zi + 0.2 * gauss();
        let yi = 1.2 * ti + 0.8 * mi + 0.5 * zi + 0.3 * gauss();
        z[i] = zi;
        t[i] = ti;
        m[i] = mi;
        y[i] = yi;
    }

    let mut b = CausalSchemaBuilder::new();
    for (name, role) in [
        ("z", RoleHint::Context),
        ("t", RoleHint::TreatmentCandidate),
        ("m", RoleHint::Context),
        ("y", RoleHint::OutcomeCandidate),
    ] {
        b.add_variable(
            name,
            ValueType::Continuous,
            SmallRoleSet::from_hint(role),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    }
    let schema = b.build().unwrap();
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(0), Arc::from(z), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(1), Arc::from(t), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(2), Arc::from(m), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(3), Arc::from(y), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
    ];
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    let mut dag = Dag::with_variables(4);
    // z→t, z→m, z→y, t→m, t→y, m→y
    for (a, b) in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)] {
        dag.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    (TabularData::new(storage), dag)
}

fn sales_temporal(n: usize, seed: u64) -> (TimeSeriesData, TemporalDag, TemporalEffectQuery) {
    let mut rng = CausalRng::from_seed(seed);
    let mut gauss = || {
        let u1 = rng.next_f64().max(1e-12);
        let u2 = rng.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };
    let mut x = vec![0.0; n];
    let mut y = vec![0.0; n];
    for i in 0..n {
        x[i] = ((i as f64) * 0.04).sin() + 0.1 * gauss();
        if i > 0 {
            y[i] = 0.85 * x[i - 1] + 0.05 * gauss();
        }
    }

    let mut b = CausalSchemaBuilder::new();
    b.add_variable(
        "promo",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    b.add_variable(
        "returns",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    let schema = b.build().unwrap();
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
    let series = TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap();

    let mut g = TemporalDag::empty();
    let p1 = ensure_lagged(&mut g, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let r0 = ensure_lagged(&mut g, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    g.insert_directed(p1, r0).unwrap();
    let q = TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
        .with_policy(TemporalPolicy::pulse(-1))
        .with_horizon_steps(1);
    (series, g, q)
}

fn main() -> Result<(), CausalError> {
    let (data, dag) = sales_static(400, 7);
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(1), VariableId::from_raw(3));

    let bayes = Study::tabular(data.clone())
        .graph(dag.clone())
        .query(query.clone())
        .inference(InferenceMode::Bayesian(BayesianConfig::laplace().n_draws(128)))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()?
        .run(&ExecutionContext::for_tests(3))?;
    assert!(bayes.estimate.ate.is_finite(), "ate={}", bayes.estimate.ate);
    assert!(bayes.posterior.is_some());
    println!("Bayesian ATE={:.4} (campaign → revenue)", bayes.estimate.ate);

    let fitted = fit_gcm(dag.clone(), &data)?;
    let path_q = PathSpecificEffectQuery::binary(VariableId::from_raw(1), VariableId::from_raw(3))
        .with_path_nodes([VariableId::from_raw(2)]);
    let path = attribute_path_specific(&fitted.model, &path_q, &ExecutionContext::for_tests(5))?;
    assert!(path.total_change.is_finite());
    println!(
        "Path decompose total_change={:.4} paths={}",
        path.total_change,
        path.path_breakdown.len()
    );

    let ite = counterfactual_ite(
        fitted.model,
        &data,
        VariableId::from_raw(1),
        VariableId::from_raw(3),
        1.0,
        0.0,
        &ExecutionContext::for_tests(7),
    )?;
    assert_eq!(ite.unit_effects.len(), 400);
    assert!(ite.mean_ite.is_finite());
    println!("ITE mean={:.4} n={}", ite.mean_ite, ite.unit_effects.len());

    // Second estimate click — still no discovery (graph supplied).
    let _ = Study::tabular(data)
        .graph(dag)
        .query(query)
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(64)))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()?
        .run(&ExecutionContext::for_tests(4))?;

    let (series, tdag, pulse_q) = sales_temporal(350, 11);
    let pulse = Study::series(series)
        .graph(tdag)
        .temporal_query(pulse_q)
        .inference(InferenceMode::Bayesian(BayesianConfig::laplace().n_draws(96)))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()?
        .run(&ExecutionContext::for_tests(13))?;
    assert!(pulse.estimate.ate.is_finite(), "pulse={}", pulse.estimate.ate);
    println!("Temporal pulse Bayesian ATE={:.4} (promo → returns)", pulse.estimate.ate);
    assert!((pulse.estimate.ate - 0.85).abs() < 0.25, "pulse={}", pulse.estimate.ate);

    println!("sales_spreadsheet_e2e: ok");
    Ok(())
}
