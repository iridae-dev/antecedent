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
//! Propensity-weighting (IPW) analyze example.
//!
//! Confounded SCM: `Z ~ N(0,1)`, `T ~ Bernoulli(sigmoid(-0.4 + 0.9 Z))`,
//! `Y = 2T + Z + noise`. True ATE = 2.
//!
//! Run: `cargo run -p antecedent --example propensity_weighting`

use std::sync::Arc;

use antecedent::RefuteSuite;
use antecedent::prelude::*;
use antecedent_core::{CausalRng, MeasurementSpec, RoleHint, SmallRoleSet, ValueType};
use antecedent_data::{Float64Column, OwnedColumn, OwnedColumnarStorage, ValidityBitmap};
use antecedent_graph::DenseNodeId;

fn confounded_scm(n: usize, seed: u64) -> (TabularData, Dag, AverageEffectQuery) {
    let mut rng = CausalRng::from_seed(seed);
    let mut z = vec![0.0; n];
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    for i in 0..n {
        let u1 = rng.next_f64().max(1e-12);
        let u2 = rng.next_f64();
        let zi = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        let p = 1.0 / (1.0 + (-(-0.4 + 0.9 * zi)).exp());
        let ti = if rng.next_f64() < p { 1.0 } else { 0.0 };
        let e = (-2.0 * rng.next_f64().max(1e-12).ln()).sqrt()
            * (2.0 * std::f64::consts::PI * rng.next_f64()).cos()
            * 0.4;
        z[i] = zi;
        t[i] = ti;
        y[i] = 2.0 * ti + zi + e;
    }

    let mut b = CausalSchemaBuilder::new();
    b.add_variable(
        "t",
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
    b.add_variable(
        "z",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::Context),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    let schema = b.build().unwrap();
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(0), Arc::from(t), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(1), Arc::from(y), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(2), Arc::from(z), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
    ];
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    let mut dag = Dag::with_variables(3);
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    (TabularData::new(storage), dag, query)
}

fn main() -> Result<(), CausalError> {
    let (data, graph, query) = confounded_scm(1200, 7);
    let result = CausalAnalysis::builder()
        .data(data)
        .graph(graph)
        .query(query)
        .identifier("backdoor.adjustment")
        .estimator("propensity.weighting")
        .bootstrap_replicates(30)
        .refute(RefuteSuite::None)
        .build()?
        .run(&ExecutionContext::for_tests(11))?;

    let report = result.estimate.overlap_report.as_ref();
    let ess = report.and_then(|r| r.ess);
    let pmin = report.map(|r| r.propensity_min);
    println!(
        "ATE={:.4} method={} estimator={} overlap_ess={:?} overlap_propensity_min={:?}",
        result.estimate.ate,
        result.estimand.method,
        result.logical_plan.estimator.as_deref().unwrap_or("?"),
        ess,
        pmin,
    );
    assert!((result.estimate.ate - 2.0).abs() < 0.35, "ate={}", result.estimate.ate);
    assert_eq!(result.logical_plan.estimator.as_deref(), Some("propensity.weighting"));
    assert!(ess.is_some());
    Ok(())
}
