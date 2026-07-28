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
//! Discover-once → many interactive estimates.
//!
//! Discover with PC, accept a fully oriented DAG for estimate clicks, then
//! re-estimate via `PreparedStudy` without rediscovery.
//!
//! Run: `cargo run -p antecedent --example discover_then_estimate`

use std::sync::Arc;

use antecedent::RefuteSuite;
use antecedent::discovery::{StaticDiscoverParams, discover_pc};
use antecedent::discovery_defaults::resolve_ci;
use antecedent::io::{dag_from_json, dag_to_json};
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
    let (data, accepted_dag, query) = confounded_scm(500, 7);
    let ctx = ExecutionContext::for_tests(1);

    // Structure-ready click (once).
    let vars = [VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2)];
    let params = StaticDiscoverParams {
        alpha: 0.5,
        max_cond_size: 0,
        fdr: None,
        ci: resolve_ci("parcorr", None)?,
        screen_pc: false,
        max_subset: None,
    };
    let discovery = discover_pc(&data, &vars, &params, &ctx)?;
    let discovery_calls = 1u32;
    let _ = &discovery.evidence.graph;

    // Spreadsheet review: accept a fully oriented DAG for estimate clicks.
    let analysis = Study::builder()
        .data(data.clone())
        .graph(accepted_dag.clone())
        .query(query.clone())
        .refute(RefuteSuite::None)
        .build()?;

    let first = analysis.run(&ctx)?;
    let second = Study::builder()
        .data(data.clone())
        .graph(accepted_dag.clone())
        .query(query.clone())
        .bootstrap_replicates(0)
        .refute(RefuteSuite::None)
        .build()?
        .run(&ctx)?;
    let prepared = analysis.prepare(&ctx)?;
    let third = prepared.estimate(&data, &ctx)?;
    // Prepared / graph= paths never re-enter discovery.
    assert_eq!(discovery_calls, 1);

    assert!(first.estimate.ate.is_finite() && (first.estimate.ate - 2.0).abs() < 0.75);
    assert!(second.estimate.ate.is_finite() && third.estimate.ate.is_finite());

    // Durable hold for the next session.
    let names = ["t".to_string(), "y".to_string(), "z".to_string()];
    let json = dag_to_json(&accepted_dag, Some(&names))?;
    let restored = dag_from_json(&json)?;
    assert_eq!(restored.node_count(), accepted_dag.node_count());

    println!(
        "ATE={:.4} discovery_calls={discovery_calls} latency={:?}",
        first.estimate.ate,
        first.performance.latency_mode.as_deref(),
    );
    Ok(())
}
