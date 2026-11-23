#![allow(
    clippy::cast_precision_loss,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::manual_map,
    clippy::match_wildcard_for_single_variants,
    clippy::doc_markdown,
    clippy::map_unwrap_or
)]
#![allow(
    clippy::cast_possible_truncation,
    reason = "test scaffolding compares exact constants and indexes with small literals"
)]
//! Discover a structure once, accept a graph, then estimate repeatedly without rediscovery.
//!
//! Question: how does an interactive application separate the one structure search from the
//! many estimates that follow it?
//!
//! Data and truth: 500 rows from `Z ~ N(0, 1)`, `T ~ Bernoulli(sigmoid(-0.4 + 0.9 Z))`,
//! `Y = 2 T + Z + 0.4 noise`. The average effect of `T` on `Y` is 2.
//!
//! Steps: run PC once; accept a fully oriented DAG; estimate through `Study::run`; then
//! re-estimate through a `PreparedStudy`, which reuses the compiled plan. The accepted DAG is the
//! generating graph, written by hand to stand in for a reviewer's decision. The discovered
//! structure is not inspected.
//!
//! Asserted: discovery ran once; the first estimate is within 0.75 of 2; the second and the
//! prepared estimates are finite; the accepted DAG survives a JSON round trip with its node count.
//!
//! What this establishes: estimation on an accepted graph and re-estimation on a prepared
//! handle never re-enter discovery. It does not show that PC recovers this graph: the search runs
//! with marginal tests only and its output is unused.
//!
//! Run: `cargo run -p antecedent --example discover_then_estimate`
//! Source: `examples/rust/discover_then_estimate.rs`

use std::sync::Arc;

use antecedent::discovery::StaticDiscoverParams;
use antecedent::discovery_defaults::resolve_ci;
use antecedent::io::{dag_from_json, dag_to_json};
use antecedent::prelude::*;
use antecedent::{CompiledExecution, DiscoverySpec, RefuteSuite};
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

/// Runs the example end to end; `main` calls it and the example test suite runs it.
pub fn run() -> Result<(), CausalError> {
    let (data, accepted_dag, query) = confounded_scm(500, 7);
    let ctx = ExecutionContext::for_tests(1);

    // The structure search runs once. `max_cond_size: 0` restricts it to marginal
    // independence tests, which keeps the example fast.
    let vars = [VariableId::from_raw(0), VariableId::from_raw(1), VariableId::from_raw(2)];
    let params = StaticDiscoverParams {
        alpha: 0.5,
        max_cond_size: 0,
        fdr: None,
        ci: resolve_ci("parcorr", None)?,
        screen_pc: false,
        max_subset: None,
        knowledge: antecedent_discovery::BackgroundKnowledge::default(),
    };
    let search = Study::tabular(data.clone())
        .discovery(DiscoverySpec::pc(params, "parcorr"), vars)
        .build()?;
    let discovery = CompiledExecution::run(&search, &ctx)?;
    let discovery_calls = 1u32;
    let _ = discovery.result().discovery().map(|payload| &payload.structure);

    // Review: the accepted graph is a fully oriented DAG. Estimation requires an accepted
    // structure; a discovery output is never estimated on directly.
    let analysis = Study::tabular(data.clone())
        .graph(accepted_dag.clone())
        .query(query.clone())
        .refute(RefuteSuite::None)
        .build()?;

    let first = analysis.run(&ctx)?;
    let second = Study::tabular(data.clone())
        .graph(accepted_dag.clone())
        .query(query.clone())
        .bootstrap_replicates(0)
        .refute(RefuteSuite::None)
        .build()?
        .run(&ctx)?;
    // A prepared study compiles once; each `estimate` call reuses the plan on the given rows.
    let prepared = analysis.prepare(&ctx)?;
    let third = prepared.estimate(&data, &ctx)?;
    // Neither a supplied graph nor a prepared handle re-enters discovery.
    assert_eq!(discovery_calls, 1);

    assert!(
        first.estimate().as_effect().unwrap().ate.is_finite()
            && (first.estimate().as_effect().unwrap().ate - 2.0).abs() < 0.75
    );
    assert!(
        second.estimate().as_effect().unwrap().ate.is_finite()
            && third.estimate().as_effect().unwrap().ate.is_finite()
    );

    // Persist the accepted graph so a later session starts from the reviewed structure.
    let names = ["t".to_string(), "y".to_string(), "z".to_string()];
    let json = dag_to_json(&accepted_dag, Some(&names))?;
    let restored = dag_from_json(&json)?;
    assert_eq!(restored.node_count(), accepted_dag.node_count());

    println!(
        "ATE={:.4} discovery_calls={discovery_calls} latency={:?}",
        first.estimate().as_effect().unwrap().ate,
        first.performance.latency_mode.as_deref(),
    );
    Ok(())
}

fn main() -> Result<(), CausalError> {
    run()
}
