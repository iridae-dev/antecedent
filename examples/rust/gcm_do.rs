//! Draw from a fitted causal model under an intervention, as a compiled simulation route.
//!
//! Question: what does the fitted model generate when `t` is set to 1 for every unit?
//!
//! Data and truth: 120 rows with `z = 0.01 i`, `t = 1` exactly when `z > 0.5`, and
//! `y = 1 + 2 t + z` plus a small deterministic ripple. A study with a `Simulate` query on the
//! supplied DAG fits one mechanism per node (held-out family selection) and draws from the model
//! under `do(t = 1)`, returning the draws with the selection record of every node.
//!
//! Asserted: 50 draws are returned, one per requested draw; the selection records of all three
//! nodes are published; and every draw of `t` equals 1.
//!
//! What this establishes: the hard intervention replaces the mechanism of `t` in every draw, and
//! the route reports which mechanism family it selected for each node. The draws describe the
//! fitted model. The example does not compare the sampled distribution of `y` with the
//! generating equation.
//!
//! Run: `cargo run -p antecedent --example gcm_do`
//! Source: `examples/rust/gcm_do.rs`

use std::sync::Arc;

use antecedent::prelude::*;
use antecedent::{CompiledExecution, SimulationOutcome, Study};
use antecedent_core::{CausalQuery, Intervention, SimulateQuery, SimulationTarget, Value};

/// Runs the example end to end; `main` calls it and the example test suite runs it.
#[allow(clippy::cast_precision_loss)]
pub fn run() -> Result<(), CausalError> {
    let n = 120usize;
    let z: Vec<f64> = (0..n).map(|i| (i as f64) * 0.01).collect();
    let t: Vec<f64> = z.iter().map(|&zi| if zi > 0.5 { 1.0 } else { 0.0 }).collect();
    let y: Vec<f64> = t
        .iter()
        .zip(z.iter())
        .enumerate()
        .map(|(i, (&ti, &zi))| 1.0 + 2.0 * ti + zi + 0.05 * (i as f64 * 1.7).sin())
        .collect();

    let schema = CausalSchemaBuilder::new()
        .continuous("t")
        .treatment()
        .continuous("y")
        .outcome()
        .continuous("z")
        .context()
        .build()?;
    let data = TabularData::try_from_schema_f64(
        schema.clone(),
        [("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())],
    )?;
    let dag = Dag::from_named_edges(&schema, &[("z", "t"), ("z", "y"), ("t", "y")])?;
    let ctx = ExecutionContext::for_tests(1);
    // `do(t = 1)`: replace the mechanism of t with the constant 1; z keeps its own mechanism and
    // y is regenerated from the intervened t.
    let interventions: Arc<[Intervention]> =
        Arc::from([Intervention::set(schema.id_of("t")?, Value::f64(1.0))]);
    let query = CausalQuery::Simulate(SimulateQuery::new(SimulationTarget::InterventionalSample {
        interventions,
        draws: 50,
        seed: 1,
    }));
    let study = Study::tabular(data).graph(dag).query(query).build()?;
    // The compiled route fits the mechanisms and samples in one execution, so the result
    // carries the selection record of the model the draws came from.
    let execution = CompiledExecution::run(&study, &ctx)?;
    let Some(payload) = execution.result().simulation() else {
        panic!("a simulation query returns a simulation");
    };
    let SimulationOutcome::Sample(draws) = &payload.outcome else {
        panic!("an interventional sample returns draws");
    };
    println!("do(t=1) sample rows = {}", draws.n_rows);
    assert_eq!(draws.n_rows, 50, "one row per requested interventional draw");
    assert_eq!(payload.mechanisms.len(), 3, "the selection record of every node is published");
    let t = draws.column(usize::try_from(schema.id_of("t")?.raw()).unwrap_or(0))?;
    assert!(t.iter().all(|&ti| (ti - 1.0).abs() < 1e-12), "do(t=1) pins every draw of t");
    Ok(())
}

fn main() -> Result<(), CausalError> {
    run()
}
