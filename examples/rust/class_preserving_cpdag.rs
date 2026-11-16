//! Estimate on a CPDAG without completing it: the graph stays a `Cpdag`.
//!
//! Question: what can be said about the effect of `t` on `y` when the direction of the `z - t`
//! edge is not known?
//!
//! Data and truth: 200 noise-free rows with `z` evenly spaced on `[0, 1)`, `t = 1` exactly when
//! `z > 0.5`, and `y = 1 + 2 t + 3 z`. The partial CPDAG has `z -> y`, `t -> y` and an undirected
//! `z - t`. Its two members disagree: with `z -> t` the effect is the adjusted 2; with `t -> z`
//! it is the unadjusted contrast, 3.5.
//!
//! Asserted: the partial CPDAG returns 2.75, the equal-weight mean over the two members; the
//! fully oriented CPDAG returns 2.
//!
//! What this establishes: a partial CPDAG estimates an envelope over its Markov equivalence
//! class instead of one silently chosen member, and a fully oriented CPDAG stays a `Cpdag`.
//! The value 2.75 summarizes the class; it is the effect under neither member. Completing the
//! graph yourself and supplying a `Dag` is a different support cell.
//!
//! Run: `cargo run -p antecedent --example class_preserving_cpdag`
//! Source: `examples/rust/class_preserving_cpdag.rs`

#![allow(clippy::cast_precision_loss)]
#![allow(
    clippy::cast_possible_truncation,
    reason = "test scaffolding compares exact constants and indexes with small literals"
)]

use antecedent::graph::Cpdag;
use antecedent::prelude::*;

fn confounded(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let z: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
    let t: Vec<f64> = z.iter().map(|&zi| if zi > 0.5 { 1.0 } else { 0.0 }).collect();
    let y: Vec<f64> = t.iter().zip(z.iter()).map(|(&ti, &zi)| 1.0 + 2.0 * ti + 3.0 * zi).collect();
    (t, y, z)
}

fn schema() -> Result<CausalSchema, CausalError> {
    Ok(CausalSchemaBuilder::new()
        .continuous("t")
        .treatment()
        .continuous("y")
        .outcome()
        .continuous("z")
        .context()
        .build()?)
}

fn node(schema: &CausalSchema, name: &str) -> Result<DenseNodeId, CausalError> {
    Ok(DenseNodeId::from_raw(schema.id_of(name)?.raw()))
}

/// Runs the example end to end; `main` calls it and the example test suite runs it.
pub fn run() -> Result<(), CausalError> {
    let (t, y, z) = confounded(200);
    let schema = schema()?;
    let columns = || [("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())];
    let data = TabularData::try_from_schema_f64(schema.clone(), columns())?;
    let data_oriented = TabularData::try_from_schema_f64(schema.clone(), columns())?;
    let query = AverageEffectQuery::binary_ate(schema.id_of("t")?, schema.id_of("y")?);

    // The z - t edge is left undirected: the data cannot orient it, and the analysis keeps it so.
    let mut partial = Cpdag::from_named_edges(&schema, &[("z", "y"), ("t", "y")])?;
    partial.insert_undirected(node(&schema, "z")?, node(&schema, "t")?)?;
    let result = Study::tabular(data)
        .graph(partial)
        .query(query.clone())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()?
        .run(&ExecutionContext::for_tests(1))?;
    println!("partial status = {:?}", result.identification.status);
    println!("partial effect = {:.4}", result.effect());
    // The two completions disagree: Z -> T adjusts for Z (effect 2), T -> Z is the crude
    // contrast (2 + 3 * (E[z | t=1] - E[z | t=0]) = 3.5). Equal weights give their mean.
    assert!((result.effect() - 2.75).abs() < 1e-6, "envelope mean {}", result.effect());

    let oriented = Cpdag::from_named_edges(&schema, &[("z", "t"), ("z", "y"), ("t", "y")])?;
    let point = Study::tabular(data_oriented)
        .graph(oriented)
        .query(query)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()?
        .run(&ExecutionContext::for_tests(1))?;
    println!("oriented status = {:?}", point.identification.status);
    // A fully oriented CPDAG adjusts for Z: the effect is the structural 2.
    assert!((point.effect() - 2.0).abs() < 1e-6, "oriented effect {}", point.effect());
    Ok(())
}

fn main() -> Result<(), CausalError> {
    run()
}
