//! Class-preserving CPDAG ATE: the graph stays a `Cpdag`.
//!
//! A partial CPDAG estimates a MEC envelope. A fully-oriented CPDAG stays a
//! `Cpdag`. Completing the graph yourself is still the `Dag` cell.
//!
//! Run: `cargo run -p antecedent --example class_preserving_cpdag`
//! Source: `examples/rust/class_preserving_cpdag.rs`

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

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
    let data = TabularData::try_from_schema_f64(
        schema.clone(),
        [("t", t.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())],
    )?;
    let query = AverageEffectQuery::binary_ate(schema.id_of("t")?, schema.id_of("y")?);

    let mut partial = Cpdag::from_named_edges(&schema, &[("z", "y"), ("t", "y")])?;
    partial.insert_undirected(node(&schema, "z")?, node(&schema, "t")?)?;
    let identified = identify(
        &AcceptedGraph::from(partial.clone()),
        &CausalQuery::AverageEffect(query.clone()),
    )?;
    println!("partial status = {:?}", identified.status());
    assert_eq!(format!("{:?}", identified.status()), "PartiallyIdentified");

    let result = Study::tabular(data)
        .graph(partial)
        .query(query.clone())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()?
        .run(&ExecutionContext::for_tests(1))?;
    println!("partial effect = {:.4}", result.effect());
    // The two completions disagree: Z -> T adjusts for Z (effect 2), T -> Z is the crude
    // contrast (2 + 3 * (E[z | t=1] - E[z | t=0]) = 3.5). Equal weights give their mean.
    assert!((result.effect() - 2.75).abs() < 1e-6, "envelope mean {}", result.effect());

    let oriented = Cpdag::from_named_edges(&schema, &[("z", "t"), ("z", "y"), ("t", "y")])?;
    let point = identify(&AcceptedGraph::from(oriented), &CausalQuery::AverageEffect(query))?;
    println!("oriented status = {:?}", point.status());
    assert_eq!(format!("{:?}", point.status()), "NonparametricallyIdentified");
    Ok(())
}

fn main() -> Result<(), CausalError> {
    run()
}
