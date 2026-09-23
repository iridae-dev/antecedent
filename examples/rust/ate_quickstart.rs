//! Static ATE: schema + named columns + named DAG → analyze.
//!
//! Run: `cargo run -p antecedent --example ate_quickstart`
//! Source: `examples/rust/ate_quickstart.rs`

use antecedent::RefuteSuite;
use antecedent::prelude::*;

/// Runs the example end to end; `main` calls it and the example test suite runs it.
#[allow(clippy::cast_precision_loss)]
pub fn run() -> Result<(), CausalError> {
    let n = 200usize;
    let z: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
    let t: Vec<f64> = z.iter().map(|&zi| if zi > 0.5 { 1.0 } else { 0.0 }).collect();
    let y: Vec<f64> = t.iter().zip(z.iter()).map(|(&ti, &zi)| 1.0 + 2.0 * ti + 3.0 * zi).collect();

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
    let query = AverageEffectQuery::binary_ate(schema.id_of("t")?, schema.id_of("y")?);

    let result = Study::tabular(data)
        .graph(dag)
        .query(query)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()?
        .run(&ExecutionContext::for_tests(1))?;

    println!("effect = {:.4}", result.effect());
    println!("status = {:?}", result.identification.status);
    // y = 1 + 2t + 3z with no noise and z confounding t: adjusting for z recovers exactly 2.
    // The crude contrast of y on t is far from it (the z-gap between arms adds 1.5).
    assert!((result.effect() - 2.0).abs() < 1e-6, "adjusted effect {}", result.effect());
    assert_eq!(format!("{:?}", result.identification.status), "NonparametricallyIdentified");
    Ok(())
}

fn main() -> Result<(), CausalError> {
    run()
}
