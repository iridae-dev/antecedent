//! Static ATE: schema + named columns + named DAG → analyze.
//!
//! Run: `cargo run -p causal --example ate_quickstart`

use causal::RefuteSuite;
use causal::prelude::*;

#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn main() -> Result<(), CausalError> {
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

    let result = CausalAnalysis::builder()
        .data(data)
        .graph(dag)
        .query(query)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()?
        .run(&ExecutionContext::for_tests(1))?;

    println!("effect = {:.4}", result.effect());
    println!("status = {:?}", result.identification.status);
    Ok(())
}
