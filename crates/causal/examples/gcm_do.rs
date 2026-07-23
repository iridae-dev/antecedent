//! GCM: fit → [`sample_do`](causal::gcm::sample_do).
//!
//! Run: `cargo run -p causal --example gcm_do`

use causal::gcm::{fit_gcm, sample_do};
use causal::prelude::*;
use causal_core::{Intervention, Value};

#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn main() -> Result<(), CausalError> {
    let n = 120usize;
    let z: Vec<f64> = (0..n).map(|i| (i as f64) * 0.01).collect();
    let t: Vec<f64> = z.iter().map(|&zi| if zi > 0.5 { 1.0 } else { 0.0 }).collect();
    let y: Vec<f64> = t.iter().zip(z.iter()).map(|(&ti, &zi)| 1.0 + 2.0 * ti + zi).collect();

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
    let fitted = fit_gcm(dag, &data)?;
    let ctx = ExecutionContext::for_tests(1);
    let mut rng = ctx.rng.stream(1);
    let draws = sample_do(
        &fitted.model,
        &[Intervention::set(schema.id_of("t")?, Value::f64(1.0))],
        50,
        &mut rng,
        &ctx,
    )?;
    println!("do(t=1) sample rows = {}", draws.n_rows);
    Ok(())
}
