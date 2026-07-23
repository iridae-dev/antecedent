//! Identify-only: graph + query without fitting.
//!
//! Run: `cargo run -p antecedent --example identify_only`

use antecedent::RefuteSuite;
use antecedent::prelude::*;

fn main() -> Result<(), CausalError> {
    let schema = CausalSchemaBuilder::new()
        .continuous("t")
        .treatment()
        .continuous("y")
        .outcome()
        .continuous("z")
        .context()
        .build()?;
    // Dummy data required by the builder; identify_only ignores rows.
    let data = TabularData::from_f64_columns([
        ("t", &[0.0_f64, 1.0][..]),
        ("y", &[0.0_f64, 1.0][..]),
        ("z", &[0.0_f64, 1.0][..]),
    ])?;
    let dag = Dag::from_named_edges(&schema, &[("z", "t"), ("z", "y"), ("t", "y")])?;
    let query = AverageEffectQuery::binary_ate(schema.id_of("t")?, schema.id_of("y")?);

    let id = CausalAnalysis::builder()
        .data(data)
        .graph(dag)
        .query(query)
        .refute(RefuteSuite::None)
        .build()?
        .identify_only()?;

    println!("status = {:?}", id.status);
    println!("estimands = {}", id.estimands.len());
    Ok(())
}
