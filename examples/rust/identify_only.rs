//! Identify-only: graph + query without fitting.
//!
//! Run: `cargo run -p antecedent --example identify_only`
//! Source: `examples/rust/identify_only.rs`

use antecedent::RefuteSuite;
use antecedent::prelude::*;

/// Runs the example end to end; `main` calls it and the example test suite runs it.
pub fn run() -> Result<(), CausalError> {
    let schema = CausalSchemaBuilder::new()
        .continuous("t")
        .treatment()
        .continuous("y")
        .outcome()
        .continuous("z")
        .context()
        .build()?;
    // Dummy data required by the builder; identify_only ignores rows.
    // For Cpdag / Pag / temporal classes use `identify(&AcceptedGraph::from(graph), &query)` —
    // that path takes no data.
    let data = TabularData::from_f64_columns([
        ("t", &[0.0_f64, 1.0][..]),
        ("y", &[0.0_f64, 1.0][..]),
        ("z", &[0.0_f64, 1.0][..]),
    ])?;
    let dag = Dag::from_named_edges(&schema, &[("z", "t"), ("z", "y"), ("t", "y")])?;
    let query = AverageEffectQuery::binary_ate(schema.id_of("t")?, schema.id_of("y")?);

    let id = Study::tabular(data)
        .graph(dag)
        .query(query)
        .refute(RefuteSuite::None)
        .build()?
        .identify_only()?;

    println!("status = {:?}", id.status);
    println!("estimands = {}", id.estimands.len());
    // z is a measured common cause of t and y, so the effect is identified by adjusting for z.
    assert_eq!(format!("{:?}", id.status), "NonparametricallyIdentified");
    assert!(!id.estimands.is_empty(), "an identified effect carries an estimand");
    Ok(())
}

fn main() -> Result<(), CausalError> {
    run()
}
