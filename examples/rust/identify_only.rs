//! Decide identification from a graph and a query, without fitting anything.
//!
//! Question: is the average effect of `t` on `y` identified under the declared graph, and by
//! which estimand?
//!
//! Data and truth: identification reads the structure and the query, never the rows, so the
//! two-row table only satisfies the builder. In the graph `z -> t`, `z -> y`, `t -> y` the
//! variable `z` is a measured common cause of treatment and outcome, so the effect is identified
//! by adjusting for `z`.
//!
//! Asserted: the status is `NonparametricallyIdentified` and at least one estimand is returned.
//!
//! What this establishes: under the declared graph the effect is a functional of the observed
//! distribution. It does not test the graph against data and it produces no estimate.
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
    // Placeholder rows: the builder requires data, and `identify_only` never reads it.
    // `identify_only` covers static DAGs and ADMGs. Cpdag, Pag and temporal classes run through
    // `Study::run`, which reports the identification beside the estimate.
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
