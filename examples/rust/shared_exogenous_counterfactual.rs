//! A counterfactual through a shared hidden cause, computed because the model is fully declared.
//!
//! A hidden frailty `u` (frail with probability 0.3) drives who is treated and how the outcome
//! responds: `x = u`, `y = x xor u`. A patient observed treated is known to be frail, and the same
//! frailty is read under `do(x = 0)`, so `P(Y_x0 = 1 | X = 1) = 1`; independent outcome noise with
//! the same law gives 0.3. The execution is sealed and replayed.
//!
//! Run: `cargo run -p antecedent --example shared_exogenous_counterfactual`
//! Source: `examples/rust/shared_exogenous_counterfactual.rs`

use antecedent::analysis::cross_world_execution::{
    CrossWorldExecution, ExogenousInference, ExogenousVariable, FactualUnit,
    FiniteExecutionRequest, FiniteQuantity, FiniteScm, FiniteVariable, FiniteWorld, NoiseCoupling,
    WorldAtom,
};
use antecedent::prelude::*;
use antecedent_contract::RequestedReading;

fn model(shared: bool) -> FiniteScm {
    let mut exogenous = vec![ExogenousVariable { name: "u".into(), pmf: vec![0.7, 0.3] }];
    if !shared {
        exogenous.push(ExogenousVariable { name: "u_y".into(), pmf: vec![0.7, 0.3] });
    }
    FiniteScm {
        population: "clinic".into(),
        history: "current_regime".into(),
        exogenous,
        variables: vec![
            FiniteVariable {
                name: "x".into(),
                levels: 2,
                parents: vec![],
                exogenous: 0,
                table: vec![0, 1],
            },
            FiniteVariable {
                name: "y".into(),
                levels: 2,
                parents: vec![0],
                exogenous: usize::from(!shared),
                // [u * 2 + x]: u = 0 gives y = x, u = 1 gives y = 1 - x.
                table: vec![0, 1, 1, 0],
            },
        ],
    }
}

fn ask(scm: FiniteScm) -> FiniteExecutionRequest {
    let world = |name: &str, level| FiniteWorld {
        name: name.into(),
        interventions: vec![(0, level)],
        routes: vec![],
    };
    FiniteExecutionRequest {
        evidence_history: scm.history.clone(),
        scm,
        units: vec![FactualUnit { unit: "patient".into(), observed: vec![(0, 1)] }],
        worlds: vec![world("untreated", 0), world("treated", 1)],
        quantity: FiniteQuantity::Probability(WorldAtom { world: 0, variable: 1, level: 1 }),
        coupling: NoiseCoupling::SharedAbducedExogenous,
        exogenous: ExogenousInference::ExactEnumeration,
        target_population: None,
        reading: RequestedReading::Point,
    }
}

/// Runs the example end to end; `main` calls it and the example test suite runs it.
pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = ExecutionContext::for_tests(1);
    let shared = CrossWorldExecution::seal(&ask(model(true)), &ctx)?;
    let independent = CrossWorldExecution::seal(&ask(model(false)), &ctx)?;
    println!("shared frailty:    {:.3}", shared.result().value);
    println!("independent noise: {:.3}", independent.result().value);
    assert_eq!(shared.result().matrix_row, "counterfactual.finite_shared_exogenous_scm.marginal");
    assert!((shared.result().value - 1.0).abs() < 1e-12);
    assert!((independent.result().value - 0.3).abs() < 1e-12);

    // A consumer recomputes the enumeration from the bytes alone.
    let bytes = shared.to_bytes("shared_exogenous_counterfactual")?;
    let replayed = CrossWorldExecution::consume(&bytes, Some(shared.identity()), &ctx)?;
    assert_eq!(replayed.result().value.to_bits(), shared.result().value.to_bits());
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    run()
}
