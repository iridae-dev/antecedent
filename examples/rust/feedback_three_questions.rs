// SPDX-License-Identifier: MIT OR Apache-2.0
//! Compare an observation, a clamp and a shock on one feedback system.
//!
//! The system is `A = 0.4 B + U_A` and `B = 0.5 A + U_B` with independent centered
//! unit-variance disturbances. No data are simulated: each answer is the Gaussian equilibrium
//! law the stated equations imply, so the three numbers are fixture truths of this model. A
//! second system shows an intervention with no unique solution, retained as failed mass.
//!
//! Run: `cargo run -p antecedent --example feedback_three_questions`
//! Source: `examples/rust/feedback_three_questions.rs`

use antecedent_core::{Intervention, ScientificWorldId, Value, VariableId};
use antecedent_model::cyclic_world::{
    CertificateDemand, CyclicVariables, CyclicWorldOutcome, CyclicWorldRequest, CyclicWorldResult,
    LinearCyclicSpecification, SELECTION_OR_DYNAMIC_LAW_ATTEMPT, evaluate_linear_world,
    evaluate_linear_worlds,
};
use antecedent_model::linear_solution::LinearStructuralMap;
use antecedent_model::scc_compile::{CompileBudget, GaussianDisturbance};

fn variable(index: u32) -> VariableId {
    VariableId::from_raw(10 + index)
}

fn world(raw: u32) -> ScientificWorldId {
    ScientificWorldId::from_raw(raw)
}

fn identity(n: usize) -> Vec<f64> {
    let mut matrix = vec![0.0; n * n];
    for index in 0..n {
        matrix[index * n + index] = 1.0;
    }
    matrix
}

fn specification(
    coefficients: Vec<f64>,
    order: usize,
) -> Result<LinearCyclicSpecification, String> {
    let describe = |error: &dyn std::fmt::Debug| format!("{error:?}");
    let ids = (0..u32::try_from(order).map_err(|e| describe(&e))?).map(variable).collect();
    LinearCyclicSpecification::try_new(
        CyclicVariables::try_new(ids).map_err(|e| describe(&e))?,
        LinearStructuralMap::try_new(coefficients, vec![0.0; order]).map_err(|e| describe(&e))?,
        GaussianDisturbance::try_new(vec![0.0; order], identity(order))
            .map_err(|e| describe(&e))?,
        "feedback",
        "specification",
        &["bench-derivation"],
    )
    .map_err(|e| describe(&e))
}

fn gaussian_covariance(result: &CyclicWorldResult) -> Result<&[f64], String> {
    match result.outcome() {
        CyclicWorldOutcome::GaussianLaw { covariance, .. } => Ok(covariance),
        other => Err(format!("expected a Gaussian law, got {other:?}")),
    }
}

fn close(left: f64, right: f64, tolerance: f64) -> bool {
    (left - right).abs() <= tolerance
}

/// Runs the example end to end; `main` calls it and the example test suite runs it.
pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    // A = 0.4 B + U_A, B = 0.5 A + U_B.
    let feedback = specification(vec![0.0, 0.4, 0.5, 0.0], 2)?;
    let requests = [
        CyclicWorldRequest::observational(world(0)),
        CyclicWorldRequest::new(world(1), [Intervention::set(variable(0), Value::f64(0.0))]),
        CyclicWorldRequest::new(world(2), [Intervention::set(variable(0), Value::f64(1.0))]),
        CyclicWorldRequest::new(world(3), [Intervention::shift(variable(0), Value::f64(1.0))]),
    ];
    let results = evaluate_linear_worlds(
        &feedback,
        &requests,
        CertificateDemand::Required,
        CompileBudget::unlimited(),
    )
    .map_err(|e| format!("{e:?}"))?;
    let answer = |id: u32| results.get(world(id)).ok_or_else(|| format!("world {id} missing"));

    let covariance = gaussian_covariance(answer(0)?)?;
    let association = covariance[1] / covariance[0];
    let low = answer(1)?.mean_of(variable(1)).ok_or("do(A = 0) has no mean")?;
    let high = answer(2)?.mean_of(variable(1)).ok_or("do(A = 1) has no mean")?;
    let shock = answer(3)?.mean_of(variable(1)).ok_or("the shocked world has no mean")?;
    assert!(close(association, 1.40625 / 1.8125, 1.0e-12));
    assert!(close(association, 0.776, 5.0e-4));
    assert!(close(high - low, 0.5, 1.0e-12));
    assert!(close(shock, 0.625, 1.0e-12));
    assert_eq!(answer(1)?.solved_by().as_str(), "acyclic_order");
    assert_eq!(answer(3)?.solved_by().as_str(), "strongly_connected_blocks");
    println!("observed slope of B on A: {association:.4}");
    println!("mean change in B when A is clamped from 0 to 1: {:.4}", high - low);
    println!("mean change in B after a unit shock to U_A: {shock:.4}");

    // A = B + C + U_A, B = A + U_B, C = A + U_C.
    let loops = specification(vec![0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0], 3)?;
    let clamped = evaluate_linear_world(
        &loops,
        &CyclicWorldRequest::new(world(4), [Intervention::set(variable(2), Value::f64(1.0))]),
        CertificateDemand::ReportUncertified,
        CompileBudget::unlimited(),
    )
    .map_err(|e| format!("{e:?}"))?;
    let CyclicWorldOutcome::NoSolution { failure } = clamped.outcome() else {
        return Err("clamping C leaves no unique solution".into());
    };
    assert_eq!(failure.next_attempt, SELECTION_OR_DYNAMIC_LAW_ATTEMPT);
    assert!(close(clamped.mass().failed(), 1.0, 0.0));
    assert!(close(clamped.mass().solved(), 0.0, 0.0));
    assert_eq!(clamped.solved_by().as_str(), "not_solved");
    println!(
        "do(C = 1): failed mass {:.1}; next attempt {:?}",
        clamped.mass().failed(),
        failure.next_attempt
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    run()
}
