//! Intrinsic causal influence: which noise term drives a target's variance.
//!
//! `x -> y -> z` with `y = 2x + e_y` (unit noise) and `z = 1.5y + e_z` (noise scale 0.5). The
//! total coefficients of the three noise terms on `z` are 3, 1.5 and 1, so their Shapley
//! shares of `Var(z)` are 9, 2.25 and 0.25. The estimate is nested Monte Carlo through batch
//! noise replay, and every interval names the one uncertainty component it covers.
//!
//! Run: `cargo run -p antecedent --example intrinsic_influence`
//! Source: `examples/rust/intrinsic_influence.rs`

use std::sync::Arc;

use antecedent::gcm::{
    ComponentStatus, IntrinsicInfluenceQuery, NoiseIndependence, UncertaintyComponent,
    attribute_intrinsic_influence,
};
use antecedent_core::{ExecutionContext, VariableId};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_model::{CompiledCausalModel, CompiledMechanismStore, MechanismSlot};

const TARGET: VariableId = VariableId::from_raw(2);
/// Closed-form contributions of the noise terms of `x`, `y` and `z` to `Var(z)`.
const TRUTH: [f64; 3] = [9.0, 2.25, 0.25];

fn linear(intercept: f64, coeffs: &[f64], sigma: f64) -> MechanismSlot {
    MechanismSlot::LinearGaussian { intercept, coeffs: Arc::from(coeffs.to_vec()), sigma }
}

/// Runs the example end to end; `main` calls it and the example test suite runs it.
pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    let mut graph = Dag::with_variables(3);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1))?;
    graph.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2))?;
    let model = CompiledCausalModel::compile(graph)?.with_mechanisms(CompiledMechanismStore {
        slots: Arc::from([
            linear(0.0, &[], 1.0),
            linear(0.0, &[2.0], 1.0),
            linear(0.0, &[1.5], 0.5),
        ]),
    });
    let ctx = ExecutionContext::for_tests(11);
    let (outer, replicates) = (400, 4);
    let query =
        IntrinsicInfluenceQuery::new(TARGET).with_simulation(outer, 20, replicates).with_seed(5);

    // The caller must say the noise terms are independent; a compiled DAG cannot certify it.
    let refused =
        attribute_intrinsic_influence(&model, NoiseIndependence::Undeclared, None, &query, &ctx)
            .expect_err("undeclared noise independence is refused");
    assert!(refused.to_string().contains("noise_independence_undeclared"), "{refused}");

    let result =
        attribute_intrinsic_influence(&model, NoiseIndependence::Declared, None, &query, &ctx)?;
    assert_eq!(result.sources.len(), 3, "the ancestors and the target itself are the players");
    for (source, truth) in result.sources.iter().zip(TRUTH) {
        // Six times the larger of the simulation error implied by the design and the reported one.
        let derived = (2.0 / (replicates as f64 * (outer as f64 - 1.0))).sqrt() * truth;
        let tolerance = 6.0 * derived.max(source.quantified_stderr().stderr);
        assert!(
            (source.contribution - truth).abs() <= tolerance,
            "{:?}: {} vs {truth}",
            source.source,
            source.contribution
        );
        // Three sources enumerate every coalition, so no permutation is sampled; the
        // simulation error is measured; mechanism fitting was not asked for, which is not zero.
        assert!(matches!(
            source.status(UncertaintyComponent::PermutationSampling),
            Some(ComponentStatus::NotApplicable { .. })
        ));
        assert!(matches!(
            source.status(UncertaintyComponent::Simulation),
            Some(ComponentStatus::Quantified { .. })
        ));
        assert!(matches!(
            source.status(UncertaintyComponent::MechanismFit),
            Some(ComponentStatus::Unquantified { .. })
        ));
    }

    // The shares add up to Var(z) up to the estimation error reported as the residual.
    let total: f64 = TRUTH.iter().sum();
    assert!(
        (result.total_variance - total).abs()
            <= 6.0 * result.total_variance_stderr.max(0.05 * total)
    );
    assert!(
        (result.total_variance - result.contribution_sum() - result.allocation_residual).abs()
            < 1e-9
    );
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    run()
}
