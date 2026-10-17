//! Noise replay: keep a model's exogenous noise, fix or redraw some coordinates, evaluate again.
//!
//! The chain `a -> b -> c` is linear Gaussian with known coefficients and noise scales, so the
//! variances once a noise coordinate is fixed have closed forms: with `u_a` fixed,
//! `Var(b) = sigma_b^2` and `Var(c) = 9 sigma_b^2 + sigma_c^2`.
//!
//! Run: `cargo run -p antecedent --example noise_replay`
//! Source: `examples/rust/noise_replay.rs`

use std::sync::Arc;

use antecedent_core::ExecutionContext;
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_model::{
    CompiledCausalModel, CompiledMechanismStore, MechanismSlot, ModelView, NoiseFix, NoiseOrigin,
    NoiseReplayer, NoiseSource, NoiseTable,
};

const B_SIGMA: f64 = 0.5;
const C_SIGMA: f64 = 2.0;

fn node(i: u32) -> DenseNodeId {
    DenseNodeId::from_raw(i)
}

fn linear(intercept: f64, coeffs: &[f64], sigma: f64) -> MechanismSlot {
    MechanismSlot::LinearGaussian { intercept, coeffs: Arc::from(coeffs.to_vec()), sigma }
}

/// `a = 1 + e_a`, `b = 0.5 + 2a + e_b`, `c = -1 + 3b + e_c`.
fn chain() -> Result<CompiledCausalModel, Box<dyn std::error::Error>> {
    let mut graph = Dag::with_variables(3);
    graph.insert_directed(node(0), node(1))?;
    graph.insert_directed(node(1), node(2))?;
    Ok(CompiledCausalModel::compile(graph)?.with_mechanisms(CompiledMechanismStore {
        slots: Arc::from([
            linear(1.0, &[], 1.0),
            linear(0.5, &[2.0], B_SIGMA),
            linear(-1.0, &[3.0], C_SIGMA),
        ]),
    }))
}

fn variance(x: &[f64]) -> f64 {
    let mean = x.iter().sum::<f64>() / x.len() as f64;
    x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (x.len() - 1) as f64
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// Runs the example end to end; `main` calls it and the example test suite runs it.
pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    let model = chain()?;
    let view = ModelView::observational(&model);
    let ctx = ExecutionContext::for_tests(21);
    let n = 20_000;

    // 1. Draw the noise once; replaying the table reproduces the sampled values bit for bit.
    let mut replayer = NoiseReplayer::new();
    let (sampled, table) = replayer.sample(&view, n, &ctx)?;
    let again = NoiseReplayer::new().replay(&view, NoiseSource::Table(&table), 0, &ctx)?;
    assert_eq!(bits(&sampled.values), bits(&again.values));

    // 2. Fix `u_a` at its reference value and redraw the other coordinates on a fresh stream.
    //    Every row now shares one `a`, so only the downstream noise moves `b` and `c`.
    let mut conditioned = table.clone();
    conditioned.fix(&model, &[node(0)], NoiseFix::Reference)?;
    conditioned.randomize(&view, &[node(1), node(2)], 1, &ctx)?;
    assert_eq!(
        conditioned.origins(),
        &[NoiseOrigin::Fixed, NoiseOrigin::Sampled { draw: 1 }, NoiseOrigin::Sampled { draw: 1 }]
    );
    let out = NoiseReplayer::new().replay(&view, NoiseSource::Table(&conditioned), 0, &ctx)?;
    assert!(out.column(0)?.iter().all(|a| (a - 1.0).abs() < 1e-12), "a is constant");
    let (var_b, var_c) = (variance(out.column(1)?), variance(out.column(2)?));
    let (want_b, want_c) = (B_SIGMA.powi(2), 9.0 * B_SIGMA.powi(2) + C_SIGMA.powi(2));
    assert!((var_b / want_b - 1.0).abs() < 0.06, "Var(b | u_a) {var_b} vs {want_b}");
    assert!((var_c / want_c - 1.0).abs() < 0.06, "Var(c | u_a) {var_c} vs {want_c}");

    // 3. A coalition reads some coordinates from the observed noise and the rest from an
    //    independent redraw without copying either table. Holding only `u_a` equals writing
    //    the observed `u_a` into the redraw.
    let mut redraw = table.clone();
    redraw.randomize(&view, &[node(0), node(1), node(2)], 1, &ctx)?;
    let mask = [true, false, false];
    let mut buffer = vec![0.0; n * 3];
    let coalition = NoiseSource::coalition(&table, &redraw, &mask)?;
    let done = replayer.replay_into(&view, coalition, 0, &mut buffer, &ctx)?;
    assert!(!done.cancelled && done.completed_nodes == 3);
    let mut by_hand: NoiseTable = redraw.clone();
    by_hand.fix(&model, &[node(0)], NoiseFix::FromTable(&table))?;
    let expected = NoiseReplayer::new().replay(&view, NoiseSource::Table(&by_hand), 0, &ctx)?;
    assert_eq!(bits(&buffer), bits(&expected.values));
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    run()
}
