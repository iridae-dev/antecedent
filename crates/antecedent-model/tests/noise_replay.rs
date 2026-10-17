//! Batch exogenous-noise replay: sampled, fixed, randomized and abducted noise
//! coordinates replayed through one compiled model without cloning it.
//!
//! The chain `a -> b -> c` is linear Gaussian with known coefficients and noise scales,
//! so fixed-coordinate variances have closed forms.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::float_cmp, reason = "fixed coordinates and hard sets replay exact literal values")]

use std::sync::Arc;

use antecedent_core::{
    CancellationToken, CausalRng, ExecutionContext, MemoryBudget, StochasticPolicy,
};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_kernels::standard_normal;
use antecedent_model::{
    CompiledCausalModel, CompiledMechanismStore, InterventionOverlay, MechanismSlot,
    MechanismWorkspace, ModelError, ModelView, NOISE_ROW_BLOCK, NoiseFix, NoiseInferenceMode,
    NoiseOrigin, NoiseReplayer, NoiseSource, NoiseTable, sample_observational,
};

const A_SIGMA: f64 = 1.0;
const B_SIGMA: f64 = 0.5;
const C_SIGMA: f64 = 2.0;

fn node(i: u32) -> DenseNodeId {
    DenseNodeId::from_raw(i)
}

fn linear(intercept: f64, coeffs: &[f64], sigma: f64) -> MechanismSlot {
    MechanismSlot::LinearGaussian { intercept, coeffs: Arc::from(coeffs.to_vec()), sigma }
}

fn chain_with(slots: Vec<MechanismSlot>) -> CompiledCausalModel {
    let n = u32::try_from(slots.len()).unwrap();
    let mut graph = Dag::with_variables(n);
    for i in 1..n {
        graph.insert_directed(node(i - 1), node(i)).unwrap();
    }
    CompiledCausalModel::compile(graph)
        .unwrap()
        .with_mechanisms(CompiledMechanismStore { slots: Arc::from(slots) })
}

/// `a = 1 + e_a`, `b = 0.5 + 2a + e_b`, `c = -1 + 3b + e_c`.
fn chain() -> CompiledCausalModel {
    chain_with(vec![
        linear(1.0, &[], A_SIGMA),
        linear(0.5, &[2.0], B_SIGMA),
        linear(-1.0, &[3.0], C_SIGMA),
    ])
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn variance(x: &[f64]) -> f64 {
    let mean = x.iter().sum::<f64>() / x.len() as f64;
    x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (x.len() - 1) as f64
}

#[test]
fn replaying_a_sampled_table_reproduces_the_draws_bit_for_bit() {
    let model = chain();
    let ctx = ExecutionContext::for_tests(11);
    let mut overlays = vec![InterventionOverlay::observational(3)];
    let mut shifted = InterventionOverlay::observational(3);
    shifted.hard_set[0] = Some(2.0);
    shifted.shifts[1] = 0.75;
    shifted.stochastic[2] = Some(StochasticPolicy::Gaussian { mean: 1.0, variance: 4.0 });
    overlays.push(shifted);

    for overlay in overlays {
        let view = ModelView::with_overlay(&model, overlay);
        let mut replayer = NoiseReplayer::new();
        let (sampled, table) = replayer.sample(&view, 10_000, &ctx).unwrap();
        // A fresh replayer (fresh scratch) and a fresh sampler on the same seed agree.
        let again =
            NoiseReplayer::new().replay(&view, NoiseSource::Table(&table), 0, &ctx).unwrap();
        assert_eq!(bits(&sampled.values), bits(&again.values));
        let (resampled, retable) = NoiseReplayer::new().sample(&view, 10_000, &ctx).unwrap();
        assert_eq!(bits(&resampled.values), bits(&sampled.values));
        assert_eq!(bits(retable.values()), bits(table.values()));
    }

    let other = ExecutionContext::for_tests(12);
    let view = ModelView::observational(&model);
    let a = NoiseTable::sample(&view, 64, &ctx).unwrap();
    let b = NoiseTable::sample(&view, 64, &other).unwrap();
    assert_ne!(bits(a.values()), bits(b.values()), "noise follows the seed");
}

#[test]
fn replay_applies_the_overlay_with_forward_sampling_precedence() {
    let model = chain();
    let ctx = ExecutionContext::for_tests(3);
    let mut overlay = InterventionOverlay::observational(3);
    overlay.hard_set[0] = Some(2.0);
    overlay.shifts[1] = 0.75;
    let view = ModelView::with_overlay(&model, overlay);
    let table = NoiseTable::sample(&view, 50, &ctx).unwrap();
    let out = NoiseReplayer::new().replay(&view, NoiseSource::Table(&table), 0, &ctx).unwrap();
    let (a, b, c) = (out.column(0).unwrap(), out.column(1).unwrap(), out.column(2).unwrap());
    let (nb, nc) = (table.column(1).unwrap(), table.column(2).unwrap());
    for r in 0..50 {
        assert_eq!(a[r], 2.0, "hard set wins over the retained noise");
        let want_b = 0.5 + 2.0 * a[r] + nb[r] + 0.75;
        assert!((b[r] - want_b).abs() < 1e-12, "shift is added after evaluation");
        assert!((c[r] - (-1.0 + 3.0 * b[r] + nc[r])).abs() < 1e-12);
    }
}

#[test]
fn forward_sampling_draws_are_unchanged() {
    // The sequential-stream sampler keeps its published draws: node `a` of a seeded
    // observational sample is `intercept + sigma * z` on the stream's first normals.
    let model = chain();
    let ctx = ExecutionContext::for_tests(1);
    let mut rng = CausalRng::from_seed(7);
    let mut ws = MechanismWorkspace::default();
    let batch = sample_observational(&model, 8, &mut rng, &mut ws, &ctx).unwrap();
    let mut expected = CausalRng::from_seed(7);
    for r in 0..8 {
        let noise = A_SIGMA * standard_normal(&mut expected);
        assert_eq!(batch.column(0).unwrap()[r].to_bits(), (1.0 + noise).to_bits());
    }
}

#[test]
fn fixed_coordinates_stay_fixed_and_randomized_coordinates_vary() {
    let model = chain();
    let ctx = ExecutionContext::for_tests(21);
    let view = ModelView::observational(&model);
    let n = 20_000;
    let mut table = NoiseTable::sample(&view, n, &ctx).unwrap();
    let original = table.clone();

    table.fix(&model, &[node(0)], NoiseFix::Reference).unwrap();
    table.randomize(&view, &[node(1), node(2)], 1, &ctx).unwrap();
    assert_eq!(
        table.origins(),
        &[NoiseOrigin::Fixed, NoiseOrigin::Sampled { draw: 1 }, NoiseOrigin::Sampled { draw: 1 }]
    );
    assert!(table.column(0).unwrap().iter().all(|&u| u == 0.0));
    assert_ne!(bits(table.column(1).unwrap()), bits(original.column(1).unwrap()));
    assert_ne!(bits(table.column(2).unwrap()), bits(original.column(2).unwrap()));

    let out = NoiseReplayer::new().replay(&view, NoiseSource::Table(&table), 0, &ctx).unwrap();
    let a = out.column(0).unwrap();
    assert!(a.iter().all(|&v| v == 1.0), "a fixed noise coordinate gives one value for every row");
    // Conditional on the fixed u_a: Var(b) = sigma_b^2, Var(c) = 9 sigma_b^2 + sigma_c^2.
    let var_b = variance(out.column(1).unwrap());
    let var_c = variance(out.column(2).unwrap());
    let want_b = B_SIGMA.powi(2);
    let want_c = 9.0 * B_SIGMA.powi(2) + C_SIGMA.powi(2);
    assert!((var_b / want_b - 1.0).abs() < 0.06, "Var(b | u_a) {var_b} vs {want_b}");
    assert!((var_c / want_c - 1.0).abs() < 0.06, "Var(c | u_a) {var_c} vs {want_c}");
}

#[test]
fn randomize_is_a_stream_family_and_leaves_other_columns_alone() {
    let model = chain();
    let ctx = ExecutionContext::for_tests(5);
    let view = ModelView::observational(&model);
    let base = NoiseTable::sample(&view, 300, &ctx).unwrap();

    let mut one = base.clone();
    one.randomize(&view, &[node(1)], 1, &ctx).unwrap();
    let mut one_again = base.clone();
    one_again.randomize(&view, &[node(1)], 1, &ctx).unwrap();
    let mut two = base.clone();
    two.randomize(&view, &[node(1)], 2, &ctx).unwrap();
    assert_eq!(bits(one.values()), bits(one_again.values()), "same draw, same noise");
    assert_ne!(bits(one.column(1).unwrap()), bits(two.column(1).unwrap()));
    for untouched in [0, 2] {
        assert_eq!(bits(one.column(untouched).unwrap()), bits(base.column(untouched).unwrap()));
    }
    let err = one.randomize(&view, &[node(1)], 0, &ctx).unwrap_err();
    assert!(matches!(err, ModelError::Shape { .. }), "{err:?}");
}

#[test]
fn a_nodes_noise_does_not_depend_on_other_nodes_or_later_row_blocks() {
    let model = chain();
    let ctx = ExecutionContext::for_tests(8);
    let view = ModelView::observational(&model);
    let short = NoiseTable::sample(&view, NOISE_ROW_BLOCK + 10, &ctx).unwrap();
    let long = NoiseTable::sample(&view, 3 * NOISE_ROW_BLOCK, &ctx).unwrap();
    for n in 0..3 {
        assert_eq!(
            bits(&short.column(n).unwrap()[..NOISE_ROW_BLOCK]),
            bits(&long.column(n).unwrap()[..NOISE_ROW_BLOCK]),
            "the first row block of node {n} is the same stream at any length"
        );
    }
    let two_nodes = chain_with(vec![linear(1.0, &[], A_SIGMA), linear(0.5, &[2.0], B_SIGMA)]);
    let narrow = NoiseTable::sample(&ModelView::observational(&two_nodes), 100, &ctx).unwrap();
    assert_eq!(bits(narrow.column(1).unwrap()), bits(&long.column(1).unwrap()[..100]));
}

#[test]
fn coalitions_replay_from_two_tables_without_copying_or_growing_scratch() {
    let model = chain();
    let ctx = ExecutionContext::for_tests(31);
    let view = ModelView::observational(&model);
    let n = 5_000;
    let observed = NoiseTable::sample(&view, n, &ctx).unwrap();
    let mut redraw = observed.clone();
    redraw.randomize(&view, &[node(0), node(1), node(2)], 1, &ctx).unwrap();

    let mut replayer = NoiseReplayer::new();
    let mut buf = vec![0.0; n * 3];
    let replay = |replayer: &mut NoiseReplayer, mask: &[bool], buf: &mut Vec<f64>| {
        let source = NoiseSource::coalition(&observed, &redraw, mask).unwrap();
        let outcome = replayer.replay_into(&view, source, 0, buf, &ctx).unwrap();
        assert!(!outcome.cancelled && outcome.completed_nodes == 3);
    };

    replay(&mut replayer, &[true, true, true], &mut buf);
    let all_fixed =
        NoiseReplayer::new().replay(&view, NoiseSource::Table(&observed), 0, &ctx).unwrap();
    assert_eq!(bits(&buf), bits(&all_fixed.values));

    replay(&mut replayer, &[false, false, false], &mut buf);
    let all_redrawn =
        NoiseReplayer::new().replay(&view, NoiseSource::Table(&redraw), 0, &ctx).unwrap();
    assert_eq!(bits(&buf), bits(&all_redrawn.values));

    // Fixing only u_a equals writing u_a into the redraw table.
    replay(&mut replayer, &[true, false, false], &mut buf);
    let mut by_hand = redraw.clone();
    by_hand.fix(&model, &[node(0)], NoiseFix::FromTable(&observed)).unwrap();
    let want = NoiseReplayer::new().replay(&view, NoiseSource::Table(&by_hand), 0, &ctx).unwrap();
    assert_eq!(bits(&buf), bits(&want.values));

    let footprint = replayer.scratch_footprint();
    for coalition in 0u32..512 {
        let mask = [coalition & 1 != 0, coalition & 2 != 0, coalition & 4 != 0];
        replay(&mut replayer, &mask, &mut buf);
    }
    assert_eq!(replayer.scratch_footprint(), footprint, "coalitions reuse one set of buffers");

    let narrow = NoiseTable::sample(&view, n - 1, &ctx).unwrap();
    assert!(NoiseSource::coalition(&observed, &narrow, &[true; 3]).is_err());
    assert!(NoiseSource::coalition(&observed, &redraw, &[true; 2]).is_err());
}

#[test]
fn cancellation_returns_the_exact_completed_prefix() {
    let model = chain();
    let ctx = ExecutionContext::for_tests(41);
    let view = ModelView::observational(&model);
    let table = NoiseTable::sample(&view, 200, &ctx).unwrap();
    let full = NoiseReplayer::new().replay(&view, NoiseSource::Table(&table), 0, &ctx).unwrap();

    let mut cancelling = ExecutionContext::for_tests(41);
    cancelling.cancellation = CancellationToken::cancel_after_checks(2);
    let mut partial = vec![0.0; 200 * 3];
    let outcome = NoiseReplayer::new()
        .replay_into(&view, NoiseSource::Table(&table), 0, &mut partial, &cancelling)
        .unwrap();
    assert!(outcome.cancelled);
    assert_eq!(outcome.completed_nodes, 2);
    assert_eq!(bits(&partial[..400]), bits(&full.values[..400]), "completed draws are exact");
    assert!(partial[400..].iter().all(|&v| v == 0.0), "the rest is untouched");

    let mut cancelling = ExecutionContext::for_tests(41);
    cancelling.cancellation = CancellationToken::cancel_after_checks(2);
    let err =
        NoiseReplayer::new().replay(&view, NoiseSource::Table(&table), 0, &cancelling).unwrap_err();
    assert_eq!(err, ModelError::Cancelled);

    let cancelled = ExecutionContext::for_tests(41);
    cancelled.cancellation.cancel();
    assert_eq!(NoiseTable::sample(&view, 10, &cancelled).unwrap_err(), ModelError::Cancelled);
}

#[test]
fn a_table_over_the_memory_limit_is_refused_before_allocating() {
    let model = chain();
    let view = ModelView::observational(&model);
    let mut ctx = ExecutionContext::for_tests(1);
    ctx.memory = MemoryBudget { soft_limit_bytes: None, hard_limit_bytes: Some(1_000) };

    let err = NoiseTable::sample(&view, 1_000, &ctx).unwrap_err();
    assert_eq!(
        err,
        ModelError::ResourceLimit {
            message: "noise replay memory budget",
            required_bytes: 1_000 * 3 * 8,
            limit_bytes: 1_000,
        }
    );
    // Sampling retains the table and the values: twice the bytes.
    ctx.memory.hard_limit_bytes = Some(1_000 * 3 * 8);
    NoiseTable::sample(&view, 1_000, &ctx).unwrap();
    let err = NoiseReplayer::new().sample(&view, 1_000, &ctx).unwrap_err();
    assert!(matches!(err, ModelError::ResourceLimit { required_bytes: 48_000, .. }), "{err:?}");
    // A shape whose byte size overflows is refused, not allocated.
    let err = NoiseTable::sample(&view, 1 << 62, &ctx).unwrap_err();
    assert!(matches!(err, ModelError::ResourceLimit { required_bytes: u64::MAX, .. }), "{err:?}");
}

#[test]
fn abducted_noise_carries_its_inference_mode_into_the_table() {
    let support = Arc::from(vec![0.0, 1.0, 2.0]);
    let probs = Arc::from(vec![0.2, 0.5, 0.3]);
    let model = chain_with(vec![
        linear(0.0, &[], 1.0),
        MechanismSlot::Discrete { support, probs, logit_coeffs: None },
    ]);
    let n = 4;
    let table = NoiseTable::from_abducted(
        &model,
        n,
        vec![0.1, -0.2, 0.3, 0.0, 0.1, 0.6, 0.9, 0.5],
        &[false, false],
    )
    .unwrap();
    assert_eq!(table.inference_mode(0), Some(NoiseInferenceMode::Invertible));
    assert_eq!(table.inference_mode(1), Some(NoiseInferenceMode::Posterior));
    assert!(table.has_posterior_noise());

    // Replaying abducted categorical noise recovers the category the uniform selects.
    let view = ModelView::observational(&model);
    let ctx = ExecutionContext::for_tests(2);
    let out = NoiseReplayer::new().replay(&view, NoiseSource::Table(&table), 0, &ctx).unwrap();
    assert_eq!(out.column(1).unwrap(), &[0.0, 1.0, 2.0, 1.0]);

    let assumed = NoiseTable::from_abducted(&model, n, vec![0.5; 8], &[true, false]).unwrap();
    assert_eq!(assumed.origin(0), Some(NoiseOrigin::Assumed));
    assert!(NoiseTable::from_abducted(&model, n, vec![0.5; 7], &[false, false]).is_err());

    // Categorical noise is a uniform: an additive reference is refused and leaves the table intact.
    let mut table = table;
    let before = bits(table.values());
    let err = table.fix(&model, &[node(1)], NoiseFix::Constant(0.0)).unwrap_err();
    assert!(matches!(err, ModelError::Numerical { .. }), "{err:?}");
    assert_eq!(bits(table.values()), before);
    table.fix(&model, &[node(1)], NoiseFix::Reference).unwrap();
    assert!(table.column(1).unwrap().iter().all(|&u| u == 0.5));
    assert_eq!(table.origin(1), Some(NoiseOrigin::Fixed));
}

#[test]
fn a_hard_set_node_without_a_fitted_mechanism_needs_no_noise() {
    let model = chain_with(vec![MechanismSlot::Vacant, linear(0.0, &[1.0], 1.0)]);
    let ctx = ExecutionContext::for_tests(2);
    let observational = ModelView::observational(&model);
    assert!(NoiseTable::sample(&observational, 10, &ctx).is_err());

    let mut overlay = InterventionOverlay::observational(2);
    overlay.hard_set[0] = Some(3.0);
    let view = ModelView::with_overlay(&model, overlay);
    let (values, table) = NoiseReplayer::new().sample(&view, 10, &ctx).unwrap();
    assert_eq!(table.origin(0), Some(NoiseOrigin::Unused));
    assert!(values.column(0).unwrap().iter().all(|&v| v == 3.0));
}
