//! Known-truth checks of the whole-chain bootstrap unions around an assumption
//! envelope. The generic union is checked against the basic-bootstrap endpoint
//! formula evaluated by hand on replicates whose endpoints are known; the
//! compiler-resampled unions (independent units, complete clusters, moving
//! blocks) are checked against the sampling standard error of a mean of known
//! variance, which the resampling must reproduce within bootstrap noise.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::propagation_chain::PropagationChain;
use antecedent::analysis::sampling_assumption_union::{
    ClusterSamplingAssumptionUnionRequest, IidSamplingAssumptionUnionRequest,
    MovingBlockSamplingAssumptionUnionRequest, SamplingAssumptionUnionRequest,
    compile_cluster_sampling_assumption_union, compile_iid_sampling_assumption_union,
    compile_moving_block_sampling_assumption_union, compile_sampling_assumption_union,
    consume_sampling_assumption_union,
};
use antecedent_contract::propagation_chain::{
    ChainReading, LawMeaning, StageKind, TableConstruction,
};
use antecedent_core::ExecutionContext;
use antecedent_io::propagation_chain_record::{
    ChainOpWire, ChainSourceWire, ChainStageWire, PropagationChainRecordWire,
};
use antecedent_io::quantity_wire::ScientificQuantityWire;

fn envelope_chain(values: &[f64]) -> PropagationChainRecordWire {
    PropagationChain::draft(
        TableConstruction::JointDraws,
        ChainReading::AssumptionEnvelope,
        "answer",
        vec![
            ChainSourceWire::draws("estimate", LawMeaning::Sampling, "joint-table", values),
            ChainSourceWire::interval("bias", 0.0, 1.0),
        ],
        vec![ChainStageWire::new(
            "answer",
            StageKind::DerivedQuantity,
            &["estimate", "bias"],
            ChainOpWire::binary("sum"),
        )],
    )
    .with_output_quantity(ScientificQuantityWire {
        version: 1,
        variable_id: "study.estimate_plus_bias".into(),
        variable_name: "Estimate plus bias".into(),
        role: "outcome".into(),
        units: "points".into(),
        population_id: "target".into(),
        regime_id: "observed".into(),
        horizon: 1,
        functional_id: "mean".into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    })
}

/// Type-7 quantile of a sorted slice, written here from its definition.
fn type7(sorted: &[f64], probability: f64) -> f64 {
    let h = (sorted.len() as f64 - 1.0) * probability;
    let low = h.floor() as usize;
    let high = (low + 1).min(sorted.len() - 1);
    sorted[low] + (h - low as f64) * (sorted[high] - sorted[low])
}

fn ids(prefix: &str, n: usize) -> Vec<String> {
    (0..n).map(|row| format!("{prefix}-{row}")).collect()
}

#[test]
fn the_union_is_the_basic_bootstrap_endpoint_formula_over_the_replicate_endpoints() {
    // Replicate i shifts the sampling draws {1, 2, 3} by s_i = (i - 50) / 100, so its
    // envelope endpoint means are 2 + s_i and 3 + s_i; the original's are 2 and 3.
    let shifts: Vec<f64> = (0..100).map(|i| (f64::from(i) - 50.0) / 100.0).collect();
    let level = 0.95;
    let ctx = ExecutionContext::for_tests(11);
    let request = SamplingAssumptionUnionRequest {
        original: envelope_chain(&[1.0, 2.0, 3.0]),
        bootstrap: shifts
            .iter()
            .map(|shift| envelope_chain(&[1.0 + shift, 2.0 + shift, 3.0 + shift]))
            .collect(),
        replicate_ids: ids("resample", 100),
        resampling_design_identity: "hand-built-shift-schedule".into(),
        level,
    };
    let plan = compile_sampling_assumption_union(request, &ctx).expect("compiles");
    let result = plan.execute(&ctx).expect("executes");
    assert!((result.original_lower - 2.0).abs() < 1e-12);
    assert!((result.original_upper - 3.0).abs() < 1e-12);
    // Basic bootstrap: lower = 2 * lower_hat - Q_{1 - a/2}(lower*), upper = 2 * upper_hat - Q_{a/2}(upper*).
    let tail = (1.0 - level) / 2.0;
    let lows: Vec<f64> = shifts.iter().map(|shift| 2.0 + shift).collect();
    let highs: Vec<f64> = shifts.iter().map(|shift| 3.0 + shift).collect();
    let expected_lower = 4.0 - type7(&lows, 1.0 - tail);
    let expected_upper = 6.0 - type7(&highs, tail);
    assert!((result.union_lower - expected_lower).abs() < 1e-9, "{}", result.union_lower);
    assert!((result.union_upper - expected_upper).abs() < 1e-9, "{}", result.union_upper);
    // The hand evaluation of the same formula at these shifts.
    assert!((expected_lower - 1.534_75).abs() < 1e-9);
    assert!((expected_upper - 3.475_25).abs() < 1e-9);
    assert_eq!(result.calibration, "not_measured");
    assert_eq!(result.chain_identities.len(), 101);
    let artifact = plan.export(&result, &ctx).expect("exports");
    let replay = consume_sampling_assumption_union(&artifact, plan.lowered().identity(), &ctx)
        .expect("replays");
    assert!((replay.union_lower - result.union_lower).abs() == 0.0);
}

fn bernoulli_half(n: usize, group: usize) -> Vec<f64> {
    // 0/1 values in runs of `group` rows: mean 1/2, variance 1/4.
    (0..n).map(|row| f64::from(u8::try_from((row / group) % 2).expect("bit"))).collect()
}

#[test]
fn iid_resampling_reproduces_the_standard_error_of_a_mean() {
    // n = 400 values of mean 1/2 and variance 1/4: se = 0.025, half-width 1.96 se = 0.049.
    let ctx = ExecutionContext::for_tests(31);
    let request = IidSamplingAssumptionUnionRequest {
        original: envelope_chain(&bernoulli_half(400, 1)),
        unit_ids: ids("unit", 400),
        bootstrap_replicates: 200,
        level: 0.95,
    };
    let result = compile_iid_sampling_assumption_union(request, &ctx)
        .expect("compiles")
        .execute(&ctx)
        .expect("executes");
    assert!((result.original_lower - 0.5).abs() < 1e-12);
    assert!((result.original_upper - 1.5).abs() < 1e-12);
    let lower_gap = result.original_lower - result.union_lower;
    let upper_gap = result.union_upper - result.original_upper;
    for gap in [lower_gap, upper_gap] {
        assert!((0.02..0.09).contains(&gap), "half-width {gap} is not near 0.049");
    }
    assert_eq!(result.calibration, "not_measured");
}

#[test]
fn cluster_resampling_widens_the_interval_for_dependent_row_pairs() {
    // 200 complete two-row clusters, each constant at 0 or 1: the mean has 200 effective
    // units, so se = 0.5 / sqrt(200) = 0.0354 and the half-width is about 0.069.
    let ctx = ExecutionContext::for_tests(37);
    let request = ClusterSamplingAssumptionUnionRequest {
        original: envelope_chain(&bernoulli_half(400, 2)),
        unit_ids: ids("unit", 400),
        cluster_ids: (0..400).map(|row| format!("cluster-{}", row / 2)).collect(),
        bootstrap_replicates: 200,
        level: 0.95,
    };
    let result = compile_cluster_sampling_assumption_union(request, &ctx)
        .expect("compiles")
        .execute(&ctx)
        .expect("executes");
    assert!((result.original_lower - 0.5).abs() < 1e-12);
    let lower_gap = result.original_lower - result.union_lower;
    let upper_gap = result.union_upper - result.original_upper;
    for gap in [lower_gap, upper_gap] {
        assert!((0.03..0.12).contains(&gap), "half-width {gap} is not near 0.069");
    }
}

#[test]
fn moving_blocks_carry_the_serial_dependence_that_independent_units_would_miss() {
    // A slow square wave (runs of 40) has far larger sampling error than 400 independent
    // units of the same marginal (half-width 0.049); blocks of the run length recover it
    // (about 0.18 for ten effectively independent blocks).
    let ctx = ExecutionContext::for_tests(41);
    let request = MovingBlockSamplingAssumptionUnionRequest {
        original: envelope_chain(&bernoulli_half(400, 40)),
        unit_ids: ids("time", 400),
        block_length: 40,
        bootstrap_replicates: 200,
        level: 0.95,
    };
    let result = compile_moving_block_sampling_assumption_union(request, &ctx)
        .expect("compiles")
        .execute(&ctx)
        .expect("executes");
    let lower_gap = result.original_lower - result.union_lower;
    let upper_gap = result.union_upper - result.original_upper;
    for gap in [lower_gap, upper_gap] {
        assert!((0.09..0.40).contains(&gap), "half-width {gap} did not carry the dependence");
    }
}

#[test]
fn malformed_schedules_are_refused() {
    let ctx = ExecutionContext::for_tests(43);
    let short = IidSamplingAssumptionUnionRequest {
        original: envelope_chain(&bernoulli_half(400, 1)),
        unit_ids: ids("unit", 400),
        bootstrap_replicates: 99,
        level: 0.95,
    };
    assert!(compile_iid_sampling_assumption_union(short, &ctx).is_err());
    let one_short = SamplingAssumptionUnionRequest {
        original: envelope_chain(&[1.0, 2.0, 3.0]),
        bootstrap: (0..99).map(|_| envelope_chain(&[1.0, 2.0, 3.0])).collect(),
        replicate_ids: ids("resample", 99),
        resampling_design_identity: "short".into(),
        level: 0.95,
    };
    assert!(compile_sampling_assumption_union(one_short, &ctx).is_err());
}
