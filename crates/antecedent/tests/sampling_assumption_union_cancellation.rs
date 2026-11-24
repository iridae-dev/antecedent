//! A whole-chain bootstrap union stopped by the caller's cancellation is a cancellation, never
//! an invalid declaration.
//!
//! Every scheme reruns the chain once per replicate and checks the caller's token between
//! replicates; each chain checks it again at its own work boundaries. A token that cancels
//! itself on its `k`-th observation stops the run at every such boundary in turn, inside a
//! chain and between replicates. Wherever it stops, the refusal is the class every chain gives
//! its own cancellation: `cancelled_no_claim` with detail `propagation_chain.cancelled`.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::propagation_chain::PropagationChain;
use antecedent::analysis::sampling_assumption_union::{
    ClusterSamplingAssumptionUnionRequest, IidSamplingAssumptionUnionRequest,
    SamplingAssumptionUnionError, SamplingAssumptionUnionRequest, SamplingAssumptionUnionResult,
    compile_cluster_sampling_assumption_union, compile_iid_sampling_assumption_union,
    compile_sampling_assumption_union,
};
use antecedent_contract::propagation_chain::{
    ChainReading, LawMeaning, StageKind, TableConstruction,
};
use antecedent_core::{CancellationToken, ExecutionContext};
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

fn ids(prefix: &str, n: usize) -> Vec<String> {
    (0..n).map(|row| format!("{prefix}-{row}")).collect()
}

/// The observation counts swept: every boundary of the original chain and of the first
/// replicates, and so at least one check between replicates.
const SWEEP: usize = 24;

/// Run `execute` with a token that cancels on its `k`-th observation for every `k` of the
/// sweep, and require each stopped run to be a cancellation. Returns the messages seen.
fn sweep(
    label: &str,
    seed: u64,
    execute: impl Fn(
        &ExecutionContext,
    ) -> Result<SamplingAssumptionUnionResult, SamplingAssumptionUnionError>,
) -> Vec<String> {
    let mut messages = Vec::new();
    for checks in 0..SWEEP {
        let mut ctx = ExecutionContext::for_tests(seed);
        ctx.cancellation = CancellationToken::cancel_after_checks(checks);
        match execute(&ctx) {
            Ok(_) => break,
            Err(SamplingAssumptionUnionError::Chain(refusal)) => {
                assert_eq!(refusal.code(), "cancelled_no_claim", "{label} at {checks}: {refusal}");
                assert_eq!(
                    refusal.detail(),
                    antecedent_core::reason_details::propagation_chain::CANCELLED,
                    "{label} at {checks}: {refusal}"
                );
                messages.push(refusal.message().to_owned());
            }
            Err(other) => panic!("{label} at {checks}: a cancelled run refused as {other}"),
        }
    }
    assert!(!messages.is_empty(), "{label}: the sweep stopped at least one run");
    messages
}

#[test]
fn a_cancelled_declared_union_is_a_cancellation() {
    let shifts: Vec<f64> = (0..100).map(|i| (f64::from(i) - 50.0) / 100.0).collect();
    let ctx = ExecutionContext::for_tests(11);
    let plan = compile_sampling_assumption_union(
        SamplingAssumptionUnionRequest {
            original: envelope_chain(&[1.0, 2.0, 3.0]),
            bootstrap: shifts
                .iter()
                .map(|shift| envelope_chain(&[1.0 + shift, 2.0 + shift, 3.0 + shift]))
                .collect(),
            replicate_ids: ids("resample", 100),
            resampling_design_identity: "hand-built-shift-schedule".into(),
            level: 0.95,
        },
        &ctx,
    )
    .expect("compiles");
    let messages = sweep("declared", 11, |ctx| plan.execute(ctx));
    assert!(
        messages.iter().any(|message| message == "whole-chain resampling was cancelled"),
        "the sweep stopped a run between replicates: {messages:?}"
    );
    // Uncancelled, the same program still runs.
    assert!(plan.execute(&ctx).is_ok());
}

#[test]
fn a_cancelled_iid_union_is_a_cancellation() {
    let ctx = ExecutionContext::for_tests(31);
    let values: Vec<f64> = (0..40).map(|row| 1.0 + f64::from(row) / 40.0).collect();
    let plan = compile_iid_sampling_assumption_union(
        IidSamplingAssumptionUnionRequest {
            original: envelope_chain(&values),
            unit_ids: ids("unit", 40),
            bootstrap_replicates: 100,
            level: 0.95,
        },
        &ctx,
    )
    .expect("compiles");
    let messages = sweep("iid", 31, |ctx| plan.execute(ctx));
    assert!(
        messages.iter().any(|message| message == "IID whole-chain bootstrap was cancelled"),
        "the sweep stopped a run between replicates: {messages:?}"
    );
}

#[test]
fn a_cancelled_cluster_union_is_a_cancellation() {
    let ctx = ExecutionContext::for_tests(37);
    let values: Vec<f64> = (0..40).map(|row| 1.0 + f64::from(row / 2) / 20.0).collect();
    let plan = compile_cluster_sampling_assumption_union(
        ClusterSamplingAssumptionUnionRequest {
            original: envelope_chain(&values),
            unit_ids: ids("unit", 40),
            cluster_ids: (0..40).map(|row| format!("cluster-{}", row / 2)).collect(),
            bootstrap_replicates: 100,
            level: 0.95,
        },
        &ctx,
    )
    .expect("compiles");
    let messages = sweep("cluster", 37, |ctx| plan.execute(ctx));
    assert!(
        messages.iter().any(|message| message == "cluster bootstrap was cancelled"),
        "the sweep stopped a run between replicates: {messages:?}"
    );
}
