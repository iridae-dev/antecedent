// SPDX-License-Identifier: MIT OR Apache-2.0
//! Carry grouped sampling through a complete analysis.
//!
//! The synthetic table has 20 pairs split between two strata. Resample complete pairs within each stratum.

use antecedent::analysis::propagation_chain::PropagationChain;
use antecedent::analysis::sampling_assumption_union::{
    StratifiedClusterSamplingAssumptionUnionRequest,
    compile_stratified_cluster_sampling_assumption_union,
    consume_stratified_cluster_sampling_assumption_union,
};
use antecedent_contract::propagation_chain::{
    ChainReading, LawMeaning, StageKind, TableConstruction,
};
use antecedent_core::ExecutionContext;
use antecedent_io::propagation_chain_record::{
    ChainOpWire, ChainSourceWire, ChainStageWire, PropagationChainRecordWire,
};
use antecedent_io::quantity_wire::ScientificQuantityWire;

fn draft(shift: f64) -> PropagationChainRecordWire {
    PropagationChain::draft(
        TableConstruction::JointDraws,
        ChainReading::AssumptionEnvelope,
        "answer",
        vec![
            ChainSourceWire::draws(
                "estimate",
                LawMeaning::Sampling,
                "joint-sampling-table",
                &[1.0 + shift, 2.0 + shift, 3.0 + shift],
            ),
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

fn main() {
    let ctx = ExecutionContext::for_tests(43);
    let mut original = draft(0.0);
    original.sources[0].values = (0..40).map(|row| (row as f64 / 40.0).to_bits()).collect();
    let request = StratifiedClusterSamplingAssumptionUnionRequest {
        original,
        unit_ids: (0..40).map(|i| format!("unit-{i}")).collect(),
        cluster_ids: (0..40).map(|i| format!("cluster-{}", i / 2)).collect(),
        stratum_ids: (0..40).map(|i| format!("stratum-{}", i / 20)).collect(),
        bootstrap_replicates: 100,
        level: 0.95,
    };
    let plan = compile_stratified_cluster_sampling_assumption_union(request, &ctx).unwrap();
    let candidate = plan.calibration_candidate(&ctx).unwrap();
    assert_eq!(candidate.result.calibration, "not_measured");
    assert_eq!(candidate.result.chain_identities.len(), 101);
    assert_eq!(
        candidate.basis.interval_method.as_ref(),
        "bonferroni_basic_stratified_cluster_bootstrap"
    );
    let artifact = plan.export(&candidate.result, &ctx).unwrap();
    let replay = consume_stratified_cluster_sampling_assumption_union(
        &artifact,
        plan.lowered().identity(),
        &ctx,
    )
    .unwrap();
    assert_eq!(candidate.result.union_lower, replay.union_lower);
    assert_eq!(candidate.result.union_upper, replay.union_upper);
    println!(
        "Original assumption endpoints: [{}, {}]",
        replay.original_lower, replay.original_upper
    );
    println!(
        "Resampled candidate endpoints: [{}, {}] ({})",
        replay.union_lower, replay.union_upper, replay.calibration
    );
}
