//! The frequentist `randomized.quantile` result is retained in the contracted analysis artifact
//! and read back by the independent consumer.
//!
//! The prepared study of the forty-row design of `randomized_quantile_known_truth.rs` (arms
//! 1..20 and 3..22 under `t -> y`, median effect exactly 2) is executed twice from the retained
//! plan. Both executions carry the same lowered program identity and publish the same effect bit
//! for bit; the artifact encoded from the first is consumed by `antecedent_io` without refusal.
//! The published effect is the hand-derived 2 to `1e-9` (the derivation is in the known-truth
//! module documentation).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::{EstimatorId, RefuteSuite, Study};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ExecutionContext, OutcomeFunctional, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};

#[test]
fn the_randomized_quantile_result_round_trips_through_the_contracted_artifact() {
    let mut outcome: Vec<f64> = (1..=20).map(f64::from).collect();
    outcome.extend((3..=22).map(f64::from));
    let mut treatment = vec![0.0; 20];
    treatment.extend(vec![1.0; 20]);
    let data =
        TabularData::from_f64_columns([("t", treatment.as_slice()), ("y", outcome.as_slice())])
            .unwrap();
    let mut graph = Dag::with_variables(2);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
        .with_outcome_functional(OutcomeFunctional::quantile(0.5));
    let prepared = Study::tabular(data.clone())
        .graph(graph)
        .query(CausalQuery::AverageEffect(query))
        .estimator(EstimatorId::RandomizedQuantile)
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ExecutionContext::for_tests(3))
        .unwrap();

    let ctx = ExecutionContext::for_tests(0xA11);
    let first = prepared.estimate(&data, &ctx).unwrap();
    let again = prepared.estimate(&data, &ctx).unwrap();
    assert!(first.lowered_program_identity().is_some());
    assert_eq!(first.lowered_program_identity(), again.lowered_program_identity());
    assert_eq!(first.effect().to_bits(), again.effect().to_bits());
    assert!((first.effect() - 2.0).abs() < 1e-9, "effect {}", first.effect());

    let bytes = prepared.encode_contracted_result(&first, "randomized-quantile", &ctx).unwrap();
    antecedent_io::consume_analysis_result(&bytes).expect("the independent consumer reads it");
}
