//! Criterion smoke bench for design MC ranking.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(missing_docs)]

use std::sync::Arc;

use antecedent_core::{ExecutionContext, VariableId};
use antecedent_design::{
    CandidateDesign, DesignCost, DesignEvaluationContext, DesignObjective, DesignRankConfig,
    DesignRanker, MeasurementPlan, SamplingPlan,
};
use antecedent_prob::{GraphIdentFlag, WeightedGraphSamples};
use criterion::{Criterion, criterion_group, criterion_main};

fn bench_rank(c: &mut Criterion) {
    let graphs = WeightedGraphSamples::new(
        vec![0.4, 0.35, 0.25],
        vec![
            GraphIdentFlag::Identified,
            GraphIdentFlag::Unidentified,
            GraphIdentFlag::Unidentified,
        ],
        vec![1, 2, 3],
    )
    .expect("graphs");
    let candidates: Vec<CandidateDesign> = (0..8)
        .map(|i| {
            if i % 2 == 0 {
                CandidateDesign::Measure(MeasurementPlan {
                    variables: Arc::from([VariableId::from_raw(i)]),
                    cost: DesignCost::zero(),
                    tag: u64::from(i),
                })
            } else {
                CandidateDesign::IncreaseSamplingRate(SamplingPlan {
                    additional_samples: 50 * u64::from(i),
                    cost: DesignCost::zero(),
                    tag: u64::from(i),
                })
            }
        })
        .collect();
    let ranker = DesignRanker::new().with_config(DesignRankConfig {
        min_batches: 2,
        max_batches: 8,
        batch_size: 4,
        rank_uncertainty_threshold: 0.2,
    });
    let ctx = ExecutionContext::for_tests(11);
    let eval = DesignEvaluationContext::<(), ()> {
        graphs: &graphs,
        effect_width: None,
        model_loglik: None,
        decisions: None,
        query_id_unlock: None,
        env_id_unlock: None,
        identified_under_intervention: None,
        graph_features: None,
    };
    c.bench_function("design_rank_eig_8_candidates", |b| {
        b.iter(|| {
            ranker
                .rank(&DesignObjective::ReduceGraphEntropy, &candidates, &eval, &ctx)
                .expect("rank")
        });
    });
}

criterion_group!(benches, bench_rank);
criterion_main!(benches);
