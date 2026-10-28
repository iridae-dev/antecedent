//! Root-cause ranking over a graph ensemble against a hand-computed weighted mean and spread.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_attribution::{
    CacheStats, ChangeAttributionResult, ComponentContribution, ComputeBudget, root_cause_rank,
};
use antecedent_core::{ComponentId, ExecutionContext, VariableId};
use antecedent_prob::{
    GraphIdentFlag, PosteriorDraws, PosteriorQuantityKind, PosteriorSchema, WeightedGraphSamples,
};

fn attribution(value: f64) -> ChangeAttributionResult {
    ChangeAttributionResult {
        outcome: VariableId::from_raw(0),
        total_change: value,
        observed_change: None,
        contributions: Arc::from([ComponentContribution {
            component: ComponentId::from_raw(0),
            contribution: value,
            stderr: None,
            ci_low: None,
            ci_high: None,
        }]),
        interactions: Arc::from([]),
        path_breakdown: Arc::from([]),
        unidentified: Arc::from([]),
        graph_sensitivity: None,
        budget: ComputeBudget::default(),
        monte_carlo_stderr: None,
        component_mc_stderr: None,
        cache_stats: CacheStats::default(),
        fit_uncertainty: None,
    }
}

fn draws(values: &[f64]) -> PosteriorDraws {
    let schema = PosteriorSchema {
        quantities: Arc::from([PosteriorQuantityKind::Scalar { name: Arc::from("contrib_0") }]),
    };
    PosteriorDraws::from_column_major(schema, values.len(), values).unwrap()
}

fn graphs(weights: &[f64]) -> WeightedGraphSamples {
    WeightedGraphSamples::new(
        weights.to_vec(),
        vec![GraphIdentFlag::Identified; weights.len()],
        (1..=weights.len() as u64).collect::<Vec<u64>>(),
    )
    .unwrap()
}

#[test]
fn half_an_ensemble_is_refused_not_silently_ignored() {
    let ctx = ExecutionContext::for_tests(1);
    let graphs = graphs(&[1.0]);
    let draws = draws(&[2.0]);
    assert!(root_cause_rank(&attribution(2.0), Some(&graphs), None, &ctx).is_err());
    assert!(root_cause_rank(&attribution(2.0), None, Some(&draws), &ctx).is_err());
    assert!(root_cause_rank(&attribution(2.0), Some(&graphs), Some(&draws), &ctx).is_ok());
    assert!(root_cause_rank(&attribution(2.0), None, None, &ctx).is_ok());
}

#[test]
fn weighted_graph_ranks_match_a_hand_computed_weighted_mean_and_spread() {
    // Weights 1:3 on contributions -2 and -6: mean -5, population variance 3.
    let ranks = root_cause_rank(
        &attribution(0.0),
        Some(&graphs(&[1.0, 3.0])),
        Some(&draws(&[-2.0, -6.0])),
        &ExecutionContext::for_tests(1),
    )
    .unwrap();
    assert!((ranks[0].score - 5.0).abs() < 1e-12, "the sign is dropped, the mean is weighted");
    assert!((ranks[0].graph_std.unwrap() - 3.0f64.sqrt()).abs() < 1e-12);
}
