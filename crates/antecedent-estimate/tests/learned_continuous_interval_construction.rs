//! Independent construction check of the learned-continuous joint outer bootstrap interval and
//! of the percentile step every bootstrap route shares.
//!
//! The percentile definition is checked on 1..=199, where the type-7 quantile at `p` is
//! `1 + 198 p` exactly: the lower end `1 + 4.95 = 5.95` and the upper end `1 + 193.05 = 194.05`.
//! The route's interval is then compared with hand percentiles of the replicate estimates it
//! retained.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "../../antecedent/tests/learned_continuous_dgp/mod.rs"]
mod dgp;

use antecedent_core::ExecutionContext;
use antecedent_estimate::{
    LearnedContinuousOptions, LearnerSpec, LinearSpec, learned_continuous_interval_internal,
    percentile_interval,
};
use dgp::{Design, Scenario, draw, graph};

fn hand_quantile(values: &[f64], p: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let h = (sorted.len() as f64 - 1.0) * p;
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the index is floor of a non-negative position below the replicate count"
    )]
    let k = h.floor() as usize;
    if k + 1 >= sorted.len() {
        return sorted[sorted.len() - 1];
    }
    sorted[k] + (h - h.floor()) * (sorted[k + 1] - sorted[k])
}

#[test]
fn the_percentile_step_is_the_type_7_quantile_of_the_sorted_replicates() {
    let values: Vec<f64> = (1..=199).rev().map(f64::from).collect();
    let (lo, hi) = percentile_interval(&values, 0.95).unwrap();
    assert!((lo - 5.95).abs() < 1e-12 && (hi - 194.05).abs() < 1e-12, "{lo} {hi}");
    assert!(percentile_interval(&values, 1.0).is_none());
}

#[test]
fn the_joint_outer_bootstrap_interval_equals_hand_percentiles_of_its_replicates() {
    let options = LearnedContinuousOptions {
        outcome: LearnerSpec::Linear(LinearSpec::default()),
        folds: 3,
        bootstrap: 199,
        ..LearnedContinuousOptions::default()
    };
    let (diagram, query, _) = graph();
    let input = draw(Design::NestedCohort, Scenario::Good, 300, 200, 11);
    let id = antecedent_identify::TransportIdentifier::new().identify(&diagram, &query).unwrap();
    let run = learned_continuous_interval_internal(
        &id,
        &input,
        &options,
        &ExecutionContext::for_tests(11),
    )
    .unwrap();
    assert_eq!(run.failures, 0, "{:?}", run.uncertainty_reason);
    let values: Vec<f64> = run.replicates.iter().map(|(_, v)| *v).collect();
    assert_eq!(values.len(), 199);
    let (lo, hi) = run.interval.expect("every replicate succeeded");
    assert!((lo - hand_quantile(&values, 0.025)).abs() < 1e-12);
    assert!((hi - hand_quantile(&values, 0.975)).abs() < 1e-12);
}
