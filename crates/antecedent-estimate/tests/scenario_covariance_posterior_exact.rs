//! The shared-scenario covariance posterior means, run from the route's exact-enumeration
//! covariance and entry sampling variances, against values derived by hand on a two-row sample.
//!
//! Two rows, scenario `a` = share of draws equal to row 0 and `b` = share equal to row 1. The
//! resampled count vectors are (2,0), (1,1), (0,2) with probabilities 1/4, 1/2, 1/4, so the
//! replicates (a, b) are (1, 0), (1/2, 1/2), (0, 1) with means (1/2, 1/2) and covariance
//! `[[1/8, -1/8], [-1/8, 1/8]]`. Each entry's centered product takes the values (1/4, 0, 1/4)
//! (or their negatives), mean 1/8 in absolute value, so the sampling variance of every entry is
//! `(1 / n_rows) * sum_w (product - mean)^2 = (1/2) * (1/4 + 1/2 + 1/4) * (1/8)^2 = 1/128`.
//! The posterior mean of an entry is `(1/v) / (1/v + 1/scale^2) * entry`.

use std::sync::Arc;

use antecedent_core::ExecutionContext;
use antecedent_estimate::posterior_mean::{PosteriorRequest, covariance_entry_posterior};
use antecedent_estimate::scenario_covariance::{
    ExactEnumerationOptions, ScenarioRowEstimator, exact_enumeration_covariance,
};

fn estimator(id: &str, row: usize) -> ScenarioRowEstimator {
    let units: Vec<Arc<str>> = vec![Arc::from("u0"), Arc::from("u1")];
    ScenarioRowEstimator::new(id, "snapshot:two-rows", units, move |counts| {
        Ok(f64::from(counts[row]) / 2.0)
    })
}

#[test]
fn scenario_covariance_posterior_equals_the_hand_values_of_the_two_row_enumeration() {
    let family = vec![estimator("a", 0), estimator("b", 1)];
    let out = exact_enumeration_covariance(
        &family,
        &ExactEnumerationOptions { max_compositions: 100, max_failure_mass: 0.0 },
        &ExecutionContext::for_tests(1),
    )
    .unwrap();
    let want_cov = [0.125, -0.125, -0.125, 0.125];
    let variance = 1.0 / 128.0;
    assert_eq!(out.covariance.len(), 4);
    for (got, want) in out.covariance.iter().zip(&want_cov) {
        assert!((got - want).abs() < 1e-12, "covariance {got} vs {want}");
    }
    for got in &out.entry_sampling_variance {
        assert!((got - variance).abs() < 1e-12, "entry sampling variance {got} vs {variance}");
    }
    for scale in [0.1, 0.5] {
        let precision = 1.0 / variance;
        let weight = precision / (precision + 1.0 / (scale * scale));
        let posterior = covariance_entry_posterior(
            &out.covariance,
            &out.entry_sampling_variance,
            PosteriorRequest::IndependentNormal { scale },
        )
        .unwrap();
        for (got, entry) in posterior.iter().zip(&want_cov) {
            assert!(
                (got - weight * entry).abs() < 1e-12,
                "scale {scale}: {got} vs {}",
                weight * entry
            );
        }
        assert!(posterior[0] < 0.125 - 1e-3, "the prior must shrink the entries");
    }
}
