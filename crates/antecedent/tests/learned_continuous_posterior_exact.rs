//! The learned continuous-outcome trial-transport posterior mean, run from the route's own
//! per-row cross-fitted scores, against the normal-normal update written out by hand.
//!
//! The scores come from the full route (identification, cross-fitted nuisances, augmented
//! inverse-odds score); their mean is tied to the reported estimate, and the expected posterior
//! mean is `(n / s^2) / (n / s^2 + 1 / scale^2) * mean` with `s^2` the sample variance of the
//! scores, at prior scales that visibly shrink the mean.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "learned_continuous_dgp/mod.rs"]
mod dgp;

use antecedent_core::ExecutionContext;
use antecedent_estimate::posterior_mean::{PosteriorRequest, learned_continuous_score_posterior};
use antecedent_estimate::{LearnedContinuousOptions, LearnerSpec, LinearSpec};
use dgp::{Design, Scenario, draw, graph};

#[test]
fn learned_continuous_posterior_mean_equals_the_hand_update_of_the_route_scores() {
    let options = LearnedContinuousOptions {
        outcome: LearnerSpec::Linear(LinearSpec::default()),
        folds: 3,
        ..LearnedContinuousOptions::default()
    };
    for design in [Design::NestedCohort, Design::IndependentSamples] {
        let (diagram, query, _) = graph();
        let input = draw(design, Scenario::Good, 800, 500, 11);
        let id =
            antecedent_identify::TransportIdentifier::new().identify(&diagram, &query).unwrap();
        let estimate = antecedent_estimate::estimate_learned_continuous(
            &id,
            &input,
            &options,
            &ExecutionContext::for_tests(11),
        )
        .unwrap();
        let n = estimate.scores.len() as f64;
        let m = estimate.scores.iter().sum::<f64>() / n;
        assert!((m - estimate.estimate).abs() < 1e-9, "{m} vs {}", estimate.estimate);
        let s2 = estimate.scores.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (n - 1.0);
        for scale in [0.05, 0.5] {
            let precision = n / s2;
            let want = precision / (precision + 1.0 / (scale * scale)) * m;
            let got = learned_continuous_score_posterior(
                &estimate.scores,
                PosteriorRequest::IndependentNormal { scale },
            )
            .unwrap();
            assert!((got - want).abs() < 1e-12, "{design:?} scale {scale}: {got} vs {want}");
            assert!((got - m).abs() > 1e-4, "the prior must move the mean");
        }
    }
}
