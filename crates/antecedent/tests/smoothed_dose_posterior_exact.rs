//! The smoothed dose-response transport posterior mean, run from the route's own per-row score
//! contributions, against the normal-normal update written out by hand for every grid dose.
//!
//! The scores come from the full route (identification, cross-fitted nuisances, kernel
//! quadrature, augmentation); each dose's score mean is tied to its reported estimate, and the
//! expected posterior mean is `(n / s^2) / (n / s^2 + 1 / scale^2) * mean` with `s^2` the
//! sample variance of that dose's scores, at prior scales that visibly shrink the mean.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "smoothed_dose_dgp/mod.rs"]
mod dgp;

use antecedent_core::ExecutionContext;
use antecedent_estimate::SmoothedDoseOptions;
use antecedent_estimate::posterior_mean::{PosteriorRequest, smoothed_dose_score_posterior};
use antecedent_identify::TransportIdentifier;
use dgp::{Design, Scenario, diagram, draw, query};

const GRID: [f64; 3] = [1.0, 2.0, 3.0];
const H: f64 = 0.5;

#[test]
fn smoothed_dose_posterior_means_equal_the_hand_update_of_the_route_scores() {
    let options = SmoothedDoseOptions { folds: 3, ..SmoothedDoseOptions::default() };
    let q = query(&GRID, H);
    let (diagram, _) = diagram();
    let id = TransportIdentifier::new().identify(&diagram, &q.transport_query()).unwrap();
    let input = draw(Design::NestedCohort, Scenario::Good, 1500, 1000, 11);
    let estimate = antecedent_estimate::estimate_smoothed_dose(
        &id,
        &q,
        &input,
        &options,
        &ExecutionContext::for_tests(11),
    )
    .unwrap();
    let grids: Vec<Vec<f64>> = estimate.grid.iter().map(|point| point.row_scores.clone()).collect();
    for scale in [0.05, 0.5] {
        let got =
            smoothed_dose_score_posterior(&grids, PosteriorRequest::IndependentNormal { scale })
                .unwrap();
        assert_eq!(got.len(), estimate.grid.len());
        for ((point, posterior), scores) in estimate.grid.iter().zip(&got).zip(&grids) {
            let n = scores.len() as f64;
            let m = scores.iter().sum::<f64>() / n;
            assert!((m - point.estimate).abs() < 1e-8, "{m} vs {}", point.estimate);
            let s2 = scores.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (n - 1.0);
            let precision = n / s2;
            let want = precision / (precision + 1.0 / (scale * scale)) * m;
            assert!((posterior - want).abs() < 1e-12, "dose {}: {posterior} vs {want}", point.dose);
            assert!((posterior - m).abs() > 1e-4, "the prior must move the mean");
        }
    }
}
