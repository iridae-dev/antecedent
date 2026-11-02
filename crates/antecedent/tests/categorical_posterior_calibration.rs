//! Prior-predictive coverage of the bounded conjugate categorical contrast.
//!
//! This ignored calibration harness measures nothing until explicitly run. Its truth is
//! drawn from the declared Normal-Inverse-Gamma prior, then a homoskedastic Gaussian
//! sample is drawn conditionally. It makes no causal or misspecification guarantee.

#![cfg(feature = "calibration-internal")]

#[path = "common/calibration.rs"]
mod calibration;
#[path = "../../antecedent-estimate/tests/common/candidate_calibration.rs"]
mod candidate;

use antecedent_estimate::categorical_treatment::CategoricalTreatmentInput;
use antecedent_estimate::categorical_treatment_posterior::{
    CategoricalPosteriorPrior, categorical_prior_predictive_calibration_candidate,
};
use calibration::{CoverageTally, RecordKey, gaussian, grid_seed, map_replicates, n_sim, uniform};

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn categorical_conjugate_contrast_prior_predictive_interval() {
    let mut tally = CoverageTally::for_record(
        RecordKey {
            test: "categorical_conjugate_contrast_prior_predictive_interval",
            dgp: "two_level_normal_inverse_gamma_prior_predictive",
            interval: "posterior_quantile",
        },
        0.95,
    );
    let prior = CategoricalPosteriorPrior {
        level_precision: 1.0,
        nuisance_precision: 0.5,
        variance_shape: 2.0,
        variance_scale: 1.0,
    };
    let runs = map_replicates(n_sim(), |rep| {
        let seed = grid_seed(0x6361_7400 + rep);
        let mut uniform = uniform(seed, 19);
        let mut normal = gaussian(seed ^ 0x7e11);
        // If G is Gamma(shape=2, rate=1), 1/G is InvGamma(2, 1).
        let variance = 1.0 / (-(1.0 - uniform()).ln() - (1.0 - uniform()).ln());
        let scale = variance.sqrt();
        let intercept = scale * normal() / prior.nuisance_precision.sqrt();
        let contrast = scale * normal() / prior.level_precision.sqrt();
        let levels: Vec<String> =
            (0..40).map(|i| if i < 20 { "control" } else { "treated" }.to_owned()).collect();
        let outcome: Vec<f64> = (0..40)
            .map(|i| intercept + (if i < 20 { 0.0 } else { contrast }) + scale * normal())
            .collect();
        let input = CategoricalTreatmentInput {
            outcome: outcome.into(),
            row_snapshot: "prior-predictive".into(),
            adjustment: vec![],
            levels: levels.into(),
        };
        categorical_prior_predictive_calibration_candidate(&input)
            .ok()
            .map(|(interval, basis)| (interval, basis, contrast))
    });
    for sample in runs {
        let Some((interval, basis, truth)) = sample else {
            tally.skip();
            continue;
        };
        candidate::bind(&mut tally, &basis);
        tally.record(Some((interval[0], interval[1])), truth);
    }
    assert_eq!(
        tally.record_id().as_deref(),
        Some(
            "cov.categorical_treatment_contrast.not_applicable.bayesian.posterior_quantile.l95.categorical_conjugate_contrast_prior_predictive_interval.v2_validation_not_applicable"
        )
    );
    tally.assert();
}
