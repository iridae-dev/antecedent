//! Oracles for the exact participant-aware posterior.
//!
//! The references are independent of the implementation under test: the beta-binomial mass is the
//! gamma-function form (`betaln`), the integrals are dense midpoint sums over the success
//! probability (and over the correlation when it is estimated) computed in Python with
//! numpy/scipy, and the two-arm case is an arbitrary-precision nested quadrature (mpmath, 25
//! digits). At a zero correlation the posterior is also compared with the conjugate Beta route.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::needless_range_loop,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::unreadable_literal,
    clippy::excessive_precision,
    reason = "the oracles index per-arm columns and weight counts, and the references are printed digits of an independent computation"
)]

use antecedent_stats::adaptive_inference::{
    AdaptiveInferenceError as Error, ArmEvidence, BetaPosterior, CorrelationModel,
    CorrelationPrior, ExactPosterior, PosteriorRequest, best_arm_probabilities, beta_binomial_pmf,
    exact_posterior, expected_losses, probability_exceeds,
};

fn never() -> bool {
    false
}

fn close(actual: f64, expected: f64, tolerance: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{what}: expected {expected} got {actual} (tolerance {tolerance})"
    );
}

fn close_all(actual: &[f64], expected: &[f64], tolerance: f64, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        close(*a, *e, tolerance, &format!("{what}[{i}]"));
    }
}

struct Case {
    responses: u32,
    priors: Vec<(f64, f64)>,
    counts: Vec<Vec<u64>>,
}

impl Case {
    fn posterior(
        &self,
        correlation: CorrelationModel,
        margin: Option<f64>,
    ) -> Result<ExactPosterior, Error> {
        let arms: Vec<ArmEvidence<'_>> = self
            .priors
            .iter()
            .zip(&self.counts)
            .map(|(p, c)| ArmEvidence { prior_alpha: p.0, prior_beta: p.1, counts: c })
            .collect();
        exact_posterior(
            &PosteriorRequest {
                responses: self.responses,
                correlation,
                arms: &arms,
                exceedance_margin: margin,
            },
            &never,
        )
    }
}

fn three_responses() -> Case {
    Case {
        responses: 3,
        priors: vec![(1.0, 1.0), (2.0, 3.0), (1.0, 1.0)],
        counts: vec![vec![3, 5, 4, 2], vec![1, 3, 6, 6], vec![2, 2, 5, 9]],
    }
}

fn six_responses() -> Case {
    Case {
        responses: 6,
        priors: vec![(3.0, 4.0), (1.0, 1.0), (1.0, 1.0)],
        counts: vec![
            vec![2, 3, 5, 8, 9, 7, 6],
            vec![1, 2, 4, 7, 10, 9, 7],
            vec![0, 1, 3, 6, 11, 12, 7],
        ],
    }
}

#[test]
fn a_two_arm_posterior_matches_the_arbitrary_precision_quadrature() {
    let case = Case {
        responses: 3,
        priors: vec![(1.0, 1.0), (2.0, 3.0)],
        counts: vec![vec![2, 1, 1, 0], vec![0, 1, 2, 1]],
    };
    let got = case.posterior(CorrelationModel::Declared(0.3), None).unwrap();
    close_all(&got.mean, &[0.307_627_009_435_763_4, 0.551_698_720_751_698_4], 1e-12, "mean");
    close_all(&got.sd, &[0.140_351_145_358_944_83, 0.131_567_918_576_018_58], 1e-12, "sd");
    close_all(
        &got.probability_best,
        &[0.108_559_743_289_141_44, 0.891_440_256_710_858_5],
        1e-12,
        "probability best",
    );
    close_all(
        &got.expected_loss,
        &[0.255_029_849_367_989_7, 0.010_958_138_052_054_783],
        1e-12,
        "expected loss",
    );
    assert!(got.probability_error < 1e-12, "{}", got.probability_error);
}

#[test]
fn the_declared_correlation_posterior_matches_the_dense_enumeration() {
    let got = three_responses().posterior(CorrelationModel::Declared(0.3), Some(0.05)).unwrap();
    close_all(&got.mean, &[0.459_121_732_387, 0.6381996247, 0.710_911_633_585], 5e-9, "mean");
    close_all(&got.sd, &[0.0902144862751, 0.0779836296874, 0.0744606150727], 5e-9, "sd");
    close_all(
        &got.probability_best,
        &[0.00769079795408, 0.246_908_501_017, 0.74540070102],
        5e-9,
        "best",
    );
    close_all(
        &got.expected_loss,
        &[0.268_263_572_017, 0.0891856797042, 0.0164736708197],
        5e-9,
        "loss",
    );
    close_all(&got.exceedance, &[0.858_595_607_782, 0.954_635_527_113], 5e-9, "exceedance");
    assert!(got.probability_error < 1e-9, "{}", got.probability_error);
    assert_eq!(got.correlation_nodes, 1);

    let six = six_responses().posterior(CorrelationModel::Declared(0.15), Some(0.0)).unwrap();
    close_all(&six.mean, &[0.593_035_685_572, 0.655_592_362_758, 0.70164447239], 5e-9, "mean");
    close_all(&six.sd, &[0.0410026836466, 0.039946501819, 0.0376495432524], 5e-9, "sd");
    close_all(
        &six.probability_best,
        &[0.0136829731291, 0.196_484_451_586, 0.789_832_575_231],
        5e-9,
        "best",
    );
    close_all(
        &six.expected_loss,
        &[0.115_021_075_262, 0.0524643980756, 0.00641228844302],
        5e-9,
        "loss",
    );
    close_all(&six.exceedance, &[0.862_464_538_913, 0.973_980_661_012], 5e-9, "exceedance");
}

#[test]
fn a_zero_correlation_reproduces_the_conjugate_beta_route() {
    let case = three_responses();
    let got = case.posterior(CorrelationModel::Declared(0.0), Some(0.05)).unwrap();
    let beta: Vec<BetaPosterior> = case
        .priors
        .iter()
        .zip(&case.counts)
        .map(|(p, counts)| {
            let successes: u64 = counts.iter().enumerate().map(|(s, &c)| s as u64 * c).sum();
            let participants: u64 = counts.iter().sum();
            let failures = participants * 3 - successes;
            BetaPosterior::from_counts(p.0, p.1, successes, failures).unwrap()
        })
        .collect();
    close_all(&got.mean, &beta.iter().map(|b| b.mean()).collect::<Vec<_>>(), 1e-10, "mean");
    close_all(&got.probability_best, &best_arm_probabilities(&beta, &never).unwrap(), 1e-8, "best");
    close_all(&got.expected_loss, &expected_losses(&beta, &never).unwrap(), 1e-8, "loss");
    for k in 1..3 {
        close(
            got.exceedance[k - 1],
            probability_exceeds(beta[k], beta[0], 0.05, &never).unwrap(),
            1e-8,
            "exceedance",
        );
    }
}

#[test]
fn the_estimated_correlation_is_integrated_jointly_with_the_arm_means() {
    let case = three_responses();
    let got = case
        .posterior(
            CorrelationModel::Estimated(CorrelationPrior { alpha: 2.0, beta: 6.0 }),
            Some(0.05),
        )
        .unwrap();
    close_all(&got.mean, &[0.457_212_074_963, 0.64570832464, 0.715_712_231_232], 2e-6, "mean");
    close_all(&got.sd, &[0.0870269136015, 0.0754783178173, 0.0715557369654], 2e-6, "sd");
    close_all(
        &got.probability_best,
        &[0.00539347084645, 0.247_187_689_525, 0.747_418_839_467],
        2e-6,
        "best",
    );
    close_all(
        &got.expected_loss,
        &[0.274_162_973_384, 0.0856667237071, 0.0156628171144],
        2e-6,
        "loss",
    );
    close_all(&got.exceedance, &[0.882_986_880_044, 0.964_580_934_209], 2e-6, "exceedance");
    close(got.correlation_mean, 0.220_994_523_757, 2e-6, "correlation mean");
    close(got.correlation_sd, 0.0812951634713, 2e-6, "correlation sd");
    assert!(got.probability_error < 1e-9, "{}", got.probability_error);

    let six = six_responses()
        .posterior(
            CorrelationModel::Estimated(CorrelationPrior { alpha: 1.0, beta: 1.0 }),
            Some(0.02),
        )
        .unwrap();
    close_all(&six.mean, &[0.593_197_754_449, 0.65575372049, 0.702_162_916_005], 2e-6, "mean");
    close_all(
        &six.probability_best,
        &[0.0131855744412, 0.192_072_426_964, 0.794_741_997_287],
        2e-6,
        "best",
    );
    close_all(&six.exceedance, &[0.774_216_693_659, 0.94569006946], 2e-6, "exceedance");
    close(six.correlation_mean, 0.143_041_976_189, 2e-6, "correlation mean");
    close(six.correlation_sd, 0.0353875875276, 2e-6, "correlation sd");
    assert!(six.probability_error < 1e-9, "{}", six.probability_error);
}

#[test]
fn the_estimated_correlation_recovers_the_truth_from_large_samples() {
    // Expected histograms of 4000 participants per arm at a correlation of 0.3.
    let responses = 6;
    let counts: Vec<Vec<u64>> = [0.35, 0.5, 0.6]
        .iter()
        .map(|&mean| {
            beta_binomial_pmf(responses, mean, 0.3)
                .iter()
                .map(|p| (p * 4000.0).round() as u64)
                .collect()
        })
        .collect();
    let case = Case { responses, priors: vec![(1.0, 1.0); 3], counts };
    let got = case
        .posterior(CorrelationModel::Estimated(CorrelationPrior { alpha: 1.0, beta: 1.0 }), None)
        .unwrap();
    assert!((got.correlation_mean - 0.3).abs() < 0.02, "{}", got.correlation_mean);
    assert!(got.correlation_sd < 0.02, "{}", got.correlation_sd);
    close_all(&got.mean, &[0.35, 0.5, 0.6], 0.01, "mean");
    // A narrow correlation posterior is an adaptive window, not a fixed grid.
    assert!(got.correlation_nodes <= 128, "{}", got.correlation_nodes);
    assert!(got.probability_error < 1e-8, "{}", got.probability_error);
    // The estimate agrees with the posterior at the correlation it estimates.
    let at_mean = case.posterior(CorrelationModel::Declared(got.correlation_mean), None).unwrap();
    close_all(&got.mean, &at_mean.mean, 2e-4, "mean at the estimate");
    close_all(&got.sd, &at_mean.sd, 2e-4, "sd at the estimate");
}

#[test]
fn without_participants_the_posterior_is_the_prior() {
    let case = Case {
        responses: 3,
        priors: vec![(1.0, 1.0); 3],
        counts: vec![vec![0; 4], vec![0; 4], vec![0; 4]],
    };
    for model in [
        CorrelationModel::Declared(0.4),
        CorrelationModel::Estimated(CorrelationPrior { alpha: 2.0, beta: 2.0 }),
    ] {
        let got = case.posterior(model, Some(0.0)).unwrap();
        close_all(&got.probability_best, &[1.0 / 3.0; 3], 1e-9, "best");
        // The expected maximum of three uniforms is 3/4.
        close_all(&got.expected_loss, &[0.25; 3], 1e-9, "loss");
        close_all(&got.mean, &[0.5; 3], 1e-10, "mean");
        close_all(&got.sd, &[(1.0_f64 / 12.0).sqrt(); 3], 1e-9, "sd");
        close_all(&got.exceedance, &[0.5, 0.5], 1e-9, "exceedance");
    }
}

#[test]
fn many_participants_do_not_underflow_or_lose_the_peak() {
    let case = Case {
        responses: 3,
        priors: vec![(1.0, 1.0); 3],
        counts: vec![
            vec![9000, 12000, 14000, 10000],
            vec![4000, 9000, 15000, 17000],
            vec![1000, 7000, 17000, 20000],
        ],
    };
    let got = case.posterior(CorrelationModel::Declared(0.2), Some(0.0)).unwrap();
    assert!((got.probability_best.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    assert!(got.probability_best[2] > 0.99);
    assert!(got.sd.iter().all(|&s| s > 0.0 && s < 0.01));
    assert!(got.probability_error < 1e-9, "{}", got.probability_error);
}

#[test]
fn invalid_requests_are_refused() {
    let case = three_responses();
    assert!(matches!(
        case.posterior(CorrelationModel::Declared(1.0), None),
        Err(Error::InvalidParameter(_))
    ));
    assert!(matches!(
        case.posterior(
            CorrelationModel::Estimated(CorrelationPrior { alpha: 0.5, beta: 1.0 }),
            None
        ),
        Err(Error::InvalidParameter(_))
    ));
    assert!(matches!(
        case.posterior(CorrelationModel::Declared(0.1), Some(1.0)),
        Err(Error::InvalidParameter(_))
    ));
    let mut wrong = three_responses();
    wrong.counts[0] = vec![1, 2];
    assert!(matches!(wrong.posterior(CorrelationModel::Declared(0.1), None), Err(Error::Shape(_))));
}
