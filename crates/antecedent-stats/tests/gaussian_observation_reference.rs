//! Gaussian observation-likelihood terms against 60-digit `mpmath` values.
//!
//! Each constant is `ln` of the stated term computed in `mpmath` (dps = 60) from the
//! definitions: exact `ln(phi(z)/sigma)`, left censoring `ln Phi((b-mu)/sigma)`, right censoring
//! `ln(1 - Phi((b-mu)/sigma))`, interval censoring `ln(Phi(u') - Phi(l'))` and truncation
//! `ln(phi(z)/sigma) - ln(Phi(u') - Phi(l'))`. They cover deep tails, intervals that straddle or
//! touch the mean, and intervals of width down to 1e-12 standard deviations, where differencing
//! two CDF values near one half or two nearby tail logarithms loses digits.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_stats::{GaussianObservation as G, gaussian_observation_log_likelihood as loglik};

fn one(observation: G, mean: f64, sigma: f64) -> f64 {
    loglik(&[observation], &[mean], sigma).unwrap()
}

fn assert_close(got: f64, expected: f64, tolerance: f64, label: &str) {
    assert!(
        (got - expected).abs() <= tolerance * (1.0 + expected.abs()),
        "{label}: {got} vs {expected}"
    );
}

#[test]
fn exact_and_censored_terms_match_the_reference_including_deep_tails() {
    assert_close(one(G::Exact(1.3), 0.4, 0.8), -1.328_607_481_890_463, 1e-13, "exact");
    assert_close(one(G::Exact(-50.0), 0.0, 1.0), -1_250.918_938_533_204_7, 1e-13, "exact tail");
    assert_close(one(G::LeftCensored(0.5), 1.0, 2.0), -0.913_061_764_811_135_06, 1e-13, "left");
    assert_close(
        one(G::LeftCensored(-40.0), 0.0, 1.0),
        -804.608_442_013_753_79,
        1e-13,
        "left tail",
    );
    assert_close(
        one(G::LeftCensored(9.0), 0.0, 1.0),
        -1.128_588_405_953_840_6e-19,
        1e-13,
        "left top",
    );
    assert_close(one(G::RightCensored(0.5), 1.0, 2.0), -0.512_984_075_409_430_43, 1e-13, "right");
    assert_close(
        one(G::RightCensored(40.0), 0.0, 1.0),
        -804.608_442_013_753_79,
        1e-13,
        "right tail",
    );
}

#[test]
fn interval_masses_keep_precision_for_narrow_straddling_and_tail_intervals() {
    // (lower, upper, mean, sigma, ln mass)
    let cases = [
        (-1e-12, 1e-12, 0.0, 1.0, -27.856_812_468_573_276),
        (-3e-9, 1e-9, 0.0, 1.0, -20.255_910_009_031_193),
        (0.0, 1e-10, 0.0, 1.0, -23.944_789_463_145_13),
        (-1e-12, 3e-12, 0.0, 2.0, -27.856_812_468_573_276),
        (-1.0, 2.0, 0.5, 1.5, -0.381_715_146_302_126_07),
        (8.0, 9.0, 0.0, 1.0, -35.013_618_593_437_148),
        (-40.0, -39.0, 0.0, 1.0, -765.083_156_564_377_54),
        (5.0, 5.000_000_001, 0.0, 1.0, -34.142_204_289_910_717),
        (1.0, 1.000_000_000_001, 0.0, 1.0, -29.049_870_752_502_803),
        (-1.000_000_000_001, -1.0, 0.0, 1.0, -29.049_870_752_502_803),
        (2.0, 2.2, 0.0, 1.0, -4.727_712_530_306_536_5),
        (2.0, 2.3, 0.0, 1.0, -4.420_682_483_108_198_7),
        (0.0, 0.2, 0.0, 1.0, -2.535_025_357_147_548_8),
        (0.01, 0.2, 0.0, 1.0, -2.586_668_938_509_589_5),
        (20.0, 20.2, 0.0, 1.0, -203.935_091_341_032_67),
    ];
    for (lower, upper, mean, sigma, expected) in cases {
        let got = one(G::IntervalCensored { lower, upper }, mean, sigma);
        assert_close(
            got,
            expected,
            1e-12,
            &format!("[{lower}, {upper}] mean {mean} sigma {sigma}"),
        );
    }
}

#[test]
fn truncated_terms_divide_by_the_interval_mass() {
    let t =
        |value, lower, upper, mean, sigma| one(G::Truncated { value, lower, upper }, mean, sigma);
    assert_close(t(0.3, -1.0, 2.0, 0.5, 1.5), -0.951_577_383_899_599_94, 1e-13, "interior");
    assert_close(t(0.0, -1e-12, 1e-12, 0.0, 1.0), 26.937_873_935_368_603, 1e-12, "narrow");
    assert_close(t(10.5, 10.0, 11.0, 0.0, 1.0), -2.812_628_307_621_547_9, 1e-12, "tail");
    assert_close(
        t(0.2, f64::NEG_INFINITY, 1.0, 0.0, 1.0),
        -0.766_184_754_181_222_85,
        1e-13,
        "no lower",
    );
    assert_close(
        t(0.2, -1.0, f64::INFINITY, 0.0, 1.0),
        -0.766_184_754_181_222_85,
        1e-13,
        "no upper",
    );
}

#[test]
fn invalid_or_impossible_terms_are_refused_not_scored() {
    assert_eq!(
        loglik(&[G::Truncated { value: 3.0, lower: -1.0, upper: 2.0 }], &[0.0], 1.0)
            .unwrap_err()
            .to_string(),
        "unsupported: truncated Gaussian observation lies outside sampling bounds"
    );
    assert_eq!(
        loglik(&[G::IntervalCensored { lower: 1.0, upper: 1.0 }], &[0.0], 1.0)
            .unwrap_err()
            .to_string(),
        "unsupported: Gaussian observation bounds must be ordered"
    );
    assert_eq!(
        loglik(&[G::IntervalCensored { lower: 2.0, upper: 1.0 }], &[0.0], 1.0)
            .unwrap_err()
            .to_string(),
        "unsupported: Gaussian observation bounds must be ordered"
    );
    assert_eq!(loglik(&[G::Exact(f64::NAN)], &[0.0], 1.0).unwrap_err().to_string(), "unsupported: Gaussian observation endpoint must be finite");
    assert_eq!(loglik(&[G::Exact(0.0)], &[0.0], 0.0).unwrap_err().to_string(), "unsupported: Gaussian means/scale must be finite and scale positive");
    // Mass below the representable log scale of the tail is zero probability, not a number.
    assert!(loglik(&[G::IntervalCensored { lower: 60.0, upper: 61.0 }], &[0.0], 1.0).is_ok());
    // The sum over rows adds the per-row terms.
    let both = loglik(&[G::Exact(1.3), G::LeftCensored(0.5)], &[0.4, 1.0], 0.8).unwrap();
    let a = one(G::Exact(1.3), 0.4, 0.8);
    let b = one(G::LeftCensored(0.5), 1.0, 0.8);
    assert!((both - (a + b)).abs() <= 1e-13);
}
