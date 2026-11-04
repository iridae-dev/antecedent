//! Independent construction check of the choice-share credible interval.
//!
//! The pooled share of one option under a Dirichlet posterior is the Beta marginal
//! `Beta(a_j, sum(a) - a_j)`. With a remaining mass of one the CDF is `x^a`, so the equal-tailed
//! interval is explicit: the `p` quantile is `p^(1/a)`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_estimate::choice_inference::dirichlet_share_interval;

#[test]
fn the_dirichlet_share_interval_equals_the_explicit_beta_quantiles() {
    // Option 0 of Dirichlet(2, 1): Beta(2, 1) with CDF x^2, quantile sqrt(p).
    let (lo, hi) = dirichlet_share_interval(&[2.0, 1.0], 0, 0.95).unwrap();
    assert!((lo - 0.025_f64.sqrt()).abs() < 1e-9, "lower {lo}");
    assert!((hi - 0.975_f64.sqrt()).abs() < 1e-9, "upper {hi}");
    // Option 1 of Dirichlet(1, 3): Beta(3, 1) with CDF x^3, quantile p^(1/3).
    let (lo, hi) = dirichlet_share_interval(&[1.0, 3.0], 1, 0.9).unwrap();
    assert!((lo - 0.05_f64.powf(1.0 / 3.0)).abs() < 1e-9, "lower {lo}");
    assert!((hi - 0.95_f64.powf(1.0 / 3.0)).abs() < 1e-9, "upper {hi}");
    // Option 0 of Dirichlet(1, 1, 1): Beta(1, 2), quantile 1 - sqrt(1 - p).
    let (lo, hi) = dirichlet_share_interval(&[1.0, 1.0, 1.0], 0, 0.95).unwrap();
    assert!(
        (lo - (1.0 - 0.975_f64.sqrt())).abs() < 1e-9
            && (hi - (1.0 - 0.025_f64.sqrt())).abs() < 1e-9
    );
    assert!(dirichlet_share_interval(&[1.0, 1.0], 5, 0.95).is_err());
}
