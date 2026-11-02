//! Whole-method calibration of the credible intervals of the two measured choice cells: the
//! equal-tailed 95% interval of the first option's share, scored against the share drawn from
//! the declared prior.
//!
//! Nothing here is measured at implementation time: every test is `#[ignore]`d and runs only at
//! the final measurement step, and no interval is licensed until its record exists.
//!
//! * Pooled (`choice_dirichlet_share_*`): one choice per respondent, the uniform Dirichlet
//!   prior, shares drawn from it, one final analysis. Axis: options {2, 3, 5}, respondents on
//!   the three-point grid around 50. The interval is the Beta marginal of the Dirichlet
//!   posterior, so the nominal level is exact for a correct posterior.
//! * Hierarchical (`choice_hierarchical_share_*`): three options, three choices per respondent,
//!   uniform population-share shapes and `ln kappa ~ Normal(ln 5, 1)`; shares, heterogeneity
//!   and respondents are drawn from that hyperprior, respondents on the grid around 30. The
//!   interval is the quadrature posterior's marginal quantiles. A replicate whose posterior the
//!   quadrature does not resolve is skipped and counted, not scored.
//!
//! The truth of both designs is drawn from the declared prior, so the reading is prior
//! predictive; no pointwise frequentist coverage is claimed, and stated choices do not identify
//! behaviour.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg(feature = "calibration-internal")]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "bounded simulation sizes and small indices"
)]

#[path = "../../antecedent/tests/common/calibration.rs"]
mod calibration;
#[path = "common/candidate_calibration.rs"]
mod candidate;
#[path = "../../antecedent/tests/common/choice_dgp.rs"]
mod dgp;

use antecedent_core::ExecutionContext;
use antecedent_estimate::choice_inference::{
    CHOICE_INTERVAL_METHOD, DIRICHLET_ESTIMATOR, DIRICHLET_PRIOR, FIRST_OPTION_SHARE_FUNCTIONAL,
    HIERARCHICAL_DEPENDENCE, HIERARCHICAL_ESTIMATOR, HIERARCHICAL_PRIOR, HierarchicalChoicePrior,
    MEASURED_LEVEL, POOLED_DEPENDENCE, calibration_basis, choice_posterior,
    dirichlet_share_interval, hierarchical_share_interval,
};
use calibration::{CoverageTally, RecordKey, grid_n, grid_seed, map_replicates, n_sim, uniform};
use dgp::{choices, dirichlet, gamma_any};

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(0x62)
}

const POOLED_DGP: &str =
    "uniform_dirichlet_shares_one_choice_per_respondent_dirichlet_multinomial_posterior";
const HIERARCHICAL_DGP: &str = "uniform_population_shares_ln_kappa_normal_ln5_sd1_three_choices_per_respondent_hierarchical_posterior";

fn pooled(test: &'static str, options: usize, expected_id: &str) {
    let n = grid_n(50);
    let prior = vec![1.0; options];
    let mut tally = CoverageTally::for_record(
        RecordKey { test, dgp: POOLED_DGP, interval: CHOICE_INTERVAL_METHOD },
        MEASURED_LEVEL,
    );
    let basis = calibration_basis(
        DIRICHLET_ESTIMATOR,
        FIRST_OPTION_SHARE_FUNCTIONAL,
        POOLED_DEPENDENCE,
        DIRICHLET_PRIOR,
        n as u64,
    );
    let results = map_replicates(n_sim(), |replicate| {
        let seed = grid_seed(0x63_0000 + replicate);
        let theta = dirichlet(&prior, seed);
        let mut next = uniform(seed, 21);
        let counts = choices(&theta, n, &mut next);
        let posterior = choice_posterior(&prior, &counts, &ctx()).ok()?;
        let interval = dirichlet_share_interval(&posterior.alpha, 0, MEASURED_LEVEL).ok()?;
        Some((interval, theta[0]))
    });
    for result in results {
        let Some((interval, truth)) = result else {
            tally.skip();
            continue;
        };
        candidate::bind(&mut tally, &basis);
        tally.record(Some(interval), truth);
    }
    assert_eq!(tally.record_id().as_deref(), Some(expected_id));
    tally.assert();
}

#[allow(clippy::cast_sign_loss, reason = "choice counts are nonnegative integers")]
fn hierarchical(test: &'static str, expected_id: &str) {
    let n = grid_n(30);
    let hyper = HierarchicalChoicePrior {
        shapes: vec![1.0; 3],
        log_concentration_mean: 5f64.ln(),
        log_concentration_sd: 1.0,
    };
    let mut tally = CoverageTally::for_record(
        RecordKey { test, dgp: HIERARCHICAL_DGP, interval: CHOICE_INTERVAL_METHOD },
        MEASURED_LEVEL,
    );
    let basis = calibration_basis(
        HIERARCHICAL_ESTIMATOR,
        FIRST_OPTION_SHARE_FUNCTIONAL,
        HIERARCHICAL_DEPENDENCE,
        HIERARCHICAL_PRIOR,
        n as u64,
    );
    let results = map_replicates(n_sim(), |replicate| {
        let seed = grid_seed(0x64_0000 + replicate);
        let mut normal = calibration::gaussian(seed);
        let mut unit = uniform(seed, 22);
        let pi = dirichlet(&hyper.shapes, seed);
        let kappa = (hyper.log_concentration_mean + hyper.log_concentration_sd * normal()).exp();
        let mut next = uniform(seed, 23);
        let counts: Vec<Vec<u32>> = (0..n)
            .map(|_| {
                let g: Vec<f64> =
                    pi.iter().map(|p| gamma_any(kappa * p, &mut normal, &mut unit)).collect();
                let total: f64 = g.iter().sum();
                let theta: Vec<f64> = g.iter().map(|x| x / total).collect();
                choices(&theta, 3, &mut next).iter().map(|c| *c as u32).collect()
            })
            .collect();
        let interval =
            hierarchical_share_interval(&hyper, &counts, 0, MEASURED_LEVEL, &ctx()).ok()?;
        Some((interval, pi[0]))
    });
    for result in results {
        let Some((interval, truth)) = result else {
            tally.skip();
            continue;
        };
        candidate::bind(&mut tally, &basis);
        tally.record(Some(interval), truth);
    }
    assert_eq!(tally.record_id().as_deref(), Some(expected_id));
    tally.assert();
}

#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn choice_dirichlet_share_options_2_l95() {
    pooled(
        "choice_dirichlet_share_options_2_l95",
        2,
        "cov.choice_experiment_share.not_applicable.bayesian.posterior_quantile.l95.choice_dirichlet_share_options_2_l95.v2_validation_not_applicable",
    );
}

#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn choice_dirichlet_share_options_3_l95() {
    pooled(
        "choice_dirichlet_share_options_3_l95",
        3,
        "cov.choice_experiment_share.not_applicable.bayesian.posterior_quantile.l95.choice_dirichlet_share_options_3_l95.v2_validation_not_applicable",
    );
}

#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn choice_dirichlet_share_options_5_l95() {
    pooled(
        "choice_dirichlet_share_options_5_l95",
        5,
        "cov.choice_experiment_share.not_applicable.bayesian.posterior_quantile.l95.choice_dirichlet_share_options_5_l95.v2_validation_not_applicable",
    );
}

#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn choice_hierarchical_share_options_3_l95() {
    hierarchical(
        "choice_hierarchical_share_options_3_l95",
        "cov.choice_experiment_share.not_applicable.bayesian.posterior_quantile.l95.choice_hierarchical_share_options_3_l95.v2_validation_not_applicable",
    );
}
