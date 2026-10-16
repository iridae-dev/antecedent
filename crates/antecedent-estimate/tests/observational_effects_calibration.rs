//! Whole-route calibration designs for the observational survival and policy-value
//! standard errors. Nothing here is measured at implementation time: every test is
//! `#[ignore]`d and runs only at the final measurement step, and the routes report their
//! standard errors as `unmeasured` and publish no interval until a record exists.
//!
//! The scored statement is the Wald interval `estimate +- 1.96 se` built from the reported
//! influence-function standard error, against a truth computed from the generating process
//! (see `common/observational_effects.rs`), over `n_sim()` independent datasets.
//!
//! Designs (nominal level 0.95, sample sizes `grid_n(1000)` and `grid_n(3000)`):
//!
//! * `survival_*`: the survival probability difference and the restricted mean survival time
//!   difference at the horizon, with confounded treatment and censoring that depends on the
//!   arm and the confounders (coarsening at random, Cox censoring model), and the
//!   covariate-free censoring special case (Nelson-Aalen censoring model).
//! * `policy_prespecified_*`: a fixed threshold policy scored on all rows with cross-fitted
//!   nuisance models; the truth is `1 + 1.5 phi(0)`.
//! * `policy_held_out_*`: the greedy policy learned on a seeded learning part and scored on
//!   the independent held-out rows; the truth is the value of the one learned rule, computed
//!   from the threshold that separates the policy's own actions on the held-out rows.
//! * `policy_cross_fitted_learning_*`: the selection-aware value of the learning procedure
//!   fed four fifths of the data, against its expectation over independent training sets
//!   simulated with the same greedy rule.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "bounded simulation sizes and small indices"
)]

#[path = "../../antecedent/tests/common/calibration.rs"]
mod calibration;
#[path = "common/observational_effects.rs"]
mod dgp;

use antecedent_core::ExecutionContext;
use antecedent_estimate::observational_policy::{
    ObservationalPolicyValue, PolicyData, PolicyEvaluation, PolicyObjective, PolicyOrigin,
};
use antecedent_estimate::observational_survival::{
    CensoringAssumption, ObservationalSurvival, SurvivalData,
};
use calibration::{CoverageTally, Z95, grid_n, grid_seed, map_replicates, n_sim};
use dgp::{
    BinaryStudy, Censoring, HORIZON, PHI_0, binary_study, phi, survival_study, survival_truth,
};

const LEVEL: f64 = 0.95;

fn wald(estimate: f64, se: f64) -> Option<(f64, f64)> {
    (estimate.is_finite() && se.is_finite() && se > 0.0)
        .then_some((estimate - Z95 * se, estimate + Z95 * se))
}

#[derive(Clone, Copy)]
enum Target {
    Survival,
    Rmst,
}

fn survival_design(
    name: &'static str,
    base: usize,
    censoring: Censoring,
    assumption: CensoringAssumption,
    target: Target,
) {
    let n = grid_n(base);
    let (s1, r1) = survival_truth(1.0, HORIZON);
    let (s0, r0) = survival_truth(0.0, HORIZON);
    let truth = match target {
        Target::Survival => s1 - s0,
        Target::Rmst => r1 - r0,
    };
    let mut tally = CoverageTally::new(name, LEVEL);
    let results = map_replicates(n_sim(), |replicate| {
        let seed = grid_seed(0x5e_0000 + replicate);
        let data = survival_study(n, seed, censoring, 1.0);
        let estimate = ObservationalSurvival::new(1.0, 0.0, HORIZON, assumption)
            .estimate(
                &SurvivalData {
                    duration: &data.duration,
                    event: &data.event,
                    treatment: &data.treatment,
                    covariates: &[&data.z1, &data.z2],
                },
                &ExecutionContext::for_tests(seed),
            )
            .ok()?;
        let contrast = match target {
            Target::Survival => estimate.survival_difference,
            Target::Rmst => estimate.rmst_difference,
        };
        wald(contrast.estimate, contrast.se)
    });
    for interval in results {
        if interval.is_none() {
            tally.skip();
        } else {
            tally.record(interval, truth);
        }
    }
    tally.assert();
}

#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn survival_difference_covariate_censoring_n1000_l95() {
    survival_design(
        "survival_difference_covariate_censoring_n1000_l95",
        1000,
        Censoring::DependsOnCovariates(0.12),
        CensoringAssumption::AtRandomGivenCovariates,
        Target::Survival,
    );
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn survival_difference_covariate_censoring_n3000_l95() {
    survival_design(
        "survival_difference_covariate_censoring_n3000_l95",
        3000,
        Censoring::DependsOnCovariates(0.12),
        CensoringAssumption::AtRandomGivenCovariates,
        Target::Survival,
    );
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn rmst_difference_covariate_censoring_n1000_l95() {
    survival_design(
        "rmst_difference_covariate_censoring_n1000_l95",
        1000,
        Censoring::DependsOnCovariates(0.12),
        CensoringAssumption::AtRandomGivenCovariates,
        Target::Rmst,
    );
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn rmst_difference_covariate_censoring_n3000_l95() {
    survival_design(
        "rmst_difference_covariate_censoring_n3000_l95",
        3000,
        Censoring::DependsOnCovariates(0.12),
        CensoringAssumption::AtRandomGivenCovariates,
        Target::Rmst,
    );
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn survival_difference_independent_censoring_n1000_l95() {
    survival_design(
        "survival_difference_independent_censoring_n1000_l95",
        1000,
        Censoring::Independent(0.15),
        CensoringAssumption::IndependentWithinArm,
        Target::Survival,
    );
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn survival_difference_independent_censoring_n3000_l95() {
    survival_design(
        "survival_difference_independent_censoring_n3000_l95",
        3000,
        Censoring::Independent(0.15),
        CensoringAssumption::IndependentWithinArm,
        Target::Survival,
    );
}

fn estimate_policy(
    data: &BinaryStudy,
    evaluation: PolicyEvaluation,
    seed: u64,
) -> Option<antecedent_estimate::observational_policy::PolicyValueEstimate> {
    ObservationalPolicyValue::new(vec![0.0, 1.0], PolicyObjective::Maximize, evaluation)
        .estimate(
            &PolicyData {
                outcome: &data.outcome,
                treatment: &data.treatment,
                covariates: &[&data.z],
            },
            &ExecutionContext::for_tests(seed),
        )
        .ok()
}

fn record_all(tally: &mut CoverageTally, results: Vec<Option<((f64, f64), f64)>>) {
    for result in results {
        match result {
            Some((interval, truth)) => tally.record(Some(interval), truth),
            None => tally.skip(),
        }
    }
    tally.assert();
}

fn policy_prespecified_design(name: &'static str, base: usize) {
    let n = grid_n(base);
    let truth = 1.0 + 1.5 * PHI_0;
    let mut tally = CoverageTally::new(name, LEVEL);
    let results = map_replicates(n_sim(), |replicate| {
        let seed = grid_seed(0x6a_0000 + replicate);
        let data = binary_study(n, seed, 1.5, 0, 1.0);
        let actions = data.z.iter().map(|&z| f64::from(u8::from(z > 0.0))).collect::<Vec<_>>();
        let estimate = estimate_policy(
            &data,
            PolicyEvaluation::Prespecified {
                actions: actions.into(),
                origin: PolicyOrigin::Prespecified,
            },
            seed,
        )?;
        Some((wald(estimate.value, estimate.se)?, truth))
    });
    record_all(&mut tally, results);
}

/// The threshold `c` and direction of a learned rule `treat iff z > c` (or `z < c`), read off
/// the actions the policy took on the scored rows; `None` when it is not a threshold rule.
fn learned_rule_value(data: &BinaryStudy, rows: &[u32], actions: &[f64]) -> Option<f64> {
    let z = |i: usize| data.z[rows[i] as usize];
    let treated: Vec<f64> =
        (0..rows.len()).filter(|&i| (actions[i] - 1.0).abs() < f64::EPSILON).map(z).collect();
    let untreated: Vec<f64> = (0..rows.len()).filter(|&i| actions[i] == 0.0).map(z).collect();
    if treated.is_empty() || untreated.is_empty() {
        return None;
    }
    let fold =
        |values: &[f64], f: fn(f64, f64) -> f64, init: f64| values.iter().copied().fold(init, f);
    let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len() as f64;
    if mean(&treated) > mean(&untreated) {
        // Treat above the threshold: it lies between the highest untreated and lowest treated z.
        let (low, high) = (
            fold(&untreated, f64::max, f64::NEG_INFINITY),
            fold(&treated, f64::min, f64::INFINITY),
        );
        (low <= high).then(|| 1.0 + 1.5 * phi(0.5 * (low + high)))
    } else {
        let (low, high) = (
            fold(&treated, f64::max, f64::NEG_INFINITY),
            fold(&untreated, f64::min, f64::INFINITY),
        );
        (low <= high).then(|| 1.0 - 1.5 * phi(0.5 * (low + high)))
    }
}

fn policy_held_out_design(name: &'static str, base: usize) {
    let n = grid_n(base);
    let mut tally = CoverageTally::new(name, LEVEL);
    let results = map_replicates(n_sim(), |replicate| {
        let seed = grid_seed(0x6b_0000 + replicate);
        let data = binary_study(n, seed, 1.5, 0, 1.0);
        let estimate =
            estimate_policy(&data, PolicyEvaluation::HeldOut { evaluation_fraction: 0.4 }, seed)?;
        let truth = learned_rule_value(&data, &estimate.row_index, &estimate.policy_actions)?;
        Some((wald(estimate.value, estimate.se)?, truth))
    });
    record_all(&mut tally, results);
}

/// Simple linear regression `y ~ 1 + z`: `(intercept, slope)`.
fn simple_ols(z: &[f64], y: &[f64]) -> (f64, f64) {
    let n = z.len() as f64;
    let (mz, my) = (z.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
    let sxy: f64 = z.iter().zip(y).map(|(a, b)| (a - mz) * (b - my)).sum();
    let sxx: f64 = z.iter().map(|a| (a - mz).powi(2)).sum();
    (my - sxy / sxx * mz, sxy / sxx)
}

/// Expected true value of the greedy rule learned from `training` rows of this process,
/// over independent training sets.
fn expected_greedy_value(training: usize, draws: u64) -> f64 {
    let mut total = 0.0;
    for draw in 0..draws {
        let data = binary_study(training, grid_seed(0x6c_0000 + draw), 1.5, 0, 1.0);
        let arm = |level: f64| {
            let rows: Vec<usize> = (0..training)
                .filter(|&i| (data.treatment[i] - level).abs() < f64::EPSILON)
                .collect();
            simple_ols(
                &rows.iter().map(|&i| data.z[i]).collect::<Vec<_>>(),
                &rows.iter().map(|&i| data.outcome[i]).collect::<Vec<_>>(),
            )
        };
        let ((a0, b0), (a1, b1)) = (arm(0.0), arm(1.0));
        let slope = b1 - b0;
        let threshold = -(a1 - a0) / slope;
        total += 1.0 + 1.5 * slope.signum() * phi(threshold);
    }
    total / draws as f64
}

fn policy_cross_fitted_design(name: &'static str, base: usize) {
    let n = grid_n(base);
    // Five folds: each fold's policy is learned on four fifths of the rows.
    let truth = expected_greedy_value(n - n / 5, 1500);
    let mut tally = CoverageTally::new(name, LEVEL);
    let results = map_replicates(n_sim(), |replicate| {
        let seed = grid_seed(0x6d_0000 + replicate);
        let data = binary_study(n, seed, 1.5, 0, 1.0);
        let estimate = estimate_policy(&data, PolicyEvaluation::CrossFitted, seed)?;
        Some((wald(estimate.value, estimate.se)?, truth))
    });
    record_all(&mut tally, results);
}

#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn policy_prespecified_n1000_l95() {
    policy_prespecified_design("policy_prespecified_n1000_l95", 1000);
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn policy_prespecified_n3000_l95() {
    policy_prespecified_design("policy_prespecified_n3000_l95", 3000);
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn policy_held_out_n1000_l95() {
    policy_held_out_design("policy_held_out_n1000_l95", 1000);
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn policy_held_out_n3000_l95() {
    policy_held_out_design("policy_held_out_n3000_l95", 3000);
}
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn policy_cross_fitted_learning_n3000_l95() {
    policy_cross_fitted_design("policy_cross_fitted_learning_n3000_l95", 3000);
}
