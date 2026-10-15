//! Coverage of the composed interval of a linear combination of two estimates.
//!
//! The whole method is the component estimators (sample means of two rows
//! columns), their sampling covariance and the licensed dependence route, and
//! the normal-approximation interval built on the composed value. Every design
//! scores the interval a sealed claim reports against the known composed truth
//! and binds the calibration key the claim itself derives
//! (`PropagationClaim::composed_interval_basis`), so a record describes exactly
//! the construction the public route resolves.
//!
//! Keys, and what each design measures:
//!
//! * `supplied_covariance` on the linear-moments route: the two estimates share
//!   rows and the covariance of the estimates is estimated from them.
//! * `declared_independent_distinct_evidence` on the linear-moments route: two
//!   independent samples with a diagonal covariance.
//! * `aligned_joint_law`: aligned bootstrap draws of the two estimates.
//!
//! A declared independence that is violated cannot be detected by the library,
//! so it is not a record under the independent key (a boundary there would
//! govern every honest declaration): it is a plain assertion that such a claim
//! under-covers, which is why the declaration is a premise.
//!
//! Ignored tests run via `scripts/gate_calibration.sh`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    reason = "bounded replicate indices in a calibration fixture"
)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "bounded simulation sizes and small indices"
)]

mod common;

use antecedent::analysis::joint_law_propagation::{
    LinearDependence, MomentInput, PropagationDraws, propagate_aligned_linear_spread,
    propagate_linear_moments_spread,
};
use antecedent_contract::{InputRole, PropagationOperation};
use antecedent_core::{CalibrationBasis, ExecutionContext};
use antecedent_stats::{NORMAL_APPROXIMATION_CONSTRUCTION, normal_interval_candidate};
use common::calibration::{
    Construction, CoverageTally, RecordKey, ScopeFacts, grid_n, grid_seed, map_replicates, n_sim,
    uniform,
};

const LEVEL: f64 = 0.95;
const BOOTSTRAP: usize = 200;

#[derive(Clone, Copy)]
enum Law {
    /// Bivariate normal, means 1.0 and 0.4, unit variances, correlation `rho`.
    Gaussian { rho: f64 },
    /// Shared-evidence skewed pair: `x = E1`, `y = 0.5 E1 + 0.5 E2`, exponential `E`.
    Skewed,
}

impl Law {
    /// The composed truth `E[x] - E[y]`.
    fn truth(self) -> f64 {
        match self {
            Self::Gaussian { .. } => 0.6,
            Self::Skewed => 0.0,
        }
    }
}

fn normals(seed: u64, tag: u64) -> impl FnMut() -> f64 {
    let mut next = uniform(seed, tag);
    move || {
        let (u, v) = (next(), next());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

fn rows(law: Law, n: usize, seed: u64, tag: u64) -> (Vec<f64>, Vec<f64>) {
    let (mut x, mut y) = (Vec::with_capacity(n), Vec::with_capacity(n));
    match law {
        Law::Gaussian { rho } => {
            let mut normal = normals(seed, tag);
            for _ in 0..n {
                let (z1, z2) = (normal(), normal());
                x.push(1.0 + z1);
                y.push(0.4 + rho * z1 + (1.0 - rho * rho).sqrt() * z2);
            }
        }
        Law::Skewed => {
            let mut next = uniform(seed, tag);
            for _ in 0..n {
                let (e1, e2) = (-next().ln(), -next().ln());
                x.push(e1);
                y.push(0.5 * e1 + 0.5 * e2);
            }
        }
    }
    (x, y)
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

/// Sample covariance matrix `[vx, c, c, vy]` of two columns.
fn covariance(x: &[f64], y: &[f64]) -> [f64; 4] {
    let (mx, my) = (mean(x), mean(y));
    let n = x.len() as f64 - 1.0;
    let (mut vx, mut vy, mut c) = (0.0, 0.0, 0.0);
    for (a, b) in x.iter().zip(y) {
        vx += (a - mx) * (a - mx);
        vy += (b - my) * (b - my);
        c += (a - mx) * (b - my);
    }
    [vx / n, c / n, c / n, vy / n]
}

fn bind_basis(tally: &mut CoverageTally, basis: &CalibrationBasis) {
    tally.bind(
        &Construction {
            query: basis.query.to_string(),
            graph_class: basis.graph_class.to_string(),
            structure: basis.structure.to_string(),
            modality: basis.modality.to_string(),
            inference: basis.inference.to_string(),
            estimator: basis.estimator.to_string(),
            interval_method: basis.interval_method.to_string(),
            se_kind: basis.se_kind.to_string(),
            dependence: basis.dependence.to_string(),
            posterior: basis.posterior.to_string(),
            functional: basis.functional.to_string(),
            identification: basis.identification.to_string(),
            reported_level: basis.level,
        },
        ScopeFacts {
            row_count: basis.row_count,
            replicates_ok: basis.replicates_ok,
            posterior_draws: basis.posterior_draws,
            unidentified_mass: basis.unidentified_mass,
        },
    );
}

/// The interval of one claim and the key it is measured under.
fn reading(
    claim: &antecedent::analysis::joint_law_propagation::PropagationClaim,
    rows: usize,
) -> (Option<(f64, f64)>, CalibrationBasis) {
    let basis = claim
        .composed_interval_basis(NORMAL_APPROXIMATION_CONSTRUCTION, rows as u64)
        .expect("composed key");
    let interval = match (claim.value("mean"), claim.value("variance")) {
        (Some(m), Some(v)) => {
            normal_interval_candidate(m, v, LEVEL).ok().map(|c| (c.lower, c.upper))
        }
        _ => None,
    };
    (interval, basis)
}

fn ctx(seed: u64) -> ExecutionContext {
    ExecutionContext::for_tests(seed)
}

fn moment(id: &str, evidence: &str, mean: f64) -> MomentInput {
    MomentInput { id: id.into(), evidence_id: evidence.into(), role: InputRole::Estimate, mean }
}

fn score(
    test: &'static str,
    dgp: &'static str,
    truth: f64,
    results: Vec<(Option<(f64, f64)>, CalibrationBasis)>,
) {
    let mut tally = CoverageTally::for_record(
        RecordKey { test, dgp, interval: NORMAL_APPROXIMATION_CONSTRUCTION },
        LEVEL,
    );
    for (interval, basis) in results {
        bind_basis(&mut tally, &basis);
        tally.record(interval, truth);
    }
    tally.assert();
}

/// Two estimates from shared rows, the covariance of the estimates estimated
/// from the same rows.
fn shared_rows_design(test: &'static str, dgp: &'static str, law: Law) {
    let n = grid_n(200);
    let results = map_replicates(n_sim(), |replicate| {
        let seed = grid_seed(0x91_0000 + replicate);
        let (x, y) = rows(law, n, seed, 1);
        let c = covariance(&x, &y).map(|v| v / n as f64);
        let claim = propagate_linear_moments_spread(
            PropagationOperation::Contrast,
            &[moment("x", "shared_rows", mean(&x)), moment("y", "shared_rows", mean(&y))],
            &[1.0, -1.0],
            &c,
            &LinearDependence::Covariance { id: "estimated_from_the_rows".into() },
            &ctx(seed),
        )
        .expect("claim");
        reading(&claim, n)
    });
    score(test, dgp, law.truth(), results);
}

/// Two independent samples with distinct evidence and a diagonal covariance.
fn independent_samples_design(test: &'static str, dgp: &'static str, law: Law) {
    let n = grid_n(200);
    let results = map_replicates(n_sim(), |replicate| {
        let seed = grid_seed(0x92_0000 + replicate);
        let (x, _) = rows(law, n, seed, 1);
        let (_, y) = rows(law, n, seed ^ 0x5eed, 2);
        let (vx, vy) = (covariance(&x, &x)[0] / n as f64, covariance(&y, &y)[0] / n as f64);
        let claim = propagate_linear_moments_spread(
            PropagationOperation::Contrast,
            &[moment("x", "sample_x", mean(&x)), moment("y", "sample_y", mean(&y))],
            &[1.0, -1.0],
            &[vx, 0.0, 0.0, vy],
            &LinearDependence::DeclaredIndependent,
            &ctx(seed),
        )
        .expect("claim");
        reading(&claim, n)
    });
    score(test, dgp, law.truth(), results);
}

/// Aligned bootstrap draws of the two estimates.
fn aligned_bootstrap_design(test: &'static str, dgp: &'static str, law: Law) {
    let n = grid_n(200);
    let results = map_replicates(n_sim(), |replicate| {
        let seed = grid_seed(0x93_0000 + replicate);
        let (x, y) = rows(law, n, seed, 1);
        let mut next = uniform(seed, 3);
        let (mut bx, mut by) = (Vec::with_capacity(BOOTSTRAP), Vec::with_capacity(BOOTSTRAP));
        for _ in 0..BOOTSTRAP {
            let (mut sx, mut sy) = (0.0, 0.0);
            for _ in 0..n {
                let i = ((next() * n as f64) as usize).min(n - 1);
                sx += x[i];
                sy += y[i];
            }
            bx.push(sx / n as f64);
            by.push(sy / n as f64);
        }
        let alignment = format!("bootstrap-{seed}");
        let px = PropagationDraws::aligned("x", "shared_rows", alignment.clone(), bx);
        let py = PropagationDraws::aligned("y", "shared_rows", alignment, by);
        let claim = propagate_aligned_linear_spread(
            PropagationOperation::Contrast,
            (1.0, -1.0),
            &px,
            &py,
            &ctx(seed),
        )
        .expect("claim");
        reading(&claim, n)
    });
    score(test, dgp, law.truth(), results);
}

#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn composed_contrast_shared_rows_gaussian_positive_correlation_l95() {
    shared_rows_design(
        "composed_contrast_shared_rows_gaussian_positive_correlation_l95",
        "bivariate_normal_rho_0.6_sample_means_covariance_estimated_from_rows",
        Law::Gaussian { rho: 0.6 },
    );
}

#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn composed_contrast_shared_rows_gaussian_negative_correlation_l95() {
    shared_rows_design(
        "composed_contrast_shared_rows_gaussian_negative_correlation_l95",
        "bivariate_normal_rho_-0.5_sample_means_covariance_estimated_from_rows",
        Law::Gaussian { rho: -0.5 },
    );
}

#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn composed_contrast_shared_rows_skewed_pair_l95() {
    shared_rows_design(
        "composed_contrast_shared_rows_skewed_pair_l95",
        "exponential_pair_shared_evidence_sample_means_covariance_estimated_from_rows",
        Law::Skewed,
    );
}

#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn composed_contrast_independent_samples_gaussian_l95() {
    independent_samples_design(
        "composed_contrast_independent_samples_gaussian_l95",
        "two_independent_normal_samples_sample_means_diagonal_covariance",
        Law::Gaussian { rho: 0.0 },
    );
}

#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn composed_contrast_aligned_bootstrap_gaussian_l95() {
    aligned_bootstrap_design(
        "composed_contrast_aligned_bootstrap_gaussian_l95",
        "bivariate_normal_rho_0.6_aligned_bootstrap_draws_of_the_sample_means",
        Law::Gaussian { rho: 0.6 },
    );
}

#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn composed_contrast_aligned_bootstrap_skewed_pair_l95() {
    aligned_bootstrap_design(
        "composed_contrast_aligned_bootstrap_skewed_pair_l95",
        "exponential_pair_shared_evidence_aligned_bootstrap_draws_of_the_sample_means",
        Law::Skewed,
    );
}

/// A violated declaration of independence is undetectable, and its interval
/// under-covers: the declaration is a premise of the licence, not something the
/// route checks. No record is emitted, because a boundary under the independent
/// key would govern every honest declaration.
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn a_violated_declared_independence_under_covers_and_emits_no_record() {
    let n = grid_n(200);
    let law = Law::Gaussian { rho: 0.9 };
    let mut tally = CoverageTally::new("a_violated_declared_independence_under_covers", LEVEL);
    let results = map_replicates(n_sim(), |replicate| {
        let seed = grid_seed(0x94_0000 + replicate);
        let (x, y) = rows(law, n, seed, 1);
        let (vx, vy) = (covariance(&x, &x)[0] / n as f64, covariance(&y, &y)[0] / n as f64);
        // The rows are dependent, but the two inputs claim distinct evidence.
        let claim = propagate_linear_moments_spread(
            PropagationOperation::Sum,
            &[moment("x", "sample_x", mean(&x)), moment("y", "sample_y", mean(&y))],
            &[1.0, 1.0],
            &[vx, 0.0, 0.0, vy],
            &LinearDependence::DeclaredIndependent,
            &ctx(seed),
        )
        .expect("claim");
        reading(&claim, n).0
    });
    for interval in results {
        tally.record(interval, 1.4);
    }
    assert!(
        tally.rate() < 0.9,
        "a violated independence declaration under-covers: {}",
        tally.rate()
    );
}
