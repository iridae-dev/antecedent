//! Known-truth recovery of the local-ratio design layer.
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;

use super::{LocalRatioColumns, LocalRatioDesign, density_test, fit_local_ratio_design};
use crate::splitmix::FixtureRng;

fn normal(rng: &mut FixtureRng) -> f64 {
    let u = rng.uniform().max(1e-300);
    let v = rng.uniform();
    (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
}

struct Sample {
    running: Vec<f64>,
    outcome: Vec<f64>,
    treatment: Vec<f64>,
    covariate: Vec<f64>,
}

impl Sample {
    fn columns(&self) -> LocalRatioColumns<'_> {
        LocalRatioColumns {
            running: &self.running,
            outcome: &self.outcome,
            treatment: &self.treatment,
        }
    }
}

/// Fuzzy discontinuity: uptake jumps from 0.2 to 0.8 at zero and the effect of uptake is 2.
fn fuzzy(seed: u64, n: usize) -> Sample {
    let mut rng = FixtureRng::new(seed);
    let mut sample = Sample {
        running: Vec::new(),
        outcome: Vec::new(),
        treatment: Vec::new(),
        covariate: Vec::new(),
    };
    for _ in 0..n {
        let r = 2.0 * rng.uniform() - 1.0;
        let uptake = if r >= 0.0 { 0.8 } else { 0.2 };
        let t = if rng.uniform() < uptake { 1.0 } else { 0.0 };
        let y = 1.0 + r + 0.5 * r * r - 0.8 * r.powi(3) + 2.0 * t + 0.5 * normal(&mut rng);
        sample.running.push(r);
        sample.treatment.push(t);
        sample.outcome.push(y);
        sample.covariate.push(r + 0.3 * normal(&mut rng));
    }
    sample
}

fn selected() -> LocalRatioDesign {
    LocalRatioDesign { select_bandwidth: true, temporal_replicates: None, density_test: true }
}

#[test]
fn fuzzy_discontinuity_with_a_selected_bandwidth_recovers_the_effect() {
    let sample = fuzzy(41, 6000);
    let fit = fit_local_ratio_design(
        &sample.columns(),
        &[&sample.covariate],
        0.0,
        0.8,
        false,
        selected(),
        &ExecutionContext::for_tests(1),
    )
    .unwrap();
    let chosen = fit.bandwidth.bandwidth;
    assert!(chosen > 0.0 && chosen <= 0.8, "{chosen}");
    assert!(fit.bandwidth.mse_optimal.is_some());
    let se = fit.fit.standard_error;
    assert!((fit.fit.estimate - 2.0).abs() < 4.0 * se, "{} ± {se}", fit.fit.estimate);
    let density = fit.density.unwrap();
    assert!(density.p_value > 1e-3, "{density:?}");
    assert_eq!(fit.balance.len(), 1);
    assert!(fit.balance[0].z.abs() < 4.0, "{:?}", fit.balance[0]);
}

#[test]
fn bunching_above_the_cutoff_is_flagged_by_the_density_test() {
    let mut sample = fuzzy(42, 6000);
    for r in &mut sample.running {
        if *r < 0.0 && *r > -0.1 {
            *r = -*r;
        }
    }
    let test = density_test(&sample.running, 0.0, 0.3).unwrap();
    assert!(test.log_difference > 0.0 && test.p_value < 0.01, "{test:?}");
}

#[test]
fn regression_kink_with_a_selected_bandwidth_recovers_the_slope_ratio() {
    let mut rng = FixtureRng::new(43);
    let mut sample = Sample {
        running: Vec::new(),
        outcome: Vec::new(),
        treatment: Vec::new(),
        covariate: Vec::new(),
    };
    for _ in 0..8000 {
        let r = 2.0 * rng.uniform() - 1.0;
        let t = 0.5 * r + r.max(0.0) + 0.05 * normal(&mut rng);
        let y = 2.0 * t + 0.3 * r * r + 0.2 * normal(&mut rng);
        sample.running.push(r);
        sample.treatment.push(t);
        sample.outcome.push(y);
    }
    let fit = fit_local_ratio_design(
        &sample.columns(),
        &[],
        0.0,
        0.9,
        true,
        LocalRatioDesign { select_bandwidth: true, temporal_replicates: None, density_test: false },
        &ExecutionContext::for_tests(1),
    )
    .unwrap();
    let se = fit.fit.standard_error;
    assert!((fit.fit.estimate - 2.0).abs() < 4.0 * se + 0.05, "{} ± {se}", fit.fit.estimate);
    assert!(fit.density.is_none());
}

/// A discontinuity in calendar time: a level shift of 1 at period 1000 with AR(1) errors.
fn in_time(seed: u64) -> Sample {
    let mut rng = FixtureRng::new(seed);
    let mut sample = Sample {
        running: Vec::new(),
        outcome: Vec::new(),
        treatment: Vec::new(),
        covariate: Vec::new(),
    };
    let mut error = 0.0;
    for period in 1..=2000_u32 {
        error = 0.7 * error + 0.5 * normal(&mut rng);
        let time = f64::from(period);
        let treated = if time >= 1000.0 { 1.0 } else { 0.0 };
        sample.running.push(time);
        sample.treatment.push(treated);
        sample.outcome.push(0.001 * time + treated + error);
    }
    sample
}

#[test]
fn a_discontinuity_in_time_gets_a_dependence_preserving_interval() {
    let sample = in_time(44);
    let design = LocalRatioDesign {
        select_bandwidth: false,
        temporal_replicates: Some(199),
        density_test: false,
    };
    let serial = fit_local_ratio_design(
        &sample.columns(),
        &[],
        1000.0,
        300.0,
        false,
        design,
        &ExecutionContext::for_tests(5),
    )
    .unwrap();
    let parallel = fit_local_ratio_design(
        &sample.columns(),
        &[],
        1000.0,
        300.0,
        false,
        design,
        &ExecutionContext::production(5, 4),
    )
    .unwrap();
    assert_eq!(serial, parallel);
    let temporal = serial.temporal.unwrap();
    let block_se = temporal.ratio.standard_error.unwrap();
    assert!(
        block_se > 1.3 * serial.fit.standard_error,
        "{block_se} vs {}",
        serial.fit.standard_error
    );
    assert!((serial.fit.estimate - 1.0).abs() < 4.0 * block_se, "{}", serial.fit.estimate);
    assert!(temporal.replicates_ok >= 190);
}

#[test]
fn the_block_bayesian_bootstrap_posterior_widens_for_serial_dependence() {
    let sample = in_time(45);
    let serial = super::posterior_temporal_local_ratio(
        &sample.columns(),
        1000.0,
        300.0,
        false,
        400,
        &ExecutionContext::for_tests(2),
    )
    .unwrap();
    let parallel = super::posterior_temporal_local_ratio(
        &sample.columns(),
        1000.0,
        300.0,
        false,
        400,
        &ExecutionContext::production(2, 4),
    )
    .unwrap();
    assert_eq!(serial, parallel);
    let independent = crate::local_polynomial_ratio::fit_local_polynomial_ratio(
        &sample.running,
        &sample.outcome,
        &sample.treatment,
        1000.0,
        300.0,
        false,
    )
    .unwrap();
    assert!(
        serial.summary.sd > 1.3 * independent.standard_error,
        "{} vs {}",
        serial.summary.sd,
        independent.standard_error
    );
    assert!((serial.summary.mean - 1.0).abs() < 4.0 * serial.summary.sd);
    let interval = serial.summary.interval;
    assert!(interval[0] < serial.summary.mean && serial.summary.mean < interval[1]);
    assert!(serial.draws_ok >= 380);
}

#[test]
fn unit_row_weights_reproduce_the_unweighted_fit_exactly() {
    let sample = fuzzy(46, 2000);
    let ones = vec![1.0; sample.running.len()];
    let weighted = crate::local_polynomial_ratio::fit_weighted_local_polynomial_ratio(
        &sample.running,
        &sample.outcome,
        &sample.treatment,
        &ones,
        0.0,
        0.8,
        false,
    )
    .unwrap();
    let plain = crate::local_polynomial_ratio::fit_local_polynomial_ratio(
        &sample.running,
        &sample.outcome,
        &sample.treatment,
        0.0,
        0.8,
        false,
    )
    .unwrap();
    assert_eq!(weighted, plain);
}

#[test]
fn temporal_blocks_keep_order_of_retained_rows_on_each_side() {
    // Running values are deliberately out of order in time and the sides interleave.
    let running = [-0.1, 0.8, -0.9, 0.1, -0.5, 2.0, 0.6, -0.3];
    let (left, right) = super::temporal_side_rows(&running, 0.0, 1.0);
    assert_eq!(left, [0, 2, 4, 7]);
    assert_eq!(right, [1, 3, 6]);
    // Both the moving-block sampler and posterior chunking use these sequences.
    assert_eq!(&left[1..3], &[2, 4]);
    assert_eq!(right.chunks(2).next().unwrap(), &[1, 3]);
}
