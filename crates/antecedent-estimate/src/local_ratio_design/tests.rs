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

/// Reduced form and first stage as the estimator weights give them, for the rows of `sample`.
fn weighted_changes(sample: &Sample, bandwidth: f64, kink: bool) -> (f64, f64) {
    let (left, right) = super::temporal_side_rows(&sample.running, 0.0, bandwidth);
    let (mut reduced_form, mut first_stage) = (0.0, 0.0);
    for (rows, sign) in [(&left, -1.0), (&right, 1.0)] {
        let distances: Vec<f64> = rows.iter().map(|&row| sample.running[row] / bandwidth).collect();
        let weights = super::estimator_weights(&distances, bandwidth, kink).unwrap();
        for (&row, weight) in rows.iter().zip(&weights) {
            reduced_form += sign * weight * sample.outcome[row];
            first_stage += sign * weight * sample.treatment[row];
        }
    }
    (reduced_form, first_stage)
}

#[test]
fn the_estimator_weights_reproduce_the_fitted_jump_and_slope_change() {
    let jump = fuzzy(47, 2000);
    let mut rng = FixtureRng::new(48);
    let mut kink = Sample {
        running: Vec::new(),
        outcome: Vec::new(),
        treatment: Vec::new(),
        covariate: Vec::new(),
    };
    for _ in 0..4000 {
        let r = 2.0 * rng.uniform() - 1.0;
        let t = 0.5 * r + r.max(0.0) + 0.02 * normal(&mut rng);
        kink.running.push(r);
        kink.treatment.push(t);
        kink.outcome.push(2.0 * t + 0.3 * r * r + 0.2 * normal(&mut rng));
    }
    for (sample, is_kink) in [(&jump, false), (&kink, true)] {
        let fit = crate::local_polynomial_ratio::fit_local_polynomial_ratio(
            &sample.running,
            &sample.outcome,
            &sample.treatment,
            0.0,
            0.8,
            is_kink,
        )
        .unwrap();
        let (reduced_form, first_stage) = weighted_changes(sample, 0.8, is_kink);
        assert!(
            (reduced_form - fit.reduced_form).abs() < 1e-6 * (1.0 + fit.reduced_form.abs()),
            "{reduced_form} vs {}",
            fit.reduced_form
        );
        assert!(
            (first_stage - fit.first_stage).abs() < 1e-6 * (1.0 + fit.first_stage.abs()),
            "{first_stage} vs {}",
            fit.first_stage
        );
    }
}

#[test]
fn the_trend_correction_recovers_the_autocorrelation_of_a_persistent_error() {
    // White noise leaves the residual maker itself: its trace is the residual degrees of
    // freedom.
    let distances: Vec<f64> = (0..200).map(|i| f64::from(i) / 200.0).collect();
    let trend = |distances: &[f64]| super::SideTrend {
        inverse: super::inverse_gram(distances, super::TREND_ORDER, |_| 1.0).unwrap(),
        basis: distances.iter().map(|&d| super::monomials(d, super::TREND_ORDER)).collect(),
    };
    let (_, square) = trend(&distances).residual_moments(0.0);
    assert!((square - 196.0).abs() < 1e-6, "{square}");
    // AR(1) errors at 0.7 about a cubic on each side: the raw residual autocorrelation is
    // biased down, the corrected coefficient is not.
    let mut rng = FixtureRng::new(49);
    let (mut raw, mut corrected) = (0.0, 0.0);
    let replicates = 200;
    for _ in 0..replicates {
        let mut error = normal(&mut rng);
        let mut series = [Vec::new(), Vec::new()];
        for side in &mut series {
            for &d in &distances {
                error = 0.7 * error + 0.51_f64.sqrt() * normal(&mut rng);
                side.push(1.0 + d - 2.0 * d.powi(3) + error);
            }
        }
        let trends = [trend(&distances), trend(&distances)];
        let residuals = [trends[0].residuals(&series[0]), trends[1].residuals(&series[1])];
        let cross: f64 =
            residuals.iter().map(|r| r.windows(2).map(|p| p[0] * p[1]).sum::<f64>()).sum();
        let square: f64 = residuals.iter().flatten().map(|r| r * r).sum();
        raw += cross / square / f64::from(replicates);
        let fitted =
            super::error_series(&trends, [series[0].as_slice(), series[1].as_slice()]).unwrap();
        corrected += fitted.model.phi[0] / f64::from(replicates);
    }
    assert!(raw < corrected - 0.015, "{raw} vs {corrected}");
    assert!((corrected - 0.7).abs() < 0.02, "{corrected}");
}

/// A level shift of 1 at period 1000 with AR(2) errors (`0.5`, `0.3`, marginal standard
/// deviation `0.7`): dependence the AR(1) error model of the interval does not carry.
fn in_time_ar2(seed: u64) -> Sample {
    let mut rng = FixtureRng::new(seed);
    let mut sample = Sample {
        running: Vec::new(),
        outcome: Vec::new(),
        treatment: Vec::new(),
        covariate: Vec::new(),
    };
    let (phi1, phi2): (f64, f64) = (0.5, 0.3);
    let marginal = 0.7;
    let innovation =
        marginal * ((1.0 + phi2) * ((1.0 - phi2).powi(2) - phi1 * phi1) / (1.0 - phi2)).sqrt();
    let (mut previous, mut before) = (0.0, 0.0);
    for period in 1..=2500_u32 {
        let error = phi1 * previous + phi2 * before + innovation * normal(&mut rng);
        before = previous;
        previous = error;
        if period <= 500 {
            continue;
        }
        let time = f64::from(period - 500);
        let treated = if time >= 1000.0 { 1.0 } else { 0.0 };
        sample.running.push(time);
        sample.treatment.push(treated);
        sample.outcome.push(0.001 * time + treated + error);
    }
    sample
}

/// Per-side trend of `rows` equally spaced distances in `[0, 1)`.
fn cubic_trend(rows: usize) -> super::SideTrend {
    let distances: Vec<f64> = (0..rows).map(|i| i as f64 / rows as f64).collect();
    super::SideTrend {
        inverse: super::inverse_gram(&distances, super::TREND_ORDER, |_| 1.0).unwrap(),
        basis: distances.iter().map(|&d| super::monomials(d, super::TREND_ORDER)).collect(),
    }
}

/// Stationary AR(2) series of `rows` points (`0.5`, `0.3`, marginal standard deviation `0.7`)
/// or AR(1) (`0.7`, `0.7`) about a cubic.
fn trend_plus_ar(rng: &mut FixtureRng, rows: usize, second_order: bool) -> Vec<f64> {
    let (phi1, phi2, marginal): (f64, f64, f64) =
        if second_order { (0.5, 0.3, 0.7) } else { (0.7, 0.0, 0.7) };
    let innovation =
        marginal * ((1.0 + phi2) * ((1.0 - phi2).powi(2) - phi1 * phi1) / (1.0 - phi2)).sqrt();
    let (mut previous, mut before) = (0.0, 0.0);
    let mut series = Vec::with_capacity(rows);
    for step in 0..rows + 300 {
        let error = phi1 * previous + phi2 * before + innovation * normal(rng);
        before = previous;
        previous = error;
        if step >= 300 {
            let d = (step - 300) as f64 / rows as f64;
            series.push(1.0 + d - 2.0 * d.powi(3) + error);
        }
    }
    series
}

#[test]
fn the_error_model_recovers_the_persistence_of_a_second_order_error_through_the_trend() {
    let trends = [cubic_trend(300), cubic_trend(300)];
    let mut rng = FixtureRng::new(52);
    for (second_order, persistence) in [(true, 0.8), (false, 0.7)] {
        let replicates = 40;
        let (mut total, mut higher) = (0.0, 0);
        for _ in 0..replicates {
            let sides = [
                trend_plus_ar(&mut rng, 300, second_order),
                trend_plus_ar(&mut rng, 300, second_order),
            ];
            let fitted =
                super::error_series(&trends, [sides[0].as_slice(), sides[1].as_slice()]).unwrap();
            total += fitted.model.phi.iter().sum::<f64>() / f64::from(replicates);
            higher += usize::from(fitted.model.phi.len() >= 2);
        }
        assert!((total - persistence).abs() < 0.04, "{second_order}: {total}");
        if second_order {
            assert!(higher >= 36, "{higher}");
        } else {
            assert!(higher <= 4, "{higher}");
        }
    }
}

#[test]
fn an_ar2_error_widens_the_interval_beyond_the_first_order_model() {
    let design = LocalRatioDesign {
        select_bandwidth: false,
        temporal_replicates: Some(199),
        density_test: false,
    };
    let sample = in_time_ar2(46);
    let fit = fit_local_ratio_design(
        &sample.columns(),
        &[],
        1000.0,
        300.0,
        false,
        design,
        &ExecutionContext::for_tests(7),
    )
    .unwrap();
    let record = fit.temporal.unwrap();
    assert!(record.ratio.interval.is_some());
    // The long-run sd of this error is about 3.3 times its row-wise sd, the independent-row
    // delta-method standard error ignores it, and the first-order model would reach only 2.4.
    let block_se = record.ratio.standard_error.unwrap();
    assert!(block_se > 1.3 * fit.fit.standard_error, "{block_se} vs {}", fit.fit.standard_error);
}
