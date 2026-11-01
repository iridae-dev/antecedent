//! Prior-predictive expectation checks of the categorical-choice evidence-to-decision lifecycle.
//!
//! Nothing here is measured at implementation time: every test is `#[ignore]`d and runs only at
//! the final measurement step. They state no coverage and issue no record; each checks an
//! expectation statement that the lifecycle's posterior reading makes about the whole procedure,
//! with the truth drawn from the prior (or, for study value, from the current posterior) by an
//! independent generator, so the nominal reading is exact for a correct posterior:
//!
//! * `posterior_probability_best_is_calibrated`: the posterior probability that each option
//!   has the largest share equals the frequency with which it does, in aggregate and within
//!   deciles of the stated probability. Axis: options {2, 3, 5}, observations {20, 100, 400},
//!   flat and informative priors. Boundary probe: 20 observations over 5 options.
//! * `sequential_selection_is_correct_at_least_at_the_threshold`: among replicates a Bayesian
//!   stopping rule selects, the selected option is the truly best option with frequency at
//!   least the rule's probability threshold, whatever the look at which it stopped (optional
//!   stopping leaves the posterior unchanged). Axis: thresholds {0.90, 0.95, 0.99}, three
//!   options, looks every 20 choices to 400. Boundary probe: 0.99, where most replicates
//!   exhaust the budget and the statement is about few selections.
//! * `study_value_matches_realized_gain`: the expected value of sample information the ranking
//!   reports equals the mean realized gain in true option value from deciding after the study
//!   rather than before. Axis: study size {10, 50, 200}, option values (1.0, 1.1, 0.9).
//!   Boundary probe: the Monte Carlo branch (study size 200 over 5 options).
//!
//! No coverage is claimed for any interval, no frequentist error rate for the stopping rule, and
//! no behavioural reading of stated choices: the lifecycle claims none of these. A prior chosen
//! after seeing outcomes, or a source that conflicts with the truth, is outside these designs.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "bounded simulation sizes and small indices"
)]

#[path = "../../antecedent/tests/common/calibration.rs"]
mod calibration;

use antecedent_design::choice_evidence::{
    ChoiceFrame, ChoiceMeaning, ChoiceObservations, ConflictPolicy, LookOutcome, SequentialRule,
    StudyCandidate, qualify_prior, rank_studies, update,
};
use calibration::{grid_seed, map_replicates, n_sim, smoke, uniform};

fn replicates() -> u32 {
    if smoke() { n_sim() } else { n_sim().max(20_000) }
}

fn never() -> bool {
    false
}

fn frame(options: usize) -> ChoiceFrame {
    ChoiceFrame {
        options: (0..options).map(|i| format!("option_{i}")).collect(),
        population: "calibration".into(),
        meaning: ChoiceMeaning::StatedPreference,
    }
}

/// Independent Marsaglia-Tsang gamma draw (shape at least one); the product sampler is not
/// reused.
fn gamma(shape: f64, normal: &mut impl FnMut() -> f64, unit: &mut impl FnMut() -> f64) -> f64 {
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = normal();
        let v = (1.0 + c * x).powi(3);
        if v > 0.0 && unit().ln() < 0.5 * x * x + d - d * v + d * v.ln() {
            return d * v;
        }
    }
}

/// A Dirichlet draw from independent gammas.
fn dirichlet(alpha: &[f64], seed: u64) -> Vec<f64> {
    let mut normal = calibration::gaussian(seed);
    let mut unit = uniform(seed, 11);
    let g: Vec<f64> = alpha.iter().map(|a| gamma(*a, &mut normal, &mut unit)).collect();
    let total: f64 = g.iter().sum();
    g.iter().map(|x| x / total).collect()
}

/// `n` categorical choices from `theta`, as counts.
fn choices(theta: &[f64], n: usize, next: &mut impl FnMut() -> f64) -> Vec<f64> {
    let mut counts = vec![0.0; theta.len()];
    for _ in 0..n {
        let mut u = next();
        let mut pick = theta.len() - 1;
        for (j, t) in theta.iter().enumerate() {
            if u < *t {
                pick = j;
                break;
            }
            u -= t;
        }
        counts[pick] += 1.0;
    }
    counts
}

fn argmax(values: &[f64]) -> usize {
    values.iter().enumerate().fold(0, |b, (i, v)| if *v > values[b] { i } else { b })
}

fn standard_error(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0) / n).sqrt()
}

fn probability_best_design(options: usize, observations: usize, concentration: f64, cell: u64) {
    let prior = vec![concentration; options];
    let t = frame(options);
    let empty = qualify_prior(&t, &prior, &[], ConflictPolicy::Report).expect("prior");
    let results = map_replicates(replicates(), |replicate| {
        let seed = grid_seed(0x5c_0000 + cell * 1_000_003 + replicate);
        let theta = dirichlet(&prior, seed);
        let mut next = uniform(seed, 12);
        let counts = choices(&theta, observations, &mut next);
        let inference = update(
            &empty,
            &ChoiceObservations { counts, observation_ids: vec![format!("replicate.{replicate}")] },
            &never,
        )
        .expect("inference");
        (inference.probability_best, argmax(&theta))
    });
    let n = results.len() as f64;
    // Aggregate: for each option, mean stated probability against the frequency of being best.
    for k in 0..options {
        let gap: Vec<f64> =
            results.iter().map(|(p, best)| p[k] - f64::from(u8::from(*best == k))).collect();
        let mean = gap.iter().sum::<f64>() / n;
        assert!(
            mean.abs() <= 4.5 * standard_error(&gap).max(1e-9),
            "options {options}, n {observations}, concentration {concentration}, option {k}: \
             probability of being best off by {mean:.5}",
        );
    }
    // Reliability: deciles of the stated probability, all options pooled.
    let mut bins = [(0.0_f64, 0.0_f64, 0.0_f64); 10];
    for (p, best) in &results {
        for (k, pk) in p.iter().enumerate() {
            let b = ((pk * 10.0) as usize).min(9);
            bins[b].0 += 1.0;
            bins[b].1 += pk;
            bins[b].2 += f64::from(u8::from(*best == k));
        }
    }
    for (b, (count, sum_p, hits)) in bins.iter().enumerate() {
        if *count < 400.0 {
            continue;
        }
        let (p, f) = (sum_p / count, hits / count);
        let tolerance = 4.5 * (p * (1.0 - p) / count).sqrt() + 1e-3;
        assert!(
            (p - f).abs() <= tolerance,
            "options {options}, n {observations}: decile {b} stated {p:.4}, observed {f:.4}",
        );
    }
}

#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn posterior_probability_best_is_calibrated() {
    let mut cell = 0;
    for options in [2, 3, 5] {
        for observations in [20, 100, 400] {
            for concentration in [1.0, 4.0] {
                probability_best_design(options, observations, concentration, cell);
                cell += 1;
            }
        }
    }
}

fn sequential_design(threshold: f64, cell: u64) {
    let (options, step, ceiling) = (3, 20_u32, 400_u32);
    let prior = vec![1.0; options];
    let t = frame(options);
    let empty = qualify_prior(&t, &prior, &[], ConflictPolicy::Report).expect("prior");
    let rule = SequentialRule {
        min_probability_best: threshold,
        max_expected_loss: 1.0,
        min_observations: f64::from(step),
        max_observations: f64::from(ceiling),
        allocation_floor: 0.0,
    };
    let results = map_replicates(replicates(), |replicate| {
        let seed = grid_seed(0x5d_0000 + cell * 1_000_003 + replicate);
        let theta = dirichlet(&prior, seed);
        let mut next = uniform(seed, 13);
        let mut counts = vec![0.0; options];
        loop {
            for (c, add) in counts.iter_mut().zip(choices(&theta, step as usize, &mut next)) {
                *c += add;
            }
            let inference = update(
                &empty,
                &ChoiceObservations {
                    counts: counts.clone(),
                    observation_ids: vec!["running".into()],
                },
                &never,
            )
            .expect("inference");
            match rule.evaluate(&inference).expect("rule") {
                LookOutcome::Select { option } => return Some(option == argmax(&theta)),
                LookOutcome::BudgetExhausted => return None,
                LookOutcome::Continue { .. } => {}
            }
        }
    });
    let selected: Vec<f64> =
        results.iter().flatten().map(|correct| f64::from(u8::from(*correct))).collect();
    assert!(!selected.is_empty(), "threshold {threshold}: nothing was ever selected");
    let m = selected.len() as f64;
    let rate = selected.iter().sum::<f64>() / m;
    let tolerance = 4.5 * (threshold * (1.0 - threshold) / m).sqrt();
    assert!(
        rate >= threshold - tolerance,
        "threshold {threshold}: selected option best in {rate:.4} of {m} selections",
    );
}

#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn sequential_selection_is_correct_at_least_at_the_threshold() {
    for (cell, threshold) in [0.90, 0.95, 0.99].into_iter().enumerate() {
        sequential_design(threshold, cell as u64);
    }
}

fn study_value_design(options: usize, study: u32, cell: u64) {
    let values: Vec<f64> = (0..options).map(|i| [1.0, 1.1, 0.9, 1.05, 0.95][i]).collect();
    let prior = vec![1.0; options];
    let t = frame(options);
    let empty = qualify_prior(&t, &prior, &[], ConflictPolicy::Report).expect("prior");
    // The decision-maker starts from a fixed, modest sample.
    let current = update(
        &empty,
        &ChoiceObservations {
            counts: (0..options).map(|i| 6.0 + 3.0 * i as f64).collect(),
            observation_ids: vec!["current".into()],
        },
        &never,
    )
    .expect("current");
    let alpha = current.posterior.alpha().to_vec();
    let total: f64 = alpha.iter().sum();
    let now = argmax(&values.iter().zip(&alpha).map(|(v, a)| v * a / total).collect::<Vec<_>>());
    let candidate = StudyCandidate { label: "study".into(), observations: study, cost: 0.0 };
    let predicted = rank_studies(&current, &values, &[candidate], 17).expect("rank")[0].evsi;
    let gains = map_replicates(replicates(), |replicate| {
        let seed = grid_seed(0x5e_0000 + cell * 1_000_003 + replicate);
        // The truth comes from the current posterior; the study then observes from it.
        let theta = dirichlet(&alpha, seed);
        let mut next = uniform(seed, 14);
        let counts = choices(&theta, study as usize, &mut next);
        let after: Vec<f64> =
            values.iter().zip(alpha.iter().zip(&counts)).map(|(v, (a, c))| v * (a + c)).collect();
        let chosen = argmax(&after);
        values[chosen] * theta[chosen] - values[now] * theta[now]
    });
    let mean = gains.iter().sum::<f64>() / gains.len() as f64;
    let se = standard_error(&gains);
    // The ranking's own Monte Carlo branch carries its standard error; the exact branch has none.
    assert!(
        (mean - predicted).abs() <= 4.5 * se.max(1e-9) + 1e-9,
        "options {options}, study {study}: realized gain {mean:.6} against predicted {predicted:.6}",
    );
}

#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn study_value_matches_realized_gain() {
    for (cell, (options, study)) in [(3, 10), (3, 50), (3, 200), (5, 200)].into_iter().enumerate() {
        study_value_design(options, study, cell as u64);
    }
}
