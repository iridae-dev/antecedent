//! Prior-predictive expectation checks of the categorical-choice inference and decision routes.
//!
//! Nothing here is measured at implementation time: every test is `#[ignore]`d and runs only at
//! the final measurement step. They state no coverage and issue no record; each checks an
//! expectation statement that the choice posterior reading makes about the whole procedure,
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
//! * `conflict_screen_flags_an_exchangeable_source_at_most_one_time_in_ten`: a source drawn from
//!   the predictive of the remaining evidence has a log Bayes factor (vague prior against the
//!   remaining evidence) whose exponential has expectation one, so by Markov's inequality the
//!   screen at `ln 10` flags it with probability at most one tenth. Axis: source size
//!   {20, 100, 400}, remaining evidence worth {5, 30} choices, three options.
//! * `procedure_law_matches_simulation`: the exact law of the sequential procedure
//!   (`choice_operating_characteristics`) agrees with the frequency of the same procedure
//!   simulated under the same truth, for each selection probability and the expected enrolment.
//!
//! The credible-interval coverage of the two measured cells, which does emit records, is
//! `antecedent-estimate/tests/choice_interval_calibration.rs`.
//!
//! No coverage is claimed for any interval, no frequentist error rate for the stopping rule, and
//! no behavioural reading of stated choices: the choice routes claim none of these. A prior chosen
//! after seeing outcomes, or a source that conflicts with the truth, is outside these designs.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "bounded simulation sizes and small indices"
)]

#[path = "common/calibration.rs"]
mod calibration;
#[path = "common/choice_dgp.rs"]
mod dgp;
use dgp::{choices, dirichlet};

use antecedent_core::ExecutionContext;
use antecedent_design::choice_evsi::{StudyCandidate, rank_studies};
use antecedent_estimate::choice_inference::{
    ChoiceConflictPolicy, ChoiceKernel, ChoiceProcedureSpec, ChoiceSourceCounts,
    ChoiceStoppingRule, choice_operating_characteristics, choice_posterior,
    choice_stopping_decision, qualify_choice_prior,
};
use calibration::{grid_seed, map_replicates, n_sim, smoke, uniform};

fn replicates() -> u32 {
    if smoke() { n_sim() } else { n_sim().max(20_000) }
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(0x5c)
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
    let results = map_replicates(replicates(), |replicate| {
        let seed = grid_seed(0x5c_0000 + cell * 1_000_003 + replicate);
        let theta = dirichlet(&prior, seed);
        let mut next = uniform(seed, 12);
        let counts = choices(&theta, observations, &mut next);
        let posterior = choice_posterior(&prior, &counts, &ctx()).expect("posterior");
        (posterior.probability_best, argmax(&theta))
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
    let rule = ChoiceStoppingRule {
        min_probability_best: Some(threshold),
        max_expected_loss: None,
        min_observations: step,
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
            let posterior = choice_posterior(&prior, &counts, &ctx()).expect("posterior");
            let decision = choice_stopping_decision(&posterior, rule).expect("rule");
            if decision.stop {
                return Some(decision.best_option == argmax(&theta));
            }
            if counts.iter().sum::<f64>() >= f64::from(ceiling) {
                return None;
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
    // The decision-maker starts from a fixed, modest sample.
    let current_counts: Vec<f64> = (0..options).map(|i| 6.0 + 3.0 * i as f64).collect();
    let current = choice_posterior(&prior, &current_counts, &ctx()).expect("current");
    let alpha = current.alpha.clone();
    let total: f64 = alpha.iter().sum();
    let now = argmax(&values.iter().zip(&alpha).map(|(v, a)| v * a / total).collect::<Vec<_>>());
    let candidate = StudyCandidate { label: "study".into(), observations: study, cost: 0.0 };
    let predicted = rank_studies(&alpha, &values, &[candidate], 17).expect("rank")[0].evsi;
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

#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn procedure_law_matches_simulation() {
    let theta = [0.5, 0.3, 0.2];
    let spec = ChoiceProcedureSpec {
        prior: vec![1.0; 3],
        batches: vec![10; 6],
        rule: ChoiceStoppingRule {
            min_probability_best: Some(0.9),
            max_expected_loss: None,
            min_observations: 10,
        },
    };
    let law = choice_operating_characteristics(
        &spec,
        &ChoiceKernel::Fixed { theta: theta.to_vec() },
        &ctx(),
    )
    .expect("law");
    let runs = map_replicates(replicates(), |replicate| {
        let seed = grid_seed(0x61_0000 + replicate);
        let mut next = uniform(seed, 18);
        let mut counts = vec![0.0; 3];
        for batch in &spec.batches {
            for (c, add) in counts.iter_mut().zip(choices(&theta, *batch as usize, &mut next)) {
                *c += add;
            }
            let posterior = choice_posterior(&spec.prior, &counts, &ctx()).expect("posterior");
            let decision = choice_stopping_decision(&posterior, spec.rule).expect("rule");
            if decision.stop {
                return (Some(decision.best_option), counts.iter().sum::<f64>());
            }
        }
        (None, counts.iter().sum::<f64>())
    });
    let m = runs.len() as f64;
    for (k, expected) in law.selection.iter().enumerate() {
        let hits = runs.iter().filter(|(pick, _)| *pick == Some(k)).count() as f64;
        let rate = hits / m;
        let tolerance = 4.5 * (expected * (1.0 - expected) / m).sqrt() + 1e-9;
        assert!(
            (rate - expected).abs() <= tolerance,
            "option {k}: {rate:.4} against {expected:.4}"
        );
    }
    let enrolled: Vec<f64> = runs.iter().map(|(_, n)| *n).collect();
    let mean = enrolled.iter().sum::<f64>() / m;
    assert!(
        (mean - law.expected_enrolled).abs() <= 4.5 * standard_error(&enrolled).max(1e-9),
        "expected enrolment {mean:.3} against {:.3}",
        law.expected_enrolled
    );
}

fn conflict_screen_design(source_size: usize, rest_worth: f64, cell: u64) {
    let initial = [1.0, 1.0, 1.0];
    // The remaining evidence: one fixed source, discounted to `rest_worth` choices.
    let other = [rest_worth * 0.5, rest_worth * 0.3, rest_worth * 0.2];
    let rest: Vec<f64> = initial.iter().zip(other).map(|(a, c)| a + c).collect();
    let flagged = map_replicates(replicates(), |replicate| {
        let seed = grid_seed(0x71_0000 + cell * 1_000_003 + replicate);
        let theta = dirichlet(&rest, seed);
        let mut next = uniform(seed, 21);
        let drawn = choices(&theta, source_size, &mut next);
        let prior = qualify_choice_prior(
            &initial,
            &[
                ChoiceSourceCounts {
                    label: "drawn",
                    counts: &drawn,
                    discount: 1.0,
                    transferred: false,
                },
                ChoiceSourceCounts {
                    label: "rest",
                    counts: &other,
                    discount: 1.0,
                    transferred: false,
                },
            ],
            ChoiceConflictPolicy::Report,
        )
        .expect("prior");
        f64::from(u8::from(prior.sources[0].conflicting))
    });
    let m = flagged.len() as f64;
    let rate = flagged.iter().sum::<f64>() / m;
    let tolerance = 4.5 * (0.1 * 0.9 / m).sqrt();
    assert!(
        rate <= 0.1 + tolerance,
        "source {source_size}, remaining evidence {rest_worth}: flagged {rate:.4} of exchangeable sources",
    );
}

#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn conflict_screen_flags_an_exchangeable_source_at_most_one_time_in_ten() {
    let mut cell = 0;
    for source_size in [20, 100, 400] {
        for rest_worth in [5.0, 30.0] {
            conflict_screen_design(source_size, rest_worth, cell);
            cell += 1;
        }
    }
}

/// A registered prior-importance check of the hierarchical two-choice study value.
#[test]
#[ignore = "calibration: run only at the final measurement step"]
fn hierarchical_study_value_matches_importance_sampling() {
    use antecedent_design::choice_evsi::rank_hierarchical_studies;
    use antecedent_stats::adaptive_inference::{
        HierarchicalChoicePosterior, HierarchicalChoicePrior,
    };

    let prior = HierarchicalChoicePrior {
        shapes: vec![1.0, 1.0],
        log_concentration_mean: 5.0_f64.ln(),
        log_concentration_sd: 1.0,
    };
    let current =
        vec![vec![3, 1], vec![1, 3], vec![2, 2], vec![4, 0], vec![0, 4], vec![3, 1], vec![2, 2]];
    let cancelled = || false;
    let base =
        HierarchicalChoicePosterior::new(&prior, &current).unwrap().summary(&cancelled).unwrap();
    let now = argmax(&base.mean);
    let candidate = StudyCandidate { label: "two choices".into(), observations: 2, cost: 0.0 };
    let predicted = rank_hierarchical_studies(&prior, &current, &[1.0, 1.0], &[candidate], &ctx())
        .unwrap()[0]
        .evsi;
    let mut choices_after = [0usize; 3];
    for (a, slot) in choices_after.iter_mut().enumerate() {
        let mut updated = current.clone();
        updated.push(vec![a as u32, (2 - a) as u32]);
        let summary = HierarchicalChoicePosterior::new(&prior, &updated)
            .unwrap()
            .summary(&cancelled)
            .unwrap();
        *slot = argmax(&summary.mean);
    }
    let sequence_probability = |p: f64, k: f64, counts: &[u32]| {
        let mut value = 1.0;
        let mut total = 0u32;
        for (probability, &count) in [p, 1.0 - p].iter().zip(counts) {
            for j in 0..count {
                value *= (k * probability + f64::from(j)) / (k + f64::from(total));
                total += 1;
            }
        }
        value
    };
    let samples = map_replicates(replicates(), |rep| {
        let seed = grid_seed(0x5f_0000 + rep);
        let mut uniform = uniform(seed, 71);
        let mut normal = calibration::gaussian(seed ^ 0x5f_7191);
        let p = uniform();
        let k = (prior.log_concentration_mean + normal()).exp();
        let weight = current.iter().map(|row| sequence_probability(p, k, row)).product::<f64>();
        let share = [p, 1.0 - p];
        let mut gain = 0.0;
        for a in 0..=2usize {
            let counts = [a as u32, (2 - a) as u32];
            let multiplicity = if a == 1 { 2.0 } else { 1.0 };
            gain += multiplicity
                * sequence_probability(p, k, &counts)
                * (share[choices_after[a]] - share[now]);
        }
        (weight, weight * gain)
    });
    let sum_weight = samples.iter().map(|x| x.0).sum::<f64>();
    let estimate = samples.iter().map(|x| x.1).sum::<f64>() / sum_weight;
    let mean_weight = sum_weight / samples.len() as f64;
    let residuals: Vec<f64> =
        samples.iter().map(|(w, g)| (g - estimate * w) / mean_weight).collect();
    let se = standard_error(&residuals);
    assert!(
        (estimate - predicted).abs() <= 4.5 * se.max(1e-9) + 1e-6,
        "importance estimate {estimate} vs quadrature {predicted}"
    );
}
