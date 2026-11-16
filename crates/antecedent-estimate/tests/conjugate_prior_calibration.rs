//! Coverage of the conjugate count posteriors and of the select-then-estimate power-prior
//! procedure.
//!
//! Nothing here is measured at implementation time: every test is `#[ignore]`d and runs only
//! at the measurement step through `scripts/measure_calibration.sh`; none emits a coverage
//! record (the procedures are not reachable through the facade) and each gates pass or fail.
//!
//! Designs (all at the reported 0.95 level, on the sample-size grid around the stated base):
//!
//! * `beta_binomial_posterior_covers_under_the_stated_prior`: truths `θ ~ Beta(2, 3)`, `k ~
//!   Binomial(n, θ)`, interval the equal-tailed 95% credible interval under that same prior.
//!   Averaged over the prior the posterior interval has exact coverage `0.95`, so the
//!   two-sided band applies. `gamma_poisson_posterior_covers_under_the_stated_prior` is the
//!   same for `λ ~ Gamma(3, 1.5)` and `n` Poisson counts.
//! * `*_misspecified_prior_matches_its_exact_coverage`: the same truths analysed under a prior
//!   that disagrees with them (`Beta(8, 2)`, `Gamma(12, 2)`). Coverage is then below nominal; its
//!   exact value is a finite sum of the closed-form marginal and posterior laws, and the test
//!   asserts that repeated sampling reproduces that value and that it is below nominal. It
//!   measures how far a wrong prior pulls the interval, not a claim that the interval covers.
//! * `selected_alpha_*`: the whole select-then-estimate procedure of
//!   `antecedent_prob::select_linear_power_prior` (five-fold cross-validated discount on an
//!   11-point grid, 100 bootstrap resamples of the selection, known residual variance 1). The
//!   target is a balanced two-arm regression with effect `1`; the source is an external Gaussian
//!   prior four times as informative as the target whose mean is displaced from the truth by a
//!   multiple `d` of the target's effect standard error. Cells:
//!   `d = 0` (a matching source) must cover at least nominally; `d = 8` (a clear conflict) must
//!   cover nominally because the selection discounts the source; at the moderate
//!   discrepancies `d = 1` and `d = 2` the procedure is scored against trusting the source
//!   outright, which it must not lose to, and the coverage of the selected, the full-trust
//!   (`α = 1`) and the target-only (`α = 0`) intervals is printed. No data-driven discount covers
//!   uniformly over every discrepancy; the moderate cells name that boundary and assert only
//!   the comparison.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "bounded simulation sizes and small indices"
)]

#[path = "../../antecedent/tests/common/calibration.rs"]
mod calibration;

use std::sync::Arc;

use antecedent_core::CausalRng;
use antecedent_kernels::{sample_gamma, standard_normal};
use antecedent_prob::{
    AlphaCriterion, AlphaSelectionOptions, BetaBinomialPosterior, BetaHyperparameters, BetaSource,
    ExternalPriorSource, ExternalPriorWeight, GammaHyperparameters, GammaPoissonPosterior,
    GaussianCoefficientPrior, LinearTarget, PriorSet, PriorSpec, select_beta_power_prior,
    select_linear_power_prior,
};
use calibration::{
    CoverageTally, REPORTED_LEVEL, coverage_mcse, grid_n, grid_seed, map_replicates, n_sim,
};

const LEVEL: f64 = REPORTED_LEVEL;
const SEED_BETA: u64 = 4_117_001;
const SEED_GAMMA: u64 = 4_117_101;
const SEED_BETA_MISSPECIFIED: u64 = 4_117_201;
const SEED_GAMMA_MISSPECIFIED: u64 = 4_117_301;
const SEED_SELECTED: u64 = 4_117_401;
const SEED_SELECTED_BETA: u64 = 4_117_501;

/// Prior the truths of the Beta designs are drawn from, and the misspecified analysis prior.
const BETA_TRUE: BetaHyperparameters = BetaHyperparameters { alpha: 2.0, beta: 3.0 };
const BETA_WRONG: BetaHyperparameters = BetaHyperparameters { alpha: 8.0, beta: 2.0 };
const GAMMA_TRUE: GammaHyperparameters = GammaHyperparameters { shape: 3.0, rate: 1.5 };
const GAMMA_WRONG: GammaHyperparameters = GammaHyperparameters { shape: 12.0, rate: 2.0 };
/// Upper bound of the total count summed in the exact Gamma-Poisson coverage.
const MAX_TOTAL_COUNT: u64 = 2_000;

fn rng(seed: u64) -> CausalRng {
    CausalRng::from_seed(grid_seed(seed))
}

fn draw_beta(prior: BetaHyperparameters, rng: &mut CausalRng) -> f64 {
    let x = sample_gamma(prior.alpha, 1.0, rng);
    let y = sample_gamma(prior.beta, 1.0, rng);
    x / (x + y)
}

fn draw_binomial(n: usize, theta: f64, rng: &mut CausalRng) -> u64 {
    (0..n).filter(|_| rng.next_f64() < theta).count() as u64
}

/// Knuth's product-of-uniforms Poisson draw (rates here are small).
fn draw_poisson(rate: f64, rng: &mut CausalRng) -> u64 {
    let limit = (-rate).exp();
    let mut count = 0;
    let mut product = rng.next_f64();
    while product > limit {
        count += 1;
        product *= rng.next_f64();
    }
    count
}

// -- Conjugate posteriors under the stated prior ------------------------------------------------

fn beta_replicate(n: usize, analysis: BetaHyperparameters, seed: u64) -> ((f64, f64), f64) {
    let mut rng = rng(seed);
    let theta = draw_beta(BETA_TRUE, &mut rng);
    let k = draw_binomial(n, theta, &mut rng);
    let post = BetaBinomialPosterior::new(analysis, k, n as u64).unwrap();
    (post.credible_interval(LEVEL).unwrap(), theta)
}

fn gamma_replicate(n: usize, analysis: GammaHyperparameters, seed: u64) -> ((f64, f64), f64) {
    let mut rng = rng(seed);
    let lambda = sample_gamma(GAMMA_TRUE.shape, GAMMA_TRUE.rate, &mut rng);
    let counts: Vec<u64> = (0..n).map(|_| draw_poisson(lambda, &mut rng)).collect();
    let post = GammaPoissonPosterior::from_counts(analysis, &counts, None).unwrap();
    (post.credible_interval(LEVEL).unwrap(), lambda)
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn beta_binomial_posterior_covers_under_the_stated_prior() {
    let n = grid_n(40);
    let mut tally = CoverageTally::new("beta_binomial_stated_prior", LEVEL);
    for (interval, truth) in
        map_replicates(n_sim(), |rep| beta_replicate(n, BETA_TRUE, SEED_BETA + rep))
    {
        tally.record(Some(interval), truth);
    }
    tally.assert();
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn gamma_poisson_posterior_covers_under_the_stated_prior() {
    let n = grid_n(30);
    let mut tally = CoverageTally::new("gamma_poisson_stated_prior", LEVEL);
    for (interval, truth) in
        map_replicates(n_sim(), |rep| gamma_replicate(n, GAMMA_TRUE, SEED_GAMMA + rep))
    {
        tally.record(Some(interval), truth);
    }
    tally.assert();
}

/// Exact coverage of the credible interval of `analysis` when truths come from `BETA_TRUE`:
/// `Σ_k P(k) · P_true(θ ∈ CI_analysis(k) | k)` over the beta-binomial marginal of `k`.
fn beta_exact_coverage(n: usize, analysis: BetaHyperparameters) -> f64 {
    let n = n as u64;
    (0..=n)
        .map(|k| {
            let truth = BetaBinomialPosterior::new(BETA_TRUE, k, n).unwrap();
            let (lo, hi) = BetaBinomialPosterior::new(analysis, k, n)
                .unwrap()
                .credible_interval(LEVEL)
                .unwrap();
            truth.log_marginal_likelihood().exp() * (truth.cdf(hi) - truth.cdf(lo))
        })
        .sum()
}

/// Exact coverage of the Gamma-Poisson interval of `analysis` under truths from `GAMMA_TRUE`:
/// the total count `S` of `n` Poisson counts is negative binomial under the true prior and
/// enters both posteriors only through `(S, n)`.
fn gamma_exact_coverage(n: usize, analysis: GammaHyperparameters) -> f64 {
    (0..=MAX_TOTAL_COUNT)
        .map(|s| {
            let truth = GammaPoissonPosterior::from_totals(GAMMA_TRUE, s, n as f64).unwrap();
            let wrong = GammaPoissonPosterior::from_totals(analysis, s, n as f64).unwrap();
            let (lo, hi) = wrong.credible_interval(LEVEL).unwrap();
            true_predictive_total(n, s) * (truth.cdf(hi) - truth.cdf(lo))
        })
        .sum()
}

/// `P(S = s)` of the total of `n` Poisson counts whose rate is `GAMMA_TRUE`-distributed
/// (negative binomial with shape `a` and success probability `b / (b + n)`), by the posterior
/// predictive of a pooled observation over exposure `n` under the prior alone.
fn true_predictive_total(n: usize, s: u64) -> f64 {
    let prior = GammaPoissonPosterior::from_totals(GAMMA_TRUE, 0, 0.0).unwrap();
    prior.predictive_pmf(s, n as f64).unwrap()
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn beta_binomial_misspecified_prior_matches_its_exact_coverage() {
    let n = grid_n(40);
    let exact = beta_exact_coverage(n, BETA_WRONG);
    assert!(exact < 0.90, "a prior this far from the truths must undercover, exact {exact}");
    eprintln!(
        "beta_binomial_misspecified_prior: n={n} exact coverage {exact:.4} (nominal {LEVEL})"
    );
    let mut tally = CoverageTally::new("beta_binomial_misspecified_prior", exact);
    for (interval, truth) in
        map_replicates(n_sim(), |rep| beta_replicate(n, BETA_WRONG, SEED_BETA_MISSPECIFIED + rep))
    {
        tally.record(Some(interval), truth);
    }
    tally.assert();
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn gamma_poisson_misspecified_prior_matches_its_exact_coverage() {
    let n = grid_n(30);
    let exact = gamma_exact_coverage(n, GAMMA_WRONG);
    assert!(exact < 0.90, "a prior this far from the truths must undercover, exact {exact}");
    eprintln!(
        "gamma_poisson_misspecified_prior: n={n} exact coverage {exact:.4} (nominal {LEVEL})"
    );
    let mut tally = CoverageTally::new("gamma_poisson_misspecified_prior", exact);
    for (interval, truth) in map_replicates(n_sim(), |rep| {
        gamma_replicate(n, GAMMA_WRONG, SEED_GAMMA_MISSPECIFIED + rep)
    }) {
        tally.record(Some(interval), truth);
    }
    tally.assert();
}

// -- The select-then-estimate procedure ---------------------------------------------------------

const EFFECT: f64 = 1.0;
const INTERCEPT: f64 = 0.5;
/// Rows of the target regression before the sample-size grid.
const BASE_ROWS: usize = 60;
const SELECTION_RESAMPLES: u32 = 100;

fn selection_grid() -> Vec<f64> {
    (0..=10).map(|i| f64::from(i) / 10.0).collect()
}

/// Intervals of one replicate: the whole procedure, the same data at `α = 1` (full trust in
/// the source) and at `α = 0` (target only).
struct Triple {
    selected: (f64, f64),
    full_trust: (f64, f64),
    target_only: (f64, f64),
}

/// Standard errors of the target's effect coefficient: `2 / √n` for the balanced design.
fn effect_se(rows: usize) -> f64 {
    2.0 / (rows as f64).sqrt()
}

fn selected_replicate(rows: usize, discrepancy: f64, seed: u64) -> Option<Triple> {
    let mut rng = rng(seed);
    let x: Vec<f64> = (0..rows).flat_map(|i| [1.0, (i % 2) as f64]).collect();
    let y: Vec<f64> = (0..rows)
        .map(|i| INTERCEPT + EFFECT * (i % 2) as f64 + standard_normal(&mut rng))
        .collect();

    let mut baseline = PriorSet::new();
    baseline.push(PriorSpec::GaussianCoefficients(GaussianCoefficientPrior {
        mean: Arc::from(vec![0.0, 0.0]),
        variance: Arc::from(vec![1.0e4, 1.0e4]),
    }));
    baseline.push(PriorSpec::KnownResidualVariance(1.0));
    let mut source_prior = PriorSet::new();
    let source_variance = 1.0 / rows as f64;
    source_prior.push(PriorSpec::GaussianCoefficients(GaussianCoefficientPrior {
        mean: Arc::from(vec![INTERCEPT, EFFECT + discrepancy * effect_se(rows)]),
        variance: Arc::from(vec![source_variance, source_variance]),
    }));
    let sources = [ExternalPriorSource {
        id: Arc::from("source"),
        prior: source_prior,
        weight: ExternalPriorWeight::power(1.0).ok()?,
        ess: None,
    }];
    let mut options = AlphaSelectionOptions::new(AlphaCriterion::CrossValidated { folds: 5 });
    options.grid = selection_grid();
    options.n_bootstrap = SELECTION_RESAMPLES;
    options.seed = seed;
    let report = select_linear_power_prior(
        &sources,
        &baseline,
        LinearTarget { x: &x, y: &y, n_rows: rows, n_cols: 2, functional: &[0.0, 1.0] },
        &options,
    )
    .ok()?;
    let row = |index: usize| {
        let r = report.sensitivity[index];
        (r.lower, r.upper)
    };
    Some(Triple {
        selected: report.interval,
        full_trust: row(report.sensitivity.len() - 1),
        target_only: row(0),
    })
}

fn score_cell(rows: usize, discrepancy: f64, seed: u64) -> [CoverageTally; 3] {
    let mut tallies = [
        CoverageTally::new(format!("selected_alpha_d{discrepancy}"), LEVEL),
        CoverageTally::new(format!("full_trust_d{discrepancy}"), LEVEL),
        CoverageTally::new(format!("target_only_d{discrepancy}"), LEVEL),
    ];
    for triple in map_replicates(n_sim(), |rep| selected_replicate(rows, discrepancy, seed + rep)) {
        match triple {
            Some(t) => {
                tallies[0].record(Some(t.selected), EFFECT);
                tallies[1].record(Some(t.full_trust), EFFECT);
                tallies[2].record(Some(t.target_only), EFFECT);
            }
            None => tallies.iter_mut().for_each(CoverageTally::skip),
        }
    }
    tallies
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn selected_alpha_covers_when_the_source_matches() {
    let rows = grid_n(BASE_ROWS);
    let [selected, ..] = score_cell(rows, 0.0, SEED_SELECTED);
    selected.assert_at_least();
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn selected_alpha_covers_under_a_clear_source_conflict() {
    let rows = grid_n(BASE_ROWS);
    let [selected, ..] = score_cell(rows, 8.0, SEED_SELECTED + 1_000);
    selected.assert();
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn selected_alpha_is_not_beaten_by_full_trust_at_moderate_discrepancy() {
    let rows = grid_n(BASE_ROWS);
    for (i, discrepancy) in [1.0_f64, 2.0].into_iter().enumerate() {
        let [selected, full_trust, target_only] =
            score_cell(rows, discrepancy, SEED_SELECTED + 2_000 + 1_000 * i as u64);
        eprintln!(
            "selected-alpha boundary d={discrepancy} rows={rows}: selected {:.3} full-trust {:.3} \
             target-only {:.3} (selected mean length {:.3}, target-only {:.3})",
            selected.rate(),
            full_trust.rate(),
            target_only.rate(),
            selected.mean_length(),
            target_only.mean_length(),
        );
        let slack = 3.0 * coverage_mcse(selected.attempts(), LEVEL);
        assert!(
            selected.rate() + slack >= full_trust.rate(),
            "d={discrepancy}: selected coverage {} is below full trust {}",
            selected.rate(),
            full_trust.rate()
        );
    }
}

fn selected_beta_replicate(
    theta: f64,
    trials: usize,
    seed: u64,
) -> Option<((f64, f64), (f64, f64))> {
    let mut rng = rng(seed);
    let outcomes: Vec<bool> = (0..trials).map(|_| rng.next_f64() < theta).collect();
    let source = BetaSource {
        base: BetaHyperparameters { alpha: 1.0, beta: 1.0 },
        successes: 300,
        trials: 1_000,
    };
    let mut options = AlphaSelectionOptions::new(AlphaCriterion::CrossValidated { folds: 5 });
    options.grid = selection_grid();
    options.n_bootstrap = SELECTION_RESAMPLES;
    options.seed = seed;
    let report = select_beta_power_prior(source, &outcomes, &options).ok()?;
    let full_trust = report.sensitivity.last().map(|r| (r.lower, r.upper))?;
    Some((report.interval, full_trust))
}

/// Source proportion 0.3 from 1,000 trials. A matching target (θ = 0.3) must be covered at
/// least nominally; a clear conflict (θ = 0.6) must not lose to trusting the source.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn selected_alpha_beta_binomial_covers_across_discrepancy() {
    let trials = grid_n(100);
    for (i, (truth, matched)) in [(0.3, true), (0.6, false)].into_iter().enumerate() {
        let mut selected = CoverageTally::new(format!("selected_alpha_beta_theta{truth}"), LEVEL);
        let mut full_trust = CoverageTally::new(format!("full_trust_beta_theta{truth}"), LEVEL);
        let cell = map_replicates(n_sim(), |rep| {
            selected_beta_replicate(truth, trials, SEED_SELECTED_BETA + 1_000 * i as u64 + rep)
        });
        for pair in cell {
            match pair {
                Some((a, b)) => {
                    selected.record(Some(a), truth);
                    full_trust.record(Some(b), truth);
                }
                None => {
                    selected.skip();
                    full_trust.skip();
                }
            }
        }
        eprintln!(
            "selected-alpha beta-binomial theta={truth} trials={trials}: selected {:.3} full-trust {:.3}",
            selected.rate(),
            full_trust.rate()
        );
        if matched {
            selected.assert_at_least();
        } else {
            let slack = 3.0 * coverage_mcse(selected.attempts(), LEVEL);
            assert!(
                selected.rate() + slack >= full_trust.rate(),
                "theta={truth}: selected coverage {} is below full trust {}",
                selected.rate(),
                full_trust.rate()
            );
        }
    }
}
