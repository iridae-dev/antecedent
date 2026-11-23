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
//!   11-point grid, known residual variance 1) and of `select_beta_power_prior`. The linear
//!   target is a balanced two-arm regression with effect `1`; the source is an external
//!   Gaussian prior four times as informative as the target whose mean is displaced from the
//!   truth by a multiple `d` of the target's effect standard error. The reported interval
//!   must cover at least nominally in every cell: the uniform interval (no declared bound)
//!   at `d = 0, 1, 2, 4, 8`, and the bias-aware interval under a declared bound of one
//!   standard error at `d = 0, 0.5, 1`, where it must also be narrower than the target-only
//!   interval. Full-trust (`α = 1`) and target-only (`α = 0`) coverage are printed as
//!   diagnostics. The count model is scored the same way against truths near and far from
//!   the source proportion.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

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
use calibration::{CoverageTally, REPORTED_LEVEL, grid_n, grid_seed, map_replicates, n_sim};

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
/// Source displacements, in target effect standard errors, scored for the uniform interval.
const DISPLACEMENTS: [f64; 5] = [0.0, 1.0, 2.0, 4.0, 8.0];
/// Displacements within the declared bound (one target effect standard error).
const BOUNDED_DISPLACEMENTS: [f64; 3] = [0.0, 0.5, 1.0];

fn selection_grid() -> Vec<f64> {
    (0..=10).map(|i| f64::from(i) / 10.0).collect()
}

/// Intervals of one replicate on the same data.
struct Quad {
    /// The reported interval (bias-aware when a bound was declared, else uniform).
    reported: (f64, f64),
    /// The target-only interval widened to the estimate.
    uniform: (f64, f64),
    /// The same data at `α = 1` (full trust in the source), a diagnostic.
    full_trust: (f64, f64),
    /// The same data at `α = 0` (target only), a diagnostic.
    target_only: (f64, f64),
}

/// Standard error of the target's effect coefficient: `2 / √n` for the balanced design.
fn effect_se(rows: usize) -> f64 {
    2.0 / (rows as f64).sqrt()
}

fn selected_replicate(
    rows: usize,
    discrepancy: f64,
    bound: Option<f64>,
    seed: u64,
) -> Option<Quad> {
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
    options.discrepancy_bound = bound;
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
    Some(Quad {
        reported: report.interval,
        uniform: report.uniform_interval,
        full_trust: row(report.sensitivity.len() - 1),
        target_only: row(0),
    })
}

/// Tallies of the reported, uniform, full-trust and target-only intervals over one cell.
fn score_cell(rows: usize, discrepancy: f64, bound: Option<f64>, seed: u64) -> [CoverageTally; 4] {
    let mut tallies = [
        CoverageTally::new(format!("selected_alpha_reported_d{discrepancy}"), LEVEL),
        CoverageTally::new(format!("selected_alpha_uniform_d{discrepancy}"), LEVEL),
        CoverageTally::new(format!("full_trust_d{discrepancy}"), LEVEL),
        CoverageTally::new(format!("target_only_d{discrepancy}"), LEVEL),
    ];
    for quad in
        map_replicates(n_sim(), |rep| selected_replicate(rows, discrepancy, bound, seed + rep))
    {
        match quad {
            Some(q) => {
                tallies[0].record(Some(q.reported), EFFECT);
                tallies[1].record(Some(q.uniform), EFFECT);
                tallies[2].record(Some(q.full_trust), EFFECT);
                tallies[3].record(Some(q.target_only), EFFECT);
            }
            None => tallies.iter_mut().for_each(CoverageTally::skip),
        }
    }
    tallies
}

fn print_cell(label: &str, tallies: &[CoverageTally; 4]) {
    eprintln!(
        "selected-alpha {label}: reported {:.3} (mean length {:.3}), uniform {:.3} ({:.3}), \
         full-trust {:.3}, target-only {:.3} ({:.3})",
        tallies[0].rate(),
        tallies[0].mean_length(),
        tallies[1].rate(),
        tallies[1].mean_length(),
        tallies[2].rate(),
        tallies[3].rate(),
        tallies[3].mean_length(),
    );
}

/// With no declared bound the reported interval is the uniform interval, which covers at
/// the nominal level at every displacement of the source, however large.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn selected_alpha_uniform_interval_covers_at_every_discrepancy() {
    let rows = grid_n(BASE_ROWS);
    for (i, discrepancy) in DISPLACEMENTS.into_iter().enumerate() {
        let tallies = score_cell(rows, discrepancy, None, SEED_SELECTED + 1_000 * i as u64);
        print_cell(&format!("uniform d={discrepancy} rows={rows}"), &tallies);
        tallies[0].assert_at_least();
    }
}

/// With a declared bound of one target standard error the bias-aware interval covers at the
/// nominal level at every displacement inside the bound, and is narrower than the
/// target-only interval there.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn selected_alpha_bias_aware_interval_covers_within_the_declared_bound() {
    let rows = grid_n(BASE_ROWS);
    let bound = effect_se(rows);
    for (i, discrepancy) in BOUNDED_DISPLACEMENTS.into_iter().enumerate() {
        let tallies =
            score_cell(rows, discrepancy, Some(bound), SEED_SELECTED + 10_000 + 1_000 * i as u64);
        print_cell(&format!("bias-aware d={discrepancy} bound={bound:.3} rows={rows}"), &tallies);
        assert!(
            tallies[0].mean_length() < tallies[3].mean_length(),
            "d={discrepancy}: the bias-aware interval is not narrower than the target-only one"
        );
        tallies[0].assert_at_least();
    }
}

fn selected_beta_replicate(
    theta: f64,
    trials: usize,
    bound: Option<f64>,
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
    options.discrepancy_bound = bound;
    options.seed = seed;
    let report = select_beta_power_prior(source, &outcomes, &options).ok()?;
    Some((report.interval, report.uniform_interval))
}

/// Source proportion 0.3 from 1,000 trials. The uniform interval (Clopper-Pearson of the
/// target trials, widened to the estimate) covers at every truth; with a declared bound of
/// 0.05 the bias-aware interval covers at truths 0.3 and 0.34, inside it.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn selected_alpha_beta_binomial_covers_across_discrepancy() {
    let trials = grid_n(100);
    let cells: [(f64, Option<f64>); 5] =
        [(0.3, None), (0.45, None), (0.7, None), (0.3, Some(0.05)), (0.34, Some(0.05))];
    for (i, (truth, bound)) in cells.into_iter().enumerate() {
        let mut reported =
            CoverageTally::new(format!("selected_alpha_beta_theta{truth}_bound{bound:?}"), LEVEL);
        let mut uniform = CoverageTally::new(format!("uniform_beta_theta{truth}"), LEVEL);
        for pair in map_replicates(n_sim(), |rep| {
            selected_beta_replicate(
                truth,
                trials,
                bound,
                SEED_SELECTED_BETA + 1_000 * i as u64 + rep,
            )
        }) {
            if let Some((a, b)) = pair {
                reported.record(Some(a), truth);
                uniform.record(Some(b), truth);
            } else {
                reported.skip();
                uniform.skip();
            }
        }
        eprintln!(
            "selected-alpha beta-binomial theta={truth} bound={bound:?} trials={trials}: \
             reported {:.3} (mean length {:.3}), uniform {:.3} ({:.3})",
            reported.rate(),
            reported.mean_length(),
            uniform.rate(),
            uniform.mean_length()
        );
        reported.assert_at_least();
    }
}
