//! Record-emitting coverage of the conjugate count routes and of the select-then-estimate power
//! priors, each measured through the public facade entry the capability route names.
//!
//! * `association.conjugate_count_posterior`: the exceedance probability and the Poisson rate of a
//!   Bayesian association with a stated Beta or Gamma prior (`AssociationEstimator::count_prior`,
//!   `Association.exceedance`), and the same association under a Beta prior built from the counts
//!   of a source (`Bayesian.prior_from_count`). The truth is drawn from the stated prior, so the
//!   equal-tailed 95% credible interval has exact coverage at the reported level and the
//!   two-sided band applies.
//! * `select_alpha_prior_beta_binomial`, `select_alpha_prior_gamma_poisson` and
//!   `select_alpha_prior_linear` (also the route `prior.composed_named_coordinates`): the whole
//!   select-then-estimate procedure of `antecedent::inference::select_beta_power_prior`,
//!   `select_gamma_power_prior` and `select_linear_power_prior`. The uniform interval (no declared
//!   discrepancy bound) is scored at a truth displaced from the source, and the bias-aware
//!   interval at a truth inside the declared bound. Both are conservative constructions, so each
//!   record is gated one-sided: coverage at least the level, with no upper band.
//!
//! The pass/fail designs of the same procedures stay in `count_conjugate_calibration.rs` and
//! `crates/antecedent-estimate/tests/conjugate_prior_calibration.rs`; they emit no record.
//!
//! Ignored tests run via `scripts/measure_calibration.sh`.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    reason = "bounded simulation sizes, small indices and one design table per test"
)]

mod common;

use std::sync::Arc;

use antecedent::inference::{
    AlphaCriterion, AlphaSelectionOptions, BetaHyperparameters, BetaSource, ExternalPriorSource,
    ExternalPriorWeight, GammaHyperparameters, GammaSource, GaussianCoefficientPrior, LinearTarget,
    PriorSet, PriorSpec, select_beta_power_prior, select_gamma_power_prior,
    select_linear_power_prior,
};
use antecedent::{
    AssociationFunctional, AssociationOutcome, AssociationQuery, BayesianConfig, CausalQuery,
    InferenceMode, Study,
};
use antecedent_core::{CausalRng, ExecutionContext, VariableId};
use antecedent_data::TabularData;
use antecedent_kernels::{sample_gamma, standard_normal};
use common::calibration::{
    Construction, CoverageSide, CoverageTally, REPORTED_LEVEL, RecordKey, ScopeFacts, grid_n,
    grid_seed, map_replicates, n_sim,
};
use common::calibration_bind::{bind_reported, constructions};

/// A replicate's construction, scope facts and validation mode as the bind helpers take them.
type Reported = (Construction, ScopeFacts, &'static str);

/// `IntervalMethod` label of an equal-tailed credible interval of a conjugate posterior.
const POSTERIOR_INTERVAL: &str = "posterior_equal_tailed";
/// Label of the selection-valid uniform interval of a power-prior selection.
const UNIFORM_INTERVAL: &str = "alpha_selected_uniform";
/// Label of the selection-valid bias-aware interval of a power-prior selection.
const BIAS_AWARE_INTERVAL: &str = "alpha_selected_bias_aware";

// ------------------------------------------------------------------------------ the draws

fn draw_beta(prior: BetaHyperparameters, rng: &mut CausalRng) -> f64 {
    let x = sample_gamma(prior.alpha, 1.0, rng);
    let y = sample_gamma(prior.beta, 1.0, rng);
    x / (x + y)
}

/// Knuth's product-of-uniforms Poisson draw (the rates here are small).
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

// ------------------------------------------------------- the association through the study

/// The one posterior interval an association study publishes for its single cell, with the
/// construction the runtime keys it under (interval and functional set to what is scored).
fn cell_interval(
    data: TabularData,
    query: AssociationQuery,
    prior: PriorSet,
    seed: u64,
    functional: &str,
) -> Option<([f64; 2], Reported)> {
    let study = Study::tabular(data)
        .query(CausalQuery::Association(query))
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().prior(prior)))
        .build()
        .ok()?;
    let result = study.run(&ExecutionContext::for_tests(seed)).ok()?;
    let payload = result.association()?;
    let summary = payload.estimate.terms.first()?.posterior?;
    let contract = study.inspect().ok()?;
    let (base, scope, _) = constructions(&contract, &result).into_iter().next()?;
    let construction = Construction {
        structure: "design".into(),
        interval_method: POSTERIOR_INTERVAL.into(),
        se_kind: String::new(),
        dependence: "iid".into(),
        functional: functional.into(),
        reported_level: REPORTED_LEVEL,
        ..base
    };
    let scope = ScopeFacts { replicates_ok: None, posterior_draws: None, ..scope };
    Some(([summary.lower, summary.upper], (construction, scope, "not_applicable")))
}

/// One exceedance replicate: `theta` is drawn from `truth_prior`, the rows exceed the threshold
/// with probability `theta`, and the study analyses them under `analysis_prior`.
fn exceedance_interval(
    n: usize,
    truth_prior: BetaHyperparameters,
    analysis_prior: BetaHyperparameters,
    seed: u64,
) -> Option<(([f64; 2], Reported), f64)> {
    let mut rng = CausalRng::from_seed(grid_seed(seed));
    let theta = draw_beta(truth_prior, &mut rng);
    let y: Vec<f64> =
        (0..n).map(|i| if rng.next_f64() < theta { 0.8 + 1e-3 * i as f64 } else { 0.1 }).collect();
    let data = TabularData::from_f64_columns([("y", &y[..])]).ok()?;
    let query = AssociationQuery::new(VariableId::from_raw(0), Vec::<VariableId>::new())
        .with_functional(AssociationFunctional::exceedance(0.5));
    let mut prior = PriorSet::new();
    prior.push(PriorSpec::Beta(analysis_prior));
    cell_interval(data, query, prior, seed, "exceedance_probability").map(|run| (run, theta))
}

/// One count-rate replicate under a Gamma prior the rate is drawn from.
fn rate_interval(
    n: usize,
    truth_prior: GammaHyperparameters,
    seed: u64,
) -> Option<(([f64; 2], Reported), f64)> {
    let mut rng = CausalRng::from_seed(grid_seed(seed));
    let lambda = sample_gamma(truth_prior.shape, truth_prior.rate, &mut rng);
    let y: Vec<f64> = (0..n).map(|_| draw_poisson(lambda, &mut rng) as f64).collect();
    let data = TabularData::from_f64_columns([("y", &y[..])]).ok()?;
    let query = AssociationQuery::new(VariableId::from_raw(0), Vec::<VariableId>::new())
        .with_functional(AssociationFunctional::Conditional)
        .with_outcome_scale(AssociationOutcome::Count);
    let mut prior = PriorSet::new();
    prior.push(PriorSpec::Gamma(truth_prior));
    cell_interval(data, query, prior, seed, "poisson_rate").map(|run| (run, lambda))
}

fn measure_study(
    test: &'static str,
    dgp: &'static str,
    runs: impl Fn(u64) -> Option<(([f64; 2], Reported), f64)> + Sync,
) {
    let mut tally = CoverageTally::for_record(
        RecordKey { test, dgp, interval: POSTERIOR_INTERVAL },
        REPORTED_LEVEL,
    );
    for run in map_replicates(n_sim(), runs) {
        let Some(((band, reported), truth)) = run else {
            tally.skip();
            continue;
        };
        bind_reported(&mut tally, &[reported]);
        tally.record(Some((band[0], band[1])), truth);
    }
    tally.assert();
}

const EXCEEDANCE_DGP: &str = "beta_drawn_exceedance_probability_known_truth";
const RATE_DGP: &str = "gamma_drawn_poisson_rate_known_truth";
const SOURCE_COUNT_DGP: &str = "beta_prior_from_source_counts_exceedance_known_truth";

const BETA_TRUE: BetaHyperparameters = BetaHyperparameters { alpha: 2.0, beta: 3.0 };
const GAMMA_TRUE: GammaHyperparameters = GammaHyperparameters { shape: 3.0, rate: 1.5 };

const SEED_EXCEEDANCE: u64 = 0x5C01_1001;
const SEED_RATE: u64 = 0x5C01_1101;
const SEED_SOURCE_COUNTS: u64 = 0x5C01_1201;

/// The exceedance probability of a cell under the Beta prior it was drawn from
/// (`AssociationEstimator::count_prior`, `Association.exceedance`).
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn association_exceedance_beta_stated_prior_posterior_equal_tailed_l95() {
    let n = grid_n(40);
    measure_study(
        "association_exceedance_beta_stated_prior_posterior_equal_tailed_l95",
        EXCEEDANCE_DGP,
        |rep| exceedance_interval(n, BETA_TRUE, BETA_TRUE, SEED_EXCEEDANCE + rep),
    );
}

/// The Poisson rate of a count cell under the Gamma prior it was drawn from.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn association_count_rate_gamma_stated_prior_posterior_equal_tailed_l95() {
    let n = grid_n(30);
    measure_study(
        "association_count_rate_gamma_stated_prior_posterior_equal_tailed_l95",
        RATE_DGP,
        |rep| rate_interval(n, GAMMA_TRUE, SEED_RATE + rep),
    );
}

/// The exceedance probability of a cell under a Beta prior built from the counts of a source
/// (`Bayesian.prior_from_count`): 12 successes of 40 source trials at full discount on a
/// `Beta(1, 1)` base, `Beta(13, 29)`, which the truth is drawn from.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn association_exceedance_prior_from_source_counts_posterior_equal_tailed_l95() {
    let n = grid_n(40);
    let source = BetaSource {
        base: BetaHyperparameters { alpha: 1.0, beta: 1.0 },
        successes: 12,
        trials: 40,
    };
    let prior = source.power_prior(1.0).expect("a valid source at full discount");
    measure_study(
        "association_exceedance_prior_from_source_counts_posterior_equal_tailed_l95",
        SOURCE_COUNT_DGP,
        |rep| exceedance_interval(n, prior, prior, SEED_SOURCE_COUNTS + rep),
    );
}

// ------------------------------------------------------ the select-then-estimate procedures

/// Construction of a facade-level procedure that has no study: the power prior chosen from the
/// target data and the interval scored.
fn selection_construction(
    estimator: &str,
    posterior: &str,
    functional: &str,
    interval: &'static str,
) -> (&'static str, Construction) {
    let construction = Construction {
        query: "PowerPriorSelection".into(),
        graph_class: "none".into(),
        structure: "fixed".into(),
        modality: "tabular".into(),
        inference: "Bayesian".into(),
        estimator: estimator.into(),
        interval_method: interval.into(),
        se_kind: String::new(),
        dependence: "iid".into(),
        posterior: posterior.into(),
        functional: functional.into(),
        identification: "point".into(),
        reported_level: REPORTED_LEVEL,
    };
    (interval, construction)
}

/// Score one selection design: `interval(rep)` is the reported interval of replicate `rep`
/// (`None` when the procedure refuses its data) and `truth` the value it must cover. The interval
/// is conservative by construction, so the record is gated one-sided.
fn measure_selection(
    (test, dgp): (&'static str, &'static str),
    construction: &(&'static str, Construction),
    rows: usize,
    truth: f64,
    target: &str,
    interval: impl Fn(u64) -> Option<(f64, f64)> + Sync,
) {
    let mut tally = CoverageTally::for_record(
        RecordKey { test, dgp, interval: construction.0 },
        REPORTED_LEVEL,
    );
    for run in map_replicates(n_sim(), interval) {
        let Some(band) = run else {
            tally.skip();
            continue;
        };
        tally.bind_validation(
            &construction.1,
            ScopeFacts {
                row_count: rows as u64,
                replicates_ok: None,
                posterior_draws: None,
                unidentified_mass: 0.0,
            },
            "not_applicable",
        );
        tally.record(Some(band), truth);
    }
    tally.assert_one_sided(CoverageSide::Lower, target);
}

fn selection_options(bound: Option<f64>, seed: u64) -> AlphaSelectionOptions {
    let mut options = AlphaSelectionOptions::new(AlphaCriterion::CrossValidated { folds: 5 });
    options.grid = (0..=10).map(|i| f64::from(i) / 10.0).collect();
    options.discrepancy_bound = bound;
    options.seed = seed;
    options
}

// -- Beta-Binomial: source proportion 0.3 from 1,000 trials ------------------------------------

const BETA_SELECT_SEED: u64 = 0x5C02_1001;

fn beta_selected(trials: usize, theta: f64, bound: Option<f64>, seed: u64) -> Option<(f64, f64)> {
    let mut rng = CausalRng::from_seed(grid_seed(seed));
    let outcomes: Vec<bool> = (0..trials).map(|_| rng.next_f64() < theta).collect();
    let source = BetaSource {
        base: BetaHyperparameters { alpha: 1.0, beta: 1.0 },
        successes: 300,
        trials: 1_000,
    };
    let report =
        select_beta_power_prior(source, &outcomes, &selection_options(bound, seed)).ok()?;
    Some(report.interval)
}

/// With no declared bound the reported interval is the uniform interval, which covers at the
/// nominal level at a truth far from the source proportion.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn select_alpha_prior_beta_binomial_uniform_interval_l95() {
    let trials = grid_n(100);
    measure_selection(
        (
            "select_alpha_prior_beta_binomial_uniform_interval_l95",
            "beta_binomial_source_0_3_truth_0_45",
        ),
        &selection_construction(
            "select_beta_power_prior",
            "conjugate_beta",
            "binomial_proportion",
            UNIFORM_INTERVAL,
        ),
        trials,
        0.45,
        "the true proportion, 0.45, displaced from the source proportion 0.3",
        |rep| beta_selected(trials, 0.45, None, BETA_SELECT_SEED + rep),
    );
}

/// With a declared bound of 0.05 the bias-aware interval covers at a truth inside it.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn select_alpha_prior_beta_binomial_bias_aware_interval_l95() {
    let trials = grid_n(100);
    measure_selection(
        (
            "select_alpha_prior_beta_binomial_bias_aware_interval_l95",
            "beta_binomial_source_0_3_truth_0_34",
        ),
        &selection_construction(
            "select_beta_power_prior",
            "conjugate_beta",
            "binomial_proportion",
            BIAS_AWARE_INTERVAL,
        ),
        trials,
        0.34,
        "the true proportion, 0.34, within the declared bound 0.05 of the source proportion 0.3",
        |rep| beta_selected(trials, 0.34, Some(0.05), BETA_SELECT_SEED + 1_000_000 + rep),
    );
}

// -- Gamma-Poisson: source rate 2 from 300 unit exposures --------------------------------------

const GAMMA_SELECT_SEED: u64 = 0x5C03_1001;

fn gamma_selected(rows: usize, lambda: f64, bound: Option<f64>, seed: u64) -> Option<(f64, f64)> {
    let mut rng = CausalRng::from_seed(grid_seed(seed));
    let counts: Vec<u64> = (0..rows).map(|_| draw_poisson(lambda, &mut rng)).collect();
    let source = GammaSource {
        base: GammaHyperparameters { shape: 1.0, rate: 1.0 },
        total_count: 600,
        exposure: 300.0,
    };
    let report = select_gamma_power_prior(source, &counts, &selection_options(bound, seed)).ok()?;
    Some(report.interval)
}

/// With no declared bound the reported interval is the uniform interval, which covers at the
/// nominal level at a rate far from the source rate.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn select_alpha_prior_gamma_poisson_uniform_interval_l95() {
    let rows = grid_n(60);
    measure_selection(
        (
            "select_alpha_prior_gamma_poisson_uniform_interval_l95",
            "gamma_poisson_source_rate_2_truth_2_6",
        ),
        &selection_construction(
            "select_gamma_power_prior",
            "conjugate_gamma",
            "poisson_rate",
            UNIFORM_INTERVAL,
        ),
        rows,
        2.6,
        "the true rate, 2.6, displaced from the source rate 2",
        |rep| gamma_selected(rows, 2.6, None, GAMMA_SELECT_SEED + rep),
    );
}

/// With a declared bound of 0.2 the bias-aware interval covers at a rate inside it.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn select_alpha_prior_gamma_poisson_bias_aware_interval_l95() {
    let rows = grid_n(60);
    measure_selection(
        (
            "select_alpha_prior_gamma_poisson_bias_aware_interval_l95",
            "gamma_poisson_source_rate_2_truth_2_1",
        ),
        &selection_construction(
            "select_gamma_power_prior",
            "conjugate_gamma",
            "poisson_rate",
            BIAS_AWARE_INTERVAL,
        ),
        rows,
        2.1,
        "the true rate, 2.1, within the declared bound 0.2 of the source rate 2",
        |rep| gamma_selected(rows, 2.1, Some(0.2), GAMMA_SELECT_SEED + 1_000_000 + rep),
    );
}

// -- Gaussian linear: a balanced two-arm regression with effect 1 ------------------------------

const LINEAR_SELECT_SEED: u64 = 0x5C04_1001;
const EFFECT: f64 = 1.0;
const INTERCEPT: f64 = 0.5;
const LINEAR_BASE_ROWS: usize = 60;

/// Standard error of the target's effect coefficient: `2 / sqrt(n)` for the balanced design.
fn effect_se(rows: usize) -> f64 {
    2.0 / (rows as f64).sqrt()
}

/// The reported interval of the effect coefficient when the external Gaussian prior (four times
/// as informative as the target) is displaced from the truth by `discrepancy` standard errors.
fn linear_selected(
    rows: usize,
    discrepancy: f64,
    bound: Option<f64>,
    seed: u64,
) -> Option<(f64, f64)> {
    let mut rng = CausalRng::from_seed(grid_seed(seed));
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
    let report = select_linear_power_prior(
        &sources,
        &baseline,
        LinearTarget { x: &x, y: &y, n_rows: rows, n_cols: 2, functional: &[0.0, 1.0] },
        &selection_options(bound, seed),
    )
    .ok()?;
    Some(report.interval)
}

/// With no declared bound the reported interval is the uniform interval, which covers at the
/// nominal level with the source displaced four standard errors.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn select_alpha_prior_linear_uniform_interval_l95() {
    let rows = grid_n(LINEAR_BASE_ROWS);
    measure_selection(
        (
            "select_alpha_prior_linear_uniform_interval_l95",
            "balanced_two_arm_effect_1_source_displaced_4_se",
        ),
        &selection_construction(
            "select_linear_power_prior",
            "conjugate_normal",
            "coefficient_effect",
            UNIFORM_INTERVAL,
        ),
        rows,
        EFFECT,
        "the true effect coefficient, 1, with the external prior displaced four standard errors",
        |rep| linear_selected(rows, 4.0, None, LINEAR_SELECT_SEED + rep),
    );
}

/// With a declared bound of one target standard error the bias-aware interval covers with the
/// source displaced half a standard error.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn select_alpha_prior_linear_bias_aware_interval_l95() {
    let rows = grid_n(LINEAR_BASE_ROWS);
    measure_selection(
        (
            "select_alpha_prior_linear_bias_aware_interval_l95",
            "balanced_two_arm_effect_1_source_displaced_half_se",
        ),
        &selection_construction(
            "select_linear_power_prior",
            "conjugate_normal",
            "coefficient_effect",
            BIAS_AWARE_INTERVAL,
        ),
        rows,
        EFFECT,
        "the true effect coefficient, 1, with the external prior displaced half a standard error inside the declared bound of one",
        |rep| {
            linear_selected(rows, 0.5, Some(effect_se(rows)), LINEAR_SELECT_SEED + 1_000_000 + rep)
        },
    );
}
