//! Conjugate count posteriors as analysis routes: the exceedance probability `P(Y > t)` and the
//! Poisson rate of a count outcome within each cell of a conditional association, compiled and
//! executed through the study under `Bayesian` with a Beta or Gamma prior.
//!
//! Known-truth checks compare every cell's posterior with the closed form (`Beta(a + k, b + n -
//! k)`, `Gamma(shape + sum y, rate + n)`) and its equal-tailed credible interval, the posterior
//! predictive of one further unit, the planned posterior algorithm, and the refusals that name
//! the next attempt. A Beta prior never changes what the cells describe: the result stays an
//! unsupported, `not_measured` description of the observed law.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    reason = "fixture counts convert exactly and exact constants are compared"
)]

use antecedent::inference::{BetaHyperparameters, GammaHyperparameters, PriorSet, PriorSpec};
use antecedent::planner::statistical::PosteriorAlgorithm;
use antecedent::{
    AssociationFunctional, AssociationOutcome, AssociationQuery, BayesianConfig, CausalQuery,
    InferenceMode, Study,
};
use antecedent_core::{ExecutionContext, VariableId};
use antecedent_data::TabularData;
use antecedent_prob::{BetaBinomialPosterior, GammaPoissonPosterior};

const LEVEL: f64 = 0.95;

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn ctx(seed: u64) -> ExecutionContext {
    ExecutionContext::for_tests(seed)
}

fn beta(alpha: f64, beta: f64) -> PriorSet {
    let mut prior = PriorSet::new();
    prior.push(PriorSpec::Beta(BetaHyperparameters { alpha, beta }));
    prior
}

fn gamma(shape: f64, rate: f64) -> PriorSet {
    let mut prior = PriorSet::new();
    prior.push(PriorSpec::Gamma(GammaHyperparameters { shape, rate }));
    prior
}

fn study(data: TabularData, query: AssociationQuery, prior: PriorSet) -> Study {
    Study::tabular(data)
        .query(CausalQuery::Association(query))
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().prior(prior)))
        .build()
        .unwrap()
}

/// Two cells (`x = 0`, `x = 1`) of 40 and 60 rows whose outcome exceeds 0.5 in 15 and 45 rows.
fn exceedance_table() -> (TabularData, [(u64, u64); 2]) {
    let mut x = Vec::new();
    let mut y = Vec::new();
    for (level, rows, above) in [(0.0, 40, 15), (1.0, 60, 45)] {
        for i in 0..rows {
            x.push(level);
            y.push(if i < above { 0.8 + 0.001 * f64::from(i) } else { 0.1 + 0.001 * f64::from(i) });
        }
    }
    (TabularData::from_f64_columns([("x", &x[..]), ("y", &y[..])]).unwrap(), [(15, 40), (45, 60)])
}

#[test]
fn the_exceedance_of_each_cell_is_the_beta_binomial_posterior_in_closed_form() {
    let (data, cells) = exceedance_table();
    let query =
        AssociationQuery::new(v(1), [v(0)]).with_functional(AssociationFunctional::exceedance(0.5));
    let study = study(data, query, beta(2.0, 3.0));

    let plan = study.compile(&ctx(1)).unwrap();
    let posterior = plan.lowered().unwrap().statistical().posterior().unwrap();
    assert_eq!(posterior.algorithm(), PosteriorAlgorithm::Conjugate);

    let result = study.run(&ctx(1)).unwrap();
    let payload = result.association().unwrap();
    let record = payload.estimate.posterior.unwrap();
    assert_eq!((record.method, record.draws), ("conjugate_beta_binomial", 0));
    assert_eq!(payload.estimate.selection.requested, "exceedance");
    assert_eq!(payload.estimate.selection.analysis.as_str(), "stratified_probability");
    assert_eq!(payload.estimate.terms.len(), 2);
    for (term, (successes, trials)) in payload.estimate.terms.iter().zip(cells) {
        let closed = BetaBinomialPosterior::new(
            BetaHyperparameters { alpha: 2.0, beta: 3.0 },
            successes,
            trials,
        )
        .unwrap();
        let (lower, upper) = closed.credible_interval(LEVEL).unwrap();
        let summary = term.posterior.unwrap();
        assert!((summary.mean - closed.mean()).abs() <= 1e-12, "{summary:?}");
        assert!((summary.sd - closed.variance().sqrt()).abs() <= 1e-12, "{summary:?}");
        assert!((summary.lower - lower).abs() <= 1e-12 && (summary.upper - upper).abs() <= 1e-12);
        // The posterior is Beta(a + k, b + n - k); the observed proportion is the point.
        let cell = term.conjugate.unwrap();
        assert_eq!(cell.family, "beta_binomial");
        assert_eq!(cell.prior, (2.0, 3.0));
        assert_eq!(cell.posterior, (2.0 + successes as f64, 3.0 + (trials - successes) as f64));
        assert_eq!((cell.count, cell.exposure), (successes, trials as f64));
        assert!((cell.log_marginal_likelihood - closed.log_marginal_likelihood()).abs() <= 1e-12);
        assert!((cell.predictive_mean - closed.predictive_mean(1)).abs() <= 1e-12);
        assert!((cell.predictive_variance - closed.predictive_variance(1)).abs() <= 1e-12);
        assert!((term.estimate - successes as f64 / trials as f64).abs() <= 1e-12);
    }
    // The cells describe the observed law; the prior never upgrades that.
    assert!(!result.association().unwrap().estimate.terms.is_empty());
}

#[test]
fn a_binary_outcomes_conditional_probability_is_the_same_posterior() {
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for (level, rows, ones) in [(0.0, 50, 10), (1.0, 50, 30)] {
        for i in 0..rows {
            x.push(level);
            y.push(f64::from(i < ones));
        }
    }
    let data = TabularData::from_f64_columns([("x", &x[..]), ("y", &y[..])]).unwrap();
    let query = AssociationQuery::new(v(1), [v(0)])
        .with_functional(AssociationFunctional::Conditional)
        .with_outcome_scale(AssociationOutcome::Binary);
    let result = study(data, query, beta(1.0, 1.0)).run(&ctx(2)).unwrap();
    let payload = result.association().unwrap();
    for (term, successes) in payload.estimate.terms.iter().zip([10_u64, 30]) {
        let closed = BetaBinomialPosterior::new(
            BetaHyperparameters { alpha: 1.0, beta: 1.0 },
            successes,
            50,
        )
        .unwrap();
        assert!((term.posterior.unwrap().mean - closed.mean()).abs() <= 1e-12);
    }
}

#[test]
fn the_rate_of_a_count_outcome_in_each_cell_is_the_gamma_poisson_posterior_in_closed_form() {
    let cell_counts: [Vec<u64>; 2] =
        [vec![0, 1, 1, 2, 0, 3, 1, 0, 2, 1], vec![4, 2, 5, 3, 6, 4, 3, 5, 2, 4, 3, 5]];
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for (level, counts) in cell_counts.iter().enumerate() {
        for &count in counts {
            x.push(level as f64);
            y.push(count as f64);
        }
    }
    let data = TabularData::from_f64_columns([("x", &x[..]), ("y", &y[..])]).unwrap();
    let query = AssociationQuery::new(v(1), [v(0)])
        .with_functional(AssociationFunctional::Conditional)
        .with_outcome_scale(AssociationOutcome::Count);
    let study = study(data, query, gamma(2.0, 0.5));
    let plan = study.compile(&ctx(3)).unwrap();
    let posterior = plan.lowered().unwrap().statistical().posterior().unwrap();
    assert_eq!(posterior.algorithm(), PosteriorAlgorithm::Conjugate);
    let result = study.run(&ctx(3)).unwrap();
    let payload = result.association().unwrap();
    assert_eq!(payload.estimate.posterior.unwrap().method, "conjugate_gamma_poisson");
    for (term, counts) in payload.estimate.terms.iter().zip(&cell_counts) {
        let closed = GammaPoissonPosterior::from_counts(
            GammaHyperparameters { shape: 2.0, rate: 0.5 },
            counts,
            None,
        )
        .unwrap();
        let (lower, upper) = closed.credible_interval(LEVEL).unwrap();
        let summary = term.posterior.unwrap();
        assert!((summary.mean - closed.mean()).abs() <= 1e-12, "{summary:?}");
        assert!((summary.sd - closed.variance().sqrt()).abs() <= 1e-12);
        assert!((summary.lower - lower).abs() <= 1e-12 && (summary.upper - upper).abs() <= 1e-12);
        let cell = term.conjugate.unwrap();
        assert_eq!(cell.family, "gamma_poisson");
        assert_eq!(
            cell.posterior,
            (2.0 + counts.iter().sum::<u64>() as f64, 0.5 + counts.len() as f64)
        );
        assert!((cell.predictive_mean - closed.predictive_mean(1.0)).abs() <= 1e-12);
        assert!((cell.predictive_variance - closed.predictive_variance(1.0)).abs() <= 1e-12);
    }
}

#[test]
fn a_prior_of_the_wrong_family_or_a_continuous_conditional_mean_is_refused_with_the_next_attempt() {
    let (data, _) = exceedance_table();
    // A Gamma prior has no reading of an exceedance probability.
    let exceedance =
        AssociationQuery::new(v(1), [v(0)]).with_functional(AssociationFunctional::exceedance(0.5));
    let refused = study(data.clone(), exceedance, gamma(2.0, 1.0)).run(&ctx(4)).unwrap_err();
    assert!(refused.to_string().contains("Beta prior"), "{refused}");
    // A continuous outcome's conditional mean has no count posterior.
    let mean =
        AssociationQuery::new(v(1), [v(0)]).with_functional(AssociationFunctional::Conditional);
    let refused = study(data, mean, beta(2.0, 3.0)).run(&ctx(4)).unwrap_err();
    assert!(refused.to_string().contains("Next attempt"), "{refused}");
}
