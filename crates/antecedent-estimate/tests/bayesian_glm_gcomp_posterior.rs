//! Bayesian GLM g-computation: the posterior of the mean-scale contrast of a Bernoulli or
//! Poisson outcome model under a Gaussian coefficient prior.
//!
//! The target is the average of `μ(active, x) − μ(control, x)` over the observed rows, evaluated
//! at every coefficient draw. It is the marginal effect the frequentist GLM adjustment reports,
//! not the link-scale coefficient. A treatment-only design has two parameters and two cells, so
//! the posterior has a closed-form reference: with a nearly flat prior on the link scale the cell
//! means have Beta (Bernoulli) or Gamma (Poisson) posteriors, and the contrast is their
//! difference. Expected values below come from those closed forms, not from the estimator.

use std::sync::Arc;

use antecedent_core::{ExecutionContext, IdentificationStatus};
use antecedent_estimate::{
    BayesianGCompWorkspace, BayesianGComputationAte, OverlapPolicy, PreparedBayesianProblem,
    SerialDependence,
};
use antecedent_prob::BayesLikelihood;
use antecedent_stats::CompiledDesign;

const ARM_ROWS: usize = 200;

fn problem(control_outcomes: &[f64], active_outcomes: &[f64]) -> PreparedBayesianProblem {
    let treatment: Vec<f64> = (0..control_outcomes.len())
        .map(|_| 0.0)
        .chain((0..active_outcomes.len()).map(|_| 1.0))
        .collect();
    let outcome: Vec<f64> =
        control_outcomes.iter().chain(active_outcomes.iter()).copied().collect();
    let design = CompiledDesign::linear_adjustment(&treatment, &[], &outcome, &[]).expect("design");
    PreparedBayesianProblem {
        design,
        method: Arc::from("backdoor.adjustment"),
        adjustment_set: Arc::from(Vec::new()),
        active: 1.0,
        control: 0.0,
        overlap: OverlapPolicy::ExplicitOverride,
        coef_names: None,
        unit_ids: None,
        serial_dependence: SerialDependence::Iid,
    }
}

fn counts(successes: usize) -> Vec<f64> {
    (0..ARM_ROWS).map(|row| if row < successes { 1.0 } else { 0.0 }).collect()
}

/// Posterior mean and standard deviation of the contrast under `likelihood`.
fn posterior_moments(likelihood: BayesLikelihood, problem: &PreparedBayesianProblem) -> (f64, f64) {
    let estimator =
        BayesianGComputationAte::new().with_likelihood(likelihood).with_n_draws(4000).with_seed(5);
    let posterior = estimator
        .fit(
            problem,
            IdentificationStatus::NonparametricallyIdentified,
            &mut BayesianGCompWorkspace::default(),
            &ExecutionContext::for_tests(1),
        )
        .expect("Laplace GLM posterior");
    let effect = posterior.effect_column().expect("effect column");
    (posterior.summaries.mean[effect], posterior.summaries.sd[effect])
}

/// Beta(a, b) variance.
fn beta_variance(a: f64, b: f64) -> f64 {
    a * b / ((a + b) * (a + b) * (a + b + 1.0))
}

#[test]
fn logit_contrast_is_the_risk_difference_with_the_beta_posterior_spread() {
    // 60/200 and 100/200 successes: risks 0.3 and 0.5, risk difference 0.2. The Haldane-like
    // flat-logit posteriors are Beta(60, 140) and Beta(100, 100).
    let problem = problem(&counts(60), &counts(100));
    let (mean, sd) = posterior_moments(BayesLikelihood::BernoulliLogit, &problem);
    let reference_sd = (beta_variance(60.0, 140.0) + beta_variance(100.0, 100.0)).sqrt();
    assert!((mean - 0.2).abs() < 0.006, "posterior mean {mean}");
    assert!((sd / reference_sd - 1.0).abs() < 0.07, "sd {sd} against {reference_sd}");
    // The treatment coefficient is the log odds ratio, ln(1 / 0.4286 ...) = 0.8473, not this
    // contrast.
    assert!((mean - 0.847_298).abs() > 0.5);
}

#[test]
fn probit_contrast_answers_the_same_probability_scale_target() {
    let problem = problem(&counts(60), &counts(100));
    let (mean, sd) = posterior_moments(BayesLikelihood::BernoulliProbit, &problem);
    let reference_sd = (beta_variance(60.0, 140.0) + beta_variance(100.0, 100.0)).sqrt();
    assert!((mean - 0.2).abs() < 0.006, "posterior mean {mean}");
    assert!((sd / reference_sd - 1.0).abs() < 0.07, "sd {sd} against {reference_sd}");
}

#[test]
fn poisson_contrast_is_the_difference_of_gamma_posterior_means() {
    // 200 rows per arm. Control counts sum to 300 (mean 1.5), active counts to 500 (mean 2.5).
    // Gamma(300, 200) and Gamma(500, 200) have variances 300 / 200^2 and 500 / 200^2.
    let control: Vec<f64> = (0..ARM_ROWS).map(|row| if row < 100 { 3.0 } else { 0.0 }).collect();
    let active: Vec<f64> = (0..ARM_ROWS).map(|row| if row < 100 { 5.0 } else { 0.0 }).collect();
    let problem = problem(&control, &active);
    let (mean, sd) = posterior_moments(BayesLikelihood::PoissonLog, &problem);
    let reference_sd = ((300.0 + 500.0) / (200.0_f64 * 200.0)).sqrt();
    assert!((mean - 1.0).abs() < 0.02, "posterior mean {mean}");
    assert!((sd / reference_sd - 1.0).abs() < 0.07, "sd {sd} against {reference_sd}");
}

#[test]
fn a_tight_prior_shrinks_the_contrast_that_a_flat_prior_leaves_alone() {
    let problem = problem(&counts(60), &counts(100));
    let tight = BayesianGComputationAte::new()
        .with_likelihood(BayesLikelihood::BernoulliLogit)
        .with_prior_scale(0.05)
        .with_n_draws(2000)
        .with_seed(5)
        .fit(
            &problem,
            IdentificationStatus::NonparametricallyIdentified,
            &mut BayesianGCompWorkspace::default(),
            &ExecutionContext::for_tests(1),
        )
        .expect("tight prior fit");
    let effect = tight.effect_column().expect("effect column");
    // N(0, 0.05^2) on every logit coefficient keeps both risks near one half and the contrast
    // near zero: the prior is part of the posterior, not a penalty on a point estimate.
    assert!(tight.summaries.mean[effect].abs() < 0.01, "{}", tight.summaries.mean[effect]);
}
