//! The Bayesian outcome likelihood: which routes fit it, which refuse it, and
//! the disclosure of a Gaussian fit to a binary or count outcome.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "test scaffolding compares exact constants and indexes with small literals"
)]

use antecedent::{BayesianConfig, CausalError, InferenceMode, RefuteSuite, Study, StudyResult};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ConditionalEffectQuery, ExecutionContext, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Cpdag, Dag, DenseNodeId};
use antecedent_prob::BayesLikelihood;

const DISCLOSURE: &str = "estimate.bayesian.gaussian_likelihood_discrete_outcome";

fn v(i: u32) -> VariableId {
    VariableId::from_raw(i)
}

fn d(i: u32) -> DenseNodeId {
    DenseNodeId::from_raw(i)
}

/// Deterministic LCG uniforms in `(0, 1)`.
fn uniforms(seed: u64) -> impl FnMut() -> f64 {
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    move || {
        state =
            state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Columns `t, y, z`: `z ~ U(-1.5, 1.5)`, `t ~ Bern(σ(z))`, and an outcome
/// chosen by `outcome` from `(t, z, u)`. Returns the data and the in-sample
/// logit risk difference `mean σ(-0.5 + 1.2 + 0.8z) − σ(-0.5 + 0.8z)`.
fn data(n: usize, seed: u64, outcome: impl Fn(f64, f64, f64) -> f64) -> (TabularData, f64) {
    let mut u = uniforms(seed);
    let (mut t, mut y, mut z) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    let mut truth = 0.0;
    for i in 0..n {
        z[i] = 3.0 * u() - 1.5;
        t[i] = f64::from(u() < sigmoid(z[i]));
        y[i] = outcome(t[i], z[i], u());
        truth += sigmoid(0.7 + 0.8 * z[i]) - sigmoid(-0.5 + 0.8 * z[i]);
    }
    let table = TabularData::from_f64_columns([("t", t.as_slice()), ("y", &y), ("z", &z)]).unwrap();
    (table, truth / n as f64)
}

fn binary(n: usize, seed: u64) -> (TabularData, f64) {
    data(n, seed, |t, z, u| f64::from(u < sigmoid(-0.5 + 1.2 * t + 0.8 * z)))
}

fn dag() -> Dag {
    let mut g = Dag::with_variables(3);
    for (a, b) in [(2, 0), (2, 1), (0, 1)] {
        g.insert_directed(d(a), d(b)).unwrap();
    }
    g
}

fn bayes(likelihood: BayesLikelihood) -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::laplace().likelihood(likelihood))
}

fn run_dag(
    table: TabularData,
    inference: InferenceMode,
) -> Result<(Study, StudyResult), CausalError> {
    let study = Study::tabular(table)
        .graph(dag())
        .query(AverageEffectQuery::binary_ate(v(0), v(1)))
        .inference(inference)
        .refute(RefuteSuite::None)
        .build()?;
    let result = study.run(&ExecutionContext::for_tests(7))?;
    Ok((study, result))
}

fn assert_refused(result: Result<Study, CausalError>, what: &str) {
    let error = result.err().unwrap_or_else(|| panic!("{what}: expected a refusal"));
    let message = error.to_string();
    assert!(message.contains("reason=likelihood_not_supported"), "{what}: {message}");
}

fn disclosed(result: &StudyResult) -> Option<String> {
    result.diagnostics.iter().find(|d| d.code.as_ref() == DISCLOSURE).map(|d| {
        d.fields
            .iter()
            .find(|(key, _)| key.as_ref() == "outcome_kind")
            .map(|(_, value)| value.to_string())
            .unwrap_or_default()
    })
}

#[test]
fn a_logit_likelihood_recovers_the_risk_difference_on_a_binary_outcome() {
    let (table, truth) = binary(3000, 11);
    let (_, logit) = run_dag(table.clone(), bayes(BayesLikelihood::BernoulliLogit)).unwrap();
    let posterior = logit.posterior.as_ref().expect("posterior");
    assert!(
        (logit.estimate.ate - truth).abs() < 0.05,
        "logit ATE {} vs in-sample risk difference {truth}",
        logit.estimate.ate
    );
    assert!(
        format!("{:?}", posterior.assumptions).contains("Bernoulli logit outcome regression"),
        "the fitted outcome model is named"
    );
    assert_eq!(disclosed(&logit), None, "a Bernoulli fit is not disclosed as a misfit");

    let (_, gaussian) = run_dag(table, bayes(BayesLikelihood::GaussianIdentity)).unwrap();
    assert_ne!(gaussian.estimate.ate.to_bits(), logit.estimate.ate.to_bits());
    assert_eq!(disclosed(&gaussian).as_deref(), Some("binary"));
}

#[test]
fn the_likelihood_is_part_of_the_inference_binding() {
    let (table, _) = binary(400, 3);
    let ctx = ExecutionContext::for_tests(1);
    let contract = |likelihood| {
        Study::tabular(table.clone())
            .graph(dag())
            .query(AverageEffectQuery::binary_ate(v(0), v(1)))
            .inference(bayes(likelihood))
            .refute(RefuteSuite::None)
            .build()
            .unwrap()
            .prepare(&ctx)
            .unwrap()
            .contract()
            .unwrap()
    };
    let gaussian = contract(BayesLikelihood::GaussianIdentity);
    let logit = contract(BayesLikelihood::BernoulliLogit);
    let probit = contract(BayesLikelihood::BernoulliProbit);
    assert_eq!(gaussian.identities.target, logit.identities.target);
    assert_ne!(gaussian.identities.inference_binding, logit.identities.inference_binding);
    assert_ne!(logit.identities.inference_binding, probit.identities.inference_binding);
    assert_eq!(logit.posterior.as_ref(), "laplace.bernoulli_logit.prior_scale=10");
}

#[test]
fn a_poisson_likelihood_fits_a_count_outcome_and_gaussian_is_disclosed() {
    let (table, _) = data(1500, 5, |t, z, u| {
        let mean = (0.2 + 0.5 * t + 0.3 * z).exp();
        let (mut k, mut product, limit) = (0.0, u, (-mean).exp());
        let mut next = uniforms((u * 1e9) as u64);
        while product > limit {
            k += 1.0;
            product *= next();
        }
        k
    });
    let (_, poisson) = run_dag(table.clone(), bayes(BayesLikelihood::PoissonLog)).unwrap();
    assert!(poisson.estimate.ate.is_finite() && poisson.estimate.ate > 0.0);
    assert_eq!(disclosed(&poisson), None);
    let (_, gaussian) = run_dag(table, bayes(BayesLikelihood::GaussianIdentity)).unwrap();
    assert_eq!(disclosed(&gaussian).as_deref(), Some("count"));
}

#[test]
fn a_continuous_outcome_is_not_disclosed() {
    let (table, _) = data(400, 9, |t, z, u| 2.0 * t + z + u);
    let (_, gaussian) = run_dag(table, bayes(BayesLikelihood::GaussianIdentity)).unwrap();
    assert_eq!(disclosed(&gaussian), None);
    let (table, _) = binary(400, 9);
    let (_, frequentist) = run_dag(table, InferenceMode::Frequentist).unwrap();
    assert_eq!(disclosed(&frequentist), None, "only a Bayesian model fit is disclosed");
}

#[test]
fn routes_that_cannot_fit_a_non_gaussian_likelihood_refuse_by_code() {
    let (table, _) = binary(200, 1);
    let logit = bayes(BayesLikelihood::BernoulliLogit);

    assert_refused(
        Study::tabular(table.clone())
            .graph(dag())
            .query(AverageEffectQuery::binary_ate(v(0), v(1)))
            .inference(InferenceMode::Bayesian(
                BayesianConfig::conjugate().likelihood(BayesLikelihood::BernoulliLogit),
            ))
            .build(),
        "conjugate backend",
    );

    let mut cpdag = Cpdag::with_variables(3);
    cpdag.insert_directed(d(2), d(1)).unwrap();
    cpdag.insert_directed(d(0), d(1)).unwrap();
    cpdag.insert_undirected(d(2), d(0)).unwrap();
    assert_refused(
        Study::tabular(table.clone())
            .graph(cpdag)
            .query(AverageEffectQuery::binary_ate(v(0), v(1)))
            .inference(logit.clone())
            .build(),
        "Cpdag envelope",
    );

    let conditional = ConditionalEffectQuery::try_new(
        AverageEffectQuery::binary_ate(v(0), v(1)).with_effect_modifiers([v(2)]),
    )
    .unwrap();
    assert_refused(
        Study::tabular(table.clone())
            .graph(dag())
            .query(CausalQuery::ConditionalEffect(conditional))
            .inference(logit)
            .build(),
        "ConditionalEffect",
    );

    assert_refused(
        Study::tabular(table)
            .graph(dag())
            .query(AverageEffectQuery::binary_ate(v(0), v(1)))
            .inference(InferenceMode::Bayesian(
                BayesianConfig::laplace()
                    .likelihood(BayesLikelihood::BernoulliLogit)
                    .prior_from_artifact(vec![0u8; 4], None),
            ))
            .build(),
        "transferred prior",
    );
}
