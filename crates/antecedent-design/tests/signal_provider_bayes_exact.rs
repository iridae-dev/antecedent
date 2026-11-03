//! External candidate signals against an independent prior-times-likelihood update.
//!
//! Model, enumerated here: the state `theta` is uniform on `{1/5, 1/2, 4/5}` and the signal
//! `y` in `{0, 1, 2}` is Binomial(2, theta): `P(y | theta) = C(2, y) theta^y (1 - theta)^(2 - y)`.
//! The predictive law `P(y) = (1/3) sum_k P(y | theta_k)` and the posterior
//! `P(theta_k | y) = P(y | theta_k) / (3 P(y))` are computed from that generative model by Bayes.
//!
//! Two supply modes are exercised, each through `ExternalSignal::prepare`:
//! * an externally computed posterior (predictive and posterior above): the likelihood the
//!   library derives must be the generating `P(y | theta_k)`, so that a fresh Bayes update over the
//!   uniform draws with that derived likelihood returns the independent posterior;
//! * a supplied likelihood table: the native update of that table over the uniform draws returns
//!   the independent posterior and predictive.
//!
//! Tolerance 1e-12: a few float operations per entry.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_contract::{
    ExternalCapability, ExternalScientificObject, ExternalTrustState, ProviderObjectIdentity,
    ScientificQuantity, SignalProviderContract,
};
use antecedent_design::preposterior::DecisionSignal;
use antecedent_design::signal::{
    ExternalLaw, ExternalSignal, ExternalSignalBody, PreparedLaw, SignalLimits, SignalProvider,
    SignalRequest,
};

mod common;

const STATES: [f64; 3] = [0.2, 0.5, 0.8];
const TOLERANCE: f64 = 1e-12;

fn close(actual: f64, expected: f64, what: &str) {
    assert!((actual - expected).abs() <= TOLERANCE, "{what}: {actual} vs {expected}");
}

fn quantity(id: &str) -> ScientificQuantity {
    ScientificQuantity {
        units: "dimensionless".into(),
        functional_id: "state".into(),
        ..common::quantity(id)
    }
}

fn request() -> SignalRequest {
    SignalRequest {
        candidate_id: "study-a".into(),
        prior_id: "prior-1".into(),
        state_quantity: quantity("schema:theta"),
        observation_quantity: quantity("schema:signal"),
        sample_size: 1,
        rng_seed: 7,
        evidence_lineage: vec!["snapshot:a".into()],
        conditional_independence: "iid_given_state".into(),
        limits: SignalLimits::default(),
    }
}

/// Binomial(2, theta) mass at `y`, written out.
fn binomial(y: usize, theta: f64) -> f64 {
    match y {
        0 => (1.0 - theta) * (1.0 - theta),
        1 => 2.0 * theta * (1.0 - theta),
        _ => theta * theta,
    }
}

fn predictive(y: usize) -> f64 {
    STATES.iter().map(|t| binomial(y, *t)).sum::<f64>() / 3.0
}

/// Independent Bayes: `P(theta_k | y)` under the uniform prior.
fn posterior(y: usize) -> Vec<f64> {
    STATES.iter().map(|t| binomial(y, *t) / 3.0 / predictive(y)).collect()
}

fn signal_over(
    capabilities: Vec<ExternalCapability>,
    law: ExternalLaw,
) -> Arc<dyn DecisionSignal<f64>> {
    let req = request();
    let object = ExternalScientificObject::Signal(SignalProviderContract {
        identity: ProviderObjectIdentity {
            provider_id: "lab".into(),
            object_id: "signal".into(),
            version_id: "v1".into(),
            snapshot_id: "snap".into(),
            request_id: req.fingerprint(),
        },
        candidate_id: req.candidate_id.clone(),
        prior_id: req.prior_id.clone(),
        observation: req.observation_quantity.clone(),
        capabilities,
    });
    let trust = ExternalTrustState::attest(&object, "lab-qa").unwrap();
    let body = ExternalSignalBody {
        sample_size: req.sample_size,
        state_quantity: req.state_quantity.clone(),
        observation_quantity: req.observation_quantity.clone(),
        law,
    };
    let provider = ExternalSignal::new(object, trust, body).unwrap();
    match provider.prepare(&req).unwrap().law {
        PreparedLaw::Likelihood(signal) => signal,
        PreparedLaw::DecisionValues(_) => panic!("expected a likelihood"),
    }
}

/// Bayes over the uniform draws from a signal's own log likelihood.
fn derived_posterior(signal: &dyn DecisionSignal<f64>, y: usize) -> (f64, Vec<f64>) {
    let mut ll = [0.0; 3];
    signal.log_likelihood(y as f64, 1, &STATES, &mut ll).unwrap();
    let likelihood: Vec<f64> = ll.iter().map(|v| v.exp()).collect();
    let predictive = likelihood.iter().sum::<f64>() / 3.0;
    (predictive, likelihood.iter().map(|l| l / 3.0 / predictive).collect())
}

fn check_against_bayes(signal: &dyn DecisionSignal<f64>) {
    for y in 0..3 {
        let (derived_predictive, derived) = derived_posterior(signal, y);
        close(derived_predictive, predictive(y), &format!("P(y = {y})"));
        for (k, (got, want)) in derived.iter().zip(posterior(y)).enumerate() {
            close(*got, want, &format!("P(theta_{k} | y = {y})"));
        }
        // And the derived likelihood is the generating binomial mass itself.
        let mut ll = [0.0; 3];
        signal.log_likelihood(y as f64, 1, &STATES, &mut ll).unwrap();
        for (k, theta) in STATES.iter().enumerate() {
            close(ll[k].exp(), binomial(y, *theta), &format!("P(y = {y} | theta_{k})"));
        }
    }
}

#[test]
fn external_posterior_derives_the_generating_likelihood_and_the_independent_posterior() {
    let law = ExternalLaw::Posterior {
        states: STATES.to_vec(),
        statistics: vec![0.0, 1.0, 2.0],
        predictive: (0..3).map(predictive).collect(),
        posterior: (0..3).map(posterior).collect(),
    };
    let signal = signal_over(
        vec![ExternalCapability::Sample, ExternalCapability::Update],
        law,
    );
    check_against_bayes(signal.as_ref());
    // Hand value: P(y = 2) = (1/25 + 1/4 + 16/25)/3 = 0.93/3 = 0.31, and
    // P(theta = 4/5 | y = 2) = (16/25) / (3 * 0.31).
    close(predictive(2), 0.31, "hand P(y = 2)");
    close(posterior(2)[2], 0.64 / 0.93, "hand posterior");
}

#[test]
fn a_supplied_likelihood_table_updates_to_the_independent_posterior() {
    let law = ExternalLaw::Likelihood {
        states: STATES.to_vec(),
        statistics: vec![0.0, 1.0, 2.0],
        probabilities: (0..3)
            .map(|y| STATES.iter().map(|t| binomial(y, *t)).collect())
            .collect(),
    };
    let signal = signal_over(
        vec![ExternalCapability::Sample, ExternalCapability::Factor, ExternalCapability::Update],
        law,
    );
    check_against_bayes(signal.as_ref());
}
