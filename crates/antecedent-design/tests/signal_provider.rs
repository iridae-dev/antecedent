//! F11: typed candidate-signal providers (native families and an external signal).
//!
//! Oracle (frozen F11 record, enumerated by hand): a binary latent state
//! `theta in {0, 1}` with prior `P(theta = 1) = 1/2`, and one signal `y` with
//! sensitivity `P(y = 1 | theta = 1) = 3/4` and specificity `P(y = 0 | theta = 0) = 3/4`.
//! Then `P(y = 1) = 1/2 * 3/4 + 1/2 * 1/4 = 1/2`, and by Bayes
//! `P(theta = 1 | y = 1) = (1/2 * 3/4) / (1/2) = 3/4` and
//! `P(theta = 1 | y = 0) = (1/2 * 1/4) / (1/2) = 1/4`.

use std::sync::Arc;

use antecedent_core::{
    ExternalCapability, ExternalScientificObject, ExternalTrustState, ProviderObjectIdentity,
    QuantityRole, ScientificQuantity, SignalProviderContract,
};
use antecedent_design::preposterior::DecisionSignal;
use antecedent_design::signal::{
    ExternalDecisionValues, ExternalLaw, ExternalSignal, ExternalSignalBody, NativeBinomialSignal,
    NativeGaussianMeanSignal, PreparedLaw, PreparedSignal, SignalError, SignalLimits,
    SignalProvider, SignalRequest, SignalTrustLabel, SignalUpdateMode,
};
use antecedent_design::{CandidateDesign, DesignCost, SamplingPlan};

fn quantity(role: QuantityRole, id: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: id.into(),
        variable_name: id.into(),
        role,
        units: "dimensionless".into(),
        population_id: "target".into(),
        regime_id: "observational".into(),
        horizon: 0,
        functional_id: "state".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

fn request(candidate: &str, n: u64) -> SignalRequest {
    SignalRequest {
        candidate_id: candidate.into(),
        prior_id: "prior-1".into(),
        state_quantity: quantity(QuantityRole::Outcome, "schema:theta"),
        observation_quantity: quantity(QuantityRole::Outcome, "schema:signal"),
        sample_size: n,
        rng_seed: 7,
        evidence_lineage: vec!["snapshot:a".into(), "snapshot:b".into()],
        conditional_independence: "iid_given_state".into(),
        limits: SignalLimits::default(),
    }
}

fn object(req: &SignalRequest, capabilities: Vec<ExternalCapability>) -> ExternalScientificObject {
    ExternalScientificObject::Signal(SignalProviderContract {
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
    })
}

fn attested(object: &ExternalScientificObject) -> ExternalTrustState {
    ExternalTrustState::attest(object, "lab-qa").unwrap()
}

fn body(req: &SignalRequest, law: ExternalLaw) -> ExternalSignalBody {
    ExternalSignalBody {
        sample_size: req.sample_size,
        state_quantity: req.state_quantity.clone(),
        observation_quantity: req.observation_quantity.clone(),
        law,
    }
}

const FACTOR_CAPS: [ExternalCapability; 3] =
    [ExternalCapability::Sample, ExternalCapability::Factor, ExternalCapability::Update];
const UPDATE_CAPS: [ExternalCapability; 2] =
    [ExternalCapability::Sample, ExternalCapability::Update];

/// The frozen F11 law as an externally computed posterior.
fn frozen_posterior() -> ExternalLaw {
    ExternalLaw::Posterior {
        states: vec![0.0, 1.0],
        statistics: vec![0.0, 1.0],
        predictive: vec![0.5, 0.5],
        // posterior[y][k] = P(theta_k | y): y = 0 -> [3/4, 1/4], y = 1 -> [1/4, 3/4].
        posterior: vec![vec![0.75, 0.25], vec![0.25, 0.75]],
    }
}

fn external_posterior(req: &SignalRequest) -> ExternalSignal {
    let obj = object(req, UPDATE_CAPS.to_vec());
    let trust = attested(&obj);
    ExternalSignal::new(obj, trust, body(req, frozen_posterior())).unwrap()
}

fn likelihood(prepared: PreparedSignal) -> Arc<dyn DecisionSignal<f64>> {
    match prepared.law {
        PreparedLaw::Likelihood(signal) => signal,
        PreparedLaw::DecisionValues(_) => panic!("expected a likelihood"),
    }
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-12
}

/// Bayes over the equally weighted states from a signal's log likelihood.
fn posterior_of_one(signal: &dyn DecisionSignal<f64>, y: f64, n: u64) -> f64 {
    let mut ll = [0.0; 2];
    signal.log_likelihood(y, n, &[0.0, 1.0], &mut ll).unwrap();
    let (w0, w1) = (ll[0].exp(), ll[1].exp());
    w1 / (w0 + w1)
}

#[test]
fn f11_known_truth_external_posterior_reproduces_the_frozen_update() {
    let req = request("study-a", 1);
    let prepared = external_posterior(&req).prepare(&req).unwrap();
    assert_eq!(prepared.receipt.update_mode, SignalUpdateMode::ExternalPosterior);
    assert!(prepared.receipt.update_mode.externally_computed());
    assert_eq!(prepared.bound_states, Some(vec![0.0, 1.0]));
    let signal = likelihood(prepared);
    assert_eq!(signal.finite_support(1), Some(vec![0.0, 1.0]));
    // P(y = 1) = 1/2 over the equally weighted states, from the derived likelihood.
    let mut ll = [0.0; 2];
    signal.log_likelihood(1.0, 1, &[0.0, 1.0], &mut ll).unwrap();
    assert!(close(0.5 * (ll[0].exp() + ll[1].exp()), 0.5));
    assert!(close(posterior_of_one(signal.as_ref(), 1.0, 1), 0.75));
    assert!(close(posterior_of_one(signal.as_ref(), 0.0, 1), 0.25));
}

#[test]
fn f11_native_update_from_a_supplied_likelihood_reproduces_the_frozen_update() {
    let req = request("study-a", 1);
    let law = ExternalLaw::Likelihood {
        states: vec![0.0, 1.0],
        statistics: vec![0.0, 1.0],
        // probabilities[y][k] = P(y | theta_k): y = 0 -> [3/4, 1/4], y = 1 -> [1/4, 3/4].
        probabilities: vec![vec![0.75, 0.25], vec![0.25, 0.75]],
    };
    let obj = object(&req, FACTOR_CAPS.to_vec());
    let trust = attested(&obj);
    let provider = ExternalSignal::new(obj, trust, body(&req, law)).unwrap();
    assert_eq!(provider.update_mode(), SignalUpdateMode::NativeUpdate);
    let prepared = provider.prepare(&req).unwrap();
    // Native update, but the likelihood is still an external, attested supply.
    assert_eq!(prepared.receipt.update_mode, SignalUpdateMode::NativeUpdate);
    assert_eq!(prepared.receipt.trust, SignalTrustLabel::ExternallyAttested);
    let signal = likelihood(prepared);
    assert!(close(posterior_of_one(signal.as_ref(), 1.0, 1), 0.75));
    assert!(close(posterior_of_one(signal.as_ref(), 0.0, 1), 0.25));
}

#[test]
fn f11_externally_computed_decision_values_are_tagged_and_not_native() {
    let req = request("study-a", 1);
    let values = ExternalDecisionValues {
        branch_probabilities: vec![0.5, 0.5],
        action_ids: vec!["guess0".into(), "guess1".into()],
        // E[U | y]: guess0 pays 1 when theta = 0. After y = 0: 3/4, 1/4; after y = 1: 1/4, 3/4.
        values: vec![vec![0.75, 0.25], vec![0.25, 0.75]],
    };
    let obj = object(&req, UPDATE_CAPS.to_vec());
    let trust = attested(&obj);
    let provider =
        ExternalSignal::new(obj, trust, body(&req, ExternalLaw::DecisionValues(values))).unwrap();
    let prepared = provider.prepare(&req).unwrap();
    assert!(matches!(prepared.law, PreparedLaw::DecisionValues(_)));
    assert!(prepared.bound_states.is_none());
    assert_eq!(prepared.receipt.update_mode, SignalUpdateMode::ExternalDecisionValues);
    assert_eq!(prepared.receipt.trust, SignalTrustLabel::ExternallyAttested);
    assert_eq!(prepared.receipt.attestor.as_deref(), Some("lab-qa"));
    assert_eq!(prepared.receipt.request_fingerprint, req.fingerprint());
    assert!(prepared.receipt.provider_identity.is_some());
}

#[test]
fn f11_native_binomial_and_equivalent_external_law_agree_pointwise() {
    // One Bernoulli trial: P(y = 1 | theta) = theta, so at theta in {1/4, 3/4}:
    // y = 0 -> [3/4, 1/4], y = 1 -> [1/4, 3/4].
    let req = request("study-a", 1);
    let native = likelihood(NativeBinomialSignal.prepare(&req).unwrap());
    let law = ExternalLaw::Likelihood {
        states: vec![0.25, 0.75],
        statistics: vec![0.0, 1.0],
        probabilities: vec![vec![0.75, 0.25], vec![0.25, 0.75]],
    };
    let obj = object(&req, FACTOR_CAPS.to_vec());
    let trust = attested(&obj);
    let external = likelihood(
        ExternalSignal::new(obj, trust, body(&req, law)).unwrap().prepare(&req).unwrap(),
    );
    for y in [0.0, 1.0] {
        let (mut a, mut b) = ([0.0; 2], [0.0; 2]);
        native.log_likelihood(y, 1, &[0.25, 0.75], &mut a).unwrap();
        external.log_likelihood(y, 1, &[0.25, 0.75], &mut b).unwrap();
        assert!(close(a[0], b[0]) && close(a[1], b[1]), "y = {y}: {a:?} vs {b:?}");
    }
}

#[test]
fn f11_native_receipts_are_native_and_differ_from_an_equivalent_external_receipt() {
    let req = request("study-a", 1);
    let native = NativeBinomialSignal.prepare(&req).unwrap();
    assert_eq!(native.receipt.trust, SignalTrustLabel::NativeLicensed);
    assert_eq!(native.receipt.family, "binomial");
    assert_eq!(native.receipt.update_mode, SignalUpdateMode::NativeUpdate);
    assert!(native.receipt.provider_identity.is_none());
    let external = external_posterior(&req).prepare(&req).unwrap();
    assert_ne!(native.receipt.identity(), external.receipt.identity());
    let gauss = NativeGaussianMeanSignal::new(0.25).unwrap().prepare(&req).unwrap();
    assert_eq!(gauss.receipt.family, "gaussian_mean");
    assert!(NativeGaussianMeanSignal::new(0.0).is_err());
    // The request's sample size is authoritative for every candidate.
    let design = CandidateDesign::IncreaseSamplingRate(SamplingPlan {
        additional_samples: 99,
        cost: DesignCost::zero(),
        tag: 0,
    });
    assert_eq!(likelihood(gauss).sample_size(&design), Some(1));
}

#[test]
fn f11_predictive_law_without_coherent_posterior_update_refuses() {
    let req = request("study-a", 1);
    // Rows sum to one, but averaging over P(y) gives P(theta = 0) = 0.575, not the prior 1/2.
    let law = ExternalLaw::Posterior {
        states: vec![0.0, 1.0],
        statistics: vec![0.0, 1.0],
        predictive: vec![0.5, 0.5],
        posterior: vec![vec![0.9, 0.1], vec![0.25, 0.75]],
    };
    let obj = object(&req, UPDATE_CAPS.to_vec());
    let trust = attested(&obj);
    let error = ExternalSignal::new(obj, trust, body(&req, law)).unwrap_err();
    assert_eq!(error, SignalError::PosteriorIncoherent);
    assert_eq!(error.to_refusal().detail, "signal_provider.posterior_incoherent");
    // A predictive law that does not normalize also refuses.
    let law = ExternalLaw::Likelihood {
        states: vec![0.0, 1.0],
        statistics: vec![0.0, 1.0],
        probabilities: vec![vec![0.75, 0.25], vec![0.25, 0.5]],
    };
    let obj = object(&req, FACTOR_CAPS.to_vec());
    let trust = attested(&obj);
    let error = ExternalSignal::new(obj, trust, body(&req, law)).unwrap_err();
    assert_eq!(error, SignalError::LawIncoherent);
}

#[test]
fn f11_wrong_candidate_prior_sample_size_or_fingerprint_refuses() {
    let req = request("study-a", 1);
    let provider = external_posterior(&req);
    let mut other = request("study-b", 1);
    other.evidence_lineage = req.evidence_lineage.clone();
    // The request fingerprint covers the candidate, so the exact-request check fires first.
    assert_eq!(provider.prepare(&other).unwrap_err(), SignalError::RequestFingerprintMismatch);
    let mut resized = request("study-a", 2);
    resized.rng_seed = req.rng_seed;
    let error = provider.prepare(&resized).unwrap_err();
    assert_eq!(error, SignalError::RequestFingerprintMismatch);
    assert_eq!(error.to_refusal().detail, "signal_provider.request_fingerprint_mismatch");

    // With a matching fingerprint, a wrong candidate id, prior id and sample size each refuse.
    let mut wrong = object(&req, UPDATE_CAPS.to_vec());
    if let ExternalScientificObject::Signal(contract) = &mut wrong {
        contract.candidate_id = "study-z".into();
    }
    let trust = attested(&wrong);
    let provider = ExternalSignal::new(wrong, trust, body(&req, frozen_posterior())).unwrap();
    assert_eq!(provider.prepare(&req).unwrap_err(), SignalError::CandidateMismatch);

    let mut wrong = object(&req, UPDATE_CAPS.to_vec());
    if let ExternalScientificObject::Signal(contract) = &mut wrong {
        contract.prior_id = "prior-z".into();
    }
    let trust = attested(&wrong);
    let provider = ExternalSignal::new(wrong, trust, body(&req, frozen_posterior())).unwrap();
    assert_eq!(provider.prepare(&req).unwrap_err(), SignalError::PriorMismatch);

    let obj = object(&req, UPDATE_CAPS.to_vec());
    let trust = attested(&obj);
    let mut mismatched = body(&req, frozen_posterior());
    mismatched.sample_size = 5;
    let provider = ExternalSignal::new(obj, trust, mismatched).unwrap();
    let error = provider.prepare(&req).unwrap_err();
    assert_eq!(error, SignalError::SampleSizeMismatch { declared: 5, requested: 1 });
    assert_eq!(error.to_refusal().detail, "signal_provider.sample_size_mismatch");
}

#[test]
fn f11_returned_posterior_quantity_must_align_with_the_request() {
    let req = request("study-a", 1);
    let obj = object(&req, UPDATE_CAPS.to_vec());
    let trust = attested(&obj);
    let mut misaligned = body(&req, frozen_posterior());
    misaligned.state_quantity.units = "log_odds".into();
    let provider = ExternalSignal::new(obj, trust, misaligned).unwrap();
    let error = provider.prepare(&req).unwrap_err();
    assert_eq!(error, SignalError::PosteriorQuantityMismatch);
    assert_eq!(error.to_refusal().detail, "signal_provider.posterior_quantity_mismatch");

    let obj = object(&req, UPDATE_CAPS.to_vec());
    let trust = attested(&obj);
    let mut wrong_observation = body(&req, frozen_posterior());
    wrong_observation.observation_quantity.variable_id = "schema:other".into();
    let error = ExternalSignal::new(obj, trust, wrong_observation).unwrap_err();
    assert_eq!(error, SignalError::ObservationQuantityMismatch);
}

#[test]
fn f11_native_trust_capability_and_object_claims_refuse() {
    let req = request("study-a", 1);
    let obj = object(&req, UPDATE_CAPS.to_vec());
    let error = ExternalSignal::new(
        obj,
        ExternalTrustState::NativeLicensed,
        body(&req, frozen_posterior()),
    )
    .unwrap_err();
    assert_eq!(error, SignalError::TrustNativeClaimed);

    // A supplied likelihood needs the Factor capability.
    let law = ExternalLaw::Likelihood {
        states: vec![0.0, 1.0],
        statistics: vec![0.0, 1.0],
        probabilities: vec![vec![0.75, 0.25], vec![0.25, 0.75]],
    };
    let obj = object(&req, UPDATE_CAPS.to_vec());
    let trust = attested(&obj);
    let error = ExternalSignal::new(obj, trust, body(&req, law)).unwrap_err();
    assert_eq!(error, SignalError::CapabilityMissing(ExternalCapability::Factor));
    assert_eq!(error.to_refusal().capability, Some(ExternalCapability::Factor));

    // A non-signal object is not a signal declaration.
    let utility = ExternalScientificObject::Utility(antecedent_core::UtilityProviderContract {
        identity: ProviderObjectIdentity {
            provider_id: "p".into(),
            object_id: "o".into(),
            version_id: "v".into(),
            snapshot_id: "s".into(),
            request_id: "r".into(),
        },
        inputs: vec![quantity(QuantityRole::Outcome, "schema:x")],
        output: quantity(QuantityRole::Utility, "schema:u"),
        action_ids: vec!["a".into()],
        stochastic: false,
        bounds: None,
        monotonicity: vec![antecedent_core::UtilityMonotonicity::Unspecified],
        capabilities: vec![ExternalCapability::EvaluateUtility],
    });
    let trust = ExternalTrustState::ExternallyAttested { attestor: "x".into() };
    let error = ExternalSignal::new(utility, trust, body(&req, frozen_posterior())).unwrap_err();
    assert_eq!(error, SignalError::InvalidObject);
}

#[test]
fn f11_request_declares_lineage_independence_sample_size_and_limits() {
    let mut req = request("study-a", 1);
    req.evidence_lineage.clear();
    assert_eq!(req.validate(), Err(SignalError::MissingLineage));
    let mut req = request("study-a", 1);
    req.conditional_independence = " ".into();
    assert_eq!(req.validate(), Err(SignalError::ConditionalIndependenceMissing));
    assert_eq!(request("study-a", 0).validate(), Err(SignalError::SampleSizeInvalid));
    let mut req = request("study-a", 100);
    req.limits.max_sample_size = 10;
    assert_eq!(req.validate(), Err(SignalError::ComputationalLimit("sample_size")));
    let mut req = request("study-a", 100);
    req.limits.max_support = 10;
    let error = NativeBinomialSignal.prepare(&req).unwrap_err();
    assert_eq!(error, SignalError::ComputationalLimit("support"));
    assert_eq!(error.to_refusal().detail, "signal_provider.computational_limit");

    // The fingerprint ignores lineage order and tracks every other field.
    let a = request("study-a", 3);
    let mut b = a.clone();
    b.evidence_lineage.reverse();
    assert_eq!(a.fingerprint(), b.fingerprint());
    let mut c = a.clone();
    c.sample_size = 4;
    assert_ne!(a.fingerprint(), c.fingerprint());
    let mut d = a.clone();
    d.rng_seed = 8;
    assert_ne!(a.fingerprint(), d.fingerprint());
}
