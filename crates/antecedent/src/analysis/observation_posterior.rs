//! Posterior of a declared model after new evidence.
//!
//! Two declared models are supported, each answered exactly:
//!
//! - a finite hypothesis space with a likelihood for each observation level
//!   ([`FiniteBayesInput`]); observations are exchangeable and conditionally
//!   independent given the hypothesis;
//! - a Beta-Bernoulli model (conjugate update; the posterior is
//!   `Beta(alpha + successes, beta + failures)` and the decision probability
//!   is the predictive probability of the next success).
//!
//! The value is exact given the declared prior and likelihood. Whether those
//! were written down or fitted to a sample is stated by the caller as an
//! [`InferenceBasis`] and travels in the claim: a posterior from a
//! sample-fitted model is a labelled point value, and neither an interval nor
//! a calibration claim is made for it ([`ObservationClaim::interval`],
//! [`ObservationClaim::calibrated_probability`] refuse).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::InferenceBasis;
use antecedent_core::ExecutionContext;

use super::observation_query::{
    ObservationClaim, ObservationQueryRefusal, ObservationQueryWire, seal_declared_query,
};

/// A declared finite hypothesis space with its likelihood.
#[derive(Clone, Debug, PartialEq)]
pub struct FiniteBayesInput {
    /// Hypothesis names.
    pub hypotheses: Vec<String>,
    /// Prior probabilities, aligned with `hypotheses`.
    pub prior: Vec<f64>,
    /// Observation level names.
    pub observation_levels: Vec<String>,
    /// Row-major `[hypothesis][observation]` likelihood.
    pub likelihood: Vec<f64>,
}

/// The posterior of a declared finite model after observations (exchangeable,
/// conditionally independent given the hypothesis).
///
/// `basis` states whether the prior and likelihood were written down
/// ([`InferenceBasis::DeclaredModel`]) or fitted to a sample
/// ([`InferenceBasis::SampleFittedPointOnly`]); the value is exact given the
/// model either way, and the reading says which statement that is.
///
/// # Errors
/// A malformed declaration, an unknown observation level, observations of
/// probability zero under every hypothesis, or cancellation.
pub fn finite_posterior(
    model: &FiniteBayesInput,
    observations: &[String],
    basis: InferenceBasis,
    ctx: &ExecutionContext,
) -> Result<ObservationClaim, ObservationQueryRefusal> {
    seal_declared_query(
        ObservationQueryWire::FinitePosterior {
            hypotheses: model.hypotheses.clone(),
            prior_bits: model.prior.iter().map(|p| p.to_bits()).collect(),
            observation_levels: model.observation_levels.clone(),
            likelihood_bits: model.likelihood.iter().map(|p| p.to_bits()).collect(),
            observations: observations.to_vec(),
        },
        basis,
        ctx,
    )
}

/// The posterior of a Beta-Bernoulli model after `successes` and `failures`,
/// with the predictive probability of the next success.
///
/// # Errors
/// Non-positive shapes, counts above `2^53`, or cancellation.
pub fn conjugate_posterior(
    alpha: f64,
    beta: f64,
    successes: u64,
    failures: u64,
    basis: InferenceBasis,
    ctx: &ExecutionContext,
) -> Result<ObservationClaim, ObservationQueryRefusal> {
    seal_declared_query(
        ObservationQueryWire::ConjugatePosterior {
            alpha_bits: alpha.to_bits(),
            beta_bits: beta.to_bits(),
            successes,
            failures,
        },
        basis,
        ctx,
    )
}
