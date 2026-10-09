//! Independently replayed, calibration-internal continuous Verma posterior candidate.
//!
//! The original point artifact supplies graph/count/checked general-ID binding.
//! This additive envelope binds the raw-coordinate truncated Beta-kernel prior,
//! bounded seed/request, all joint draws, credible quantiles, covariance and modern
//! diagnostics. Scope additionally requires the original checked point pilot to
//! converge to an interior fit: this reuse binds its identification authority and
//! is an explicit pilot eligibility restriction, not a condition for existence of
//! a Bayesian posterior. The posterior is fitted from counts, never the MLE.
//! A consumer reconstructs every numerical output from bound premises.
//! Digests bind declarations, not the authenticity of supplied observations.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{
    IoError,
    nested_markov_artifact::{
        NestedMarkovArtifactWire, NestedMarkovConsumeLimits, NestedMarkovExpectation,
    },
};
use antecedent_core::{ExecutionContext, IdentityDomain};
use antecedent_estimate::nested_markov_binary::{FitOptions, NestedMarkovInput};
use antecedent_learn::nested_markov_bayesian::{self, BayesianRefusal, Options, Posterior, Prior};
use serde::{Deserialize, Serialize};

/// Frozen Bayesian envelope version (separate from the original point artifact).
pub const VERSION: u32 = 1;
/// Internal posterior artifact marker, never licensed public inference.
pub const FEATURE: &str = "binary_verma_continuous_beta_slice_posterior_internal_v1";
/// Exact declared sampling model.
pub const SAMPLING: &str = "iid_multinomial_integer_counts_correctly_specified_verma_nested_markov";

/// Consumer-owned bounds; the artifact cannot raise any of these.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    /// Encoded bytes, including retained posterior draws.
    pub bytes: usize,
    /// Number of retained chains.
    pub chains: usize,
    /// Warmup sweeps per chain.
    pub warmup: usize,
    /// Retained draws per chain.
    pub draws: usize,
    /// Whole sampler proposal count.
    pub proposals: usize,
    /// Original point-replay bounds.
    pub point: NestedMarkovConsumeLimits,
}
impl Default for Limits {
    fn default() -> Self {
        Self {
            bytes: 16 * 1024 * 1024,
            chains: 4,
            warmup: 4096,
            draws: 4096,
            proposals: 5_000_000,
            point: NestedMarkovConsumeLimits::default(),
        }
    }
}
/// Consumer identity expectation for the full Bayesian premises, plus original data.
#[derive(Clone, Debug, Default)]
pub struct Expectation {
    /// Expected Bayesian premises digest.
    pub premises_digest: Option<String>,
    /// Expected original graph/count identity.
    pub point: NestedMarkovExpectation,
}
/// Posterior candidate envelope. Every standing field is checked on independent replay.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Artifact {
    /// Envelope version.
    pub version: u32,
    /// Required marker.
    pub feature: String,
    /// Frozen numerical method identity.
    pub method: String,
    /// Frozen random stream and chain derivation identity.
    pub rng: String,
    /// Exact eleven-parameter/three-estimand column order.
    pub coordinates: Vec<String>,
    /// Declared sampling assumption.
    pub sampling: String,
    /// Must remain unmeasured; an artifact cannot activate calibration.
    pub calibration: String,
    /// Inferential standing, distinct from nonparametric identification.
    pub inference: String,
    /// Original graph, checked ID, counts and point receipt.
    pub point: NestedMarkovArtifactWire,
    /// Original-coordinate prior, not independent conditional-width priors.
    pub prior: Prior,
    /// Frozen bounded sampler request.
    pub options: Options,
    /// Joint posterior candidate receipt.
    pub posterior: Posterior,
    /// Full Bayesian premises digest (also binds original point/data digests).
    pub premises_digest: String,
}
#[derive(Serialize)]
struct Premises<'a> {
    feature: &'static str,
    method: &'static str,
    rng: &'static str,
    coordinates: &'static [&'static str],
    sampling: &'static str,
    point_premises_digest: &'a str,
    point_data: &'a str,
    prior: &'a Prior,
    options: &'a Options,
}
fn refusal(e: &BayesianRefusal) -> IoError {
    let code = match e {
        BayesianRefusal::Invalid(_) => antecedent_core::reason_code!("invalid_argument"),
        BayesianRefusal::Budget(_) => antecedent_core::reason_code!("transport_budget_cancel"),
        BayesianRefusal::Numerical(_) | BayesianRefusal::Nonconvergence => {
            antecedent_core::reason_code!("transport_numerical_failure")
        }
    };
    IoError::Refused { code, message: format!("nested_markov.bayesian: {e}") }
}
fn invalid(message: &str) -> IoError {
    IoError::Refused {
        code: antecedent_core::reason_code!("invalid_argument"),
        message: format!("nested_markov.bayesian_artifact: {message}"),
    }
}
fn integer_counts(point: &NestedMarkovArtifactWire) -> Result<[u64; 16], IoError> {
    if point.regimes.len() != 1 || point.regimes[0].cells.len() != 16 {
        return Err(invalid("one sixteen-cell observational regime required"));
    }
    let mut out = [0; 16];
    for (i, n) in point.regimes[0].cells.iter().enumerate() {
        if !n.is_finite() || *n <= 0.0 || *n > 1_000_000_000.0 || n.fract() != 0.0 {
            return Err(invalid("positive integer multinomial counts required"));
        }
        #[allow(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "finite positive bounded integer checked above"
        )]
        {
            out[i] = *n as u64;
        }
    }
    Ok(out)
}
impl Artifact {
    /// Bind a checked point pilot and fit the independent continuous posterior.
    /// # Errors
    /// Original graph/ID/fit refusal or Bayesian declaration/budget/diagnostic refusal.
    pub fn build(
        input: &NestedMarkovInput,
        fit_options: &FitOptions,
        prior: Prior,
        options: Options,
        ctx: &ExecutionContext,
    ) -> Result<Self, IoError> {
        let (point, _) = NestedMarkovArtifactWire::build(input, fit_options, ctx)?;
        let posterior =
            nested_markov_bayesian::fit(&integer_counts(&point)?, &prior, &options, ctx)
                .map_err(|e| refusal(&e))?;
        let mut artifact = Self {
            version: VERSION,
            feature: FEATURE.into(),
            method: nested_markov_bayesian::METHOD.into(),
            rng: nested_markov_bayesian::RNG.into(),
            coordinates: nested_markov_bayesian::COORDINATES.iter().map(|s| (*s).into()).collect(),
            sampling: SAMPLING.into(),
            calibration: "unmeasured".into(),
            inference: "posterior_candidate_withheld_calibration_unmeasured".into(),
            point,
            prior,
            options,
            posterior,
            premises_digest: String::new(),
        };
        artifact.premises_digest = artifact.expected_premises_digest()?;
        Ok(artifact)
    }
    /// Recompute premises identity; this does not certify any declaration.
    /// # Errors
    /// Encoding failure.
    pub fn expected_premises_digest(&self) -> Result<String, IoError> {
        let premises = Premises {
            feature: FEATURE,
            method: nested_markov_bayesian::METHOD,
            rng: nested_markov_bayesian::RNG,
            coordinates: &nested_markov_bayesian::COORDINATES,
            sampling: SAMPLING,
            point_premises_digest: &self.point.premises_digest,
            point_data: &self.point.data_digest,
            prior: &self.prior,
            options: &self.options,
        };
        Ok(crate::identity::digest_wire(IdentityDomain::TransportCertificate, &premises)?.to_hex())
    }
    /// Export internal candidate evidence, never a public interval result.
    /// # Errors
    /// Encoding failure.
    pub fn export(&self) -> Result<Vec<u8>, IoError> {
        crate::to_cbor(self)
    }
    /// Independently replay checked identification/fit and the ENTIRE posterior.
    /// # Errors
    /// Consumer limit, identity, semantic standing or receipt mismatch; engine refusal.
    pub fn consume(
        bytes: &[u8],
        expected: &Expectation,
        limits: Limits,
        ctx: &ExecutionContext,
    ) -> Result<Self, IoError> {
        if bytes.len() > limits.bytes {
            return Err(invalid("consumer byte limit"));
        }
        let wire: Self = crate::from_cbor(bytes)?;
        if wire.version != VERSION
            || wire.feature != FEATURE
            || wire.method != nested_markov_bayesian::METHOD
            || wire.rng != nested_markov_bayesian::RNG
            || wire.coordinates != nested_markov_bayesian::COORDINATES
            || wire.sampling != SAMPLING
            || wire.calibration != "unmeasured"
            || wire.inference != "posterior_candidate_withheld_calibration_unmeasured"
        {
            return Err(invalid("unsupported format or inferential standing"));
        }
        let o = &wire.options;
        if o.chains > limits.chains
            || o.warmup > limits.warmup
            || o.draws > limits.draws
            || o.max_proposals > limits.proposals
            || wire.posterior.samples.len() != o.chains.saturating_mul(o.draws)
            || wire.posterior.covariance.len() != 196
            || wire.posterior.diagnostics.len() != 14
        {
            return Err(invalid("consumer sampler or receipt shape limit"));
        }
        if wire.expected_premises_digest()? != wire.premises_digest
            || expected.premises_digest.as_ref().is_some_and(|d| *d != wire.premises_digest)
        {
            return Err(invalid("Bayesian premises identity mismatch"));
        }
        let (point, _) = NestedMarkovArtifactWire::consume_expecting(
            &wire.point.export()?,
            &expected.point,
            limits.point,
            ctx,
        )?;
        let replay =
            nested_markov_bayesian::fit(&integer_counts(&point)?, &wire.prior, &wire.options, ctx)
                .map_err(|e| refusal(&e))?;
        if crate::to_cbor(&replay)? != crate::to_cbor(&wire.posterior)? {
            return Err(invalid("posterior receipt does not replay"));
        }
        Ok(wire)
    }
}
