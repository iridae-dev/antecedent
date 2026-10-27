//! Replay artifact for the treatment-propensity nuisance product of a prepared study.
//!
//! Format version 1, kind marker `array_fit_replay_v1`. The request carries the design the
//! product was fitted on; the stored result is the posterior summary of that nuisance, labelled
//! with its target, the meaning of its uncertainty and the premises declared. It is unlicensed
//! with calibration unmeasured, and a consumer refuses any other value.
//!
//! The product is a posterior of the treatment model only. Its scores are in-sample summaries,
//! not out-of-fold effect scores, so this artifact is a different kind from the cross-fitted
//! effect-score table: that table's loader does not read it, and this loader reads nothing else.
//!
//! A consumer trusts nothing in the result. It re-validates the stored request, refits the
//! posterior from the embedded design and accepts only a result identical, bit for bit, to the
//! stored one. A digest guards the stored request; a changed array, setting or stored value is
//! refused even when the digest is re-sealed, because the replayed result then differs.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::IdentityDomain;
use antecedent_estimate::EstimationError;
use antecedent_estimate::penalized_propensity_posterior::{
    PropensityCoefficientPrior, StaticPropensityDesignRef,
};
use antecedent_estimate::temporal_penalized_score::{
    BayesianTemporalPropensity, FrequentistTemporalPropensity, TemporalPropensityDesignRef,
    TrajectoryDependence,
};
use serde::{Deserialize, Serialize};

use crate::IoError;

/// The artifact format this reader writes and accepts.
pub const ARRAY_FIT_ARTIFACT_VERSION: u32 = 1;
/// The feature marker of the accepted format.
pub const ARRAY_FIT_ARTIFACT_FEATURE: &str = "array_fit_replay_v1";
/// Licensing state of this product.
pub const ARRAY_FIT_LICENSING: &str = "unlicensed";
/// Calibration state of this product.
pub const ARRAY_FIT_CALIBRATION: &str = "unmeasured";

/// Nuisance posterior of a static logistic propensity.
pub const ID_BAYESIAN_PROPENSITY: &str = "bayesian.propensity.coefficient_posterior";
/// Ridge-penalized temporal propensity score.
pub const ID_TEMPORAL_PROPENSITY: &str = "temporal.penalized_propensity";
/// Laplace nuisance posterior of the temporal propensity.
pub const ID_BAYESIAN_TEMPORAL_PROPENSITY: &str = "bayesian.temporal.propensity_posterior";

/// Why an array-fit artifact was refused.
#[derive(Clone, Debug, Eq, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum ArrayFitError {
    /// The caller's real native admission pool refused an allocation or canceled work.
    #[error(transparent)]
    Admission(#[from] antecedent_core::execution::AdmissionError),
    /// The feature marker, a field spelling or the result shape is not this format's.
    #[error("unsupported semantics: {0}")]
    UnsupportedSemantics(&'static str),
    /// A stored collection exceeds the consumer's bound.
    #[error("consumer limit exceeded: {0}")]
    LimitsExceeded(&'static str),
    /// The estimator refused the stored request with its own typed reason.
    #[error("{detail}: {message}")]
    Refused {
        /// Registered reason code.
        code: &'static str,
        /// Namespaced detail, such as `white_innovation.multi_step`.
        detail: String,
        /// What was refused and the next attempt.
        message: String,
    },
    /// The recomputed result differs from the stored one.
    #[error("the stored result does not replay")]
    ResultMismatch,
    /// The request digest does not match the stored request.
    #[error("request digest mismatch")]
    RequestMismatch,
    /// The bytes do not decode.
    #[error("artifact does not decode: {0}")]
    Undecodable(String),
}

impl ArrayFitError {
    /// `(reason code, detail)` of this refusal.
    #[must_use]
    pub fn refusal(&self) -> (&'static str, String) {
        let invalid = antecedent_core::reason_code!("invalid_argument");
        match self {
            Self::Admission(_) => (invalid, "array_fit.operational_admission".to_owned()),
            Self::UnsupportedSemantics(_) => (
                antecedent_core::reason_code!("route_not_supported"),
                "array_fit.unsupported_semantics".to_owned(),
            ),
            Self::LimitsExceeded(_) => (invalid, "array_fit.consumer_limit_exceeded".to_owned()),
            Self::Refused { code, detail, .. } => (*code, detail.clone()),
            Self::ResultMismatch => (invalid, "array_fit.result_replay_mismatch".to_owned()),
            Self::RequestMismatch => (invalid, "array_fit.request_digest_mismatch".to_owned()),
            Self::Undecodable(_) => (invalid, "array_fit.undecodable".to_owned()),
        }
    }
}

impl From<ArrayFitError> for IoError {
    fn from(error: ArrayFitError) -> Self {
        let (code, detail) = error.refusal();
        Self::Refused { code, message: format!("{detail}: array fit artifact: {error}") }
    }
}

impl From<EstimationError> for ArrayFitError {
    fn from(error: EstimationError) -> Self {
        match error {
            EstimationError::Refused { code, message }
            | EstimationError::RefusedWithFields { code, message, .. } => {
                Self::Refused {
                    code,
                    detail: message.detail().unwrap_or("array_fit.refused").to_owned(),
                    message: message.explanation().to_owned(),
                }
            }
            other => Self::Refused {
                code: antecedent_core::reason_code!("invalid_argument"),
                detail: "array_fit.fit_refused".to_owned(),
                message: other.to_string(),
            },
        }
    }
}

/// Bounds a consumer imposes. Nothing the artifact stores raises them.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ArrayFitConsumeLimits {
    /// Most real numbers stored in the request.
    pub max_numbers: usize,
    /// Most posterior draws requested.
    pub max_draws: usize,
    /// Most design columns.
    pub max_columns: usize,
}

impl Default for ArrayFitConsumeLimits {
    fn default() -> Self {
        Self { max_numbers: 4_000_000, max_draws: 200_000, max_columns: 1_024 }
    }
}

/// Static logistic propensity design and the Gaussian coefficient prior.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PropensityPosteriorWire {
    /// Column-major design; column 0 is an intercept of ones.
    pub x_colmajor: Vec<f64>,
    /// Treatment coded 0 or 1, one per row.
    pub treatment: Vec<f64>,
    /// Rows.
    pub nrows: usize,
    /// Standard deviation of the Gaussian prior on every coefficient.
    pub prior_sd: f64,
    /// Posterior draws.
    pub draws: usize,
    /// Seed of the draw stream.
    pub seed: u64,
    /// Design rows `[1, x1, ...]` whose score summaries are reported.
    pub rows: Vec<Vec<f64>>,
    /// Positivity band `(clip, 1 - clip)` of the outside-band probability.
    pub clip: f64,
}

/// Lag-aligned treatment design of independent subjects: intercept, then lags; treatment 0 or 1.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PropensityDesignWire {
    /// Column-major design; column 0 is an intercept of ones.
    pub x_colmajor: Vec<f64>,
    /// Treatment coded 0 or 1, one per row.
    pub treatment: Vec<f64>,
    /// Rows, one independent subject each.
    pub nrows: usize,
    /// Lag histories (one value per slope column) whose scores are reported.
    pub histories: Vec<Vec<f64>>,
}

/// Ridge-penalized temporal propensity score.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PenalizedScoreWire {
    /// The design.
    pub design: PropensityDesignWire,
    /// Ridge penalty on the slopes; the intercept is unpenalized.
    pub penalty: f64,
}

/// Laplace nuisance posterior of the temporal propensity.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PropensityScorePosteriorWire {
    /// The design.
    pub design: PropensityDesignWire,
    /// Prior standard deviation of the intercept.
    pub intercept_sd: f64,
    /// Prior standard deviation of each slope.
    pub slope_sd: f64,
    /// Posterior draws.
    pub draws: usize,
    /// Seed of the draw stream.
    pub seed: u64,
}

/// One estimator and every array it reads.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ArrayFitRequestWire {
    /// `bayesian.propensity.coefficient_posterior`.
    PropensityPosterior(PropensityPosteriorWire),
    /// `temporal.penalized_propensity`.
    TemporalPropensity(PenalizedScoreWire),
    /// `bayesian.temporal.propensity_posterior`.
    TemporalPropensityPosterior(PropensityScorePosteriorWire),
}

/// A named real of a result.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NamedValueWire {
    /// Stable name, such as `effect` or `row0.mean`.
    pub name: String,
    /// The value.
    pub value: f64,
}

/// The stored result of one fit.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ArrayFitResultWire {
    /// Estimator id.
    pub estimator_id: String,
    /// What the numbers are about.
    pub target: String,
    /// What the uncertainty is: a standard error, a posterior, or a nuisance posterior.
    pub uncertainty: String,
    /// Always `unlicensed`.
    pub licensing: String,
    /// Always `unmeasured`.
    pub calibration: String,
    /// Named reals, in a fixed order.
    pub values: Vec<NamedValueWire>,
    /// Posterior draws of the target; empty when the result is not a posterior of the target.
    pub draws: Vec<f64>,
    /// Premises the caller declares and the estimator does not establish.
    pub assumptions: Vec<String>,
    /// Diagnostics the fit reported.
    pub diagnostics: Vec<String>,
}

/// Versioned array-fit result with the request needed for replay.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ArrayFitArtifactWire {
    /// Independent format version.
    pub version: u32,
    /// Required feature marker.
    pub required_features: Vec<String>,
    /// The request, arrays included.
    pub request: ArrayFitRequestWire,
    /// The stored result.
    pub result: ArrayFitResultWire,
    /// Digest of the canonical request.
    pub request_digest: String,
}

/// Bounded framing recognition only: no request/result arrays or fit are materialized.
/// A recognized header grants no semantic acceptance; use the independent consumer.
///
/// # Errors
/// Malformed CBOR; interpretation and replay belong to the consumer.
pub fn inspect_array_fit_header(bytes: &[u8]) -> Result<bool, IoError> {
    #[derive(Deserialize)]
    struct Header {
        #[serde(default)]
        required_features: FeatureMarker,
    }
    let header: Header = crate::convert::from_cbor(bytes)?;
    Ok(header.required_features.0)
}

#[derive(Default)]
struct FeatureMarker(bool);

impl<'de> Deserialize<'de> for FeatureMarker {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self,D::Error> {
        struct Marker;
        impl<'de> serde::de::Visitor<'de> for Marker {
            type Value=FeatureMarker;
            fn expecting(&self,f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str("a required-features discriminator") }
            fn visit_seq<A: serde::de::SeqAccess<'de>>(self,mut seq:A) -> Result<Self::Value,A::Error> {
                // The established discriminator is the first feature; later features
                // remain the independent consumer's validation responsibility.
                let first=seq.next_element::<ExactMarker>()?.is_some_and(|marker|marker.0);
                while seq.next_element::<serde::de::IgnoredAny>()?.is_some() {}
                Ok(FeatureMarker(first))
            }
        }
        deserializer.deserialize_seq(Marker)
    }
}

struct ExactMarker(bool);
impl<'de> Deserialize<'de> for ExactMarker {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self,D::Error> {
        struct Marker;
        impl serde::de::Visitor<'_> for Marker {
            type Value=ExactMarker;
            fn expecting(&self,f:&mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str("a feature name") }
            fn visit_str<E:serde::de::Error>(self,value:&str) -> Result<Self::Value,E> {
                Ok(ExactMarker(value==ARRAY_FIT_ARTIFACT_FEATURE))
            }
        }
        deserializer.deserialize_str(Marker)
    }
}

struct Meta {
    id: &'static str,
    target: &'static str,
    uncertainty: &'static str,
}

type Fit = Result<ArrayFitResultWire, ArrayFitError>;

fn value(name: &str, value: f64) -> NamedValueWire {
    NamedValueWire { name: name.to_owned(), value }
}

fn indexed(prefix: &str, index: usize, field: &str, v: f64) -> NamedValueWire {
    value(&format!("{prefix}{index}.{field}"), v)
}

fn declared(items: &[&str]) -> Vec<String> {
    items.iter().map(|item| (*item).to_owned()).collect()
}

fn finish(
    meta: &Meta,
    values: Vec<NamedValueWire>,
    draws: Vec<f64>,
    assumptions: Vec<String>,
    diagnostics: Vec<String>,
) -> ArrayFitResultWire {
    ArrayFitResultWire {
        estimator_id: meta.id.to_owned(),
        target: meta.target.to_owned(),
        uncertainty: meta.uncertainty.to_owned(),
        licensing: ARRAY_FIT_LICENSING.to_owned(),
        calibration: ARRAY_FIT_CALIBRATION.to_owned(),
        values,
        draws,
        assumptions,
        diagnostics,
    }
}

fn run_propensity(wire: &PropensityPosteriorWire) -> Fit {
    let design = StaticPropensityDesignRef::try_new(
        &wire.x_colmajor,
        &wire.treatment,
        wire.nrows,
    )?;
    let posterior =
        PropensityCoefficientPrior::new(wire.prior_sd, wire.draws, wire.seed)?.fit_borrowed(&design)?;
    let mut values = vec![value("prior_sd", posterior.prior_sd)];
    for (j, mode) in posterior.mode.iter().enumerate() {
        values.push(value(&format!("mode.{j}"), *mode));
    }
    for (i, row) in wire.rows.iter().enumerate() {
        let score = posterior.score_at(row, wire.clip)?;
        values.push(indexed("row", i, "mean", score.mean));
        values.push(indexed("row", i, "at_mode", score.at_mode));
        values.push(indexed("row", i, "sd", score.sd));
        values.push(indexed("row", i, "outside_band", score.outside_band));
    }
    let meta = Meta {
        id: ID_BAYESIAN_PROPENSITY,
        target: "propensity_score_nuisance",
        uncertainty: "nuisance_posterior_laplace_not_an_effect_posterior",
    };
    let assumptions = declared(&[
        "Gaussian prior on every coefficient, intercept included",
        "a posterior of the propensity is not a posterior of any weighting effect",
    ]);
    Ok(finish(&meta, values, Vec::new(), assumptions, Vec::new()))
}

const SCORE_PREMISES: [&str; 3] = [
    "each row is one independent subject at one decision time",
    "a nuisance score: not a posterior or an interval of any effect",
    "in-sample summary of the treatment model, not an out-of-fold effect score",
];

fn score_design(wire: &PropensityDesignWire) -> Result<TemporalPropensityDesignRef<'_>, ArrayFitError> {
    Ok(TemporalPropensityDesignRef::try_new(
        &wire.x_colmajor,
        &wire.treatment,
        wire.nrows,
        TrajectoryDependence::IndependentSubjects,
    )?)
}

fn run_score(wire: &PenalizedScoreWire) -> Fit {
    let design = score_design(&wire.design)?;
    let score = FrequentistTemporalPropensity::new(wire.penalty)?.fit_borrowed(&design)?;
    let mut values = vec![value("penalty", score.penalty)];
    for (j, coefficient) in score.coefficients.iter().enumerate() {
        values.push(value(&format!("coefficient.{j}"), *coefficient));
    }
    for (i, history) in wire.design.histories.iter().enumerate() {
        values.push(indexed("score", i, "at_mode", score.at(history)?));
    }
    let meta = Meta {
        id: ID_TEMPORAL_PROPENSITY,
        target: "propensity_score_nuisance",
        uncertainty: "penalized_mode_no_uncertainty",
    };
    Ok(finish(&meta, values, Vec::new(), declared(&SCORE_PREMISES), Vec::new()))
}

fn run_score_posterior(wire: &PropensityScorePosteriorWire) -> Fit {
    let design = score_design(&wire.design)?;
    let posterior =
        BayesianTemporalPropensity::new(wire.intercept_sd, wire.slope_sd, wire.draws, wire.seed)?
            .fit_borrowed(&design)?;
    let mut values =
        vec![value("intercept_sd", posterior.intercept_sd), value("slope_sd", posterior.slope_sd)];
    for (j, mode) in posterior.mode.iter().enumerate() {
        values.push(value(&format!("mode.{j}"), *mode));
    }
    for (i, history) in wire.design.histories.iter().enumerate() {
        values.push(indexed("score", i, "mean", posterior.at(history)?));
        values.push(indexed("score", i, "at_mode", posterior.at_mode(history)?));
    }
    let meta = Meta {
        id: ID_BAYESIAN_TEMPORAL_PROPENSITY,
        target: "propensity_score_nuisance",
        uncertainty: "nuisance_posterior_laplace_not_an_effect_posterior",
    };
    Ok(finish(&meta, values, Vec::new(), declared(&SCORE_PREMISES), Vec::new()))
}

fn score_numbers(design: &PropensityDesignWire) -> usize {
    let histories: usize = design.histories.iter().map(Vec::len).sum();
    design.x_colmajor.len() + design.treatment.len() + histories
}

fn design_columns(x_len: usize, nrows: usize) -> usize {
    x_len.checked_div(nrows).unwrap_or(0)
}

#[cfg(test)]
std::thread_local! { static ARRAY_EVALUATIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }

impl ArrayFitRequestWire {
    /// The estimator id this request runs.
    #[must_use]
    pub const fn estimator_id(&self) -> &'static str {
        match self {
            Self::PropensityPosterior(_) => ID_BAYESIAN_PROPENSITY,
            Self::TemporalPropensity(_) => ID_TEMPORAL_PROPENSITY,
            Self::TemporalPropensityPosterior(_) => ID_BAYESIAN_TEMPORAL_PROPENSITY,
        }
    }

    fn numbers(&self) -> usize {
        match self {
            Self::PropensityPosterior(p) => {
                let rows: usize = p.rows.iter().map(Vec::len).sum();
                p.x_colmajor.len() + p.treatment.len() + rows
            }
            Self::TemporalPropensity(p) => score_numbers(&p.design),
            Self::TemporalPropensityPosterior(p) => score_numbers(&p.design),
        }
    }

    fn draws(&self) -> usize {
        match self {
            Self::PropensityPosterior(p) => p.draws,
            Self::TemporalPropensityPosterior(p) => p.draws,
            Self::TemporalPropensity(_) => 0,
        }
    }

    fn columns(&self) -> usize {
        match self {
            Self::PropensityPosterior(p) => design_columns(p.x_colmajor.len(), p.nrows),
            Self::TemporalPropensity(p) => {
                design_columns(p.design.x_colmajor.len(), p.design.nrows)
            }
            Self::TemporalPropensityPosterior(p) => {
                design_columns(p.design.x_colmajor.len(), p.design.nrows)
            }
        }
    }

    fn check_limits(&self, limits: &ArrayFitConsumeLimits) -> Result<(), ArrayFitError> {
        if self.numbers() > limits.max_numbers {
            return Err(ArrayFitError::LimitsExceeded("numbers"));
        }
        if self.draws() > limits.max_draws {
            return Err(ArrayFitError::LimitsExceeded("draws"));
        }
        if self.columns() > limits.max_columns {
            return Err(ArrayFitError::LimitsExceeded("columns"));
        }
        Ok(())
    }

    fn evaluate(&self) -> Fit {
        #[cfg(test)]
        ARRAY_EVALUATIONS.with(|count|count.set(count.get()+1));
        match self {
            Self::PropensityPosterior(p) => run_propensity(p),
            Self::TemporalPropensity(p) => run_score(p),
            Self::TemporalPropensityPosterior(p) => run_score_posterior(p),
        }
    }

    fn digest(&self) -> Result<String, IoError> {
        crate::digest_hex(IdentityDomain::Claim, self)
    }
}

impl ArrayFitArtifactWire {
    /// Fit the request and seal the result with the request digest.
    ///
    /// # Errors
    /// The estimator's typed refusal (`white_innovation.*`, `array_fit.*`, ...), or a request
    /// above the default consumer bounds.
    pub fn seal(request: ArrayFitRequestWire) -> Result<Self, ArrayFitError> {
        request.check_limits(&ArrayFitConsumeLimits::default())?;
        let result = request.evaluate()?;
        let undecodable = |e: IoError| ArrayFitError::Undecodable(e.to_string());
        let wire = Self {
            version: ARRAY_FIT_ARTIFACT_VERSION,
            required_features: vec![ARRAY_FIT_ARTIFACT_FEATURE.to_owned()],
            request_digest: request.digest().map_err(undecodable)?,
            request,
            result,
        };
        wire.validate_shape()?;
        Ok(wire)
    }

    /// The digest the stored request should carry. Recomputing it grants nothing: a consumer
    /// still refits the whole result.
    ///
    /// # Errors
    /// The request does not encode.
    pub fn expected_request_digest(&self) -> Result<String, IoError> {
        self.request.digest()
    }

    fn validate_shape(&self) -> Result<(), ArrayFitError> {
        let unsupported = ArrayFitError::UnsupportedSemantics;
        if self.required_features != [ARRAY_FIT_ARTIFACT_FEATURE] {
            return Err(unsupported("required features"));
        }
        if self.result.licensing != ARRAY_FIT_LICENSING
            || self.result.calibration != ARRAY_FIT_CALIBRATION
        {
            return Err(unsupported("this family is unlicensed and its calibration unmeasured"));
        }
        if self.result.estimator_id != self.request.estimator_id() {
            return Err(unsupported("the result belongs to another estimator"));
        }
        Ok(())
    }

    /// Encode as CBOR.
    ///
    /// # Errors
    /// Encoding failure.
    pub fn export(&self) -> Result<Vec<u8>, IoError> {
        crate::to_cbor(self)
    }

    /// Encode as CBOR into `writer`, producing exactly the bytes of [`Self::export`].
    ///
    /// # Errors
    /// Encoding or writer failure.
    pub fn export_into<W: std::io::Write>(&self, writer: W) -> Result<(), IoError> {
        ciborium::into_writer(self, writer).map_err(IoError::cbor_encode)
    }

    /// Decode, refusing any other version before the payload is interpreted.
    ///
    /// # Errors
    /// [`IoError::UnsupportedVersion`], or a decoding or shape failure.
    pub fn decode(bytes: &[u8]) -> Result<Self, IoError> {
        let wire: Self = crate::decode_with_version(bytes, |version| {
            if version == ARRAY_FIT_ARTIFACT_VERSION {
                Ok(())
            } else {
                Err(IoError::UnsupportedVersion { version })
            }
        })?;
        wire.validate_shape()?;
        Ok(wire)
    }

    /// Decode and refit everything under the consumer's limits; accept only a result
    /// identical to the stored one.
    ///
    /// # Errors
    /// Any reconstruction failure, with the reason code and `array_fit.*` detail.
    pub fn consume_with_limits(
        bytes: &[u8],
        limits: ArrayFitConsumeLimits,
    ) -> Result<Self, IoError> {
        Self::consume_typed(bytes, limits).map_err(IoError::from)
    }

    /// [`Self::consume_with_limits`] with the typed refusal kind.
    ///
    /// # Errors
    /// As [`Self::consume_with_limits`].
    pub fn consume_typed(
        bytes: &[u8],
        limits: ArrayFitConsumeLimits,
    ) -> Result<Self, ArrayFitError> {
        Self::consume_typed_expected(bytes, limits, None)
    }

    /// Independently consume, also binding an explicit producer request digest before fitting.
    ///
    /// # Errors
    /// As [`Self::consume_typed`], including producer expectation mismatch.
    pub fn consume_typed_expected(bytes: &[u8], limits: ArrayFitConsumeLimits, expected: Option<&str>) -> Result<Self, ArrayFitError> {
        let wire = Self::decode(bytes).map_err(|e| match e {
            IoError::Refused { .. } => ArrayFitError::UnsupportedSemantics("shape"),
            other => ArrayFitError::Undecodable(other.to_string()),
        })?;
        wire.request.check_limits(&limits)?;
        if expected.is_some_and(|expected| expected != wire.request_digest) {
            return Err(ArrayFitError::RequestMismatch);
        }
        let undecodable = |e: IoError| ArrayFitError::Undecodable(e.to_string());
        crate::require_digest(&wire.request_digest, wire.request.digest(), undecodable, || {
            ArrayFitError::RequestMismatch
        })?;
        let fresh = wire.request.evaluate()?;
        let stored = crate::to_cbor(&wire.result).map_err(undecodable)?;
        let recomputed = crate::to_cbor(&fresh).map_err(undecodable)?;
        if stored != recomputed {
            return Err(ArrayFitError::ResultMismatch);
        }
        Ok(wire)
    }
}

#[cfg(test)]
mod header_and_replay_regression {
    use super::*;
    #[derive(Serialize)]
    struct OriginalRecord<'a> {
        version:u32, required_features:&'a [String], request:&'a ArrayFitRequestWire,
        result:&'a ArrayFitResultWire, request_digest:&'a str,
    }
    #[test]
    fn exact_historical_record_encoding_and_one_independent_replay() {
        let mut x=vec![1.0;200]; x.extend((0..200).map(|i|if i>=100 {1.0}else{0.0}));
        let treatment=(0..200).map(|i|if i<30 || (100..160).contains(&i) {1.0}else{0.0}).collect();
        let request=ArrayFitRequestWire::PropensityPosterior(PropensityPosteriorWire { x_colmajor:x,treatment,nrows:200,prior_sd:2.0,draws:8,seed:13,rows:vec![vec![1.0,0.0]],clip:0.05 });
        let before=ARRAY_EVALUATIONS.with(std::cell::Cell::get);
        let wire=ArrayFitArtifactWire::seal(request).unwrap();
        assert_eq!(ARRAY_EVALUATIONS.with(std::cell::Cell::get),before+1);
        let original=OriginalRecord {version:wire.version,required_features:&wire.required_features,request:&wire.request,result:&wire.result,request_digest:&wire.request_digest};
        let bytes=wire.export().unwrap();
        assert_eq!(bytes,crate::to_cbor(&original).unwrap());
        assert!(inspect_array_fit_header(&bytes).unwrap());
        assert_eq!(ARRAY_EVALUATIONS.with(std::cell::Cell::get),before+1);
        assert_eq!(ArrayFitArtifactWire::consume_typed_expected(&bytes,ArrayFitConsumeLimits::default(),Some(&"0".repeat(64))).unwrap_err(),ArrayFitError::RequestMismatch);
        assert_eq!(ARRAY_EVALUATIONS.with(std::cell::Cell::get),before+1);
        let consumed=ArrayFitArtifactWire::consume_typed_expected(&bytes,ArrayFitConsumeLimits::default(),Some(&wire.request_digest)).unwrap();
        assert_eq!(ARRAY_EVALUATIONS.with(std::cell::Cell::get),before+2);
        assert_eq!(consumed,wire); assert_eq!(consumed.export().unwrap(),bytes);
    }
}
