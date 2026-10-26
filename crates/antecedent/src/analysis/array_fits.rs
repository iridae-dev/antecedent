//! The treatment-propensity nuisance product: a retained, sealed fit with a replay artifact.
//!
//! A prepared average-effect study knows its treatment and certified adjustment set;
//! [`crate::PreparedStudy::treatment_score_posterior`] fits the Laplace posterior of the logistic
//! treatment model on those columns and returns it as an [`ArrayFit`]. The product is a posterior
//! of a nuisance, never of an effect, it is unlicensed with calibration unmeasured, and the
//! result and the artifact say so.
//!
//! [`ArrayFit::consume`] refits from the exported artifact and refuses any resealed change of
//! request, array or stored value. The artifact is a different kind from the cross-fitted
//! effect-score table, which does not read it.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub use antecedent_io::array_fit_artifact::{
    ARRAY_FIT_ARTIFACT_FEATURE, ARRAY_FIT_ARTIFACT_VERSION, ARRAY_FIT_CALIBRATION,
    ARRAY_FIT_LICENSING, ArrayFitArtifactWire, ArrayFitConsumeLimits, ArrayFitError,
    ArrayFitRequestWire, ArrayFitResultWire, ID_BAYESIAN_PROPENSITY,
    ID_BAYESIAN_TEMPORAL_PROPENSITY, ID_TEMPORAL_PROPENSITY, NamedValueWire, PenalizedScoreWire,
    PropensityDesignWire, PropensityPosteriorWire, PropensityScorePosteriorWire,
};

use antecedent_estimate::EstimationError;

use crate::error::CausalError;

/// A retained, sealed nuisance product.
#[derive(Clone, Debug, PartialEq)]
pub struct ArrayFit {
    artifact: ArrayFitArtifactWire,
}

impl ArrayFit {
    /// Fit a declared request and seal it with its digest.
    ///
    /// # Errors
    /// A refused design or prior, a request above the default consumer bounds, or a Laplace fit
    /// that does not publish a posterior (`array_fit.fit_refused` or the estimator's own detail).
    pub fn fit(request: ArrayFitRequestWire) -> Result<Self, ArrayFitError> {
        Ok(Self { artifact: ArrayFitArtifactWire::seal(request)? })
    }

    /// Consume an exported artifact by refitting from its embedded arrays.
    ///
    /// # Errors
    /// Corruption, an unknown version, a changed digest or request, or a stored result that
    /// does not replay.
    pub fn consume(bytes: &[u8], limits: ArrayFitConsumeLimits) -> Result<Self, ArrayFitError> {
        Ok(Self { artifact: ArrayFitArtifactWire::consume_typed(bytes, limits)? })
    }

    /// Export the artifact.
    ///
    /// # Errors
    /// Encoding failure.
    pub fn export(&self) -> Result<Vec<u8>, ArrayFitError> {
        self.artifact.export().map_err(|error| ArrayFitError::Undecodable(error.to_string()))
    }

    /// The estimator id.
    #[must_use]
    pub fn estimator_id(&self) -> &str {
        &self.artifact.result.estimator_id
    }

    /// The stored request, arrays included.
    #[must_use]
    pub fn request(&self) -> &ArrayFitRequestWire {
        &self.artifact.request
    }

    /// The stored result.
    #[must_use]
    pub fn result(&self) -> &ArrayFitResultWire {
        &self.artifact.result
    }

    /// One named real of the result.
    #[must_use]
    pub fn value(&self, name: &str) -> Option<f64> {
        self.artifact.result.values.iter().find(|v| v.name == name).map(|v| v.value)
    }

    /// Posterior draws of the target; empty when the result is not a posterior of the target.
    #[must_use]
    pub fn draws(&self) -> &[f64] {
        &self.artifact.result.draws
    }

    /// The request digest.
    #[must_use]
    pub fn digest(&self) -> &str {
        &self.artifact.request_digest
    }

    /// The underlying artifact.
    #[must_use]
    pub fn artifact(&self) -> &ArrayFitArtifactWire {
        &self.artifact
    }
}

impl From<ArrayFitError> for CausalError {
    fn from(error: ArrayFitError) -> Self {
        let (code, detail) = error.refusal();
        let explanation = match &error {
            ArrayFitError::Refused { message, .. } => message.clone(),
            other => other.to_string(),
        };
        CausalError::Estimate(EstimationError::refused_detail(code, detail, explanation))
    }
}

/// How a prepared study's treatment propensity is summarized as a nuisance product.
///
/// The prior is `N(0, prior_sd^2)` on every logit coefficient, the intercept included, and the
/// product is a posterior of that nuisance, never of an effect.
#[derive(Clone, Debug, PartialEq)]
pub struct TreatmentScoreSpec {
    /// Standard deviation of the Gaussian prior on every coefficient.
    pub prior_sd: f64,
    /// Posterior draws.
    pub draws: usize,
    /// Seed of the draw stream.
    pub seed: u64,
    /// Positivity band `(clip, 1 - clip)` of the outside-band probability.
    pub clip: f64,
    /// Rows of the prepared data whose score posterior is reported.
    pub summary_rows: Vec<usize>,
}

/// The treatment model fitted on the independent-subject rows of a prepared panel study.
#[derive(Clone, Debug, PartialEq)]
pub enum TemporalScoreModel {
    /// Ridge-penalized logistic fit; the intercept is unpenalized and the score is the inverse
    /// link at the penalized mode, with no uncertainty.
    Penalized {
        /// Penalty on half the squared norm of the slopes.
        penalty: f64,
    },
    /// Laplace posterior under independent `N(0, sd^2)` priors on the intercept and the slopes.
    Posterior {
        /// Prior standard deviation of the intercept.
        intercept_sd: f64,
        /// Prior standard deviation of each slope.
        slope_sd: f64,
        /// Posterior draws.
        draws: usize,
        /// Seed of the draw stream.
        seed: u64,
    },
}

/// How a prepared panel study's temporal treatment propensity is summarized as a nuisance
/// product: the model and the lag histories whose scores are reported.
#[derive(Clone, Debug, PartialEq)]
pub struct TemporalTreatmentScoreSpec {
    /// The treatment model.
    pub model: TemporalScoreModel,
    /// Lag histories, one value per adjustment lag column, whose scores are reported.
    pub histories: Vec<Vec<f64>>,
}
