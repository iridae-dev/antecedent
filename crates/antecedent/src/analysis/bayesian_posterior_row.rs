//! The unlicensed Bayesian randomized-dose grid posterior and its portable artifact.
//!
//! The dose-grid row has no `Study` query (its frequentist row,
//! [`crate::analysis::dose_grid_functional`], is a standalone family too), so its Bayesian
//! counterpart is evaluated and sealed here on the same coordinate: the level, derivative or
//! contrast of a randomized-dose response curve, as the joint posterior over the grid under a
//! declared polynomial coefficient prior with the covariance across grid points retained. An
//! observational dose is refused (`dose_grid.graph_not_certified`). The survival and fixed
//! group-dose policy counterparts are `Study` routes (`bayesian.piecewise_survival`,
//! `bayesian.continuous_dose_policy`) and the smoothed dose transport posterior is
//! [`crate::PreparedSmoothedDose::posterior`].
//!
//! The row is explicitly unlicensed and its calibration is `unmeasured`; the result says so and
//! no support-matrix coordinate is claimed. [`BayesianPosteriorRow::evaluate`] runs and seals the
//! row; [`BayesianPosteriorRow::consume`] recomputes the posterior from the exported table and
//! refuses any resealed change of premise, table or value.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub use antecedent_io::bayesian_posterior_row_artifact::{
    DOSE_GRID_POSTERIOR_ESTIMATOR, DoseGridPosteriorWire, POSTERIOR_ROW_ARTIFACT_FEATURE,
    POSTERIOR_ROW_ARTIFACT_VERSION, POSTERIOR_ROW_CALIBRATION, POSTERIOR_ROW_UNCERTAINTY,
    PosteriorRowArtifactError, PosteriorRowArtifactWire, PosteriorRowConsumeLimits,
    PosteriorRowRequestWire, PosteriorRowResultWire, ResidualPriorWire,
};

use crate::error::CausalError;

/// A sealed or consumed Bayesian posterior row.
#[derive(Clone, Debug, PartialEq)]
pub struct BayesianPosteriorRow {
    artifact: PosteriorRowArtifactWire,
}

impl BayesianPosteriorRow {
    /// Run the posterior of a declared request and seal it with its digests.
    ///
    /// # Errors
    /// A typed refusal whose detail is one of `dose_grid.graph_not_certified`,
    /// `dose_grid.unsupported_dose`, `dose_grid.invalid_request` or `dose_grid.refused`; see
    /// [`PosteriorRowArtifactError::refusal`].
    pub fn evaluate(request: PosteriorRowRequestWire) -> Result<Self, PosteriorRowArtifactError> {
        Ok(Self { artifact: PosteriorRowArtifactWire::seal(request)? })
    }

    /// Consume an exported artifact by recomputing the posterior from its embedded table.
    ///
    /// # Errors
    /// Corruption, an unknown version, a changed digest or premise, or a stored result that
    /// does not replay bit for bit.
    pub fn consume(
        bytes: &[u8],
        limits: PosteriorRowConsumeLimits,
    ) -> Result<Self, PosteriorRowArtifactError> {
        Ok(Self { artifact: PosteriorRowArtifactWire::consume_typed(bytes, limits)? })
    }

    /// Export the artifact.
    ///
    /// # Errors
    /// Encoding failure.
    pub fn export(&self) -> Result<Vec<u8>, PosteriorRowArtifactError> {
        self.artifact
            .export()
            .map_err(|error| PosteriorRowArtifactError::Undecodable(error.to_string()))
    }

    /// The estimator id of this row, for example `bayesian.dose_grid`.
    #[must_use]
    pub fn estimator(&self) -> &str {
        &self.artifact.result.estimator
    }

    /// The stored request (premises and compact table).
    #[must_use]
    pub const fn request(&self) -> &PosteriorRowRequestWire {
        &self.artifact.request
    }

    /// The retained posterior: summaries, joint covariance and named diagnostics.
    #[must_use]
    pub const fn result(&self) -> &PosteriorRowResultWire {
        &self.artifact.result
    }

    /// `(premises digest, data digest)`.
    #[must_use]
    pub fn digests(&self) -> (&str, &str) {
        (&self.artifact.premises_digest, &self.artifact.data_digest)
    }

    /// The underlying artifact.
    #[must_use]
    pub const fn artifact(&self) -> &PosteriorRowArtifactWire {
        &self.artifact
    }
}

impl From<PosteriorRowArtifactError> for CausalError {
    fn from(error: PosteriorRowArtifactError) -> Self {
        let (code, detail) = error.refusal();
        CausalError::Compile {
            message: format!("{}{code}: {detail}: {error}", crate::error::REASON_PREFIX),
        }
    }
}
