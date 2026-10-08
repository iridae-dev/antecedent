//! 2.3 B1 (X4): one complete response-grid row, a named functional of a randomized-dose
//! response curve with per-dose support, and its portable, independently consumable
//! artifact.
//!
//! Exactly one named functional is requested per call (`level`, `derivative` or
//! `contrast`), on a fixed, caller-declared bandwidth. Level and contrast claims,
//! derivative claims and the simultaneous band are separate: a derivative needs a
//! derivative claim, a level or contrast needs a pointwise-level claim, and the
//! simultaneous band is closed (`dose_grid.simultaneous_band_closed`). Every interval
//! is pointwise with calibration `unmeasured`: no coverage claim is made, and the
//! smoothing bias is not included. A dose that is not supported (outside the observed
//! range, or with too little local weight) is refused, never extrapolated;
//! [`dose_support_table`] shows the per-dose labels without refusing.
//!
//! [`DoseGridFunctional::evaluate`] runs and seals the row;
//! [`DoseGridFunctional::consume`] recomputes the local-quadratic fit from the
//! exported table and refuses any resealed change of premise, table or value.
//! A `quadratic_mean` request explicitly attests an exact quadratic conditional mean;
//! polynomial reproduction accounts for its zero smoothing bias, with sampling
//! calibration still unmeasured. Its artifact binds this additional premise and feature.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub use antecedent_estimate::dose_grid_functional::{
    ClaimSet, DoseDesign, DoseFunctional, IntervalCalibration,
};
pub use antecedent_io::dose_grid_functional_artifact::{
    DOSE_GRID_ARTIFACT_FEATURE, DOSE_GRID_ARTIFACT_VERSION, DOSE_GRID_BAND_CLOSED,
    DOSE_GRID_CALIBRATION, DOSE_GRID_INFERENCE_CLAIM, DOSE_GRID_QUADRATIC_FEATURE, DoseClaimsWire,
    DoseContrastWire, DoseFunctionalWire, DoseGridArtifactError, DoseGridArtifactWire,
    DoseGridConsumeLimits, DoseGridRequestWire, DoseGridResultWire, DosePointWire, DoseSupportWire,
    dose_support_table,
};

use crate::error::CausalError;

/// A sealed or consumed dose-grid functional answer.
#[derive(Clone, Debug, PartialEq)]
pub struct DoseGridFunctional {
    artifact: DoseGridArtifactWire,
}

impl DoseGridFunctional {
    /// Run the named functional on a declared request and seal it with its digests.
    ///
    /// # Errors
    /// A typed refusal whose detail is one of `dose_grid.graph_not_certified`,
    /// `dose_grid.simultaneous_band_closed`, `dose_grid.derivative_without_claim`,
    /// `dose_grid.level_without_claim`, `dose_grid.invalid_request`,
    /// `dose_grid.bandwidth_outside_range`, `dose_grid.unsupported_dose`,
    /// `dose_grid.insufficient_local_weight` or `dose_grid.non_finite_result`; see
    /// [`DoseGridArtifactError::refusal`].
    pub fn evaluate(request: DoseGridRequestWire) -> Result<Self, DoseGridArtifactError> {
        Ok(Self { artifact: DoseGridArtifactWire::seal(request)? })
    }

    /// Consume an exported artifact by recomputing the local-quadratic fit from its
    /// embedded dose/outcome table.
    ///
    /// # Errors
    /// Corruption, an unknown version, a changed digest or premise, or a stored result
    /// that does not replay bit for bit.
    pub fn consume(
        bytes: &[u8],
        limits: DoseGridConsumeLimits,
    ) -> Result<Self, DoseGridArtifactError> {
        Ok(Self { artifact: DoseGridArtifactWire::consume_typed(bytes, limits)? })
    }

    /// Export the artifact.
    ///
    /// # Errors
    /// Encoding failure.
    pub fn export(&self) -> Result<Vec<u8>, DoseGridArtifactError> {
        self.artifact
            .export()
            .map_err(|error| DoseGridArtifactError::Undecodable(error.to_string()))
    }

    /// The stored request (premises and compact table).
    #[must_use]
    pub fn request(&self) -> &DoseGridRequestWire {
        &self.artifact.request
    }

    /// The result.
    #[must_use]
    pub fn result(&self) -> &DoseGridResultWire {
        &self.artifact.result
    }

    /// `(premises digest, data digest)`.
    #[must_use]
    pub fn digests(&self) -> (&str, &str) {
        (&self.artifact.premises_digest, &self.artifact.data_digest)
    }

    /// The underlying artifact.
    #[must_use]
    pub fn artifact(&self) -> &DoseGridArtifactWire {
        &self.artifact
    }
}

impl From<DoseGridArtifactError> for CausalError {
    fn from(error: DoseGridArtifactError) -> Self {
        let (code, detail) = error.refusal();
        CausalError::Compile {
            message: format!("{}{code}: {detail}: {error}", crate::error::REASON_PREFIX),
        }
    }
}
