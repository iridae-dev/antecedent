//! B4 categorical treatments: ordered and unordered regimes with a declared reference level,
//! per-level and requested pairwise contrasts from the dummy-coded regression's full covariance,
//! Holm multiplicity over the declared family, a declared monotonicity test for ordered levels,
//! and a portable, independently consumable artifact.
//!
//! A declared level with no rows (`categorical_treatment.absent_level`) or too few
//! (`categorical_treatment.sparse_level`) refuses with the level named; a level is never silently
//! dropped or merged. Unordered levels are canonicalised (sorted) before any arithmetic, so the
//! declared order of an unordered level list does not change a number.
//!
//! Every p-value (Wald, Holm, chi-square, monotonicity) is asymptotic and its calibration is
//! **unmeasured**. The monotonicity p-value is a conservative union-intersection bound whose
//! rejection is evidence against monotonicity; failing to reject does not prove it.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub use antecedent_estimate::categorical_treatment::{
    CATEGORICAL_OMNIBUS_NULL, LevelScale, MONOTONICITY_NON_DECREASING_NULL,
    MONOTONICITY_NON_INCREASING_NULL, MonotonicityDirection,
};
pub use antecedent_estimate::effect_constancy::CalibrationStatus;
pub use antecedent_io::categorical_treatment_artifact::{
    AdjacentStepWire, CATEGORICAL_HC_ROW_CAP, CATEGORICAL_TREATMENT_ARTIFACT_VERSION,
    CATEGORICAL_TREATMENT_CALIBRATION, CategoricalResultWire, CategoricalSpecWire,
    CategoricalTreatmentArtifact, CategoricalTreatmentArtifactError, CategoricalTreatmentIdentity,
    CategoricalTreatmentReportWire, CategoricalTreatmentRequest, LevelContrastWire, LevelCountWire,
    MonotonicityWire, PairContrastWire,
};

use antecedent_estimate::categorical_treatment::CategoricalTreatmentFit;

use crate::error::CausalError;

/// A sealed or consumed categorical-treatment answer.
#[derive(Clone, Debug, PartialEq)]
pub struct CategoricalTreatment {
    artifact: CategoricalTreatmentArtifact,
}

impl CategoricalTreatment {
    /// Fit the declared categorical regime and seal the result with its identity.
    ///
    /// # Errors
    /// `arm_not_populated` for an absent or sparse level, `invalid_argument` for an undeclared
    /// level or reference, `route_not_supported` for a monotonicity test on unordered levels, or
    /// a bound exceeded; see [`CategoricalTreatmentArtifactError::refusal`].
    pub fn evaluate(
        request: &CategoricalTreatmentRequest,
    ) -> Result<Self, CategoricalTreatmentArtifactError> {
        Ok(Self { artifact: CategoricalTreatmentArtifact::seal(request)? })
    }

    /// Consume an exported artifact by recomputing every coefficient, contrast, Holm value and
    /// test from its embedded design.
    ///
    /// `expected` is an identity the consumer retained independently.
    ///
    /// # Errors
    /// Corruption, an unknown major version, a changed identity, or a stored result that does
    /// not replay.
    pub fn consume(
        bytes: &[u8],
        expected: Option<&CategoricalTreatmentIdentity>,
    ) -> Result<Self, CategoricalTreatmentArtifactError> {
        Ok(Self { artifact: CategoricalTreatmentArtifact::from_bytes(bytes, expected)? })
    }

    /// Export the checksummed artifact.
    ///
    /// # Errors
    /// An empty id or an oversized payload.
    pub fn export(&self, artifact_id: &str) -> Result<Vec<u8>, CategoricalTreatmentArtifactError> {
        self.artifact.to_bytes(artifact_id)
    }

    /// The core fit.
    #[must_use]
    pub fn fit(&self) -> &CategoricalTreatmentFit {
        self.artifact.fit()
    }

    /// The identity digests.
    #[must_use]
    pub fn identity(&self) -> &CategoricalTreatmentIdentity {
        self.artifact.identity()
    }

    /// The self-describing report.
    #[must_use]
    pub fn report(&self) -> CategoricalTreatmentReportWire {
        self.artifact.report()
    }

    /// The underlying artifact.
    #[must_use]
    pub fn artifact(&self) -> &CategoricalTreatmentArtifact {
        &self.artifact
    }
}

impl From<CategoricalTreatmentArtifactError> for CausalError {
    fn from(error: CategoricalTreatmentArtifactError) -> Self {
        match error.refusal() {
            Some((code, detail, text)) => CausalError::Compile {
                message: format!("{}{code}: {detail}: {text}", crate::error::REASON_PREFIX),
            },
            None => match error {
                CategoricalTreatmentArtifactError::LimitsExceeded(_) => {
                    CausalError::Compile { message: error.to_string() }
                }
                other => {
                    CausalError::Serialization(antecedent_io::IoError::Convert(other.to_string()))
                }
            },
        }
    }
}
