//! B4 joint vector-treatment coefficients: several treatments estimated *together* by one
//! regression that shares ONE adjustment set and ONE row snapshot, with the named coefficient
//! vector, its full covariance (off-diagonals included), declared contrasts and a portable,
//! independently consumable artifact.
//!
//! A treatment that declares a different adjustment set, row snapshot or row count than the
//! shared one is refused with `route_not_supported` (`vector_treatment.adjustment_set_mismatch`,
//! `.row_snapshot_mismatch`, `.row_count_mismatch`); nothing is dropped silently.
//!
//! Every p-value (Wald, Holm, joint chi-square) is asymptotic and its calibration is
//! **unmeasured**: no coverage claim is made and no interval is produced. The caller owns that
//! the shared adjustment set is valid.
//!
//! [`VectorTreatment::evaluate`] fits and seals the result with its embedded compact design;
//! [`VectorTreatment::consume`] recomputes it from an exported artifact and, given a retained
//! [`VectorTreatmentIdentity`], refuses a resealed change of snapshot, adjustment set,
//! coefficient order, design, covariance kind, contrasts or result.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub use antecedent_estimate::effect_constancy::CalibrationStatus;
pub use antecedent_estimate::vector_treatment::{
    VECTOR_TREATMENT_INFERENCE_CLAIM, VECTOR_TREATMENT_MAX_TREATMENTS, VECTOR_TREATMENT_NULL,
    VectorCovariance, VectorTreatmentFit,
};
pub use antecedent_io::vector_treatment_artifact::{
    AdjustmentColumnRequest, CoefficientWire, ContrastResultWire, ContrastWire, TreatmentRequest,
    VECTOR_TREATMENT_ARTIFACT_VERSION, VECTOR_TREATMENT_CALIBRATION, VECTOR_TREATMENT_HC_ROW_CAP,
    VectorResultWire, VectorTreatmentArtifact, VectorTreatmentArtifactError,
    VectorTreatmentIdentity, VectorTreatmentReportWire, VectorTreatmentRequest, WaldWire,
};

use crate::error::CausalError;

/// A sealed or consumed joint vector-treatment answer.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorTreatment {
    artifact: VectorTreatmentArtifact,
}

impl VectorTreatment {
    /// Fit the declared treatments jointly and seal the result with its identity.
    ///
    /// # Errors
    /// `route_not_supported` for incompatible treatments, `design_rank_deficient`,
    /// `invalid_argument` or a bound exceeded; see [`VectorTreatmentArtifactError::refusal`].
    pub fn evaluate(
        request: &VectorTreatmentRequest,
    ) -> Result<Self, VectorTreatmentArtifactError> {
        Ok(Self { artifact: VectorTreatmentArtifact::seal(request)? })
    }

    /// Consume an exported artifact by recomputing the coefficients, full covariance, contrasts
    /// and joint test from its embedded design.
    ///
    /// `expected` is an identity the consumer retained independently.
    ///
    /// # Errors
    /// Corruption, an unknown major version, a changed identity, or a stored result that does
    /// not replay.
    pub fn consume(
        bytes: &[u8],
        expected: Option<&VectorTreatmentIdentity>,
    ) -> Result<Self, VectorTreatmentArtifactError> {
        Ok(Self { artifact: VectorTreatmentArtifact::from_bytes(bytes, expected)? })
    }

    /// Export the checksummed artifact.
    ///
    /// # Errors
    /// An empty id or an oversized payload.
    pub fn export(&self, artifact_id: &str) -> Result<Vec<u8>, VectorTreatmentArtifactError> {
        self.artifact.to_bytes(artifact_id)
    }

    /// The core fit.
    #[must_use]
    pub fn fit(&self) -> &VectorTreatmentFit {
        self.artifact.fit()
    }

    /// The identity digests.
    #[must_use]
    pub fn identity(&self) -> &VectorTreatmentIdentity {
        self.artifact.identity()
    }

    /// The self-describing report.
    #[must_use]
    pub fn report(&self) -> VectorTreatmentReportWire {
        self.artifact.report()
    }

    /// The underlying artifact.
    #[must_use]
    pub fn artifact(&self) -> &VectorTreatmentArtifact {
        &self.artifact
    }
}

impl From<VectorTreatmentArtifactError> for CausalError {
    fn from(error: VectorTreatmentArtifactError) -> Self {
        match error.refusal() {
            Some((code, detail, text)) => CausalError::Compile {
                message: format!("{}{code}: {detail}: {text}", crate::error::REASON_PREFIX),
            },
            None => match error {
                VectorTreatmentArtifactError::LimitsExceeded(_) => {
                    CausalError::Compile { message: error.to_string() }
                }
                other => {
                    CausalError::Serialization(antecedent_io::IoError::Convert(other.to_string()))
                }
            },
        }
    }
}
