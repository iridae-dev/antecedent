//! Categorical treatments: ordered and unordered regimes with a declared reference level,
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
//! The fit has no row ceiling, and every covariance kind (HC0 to HC3 included) exports a
//! summary of fixed size bound to the rows by a digest. [`CategoricalTreatment::evaluate_with`]
//! and [`CategoricalTreatment::consume_with`] run under an [`ExecutionContext`] memory budget and
//! cancellation, [`CategoricalTreatment::export_within`] under an export byte budget, and
//! [`CategoricalTreatment::verify_data`] checks held rows against the digest.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub use antecedent_estimate::categorical_treatment::{
    CATEGORICAL_OMNIBUS_NULL, LevelScale, MONOTONICITY_NON_DECREASING_NULL,
    MONOTONICITY_NON_INCREASING_NULL, MonotonicityDirection,
};
pub use antecedent_estimate::effect_constancy::CalibrationStatus;
pub use antecedent_io::categorical_treatment_artifact::{
    AdjacentStepWire, CATEGORICAL_DECLARED_ROWS_FORMAT, CATEGORICAL_HC_ROW_CAP,
    CATEGORICAL_TREATMENT_ARTIFACT_VERSION, CATEGORICAL_TREATMENT_CALIBRATION,
    CategoricalPosteriorDeclaration, CategoricalPosteriorWire, CategoricalResultWire,
    CategoricalRowsRequest, CategoricalSpecWire, CategoricalTreatmentArtifact,
    CategoricalTreatmentArtifactError, CategoricalTreatmentIdentity,
    CategoricalTreatmentReportWire, CategoricalTreatmentRequest, CovarianceKindWire,
    DeclaredRowsWire, LevelContrastWire, LevelCountWire, LevelScaleWire, MonotonicityDirectionWire,
    MonotonicityWire, PairContrastWire, PosteriorContrastWire, PosteriorMonotonicityWire,
};

use antecedent_core::ExecutionContext;
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

    /// Fit under the computation budget of `ctx`: its hard memory limit bounds the planned
    /// working set and a cancelled context stops the run. No row ceiling applies.
    ///
    /// # Errors
    /// The refusals of [`Self::evaluate`], `categorical_treatment.memory_budget_exceeded` and
    /// `categorical_treatment.cancelled`.
    pub fn evaluate_with(
        request: &CategoricalTreatmentRequest,
        ctx: &ExecutionContext,
    ) -> Result<Self, CategoricalTreatmentArtifactError> {
        Ok(Self { artifact: CategoricalTreatmentArtifact::seal_with(request, ctx)? })
    }

    /// Fit the declared regime under declared sampling weights, clusters or a Newey-West lag.
    ///
    /// Sampling weights follow the shared contract (rows of weight zero are deleted and counted,
    /// the rest are normalized to mean one, a sandwich covariance is required); clusters are read
    /// by the `cluster_robust` covariance and the lag by `newey_west`. A declaration its covariance
    /// kind does not read, or a kind whose declaration is missing, is refused, never dropped. The
    /// Gaussian posterior is refused for declared rows. With nothing declared this is
    /// [`Self::evaluate_with`].
    ///
    /// # Errors
    /// The refusals of [`Self::evaluate_with`]; `sampling_weights_invalid`; and
    /// `vector_treatment.weights_need_robust_covariance`, `.cluster_ids_required`,
    /// `.cluster_unread`, `.too_few_clusters`, `.lag_required`, `.lag_unread`, `.invalid_lag`.
    pub fn evaluate_declared(
        request: &CategoricalTreatmentRequest,
        rows: &CategoricalRowsRequest,
        ctx: &ExecutionContext,
    ) -> Result<Self, CategoricalTreatmentArtifactError> {
        Ok(Self { artifact: CategoricalTreatmentArtifact::seal_declared(request, rows, ctx)? })
    }

    /// Check rows still held, with their declared weights and clusters, against the digest.
    ///
    /// # Errors
    /// As [`Self::verify_data`].
    pub fn verify_data_declared(
        &self,
        request: &CategoricalTreatmentRequest,
        rows: &CategoricalRowsRequest,
    ) -> Result<(), CategoricalTreatmentArtifactError> {
        self.artifact.verify_data_declared(request, rows)
    }

    /// Attach the Bayesian counterpart of the fit under a declared prior: the posterior of the
    /// same dummy-coefficient contrasts under Gaussian homoskedastic errors, a Student-t
    /// marginal for each level and requested pair, and the Bonferroni/Frechet bounds of the
    /// declared monotonicity. The estimand is the superpopulation coefficient contrast of the
    /// stated linear model, not the finite-population average of the rows' effects; the section
    /// is unlicensed, its calibration is unmeasured, and it is recomputed from the stored design
    /// when the artifact is consumed. The covariance kind of the declared design is not read:
    /// the model-based likelihood is the posterior's.
    ///
    /// # Errors
    /// `invalid_argument` for a prior or credible level outside its range.
    pub fn with_posterior(
        self,
        declaration: &CategoricalPosteriorDeclaration,
    ) -> Result<Self, CategoricalTreatmentArtifactError> {
        Ok(Self { artifact: self.artifact.with_posterior(declaration)? })
    }

    /// The posterior section, when one was attached.
    #[must_use]
    pub fn posterior(&self) -> Option<&CategoricalPosteriorWire> {
        self.artifact.posterior()
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

    /// Consume within the replay budget of `ctx`.
    ///
    /// # Errors
    /// The refusals of [`Self::consume`], `categorical_treatment.memory_budget_exceeded` and
    /// `categorical_treatment.cancelled`.
    pub fn consume_with(
        bytes: &[u8],
        expected: Option<&CategoricalTreatmentIdentity>,
        ctx: &ExecutionContext,
    ) -> Result<Self, CategoricalTreatmentArtifactError> {
        Ok(Self { artifact: CategoricalTreatmentArtifact::from_bytes_with(bytes, expected, ctx)? })
    }

    /// Export the checksummed artifact.
    ///
    /// # Errors
    /// An empty id or an oversized payload.
    pub fn export(&self, artifact_id: &str) -> Result<Vec<u8>, CategoricalTreatmentArtifactError> {
        self.artifact.to_bytes(artifact_id)
    }

    /// Export within `export_budget` bytes. A refusal
    /// (`categorical_treatment.export_budget_exceeded`) leaves the fit available.
    ///
    /// # Errors
    /// An empty id or `categorical_treatment.export_budget_exceeded`.
    pub fn export_within(
        &self,
        artifact_id: &str,
        export_budget: usize,
    ) -> Result<Vec<u8>, CategoricalTreatmentArtifactError> {
        self.artifact.to_bytes_within(artifact_id, export_budget)
    }

    /// Check rows still held against the digest the artifact's summary was computed from.
    ///
    /// # Errors
    /// `IdentityMismatch` (`data`) for other rows; `categorical_treatment.data_not_bound` for a
    /// legacy artifact that embeds its rows.
    pub fn verify_data(
        &self,
        request: &CategoricalTreatmentRequest,
    ) -> Result<(), CategoricalTreatmentArtifactError> {
        self.artifact.verify_data(request)
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
