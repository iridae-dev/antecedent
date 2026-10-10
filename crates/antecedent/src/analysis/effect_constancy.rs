//! F18 `EffectConstancy`: a global heterogeneity test of one effect across a declared region or
//! time partition, with a portable, independently consumable artifact.
//!
//! The null is frozen (the effect is equal across all declared partitions) and the estimand is
//! shared: a partition that changes the estimand, units, regime or population refuses the
//! global test with `route_not_supported` / `effect_constancy.incompatible_partitions`.
//! Dependence between partition estimates is declared, never assumed from silence.
//!
//! The claim is **point only** and the calibration coordinate is **unmeasured**: failing to
//! reject the null never proves constancy, and the test's Type I error and power have not been
//! measured. The per-partition effects are part of the result so transport diagnostics, prior
//! transfer and policy generalization can consume them directly.
//!
//! [`EffectConstancy::evaluate`] produces and seals the result; [`EffectConstancy::consume`]
//! recomputes it from an exported artifact and, given a retained
//! [`EffectConstancyIdentity`], refuses a resealed change of partition identity, estimand,
//! covariance, null or multiplicity family.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub use antecedent_estimate::effect_constancy::{
    CalibrationStatus, ConstancyConclusion, ContrastFamily, EFFECT_CONSTANCY_NULL,
    EffectConstancyResult, EffectEstimandIdentity, HeterogeneityStatistic, NON_REJECTION_CAVEAT,
    POWER_CAVEAT, PartitionDependence, PartitionSupport,
};
pub use antecedent_io::effect_constancy_artifact::{
    ContrastWire, DependenceWire, EFFECT_CONSTANCY_ARTIFACT_VERSION, EFFECT_CONSTANCY_CALIBRATION,
    EffectConstancyArtifact, EffectConstancyArtifactError, EffectConstancyIdentity,
    EffectConstancyReportWire, EffectConstancyRequestWire, EffectEstimandWire, FamilyWire,
    PartitionReportWire, PartitionWire, ResultWire,
};

use crate::error::CausalError;

/// A sealed or consumed effect-constancy answer.
#[derive(Clone, Debug, PartialEq)]
pub struct EffectConstancy {
    artifact: EffectConstancyArtifact,
}

impl EffectConstancy {
    /// Run the global constancy test on a declared request and seal it with its identity.
    ///
    /// # Errors
    /// `route_not_supported` (`effect_constancy.incompatible_partitions`,
    /// `effect_constancy.unsupported_partition`), `invalid_argument`
    /// (`effect_constancy.too_few_partitions`, `.invalid_covariance`, `.unknown_reference`, ...)
    /// or a bound exceeded; see [`EffectConstancyArtifactError::refusal`].
    pub fn evaluate(
        request: &EffectConstancyRequestWire,
    ) -> Result<Self, EffectConstancyArtifactError> {
        Ok(Self { artifact: EffectConstancyArtifact::seal(request)? })
    }

    /// Consume an exported artifact by recomputing the statistic, p-value and Holm values from
    /// its embedded effects, standard errors and covariance.
    ///
    /// `expected` is an identity the consumer retained independently.
    ///
    /// # Errors
    /// Corruption, an unknown major version, a changed identity, or a stored result that does
    /// not replay.
    pub fn consume(
        bytes: &[u8],
        expected: Option<&EffectConstancyIdentity>,
    ) -> Result<Self, EffectConstancyArtifactError> {
        Ok(Self { artifact: EffectConstancyArtifact::from_bytes(bytes, expected)? })
    }

    /// Export the checksummed artifact.
    ///
    /// # Errors
    /// An empty id or an oversized payload.
    pub fn export(&self, artifact_id: &str) -> Result<Vec<u8>, EffectConstancyArtifactError> {
        self.artifact.to_bytes(artifact_id)
    }

    /// The core test result.
    #[must_use]
    pub fn result(&self) -> &EffectConstancyResult {
        self.artifact.result()
    }

    /// The identity digests.
    #[must_use]
    pub fn identity(&self) -> &EffectConstancyIdentity {
        self.artifact.identity()
    }

    /// The self-describing report.
    #[must_use]
    pub fn report(&self) -> EffectConstancyReportWire {
        self.artifact.report()
    }

    /// The underlying artifact.
    #[must_use]
    pub fn artifact(&self) -> &EffectConstancyArtifact {
        &self.artifact
    }
}

impl From<EffectConstancyArtifactError> for CausalError {
    fn from(error: EffectConstancyArtifactError) -> Self {
        match error.refusal() {
            Some((code, detail, text)) => CausalError::Compile {
                message: format!("{}{code}: {detail}: {text}", crate::error::REASON_PREFIX),
            },
            None => match error {
                EffectConstancyArtifactError::LimitsExceeded(_) => {
                    CausalError::Compile { message: error.to_string() }
                }
                other => {
                    CausalError::Serialization(antecedent_io::IoError::Convert(other.to_string()))
                }
            },
        }
    }
}
