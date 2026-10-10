//! B4 latent-class (finite mixture) regime effects: class-specific effects, class weights, the
//! mixture-average effect and the per-unit posterior class responsibilities, with a portable,
//! independently consumable artifact.
//!
//! The number of classes `K` (2 to 4) is declared, never selected. Classes are reported in a
//! canonical order (ascending class effect, then intercept, then weight) and the artifact keeps
//! the mapping to the raw EM components, so label switching cannot change the reported result.
//! Identification rests on a declared within-class conditional randomization and the
//! Gaussian-linear class model; the fit refuses without the randomization declaration and
//! refuses weak or degenerate classes (`latent_class.weak_class`,
//! `latent_class.degenerate_class`).
//!
//! Standard errors are a seeded, label-aligned bootstrap with **unmeasured** calibration: no
//! coverage or interval claim is made.
//!
//! [`LatentClassEffects::fit`] fits and seals; [`LatentClassEffects::consume`] refits from the
//! artifact's embedded dataset under the stored seed and configuration and checks the canonical
//! output bit for bit, refusing a resealed change when given a retained
//! [`LatentClassIdentity`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub use antecedent_estimate::latent_class_effects::{
    LATENT_CLASS_CAVEAT, LATENT_CLASS_INFERENCE_CLAIM, LATENT_CLASS_MAX_CLASSES, LatentClassConfig,
    LatentClassResult, PremiseStatus,
};
pub use antecedent_io::latent_class_artifact::{
    LATENT_CLASS_ARTIFACT_MAX_ROWS, LATENT_CLASS_ARTIFACT_VERSION, LatentClassArtifact,
    LatentClassArtifactError, LatentClassBootstrapWire, LatentClassConfigWire, LatentClassDataWire,
    LatentClassIdentity, LatentClassLikelihoodWire, LatentClassMeta, LatentClassPremiseRecord,
    LatentClassReportWire, LatentClassRequestWire, LatentClassWire, ResponsibilitiesSummaryWire,
};

use crate::error::CausalError;

/// A sealed or consumed latent-class answer.
#[derive(Clone, Debug, PartialEq)]
pub struct LatentClassEffects {
    artifact: LatentClassArtifact,
}

impl LatentClassEffects {
    /// Fit the mixture on a declared request and seal the answer with its identity.
    ///
    /// # Errors
    /// Refusals `latent_class.randomization_not_declared`, `.weak_class`, `.degenerate_class`,
    /// `.not_converged`, `.likelihood_not_monotone`, `.treatment_constant`,
    /// `.design_support_too_small` or an input-validation detail, and bounds exceeded; see
    /// [`LatentClassArtifactError::refusal`].
    pub fn fit(request: &LatentClassRequestWire) -> Result<Self, LatentClassArtifactError> {
        Ok(Self { artifact: LatentClassArtifact::seal(request)? })
    }

    /// Consume an exported artifact by refitting under the stored seed and configuration.
    ///
    /// `expected` is an identity the consumer retained independently.
    ///
    /// # Errors
    /// Corruption, an unknown major version, a changed identity, or a canonical record that does
    /// not replay bit for bit.
    pub fn consume(
        bytes: &[u8],
        expected: Option<&LatentClassIdentity>,
    ) -> Result<Self, LatentClassArtifactError> {
        Ok(Self { artifact: LatentClassArtifact::from_bytes(bytes, expected)? })
    }

    /// Export the checksummed artifact.
    ///
    /// # Errors
    /// An empty id or an oversized payload.
    pub fn export(&self, artifact_id: &str) -> Result<Vec<u8>, LatentClassArtifactError> {
        self.artifact.to_bytes(artifact_id)
    }

    /// The core result, with the full responsibilities matrix.
    #[must_use]
    pub fn result(&self) -> &LatentClassResult {
        self.artifact.result()
    }

    /// The identity digests.
    #[must_use]
    pub fn identity(&self) -> &LatentClassIdentity {
        self.artifact.identity()
    }

    /// The report: the record plus the full responsibilities.
    #[must_use]
    pub fn report(&self) -> LatentClassReportWire {
        self.artifact.report()
    }

    /// The underlying artifact.
    #[must_use]
    pub fn artifact(&self) -> &LatentClassArtifact {
        &self.artifact
    }
}

impl From<LatentClassArtifactError> for CausalError {
    fn from(error: LatentClassArtifactError) -> Self {
        match error.refusal() {
            Some((code, detail, text)) => CausalError::Compile {
                message: format!("{}{code}: {detail}: {text}", crate::error::REASON_PREFIX),
            },
            None => match error {
                LatentClassArtifactError::LimitsExceeded(_) => {
                    CausalError::Compile { message: error.to_string() }
                }
                other => {
                    CausalError::Serialization(antecedent_io::IoError::Convert(other.to_string()))
                }
            },
        }
    }
}
