//! B4 nonlinear continuous-mediator mediation: natural direct, indirect and total effects of a
//! binary treatment through a continuous mediator with a nonlinear outcome model, with a
//! portable, independently consumable artifact.
//!
//! Only the natural-effects estimand is open. It needs cross-world independence and sequential
//! ignorability (no unmeasured treatment-outcome, treatment-mediator or mediator-outcome
//! confounding and no treatment-induced mediator-outcome confounder); a declared violation
//! refuses (`effect_not_identified` / `cross_world_not_identified`). The interventional
//! (randomized-draw) effects are closed:
//! `nonlinear_mediation.interventional_effects_closed`.
//!
//! The integral over the cross-world mediator law is a deterministic Gauss-Hermite rule; the
//! estimate is evaluated at `n` and `2n` nodes and the disagreement is the reported integration
//! error. Bootstrap standard errors use a fixed seed and replicate ids. Calibration is
//! **unmeasured**: no coverage claim and no public interval; the point estimates and the
//! diagnostics are the claim.
//!
//! [`NonlinearMediation::estimate`] runs and seals the estimate; [`NonlinearMediation::consume`]
//! re-estimates from the artifact's embedded dataset and, given a retained
//! [`NonlinearMediationIdentity`], refuses a resealed change.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub use antecedent_estimate::nonlinear_mediation::{
    NONLINEAR_MEDIATION_CALIBRATION, NONLINEAR_MEDIATION_INTERVAL_STATUS,
    NONLINEAR_MEDIATION_MAX_DEGREE, NONLINEAR_MEDIATION_MAX_NODES, NonlinearMediationConfig,
    NonlinearMediationEstimand, NonlinearMediationEstimate, NonlinearMediationPremises,
};
pub use antecedent_io::nonlinear_mediation_artifact::{
    MediationBootstrapWire, MediationConfigWire, MediationDataWire, MediationMediatorWire,
    MediationModelSpecs, MediationOverlapWire, MediationPremiseRecord, MediationPremisesWire,
    MediationResultWire, NONLINEAR_MEDIATION_ARTIFACT_MAX_ROWS,
    NONLINEAR_MEDIATION_ARTIFACT_VERSION, NonlinearMediationArtifact,
    NonlinearMediationArtifactError, NonlinearMediationIdentity, NonlinearMediationMeta,
    NonlinearMediationRequestWire,
};

use crate::error::CausalError;

/// A sealed or consumed nonlinear-mediation answer.
#[derive(Clone, Debug, PartialEq)]
pub struct NonlinearMediation {
    artifact: NonlinearMediationArtifact,
}

impl NonlinearMediation {
    /// Estimate the natural effects on a declared request and seal them with their identity.
    ///
    /// # Errors
    /// Refusals `nonlinear_mediation.confounding`, `.treatment_induced_confounding`,
    /// `.cross_world_independence_not_declared`, `.interventional_effects_closed`, `.overlap`,
    /// `.integration_error`, `.rank_deficient` or an input-validation detail, and bounds
    /// exceeded; see [`NonlinearMediationArtifactError::refusal`].
    pub fn estimate(
        request: &NonlinearMediationRequestWire,
    ) -> Result<Self, NonlinearMediationArtifactError> {
        Ok(Self { artifact: NonlinearMediationArtifact::seal(request)? })
    }

    /// Consume an exported artifact by re-estimating from its embedded dataset, premises and
    /// configuration under the stored seed.
    ///
    /// `expected` is an identity the consumer retained independently.
    ///
    /// # Errors
    /// Corruption, an unknown major version, a changed identity, or a stored record that does
    /// not replay.
    pub fn consume(
        bytes: &[u8],
        expected: Option<&NonlinearMediationIdentity>,
    ) -> Result<Self, NonlinearMediationArtifactError> {
        Ok(Self { artifact: NonlinearMediationArtifact::from_bytes(bytes, expected)? })
    }

    /// Export the checksummed artifact.
    ///
    /// # Errors
    /// An empty id or an oversized payload.
    pub fn export(&self, artifact_id: &str) -> Result<Vec<u8>, NonlinearMediationArtifactError> {
        self.artifact.to_bytes(artifact_id)
    }

    /// The core estimate.
    #[must_use]
    pub fn result(&self) -> &NonlinearMediationEstimate {
        self.artifact.estimate()
    }

    /// The identity digests.
    #[must_use]
    pub fn identity(&self) -> &NonlinearMediationIdentity {
        self.artifact.identity()
    }

    /// The self-describing report record.
    #[must_use]
    pub fn report(&self) -> &NonlinearMediationMeta {
        self.artifact.meta()
    }

    /// The underlying artifact.
    #[must_use]
    pub fn artifact(&self) -> &NonlinearMediationArtifact {
        &self.artifact
    }
}

impl From<NonlinearMediationArtifactError> for CausalError {
    fn from(error: NonlinearMediationArtifactError) -> Self {
        match error.refusal() {
            Some((code, detail, text)) => CausalError::Compile {
                message: format!("{}{code}: {detail}: {text}", crate::error::REASON_PREFIX),
            },
            None => match error {
                NonlinearMediationArtifactError::LimitsExceeded(_) => {
                    CausalError::Compile { message: error.to_string() }
                }
                other => {
                    CausalError::Serialization(antecedent_io::IoError::Convert(other.to_string()))
                }
            },
        }
    }
}
