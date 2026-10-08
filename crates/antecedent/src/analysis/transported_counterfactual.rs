//! 2.3.0 A5: the transported static path-specific counterfactual (narrow class).
//!
//! One cell is licensed: a source structural fit with **additive-noise** equations affine in
//! the covariates, finite-support source and target covariate laws, and a selection diagram
//! whose selection nodes point only at covariates or the treatment. The answer is the
//! target-population path-specific edge-intervention contrast
//! `theta_T = sum_z P_T(z) G(z)`, with `G(z)` the unit-level contrast of the two edge
//! assignments (the natural direct, indirect or total effect among them) computed from the
//! source equations alone, plus the source answer for comparison, the per-unit contrasts and a
//! derivation recording which premises were **checked** (acyclic well-formed model, no
//! selection on a mediator or outcome mechanism, covariates are pre-treatment roots, target
//! support inside source support, regime evidence present) and which were only **declared**
//! (additive noise, shared noise laws, unit-level cross-world independence, source fit equal
//! to the structural equations). The theorem and its proof are in
//! `antecedent_counterfactual::transported_path_specific`.
//!
//! The declared premises default to undeclared at every layer: omitting one refuses with the
//! core's `transported_counterfactual.*` detail. A selection on a mediator or outcome
//! mechanism refuses (`selection_on_mechanism`) and, when the contrast is sensitive to the
//! selected mechanism, retains the explicit two-model witness
//! (`transport_proven_non_transportable`).
//!
//! The GENERAL class (nonparametric mechanisms, recanting witnesses, nonlinear additive noise)
//! stays closed: [`super::temporal_counterfactual::transported_path_specific`] never
//! evaluates. This cell is a different, narrower theorem and does not open it.
//!
//! The result is exported as the `transported_counterfactual_v1` artifact; a consumer
//! recomputes it ([`TransportedCounterfactual::consume`]) and refuses any resealed change of a
//! premise, law, coefficient, selection or assignment when it retains the identity.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub use antecedent_counterfactual::transported_path_specific::{
    AdditiveLinearScm, Affine, CovariateLaw, DeclaredAssumptions, EdgeAssignment, LinearMechanism,
    NonRecoverableWitness, PathSpecificQuery, SelectionDiagram, SelectionNode,
    TransportedPathSpecificInput, TransportedPathSpecificRefusal, TransportedPathSpecificResult,
    evaluate_transported_path_specific,
};
use antecedent_io::transported_counterfactual_artifact as transported_wire;
pub use transported_wire::{
    AffineWire, AssignmentWire, DerivationWire, EvidenceWire, LawWire, MechanismWire, ModelWire,
    NonRecoverableWitnessWire, PremisesWire, RefusalWire, SelectionWire, SupportPointWire,
    TRANSPORTED_COUNTERFACTUAL_ARTIFACT_VERSION, TransportedCounterfactualArtifact,
    TransportedCounterfactualArtifactError, TransportedCounterfactualIdentity,
    TransportedCounterfactualReportWire, TransportedCounterfactualRequestWire,
    TransportedCounterfactualResultWire, UnitContrastWire,
};

use crate::error::CausalError;

/// A sealed or consumed transported path-specific counterfactual.
#[derive(Clone, Debug, PartialEq)]
pub struct TransportedCounterfactual {
    artifact: TransportedCounterfactualArtifact,
}

impl TransportedCounterfactual {
    /// Evaluate the transported contrast and seal it with its identity.
    ///
    /// # Errors
    /// `cell_not_licensed` (`transported_counterfactual.nonadditive_mechanism`,
    /// `.noise_law_not_shared`, `.cross_world_independence_missing` for an undeclared premise;
    /// `.selection_on_mechanism` without a witness), `transport_proven_non_transportable`
    /// (`.selection_on_mechanism` with the two-model witness retained in
    /// [`RefusalWire::witness`]), `transport_support_failure` (`.overlap_failure`),
    /// `transport_missing_evidence` (`.factor_missing`) or `invalid_argument` (`.invalid_model`,
    /// `.invalid_query`, `.invalid_law`, `.invalid_diagram`, `.invalid_factor`).
    pub fn evaluate(
        request: &TransportedCounterfactualRequestWire,
    ) -> Result<Self, TransportedCounterfactualArtifactError> {
        Ok(Self { artifact: TransportedCounterfactualArtifact::seal(request)? })
    }

    /// Consume an exported artifact by recomputing the contrast.
    ///
    /// `expected` is an identity the consumer retained independently; a changed premise, law,
    /// coefficient, selection or assignment is refused even when the artifact was resealed.
    ///
    /// # Errors
    /// Corruption, an unknown major version, a core refusal (including false-labelled declared
    /// premises), a changed identity, or a stored contrast that does not replay.
    pub fn consume(
        bytes: &[u8],
        expected: Option<&TransportedCounterfactualIdentity>,
    ) -> Result<Self, TransportedCounterfactualArtifactError> {
        Ok(Self { artifact: TransportedCounterfactualArtifact::from_bytes(bytes, expected)? })
    }

    /// Export the checksummed artifact.
    ///
    /// # Errors
    /// An empty id or an oversized payload.
    pub fn export(
        &self,
        artifact_id: &str,
    ) -> Result<Vec<u8>, TransportedCounterfactualArtifactError> {
        self.artifact.to_bytes(artifact_id)
    }

    /// The target contrast `sum_z P_T(z) G(z)`.
    #[must_use]
    pub const fn target_contrast(&self) -> f64 {
        self.artifact.result().target_contrast
    }

    /// The source contrast `sum_z P_S(z) G(z)`.
    #[must_use]
    pub const fn source_contrast(&self) -> f64 {
        self.artifact.result().source_contrast
    }

    /// The answer: contrasts, per-unit contrasts and the derivation.
    #[must_use]
    pub const fn result(&self) -> &TransportedCounterfactualResultWire {
        self.artifact.result()
    }

    /// The canonical request.
    #[must_use]
    pub const fn request(&self) -> &TransportedCounterfactualRequestWire {
        self.artifact.request()
    }

    /// The identity digests.
    #[must_use]
    pub const fn identity(&self) -> &TransportedCounterfactualIdentity {
        self.artifact.identity()
    }

    /// The self-describing report.
    #[must_use]
    pub fn report(&self) -> TransportedCounterfactualReportWire {
        self.artifact.report()
    }

    /// The underlying artifact.
    #[must_use]
    pub const fn artifact(&self) -> &TransportedCounterfactualArtifact {
        &self.artifact
    }
}

impl From<TransportedCounterfactualArtifactError> for CausalError {
    fn from(error: TransportedCounterfactualArtifactError) -> Self {
        match error.refusal() {
            Some(refusal) => {
                let mut message =
                    format!("{}{}: {}", crate::error::REASON_PREFIX, refusal.code, refusal.detail);
                if let Some(offending) = &refusal.offending {
                    message.push_str(": ");
                    message.push_str(offending);
                }
                CausalError::Compile { message }
            }
            None => match error {
                TransportedCounterfactualArtifactError::LimitsExceeded(_) => {
                    CausalError::Compile { message: error.to_string() }
                }
                other => {
                    CausalError::Serialization(antecedent_io::IoError::Convert(other.to_string()))
                }
            },
        }
    }
}
