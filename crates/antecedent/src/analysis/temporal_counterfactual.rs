//! Fixed-population temporal counterfactual and the closed transported route.
//!
//! One cell is licensed: on a fixed, fully observed, Markovian two-slice DAG with
//! linear-Gaussian additive-noise mechanisms, each unit's exogenous history is abduced once from
//! its factual trajectory and **both** named two-step action histories are replayed against that
//! same history ([`TemporalCounterfactual::evaluate`]). The answer is a point: per-unit final
//! outcomes in both worlds and their sample mean contrast, with a shared-abduction receipt that
//! an independent consumer replays ([`TemporalCounterfactual::consume`]).
//!
//! The transported path-specific counterfactual stays **closed**:
//! [`transported_path_specific`] never evaluates, it reports exactly which prerequisite gate is
//! missing and which regime factors are absent. Neither a transported mean nor a source
//! counterfactual alone licenses their composition.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;
use std::convert::Infallible;

use antecedent_core::ExecutionContext;
pub use antecedent_counterfactual::temporal_cross_world::{
    MAX_HORIZON, MAX_UNITS, MAX_WORLDS, MECHANISM_CLASS, SharedAbductionReceipt,
    TemporalCounterfactualResult, TemporalNode,
};
pub use antecedent_counterfactual::transported_gate::{
    PopulationRole, RegimeFactorKey, TransportedCounterfactualGate,
    TransportedCounterfactualPrerequisites, TransportedCounterfactualRefusal,
    refuse_transported_counterfactual,
};
use antecedent_io::temporal_counterfactual_artifact as temporal_counterfactual_wire;
pub use temporal_counterfactual_wire::{
    ActionHistoryWire, FactualUnitWire, FitWire, MechanismWire, ReceiptWire, RefusalWire,
    TEMPORAL_COUNTERFACTUAL_ARTIFACT_VERSION, TemporalCounterfactualArtifact,
    TemporalCounterfactualArtifactError, TemporalCounterfactualIdentity,
    TemporalCounterfactualReportWire, TemporalCounterfactualRequestWire, TemporalGraphWire,
    UnitOutcomeWire, WitnessWire,
};

pub use antecedent_counterfactual::temporal_cross_world::general::{
    GeneralAbductionReceipt, GeneralTemporalResult,
};
pub use antecedent_io::temporal_counterfactual_general_artifact::{
    GENERAL_TEMPORAL_COUNTERFACTUAL_ARTIFACT_VERSION, GeneralFactualUnitWire, GeneralFitWire,
    GeneralMechanismWire, GeneralTemporalCounterfactualArtifact,
    GeneralTemporalCounterfactualArtifactError, GeneralTemporalIdentity, GeneralTemporalReportWire,
    GeneralTemporalRequestWire, GeneralWorldWire,
};

use crate::error::CausalError;

/// Stage id reported on a cancellation raised by this cell.
pub const STAGE_TEMPORAL_COUNTERFACTUAL: &str = "temporal_counterfactual";

/// A sealed or consumed temporal counterfactual answer.
#[derive(Clone, Debug, PartialEq)]
pub struct TemporalCounterfactual {
    artifact: TemporalCounterfactualArtifact,
}

impl TemporalCounterfactual {
    /// Abduce each unit once, replay both action histories against that history, and seal the
    /// answer with its shared-abduction receipt.
    ///
    /// # Errors
    /// `route_not_supported` (`temporal_counterfactual.unpaired_histories`, `.time_misaligned`,
    /// `.shared_history_missing`, `.latent_confounding`, `.refuting_history`), `invalid_argument`
    /// (`.horizon_exceeded`, `.too_many_units`, `.invalid_graph`, `.fit_mismatch`,
    /// `.history_name_invalid`, `.non_finite_history`) or cancellation. A refusal retains its
    /// witness in [`RefusalWire::witness`].
    pub fn evaluate(
        request: &TemporalCounterfactualRequestWire,
        ctx: &ExecutionContext,
    ) -> Result<Self, TemporalCounterfactualArtifactError> {
        Ok(Self { artifact: TemporalCounterfactualArtifact::seal(request, ctx)? })
    }

    /// Consume an exported artifact by recomputing both worlds.
    ///
    /// `expected` is an identity the consumer retained independently; a changed action time,
    /// unit history, snapshot or mechanism fit is refused even when the artifact was resealed.
    ///
    /// # Errors
    /// Corruption, an unknown major version, a changed identity, a stored outcome or receipt
    /// that does not replay, cancellation or a core refusal.
    pub fn consume(
        bytes: &[u8],
        expected: Option<&TemporalCounterfactualIdentity>,
        ctx: &ExecutionContext,
    ) -> Result<Self, TemporalCounterfactualArtifactError> {
        Ok(Self { artifact: TemporalCounterfactualArtifact::from_bytes(bytes, expected, ctx)? })
    }

    /// Export the checksummed artifact.
    ///
    /// # Errors
    /// An empty id or an oversized payload.
    pub fn export(
        &self,
        artifact_id: &str,
    ) -> Result<Vec<u8>, TemporalCounterfactualArtifactError> {
        self.artifact.to_bytes(artifact_id)
    }

    /// The core result.
    #[must_use]
    pub fn result(&self) -> &TemporalCounterfactualResult {
        self.artifact.result()
    }

    /// The shared-abduction receipt.
    #[must_use]
    pub fn receipt(&self) -> &SharedAbductionReceipt {
        &self.artifact.result().receipt
    }

    /// The identity digests.
    #[must_use]
    pub fn identity(&self) -> &TemporalCounterfactualIdentity {
        self.artifact.identity()
    }

    /// The self-describing report.
    #[must_use]
    pub fn report(&self) -> TemporalCounterfactualReportWire {
        self.artifact.report()
    }

    /// The underlying artifact.
    #[must_use]
    pub fn artifact(&self) -> &TemporalCounterfactualArtifact {
        &self.artifact
    }
}

/// A sealed or consumed temporal counterfactual answer for any horizon and any number of named
/// action histories (the general counterpart of [`TemporalCounterfactual`]).
#[derive(Clone, Debug, PartialEq)]
pub struct GeneralTemporalCounterfactual {
    artifact: GeneralTemporalCounterfactualArtifact,
}

impl GeneralTemporalCounterfactual {
    /// Abduce each unit once, replay every named action history against that history, and seal
    /// the answer with its shared-abduction receipt.
    ///
    /// # Errors
    /// The refusals of [`TemporalCounterfactual::evaluate`] for the general layout, or
    /// cancellation. A refusal retains its witness in the refusal wire.
    pub fn evaluate(
        request: &GeneralTemporalRequestWire,
        ctx: &ExecutionContext,
    ) -> Result<Self, GeneralTemporalCounterfactualArtifactError> {
        Ok(Self { artifact: GeneralTemporalCounterfactualArtifact::seal(request, ctx)? })
    }

    /// Consume an exported artifact by recomputing every world.
    ///
    /// # Errors
    /// Corruption, an unknown major version, a changed identity, a stored outcome or receipt
    /// that does not replay, cancellation or a core refusal.
    pub fn consume(
        bytes: &[u8],
        expected: Option<&GeneralTemporalIdentity>,
        ctx: &ExecutionContext,
    ) -> Result<Self, GeneralTemporalCounterfactualArtifactError> {
        Ok(Self {
            artifact: GeneralTemporalCounterfactualArtifact::from_bytes(bytes, expected, ctx)?,
        })
    }

    /// Export the checksummed artifact.
    ///
    /// # Errors
    /// An empty id or an oversized payload.
    pub fn export(
        &self,
        artifact_id: &str,
    ) -> Result<Vec<u8>, GeneralTemporalCounterfactualArtifactError> {
        self.artifact.to_bytes(artifact_id)
    }

    /// The core result.
    #[must_use]
    pub fn result(&self) -> &GeneralTemporalResult {
        self.artifact.result()
    }

    /// The shared-abduction receipt.
    #[must_use]
    pub fn receipt(&self) -> &GeneralAbductionReceipt {
        &self.artifact.result().receipt
    }

    /// The identity digests.
    #[must_use]
    pub fn identity(&self) -> &GeneralTemporalIdentity {
        self.artifact.identity()
    }

    /// The self-describing report.
    #[must_use]
    pub fn report(&self) -> GeneralTemporalReportWire {
        self.artifact.report()
    }

    /// The underlying artifact.
    #[must_use]
    pub fn artifact(&self) -> &GeneralTemporalCounterfactualArtifact {
        &self.artifact
    }
}

/// The closed transported path-specific counterfactual route.
///
/// Always refuses: `transported_counterfactual.route_frozen` (`cell_not_licensed`) while a
/// prerequisite gate is missing (and also when every gate passes and every required regime
/// factor is present, because the joint theorem has not passed), and
/// `transported_counterfactual.factor_missing` (`transport_missing_evidence`) when every gate
/// passes but a required source or target regime factor is absent from `supplied`. The missing
/// gates and factors are retained on the refusal.
///
/// # Errors
/// Always: the typed [`TransportedCounterfactualRefusal`].
#[allow(
    clippy::result_large_err,
    reason = "the typed refusal retains the missing gates and factors by value on an always-refusing cold path; boxing would only complicate the public error contract"
)]
pub fn transported_path_specific(
    prerequisites: &TransportedCounterfactualPrerequisites,
    required_factors: &[RegimeFactorKey],
    supplied: &BTreeMap<RegimeFactorKey, String>,
) -> Result<Infallible, TransportedCounterfactualRefusal> {
    Err(refuse_transported_counterfactual(prerequisites, required_factors, supplied))
}

impl From<TemporalCounterfactualArtifactError> for CausalError {
    fn from(error: TemporalCounterfactualArtifactError) -> Self {
        if matches!(error, TemporalCounterfactualArtifactError::Cancelled) {
            return CausalError::Cancelled { stage: STAGE_TEMPORAL_COUNTERFACTUAL };
        }
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
                TemporalCounterfactualArtifactError::LimitsExceeded(_) => {
                    CausalError::Compile { message: error.to_string() }
                }
                other => {
                    CausalError::Serialization(antecedent_io::IoError::Convert(other.to_string()))
                }
            },
        }
    }
}

impl From<GeneralTemporalCounterfactualArtifactError> for CausalError {
    fn from(error: GeneralTemporalCounterfactualArtifactError) -> Self {
        if matches!(error, GeneralTemporalCounterfactualArtifactError::Cancelled) {
            return CausalError::Cancelled { stage: STAGE_TEMPORAL_COUNTERFACTUAL };
        }
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
                GeneralTemporalCounterfactualArtifactError::LimitsExceeded(_) => {
                    CausalError::Compile { message: error.to_string() }
                }
                other => {
                    CausalError::Serialization(antecedent_io::IoError::Convert(other.to_string()))
                }
            },
        }
    }
}
