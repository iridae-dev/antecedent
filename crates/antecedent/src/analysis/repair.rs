//! Identification-repair facade: which proposed studies make a failed causal
//! contract identify.
//!
//! A failed contract is a [`RepairFamily`]: a transport contract that misses
//! evidence ([`TransportRepairFamily`]) or a fixed-graph back-door contract
//! whose adjustment covariates were never measured jointly
//! ([`BackdoorRepairFamily`]), or a checked z-transport proof whose required
//! factors are missing ([`ZTransportRepairFamily`]). The family exposes its unresolved
//! [`EvidenceObligation`]s, each retaining its source proof step, and
//! [`repair_contract`] applies the hypothetical evidence of
//! [`DurableStudyCandidate`]s (one study, or a bounded subset) and re-runs the
//! family's own theorem checker on it. A candidate that only matches a variable
//! name never repairs a contract, and a search that a budget stopped keeps its
//! unevaluated subsets: exhaustion is never impossibility.
//!
//! [`repair_with_artifact`] also seals the finished report as a portable
//! `repair_search_receipt_v1` artifact that [`consume_repair_artifact`] replays
//! independently. The 2.2 specialized planners ([`crate::design::StudyPlanResult`]
//! and the transport and z-transport planners) are untouched.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
pub use antecedent_core::{
    EvidenceObligation, EvidenceObligationKind, ObligationProvenance, ObligationRegime,
};
pub use antecedent_design::{
    BackdoorRepairFamily, DurableStudyCandidate, ExpectedEvidence, FamilyVerdict,
    RepairArtifactError, RepairClassification, RepairConsumeLimits, RepairError, RepairFamily,
    RepairFamilyRef, RepairLimits, RepairObjective, RepairOutcome, RepairReceipt, RepairReport,
    RepairReportArtifact, StudyCandidateError, StudyCostDeclaration, StudyKind,
    TransportRepairFamily, UnitRules, ZTransportRepairFamily,
};

/// A refused repair or artifact operation: the registered reason code and the
/// stable `identification_repair.*` or `repair_artifact.*` detail.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum RepairFacadeError {
    /// The repair request was refused.
    #[error(transparent)]
    Repair(#[from] RepairError),
    /// The artifact could not be built, read or replayed.
    #[error(transparent)]
    Artifact(#[from] RepairArtifactError),
}

impl RepairFacadeError {
    /// Registered runtime-refusal reason code.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        match self {
            Self::Repair(error) => error.code,
            Self::Artifact(error) => error.code,
        }
    }

    /// Stable detail.
    #[must_use]
    pub const fn detail(&self) -> &'static str {
        match self {
            Self::Repair(error) => error.detail,
            Self::Artifact(error) => error.detail,
        }
    }
}

/// The unresolved evidence obligations of a failed contract, each with its
/// source proof step. A study never satisfies an
/// [`EvidenceObligationKind::EstablishAssumption`] obligation.
#[must_use]
pub fn unresolved_obligations(family: &dyn RepairFamily) -> Vec<EvidenceObligation> {
    family.unresolved_obligations()
}

/// Rank the candidates (and bounded subsets of them) that repair the contract.
///
/// # Errors
/// `identification_repair.bounds_exceeded` for a request beyond the declared
/// bounds, `identification_repair.cost_units_mismatch` when cost ranking
/// compares different unit labels and `identification_repair.budget` when the
/// budget is stopped before the search. A stop during the search is a report.
pub fn repair_contract(
    family: &dyn RepairFamily,
    candidates: &[DurableStudyCandidate],
    objective: RepairObjective,
    limits: RepairLimits,
    ctx: &ExecutionContext,
) -> Result<RepairReport, RepairError> {
    antecedent_design::repair(family, candidates, objective, limits, ctx)
}

/// [`repair_contract`] and the sealed artifact of its report.
///
/// # Errors
/// A refused repair, or a report that cannot be exported (cancellation stopped
/// it, or its graph cannot be stored).
pub fn repair_with_artifact(
    family: RepairFamilyRef<'_>,
    candidates: &[DurableStudyCandidate],
    objective: RepairObjective,
    limits: RepairLimits,
    ctx: &ExecutionContext,
) -> Result<(RepairReport, RepairReportArtifact), RepairFacadeError> {
    let owner: &dyn RepairFamily = match family {
        RepairFamilyRef::Transport(f) => f,
        RepairFamilyRef::Backdoor(f) => f,
        RepairFamilyRef::ZTransport(f) => f,
    };
    let report = repair_contract(owner, candidates, objective, limits, ctx)?;
    let artifact = RepairReportArtifact::build(family, candidates, &report)?;
    Ok((report, artifact))
}

/// Read a `repair_search_receipt_v1` container and replay it under the
/// consumer's `limits`: every stored classification is recomputed by the
/// family's theorem checker and the whole search must reproduce the stored
/// report. Returns the artifact and the replayed report.
///
/// # Errors
/// An unsupported version, stored limits beyond `limits`, a digest or replay
/// that does not match, or a replay that cancellation stopped.
pub fn consume_repair_artifact(
    bytes: &[u8],
    limits: RepairConsumeLimits,
    ctx: &ExecutionContext,
) -> Result<(RepairReportArtifact, RepairReport), RepairArtifactError> {
    let artifact = RepairReportArtifact::from_bytes(bytes)?;
    let report = artifact.consume(limits, ctx)?;
    Ok((artifact, report))
}
