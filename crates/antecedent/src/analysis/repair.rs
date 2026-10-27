//! Identification-repair facade: which proposed studies make a failed causal
//! contract identify.
//!
//! A failed contract is a [`RepairFamily`]: a transport contract that misses
//! evidence ([`TransportRepairFamily`]) or a fixed-graph back-door contract
//! whose adjustment covariates were never measured jointly
//! ([`BackdoorRepairFamily`]), or a checked z-transport proof whose required
//! factors are missing ([`ZTransportRepairFamily`]). A failed identification
//! also yields one family per identifier: front-door ([`FrontDoorRepairFamily`]),
//! instrument ([`InstrumentRepairFamily`]), hedge ([`HedgeRepairFamily`]) and
//! sensitivity ([`SensitivityRepairFamily`]), and
//! [`identification_failure_obligations`] reads all of them from one failed
//! identification result. The family exposes its unresolved
//! [`EvidenceObligation`]s, each retaining its source proof step, and
//! [`repair_contract`] applies the hypothetical evidence of
//! [`DurableStudyCandidate`]s (one study, or a bounded subset) and re-runs the
//! family's own theorem checker on it. A candidate that only matches a variable
//! name never repairs a contract, and a search that a budget stopped keeps its
//! unevaluated subsets: exhaustion is never impossibility.
//!
//! [`RepairReportArtifact::build`] seals a finished report as a portable
//! `repair_search_receipt_v1` artifact that [`consume_repair_artifact`] replays
//! independently; a report is kept even when its artifact cannot be built. The specialized planners ([`crate::design::StudyPlanResult`]
//! and the transport and z-transport planners) are untouched.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub use antecedent_contract::{
    EvidenceObligation, EvidenceObligationKind, ObligationProvenance, ObligationRegime,
};
use antecedent_core::ExecutionContext;
pub use antecedent_design::obligation_adapters::identification_failure_obligations;
pub use antecedent_design::{
    BackdoorRepairFamily, DurableStudyCandidate, ExpectedEvidence, FamilyVerdict,
    FrontDoorRepairFamily, HedgeRepairFamily, InstrumentRepairFamily, LatentCause,
    RepairArtifactError, RepairClassification, RepairConsumeLimits, RepairError, RepairFamily,
    RepairFamilyRef, RepairLimits, RepairObjective, RepairOutcome, RepairReceipt, RepairReport,
    RepairReportArtifact, SensitivityFinding, SensitivityMethod, SensitivityRepairFamily,
    StudyCandidateError, StudyCostDeclaration, StudyKind, TransportRepairFamily, UnitRules,
    ZTransportRepairFamily, latent_expansion,
};

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
